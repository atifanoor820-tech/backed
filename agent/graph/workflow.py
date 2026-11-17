"""LangGraph workflow: interpret -> plan_review -> approval -> execute (bounded loop) -> verify -> report."""

import concurrent.futures
import json
import re
from pathlib import Path
from typing import Literal

from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.config import get_settings
from agent.events import add_message, cancellations, emit, set_status
from agent.interpreter import finalize_spec, interpret
from agent.llm import gemini
from agent.memory import delete_memory, find_memory, list_memories, looks_secret, save_memory
from agent.autonomy import get_level
from agent.safety import SafetyError, low_confidence_block, step_needs_approval, workspace_roots
from agent.state import GraphState, PlanStep, Risk, TaskSpec, ToolResult
from agent.tools import TOOLS, ToolContext, compact_catalog

_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="abbas-tool")


def _cancelled(state: GraphState) -> bool:
    return cancellations.is_cancelled(state["task_id"])


def _apply_memory_updates(tid: str, spec: TaskSpec) -> None:
    store = get_store()
    if store is None:
        return
    for fact in spec.remember:
        if looks_secret(fact):
            emit(tid, "memory_saved", "Not saved: it looks like a password, key or card number.", {"id": None})
            spec.reply = (spec.reply or "") + "\n(Main passwords, keys ya card numbers yaad nahi rakhta.)"
        elif saved := save_memory(store, fact, tid):
            emit(tid, "memory_saved", saved["text"], {"id": saved["id"]})
    for ref in spec.forget:
        target = ref if store.get(("memories", get_settings().user_id), ref) else (find_memory(store, ref) or {}).get("id")
        if target and delete_memory(store, target):
            emit(tid, "memory_deleted", ref, {"id": target})
        else:
            emit(tid, "memory_deleted", f"nothing matching '{ref}' to forget", {"id": None})


def node_interpret(state: GraphState) -> GraphState:
    tid = state["task_id"]
    emit(tid, "planning_started", "Interpreting the request")
    store = get_store()
    memories = list_memories(store) if store is not None else []
    history = state.get("history", [])[-2 * get_settings().history_turns:]
    spec, warning = interpret(state["request"], tid, history=history, context=state.get("context", {}),
                              memories=memories)
    if warning:
        emit(tid, "tool_failed", warning)
    _apply_memory_updates(tid, spec)
    emit(tid, "intent_resolved", spec.interpreted_objective,
         {"language": spec.language, "interpreter": spec.interpreter, "normalized": spec.normalized_request})
    if spec.plan:
        emit(tid, "planning_completed", f"{len(spec.plan)} step plan, {spec.risk_level.value} risk",
             {"plan": [s.model_dump(mode="json") for s in spec.plan]})
    auto = [s for s in spec.plan if s.risk != Risk.low and not step_needs_approval(s)]
    if auto:  # audit trail: what ran without asking, and why
        emit(tid, "approval_granted", f"Auto-approved by autonomy level '{get_level()}': "
             + "; ".join(f"{s.description} ({s.risk.value})" for s in auto),
             {"auto": True, "autonomy": get_level()})
    set_status(tid, "running", spec=spec.model_dump(mode="json"))
    return {"spec": spec.model_dump(mode="json"), "step_index": 0, "results": [], "attempts": 0}


def route_after_interpret(state: GraphState) -> Literal["respond", "blocked", "request_approval", "execute"]:
    spec = TaskSpec.model_validate(state["spec"])
    if spec.reply and not spec.plan:
        return "respond"
    if spec.clarification_question or not spec.plan:
        return "blocked"
    if low_confidence_block(state.get("source", "text"), state.get("transcript_confidence"), spec.risk_level):
        return "blocked"
    return "request_approval" if spec.requires_approval else "execute"


def node_respond(state: GraphState) -> GraphState:
    spec = TaskSpec.model_validate(state["spec"])
    return {"status": "completed", "report": spec.reply or ""}


def node_blocked(state: GraphState) -> GraphState:
    spec = TaskSpec.model_validate(state["spec"])
    reason = spec.clarification_question or low_confidence_block(
        state.get("source", "text"), state.get("transcript_confidence"), spec.risk_level) or "Nothing to execute."
    emit(state["task_id"], "task_blocked", reason)
    return {"status": "blocked", "report": reason}


def node_request_approval(state: GraphState) -> GraphState:
    spec = TaskSpec.model_validate(state["spec"])
    tid = state["task_id"]
    pending = spec.plan[state.get("step_index", 0):]
    emit(tid, "approval_required", spec.approval_reason or "Approval required",
         {"steps": [s.model_dump(mode="json") for s in pending if s.risk != Risk.low]})
    set_status(tid, "awaiting_approval")
    return {}


def node_approval(state: GraphState) -> GraphState:
    # LangGraph re-runs this node from the top on resume, so it must have no side effects before interrupt().
    spec = TaskSpec.model_validate(state["spec"])
    tid = state["task_id"]
    decision = interrupt({"reason": spec.approval_reason})
    if decision.get("approved") and not _cancelled(state):
        emit(tid, "approval_granted", "Approved by user")
        set_status(tid, "running")
        return {"status": "running"}
    if _cancelled(state):
        return {"status": "cancelled", "report": "Stopped on your request before running anything."}
    emit(tid, "approval_rejected", "Rejected by user")
    return {"status": "cancelled", "report": "You declined the plan, so I did not run it."}


def route_after_approval(state: GraphState) -> Literal["execute", "finish"]:
    return "execute" if state.get("status") == "running" else "finish"


def node_execute(state: GraphState) -> GraphState:
    """Run exactly one bounded step and record the observed result."""
    tid, idx = state["task_id"], state["step_index"]
    spec = TaskSpec.model_validate(state["spec"])
    if _cancelled(state):
        return {"status": "cancelled", "report": "Stopped on your request."}
    step = spec.plan[idx]
    tool = TOOLS[step.tool]
    emit(tid, "tool_started", step.description or step.tool, {"tool": step.tool, "args": step.args, "step": idx})
    try:
        args = tool.input_model.model_validate(step.args)
        fut = _pool.submit(tool.run, args, ToolContext(task_id=tid, step_index=idx))
        result = fut.result(timeout=tool.timeout_s or get_settings().tool_timeout_s)
    except concurrent.futures.TimeoutError:
        result = ToolResult(ok=False, summary=f"{step.tool} timed out after {tool.timeout_s or get_settings().tool_timeout_s:.0f}s")
    except SafetyError as e:
        result = ToolResult(ok=False, summary=f"blocked by safety policy: {e}")
    except Exception as e:
        result = ToolResult(ok=False, summary=f"{step.tool} error: {type(e).__name__}: {e}")

    attempts = state.get("attempts", 0)
    if not result.ok and tool.retryable and attempts < 1 and not _cancelled(state):
        emit(tid, "tool_failed", f"{result.summary} — retrying once", {"tool": step.tool, "step": idx})
        return {"attempts": attempts + 1}

    emit(tid, "tool_completed" if result.ok else "tool_failed", result.summary,
         {"tool": step.tool, "step": idx, "ok": result.ok, "data": result.data, "evidence": result.evidence})
    results = [*state.get("results", []), {"step": idx, "tool": step.tool, **result.model_dump()}]
    return {"results": results, "step_index": idx + 1, "attempts": 0}


def route_after_execute(state: GraphState) -> Literal["execute", "verify", "finish"]:
    if state.get("status") == "cancelled":
        return "finish"
    spec = TaskSpec.model_validate(state["spec"])
    results = state.get("results", [])
    if results and not results[-1]["ok"] and len(results) == state["step_index"]:
        return "verify"  # stop the plan at the first failed step
    limit = min(len(spec.plan), get_settings().max_total_steps)
    return "execute" if state["step_index"] < limit else "verify"


def node_verify(state: GraphState) -> GraphState:
    spec = TaskSpec.model_validate(state["spec"])
    results = state.get("results", [])
    done = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    skipped = len(spec.plan) - len(results)
    lines = []
    for r in results:
        lines.append(f"{'✔' if r['ok'] else '✘'} {r['summary']}")
        data = r.get("data") or {}
        if r["ok"] and r["tool"] == "workspace.read" and data.get("content"):  # the user asked what it says
            lines.append(data["content"][:1500] + ("…" if data.get("truncated") or len(data["content"]) > 1500 else ""))
        elif r["ok"] and r["tool"] == "workspace.list":
            lines.append(", ".join(e["name"] for e in data.get("entries", [])[:40]) or "(empty)")
    if skipped:
        lines.append(f"… {skipped} step(s) not run")
    status = "completed" if results and not failed and not skipped else "failed"
    head = "Done." if status == "completed" else f"Incomplete: {len(done)}/{len(spec.plan)} steps succeeded."
    s = get_settings()
    if spec.interpreter == "llm" and (state.get("rounds", 0) >= s.max_rounds or len(results) >= s.max_total_steps):
        # A multi-round goal is only "done" when the reviewer confirms it; hitting the budget is not success.
        status = "failed"
        head = (f"Stopped after {state.get('rounds', 0)} rounds / {len(results)} steps (the safety limit) before the "
                "goal was confirmed complete. Say 'continue' to keep going.")
    return {"status": status, "report": head + "\n" + "\n".join(lines)}


# Words that mark a request as a multi-part goal rather than a single command.
GOAL_RE = re.compile(r"\b(then|phir|uske baad|us ke baad|after that|project|website|site|frontend|backend|"
                     r"research|build|deploy|code|coding|install|setup|competitors?)\b", re.I)
WORK_TOOLS = {"code.write", "terminal.run", "research.brief"}
CLAUSE_RE = re.compile(r"\b(?:aur|and|then|phir|uske baad|after that)\b|[;]", re.I)


def needs_review(spec: TaskSpec, results: list[dict]) -> bool:
    """Review costs a full model call (~3-10 s), so only goals that may be unfinished get one.

    The planner's own multi_round flag is not trusted alone (it once planned 1 step of a 5-part goal with the
    flag off), so a request also counts as a goal by its length, its wording or the kind of work it started.
    """
    if spec.multi_round or any(not r["ok"] for r in results):
        return True
    if any(r["tool"] in WORK_TOOLS for r in results):
        return True
    request = spec.original_request
    if len(request.split()) >= 12 or GOAL_RE.search(request):
        return True
    # "X kholo aur Y click karo" is two actions: fine if two steps ran, suspicious if only one did.
    parts = 1 + len(CLAUSE_RE.findall(request))
    return len(results) < parts


def route_after_verify(state: GraphState) -> Literal["review", "finish"]:
    """Goals (and anything that failed) are reviewed against the user's request before they may end;
    simple commands finish as soon as their steps succeed."""
    spec = TaskSpec.model_validate(state["spec"])
    s = get_settings()
    results = state.get("results", [])
    if _cancelled(state) or not s.gemini_enabled:
        return "finish"
    if spec.interpreter != "llm" and all(r["ok"] for r in results):
        return "finish"  # an instant (rule-planned) command that worked; failures still get repaired below
    if state.get("rounds", 0) >= s.max_rounds or len(results) >= s.max_total_steps:
        return "finish"
    return "review" if needs_review(spec, results) else "finish"


REVIEW_SYSTEM = """You are Abbas's progress reviewer. You get the user's GOAL and every step run so far with its
result. Decide what happens next. Return JSON:
{"done": bool,                     // true only if the WHOLE goal is achieved and verified by the results
 "summary": "<for the user, in their language: what was achieved, where (paths/URLs), what is left>",
 "remaining": ["<parts of the goal not done yet>"],
 "next_plan": [{"tool": ..., "args": {...}, "description": ...}],   // next 1-6 steps, empty if done
 "question": "<only if you truly need the user's input to continue, else null>"}
Rules: use ONLY the tools listed. Never claim something worked unless a result shows it, and quote file paths
exactly as the results show them. Code is only done after "npm run build" succeeds following the last code change.
If code.write reports packages that are not installed, run npm install for them. If a step failed, fix the cause
(e.g. read the build error in output_tail and call code.write with the exact error to fix it) instead of repeating
it unchanged. Prefer several focused code.write steps (layout, each section, components) over one huge file. If a path is outside the allowed folders, ask the user to add it in Settings -> Folders.
Do not redo steps that already succeeded. Tools:
%s"""


def _result_digest(results: list[dict], plan: list[dict]) -> str:
    lines = []
    for r in results[-25:]:
        args = plan[r["step"]]["args"] if r["step"] < len(plan) else {}
        data = r.get("data") or {}
        extra = {k: data[k] for k in ("path", "url", "title", "files", "notes", "summary", "exit_code", "window", "project")
                 if data.get(k) not in (None, "", [])}
        # What the step actually observed, so the reviewer doesn't re-run work to "see" it.
        if data.get("content"):
            extra["content"] = str(data["content"])[:1500]
        if data.get("text"):
            extra["text"] = str(data["text"])[:1500]
        if data.get("entries"):
            extra["entries"] = [e.get("name") for e in data["entries"][:40]]
        if data.get("results"):
            extra["results"] = [{"title": x.get("title"), "url": x.get("url")} for x in data["results"][:8]]
        if not r["ok"] or r["tool"] == "terminal.run":
            extra["output_tail"] = str(data.get("output_tail", ""))[-2500:]
        lines.append(f"[{r['step']}] {r['tool']} {json.dumps(args)[:300]} -> {'OK' if r['ok'] else 'FAILED'}: "
                     f"{r['summary']}" + (f"\n    data: {json.dumps(extra)[:3000]}" if extra else ""))
    return "\n".join(lines)


def _build_gate(results: list[dict], plan: list[dict]) -> tuple[str, str] | None:
    """None if no code was written or a build passed after the last code change.
    ("unverified", project) if nothing was built since; ("failing", project) if the last build failed."""
    last_code = None
    for r in results:
        if r["ok"] and r["tool"] == "code.write":
            last_code = r
    if last_code is None:
        return None
    project = (last_code.get("data") or {}).get("project") or plan[last_code["step"]]["args"].get("project_dir", "")
    builds = [r for r in results if r["step"] > last_code["step"] and r["tool"] == "terminal.run"
              and re.search(r"\b(build|tsc)\b", str(plan[r["step"]]["args"].get("command", "")))]
    if not builds:
        return ("unverified", project)
    return None if builds[-1]["ok"] else ("failing", project)


_STARTER_MARKERS = ("utm_source=create-next-app", "Get started by editing", "To get started, edit the page")


def _starter_homepage(project: str) -> str | None:
    """Relative path of a homepage that is still the create-next-app starter, if any."""
    for rel in ("src/app/page.tsx", "app/page.tsx", "src/app/page.jsx", "app/page.jsx"):
        p = Path(project) / rel
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            return rel if any(m in text for m in _STARTER_MARKERS) else None
    return None


def node_review(state: GraphState) -> GraphState:
    tid = state["task_id"]
    spec = TaskSpec.model_validate(state["spec"])
    s = get_settings()
    rounds = state.get("rounds", 0) + 1
    results = state.get("results", [])
    emit(tid, "planning_started", f"Checking progress on the goal (round {rounds}/{s.max_rounds})")
    plan_dump = [p.model_dump(mode="json") for p in spec.plan]
    user = (f"GOAL: {spec.original_request}\nInterpreted as: {spec.interpreted_objective}\n"
            f"Allowed folders: {', '.join(str(p) for p in workspace_roots())}\n"
            f"Round {rounds} of {s.max_rounds}; {s.max_total_steps - len(results)} steps left in the budget.\n\n"
            f"STEPS SO FAR:\n{_result_digest(results, plan_dump)}")
    system = REVIEW_SYSTEM % json.dumps(compact_catalog())
    correction = ""
    for attempt in range(2):  # one retry when the proposed plan fails validation
        try:
            out = dict(gemini.generate_json(system, user + correction, timeout_s=90))  # we amend it below
        except Exception as e:
            report = state.get("report", "")
            return {"rounds": rounds, "report": f"{report}\n(Couldn't review progress: {str(e)[:150]})"}

        summary = str(out.get("summary") or "").strip()
        remaining = [str(x) for x in (out.get("remaining") or [])][:10]
        gate = _build_gate(results, plan_dump)
        if out.get("done") and gate:
            # The model may not declare code "done" on its own word: the project must build after the last change.
            kind, project = gate
            out["done"] = False
            if kind == "unverified":
                summary = (summary + "\n" if summary else "") + "Verifying that the project builds before calling it done."
                out["next_plan"] = [{"tool": "terminal.run", "args": {"command": "npm run build", "cwd": project},
                                     "description": "Build the project to verify the code compiles"}]
            else:
                remaining = remaining or ["fix the build errors shown in the last build output"]
        code_projects = {(r.get("data") or {}).get("project") for r in results if r["ok"] and r["tool"] == "code.write"}
        starter = next(((p, rel) for p in code_projects if p and (rel := _starter_homepage(p))), None)
        if out.get("done") and starter:
            # A site is not done while its homepage is still the create-next-app placeholder.
            project, rel = starter
            out["done"] = False
            summary = (summary + "\n" if summary else "") + f"The homepage ({rel}) is still the Next.js starter page - building it."
            context = ["docs/research.md"] if (Path(project) / "docs/research.md").exists() else []
            out["next_plan"] = [
                {"tool": "code.write", "args": {"project_dir": project, "files": [rel], "context_files": context,
                 "instructions": "Replace the create-next-app starter homepage with the real homepage for this site, "
                                 "reusing the existing layout, components and theme (read them in the snapshot)."},
                 "description": "Build the real homepage"},
                {"tool": "terminal.run", "args": {"command": "npm run build", "cwd": project},
                 "description": "Verify the build"}]
        if out.get("done") or out.get("question"):
            break
        nxt = TaskSpec(original_request=spec.original_request, normalized_request=spec.normalized_request,
                       language=spec.language, interpreted_objective=spec.interpreted_objective, interpreter="llm",
                       plan=[PlanStep(**{k: v for k, v in p.items() if k in ("tool", "args", "description")})
                             for p in (out.get("next_plan") or []) if isinstance(p, dict) and p.get("tool")][: s.max_steps])
        nxt = finalize_spec(nxt, tid)
        if nxt.plan or not nxt.clarification_question or attempt == 1:
            break
        emit(tid, "review_completed", f"Proposed plan was invalid ({nxt.clarification_question}); asking for a corrected one",
             {"done": False, "remaining": remaining, "round": rounds})
        correction = (f"\n\nYOUR PREVIOUS next_plan WAS REJECTED: {nxt.clarification_question}\n"
                      "Return the same JSON again with a corrected next_plan in which every step has all required args.")

    emit(tid, "review_completed", summary[:500] or "Reviewed progress",
         {"done": bool(out.get("done")), "remaining": remaining, "round": rounds})
    checklist = "\n".join(f"{'✔' if r['ok'] else '✘'} {r['summary']}" for r in results)
    if out.get("done"):
        return {"rounds": rounds, "status": "completed", "report": f"{summary}\n{checklist}"}
    if out.get("question"):
        emit(tid, "task_blocked", str(out["question"]))
        return {"rounds": rounds, "status": "blocked", "report": f"{out['question']}\n{checklist}"}

    if not nxt.plan:
        left = "\n".join(f"• {x}" for x in remaining) or "• (no further plan)"
        why = f"\n{nxt.clarification_question}" if nxt.clarification_question else ""
        return {"rounds": rounds, "status": "failed",
                "report": f"{summary or 'I could not finish the goal.'}\nStill to do:\n{left}{why}\n{checklist}"}

    done_steps = spec.plan[: state["step_index"]]
    spec.plan = done_steps + nxt.plan
    spec.requires_approval, spec.approval_reason = nxt.requires_approval, nxt.approval_reason
    spec.risk_level = nxt.risk_level
    emit(tid, "planning_completed", f"Next: {len(nxt.plan)} step(s) — " + "; ".join(p.description or p.tool for p in nxt.plan),
         {"plan": [p.model_dump(mode="json") for p in nxt.plan], "round": rounds})
    auto = [p for p in nxt.plan if p.risk != Risk.low and not step_needs_approval(p)]
    if auto:
        emit(tid, "approval_granted", f"Auto-approved by autonomy level '{get_level()}': "
             + "; ".join(f"{p.description} ({p.risk.value})" for p in auto), {"auto": True, "autonomy": get_level()})
    set_status(tid, "running", spec=spec.model_dump(mode="json"))
    return {"rounds": rounds, "status": "running", "spec": spec.model_dump(mode="json"), "attempts": 0}


def route_after_review(state: GraphState) -> Literal["request_approval", "execute", "finish"]:
    if state.get("status") != "running":
        return "finish"
    spec = TaskSpec.model_validate(state["spec"])
    return "request_approval" if spec.requires_approval else "execute"


def node_announce(state: GraphState) -> GraphState:
    """Publish the outcome. Runs after `finish`, so the turn is already checkpointed into conversation memory
    by the time anyone sees the task as completed."""
    tid = state["task_id"]
    status = state.get("status", "failed")
    report = state.get("report", "")
    set_status(tid, status, result=report)
    add_message(tid, "assistant", report)
    terminal = {"completed": "task_completed", "cancelled": "task_cancelled", "failed": "task_failed"}
    if status in terminal:  # "blocked" already emitted task_blocked
        emit(tid, terminal[status], report[:500], {"status": status})
    return {}


def node_finish(state: GraphState) -> GraphState:
    tid = state["task_id"]
    status = state.get("status", "failed")
    report = state.get("report", "")
    # Short-term memory: this turn goes into the conversation thread's state.
    return {
        "history": [{"role": "user", "content": state["request"], "task_id": tid},
                    {"role": "assistant", "content": report[:2000], "task_id": tid, "status": status}],
        "context": _context_from(state),
    }


def _context_from(state: GraphState) -> dict:
    """Working context for resolving "that page", "same window", "wahi file" in the next turn."""
    ctx: dict = {}
    spec = state.get("spec") or {}
    plan = spec.get("plan") or []
    for r in state.get("results", []):
        if not r["ok"]:
            continue
        tool, data = r["tool"], r.get("data") or {}
        args = plan[r["step"]]["args"] if r["step"] < len(plan) else {}
        if tool.startswith("browser.") and data.get("url"):
            ctx.update(surface="browser", url=data["url"], page_title=data.get("title"))
        elif tool == "desktop.launch_app":
            ctx.update(surface="desktop", app=args.get("name"),
                       window=(data.get("windows") or [{}])[0].get("title") or args.get("name"))
        elif tool in ("desktop.click", "desktop.type", "desktop.read_window"):
            ctx.update(surface="desktop", window=data.get("window") or args.get("window"))
        elif tool.startswith("workspace.") and data.get("path"):
            ctx["file" if tool != "workspace.list" else "folder"] = data["path"]
    return ctx


def build_graph(checkpointer=None, store=None):
    g = StateGraph(GraphState)
    g.add_node("interpret", node_interpret)
    g.add_node("respond", node_respond)
    g.add_node("blocked", node_blocked)
    g.add_node("request_approval", node_request_approval)
    g.add_node("approval", node_approval)
    g.add_node("execute", node_execute)
    g.add_node("verify", node_verify)
    g.add_node("review", node_review)
    g.add_node("finish", node_finish)
    g.add_node("announce", node_announce)
    g.add_edge(START, "interpret")
    g.add_conditional_edges("interpret", route_after_interpret)
    g.add_edge("respond", "finish")
    g.add_edge("blocked", "finish")
    g.add_edge("request_approval", "approval")
    g.add_conditional_edges("approval", route_after_approval)
    g.add_conditional_edges("execute", route_after_execute)
    g.add_conditional_edges("verify", route_after_verify)
    g.add_conditional_edges("review", route_after_review)
    g.add_edge("finish", "announce")
    g.add_edge("announce", END)
    return g.compile(checkpointer=checkpointer, store=store)
