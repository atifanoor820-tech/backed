"""Turn an utterance into a validated TaskSpec.

Two interpreters share one finalisation step:
* rules: deterministic patterns for a documented command set (always available, offline);
* llm:   OpenAI-compatible chat model producing JSON, used when ABBAS_LLM_* is configured.
Risk is always computed from the tool registry, never taken from model output.
"""

import json
import logging
import re

import httpx
from pydantic import ValidationError

from agent.config import get_settings
from agent.dashi_config import is_dashi_active, parse_accounts_text
from agent.llm import gemini
from agent.memory import looks_secret
from agent.reflex.decide import yes_no
from agent.safety import SafetyError, overall_risk, resolve_in_workspace, step_needs_approval, workspace_roots
from agent.state import PlanStep, Risk, TaskSpec
from agent.tools import TOOLS, tool_catalog
from agent.tools.desktop import app_allowlist
from voice.language import detect_language, normalize, strip_wake_name

log = logging.getLogger("abbas.interpreter")

URL_RE = re.compile(r"https?://[^\s\"'<>]+|\b(?:www\.)?[a-z0-9-]+\.(?:com|org|net|io|dev|pk|co|app|ai)(?:/[^\s]*)?\b", re.I)
FILE_RE = r"([\w\-./\\]+\.[a-z0-9]{1,6})"
APP_ALIASES = {"calc": "calculator", "file explorer": "explorer", "explorer": "explorer", "notepad": "notepad",
               "calculator": "calculator", "antigravity": "antigravity", "anti gravity": "antigravity"}
OPEN_WORDS = r"\b(open|launch|start|run|kholo|khol do|khol|chalao|chala do)\b"
STOP_RE = re.compile(r"^(stop|cancel|ruko|ruk jao|bas karo|bas|band karo|stop it)\b", re.I)

CAPABILITIES = (
    "Without Gemini I understand a fixed set of commands: list/read/create workspace files, list windows, "
    "open allow-listed apps ('notepad kholo'), click or type in an app ('click \"7\" in Calculator', "
    "'type \"hello\" in Notepad'), open or inspect a website, click on the open page ('click \"More information\"'), "
    "take a screenshot, and check coding tools. Set ABBAS_GEMINI_API_KEY for open-ended requests."
)


def is_stop_command(text: str) -> bool:
    return bool(STOP_RE.match(normalize(strip_wake_name(text))))


def _risk_reflex(task_id: str | None, spec: TaskSpec, step: PlanStep) -> None:
    """Ask the reflex layer if this concrete action is hard to undo. It can only raise risk."""
    p, engine = yes_no(
        task_id,
        {"user_request": spec.original_request, "action": step.tool, "arguments": step.args},
        "Could this action delete data, send or publish something, make a purchase, submit a form, "
        "or otherwise be hard to undo?",
        purpose="risk",
    )
    if p is None:
        if step.risk == Risk.low:
            step.risk = Risk.medium  # nobody could judge it: stay conservative
    elif p >= get_settings().jev_risk_threshold:
        step.risk = Risk.high


def finalize_spec(spec: TaskSpec, task_id: str | None = None) -> TaskSpec:
    """Validate tools and arguments, then set risk and approval from the registry, reflexes and autonomy."""
    valid: list[PlanStep] = []
    for step in spec.plan:
        tool = TOOLS.get(step.tool)
        if tool is None:
            spec.clarification_question = f"I planned an unknown tool '{step.tool}', so I stopped before executing."
            valid = []
            break
        try:
            step.risk = tool.risk_for(step.args)
            step.args = tool.input_model.model_validate(step.args).model_dump(mode="json")
        except ValidationError as e:
            spec.clarification_question = f"I could not build valid arguments for {step.tool}: {e.errors()[0]['msg']}"
            valid = []
            break
        if tool.risk_reflex:
            _risk_reflex(task_id, spec, step)
        valid.append(step)
    spec.plan = valid
    spec.required_tools = sorted({s.tool for s in valid})
    spec.risk_level = overall_risk(valid)
    risky = [s for s in valid if step_needs_approval(s)]
    spec.requires_approval = bool(risky)
    spec.approval_reason = "; ".join(f"{s.description} ({s.risk.value} risk)" for s in risky) or None
    return spec


SAVE_ACCOUNTS_RE = re.compile(r"\b(save|store|container|settings?|vault|yaad|rakh|rakho|rakhlo|rakh lo|list mein|mein daal)\b", re.I)
ACCOUNT_ACTION_RE = re.compile(r"\b(update|refresh|re-?add|delete|delte|remove|hata|hatao|badlo|badal)\b", re.I)


def _dashi_save_spec(request: str, norm: str, base: dict) -> TaskSpec | None:
    """'ye accounts save karo: 5922 / pass / FxPro-MT5 Demo ...' -> dashi.save_accounts. Deterministic on purpose:
    credentials are parsed in code, never by a model, and a save never touches ShowMyTrades."""
    accounts = [a for a in parse_accounts_text(request) if a.get("password") or a.get("server")]
    if not accounts:
        return None
    if ACCOUNT_ACTION_RE.search(norm) and not re.search(r"\b(save|store|container|settings?|vault)\b", norm):
        return None  # "update 5922 password x" is an update that also saves the credentials
    wants_save = bool(SAVE_ACCOUNTS_RE.search(norm))
    many = len(accounts) > 1 and not ACCOUNT_ACTION_RE.search(norm) and not re.search(r"\b(add|connect)\b", norm)
    if not (wants_save or many):
        return None
    numbers = ", ".join(a["account_number"] for a in accounts)
    return TaskSpec(**{**base, "interpreter": "rules"},
                    interpreted_objective=f"save {len(accounts)} trading account(s) to Dashi settings",
                    required_outcome="accounts listed in Settings → Dashi",
                    plan=[PlanStep(tool="dashi.save_accounts", args={"text": request},
                                   description=f"Save trading account(s) {numbers} in Settings → Dashi")])


REMEMBER_RE = re.compile(r"^(?:please\s+)?(?:remember|yaad rakho|yaad rakhna|yaad kar lo)\s*(?:that|ke|ki|k)?\s*[:,-]?\s*(.+)$", re.I)
REMEMBER_TAIL_RE = re.compile(r"^(.+?)\s*[,-]?\s*(?:ye|yeh|is ko|isko)?\s*yaad rakh(?:o|na)$", re.I)
FORGET_RE = re.compile(r"^(?:forget|bhool jao|bhul jao|bhool jaao|yaad se hata do)\s*(?:that|ke|ki|about)?\s*[:,-]?\s*(.+)$", re.I)
RECALL_RE = re.compile(r"\b(what do you remember|what do you know about me|kya yaad hai|tumhe kya yaad|aap ko kya yaad)\b", re.I)


def _rules(request: str, norm: str, language: str, context: dict | None = None,
           memories: list[dict] | None = None) -> TaskSpec:
    base = dict(original_request=request, normalized_request=norm, language=language, interpreter="rules")
    context, memories = context or {}, memories or []
    urdu = language != "en"

    if spec := _dashi_save_spec(request, norm, base):
        return spec

    # Long-term memory commands
    if m := (REMEMBER_RE.match(request.strip()) or REMEMBER_TAIL_RE.match(request.strip().rstrip(".!"))):
        fact = m.group(1).strip().rstrip(".!")
        if looks_secret(fact):
            return TaskSpec(**base, interpreted_objective="refuse to store a secret",
                            reply=("Main passwords, keys ya card numbers yaad nahi rakhta. Inhe password manager mein rakhein."
                                   if urdu else "I don't store passwords, keys or card numbers. Keep them in a password manager."))
        return TaskSpec(**base, interpreted_objective="remember a fact", remember=[fact],
                        reply=(f"Theek hai, yaad kar liya: {fact}" if urdu else f"Got it, I'll remember: {fact}"))
    if m := FORGET_RE.match(request.strip()):
        what = m.group(1).strip().rstrip(".!")
        return TaskSpec(**base, interpreted_objective="forget a fact", forget=[what],
                        reply=(f"Theek hai, bhool gaya: {what}" if urdu else f"Okay, I've forgotten: {what}"))
    if RECALL_RE.search(norm):
        listing = "\n".join(f"• {m['text']}" for m in memories)
        empty = "Abhi mujhe aap ke baare mein kuch yaad nahi." if urdu else "I don't have anything saved about you yet."
        return TaskSpec(**base, interpreted_objective="list memories",
                        reply=(("Mujhe ye yaad hai:\n" if urdu else "Here's what I remember:\n") + listing) if memories else empty)

    # "read that file again" / "wahi file parho"
    if context.get("file") and re.search(r"\b(wahi|wohi|usi|us|that|same|it|again|dobara)\b", norm) and \
            re.search(r"\b(read|parho|dikhao|show)\b", norm) and not re.search(FILE_RE, request, re.I):
        return TaskSpec(**base, interpreted_objective="read the last file again", target=context["file"],
                        plan=[PlanStep(tool="workspace.read", args={"path": context["file"]},
                                       description="Read the same file again")])

    if re.fullmatch(r"(hello|hi|hey|salam|salaam|assalam(u|o)? ?alaikum|aoa)( there)?( abbas| abbad)?[\s!.,]*", norm):
        return TaskSpec(**base, interpreted_objective="greeting", reply="Salam! Abbas here. " + CAPABILITIES)
    if re.search(r"\b(help|what can you do|kya kar sakte)\b", norm):
        return TaskSpec(**base, interpreted_objective="describe capabilities", reply=CAPABILITIES)

    if re.search(r"\b(codex|antigravity|coding tools?)\b", norm) and re.search(
            r"\b(check|installed|available|hai|hain|kya)\b", norm) and not re.search(OPEN_WORDS, norm):
        return TaskSpec(**base, interpreted_objective="check coding executors", required_outcome="list of available tools",
                        plan=[PlanStep(tool="coding.check_executors", description="Check installed coding tools")])

    if re.search(r"\b(screenshot|screen ?shot|screen capture)\b", norm):
        return TaskSpec(**base, interpreted_objective="capture the screen", required_outcome="screenshot",
                        plan=[PlanStep(tool="screen.screenshot", description="Capture the screen")])

    # click "X" in <window>  /  <window> mein "X" click karo|dabao
    quoted = re.search(r"[\"“]([^\"”]{1,200})[\"”]", request)
    window = None
    if quoted:
        before, after = request[: quoted.start()], request[quoted.end():]
        # English: ... "X" in Calculator   |   Roman Urdu: Calculator mein "X" ...
        if m := re.search(r"\b(?:in|on|inside)\s+(?:the\s+)?([\w .-]{2,40}?)(?:\s+(?:window|app))?\s*[.?!]*\s*$", after, re.I):
            window = m.group(1).strip()
        elif m := re.search(r"([\w.-]+(?:\s[\w.-]+)?)\s+(?:mein|me|main)\s*$", before, re.I):
            window = m.group(1).strip()
        if window and re.fullmatch(r"(the\s+)?(page|website|site|browser)", window, re.I):
            window = None
    else:
        # Spoken commands have no quotes: "click Seven in Calculator", "Calculator mein Plus dabao".
        text = request.strip().rstrip(".?!")
        if m := re.match(r"(?:click|press|tap)\s+(?:on\s+)?(?:the\s+)?(.+?)\s+(?:button\s+)?(?:in|on)\s+(?:the\s+)?"
                         r"([\w.-]+(?:\s[\w.-]+)?)(?:\s+(?:window|app))?$", text, re.I) \
                or re.match(r"(?:type|write|enter)\s+(.+?)\s+(?:in|into)\s+(?:the\s+)?([\w.-]+(?:\s[\w.-]+)?)"
                            r"(?:\s+(?:window|app))?$", text, re.I):
            quoted, window = m, m.group(2).strip()
        elif m := re.match(r"([\w.-]+(?:\s[\w.-]+)?)\s+(?:mein|me|main)\s+(.+?)\s+"
                           r"(?:dabao|daba do|click karo|press karo|likho|likh do|type karo)$", text, re.I):
            quoted, window = re.match(r"(.*)", m.group(2)), m.group(1).strip()
        if window and re.fullmatch(r"(the\s+)?(page|website|site|browser)", window, re.I):
            quoted, window = None, None
        # Follow-ups that name no window: "click Plus", "Equals dabao", "salam likho".
        if not quoted and (m := re.match(r"(?:click|press|tap|type|write)\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+button)?$", text, re.I)
                           or re.match(r"(.+?)\s+(?:button\s+)?(?:dabao|daba do|par click karo|click karo|press karo|"
                                       r"likho|likh do|type karo)$", text, re.I)):
            quoted = m
    # A follow-up with no window acts on the desktop window used last in this conversation.
    if quoted and not window and context.get("surface") == "desktop" and context.get("window"):
        window = context["window"]
    if quoted and re.search(r"\b(click|press|tap|dabao|daba do|click karo)\b", norm):
        target = quoted.group(1)
        if window:
            return TaskSpec(**base, interpreted_objective=f"click '{target}' in {window}", target=window,
                            plan=[PlanStep(tool="desktop.click", args={"window": window, "target": target},
                                           description=f"Click '{target}' in {window}")])
        return TaskSpec(**base, interpreted_objective=f"click '{target}' on the open page",
                        plan=[PlanStep(tool="browser.click", args={"target": target}, description=f"Click '{target}' on the page")])
    if quoted and window and re.search(r"\b(type|write|enter|likho|likh do|type karo)\b", norm):
        return TaskSpec(**base, interpreted_objective=f"type text into {window}", target=window,
                        plan=[PlanStep(tool="desktop.type", args={"window": window, "text": quoted.group(1)},
                                       description=f"Type into {window}")])

    if m := URL_RE.search(request):
        url = m.group(0).rstrip(".,)")
        if not url.startswith("http"):
            url = "https://" + url
        if re.search(r"\b(open|go to|visit|kholo|khol do|jao|chalao)\b", norm) and not re.search(r"\b(check|inspect|issues?)\b", norm):
            return TaskSpec(**base, interpreted_objective=f"open {url} in the browser", target=url,
                            required_outcome="page open in Abbas's browser",
                            plan=[PlanStep(tool="browser.open", args={"url": url}, description=f"Open {url}")])
        return TaskSpec(**base, interpreted_objective=f"inspect website {url}", target=url,
                        required_outcome="page summary, detected issues and a screenshot",
                        plan=[PlanStep(tool="browser.inspect", args={"url": url}, description=f"Inspect {url} in a browser")])

    if re.search(r"\bwindows?\b", norm) and re.search(r"\b(list|show|which|open|dikhao|kaun|konsi|kon si|batao)\b", norm):
        return TaskSpec(**base, interpreted_objective="list open windows", required_outcome="window titles",
                        plan=[PlanStep(tool="desktop.list_windows", description="List visible windows")])

    # Create / write file
    write = re.search(rf"\b(?:create|make|write|new|banao|bana do|bana)\b.*?\bfile\b\s*(?:named|called|name)?\s*{FILE_RE}"
                      r"|" + rf"{FILE_RE}\s+(?:naam ki\s+)?file\s+(?:banao|bana do|create)", request, re.I)
    if write:
        path = write.group(1) or write.group(2)
        content_m = re.search(r"(?:with|containing|content|likho|text)\s*(?:text|content)?\s*[:\-]?\s*[\"“](.+?)[\"”]", request, re.I) \
            or re.search(r"(?:with|containing|content|likho)\s*(?:text|content)?\s*[:\-]\s*(.+)$", request, re.I)
        if not content_m:
            return TaskSpec(**base, interpreted_objective=f"create file {path}", target=path,
                            clarification_question=f'What should {path} contain? e.g. create file {path} with "hello"')
        try:
            exists = resolve_in_workspace(path).exists()
        except SafetyError:
            exists = False  # the tool will refuse the path at execution time
        return TaskSpec(**base, interpreted_objective=f"{'overwrite' if exists else 'create'} file {path}", target=path,
                        required_outcome=f"{path} exists with the given content",
                        plan=[PlanStep(tool="workspace.write",
                                       args={"path": path, "content": content_m.group(1), "overwrite": exists},
                                       description=f"{'Overwrite existing' if exists else 'Write'} {path} in the workspace")])

    if (m := re.search(FILE_RE, request, re.I)) and re.search(r"\b(read|show|open|parho|dikhao|content|kya likha)\b", norm):
        path = m.group(1)
        if not re.search(r"\.(exe|com|bat|cmd)$", path, re.I):
            return TaskSpec(**base, interpreted_objective=f"read file {path}", target=path, required_outcome="file content",
                            plan=[PlanStep(tool="workspace.read", args={"path": path}, description=f"Read {path}")])

    if re.search(r"\bfiles?\b", norm) and re.search(r"\b(list|show|dikhao|kaun|konsi|batao|what)\b", norm):
        return TaskSpec(**base, interpreted_objective="list workspace files", required_outcome="file listing",
                        plan=[PlanStep(tool="workspace.list", description="List workspace files")])

    # Dashi / ShowMyTrades commands
    if re.search(r"\b(dashi|switch to dashi|switch dashi|dashi on|dashi agent)\b", norm) and re.search(r"\b(on|switch|activate|karo|chalao|start)\b", norm):
        return TaskSpec(**base, interpreted_objective="switch active agent to Dashi",
                        plan=[PlanStep(tool="dashi.switch_agent", args={"agent_name": "dashi"},
                                       description="Switch active agent to Dashi (ShowMyTrades Agent)")])

    if re.search(r"\b(switch to abbas|switch abbas|abbas on|abbas agent)\b", norm) and re.search(r"\b(on|switch|activate|karo|chalao|start)\b", norm):
        return TaskSpec(**base, interpreted_objective="switch active agent to Abbas",
                        plan=[PlanStep(tool="dashi.switch_agent", args={"agent_name": "abbas"},
                                       description="Switch active agent to Abbas")])

    dashi_mode = is_dashi_active()
    has_trade_keyword = bool(re.search(r"\b(account|acc|trading|showmytrades|show my trades|dashi|three dots?|mt5)\b", norm))
    acc_m = re.search(r"\b(\d{5,12})\b", request)
    acc_num = acc_m.group(1) if acc_m else ""
    given = next((a for a in parse_accounts_text(request) if a["account_number"] == acc_num), {}) if acc_num else {}
    # Credentials typed with the command ("update 5922 password x server y"); a lone stray word is not a password.
    creds = {k: given[k] for k in ("password", "server", "broker") if given.get(k)} \
        if given.get("server") or re.search(r"\b(password|pass|pwd|pw)\b", norm) else {}

    is_delete = bool(re.search(r"\b(delete|delte|detle|remove|hata|hatao|khatam)\b", norm))
    is_add = bool(re.search(r"\b(add|daal|daalo|connect|insert|naya account)\b", norm))
    is_update = bool(re.search(r"\b(update|refresh|re-?add|dobara add|badal|badlo)\b", norm))

    # Several account numbers and no credentials typed: one step per account.
    all_nums = list(dict.fromkeys(re.findall(r"\b(\d{5,12})\b", request)))
    no_creds = "/" not in request and not re.search(r"\b(password|pass|pwd|pw|server)\b", norm)
    is_check = bool(re.search(r"\b(check|chek|verify|status|jaanch)\b", norm))

    if (dashi_mode or has_trade_keyword) and is_check and not (is_delete or is_add or is_update):
        nums = all_nums if no_creds else []
        what = ", ".join(nums) if nums else "every saved account"
        return TaskSpec(**base, interpreted_objective=f"check {what} on ShowMyTrades",
                        plan=[PlanStep(tool="dashi.check_accounts", args={"account_numbers": nums},
                                       description=f"Check {what} on ShowMyTrades one by one (read-only)")])

    if (dashi_mode or has_trade_keyword or acc_m) and len(all_nums) > 1 and no_creds and is_update and not is_delete:
        return TaskSpec(**base, interpreted_objective=f"update {len(all_nums)} trading accounts on ShowMyTrades",
                        plan=[PlanStep(tool="dashi.update_account", args={"account_number": n},
                                       description=f"Find account {n}, delete it and re-add it, then set Published")
                              for n in all_nums])

    if (dashi_mode or has_trade_keyword or acc_m):
        # 1. Action: DELETE ONLY
        if is_delete and not is_update:
            return TaskSpec(**base, interpreted_objective=f"delete trading account {acc_num or 'target'} from ShowMyTrades",
                            plan=[PlanStep(tool="dashi.delete_account", args={"account_number": acc_num},
                                           description=f"Scan ShowMyTrades profiles, find account {acc_num or 'target'} and permanently delete it")])

        # 2. Action: ADD ONLY
        if is_add and not is_update and not is_delete:
            return TaskSpec(**base, interpreted_objective=f"add trading account {acc_num or 'new'} to ShowMyTrades",
                            plan=[PlanStep(tool="dashi.add_account", args={"account_number": acc_num, **creds},
                                           description=f"Add MT5 Direct Connection for account {acc_num or 'new'} to ShowMyTrades")])

        # 3. Action: UPDATE (DELETE OLD + RE-ADD FRESH)
        if is_update or (acc_m and not is_delete and not is_add):
            return TaskSpec(**base, interpreted_objective=f"update trading account {acc_num or 'target'} on ShowMyTrades across profiles",
                            plan=[PlanStep(tool="dashi.update_account", args={"account_number": acc_num, **creds},
                                           description=f"Scan ShowMyTrades profiles, find account {acc_num or 'target'}, delete and re-add with fresh credentials")])

    if re.search(r"\b(showmytrades|show my trades|dashi)\b", norm) and re.search(r"\b(profiles?|accounts?|status|list|dikhao|batao)\b", norm):
        return TaskSpec(**base, interpreted_objective="list ShowMyTrades profiles and accounts",
                        plan=[PlanStep(tool="dashi.list_profiles", args={}, description="List configured ShowMyTrades profiles")])

    if re.search(OPEN_WORDS, norm):
        known = {**APP_ALIASES, **{k: k for k in app_allowlist()}}
        for alias in sorted(known, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", norm):
                app = known[alias]
                return TaskSpec(**base, interpreted_objective=f"open {app}", target=app,
                                required_outcome=f"{app} window visible",
                                plan=[PlanStep(tool="desktop.launch_app", args={"name": app}, description=f"Launch {app}")])
        # Any other app name: "open Spotify", "chrome kholo", "VS Code chalao"
        text = request.strip().rstrip(".!?")
        m = (re.match(r"(?:please\s+)?(?:open|launch|start|run)\s+(?:the\s+)?(.{2,40}?)(?:\s+app)?$", text, re.I)
             or re.match(r"(.{2,40}?)\s+(?:app\s+)?(?:kholo|khol do|chalao|chala do|open karo|start karo)$", text, re.I))
        if m and not re.search(r"\b(file|folder|page|website|link|tab)\b", m.group(1), re.I):
            app = m.group(1).strip()
            return TaskSpec(**base, interpreted_objective=f"open {app}", target=app, required_outcome=f"{app} window visible",
                            plan=[PlanStep(tool="desktop.launch_app", args={"name": app}, description=f"Launch {app}")])

    return TaskSpec(**base, interpreted_objective="unrecognised request",
                    clarification_question="I understood your words but this needs open-ended planning. " + CAPABILITIES)


_SYSTEM = """You are Abbas (nickname Abbad), a Windows desktop agent, with a specialized trading mode named Dashi.
Convert the user's request (English, Urdu or Roman Urdu) into JSON with keys: interpreted_objective (string), target (string|null),
required_outcome (string), constraints (string[]), plan (array of {tool, args, description}), reply (string|null; only for pure
conversation with an empty plan), clarification_question (string|null; set when ambiguity materially changes the action or no
tool fits). Use ONLY these tools, with args matching input_schema:
%s
Choosing the action layer:
- ShowMyTrades & Trading automation: dashi.save_accounts (user gives account numbers/passwords/servers to KEEP; pass
  the user's whole message as `text`; this is the one place credentials may be stored), dashi.delete_account (use when user asks to DELETE only: scans profiles, locates account, deletes it without adding), dashi.add_account (use when user asks to ADD only: adds MetaTrader 5 Direct Connection with credentials without deleting), dashi.update_account (use when user asks to UPDATE/re-add: deletes old connection and adds fresh one), dashi.check_accounts (READ-ONLY: look up saved accounts on ShowMyTrades one by one and report Published/Private), dashi.list_profiles, dashi.switch_agent.
- Web pages: browser.open, then browser.click / browser.type / browser.read (Playwright).
- Desktop apps: desktop.launch_app, then desktop.click / desktop.type / desktop.read_window (UI Automation).
- screen.click / screen.type / screen.hotkey (PyAutoGUI) ONLY when the target cannot be reached the ways above.
Describe click/type targets in plain English words a human would use ("the 7 button", "search box"); a separate
fast model picks the exact element, so never guess ids, coordinates or selectors.
Never invent tools or shell commands. Keep plans short (at most %d steps). Write reply/clarification in the user's
language; write descriptions and targets in English.

Bigger goals (build a website/app, research then act, several apps in a row):
- Set "multi_round": true and plan only the FIRST few concrete steps (up to 6). After they run, a reviewer sees the
  results and plans the next steps until the whole goal is done - so never squeeze a big goal into one step, and
  never treat opening an app as finishing the goal.
- Allowed folders are listed in the context; paths outside them will fail, so ask the user to add the folder in
  Settings -> Folders if they name one that is not allowed.
- Building a Next.js site: (1) files.make_folder for the parent folder; (2) terminal.run in that parent folder:
  npx create-next-app@latest <name> --ts --tailwind --eslint --app --src-dir --import-alias "@/*" --use-npm --yes
  (name: lowercase-with-dashes; if that folder already exists, pick a new name); (3) research.brief with `urls` set
  to the homepages of 3-5 real, well-known competitors you know, and save_to "<project>/docs/research.md"
  (research AFTER creating the project, so the brief doesn't block create-next-app); (4) ide.open the project if the user mentions
  Antigravity or wants to watch; (5) code.write in several focused steps (theme+layout, then sections/pages) with
  context_files ["docs/research.md"]; (6) terminal.run "npm run build" in the project to verify; fix errors with
  code.write; (7) optionally terminal.run "npm run dev" then browser.open the printed localhost URL to look.

Memory:
- You receive the recent conversation, a working context (the page, window and file used last) and long-term
  memories. Use them to resolve references like "us page par", "that file", "same window", "wahi", "dobara".
  If a follow-up names no window, use the context window; if the reference is unclear, ask.
- Answer questions about earlier turns or saved memories with reply.
- Add "remember": [short facts in English] ONLY when the user asks you to remember something or clearly states a
  lasting personal fact or preference (name, their website, preferred language). Never store secrets, passwords
  or keys - refuse those in reply.
- Add "forget": [memory ids] when the user asks you to forget something.
Respond with JSON only."""

_SPEC_KEYS = ("interpreted_objective", "target", "required_outcome", "constraints", "plan", "reply",
              "clarification_question", "remember", "forget", "multi_round")


def _spec_from_json(raw: dict, request: str, norm: str, language: str) -> TaskSpec:
    raw["plan"] = [{k: v for k, v in step.items() if k in ("tool", "args", "description")}
                   for step in raw.get("plan") or [] if isinstance(step, dict)]
    raw.setdefault("interpreted_objective", request)
    return TaskSpec(original_request=request, normalized_request=norm, language=language, interpreter="llm",
                    **{k: raw[k] for k in _SPEC_KEYS if raw.get(k) is not None})


def _memory_prompt(history: list[dict], context: dict, memories: list[dict]) -> str:
    turns = "\n".join(f"{t['role']}: {t['content'][:600]}" for t in history) or "(new conversation)"
    mems = "\n".join(f"[{m['id']}] {m['text']}" for m in memories) or "(none)"
    folders = "; ".join(str(p) for p in workspace_roots())
    return (f"Recent conversation:\n{turns}\n\nWorking context: {json.dumps(context)}\n\n"
            f"Long-term memories:\n{mems}\n\nAllowed folders: {folders}\n\n")


def _gemini(request: str, norm: str, language: str, memory: str = "") -> TaskSpec:
    system = _SYSTEM % (json.dumps(tool_catalog()), get_settings().max_steps)
    raw = gemini.generate_json(system, f"{memory}Detected language: {language}\nRequest: {request}")
    return _spec_from_json(raw, request, norm, language)


def _llm(request: str, norm: str, language: str, memory: str = "") -> TaskSpec:
    s = get_settings()
    resp = httpx.post(
        f"{s.llm_base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {s.llm_api_key}"},
        json={
            "model": s.llm_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM % (json.dumps(tool_catalog()), s.max_steps)},
                {"role": "user", "content": f"{memory}Detected language: {language}\nRequest: {request}"},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return _spec_from_json(json.loads(resp.json()["choices"][0]["message"]["content"]), request, norm, language)


FAST_MAX_WORDS = 7
COMPOUND_RE = re.compile(r"\b(aur|and|then|phir|uske|after|also|bhi|but|lekin)\b|[,;]", re.I)


def _fast_path(cleaned: str, norm: str, language: str, context: dict, memories: list[dict]) -> TaskSpec | None:
    """Short, single, unambiguous commands ("files dikhao", "notepad kholo", "Plus dabao") are planned by the rule
    engine in ~0 ms instead of a 3-5 s model call. Anything compound, longer, or conversational goes to the model."""
    if len(cleaned.split()) > FAST_MAX_WORDS or COMPOUND_RE.search(cleaned):
        return None
    spec = _rules(cleaned, norm, language, context, memories)
    if not spec.plan or spec.clarification_question or len(spec.plan) != 1:
        return None
    step = spec.plan[0]
    if step.tool == "browser.inspect" and not re.search(r"\b(check|inspect|issues?|test|audit|jaanch)\b", norm):
        return None  # "example.com par cats search karo" contains a URL but is not an inspection
    if step.tool == "desktop.launch_app":
        # Only names we know are apps; "open youtube" probably means the website, which the model handles better.
        known = {*APP_ALIASES.values(), *app_allowlist()}
        if str(step.args.get("name", "")).lower() not in known:
            return None
    return spec


DASHI_HELP = ("I'm Dashi. I only handle ShowMyTrades trading accounts: save accounts "
              "('592235145 / password / server save karo'), update ('592235145 update karo', or several numbers), "
              "add, delete, and check ('saved accounts check karo').")


def _dashi_only(spec: TaskSpec) -> TaskSpec:
    """With ABBAS_DASHI_ONLY, anything that isn't a Dashi step becomes a short help reply instead of running."""
    if not get_settings().dashi_only:
        return spec
    if spec.plan and all(st.tool.startswith("dashi.") for st in spec.plan):
        return spec
    return spec.model_copy(update={"plan": [], "clarification_question": DASHI_HELP,
                                   "interpreted_objective": "not a Dashi request"})


def interpret(request: str, task_id: str | None = None, history: list[dict] | None = None,
              context: dict | None = None, memories: list[dict] | None = None) -> tuple[TaskSpec, str | None]:
    """Returns (spec, warning). The warning explains a fallback to the rule engine, if any."""
    cleaned = strip_wake_name(request) or request
    language = detect_language(cleaned).language
    norm = normalize(cleaned)
    s = get_settings()
    warning = None
    if save := _dashi_save_spec(cleaned, norm, dict(original_request=request, normalized_request=norm, language=language)):
        return finalize_spec(save, task_id), None
    if s.llm_enabled and (fast := _fast_path(cleaned, norm, language, context or {}, memories or [])):
        fast.original_request = request
        return finalize_spec(_dashi_only(fast), task_id), None
    if s.llm_enabled:
        name = "Gemini" if s.gemini_enabled else "LLM"
        try:
            memory = _memory_prompt(history or [], context or {}, memories or [])
            spec = (_gemini if s.gemini_enabled else _llm)(cleaned, norm, language, memory)
            spec.original_request = request
            return finalize_spec(_dashi_only(spec), task_id), None
        except Exception as e:  # network, auth, bad JSON, schema mismatch
            log.warning("%s interpretation failed: %s", name, e)
            warning = f"{name} planning failed ({type(e).__name__}: {str(e)[:150]}); used the rule engine instead."
    spec = _rules(cleaned, norm, language, context, memories)
    spec.original_request = request
    return finalize_spec(_dashi_only(spec), task_id), warning


__all__ = ["interpret", "is_stop_command", "Risk"]
