"""Goal loop (plan -> act -> review), developer tools and web research parsing."""

import copy
import time
from pathlib import Path

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agent import runtime
from agent.config import get_settings
from agent.llm import gemini
from agent.runner import TaskRunner
from agent.safety import SafetyError, resolve_in_workspace
from agent.tools import TOOLS, dev, web
from agent.tools.registry import ToolContext
from database.models import Task
from database.session import session_scope


@pytest.fixture(autouse=True)
def clean_settings():
    path = get_settings().data_dir / "settings.json"
    path.unlink(missing_ok=True)
    yield
    path.unlink(missing_ok=True)


def wait(task_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with session_scope() as s:
            t = s.get(Task, task_id)
            if t and t.status in ("completed", "failed", "blocked", "cancelled", "awaiting_approval"):
                return t
        time.sleep(0.05)
    raise AssertionError("timeout")


def scripted_gemini(monkeypatch, planner: dict, reviews: list[dict]):
    """Planner call returns `planner`; each review call returns the next item in `reviews`."""
    calls = {"review": 0}

    def fake(system, user, **kw):
        if "progress reviewer" in system:
            calls["review"] += 1
            return copy.deepcopy(reviews[min(calls["review"], len(reviews)) - 1])  # a real model answers afresh
        return copy.deepcopy(planner)

    monkeypatch.setattr(gemini, "generate_json", fake)
    return calls


@pytest.fixture
def runner(settings):
    settings(gemini_api_key="g", jev_api_key="")
    runtime.update(autonomy="full")
    return TaskRunner(InMemorySaver(), InMemoryStore())


def test_goal_runs_several_rounds_until_reviewer_confirms(runner, monkeypatch, workspace):
    calls = scripted_gemini(monkeypatch, {
        "interpreted_objective": "make two files", "multi_round": True,
        "plan": [{"tool": "workspace.write", "args": {"path": "loop/a.txt", "content": "A"}, "description": "write a"}],
    }, [
        {"done": False, "summary": "a done", "remaining": ["b"],
         "next_plan": [{"tool": "workspace.write", "args": {"path": "loop/b.txt", "content": "B"}, "description": "write b"}]},
        {"done": True, "summary": "Both files are written in loop/.", "remaining": [], "next_plan": []},
    ])
    t = wait(runner.create("do the two-file goal")[0])
    assert t.status == "completed", t.result
    assert (workspace / "loop/a.txt").read_text() == "A" and (workspace / "loop/b.txt").read_text() == "B"
    assert t.result.startswith("Both files are written") and calls["review"] == 2
    assert [p["tool"] for p in t.spec["plan"]] == ["workspace.write", "workspace.write"]


def test_failed_step_is_repaired_instead_of_ending(runner, monkeypatch, workspace):
    scripted_gemini(monkeypatch, {
        "interpreted_objective": "read note", "multi_round": False,
        "plan": [{"tool": "workspace.read", "args": {"path": "fixme.txt"}, "description": "read"}],
    }, [
        {"done": False, "summary": "file missing; creating it", "next_plan": [
            {"tool": "workspace.write", "args": {"path": "fixme.txt", "content": "hi"}, "description": "create"},
            {"tool": "workspace.read", "args": {"path": "fixme.txt"}, "description": "read again"}]},
        {"done": True, "summary": "Created and read fixme.txt.", "next_plan": []},
    ])
    t = wait(runner.create("read fixme.txt")[0])
    assert t.status == "completed" and (workspace / "fixme.txt").read_text() == "hi"


def test_round_limit_is_reported_as_incomplete(runner, monkeypatch, settings):
    settings(max_rounds=2)
    n = {"i": 0}

    def step():
        n["i"] += 1
        return {"tool": "workspace.write", "args": {"path": f"lim/{n['i']}.txt", "content": "x"}, "description": "w"}

    monkeypatch.setattr(gemini, "generate_json", lambda system, user, **kw: (
        {"done": False, "summary": "more", "next_plan": [step()]} if "progress reviewer" in system
        else {"interpreted_objective": "endless", "multi_round": True, "plan": [step()]}))
    t = wait(runner.create("never ending goal")[0])
    assert t.status == "failed" and "safety limit" in t.result


def test_goal_continues_even_when_planner_forgets_multi_round(runner, monkeypatch, workspace):
    """Regression: Gemini planned 1 step of a 5-part goal with multi_round=false and Abbas said 'Done'."""
    calls = scripted_gemini(monkeypatch, {
        "interpreted_objective": "folder then file", "multi_round": False,
        "plan": [{"tool": "files.make_folder", "args": {"path": str(workspace / "goal")}, "description": "folder"}],
    }, [
        {"done": False, "summary": "folder made", "next_plan": [
            {"tool": "workspace.write", "args": {"path": "goal/app.txt", "content": "built"}, "description": "build"}]},
        {"done": True, "summary": "Folder and app created.", "next_plan": []},
    ])
    t = wait(runner.create("make a folder and build the app in it")[0])
    assert t.status == "completed" and (workspace / "goal/app.txt").exists() and calls["review"] == 2


def test_invalid_reviewer_plan_gets_one_correction(runner, monkeypatch, workspace):
    """Regression: the reviewer proposed terminal.run without cwd and the whole goal failed."""
    calls = scripted_gemini(monkeypatch, {
        "interpreted_objective": "folder then list", "plan": [
            {"tool": "files.make_folder", "args": {"path": str(workspace / "fixplan")}, "description": "folder"}],
    }, [
        {"done": False, "summary": "next", "next_plan": [
            {"tool": "terminal.run", "args": {"command": "node --version"}, "description": "missing cwd"}]},
        {"done": False, "summary": "corrected", "next_plan": [
            {"tool": "terminal.run", "args": {"command": "node --version", "cwd": str(workspace / "fixplan")},
             "description": "with cwd"}]},
        {"done": True, "summary": "Folder made and node checked.", "next_plan": []},
    ])
    t = wait(runner.create("make folder and check node")[0], timeout=60)
    assert t.status == "completed", t.result
    assert calls["review"] == 3 and t.spec["plan"][-1]["args"]["cwd"].endswith("fixplan")


def test_reviewer_question_blocks(runner, monkeypatch):
    scripted_gemini(monkeypatch, {"interpreted_objective": "x", "multi_round": True,
                                  "plan": [{"tool": "workspace.list", "args": {}, "description": "look"}]},
                    [{"done": False, "summary": "", "question": "Which folder should I use?", "next_plan": []}])
    t = wait(runner.create("build something")[0])
    assert t.status == "blocked" and t.result.startswith("Which folder")


# --- folders ------------------------------------------------------------------------------------

def test_extra_folders_validation_and_access(tmp_path):
    with pytest.raises(ValueError):
        runtime.update(extra_folders=["C:\\Windows\\Temp"])
    with pytest.raises(ValueError):
        runtime.update(extra_folders=["C:\\"])
    with pytest.raises(ValueError):
        runtime.update(extra_folders=["relative\\path"])
    target = tmp_path / "abbas work"
    with pytest.raises(SafetyError):
        resolve_in_workspace(str(target / "site"))
    runtime.update(extra_folders=[str(target)])
    assert resolve_in_workspace(str(target / "site")) == (target / "site").resolve()
    r = TOOLS["files.make_folder"].run(dev.FolderArgs(path=str(target)), ToolContext(None, 0))
    assert r.ok and target.is_dir()


# --- terminal -----------------------------------------------------------------------------------

def test_command_parsing_rules():
    argv = dev.parse_command('npx create-next-app@latest my-site --ts --import-alias "@/*" --yes')
    assert Path(argv[0]).stem.lower() == "npx" and "@/*" in argv
    for bad in ["npm install && del /s *", "powershell -c whoami", "git push origin main", "npm publish",
                "node a.js > out.txt", "git reset --hard HEAD", "cmd /c dir"]:
        with pytest.raises(SafetyError):
            dev.parse_command(bad)


def test_launched_programs_do_not_inherit_server_port(monkeypatch, workspace):
    from agent.config import child_env

    monkeypatch.setenv("PORT", "8000")
    monkeypatch.setenv("ABBAS_GEMINI_API_KEY", "secret")
    env = child_env(CI="1")
    assert "PORT" not in env and "ABBAS_GEMINI_API_KEY" not in env and env["CI"] == "1" and "PATH" in env
    r = TOOLS["terminal.run"].run(dev.RunArgs(command="node -e console.log(process.env.PORT)", cwd=str(workspace)),
                                  ToolContext(None, 0))
    assert r.ok and r.data["output_tail"].strip() == "undefined"


def test_terminal_runs_and_reports_exit_codes(workspace):
    ok = TOOLS["terminal.run"].run(dev.RunArgs(command="node --version", cwd=str(workspace)), ToolContext(None, 0))
    assert ok.ok and ok.data["output_tail"].startswith("v")
    bad = TOOLS["terminal.run"].run(dev.RunArgs(command="node -e process.exit(3)", cwd=str(workspace)), ToolContext(None, 0))
    assert not bad.ok and bad.data["exit_code"] == 3


# --- code writing -------------------------------------------------------------------------------

def test_code_write_stays_inside_the_project(monkeypatch, workspace, settings):
    settings(gemini_api_key="g")
    proj = workspace / "proj"
    (proj / "docs").mkdir(parents=True, exist_ok=True)
    (proj / "package.json").write_text('{"name":"proj","dependencies":{"next":"15"}}')
    (proj / "docs/research.md").write_text("# Brief\nUse a dark theme.")
    seen = {}

    def fake(system, parts, **kw):
        seen["user"] = parts[0]["text"]
        return ("=== FILE: src/app/page.tsx ===\n"
                'export default function P() {\n  return <h1>{"{json-looking} \\"quotes\\""}</h1>\n}\n'
                "=== END FILE ===\n"
                "=== FILE: ../escape.txt ===\nnope\n=== END FILE ===\n"
                "=== NOTES ===\nnext: pricing")

    monkeypatch.setattr(gemini, "generate", fake)
    r = TOOLS["code.write"].run(dev.CodeArgs(project_dir=str(proj), instructions="homepage",
                                             context_files=["docs/research.md"]), ToolContext(None, 0))
    assert r.ok and r.data["files"] == ["src/app/page.tsx"] and r.data["notes"] == "next: pricing"
    assert '\\"quotes\\"' in (proj / "src/app/page.tsx").read_text()  # code with braces/quotes survives intact
    assert not (workspace / "escape.txt").exists()
    assert "Use a dark theme." in seen["user"]  # the research brief was given to the coder


def test_missing_packages_are_reported(workspace):
    proj = workspace / "pkgs"
    (proj / "src").mkdir(parents=True, exist_ok=True)
    (proj / "package.json").write_text('{"dependencies":{"next":"16","react":"19"},"devDependencies":{"@types/node":"20"}}')
    (proj / "src/a.tsx").write_text('import { X } from "lucide-react"\nimport Link from "next/link"\n'
                                    'import { m } from "@/lib/m"\nimport "./x.css"\nimport { z } from "@radix-ui/react-tabs/dist"\n'
                                    'import fs from "node:fs"\nimport type { T } from "@types/node"')
    assert dev.missing_packages(proj, ["src/a.tsx"]) == ["@radix-ui/react-tabs", "lucide-react"]


def _code_project(workspace, name, build_ok: bool):
    proj = workspace / name
    proj.mkdir(parents=True, exist_ok=True)
    script = "node --version" if build_ok else "node -e process.exit(1)"
    (proj / "package.json").write_text('{"name":"%s","scripts":{"build":"%s"},"dependencies":{}}' % (name, script))
    return proj


def _fake_coder(monkeypatch):
    monkeypatch.setattr(gemini, "generate", lambda system, parts, **kw:
                        "=== FILE: page.tsx ===\nexport default function P() { return null }\n=== END FILE ===")


def test_done_is_refused_until_the_build_passes(runner, monkeypatch, workspace):
    proj = _code_project(workspace, "gate-ok", build_ok=True)
    _fake_coder(monkeypatch)
    calls = scripted_gemini(monkeypatch, {
        "interpreted_objective": "write the page", "plan": [
            {"tool": "code.write", "args": {"project_dir": str(proj), "instructions": "page"}, "description": "code"}],
    }, [{"done": True, "summary": "All done (claimed too early).", "next_plan": []},
        {"done": True, "summary": "Page written and the build passes.", "next_plan": []}])
    t = wait(runner.create("write the page")[0], timeout=90)
    assert t.status == "completed", t.result
    tools = [p["tool"] for p in t.spec["plan"]]
    assert tools == ["code.write", "terminal.run"] and t.spec["plan"][1]["args"]["command"] == "npm run build"
    assert calls["review"] == 2 and t.result.startswith("Page written and the build passes")


def test_starter_homepage_is_never_reported_done(runner, monkeypatch, workspace):
    """Regression: pages were written and the build passed, but / was still the create-next-app page."""
    proj = _code_project(workspace, "starter", build_ok=True)
    (proj / "src/app").mkdir(parents=True, exist_ok=True)
    (proj / "src/app/page.tsx").write_text('<a href="https://vercel.com/new?utm_source=create-next-app">Deploy</a>')
    written = []

    def coder(system, parts, **kw):
        written.append(parts[0]["text"])
        target = "src/app/page.tsx" if "starter homepage" in parts[0]["text"] else "src/app/rules/page.tsx"
        return f"=== FILE: {target} ===\nexport default function P() {{ return <main>Real</main> }}\n=== END FILE ==="

    monkeypatch.setattr(gemini, "generate", coder)
    scripted_gemini(monkeypatch, {
        "interpreted_objective": "site", "plan": [
            {"tool": "code.write", "args": {"project_dir": str(proj), "instructions": "rules page"}, "description": "c"},
            {"tool": "terminal.run", "args": {"command": "npm run build", "cwd": str(proj)}, "description": "b"}],
    }, [{"done": True, "summary": "Site finished.", "next_plan": []}])
    t = wait(runner.create("build the site")[0], timeout=90)
    assert t.status == "completed", t.result
    assert "Real" in (proj / "src/app/page.tsx").read_text()
    assert [p["tool"] for p in t.spec["plan"]][-2:] == ["code.write", "terminal.run"]


def test_failing_build_is_never_reported_done(runner, monkeypatch, workspace):
    proj = _code_project(workspace, "gate-bad", build_ok=False)
    _fake_coder(monkeypatch)
    scripted_gemini(monkeypatch, {
        "interpreted_objective": "write the page", "plan": [
            {"tool": "code.write", "args": {"project_dir": str(proj), "instructions": "page"}, "description": "code"},
            {"tool": "terminal.run", "args": {"command": "npm run build", "cwd": str(proj)}, "description": "build"}],
    }, [{"done": True, "summary": "Everything works!", "next_plan": []}])
    t = wait(runner.create("write the page")[0], timeout=90)
    assert t.status == "failed" and "build" in t.result.lower()


# --- web ----------------------------------------------------------------------------------------

DDG = """<div class="result results_links results_links_deep web-result">
<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fftmo.com%2Fen%2F&amp;rut=x">FTMO &amp; Co</a>
<a class="result__snippet" href="#">Funded <b>trader</b> program</a></div>
<div class="result results_links results_links_deep web-result">
<a rel="nofollow" class="result__a" href="https://www.youtube.com/watch?v=1">video</a></div>"""


def test_search_parsing(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(200, text=DDG, request=httpx.Request("POST", "https://x")))
    assert web.search("prop firm") == [{"title": "FTMO & Co", "url": "https://ftmo.com/en/", "snippet": "Funded trader program"}]
