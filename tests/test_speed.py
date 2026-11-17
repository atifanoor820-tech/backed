"""Latency paths: simple commands skip the model and the review round; goals still get reviewed."""

import pytest

from agent import interpreter
from agent.graph.workflow import needs_review
from agent.llm import gemini
from agent.state import TaskSpec


def spec(request: str, multi_round: bool = False) -> TaskSpec:
    return TaskSpec(original_request=request, normalized_request=request.lower(), interpreted_objective="x",
                    interpreter="llm", multi_round=multi_round)


def ok(tool: str = "workspace.list") -> dict:
    return {"step": 0, "tool": tool, "ok": True, "summary": "", "data": {}}


@pytest.mark.parametrize("request_,results,expected", [
    ("files dikhao", [ok()], False),
    ("notes.txt mein kya likha hai?", [ok("workspace.read")], False),
    ("example.com kholo aur Learn more par click karo", [ok("browser.open"), ok("browser.click")], False),
    ("make folder and check node", [ok("files.make_folder")], True),           # 2 parts, 1 step ran
    ("prop firm ka frontend banao Next.js project", [ok("files.make_folder")], True),  # goal wording
    ("open notepad", [{**ok("desktop.launch_app"), "ok": False}], True),         # failures get repaired
    ("write the page", [ok("code.write")], True),                               # code always reviewed
])
def test_needs_review(request_, results, expected):
    assert needs_review(spec(request_), results) is expected


def test_planner_flag_still_forces_review():
    assert needs_review(spec("files dikhao", multi_round=True), [ok()])


@pytest.mark.parametrize("text,tool", [
    ("files dikhao", "workspace.list"),
    ("Abbad, which windows are open?", "desktop.list_windows"),
    ("notepad kholo", "desktop.launch_app"),
    ("Plus dabao", "browser.click"),
])
def test_short_clear_commands_skip_the_planner(settings, monkeypatch, text, tool):
    settings(gemini_api_key="g", jev_api_key="")

    def fake(system, user, **kw):
        if "Abbas (nickname Abbad)" in system:
            pytest.fail("the slow planning call must be skipped")
        return {"p_yes": 0.0}  # the quick risk check on clicks is kept on purpose

    monkeypatch.setattr(gemini, "generate_json", fake)
    s, warning = interpreter.interpret(text)
    assert s.interpreter == "rules" and s.plan[0].tool == tool and warning is None


def test_url_without_inspect_words_is_not_an_inspection(settings, monkeypatch):
    settings(gemini_api_key="g", jev_api_key="")
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **k: {"interpreted_objective": "search", "plan": [
        {"tool": "browser.open", "args": {"url": "https://example.com"}, "description": "open"}]})
    s, _ = interpreter.interpret("example.com par cats search karo")
    assert s.interpreter == "llm"
    s, _ = interpreter.interpret("example.com check karo")
    assert s.interpreter == "rules" and s.plan[0].tool == "browser.inspect"


@pytest.mark.parametrize("text", [
    "files dikhao aur notes.txt parho",      # compound
    "open youtube",                           # unknown app name: probably a website
    "salam kaise ho",                         # conversation
    "mujhe ek acha sa prop firm ka landing page bana kar do",  # long
])
def test_other_requests_still_go_to_the_model(settings, monkeypatch, text):
    settings(gemini_api_key="g", jev_api_key="")
    called = []
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **k: called.append(1) or {"interpreted_objective": "x",
                                                                                    "reply": "ok"})
    interpreter.interpret(text)
    assert called
