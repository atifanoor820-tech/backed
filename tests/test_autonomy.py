import time

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy import select

from agent import autonomy, interpreter
from agent.llm import gemini
from agent.runner import TaskRunner
from agent.safety import low_confidence_block
from agent.state import Risk
from agent.tools import desktop
from database.models import Task, TaskEvent
from database.session import session_scope


@pytest.fixture
def level():
    yield autonomy.set_level
    autonomy.set_level("ask")


def wait(task_id, statuses=("completed", "failed", "blocked", "cancelled", "awaiting_approval"), timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with session_scope() as s:
            t = s.get(Task, task_id)
            if t and t.status in statuses:
                return t
        time.sleep(0.05)
    raise AssertionError("timeout")


def test_approval_matrix():
    n = autonomy.needs_approval
    assert [n(Risk.low, "ask"), n(Risk.medium, "ask"), n(Risk.high, "ask")] == [False, True, True]
    assert [n(Risk.low, "trusted"), n(Risk.medium, "trusted"), n(Risk.high, "trusted")] == [False, False, True]
    assert [n(Risk.low, "full"), n(Risk.medium, "full"), n(Risk.high, "full")] == [False, False, False]


def test_level_persists_and_validates(level):
    level("trusted")
    assert autonomy.get_level() == "trusted"
    with pytest.raises(ValueError):
        level("god-mode")


def test_full_mode_runs_without_asking_and_is_audited(level, settings, workspace):
    settings(gemini_api_key="", jev_api_key="")
    level("full")
    (workspace / "auto.txt").write_text("v1")
    runner = TaskRunner(InMemorySaver(), InMemoryStore())
    tid = runner.create('create file auto.txt with "v2"')[0]  # overwrite = medium risk
    t = wait(tid)
    assert t.status == "completed" and (workspace / "auto.txt").read_text() == "v2"
    with session_scope() as s:
        events = {e.type: e for e in s.scalars(select(TaskEvent).where(TaskEvent.task_id == tid))}
    assert "approval_required" not in events
    assert events["approval_granted"].data == {"auto": True, "autonomy": "full"}


def test_trusted_still_asks_for_hard_to_undo(level, settings, monkeypatch):
    settings(gemini_api_key="g", jev_api_key="")
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **k: {
        "interpreted_objective": "save", "plan": [{"tool": "screen.hotkey", "args": {"keys": ["ctrl", "s"]}}]})
    level("trusted")
    assert interpreter.interpret("save karo")[0].requires_approval  # screen.hotkey is high risk
    level("full")
    assert not interpreter.interpret("save karo")[0].requires_approval


def test_hard_limits_stay_in_full_mode(level):
    level("full")
    # A misheard voice command never authorises a risky action, at any autonomy level.
    assert low_confidence_block("voice", 0.3, Risk.medium)


def test_any_installed_app_is_resolved(monkeypatch, settings):
    settings(gemini_api_key="", jev_api_key="")
    apps = [{"name": "Google Chrome", "id": "Chrome"}, {"name": "Spotify", "id": "SpotifyAB!Spotify"},
            {"name": "Visual Studio Code", "id": "Microsoft.VisualStudioCode"}]
    monkeypatch.setattr(desktop, "start_apps", lambda max_age_s=600: apps)
    assert desktop.resolve_start_app(None, "spotify")["id"] == "SpotifyAB!Spotify"
    assert desktop.resolve_start_app(None, "chrome")["name"] == "Google Chrome"
    assert desktop.resolve_start_app(None, "visual studio")["name"] == "Visual Studio Code"
    assert desktop.resolve_start_app(None, "photoshop") is None


@pytest.mark.parametrize("text,app", [("Spotify kholo", "Spotify"), ("open google chrome", "google chrome"),
                                      ("VS Code chalao", "VS Code")])
def test_open_any_app_rule(settings, text, app):
    settings(gemini_api_key="", jev_api_key="")
    spec, _ = interpreter.interpret(text)
    assert spec.plan[0].tool == "desktop.launch_app" and spec.plan[0].args["name"] == app
