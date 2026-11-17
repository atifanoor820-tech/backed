import subprocess

import pytest

from agent import runtime
from agent.config import get_settings
from agent.tools import desktop, screen
from agent.tools.registry import ToolContext

APPS = [{"name": "Paint", "id": "Microsoft.Paint!App"}, {"name": "Spotify", "id": "Spotify!App"},
        {"name": "Notepad", "id": "Microsoft.WindowsNotepad!App"}]


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    path = get_settings().data_dir / "settings.json"
    saved = path.read_text() if path.exists() else None
    path.unlink(missing_ok=True)
    monkeypatch.setattr(desktop, "start_apps", lambda max_age_s=600: APPS)
    yield
    path.unlink(missing_ok=True)
    if saved is not None:
        path.write_text(saved)


def test_defaults_and_validation():
    rt = runtime.get_runtime()
    assert rt["autonomy"] == "ask" and rt["app_access"] == "all"
    assert runtime.update(screen_actions=False)["screen_actions"] is False
    with pytest.raises(ValueError):
        runtime.update(autonomy="yolo")
    with pytest.raises(ValueError):
        runtime.update(screen_actions="nope")
    with pytest.raises(ValueError):
        runtime.update(workspaces="C:/")  # .env-only settings cannot be changed from the dashboard


def test_app_rules_in_both_modes():
    runtime.set_app_rules({"Spotify!App": False})
    assert runtime.app_allowed("Microsoft.Paint!App") and not runtime.app_allowed("Spotify!App")
    runtime.update(app_access="selected")
    runtime.set_app_rules({"Microsoft.Paint!App": True})
    assert runtime.app_allowed("Microsoft.Paint!App")
    assert not runtime.app_allowed("Some.Other!App")  # not switched on -> blocked in "selected" mode


def test_catalog_merges_allowlist_and_installed_apps():
    runtime.set_app_rules({"Spotify!App": False})
    cat = {a["name"]: a for a in desktop.app_catalog()}
    assert cat["Notepad"]["source"] == "allowlist" and cat["Notepad"]["id"] == "Microsoft.WindowsNotepad!App"
    assert cat["Paint"]["allowed"] and not cat["Spotify"]["allowed"]
    assert "Calculator" in cat  # allow-list entry without a Start-menu match gets an exe: id
    assert cat["Calculator"]["id"] == "exe:calculator"


def test_switched_off_app_is_never_started(monkeypatch):
    runtime.set_app_rules({"Spotify!App": False})
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a))
    monkeypatch.setattr(desktop, "visible_windows", lambda: [])
    r = desktop._launch(desktop.LaunchArgs(name="spotify"), ToolContext(task_id=None, step_index=0))
    assert not r.ok and "switched off" in r.summary and started == []

    runtime.set_app_rules({"Microsoft.WindowsNotepad!App": False})
    r = desktop._launch(desktop.LaunchArgs(name="notepad"), ToolContext(task_id=None, step_index=0))
    assert not r.ok and "switched off" in r.summary  # allow-list apps obey the same switch


def test_any_app_switch(monkeypatch):
    runtime.update(allow_any_app=False)
    monkeypatch.setattr(desktop, "visible_windows", lambda: [])
    r = desktop._launch(desktop.LaunchArgs(name="paint"), ToolContext(task_id=None, step_index=0))
    assert not r.ok and "Open any installed app" in r.summary
    assert {a["source"] for a in desktop.app_catalog()} == {"allowlist"}


def test_screen_control_switch():
    runtime.update(screen_actions=False)
    r = screen._click(screen.SClickArgs(target="Save"), ToolContext(task_id=None, step_index=0))
    assert not r.ok and "switched off in Settings" in r.summary
