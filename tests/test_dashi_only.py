"""ABBAS_DASHI_ONLY: Dashi commands still run; everything else gets a help reply, and Abbas can't be switched on."""

import pytest

from agent.dashi_config import is_dashi_active, set_active_agent
from agent.interpreter import DASHI_HELP, interpret


@pytest.fixture
def dashi_only(settings):
    return settings(dashi_only=True)


def test_dashi_commands_still_plan(dashi_only):
    spec, _ = interpret("592268636 update karo")
    assert [s.tool for s in spec.plan] == ["dashi.update_account"]

    spec, _ = interpret("592268636 aur 40001234 update karo")
    assert [s.args["account_number"] for s in spec.plan] == ["592268636", "40001234"]

    spec, _ = interpret("saved accounts check karo")
    assert [s.tool for s in spec.plan] == ["dashi.check_accounts"]


def test_other_requests_get_help_instead_of_running(dashi_only):
    for text in ("notepad kholo", "files dikhao", "example.com kholo"):
        spec, _ = interpret(text)
        assert spec.plan == []
        assert spec.clarification_question == DASHI_HELP


def test_dashi_is_always_active_and_abbas_cannot_be_switched_on(dashi_only):
    assert is_dashi_active()
    with pytest.raises(ValueError, match="turned off"):
        set_active_agent("abbas")


def test_flag_off_keeps_abbas(settings):
    settings(dashi_only=False)
    spec, _ = interpret("notepad kholo")
    assert [s.tool for s in spec.plan] == ["desktop.launch_app"]


def test_fresh_server_without_config_file_reports_dashi(dashi_only, tmp_path, monkeypatch):
    import agent.dashi_config as dc
    from agent.dashi_config import load_dashi_config

    monkeypatch.setattr(dc, "_config_path", lambda: tmp_path / "dashi_config.json")
    assert load_dashi_config()["active_agent"] == "dashi"
