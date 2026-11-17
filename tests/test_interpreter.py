import pytest

from agent.interpreter import interpret, is_stop_command
from agent.safety import SafetyError, low_confidence_block, resolve_in_workspace
from agent.state import Risk


def tools(text):
    spec, _ = interpret(text)
    return [s.tool for s in spec.plan], spec


def test_open_app_needs_approval():
    names, spec = tools("Abbad, open notepad")
    assert names == ["desktop.launch_app"]
    assert spec.requires_approval and spec.risk_level == Risk.medium


def test_roman_urdu_open():
    names, spec = tools("notepad kholo")
    assert names == ["desktop.launch_app"] and spec.target == "notepad"


def test_website_inspection_is_low_risk():
    names, spec = tools("Browser mein https://example.com check karo")
    assert names == ["browser.inspect"] and not spec.requires_approval


def test_list_files_and_windows():
    assert tools("files dikhao")[0] == ["workspace.list"]
    assert tools("which windows are open?")[0] == ["desktop.list_windows"]


def test_create_file_and_overwrite_risk(workspace):
    names, spec = tools('create file notes.txt with "hello world"')
    assert names == ["workspace.write"] and spec.plan[0].args["content"] == "hello world"
    assert spec.risk_level == Risk.low
    (workspace / "exists.txt").write_text("x")
    _, spec2 = tools('create file exists.txt with "new"')
    assert spec2.risk_level == Risk.medium and spec2.requires_approval


def test_create_file_without_content_asks():
    _, spec = tools("create file todo.txt")
    assert not spec.plan and spec.clarification_question


def test_unknown_request_is_blocked_not_faked():
    _, spec = tools("research my competitors and build the whole product")
    assert not spec.plan and spec.clarification_question


def test_stop_commands():
    assert is_stop_command("Abbad, stop")
    assert is_stop_command("ruko")
    assert is_stop_command("bas karo")
    assert is_stop_command("A bass. Stop.")
    assert not is_stop_command("open notepad")


def test_workspace_escape_blocked():
    with pytest.raises(SafetyError):
        resolve_in_workspace("../../Windows/system32/x.txt")
    with pytest.raises(SafetyError):
        resolve_in_workspace("C:/Windows/win.ini")


def test_low_confidence_voice_cannot_authorise_risky_action():
    assert low_confidence_block("voice", 0.3, Risk.medium)
    assert low_confidence_block("voice", 0.3, Risk.low) is None
    assert low_confidence_block("text", None, Risk.high) is None
