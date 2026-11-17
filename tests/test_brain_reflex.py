"""Gemini (brain) and Jev (reflex) integration, with HTTP mocked - no real keys needed."""

import json

import httpx
import pytest

from agent import interpreter
from agent.llm import gemini
from agent.reflex import decide, jev
from agent.state import Risk


def test_jev_request_shape(settings):
    settings(jev_api_key="k", jev_model="jev-1.14.0")
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": {"q": {
            "type": "choice", "choice": "c1", "confidence": 0.93, "probabilities": {"c0": 0.07, "c1": 0.93}}},
            "usage": {"input_tokens": 50, "output_tokens": 3}})

    out = jev.ask({"controls": ["a", "b"]}, {"q": {"type": "choice", "instructions": "which?",
                                                   "criteria": {"c0": "a", "c1": "b"}}},
                  transport=httpx.MockTransport(handler))
    assert seen["url"].endswith("/v1/systemone")
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "jev-1.14.0" and seen["body"]["questions"]["q"]["type"] == "choice"
    assert out["answers"]["q"]["choice"] == "c1"


def test_jev_http_error_is_raised(settings):
    settings(jev_api_key="k")
    t = httpx.MockTransport(lambda r: httpx.Response(422, json={"detail": [{"msg": "bad"}]}))
    with pytest.raises(jev.JevError, match="422"):
        jev.ask("s", {"q": {"type": "noul", "instructions": "?"}}, transport=t)


def test_gemini_locate_converts_box(settings):
    settings(gemini_api_key="g")
    reply = {"candidates": [{"content": {"parts": [{"text": json.dumps({"found": True, "box_2d": [100, 200, 300, 400]})}]}}]}
    box = gemini.locate(b"png", "Save button", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply)))
    assert box == {"top": 0.1, "left": 0.2, "bottom": 0.3, "right": 0.4, "confidence": None}


def test_gemini_falls_back_to_next_free_model_on_quota(settings):
    settings(gemini_api_key="g", gemini_model="gemini-3.8-flash", gemini_fallback_models="gemini-3.5-flash-lite")
    called = []

    def handler(req: httpx.Request):
        called.append(req.url.path.split("/")[-1])
        if "3.8-flash" in req.url.path:
            return httpx.Response(429, json={"error": {"message": "quota"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]})

    assert gemini.generate_json("s", "u", transport=httpx.MockTransport(handler)) == {"ok": True}
    assert called == ["gemini-3.8-flash:generateContent", "gemini-3.5-flash-lite:generateContent"]


def test_gemini_skips_overloaded_and_slow_models(settings):
    settings(gemini_api_key="g", gemini_model="a", gemini_fallback_models="b,c")

    def handler(req: httpx.Request):
        if "/a:" in req.url.path:
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        if "/b:" in req.url.path:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"m": "c"}'}]}}]})

    assert gemini.generate_json("s", "u", transport=httpx.MockTransport(handler)) == {"m": "c"}


def test_fast_decisions_use_fast_model(settings):
    settings(gemini_api_key="g", gemini_model="slow", gemini_fast_model="quick", gemini_fallback_models="")
    seen = []

    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    gemini.generate_json("s", "u", transport=httpx.MockTransport(handler), fast=True)
    assert seen[0].endswith("/quick:generateContent")


def test_gemini_all_quotas_exhausted(settings):
    settings(gemini_api_key="g")
    t = httpx.MockTransport(lambda r: httpx.Response(429, json={"error": {"message": "quota"}}))
    with pytest.raises(gemini.GeminiError, match="free-tier quota exhausted"):
        gemini.generate_json("s", "u", transport=t)


OPTIONS = {"c0": 'Button "Seven" (id=num7Button)', "c1": 'Button "Eight" (id=num8Button)', "c2": 'Button "Plus"'}


def test_choose_prefers_jev(settings, monkeypatch):
    settings(jev_api_key="k", gemini_api_key="g")
    monkeypatch.setattr(jev, "ask", lambda *a, **k: {"answers": {"q": {"choice": "c0", "confidence": 0.97,
                                                                       "probabilities": {"c0": 0.97}}}})
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **k: pytest.fail("Gemini must not be called"))
    d = decide.choose(None, {}, "which is 7?", OPTIONS, target="7")
    assert (d.choice, d.engine) == ("c0", "jev") and d.confident()


def test_choose_falls_back_to_gemini_then_local(settings, monkeypatch):
    settings(jev_api_key="k", gemini_api_key="g")

    def boom(*a, **k):
        raise jev.JevError("529 overloaded")

    monkeypatch.setattr(jev, "ask", boom)
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **k: {"choice": "c1", "confidence": 0.8})
    assert decide.choose(None, {}, "q", OPTIONS, target="eight").engine == "gemini"

    settings(jev_api_key="", gemini_api_key="")
    d = decide.choose(None, {}, "q", OPTIONS, target="Seven")
    assert (d.choice, d.engine) == ("c0", "local") and d.confident()


def test_local_choice_is_not_confident_on_nonsense(settings):
    settings(jev_api_key="", gemini_api_key="")
    assert not decide.choose(None, {}, "q", OPTIONS, target="Subscribe to newsletter").confident()


def test_risk_reflex_raises_but_never_lowers(settings, monkeypatch):
    settings(jev_api_key="k")
    monkeypatch.setattr(jev, "ask", lambda *a, **k: {"answers": {"q": {"noul": 0.92}}})
    spec, _ = interpreter.interpret('click "Delete account"')
    assert spec.plan[0].tool == "browser.click" and spec.plan[0].risk == Risk.high and spec.requires_approval

    monkeypatch.setattr(jev, "ask", lambda *a, **k: {"answers": {"q": {"noul": 0.02}}})
    spec, _ = interpreter.interpret('click "7" in Calculator')
    assert spec.plan[0].risk == Risk.medium  # desktop.click stays medium even when Jev says safe


def test_without_any_engine_browser_clicks_need_approval(settings):
    settings(jev_api_key="", gemini_api_key="")
    spec, _ = interpreter.interpret('click "More information"')
    assert spec.plan[0].risk == Risk.medium and spec.requires_approval


def test_gemini_plans_and_registry_sets_risk(settings, monkeypatch):
    settings(gemini_api_key="g")
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **k: {
        "interpreted_objective": "search for cats",
        "plan": [{"tool": "browser.open", "args": {"url": "https://example.com"}, "description": "open", "risk": "low"},
                 {"tool": "browser.type", "args": {"target": "search box", "text": "cats", "submit": True},
                  "description": "search"}]})
    monkeypatch.setattr(jev, "ask", lambda *a, **k: pytest.fail("Jev not configured"))
    spec, warning = interpreter.interpret("Abbad, example.com par cats search karo")
    assert warning is None and spec.interpreter == "llm"
    assert [s.tool for s in spec.plan] == ["browser.open", "browser.type"]
    assert spec.plan[1].risk == Risk.medium  # no reflex engine besides Gemini's own planning -> conservative


def test_gemini_unknown_tool_blocks(settings, monkeypatch):
    settings(gemini_api_key="g")
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **k: {
        "interpreted_objective": "x", "plan": [{"tool": "shell.run", "args": {"cmd": "del *"}}]})
    spec, _ = interpreter.interpret("delete everything")
    assert not spec.plan and "unknown tool" in spec.clarification_question


def test_gemini_failure_falls_back_to_rules(settings, monkeypatch):
    settings(gemini_api_key="g")

    def down(*a, **k):
        raise gemini.GeminiError("Gemini HTTP 503")

    monkeypatch.setattr(gemini, "generate_json", down)
    spec, warning = interpreter.interpret("please show me all the files in my workspace folder")  # too long for the instant path
    assert spec.plan[0].tool == "workspace.list" and "Gemini planning failed" in warning


@pytest.mark.parametrize("text,tool,args", [
    ('click "7" in Calculator', "desktop.click", {"window": "Calculator", "target": "7"}),
    ('Calculator mein "Plus" dabao', "desktop.click", {"window": "Calculator", "target": "Plus"}),
    ('type "hello world" in Notepad', "desktop.type", {"window": "Notepad", "text": "hello world"}),
    # Spoken commands arrive without quotes
    ("click Seven in Calculator", "desktop.click", {"window": "Calculator", "target": "Seven"}),
    ("Abbad, press the equals button in calculator.", "desktop.click", {"window": "calculator", "target": "equals"}),
    ("Calculator mein Plus dabao", "desktop.click", {"window": "Calculator", "target": "Plus"}),
    ("type hello there into Notepad", "desktop.type", {"window": "Notepad", "text": "hello there"}),
    ("Notepad mein salam likho", "desktop.type", {"window": "Notepad", "text": "salam"}),
    ("open https://example.com", "browser.open", {"url": "https://example.com/"}),
    ("screenshot lo", "screen.screenshot", {}),
])
def test_rule_commands_for_action_tools(settings, text, tool, args):
    settings(jev_api_key="", gemini_api_key="")
    spec, _ = interpreter.interpret(text)
    assert spec.plan[0].tool == tool
    for k, v in args.items():
        assert spec.plan[0].args[k] == v


def test_blocked_hotkeys():
    from pydantic import ValidationError

    from agent.tools.screen import HotkeyArgs

    with pytest.raises(ValidationError):
        HotkeyArgs(keys=["alt", "f4"])
    with pytest.raises(ValidationError):
        HotkeyArgs(keys=["ctrl", "rm -rf"])
    assert HotkeyArgs(keys=["CTRL", "s"]).keys == ["ctrl", "s"]
