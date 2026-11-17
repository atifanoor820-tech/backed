"""pywinauto (Windows UI Automation) actions: read, click and type into real application controls.

The target control is picked by the decision layer (Jev -> Gemini -> local matcher) from
a text list of the window's controls, because Jev reasons over text state.
"""

import re
import sys
import time

from pydantic import BaseModel, Field

from agent.desktop_env import ensure_default_desktop

from agent.reflex.decide import choose, yes_no
from agent.state import Risk, ToolResult
from agent.tools.registry import Tool, ToolContext, register

ACTIONABLE = {
    "Button", "Edit", "Document", "MenuItem", "ListItem", "TabItem", "Hyperlink", "CheckBox",
    "RadioButton", "ComboBox", "TreeItem", "SplitButton", "Text",
}
TYPEABLE = {"Edit", "Document", "ComboBox"}
MAX_CONTROLS = 400


class WindowNotFound(Exception):
    pass


def _windows():
    if sys.platform == "win32":
        ensure_default_desktop()
    from pywinauto import Desktop

    out = []
    for w in Desktop(backend="uia").windows():
        try:
            if w.window_text() and w.is_visible():
                out.append(w)
        except Exception:
            continue
    return out


def find_window(task_id: str, hint: str):
    wins = _windows()
    hint_l = hint.lower().strip()
    exact = [w for w in wins if hint_l and hint_l in w.window_text().lower()]
    if len(exact) == 1:
        return exact[0]
    pool = exact or wins
    options = {f"w{i}": f'window "{w.window_text()}"' for i, w in enumerate(pool)}
    d = choose(task_id, {"open_windows": list(options.values())}, f"Which window is the user referring to as '{hint}'?",
               options, target=hint, purpose="window")
    if not d.confident():
        raise WindowNotFound(f"no confident match for window '{hint}' (best: {d.top()})")
    return pool[int(d.choice[1:])]


def describe_controls(win, kinds: set[str] = ACTIONABLE) -> tuple[dict[str, str], dict[str, object]]:
    descs, ctrls = {}, {}
    for c in win.descendants():
        if len(descs) >= MAX_CONTROLS:
            break
        try:
            info = c.element_info
            kind = info.control_type
            if kind not in kinds or not c.is_visible():
                continue
            name = (info.name or "").strip()
            auto_id = info.automation_id or ""
            if not name and not auto_id:
                continue
            key = f"c{len(descs)}"
            descs[key] = f'{kind} "{name}"' + (f" (id={auto_id})" if auto_id else "")
            ctrls[key] = c
        except Exception:
            continue
    return descs, ctrls


def window_snapshot(win, limit: int = 60) -> list[str]:
    """Short text view of a window, used as state for verification questions."""
    lines = []
    for c in win.descendants():
        try:
            info = c.element_info
            if info.name and info.control_type in ("Text", "Edit", "Document", "Button", "TitleBar"):
                lines.append(f"{info.control_type}: {info.name.strip()[:200]}")
        except Exception:
            continue
        if len(lines) >= limit:
            break
    return lines


def _pick_control(ctx: ToolContext, win, target: str, kinds: set[str], verb: str):
    descs, ctrls = describe_controls(win, kinds)
    if not descs:
        raise WindowNotFound(f"window '{win.window_text()}' exposes no usable controls; try a screen.* action")
    d = choose(ctx.task_id, {"window": win.window_text(), "controls": list(descs.values())},
               f"Which control should be {verb} to do: {target}?", descs, target=target, purpose="control")
    if not d.confident():
        cands = ", ".join(f"{descs[k]} {p:.0%}" for k, p in d.top())
        raise WindowNotFound(f"not confident which control is '{target}' ({d.engine}, {d.confidence:.0%}); candidates: {cands}")
    return ctrls[d.choice], descs[d.choice], d


def _verify(ctx: ToolContext, win, goal: str) -> tuple[bool | None, list[str]]:
    time.sleep(0.4)
    try:
        after = window_snapshot(win)
    except Exception:
        return None, []
    p, engine = yes_no(ctx.task_id, {"window": win.window_text(), "visible_text": after},
                       f"Does the window now show that this was done: {goal}?", purpose="verify")
    return (None if p is None else p >= 0.5), after


class ReadArgs(BaseModel):
    window: str = Field(description="Window title or a description of it, e.g. 'Calculator'")


class ClickArgs(BaseModel):
    window: str
    target: str = Field(description="The control to click, in words, e.g. 'the 7 button' or 'File menu'")


class TypeArgs(BaseModel):
    window: str
    text: str = Field(max_length=5000)
    target: str | None = Field(None, description="Field to type into; omit to type into the focused control")
    press_enter: bool = False


def _read(args: ReadArgs, ctx: ToolContext) -> ToolResult:
    try:
        win = find_window(ctx.task_id, args.window)
    except WindowNotFound as e:
        return ToolResult(ok=False, summary=str(e))
    lines = window_snapshot(win, limit=120)
    return ToolResult(ok=True, summary=f"read {len(lines)} text elements from '{win.window_text()}'",
                      data={"window": win.window_text(), "text": lines}, evidence=lines[:10])


def _click(args: ClickArgs, ctx: ToolContext) -> ToolResult:
    try:
        win = find_window(ctx.task_id, args.window)
        win.set_focus()
        ctrl, desc, d = _pick_control(ctx, win, args.target, ACTIONABLE, "clicked")
    except WindowNotFound as e:
        return ToolResult(ok=False, summary=str(e))
    try:
        ctrl.invoke()
        how = "invoke"
    except Exception:
        ctrl.click_input()
        how = "mouse click"
    verified, after = _verify(ctx, win, f"clicked {args.target}")
    msg = f"clicked {desc} in '{win.window_text()}' via {how} ({d.engine} {d.confidence:.0%})"
    if verified is False:
        return ToolResult(ok=False, summary=msg + ", but the window does not show the expected result",
                          data={"after": after}, evidence=after[:10])
    return ToolResult(ok=True, summary=msg + ("" if verified else " (result not independently verified)"),
                      data={"control": desc, "after": after, "engine": d.engine, "window": win.window_text()},
                      evidence=after[:10])


def _escape_keys(text: str) -> str:
    return re.sub(r"([{}+^%~()])", r"{\1}", text)


def _type(args: TypeArgs, ctx: ToolContext) -> ToolResult:
    try:
        win = find_window(ctx.task_id, args.window)
        win.set_focus()
        if args.target:
            ctrl, desc, _ = _pick_control(ctx, win, args.target, TYPEABLE, "typed into")
        else:
            descs, ctrls = describe_controls(win, TYPEABLE)
            if not ctrls:
                raise WindowNotFound(f"no text field found in '{win.window_text()}'")
            key = next(iter(ctrls))
            ctrl, desc = ctrls[key], descs[key]
    except WindowNotFound as e:
        return ToolResult(ok=False, summary=str(e))
    ctrl.set_focus()
    ctrl.type_keys(_escape_keys(args.text) + ("{ENTER}" if args.press_enter else ""),
                   with_spaces=True, with_newlines=True, pause=0.01)
    time.sleep(0.3)
    try:
        current = ctrl.window_text() or ""
        try:
            current = ctrl.get_value() or current
        except Exception:
            pass
    except Exception:
        current = ""
    ok = args.text.strip()[:50] in current if current else None
    summary = f"typed {len(args.text)} characters into {desc} in '{win.window_text()}'"
    if ok is False:
        return ToolResult(ok=False, summary=summary + ", but the field does not contain the text",
                          data={"field_value": current[:500]})
    return ToolResult(ok=True, summary=summary + (" (verified in field)" if ok else " (field value not readable)"),
                      data={"field_value": current[:500], "window": win.window_text()},
                      evidence=[current[:200]] if current else [])


register(Tool("desktop.read_window", "Read the visible text of a desktop application window.", ReadArgs, Risk.low, _read))
register(Tool("desktop.click", "Click a named control (button, menu, tab, link) in a desktop window via UI Automation.",
              ClickArgs, Risk.medium, _click, risk_reflex=True))
register(Tool("desktop.type", "Type text into a field of a desktop window via UI Automation.",
              TypeArgs, Risk.medium, _type, risk_reflex=True))
