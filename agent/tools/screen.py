"""PyAutoGUI visual fallback: screenshot -> locate target (Jev box / Gemini vision) -> mouse/keyboard.

Used only when a control is not reachable through Playwright or UI Automation. Every action
is approval-gated, PyAutoGUI's fail-safe stays on (slam the mouse into a screen corner to abort),
and each action is checked by comparing before/after screenshots.
"""

import io
import sys
import time

from pydantic import BaseModel, Field, field_validator

from agent import runtime
from agent.config import get_settings
from agent.desktop_env import ensure_default_desktop
from agent.events import emit
from agent.reflex.decide import locate
from agent.state import Risk, ToolResult
from agent.tools.registry import Tool, ToolContext, register

BLOCKED_HOTKEYS = {("ctrl", "alt", "delete"), ("win", "l"), ("alt", "f4"), ("win", "r"), ("ctrl", "shift", "esc")}
ALLOWED_KEYS = {
    "ctrl", "alt", "shift", "win", "enter", "esc", "tab", "space", "backspace", "delete", "home", "end",
    "pageup", "pagedown", "up", "down", "left", "right", *[f"f{i}" for i in range(1, 13)],
    *"abcdefghijklmnopqrstuvwxyz0123456789",
}


def _gui():
    if sys.platform == "win32":
        ensure_default_desktop()
    import pyautogui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    return pyautogui


def _grab(ctx: ToolContext | None = None, suffix: str = "screen"):
    img = _gui().screenshot()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    name = None
    if ctx:
        shots = get_settings().data_dir / "screenshots"
        shots.mkdir(parents=True, exist_ok=True)
        name = f"{ctx.task_id}_{ctx.step_index}_{suffix}.png"
        img.save(shots / name)
        emit(ctx.task_id, "screenshot_captured", name, {"path": name})
    return img, buf.getvalue(), name


def _changed_fraction(a, b) -> float:
    from PIL import ImageChops

    small = (320, 180)
    diff = ImageChops.difference(a.convert("L").resize(small), b.convert("L").resize(small))
    return sum(1 for px in diff.getdata() if px > 24) / (small[0] * small[1])


def _disabled() -> ToolResult | None:
    if not runtime.get("screen_actions"):
        return ToolResult(ok=False, summary="screen control (mouse/keyboard) is switched off in Settings")
    return None


class ShotArgs(BaseModel):
    pass


class SClickArgs(BaseModel):
    target: str = Field(description="What to click, described visually, e.g. 'the blue Save button'")
    double: bool = False


class STypeArgs(BaseModel):
    text: str = Field(max_length=2000)
    press_enter: bool = False


class HotkeyArgs(BaseModel):
    keys: list[str] = Field(min_length=1, max_length=4, description="e.g. ['ctrl', 's']")

    @field_validator("keys")
    @classmethod
    def _check(cls, v: list[str]) -> list[str]:
        keys = [k.lower().strip() for k in v]
        bad = [k for k in keys if k not in ALLOWED_KEYS]
        if bad:
            raise ValueError(f"keys not allowed: {bad}")
        if tuple(keys) in BLOCKED_HOTKEYS:
            raise ValueError(f"hotkey {'+'.join(keys)} is blocked")
        return keys


def _shot(args: ShotArgs, ctx: ToolContext) -> ToolResult:
    img, _, name = _grab(ctx, "screen")
    return ToolResult(ok=True, summary=f"captured the screen ({img.width}x{img.height})",
                      data={"screenshot": name, "size": [img.width, img.height]}, evidence=[name])


def _click(args: SClickArgs, ctx: ToolContext) -> ToolResult:
    if r := _disabled():
        return r
    before, png, _ = _grab(ctx, "before")
    box, engine = locate(ctx.task_id, png, args.target)
    if box is None:
        return ToolResult(ok=False, summary=f"could not find '{args.target}' on screen ({engine}). "
                          "Visual location needs Gemini (or Jev with ABBAS_JEV_IMAGES=true).")
    conf = box.get("confidence")
    if conf is not None and conf < get_settings().jev_min_confidence:
        return ToolResult(ok=False, summary=f"found '{args.target}' but only {conf:.0%} confident; not clicking")
    x = int((box["left"] + box["right"]) / 2 * before.width)
    y = int((box["top"] + box["bottom"]) / 2 * before.height)
    gui = _gui()
    gui.moveTo(x, y, duration=0.25)
    (gui.doubleClick if args.double else gui.click)(x, y)
    time.sleep(0.8)
    after, _, _ = _grab(ctx, "after")
    changed = _changed_fraction(before, after)
    return ToolResult(ok=True, summary=f"{'double-' if args.double else ''}clicked '{args.target}' at ({x}, {y}) via {engine}; "
                      f"{changed:.1%} of the screen changed",
                      data={"x": x, "y": y, "box": box, "engine": engine, "screen_changed": changed},
                      evidence=[f"({x}, {y})"])


def _type(args: STypeArgs, ctx: ToolContext) -> ToolResult:
    if r := _disabled():
        return r
    gui = _gui()
    before, _, _ = _grab()
    if args.text.isascii():
        gui.write(args.text, interval=0.02)
    else:  # pyautogui.write only handles ASCII; paste Urdu and other scripts via the clipboard
        import pyperclip

        pyperclip.copy(args.text)
        gui.hotkey("ctrl", "v")
    if args.press_enter:
        gui.press("enter")
    time.sleep(0.5)
    after, _, _ = _grab(ctx, "after")
    changed = _changed_fraction(before, after)
    return ToolResult(ok=changed > 0, summary=f"typed {len(args.text)} characters into the focused window; "
                      f"{changed:.1%} of the screen changed", data={"screen_changed": changed})


def _hotkey(args: HotkeyArgs, ctx: ToolContext) -> ToolResult:
    if r := _disabled():
        return r
    gui = _gui()
    before, _, _ = _grab()
    gui.hotkey(*args.keys)
    time.sleep(0.6)
    after, _, _ = _grab(ctx, "after")
    changed = _changed_fraction(before, after)
    return ToolResult(ok=True, summary=f"pressed {'+'.join(args.keys)}; {changed:.1%} of the screen changed",
                      data={"screen_changed": changed})


register(Tool("screen.screenshot", "Capture the whole screen.", ShotArgs, Risk.low, _shot))
register(Tool("screen.click", "Visual fallback: find an element on screen and click it with the mouse (PyAutoGUI).",
              SClickArgs, Risk.medium, _click, risk_reflex=True))
register(Tool("screen.type", "Visual fallback: type text into whatever window has focus (PyAutoGUI).",
              STypeArgs, Risk.medium, _type, risk_reflex=True))
register(Tool("screen.hotkey", "Visual fallback: press a keyboard shortcut, e.g. ctrl+s (PyAutoGUI).",
              HotkeyArgs, Risk.high, _hotkey))
