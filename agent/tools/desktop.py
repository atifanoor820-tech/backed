"""Windows desktop tools: window discovery (pywinauto) and allow-listed app launch."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

from agent import runtime
from agent.config import child_env, get_settings
from agent.desktop_env import bring_to_front, ensure_default_desktop, spawn_on_desktop
from agent.state import Risk, ToolResult
from agent.tools.registry import Tool, ToolContext, register


def visible_windows() -> list[dict]:
    if sys.platform != "win32":
        return []
    ensure_default_desktop()
    from pywinauto import Desktop

    out = []
    for w in Desktop(backend="uia").windows():
        try:
            title = w.window_text()
            if title and w.is_visible():
                out.append({"title": title, "handle": w.handle, "pid": w.process_id()})
        except Exception:
            continue
    return out


class NoArgs(BaseModel):
    pass


class LaunchArgs(BaseModel):
    name: str = Field(description="Application name as a person would say it, e.g. notepad, chrome, spotify, vs code")


def find_antigravity() -> str | None:
    found = shutil.which("antigravity")
    if found:
        return found
    for base in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles", "")):
        for rel in ("Programs/Antigravity/Antigravity.exe", "Antigravity/Antigravity.exe"):
            p = Path(base) / rel
            if base and p.exists():
                return str(p)
    return None


def app_allowlist() -> dict[str, str]:
    apps = dict(get_settings().app_allowlist)
    if "antigravity" not in apps and (ag := find_antigravity()):
        apps["antigravity"] = ag
    return apps


def _list_windows(args: NoArgs, ctx: ToolContext) -> ToolResult:
    wins = visible_windows()
    return ToolResult(ok=True, summary=f"{len(wins)} visible top-level windows",
                      data={"windows": wins}, evidence=[w["title"] for w in wins[:15]])


_start_apps: tuple[float, list[dict]] = (0.0, [])


def start_apps(max_age_s: float = 600) -> list[dict]:
    """Installed apps from the Start menu (desktop and Store apps), as [{"name", "id"}]. Cached."""
    global _start_apps
    if sys.platform != "win32":
        return []
    if time.monotonic() - _start_apps[0] < max_age_s and _start_apps[1]:
        return _start_apps[1]
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"],
        capture_output=True, text=True, timeout=30,
    )
    try:
        rows = json.loads(out.stdout or "[]")
    except ValueError:
        rows = []
    apps = [{"name": r["Name"], "id": r["AppID"]} for r in (rows if isinstance(rows, list) else [rows]) if r.get("AppID")]
    _start_apps = (time.monotonic(), apps)
    return apps


def resolve_start_app(task_id: str | None, name: str) -> dict | None:
    from agent.reflex.decide import choose

    apps = start_apps()
    want = name.strip().lower()
    for test in (lambda n: n == want, lambda n: n.startswith(want), lambda n: want in n):
        hits = sorted((a for a in apps if test(a["name"].lower())), key=lambda a: len(a["name"]))
        if hits:
            return hits[0]
    # Let the reflex layer resolve nicknames ("vs code" -> "Visual Studio Code").
    options = {f"a{i}": a["name"] for i, a in enumerate(apps[:255])}
    d = choose(task_id, {"installed_apps": list(options.values())}, f"Which installed app is '{name}'?",
               options, target=name, purpose="app")
    return apps[int(d.choice[1:])] if d.confident() else None


def _allowlist_id(key: str, apps: list[dict]) -> tuple[str, str]:
    """(rule id, display name) for an ABBAS_ALLOWED_APPS entry; shares the id of the matching Start-menu app."""
    match = next((a for a in apps if a["name"].lower() == key), None)
    return (match["id"], match["name"]) if match else (f"exe:{key}", key.title())


def app_catalog() -> list[dict]:
    """Every app Abbas could open, with whether the user allows it - for the Settings page."""
    rules, access = runtime.app_rules(), runtime.get("app_access")
    any_app = runtime.get("allow_any_app")
    apps = start_apps()
    out: dict[str, dict] = {}
    for key, exe in app_allowlist().items():
        rid, name = _allowlist_id(key, apps)
        out[rid] = {"id": rid, "name": name, "source": "allowlist", "command": exe}
    if any_app:
        for a in apps:
            out.setdefault(a["id"], {"id": a["id"], "name": a["name"], "source": "start_menu", "command": None})
    items = [{**a, "allowed": runtime.app_allowed(a["id"], rules, access)} for a in out.values()]
    return sorted(items, key=lambda a: (a["source"] != "allowlist", a["name"].lower()))


def _launch(args: LaunchArgs, ctx: ToolContext) -> ToolResult:
    ensure_default_desktop()
    apps = app_allowlist()
    key = args.name.strip().lower()
    before_windows = visible_windows()
    before = {w["handle"] for w in before_windows}
    any_app = runtime.get("allow_any_app")
    if key in apps:
        rule_id, _ = _allowlist_id(key, start_apps())
        label, cmd = args.name, [apps[key]]
    elif any_app and (app := resolve_start_app(ctx.task_id, args.name)):
        rule_id, label, cmd = app["id"], app["name"], ["explorer.exe", f"shell:AppsFolder\\{app['id']}"]
    else:
        hint = "" if any_app else " (turn on 'Open any installed app' in Settings)"
        return ToolResult(ok=False, summary=f"couldn't find an installed app called '{args.name}'{hint}")
    if not runtime.app_allowed(rule_id):
        return ToolResult(ok=False, summary=f"'{label}' is switched off in Settings → Desktop apps, so I didn't open it")
    try:
        spawn_on_desktop(cmd, child_env())
    except OSError as e:
        return ToolResult(ok=False, summary=f"could not start {label}: {e}")

    # Helper to check if a window matches the target app label
    label_tokens = [t for t in label.lower().split() if len(t) >= 3] or [label.lower()]

    def is_match(w: dict) -> bool:
        title = w["title"].lower()
        return any(tok in title for tok in label_tokens)

    # Verify: a new top-level window appears, or an existing one of this app comes to the front.
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        time.sleep(0.5)
        now = visible_windows()
        new = [w for w in now if w["handle"] not in before]
        if new:
            bring_to_front(new[0]["handle"])
            return ToolResult(ok=True, summary=f"opened {label}: window '{new[0]['title']}' appeared",
                              data={"app": label, "windows": new}, evidence=[w["title"] for w in new])
        # If 2.5s elapsed and an existing matching window is found, bring it to front
        if time.monotonic() > deadline - 9.5:
            matching = [w for w in now if is_match(w)]
            if matching:
                bring_to_front(matching[0]["handle"])
                return ToolResult(ok=True, summary=f"{label} is open ('{matching[0]['title']}'); brought to front",
                                  data={"app": label, "windows": matching}, evidence=[matching[0]["title"]])

    now = visible_windows()
    matching = [w for w in now if is_match(w)] or [w for w in before_windows if is_match(w)]
    if matching:
        bring_to_front(matching[0]["handle"])
        return ToolResult(ok=True, summary=f"{label} was already open ('{matching[0]['title']}'); brought to front",
                          data={"app": label, "windows": matching}, evidence=[matching[0]["title"]])
    return ToolResult(ok=False, summary=f"started {label} but no window appeared within 12s")


register(Tool("desktop.list_windows", "List visible top-level windows on the desktop.", NoArgs, Risk.low, _list_windows))
register(Tool("desktop.launch_app", "Open any installed desktop app by name (e.g. Chrome, Spotify, Paint, VS Code) "
              "and confirm its window appears.",
              LaunchArgs, Risk.medium, _launch))
