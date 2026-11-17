"""Settings the user changes from the dashboard while Abbas runs (persisted in the database, app_config).

Values not set here fall back to the .env defaults in agent.config. Also holds the per-app rules
for desktop.launch_app:

  app_access = "all"       every installed app may be opened, except ones switched off
  app_access = "selected"  only apps switched on may be opened
"""

import logging
import threading
from pathlib import Path
from typing import Any, Literal

from agent.config import get_settings

log = logging.getLogger("abbas.runtime")
_lock = threading.Lock()

Autonomy = Literal["ask", "trusted", "full"]
AppAccess = Literal["all", "selected"]
EDITABLE = {
    "autonomy": ("ask", "trusted", "full"),
    "app_access": ("all", "selected"),
    "screen_actions": bool,
    "browser_headless": bool,
    "allow_any_app": bool,
    "extra_folders": list,
}


def _protected(path: Path) -> bool:
    """System locations that can never be granted, whatever the user adds."""
    import os

    p = path.resolve()
    windir = Path(os.environ.get("SystemRoot", "C:/Windows")).resolve()
    system_drive = Path(os.environ.get("SystemDrive", "C:") + "/").resolve()
    blocked = [windir, Path(os.environ.get("ProgramFiles", "C:/Program Files")),
               Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")),
               Path(os.environ.get("ProgramData", "C:/ProgramData"))]
    if p == system_drive:  # all of C:\ is too broad; C:\Users\you\Projects is fine
        return True
    return any(p == b.resolve() or b.resolve() in p.parents for b in blocked)


def validate_folders(folders: list) -> list[str]:
    out = []
    for f in folders:
        if not isinstance(f, str) or not f.strip():
            raise ValueError("folders must be non-empty paths")
        p = Path(f.strip()).expanduser()
        if not p.is_absolute():
            raise ValueError(f"'{f}' must be a full path like D:\\Projects")
        if _protected(p):
            raise ValueError(f"'{f}' is a protected system location and can't be given to Abbas")
        out.append(str(p.resolve()))
    return list(dict.fromkeys(out))


SETTINGS_KEY = "settings"  # table app_config; legacy file data/settings.json is imported once


def _read() -> dict:
    from agent import config_store

    return config_store.load(SETTINGS_KEY) or {}


def _write(data: dict) -> None:
    from agent import config_store

    config_store.save(SETTINGS_KEY, data)


def _defaults() -> dict:
    s = get_settings()
    return {"autonomy": s.autonomy if s.autonomy in EDITABLE["autonomy"] else "ask", "app_access": "all",
            "screen_actions": s.screen_actions, "browser_headless": s.browser_headless,
            "allow_any_app": s.allow_any_app, "extra_folders": []}


def get_runtime() -> dict:
    data = _read()
    return {**_defaults(), **{k: v for k, v in data.items() if k in EDITABLE}}


def get(key: str) -> Any:
    return get_runtime()[key]


def update(**changes: Any) -> dict:
    for key, value in changes.items():
        rule = EDITABLE.get(key)
        if rule is None:
            raise ValueError(f"'{key}' cannot be changed at runtime")
        if rule is bool and not isinstance(value, bool):
            raise ValueError(f"'{key}' must be true or false")
        if rule is list:
            if not isinstance(value, list):
                raise ValueError(f"'{key}' must be a list")
            changes[key] = value = validate_folders(value)
        if isinstance(rule, tuple) and value not in rule:
            raise ValueError(f"'{key}' must be one of {rule}")
    with _lock:
        data = _read()
        data.update(changes)
        _write(data)
    log.warning("runtime settings changed: %s", changes)
    return get_runtime()


# --- desktop app rules ------------------------------------------------------------------------------

def app_rules() -> dict[str, bool]:
    return dict(_read().get("apps", {}))


def set_app_rules(rules: dict[str, bool]) -> dict[str, bool]:
    with _lock:
        data = _read()
        apps = data.setdefault("apps", {})
        apps.update({k: bool(v) for k, v in rules.items()})
        _write(data)
    return app_rules()


def app_allowed(app_id: str, rules: dict[str, bool] | None = None, access: str | None = None) -> bool:
    rules = app_rules() if rules is None else rules
    if app_id in rules:
        return rules[app_id]
    return (access or get("app_access")) == "all"
