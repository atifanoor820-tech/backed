"""JSON documents kept in the database (table app_config), keyed by name.

Replaces data/*.json files so Dashi's profiles/accounts and the dashboard settings live in Postgres.
On first read, a matching legacy file in data/ is imported once, so nothing saved before is lost.
"""

import json
from pathlib import Path
from typing import Any

from agent.config import get_settings
from database.models import AppConfig
from database.session import session_scope


def _legacy_file(key: str) -> Path:
    return get_settings().data_dir / f"{key}.json"


def load(key: str) -> dict[str, Any] | None:
    with session_scope() as s:
        row = s.get(AppConfig, key)
        if row is not None:
            return dict(row.value)
    legacy = _legacy_file(key)
    if legacy.is_file():
        try:
            data = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        save(key, data)
        return data
    return None


def delete(key: str) -> None:
    with session_scope() as s:
        row = s.get(AppConfig, key)
        if row is not None:
            s.delete(row)


def save(key: str, value: dict[str, Any]) -> dict[str, Any]:
    with session_scope() as s:
        row = s.get(AppConfig, key)
        if row is None:
            s.add(AppConfig(key=key, value=value))
        else:
            row.value = value
    return value
