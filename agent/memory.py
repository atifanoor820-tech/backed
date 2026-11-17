"""LangGraph persistence: short-term memory (checkpointer) and long-term memory (store).

* Short-term: every conversation is a LangGraph thread (thread_id = conversation id). The checkpointer
  keeps that thread's state - recent turns and the working context (last page, window, file) - so a
  follow-up such as "us page par click karo" can be resolved.
* Long-term: facts the user asks Abbas to remember live in the LangGraph store under
  ("memories", user_id) and are shown to the planner in every conversation.

Both use PostgreSQL when ABBAS_DATABASE_URL points at Postgres, otherwise local SQLite files.
"""

import logging
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from langgraph.store.base import BaseStore

from agent.config import get_settings

log = logging.getLogger("abbas.memory")


@dataclass
class Persistence:
    checkpointer: object
    store: BaseStore
    backend: str  # postgres | sqlite


_lock = threading.Lock()
_persistence: Persistence | None = None


def _build() -> Persistence:
    s = get_settings()
    if s.is_postgres:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(s.pg_conninfo, min_size=1, max_size=10, open=True,
                              kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        saver, store = PostgresSaver(pool), PostgresStore(pool)
        saver.setup()
        store.setup()
        return Persistence(saver, store, "postgres")

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.sqlite import SqliteStore

    saver = SqliteSaver(sqlite3.connect(s.data_dir / "checkpoints.db", check_same_thread=False))
    store = SqliteStore(sqlite3.connect(s.data_dir / "memory.db", check_same_thread=False, isolation_level=None))
    store.setup()
    return Persistence(saver, store, "sqlite")


def get_persistence() -> Persistence:
    global _persistence
    with _lock:
        if _persistence is None:
            _persistence = _build()
        return _persistence


def _ns() -> tuple[str, str]:
    return ("memories", get_settings().user_id)


def list_memories(store: BaseStore, limit: int = 100) -> list[dict]:
    items = store.search(_ns(), limit=limit)
    return sorted(({"id": i.key, **i.value} for i in items), key=lambda m: m.get("created_at", ""))


_SECRET_RE = re.compile(
    r"(password|passwd|pass ?word|api[ _-]?key|secret|token|otp|pin code|cvv|card number|\bpin\b)"
    r"|\b(?:\d[ -]?){13,19}\b"  # card-like numbers
    r"|\b(AIza[\w-]{20,}|AQ\.[\w-]{20,}|sk-[\w-]{20,}|ghp_\w{20,})", re.I)


def looks_secret(text: str) -> bool:
    return bool(_SECRET_RE.search(text))


def save_memory(store: BaseStore, text: str, task_id: str | None = None) -> dict | None:
    """Store a fact. Returns None for empty text or anything that looks like a credential."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text or looks_secret(text):
        return None
    existing = {m["text"].lower(): m for m in list_memories(store)}
    if text.lower() in existing:
        return existing[text.lower()]
    item = {"text": text[:500], "task_id": task_id, "created_at": datetime.now(timezone.utc).isoformat()}
    key = str(uuid.uuid4())
    store.put(_ns(), key, item)
    return {"id": key, **item}


def delete_memory(store: BaseStore, key: str) -> bool:
    if store.get(_ns(), key) is None:
        return False
    store.delete(_ns(), key)
    return True


def find_memory(store: BaseStore, description: str) -> dict | None:
    """Best local match for 'forget X' when no model chose an id."""
    words = set(re.findall(r"\w+", description.lower()))
    best, score = None, 0.0
    for m in list_memories(store):
        mw = set(re.findall(r"\w+", m["text"].lower()))
        s = len(words & mw) / max(len(words), 1)
        if s > score:
            best, score = m, s
    return best if score >= 0.5 else None
