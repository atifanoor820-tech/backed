"""Prepare PostgreSQL for Abbas: create the database if needed, run migrations, set up LangGraph memory tables.

1. Put your connection URL in .env (never commit it):
       ABBAS_DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5432/abbas
2. Run:
       uv run python scripts/setup_postgres.py
"""

import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import ROOT, get_settings  # noqa: E402


def main() -> int:
    s = get_settings()
    if not s.is_postgres:
        print("ABBAS_DATABASE_URL is not a postgresql:// URL. Add it to .env first (see this script's docstring).")
        return 1

    import psycopg
    from psycopg import sql

    parts = urlsplit(s.pg_conninfo)
    dbname = parts.path.lstrip("/") or "abbas"
    admin_url = urlunsplit(parts._replace(path="/postgres"))
    safe_target = f"{parts.hostname}:{parts.port or 5432}/{dbname}"

    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=10) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
            if exists:
                print(f"Database '{dbname}' already exists on {parts.hostname}.")
            else:
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
                print(f"Created database '{dbname}'.")
    except psycopg.OperationalError as e:
        print(f"Could not connect to {safe_target}: {str(e).splitlines()[0]}")
        print("Check the password/port in ABBAS_DATABASE_URL and that the PostgreSQL service is running.")
        return 1

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "database" / "migrations"))
    command.upgrade(cfg, "head")
    print("App tables migrated (conversations, tasks, task_events, messages).")

    from agent.memory import get_persistence

    p = get_persistence()
    print(f"LangGraph memory ready on {p.backend}: checkpoints (per-conversation) + store (long-term memories).")
    print(f"Done: Abbas will use {safe_target}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
