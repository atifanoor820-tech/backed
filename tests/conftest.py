import os
import sys
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="abbas-test-"))
os.environ["ABBAS_DATA_DIR"] = str(_tmp / "data")
os.environ["ABBAS_WORKSPACES"] = str(_tmp / "ws")
os.environ["ABBAS_LLM_API_KEY"] = ""
os.environ["ABBAS_GEMINI_API_KEY"] = ""
os.environ["ABBAS_JEV_API_KEY"] = ""
os.environ["ABBAS_BROWSER_HEADLESS"] = "true"
os.environ["ABBAS_DASHI_ONLY"] = "false"  # the suite covers Abbas too; tests/test_dashi_only.py turns it on
# Run the whole suite against PostgreSQL with: ABBAS_TEST_DATABASE_URL=postgresql://... uv run pytest
os.environ["ABBAS_DATABASE_URL"] = os.environ.get("ABBAS_TEST_DATABASE_URL", "")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from database.models import Base  # noqa: E402
from database.session import engine  # noqa: E402

Base.metadata.create_all(engine)


@pytest.fixture
def workspace() -> Path:
    return _tmp / "ws"


@pytest.fixture
def settings(monkeypatch):
    """Settings object whose fields a test may override (restored afterwards)."""
    from agent.config import get_settings

    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
        return s

    return set_
