"""Entry point for hosts that look for `main.py` with an `app` (Vercel's FastAPI preset).

Locally: uv run uvicorn services.agent_api.main:app --port 8000
"""

from services.agent_api.main import app

__all__ = ["app"]
