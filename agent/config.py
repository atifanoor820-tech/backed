"""Central configuration, read from environment / .env (prefix ABBAS_)."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
# Serverless hosts (Vercel sets VERCEL=1) ship the code read-only; only /tmp is writable, and it is wiped
# between instances. Use ABBAS_DATABASE_URL (Postgres, e.g. Neon) there so tasks and accounts persist.
ON_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
WRITABLE_ROOT = Path("/tmp/abbas") if ON_SERVERLESS else ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ABBAS_", env_file=ROOT / ".env", extra="ignore")

    # Storage
    data_dir: Path = WRITABLE_ROOT / "data"
    database_url: str = ""  # defaults to sqlite in data_dir

    # Folders the filesystem tools may touch. Semicolon separated.
    workspaces: str = str(WRITABLE_ROOT / "workspace")

    # Brain: Gemini plans, writes replies and reads screenshots when needed.
    gemini_api_key: str = ""
    # All free-tier models. 3.6 Flash plans; 3.5 Flash-Lite (~1 s) makes quick decisions when Jev is absent.
    gemini_model: str = "gemini-3.6-flash"
    gemini_fast_model: str = "gemini-3.5-flash-lite"
    # Tried in order when a model is out of free quota (429), overloaded (503) or too slow.
    gemini_fallback_models: str = "gemini-3.5-flash-lite,gemini-3.8-flash"
    gemini_timeout_s: float = 25.0
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Reflexes: Jev (TypeSafe) answers typed questions - which element, is it risky, did it work.
    jev_api_key: str = ""
    jev_model: str = "jev-1.14.0"  # pinned, because confidence thresholds are tuned against a version
    jev_base_url: str = "https://api.typesafe.ai/v1"
    jev_min_confidence: float = 0.6
    jev_risk_threshold: float = 0.35  # P(hard to undo) above this raises a step to high risk
    jev_images: bool = False  # opt in to Jev's experimental "box" questions on screenshots

    # Optional alternative brain: any OpenAI-compatible endpoint (used only if Gemini is not configured).
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = ""

    # Action layer
    browser_headless: bool = False  # show the browser Abbas drives
    # Installed browser to drive (fresh profile, your own data untouched): msedge | chrome | "" = bundled Chromium
    browser_channel: str = "msedge"
    # Remote browser for servers with no Chrome (Vercel): a Chrome DevTools websocket URL, e.g.
    # wss://production-sfo.browserless.io?token=YOUR_TOKEN. When set, Dashi connects to it instead of launching one.
    browser_ws_endpoint: str = ""
    # On serverless hosts a task runs inside its HTTP request; stop waiting a bit before the host's limit (300 s).
    serverless_wait_s: float = 280.0
    screen_actions: bool = True  # allow PyAutoGUI mouse/keyboard (always approval-gated)

    # Apps the agent may launch: "name=path;name2=path2"
    allowed_apps: str = "notepad=notepad.exe;calculator=calc.exe;explorer=explorer.exe"

    # ask | trusted | full - default only; the dashboard setting (data/settings.json) wins
    autonomy: str = "ask"
    # Let desktop.launch_app open any installed Start-menu app, not just ABBAS_ALLOWED_APPS
    allow_any_app: bool = True

    # Execution limits
    max_steps: int = 8  # steps per planning round
    max_rounds: int = 10  # plan -> act -> review cycles for one goal
    max_total_steps: int = 40
    history_turns: int = 10  # conversation turns given to the planner
    user_id: str = "default"  # namespace for long-term memories
    tool_timeout_s: float = 45.0

    # Voice
    asr_engine: str = "whisper"  # whisper | elevenlabs
    asr_model: str = "base"  # tiny | base | small | medium | large-v3
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_language: str = ""  # empty = auto-detect
    asr_min_confidence: float = 0.55
    tts_engine: str = "kokoro"  # kokoro | piper | elevenlabs | none
    tts_voice: str = "am_michael"
    tts_speed: float = 1.0
    piper_model_path: str = ""
    max_recording_s: int = 30

    # ElevenLabs (Cloud TTS and STT)
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"  # George (premade free plan voice)
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    elevenlabs_base_url: str = "https://api.elevenlabs.io/v1"

    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    # ABBAS_DASHI_ONLY: only Dashi (ShowMyTrades accounts) runs; Abbas desktop commands are turned off.
    dashi_only: bool = True

    @property
    def db_url(self) -> str:
        """SQLAlchemy URL. postgresql://user:pass@host/db is mapped to the psycopg 3 driver."""
        url = self.database_url or f"sqlite:///{(self.data_dir / 'abbas.db').as_posix()}"
        if url.startswith(("postgresql://", "postgres://")):
            url = "postgresql+psycopg://" + url.split("://", 1)[1]
        return url

    @property
    def is_postgres(self) -> bool:
        return self.db_url.startswith("postgresql")

    @property
    def pg_conninfo(self) -> str:
        """Plain libpq URL for psycopg / LangGraph (no SQLAlchemy driver suffix)."""
        return "postgresql://" + self.db_url.split("://", 1)[1]

    @property
    def workspace_paths(self) -> list[Path]:
        return [Path(p).expanduser().resolve() for p in self.workspaces.split(";") if p.strip()]

    @property
    def app_allowlist(self) -> dict[str, str]:
        pairs = (item.split("=", 1) for item in self.allowed_apps.split(";") if "=" in item)
        return {k.strip().lower(): v.strip() for k, v in pairs}

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_model)

    @property
    def jev_enabled(self) -> bool:
        return bool(self.jev_api_key and self.jev_model)

    @property
    def llm_enabled(self) -> bool:
        return self.gemini_enabled or bool(self.llm_api_key and self.llm_model)


_SERVER_ENV = {"PORT", "HOST", "UVICORN_PORT", "UVICORN_HOST", "WEB_CONCURRENCY"}


def child_env(**extra: str) -> dict[str, str]:
    """Environment for programs Abbas launches: without the API server's own PORT/HOST (a launched IDE once
    passed PORT=8000 on to `next dev`, which then took over Abbas's port) and without any ABBAS_* secrets."""
    env = {k: v for k, v in os.environ.items() if k.upper() not in _SERVER_ENV and not k.upper().startswith("ABBAS_")}
    env.update(extra)
    return env


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    for ws in s.workspace_paths:
        ws.mkdir(parents=True, exist_ok=True)
    return s
