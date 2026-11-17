"""Abbas agent API: tasks, approvals, cancellation, SSE event stream, voice endpoints.

Run: uv run uvicorn services.agent_api.main:app --port 8000
"""

import asyncio
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from alembic import command
from alembic.config import Config
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select

from agent import runtime
from agent.autonomy import get_level, set_level
from agent.tools.browser import close_session as close_browser_session
from agent.tools.desktop import app_catalog
from agent.config import ROOT, get_settings
from agent.interpreter import is_stop_command
from agent.memory import delete_memory, get_persistence, list_memories, looks_secret, save_memory
from agent.tools import tool_catalog
from agent.tools.desktop import app_allowlist
from agent import auth
from agent.user_context import as_user
from database.models import Conversation, Message, Task, TaskEvent, User
from database.session import session_scope
from voice.asr import ModelMissing, asr
from voice.language import detect_language
from voice.tts import TTSUnavailable, get_tts, speakable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
ACTIVE = ("created", "running", "awaiting_approval")
runner = None


def migrate() -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "database" / "migrations"))
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runner
    from agent.runner import TaskRunner
    from agent.desktop_env import ensure_default_desktop

    ensure_default_desktop()
    migrate()
    auth.ensure_admin()
    with session_scope() as s:  # tasks interrupted by a restart cannot resume their thread
        for t in s.scalars(select(Task).where(Task.status.in_(("created", "running")))):
            t.status, t.result = "failed", "Interrupted by a service restart."
    runner = TaskRunner()
    yield


app = FastAPI(title="Abbas Agent API", version="0.1.0", lifespan=lifespan)
# Browsers may call this API from the deployed frontend (hardcoded here, no .env needed), its Vercel
# preview URLs (abbas-<hash>-<team>.vercel.app), or a local dashboard on any port.
FRONTEND_ORIGINS = ["https://abbas-d97t.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS + [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://abbas-[a-z0-9-]+\.vercel\.app",
    allow_methods=["*"], allow_headers=["*"])


class TaskIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    source: str = "text"
    confidence: float | None = None
    conversation_id: str | None = None


class MemoryIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class AutonomyIn(BaseModel):
    level: Literal["ask", "trusted", "full"]


class SettingsIn(BaseModel):
    autonomy: Literal["ask", "trusted", "full"] | None = None
    app_access: Literal["all", "selected"] | None = None
    screen_actions: bool | None = None
    browser_headless: bool | None = None
    allow_any_app: bool | None = None
    extra_folders: list[str] | None = Field(None, max_length=50)


class AppRulesIn(BaseModel):
    rules: dict[str, bool] = Field(min_length=1, max_length=1000)


class ApprovalIn(BaseModel):
    approved: bool


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class DashiConfigIn(BaseModel):
    active_agent: str | None = None
    profiles: list[dict] | None = None
    trading_accounts: list[dict] | None = None


class DashiImportIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class DashiSwitchIn(BaseModel):
    agent: str = Field(description="'abbas' or 'dashi'")


class SignupIn(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=200)
    name: str = Field("", max_length=100)


class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=200)


def _token(request: Request) -> str | None:
    h = request.headers.get("authorization", "")
    if h.lower().startswith("bearer "):
        return h[7:].strip()
    return request.query_params.get("token")  # EventSource and <img> can't send headers


def current_user(request: Request) -> dict:
    user = auth.user_for_token(_token(request))
    if user is None:
        raise HTTPException(401, "Please log in.")
    return user


def admin_user(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(403, "Only the admin can do this.")
    return user


def scope(request: Request, user: dict = Depends(current_user)) -> dict:
    """Whose data a request reads or writes. A user: their own. The admin: everyone ("only" None), or one user
    picked with ?as_user=<id>. "own" is the user that owns what the request creates or edits."""
    picked = request.query_params.get("as_user") or None
    if user["role"] == "admin":
        return {"user": user, "only": picked, "own": picked or user["id"], "admin": True}
    return {"user": user, "only": user["id"], "own": user["id"], "admin": False}


def _conv_allowed(sc: dict, conversation_id: str | None) -> bool:
    if not conversation_id:
        return True
    with session_scope() as s:
        c = s.get(Conversation, conversation_id)
        if c is None:
            return True  # a new conversation id; it will be created for this user
        return sc["only"] is None or c.user_id == sc["only"]


def _task_allowed(sc: dict, task_id: str) -> None:
    with session_scope() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise HTTPException(404, "task not found")
        owner = s.scalar(select(Conversation.user_id).where(Conversation.id == t.conversation_id))
    if sc["only"] is not None and owner != sc["only"]:
        raise HTTPException(404, "task not found")


def _task_dict(t: Task) -> dict:
    return {"id": t.id, "conversation_id": t.conversation_id, "request": t.original_request, "source": t.source,
            "language": t.language,
            "status": t.status, "spec": t.spec, "result": t.result,
            "created_at": t.created_at.isoformat(), "updated_at": t.updated_at.isoformat()}


def _event_dict(e: TaskEvent) -> dict:
    return {"id": e.id, "task_id": e.task_id, "type": e.type, "message": e.message, "data": e.data,
            "created_at": e.created_at.isoformat()}


def _cancel_active(only: str | None = None) -> list[str]:
    with session_scope() as s:
        q = select(Task.id).where(Task.status.in_(ACTIVE))
        if only is not None:
            q = q.join(Conversation, Conversation.id == Task.conversation_id).where(Conversation.user_id == only)
        ids = list(s.scalars(q))
    for tid in ids:
        runner.cancel(tid)
    return ids


def _submit(text: str, source: str, confidence: float | None, meta: dict | None = None,
            conversation_id: str | None = None, sc: dict | None = None) -> dict:
    sc = sc or {"only": None, "own": None}
    if not _conv_allowed(sc, conversation_id):
        raise HTTPException(404, "conversation not found")
    if is_stop_command(text):
        return {"stopped": _cancel_active(sc["only"])}
    task_id, conv_id = runner.create(text, source=source, confidence=confidence, transcript_meta=meta,
                                     conversation_id=conversation_id, user_id=sc["own"])
    return {"task_id": task_id, "conversation_id": conv_id}


@app.get("/api/health")
def health():
    return {"ok": True}


# --- Accounts --------------------------------------------------------------------------------------

@app.post("/api/auth/signup")
def signup(body: SignupIn):
    try:
        user = auth.create_user(body.email, body.password, body.name)
    except auth.AuthError as e:
        raise HTTPException(400, str(e)) from e
    return {"token": auth.create_session(user["id"]), "user": user}


@app.post("/api/auth/login")
def login(body: LoginIn):
    try:
        user = auth.authenticate(body.email, body.password)
    except auth.AuthError as e:
        raise HTTPException(401, str(e)) from e
    return {"token": auth.create_session(user["id"]), "user": user}


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    return user


@app.post("/api/auth/logout")
def logout(request: Request):
    auth.end_session(_token(request))
    return {"ok": True}


@app.get("/api/admin/users")
def admin_users(_: dict = Depends(admin_user)):
    users = auth.list_users()
    with session_scope() as s:
        counts = dict(s.execute(select(Conversation.user_id, func.count()).group_by(Conversation.user_id)).all())
    return [{**u, "conversations": counts.get(u["id"], 0)} for u in users]


@app.get("/api/status")
def status(sc: dict = Depends(scope)):
    s = get_settings()
    from agent.dashi_config import load_dashi_config
    with as_user(sc["own"]):
        dashi_cfg = load_dashi_config()
        saved_memories = len(list_memories(runner.store)) if runner else 0
    active_agent = dashi_cfg.get("active_agent", "abbas")
    agent_display = "Dashi" if active_agent == "dashi" else "Abbas"
    agent_nick = "Dashi" if active_agent == "dashi" else "Abbad"
    return {
        "agent": {"name": agent_display, "nickname": agent_nick, "active": active_agent,
                  "dashi_only": get_settings().dashi_only},
        "reasoning": {"mode": "gemini" if s.gemini_enabled else "llm" if s.llm_enabled else "rules",
                      "model": s.gemini_model if s.gemini_enabled else (s.llm_model or None)},
        "reflex": {"engine": "jev" if s.jev_enabled else "gemini" if s.gemini_enabled else "local",
                   "model": s.jev_model if s.jev_enabled else None, "min_confidence": s.jev_min_confidence,
                   "images": s.jev_images},
        "actions": {"browser_headless": runtime.get("browser_headless"), "screen_actions": runtime.get("screen_actions"),
                    "autonomy": get_level(), "any_app": runtime.get("allow_any_app")},
        "memory": {"backend": get_persistence().backend, "database": "postgres" if s.is_postgres else "sqlite",
                   "saved": saved_memories},
        "asr": asr.status(),
        "tts": get_tts().status(),
        "workspaces": [str(p) for p in s.workspace_paths],
        "allowed_apps": sorted(app_allowlist()),
        "tools": [{k: t[k] for k in ("name", "description", "risk")} for t in tool_catalog()],
        "limits": {"max_steps": s.max_steps, "tool_timeout_s": s.tool_timeout_s, "max_recording_s": s.max_recording_s,
                   "asr_min_confidence": s.asr_min_confidence},
    }


@app.post("/api/tasks")
def create_task(body: TaskIn, sc: dict = Depends(scope)):
    return _submit(body.text.strip(), body.source, body.confidence, conversation_id=body.conversation_id, sc=sc)


@app.get("/api/tasks")
def list_tasks(limit: int = 50, conversation_id: str | None = None, sc: dict = Depends(scope)):
    with session_scope() as s:
        q = select(Task).order_by(Task.created_at.desc()).limit(limit)
        if sc["only"] is not None:
            q = q.join(Conversation, Conversation.id == Task.conversation_id).where(Conversation.user_id == sc["only"])
        if conversation_id:
            q = q.where(Task.conversation_id == conversation_id)
        return [_task_dict(t) for t in s.scalars(q)]


# --- Conversations (each one is a LangGraph thread = short-term memory) ---------------------------

@app.get("/api/conversations")
def list_conversations(limit: int = 50, sc: dict = Depends(scope)):
    with session_scope() as s:
        counts = dict(s.execute(select(Task.conversation_id, func.count()).group_by(Task.conversation_id)).all())
        q = select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
        if sc["only"] is not None:
            q = q.where(Conversation.user_id == sc["only"])
        convs = list(s.scalars(q))
        emails = dict(s.execute(select(User.id, User.email)).all()) if sc["admin"] else {}
        return [{"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat(), "tasks": counts.get(c.id, 0),
                 **({"owner": emails.get(c.user_id, "(no owner)")} if sc["admin"] else {})}
                for c in convs]


@app.post("/api/conversations")
def new_conversation(sc: dict = Depends(scope)):
    with session_scope() as s:
        c = Conversation(id=str(uuid.uuid4()), user_id=sc["own"])
        s.add(c)
        s.flush()  # fills the column defaults (updated_at)
        return {"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat(), "tasks": 0}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str, sc: dict = Depends(scope)):
    with session_scope() as s:
        c = s.get(Conversation, conversation_id)
        if c is None or (sc["only"] is not None and c.user_id != sc["only"]):
            raise HTTPException(404, "conversation not found")
        out = {"id": c.id, "title": c.title}
    return {**out, **runner.thread_state(conversation_id)}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, sc: dict = Depends(scope)):
    with session_scope() as s:
        c = s.get(Conversation, conversation_id)
        if c is None or (sc["only"] is not None and c.user_id != sc["only"]):
            raise HTTPException(404, "conversation not found")
        task_ids = list(s.scalars(select(Task.id).where(Task.conversation_id == conversation_id)))
        if s.scalar(select(func.count()).where(Task.conversation_id == conversation_id, Task.status.in_(ACTIVE))):
            raise HTTPException(409, "stop the running task before deleting this conversation")
        s.execute(delete(TaskEvent).where(TaskEvent.task_id.in_(task_ids)))
        s.execute(delete(Message).where(Message.conversation_id == conversation_id))
        s.execute(delete(Task).where(Task.conversation_id == conversation_id))
        s.execute(delete(Conversation).where(Conversation.id == conversation_id))
    get_persistence().checkpointer.delete_thread(conversation_id)  # LangGraph short-term memory
    for tid in task_ids:
        for shot in (get_settings().data_dir / "screenshots").glob(f"{tid}_*.png"):
            shot.unlink(missing_ok=True)
    return {"ok": True}


# --- Autonomy -------------------------------------------------------------------------------------

@app.get("/api/settings/autonomy")
def autonomy(_: dict = Depends(current_user)):
    return {"level": get_level()}


@app.put("/api/settings/autonomy")
def update_autonomy(body: AutonomyIn, _: dict = Depends(admin_user)):
    return {"level": set_level(body.level)}


def _settings_view() -> dict:
    s = get_settings()
    asr_desc = f"ElevenLabs ({s.elevenlabs_model_id})" if s.asr_engine == "elevenlabs" else f"Whisper {s.asr_model}"
    tts_desc = f"ElevenLabs ({s.tts_voice})" if s.tts_engine == "elevenlabs" else f"{s.tts_engine} · voice {s.tts_voice}"
    return {
        "editable": runtime.get_runtime(),
        "fixed": {  # set in .env; shown read-only on the Settings page
            "workspaces": [str(p) for p in s.workspace_paths],
            "browser_channel": s.browser_channel or "bundled chromium",
            "gemini_model": s.gemini_model if s.gemini_enabled else None,
            "jev_model": s.jev_model if s.jev_enabled else None,
            "database": "postgres" if s.is_postgres else "sqlite",
            "asr_engine": s.asr_engine,
            "asr_model": s.asr_model,
            "tts_engine": s.tts_engine,
            "tts_voice": s.tts_voice,
            "speech_summary": f"{asr_desc} · {tts_desc}",
        },
    }


@app.get("/api/settings")
def read_settings(_: dict = Depends(current_user)):
    return _settings_view()


@app.patch("/api/settings")
def change_settings(body: SettingsIn, _: dict = Depends(admin_user)):
    changes = body.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(422, "nothing to change")
    before = runtime.get_runtime()
    try:
        runtime.update(**changes)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if "browser_headless" in changes and changes["browser_headless"] != before["browser_headless"]:
        close_browser_session()  # the next browser step opens in the new mode
    return _settings_view()


@app.get("/api/apps")
async def list_apps(_: dict = Depends(current_user)):
    apps = await run_in_threadpool(app_catalog)
    return {"access": runtime.get("app_access"), "any_app": runtime.get("allow_any_app"),
            "allowed": sum(a["allowed"] for a in apps), "total": len(apps), "apps": apps}


@app.put("/api/apps")
def change_apps(body: AppRulesIn, _: dict = Depends(admin_user)):
    runtime.set_app_rules(body.rules)
    return {"ok": True, "changed": len(body.rules)}


# --- Dashi (ShowMyTrades Agent) -----------------------------------------------------------------

@app.get("/api/dashi")
def get_dashi(sc: dict = Depends(scope)):
    from agent.dashi_config import load_dashi_config
    return load_dashi_config(sc["own"])


@app.put("/api/dashi")
def update_dashi(body: DashiConfigIn, sc: dict = Depends(scope)):
    from agent.dashi_config import load_dashi_config, save_dashi_config
    cfg = load_dashi_config(sc["own"])
    data = body.model_dump(exclude_none=True)
    cfg.update(data)
    save_dashi_config(cfg, sc["own"])
    return load_dashi_config(sc["own"])


@app.post("/api/dashi/accounts/import")
def import_dashi_accounts(body: DashiImportIn, sc: dict = Depends(scope)):
    from agent.dashi_config import load_dashi_config, parse_accounts_text, upsert_trading_accounts
    parsed = parse_accounts_text(body.text)
    with as_user(sc["own"]):
        added, updated = upsert_trading_accounts(parsed)
        return {"found": len(parsed), "added": added, "updated": updated, "config": load_dashi_config()}


@app.post("/api/dashi/switch")
def switch_dashi(body: DashiSwitchIn, _: dict = Depends(admin_user)):
    from agent.dashi_config import set_active_agent
    try:
        name = set_active_agent(body.agent)
    except ValueError as e:  # unknown agent, or Abbas while ABBAS_DASHI_ONLY is on
        raise HTTPException(400, str(e)) from e
    return {"active_agent": name}


# --- Long-term memory (LangGraph store) ----------------------------------------------------------

@app.get("/api/memories")
def memories(sc: dict = Depends(scope)):
    with as_user(sc["own"]):
        return list_memories(runner.store)


@app.post("/api/memories")
def add_memory(body: MemoryIn, sc: dict = Depends(scope)):
    if looks_secret(body.text):
        raise HTTPException(422, "That looks like a password, key or card number; Abbas does not store those.")
    with as_user(sc["own"]):
        return save_memory(runner.store, body.text)


@app.delete("/api/memories/{memory_id}")
def remove_memory(memory_id: str, sc: dict = Depends(scope)):
    with as_user(sc["own"]):
        found = delete_memory(runner.store, memory_id)
    if not found:
        raise HTTPException(404, "memory not found")
    return {"ok": True}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str, sc: dict = Depends(scope)):
    _task_allowed(sc, task_id)
    with session_scope() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise HTTPException(404, "task not found")
        events = s.scalars(select(TaskEvent).where(TaskEvent.task_id == task_id).order_by(TaskEvent.id))
        return {**_task_dict(t), "events": [_event_dict(e) for e in events]}


@app.post("/api/tasks/{task_id}/approve")
def approve(task_id: str, body: ApprovalIn, sc: dict = Depends(scope)):
    _task_allowed(sc, task_id)
    with session_scope() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise HTTPException(404, "task not found")
        if t.status != "awaiting_approval":
            raise HTTPException(409, f"task is {t.status}, not awaiting approval")
        t.status = "running"  # claim it so a double click cannot resume twice
    runner.resolve_approval(task_id, body.approved)
    return {"ok": True}


@app.post("/api/tasks/{task_id}/cancel")
def cancel(task_id: str, sc: dict = Depends(scope)):
    _task_allowed(sc, task_id)
    runner.cancel(task_id)
    return {"ok": True}


@app.post("/api/stop")
def stop_all(sc: dict = Depends(scope)):
    return {"stopped": _cancel_active(sc["only"])}


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str, sc: dict = Depends(scope)):
    _task_allowed(sc, task_id)
    with session_scope() as s:
        t = s.get(Task, task_id)
        if t is None:
            raise HTTPException(404, "task not found")
        if t.status in ACTIVE:
            raise HTTPException(409, "cancel the task before deleting it")
        conv = t.conversation_id
        if conv and s.scalar(select(func.count()).where(Task.conversation_id == conv, Task.status.in_(ACTIVE))):
            raise HTTPException(409, "wait until this conversation's current task finishes")
        s.execute(delete(TaskEvent).where(TaskEvent.task_id == task_id))
        s.execute(delete(Message).where(Message.task_id == task_id))
        s.delete(t)
    if conv:
        runner.forget_task(task_id, conv)
    for shot in (get_settings().data_dir / "screenshots").glob(f"{task_id}_*.png"):
        shot.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/events/stream")
async def stream(after: int = 0, sc: dict = Depends(scope)):
    """Server-sent events built from the persisted event log, limited to the signed-in user's tasks."""
    only = sc["only"]

    async def gen():
        last = after
        if last == 0:
            with session_scope() as s:
                last = s.scalar(select(TaskEvent.id).order_by(TaskEvent.id.desc()).limit(1)) or 0
        idle = 0
        while True:
            with session_scope() as s:
                q = select(TaskEvent).where(TaskEvent.id > last).order_by(TaskEvent.id).limit(200)
                rows = [_event_dict(e) for e in s.scalars(q)]
                mine = None
                if only is not None and rows:
                    mine = set(s.scalars(select(Task.id).join(Conversation, Conversation.id == Task.conversation_id)
                                         .where(Conversation.user_id == only,
                                                Task.id.in_({r["task_id"] for r in rows}))))
            for r in rows:
                last = r["id"]
                if mine is not None and r["task_id"] not in mine:
                    continue
                yield f"id: {r['id']}\nevent: task_event\ndata: {json.dumps(r)}\n\n"
            idle = 0 if rows else idle + 1
            if idle and idle % 120 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.1)  # events reach the dashboard within ~100 ms

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/screenshots/{name}")
def screenshot(name: str, sc: dict = Depends(scope)):
    if not re.fullmatch(r"[0-9a-f-]{36}_[\w-]+\.png", name):
        raise HTTPException(400, "bad name")
    _task_allowed(sc, name[:36])
    path = get_settings().data_dir / "screenshots" / name
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/voice/transcribe")
async def transcribe(audio: UploadFile = File(...), submit: bool = Form(False),
                     conversation_id: str | None = Form(None), sc: dict = Depends(scope)):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "recording too large")
    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    try:
        tr = await run_in_threadpool(asr.transcribe_bytes, data, suffix)
    except ModelMissing as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(422, f"could not transcribe audio: {type(e).__name__}: {e}")
    out = {"transcript": tr.to_dict(), "language_guess": detect_language(tr.text).language}
    if not tr.text:
        out["note"] = "No speech detected."
    elif submit:
        meta = {"confidence": tr.confidence, "language": tr.language, "language_probability": tr.language_probability,
                "duration_s": tr.duration_s, "elapsed_s": tr.elapsed_s, "segments": tr.segments}
        out.update(_submit(tr.text, "voice", tr.confidence, meta, conversation_id=conversation_id, sc=sc))
    return out


@app.post("/api/voice/speak")
async def speak(body: SpeakIn, _: dict = Depends(current_user)):
    tts = get_tts()
    try:
        wav = await run_in_threadpool(tts.synthesize, speakable(body.text))
    except TTSUnavailable as e:
        raise HTTPException(503, str(e))
    return Response(wav, media_type="audio/wav")
