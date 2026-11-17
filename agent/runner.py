"""Runs graph invocations off the request thread and exposes start / approve / cancel.

Each conversation is one LangGraph thread (thread_id = conversation id), so the checkpointer carries
the conversation's history and working context from one request to the next. Turns of the same
conversation run strictly in order on that conversation's own worker; different conversations run
in parallel.
"""

import concurrent.futures
import logging
import sys
import threading
import uuid

from langgraph.types import Command
from sqlalchemy import select

from agent.desktop_env import ensure_default_desktop

from agent.events import add_message, cancellations, emit, set_status
from agent.graph.workflow import build_graph
from agent.memory import get_persistence
from database.models import Conversation, Message, Task, TaskEvent, utcnow
from database.session import session_scope

log = logging.getLogger("abbas.runner")

# Fields that describe one turn; reset on every new request so nothing leaks from the previous one.
TURN_RESET = {"spec": None, "step_index": 0, "results": [], "attempts": 0, "rounds": 0, "status": "running", "report": ""}


class TaskRunner:
    def __init__(self, checkpointer=None, store=None) -> None:
        if checkpointer is None or store is None:
            p = get_persistence()
            checkpointer, store = checkpointer or p.checkpointer, store or p.store
        self.store = store
        self.graph = build_graph(checkpointer, store)
        self._workers: dict[str, concurrent.futures.ThreadPoolExecutor] = {}
        self._lock = threading.Lock()

    def _worker(self, thread_id: str) -> concurrent.futures.ThreadPoolExecutor:
        with self._lock:
            if thread_id not in self._workers:
                self._workers[thread_id] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"abbas-conv-{thread_id[:8]}")
            return self._workers[thread_id]

    @staticmethod
    def _thread_of(task_id: str) -> str:
        with session_scope() as s:
            t = s.get(Task, task_id)
            return (t.conversation_id if t and t.conversation_id else task_id)

    def _invoke(self, task_id: str, thread_id: str, payload, starting: bool = False) -> None:
        if sys.platform == "win32":
            ensure_default_desktop()
        try:
            if starting:
                if cancellations.is_cancelled(task_id):
                    set_status(task_id, "cancelled", result="Cancelled before it started.")
                    emit(task_id, "task_cancelled", "Cancelled before it started.")
                    return
                emit(task_id, "task_started")
                set_status(task_id, "running")
            self.graph.invoke(payload, {"configurable": {"thread_id": thread_id}, "recursion_limit": 50})
        except Exception as e:
            log.exception("task %s crashed", task_id)
            msg = f"Internal error: {type(e).__name__}: {e}"
            set_status(task_id, "failed", result=msg)
            emit(task_id, "task_failed", msg)

    def ensure_conversation(self, conversation_id: str | None, first_request: str) -> str:
        with session_scope() as s:
            conv = s.get(Conversation, conversation_id) if conversation_id else None
            if conv is None:
                conv = Conversation(id=conversation_id or str(uuid.uuid4()), title=first_request[:80])
                s.add(conv)
            elif conv.title == "New conversation":
                conv.title = first_request[:80]
            conv.updated_at = utcnow()
            return conv.id

    def create(self, request: str, source: str = "text", confidence: float | None = None,
               transcript_meta: dict | None = None, conversation_id: str | None = None) -> tuple[str, str]:
        task_id = str(uuid.uuid4())
        with session_scope() as s:
            conv = s.get(Conversation, conversation_id) if conversation_id else None
            if conv is None:
                conv = Conversation(id=conversation_id or str(uuid.uuid4()), title=request[:80])
                s.add(conv)
            elif conv.title == "New conversation":
                conv.title = request[:80]
            conv.updated_at = utcnow()
            conv_id = conv.id

            pending = list(s.scalars(select(Task.id).where(
                Task.conversation_id == conv_id, Task.status == "awaiting_approval")))
            for old in pending:
                s.add(TaskEvent(task_id=old, type="approval_rejected", message="Superseded by a new message in this conversation"))
                old_task = s.get(Task, old)
                if old_task:
                    old_task.status = "cancelled"
                    old_task.result = "Superseded by a new message"
                cancellations.cancel(old)

            s.add(Task(id=task_id, conversation_id=conv_id, original_request=request, source=source, status="created"))
            s.flush()
            s.add(Message(task_id=task_id, conversation_id=conv_id, role="user", content=request))
            s.add(TaskEvent(task_id=task_id, type="task_created", message=request, data={"source": source, "conversation_id": conv_id}))
            if transcript_meta:
                s.add(TaskEvent(task_id=task_id, type="transcription_completed", message=request, data=transcript_meta))

        self._worker(conv_id).submit(self._invoke, task_id, conv_id, {
            **TURN_RESET, "conversation_id": conv_id, "task_id": task_id, "request": request,
            "source": source, "transcript_confidence": confidence,
        }, True)
        return task_id, conv_id

    def resolve_approval(self, task_id: str, approved: bool) -> None:
        thread_id = self._thread_of(task_id)
        self._worker(thread_id).submit(self._invoke, task_id, thread_id, Command(resume={"approved": approved}))

    def cancel(self, task_id: str) -> None:
        cancellations.cancel(task_id)
        with session_scope() as s:
            task = s.get(Task, task_id)
            status = task.status if task else None
        if status == "awaiting_approval":
            # The graph is paused at the approval interrupt; resume it with a rejection.
            self.resolve_approval(task_id, approved=False)
        elif status in ("created", "running"):
            emit(task_id, "task_cancelled", "Cancellation requested; stopping after the current step")

    def forget_task(self, task_id: str, conversation_id: str) -> None:
        """Remove a deleted task's turns from the conversation's LangGraph memory."""
        config = {"configurable": {"thread_id": conversation_id}}
        history = (self.graph.get_state(config).values or {}).get("history")
        if history:
            kept = [t for t in history if t.get("task_id") != task_id]
            if len(kept) != len(history):
                # as the last node, so nothing is left pending on the thread
                self.graph.update_state(config, {"history": {"replace": kept}}, as_node="announce")

    def thread_state(self, conversation_id: str) -> dict:
        snap = self.graph.get_state({"configurable": {"thread_id": conversation_id}})
        values = snap.values or {}
        return {"history": values.get("history", []), "context": values.get("context", {})}
