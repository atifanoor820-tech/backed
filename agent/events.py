"""Persisted task events. The dashboard stream is built only from these rows."""

import logging
import threading

from sqlalchemy import select

from database.models import Message, Task, TaskEvent
from database.session import session_scope

log = logging.getLogger("abbas.events")

EVENT_TYPES = {
    "task_created", "task_started", "listening_started", "transcription_completed", "intent_resolved",
    "planning_started", "planning_completed", "research_started", "research_completed", "tool_started",
    "tool_completed", "tool_failed", "file_created", "file_updated", "screenshot_captured",
    "approval_required", "approval_granted", "approval_rejected", "test_started", "test_completed",
    "task_blocked", "task_cancelled", "task_completed", "task_failed", "decision_made",
    "memory_saved", "memory_deleted", "tool_output", "review_completed", "progress",
}


def emit(task_id: str, type_: str, message: str = "", data: dict | None = None) -> None:
    if type_ not in EVENT_TYPES:
        raise ValueError(f"unknown event type {type_}")
    if not task_id:  # tool used outside a task (scripts, tests): nothing to attach the event to
        return
    with session_scope() as s:
        s.add(TaskEvent(task_id=task_id, type=type_, message=message, data=data))
    log.info("[%s] %s %s", task_id[:8], type_, message)


def set_status(task_id: str, status: str, result: str | None = None, spec: dict | None = None) -> None:
    with session_scope() as s:
        task = s.get(Task, task_id)
        if task is None:
            return
        task.status = status
        if result is not None:
            task.result = result
        if spec is not None:
            task.spec = spec
            task.language = spec.get("language", task.language)


def add_message(task_id: str | None, role: str, content: str) -> None:
    with session_scope() as s:
        conv = s.scalar(select(Task.conversation_id).where(Task.id == task_id)) if task_id else None
        s.add(Message(task_id=task_id, conversation_id=conv, role=role, content=content))


class CancelRegistry:
    """Cooperative cancellation flags checked between graph steps."""

    def __init__(self) -> None:
        self._flags: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def flag(self, task_id: str) -> threading.Event:
        with self._lock:
            return self._flags.setdefault(task_id, threading.Event())

    def cancel(self, task_id: str) -> None:
        self.flag(task_id).set()

    def is_cancelled(self, task_id: str) -> bool:
        return self.flag(task_id).is_set()


cancellations = CancelRegistry()
