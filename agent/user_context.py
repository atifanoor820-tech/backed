"""Who the current work belongs to.

API requests and task runs set the signed-in user here, so per-user data (Dashi's trading accounts and
ShowMyTrades profiles, long-term memories) is read and written for that user only. Unset = single-user mode.
"""

import contextvars
from contextlib import contextmanager

from sqlalchemy import select

from database.models import Conversation, Task
from database.session import session_scope

CURRENT_USER: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_user", default=None)


def current_user_id() -> str | None:
    return CURRENT_USER.get()


@contextmanager
def as_user(user_id: str | None):
    token = CURRENT_USER.set(user_id)
    try:
        yield
    finally:
        CURRENT_USER.reset(token)


def owner_of_conversation(conversation_id: str | None) -> str | None:
    if not conversation_id:
        return None
    with session_scope() as s:
        return s.scalar(select(Conversation.user_id).where(Conversation.id == conversation_id))


def owner_of_task(task_id: str | None) -> str | None:
    if not task_id:
        return None
    with session_scope() as s:
        return s.scalar(select(Conversation.user_id).join(Task, Task.conversation_id == Conversation.id)
                        .where(Task.id == task_id))
