"""Structured task representation and LangGraph state."""

from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class Risk(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]


class PlanStep(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    risk: Risk = Risk.low


class TaskSpec(BaseModel):
    original_request: str
    normalized_request: str
    language: str = "en"
    interpreted_objective: str
    target: str | None = None
    required_outcome: str = ""
    constraints: list[str] = Field(default_factory=list)
    plan: list[PlanStep] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    risk_level: Risk = Risk.low
    requires_approval: bool = False
    approval_reason: str | None = None
    # Set when the request cannot be executed as understood.
    clarification_question: str | None = None
    # Direct conversational answer when no tools are needed.
    reply: str | None = None
    # Long-term memory updates: facts to store, and ids (or descriptions) of memories to delete.
    remember: list[str] = Field(default_factory=list)
    forget: list[str] = Field(default_factory=list)
    # True for goals that need several plan -> act -> review rounds (build a site, research then act, ...)
    multi_round: bool = False
    interpreter: Literal["rules", "llm"] = "rules"


class ToolResult(BaseModel):
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    # Paths/strings that prove the outcome (screenshots, file paths, window titles).
    evidence: list[str] = Field(default_factory=list)


TaskStatus = Literal[
    "created", "running", "awaiting_approval", "blocked", "cancelled", "completed", "failed"
]


MAX_HISTORY = 100  # messages kept in a conversation thread's state


def add_turns(old: list[dict] | None, new: list[dict] | dict | None) -> list[dict]:
    """Reducer: append this turn's messages to the thread history, keeping the most recent ones.
    {"replace": [...]} overwrites the history (used when the user deletes a turn)."""
    if isinstance(new, dict) and "replace" in new:
        return list(new["replace"])[-MAX_HISTORY:]
    return ((old or []) + (new or []))[-MAX_HISTORY:]


def merge_context(old: dict | None, new: dict | None) -> dict:
    """Reducer: working context (last page, window, file) is updated key by key."""
    return {**(old or {}), **(new or {})}


class GraphState(TypedDict, total=False):
    # Persist across turns of a conversation (LangGraph thread = conversation):
    conversation_id: str
    history: Annotated[list[dict], add_turns]  # [{role, content, task_id}]
    context: Annotated[dict, merge_context]  # {surface, url, page_title, window, app, file}
    # Reset at the start of every turn:
    task_id: str
    request: str
    source: str  # text | voice
    transcript_confidence: float | None
    spec: dict
    step_index: int
    results: list[dict]
    attempts: int
    rounds: int  # review rounds completed for this goal
    status: TaskStatus
    report: str
