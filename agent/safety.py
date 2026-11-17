"""Workspace sandboxing and approval rules."""

from pathlib import Path

from agent.config import get_settings
from agent.state import PlanStep, Risk


class SafetyError(Exception):
    pass


def workspace_roots() -> list[Path]:
    """Folders from .env plus the ones the user added on the Settings page."""
    from agent import runtime

    extra = [Path(p).expanduser().resolve() for p in runtime.get("extra_folders")]
    return list(dict.fromkeys([*get_settings().workspace_paths, *extra]))


def resolve_in_workspace(path: str, workspace: str | None = None) -> Path:
    """Resolve `path` and refuse anything outside the authorised workspaces."""
    roots = workspace_roots()
    base = Path(workspace).resolve() if workspace else roots[0]
    if not any(base == r or r in base.parents for r in roots):
        raise SafetyError(f"{base} is not an authorised workspace")
    p = Path(path)
    target = (p if p.is_absolute() else base / p).resolve()
    if not any(target == r or r in target.parents for r in roots):
        raise SafetyError(f"path '{path}' is outside the authorised workspaces")
    return target


def step_needs_approval(step: PlanStep) -> bool:
    from agent.autonomy import needs_approval

    return needs_approval(step.risk)


def overall_risk(steps: list[PlanStep]) -> Risk:
    return max((s.risk for s in steps), key=lambda r: r.rank, default=Risk.low)


def low_confidence_block(source: str, confidence: float | None, risk: Risk) -> str | None:
    """A shaky voice transcript may never authorise a risky action."""
    threshold = get_settings().asr_min_confidence
    if source == "voice" and risk.rank >= Risk.medium.rank and confidence is not None and confidence < threshold:
        return (
            f"I am not confident I heard that correctly (confidence {confidence:.2f}). "
            "Because this action changes something on your computer, please repeat it or type it."
        )
    return None
