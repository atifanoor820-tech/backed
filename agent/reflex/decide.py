"""Fast decisions for the action layer.

Every decision tries Jev first, then Gemini, then a local string matcher. The engine
that answered and its confidence are recorded as a `decision_made` event, so the
dashboard shows exactly why Abbas clicked what it clicked.
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass, field

from agent.config import get_settings
from agent.events import emit
from agent.llm import gemini
from agent.reflex import jev

log = logging.getLogger("abbas.reflex")
MAX_OPTIONS = 255  # Jev's limit for choice criteria


@dataclass
class Decision:
    choice: str | None
    confidence: float
    engine: str  # jev | gemini | local | none
    probabilities: dict[str, float] = field(default_factory=dict)

    def confident(self) -> bool:
        return self.choice is not None and self.confidence >= get_settings().jev_min_confidence

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:n]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 or w.isdigit()}


def _local_similarity(target: str, desc: str) -> float:
    t, d = target.lower().strip(), desc.lower()
    if not t:
        return 0.0
    name = re.search(r'"([^"]*)"', d)
    if name and name.group(1).strip() == t:
        return 1.0
    tw, dw = _words(t), _words(d)
    overlap = len(tw & dw) / len(tw) if tw else 0.0
    return max(overlap * 0.9, difflib.SequenceMatcher(None, t, name.group(1) if name else d).ratio() * 0.8)


def _local_choice(target: str, options: dict[str, str]) -> Decision:
    scores = {k: _local_similarity(target, v or k) for k, v in options.items()}
    if not scores:
        return Decision(None, 0.0, "local")
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    # Penalise ties: two equally good matches are not a confident decision.
    margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0)
    conf = ranked[0] if margin >= 0.15 else ranked[0] * 0.5
    total = sum(scores.values()) or 1
    return Decision(best, round(conf, 3), "local", {k: round(v / total, 3) for k, v in scores.items()})


def _prefilter(target: str, options: dict[str, str]) -> dict[str, str]:
    if len(options) <= MAX_OPTIONS:
        return options
    ranked = sorted(options, key=lambda k: -_local_similarity(target, options[k] or k))
    return {k: options[k] for k in ranked[:MAX_OPTIONS]}


def choose(task_id: str | None, state, question: str, options: dict[str, str], target: str = "",
           purpose: str = "choice") -> Decision:
    """Pick one of `options` (id -> description). `target` is the plain description used by the local fallback."""
    options = _prefilter(target or question, options)
    s = get_settings()
    decision: Decision | None = None
    errors = []
    if s.jev_enabled and options:
        try:
            out = jev.ask(state, {"q": {"type": "choice", "instructions": question, "criteria": options}})["answers"]["q"]
            decision = Decision(out["choice"], float(out["confidence"]), "jev", out.get("probabilities", {}))
        except Exception as e:
            errors.append(f"jev: {e}")
    if decision is None and s.gemini_enabled and options:
        try:
            out = gemini.generate_json(
                "You choose exactly one option id for a desktop automation step. Return JSON "
                '{"choice": <option id or null>, "confidence": <0..1>}. Return null if no option fits.',
                f"State:\n{json.dumps(state)[:20000]}\n\nQuestion: {question}\n\nOptions:\n{json.dumps(options)}",
                fast=True,
            )
            choice = out.get("choice") if out.get("choice") in options else None
            decision = Decision(choice, float(out.get("confidence") or 0), "gemini",
                                {choice: float(out.get("confidence") or 0)} if choice else {})
        except Exception as e:
            errors.append(f"gemini: {e}")
    if decision is None:
        decision = _local_choice(target or question, options)
    _record(task_id, purpose, question, decision, errors)
    return decision


def yes_no(task_id: str | None, state, question: str, purpose: str = "check") -> tuple[float | None, str]:
    """Probability that the answer is yes, or None when no engine can judge it."""
    s = get_settings()
    errors = []
    p, engine = None, "none"
    if s.jev_enabled:
        try:
            p = float(jev.ask(state, {"q": {"type": "noul", "instructions": question}})["answers"]["q"]["noul"])
            engine = "jev"
        except Exception as e:
            errors.append(f"jev: {e}")
    if p is None and s.gemini_enabled:
        try:
            out = gemini.generate_json('Answer with JSON {"p_yes": <0..1>}.',
                                       f"State:\n{json.dumps(state)[:20000]}\n\nQuestion: {question}", fast=True)
            p, engine = float(out["p_yes"]), "gemini"
        except Exception as e:
            errors.append(f"gemini: {e}")
    if task_id:
        emit(task_id, "decision_made", f"{question} → {'unknown' if p is None else f'{p:.0%} yes'} ({engine})",
             {"purpose": purpose, "engine": engine, "p_yes": p, "errors": errors or None})
    return p, engine


def locate(task_id: str | None, png: bytes, target: str) -> tuple[dict | None, str]:
    """Bounding box of `target` on a screenshot (fractions of width/height)."""
    s = get_settings()
    errors = []
    found, engine = None, "none"
    if s.jev_enabled and s.jev_images:
        try:
            found, engine = jev.box(png, target), "jev"
        except Exception as e:
            errors.append(f"jev: {e}")
    if found is None and s.gemini_enabled:
        try:
            found, engine = gemini.locate(png, target), "gemini"
        except Exception as e:
            errors.append(f"gemini: {e}")
    if task_id:
        emit(task_id, "decision_made", f"locate '{target}' on screen → {'found' if found else 'not found'} ({engine})",
             {"purpose": "locate", "engine": engine, "box": found, "errors": errors or None})
    return found, engine


def _record(task_id: str | None, purpose: str, question: str, d: Decision, errors: list[str]) -> None:
    if not task_id:
        return
    emit(task_id, "decision_made",
         f"{question} → {d.choice or 'none'} ({d.confidence:.0%}, {d.engine})",
         {"purpose": purpose, "engine": d.engine, "choice": d.choice, "confidence": d.confidence,
          "top": d.top(), "errors": errors or None})
