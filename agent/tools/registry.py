"""Typed tool registry. Every tool has a narrow purpose, a Pydantic input model and a risk level."""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from agent.state import Risk, ToolResult


@dataclass
class ToolContext:
    task_id: str
    step_index: int


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    risk: Risk
    run: Callable[[BaseModel, ToolContext], ToolResult]
    # Optional: refine risk from concrete arguments (e.g. overwrite vs. create).
    assess: Callable[[BaseModel], Risk] | None = None
    retryable: bool = False
    # Ask the reflex layer whether this concrete action is hard to undo; it may only raise risk.
    risk_reflex: bool = False
    # Overrides ABBAS_TOOL_TIMEOUT_S for slow tools (installs, builds, research).
    timeout_s: float | None = None

    def risk_for(self, args: dict) -> Risk:
        parsed = self.input_model.model_validate(args)
        return self.assess(parsed) if self.assess else self.risk


TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    TOOLS[tool.name] = tool
    return tool


def compact_catalog() -> list[dict]:
    """Tool names, purpose and arguments only - about a fifth of the full JSON schemas, for faster model calls."""
    out = []
    for t in TOOLS.values():
        schema = t.input_model.model_json_schema()
        required = set(schema.get("required", []))
        args = {name: ("" if name in required else "?") + str(p.get("type") or p.get("anyOf", [{}])[0].get("type", "any"))
                for name, p in schema.get("properties", {}).items()}
        out.append({"name": t.name, "does": t.description, "args": args})
    return out


def tool_catalog() -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "risk": t.risk.value,
            "input_schema": t.input_model.model_json_schema(),
        }
        for t in TOOLS.values()
    ]
