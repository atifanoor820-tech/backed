"""Coding executor discovery. Execution adapters are added only once an executor is available."""

import shutil
import subprocess

from pydantic import BaseModel

from agent.state import Risk, ToolResult
from agent.tools.desktop import find_antigravity
from agent.tools.registry import Tool, ToolContext, register


class NoArgs(BaseModel):
    pass


def _version(exe: str) -> str | None:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15)
        return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else None
    except Exception:
        return None


def _check(args: NoArgs, ctx: ToolContext) -> ToolResult:
    found = {}
    codex = shutil.which("codex")
    found["codex"] = {"path": codex, "version": _version(codex) if codex else None}
    found["antigravity"] = {"path": find_antigravity(), "version": None}
    found["git"] = {"path": shutil.which("git"), "version": None}
    found["node"] = {"path": shutil.which("node"), "version": None}
    parts = [f"{k}: {'available' if v['path'] else 'not found'}" for k, v in found.items()]
    return ToolResult(ok=True, summary="; ".join(parts), data=found,
                      evidence=[v["path"] for v in found.values() if v["path"]])


register(Tool("coding.check_executors", "Check which coding tools (Codex, Antigravity, git, node) are installed.",
              NoArgs, Risk.low, _check))
