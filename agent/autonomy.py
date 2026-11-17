"""How freely Abbas may act without asking.

  ask      every medium/high-risk step needs approval (default)
  trusted  medium-risk steps run on their own; high-risk ones (flagged hard to undo) still ask
  full     nothing asks

Limits that apply at every level: emergency stop, PyAutoGUI fail-safe, blocked hotkeys, file tools
confined to the workspaces, low-confidence voice never acting, and the reflex layer refusing to
click when it is not confident.
"""

from agent import runtime
from agent.runtime import Autonomy as Level
from agent.state import Risk

LEVELS: tuple[Level, ...] = ("ask", "trusted", "full")


def get_level() -> Level:
    return runtime.get("autonomy")


def set_level(level: Level) -> Level:
    return runtime.update(autonomy=level)["autonomy"]


def needs_approval(risk: Risk, level: Level | None = None) -> bool:
    level = level or get_level()
    if level == "full":
        return False
    if level == "trusted":
        return risk == Risk.high
    return risk.rank >= Risk.medium.rank
