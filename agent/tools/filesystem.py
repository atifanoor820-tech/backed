from pydantic import BaseModel, Field

from agent.events import emit
from agent.safety import resolve_in_workspace
from agent.state import Risk, ToolResult
from agent.tools.registry import Tool, ToolContext, register


class ListArgs(BaseModel):
    path: str = Field(".", description="Folder inside the workspace")


class ReadArgs(BaseModel):
    path: str
    max_chars: int = Field(20_000, le=200_000)


class WriteArgs(BaseModel):
    path: str
    content: str
    overwrite: bool = False


def _list(args: ListArgs, ctx: ToolContext) -> ToolResult:
    folder = resolve_in_workspace(args.path)
    if not folder.is_dir():
        return ToolResult(ok=False, summary=f"'{args.path}' is not a folder in the workspace")
    entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))[:200]
    items = [{"name": p.name, "type": "file" if p.is_file() else "dir",
              "size": p.stat().st_size if p.is_file() else None} for p in entries]
    return ToolResult(ok=True, summary=f"{len(items)} entries in {folder}", data={"path": str(folder), "entries": items},
                      evidence=[str(folder)])


def _read(args: ReadArgs, ctx: ToolContext) -> ToolResult:
    f = resolve_in_workspace(args.path)
    if not f.is_file():
        return ToolResult(ok=False, summary=f"file '{args.path}' not found in the workspace")
    text = f.read_text(encoding="utf-8", errors="replace")
    return ToolResult(ok=True, summary=f"read {min(len(text), args.max_chars)} characters from {f.name}",
                      data={"path": str(f), "content": text[: args.max_chars], "truncated": len(text) > args.max_chars},
                      evidence=[str(f)])


def _write_risk(args: WriteArgs) -> Risk:
    try:
        return Risk.medium if resolve_in_workspace(args.path).exists() else Risk.low
    except Exception:
        return Risk.high


def _write(args: WriteArgs, ctx: ToolContext) -> ToolResult:
    f = resolve_in_workspace(args.path)
    existed = f.exists()
    if existed and not args.overwrite:
        return ToolResult(ok=False, summary=f"{f.name} already exists; overwrite was not requested")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(args.content, encoding="utf-8")
    # Verify by reading back.
    ok = f.read_text(encoding="utf-8") == args.content
    emit(ctx.task_id, "file_updated" if existed else "file_created", str(f), {"path": str(f)})
    return ToolResult(ok=ok, summary=f"{'updated' if existed else 'created'} {f} ({len(args.content)} chars, verified by read-back)"
                      if ok else f"write to {f} could not be verified", data={"path": str(f)}, evidence=[str(f)])


register(Tool("workspace.list", "List files and folders inside the authorised workspace.", ListArgs, Risk.low, _list))
register(Tool("workspace.read", "Read a text file inside the authorised workspace.", ReadArgs, Risk.low, _read))
register(Tool("workspace.write", "Create a text file in the workspace. Overwriting an existing file needs approval.",
              WriteArgs, Risk.low, _write, assess=_write_risk))
