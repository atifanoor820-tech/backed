"""Developer tools: folders, a restricted terminal, AI code writing, and opening projects in Antigravity IDE."""

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

from agent.config import child_env
from agent.events import cancellations, emit
from agent.llm import gemini
from agent.safety import SafetyError, resolve_in_workspace
from agent.state import Risk, ToolResult
from agent.tools.desktop import visible_windows
from agent.tools.registry import Tool, ToolContext, register

ALLOWED_PROGRAMS = {"npm", "npx", "node", "git", "pnpm", "yarn"}
SHELL_METACHARS = re.compile(r"[&|;<>^`$%]|\r|\n")
REFUSED = [
    (re.compile(r"^git\s+push\b"), "pushing to a remote publishes your code; run git push yourself"),
    (re.compile(r"^(npm|pnpm|yarn)\s+publish\b"), "publishing a package is public and permanent; do it yourself"),
    (re.compile(r"^git\s+(reset\s+--hard|clean\s+-[a-z]*f)"), "this permanently discards work; do it yourself"),
]
DEV_SERVER = re.compile(r"^(npm|pnpm|yarn)\s+(run\s+)?(dev|start)\b|^npx\s+next\s+(dev|start)\b")
SKIP_DIRS = {"node_modules", ".next", ".git", "dist", "build", ".turbo", "out", "coverage"}
_servers: dict[str, subprocess.Popen] = {}


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)


def parse_command(command: str) -> list[str]:
    """Split a command into argv, allowing only known developer programs and no shell syntax."""
    command = command.strip()
    if SHELL_METACHARS.search(command.replace('"@/*"', "")):
        raise SafetyError("shell operators (&&, |, >, ;, $, %) are not allowed - run one command per step")
    argv = [a[1:-1] if len(a) > 1 and a[0] == a[-1] == '"' else a for a in shlex.split(command, posix=False)]
    if not argv:
        raise SafetyError("empty command")
    prog = Path(argv[0]).stem.lower()
    if prog not in ALLOWED_PROGRAMS:
        raise SafetyError(f"'{argv[0]}' is not allowed; terminal.run only runs {', '.join(sorted(ALLOWED_PROGRAMS))}")
    normalized = " ".join([prog, *argv[1:]])
    for pattern, why in REFUSED:
        if pattern.search(normalized):
            raise SafetyError(why)
    exe = shutil.which(prog)
    if not exe:
        raise SafetyError(f"{prog} is not installed or not on PATH")
    return [exe, *argv[1:]]


class FolderArgs(BaseModel):
    path: str = Field(description="Full folder path inside an allowed folder, e.g. D:\\abbas work")


class RunArgs(BaseModel):
    command: str = Field(description="One command, e.g. 'npx create-next-app@latest my-site --ts --tailwind --app --yes'")
    cwd: str = Field(description="Folder to run in (must exist, inside an allowed folder)")
    timeout_s: int = Field(600, ge=10, le=1200)


class CodeArgs(BaseModel):
    project_dir: str = Field(description="Root of the project, e.g. D:\\abbas work\\prop-firm")
    instructions: str = Field(description="What to build or fix in this step, in detail")
    files: list[str] = Field(default_factory=list, description="Files (relative to project) to create or rewrite")
    context_files: list[str] = Field(default_factory=list, description="Files to read first, e.g. docs/research.md")
    max_files: int = Field(6, ge=1, le=10)


class IdeArgs(BaseModel):
    path: str = Field(description="Project folder to open in Antigravity IDE")


def _folder(args: FolderArgs, ctx: ToolContext) -> ToolResult:
    p = resolve_in_workspace(args.path)
    existed = p.exists()
    p.mkdir(parents=True, exist_ok=True)
    if not p.is_dir():
        return ToolResult(ok=False, summary=f"{p} exists but is not a folder")
    if not existed:
        emit(ctx.task_id, "file_created", str(p), {"path": str(p)})
    return ToolResult(ok=True, summary=f"{'folder already existed' if existed else 'created folder'} {p}",
                      data={"path": str(p)}, evidence=[str(p)])


def _run(args: RunArgs, ctx: ToolContext) -> ToolResult:
    cwd = resolve_in_workspace(args.cwd)
    if not cwd.is_dir():
        return ToolResult(ok=False, summary=f"folder {cwd} does not exist; create it first")
    argv = parse_command(args.command)
    env = child_env(CI="1", npm_config_yes="true", NEXT_TELEMETRY_DISABLED="1", FORCE_COLOR="0", NO_COLOR="1",
                    BROWSER="none")
    background = bool(DEV_SERVER.search(args.command.strip()))
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    lines: list[str] = []
    ready = threading.Event()

    def pump():
        last_emit, pending = 0.0, []
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            lines.append(line)
            pending.append(line)
            if background and re.search(r"(Ready in|Local:\s+https?://|started server on)", line, re.I):
                ready.set()
            if time.monotonic() - last_emit > 1.5:
                emit(ctx.task_id, "tool_output", "\n".join(pending[-12:]), {"step": ctx.step_index})
                pending, last_emit = [], time.monotonic()
        if pending:
            emit(ctx.task_id, "tool_output", "\n".join(pending[-12:]), {"step": ctx.step_index})

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    started = time.monotonic()
    while proc.poll() is None:
        if cancellations.is_cancelled(ctx.task_id):
            _kill_tree(proc)
            return ToolResult(ok=False, summary=f"stopped `{args.command}` on your request")
        if background and ready.is_set():
            url = next((m.group(0) for line in lines if (m := re.search(r"https?://localhost:\d+", line))), None)
            _servers[str(cwd)] = proc
            return ToolResult(ok=True, summary=f"dev server running{' at ' + url if url else ''} (left running in the background)",
                              data={"url": url, "pid": proc.pid, "output_tail": "\n".join(lines[-15:])},
                              evidence=[url] if url else [])
        if time.monotonic() - started > (60 if background else args.timeout_s):
            _kill_tree(proc)
            return ToolResult(ok=False, summary=f"`{args.command}` did not finish within {args.timeout_s}s and was stopped",
                              data={"output_tail": "\n".join(lines[-40:])})
        time.sleep(0.2)
    reader.join(timeout=5)
    tail = "\n".join(lines[-40:])
    ok = proc.returncode == 0
    return ToolResult(ok=ok, summary=f"`{args.command}` {'finished' if ok else f'failed (exit {proc.returncode})'}"
                      f" in {time.monotonic() - started:.0f}s",
                      data={"exit_code": proc.returncode, "output_tail": tail[-6000:], "cwd": str(cwd)},
                      evidence=[tail.splitlines()[-1]] if tail else [])


def project_snapshot(root: Path, wanted: list[str], context: list[str], budget: int = 45_000) -> str:
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            paths.append(Path(dirpath, f).relative_to(root).as_posix())
            if len(paths) > 300:
                break
    parts = ["Project files:\n" + "\n".join(sorted(paths)[:300])]
    key = ["package.json", "tsconfig.json", "next.config.ts", "next.config.mjs", "src/app/layout.tsx", "app/layout.tsx",
           "src/app/globals.css", "app/globals.css", "src/app/page.tsx", "app/page.tsx", "components.json"]
    for rel in dict.fromkeys([*context, *key, *wanted]):
        p = (root / rel).resolve()
        if root not in p.parents or not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        limit = 12_000 if rel in context else 6_000
        chunk = f"\n--- {rel} ---\n{text[:limit]}"
        if sum(len(x) for x in parts) + len(chunk) > budget:
            break
        parts.append(chunk)
    return "\n".join(parts)


CODE_SYSTEM = """You are a senior front-end engineer working inside an existing Next.js project (App Router,
TypeScript). Read the project snapshot to learn its setup (Next/React/Tailwind versions, src/ dir, import alias)
and follow it exactly - e.g. Tailwind v4 uses @import "tailwindcss" in globals.css and no tailwind.config.
Write production-quality, complete files: no placeholders, no TODOs, no lorem ipsum (write real copy from the
brief), accessible markup, responsive layouts, semantic HTML. Use only packages already in package.json plus
lucide-react if it is installed; do not use remote images (use CSS gradients/SVG). Keep each file under ~300 lines;
split into components under src/components/ (or components/ if there is no src/). Server components by default;
add "use client" only where you need state or effects. Never call new Date(), Date.now() or Math.random() while a
server component renders (Next.js 16 refuses to prerender it) - write the year as a literal or compute it in a
client component.
Output format - plain text, NOT JSON, NOT markdown fences:
=== FILE: <path relative to project root> ===
<the complete file content>
=== END FILE ===
(repeat for each file), then finally:
=== NOTES ===
<what you did, and what should be built next>"""

_BLOCK = re.compile(r"^=== FILE: (.+?) ===\s*\n(.*?)\n=== END FILE ===", re.S | re.M)


def parse_file_blocks(text: str) -> dict:
    """Parse the coder's delimited output. Code inside JSON strings proved fragile (escaping, truncation)."""
    files = []
    for path, content in _BLOCK.findall(text):
        content = re.sub(r"\A```[\w-]*\n|\n```\s*\Z", "", content)  # tolerate stray fences
        files.append({"path": path.strip(), "content": content.rstrip() + "\n"})
    notes = text.split("=== NOTES ===", 1)[1].strip() if "=== NOTES ===" in text else ""
    if not files and text.lstrip().startswith("{"):  # older JSON-style answer
        try:
            data = json.loads(text)
            return {"files": data.get("files") or [], "notes": data.get("notes", "")}
        except ValueError:
            pass
    return {"files": files, "notes": notes[:2000]}


def _code(args: CodeArgs, ctx: ToolContext) -> ToolResult:
    root = resolve_in_workspace(args.project_dir)
    if not (root / "package.json").exists():
        return ToolResult(ok=False, summary=f"{root} is not a project yet (no package.json); create it first")
    snapshot = project_snapshot(root, args.files, args.context_files)
    want = f"\nCreate or rewrite these files: {', '.join(args.files)}" if args.files else ""
    try:
        text = gemini.generate(
            CODE_SYSTEM, [{"text": f"{snapshot}\n\nTASK (at most {args.max_files} files):\n{args.instructions}{want}"}],
            json_mode=False, timeout_s=240, max_output_tokens=60_000,
        )
    except Exception as e:
        return ToolResult(ok=False, summary=f"code generation failed: {e}")
    out = parse_file_blocks(text)
    written = []
    for f in out["files"][: args.max_files]:
        rel = str(f.get("path", "")).replace("\\", "/").lstrip("/")
        content = f.get("content")
        if not rel or not isinstance(content, str):
            continue
        target = (root / rel).resolve()
        if root not in target.parents:
            continue  # never write outside the project
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel)
        emit(ctx.task_id, "file_updated" if existed else "file_created", str(target), {"path": str(target)})
    if not written:
        return ToolResult(ok=False, summary="the code model returned no usable files", data={"notes": out.get("notes")})
    missing = missing_packages(root, written)
    warn = f"; imports packages that are NOT installed: {', '.join(missing)} (run: npm install {' '.join(missing)})" if missing else ""
    return ToolResult(ok=True, summary=f"wrote {len(written)} file(s): {', '.join(written)}{warn}",
                      data={"files": written, "notes": out.get("notes", ""), "project": str(root),
                            "missing_packages": missing}, evidence=written)


_IMPORT = re.compile(r"""(?:^|\s)(?:import|export)\s[^'"]*?from\s+['"]([^'"]+)['"]|import\s+['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)""", re.M)
_BUILTIN_PREFIXES = ("node:", ".", "/", "@/", "~/")
_NODE_BUILTINS = {"fs", "path", "os", "crypto", "url", "util", "stream", "http", "https", "events", "child_process", "buffer"}


def missing_packages(root: Path, files: list[str]) -> list[str]:
    """Bare imports in the written files that package.json does not provide."""
    try:
        pkg = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    have = {*pkg.get("dependencies", {}), *pkg.get("devDependencies", {})}
    missing = set()
    for rel in files:
        if not rel.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs")):
            continue
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        for groups in _IMPORT.findall(text):
            spec = next((g for g in groups if g), "")
            if not spec or spec.startswith(_BUILTIN_PREFIXES):
                continue
            parts = spec.split("/")
            name = "/".join(parts[:2]) if spec.startswith("@") else parts[0]
            if name not in have and name not in _NODE_BUILTINS:
                missing.add(name)
    return sorted(missing)


def find_antigravity_cli() -> str | None:
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"
    for p in (base / "Antigravity IDE" / "_" / "bin" / "antigravity-ide.cmd", base / "Antigravity" / "bin" / "antigravity.cmd"):
        if p.exists():
            return str(p)
    return shutil.which("antigravity-ide") or shutil.which("antigravity")


def _ide(args: IdeArgs, ctx: ToolContext) -> ToolResult:
    path = resolve_in_workspace(args.path)
    if not path.exists():
        return ToolResult(ok=False, summary=f"{path} does not exist")
    cli = find_antigravity_cli()
    if not cli:
        return ToolResult(ok=False, summary="Antigravity IDE is not installed (no antigravity-ide launcher found)")
    before = {w["handle"] for w in visible_windows()}
    subprocess.Popen([cli, "--new-window", str(path)], close_fds=True, env=child_env(),
                     creationflags=subprocess.CREATE_NO_WINDOW)
    name = path.name.lower()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        time.sleep(1)
        for w in visible_windows():
            # Titles look like "page.tsx - prop-firm - Antigravity IDE": compare whole parts, so "prop-firm"
            # does not match an already-open "prop-firm-app" window.
            parts = [p.strip().lower() for p in w["title"].split(" - ")]
            if name in parts and ("antigravity ide" in parts or w["handle"] not in before):
                return ToolResult(ok=True, summary=f"opened {path} in Antigravity IDE ('{w['title']}')",
                                  data={"window": w["title"], "path": str(path)}, evidence=[w["title"]])
    return ToolResult(ok=False, summary=f"launched Antigravity IDE but no window for '{path.name}' appeared within 30s")


register(Tool("files.make_folder", "Create a folder (and parents) inside an allowed folder.", FolderArgs, Risk.low, _folder))
register(Tool("terminal.run", "Run ONE developer command (npm, npx, node, git, pnpm, yarn) in a project folder and "
              "return its output. Dev servers (npm run dev) are left running in the background.",
              RunArgs, Risk.medium, _run, risk_reflex=True, timeout_s=1300))
register(Tool("code.write", "Have the coding model create or rewrite up to 10 files in an existing project, "
              "following instructions and context files such as docs/research.md.",
              CodeArgs, Risk.medium, _code, timeout_s=300))
register(Tool("ide.open", "Open a project folder in Antigravity IDE so the user can watch the code.",
              IdeArgs, Risk.low, _ide, timeout_s=45))
