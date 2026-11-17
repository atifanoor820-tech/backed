"""Playwright browser tools: one-shot page inspection plus an interactive session (open/click/type/read)."""

import concurrent.futures
import sys

from pydantic import BaseModel, Field, HttpUrl

from agent import runtime
from agent.config import get_settings
from agent.desktop_env import bring_to_front, ensure_default_desktop
from agent.events import emit
from agent.reflex.decide import choose
from agent.state import Risk, ToolResult
from agent.tools.registry import Tool, ToolContext, register


class InspectArgs(BaseModel):
    url: HttpUrl
    full_page: bool = False
    wait_ms: int = Field(1500, le=15_000)


def _inspect(args: InspectArgs, ctx: ToolContext) -> ToolResult:
    from playwright.sync_api import Error as PwError
    from playwright.sync_api import sync_playwright

    shots = get_settings().data_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    shot = shots / f"{ctx.task_id}_{ctx.step_index}.png"
    console_errors: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1366, "height": 850})
            page.on("console", lambda m: m.type == "error" and console_errors.append(m.text[:300]))
            page.on("pageerror", lambda e: console_errors.append(str(e)[:300]))
            resp = page.goto(str(args.url), wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(args.wait_ms)
            info = page.evaluate("""() => ({
                title: document.title,
                description: document.querySelector('meta[name="description"]')?.content || null,
                headings: [...document.querySelectorAll('h1,h2')].slice(0, 12).map(h => h.innerText.trim()).filter(Boolean),
                links: document.querySelectorAll('a[href]').length,
                images_missing_alt: [...document.images].filter(i => !i.alt).length,
                buttons: document.querySelectorAll('button').length,
                text_sample: document.body?.innerText.slice(0, 1500) || '',
            })""")
            page.screenshot(path=str(shot), full_page=args.full_page)
            browser.close()
    except PwError as e:
        msg = str(e).splitlines()[0]
        if "Executable doesn't exist" in str(e):
            msg = "Chromium for Playwright is not installed. Run: uv run playwright install chromium"
        return ToolResult(ok=False, summary=f"browser inspection failed: {msg}")
    status = resp.status if resp else None
    emit(ctx.task_id, "screenshot_captured", str(shot), {"path": shot.name})
    issues = []
    if status and status >= 400:
        issues.append(f"HTTP {status}")
    if console_errors:
        issues.append(f"{len(console_errors)} console errors")
    if info["images_missing_alt"]:
        issues.append(f"{info['images_missing_alt']} images without alt text")
    if not info["description"]:
        issues.append("no meta description")
    return ToolResult(
        ok=status is not None and status < 400,
        summary=f"'{info['title']}' (HTTP {status}); " + ("issues: " + ", ".join(issues) if issues else "no obvious issues"),
        data={**info, "status": status, "console_errors": console_errors, "issues": issues, "screenshot": shot.name,
              "url": str(args.url)},
        evidence=[str(shot)],
    )


register(Tool("browser.inspect", "Open a URL in a headless browser, report title, headings, issues and a screenshot.",
              InspectArgs, Risk.low, _inspect, retryable=True))


# --- Interactive session: one visible browser page that persists between steps and tasks ----------

# Labels every visible interactive element with data-abbas-id and returns a text description of each.
_COLLECT_JS = """(onlyTypeable) => {
  const sel = onlyTypeable
    ? 'input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=checkbox]):not([type=radio]), textarea, [contenteditable=""], [contenteditable=true], select'
    : 'a[href], button, input:not([type=hidden]), textarea, select, summary, [role=button], [role=link], [role=tab], [role=menuitem], [role=checkbox], [onclick], [contenteditable=true]';
  const out = [];
  let n = 0;
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    if (r.width < 2 || r.height < 2 || st.visibility === 'hidden' || st.display === 'none') continue;
    const label = (el.getAttribute('aria-label') || el.innerText || el.value || el.placeholder ||
                   el.title || el.name || el.alt || '').trim().replace(/\\s+/g, ' ').slice(0, 120);
    const id = 'e' + (n++);
    el.setAttribute('data-abbas-id', id);
    const tag = el.tagName.toLowerCase();
    const kind = el.getAttribute('role') || (tag === 'input' ? 'input[' + (el.type || 'text') + ']' : tag);
    out.push({id, desc: kind + ' "' + label + '"' + (el.href ? ' -> ' + el.getAttribute('href').slice(0, 80) : '')});
    if (out.length >= 400) break;
  }
  return out;
}"""

_session_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="abbas-browser")
_session: dict = {}


def _on_session(fn, *a):
    """Run `fn` on the dedicated Playwright thread (the sync API is bound to one thread)."""
    def _run():
        if sys.platform == "win32":
            ensure_default_desktop()
        return fn(*a)

    return _session_pool.submit(_run).result(timeout=get_settings().tool_timeout_s)


def _page():
    from playwright.sync_api import sync_playwright

    page = _session.get("page")
    if page is not None and not page.is_closed():
        return page
    if "pw" not in _session:
        _session["pw"] = sync_playwright().start()
    browser = _session.get("browser")
    if browser is None or not browser.is_connected():
        s = get_settings()
        headless = runtime.get("browser_headless")
        try:
            browser = _session["pw"].chromium.launch(headless=headless, channel=s.browser_channel or None)
        except Exception:
            if not s.browser_channel:
                raise
            browser = _session["pw"].chromium.launch(headless=headless)  # bundled Chromium
        _session["browser"] = browser
    page = _session["page"] = browser.new_page(viewport={"width": 1366, "height": 850})
    return page


def close_session() -> None:
    """Close Abbas's browser so the next browser step starts fresh (e.g. after the visible/hidden setting changes)."""
    def _close():
        browser = _session.pop("browser", None)
        _session.pop("page", None)
        if browser is not None and browser.is_connected():
            browser.close()

    if _session.get("browser") is not None:
        _on_session(_close)


def _snap(ctx: ToolContext, page, suffix: str) -> str:
    shots = get_settings().data_dir / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    name = f"{ctx.task_id}_{ctx.step_index}_{suffix}.png"
    page.screenshot(path=str(shots / name))
    emit(ctx.task_id, "screenshot_captured", name, {"path": name})
    return name


def _page_state(page, chars: int = 3000) -> dict:
    return {"url": page.url, "title": page.title(),
            "text": page.evaluate("() => document.body ? document.body.innerText.slice(0, %d) : ''" % chars)}


class OpenArgs(BaseModel):
    url: HttpUrl


class BClickArgs(BaseModel):
    target: str = Field(description="What to click, in words, e.g. 'the Sign in button' or 'More information link'")


class BTypeArgs(BaseModel):
    target: str = Field(description="Which field, e.g. 'search box'")
    text: str = Field(max_length=5000)
    submit: bool = False


class BReadArgs(BaseModel):
    max_chars: int = Field(4000, le=50_000)


def _choose_element(ctx: ToolContext, page, target: str, typeable: bool):
    elements = page.evaluate(_COLLECT_JS, typeable)
    if not elements:
        return None, None, "no interactive elements found on the page"
    options = {e["id"]: e["desc"] for e in elements}
    d = choose(ctx.task_id, {"url": page.url, "title": page.title(), "elements": list(options.values())},
               f"Which element should be {'typed into' if typeable else 'clicked'} to do: {target}?",
               options, target=target, purpose="element")
    if not d.confident():
        cands = ", ".join(f"{options[k]} {p:.0%}" for k, p in d.top())
        return None, d, f"not confident which element is '{target}' ({d.engine}, {d.confidence:.0%}); candidates: {cands}"
    return d.choice, d, options[d.choice]


def _do_open(args: OpenArgs, ctx: ToolContext) -> ToolResult:
    page = _page()
    resp = page.goto(str(args.url), wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_timeout(800)
    st = _page_state(page, 800)
    shot = _snap(ctx, page, "open")
    status = resp.status if resp else None
    return ToolResult(ok=status is not None and status < 400, summary=f"opened '{st['title']}' (HTTP {status}) at {page.url}",
                      data={**st, "status": status, "screenshot": shot}, evidence=[page.url, shot])


def _do_click(args: BClickArgs, ctx: ToolContext) -> ToolResult:
    page = _page()
    before = _page_state(page, 1500)
    eid, d, desc = _choose_element(ctx, page, args.target, typeable=False)
    if eid is None:
        return ToolResult(ok=False, summary=desc)
    page.locator(f'[data-abbas-id="{eid}"]').first.click(timeout=10_000)
    try:
        page.wait_for_load_state("domcontentloaded", timeout=8_000)
    except Exception:
        pass
    page.wait_for_timeout(600)
    after = _page_state(page, 1500)
    shot = _snap(ctx, page, "click")
    changed = after["url"] != before["url"] or after["text"] != before["text"]
    return ToolResult(ok=True, summary=f"clicked {desc} ({d.engine} {d.confidence:.0%}); "
                      + (f"page is now '{after['title']}'" if changed else "page content did not visibly change"),
                      data={"element": desc, "url": after["url"], "title": after["title"], "changed": changed,
                            "screenshot": shot}, evidence=[after["url"], shot])


def _do_type(args: BTypeArgs, ctx: ToolContext) -> ToolResult:
    page = _page()
    eid, d, desc = _choose_element(ctx, page, args.target, typeable=True)
    if eid is None:
        return ToolResult(ok=False, summary=desc)
    loc = page.locator(f'[data-abbas-id="{eid}"]').first
    loc.fill(args.text, timeout=10_000)
    value = loc.input_value() if loc.evaluate("e => 'value' in e") else loc.inner_text()
    if args.submit:
        loc.press("Enter")
        try:
            page.wait_for_load_state("domcontentloaded", timeout=8_000)
        except Exception:
            pass
    shot = _snap(ctx, page, "type")
    ok = args.submit or value == args.text
    return ToolResult(ok=ok, summary=f"typed into {desc} ({d.engine} {d.confidence:.0%})"
                      + (" and submitted" if args.submit else " (value verified)" if ok else " but the value did not stick"),
                      data={"element": desc, "url": page.url, "screenshot": shot}, evidence=[shot])


def _do_read(args: BReadArgs, ctx: ToolContext) -> ToolResult:
    if _session.get("page") is None:
        return ToolResult(ok=False, summary="no browser page is open; open a URL first")
    st = _page_state(_page(), args.max_chars)
    return ToolResult(ok=True, summary=f"read '{st['title']}' ({len(st['text'])} chars)", data=st, evidence=[st["url"]])


def _sessioned(fn):
    def run(args, ctx):
        from playwright.sync_api import Error as PwError

        try:
            return _on_session(fn, args, ctx)
        except PwError as e:
            msg = str(e).splitlines()[0]
            if "Executable doesn't exist" in str(e):
                msg = "Chromium for Playwright is not installed. Run: uv run playwright install chromium"
            return ToolResult(ok=False, summary=f"browser action failed: {msg}")
    return run


register(Tool("browser.open", "Open a URL in Abbas's visible browser window (the page stays open for later steps).",
              OpenArgs, Risk.low, _sessioned(_do_open), retryable=True))
register(Tool("browser.click", "Click an element on the open browser page, described in words.",
              BClickArgs, Risk.low, _sessioned(_do_click), risk_reflex=True))
register(Tool("browser.type", "Type text into a field on the open browser page; optionally press Enter.",
              BTypeArgs, Risk.low, _sessioned(_do_type), risk_reflex=True))
register(Tool("browser.read", "Read the title, URL and visible text of the open browser page.",
              BReadArgs, Risk.low, _sessioned(_do_read)))
