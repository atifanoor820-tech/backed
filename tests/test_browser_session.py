"""Real Playwright session (headless) against a local page; element choice by the local matcher."""

import http.server
import threading

import pytest

from agent.tools import TOOLS, ToolContext
from agent.tools.browser import BClickArgs, BReadArgs, BTypeArgs, OpenArgs

PAGE = b"""<!doctype html><html><head><title>Abbas test page</title></head><body>
<h1 id="status">Waiting</h1>
<input placeholder="Search box" id="q">
<button onclick="document.getElementById('status').innerText='Hello '+document.getElementById('q').value">Greet me</button>
<button onclick="document.getElementById('status').innerText='Wrong button'">Unsubscribe</button>
</body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, *a):
        pass


TASK_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture(scope="module", autouse=True)
def task_row():
    from database.models import Task
    from database.session import session_scope

    with session_scope() as s:
        if s.get(Task, TASK_ID) is None:
            s.add(Task(id=TASK_ID, original_request="browser test"))


@pytest.fixture(scope="module")
def site():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()


def test_open_type_click_read(site, settings):
    settings(jev_api_key="", gemini_api_key="")
    ctx = ToolContext(task_id="00000000-0000-0000-0000-000000000001", step_index=0)
    r = TOOLS["browser.open"].run(OpenArgs(url=site), ctx)
    if not r.ok and "not installed" in r.summary:
        pytest.skip(r.summary)
    assert r.ok and "Abbas test page" in r.summary

    r = TOOLS["browser.type"].run(BTypeArgs(target="search box", text="Abbad"), ctx)
    assert r.ok, r.summary

    r = TOOLS["browser.click"].run(BClickArgs(target="Greet me"), ctx)
    assert r.ok and r.data["changed"], r.summary

    r = TOOLS["browser.read"].run(BReadArgs(), ctx)
    assert "Hello Abbad" in r.data["text"]


def test_unclear_target_is_refused(site, settings):
    settings(jev_api_key="", gemini_api_key="")
    ctx = ToolContext(task_id="00000000-0000-0000-0000-000000000001", step_index=1)
    TOOLS["browser.open"].run(OpenArgs(url=site), ctx)
    r = TOOLS["browser.click"].run(BClickArgs(target="the shopping cart"), ctx)
    assert not r.ok and "not confident" in r.summary
