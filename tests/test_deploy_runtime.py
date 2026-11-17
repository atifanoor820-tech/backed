"""Deployed (serverless) behaviour: remote browser for Dashi, tasks finished inside the request."""

import concurrent.futures
import time

import pytest

import agent.config as config
import agent.runner as runner
from agent.tools import dashi


class FakeChromium:
    def __init__(self):
        self.calls = []

    def connect_over_cdp(self, url, timeout):
        self.calls.append(("cdp", url))
        return "remote-browser"

    def launch(self, headless, channel):
        self.calls.append(("launch", channel))
        return "local-browser"


class FakePlaywright:
    def __init__(self):
        self.chromium = FakeChromium()


def test_remote_browser_is_used_when_endpoint_is_set(settings):
    settings(browser_ws_endpoint="wss://example.test?token=x")
    p = FakePlaywright()
    browser, headless = dashi._open_browser(p)
    assert browser == "remote-browser" and headless is True
    assert p.chromium.calls == [("cdp", "wss://example.test?token=x")]


def test_server_without_browser_explains_what_to_set(settings, monkeypatch):
    settings(browser_ws_endpoint="")
    monkeypatch.setattr(config, "ON_SERVERLESS", True)
    with pytest.raises(RuntimeError, match="ABBAS_BROWSER_WS_ENDPOINT"):
        dashi._open_browser(FakePlaywright())


def test_local_machine_launches_its_own_browser(settings, monkeypatch):
    settings(browser_ws_endpoint="")
    monkeypatch.setattr(config, "ON_SERVERLESS", False)
    browser, _ = dashi._open_browser(FakePlaywright())
    assert browser == "local-browser"


def test_serverless_waits_for_the_task_and_local_does_not(monkeypatch):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    monkeypatch.setattr(config, "ON_SERVERLESS", False)
    fut = pool.submit(time.sleep, 0.3)
    start = time.time()
    runner._finish_inline(fut)
    assert time.time() - start < 0.2  # returned at once

    monkeypatch.setattr(config, "ON_SERVERLESS", True)
    fut = pool.submit(time.sleep, 0.3)
    runner._finish_inline(fut)
    assert fut.done()  # waited for it
