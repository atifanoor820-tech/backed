"""Conversation memory (LangGraph thread state) and long-term memory (LangGraph store)."""

import os
import time
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agent import interpreter
from agent.llm import gemini
from agent.memory import list_memories
from agent.runner import TaskRunner
from database.models import Task
from database.session import session_scope

TERMINAL = {"completed", "failed", "blocked", "cancelled"}


def wait(task_id, statuses=TERMINAL, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with session_scope() as s:
            t = s.get(Task, task_id)
            if t and t.status in statuses:
                return t
        time.sleep(0.05)
    raise AssertionError("task did not finish")


@pytest.fixture
def runner(settings):
    settings(gemini_api_key="", jev_api_key="")
    return TaskRunner(InMemorySaver(), InMemoryStore())


def test_follow_up_uses_conversation_context(runner, workspace):
    (workspace / "plan.txt").write_text("step one")
    t1 = runner.create("read file plan.txt")[0]
    conv = wait(t1).conversation_id
    t2 = runner.create("wahi file dobara parho", conversation_id=conv)[0]
    t = wait(t2)
    assert t.status == "completed", t.result
    assert t.spec["plan"][0]["args"]["path"].endswith("plan.txt")

    state = runner.thread_state(conv)
    assert [m["role"] for m in state["history"]] == ["user", "assistant", "user", "assistant"]
    assert state["context"]["file"].endswith("plan.txt")


def test_conversations_are_isolated(runner, workspace):
    (workspace / "a.txt").write_text("a")
    conv_a = wait(runner.create("read file a.txt")[0]).conversation_id
    t = wait(runner.create("wahi file dobara parho")[0])  # new conversation: no "that file" to refer to
    assert t.conversation_id != conv_a and t.status == "blocked"


def test_long_term_memory_crosses_conversations(runner):
    t1 = wait(runner.create("yaad rakho ke meri website abbas.pk hai")[0])
    assert t1.status == "completed" and "yaad kar liya" in t1.result
    assert any("abbas.pk" in m["text"] for m in list_memories(runner.store))

    t2 = wait(runner.create("tumhe kya yaad hai?")[0])  # different conversation
    assert t2.conversation_id != t1.conversation_id and "abbas.pk" in t2.result

    wait(runner.create("bhool jao meri website abbas.pk")[0])
    assert not list_memories(runner.store)


def test_secrets_are_never_stored(runner):
    t = wait(runner.create("yaad rakho mera password hunter2 hai")[0])
    assert "password manager" in t.result
    assert not list_memories(runner.store)


def test_gemini_sees_history_context_and_memories(runner, settings, monkeypatch, workspace):
    (workspace / "g.txt").write_text("x")
    prompts = []

    def fake(system, user, **kw):
        prompts.append(user)
        if "remember my name" in user.lower().split("request:")[-1]:
            return {"interpreted_objective": "remember", "reply": "ok", "remember": ["User's name is Abbas Ali"]}
        return {"interpreted_objective": "answer", "reply": "Aap ka naam Abbas Ali hai."}

    settings(gemini_api_key="g")
    monkeypatch.setattr(gemini, "generate_json", fake)
    conv = wait(runner.create("please remember my name is Abbas Ali")[0]).conversation_id
    t = wait(runner.create("mera naam kya hai?", conversation_id=conv)[0])
    assert t.result == "Aap ka naam Abbas Ali hai."
    last = prompts[-1]
    assert "user: please remember my name is Abbas Ali" in last  # short-term history
    assert "User's name is Abbas Ali" in last  # long-term memory


def test_gemini_secret_memory_is_dropped(runner, settings, monkeypatch):
    settings(gemini_api_key="g")
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **k: {
        "interpreted_objective": "x", "reply": "ok", "remember": ["API key is AIzaSyD-abcdefghijklmnopqrstuvwxyz"]})
    wait(runner.create("save my key")[0])
    assert not list_memories(runner.store)


def test_deleting_a_task_removes_it_from_conversation_memory(runner, workspace):
    (workspace / "d.txt").write_text("d")
    t1 = runner.create("read file d.txt")[0]
    conv = wait(t1).conversation_id
    wait(runner.create("files dikhao", conversation_id=conv)[0])
    runner.forget_task(t1, conv)
    assert all(m["task_id"] != t1 for m in runner.thread_state(conv)["history"])
    assert len(runner.thread_state(conv)["history"]) == 2


def test_desktop_follow_up_without_window_name(settings):
    settings(gemini_api_key="", jev_api_key="")
    spec, _ = interpreter.interpret("Plus dabao", context={"surface": "desktop", "window": "Calculator"})
    assert spec.plan[0].tool == "desktop.click" and spec.plan[0].args == {"window": "Calculator", "target": "Plus"}
    spec, _ = interpreter.interpret("click Equals", context={"surface": "browser", "url": "https://x.y"})
    assert spec.plan[0].tool == "browser.click"


PG = os.environ.get("ABBAS_TEST_DATABASE_URL", "")


@pytest.mark.skipif(not PG.startswith("postgres"), reason="set ABBAS_TEST_DATABASE_URL to a PostgreSQL database")
def test_postgres_memory_survives_restart(settings, workspace):
    from agent import memory

    settings(gemini_api_key="", jev_api_key="")
    memory._persistence = None
    p = memory.get_persistence()
    assert p.backend == "postgres"
    (workspace / "pg.txt").write_text("pg")
    first = TaskRunner(p.checkpointer, p.store)
    tag = uuid.uuid4().hex[:8]
    conv = wait(first.create("read file pg.txt")[0]).conversation_id
    wait(first.create(f"yaad rakho ke test tag {tag} hai", conversation_id=conv)[0])

    memory._persistence = None  # simulate a service restart: new pool, new saver, new store
    p2 = memory.get_persistence()
    second = TaskRunner(p2.checkpointer, p2.store)
    assert len(second.thread_state(conv)["history"]) == 4
    t = wait(second.create("wahi file dobara parho", conversation_id=conv)[0])
    assert t.status == "completed"
    assert any(tag in m["text"] for m in list_memories(p2.store))
