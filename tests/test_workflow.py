"""End-to-end graph runs against the real tools (filesystem) and the SQLite event log."""

import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from sqlalchemy import select

from agent.runner import TaskRunner
from database.models import Task, TaskEvent
from database.session import session_scope


def wait_for(task_id, statuses, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with session_scope() as s:
            t = s.get(Task, task_id)
            if t and t.status in statuses:
                return t
        time.sleep(0.1)
    raise AssertionError(f"task stuck in {t.status if t else None}")


def event_types(task_id):
    with session_scope() as s:
        return [e.type for e in s.scalars(select(TaskEvent).where(TaskEvent.task_id == task_id).order_by(TaskEvent.id))]


runner = TaskRunner(InMemorySaver(), InMemoryStore())


def test_create_file_completes_and_is_verified(workspace):
    tid = runner.create('create file hello.txt with "salam abbas"')[0]
    t = wait_for(tid, {"completed", "failed"})
    assert t.status == "completed", t.result
    assert (workspace / "hello.txt").read_text() == "salam abbas"
    types = event_types(tid)
    assert {"intent_resolved", "tool_started", "file_created", "tool_completed", "task_completed"} <= set(types)


def test_overwrite_waits_for_approval_then_rejects(workspace):
    (workspace / "keep.txt").write_text("original")
    tid = runner.create('create file keep.txt with "replaced"')[0]
    wait_for(tid, {"awaiting_approval"})
    runner.resolve_approval(tid, approved=False)
    t = wait_for(tid, {"cancelled", "completed", "failed"})
    assert t.status == "cancelled"
    assert (workspace / "keep.txt").read_text() == "original"
    types = event_types(tid)
    assert "approval_rejected" in types
    assert types.count("approval_required") == 1  # resume must not replay the announcement


def test_approved_overwrite_runs(workspace):
    (workspace / "edit.txt").write_text("v1")
    tid = runner.create('create file edit.txt with "v2"')[0]
    wait_for(tid, {"awaiting_approval"})
    runner.resolve_approval(tid, approved=True)
    t = wait_for(tid, {"completed", "failed"})
    assert t.status == "completed", t.result
    assert (workspace / "edit.txt").read_text() == "v2"
    assert "file_updated" in event_types(tid)


def test_tool_refuses_unplanned_overwrite(workspace):
    from agent.tools import TOOLS, ToolContext
    from agent.tools.filesystem import WriteArgs

    (workspace / "guard.txt").write_text("v1")
    r = TOOLS["workspace.write"].run(WriteArgs(path="guard.txt", content="v2"), ToolContext("t", 0))
    assert not r.ok and (workspace / "guard.txt").read_text() == "v1"


def test_cancel_while_awaiting_approval(workspace):
    (workspace / "keep2.txt").write_text("original")
    tid = runner.create('create file keep2.txt with "replaced"')[0]
    wait_for(tid, {"awaiting_approval"})
    runner.cancel(tid)
    assert wait_for(tid, {"cancelled", "completed", "failed"}).status == "cancelled"


def test_read_report_shows_the_content(workspace):
    (workspace / "show.txt").write_text("Salam from the file")
    t = wait_for(runner.create("read file show.txt")[0], {"completed", "failed"})
    assert "Salam from the file" in t.result


def test_failure_is_reported_not_hidden():
    tid = runner.create("read file missing.txt")[0]
    t = wait_for(tid, {"completed", "failed"})
    assert t.status == "failed" and "not found" in t.result


def test_unrecognised_request_blocks():
    tid = runner.create("build me a full stack crm with auth")[0]
    t = wait_for(tid, {"blocked", "completed", "failed"})
    assert t.status == "blocked"
