"""Tests for Agent Fabric AgentScheduler."""

from __future__ import annotations

import pytest
from datetime import datetime

from aragora.fabric.scheduler import AgentScheduler
from aragora.fabric.models import Priority, Task, TaskStatus


@pytest.fixture
def scheduler():
    return AgentScheduler()


def make_task(id: str = "t1", type: str = "debate") -> Task:
    return Task(id=id, type=type, payload={"data": "test"})


class TestSchedule:
    @pytest.mark.asyncio
    async def test_schedule_task(self, scheduler):
        task = make_task()
        handle = await scheduler.schedule(task, "agent-1")
        assert handle.task_id == "t1"
        assert handle.agent_id == "agent-1"
        assert handle.status == TaskStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_schedule_with_priority(self, scheduler):
        handle = await scheduler.schedule(make_task(), "agent-1", priority=Priority.HIGH)
        assert handle.status == TaskStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_schedule_multiple_tasks(self, scheduler):
        for i in range(5):
            await scheduler.schedule(make_task(id=f"t{i}"), "agent-1")
        pending = await scheduler.list_pending("agent-1")
        assert len(pending) == 5

    @pytest.mark.asyncio
    async def test_queue_full_raises(self):
        scheduler = AgentScheduler(max_queue_depth=2)
        await scheduler.schedule(make_task(id="t1"), "agent-1")
        await scheduler.schedule(make_task(id="t2"), "agent-1")
        with pytest.raises(ValueError, match="Queue full"):
            await scheduler.schedule(make_task(id="t3"), "agent-1")


class TestDependencies:
    @pytest.mark.asyncio
    async def test_valid_dependency(self, scheduler):
        h1 = await scheduler.schedule(make_task(id="t1"), "agent-1")
        h2 = await scheduler.schedule(make_task(id="t2"), "agent-1", depends_on=["t1"])
        # t2 should be pending since t1 is not yet complete
        assert h2.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_invalid_dependency_raises(self, scheduler):
        with pytest.raises(ValueError, match="not found"):
            await scheduler.schedule(make_task(id="t2"), "agent-1", depends_on=["nonexistent"])

    @pytest.mark.asyncio
    async def test_dependency_resolution(self, scheduler):
        await scheduler.schedule(make_task(id="t1"), "agent-1")
        await scheduler.schedule(make_task(id="t2"), "agent-1", depends_on=["t1"])

        # Pop and complete t1
        task = await scheduler.pop_next("agent-1")
        assert task.id == "t1"
        await scheduler.complete_task("t1", result={"ok": True})

        # t2 should now be scheduled
        h2 = await scheduler.get_handle("t2")
        assert h2.status == TaskStatus.SCHEDULED


class TestPopNext:
    @pytest.mark.asyncio
    async def test_pop_returns_task(self, scheduler):
        await scheduler.schedule(make_task(), "agent-1")
        task = await scheduler.pop_next("agent-1")
        assert task is not None
        assert task.id == "t1"

    @pytest.mark.asyncio
    async def test_pop_empty_queue(self, scheduler):
        task = await scheduler.pop_next("agent-1")
        assert task is None

    @pytest.mark.asyncio
    async def test_pop_marks_running(self, scheduler):
        await scheduler.schedule(make_task(), "agent-1")
        await scheduler.pop_next("agent-1")
        status = await scheduler.get_status("t1")
        assert status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_priority_ordering(self, scheduler):
        await scheduler.schedule(make_task(id="low"), "agent-1", priority=Priority.LOW)
        await scheduler.schedule(make_task(id="critical"), "agent-1", priority=Priority.CRITICAL)
        await scheduler.schedule(make_task(id="normal"), "agent-1", priority=Priority.NORMAL)

        # Critical should come first
        t1 = await scheduler.pop_next("agent-1")
        assert t1.id == "critical"

        t2 = await scheduler.pop_next("agent-1")
        assert t2.id == "normal"

        t3 = await scheduler.pop_next("agent-1")
        assert t3.id == "low"


class TestCompleteTask:
    @pytest.mark.asyncio
    async def test_complete_success(self, scheduler):
        await scheduler.schedule(make_task(), "agent-1")
        await scheduler.pop_next("agent-1")
        await scheduler.complete_task("t1", result={"output": "done"})

        handle = await scheduler.get_handle("t1")
        assert handle.status == TaskStatus.COMPLETED
        assert handle.result == {"output": "done"}
        assert handle.completed_at is not None

    @pytest.mark.asyncio
    async def test_complete_failure(self, scheduler):
        await scheduler.schedule(make_task(), "agent-1")
        await scheduler.pop_next("agent-1")
        await scheduler.complete_task("t1", error="Something broke")

        handle = await scheduler.get_handle("t1")
        assert handle.status == TaskStatus.FAILED
        assert handle.error == "Something broke"

    @pytest.mark.asyncio
    async def test_complete_nonexistent_no_error(self, scheduler):
        await scheduler.complete_task("nonexistent")


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_pending(self, scheduler):
        await scheduler.schedule(make_task(), "agent-1")
        result = await scheduler.cancel("t1")
        assert result is True

        handle = await scheduler.get_handle("t1")
        assert handle.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_fails(self, scheduler):
        await scheduler.schedule(make_task(), "agent-1")
        await scheduler.pop_next("agent-1")
        await scheduler.complete_task("t1", result={})

        result = await scheduler.cancel("t1")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, scheduler):
        result = await scheduler.cancel("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_is_cancelled(self, scheduler):
        await scheduler.schedule(make_task(), "agent-1")
        assert not scheduler.is_cancelled("t1")
        await scheduler.cancel("t1")
        # Cancel token should be set
        assert scheduler.is_cancelled("t1")


class TestListTasks:
    @pytest.mark.asyncio
    async def test_list_pending(self, scheduler):
        for i in range(3):
            await scheduler.schedule(make_task(id=f"t{i}"), "agent-1")
        pending = await scheduler.list_pending("agent-1")
        assert len(pending) == 3

    @pytest.mark.asyncio
    async def test_list_running(self, scheduler):
        await scheduler.schedule(make_task(id="t1"), "agent-1")
        await scheduler.schedule(make_task(id="t2"), "agent-1")
        await scheduler.pop_next("agent-1")

        running = await scheduler.list_running("agent-1")
        assert len(running) == 1
        assert running[0].id == "t1"


class TestCallbacks:
    @pytest.mark.asyncio
    async def test_on_complete_callback(self, scheduler):
        await scheduler.schedule(make_task(), "agent-1")
        await scheduler.pop_next("agent-1")

        callback_results = []

        async def callback(handle):
            callback_results.append(handle.status)

        scheduler.on_complete("t1", callback)
        await scheduler.complete_task("t1", result={})

        assert len(callback_results) == 1
        assert callback_results[0] == TaskStatus.COMPLETED


class TestStats:
    @pytest.mark.asyncio
    async def test_stats(self, scheduler):
        await scheduler.schedule(make_task(id="t1"), "agent-1")
        await scheduler.pop_next("agent-1")
        await scheduler.complete_task("t1", result={})

        stats = await scheduler.get_stats()
        assert stats["tasks_scheduled"] == 1
        assert stats["tasks_completed"] == 1
        assert stats["tasks_failed"] == 0
