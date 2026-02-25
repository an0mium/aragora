"""Tests for aragora.debate.operator_intervention.

Covers DebateInterventionManager logic:
- register / unregister
- pause / resume lifecycle
- restart scheduling and consumption
- inject_context storage and consumption
- get_status / list_active
- asyncio.Event-based wait_if_paused
- edge cases (double pause, resume when running, etc.)
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch

import pytest

from aragora.debate.operator_intervention import (
    DebateInterventionManager,
    InterventionStatus,
    _reset_operator_manager,
    get_operator_manager,
)


@pytest.fixture
def manager():
    """Create a fresh manager for each test."""
    return DebateInterventionManager()


@pytest.fixture(autouse=True)
def _reset_global():
    """Reset the global singleton after each test."""
    yield
    _reset_operator_manager()


# =========================================================================
# Registration
# =========================================================================


class TestRegistration:
    def test_register_new_debate(self, manager):
        manager.register("d1", total_rounds=5)
        status = manager.get_status("d1")
        assert status is not None
        assert status.debate_id == "d1"
        assert status.state == "running"
        assert status.total_rounds == 5
        assert status.current_round == 0

    def test_register_updates_total_rounds(self, manager):
        manager.register("d1", total_rounds=3)
        manager.register("d1", total_rounds=7)
        status = manager.get_status("d1")
        assert status.total_rounds == 7

    def test_unregister_removes_debate(self, manager):
        manager.register("d1", total_rounds=3)
        manager.unregister("d1")
        assert manager.get_status("d1") is None

    def test_unregister_nonexistent_is_noop(self, manager):
        manager.unregister("nonexistent")  # Should not raise

    def test_update_round(self, manager):
        manager.register("d1", total_rounds=5)
        manager.update_round("d1", 3)
        status = manager.get_status("d1")
        assert status.current_round == 3

    def test_update_round_nonexistent_is_noop(self, manager):
        manager.update_round("nonexistent", 1)  # Should not raise


# =========================================================================
# Pause / Resume
# =========================================================================


class TestPauseResume:
    def test_pause_running_debate(self, manager):
        manager.register("d1", total_rounds=5)
        result = manager.pause("d1", reason="Checking results")
        assert result is True
        status = manager.get_status("d1")
        assert status.state == "paused"
        assert status.pause_reason == "Checking results"
        assert status.paused_at is not None

    def test_pause_records_intervention(self, manager):
        manager.register("d1", total_rounds=5)
        manager.pause("d1", reason="Review")
        status = manager.get_status("d1")
        assert len(status.interventions) == 1
        assert status.interventions[0]["action"] == "pause"
        assert status.interventions[0]["reason"] == "Review"

    def test_pause_empty_reason(self, manager):
        manager.register("d1")
        result = manager.pause("d1")
        assert result is True
        status = manager.get_status("d1")
        assert status.pause_reason is None

    def test_pause_nonexistent_returns_false(self, manager):
        assert manager.pause("nonexistent") is False

    def test_pause_already_paused_returns_false(self, manager):
        manager.register("d1")
        manager.pause("d1")
        assert manager.pause("d1") is False

    def test_pause_completed_returns_false(self, manager):
        manager.register("d1")
        manager.mark_completed("d1")
        assert manager.pause("d1") is False

    def test_resume_paused_debate(self, manager):
        manager.register("d1")
        manager.pause("d1", reason="test")
        result = manager.resume("d1")
        assert result is True
        status = manager.get_status("d1")
        assert status.state == "running"
        assert status.paused_at is None
        assert status.pause_reason is None

    def test_resume_records_intervention(self, manager):
        manager.register("d1")
        manager.pause("d1")
        manager.resume("d1")
        status = manager.get_status("d1")
        assert len(status.interventions) == 2
        assert status.interventions[1]["action"] == "resume"

    def test_resume_nonexistent_returns_false(self, manager):
        assert manager.resume("nonexistent") is False

    def test_resume_running_returns_false(self, manager):
        manager.register("d1")
        assert manager.resume("d1") is False

    def test_pause_resume_cycle(self, manager):
        manager.register("d1")
        assert manager.pause("d1") is True
        assert manager.resume("d1") is True
        assert manager.pause("d1", reason="second") is True
        assert manager.resume("d1") is True
        status = manager.get_status("d1")
        assert status.state == "running"
        assert len(status.interventions) == 4


# =========================================================================
# Restart
# =========================================================================


class TestRestart:
    def test_restart_from_beginning(self, manager):
        manager.register("d1", total_rounds=5)
        manager.update_round("d1", 3)
        result = manager.restart("d1", from_round=0)
        assert result is True
        status = manager.get_status("d1")
        assert status.state == "running"

    def test_restart_from_specific_round(self, manager):
        manager.register("d1", total_rounds=5)
        result = manager.restart("d1", from_round=2)
        assert result is True

    def test_restart_records_intervention(self, manager):
        manager.register("d1")
        manager.restart("d1", from_round=3)
        status = manager.get_status("d1")
        assert len(status.interventions) == 1
        assert status.interventions[0]["action"] == "restart"
        assert status.interventions[0]["details"]["from_round"] == 3

    def test_restart_paused_debate_resumes(self, manager):
        manager.register("d1")
        manager.pause("d1")
        result = manager.restart("d1", from_round=0)
        assert result is True
        status = manager.get_status("d1")
        assert status.state == "running"
        assert status.paused_at is None

    def test_restart_completed_returns_false(self, manager):
        manager.register("d1")
        manager.mark_completed("d1")
        assert manager.restart("d1") is False

    def test_restart_nonexistent_returns_false(self, manager):
        assert manager.restart("nonexistent") is False

    def test_consume_restart(self, manager):
        manager.register("d1")
        manager.restart("d1", from_round=2)
        result = manager.consume_restart("d1")
        assert result == 2
        # Second consume returns None (already consumed)
        assert manager.consume_restart("d1") is None

    def test_consume_restart_no_restart_pending(self, manager):
        manager.register("d1")
        assert manager.consume_restart("d1") is None

    def test_consume_restart_nonexistent(self, manager):
        assert manager.consume_restart("nonexistent") is None

    def test_restart_negative_round_clamps_to_zero(self, manager):
        manager.register("d1")
        manager.restart("d1", from_round=-5)
        result = manager.consume_restart("d1")
        assert result == 0


# =========================================================================
# Inject Context
# =========================================================================


class TestInjectContext:
    def test_inject_context_running(self, manager):
        manager.register("d1")
        result = manager.inject_context("d1", "Additional info here")
        assert result is True

    def test_inject_context_paused(self, manager):
        manager.register("d1")
        manager.pause("d1")
        result = manager.inject_context("d1", "While paused")
        assert result is True

    def test_inject_context_completed_returns_false(self, manager):
        manager.register("d1")
        manager.mark_completed("d1")
        assert manager.inject_context("d1", "Too late") is False

    def test_inject_context_failed_returns_false(self, manager):
        manager.register("d1")
        manager.mark_failed("d1")
        assert manager.inject_context("d1", "Too late") is False

    def test_inject_context_nonexistent_returns_false(self, manager):
        assert manager.inject_context("nonexistent", "hello") is False

    def test_inject_empty_context_returns_false(self, manager):
        manager.register("d1")
        assert manager.inject_context("d1", "") is False
        assert manager.inject_context("d1", "   ") is False

    def test_inject_context_records_intervention(self, manager):
        manager.register("d1")
        manager.inject_context("d1", "Extra context")
        status = manager.get_status("d1")
        assert len(status.interventions) == 1
        assert status.interventions[0]["action"] == "inject_context"

    def test_consume_injected_contexts(self, manager):
        manager.register("d1")
        manager.inject_context("d1", "First")
        manager.inject_context("d1", "Second")
        contexts = manager.consume_injected_contexts("d1")
        assert contexts == ["First", "Second"]
        # Second consume returns empty
        assert manager.consume_injected_contexts("d1") == []

    def test_consume_injected_contexts_nonexistent(self, manager):
        assert manager.consume_injected_contexts("nonexistent") == []

    def test_inject_context_strips_whitespace(self, manager):
        manager.register("d1")
        manager.inject_context("d1", "  trimmed  ")
        contexts = manager.consume_injected_contexts("d1")
        assert contexts == ["trimmed"]


# =========================================================================
# Status / List Active
# =========================================================================


class TestStatusAndListing:
    def test_get_status_returns_none_for_unknown(self, manager):
        assert manager.get_status("unknown") is None

    def test_get_status_returns_intervention_status(self, manager):
        manager.register("d1", total_rounds=5)
        manager.update_round("d1", 2)
        status = manager.get_status("d1")
        assert isinstance(status, InterventionStatus)
        assert status.debate_id == "d1"
        assert status.state == "running"
        assert status.current_round == 2
        assert status.total_rounds == 5
        assert status.paused_at is None
        assert status.interventions == []

    def test_get_status_to_dict(self, manager):
        manager.register("d1", total_rounds=3)
        status = manager.get_status("d1")
        d = status.to_dict()
        assert d["debate_id"] == "d1"
        assert d["state"] == "running"
        assert d["total_rounds"] == 3

    def test_list_active_empty(self, manager):
        assert manager.list_active() == []

    def test_list_active_excludes_completed(self, manager):
        manager.register("d1")
        manager.register("d2")
        manager.mark_completed("d1")
        active = manager.list_active()
        assert len(active) == 1
        assert active[0].debate_id == "d2"

    def test_list_active_excludes_failed(self, manager):
        manager.register("d1")
        manager.register("d2")
        manager.mark_failed("d1")
        active = manager.list_active()
        assert len(active) == 1
        assert active[0].debate_id == "d2"

    def test_list_active_includes_paused(self, manager):
        manager.register("d1")
        manager.register("d2")
        manager.pause("d1")
        active = manager.list_active()
        assert len(active) == 2
        states = {s.debate_id: s.state for s in active}
        assert states["d1"] == "paused"
        assert states["d2"] == "running"


# =========================================================================
# Mark completed / failed
# =========================================================================


class TestMarkStates:
    def test_mark_completed(self, manager):
        manager.register("d1")
        manager.mark_completed("d1")
        status = manager.get_status("d1")
        assert status.state == "completed"

    def test_mark_failed(self, manager):
        manager.register("d1")
        manager.mark_failed("d1")
        status = manager.get_status("d1")
        assert status.state == "failed"

    def test_mark_completed_nonexistent_is_noop(self, manager):
        manager.mark_completed("nonexistent")  # Should not raise

    def test_mark_failed_nonexistent_is_noop(self, manager):
        manager.mark_failed("nonexistent")  # Should not raise


# =========================================================================
# asyncio.Event-based pause waiting
# =========================================================================


class TestWaitIfPaused:
    @pytest.mark.asyncio
    async def test_wait_if_paused_returns_immediately_when_running(self, manager):
        manager.register("d1")
        # Should return immediately (not block)
        await asyncio.wait_for(manager.wait_if_paused("d1"), timeout=1.0)

    @pytest.mark.asyncio
    async def test_wait_if_paused_blocks_when_paused(self, manager):
        manager.register("d1")
        manager.pause("d1")

        # This should block
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(manager.wait_if_paused("d1"), timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_if_paused_unblocks_on_resume(self, manager):
        manager.register("d1")
        manager.pause("d1")

        async def resume_after_delay():
            await asyncio.sleep(0.05)
            manager.resume("d1")

        task = asyncio.create_task(resume_after_delay())
        # Should unblock once resume is called
        await asyncio.wait_for(manager.wait_if_paused("d1"), timeout=1.0)
        await task

    @pytest.mark.asyncio
    async def test_wait_if_paused_nonexistent_returns_immediately(self, manager):
        await asyncio.wait_for(
            manager.wait_if_paused("nonexistent"), timeout=1.0
        )

    @pytest.mark.asyncio
    async def test_wait_if_paused_unblocks_on_restart(self, manager):
        manager.register("d1")
        manager.pause("d1")

        async def restart_after_delay():
            await asyncio.sleep(0.05)
            manager.restart("d1", from_round=0)

        task = asyncio.create_task(restart_after_delay())
        await asyncio.wait_for(manager.wait_if_paused("d1"), timeout=1.0)
        await task

    @pytest.mark.asyncio
    async def test_wait_if_paused_unblocks_on_mark_completed(self, manager):
        manager.register("d1")
        manager.pause("d1")

        async def complete_after_delay():
            await asyncio.sleep(0.05)
            manager.mark_completed("d1")

        task = asyncio.create_task(complete_after_delay())
        await asyncio.wait_for(manager.wait_if_paused("d1"), timeout=1.0)
        await task

    @pytest.mark.asyncio
    async def test_wait_if_paused_unblocks_on_mark_failed(self, manager):
        manager.register("d1")
        manager.pause("d1")

        async def fail_after_delay():
            await asyncio.sleep(0.05)
            manager.mark_failed("d1")

        task = asyncio.create_task(fail_after_delay())
        await asyncio.wait_for(manager.wait_if_paused("d1"), timeout=1.0)
        await task


# =========================================================================
# Singleton
# =========================================================================


class TestSingleton:
    def test_get_operator_manager_returns_same_instance(self):
        m1 = get_operator_manager()
        m2 = get_operator_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = get_operator_manager()
        _reset_operator_manager()
        m2 = get_operator_manager()
        assert m1 is not m2


# =========================================================================
# Thread safety
# =========================================================================


class TestThreadSafety:
    def test_concurrent_pause_resume(self, manager):
        """Multiple threads can safely pause/resume without corruption."""
        manager.register("d1", total_rounds=10)
        errors: list[Exception] = []

        def worker(n: int):
            try:
                for _ in range(50):
                    manager.pause("d1")
                    manager.resume("d1")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Some pause/resume calls may fail (wrong state) but no crashes
        assert len(errors) == 0

    def test_concurrent_register_unregister(self, manager):
        """Concurrent register/unregister does not crash."""
        errors: list[Exception] = []

        def worker(n: int):
            try:
                for i in range(50):
                    did = f"d-{n}-{i}"
                    manager.register(did)
                    manager.unregister(did)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0


# =========================================================================
# InterventionRecord serialization
# =========================================================================


class TestInterventionRecord:
    def test_record_to_dict(self):
        from aragora.debate.operator_intervention import InterventionRecord

        record = InterventionRecord(
            action="pause",
            timestamp="2026-01-01T00:00:00+00:00",
            reason="test reason",
            details={"key": "value"},
        )
        d = record.to_dict()
        assert d["action"] == "pause"
        assert d["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert d["reason"] == "test reason"
        assert d["details"] == {"key": "value"}
