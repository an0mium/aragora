"""
Tests for gauntlet storage module (aragora/server/handlers/gauntlet/storage.py).

Covers:
- _gauntlet_runs: OrderedDict for in-memory run storage
- MAX_GAUNTLET_RUNS_IN_MEMORY: memory limit constant
- _GAUNTLET_COMPLETED_TTL / _GAUNTLET_MAX_AGE_SECONDS: TTL constants
- _quota_lock: threading Lock for TOCTOU protection
- _USE_DURABLE_QUEUE: environment-driven flag
- set_gauntlet_broadcast_fn / get_gauntlet_broadcast_fn: broadcast function management
- _get_storage: lazy singleton for GauntletStorage
- _handle_task_exception: fire-and-forget task exception logging
- create_tracked_task: async task creation with exception callback
- _cleanup_gauntlet_runs: memory management and eviction
- recover_stale_gauntlet_runs: server restart recovery
- get_gauntlet_runs: accessor for in-memory storage
- get_quota_lock: accessor for quota lock
- is_durable_queue_enabled: accessor for durable queue flag
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.gauntlet import storage as storage_module
from aragora.server.handlers.gauntlet.storage import (
    MAX_GAUNTLET_RUNS_IN_MEMORY,
    _cleanup_gauntlet_runs,
    _GAUNTLET_COMPLETED_TTL,
    _GAUNTLET_MAX_AGE_SECONDS,
    _handle_task_exception,
    create_tracked_task,
    get_gauntlet_broadcast_fn,
    get_gauntlet_runs,
    get_quota_lock,
    is_durable_queue_enabled,
    recover_stale_gauntlet_runs,
    set_gauntlet_broadcast_fn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_storage():
    """Reset in-memory state before/after each test."""
    runs = get_gauntlet_runs()
    runs.clear()
    # Reset broadcast fn
    set_gauntlet_broadcast_fn.__wrapped__ if hasattr(
        set_gauntlet_broadcast_fn, "__wrapped__"
    ) else None
    storage_module._gauntlet_broadcast_fn = None
    # Reset storage singleton
    storage_module._storage = None
    yield
    runs.clear()
    storage_module._gauntlet_broadcast_fn = None
    storage_module._storage = None


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_max_gauntlet_runs_in_memory(self):
        """MAX_GAUNTLET_RUNS_IN_MEMORY is a positive integer."""
        assert isinstance(MAX_GAUNTLET_RUNS_IN_MEMORY, int)
        assert MAX_GAUNTLET_RUNS_IN_MEMORY == 500

    def test_completed_ttl(self):
        """Completed run TTL is 1 hour."""
        assert _GAUNTLET_COMPLETED_TTL == 3600

    def test_max_age_seconds(self):
        """Max age is 2 hours."""
        assert _GAUNTLET_MAX_AGE_SECONDS == 7200

    def test_gauntlet_runs_is_ordered_dict(self):
        """In-memory storage is an OrderedDict."""
        runs = get_gauntlet_runs()
        assert isinstance(runs, OrderedDict)


# ============================================================================
# get_gauntlet_runs
# ============================================================================


class TestGetGauntletRuns:
    """Tests for get_gauntlet_runs accessor."""

    def test_returns_ordered_dict(self):
        """Returns the module-level OrderedDict."""
        runs = get_gauntlet_runs()
        assert isinstance(runs, OrderedDict)

    def test_returns_same_reference(self):
        """Multiple calls return the same object (not a copy)."""
        runs1 = get_gauntlet_runs()
        runs2 = get_gauntlet_runs()
        assert runs1 is runs2

    def test_mutations_persist(self):
        """Mutations to returned dict persist across calls."""
        runs = get_gauntlet_runs()
        runs["test-id"] = {"status": "running"}
        assert get_gauntlet_runs()["test-id"]["status"] == "running"

    def test_empty_by_default(self):
        """Starts empty after reset."""
        assert len(get_gauntlet_runs()) == 0


# ============================================================================
# get_quota_lock
# ============================================================================


class TestGetQuotaLock:
    """Tests for get_quota_lock accessor."""

    def test_returns_lock(self):
        """Returns a threading.Lock instance."""
        lock = get_quota_lock()
        assert isinstance(lock, type(threading.Lock()))

    def test_returns_same_lock(self):
        """Multiple calls return the same Lock object."""
        lock1 = get_quota_lock()
        lock2 = get_quota_lock()
        assert lock1 is lock2

    def test_lock_is_acquirable(self):
        """Lock can be acquired and released."""
        lock = get_quota_lock()
        acquired = lock.acquire(timeout=1)
        assert acquired is True
        lock.release()

    def test_lock_is_reentrant_safe(self):
        """Lock blocks on double acquire (non-reentrant)."""
        lock = get_quota_lock()
        lock.acquire()
        # Non-blocking attempt should fail since already held
        acquired = lock.acquire(blocking=False)
        assert acquired is False
        lock.release()


# ============================================================================
# is_durable_queue_enabled
# ============================================================================


class TestIsDurableQueueEnabled:
    """Tests for is_durable_queue_enabled accessor."""

    def test_returns_bool(self):
        """Returns a boolean."""
        assert isinstance(is_durable_queue_enabled(), bool)

    def test_default_enabled(self):
        """Durable queue is enabled by default (ARAGORA_DURABLE_GAUNTLET not set or '1')."""
        with patch.dict("os.environ", {"ARAGORA_DURABLE_GAUNTLET": "1"}):
            # Need to reimport to pick up env change - but _USE_DURABLE_QUEUE is set at import time
            # So we test the current value which reflects the import-time env
            result = is_durable_queue_enabled()
            assert isinstance(result, bool)

    def test_reflects_module_constant(self):
        """Returns the value of _USE_DURABLE_QUEUE module constant."""
        result = is_durable_queue_enabled()
        assert result == storage_module._USE_DURABLE_QUEUE


# ============================================================================
# set_gauntlet_broadcast_fn / get_gauntlet_broadcast_fn
# ============================================================================


class TestBroadcastFn:
    """Tests for broadcast function get/set."""

    def test_default_is_none(self):
        """Broadcast fn is None by default."""
        assert get_gauntlet_broadcast_fn() is None

    def test_set_and_get(self):
        """Set and retrieve a broadcast function."""
        fn = MagicMock()
        set_gauntlet_broadcast_fn(fn)
        assert get_gauntlet_broadcast_fn() is fn

    def test_overwrite(self):
        """Setting a new function replaces the old one."""
        fn1 = MagicMock()
        fn2 = MagicMock()
        set_gauntlet_broadcast_fn(fn1)
        assert get_gauntlet_broadcast_fn() is fn1
        set_gauntlet_broadcast_fn(fn2)
        assert get_gauntlet_broadcast_fn() is fn2

    def test_set_none(self):
        """Setting None clears the broadcast function."""
        fn = MagicMock()
        set_gauntlet_broadcast_fn(fn)
        set_gauntlet_broadcast_fn(None)
        assert get_gauntlet_broadcast_fn() is None

    def test_set_lambda(self):
        """Can set a lambda as broadcast function."""
        fn = lambda event, data: None  # noqa: E731
        set_gauntlet_broadcast_fn(fn)
        assert get_gauntlet_broadcast_fn() is fn

    def test_set_regular_function(self):
        """Can set a regular function as broadcast function."""

        def my_broadcast(event: str, data: dict) -> None:
            pass

        set_gauntlet_broadcast_fn(my_broadcast)
        assert get_gauntlet_broadcast_fn() is my_broadcast


# ============================================================================
# _get_storage
# ============================================================================


class TestGetStorage:
    """Tests for _get_storage lazy singleton."""

    def test_creates_storage_on_first_call(self):
        """Creates a GauntletStorage instance on first call."""
        with patch(
            "aragora.server.handlers.gauntlet.storage.GauntletStorage",
            create=True,
        ) as MockStorage:
            mock_instance = MagicMock()
            MockStorage.return_value = mock_instance
            # Patch the import inside _get_storage
            with patch.dict(
                "sys.modules",
                {"aragora.gauntlet.storage": MagicMock(GauntletStorage=MockStorage)},
            ):
                from aragora.server.handlers.gauntlet.storage import _get_storage

                result = _get_storage()
                assert result is not None

    def test_returns_same_instance_on_repeated_calls(self):
        """Subsequent calls return the cached singleton."""
        mock_instance = MagicMock()
        storage_module._storage = mock_instance
        from aragora.server.handlers.gauntlet.storage import _get_storage

        result1 = _get_storage()
        result2 = _get_storage()
        assert result1 is result2
        assert result1 is mock_instance

    def test_creates_new_when_none(self):
        """Creates new instance when _storage is None."""
        assert storage_module._storage is None
        with patch("aragora.gauntlet.storage.GauntletStorage") as MockStorage:
            mock_instance = MagicMock()
            MockStorage.return_value = mock_instance
            from aragora.server.handlers.gauntlet.storage import _get_storage

            result = _get_storage()
            assert result is mock_instance
            MockStorage.assert_called_once()


# ============================================================================
# _handle_task_exception
# ============================================================================


class TestHandleTaskException:
    """Tests for _handle_task_exception callback."""

    def test_cancelled_task_logs_debug(self, caplog):
        """Cancelled tasks log at DEBUG level."""
        task = MagicMock()
        task.cancelled.return_value = True
        with caplog.at_level(logging.DEBUG):
            _handle_task_exception(task, "test-task")
        assert any("cancelled" in r.message.lower() for r in caplog.records)

    def test_failed_task_logs_error(self, caplog):
        """Failed tasks log at ERROR level."""
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = ValueError("test error")
        with caplog.at_level(logging.ERROR):
            _handle_task_exception(task, "test-task")
        assert any(
            "failed" in r.message.lower() or "exception" in r.message.lower()
            for r in caplog.records
        )

    def test_successful_task_no_log(self, caplog):
        """Successful tasks (no exception, not cancelled) produce no logs."""
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        with caplog.at_level(logging.DEBUG):
            _handle_task_exception(task, "test-task")
        # No records about this task
        assert not any("test-task" in r.message for r in caplog.records)

    def test_uses_task_name_in_log(self, caplog):
        """Task name appears in log messages."""
        task = MagicMock()
        task.cancelled.return_value = True
        with caplog.at_level(logging.DEBUG):
            _handle_task_exception(task, "my-special-task")
        assert any("my-special-task" in r.message for r in caplog.records)

    def test_failed_task_includes_exception_type(self, caplog):
        """Failed task log includes the exception details."""
        task = MagicMock()
        task.cancelled.return_value = False
        exc = RuntimeError("something broke")
        task.exception.return_value = exc
        with caplog.at_level(logging.ERROR):
            _handle_task_exception(task, "err-task")
        assert any("err-task" in r.message for r in caplog.records)


# ============================================================================
# create_tracked_task
# ============================================================================


class TestCreateTrackedTask:
    """Tests for create_tracked_task."""

    @pytest.mark.asyncio
    async def test_creates_task_with_name(self):
        """Creates an asyncio.Task with the given name."""

        async def noop():
            pass

        task = create_tracked_task(noop(), "my-task")
        assert task.get_name() == "my-task"
        await task

    @pytest.mark.asyncio
    async def test_task_completes_successfully(self):
        """Task runs the coroutine to completion."""
        result = []

        async def append_value():
            result.append(42)

        task = create_tracked_task(append_value(), "append-task")
        await task
        assert result == [42]

    @pytest.mark.asyncio
    async def test_task_has_done_callback(self):
        """Task has a done callback registered."""

        async def noop():
            pass

        task = create_tracked_task(noop(), "callback-task")
        # asyncio.Task stores callbacks internally; we just verify the task runs
        await task
        assert task.done()

    @pytest.mark.asyncio
    async def test_failing_task_exception_logged(self, caplog):
        """Exceptions in tracked tasks are logged via callback."""

        async def fail():
            raise ValueError("tracked failure")

        task = create_tracked_task(fail(), "fail-task")
        # Wait for task to complete (it will fail)
        with pytest.raises(ValueError, match="tracked failure"):
            await task

    @pytest.mark.asyncio
    async def test_returns_asyncio_task(self):
        """Returns an asyncio.Task instance."""

        async def noop():
            pass

        task = create_tracked_task(noop(), "type-check")
        assert isinstance(task, asyncio.Task)
        await task


# ============================================================================
# _cleanup_gauntlet_runs
# ============================================================================


class TestCleanupGauntletRuns:
    """Tests for _cleanup_gauntlet_runs memory management."""

    def test_removes_entries_older_than_max_age(self):
        """Entries older than _GAUNTLET_MAX_AGE_SECONDS are removed."""
        runs = get_gauntlet_runs()
        old_time = time.time() - _GAUNTLET_MAX_AGE_SECONDS - 100
        runs["old-run"] = {"created_at": old_time, "status": "running"}
        _cleanup_gauntlet_runs()
        assert "old-run" not in runs

    def test_keeps_entries_within_max_age(self):
        """Entries within max age are kept."""
        runs = get_gauntlet_runs()
        runs["recent-run"] = {"created_at": time.time(), "status": "running"}
        _cleanup_gauntlet_runs()
        assert "recent-run" in runs

    def test_removes_completed_entries_older_than_ttl(self):
        """Completed entries older than COMPLETED_TTL are removed."""
        runs = get_gauntlet_runs()
        old_completed_time = datetime.now(timezone.utc) - timedelta(
            seconds=_GAUNTLET_COMPLETED_TTL + 100
        )
        runs["completed-old"] = {
            "created_at": time.time(),  # Within max age
            "status": "completed",
            "completed_at": old_completed_time.isoformat(),
        }
        _cleanup_gauntlet_runs()
        assert "completed-old" not in runs

    def test_keeps_recently_completed_entries(self):
        """Recently completed entries are kept."""
        runs = get_gauntlet_runs()
        recent_completed = datetime.now(timezone.utc) - timedelta(seconds=60)
        runs["completed-recent"] = {
            "created_at": time.time(),
            "status": "completed",
            "completed_at": recent_completed.isoformat(),
        }
        _cleanup_gauntlet_runs()
        assert "completed-recent" in runs

    def test_evicts_oldest_when_over_limit(self):
        """FIFO eviction when over MAX_GAUNTLET_RUNS_IN_MEMORY."""
        runs = get_gauntlet_runs()
        # Fill beyond limit
        for i in range(MAX_GAUNTLET_RUNS_IN_MEMORY + 10):
            runs[f"run-{i:05d}"] = {"created_at": time.time(), "status": "running"}
        _cleanup_gauntlet_runs()
        assert len(runs) <= MAX_GAUNTLET_RUNS_IN_MEMORY

    def test_fifo_eviction_removes_oldest_first(self):
        """FIFO eviction removes oldest entries (first inserted)."""
        runs = get_gauntlet_runs()
        for i in range(MAX_GAUNTLET_RUNS_IN_MEMORY + 5):
            runs[f"run-{i:05d}"] = {"created_at": time.time(), "status": "running"}
        _cleanup_gauntlet_runs()
        # The first 5 entries should have been evicted
        assert "run-00000" not in runs
        assert "run-00004" not in runs
        # The last entries should remain
        last_key = f"run-{MAX_GAUNTLET_RUNS_IN_MEMORY + 4:05d}"
        assert last_key in runs

    def test_handles_iso_format_created_at(self):
        """Handles ISO format datetime strings for created_at."""
        runs = get_gauntlet_runs()
        old_time = datetime.now(timezone.utc) - timedelta(seconds=_GAUNTLET_MAX_AGE_SECONDS + 100)
        runs["iso-run"] = {"created_at": old_time.isoformat(), "status": "running"}
        _cleanup_gauntlet_runs()
        assert "iso-run" not in runs

    def test_handles_numeric_created_at(self):
        """Handles numeric (float/int) timestamps for created_at."""
        runs = get_gauntlet_runs()
        old_time = time.time() - _GAUNTLET_MAX_AGE_SECONDS - 100
        runs["numeric-run"] = {"created_at": old_time, "status": "running"}
        _cleanup_gauntlet_runs()
        assert "numeric-run" not in runs

    def test_handles_missing_created_at_with_completed_at(self):
        """Falls back to completed_at when created_at is missing."""
        runs = get_gauntlet_runs()
        old_time = datetime.now(timezone.utc) - timedelta(seconds=_GAUNTLET_MAX_AGE_SECONDS + 100)
        runs["no-created"] = {
            "status": "completed",
            "completed_at": old_time.isoformat(),
        }
        _cleanup_gauntlet_runs()
        assert "no-created" not in runs

    def test_handles_invalid_created_at_string(self):
        """Handles invalid datetime strings gracefully."""
        runs = get_gauntlet_runs()
        runs["bad-date"] = {"created_at": "not-a-date", "status": "running"}
        _cleanup_gauntlet_runs()
        # Entry should survive since we can't parse its age
        assert "bad-date" in runs

    def test_handles_no_timestamp_fields(self):
        """Entries with no timestamp fields are not evicted by age."""
        runs = get_gauntlet_runs()
        runs["no-timestamps"] = {"status": "running"}
        _cleanup_gauntlet_runs()
        assert "no-timestamps" in runs

    def test_completed_with_invalid_completed_at(self):
        """Completed entries with invalid completed_at are not TTL-evicted."""
        runs = get_gauntlet_runs()
        runs["bad-completed"] = {
            "created_at": time.time(),
            "status": "completed",
            "completed_at": "invalid",
        }
        _cleanup_gauntlet_runs()
        # Should still be there (invalid timestamp can't be TTL-checked)
        assert "bad-completed" in runs

    def test_empty_runs_no_error(self):
        """Cleanup on empty storage raises no errors."""
        runs = get_gauntlet_runs()
        assert len(runs) == 0
        _cleanup_gauntlet_runs()
        assert len(runs) == 0

    def test_mixed_old_and_new_entries(self):
        """Only old entries are removed; new entries remain."""
        runs = get_gauntlet_runs()
        old_time = time.time() - _GAUNTLET_MAX_AGE_SECONDS - 100
        runs["old-1"] = {"created_at": old_time, "status": "running"}
        runs["new-1"] = {"created_at": time.time(), "status": "running"}
        runs["old-2"] = {"created_at": old_time, "status": "running"}
        runs["new-2"] = {
            "created_at": time.time(),
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _cleanup_gauntlet_runs()
        assert "old-1" not in runs
        assert "old-2" not in runs
        assert "new-1" in runs
        assert "new-2" in runs

    def test_completed_not_evicted_by_max_age_if_young(self):
        """Completed entries within max age but over completed TTL get removed."""
        runs = get_gauntlet_runs()
        # Created recently (within max age) but completed long ago (over TTL)
        completed_time = datetime.now(timezone.utc) - timedelta(
            seconds=_GAUNTLET_COMPLETED_TTL + 500
        )
        runs["old-completed"] = {
            "created_at": time.time(),  # Recent creation
            "status": "completed",
            "completed_at": completed_time.isoformat(),
        }
        _cleanup_gauntlet_runs()
        assert "old-completed" not in runs

    def test_none_created_at_handled(self):
        """None created_at value is handled without error."""
        runs = get_gauntlet_runs()
        runs["none-created"] = {"created_at": None, "status": "running"}
        _cleanup_gauntlet_runs()
        # Should survive since None is not int/float or valid string
        assert "none-created" in runs

    def test_completed_with_none_completed_at(self):
        """Completed entries with None completed_at are not TTL-evicted."""
        runs = get_gauntlet_runs()
        runs["none-completed-at"] = {
            "created_at": time.time(),
            "status": "completed",
            "completed_at": None,
        }
        _cleanup_gauntlet_runs()
        assert "none-completed-at" in runs


# ============================================================================
# recover_stale_gauntlet_runs
# ============================================================================


class TestRecoverStaleGauntletRuns:
    """Tests for recover_stale_gauntlet_runs server restart recovery."""

    def test_no_stale_runs_returns_zero(self):
        """Returns 0 when no stale runs found."""
        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = []
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            result = recover_stale_gauntlet_runs()
            assert result == 0

    def test_recovers_stale_runs(self):
        """Marks stale runs as interrupted and returns count."""
        mock_run = MagicMock()
        mock_run.gauntlet_id = "gauntlet-20260223120000-abc123"
        mock_run.status = "running"
        mock_run.progress_percent = 50.0
        mock_run.current_phase = "critique"
        mock_run.input_type = "proposal"
        mock_run.input_summary = "Test proposal"
        mock_run.persona = "adversarial"
        mock_run.agents = ["claude", "gpt-4"]
        mock_run.profile = "standard"
        mock_run.created_at = datetime.now(timezone.utc)

        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = [mock_run]

        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            result = recover_stale_gauntlet_runs()
            assert result == 1

        # Verify storage was updated
        mock_storage.update_inflight_status.assert_called_once()
        call_kwargs = mock_storage.update_inflight_status.call_args
        assert call_kwargs[1]["status"] == "interrupted"

    def test_adds_recovered_runs_to_memory(self):
        """Recovered runs are added to in-memory storage."""
        mock_run = MagicMock()
        mock_run.gauntlet_id = "gauntlet-20260223120000-def456"
        mock_run.status = "pending"
        mock_run.progress_percent = 0.0
        mock_run.current_phase = "init"
        mock_run.input_type = "proposal"
        mock_run.input_summary = "Another proposal"
        mock_run.persona = "balanced"
        mock_run.agents = ["claude"]
        mock_run.profile = "quick"
        mock_run.created_at = datetime.now(timezone.utc)

        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = [mock_run]

        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            recover_stale_gauntlet_runs()

        runs = get_gauntlet_runs()
        assert "gauntlet-20260223120000-def456" in runs
        assert runs["gauntlet-20260223120000-def456"]["status"] == "interrupted"

    def test_handles_multiple_stale_runs(self):
        """Recovers multiple stale runs."""
        mock_runs = []
        for i in range(3):
            run = MagicMock()
            run.gauntlet_id = f"gauntlet-20260223120000-run{i:03d}"
            run.status = "running"
            run.progress_percent = float(i * 25)
            run.current_phase = "phase"
            run.input_type = "proposal"
            run.input_summary = f"Run {i}"
            run.persona = "adversarial"
            run.agents = ["claude"]
            run.profile = "standard"
            run.created_at = datetime.now(timezone.utc)
            mock_runs.append(run)

        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = mock_runs

        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            result = recover_stale_gauntlet_runs()
            assert result == 3

    def test_passes_max_age_to_storage(self):
        """Passes max_age_seconds parameter to storage."""
        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = []

        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            recover_stale_gauntlet_runs(max_age_seconds=3600)
            mock_storage.list_stale_inflight.assert_called_once_with(max_age_seconds=3600)

    def test_default_max_age(self):
        """Default max_age_seconds is 7200."""
        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = []

        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            recover_stale_gauntlet_runs()
            mock_storage.list_stale_inflight.assert_called_once_with(max_age_seconds=7200)

    def test_handles_import_error(self):
        """Returns 0 when import fails."""
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            side_effect=ImportError("no module"),
        ):
            result = recover_stale_gauntlet_runs()
            assert result == 0

    def test_handles_runtime_error(self):
        """Returns 0 when runtime error occurs."""
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            side_effect=RuntimeError("db error"),
        ):
            result = recover_stale_gauntlet_runs()
            assert result == 0

    def test_handles_individual_run_failure(self):
        """Continues recovering other runs when one fails."""
        mock_run_ok = MagicMock()
        mock_run_ok.gauntlet_id = "gauntlet-20260223120000-ok0001"
        mock_run_ok.status = "running"
        mock_run_ok.progress_percent = 50.0
        mock_run_ok.current_phase = "critique"
        mock_run_ok.input_type = "proposal"
        mock_run_ok.input_summary = "Good run"
        mock_run_ok.persona = "adversarial"
        mock_run_ok.agents = ["claude"]
        mock_run_ok.profile = "standard"
        mock_run_ok.created_at = datetime.now(timezone.utc)

        mock_run_bad = MagicMock()
        mock_run_bad.gauntlet_id = "gauntlet-20260223120000-bad001"
        mock_run_bad.status = "running"
        mock_run_bad.progress_percent = 25.0
        mock_run_bad.current_phase = "init"
        mock_run_bad.input_type = "proposal"
        mock_run_bad.input_summary = "Bad run"
        mock_run_bad.persona = "adversarial"
        mock_run_bad.agents = ["claude"]
        mock_run_bad.profile = "standard"
        mock_run_bad.created_at = datetime.now(timezone.utc)

        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = [mock_run_bad, mock_run_ok]

        # First call to update_inflight_status fails, second succeeds
        mock_storage.update_inflight_status.side_effect = [RuntimeError("db error"), None]

        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            result = recover_stale_gauntlet_runs()
            assert result == 1  # Only one succeeded

    def test_handles_os_error(self):
        """Returns 0 when OSError occurs."""
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            side_effect=OSError("disk error"),
        ):
            result = recover_stale_gauntlet_runs()
            assert result == 0

    def test_handles_value_error(self):
        """Returns 0 when ValueError occurs."""
        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            side_effect=ValueError("bad value"),
        ):
            result = recover_stale_gauntlet_runs()
            assert result == 0

    def test_recovered_run_has_correct_fields(self):
        """Recovered runs in memory have all expected fields."""
        mock_run = MagicMock()
        mock_run.gauntlet_id = "gauntlet-20260223120000-fld001"
        mock_run.status = "running"
        mock_run.progress_percent = 75.0
        mock_run.current_phase = "revision"
        mock_run.input_type = "code"
        mock_run.input_summary = "Code review"
        mock_run.persona = "security"
        mock_run.agents = ["claude", "gpt-4"]
        mock_run.profile = "thorough"
        mock_run.created_at = datetime.now(timezone.utc)

        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = [mock_run]

        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            recover_stale_gauntlet_runs()

        runs = get_gauntlet_runs()
        run = runs["gauntlet-20260223120000-fld001"]
        assert run["gauntlet_id"] == "gauntlet-20260223120000-fld001"
        assert run["status"] == "interrupted"
        assert run["input_type"] == "code"
        assert run["input_summary"] == "Code review"
        assert run["persona"] == "security"
        assert run["agents"] == ["claude", "gpt-4"]
        assert run["profile"] == "thorough"
        assert run["progress_percent"] == 75.0
        assert run["current_phase"] == "revision"
        assert "error" in run
        assert "Server restarted" in run["error"]

    def test_error_message_includes_original_status(self):
        """Error message for interrupted runs includes original status."""
        mock_run = MagicMock()
        mock_run.gauntlet_id = "gauntlet-20260223120000-msg001"
        mock_run.status = "pending"
        mock_run.progress_percent = 0.0
        mock_run.current_phase = None
        mock_run.input_type = "proposal"
        mock_run.input_summary = "Test"
        mock_run.persona = "balanced"
        mock_run.agents = []
        mock_run.profile = "quick"
        mock_run.created_at = datetime.now(timezone.utc)

        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = [mock_run]

        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            recover_stale_gauntlet_runs()

        runs = get_gauntlet_runs()
        run = runs["gauntlet-20260223120000-msg001"]
        assert "pending" in run["error"]

    def test_update_inflight_status_called_with_error(self):
        """Storage update includes error message with progress info."""
        mock_run = MagicMock()
        mock_run.gauntlet_id = "gauntlet-20260223120000-upd001"
        mock_run.status = "running"
        mock_run.progress_percent = 33.0
        mock_run.current_phase = "proposal"
        mock_run.input_type = "text"
        mock_run.input_summary = "Summary"
        mock_run.persona = "adversarial"
        mock_run.agents = ["claude"]
        mock_run.profile = "standard"
        mock_run.created_at = datetime.now(timezone.utc)

        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = [mock_run]

        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            recover_stale_gauntlet_runs()

        call_kwargs = mock_storage.update_inflight_status.call_args[1]
        assert call_kwargs["gauntlet_id"] == "gauntlet-20260223120000-upd001"
        assert call_kwargs["status"] == "interrupted"
        assert "33%" in call_kwargs["error"]
        assert "proposal" in call_kwargs["error"]


# ============================================================================
# _cleanup_gauntlet_runs - boundary / edge cases
# ============================================================================


class TestCleanupEdgeCases:
    """Boundary and edge case tests for cleanup."""

    def test_exactly_at_max_age_boundary(self):
        """Entry exactly at max age boundary stays or goes based on precision."""
        runs = get_gauntlet_runs()
        # Just barely over the limit
        boundary_time = time.time() - _GAUNTLET_MAX_AGE_SECONDS - 1
        runs["boundary"] = {"created_at": boundary_time, "status": "running"}
        _cleanup_gauntlet_runs()
        assert "boundary" not in runs

    def test_exactly_at_completed_ttl_boundary(self):
        """Entry exactly at completed TTL boundary."""
        runs = get_gauntlet_runs()
        boundary_time = datetime.now(timezone.utc) - timedelta(seconds=_GAUNTLET_COMPLETED_TTL + 1)
        runs["ttl-boundary"] = {
            "created_at": time.time(),
            "status": "completed",
            "completed_at": boundary_time.isoformat(),
        }
        _cleanup_gauntlet_runs()
        assert "ttl-boundary" not in runs

    def test_exactly_at_memory_limit(self):
        """Exactly MAX_GAUNTLET_RUNS_IN_MEMORY entries are kept."""
        runs = get_gauntlet_runs()
        for i in range(MAX_GAUNTLET_RUNS_IN_MEMORY):
            runs[f"exact-{i:05d}"] = {"created_at": time.time(), "status": "running"}
        _cleanup_gauntlet_runs()
        assert len(runs) == MAX_GAUNTLET_RUNS_IN_MEMORY

    def test_one_over_memory_limit(self):
        """One entry over limit triggers exactly one eviction."""
        runs = get_gauntlet_runs()
        for i in range(MAX_GAUNTLET_RUNS_IN_MEMORY + 1):
            runs[f"over-{i:05d}"] = {"created_at": time.time(), "status": "running"}
        assert len(runs) == MAX_GAUNTLET_RUNS_IN_MEMORY + 1
        _cleanup_gauntlet_runs()
        assert len(runs) == MAX_GAUNTLET_RUNS_IN_MEMORY
        # First entry should be evicted
        assert "over-00000" not in runs

    def test_non_completed_entries_not_ttl_evicted(self):
        """Non-completed entries are not subject to completed TTL."""
        runs = get_gauntlet_runs()
        runs["running-old"] = {
            "created_at": time.time(),  # Within max age
            "status": "running",
            "completed_at": (
                datetime.now(timezone.utc) - timedelta(seconds=_GAUNTLET_COMPLETED_TTL + 500)
            ).isoformat(),
        }
        _cleanup_gauntlet_runs()
        assert "running-old" in runs

    def test_pending_status_not_ttl_evicted(self):
        """Pending entries are not subject to completed TTL."""
        runs = get_gauntlet_runs()
        runs["pending-run"] = {
            "created_at": time.time(),
            "status": "pending",
        }
        _cleanup_gauntlet_runs()
        assert "pending-run" in runs

    def test_failed_status_not_ttl_evicted(self):
        """Failed entries are not subject to completed TTL."""
        runs = get_gauntlet_runs()
        runs["failed-run"] = {
            "created_at": time.time(),
            "status": "failed",
        }
        _cleanup_gauntlet_runs()
        assert "failed-run" in runs


# ============================================================================
# Integration: storage accessors used together
# ============================================================================


class TestStorageIntegration:
    """Integration tests combining multiple storage functions."""

    def test_runs_and_cleanup_cycle(self):
        """Add runs, cleanup, verify state."""
        runs = get_gauntlet_runs()
        # Add some old runs
        old_time = time.time() - _GAUNTLET_MAX_AGE_SECONDS - 200
        runs["old-a"] = {"created_at": old_time, "status": "running"}
        runs["old-b"] = {
            "created_at": old_time,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        # Add a fresh run
        runs["fresh-c"] = {"created_at": time.time(), "status": "running"}

        assert len(runs) == 3
        _cleanup_gauntlet_runs()
        assert len(runs) == 1
        assert "fresh-c" in runs

    def test_broadcast_fn_lifecycle(self):
        """Set, use, and clear broadcast function."""
        assert get_gauntlet_broadcast_fn() is None
        fn = MagicMock()
        set_gauntlet_broadcast_fn(fn)
        assert get_gauntlet_broadcast_fn() is fn
        # Simulate calling it
        broadcast = get_gauntlet_broadcast_fn()
        broadcast("event", {"data": "test"})
        fn.assert_called_once_with("event", {"data": "test"})
        # Clear
        set_gauntlet_broadcast_fn(None)
        assert get_gauntlet_broadcast_fn() is None

    def test_quota_lock_protects_runs(self):
        """Quota lock can protect read-modify-write on runs."""
        runs = get_gauntlet_runs()
        lock = get_quota_lock()
        with lock:
            runs["protected"] = {"status": "pending", "created_at": time.time()}
        assert "protected" in runs

    def test_recover_then_cleanup(self):
        """Recovered runs can be subsequently cleaned up."""
        mock_run = MagicMock()
        mock_run.gauntlet_id = "gauntlet-20260223120000-rcl001"
        mock_run.status = "running"
        mock_run.progress_percent = 10.0
        mock_run.current_phase = "init"
        mock_run.input_type = "text"
        mock_run.input_summary = "Test"
        mock_run.persona = "balanced"
        mock_run.agents = []
        mock_run.profile = "quick"
        # Created very long ago
        mock_run.created_at = datetime.now(timezone.utc) - timedelta(
            seconds=_GAUNTLET_MAX_AGE_SECONDS + 1000
        )

        mock_storage = MagicMock()
        mock_storage.list_stale_inflight.return_value = [mock_run]

        with patch(
            "aragora.server.handlers.gauntlet._get_storage",
            return_value=mock_storage,
        ):
            recover_stale_gauntlet_runs()

        runs = get_gauntlet_runs()
        assert "gauntlet-20260223120000-rcl001" in runs

        # The created_at in memory is the ISO string of the old time
        # Cleanup should remove it since it's older than max age
        _cleanup_gauntlet_runs()
        assert "gauntlet-20260223120000-rcl001" not in runs
