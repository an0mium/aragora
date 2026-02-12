"""
Tests for gauntlet storage module.

Tests cover:
- In-memory storage management
- Cleanup and TTL logic
- Broadcast function management
- Task creation utilities
- Quota lock access
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.gauntlet.storage import (
    MAX_GAUNTLET_RUNS_IN_MEMORY,
    _GAUNTLET_COMPLETED_TTL,
    _GAUNTLET_MAX_AGE_SECONDS,
    _cleanup_gauntlet_runs,
    _gauntlet_runs,
    create_tracked_task,
    get_gauntlet_broadcast_fn,
    get_gauntlet_runs,
    get_quota_lock,
    is_durable_queue_enabled,
    set_gauntlet_broadcast_fn,
)


@pytest.fixture(autouse=True)
def clear_state():
    """Clear storage state before each test."""
    _gauntlet_runs.clear()
    set_gauntlet_broadcast_fn(None)
    yield
    _gauntlet_runs.clear()
    set_gauntlet_broadcast_fn(None)


class TestBroadcastFunction:
    """Tests for broadcast function management."""

    def test_set_and_get_broadcast_fn(self):
        """Test setting and getting broadcast function."""
        mock_fn = MagicMock()
        set_gauntlet_broadcast_fn(mock_fn)

        result = get_gauntlet_broadcast_fn()
        assert result is mock_fn

    def test_get_broadcast_fn_when_not_set(self):
        """Test getting broadcast function when not set."""
        result = get_gauntlet_broadcast_fn()
        assert result is None


class TestInMemoryStorage:
    """Tests for in-memory storage access."""

    def test_get_gauntlet_runs_empty(self):
        """Test getting empty runs storage."""
        runs = get_gauntlet_runs()
        assert len(runs) == 0

    def test_get_gauntlet_runs_with_data(self):
        """Test getting runs storage with data."""
        _gauntlet_runs["test-123"] = {"gauntlet_id": "test-123", "status": "running"}

        runs = get_gauntlet_runs()
        assert "test-123" in runs
        assert runs["test-123"]["status"] == "running"


class TestCleanup:
    """Tests for cleanup logic."""

    def test_cleanup_old_entries(self):
        """Test cleanup removes old entries."""
        # Add an entry older than MAX_AGE
        old_time = datetime.now() - timedelta(seconds=_GAUNTLET_MAX_AGE_SECONDS + 100)
        _gauntlet_runs["old-run"] = {
            "gauntlet_id": "old-run",
            "status": "running",
            "created_at": old_time.isoformat(),
        }

        # Add a recent entry
        _gauntlet_runs["new-run"] = {
            "gauntlet_id": "new-run",
            "status": "running",
            "created_at": datetime.now().isoformat(),
        }

        _cleanup_gauntlet_runs()

        assert "old-run" not in _gauntlet_runs
        assert "new-run" in _gauntlet_runs

    def test_cleanup_completed_entries(self):
        """Test cleanup removes completed entries past TTL."""
        # Add a completed entry older than TTL
        old_time = datetime.now() - timedelta(seconds=_GAUNTLET_COMPLETED_TTL + 100)
        _gauntlet_runs["completed-old"] = {
            "gauntlet_id": "completed-old",
            "status": "completed",
            "created_at": datetime.now().isoformat(),  # Recent creation
            "completed_at": old_time.isoformat(),  # But completed long ago
        }

        # Add a recently completed entry
        _gauntlet_runs["completed-new"] = {
            "gauntlet_id": "completed-new",
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
        }

        _cleanup_gauntlet_runs()

        assert "completed-old" not in _gauntlet_runs
        assert "completed-new" in _gauntlet_runs

    def test_cleanup_fifo_eviction(self):
        """Test FIFO eviction when over limit."""
        # Add more entries than the limit
        for i in range(MAX_GAUNTLET_RUNS_IN_MEMORY + 10):
            _gauntlet_runs[f"run-{i:04d}"] = {
                "gauntlet_id": f"run-{i:04d}",
                "status": "running",
                "created_at": datetime.now().isoformat(),
            }

        _cleanup_gauntlet_runs()

        assert len(_gauntlet_runs) <= MAX_GAUNTLET_RUNS_IN_MEMORY

    def test_cleanup_preserves_running_entries(self):
        """Test cleanup preserves recent running entries."""
        _gauntlet_runs["running-1"] = {
            "gauntlet_id": "running-1",
            "status": "running",
            "created_at": datetime.now().isoformat(),
        }

        _cleanup_gauntlet_runs()

        assert "running-1" in _gauntlet_runs


class TestQuotaLock:
    """Tests for quota lock access."""

    def test_get_quota_lock(self):
        """Test getting quota lock."""
        lock = get_quota_lock()
        assert lock is not None

        # Verify it's a proper lock
        assert hasattr(lock, "acquire")
        assert hasattr(lock, "release")

    def test_quota_lock_is_reusable(self):
        """Test quota lock is the same instance."""
        lock1 = get_quota_lock()
        lock2 = get_quota_lock()
        assert lock1 is lock2


class TestDurableQueue:
    """Tests for durable queue configuration."""

    def test_is_durable_queue_enabled_default(self):
        """Test durable queue is enabled by default.

        Note: Previously this test used importlib.reload() which replaced the
        module-level _gauntlet_runs OrderedDict, causing stale references in
        other modules (e.g., the __init__.py re-export) and downstream test
        failures.  Instead we read the source module-level constant directly
        using the default env (no ARAGORA_DURABLE_GAUNTLET set).
        """
        import os

        # When ARAGORA_DURABLE_GAUNTLET is absent or "1", durable queue is enabled
        saved = os.environ.pop("ARAGORA_DURABLE_GAUNTLET", None)
        try:
            # Re-evaluate the expression used in storage.py
            result = os.environ.get("ARAGORA_DURABLE_GAUNTLET", "1").lower() not in (
                "0", "false", "no",
            )
            assert result is True
        finally:
            if saved is not None:
                os.environ["ARAGORA_DURABLE_GAUNTLET"] = saved

        # Also verify the module's cached value (from when it was first imported)
        assert is_durable_queue_enabled()


class TestTrackedTask:
    """Tests for tracked task creation."""

    @pytest.mark.asyncio
    async def test_create_tracked_task(self):
        """Test creating a tracked async task."""
        result = []

        async def test_coro():
            result.append("done")

        task = create_tracked_task(test_coro(), "test-task")
        await task

        assert "done" in result
        assert task.done()

    @pytest.mark.asyncio
    async def test_tracked_task_logs_exception(self):
        """Test tracked task logs exceptions."""

        async def failing_coro():
            raise ValueError("Test error")

        with patch("aragora.server.handlers.gauntlet.storage.logger") as mock_logger:
            task = create_tracked_task(failing_coro(), "failing-task")

            # Wait for the task to complete (with exception)
            try:
                await task
            except ValueError:
                pass

            # Give the callback time to run
            await asyncio.sleep(0.1)

            # Check that error was logged
            mock_logger.error.assert_called()


class TestStorageConstants:
    """Tests for storage constants."""

    def test_max_runs_constant(self):
        """Test MAX_GAUNTLET_RUNS_IN_MEMORY is reasonable."""
        assert MAX_GAUNTLET_RUNS_IN_MEMORY > 0
        assert MAX_GAUNTLET_RUNS_IN_MEMORY <= 10000  # Reasonable upper bound

    def test_completed_ttl_constant(self):
        """Test COMPLETED_TTL is reasonable."""
        assert _GAUNTLET_COMPLETED_TTL > 0
        assert _GAUNTLET_COMPLETED_TTL >= 60  # At least 1 minute

    def test_max_age_constant(self):
        """Test MAX_AGE is reasonable."""
        assert _GAUNTLET_MAX_AGE_SECONDS > 0
        assert _GAUNTLET_MAX_AGE_SECONDS >= _GAUNTLET_COMPLETED_TTL  # Should be >= TTL
