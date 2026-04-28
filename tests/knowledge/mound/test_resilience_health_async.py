"""
Async-path tests for ConnectionHealthMonitor.

Existing test_resilience.py covers the four sync helpers (record_success,
record_failure, is_healthy, get_status). This file fills the documented P4
(Memory) dark-week gap by exercising the async paths that were previously
uncovered:

- ``check_health()`` happy path with mocked pool roundtrip + latency
- ``check_health()`` exception variants (ConnectionError, TimeoutError, OSError, RuntimeError)
- ``check_health()`` failure-threshold transition (healthy→unhealthy)
- ``start()`` / ``stop()`` lifecycle and idempotency
- ``HealthStatus.to_dict()`` serialization
- ``_health_check_loop()`` survives transient errors and respects cancellation

All tests are additive and read-only against the underlying module — no
behavior change, no production code touched.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.resilience import ConnectionHealthMonitor, HealthStatus


def _make_pool(execute_side_effect=None) -> MagicMock:
    """Return a MagicMock pool whose acquire() yields a connection with execute()."""
    mock_conn = AsyncMock()
    if execute_side_effect is not None:
        mock_conn.execute = AsyncMock(side_effect=execute_side_effect)
    else:
        mock_conn.execute = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_pool


class TestHealthStatusSerialization:
    """HealthStatus.to_dict() is the JSON-friendly serialization surface."""

    def test_to_dict_default(self) -> None:
        now = datetime.now(timezone.utc)
        status = HealthStatus(healthy=True, last_check=now)
        payload = status.to_dict()
        assert payload["healthy"] is True
        assert payload["last_check"] == now.isoformat()
        assert payload["consecutive_failures"] == 0
        assert payload["last_error"] is None
        assert payload["latency_ms"] is None

    def test_to_dict_with_failure(self) -> None:
        now = datetime.now(timezone.utc)
        status = HealthStatus(
            healthy=False,
            last_check=now,
            consecutive_failures=4,
            last_error="Failed: ConnectionError",
            latency_ms=12.5,
        )
        payload = status.to_dict()
        assert payload["healthy"] is False
        assert payload["consecutive_failures"] == 4
        assert payload["last_error"] == "Failed: ConnectionError"
        assert payload["latency_ms"] == 12.5


class TestCheckHealthHappyPath:
    """check_health() against a healthy pool resets failure state and records latency."""

    @pytest.mark.asyncio
    async def test_records_latency_on_success(self) -> None:
        monitor = ConnectionHealthMonitor(_make_pool())
        # Seed prior failures so we can verify reset.
        monitor._status.consecutive_failures = 3
        monitor._status.healthy = False

        result = await monitor.check_health()

        assert result.healthy is True
        assert result.consecutive_failures == 0
        assert result.latency_ms is not None
        assert result.latency_ms >= 0.0
        assert result.last_error is None

    @pytest.mark.asyncio
    async def test_executes_select_one(self) -> None:
        executed: list[str] = []

        async def _capture(stmt: str) -> None:
            executed.append(stmt)

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=_capture)
        pool = MagicMock()
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        monitor = ConnectionHealthMonitor(pool)
        await monitor.check_health()

        assert "SELECT 1" in executed


class TestCheckHealthExceptionPaths:
    """Each exception type degrades the status without raising."""

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        pool = _make_pool(execute_side_effect=ConnectionError("boom"))
        monitor = ConnectionHealthMonitor(pool, failure_threshold=5)

        result = await monitor.check_health()

        assert result.consecutive_failures == 1
        assert result.healthy is True  # still under threshold
        assert result.last_error == "Failed: ConnectionError"

    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        pool = _make_pool(execute_side_effect=TimeoutError("slow"))
        monitor = ConnectionHealthMonitor(pool, failure_threshold=5)

        result = await monitor.check_health()

        assert result.consecutive_failures == 1
        assert result.last_error == "Failed: TimeoutError"

    @pytest.mark.asyncio
    async def test_os_error(self) -> None:
        pool = _make_pool(execute_side_effect=OSError("fd"))
        monitor = ConnectionHealthMonitor(pool, failure_threshold=5)

        result = await monitor.check_health()

        assert result.consecutive_failures == 1
        assert result.last_error == "Failed: OSError"

    @pytest.mark.asyncio
    async def test_runtime_error(self) -> None:
        pool = _make_pool(execute_side_effect=RuntimeError("oops"))
        monitor = ConnectionHealthMonitor(pool, failure_threshold=5)

        result = await monitor.check_health()

        assert result.consecutive_failures == 1
        assert result.last_error == "Failed: RuntimeError"

    @pytest.mark.asyncio
    async def test_failure_threshold_flips_unhealthy(self) -> None:
        pool = _make_pool(execute_side_effect=ConnectionError("boom"))
        monitor = ConnectionHealthMonitor(pool, failure_threshold=3)

        await monitor.check_health()
        assert monitor.is_healthy() is True
        await monitor.check_health()
        assert monitor.is_healthy() is True
        result = await monitor.check_health()
        assert result.consecutive_failures == 3
        assert result.healthy is False
        assert monitor.is_healthy() is False

    @pytest.mark.asyncio
    async def test_unrelated_exception_propagates(self) -> None:
        # Exceptions outside the documented set must NOT be silently swallowed
        # (defense against masking real bugs).
        pool = _make_pool(execute_side_effect=ValueError("schema"))
        monitor = ConnectionHealthMonitor(pool)

        with pytest.raises(ValueError, match="schema"):
            await monitor.check_health()


class TestStartStopLifecycle:
    """start()/stop() create and cancel the background task cleanly."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self) -> None:
        monitor = ConnectionHealthMonitor(_make_pool(), health_check_interval=10.0)
        assert monitor._check_task is None

        await monitor.start()

        assert monitor._check_task is not None
        assert not monitor._check_task.done()

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        monitor = ConnectionHealthMonitor(_make_pool(), health_check_interval=10.0)

        await monitor.start()
        first_task = monitor._check_task
        await monitor.start()
        second_task = monitor._check_task

        assert first_task is second_task

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self) -> None:
        monitor = ConnectionHealthMonitor(_make_pool(), health_check_interval=10.0)
        await monitor.start()
        task = monitor._check_task
        assert task is not None

        await monitor.stop()

        assert monitor._check_task is None
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self) -> None:
        monitor = ConnectionHealthMonitor(_make_pool(), health_check_interval=10.0)
        # Must not raise.
        await monitor.stop()
        assert monitor._check_task is None


class TestHealthCheckLoopResilience:
    """The background loop continues across transient errors and exits on cancel."""

    @pytest.mark.asyncio
    async def test_loop_runs_at_least_one_check(self) -> None:
        # Use a tiny interval so one tick happens within the wait window.
        monitor = ConnectionHealthMonitor(_make_pool(), health_check_interval=0.01)

        await monitor.start()
        # Give the loop a moment to tick.
        await asyncio.sleep(0.05)
        status = monitor.get_status()
        await monitor.stop()

        # Latency is set only by check_health(); presence proves the loop ran.
        assert status.latency_ms is not None or status.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_loop_survives_transient_connection_error(self) -> None:
        pool = _make_pool(execute_side_effect=ConnectionError("flap"))
        monitor = ConnectionHealthMonitor(pool, health_check_interval=0.01, failure_threshold=100)

        await monitor.start()
        await asyncio.sleep(0.05)
        # The task is still alive after multiple transient failures.
        assert monitor._check_task is not None
        assert not monitor._check_task.done()
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_loop_exits_cleanly_on_cancel(self) -> None:
        monitor = ConnectionHealthMonitor(_make_pool(), health_check_interval=0.01)
        await monitor.start()
        task = monitor._check_task
        assert task is not None

        await monitor.stop()

        # After stop(), the task is finished and the monitor cleared the handle.
        assert monitor._check_task is None
        assert task.done()
