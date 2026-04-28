"""Tests for ``aragora.knowledge.mound.metrics_health_bridge``.

The bridge wires ``ConnectionHealthMonitor`` snapshots into the
``KMMetrics.record(...)`` surface. These tests exercise the
public lifecycle (``start``/``stop``/``tick_once``) and the
translation from ``HealthStatus`` to operation samples without
touching any database, network, or file system.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aragora.knowledge.mound.metrics import KMMetrics, OperationType
from aragora.knowledge.mound.metrics_health_bridge import (
    KMMetricsHealthBridge,
    build_bridge,
)
from aragora.knowledge.mound.resilience.health import (
    ConnectionHealthMonitor,
    HealthStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingSink:
    """Minimal sink capturing every ``record(...)`` call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(
        self,
        operation: OperationType,
        latency_ms: float,
        success: bool = True,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.calls.append(
            {
                "operation": operation,
                "latency_ms": latency_ms,
                "success": success,
                "error": error,
                "metadata": metadata,
            }
        )


def _make_status(
    *,
    healthy: bool = True,
    latency_ms: float | None = 12.5,
    consecutive_failures: int = 0,
    last_error: str | None = None,
) -> HealthStatus:
    return HealthStatus(
        healthy=healthy,
        last_check=datetime.now(UTC),
        latency_ms=latency_ms,
        consecutive_failures=consecutive_failures,
        last_error=last_error,
    )


def _make_monitor_with_status(status: HealthStatus) -> ConnectionHealthMonitor:
    """Build a ConnectionHealthMonitor whose ``check_health`` returns
    ``status`` without actually touching the pool.
    """
    monitor = ConnectionHealthMonitor.__new__(ConnectionHealthMonitor)
    monitor._pool = None  # noqa: SLF001
    monitor._failure_threshold = 5  # noqa: SLF001
    monitor._recovery_timeout = 30.0  # noqa: SLF001
    monitor._health_check_interval = 10.0  # noqa: SLF001
    monitor._status = status  # noqa: SLF001
    monitor._check_task = None  # noqa: SLF001
    monitor.check_health = AsyncMock(return_value=status)  # type: ignore[method-assign]
    return monitor


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_non_positive_interval(self) -> None:
        monitor = _make_monitor_with_status(_make_status())
        sink = _RecordingSink()
        with pytest.raises(ValueError, match="interval_s"):
            KMMetricsHealthBridge(monitor, sink, interval_s=0)
        with pytest.raises(ValueError, match="interval_s"):
            KMMetricsHealthBridge(monitor, sink, interval_s=-1.0)

    def test_initial_state(self) -> None:
        monitor = _make_monitor_with_status(_make_status())
        sink = _RecordingSink()
        bridge = KMMetricsHealthBridge(monitor, sink, interval_s=1.0)
        assert bridge.is_running is False
        assert bridge.tick_count == 0


# ---------------------------------------------------------------------------
# tick_once: the deterministic surface
# ---------------------------------------------------------------------------


class TestTickOnce:
    @pytest.mark.asyncio
    async def test_records_healthy_snapshot(self) -> None:
        status = _make_status(healthy=True, latency_ms=42.0)
        monitor = _make_monitor_with_status(status)
        sink = _RecordingSink()
        bridge = KMMetricsHealthBridge(monitor, sink, interval_s=1.0)

        result = await bridge.tick_once()

        assert result is status
        assert bridge.tick_count == 1
        assert len(sink.calls) == 1
        call = sink.calls[0]
        assert call["operation"] == OperationType.QUERY
        assert call["latency_ms"] == 42.0
        assert call["success"] is True
        assert call["error"] is None
        assert call["metadata"]["source"] == "connection_health_monitor"
        assert call["metadata"]["consecutive_failures"] == 0

    @pytest.mark.asyncio
    async def test_records_unhealthy_snapshot(self) -> None:
        status = _make_status(
            healthy=False,
            latency_ms=None,
            consecutive_failures=7,
            last_error="ConnectionRefusedError",
        )
        monitor = _make_monitor_with_status(status)
        sink = _RecordingSink()
        bridge = KMMetricsHealthBridge(monitor, sink, interval_s=1.0)

        await bridge.tick_once()

        call = sink.calls[0]
        assert call["success"] is False
        assert call["latency_ms"] == 0.0  # None coerced to 0.0
        assert call["error"] == "ConnectionRefusedError"
        assert call["metadata"]["consecutive_failures"] == 7

    @pytest.mark.asyncio
    async def test_custom_operation_type(self) -> None:
        status = _make_status()
        monitor = _make_monitor_with_status(status)
        sink = _RecordingSink()
        bridge = KMMetricsHealthBridge(
            monitor, sink, interval_s=1.0, operation_type=OperationType.SYNC
        )

        await bridge.tick_once()

        assert sink.calls[0]["operation"] == OperationType.SYNC

    @pytest.mark.asyncio
    async def test_sink_failure_does_not_propagate(self) -> None:
        class _BrokenSink:
            def record(
                self,
                operation: OperationType,
                latency_ms: float,
                success: bool = True,
                error: str | None = None,
                metadata: dict[str, Any] | None = None,
            ) -> None:
                raise RuntimeError("sink boom")

        status = _make_status()
        monitor = _make_monitor_with_status(status)
        bridge = KMMetricsHealthBridge(monitor, _BrokenSink(), interval_s=1.0)

        # Must not raise — bridge swallows sink failures.
        result = await bridge.tick_once()
        assert result is status
        assert bridge.tick_count == 1


# ---------------------------------------------------------------------------
# Lifecycle: start / stop
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_then_stop_is_clean(self) -> None:
        monitor = _make_monitor_with_status(_make_status())
        sink = _RecordingSink()
        bridge = KMMetricsHealthBridge(monitor, sink, interval_s=10.0)

        await bridge.start()
        assert bridge.is_running is True
        await bridge.stop()
        assert bridge.is_running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        monitor = _make_monitor_with_status(_make_status())
        bridge = KMMetricsHealthBridge(monitor, _RecordingSink(), interval_s=10.0)
        await bridge.start()
        first_task = bridge._task  # noqa: SLF001
        await bridge.start()  # second call should be no-op
        assert bridge._task is first_task  # noqa: SLF001
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_clean(self) -> None:
        monitor = _make_monitor_with_status(_make_status())
        bridge = KMMetricsHealthBridge(monitor, _RecordingSink(), interval_s=10.0)
        # Must not raise.
        await bridge.stop()
        assert bridge.is_running is False

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        monitor = _make_monitor_with_status(_make_status())
        bridge = KMMetricsHealthBridge(monitor, _RecordingSink(), interval_s=10.0)
        await bridge.start()
        await bridge.stop()
        # Second stop must not raise.
        await bridge.stop()
        assert bridge.is_running is False

    @pytest.mark.asyncio
    async def test_poll_loop_records_at_least_once(self) -> None:
        monitor = _make_monitor_with_status(_make_status(latency_ms=3.0))
        sink = _RecordingSink()
        bridge = KMMetricsHealthBridge(monitor, sink, interval_s=0.01)

        await bridge.start()
        # Give the loop two intervals to complete.
        await asyncio.sleep(0.05)
        await bridge.stop()

        assert bridge.tick_count >= 1
        assert len(sink.calls) >= 1
        assert all(call["operation"] == OperationType.QUERY for call in sink.calls)


# ---------------------------------------------------------------------------
# Integration with real KMMetrics
# ---------------------------------------------------------------------------


class TestKMMetricsIntegration:
    @pytest.mark.asyncio
    async def test_build_bridge_returns_unstarted_bridge(self) -> None:
        monitor = _make_monitor_with_status(_make_status())
        metrics = KMMetrics()
        bridge = build_bridge(monitor, metrics, interval_s=5.0)
        assert isinstance(bridge, KMMetricsHealthBridge)
        assert bridge.is_running is False

    @pytest.mark.asyncio
    async def test_kmmetrics_records_bridge_sample(self) -> None:
        status = _make_status(healthy=True, latency_ms=8.5)
        monitor = _make_monitor_with_status(status)
        metrics = KMMetrics()
        bridge = build_bridge(monitor, metrics, interval_s=10.0)

        await bridge.tick_once()

        # KMMetrics get_stats() returns a dict keyed by op-value when no op
        # is specified, only including operations with count > 0.
        stats = metrics.get_stats()
        assert "query" in stats
        assert stats["query"]["count"] >= 1
        assert stats["query"]["success_count"] >= 1

    @pytest.mark.asyncio
    async def test_unhealthy_bridge_sample_marks_query_error(self) -> None:
        status = _make_status(
            healthy=False, latency_ms=None, last_error="OSError", consecutive_failures=3
        )
        monitor = _make_monitor_with_status(status)
        metrics = KMMetrics()
        bridge = build_bridge(monitor, metrics, interval_s=10.0)

        await bridge.tick_once()

        stats = metrics.get_stats()
        assert "query" in stats
        assert stats["query"]["error_count"] >= 1
