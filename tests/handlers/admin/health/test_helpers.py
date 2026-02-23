"""Comprehensive tests for health check helper functions.

Tests the four public functions in aragora/server/handlers/admin/health/helpers.py:

  TestSyncStatus                 - sync_status() Supabase sync service status
  TestSyncStatusImportError      - sync_status() when sync_service module unavailable
  TestSyncStatusExceptions       - sync_status() error handling branches
  TestSlowDebatesStatus          - slow_debates_status() slow-running debate detection
  TestSlowDebatesStateManager    - slow_debates_status() state manager branches
  TestSlowDebatesMonitor         - slow_debates_status() performance monitor branches
  TestSlowDebatesStatusLogic     - slow_debates_status() overall status determination
  TestSlowDebatesThreshold       - slow_debates_status() configurable threshold
  TestSlowDebatesEdgeCases       - slow_debates_status() edge cases and boundary conditions
  TestCircuitBreakersStatus      - circuit_breakers_status() circuit breaker metrics
  TestCircuitBreakersImportError - circuit_breakers_status() when resilience unavailable
  TestCircuitBreakersExceptions  - circuit_breakers_status() error handling branches
  TestComponentHealthStatus      - component_health_status() HealthRegistry report
  TestComponentHealthImportError - component_health_status() when unavailable
  TestComponentHealthExceptions  - component_health_status() error handling branches
  TestSecurity                   - security-related tests (injection, traversal)
  TestReturnTypes                - verify HandlerResult structure

100+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
import sys
import time
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health.helpers import (
    sync_status,
    slow_debates_status,
    circuit_breakers_status,
    component_health_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_mock_handler() -> MagicMock:
    """Create a mock handler object."""
    return MagicMock()


@contextmanager
def _block_module(*module_paths: str):
    """Temporarily block modules so that ``from X import Y`` raises ImportError."""
    saved = {}
    for mod in module_paths:
        saved[mod] = sys.modules.get(mod, _SENTINEL)
        sys.modules[mod] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        for mod in module_paths:
            if saved[mod] is _SENTINEL:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = saved[mod]


_SENTINEL = object()


@contextmanager
def _mock_module(module_path: str, **attrs):
    """Install a fake module with the given attributes, then restore."""
    saved = sys.modules.get(module_path, _SENTINEL)
    fake = types.ModuleType(module_path)
    for k, v in attrs.items():
        setattr(fake, k, v)
    sys.modules[module_path] = fake
    try:
        yield fake
    finally:
        if saved is _SENTINEL:
            sys.modules.pop(module_path, None)
        else:
            sys.modules[module_path] = saved


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_sync_status_obj(
    enabled: bool = True,
    running: bool = True,
    queue_size: int = 0,
    synced_count: int = 42,
    failed_count: int = 0,
    last_sync_at: datetime | None = None,
    last_error: str | None = None,
):
    """Create a mock sync status object."""
    status = MagicMock()
    status.enabled = enabled
    status.running = running
    status.queue_size = queue_size
    status.synced_count = synced_count
    status.failed_count = failed_count
    status.last_sync_at = last_sync_at
    status.last_error = last_error
    return status


def _make_health_report(
    overall_healthy: bool = True,
    summary: dict | None = None,
    components: dict | None = None,
    checked_at: datetime | None = None,
):
    """Create a mock health report."""
    report = MagicMock()
    report.overall_healthy = overall_healthy
    report.summary = summary or {"total": 2, "healthy": 2, "unhealthy": 0}
    report.components = components or {}
    report.checked_at = checked_at or datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
    return report


def _make_component_status(
    healthy: bool = True,
    consecutive_failures: int = 0,
    last_error: str | None = None,
    latency_ms: float = 5.0,
    last_check: datetime | None = None,
    metadata: dict | None = None,
):
    """Create a mock component status."""
    status = MagicMock()
    status.healthy = healthy
    status.consecutive_failures = consecutive_failures
    status.last_error = last_error
    status.latency_ms = latency_ms
    status.last_check = last_check or datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
    status.metadata = metadata or {}
    return status


def _make_sync_service(status_obj=None):
    """Create a mock sync service that returns the given status."""
    sync = MagicMock()
    if status_obj is None:
        status_obj = _make_sync_status_obj()
    sync.get_status.return_value = status_obj
    return sync


def _patch_sync_service(sync_service=None, status_obj=None):
    """Context manager to install a mock sync_service module."""
    if sync_service is None:
        sync_service = _make_sync_service(status_obj)
    return _mock_module(
        "aragora.persistence.sync_service",
        get_sync_service=MagicMock(return_value=sync_service),
    )


def _make_lock():
    """Create a mock context manager lock."""
    lock = MagicMock()
    lock.__enter__ = MagicMock(return_value=None)
    lock.__exit__ = MagicMock(return_value=False)
    return lock


def _patch_state_manager(debates=None, lock=None, error=None):
    """Context manager to install a mock state_manager module."""
    if error:
        # Install module but make get_active_debates raise
        get_debates = MagicMock(side_effect=error)
    else:
        get_debates = MagicMock(return_value=debates or {})
    if lock is None:
        lock = _make_lock()
    return _mock_module(
        "aragora.server.stream.state_manager",
        get_active_debates=get_debates,
        get_active_debates_lock=MagicMock(return_value=lock),
    )


def _patch_performance_monitor(
    slow_debates=None,
    current_slow_debates=None,
    error=None,
):
    """Context manager to install a mock performance_monitor module."""
    if error:
        get_monitor = MagicMock(side_effect=error)
    else:
        monitor = MagicMock()
        monitor.get_slow_debates.return_value = slow_debates or []
        monitor.get_current_slow_debates.return_value = current_slow_debates or []
        get_monitor = MagicMock(return_value=monitor)
    return _mock_module(
        "aragora.debate.performance_monitor",
        get_debate_monitor=get_monitor,
    )


def _patch_resilience(metrics=None, error=None):
    """Context manager to install a mock resilience module."""
    if error:
        get_metrics = MagicMock(side_effect=error)
    else:
        get_metrics = MagicMock(return_value=metrics or {})
    return _mock_module(
        "aragora.resilience",
        get_circuit_breaker_metrics=get_metrics,
    )


def _patch_health_registry(report=None, error=None):
    """Context manager to install a mock resilience.health module."""
    if error:
        get_registry = MagicMock(side_effect=error)
    else:
        registry = MagicMock()
        registry.get_report.return_value = report or _make_health_report()
        get_registry = MagicMock(return_value=registry)
    return _mock_module(
        "aragora.resilience.health",
        get_global_health_registry=get_registry,
    )


# ============================================================================
# TestSyncStatus - sync_status() function
# ============================================================================


class TestSyncStatus:
    """Tests for sync_status() - Supabase sync service status."""

    def test_healthy_sync_with_all_fields(self):
        """Returns full status when sync service is healthy."""
        sync_time = datetime(2026, 2, 23, 10, 30, 0, tzinfo=timezone.utc)
        status_obj = _make_sync_status_obj(
            enabled=True,
            running=True,
            queue_size=5,
            synced_count=100,
            failed_count=2,
            last_sync_at=sync_time,
            last_error=None,
        )
        with _patch_sync_service(status_obj=status_obj):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert _status(result) == 200
        assert body["enabled"] is True
        assert body["running"] is True
        assert body["queue_size"] == 5
        assert body["synced_count"] == 100
        assert body["failed_count"] == 2
        assert body["last_sync_at"] is not None
        assert "Z" in body["last_sync_at"]
        assert body["last_error"] is None
        assert "timestamp" in body

    def test_sync_disabled(self):
        """Returns disabled status when sync service not enabled."""
        status_obj = _make_sync_status_obj(
            enabled=False, running=False, queue_size=0, synced_count=0, failed_count=0
        )
        with _patch_sync_service(status_obj=status_obj):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["enabled"] is False
        assert body["running"] is False

    def test_sync_with_last_error(self):
        """Returns last_error when sync has encountered errors."""
        status_obj = _make_sync_status_obj(last_error="Connection timeout")
        with _patch_sync_service(status_obj=status_obj):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["last_error"] == "Connection timeout"

    def test_sync_with_no_last_sync_at(self):
        """Returns null last_sync_at when never synced."""
        status_obj = _make_sync_status_obj(last_sync_at=None)
        with _patch_sync_service(status_obj=status_obj):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["last_sync_at"] is None

    def test_sync_timestamp_present(self):
        """Response includes ISO timestamp."""
        with _patch_sync_service():
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")

    def test_sync_large_queue_size(self):
        """Handles large queue sizes."""
        status_obj = _make_sync_status_obj(queue_size=999999)
        with _patch_sync_service(status_obj=status_obj):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["queue_size"] == 999999

    def test_sync_zero_counts(self):
        """Handles zero synced/failed counts."""
        status_obj = _make_sync_status_obj(synced_count=0, failed_count=0)
        with _patch_sync_service(status_obj=status_obj):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["synced_count"] == 0
        assert body["failed_count"] == 0

    def test_sync_last_sync_at_formatted_correctly(self):
        """last_sync_at includes isoformat with Z suffix."""
        dt = datetime(2026, 1, 15, 8, 45, 30, tzinfo=timezone.utc)
        status_obj = _make_sync_status_obj(last_sync_at=dt)
        with _patch_sync_service(status_obj=status_obj):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["last_sync_at"] == "2026-01-15T08:45:30+00:00Z"


# ============================================================================
# TestSyncStatusImportError - import failure branch
# ============================================================================


class TestSyncStatusImportError:
    """Tests for sync_status() when sync_service module is not available."""

    def test_import_error_returns_disabled(self):
        """Returns enabled=False and error message on ImportError."""
        with _block_module("aragora.persistence.sync_service"):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["enabled"] is False
        assert "not available" in body["error"]
        assert "timestamp" in body

    def test_import_error_status_200(self):
        """Import error still returns 200 (not a server error)."""
        with _block_module("aragora.persistence.sync_service"):
            result = sync_status(_make_mock_handler())

        assert _status(result) == 200

    def test_import_error_has_timestamp(self):
        """Import error response has a valid timestamp."""
        with _block_module("aragora.persistence.sync_service"):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["timestamp"].endswith("Z")


# ============================================================================
# TestSyncStatusExceptions - exception handling branches
# ============================================================================


class TestSyncStatusExceptions:
    """Tests for sync_status() error handling."""

    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError],
    )
    def test_handled_exception_returns_disabled(self, exc_class):
        """Each handled exception type returns enabled=False."""
        mock_get = MagicMock(side_effect=exc_class("test error"))
        with _mock_module(
            "aragora.persistence.sync_service",
            get_sync_service=mock_get,
        ):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["enabled"] is False
        assert body["error"] == "Health check failed"
        assert "timestamp" in body

    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError],
    )
    def test_handled_exception_returns_200(self, exc_class):
        """Handled exceptions still return 200."""
        mock_get = MagicMock(side_effect=exc_class("test error"))
        with _mock_module(
            "aragora.persistence.sync_service",
            get_sync_service=mock_get,
        ):
            result = sync_status(_make_mock_handler())

        assert _status(result) == 200

    def test_get_status_raises_attribute_error(self):
        """AttributeError from get_status() is handled."""
        sync = MagicMock()
        sync.get_status.side_effect = AttributeError("no attribute")
        with _mock_module(
            "aragora.persistence.sync_service",
            get_sync_service=MagicMock(return_value=sync),
        ):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["enabled"] is False

    def test_get_status_raises_runtime_error(self):
        """RuntimeError from get_status() is handled."""
        sync = MagicMock()
        sync.get_status.side_effect = RuntimeError("backend down")
        with _mock_module(
            "aragora.persistence.sync_service",
            get_sync_service=MagicMock(return_value=sync),
        ):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert body["error"] == "Health check failed"


# ============================================================================
# TestSlowDebatesStatus - slow_debates_status() function
# ============================================================================


class TestSlowDebatesStatus:
    """Tests for slow_debates_status() - slow-running debate detection."""

    def test_no_active_no_historical_healthy(self):
        """Returns healthy when no slow debates found."""
        with _patch_state_manager(debates={}), \
             _patch_performance_monitor():
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"
        assert body["current_slow_count"] == 0
        assert body["recent_slow_count"] == 0

    def test_response_includes_threshold(self):
        """Response includes slow_threshold_seconds."""
        with patch.dict("os.environ", {"ARAGORA_SLOW_DEBATE_THRESHOLD": "45"}), \
             _block_module(
                 "aragora.server.stream.state_manager",
                 "aragora.debate.performance_monitor",
             ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["slow_threshold_seconds"] == 45.0

    def test_timestamp_present(self):
        """Response includes ISO timestamp."""
        with _block_module(
            "aragora.server.stream.state_manager",
            "aragora.debate.performance_monitor",
        ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")


# ============================================================================
# TestSlowDebatesStateManager - state manager branches
# ============================================================================


class TestSlowDebatesStateManager:
    """Tests for slow_debates_status() state manager integration."""

    def test_state_manager_import_error(self):
        """Handles ImportError from state_manager gracefully."""
        with _block_module(
            "aragora.server.stream.state_manager",
            "aragora.debate.performance_monitor",
        ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["errors"] is not None
        assert any("state_manager not available" in e for e in body["errors"])

    @pytest.mark.parametrize("exc_class", [OSError, RuntimeError, ValueError])
    def test_state_manager_exceptions(self, exc_class):
        """State manager exceptions are caught and logged."""
        with _patch_state_manager(error=exc_class("fail")), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["errors"] is not None
        assert any("state_manager error" in e for e in body["errors"])

    def test_active_debate_below_threshold_not_listed(self):
        """Debates below threshold are not included in current_slow."""
        now = time.time()
        debates = {
            "debate-1": {
                "start_time": now - 5,  # 5 seconds, below 30s default threshold
                "task": "Short debate",
                "agents": ["agent1"],
                "current_round": 1,
                "total_rounds": 3,
            }
        }
        with _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["current_slow_count"] == 0

    def test_active_debate_above_threshold_listed(self):
        """Debates above threshold are included in current_slow."""
        now = time.time()
        debates = {
            "debate-slow-1": {
                "start_time": now - 60,  # 60 seconds, above 30s default
                "task": "Slow running debate",
                "agents": ["agent1", "agent2"],
                "current_round": 2,
                "total_rounds": 3,
            }
        }
        with _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["current_slow_count"] == 1
        slow = body["current_slow"][0]
        assert slow["debate_id"] == "debate-slow-1"
        assert slow["duration_seconds"] >= 59
        assert slow["agents"] == ["agent1", "agent2"]
        assert slow["current_round"] == 2
        assert slow["total_rounds"] == 3
        assert "started_at" in slow

    def test_task_truncated_to_100_chars(self):
        """Debate task is truncated to 100 characters."""
        now = time.time()
        long_task = "A" * 200
        debates = {
            "debate-1": {
                "start_time": now - 60,
                "task": long_task,
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            }
        }
        with _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert len(body["current_slow"][0]["task"]) == 100

    def test_multiple_slow_debates_sorted_by_duration(self):
        """Multiple slow debates are sorted by duration descending."""
        now = time.time()
        debates = {
            "debate-short": {
                "start_time": now - 40,
                "task": "Short slow",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            },
            "debate-long": {
                "start_time": now - 120,
                "task": "Long slow",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            },
            "debate-medium": {
                "start_time": now - 60,
                "task": "Medium slow",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            },
        }
        with _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["current_slow_count"] == 3
        durations = [d["duration_seconds"] for d in body["current_slow"]]
        assert durations == sorted(durations, reverse=True)

    def test_debate_missing_optional_fields(self):
        """Handles debates missing optional fields gracefully."""
        now = time.time()
        debates = {
            "debate-1": {
                "start_time": now - 60,
                # Missing: task, agents, current_round, total_rounds
            }
        }
        with _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["current_slow_count"] == 1
        slow = body["current_slow"][0]
        assert slow["task"] == ""
        assert slow["agents"] == []
        assert slow["current_round"] == 0
        assert slow["total_rounds"] == 0

    def test_debate_missing_start_time_uses_now(self):
        """Debates missing start_time use now, resulting in 0 duration."""
        debates = {
            "debate-1": {
                # No start_time - defaults to now, so duration is ~0
                "task": "Test",
                "agents": [],
            }
        }
        with _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        # Duration is near 0, so below threshold; should not be listed
        assert body["current_slow_count"] == 0

    def test_started_at_formatted_with_z_suffix(self):
        """started_at field includes Z suffix."""
        now = time.time()
        debates = {
            "debate-1": {
                "start_time": now - 60,
                "task": "Test",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            }
        }
        with _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["current_slow"][0]["started_at"].endswith("Z")

    def test_duration_rounded_to_2_decimals(self):
        """Duration is rounded to 2 decimal places."""
        now = time.time()
        debates = {
            "debate-1": {
                "start_time": now - 60.123456,
                "task": "Test",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            }
        }
        with _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        dur = body["current_slow"][0]["duration_seconds"]
        # Should be rounded to 2 decimal places
        assert dur == round(dur, 2)


# ============================================================================
# TestSlowDebatesMonitor - performance monitor branches
# ============================================================================


class TestSlowDebatesMonitor:
    """Tests for slow_debates_status() performance monitor integration."""

    def test_monitor_import_error(self):
        """Handles ImportError from performance_monitor gracefully."""
        with _patch_state_manager(debates={}), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["errors"] is not None
        assert any("performance_monitor not available" in e for e in body["errors"])

    @pytest.mark.parametrize("exc_class", [OSError, RuntimeError, ValueError])
    def test_monitor_exceptions(self, exc_class):
        """Performance monitor exceptions are caught and logged."""
        with _patch_state_manager(debates={}), \
             _patch_performance_monitor(error=exc_class("fail")):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["errors"] is not None
        assert any("performance_monitor error" in e for e in body["errors"])

    def test_monitor_returns_historical_slow(self):
        """Performance monitor historical slow debates are included."""
        historical = [
            {"debate_id": "hist-1", "duration_seconds": 60},
            {"debate_id": "hist-2", "duration_seconds": 45},
        ]
        with _patch_state_manager(debates={}), \
             _patch_performance_monitor(slow_debates=historical):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["recent_slow_count"] == 2

    def test_monitor_deduplicates_current_slow(self):
        """Monitor current debates are deduplicated with state manager."""
        now = time.time()
        debates = {
            "debate-dup": {
                "start_time": now - 60,
                "task": "Duplicate",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            }
        }
        with _patch_state_manager(debates=debates), \
             _patch_performance_monitor(
                 current_slow_debates=[
                     {"debate_id": "debate-dup", "duration_seconds": 60},
                 ],
             ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        # debate-dup already in current_slow from state_manager, should not be duplicated
        assert body["current_slow_count"] == 1

    def test_monitor_adds_non_duplicate_current(self):
        """Monitor current debates not in state_manager are added."""
        with _patch_state_manager(debates={}), \
             _patch_performance_monitor(
                 current_slow_debates=[
                     {"debate_id": "monitor-only", "duration_seconds": 90},
                 ],
             ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["current_slow_count"] == 1
        assert body["current_slow"][0]["debate_id"] == "monitor-only"

    def test_monitor_limit_20(self):
        """get_slow_debates is called with limit=20."""
        monitor = MagicMock()
        monitor.get_slow_debates.return_value = []
        monitor.get_current_slow_debates.return_value = []
        get_monitor = MagicMock(return_value=monitor)

        with _patch_state_manager(debates={}), \
             _mock_module(
                 "aragora.debate.performance_monitor",
                 get_debate_monitor=get_monitor,
             ):
            slow_debates_status(_make_mock_handler())

        monitor.get_slow_debates.assert_called_once_with(limit=20)


# ============================================================================
# TestSlowDebatesStatusLogic - status determination
# ============================================================================


class TestSlowDebatesStatusLogic:
    """Tests for slow_debates_status() overall status determination."""

    def test_healthy_when_no_slow_and_no_errors(self):
        """Status is 'healthy' when no slow debates and no errors."""
        with _patch_state_manager(debates={}), \
             _patch_performance_monitor():
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "healthy"

    def test_degraded_when_slow_debates_exist(self):
        """Status is 'degraded' when slow debates are found."""
        now = time.time()
        debates = {
            "debate-slow": {
                "start_time": now - 60,
                "task": "Slow",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            }
        }
        with _patch_state_manager(debates=debates), \
             _patch_performance_monitor():
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "degraded"

    def test_partial_when_errors_but_no_slow(self):
        """Status is 'partial' when errors but no slow debates."""
        with _block_module(
            "aragora.server.stream.state_manager",
            "aragora.debate.performance_monitor",
        ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "partial"

    def test_degraded_when_historical_slow_exists(self):
        """Status is 'degraded' when recent historical slow debates exist."""
        with _patch_state_manager(debates={}), \
             _patch_performance_monitor(
                 slow_debates=[{"debate_id": "hist-1", "duration_seconds": 60}],
             ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "degraded"

    def test_errors_null_when_no_errors(self):
        """Errors field is null when no errors occurred."""
        with _patch_state_manager(debates={}), \
             _patch_performance_monitor():
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["errors"] is None

    def test_degraded_overrides_partial_when_slow_and_errors(self):
        """Status is 'degraded' (not 'partial') when slow debates exist with errors."""
        now = time.time()
        debates = {
            "debate-slow": {
                "start_time": now - 60,
                "task": "Slow",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            }
        }
        with _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "degraded"


# ============================================================================
# TestSlowDebatesThreshold - configurable threshold
# ============================================================================


class TestSlowDebatesThreshold:
    """Tests for slow_debates_status() threshold configuration."""

    def test_default_threshold_30(self):
        """Default threshold is 30 seconds."""
        with _block_module(
            "aragora.server.stream.state_manager",
            "aragora.debate.performance_monitor",
        ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["slow_threshold_seconds"] == 30.0

    def test_custom_threshold_from_env(self):
        """Custom threshold from ARAGORA_SLOW_DEBATE_THRESHOLD env var."""
        with patch.dict("os.environ", {"ARAGORA_SLOW_DEBATE_THRESHOLD": "10"}), \
             _block_module(
                 "aragora.server.stream.state_manager",
                 "aragora.debate.performance_monitor",
             ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["slow_threshold_seconds"] == 10.0

    def test_threshold_affects_classification(self):
        """Lower threshold correctly classifies more debates as slow."""
        now = time.time()
        debates = {
            "debate-1": {
                "start_time": now - 15,  # 15s - not slow at 30s, slow at 10s
                "task": "Test",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            }
        }
        with patch.dict("os.environ", {"ARAGORA_SLOW_DEBATE_THRESHOLD": "10"}), \
             _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["current_slow_count"] == 1

    def test_high_threshold_excludes_debates(self):
        """Higher threshold excludes debates that would otherwise be slow."""
        now = time.time()
        debates = {
            "debate-1": {
                "start_time": now - 60,  # 60s - slow at 30s, not slow at 120s
                "task": "Test",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            }
        }
        with patch.dict("os.environ", {"ARAGORA_SLOW_DEBATE_THRESHOLD": "120"}), \
             _patch_state_manager(debates=debates), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["current_slow_count"] == 0


# ============================================================================
# TestSlowDebatesEdgeCases
# ============================================================================


class TestSlowDebatesEdgeCases:
    """Edge cases and boundary conditions for slow_debates_status()."""

    def test_current_slow_capped_at_20(self):
        """current_slow response is capped at 20 entries."""
        now = time.time()
        debates = {}
        for i in range(25):
            debates[f"debate-{i}"] = {
                "start_time": now - (60 + i),
                "task": f"Debate {i}",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            }
        with _patch_state_manager(debates=debates), \
             _patch_performance_monitor():
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert len(body["current_slow"]) == 20
        # But total count reflects actual count
        assert body["current_slow_count"] == 25

    def test_recent_slow_capped_at_20(self):
        """recent_slow response is capped at 20 entries."""
        recent = [{"debate_id": f"hist-{i}"} for i in range(25)]
        with _patch_state_manager(debates={}), \
             _patch_performance_monitor(slow_debates=recent):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert len(body["recent_slow"]) == 20

    def test_empty_debates_dict(self):
        """Handles empty active debates dict."""
        with _patch_state_manager(debates={}), \
             _patch_performance_monitor():
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["current_slow_count"] == 0
        assert body["status"] == "healthy"

    def test_both_sources_contribute_slow(self):
        """Both state_manager and monitor contribute to current_slow."""
        now = time.time()
        debates = {
            "state-debate": {
                "start_time": now - 60,
                "task": "State",
                "agents": [],
                "current_round": 1,
                "total_rounds": 1,
            }
        }
        with _patch_state_manager(debates=debates), \
             _patch_performance_monitor(
                 current_slow_debates=[
                     {"debate_id": "monitor-debate", "duration_seconds": 45},
                 ],
             ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["current_slow_count"] == 2
        ids = [d["debate_id"] for d in body["current_slow"]]
        assert "state-debate" in ids
        assert "monitor-debate" in ids

    def test_both_modules_fail(self):
        """Both state_manager and monitor failing yields partial status."""
        with _block_module(
            "aragora.server.stream.state_manager",
            "aragora.debate.performance_monitor",
        ):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "partial"
        assert len(body["errors"]) == 2


# ============================================================================
# TestCircuitBreakersStatus - circuit_breakers_status() function
# ============================================================================


class TestCircuitBreakersStatus:
    """Tests for circuit_breakers_status() - circuit breaker metrics."""

    def test_healthy_metrics(self):
        """Returns full metrics when resilience module is available."""
        metrics = {
            "health": {"status": "healthy", "high_failure_circuits": []},
            "summary": {"open": 0, "closed": 5, "half_open": 0},
            "circuit_breakers": {
                "agent_api": {"state": "closed", "failures": 0},
                "database": {"state": "closed", "failures": 0},
            },
        }
        with _patch_resilience(metrics=metrics):
            result = circuit_breakers_status(_make_mock_handler())

        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"
        assert body["summary"]["closed"] == 5
        assert "agent_api" in body["circuit_breakers"]
        assert "timestamp" in body

    def test_degraded_metrics(self):
        """Returns degraded status when circuits are open."""
        metrics = {
            "health": {"status": "degraded", "high_failure_circuits": ["db"]},
            "summary": {"open": 1, "closed": 3, "half_open": 1},
            "circuit_breakers": {},
        }
        with _patch_resilience(metrics=metrics):
            result = circuit_breakers_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "degraded"

    def test_empty_metrics(self):
        """Handles empty metrics dict gracefully."""
        with _patch_resilience(metrics={}):
            result = circuit_breakers_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "unknown"
        assert body["summary"] == {}
        assert body["circuit_breakers"] == {}

    def test_health_status_from_nested_dict(self):
        """Status is extracted from health.status nested path."""
        metrics = {
            "health": {"status": "critical"},
            "summary": {},
            "circuit_breakers": {},
        }
        with _patch_resilience(metrics=metrics):
            result = circuit_breakers_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "critical"

    def test_missing_health_key_defaults_unknown(self):
        """Defaults to 'unknown' when health key missing."""
        metrics = {
            "summary": {"open": 0},
            "circuit_breakers": {},
        }
        with _patch_resilience(metrics=metrics):
            result = circuit_breakers_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "unknown"

    def test_health_dict_without_status_defaults_unknown(self):
        """Defaults to 'unknown' when health dict has no status key."""
        metrics = {
            "health": {"other_field": "value"},
            "summary": {},
            "circuit_breakers": {},
        }
        with _patch_resilience(metrics=metrics):
            result = circuit_breakers_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "unknown"

    def test_timestamp_present(self):
        """Response includes ISO timestamp."""
        with _patch_resilience(metrics={"health": {"status": "ok"}, "summary": {}, "circuit_breakers": {}}):
            result = circuit_breakers_status(_make_mock_handler())

        body = _body(result)
        assert body["timestamp"].endswith("Z")


# ============================================================================
# TestCircuitBreakersImportError - import failure branch
# ============================================================================


class TestCircuitBreakersImportError:
    """Tests for circuit_breakers_status() when resilience module unavailable."""

    def test_import_error_returns_unavailable(self):
        """Returns status=unavailable on ImportError."""
        with _block_module("aragora.resilience"):
            result = circuit_breakers_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "unavailable"
        assert "not available" in body["error"]
        assert "timestamp" in body

    def test_import_error_status_200(self):
        """Import error still returns 200."""
        with _block_module("aragora.resilience"):
            result = circuit_breakers_status(_make_mock_handler())

        assert _status(result) == 200


# ============================================================================
# TestCircuitBreakersExceptions - error handling branches
# ============================================================================


class TestCircuitBreakersExceptions:
    """Tests for circuit_breakers_status() error handling."""

    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError],
    )
    def test_handled_exception_returns_error(self, exc_class):
        """Each handled exception returns status=error."""
        with _patch_resilience(error=exc_class("test error")):
            result = circuit_breakers_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "error"
        assert body["error"] == "Health check failed"
        assert "timestamp" in body

    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError],
    )
    def test_handled_exception_returns_200(self, exc_class):
        """Handled exceptions still return 200."""
        with _patch_resilience(error=exc_class("test")):
            result = circuit_breakers_status(_make_mock_handler())

        assert _status(result) == 200


# ============================================================================
# TestComponentHealthStatus - component_health_status() function
# ============================================================================


class TestComponentHealthStatus:
    """Tests for component_health_status() - HealthRegistry report."""

    def test_healthy_report(self):
        """Returns healthy status with component details."""
        comp = _make_component_status(
            healthy=True,
            latency_ms=3.5,
            metadata={"version": "1.0"},
        )
        report = _make_health_report(
            overall_healthy=True,
            summary={"total": 1, "healthy": 1, "unhealthy": 0},
            components={"database": comp},
        )
        with _patch_health_registry(report=report):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"
        assert body["overall_healthy"] is True
        assert body["summary"]["total"] == 1
        assert "database" in body["components"]
        db = body["components"]["database"]
        assert db["healthy"] is True
        assert db["latency_ms"] == 3.5
        assert db["metadata"]["version"] == "1.0"
        assert "timestamp" in body
        assert "checked_at" in body

    def test_degraded_report(self):
        """Returns degraded when some components unhealthy."""
        healthy_comp = _make_component_status(healthy=True)
        unhealthy_comp = _make_component_status(
            healthy=False,
            consecutive_failures=3,
            last_error="Connection refused",
        )
        report = _make_health_report(
            overall_healthy=False,
            summary={"total": 2, "healthy": 1, "unhealthy": 1},
            components={"cache": healthy_comp, "database": unhealthy_comp},
        )
        with _patch_health_registry(report=report):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "degraded"
        assert body["overall_healthy"] is False
        assert body["components"]["database"]["healthy"] is False
        assert body["components"]["database"]["consecutive_failures"] == 3
        assert body["components"]["database"]["last_error"] == "Connection refused"

    def test_empty_components(self):
        """Handles report with no components."""
        report = _make_health_report(
            overall_healthy=True,
            summary={"total": 0, "healthy": 0, "unhealthy": 0},
            components={},
        )
        with _patch_health_registry(report=report):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "healthy"
        assert body["components"] == {}

    def test_component_last_check_format(self):
        """Component last_check is ISO formatted with Z suffix."""
        dt = datetime(2026, 2, 23, 15, 30, 0, tzinfo=timezone.utc)
        comp = _make_component_status(last_check=dt)
        report = _make_health_report(components={"redis": comp})
        with _patch_health_registry(report=report):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert body["components"]["redis"]["last_check"].endswith("Z")

    def test_checked_at_format(self):
        """Report checked_at is ISO formatted with Z suffix."""
        dt = datetime(2026, 2, 23, 15, 30, 0, tzinfo=timezone.utc)
        report = _make_health_report(checked_at=dt)
        with _patch_health_registry(report=report):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert body["checked_at"].endswith("Z")

    def test_multiple_components(self):
        """Handles multiple components in report."""
        comps = {
            "database": _make_component_status(healthy=True, latency_ms=2.0),
            "cache": _make_component_status(healthy=True, latency_ms=0.5),
            "queue": _make_component_status(healthy=False, consecutive_failures=5),
        }
        report = _make_health_report(overall_healthy=False, components=comps)
        with _patch_health_registry(report=report):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert len(body["components"]) == 3
        assert body["components"]["queue"]["healthy"] is False

    def test_component_with_null_last_error(self):
        """Component with no last_error shows None."""
        comp = _make_component_status(last_error=None)
        report = _make_health_report(components={"svc": comp})
        with _patch_health_registry(report=report):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert body["components"]["svc"]["last_error"] is None

    def test_component_with_zero_failures(self):
        """Component with zero consecutive failures."""
        comp = _make_component_status(consecutive_failures=0)
        report = _make_health_report(components={"svc": comp})
        with _patch_health_registry(report=report):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert body["components"]["svc"]["consecutive_failures"] == 0


# ============================================================================
# TestComponentHealthImportError - import failure branch
# ============================================================================


class TestComponentHealthImportError:
    """Tests for component_health_status() when resilience_patterns unavailable."""

    def test_import_error_returns_unavailable(self):
        """Returns status=unavailable on ImportError."""
        with _block_module("aragora.resilience.health"):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "unavailable"
        assert "not available" in body["error"]
        assert "timestamp" in body

    def test_import_error_status_200(self):
        """Import error still returns 200."""
        with _block_module("aragora.resilience.health"):
            result = component_health_status(_make_mock_handler())

        assert _status(result) == 200


# ============================================================================
# TestComponentHealthExceptions - error handling branches
# ============================================================================


class TestComponentHealthExceptions:
    """Tests for component_health_status() error handling."""

    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError],
    )
    def test_handled_exception_returns_error(self, exc_class):
        """Each handled exception returns status=error."""
        with _patch_health_registry(error=exc_class("test error")):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "error"
        assert body["error"] == "Health check failed"
        assert "timestamp" in body

    @pytest.mark.parametrize(
        "exc_class",
        [TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError],
    )
    def test_handled_exception_returns_200(self, exc_class):
        """Handled exceptions still return 200."""
        with _patch_health_registry(error=exc_class("test")):
            result = component_health_status(_make_mock_handler())

        assert _status(result) == 200

    def test_report_get_report_raises(self):
        """Exception from registry.get_report() is handled."""
        registry = MagicMock()
        registry.get_report.side_effect = RuntimeError("registry broken")
        get_registry = MagicMock(return_value=registry)
        with _mock_module(
            "aragora.resilience.health",
            get_global_health_registry=get_registry,
        ):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "error"


# ============================================================================
# TestSecurity - security-related tests
# ============================================================================


class TestSecurity:
    """Security-related tests for helper functions."""

    def test_sync_status_no_internal_leakage_on_error(self):
        """sync_status does not leak internal error messages."""
        mock_get = MagicMock(side_effect=RuntimeError("SECRET_INTERNAL_PATH/db.sqlite"))
        with _mock_module(
            "aragora.persistence.sync_service",
            get_sync_service=mock_get,
        ):
            result = sync_status(_make_mock_handler())

        body = _body(result)
        assert "SECRET_INTERNAL_PATH" not in json.dumps(body)
        assert body["error"] == "Health check failed"

    def test_circuit_breakers_no_internal_leakage_on_error(self):
        """circuit_breakers_status does not leak internal error messages."""
        with _patch_resilience(error=ValueError("internal db password: hunter2")):
            result = circuit_breakers_status(_make_mock_handler())

        body = _body(result)
        assert "hunter2" not in json.dumps(body)
        assert body["error"] == "Health check failed"

    def test_component_health_no_internal_leakage_on_error(self):
        """component_health_status does not leak internal error messages."""
        with _patch_health_registry(error=OSError("filesystem /secret/data not found")):
            result = component_health_status(_make_mock_handler())

        body = _body(result)
        assert "/secret/data" not in json.dumps(body)
        assert body["error"] == "Health check failed"

    def test_slow_debates_error_type_only(self):
        """slow_debates_status only exposes exception type, not message."""
        with _patch_state_manager(error=RuntimeError("secret data")), \
             _block_module("aragora.debate.performance_monitor"):
            result = slow_debates_status(_make_mock_handler())

        body = _body(result)
        errors_str = json.dumps(body.get("errors", []))
        assert "secret data" not in errors_str
        assert "RuntimeError" in errors_str

    def test_all_responses_have_timestamp(self):
        """All function responses include timestamp."""
        with _block_module(
            "aragora.persistence.sync_service",
            "aragora.resilience",
            "aragora.resilience.health",
            "aragora.server.stream.state_manager",
            "aragora.debate.performance_monitor",
        ):
            for func in [sync_status, circuit_breakers_status, component_health_status, slow_debates_status]:
                result = func(_make_mock_handler())
                body = _body(result)
                assert "timestamp" in body, f"{func.__name__} missing timestamp"

    def test_handler_argument_ignored(self):
        """All functions accept handler arg but don't depend on it."""
        with _block_module(
            "aragora.persistence.sync_service",
            "aragora.resilience",
            "aragora.resilience.health",
            "aragora.server.stream.state_manager",
            "aragora.debate.performance_monitor",
        ):
            for func in [sync_status, circuit_breakers_status, component_health_status, slow_debates_status]:
                result = func(None)
                assert _status(result) == 200


# ============================================================================
# TestReturnTypes - verify HandlerResult structure
# ============================================================================


class TestReturnTypes:
    """Verify all functions return proper HandlerResult objects."""

    def test_sync_status_returns_handler_result(self):
        """sync_status returns a HandlerResult."""
        from aragora.server.handlers.utils.responses import HandlerResult

        with _block_module("aragora.persistence.sync_service"):
            result = sync_status(_make_mock_handler())

        assert isinstance(result, HandlerResult)
        assert result.content_type == "application/json"

    def test_circuit_breakers_returns_handler_result(self):
        """circuit_breakers_status returns a HandlerResult."""
        from aragora.server.handlers.utils.responses import HandlerResult

        with _block_module("aragora.resilience"):
            result = circuit_breakers_status(_make_mock_handler())

        assert isinstance(result, HandlerResult)

    def test_component_health_returns_handler_result(self):
        """component_health_status returns a HandlerResult."""
        from aragora.server.handlers.utils.responses import HandlerResult

        with _block_module("aragora.resilience.health"):
            result = component_health_status(_make_mock_handler())

        assert isinstance(result, HandlerResult)

    def test_slow_debates_returns_handler_result(self):
        """slow_debates_status returns a HandlerResult."""
        from aragora.server.handlers.utils.responses import HandlerResult

        with _block_module(
            "aragora.server.stream.state_manager",
            "aragora.debate.performance_monitor",
        ):
            result = slow_debates_status(_make_mock_handler())

        assert isinstance(result, HandlerResult)

    def test_all_results_have_json_body(self):
        """All results have valid JSON bodies."""
        with _block_module(
            "aragora.persistence.sync_service",
            "aragora.resilience",
            "aragora.resilience.health",
            "aragora.server.stream.state_manager",
            "aragora.debate.performance_monitor",
        ):
            for func in [sync_status, circuit_breakers_status, component_health_status, slow_debates_status]:
                result = func(_make_mock_handler())
                body = json.loads(result.body)
                assert isinstance(body, dict)

    def test_all_results_status_200(self):
        """All error/fallback results return 200 (not 5xx)."""
        with _block_module(
            "aragora.persistence.sync_service",
            "aragora.resilience",
            "aragora.resilience.health",
            "aragora.server.stream.state_manager",
            "aragora.debate.performance_monitor",
        ):
            for func in [sync_status, circuit_breakers_status, component_health_status, slow_debates_status]:
                result = func(_make_mock_handler())
                assert _status(result) == 200, f"{func.__name__} returned non-200"
