"""Comprehensive tests for background worker and job queue health handlers.

Tests the three public functions in aragora/server/handlers/admin/health/workers.py:

  TestWorkerHealthStatus             - worker_health_status() top-level orchestrator
  TestWorkerGauntletBranch           - gauntlet worker check branches
  TestWorkerNotificationBranch       - notification worker check branches
  TestWorkerConsensusHealingBranch   - consensus healing worker check branches
  TestWorkerOverallStatus            - overall status determination logic
  TestJobQueueHealthStatus           - job_queue_health_status() queue check
  TestJobQueueConnectivity           - queue connectivity error branches
  TestJobQueueThresholds             - configurable threshold and warning logic
  TestJobQueueBackendDetection       - backend type detection (sqlite/postgres/redis)
  TestJobQueueEnvironmentConfig      - environment variable threshold configuration
  TestCombinedWorkerQueueHealth      - combined_worker_queue_health() aggregation
  TestCombinedStatusPriority         - worst-status aggregation priority logic
  TestDataclasses                    - WorkerStatus and QueueStatus dataclasses
  TestEdgeCases                      - edge cases and boundary conditions
  TestSecurity                       - security-related tests (injection, traversal)

100+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health.workers import (
    WorkerStatus,
    QueueStatus,
    worker_health_status,
    job_queue_health_status,
    combined_worker_queue_health,
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


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_gauntlet_worker(
    running: bool = True,
    active_jobs: list | None = None,
    max_concurrent: int = 5,
    worker_id: str = "gauntlet-001",
):
    """Create a mock gauntlet worker."""
    worker = MagicMock()
    worker._running = running
    worker._active_jobs = active_jobs if active_jobs is not None else []
    worker.max_concurrent = max_concurrent
    worker.worker_id = worker_id
    return worker


def _make_notification_dispatcher(
    has_worker_task: bool = True,
    worker_done: bool = False,
    queue_enabled: bool = True,
    max_concurrent_deliveries: int = 10,
    has_config: bool = True,
):
    """Create a mock notification dispatcher."""
    dispatcher = MagicMock()

    # Set up _worker_task
    if has_worker_task:
        task = MagicMock()
        task.done.return_value = worker_done
        dispatcher._worker_task = task
    else:
        dispatcher._worker_task = None

    # Set up _config
    if has_config:
        config = MagicMock()
        config.queue_enabled = queue_enabled
        config.max_concurrent_deliveries = max_concurrent_deliveries
        dispatcher._config = config
    else:
        dispatcher._config = None

    return dispatcher


def _make_healing_worker(
    running: bool = True,
    candidates_processed: int = 42,
    healed_count: int = 7,
):
    """Create a mock consensus healing worker."""
    worker = MagicMock()
    worker._running = running
    worker._candidates_processed = candidates_processed
    worker._healed_count = healed_count
    return worker


_DEFAULT_STATS = {
    "pending": 5,
    "processing": 2,
    "completed": 100,
    "failed": 1,
    "cancelled": 3,
    "total": 111,
}


def _make_job_store(
    class_name: str = "SQLiteJobStore",
    stats: dict | None = None,
):
    """Create a mock job store."""
    store = MagicMock()
    type(store).__name__ = class_name

    result_stats = stats if stats is not None else dict(_DEFAULT_STATS)

    async def mock_get_stats():
        return result_stats

    store.get_stats = mock_get_stats
    return store


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_P_GAUNTLET = "aragora.server.handlers.admin.health.workers.get_gauntlet_worker"
_P_NOTIFICATION = "aragora.server.handlers.admin.health.workers.get_default_notification_dispatcher"
_P_HEALING = "aragora.server.handlers.admin.health.workers.get_consensus_healing_worker"
_P_JOB_STORE = "aragora.server.handlers.admin.health.workers.get_job_store"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_handler():
    """Default mock handler."""
    return _make_mock_handler()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove queue threshold env vars to ensure defaults."""
    monkeypatch.delenv("ARAGORA_QUEUE_PENDING_WARNING", raising=False)
    monkeypatch.delenv("ARAGORA_QUEUE_PENDING_CRITICAL", raising=False)
    monkeypatch.delenv("ARAGORA_QUEUE_PROCESSING_WARNING", raising=False)


# ============================================================================
# TestDataclasses - WorkerStatus and QueueStatus
# ============================================================================


class TestDataclasses:
    """Tests for the WorkerStatus and QueueStatus dataclasses."""

    def test_worker_status_defaults(self):
        """WorkerStatus has sensible defaults for optional fields."""
        ws = WorkerStatus(
            name="test",
            running=True,
            active_jobs=3,
            jobs_processed=100,
            jobs_failed=2,
            uptime_seconds=3600.0,
        )
        assert ws.name == "test"
        assert ws.running is True
        assert ws.active_jobs == 3
        assert ws.jobs_processed == 100
        assert ws.jobs_failed == 2
        assert ws.uptime_seconds == 3600.0
        assert ws.last_heartbeat is None
        assert ws.error is None

    def test_worker_status_with_all_fields(self):
        """WorkerStatus can set all fields."""
        now = datetime.now(timezone.utc)
        ws = WorkerStatus(
            name="gauntlet",
            running=False,
            active_jobs=0,
            jobs_processed=50,
            jobs_failed=5,
            uptime_seconds=1800.0,
            last_heartbeat=now,
            error="Connection lost",
        )
        assert ws.last_heartbeat == now
        assert ws.error == "Connection lost"

    def test_queue_status_defaults(self):
        """QueueStatus has sensible defaults."""
        qs = QueueStatus(
            name="main",
            connected=True,
            pending=10,
            processing=3,
            completed=200,
            failed=5,
            total=218,
        )
        assert qs.name == "main"
        assert qs.connected is True
        assert qs.error is None

    def test_queue_status_with_error(self):
        """QueueStatus can have an error."""
        qs = QueueStatus(
            name="main",
            connected=False,
            pending=0,
            processing=0,
            completed=0,
            failed=0,
            total=0,
            error="Redis connection refused",
        )
        assert qs.connected is False
        assert qs.error == "Redis connection refused"


# ============================================================================
# TestWorkerHealthStatus - worker_health_status() top-level
# ============================================================================


class TestWorkerHealthStatus:
    """Tests for worker_health_status() orchestrator."""

    def _patch_all_imports(self, gauntlet=None, notification=None, healing=None):
        """Return a context manager patching all three worker imports."""
        import contextlib

        @contextlib.contextmanager
        def _cm():
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.startup.workers": MagicMock(
                        get_gauntlet_worker=MagicMock(return_value=gauntlet)
                    ),
                    "aragora.control_plane.notifications": MagicMock(
                        get_default_notification_dispatcher=MagicMock(return_value=notification)
                    ),
                    "aragora.queue.workers": MagicMock(
                        get_consensus_healing_worker=MagicMock(return_value=healing)
                    ),
                },
            ):
                yield

        return _cm()

    def test_returns_200(self, mock_handler):
        """worker_health_status always returns HTTP 200."""
        with self._patch_all_imports():
            result = worker_health_status(mock_handler)
        assert _status(result) == 200

    def test_response_has_required_fields(self, mock_handler):
        """Response contains status, summary, workers, errors, timestamp."""
        with self._patch_all_imports():
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert "status" in body
        assert "summary" in body
        assert "workers" in body
        assert "timestamp" in body

    def test_timestamp_is_iso_format(self, mock_handler):
        """Timestamp is in ISO 8601 format ending with Z."""
        with self._patch_all_imports():
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["timestamp"].endswith("Z")

    def test_summary_counts(self, mock_handler):
        """Summary includes total, running, and stopped counts."""
        with self._patch_all_imports():
            result = worker_health_status(mock_handler)
        body = _body(result)
        summary = body["summary"]
        assert "total_workers" in summary
        assert "running_workers" in summary
        assert "stopped_workers" in summary
        assert summary["stopped_workers"] == summary["total_workers"] - summary["running_workers"]

    def test_errors_null_when_no_errors(self, mock_handler):
        """Errors field is null when there are no errors."""
        with self._patch_all_imports():
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["errors"] is None


# ============================================================================
# TestWorkerGauntletBranch - gauntlet worker check
# ============================================================================


class TestWorkerGauntletBranch:
    """Tests for the gauntlet worker check branch."""

    def test_gauntlet_running(self, mock_handler):
        """Running gauntlet worker reported correctly."""
        gauntlet = _make_gauntlet_worker(running=True, active_jobs=["j1", "j2"])
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup.workers": MagicMock(
                    get_gauntlet_worker=MagicMock(return_value=gauntlet)
                ),
                "aragora.control_plane.notifications": MagicMock(
                    get_default_notification_dispatcher=MagicMock(return_value=None)
                ),
                "aragora.queue.workers": MagicMock(
                    get_consensus_healing_worker=MagicMock(return_value=None)
                ),
            },
        ):
            result = worker_health_status(mock_handler)
        body = _body(result)
        gauntlet_info = next(w for w in body["workers"] if w["name"] == "gauntlet")
        assert gauntlet_info["running"] is True
        assert gauntlet_info["active_jobs"] == 2
        assert gauntlet_info["max_concurrent"] == 5
        assert gauntlet_info["worker_id"] == "gauntlet-001"

    def test_gauntlet_not_initialized(self, mock_handler):
        """None gauntlet worker returns not-initialized note."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup.workers": MagicMock(
                    get_gauntlet_worker=MagicMock(return_value=None)
                ),
                "aragora.control_plane.notifications": MagicMock(
                    get_default_notification_dispatcher=MagicMock(return_value=None)
                ),
                "aragora.queue.workers": MagicMock(
                    get_consensus_healing_worker=MagicMock(return_value=None)
                ),
            },
        ):
            result = worker_health_status(mock_handler)
        body = _body(result)
        gauntlet_info = next(w for w in body["workers"] if w["name"] == "gauntlet")
        assert gauntlet_info["running"] is False
        assert gauntlet_info["active_jobs"] == 0
        assert "not initialized" in gauntlet_info["note"].lower()

    def test_gauntlet_import_error(self, mock_handler):
        """ImportError on gauntlet adds to errors list."""
        import sys as _sys

        # Remove the module so the import inside the function fails
        saved = {}
        for k in list(_sys.modules.keys()):
            if k.startswith("aragora.server.startup"):
                saved[k] = _sys.modules.pop(k)
        try:
            import builtins

            original_import = builtins.__import__

            def fail_gauntlet(name, *args, **kwargs):
                if name == "aragora.server.startup.workers":
                    raise ImportError("No module named 'aragora.server.startup.workers'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fail_gauntlet):
                # Also ensure the other modules work
                with patch.dict(
                    "sys.modules",
                    {
                        "aragora.control_plane.notifications": MagicMock(
                            get_default_notification_dispatcher=MagicMock(return_value=None)
                        ),
                        "aragora.queue.workers": MagicMock(
                            get_consensus_healing_worker=MagicMock(return_value=None)
                        ),
                    },
                ):
                    result = worker_health_status(mock_handler)
        finally:
            _sys.modules.update(saved)

        body = _body(result)
        assert body["errors"] is not None
        assert any("gauntlet_worker import failed" in e for e in body["errors"])

    def test_gauntlet_attribute_error(self, mock_handler):
        """AttributeError on gauntlet worker access adds to errors."""

        class BadWorker:
            """Worker that raises AttributeError on _running access."""

            @property
            def _running(self):
                raise AttributeError("no _running")

        bad_worker = BadWorker()
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup.workers": MagicMock(
                    get_gauntlet_worker=MagicMock(return_value=bad_worker)
                ),
                "aragora.control_plane.notifications": MagicMock(
                    get_default_notification_dispatcher=MagicMock(return_value=None)
                ),
                "aragora.queue.workers": MagicMock(
                    get_consensus_healing_worker=MagicMock(return_value=None)
                ),
            },
        ):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["errors"] is not None
        assert any("gauntlet_worker access error" in e for e in body["errors"])

    def test_gauntlet_runtime_error(self, mock_handler):
        """RuntimeError on gauntlet worker access adds to errors."""

        def raise_runtime():
            raise RuntimeError("worker crashed")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup.workers": MagicMock(
                    get_gauntlet_worker=MagicMock(side_effect=RuntimeError("crashed"))
                ),
                "aragora.control_plane.notifications": MagicMock(
                    get_default_notification_dispatcher=MagicMock(return_value=None)
                ),
                "aragora.queue.workers": MagicMock(
                    get_consensus_healing_worker=MagicMock(return_value=None)
                ),
            },
        ):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["errors"] is not None
        assert any("gauntlet_worker access error" in e for e in body["errors"])

    def test_gauntlet_stopped(self, mock_handler):
        """Stopped gauntlet worker shows running=False."""
        gauntlet = _make_gauntlet_worker(running=False)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup.workers": MagicMock(
                    get_gauntlet_worker=MagicMock(return_value=gauntlet)
                ),
                "aragora.control_plane.notifications": MagicMock(
                    get_default_notification_dispatcher=MagicMock(return_value=None)
                ),
                "aragora.queue.workers": MagicMock(
                    get_consensus_healing_worker=MagicMock(return_value=None)
                ),
            },
        ):
            result = worker_health_status(mock_handler)
        body = _body(result)
        gauntlet_info = next(w for w in body["workers"] if w["name"] == "gauntlet")
        assert gauntlet_info["running"] is False


# ============================================================================
# TestWorkerNotificationBranch - notification worker check
# ============================================================================


class TestWorkerNotificationBranch:
    """Tests for the notification worker check branch."""

    def _patch_others(self):
        """Patch gauntlet and healing to return None."""
        return {
            "aragora.server.startup.workers": MagicMock(
                get_gauntlet_worker=MagicMock(return_value=None)
            ),
            "aragora.queue.workers": MagicMock(
                get_consensus_healing_worker=MagicMock(return_value=None)
            ),
        }

    def test_notification_running(self, mock_handler):
        """Running notification worker reported correctly."""
        dispatcher = _make_notification_dispatcher(
            has_worker_task=True, worker_done=False, queue_enabled=True
        )
        modules = self._patch_others()
        modules["aragora.control_plane.notifications"] = MagicMock(
            get_default_notification_dispatcher=MagicMock(return_value=dispatcher)
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        notif = next(w for w in body["workers"] if w["name"] == "notification")
        assert notif["running"] is True
        assert notif["queue_enabled"] is True
        assert notif["max_concurrent"] == 10

    def test_notification_worker_task_done(self, mock_handler):
        """Worker task .done() == True means not running."""
        dispatcher = _make_notification_dispatcher(has_worker_task=True, worker_done=True)
        modules = self._patch_others()
        modules["aragora.control_plane.notifications"] = MagicMock(
            get_default_notification_dispatcher=MagicMock(return_value=dispatcher)
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        notif = next(w for w in body["workers"] if w["name"] == "notification")
        assert notif["running"] is False

    def test_notification_no_worker_task(self, mock_handler):
        """No worker task means dispatcher not initialized."""
        dispatcher = _make_notification_dispatcher(has_worker_task=False)
        # Remove _worker_task attribute to trigger the hasattr check
        del dispatcher._worker_task
        modules = self._patch_others()
        modules["aragora.control_plane.notifications"] = MagicMock(
            get_default_notification_dispatcher=MagicMock(return_value=dispatcher)
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        notif = next(w for w in body["workers"] if w["name"] == "notification")
        assert notif["running"] is False
        assert (
            "not initialized" in notif.get("note", "").lower()
            or "no worker task" in notif.get("note", "").lower()
        )

    def test_notification_dispatcher_none(self, mock_handler):
        """None dispatcher gives not-initialized note."""
        modules = self._patch_others()
        modules["aragora.control_plane.notifications"] = MagicMock(
            get_default_notification_dispatcher=MagicMock(return_value=None)
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        notif = next(w for w in body["workers"] if w["name"] == "notification")
        assert notif["running"] is False

    def test_notification_import_error(self, mock_handler):
        """ImportError on notifications module gives module-not-available note."""
        import sys as _sys

        saved = {}
        for k in list(_sys.modules.keys()):
            if k.startswith("aragora.control_plane.notifications"):
                saved[k] = _sys.modules.pop(k)
        try:
            import builtins

            original_import = builtins.__import__

            def fail_notification(name, *args, **kwargs):
                if name == "aragora.control_plane.notifications":
                    raise ImportError("no module")
                return original_import(name, *args, **kwargs)

            others = self._patch_others()
            with patch("builtins.__import__", side_effect=fail_notification):
                with patch.dict("sys.modules", others):
                    result = worker_health_status(mock_handler)
        finally:
            _sys.modules.update(saved)

        body = _body(result)
        notif = next(w for w in body["workers"] if w["name"] == "notification")
        assert notif["running"] is False
        assert "not available" in notif.get("note", "").lower()

    def test_notification_attribute_error(self, mock_handler):
        """AttributeError on dispatcher access adds to errors."""
        modules = self._patch_others()
        modules["aragora.control_plane.notifications"] = MagicMock(
            get_default_notification_dispatcher=MagicMock(side_effect=AttributeError("bad attr"))
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["errors"] is not None
        assert any("notification_worker access error" in e for e in body["errors"])

    def test_notification_runtime_error(self, mock_handler):
        """RuntimeError on dispatcher access adds to errors."""
        modules = self._patch_others()
        modules["aragora.control_plane.notifications"] = MagicMock(
            get_default_notification_dispatcher=MagicMock(side_effect=RuntimeError("rt error"))
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["errors"] is not None
        assert any("notification_worker access error" in e for e in body["errors"])

    def test_notification_no_config(self, mock_handler):
        """Dispatcher with no config returns queue_enabled=False, max_concurrent=0."""
        dispatcher = _make_notification_dispatcher(has_config=False)
        modules = self._patch_others()
        modules["aragora.control_plane.notifications"] = MagicMock(
            get_default_notification_dispatcher=MagicMock(return_value=dispatcher)
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        notif = next(w for w in body["workers"] if w["name"] == "notification")
        assert notif["queue_enabled"] is False
        assert notif["max_concurrent"] == 0


# ============================================================================
# TestWorkerConsensusHealingBranch - consensus healing worker check
# ============================================================================


class TestWorkerConsensusHealingBranch:
    """Tests for the consensus healing worker check branch."""

    def _patch_others(self):
        """Patch gauntlet and notification to return None."""
        return {
            "aragora.server.startup.workers": MagicMock(
                get_gauntlet_worker=MagicMock(return_value=None)
            ),
            "aragora.control_plane.notifications": MagicMock(
                get_default_notification_dispatcher=MagicMock(return_value=None)
            ),
        }

    def test_healing_running(self, mock_handler):
        """Running healing worker reported correctly."""
        worker = _make_healing_worker(running=True, candidates_processed=42, healed_count=7)
        modules = self._patch_others()
        modules["aragora.queue.workers"] = MagicMock(
            get_consensus_healing_worker=MagicMock(return_value=worker)
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        healing = next(w for w in body["workers"] if w["name"] == "consensus_healing")
        assert healing["running"] is True
        assert healing["candidates_processed"] == 42
        assert healing["healed_count"] == 7

    def test_healing_not_initialized(self, mock_handler):
        """None healing worker gives not-initialized note."""
        modules = self._patch_others()
        modules["aragora.queue.workers"] = MagicMock(
            get_consensus_healing_worker=MagicMock(return_value=None)
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        healing = next(w for w in body["workers"] if w["name"] == "consensus_healing")
        assert healing["running"] is False
        assert "not initialized" in healing.get("note", "").lower()

    def test_healing_import_error(self, mock_handler):
        """ImportError on queue.workers gives module-not-available note."""
        import sys as _sys

        saved = {}
        for k in list(_sys.modules.keys()):
            if k.startswith("aragora.queue.workers"):
                saved[k] = _sys.modules.pop(k)
        try:
            import builtins

            original_import = builtins.__import__

            def fail_healing(name, *args, **kwargs):
                if name == "aragora.queue.workers":
                    raise ImportError("no module")
                return original_import(name, *args, **kwargs)

            others = self._patch_others()
            with patch("builtins.__import__", side_effect=fail_healing):
                with patch.dict("sys.modules", others):
                    result = worker_health_status(mock_handler)
        finally:
            _sys.modules.update(saved)

        body = _body(result)
        healing = next(w for w in body["workers"] if w["name"] == "consensus_healing")
        assert healing["running"] is False
        assert "not available" in healing.get("note", "").lower()

    def test_healing_attribute_error(self, mock_handler):
        """AttributeError on healing worker adds to errors."""
        modules = self._patch_others()
        modules["aragora.queue.workers"] = MagicMock(
            get_consensus_healing_worker=MagicMock(side_effect=AttributeError("bad"))
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["errors"] is not None
        assert any("consensus_healing_worker access error" in e for e in body["errors"])

    def test_healing_runtime_error(self, mock_handler):
        """RuntimeError on healing worker adds to errors."""
        modules = self._patch_others()
        modules["aragora.queue.workers"] = MagicMock(
            get_consensus_healing_worker=MagicMock(side_effect=RuntimeError("crashed"))
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["errors"] is not None
        assert any("consensus_healing_worker access error" in e for e in body["errors"])

    def test_healing_missing_getattr_fields(self, mock_handler):
        """Healing worker without _candidates_processed/_healed_count uses defaults."""
        worker = MagicMock()
        worker._running = True
        # Remove the attributes so getattr falls back to 0
        del worker._candidates_processed
        del worker._healed_count
        modules = self._patch_others()
        modules["aragora.queue.workers"] = MagicMock(
            get_consensus_healing_worker=MagicMock(return_value=worker)
        )
        with patch.dict("sys.modules", modules):
            result = worker_health_status(mock_handler)
        body = _body(result)
        healing = next(w for w in body["workers"] if w["name"] == "consensus_healing")
        assert healing["candidates_processed"] == 0
        assert healing["healed_count"] == 0


# ============================================================================
# TestWorkerOverallStatus - overall status determination
# ============================================================================


class TestWorkerOverallStatus:
    """Tests for overall health status determination logic."""

    def _make_modules(self, gauntlet=None, notification=None, healing=None):
        """Build sys.modules patch dict."""
        return {
            "aragora.server.startup.workers": MagicMock(
                get_gauntlet_worker=MagicMock(return_value=gauntlet)
            ),
            "aragora.control_plane.notifications": MagicMock(
                get_default_notification_dispatcher=MagicMock(return_value=notification)
            ),
            "aragora.queue.workers": MagicMock(
                get_consensus_healing_worker=MagicMock(return_value=healing)
            ),
        }

    def test_all_running_healthy(self, mock_handler):
        """All workers running yields status=healthy."""
        g = _make_gauntlet_worker(running=True)
        n = _make_notification_dispatcher(has_worker_task=True, worker_done=False)
        h = _make_healing_worker(running=True)
        with patch.dict("sys.modules", self._make_modules(g, n, h)):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "healthy"

    def test_none_running_unhealthy(self, mock_handler):
        """No workers running yields status=unhealthy."""
        with patch.dict("sys.modules", self._make_modules(None, None, None)):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_some_running_degraded(self, mock_handler):
        """Some workers running yields status=degraded."""
        g = _make_gauntlet_worker(running=True)
        with patch.dict("sys.modules", self._make_modules(g, None, None)):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "degraded"

    def test_all_stopped_workers_unhealthy(self, mock_handler):
        """All workers present but stopped yields unhealthy."""
        g = _make_gauntlet_worker(running=False)
        h = _make_healing_worker(running=False)
        # Notification: worker task done
        n = _make_notification_dispatcher(has_worker_task=True, worker_done=True)
        with patch.dict("sys.modules", self._make_modules(g, n, h)):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_one_of_three_running_is_degraded(self, mock_handler):
        """One running out of three is degraded."""
        g = _make_gauntlet_worker(running=False)
        h = _make_healing_worker(running=True)
        n = _make_notification_dispatcher(has_worker_task=True, worker_done=True)
        with patch.dict("sys.modules", self._make_modules(g, n, h)):
            result = worker_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "degraded"
        assert body["summary"]["running_workers"] == 1


# ============================================================================
# TestJobQueueHealthStatus - job_queue_health_status()
# ============================================================================


class TestJobQueueHealthStatus:
    """Tests for job_queue_health_status() function."""

    def test_healthy_queue(self, mock_handler):
        """Healthy queue with low counts returns status=healthy."""
        store = _make_job_store(
            stats={
                "pending": 5,
                "processing": 2,
                "completed": 100,
                "failed": 0,
                "cancelled": 3,
                "total": 110,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"
        assert body["connected"] is True
        assert body["warnings"] is None

    def test_response_has_required_fields(self, mock_handler):
        """Response contains all expected fields."""
        store = _make_job_store()
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert "status" in body
        assert "connected" in body
        assert "backend" in body
        assert "stats" in body
        assert "thresholds" in body
        assert "timestamp" in body

    def test_stats_fields(self, mock_handler):
        """Stats contain pending, processing, completed, failed, cancelled, total."""
        store = _make_job_store(
            stats={
                "pending": 5,
                "processing": 2,
                "completed": 100,
                "failed": 1,
                "cancelled": 3,
                "total": 111,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        stats = body["stats"]
        assert stats["pending"] == 5
        assert stats["processing"] == 2
        assert stats["completed"] == 100
        assert stats["failed"] == 1
        assert stats["cancelled"] == 3
        assert stats["total"] == 111


# ============================================================================
# TestJobQueueBackendDetection - backend type detection
# ============================================================================


class TestJobQueueBackendDetection:
    """Tests for backend type detection from class name."""

    @pytest.mark.parametrize(
        "class_name,expected_backend",
        [
            ("SQLiteJobStore", "sqlite"),
            ("PostgresJobStore", "postgresql"),
            ("RedisJobStore", "redis"),
            ("InMemoryStore", "InMemoryStore"),
            ("CustomBackend", "CustomBackend"),
        ],
    )
    def test_backend_detection(self, mock_handler, class_name, expected_backend):
        """Backend type is correctly detected from class name."""
        store = _make_job_store(
            class_name=class_name,
            stats={
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            },
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["backend"] == expected_backend


# ============================================================================
# TestJobQueueConnectivity - queue connectivity error branches
# ============================================================================


class TestJobQueueConnectivity:
    """Tests for queue connectivity error handling."""

    def test_import_error_shows_unhealthy(self, mock_handler):
        """ImportError on job_queue_store results in unhealthy."""
        import sys as _sys

        saved = {}
        for k in list(_sys.modules.keys()):
            if k.startswith("aragora.storage.job_queue_store"):
                saved[k] = _sys.modules.pop(k)
        try:
            import builtins

            original_import = builtins.__import__

            def fail_import(name, *args, **kwargs):
                if name == "aragora.storage.job_queue_store":
                    raise ImportError("no module")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fail_import):
                result = job_queue_health_status(mock_handler)
        finally:
            _sys.modules.update(saved)

        body = _body(result)
        assert body["status"] == "unhealthy"
        assert body["connected"] is False
        assert body["errors"] is not None

    def test_connection_error(self, mock_handler):
        """ConnectionError shows unhealthy with connectivity error."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(side_effect=ConnectionError("refused"))
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"
        assert body["connected"] is False
        assert any("connectivity" in e.lower() for e in body["errors"])

    def test_timeout_error(self, mock_handler):
        """TimeoutError shows unhealthy with connectivity error."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(side_effect=TimeoutError("timed out"))
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"
        assert body["connected"] is False

    def test_os_error(self, mock_handler):
        """OSError shows unhealthy with connectivity error."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(side_effect=OSError("disk error"))
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"
        assert body["connected"] is False

    def test_runtime_error(self, mock_handler):
        """RuntimeError shows unhealthy with status error."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(side_effect=RuntimeError("bad"))
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"
        assert body["connected"] is False
        assert any("status error" in e.lower() for e in body["errors"])

    def test_value_error(self, mock_handler):
        """ValueError shows unhealthy with status error."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(side_effect=ValueError("bad value"))
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"
        assert body["connected"] is False

    def test_type_error(self, mock_handler):
        """TypeError shows unhealthy with status error."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(side_effect=TypeError("bad type"))
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_key_error(self, mock_handler):
        """KeyError shows unhealthy with status error."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(side_effect=KeyError("missing"))
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_attribute_error(self, mock_handler):
        """AttributeError shows unhealthy with status error."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(side_effect=AttributeError("no attr"))
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_backend_unknown_when_disconnected(self, mock_handler):
        """Backend is 'unknown' when store cannot be loaded."""
        import sys as _sys

        saved = {}
        for k in list(_sys.modules.keys()):
            if k.startswith("aragora.storage.job_queue_store"):
                saved[k] = _sys.modules.pop(k)
        try:
            import builtins

            original_import = builtins.__import__

            def fail_import(name, *args, **kwargs):
                if name == "aragora.storage.job_queue_store":
                    raise ImportError("nope")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fail_import):
                result = job_queue_health_status(mock_handler)
        finally:
            _sys.modules.update(saved)

        body = _body(result)
        assert body["backend"] == "unknown"


# ============================================================================
# TestJobQueueThresholds - threshold and warning logic
# ============================================================================


class TestJobQueueThresholds:
    """Tests for configurable threshold and warning logic."""

    def _make_store_with_stats(self, pending=0, processing=0, failed=0):
        """Create a store mock with specific stats."""
        return _make_job_store(
            stats={
                "pending": pending,
                "processing": processing,
                "completed": 100,
                "failed": failed,
                "cancelled": 0,
                "total": pending + processing + 100 + failed,
            }
        )

    def _run_with_store(self, store, handler):
        """Run job_queue_health_status with the given store."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            return job_queue_health_status(handler)

    def test_no_warnings_below_thresholds(self, mock_handler):
        """No warnings when all counts below thresholds."""
        store = self._make_store_with_stats(pending=10, processing=5, failed=0)
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        assert body["warnings"] is None
        assert body["status"] == "healthy"

    def test_pending_warning_threshold(self, mock_handler):
        """Pending count at warning threshold triggers warning."""
        store = self._make_store_with_stats(pending=50, processing=0, failed=0)
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        assert body["warnings"] is not None
        assert any("pending" in w.lower() for w in body["warnings"])
        assert body["status"] == "degraded"

    def test_pending_above_warning_below_critical(self, mock_handler):
        """Pending count between warning and critical thresholds triggers warning."""
        store = self._make_store_with_stats(pending=100, processing=0, failed=0)
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        assert body["warnings"] is not None
        assert any("Warning" in w for w in body["warnings"])
        assert body["status"] == "degraded"

    def test_pending_critical_threshold(self, mock_handler):
        """Pending count at critical threshold triggers critical status."""
        store = self._make_store_with_stats(pending=200, processing=0, failed=0)
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        assert body["warnings"] is not None
        assert any("Critical" in w for w in body["warnings"])
        assert body["status"] == "critical"

    def test_pending_above_critical(self, mock_handler):
        """Pending count above critical threshold triggers critical status."""
        store = self._make_store_with_stats(pending=500, processing=0, failed=0)
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        assert body["status"] == "critical"

    def test_processing_warning_threshold(self, mock_handler):
        """Processing count at warning threshold triggers warning."""
        store = self._make_store_with_stats(pending=0, processing=20, failed=0)
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        assert body["warnings"] is not None
        assert any("processing" in w.lower() for w in body["warnings"])
        assert body["status"] == "degraded"

    def test_failed_jobs_warning(self, mock_handler):
        """Failed jobs add a warning."""
        store = self._make_store_with_stats(pending=0, processing=0, failed=5)
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        assert body["warnings"] is not None
        assert any("failed" in w.lower() for w in body["warnings"])
        assert body["status"] == "degraded"

    def test_zero_failed_no_warning(self, mock_handler):
        """Zero failed jobs does not add a warning."""
        store = self._make_store_with_stats(pending=0, processing=0, failed=0)
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        assert body["warnings"] is None

    def test_multiple_warnings(self, mock_handler):
        """Multiple threshold violations produce multiple warnings."""
        store = self._make_store_with_stats(pending=100, processing=25, failed=3)
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        assert body["warnings"] is not None
        assert len(body["warnings"]) == 3  # pending warning + processing warning + failed warning

    def test_thresholds_in_response(self, mock_handler):
        """Thresholds are included in response."""
        store = self._make_store_with_stats()
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        thresholds = body["thresholds"]
        assert thresholds["pending_warning"] == 50
        assert thresholds["pending_critical"] == 200
        assert thresholds["processing_warning"] == 20

    def test_errors_null_when_no_errors(self, mock_handler):
        """Errors field is null when there are no errors."""
        store = self._make_store_with_stats()
        result = self._run_with_store(store, mock_handler)
        body = _body(result)
        assert body["errors"] is None


# ============================================================================
# TestJobQueueEnvironmentConfig - environment variable threshold config
# ============================================================================


class TestJobQueueEnvironmentConfig:
    """Tests for environment variable threshold configuration."""

    def _make_store_with_stats(self, pending=0, processing=0, failed=0):
        return _make_job_store(
            stats={
                "pending": pending,
                "processing": processing,
                "completed": 0,
                "failed": failed,
                "cancelled": 0,
                "total": pending + processing + failed,
            }
        )

    def _run(self, store, handler):
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            return job_queue_health_status(handler)

    def test_custom_pending_warning_threshold(self, mock_handler, monkeypatch):
        """Custom ARAGORA_QUEUE_PENDING_WARNING changes threshold."""
        monkeypatch.setenv("ARAGORA_QUEUE_PENDING_WARNING", "10")
        store = self._make_store_with_stats(pending=15)
        result = self._run(store, mock_handler)
        body = _body(result)
        assert body["thresholds"]["pending_warning"] == 10
        assert body["warnings"] is not None
        assert any("pending" in w.lower() for w in body["warnings"])

    def test_custom_pending_critical_threshold(self, mock_handler, monkeypatch):
        """Custom ARAGORA_QUEUE_PENDING_CRITICAL changes threshold."""
        monkeypatch.setenv("ARAGORA_QUEUE_PENDING_CRITICAL", "100")
        store = self._make_store_with_stats(pending=100)
        result = self._run(store, mock_handler)
        body = _body(result)
        assert body["thresholds"]["pending_critical"] == 100
        assert body["status"] == "critical"

    def test_custom_processing_warning_threshold(self, mock_handler, monkeypatch):
        """Custom ARAGORA_QUEUE_PROCESSING_WARNING changes threshold."""
        monkeypatch.setenv("ARAGORA_QUEUE_PROCESSING_WARNING", "5")
        store = self._make_store_with_stats(processing=5)
        result = self._run(store, mock_handler)
        body = _body(result)
        assert body["thresholds"]["processing_warning"] == 5
        assert body["warnings"] is not None


# ============================================================================
# TestCombinedWorkerQueueHealth - combined_worker_queue_health()
# ============================================================================


class TestCombinedWorkerQueueHealth:
    """Tests for combined_worker_queue_health() aggregation."""

    def _patch_all(self, gauntlet=None, notification=None, healing=None, store=None):
        """Patch all worker and store imports."""
        modules = {
            "aragora.server.startup.workers": MagicMock(
                get_gauntlet_worker=MagicMock(return_value=gauntlet)
            ),
            "aragora.control_plane.notifications": MagicMock(
                get_default_notification_dispatcher=MagicMock(return_value=notification)
            ),
            "aragora.queue.workers": MagicMock(
                get_consensus_healing_worker=MagicMock(return_value=healing)
            ),
        }
        if store is not None:
            modules["aragora.storage.job_queue_store"] = MagicMock(
                get_job_store=MagicMock(return_value=store)
            )
        return modules

    def test_combined_returns_200(self, mock_handler):
        """Combined endpoint returns 200."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            }
        )
        with patch.dict("sys.modules", self._patch_all(store=store)):
            result = combined_worker_queue_health(mock_handler)
        assert _status(result) == 200

    def test_combined_has_all_sections(self, mock_handler):
        """Combined response has status, workers, job_queue, timestamp."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            }
        )
        with patch.dict("sys.modules", self._patch_all(store=store)):
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert "status" in body
        assert "workers" in body
        assert "job_queue" in body
        assert "timestamp" in body

    def test_combined_workers_section_has_worker_data(self, mock_handler):
        """Workers section contains worker health data."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            }
        )
        with patch.dict("sys.modules", self._patch_all(store=store)):
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert "status" in body["workers"]
        assert "workers" in body["workers"]

    def test_combined_queue_section_has_queue_data(self, mock_handler):
        """Job queue section contains queue health data."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            }
        )
        with patch.dict("sys.modules", self._patch_all(store=store)):
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert "status" in body["job_queue"]
        assert "connected" in body["job_queue"]


# ============================================================================
# TestCombinedStatusPriority - worst-status aggregation logic
# ============================================================================


class TestCombinedStatusPriority:
    """Tests for the worst-status aggregation in combined health."""

    def _patch_with_results(self, worker_result, queue_result):
        """Patch worker_health_status and job_queue_health_status directly."""
        return (
            patch(
                "aragora.server.handlers.admin.health.workers.worker_health_status",
                return_value=worker_result,
            ),
            patch(
                "aragora.server.handlers.admin.health.workers.job_queue_health_status",
                return_value=queue_result,
            ),
        )

    def _make_result(self, status: str, data: dict | None = None):
        """Create a HandlerResult with JSON body."""
        from aragora.server.handlers.base import json_response

        body_data = data or {}
        body_data["status"] = status
        return json_response(body_data)

    def test_both_healthy_yields_healthy(self, mock_handler):
        """Both healthy yields overall healthy."""
        w = self._make_result(
            "healthy", {"summary": {}, "workers": [], "errors": None, "timestamp": "t"}
        )
        q = self._make_result(
            "healthy",
            {
                "connected": True,
                "backend": "sqlite",
                "stats": {},
                "thresholds": {},
                "warnings": None,
                "errors": None,
                "timestamp": "t",
            },
        )
        p1, p2 = self._patch_with_results(w, q)
        with p1, p2:
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert body["status"] == "healthy"

    def test_worker_degraded_queue_healthy_yields_degraded(self, mock_handler):
        """Workers degraded, queue healthy yields overall degraded."""
        w = self._make_result(
            "degraded", {"summary": {}, "workers": [], "errors": None, "timestamp": "t"}
        )
        q = self._make_result(
            "healthy",
            {
                "connected": True,
                "backend": "sqlite",
                "stats": {},
                "thresholds": {},
                "warnings": None,
                "errors": None,
                "timestamp": "t",
            },
        )
        p1, p2 = self._patch_with_results(w, q)
        with p1, p2:
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert body["status"] == "degraded"

    def test_worker_healthy_queue_critical_yields_critical(self, mock_handler):
        """Workers healthy, queue critical yields overall critical."""
        w = self._make_result(
            "healthy", {"summary": {}, "workers": [], "errors": None, "timestamp": "t"}
        )
        q = self._make_result(
            "critical",
            {
                "connected": True,
                "backend": "sqlite",
                "stats": {},
                "thresholds": {},
                "warnings": [],
                "errors": None,
                "timestamp": "t",
            },
        )
        p1, p2 = self._patch_with_results(w, q)
        with p1, p2:
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert body["status"] == "critical"

    def test_both_unhealthy_yields_unhealthy(self, mock_handler):
        """Both unhealthy yields overall unhealthy."""
        w = self._make_result(
            "unhealthy", {"summary": {}, "workers": [], "errors": [], "timestamp": "t"}
        )
        q = self._make_result(
            "unhealthy",
            {
                "connected": False,
                "backend": "unknown",
                "stats": {},
                "thresholds": {},
                "warnings": None,
                "errors": [],
                "timestamp": "t",
            },
        )
        p1, p2 = self._patch_with_results(w, q)
        with p1, p2:
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_worker_unhealthy_queue_degraded_yields_unhealthy(self, mock_handler):
        """Workers unhealthy (priority 3) beats queue degraded (priority 1)."""
        w = self._make_result(
            "unhealthy", {"summary": {}, "workers": [], "errors": [], "timestamp": "t"}
        )
        q = self._make_result(
            "degraded",
            {
                "connected": True,
                "backend": "sqlite",
                "stats": {},
                "thresholds": {},
                "warnings": [],
                "errors": None,
                "timestamp": "t",
            },
        )
        p1, p2 = self._patch_with_results(w, q)
        with p1, p2:
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_worker_critical_queue_unhealthy_yields_unhealthy(self, mock_handler):
        """Queue unhealthy (priority 3) beats worker critical (priority 2)."""
        w = self._make_result(
            "critical", {"summary": {}, "workers": [], "errors": [], "timestamp": "t"}
        )
        q = self._make_result(
            "unhealthy",
            {
                "connected": False,
                "backend": "unknown",
                "stats": {},
                "thresholds": {},
                "warnings": None,
                "errors": [],
                "timestamp": "t",
            },
        )
        p1, p2 = self._patch_with_results(w, q)
        with p1, p2:
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_unknown_status_gets_priority_4(self, mock_handler):
        """Unknown status gets priority 4 (highest/worst)."""
        w = self._make_result(
            "unknown_status", {"summary": {}, "workers": [], "errors": [], "timestamp": "t"}
        )
        q = self._make_result(
            "healthy",
            {
                "connected": True,
                "backend": "sqlite",
                "stats": {},
                "thresholds": {},
                "warnings": None,
                "errors": None,
                "timestamp": "t",
            },
        )
        p1, p2 = self._patch_with_results(w, q)
        with p1, p2:
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert body["status"] == "unknown_status"

    def test_equal_priority_uses_worker_status(self, mock_handler):
        """When priorities are equal, worker status is chosen (>= comparison)."""
        w = self._make_result(
            "degraded", {"summary": {}, "workers": [], "errors": None, "timestamp": "t"}
        )
        q = self._make_result(
            "degraded",
            {
                "connected": True,
                "backend": "sqlite",
                "stats": {},
                "thresholds": {},
                "warnings": [],
                "errors": None,
                "timestamp": "t",
            },
        )
        p1, p2 = self._patch_with_results(w, q)
        with p1, p2:
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert body["status"] == "degraded"


# ============================================================================
# TestEdgeCases - edge cases and boundary conditions
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def _make_modules(self, gauntlet=None, notification=None, healing=None):
        return {
            "aragora.server.startup.workers": MagicMock(
                get_gauntlet_worker=MagicMock(return_value=gauntlet)
            ),
            "aragora.control_plane.notifications": MagicMock(
                get_default_notification_dispatcher=MagicMock(return_value=notification)
            ),
            "aragora.queue.workers": MagicMock(
                get_consensus_healing_worker=MagicMock(return_value=healing)
            ),
        }

    def test_gauntlet_zero_active_jobs(self, mock_handler):
        """Gauntlet with empty active_jobs list shows 0."""
        g = _make_gauntlet_worker(running=True, active_jobs=[])
        with patch.dict("sys.modules", self._make_modules(gauntlet=g)):
            result = worker_health_status(mock_handler)
        body = _body(result)
        gauntlet_info = next(w for w in body["workers"] if w["name"] == "gauntlet")
        assert gauntlet_info["active_jobs"] == 0

    def test_gauntlet_many_active_jobs(self, mock_handler):
        """Gauntlet with many active jobs shows correct count."""
        g = _make_gauntlet_worker(running=True, active_jobs=list(range(100)))
        with patch.dict("sys.modules", self._make_modules(gauntlet=g)):
            result = worker_health_status(mock_handler)
        body = _body(result)
        gauntlet_info = next(w for w in body["workers"] if w["name"] == "gauntlet")
        assert gauntlet_info["active_jobs"] == 100

    def test_queue_pending_at_boundary_49(self, mock_handler):
        """49 pending jobs (just below warning) yields healthy."""
        store = _make_job_store(
            stats={
                "pending": 49,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 49,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "healthy"

    def test_queue_pending_at_boundary_50(self, mock_handler):
        """50 pending jobs (at warning) yields degraded."""
        store = _make_job_store(
            stats={
                "pending": 50,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 50,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "degraded"

    def test_queue_pending_at_boundary_199(self, mock_handler):
        """199 pending jobs (just below critical) yields degraded."""
        store = _make_job_store(
            stats={
                "pending": 199,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 199,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "degraded"

    def test_queue_pending_at_boundary_200(self, mock_handler):
        """200 pending jobs (at critical) yields critical."""
        store = _make_job_store(
            stats={
                "pending": 200,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 200,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "critical"

    def test_queue_processing_at_boundary_19(self, mock_handler):
        """19 processing jobs (just below warning) yields healthy."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 19,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 19,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "healthy"

    def test_queue_processing_at_boundary_20(self, mock_handler):
        """20 processing jobs (at warning) yields degraded."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 20,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 20,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["status"] == "degraded"

    def test_queue_stats_missing_keys_default_to_zero(self, mock_handler):
        """Missing keys in queue stats default to 0."""
        store = _make_job_store(stats={})
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["stats"]["pending"] == 0
        assert body["stats"]["processing"] == 0
        assert body["stats"]["completed"] == 0
        assert body["stats"]["failed"] == 0
        assert body["stats"]["cancelled"] == 0
        assert body["stats"]["total"] == 0

    def test_combined_timestamp_format(self, mock_handler):
        """Combined endpoint timestamp ends with Z."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            }
        )
        modules = {
            "aragora.server.startup.workers": MagicMock(
                get_gauntlet_worker=MagicMock(return_value=None)
            ),
            "aragora.control_plane.notifications": MagicMock(
                get_default_notification_dispatcher=MagicMock(return_value=None)
            ),
            "aragora.queue.workers": MagicMock(
                get_consensus_healing_worker=MagicMock(return_value=None)
            ),
            "aragora.storage.job_queue_store": MagicMock(
                get_job_store=MagicMock(return_value=store)
            ),
        }
        with patch.dict("sys.modules", modules):
            result = combined_worker_queue_health(mock_handler)
        body = _body(result)
        assert body["timestamp"].endswith("Z")

    def test_worker_handler_arg_ignored(self, mock_handler):
        """Handler argument is passed through but not used meaningfully."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup.workers": MagicMock(
                    get_gauntlet_worker=MagicMock(return_value=None)
                ),
                "aragora.control_plane.notifications": MagicMock(
                    get_default_notification_dispatcher=MagicMock(return_value=None)
                ),
                "aragora.queue.workers": MagicMock(
                    get_consensus_healing_worker=MagicMock(return_value=None)
                ),
            },
        ):
            # Any handler value should work
            result1 = worker_health_status(None)
            result2 = worker_health_status("string handler")
            result3 = worker_health_status(42)
        assert _status(result1) == 200
        assert _status(result2) == 200
        assert _status(result3) == 200


# ============================================================================
# TestSecurity - security-related tests
# ============================================================================


class TestSecurity:
    """Security-related tests for worker health endpoints."""

    def test_no_sensitive_data_in_worker_response(self, mock_handler):
        """Worker response does not contain passwords, tokens, or keys."""
        g = _make_gauntlet_worker(running=True)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup.workers": MagicMock(
                    get_gauntlet_worker=MagicMock(return_value=g)
                ),
                "aragora.control_plane.notifications": MagicMock(
                    get_default_notification_dispatcher=MagicMock(return_value=None)
                ),
                "aragora.queue.workers": MagicMock(
                    get_consensus_healing_worker=MagicMock(return_value=None)
                ),
            },
        ):
            result = worker_health_status(mock_handler)
        body_str = json.dumps(_body(result)).lower()
        assert "password" not in body_str
        assert "secret" not in body_str
        assert "token" not in body_str
        assert "api_key" not in body_str

    def test_no_sensitive_data_in_queue_response(self, mock_handler):
        """Queue response does not contain passwords, tokens, or keys."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body_str = json.dumps(_body(result)).lower()
        assert "password" not in body_str
        assert "secret" not in body_str
        assert "api_key" not in body_str

    def test_error_messages_do_not_leak_internal_paths(self, mock_handler):
        """Error messages from gauntlet import failure do not expose system paths."""
        import sys as _sys

        saved = {}
        for k in list(_sys.modules.keys()):
            if k.startswith("aragora.server.startup"):
                saved[k] = _sys.modules.pop(k)
        try:
            import builtins

            original_import = builtins.__import__

            def fail_gauntlet(name, *args, **kwargs):
                if name == "aragora.server.startup.workers":
                    raise ImportError("No module named 'aragora.server.startup.workers'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fail_gauntlet):
                with patch.dict(
                    "sys.modules",
                    {
                        "aragora.control_plane.notifications": MagicMock(
                            get_default_notification_dispatcher=MagicMock(return_value=None)
                        ),
                        "aragora.queue.workers": MagicMock(
                            get_consensus_healing_worker=MagicMock(return_value=None)
                        ),
                    },
                ):
                    result = worker_health_status(mock_handler)
        finally:
            _sys.modules.update(saved)

        body = _body(result)
        # Errors should exist but not contain full filesystem paths
        if body["errors"]:
            for err in body["errors"]:
                assert "/home/" not in err
                assert "/usr/" not in err
                assert "C:\\" not in err

    def test_sanitized_attribute_error_type_only(self, mock_handler):
        """AttributeError in worker access only shows type name, not message."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup.workers": MagicMock(
                    get_gauntlet_worker=MagicMock(
                        side_effect=AttributeError("internal detail should not leak")
                    )
                ),
                "aragora.control_plane.notifications": MagicMock(
                    get_default_notification_dispatcher=MagicMock(return_value=None)
                ),
                "aragora.queue.workers": MagicMock(
                    get_consensus_healing_worker=MagicMock(return_value=None)
                ),
            },
        ):
            result = worker_health_status(mock_handler)
        body = _body(result)
        # The error includes type(e).__name__ which is "AttributeError"
        assert body["errors"] is not None
        for err in body["errors"]:
            assert "internal detail should not leak" not in err

    def test_queue_connectivity_error_sanitized(self, mock_handler):
        """Connection errors don't leak internal details."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(
                        side_effect=ConnectionError("redis://password:secret@host:6379")
                    )
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["errors"] is not None
        for err in body["errors"]:
            assert "password" not in err
            assert "secret" not in err

    def test_queue_runtime_error_sanitized(self, mock_handler):
        """Runtime errors don't leak internal details."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(
                        side_effect=RuntimeError("internal: /etc/passwd readable")
                    )
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["errors"] is not None
        for err in body["errors"]:
            assert "/etc/passwd" not in err


# ============================================================================
# TestWorkerNamesConsistency - worker name consistency checks
# ============================================================================


class TestWorkerNamesConsistency:
    """Tests that all three known worker names always appear."""

    def _make_modules(self, gauntlet=None, notification=None, healing=None):
        return {
            "aragora.server.startup.workers": MagicMock(
                get_gauntlet_worker=MagicMock(return_value=gauntlet)
            ),
            "aragora.control_plane.notifications": MagicMock(
                get_default_notification_dispatcher=MagicMock(return_value=notification)
            ),
            "aragora.queue.workers": MagicMock(
                get_consensus_healing_worker=MagicMock(return_value=healing)
            ),
        }

    def test_all_three_worker_names_present(self, mock_handler):
        """All three worker names always appear in the response."""
        with patch.dict("sys.modules", self._make_modules()):
            result = worker_health_status(mock_handler)
        body = _body(result)
        names = {w["name"] for w in body["workers"]}
        assert "gauntlet" in names
        assert "notification" in names
        assert "consensus_healing" in names

    def test_worker_names_with_running_workers(self, mock_handler):
        """Worker names present even when all are running."""
        g = _make_gauntlet_worker(running=True)
        n = _make_notification_dispatcher(has_worker_task=True, worker_done=False)
        h = _make_healing_worker(running=True)
        with patch.dict("sys.modules", self._make_modules(g, n, h)):
            result = worker_health_status(mock_handler)
        body = _body(result)
        names = [w["name"] for w in body["workers"]]
        assert len(names) == 3

    def test_worker_order_is_deterministic(self, mock_handler):
        """Workers always appear in gauntlet, notification, consensus_healing order."""
        with patch.dict("sys.modules", self._make_modules()):
            result = worker_health_status(mock_handler)
        body = _body(result)
        names = [w["name"] for w in body["workers"]]
        assert names == ["gauntlet", "notification", "consensus_healing"]


# ============================================================================
# TestJobQueueTimestamp - timestamp format checks
# ============================================================================


class TestJobQueueTimestamp:
    """Tests for timestamp formatting in queue responses."""

    def test_queue_timestamp_ends_with_z(self, mock_handler):
        """Queue health timestamp ends with Z."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert body["timestamp"].endswith("Z")

    def test_queue_timestamp_is_parseable(self, mock_handler):
        """Queue health timestamp is parseable as ISO 8601."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 0,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        ts = body["timestamp"].rstrip("Z")
        # Should be parseable without error
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_worker_timestamp_is_parseable(self, mock_handler):
        """Worker health timestamp is parseable as ISO 8601."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup.workers": MagicMock(
                    get_gauntlet_worker=MagicMock(return_value=None)
                ),
                "aragora.control_plane.notifications": MagicMock(
                    get_default_notification_dispatcher=MagicMock(return_value=None)
                ),
                "aragora.queue.workers": MagicMock(
                    get_consensus_healing_worker=MagicMock(return_value=None)
                ),
            },
        ):
            result = worker_health_status(mock_handler)
        body = _body(result)
        ts = body["timestamp"].rstrip("Z")
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None


# ============================================================================
# TestCriticalPendingWithWarning - critical pending also has critical warning
# ============================================================================


class TestCriticalPendingWithWarning:
    """Test that critical pending threshold produces a Critical warning message."""

    def test_critical_warning_message_includes_threshold(self, mock_handler):
        """Critical pending warning includes the threshold value."""
        store = _make_job_store(
            stats={
                "pending": 250,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 250,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert any("200" in w for w in body["warnings"])

    def test_warning_message_includes_count(self, mock_handler):
        """Warning message includes the actual count of pending jobs."""
        store = _make_job_store(
            stats={
                "pending": 75,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 75,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert any("75" in w for w in body["warnings"])

    def test_processing_warning_mentions_stuck_jobs(self, mock_handler):
        """Processing warning mentions stuck jobs."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 30,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "total": 30,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert any("stuck" in w.lower() for w in body["warnings"])

    def test_failed_warning_includes_count(self, mock_handler):
        """Failed jobs warning includes the count."""
        store = _make_job_store(
            stats={
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 42,
                "cancelled": 0,
                "total": 42,
            }
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.job_queue_store": MagicMock(
                    get_job_store=MagicMock(return_value=store)
                )
            },
        ):
            result = job_queue_health_status(mock_handler)
        body = _body(result)
        assert any("42" in w for w in body["warnings"])
