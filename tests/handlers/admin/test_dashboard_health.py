"""Tests for aragora.server.handlers.admin.dashboard_health module.

Comprehensive coverage of all three public functions:

  TestGetSystemHealth          - get_system_health()
  TestGetConnectorType         - get_connector_type()
  TestGetConnectorHealth       - get_connector_health()

Coverage: happy paths, error handling, edge cases, import failures,
graceful degradation, connector classification, health scoring.
Target: 40+ tests, 0 failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.dashboard_health import (
    get_connector_health,
    get_connector_type,
    get_system_health,
)


# ===========================================================================
# Helpers
# ===========================================================================


class _FakeSyncStatus(Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FakeSyncSchedule:
    enabled: bool = True


@dataclass
class FakeSyncHistory:
    status: _FakeSyncStatus = _FakeSyncStatus.COMPLETED
    duration_seconds: float | None = 1.5
    items_synced: int = 10


@dataclass
class FakeSyncJob:
    id: str = "job-1"
    connector_id: str = "conn-1"
    schedule: FakeSyncSchedule = field(default_factory=FakeSyncSchedule)
    connector: Any = None
    consecutive_failures: int = 0
    current_run_id: str | None = None
    last_run: datetime | None = None
    next_run: datetime | None = None


class FakeScheduler:
    """Minimal mock of SyncScheduler used by get_connector_health."""

    def __init__(
        self,
        stats: dict | None = None,
        jobs: list | None = None,
        history: list | None = None,
        scheduler_task: Any = None,
    ):
        self._stats = stats or {"total_jobs": 0, "running_syncs": 0, "success_rate": 1.0}
        self._jobs = jobs or []
        self._history = history or []
        self._scheduler_task = scheduler_task

    def get_stats(self) -> dict:
        return self._stats

    def list_jobs(self) -> list:
        return self._jobs

    def get_history(self, job_id: str | None = None, limit: int = 100) -> list:
        return self._history


# ===========================================================================
# TestGetSystemHealth
# ===========================================================================


class TestGetSystemHealth:
    """Tests for get_system_health()."""

    def test_returns_dict_with_required_keys(self):
        """Return value has all four required keys."""
        with patch(
            "aragora.server.handlers.admin.dashboard_health.is_prometheus_available",
            create=True,
        ):
            result = get_system_health()
        assert "uptime_seconds" in result
        assert "cache_entries" in result
        assert "active_websocket_connections" in result
        assert "prometheus_available" in result

    def test_default_values_on_import_error(self):
        """All fields fall back to safe defaults when imports fail."""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("no module"),
        ):
            result = get_system_health()
        assert result["uptime_seconds"] == 0
        assert result["cache_entries"] == 0
        assert result["active_websocket_connections"] == 0
        assert result["prometheus_available"] is False

    def test_prometheus_available_true(self):
        """prometheus_available reflects the imported helper."""
        mock_prom = MagicMock(return_value=True)
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus": MagicMock(is_prometheus_available=mock_prom)},
        ):
            with patch(
                "aragora.server.handlers.base._cache",
                new=None,
            ):
                result = get_system_health()
        assert result["prometheus_available"] is True

    def test_prometheus_available_false(self):
        """prometheus_available reflects the imported helper returning False."""
        mock_prom = MagicMock(return_value=False)
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus": MagicMock(is_prometheus_available=mock_prom)},
        ):
            with patch(
                "aragora.server.handlers.base._cache",
                new=None,
            ):
                result = get_system_health()
        assert result["prometheus_available"] is False

    def test_cache_entries_populated_from_cache(self):
        """cache_entries equals len(_cache) when _cache is non-empty."""
        fake_cache = {"a": 1, "b": 2, "c": 3}
        mock_prom = MagicMock(return_value=False)
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus": MagicMock(is_prometheus_available=mock_prom)},
        ):
            with patch(
                "aragora.server.handlers.base._cache",
                new=fake_cache,
            ):
                result = get_system_health()
        assert result["cache_entries"] == 3

    def test_cache_entries_zero_when_cache_none(self):
        """cache_entries stays 0 when _cache is None."""
        mock_prom = MagicMock(return_value=False)
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus": MagicMock(is_prometheus_available=mock_prom)},
        ):
            with patch(
                "aragora.server.handlers.base._cache",
                new=None,
            ):
                result = get_system_health()
        assert result["cache_entries"] == 0

    def test_cache_entries_zero_when_cache_empty(self):
        """cache_entries is 0 for an empty cache dict."""
        mock_prom = MagicMock(return_value=False)
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus": MagicMock(is_prometheus_available=mock_prom)},
        ):
            with patch(
                "aragora.server.handlers.base._cache",
                new={},
            ):
                result = get_system_health()
        assert result["cache_entries"] == 0

    def test_handles_type_error(self):
        """TypeError during health collection returns defaults."""
        mock_prom = MagicMock(side_effect=TypeError("bad type"))
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus": MagicMock(is_prometheus_available=mock_prom)},
        ):
            result = get_system_health()
        assert result["prometheus_available"] is False

    def test_handles_value_error(self):
        """ValueError during health collection returns defaults."""
        mock_prom = MagicMock(side_effect=ValueError("bad value"))
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus": MagicMock(is_prometheus_available=mock_prom)},
        ):
            result = get_system_health()
        assert result["prometheus_available"] is False

    def test_handles_runtime_error(self):
        """RuntimeError during health collection returns defaults."""
        mock_prom = MagicMock(side_effect=RuntimeError("runtime"))
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus": MagicMock(is_prometheus_available=mock_prom)},
        ):
            result = get_system_health()
        assert result["prometheus_available"] is False

    def test_handles_attribute_error(self):
        """AttributeError during health collection returns defaults."""
        mock_prom = MagicMock(side_effect=AttributeError("attr"))
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus": MagicMock(is_prometheus_available=mock_prom)},
        ):
            result = get_system_health()
        assert result["prometheus_available"] is False

    def test_handles_key_error(self):
        """KeyError during health collection returns defaults."""
        mock_prom = MagicMock(side_effect=KeyError("missing key"))
        with patch.dict(
            "sys.modules",
            {"aragora.server.prometheus": MagicMock(is_prometheus_available=mock_prom)},
        ):
            result = get_system_health()
        assert result["prometheus_available"] is False

    def test_uptime_seconds_always_zero(self):
        """uptime_seconds is always 0 (placeholder)."""
        result = get_system_health()
        assert result["uptime_seconds"] == 0

    def test_active_websocket_connections_always_zero(self):
        """active_websocket_connections is always 0 (placeholder)."""
        result = get_system_health()
        assert result["active_websocket_connections"] == 0


# ===========================================================================
# TestGetConnectorType
# ===========================================================================


class TestGetConnectorType:
    """Tests for get_connector_type()."""

    def test_none_returns_unknown(self):
        """None connector returns 'unknown'."""
        assert get_connector_type(None) == "unknown"

    def test_falsy_empty_string_returns_unknown(self):
        """Empty string (falsy) connector returns 'unknown'."""
        assert get_connector_type("") == "unknown"

    def test_falsy_zero_returns_unknown(self):
        """0 (falsy) connector returns 'unknown'."""
        assert get_connector_type(0) == "unknown"

    def test_github_enterprise_connector(self):
        """GitHubEnterpriseConnector maps to 'github'."""
        obj = type("GitHubEnterpriseConnector", (), {})()
        assert get_connector_type(obj) == "github"

    def test_s3_connector(self):
        """S3Connector maps to 's3'."""
        obj = type("S3Connector", (), {})()
        assert get_connector_type(obj) == "s3"

    def test_postgresql_connector(self):
        """PostgreSQLConnector maps to 'postgresql'."""
        obj = type("PostgreSQLConnector", (), {})()
        assert get_connector_type(obj) == "postgresql"

    def test_mongodb_connector(self):
        """MongoDBConnector maps to 'mongodb'."""
        obj = type("MongoDBConnector", (), {})()
        assert get_connector_type(obj) == "mongodb"

    def test_fhir_connector(self):
        """FHIRConnector maps to 'fhir'."""
        obj = type("FHIRConnector", (), {})()
        assert get_connector_type(obj) == "fhir"

    def test_unmapped_connector_strips_connector_suffix(self):
        """Unknown *Connector class strips the 'connector' suffix."""
        obj = type("SlackConnector", (), {})()
        assert get_connector_type(obj) == "slack"

    def test_unmapped_connector_no_suffix(self):
        """Class name without 'connector' is returned as-is (lowered)."""
        obj = type("RedisCache", (), {})()
        assert get_connector_type(obj) == "rediscache"

    def test_case_insensitive_mapping(self):
        """Mapping lookup is case-insensitive via lower()."""
        # Class name matches mapping after lowering
        obj = type("GITHUBENTERPRISECONNECTOR", (), {})()
        assert get_connector_type(obj) == "github"

    def test_plain_object(self):
        """Plain object returns lowered class name."""
        obj = object()
        assert get_connector_type(obj) == "object"

    def test_connector_suffix_only(self):
        """Class named exactly 'Connector' strips to empty string."""
        obj = type("Connector", (), {})()
        assert get_connector_type(obj) == ""


# ===========================================================================
# TestGetConnectorHealth
# ===========================================================================


class TestGetConnectorHealth:
    """Tests for get_connector_health()."""

    # -- Default / error paths --

    def test_returns_default_on_import_error(self):
        """ImportError from get_scheduler returns safe defaults."""
        with patch(
            "aragora.server.handlers.admin.dashboard_health.get_scheduler",
            create=True,
            side_effect=ImportError("no module"),
        ):
            # Force the import path to fail
            result = self._call_with_import_error()
        assert result["summary"]["total_connectors"] == 0
        assert result["connectors"] == []

    def _call_with_import_error(self):
        """Call get_connector_health with import failure."""
        import sys

        saved = sys.modules.get("aragora.server.handlers.connectors")
        try:
            sys.modules["aragora.server.handlers.connectors"] = None  # type: ignore
            return get_connector_health()
        finally:
            if saved is not None:
                sys.modules["aragora.server.handlers.connectors"] = saved
            else:
                sys.modules.pop("aragora.server.handlers.connectors", None)

    def test_default_summary_on_import_error(self):
        """Default summary has all required keys on import failure."""
        result = self._call_with_import_error()
        summary = result["summary"]
        assert summary["total_connectors"] == 0
        assert summary["healthy"] == 0
        assert summary["degraded"] == 0
        assert summary["unhealthy"] == 0
        assert summary["health_score"] == 100
        assert summary["scheduler_running"] is False
        assert summary["running_syncs"] == 0
        assert summary["success_rate"] == 1.0

    def test_empty_scheduler_no_jobs(self):
        """Scheduler with no jobs gives empty connectors list, health_score=100."""
        scheduler = FakeScheduler()
        with patch(
            "aragora.server.handlers.admin.dashboard_health.get_scheduler",
            create=True,
        ):
            result = self._call_with_scheduler(scheduler)
        assert result["summary"]["total_connectors"] == 0
        assert result["connectors"] == []
        assert result["summary"]["health_score"] == 100

    def _call_with_scheduler(self, scheduler: FakeScheduler) -> dict:
        """Call get_connector_health with a FakeScheduler injected."""
        with patch(
            "aragora.server.handlers.connectors.get_scheduler",
            return_value=scheduler,
        ):
            return get_connector_health()

    def test_healthy_connector(self):
        """Job with 0 failures, enabled, no current_run => healthy + connected."""
        job = FakeSyncJob()
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 1.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["healthy"] == 1
        assert result["summary"]["degraded"] == 0
        assert result["summary"]["unhealthy"] == 0
        assert result["summary"]["health_score"] == 100
        conn = result["connectors"][0]
        assert conn["health"] == "healthy"
        assert conn["status"] == "connected"

    def test_degraded_connector_one_failure(self):
        """Job with 1 failure => degraded."""
        job = FakeSyncJob(consecutive_failures=1)
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 0.9},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["degraded"] == 1
        conn = result["connectors"][0]
        assert conn["health"] == "degraded"

    def test_degraded_connector_two_failures(self):
        """Job with 2 failures => still degraded (threshold is 3)."""
        job = FakeSyncJob(consecutive_failures=2)
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 0.8},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["degraded"] == 1
        assert result["connectors"][0]["health"] == "degraded"

    def test_unhealthy_connector_three_failures(self):
        """Job with 3+ failures => unhealthy + error status."""
        job = FakeSyncJob(consecutive_failures=3)
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 0.5},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["unhealthy"] == 1
        conn = result["connectors"][0]
        assert conn["health"] == "unhealthy"
        assert conn["status"] == "error"

    def test_unhealthy_connector_many_failures(self):
        """Job with many failures => unhealthy."""
        job = FakeSyncJob(consecutive_failures=10)
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 0.1},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["unhealthy"] == 1
        assert result["connectors"][0]["health"] == "unhealthy"

    def test_syncing_status_when_current_run_id(self):
        """Job with current_run_id => status 'syncing'."""
        job = FakeSyncJob(current_run_id="run-abc")
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 1, "success_rate": 1.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["connectors"][0]["status"] == "syncing"

    def test_disconnected_status_when_disabled(self):
        """Job with schedule disabled => status 'disconnected'."""
        job = FakeSyncJob(schedule=FakeSyncSchedule(enabled=False))
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 1.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["connectors"][0]["status"] == "disconnected"

    def test_history_metrics_calculated(self):
        """error_rate, avg_sync_duration, items_synced computed from history."""
        job = FakeSyncJob()
        history = [
            FakeSyncHistory(status=_FakeSyncStatus.COMPLETED, duration_seconds=2.0, items_synced=5),
            FakeSyncHistory(status=_FakeSyncStatus.FAILED, duration_seconds=1.0, items_synced=0),
            FakeSyncHistory(
                status=_FakeSyncStatus.COMPLETED, duration_seconds=3.0, items_synced=15
            ),
        ]
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 0.67},
            jobs=[job],
            history=history,
        )
        result = self._call_with_scheduler(scheduler)
        conn = result["connectors"][0]
        # 1 of 3 failed => error_rate ~33.3
        assert conn["error_rate"] == pytest.approx(33.3, abs=0.1)
        # uptime = 100 - error_rate
        assert conn["uptime"] == pytest.approx(66.7, abs=0.1)
        # avg duration: (2+1+3)/3 = 2.0
        assert conn["avg_sync_duration"] == pytest.approx(2.0, abs=0.1)
        # total items: 5+0+15 = 20
        assert conn["items_synced"] == 20

    def test_empty_history_gives_zero_metrics(self):
        """No history => error_rate=0, avg_duration=0, items_synced=0."""
        job = FakeSyncJob()
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 1.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        conn = result["connectors"][0]
        assert conn["error_rate"] == 0.0
        assert conn["uptime"] == 100.0
        assert conn["avg_sync_duration"] == 0.0
        assert conn["items_synced"] == 0

    def test_connector_name_from_connector_attribute(self):
        """connector_name comes from connector.name when available."""
        connector = MagicMock()
        connector.name = "My GitHub"
        type(connector).__name__ = "GitHubEnterpriseConnector"
        job = FakeSyncJob(connector=connector, connector_id="gh-1")
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 1.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        conn = result["connectors"][0]
        assert conn["connector_name"] == "My GitHub"
        assert conn["connector_type"] == "github"

    def test_connector_name_falls_back_to_connector_id(self):
        """Without a connector object, name defaults to connector_id."""
        job = FakeSyncJob(connector=None, connector_id="fallback-id")
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 1.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["connectors"][0]["connector_name"] == "fallback-id"
        assert result["connectors"][0]["connector_type"] == "unknown"

    def test_last_sync_and_next_sync_iso_format(self):
        """last_sync and next_sync are ISO-formatted when present."""
        now = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
        later = datetime(2026, 2, 23, 13, 0, 0, tzinfo=timezone.utc)
        job = FakeSyncJob(last_run=now, next_run=later)
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 1.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        conn = result["connectors"][0]
        assert conn["last_sync"] == now.isoformat()
        assert conn["next_sync"] == later.isoformat()

    def test_last_sync_none_when_no_run(self):
        """last_sync and next_sync are None when no runs."""
        job = FakeSyncJob(last_run=None, next_run=None)
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 1.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        conn = result["connectors"][0]
        assert conn["last_sync"] is None
        assert conn["next_sync"] is None

    def test_health_score_mixed_connectors(self):
        """health_score is (healthy/total)*100 rounded."""
        jobs = [
            FakeSyncJob(id="j1", consecutive_failures=0),  # healthy
            FakeSyncJob(id="j2", consecutive_failures=1),  # degraded
            FakeSyncJob(id="j3", consecutive_failures=5),  # unhealthy
        ]
        scheduler = FakeScheduler(
            stats={"total_jobs": 3, "running_syncs": 0, "success_rate": 0.67},
            jobs=jobs,
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        # 1 healthy out of 3 => 33%
        assert result["summary"]["health_score"] == 33
        assert result["summary"]["healthy"] == 1
        assert result["summary"]["degraded"] == 1
        assert result["summary"]["unhealthy"] == 1

    def test_health_score_all_healthy(self):
        """All healthy connectors gives health_score=100."""
        jobs = [
            FakeSyncJob(id="j1", consecutive_failures=0),
            FakeSyncJob(id="j2", consecutive_failures=0),
        ]
        scheduler = FakeScheduler(
            stats={"total_jobs": 2, "running_syncs": 0, "success_rate": 1.0},
            jobs=jobs,
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["health_score"] == 100

    def test_health_score_all_unhealthy(self):
        """All unhealthy connectors gives health_score=0."""
        jobs = [
            FakeSyncJob(id="j1", consecutive_failures=5),
            FakeSyncJob(id="j2", consecutive_failures=3),
        ]
        scheduler = FakeScheduler(
            stats={"total_jobs": 2, "running_syncs": 0, "success_rate": 0.0},
            jobs=jobs,
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["health_score"] == 0

    def test_scheduler_running_true(self):
        """scheduler_running=True when _scheduler_task is not None."""
        scheduler = FakeScheduler(scheduler_task="running")
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["scheduler_running"] is True

    def test_scheduler_running_false(self):
        """scheduler_running=False when _scheduler_task is None."""
        scheduler = FakeScheduler(scheduler_task=None)
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["scheduler_running"] is False

    def test_running_syncs_from_stats(self):
        """running_syncs is taken from scheduler stats."""
        scheduler = FakeScheduler(
            stats={"total_jobs": 5, "running_syncs": 3, "success_rate": 0.9},
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["running_syncs"] == 3

    def test_success_rate_from_stats(self):
        """success_rate is taken from scheduler stats."""
        scheduler = FakeScheduler(
            stats={"total_jobs": 5, "running_syncs": 0, "success_rate": 0.75},
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["success_rate"] == 0.75

    def test_total_connectors_from_stats(self):
        """total_connectors comes from stats total_jobs."""
        scheduler = FakeScheduler(
            stats={"total_jobs": 7, "running_syncs": 0, "success_rate": 1.0},
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["total_connectors"] == 7

    def test_handles_type_error(self):
        """TypeError from scheduler returns defaults."""
        scheduler = FakeScheduler()
        scheduler.get_stats = MagicMock(side_effect=TypeError("bad type"))
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["total_connectors"] == 0
        assert result["connectors"] == []

    def test_handles_attribute_error(self):
        """AttributeError from scheduler returns defaults."""
        scheduler = FakeScheduler()
        scheduler.get_stats = MagicMock(side_effect=AttributeError("bad attr"))
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["total_connectors"] == 0

    def test_handles_runtime_error(self):
        """RuntimeError from scheduler returns defaults."""
        scheduler = FakeScheduler()
        scheduler.get_stats = MagicMock(side_effect=RuntimeError("runtime"))
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["total_connectors"] == 0

    def test_handles_value_error(self):
        """ValueError from scheduler returns defaults."""
        scheduler = FakeScheduler()
        scheduler.get_stats = MagicMock(side_effect=ValueError("bad value"))
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["total_connectors"] == 0

    def test_handles_key_error(self):
        """KeyError from scheduler returns defaults."""
        scheduler = FakeScheduler()
        scheduler.get_stats = MagicMock(side_effect=KeyError("missing"))
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["total_connectors"] == 0

    def test_handles_os_error(self):
        """OSError from scheduler returns defaults."""
        scheduler = FakeScheduler()
        scheduler.get_stats = MagicMock(side_effect=OSError("disk"))
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["total_connectors"] == 0

    def test_connector_id_in_output(self):
        """Each connector entry has correct connector_id."""
        job = FakeSyncJob(connector_id="my-conn-42")
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 1.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["connectors"][0]["connector_id"] == "my-conn-42"

    def test_consecutive_failures_in_output(self):
        """consecutive_failures is included in connector output."""
        job = FakeSyncJob(consecutive_failures=7)
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 0.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["connectors"][0]["consecutive_failures"] == 7

    def test_history_with_none_duration(self):
        """History entries with None duration count as 0."""
        job = FakeSyncJob()
        history = [
            FakeSyncHistory(
                status=_FakeSyncStatus.COMPLETED, duration_seconds=None, items_synced=5
            ),
            FakeSyncHistory(
                status=_FakeSyncStatus.COMPLETED, duration_seconds=4.0, items_synced=10
            ),
        ]
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 1.0},
            jobs=[job],
            history=history,
        )
        result = self._call_with_scheduler(scheduler)
        conn = result["connectors"][0]
        # avg duration: (0+4)/2 = 2.0
        assert conn["avg_sync_duration"] == 2.0
        assert conn["items_synced"] == 15

    def test_syncing_overrides_failure_status(self):
        """current_run_id makes status 'syncing' even with failures."""
        job = FakeSyncJob(consecutive_failures=5, current_run_id="run-123")
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 1, "success_rate": 0.5},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["connectors"][0]["status"] == "syncing"
        # Health is still unhealthy though
        assert result["connectors"][0]["health"] == "unhealthy"

    def test_connector_without_name_attr(self):
        """Connector without 'name' attribute falls back to connector_id."""
        connector = MagicMock(spec=[])  # no attributes
        type(connector).__name__ = "CustomConnector"
        # Since spec=[] means no attrs, getattr will use default
        job = FakeSyncJob(connector=connector, connector_id="custom-1")
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 1.0},
            jobs=[job],
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        conn = result["connectors"][0]
        # getattr(connector, "name", job.connector_id) with no name attr => "custom-1"
        assert conn["connector_name"] == "custom-1"
        assert conn["connector_type"] == "custom"

    def test_multiple_jobs_different_states(self):
        """Multiple jobs in different health states are all listed."""
        jobs = [
            FakeSyncJob(id="j1", connector_id="c1", consecutive_failures=0),
            FakeSyncJob(id="j2", connector_id="c2", consecutive_failures=2),
            FakeSyncJob(id="j3", connector_id="c3", consecutive_failures=3),
            FakeSyncJob(id="j4", connector_id="c4", consecutive_failures=0, current_run_id="r1"),
        ]
        scheduler = FakeScheduler(
            stats={"total_jobs": 4, "running_syncs": 1, "success_rate": 0.8},
            jobs=jobs,
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert len(result["connectors"]) == 4
        statuses = {c["connector_id"]: c["status"] for c in result["connectors"]}
        assert statuses["c1"] == "connected"
        assert statuses["c2"] == "connected"  # degraded health but connected status
        assert statuses["c3"] == "error"
        assert statuses["c4"] == "syncing"

    def test_health_score_50_percent(self):
        """Two healthy out of four gives health_score=50."""
        jobs = [
            FakeSyncJob(id="j1", consecutive_failures=0),
            FakeSyncJob(id="j2", consecutive_failures=0),
            FakeSyncJob(id="j3", consecutive_failures=3),
            FakeSyncJob(id="j4", consecutive_failures=3),
        ]
        scheduler = FakeScheduler(
            stats={"total_jobs": 4, "running_syncs": 0, "success_rate": 0.5},
            jobs=jobs,
            history=[],
        )
        result = self._call_with_scheduler(scheduler)
        assert result["summary"]["health_score"] == 50

    def test_all_failed_history(self):
        """All-failed history gives error_rate=100, uptime=0."""
        job = FakeSyncJob()
        history = [
            FakeSyncHistory(status=_FakeSyncStatus.FAILED, duration_seconds=1.0, items_synced=0),
            FakeSyncHistory(status=_FakeSyncStatus.FAILED, duration_seconds=2.0, items_synced=0),
        ]
        scheduler = FakeScheduler(
            stats={"total_jobs": 1, "running_syncs": 0, "success_rate": 0.0},
            jobs=[job],
            history=history,
        )
        result = self._call_with_scheduler(scheduler)
        conn = result["connectors"][0]
        assert conn["error_rate"] == 100.0
        assert conn["uptime"] == 0.0
        assert conn["items_synced"] == 0
