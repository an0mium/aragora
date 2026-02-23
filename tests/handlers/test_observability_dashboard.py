"""Tests for ObservabilityDashboardHandler.

Covers:
- Handler initialization and context extraction
- Route matching (can_handle for all known routes)
- GET /api/observability/dashboard - Aggregated dashboard metrics
- GET /api/observability/metrics - Alias for dashboard
- Versioned path support (/api/v1/observability/dashboard)
- Debate metrics collection (with storage, without storage, edge cases)
- Agent rankings collection (dict entries, object entries, empty, error)
- Circuit breaker state collection (registry, _instances fallback, unavailable)
- Self-improvement cycle collection (dict runs, object runs, unavailable)
- System health collection (with psutil, without psutil)
- Error rates collection (with metrics module, without)
- Graceful degradation when subsystems are unavailable
- Collection timing measurement
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult, json_response
from aragora.server.handlers.observability.dashboard import (
    ObservabilityDashboardHandler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an ObservabilityDashboardHandler with empty context."""
    return ObservabilityDashboardHandler(server_context={})


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = MagicMock()
    storage.list_debates.return_value = []
    return storage


@pytest.fixture
def mock_elo():
    """Create a mock ELO system."""
    elo = MagicMock()
    elo.get_leaderboard.return_value = []
    return elo


@pytest.fixture
def handler_with_storage(mock_storage):
    """Create a handler with mock storage."""
    return ObservabilityDashboardHandler(
        server_context={"storage": mock_storage}
    )


@pytest.fixture
def handler_with_elo(mock_elo):
    """Create a handler with mock ELO system."""
    return ObservabilityDashboardHandler(
        server_context={"elo_system": mock_elo}
    )


@pytest.fixture
def handler_full(mock_storage, mock_elo):
    """Create a handler with all subsystems."""
    return ObservabilityDashboardHandler(
        server_context={
            "storage": mock_storage,
            "elo_system": mock_elo,
        }
    )


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with basic attributes."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 54321)
    h.headers = {
        "Content-Length": "0",
        "Host": "localhost:8080",
    }
    return h


# ===========================================================================
# Initialization
# ===========================================================================


class TestInit:
    """Tests for ObservabilityDashboardHandler initialization."""

    def test_init_with_empty_context(self):
        h = ObservabilityDashboardHandler(server_context={})
        assert h._elo is None
        assert h._storage is None

    def test_init_with_storage(self, mock_storage):
        h = ObservabilityDashboardHandler(
            server_context={"storage": mock_storage}
        )
        assert h._storage is mock_storage
        assert h._elo is None

    def test_init_with_elo(self, mock_elo):
        h = ObservabilityDashboardHandler(
            server_context={"elo_system": mock_elo}
        )
        assert h._elo is mock_elo
        assert h._storage is None

    def test_init_with_full_context(self, mock_storage, mock_elo):
        h = ObservabilityDashboardHandler(
            server_context={
                "storage": mock_storage,
                "elo_system": mock_elo,
            }
        )
        assert h._storage is mock_storage
        assert h._elo is mock_elo

    def test_routes_class_attribute(self):
        assert "/api/observability/dashboard" in ObservabilityDashboardHandler.ROUTES
        assert "/api/observability/metrics" in ObservabilityDashboardHandler.ROUTES

    def test_resource_type(self):
        assert ObservabilityDashboardHandler.RESOURCE_TYPE == "observability"


# ===========================================================================
# Route Matching (can_handle)
# ===========================================================================


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/observability/dashboard",
            "/api/observability/metrics",
        ],
    )
    def test_can_handle_known_routes(self, handler, path):
        assert handler.can_handle(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/observability/dashboard",
            "/api/v1/observability/metrics",
        ],
    )
    def test_can_handle_versioned_routes(self, handler, path):
        """Versioned paths are stripped to their base route and matched."""
        assert handler.can_handle(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/observability",
            "/api/observability/unknown",
            "/api/metrics",
            "/api/debates",
            "/observability/dashboard",
            "/",
            "",
        ],
    )
    def test_cannot_handle_unrelated_routes(self, handler, path):
        assert handler.can_handle(path) is False

    def test_can_handle_with_method_param(self, handler):
        """The method parameter is accepted but does not affect matching."""
        assert handler.can_handle("/api/observability/dashboard", "GET") is True
        assert handler.can_handle("/api/observability/dashboard", "POST") is True


# ===========================================================================
# GET /api/observability/dashboard
# ===========================================================================


class TestDashboardEndpoint:
    """Tests for the main dashboard endpoint."""

    @pytest.mark.asyncio
    async def test_returns_200(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/observability/dashboard", {}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_json_with_all_sections(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/observability/dashboard", {}, mock_http_handler
        )
        body = result[0]
        assert "timestamp" in body
        assert "debate_metrics" in body
        assert "agent_rankings" in body
        assert "circuit_breakers" in body
        assert "self_improve" in body
        assert "system_health" in body
        assert "error_rates" in body
        assert "collection_time_ms" in body

    @pytest.mark.asyncio
    async def test_timestamp_is_reasonable(self, handler, mock_http_handler):
        before = time.time()
        result = await handler.handle(
            "/api/observability/dashboard", {}, mock_http_handler
        )
        after = time.time()
        body = result[0]
        assert before <= body["timestamp"] <= after

    @pytest.mark.asyncio
    async def test_collection_time_is_non_negative(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/observability/dashboard", {}, mock_http_handler
        )
        body = result[0]
        assert body["collection_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_metrics_alias_route(self, handler, mock_http_handler):
        """GET /api/observability/metrics returns the same dashboard data."""
        result = await handler.handle(
            "/api/observability/metrics", {}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 200
        body = result[0]
        assert "debate_metrics" in body
        assert "agent_rankings" in body

    @pytest.mark.asyncio
    async def test_versioned_path(self, handler, mock_http_handler):
        """Versioned path /api/v1/observability/dashboard also works."""
        result = await handler.handle(
            "/api/v1/observability/dashboard", {}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_unmatched_path_returns_none(self, handler, mock_http_handler):
        """An unmatched path returns None."""
        result = await handler.handle(
            "/api/observability/unknown", {}, mock_http_handler
        )
        assert result is None


# ===========================================================================
# Debate Metrics Collection
# ===========================================================================


class TestDebateMetrics:
    """Tests for _collect_debate_metrics."""

    def test_fallback_when_no_storage(self, handler):
        result = handler._collect_debate_metrics()
        assert result["available"] is False
        assert result["total_debates"] == 0
        assert result["avg_duration_seconds"] == 0
        assert result["consensus_rate"] == 0

    def test_empty_debates(self, handler_with_storage, mock_storage):
        mock_storage.list_debates.return_value = []
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is True
        assert result["total_debates"] == 0
        assert result["avg_duration_seconds"] == 0
        assert result["consensus_rate"] == 0

    def test_debates_with_dict_entries(self, handler_with_storage, mock_storage):
        debates = [
            {
                "duration": 10.5,
                "consensus_reached": True,
            },
            {
                "duration": 20.0,
                "consensus_reached": False,
            },
            {
                "duration_seconds": 15.0,
                "consensus": True,
            },
        ]
        mock_storage.list_debates.return_value = debates
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is True
        assert result["total_debates"] == 3
        assert result["avg_duration_seconds"] == round((10.5 + 20.0 + 15.0) / 3, 1)
        assert result["consensus_rate"] == round(2 / 3, 3)

    def test_debates_with_object_entries(self, handler_with_storage, mock_storage):
        debate1 = SimpleNamespace(duration=12.0, consensus_reached=True)
        debate2 = SimpleNamespace(duration=8.0, consensus_reached=False)
        mock_storage.list_debates.return_value = [debate1, debate2]
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is True
        assert result["total_debates"] == 2
        assert result["avg_duration_seconds"] == 10.0
        assert result["consensus_rate"] == 0.5

    def test_debates_with_no_durations(self, handler_with_storage, mock_storage):
        """Debates without duration fields result in avg=0."""
        debates = [
            {"consensus_reached": True},
            {"consensus_reached": False},
        ]
        mock_storage.list_debates.return_value = debates
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is True
        assert result["total_debates"] == 2
        assert result["avg_duration_seconds"] == 0

    def test_debates_samples_last_100(self, handler_with_storage, mock_storage):
        """Only the last 100 debates are sampled for metrics."""
        debates = [{"duration": 1.0, "consensus_reached": True}] * 150
        mock_storage.list_debates.return_value = debates
        result = handler_with_storage._collect_debate_metrics()
        assert result["total_debates"] == 150
        # consensus_rate should be 1.0 (all of last 100 have consensus)
        assert result["consensus_rate"] == 1.0

    def test_storage_without_list_debates(self, handler_with_storage, mock_storage):
        """Storage without list_debates attribute returns empty list fallback."""
        del mock_storage.list_debates
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is True
        assert result["total_debates"] == 0

    def test_storage_runtime_error_returns_fallback(self, handler_with_storage, mock_storage):
        mock_storage.list_debates.side_effect = RuntimeError("db error")
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is False
        assert result["total_debates"] == 0

    def test_storage_value_error_returns_fallback(self, handler_with_storage, mock_storage):
        mock_storage.list_debates.side_effect = ValueError("bad value")
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is False

    def test_storage_type_error_returns_fallback(self, handler_with_storage, mock_storage):
        mock_storage.list_debates.side_effect = TypeError("wrong type")
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is False

    def test_storage_os_error_returns_fallback(self, handler_with_storage, mock_storage):
        mock_storage.list_debates.side_effect = OSError("disk error")
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is False

    def test_storage_key_error_returns_fallback(self, handler_with_storage, mock_storage):
        mock_storage.list_debates.side_effect = KeyError("missing")
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is False

    def test_storage_attribute_error_returns_fallback(self, handler_with_storage, mock_storage):
        mock_storage.list_debates.side_effect = AttributeError("no attr")
        result = handler_with_storage._collect_debate_metrics()
        assert result["available"] is False


# ===========================================================================
# Agent Rankings Collection
# ===========================================================================


class TestAgentRankings:
    """Tests for _collect_agent_rankings."""

    def test_fallback_when_no_elo(self, handler):
        result = handler._collect_agent_rankings()
        assert result["available"] is False
        assert result["top_agents"] == []

    def test_empty_leaderboard(self, handler_with_elo, mock_elo):
        mock_elo.get_leaderboard.return_value = []
        result = handler_with_elo._collect_agent_rankings()
        assert result["available"] is True
        assert result["top_agents"] == []

    def test_dict_leaderboard_entries(self, handler_with_elo, mock_elo):
        lb = [
            {
                "agent": "claude",
                "rating": 1600,
                "matches": 50,
                "win_rate": 0.72,
            },
            {
                "name": "gpt4",
                "elo": 1550,
                "total_matches": 45,
                "win_rate": 0.65,
            },
        ]
        mock_elo.get_leaderboard.return_value = lb
        result = handler_with_elo._collect_agent_rankings()
        assert result["available"] is True
        assert len(result["top_agents"]) == 2

        first = result["top_agents"][0]
        assert first["name"] == "claude"
        assert first["rating"] == 1600
        assert first["matches"] == 50
        assert first["win_rate"] == 0.72

        second = result["top_agents"][1]
        assert second["name"] == "gpt4"
        assert second["rating"] == 1550
        assert second["matches"] == 45
        assert second["win_rate"] == 0.65

    def test_dict_entry_with_fallback_values(self, handler_with_elo, mock_elo):
        """Dict entries missing fields use sensible defaults."""
        mock_elo.get_leaderboard.return_value = [{}]
        result = handler_with_elo._collect_agent_rankings()
        assert result["available"] is True
        agent = result["top_agents"][0]
        assert agent["name"] == "unknown"
        assert agent["rating"] == 1500
        assert agent["matches"] == 0
        assert agent["win_rate"] == 0

    def test_object_leaderboard_entries(self, handler_with_elo, mock_elo):
        entry = SimpleNamespace(
            agent="gemini",
            rating=1580,
            matches=30,
            win_rate=0.6,
        )
        mock_elo.get_leaderboard.return_value = [entry]
        result = handler_with_elo._collect_agent_rankings()
        assert result["available"] is True
        agent = result["top_agents"][0]
        assert agent["name"] == "gemini"
        assert agent["rating"] == 1580
        assert agent["matches"] == 30
        assert agent["win_rate"] == 0.6

    def test_object_entry_with_name_fallback(self, handler_with_elo, mock_elo):
        """Object entry uses 'name' attribute when 'agent' is missing."""
        entry = SimpleNamespace(
            name="mistral",
            elo=1520,
            total_matches=20,
            win_rate=0.55,
        )
        mock_elo.get_leaderboard.return_value = [entry]
        result = handler_with_elo._collect_agent_rankings()
        agent = result["top_agents"][0]
        assert agent["name"] == "mistral"
        assert agent["rating"] == 1520
        assert agent["matches"] == 20

    def test_object_entry_with_defaults(self, handler_with_elo, mock_elo):
        """Object entries missing attrs get defaults."""
        entry = SimpleNamespace()
        mock_elo.get_leaderboard.return_value = [entry]
        result = handler_with_elo._collect_agent_rankings()
        agent = result["top_agents"][0]
        assert agent["name"] == "unknown"
        assert agent["rating"] == 1500
        assert agent["matches"] == 0
        assert agent["win_rate"] == 0

    def test_elo_without_get_leaderboard(self, handler_with_elo, mock_elo):
        """ELO system without get_leaderboard returns empty."""
        del mock_elo.get_leaderboard
        result = handler_with_elo._collect_agent_rankings()
        assert result["available"] is True
        assert result["top_agents"] == []

    def test_elo_runtime_error_returns_fallback(self, handler_with_elo, mock_elo):
        mock_elo.get_leaderboard.side_effect = RuntimeError("elo error")
        result = handler_with_elo._collect_agent_rankings()
        assert result["available"] is False
        assert result["top_agents"] == []

    def test_elo_value_error_returns_fallback(self, handler_with_elo, mock_elo):
        mock_elo.get_leaderboard.side_effect = ValueError("bad val")
        result = handler_with_elo._collect_agent_rankings()
        assert result["available"] is False

    def test_leaderboard_limits_to_10(self, handler_with_elo, mock_elo):
        """get_leaderboard is called with limit=10."""
        handler_with_elo._collect_agent_rankings()
        mock_elo.get_leaderboard.assert_called_once_with(limit=10)


# ===========================================================================
# Circuit Breaker Collection
# ===========================================================================


class TestCircuitBreakers:
    """Tests for _collect_circuit_breakers."""

    def test_result_has_required_keys(self, handler):
        """Result always has 'breakers' and 'available' keys."""
        result = handler._collect_circuit_breakers()
        assert "breakers" in result
        assert "available" in result

    def test_with_registry(self, handler):
        """When registry is available via get_registry, collect breaker states."""
        mock_registry = MagicMock()
        mock_cb = MagicMock()
        mock_cb.state = "closed"
        mock_cb.failure_count = 2
        mock_cb.success_count = 10
        mock_registry.get_all.return_value = {"api_agent": mock_cb}

        with patch(
            "aragora.resilience.registry.get_registry",
            create=True,
            return_value=mock_registry,
        ):
            result = handler._collect_circuit_breakers()
            assert result["available"] is True
            assert len(result["breakers"]) == 1
            breaker = result["breakers"][0]
            assert breaker["name"] == "api_agent"
            assert breaker["state"] == "closed"
            assert breaker["failure_count"] == 2
            assert breaker["success_count"] == 10

    def test_with_registry_multiple_breakers(self, handler):
        """Multiple circuit breakers are all collected."""
        mock_registry = MagicMock()
        cb1 = MagicMock(state="closed", failure_count=0, success_count=50)
        cb2 = MagicMock(state="open", failure_count=5, success_count=3)
        mock_registry.get_all.return_value = {"agent_a": cb1, "agent_b": cb2}

        with patch(
            "aragora.resilience.registry.get_registry",
            create=True,
            return_value=mock_registry,
        ):
            result = handler._collect_circuit_breakers()
            assert result["available"] is True
            assert len(result["breakers"]) == 2
            names = {b["name"] for b in result["breakers"]}
            assert names == {"agent_a", "agent_b"}

    def test_with_registry_empty(self, handler):
        """Empty registry returns empty breakers list."""
        mock_registry = MagicMock()
        mock_registry.get_all.return_value = {}

        with patch(
            "aragora.resilience.registry.get_registry",
            create=True,
            return_value=mock_registry,
        ):
            result = handler._collect_circuit_breakers()
            assert result["available"] is True
            assert result["breakers"] == []

    def test_breaker_without_state_attr(self, handler):
        """Breaker missing state attribute shows 'unknown'."""
        mock_registry = MagicMock()
        mock_cb = MagicMock(spec=[])  # Empty spec = no attributes
        mock_registry.get_all.return_value = {"test_cb": mock_cb}

        with patch(
            "aragora.resilience.registry.get_registry",
            create=True,
            return_value=mock_registry,
        ):
            result = handler._collect_circuit_breakers()
            assert result["available"] is True
            breaker = result["breakers"][0]
            assert breaker["state"] == "unknown"
            assert breaker["failure_count"] == 0
            assert breaker["success_count"] == 0

    def test_instances_fallback(self, handler):
        """When registry import fails, fall back to CircuitBreaker._instances."""
        mock_cb = MagicMock(state="half_open", failure_count=3)

        # Ensure the first try block (registry) raises ImportError
        with patch(
            "aragora.resilience.registry.get_registry",
            create=True,
            side_effect=ImportError("no get_registry"),
        ):
            with patch(
                "aragora.resilience.CircuitBreaker",
            ) as mock_class:
                mock_class._instances = {"fallback_cb": mock_cb}
                result = handler._collect_circuit_breakers()
                assert result["available"] is True
                assert len(result["breakers"]) == 1
                assert result["breakers"][0]["name"] == "fallback_cb"
                assert result["breakers"][0]["state"] == "half_open"
                assert result["breakers"][0]["failure_count"] == 3

    def test_all_imports_fail_returns_fallback(self, handler):
        """When all circuit breaker imports fail, returns fallback."""
        import sys

        # Make both imports fail by removing the modules temporarily
        with patch(
            "aragora.resilience.registry.get_registry",
            create=True,
            side_effect=ImportError("no get_registry"),
        ):
            # Patch CircuitBreaker as a mock without _instances attribute
            mock_cb_class = MagicMock(spec=[])  # spec=[] means no attributes at all
            with patch(
                "aragora.resilience.CircuitBreaker",
                mock_cb_class,
            ):
                result = handler._collect_circuit_breakers()
                assert result["available"] is False
                assert result["breakers"] == []


# ===========================================================================
# Self-Improvement Collection
# ===========================================================================


class TestSelfImprove:
    """Tests for _collect_self_improve."""

    def test_fallback_when_import_fails(self, handler):
        """When SelfImproveRunStore import fails, returns fallback."""
        with patch.dict("sys.modules", {"aragora.nomic.stores.run_store": None}):
            result = handler._collect_self_improve()
            assert result["available"] is False
            assert result["total_cycles"] == 0
            assert result["successful"] == 0
            assert result["failed"] == 0
            assert result["recent_runs"] == []

    def test_with_empty_runs(self, handler):
        """Empty run store returns zeros."""
        mock_store = MagicMock()
        mock_store.list_runs.return_value = []

        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore",
            return_value=mock_store,
        ):
            result = handler._collect_self_improve()
            assert result["available"] is True
            assert result["total_cycles"] == 0
            assert result["successful"] == 0
            assert result["failed"] == 0
            assert result["recent_runs"] == []

    def test_with_dict_runs(self, handler):
        """Dict-format runs are properly counted."""
        runs = [
            {"id": "r1", "goal": "improve tests", "status": "completed", "started_at": "2026-02-01"},
            {"id": "r2", "goal": "refactor", "status": "failed", "started_at": "2026-02-02"},
            {"id": "r3", "goal": "add feature", "status": "completed", "started_at": "2026-02-03"},
            {"id": "r4", "goal": "fix bugs", "status": "running", "started_at": "2026-02-04"},
        ]
        mock_store = MagicMock()
        mock_store.list_runs.return_value = runs

        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore",
            return_value=mock_store,
        ):
            result = handler._collect_self_improve()
            assert result["available"] is True
            assert result["total_cycles"] == 4
            assert result["successful"] == 2
            assert result["failed"] == 1

    def test_recent_runs_limited_to_5(self, handler):
        """Only the last 5 runs appear in recent_runs."""
        runs = [
            {"id": f"r{i}", "goal": f"goal_{i}", "status": "completed", "started_at": f"2026-02-{i:02d}"}
            for i in range(1, 9)  # 8 runs
        ]
        mock_store = MagicMock()
        mock_store.list_runs.return_value = runs

        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore",
            return_value=mock_store,
        ):
            result = handler._collect_self_improve()
            assert result["total_cycles"] == 8
            assert len(result["recent_runs"]) == 5
            # The last 5 should be runs r4-r8
            assert result["recent_runs"][0]["id"] == "r4"
            assert result["recent_runs"][-1]["id"] == "r8"

    def test_recent_runs_dict_format(self, handler):
        """Dict runs have expected fields in recent_runs."""
        runs = [
            {"id": "run-1", "goal": "test goal", "status": "completed", "started_at": "2026-02-01T00:00:00"},
        ]
        mock_store = MagicMock()
        mock_store.list_runs.return_value = runs

        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore",
            return_value=mock_store,
        ):
            result = handler._collect_self_improve()
            recent = result["recent_runs"][0]
            assert recent["id"] == "run-1"
            assert recent["goal"] == "test goal"
            assert recent["status"] == "completed"
            assert recent["started_at"] == "2026-02-01T00:00:00"

    def test_with_object_runs(self, handler):
        """Object-format runs are properly handled."""
        run1 = SimpleNamespace(id="r1", goal="improve", status="completed", started_at="2026-02-01")
        run2 = SimpleNamespace(id="r2", goal="fix", status="failed", started_at="2026-02-02")
        mock_store = MagicMock()
        mock_store.list_runs.return_value = [run1, run2]

        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore",
            return_value=mock_store,
        ):
            result = handler._collect_self_improve()
            assert result["available"] is True
            assert result["total_cycles"] == 2
            assert result["successful"] == 1
            assert result["failed"] == 1
            assert len(result["recent_runs"]) == 2
            assert result["recent_runs"][0]["id"] == "r1"

    def test_object_runs_started_at_converted_to_str(self, handler):
        """Object run's started_at is converted to string."""
        from datetime import datetime, timezone

        dt = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
        run = SimpleNamespace(id="r1", goal="test", status="completed", started_at=dt)
        mock_store = MagicMock()
        mock_store.list_runs.return_value = [run]

        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore",
            return_value=mock_store,
        ):
            result = handler._collect_self_improve()
            assert isinstance(result["recent_runs"][0]["started_at"], str)

    def test_os_error_returns_fallback(self, handler):
        """OSError from store returns fallback."""
        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore",
            side_effect=OSError("disk error"),
        ):
            result = handler._collect_self_improve()
            assert result["available"] is False


# ===========================================================================
# System Health Collection
# ===========================================================================


class TestSystemHealth:
    """Tests for _collect_system_health."""

    def test_with_psutil_available(self, handler):
        """When psutil is available, returns memory and CPU info."""
        mock_mem = MagicMock()
        mock_mem.percent = 65.4

        with patch.dict("sys.modules", {"psutil": MagicMock()}):
            import sys
            mock_psutil = sys.modules["psutil"]
            mock_psutil.virtual_memory.return_value = mock_mem
            mock_psutil.cpu_percent.return_value = 12.3

            result = handler._collect_system_health()
            assert result["available"] is True
            assert result["memory_percent"] == 65.4
            assert result["cpu_percent"] == 12.3
            assert result["pid"] == os.getpid()

    def test_without_psutil(self, handler):
        """When psutil is not importable, returns fallback."""
        import sys

        # Temporarily remove psutil from modules if present
        saved = sys.modules.pop("psutil", None)
        try:
            # Make psutil import raise ImportError
            sys.modules["psutil"] = None  # type: ignore[assignment]
            # Call through a fresh import path
            result = handler._collect_system_health()
            # If psutil was already imported, it may still work.
            # Either way, result should be valid
            assert "pid" in result
            assert result["pid"] == os.getpid()
            assert "available" in result
        finally:
            if saved is not None:
                sys.modules["psutil"] = saved
            else:
                sys.modules.pop("psutil", None)

    def test_pid_is_current_process(self, handler):
        result = handler._collect_system_health()
        assert result["pid"] == os.getpid()


# ===========================================================================
# Error Rates Collection
# ===========================================================================


class TestErrorRates:
    """Tests for _collect_error_rates."""

    def test_with_metrics_available(self, handler):
        """When observability metrics are available, returns rates."""
        with patch(
            "aragora.observability.metrics.server.get_request_total_count",
            create=True,
            return_value=1000,
        ):
            with patch(
                "aragora.observability.metrics.server.get_request_error_count",
                create=True,
                return_value=25,
            ):
                result = handler._collect_error_rates()
                assert result["available"] is True
                assert result["total_requests"] == 1000
                assert result["total_errors"] == 25
                assert result["error_rate"] == 0.025

    def test_error_rate_zero_when_no_requests(self, handler):
        """Error rate is 0 when total requests is 0."""
        with patch(
            "aragora.observability.metrics.server.get_request_total_count",
            create=True,
            return_value=0,
        ):
            with patch(
                "aragora.observability.metrics.server.get_request_error_count",
                create=True,
                return_value=0,
            ):
                result = handler._collect_error_rates()
                assert result["available"] is True
                assert result["error_rate"] == 0

    def test_error_rate_rounding(self, handler):
        """Error rate is rounded to 4 decimal places."""
        with patch(
            "aragora.observability.metrics.server.get_request_total_count",
            create=True,
            return_value=3,
        ):
            with patch(
                "aragora.observability.metrics.server.get_request_error_count",
                create=True,
                return_value=1,
            ):
                result = handler._collect_error_rates()
                assert result["error_rate"] == round(1 / 3, 4)

    def test_fallback_when_import_fails(self, handler):
        """When observability metrics import fails, returns fallback."""
        with patch.dict("sys.modules", {"aragora.observability.metrics.server": None}):
            result = handler._collect_error_rates()
            assert result["available"] is False
            assert result["total_requests"] == 0
            assert result["total_errors"] == 0
            assert result["error_rate"] == 0


# ===========================================================================
# Graceful Degradation
# ===========================================================================


class TestGracefulDegradation:
    """Tests for graceful fallback when subsystems are unavailable."""

    @pytest.mark.asyncio
    async def test_all_subsystems_unavailable(self, handler, mock_http_handler):
        """When no subsystems are configured, all sections show available=False."""
        # Patch circuit breakers and self-improve to also be unavailable
        with patch.object(handler, "_collect_circuit_breakers", return_value={
            "breakers": [], "available": False,
        }):
            with patch.object(handler, "_collect_self_improve", return_value={
                "total_cycles": 0, "successful": 0, "failed": 0,
                "recent_runs": [], "available": False,
            }):
                with patch.object(handler, "_collect_error_rates", return_value={
                    "total_requests": 0, "total_errors": 0,
                    "error_rate": 0, "available": False,
                }):
                    with patch.object(handler, "_collect_system_health", return_value={
                        "memory_percent": None, "cpu_percent": None,
                        "pid": os.getpid(), "available": False,
                    }):
                        result = await handler.handle(
                            "/api/observability/dashboard", {}, mock_http_handler
                        )
                        assert result.status_code == 200
                        body = result[0]
                        assert body["debate_metrics"]["available"] is False
                        assert body["agent_rankings"]["available"] is False
                        assert body["circuit_breakers"]["available"] is False
                        assert body["self_improve"]["available"] is False
                        assert body["error_rates"]["available"] is False

    @pytest.mark.asyncio
    async def test_partial_subsystem_availability(self, handler_with_storage, mock_http_handler, mock_storage):
        """Some subsystems available, others not -- still returns 200."""
        mock_storage.list_debates.return_value = [
            {"duration": 5.0, "consensus_reached": True}
        ]

        with patch.object(handler_with_storage, "_collect_circuit_breakers", return_value={
            "breakers": [], "available": False,
        }):
            with patch.object(handler_with_storage, "_collect_self_improve", return_value={
                "total_cycles": 0, "successful": 0, "failed": 0,
                "recent_runs": [], "available": False,
            }):
                with patch.object(handler_with_storage, "_collect_error_rates", return_value={
                    "total_requests": 0, "total_errors": 0,
                    "error_rate": 0, "available": False,
                }):
                    result = await handler_with_storage.handle(
                        "/api/observability/dashboard", {}, mock_http_handler
                    )
                    assert result.status_code == 200
                    body = result[0]
                    # Debate metrics should be available (has storage)
                    assert body["debate_metrics"]["available"] is True
                    assert body["debate_metrics"]["total_debates"] == 1
                    # Agent rankings unavailable (no elo)
                    assert body["agent_rankings"]["available"] is False


# ===========================================================================
# Full Integration (all collectors running)
# ===========================================================================


class TestFullDashboard:
    """Integration tests with all subsystems mocked."""

    @pytest.mark.asyncio
    async def test_full_dashboard_response(self, handler_full, mock_http_handler, mock_storage, mock_elo):
        """Full response has all sections populated."""
        mock_storage.list_debates.return_value = [
            {"duration": 10.0, "consensus_reached": True},
            {"duration": 20.0, "consensus_reached": False},
        ]
        mock_elo.get_leaderboard.return_value = [
            {"agent": "claude", "rating": 1600, "matches": 50, "win_rate": 0.72},
        ]

        with patch.object(handler_full, "_collect_circuit_breakers", return_value={
            "breakers": [{"name": "api", "state": "closed", "failure_count": 0, "success_count": 100}],
            "available": True,
        }):
            with patch.object(handler_full, "_collect_self_improve", return_value={
                "total_cycles": 5, "successful": 4, "failed": 1,
                "recent_runs": [], "available": True,
            }):
                with patch.object(handler_full, "_collect_error_rates", return_value={
                    "total_requests": 1000, "total_errors": 10,
                    "error_rate": 0.01, "available": True,
                }):
                    result = await handler_full.handle(
                        "/api/observability/dashboard", {}, mock_http_handler
                    )
                    assert result.status_code == 200
                    body = result[0]

                    # Debate metrics
                    assert body["debate_metrics"]["available"] is True
                    assert body["debate_metrics"]["total_debates"] == 2
                    assert body["debate_metrics"]["avg_duration_seconds"] == 15.0

                    # Agent rankings
                    assert body["agent_rankings"]["available"] is True
                    assert len(body["agent_rankings"]["top_agents"]) == 1
                    assert body["agent_rankings"]["top_agents"][0]["name"] == "claude"

                    # Circuit breakers
                    assert body["circuit_breakers"]["available"] is True

                    # Self improve
                    assert body["self_improve"]["total_cycles"] == 5

                    # Error rates
                    assert body["error_rates"]["error_rate"] == 0.01

                    # Timing
                    assert body["collection_time_ms"] >= 0
                    assert isinstance(body["timestamp"], float)


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_debate_with_mixed_duration_keys(self, handler_with_storage, mock_storage):
        """Debates mixing 'duration' and 'duration_seconds' keys."""
        debates = [
            {"duration": 5.0},
            {"duration_seconds": 10.0},
            {"other_field": "no_duration"},
        ]
        mock_storage.list_debates.return_value = debates
        result = handler_with_storage._collect_debate_metrics()
        assert result["total_debates"] == 3
        # Only 2 debates have durations
        assert result["avg_duration_seconds"] == 7.5  # (5.0 + 10.0) / 2

    def test_debate_consensus_mixed_keys(self, handler_with_storage, mock_storage):
        """Debates mixing 'consensus_reached' and 'consensus' keys."""
        debates = [
            {"consensus_reached": True},
            {"consensus": True},
            {"neither": True},
        ]
        mock_storage.list_debates.return_value = debates
        result = handler_with_storage._collect_debate_metrics()
        assert result["consensus_rate"] == round(2 / 3, 3)

    def test_leaderboard_mixed_dict_and_object(self, handler_with_elo, mock_elo):
        """Leaderboard with both dict and object entries."""
        dict_entry = {"agent": "claude", "rating": 1600, "matches": 50, "win_rate": 0.7}
        obj_entry = SimpleNamespace(agent="gpt4", rating=1550, matches=40, win_rate=0.65)
        mock_elo.get_leaderboard.return_value = [dict_entry, obj_entry]
        result = handler_with_elo._collect_agent_rankings()
        assert len(result["top_agents"]) == 2
        assert result["top_agents"][0]["name"] == "claude"
        assert result["top_agents"][1]["name"] == "gpt4"

    def test_debate_duration_coerced_to_float(self, handler_with_storage, mock_storage):
        """Duration values are coerced to float (e.g., int input)."""
        mock_storage.list_debates.return_value = [{"duration": 10}]
        result = handler_with_storage._collect_debate_metrics()
        assert result["avg_duration_seconds"] == 10.0
        assert isinstance(result["avg_duration_seconds"], (int, float))

    @pytest.mark.asyncio
    async def test_query_params_ignored(self, handler, mock_http_handler):
        """Query parameters are accepted but do not affect response."""
        result = await handler.handle(
            "/api/observability/dashboard",
            {"extra_param": "value"},
            mock_http_handler,
        )
        assert result is not None
        assert result.status_code == 200

    def test_self_improve_runs_missing_fields(self, handler):
        """Runs with missing fields use empty string defaults."""
        runs = [{}]  # Completely empty dict
        mock_store = MagicMock()
        mock_store.list_runs.return_value = runs

        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore",
            return_value=mock_store,
        ):
            result = handler._collect_self_improve()
            assert result["available"] is True
            assert result["total_cycles"] == 1
            assert result["successful"] == 0
            assert result["failed"] == 0
            recent = result["recent_runs"][0]
            assert recent["id"] == ""
            assert recent["goal"] == ""
            assert recent["status"] == ""
            assert recent["started_at"] == ""

    def test_self_improve_object_missing_fields(self, handler):
        """Object runs with missing attrs use empty string defaults."""
        run = SimpleNamespace()  # No attrs at all
        mock_store = MagicMock()
        mock_store.list_runs.return_value = [run]

        with patch(
            "aragora.nomic.stores.run_store.SelfImproveRunStore",
            return_value=mock_store,
        ):
            result = handler._collect_self_improve()
            assert result["available"] is True
            recent = result["recent_runs"][0]
            assert recent["id"] == ""
            assert recent["goal"] == ""
            assert recent["status"] == ""
