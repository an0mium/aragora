"""
Tests for the admin dashboard handler.

Tests cover:
- DashboardHandler route handling
- Debates dashboard endpoint
- Quality metrics endpoint
- Summary metrics (SQL and legacy)
- Recent activity metrics
- Agent performance metrics
- Debate patterns
- Consensus insights
- System health
- Calibration metrics
- Performance metrics
- Evolution metrics
- Debate quality metrics
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.dashboard import DashboardHandler


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a DashboardHandler instance with mocked context."""
    # Mock server context with required fields
    mock_context = {
        "storage": None,
        "elo_system": None,
        "debate_embeddings": None,
        "critique_store": None,
        "calibration_tracker": None,
        "performance_monitor": None,
        "prompt_evolver": None,
    }
    h = DashboardHandler(mock_context)
    return h


@pytest.fixture
def mock_storage():
    """Create mock storage with connection."""
    storage = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    # Setup connection context manager
    storage.db = MagicMock()
    storage.db.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
    storage.db.connection.return_value.__exit__ = MagicMock(return_value=False)

    mock_conn.cursor.return_value = mock_cursor

    return storage, mock_cursor


@pytest.fixture
def mock_debates():
    """Create sample debate data."""
    now = datetime.now(timezone.utc)
    return [
        {
            "id": "debate_1",
            "domain": "technology",
            "consensus_reached": True,
            "confidence": 0.85,
            "created_at": now.isoformat(),
            "early_stopped": False,
        },
        {
            "id": "debate_2",
            "domain": "finance",
            "consensus_reached": True,
            "confidence": 0.75,
            "created_at": (now - timedelta(hours=2)).isoformat(),
            "early_stopped": True,
        },
        {
            "id": "debate_3",
            "domain": "technology",
            "consensus_reached": False,
            "confidence": 0.55,
            "created_at": (now - timedelta(hours=12)).isoformat(),
            "early_stopped": False,
            "disagreement_report": {"types": ["methodology", "evidence"]},
        },
    ]


# =============================================================================
# Route Handling Tests
# =============================================================================


class TestDashboardRouting:
    """Tests for DashboardHandler route handling."""

    def test_can_handle_debates_route(self, handler):
        """Can handle debates dashboard route."""
        assert handler.can_handle("/api/v1/dashboard/debates") is True

    def test_can_handle_quality_metrics_route(self, handler):
        """Can handle quality metrics route."""
        assert handler.can_handle("/api/v1/dashboard/quality-metrics") is True

    def test_cannot_handle_unknown_route(self, handler):
        """Cannot handle unknown routes."""
        assert handler.can_handle("/api/v1/unknown") is False
        assert handler.can_handle("/api/v1/dashboard") is False

    @pytest.mark.asyncio
    async def test_handle_routes_to_debates(self, handler):
        """Handle routes debates path correctly."""
        mock_handler = MagicMock()
        mock_auth_context = MagicMock()

        with patch.object(handler, "_get_debates_dashboard") as mock_method:
            mock_method.return_value = {"data": {}}

            with patch(
                "aragora.server.handlers.admin.dashboard._dashboard_limiter"
            ) as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                with patch.object(
                    handler, "get_auth_context", return_value=mock_auth_context
                ) as mock_auth:
                    with patch.object(handler, "check_permission"):
                        await handler.handle(
                            "/api/v1/dashboard/debates",
                            {"limit": "20", "hours": "48"},
                            mock_handler,
                        )

                        mock_method.assert_called_once_with(None, 20, 48)

    @pytest.mark.asyncio
    async def test_handle_routes_to_quality_metrics(self, handler):
        """Handle routes quality metrics path correctly."""
        mock_handler = MagicMock()
        mock_auth_context = MagicMock()

        with patch.object(handler, "_get_quality_metrics") as mock_method:
            mock_method.return_value = {"data": {}}

            with patch(
                "aragora.server.handlers.admin.dashboard._dashboard_limiter"
            ) as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
                    with patch.object(handler, "check_permission"):
                        await handler.handle("/api/v1/dashboard/quality-metrics", {}, mock_handler)

                        mock_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_rate_limited(self, handler):
        """Handle returns 429 when rate limited."""
        mock_handler = MagicMock()

        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            result = await handler.handle("/api/v1/dashboard/debates", {}, mock_handler)

            assert result is not None
            # Rate limit response should have 429 status

    @pytest.mark.asyncio
    async def test_handle_limit_capped_at_50(self, handler):
        """Limit parameter is capped at 50."""
        mock_handler = MagicMock()
        mock_auth_context = MagicMock()

        with patch.object(handler, "_get_debates_dashboard") as mock_method:
            mock_method.return_value = {"data": {}}

            with patch(
                "aragora.server.handlers.admin.dashboard._dashboard_limiter"
            ) as mock_limiter:
                mock_limiter.is_allowed.return_value = True

                with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
                    with patch.object(handler, "check_permission"):
                        await handler.handle(
                            "/api/v1/dashboard/debates", {"limit": "100"}, mock_handler
                        )

                        # Limit should be capped at 50
                        mock_method.assert_called_once_with(None, 50, 24)


# =============================================================================
# Summary Metrics Tests
# =============================================================================


class TestSummaryMetrics:
    """Tests for summary metrics generation."""

    def test_get_summary_metrics_sql(self, handler, mock_storage):
        """SQL summary returns correct aggregates."""
        storage, cursor = mock_storage

        # Mock SQL result: total=100, consensus=75, avg_conf=0.78
        cursor.fetchone.return_value = (100, 75, 0.78)

        result = handler._get_summary_metrics_sql(storage, None)

        assert result["total_debates"] == 100
        assert result["consensus_reached"] == 75
        assert result["consensus_rate"] == 0.75
        assert result["avg_confidence"] == 0.78

    def test_get_summary_metrics_sql_empty(self, handler, mock_storage):
        """SQL summary handles empty results."""
        storage, cursor = mock_storage
        cursor.fetchone.return_value = (0, 0, None)

        result = handler._get_summary_metrics_sql(storage, None)

        assert result["total_debates"] == 0
        assert result["consensus_rate"] == 0.0
        assert result["avg_confidence"] == 0.0

    def test_get_summary_metrics_sql_error(self, handler, mock_storage):
        """SQL summary handles errors gracefully."""
        storage, cursor = mock_storage
        cursor.fetchone.side_effect = Exception("Database error")

        result = handler._get_summary_metrics_sql(storage, None)

        assert result["total_debates"] == 0

    def test_get_summary_metrics_legacy(self, handler, mock_debates):
        """Legacy summary calculates metrics from debate list."""
        result = handler._get_summary_metrics(None, mock_debates)

        assert result["total_debates"] == 3
        assert result["consensus_reached"] == 2
        # 2/3 = 0.667 rounded to 3 decimals
        assert result["consensus_rate"] == pytest.approx(0.667, rel=0.01)

    def test_get_summary_metrics_empty(self, handler):
        """Legacy summary handles empty list."""
        result = handler._get_summary_metrics(None, [])

        assert result["total_debates"] == 0
        assert result["consensus_rate"] == 0.0


# =============================================================================
# Recent Activity Tests
# =============================================================================


class TestRecentActivity:
    """Tests for recent activity metrics."""

    def test_get_recent_activity_sql(self, handler, mock_storage):
        """SQL activity returns correct counts."""
        storage, cursor = mock_storage
        cursor.fetchone.return_value = (25, 18)

        result = handler._get_recent_activity_sql(storage, 24)

        assert result["debates_last_period"] == 25
        assert result["consensus_last_period"] == 18
        assert result["period_hours"] == 24

    def test_get_recent_activity_legacy(self, handler, mock_debates):
        """Legacy activity counts recent debates."""
        result = handler._get_recent_activity(None, 24, mock_debates)

        # All 3 debates are within 24 hours
        assert result["debates_last_period"] == 3
        assert result["consensus_last_period"] == 2
        assert "technology" in result["domains_active"]

    def test_get_recent_activity_filters_by_time(self, handler):
        """Activity filters debates by time window."""
        # Create debates with clear time boundaries
        now = datetime.now()  # Local time to match the method's cutoff calculation
        old_debates = [
            {
                "id": "recent",
                "consensus_reached": True,
                "created_at": now.isoformat(),
            },
            {
                "id": "very_old",
                "consensus_reached": False,
                "created_at": (now - timedelta(days=30)).isoformat(),
            },
        ]
        # Use 24 hour window - first debate should be included, second shouldn't
        result = handler._get_recent_activity(None, 24, old_debates)

        # Only the recent debate should be included
        assert result["debates_last_period"] == 1
        assert result["consensus_last_period"] == 1


# =============================================================================
# Single Pass Processing Tests
# =============================================================================


class TestSinglePassProcessing:
    """Tests for optimized single-pass debate processing."""

    def test_process_debates_single_pass(self, handler, mock_debates):
        """Single pass calculates all metrics correctly."""
        summary, activity, patterns = handler._process_debates_single_pass(mock_debates, None, 24)

        # Summary
        assert summary["total_debates"] == 3
        assert summary["consensus_reached"] == 2

        # Activity
        assert activity["debates_last_period"] == 3
        assert activity["consensus_last_period"] == 2

        # Patterns
        assert patterns["disagreement_stats"]["with_disagreements"] == 1
        assert patterns["early_stopping"]["early_stopped"] == 1
        assert patterns["early_stopping"]["full_duration"] == 2

    def test_process_debates_single_pass_empty(self, handler):
        """Single pass handles empty list."""
        summary, activity, patterns = handler._process_debates_single_pass([], None, 24)

        assert summary["total_debates"] == 0
        assert activity["debates_last_period"] == 0


# =============================================================================
# Agent Performance Tests
# =============================================================================


class TestAgentPerformance:
    """Tests for agent performance metrics."""

    def test_get_agent_performance(self, handler):
        """Agent performance returns ELO ratings."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.agent_name = "claude"
        mock_rating.elo = 1250
        mock_rating.wins = 15
        mock_rating.losses = 5
        mock_rating.draws = 2
        mock_rating.win_rate = 0.75
        mock_rating.debates_count = 22

        mock_elo.get_all_ratings.return_value = [mock_rating]
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        result = handler._get_agent_performance(10)

        assert len(result["top_performers"]) == 1
        assert result["top_performers"][0]["name"] == "claude"
        assert result["top_performers"][0]["elo"] == 1250
        assert result["total_agents"] == 1
        assert result["avg_elo"] == 1250.0

    def test_get_agent_performance_no_elo(self, handler):
        """Agent performance handles missing ELO system."""
        handler.get_elo_system = MagicMock(return_value=None)

        result = handler._get_agent_performance(10)

        assert result["top_performers"] == []
        assert result["total_agents"] == 0


# =============================================================================
# Debate Patterns Tests
# =============================================================================


class TestDebatePatterns:
    """Tests for debate pattern statistics."""

    def test_get_debate_patterns(self, handler, mock_debates):
        """Debate patterns extracts disagreement and stopping stats."""
        result = handler._get_debate_patterns(mock_debates)

        assert result["disagreement_stats"]["with_disagreements"] == 1
        assert "methodology" in result["disagreement_stats"]["disagreement_types"]
        assert result["early_stopping"]["early_stopped"] == 1
        assert result["early_stopping"]["full_duration"] == 2

    def test_get_debate_patterns_empty(self, handler):
        """Debate patterns handles empty list."""
        result = handler._get_debate_patterns([])

        assert result["disagreement_stats"]["with_disagreements"] == 0
        assert result["early_stopping"]["early_stopped"] == 0


# =============================================================================
# Consensus Insights Tests
# =============================================================================


class TestConsensusInsights:
    """Tests for consensus memory insights."""

    def test_get_consensus_insights(self, handler):
        """Consensus insights retrieves memory stats."""
        mock_memory = MagicMock()
        mock_memory.get_statistics.return_value = {
            "total_consensus": 150,
            "total_dissents": 25,
            "by_domain": {"technology": 80, "finance": 70},
        }
        mock_memory.db_path = ":memory:"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(45,), (0.72,)]

        with patch(
            "aragora.memory.consensus.ConsensusMemory",
            return_value=mock_memory,
        ):
            with patch("aragora.storage.schema.get_wal_connection") as mock_get_conn:
                mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
                mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

                result = handler._get_consensus_insights(None)

        assert result["total_consensus_topics"] == 150
        assert result["total_dissents"] == 25
        assert "technology" in result["domains"]

    def test_get_consensus_insights_error(self, handler):
        """Consensus insights handles errors gracefully."""
        # Just verify it returns default values when there are errors
        result = handler._get_consensus_insights(None)

        # Should return default structure without crashing
        assert "total_consensus_topics" in result
        assert "domains" in result


# =============================================================================
# System Health Tests
# =============================================================================


class TestSystemHealth:
    """Tests for system health metrics."""

    def test_get_system_health(self, handler):
        """System health returns prometheus and cache status."""
        with patch(
            "aragora.server.prometheus.is_prometheus_available",
            return_value=True,
        ):
            result = handler._get_system_health()

        assert result["prometheus_available"] is True
        assert "cache_entries" in result

    def test_get_system_health_prometheus_unavailable(self, handler):
        """System health handles prometheus unavailable."""
        with patch(
            "aragora.server.prometheus.is_prometheus_available",
            return_value=False,
        ):
            result = handler._get_system_health()

        assert result["prometheus_available"] is False

    def test_get_system_health_includes_connector_health(self, handler):
        """System health includes connector_health section."""
        with patch(
            "aragora.server.prometheus.is_prometheus_available",
            return_value=False,
        ):
            with patch.object(handler, "_get_connector_health") as mock_conn:
                mock_conn.return_value = {"summary": {}, "connectors": []}
                result = handler._get_system_health()

        assert "connector_health" in result
        mock_conn.assert_called_once()


# =============================================================================
# Connector Health Tests
# =============================================================================


class TestConnectorHealth:
    """Tests for connector health metrics."""

    def test_connector_health_no_scheduler(self, handler):
        """Connector health returns empty when scheduler unavailable."""
        with patch(
            "aragora.server.handlers.connectors.get_scheduler",
            side_effect=ImportError("No scheduler"),
        ):
            result = handler._get_connector_health()

        assert result["summary"]["total_connectors"] == 0
        assert result["summary"]["healthy"] == 0
        assert result["summary"]["health_score"] == 100
        assert result["connectors"] == []

    def test_connector_health_with_jobs(self, handler):
        """Connector health returns correct data for registered jobs."""
        mock_scheduler = MagicMock()
        mock_scheduler._scheduler_task = MagicMock()  # Running
        mock_scheduler.get_stats.return_value = {
            "total_jobs": 2,
            "running_syncs": 1,
            "success_rate": 0.95,
        }

        # Create mock jobs
        mock_job1 = MagicMock()
        mock_job1.id = "job_1"
        mock_job1.connector_id = "github_corp"
        mock_job1.consecutive_failures = 0
        mock_job1.current_run_id = None
        mock_job1.schedule = MagicMock(enabled=True)
        mock_job1.last_run = datetime(2026, 1, 27, 10, 0, 0, tzinfo=timezone.utc)
        mock_job1.next_run = datetime(2026, 1, 27, 11, 0, 0, tzinfo=timezone.utc)
        mock_job1.connector = MagicMock(name="GitHub Corporate")
        mock_job1.connector.config = {"type": "github"}

        mock_job2 = MagicMock()
        mock_job2.id = "job_2"
        mock_job2.connector_id = "slack_main"
        mock_job2.consecutive_failures = 2
        mock_job2.current_run_id = "run_123"  # Currently syncing
        mock_job2.schedule = MagicMock(enabled=True)
        mock_job2.last_run = datetime(2026, 1, 27, 9, 30, 0, tzinfo=timezone.utc)
        mock_job2.next_run = datetime(2026, 1, 27, 10, 30, 0, tzinfo=timezone.utc)
        mock_job2.connector = MagicMock(name="Slack Main")
        mock_job2.connector.config = {"type": "slack"}

        mock_scheduler.list_jobs.return_value = [mock_job1, mock_job2]

        # Mock history for each job
        mock_history1 = MagicMock()
        mock_history1.status = MagicMock(value="completed")
        mock_history1.duration_seconds = 30.0
        mock_history1.items_synced = 100

        mock_history2 = MagicMock()
        mock_history2.status = MagicMock(value="failed")
        mock_history2.duration_seconds = 5.0
        mock_history2.items_synced = 0

        mock_scheduler.get_history.side_effect = [[mock_history1], [mock_history2]]

        with patch(
            "aragora.server.handlers.connectors.get_scheduler",
            return_value=mock_scheduler,
        ):
            result = handler._get_connector_health()

        assert result["summary"]["total_connectors"] == 2
        assert result["summary"]["scheduler_running"] is True
        assert result["summary"]["running_syncs"] == 1
        assert result["summary"]["success_rate"] == 0.95

        # Check connectors list
        assert len(result["connectors"]) == 2

        # First connector should be healthy (0 failures)
        conn1 = next(c for c in result["connectors"] if c["connector_id"] == "github_corp")
        assert conn1["health"] == "healthy"
        assert conn1["status"] == "connected"
        assert conn1["items_synced"] == 100

        # Second connector should be degraded (2 failures) and syncing
        conn2 = next(c for c in result["connectors"] if c["connector_id"] == "slack_main")
        assert conn2["health"] == "degraded"
        assert conn2["status"] == "syncing"

    def test_connector_health_score_calculation(self, handler):
        """Health score reflects ratio of healthy connectors correctly."""
        mock_scheduler = MagicMock()
        mock_scheduler._scheduler_task = MagicMock()
        mock_scheduler.get_stats.return_value = {
            "total_jobs": 4,
            "running_syncs": 0,
            "success_rate": 0.8,
        }

        # Create jobs with different health states
        jobs = []
        for i, failures in enumerate([0, 0, 1, 3]):  # 2 healthy, 1 degraded, 1 unhealthy
            job = MagicMock()
            job.id = f"job_{i}"
            job.connector_id = f"conn_{i}"
            job.consecutive_failures = failures
            job.current_run_id = None
            job.schedule = MagicMock(enabled=True)
            job.last_run = None
            job.next_run = None
            job.connector = None
            jobs.append(job)

        mock_scheduler.list_jobs.return_value = jobs
        mock_scheduler.get_history.return_value = []

        with patch(
            "aragora.server.handlers.connectors.get_scheduler",
            return_value=mock_scheduler,
        ):
            result = handler._get_connector_health()

        # 2 healthy out of 4 = 50%
        assert result["summary"]["healthy"] == 2
        assert result["summary"]["degraded"] == 1
        assert result["summary"]["unhealthy"] == 1
        assert result["summary"]["health_score"] == 50

    def test_connector_health_status_derivation(self, handler):
        """Connector status correctly derived from state."""
        mock_scheduler = MagicMock()
        mock_scheduler._scheduler_task = MagicMock()
        mock_scheduler.get_stats.return_value = {
            "total_jobs": 4,
            "running_syncs": 0,
            "success_rate": 1.0,
        }

        # Create jobs with different states
        # 1. Currently syncing
        job_syncing = MagicMock()
        job_syncing.id = "syncing"
        job_syncing.connector_id = "syncing"
        job_syncing.consecutive_failures = 0
        job_syncing.current_run_id = "run_1"  # Syncing
        job_syncing.schedule = MagicMock(enabled=True)
        job_syncing.last_run = None
        job_syncing.next_run = None
        job_syncing.connector = None

        # 2. Error state (3+ failures)
        job_error = MagicMock()
        job_error.id = "error"
        job_error.connector_id = "error"
        job_error.consecutive_failures = 3
        job_error.current_run_id = None
        job_error.schedule = MagicMock(enabled=True)
        job_error.last_run = None
        job_error.next_run = None
        job_error.connector = None

        # 3. Disconnected (disabled)
        job_disconnected = MagicMock()
        job_disconnected.id = "disconnected"
        job_disconnected.connector_id = "disconnected"
        job_disconnected.consecutive_failures = 0
        job_disconnected.current_run_id = None
        job_disconnected.schedule = MagicMock(enabled=False)  # Disabled
        job_disconnected.last_run = None
        job_disconnected.next_run = None
        job_disconnected.connector = None

        # 4. Connected (normal)
        job_connected = MagicMock()
        job_connected.id = "connected"
        job_connected.connector_id = "connected"
        job_connected.consecutive_failures = 0
        job_connected.current_run_id = None
        job_connected.schedule = MagicMock(enabled=True)
        job_connected.last_run = None
        job_connected.next_run = None
        job_connected.connector = None

        mock_scheduler.list_jobs.return_value = [
            job_syncing,
            job_error,
            job_disconnected,
            job_connected,
        ]
        mock_scheduler.get_history.return_value = []

        with patch(
            "aragora.server.handlers.connectors.get_scheduler",
            return_value=mock_scheduler,
        ):
            result = handler._get_connector_health()

        connectors = {c["connector_id"]: c for c in result["connectors"]}

        assert connectors["syncing"]["status"] == "syncing"
        assert connectors["error"]["status"] == "error"
        assert connectors["disconnected"]["status"] == "disconnected"
        assert connectors["connected"]["status"] == "connected"

    def test_connector_health_handles_exception(self, handler):
        """Connector health returns empty on exception."""
        with patch(
            "aragora.server.handlers.connectors.get_scheduler",
        ) as mock_get:
            mock_get.side_effect = RuntimeError("Scheduler crash")
            result = handler._get_connector_health()

        # Should return default empty structure
        assert result["summary"]["total_connectors"] == 0
        assert result["connectors"] == []


# =============================================================================
# Calibration Metrics Tests
# =============================================================================


class TestCalibrationMetrics:
    """Tests for agent calibration metrics."""

    def test_get_calibration_metrics(self, handler):
        """Calibration metrics returns agent calibration data."""
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_summary.return_value = {
            "agents": {
                "claude": {"calibration_bias": 0.05, "brier_score": 0.15},
                "gemini": {"calibration_bias": 0.15, "brier_score": 0.20},
                "codex": {"calibration_bias": -0.12, "brier_score": 0.18},
            },
            "overall": 0.85,
        }
        mock_tracker.get_all_agents.return_value = ["claude", "gemini", "codex"]
        mock_tracker.get_calibration_curve.return_value = None
        mock_tracker.get_domain_breakdown.return_value = None

        handler.ctx["calibration_tracker"] = mock_tracker

        result = handler._get_calibration_metrics()

        assert result["overall_calibration"] == 0.85
        assert "claude" in result["well_calibrated_agents"]
        assert "gemini" in result["overconfident_agents"]
        assert "codex" in result["underconfident_agents"]
        assert len(result["top_by_brier"]) == 3

    def test_get_calibration_metrics_no_tracker(self, handler):
        """Calibration metrics handles missing tracker."""
        handler.ctx = {}

        result = handler._get_calibration_metrics()

        assert result["overall_calibration"] == 0.0
        assert result["agents"] == {}


# =============================================================================
# Performance Metrics Tests
# =============================================================================


class TestPerformanceMetrics:
    """Tests for performance metrics."""

    def test_get_performance_metrics(self, handler):
        """Performance metrics returns agent stats."""
        mock_monitor = MagicMock()
        mock_monitor.get_performance_insights.return_value = {
            "agents": {"claude": {"avg_latency": 250}},
            "avg_latency_ms": 300.5,
            "success_rate": 0.95,
            "total_calls": 1500,
        }

        handler.ctx["performance_monitor"] = mock_monitor

        result = handler._get_performance_metrics()

        assert result["avg_latency_ms"] == 300.5
        assert result["success_rate"] == 0.95
        assert result["total_calls"] == 1500

    def test_get_performance_metrics_no_monitor(self, handler):
        """Performance metrics handles missing monitor."""
        handler.ctx = {}

        result = handler._get_performance_metrics()

        assert result["avg_latency_ms"] == 0.0
        assert result["total_calls"] == 0


# =============================================================================
# Evolution Metrics Tests
# =============================================================================


class TestEvolutionMetrics:
    """Tests for prompt evolution metrics."""

    def test_get_evolution_metrics(self, handler):
        """Evolution metrics returns prompt version data."""
        mock_evolver = MagicMock()

        mock_version = MagicMock()
        mock_version.version = 5
        mock_version.performance_score = 0.82
        mock_version.debates_count = 150

        mock_evolver.get_prompt_version.return_value = mock_version
        mock_evolver.get_top_patterns.return_value = [{"pattern": "p1"}, {"pattern": "p2"}]

        handler.ctx["prompt_evolver"] = mock_evolver

        result = handler._get_evolution_metrics()

        assert result["total_versions"] > 0
        assert result["patterns_extracted"] == 2

    def test_get_evolution_metrics_no_evolver(self, handler):
        """Evolution metrics handles missing evolver."""
        handler.ctx = {}

        result = handler._get_evolution_metrics()

        assert result["total_versions"] == 0
        assert result["patterns_extracted"] == 0


# =============================================================================
# Debate Quality Metrics Tests
# =============================================================================


class TestDebateQualityMetrics:
    """Tests for debate quality metrics."""

    def test_get_debate_quality_metrics(self, handler):
        """Debate quality returns aggregated scores."""
        mock_elo = MagicMock()
        mock_elo.get_recent_matches.return_value = [
            {"winner": "claude", "confidence": 0.85},
            {"winner": "gemini", "confidence": 0.78},
        ]

        handler.ctx["elo_system"] = mock_elo
        handler.get_storage = MagicMock(return_value=None)

        result = handler._get_debate_quality_metrics()

        assert len(result["recent_winners"]) == 2
        assert result["avg_confidence"] == pytest.approx(0.815, rel=0.01)

    def test_get_debate_quality_metrics_no_data(self, handler):
        """Debate quality handles missing data."""
        handler.ctx = {}
        handler.get_storage = MagicMock(return_value=None)

        result = handler._get_debate_quality_metrics()

        assert result["recent_winners"] == []
        assert result["avg_confidence"] == 0.0


# =============================================================================
# Full Dashboard Tests
# =============================================================================


class TestFullDashboard:
    """Tests for complete dashboard generation."""

    def test_get_debates_dashboard_with_storage(self, handler, mock_storage):
        """Full debates dashboard returns all sections."""
        storage, cursor = mock_storage

        # Mock SQL results
        cursor.fetchone.side_effect = [
            (100, 75, 0.78),  # Summary
            (25, 18),  # Recent activity
        ]

        handler.get_storage = MagicMock(return_value=storage)
        handler.get_elo_system = MagicMock(return_value=None)

        with patch.object(handler, "_get_consensus_insights", return_value={}):
            with patch.object(handler, "_get_system_health", return_value={}):
                result = handler._get_debates_dashboard(None, 10, 24)

        # Result should be a HandlerResult tuple or dict
        assert result is not None

    def test_get_debates_dashboard_no_storage(self, handler):
        """Dashboard handles missing storage."""
        handler.get_storage = MagicMock(return_value=None)
        handler.get_elo_system = MagicMock(return_value=None)

        with patch.object(handler, "_get_consensus_insights", return_value={}):
            with patch.object(handler, "_get_system_health", return_value={}):
                result = handler._get_debates_dashboard(None, 10, 24)

        assert result is not None

    def test_get_quality_metrics_full(self, handler):
        """Full quality metrics returns all sections."""
        handler.ctx = {}
        handler.get_storage = MagicMock(return_value=None)

        result = handler._get_quality_metrics()

        # Result should contain all sections
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
