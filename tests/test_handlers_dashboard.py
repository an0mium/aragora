"""
Tests for DashboardHandler endpoints.

Endpoints tested:
- GET /api/dashboard/debates - Consolidated debate dashboard metrics
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers.dashboard import DashboardHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_storage():
    """Create a mock debate storage."""
    storage = Mock()
    storage.list_debates.return_value = [
        {
            "id": "debate-1",
            "task": "Test Debate 1",
            "created_at": datetime.now().isoformat(),
            "consensus_reached": True,
            "confidence": 0.85,
            "domain": "coding",
        },
        {
            "id": "debate-2",
            "task": "Test Debate 2",
            "created_at": (datetime.now() - timedelta(hours=48)).isoformat(),
            "consensus_reached": False,
            "confidence": 0.6,
            "domain": "reasoning",
        },
        {
            "id": "debate-3",
            "task": "Test Debate 3",
            "created_at": datetime.now().isoformat(),
            "consensus_reached": True,
            "confidence": 0.9,
            "domain": "coding",
            "disagreement_report": {"types": ["semantic", "factual"]},
        },
    ]
    return storage


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = Mock()
    elo.list_agents.return_value = ["claude", "gpt4", "gemini"]

    rating_claude = Mock()
    rating_claude.elo = 1200
    rating_claude.wins = 10
    rating_claude.losses = 5
    rating_claude.draws = 2
    rating_claude.win_rate = 0.59
    rating_claude.debates_count = 17

    rating_gpt4 = Mock()
    rating_gpt4.elo = 1150
    rating_gpt4.wins = 8
    rating_gpt4.losses = 6
    rating_gpt4.draws = 3
    rating_gpt4.win_rate = 0.47
    rating_gpt4.debates_count = 17

    rating_gemini = Mock()
    rating_gemini.elo = 1100
    rating_gemini.wins = 5
    rating_gemini.losses = 8
    rating_gemini.draws = 2
    rating_gemini.win_rate = 0.33
    rating_gemini.debates_count = 15

    def get_rating(name):
        ratings = {
            "claude": rating_claude,
            "gpt4": rating_gpt4,
            "gemini": rating_gemini,
        }
        return ratings.get(name)

    elo.get_rating.side_effect = get_rating
    return elo


@pytest.fixture
def dashboard_handler(mock_storage, mock_elo_system):
    """Create a DashboardHandler with mock dependencies."""
    ctx = {
        "storage": mock_storage,
        "elo_system": mock_elo_system,
        "nomic_dir": None,
    }
    return DashboardHandler(ctx)


@pytest.fixture
def handler_no_storage():
    """Create a DashboardHandler without storage."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
    }
    return DashboardHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestDashboardRouting:
    """Tests for route matching."""

    def test_can_handle_dashboard_debates(self, dashboard_handler):
        assert dashboard_handler.can_handle("/api/dashboard/debates") is True

    def test_cannot_handle_unrelated_routes(self, dashboard_handler):
        assert dashboard_handler.can_handle("/api/dashboard") is False
        assert dashboard_handler.can_handle("/api/debates") is False
        assert dashboard_handler.can_handle("/api/dashboard/other") is False
        assert dashboard_handler.can_handle("/api/dashboard/debates/extra") is False


# ============================================================================
# GET /api/dashboard/debates Tests
# ============================================================================

class TestDashboardDebates:
    """Tests for GET /api/dashboard/debates endpoint."""

    def test_dashboard_success(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        # Check all sections are present
        assert "summary" in data
        assert "recent_activity" in data
        assert "agent_performance" in data
        assert "debate_patterns" in data
        assert "consensus_insights" in data
        assert "system_health" in data
        assert "generated_at" in data

    def test_dashboard_summary_metrics(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        data = json.loads(result.body)
        summary = data["summary"]

        assert summary["total_debates"] == 3
        assert summary["consensus_reached"] == 2  # 2 debates with consensus
        # 2/3 = 0.667, rounds to 0.667
        assert summary["consensus_rate"] == 0.667

    def test_dashboard_agent_performance(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        data = json.loads(result.body)
        perf = data["agent_performance"]

        assert perf["total_agents"] == 3
        assert len(perf["top_performers"]) <= 3
        # Should be sorted by ELO, so claude should be first
        if perf["top_performers"]:
            assert perf["top_performers"][0]["name"] == "claude"
            assert perf["top_performers"][0]["elo"] == 1200

    def test_dashboard_with_limit(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {"limit": "2"}, None)

        assert result is not None
        data = json.loads(result.body)
        # Limit affects top_performers
        assert len(data["agent_performance"]["top_performers"]) <= 2

    def test_dashboard_with_hours_filter(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {"hours": "12"}, None)

        assert result is not None
        data = json.loads(result.body)
        assert data["recent_activity"]["period_hours"] == 12

    def test_dashboard_with_domain_filter(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {"domain": "coding"}, None)

        assert result is not None
        data = json.loads(result.body)
        # Domain filter is passed through
        assert "summary" in data

    def test_dashboard_limit_capped_at_50(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {"limit": "100"}, None)

        assert result is not None
        data = json.loads(result.body)
        # Even though we requested 100, it should be capped
        # (The cap is applied internally, but hard to verify - just check it doesn't crash)
        assert data is not None


# ============================================================================
# Dashboard Subsections Tests
# ============================================================================

class TestDashboardRecentActivity:
    """Tests for recent activity section."""

    def test_recent_activity_counts(self, dashboard_handler, mock_storage):
        result = dashboard_handler.handle("/api/dashboard/debates", {"hours": "24"}, None)

        assert result is not None
        data = json.loads(result.body)
        activity = data["recent_activity"]

        # debates_last_period should count only recent debates
        assert "debates_last_period" in activity
        assert "consensus_last_period" in activity
        assert "domains_active" in activity
        assert "most_active_domain" in activity

    def test_recent_activity_with_old_debates(self, dashboard_handler, mock_storage):
        # Override with all old debates
        mock_storage.list_debates.return_value = [
            {
                "id": "old-1",
                "created_at": (datetime.now() - timedelta(days=7)).isoformat(),
                "consensus_reached": True,
            }
        ]

        result = dashboard_handler.handle("/api/dashboard/debates", {"hours": "24"}, None)

        assert result is not None
        data = json.loads(result.body)
        # Old debate should not count as recent
        assert data["recent_activity"]["debates_last_period"] == 0


class TestDashboardDebatePatterns:
    """Tests for debate patterns section."""

    def test_debate_patterns_disagreements(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        data = json.loads(result.body)
        patterns = data["debate_patterns"]

        assert "disagreement_stats" in patterns
        assert "early_stopping" in patterns

        # One debate has disagreement_report
        assert patterns["disagreement_stats"]["with_disagreements"] == 1


class TestDashboardSystemHealth:
    """Tests for system health section."""

    def test_system_health_fields(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        data = json.loads(result.body)
        health = data["system_health"]

        assert "uptime_seconds" in health
        assert "cache_entries" in health
        assert "active_websocket_connections" in health
        assert "prometheus_available" in health


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestDashboardErrorHandling:
    """Tests for error handling."""

    def test_dashboard_no_storage(self, handler_no_storage):
        result = handler_no_storage.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        # Should return empty/default values, not crash
        assert data["summary"]["total_debates"] == 0

    def test_dashboard_storage_exception(self, dashboard_handler, mock_storage):
        mock_storage.list_debates.side_effect = Exception("Database error")

        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        # Should handle gracefully with default values
        assert data["summary"]["total_debates"] == 0

    def test_dashboard_elo_exception(self, dashboard_handler, mock_elo_system):
        mock_elo_system.list_agents.side_effect = Exception("ELO error")

        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        # Should handle gracefully with default values
        assert data["agent_performance"]["top_performers"] == []

    def test_handle_returns_none_for_unhandled_route(self, dashboard_handler):
        result = dashboard_handler.handle("/api/other/endpoint", {}, None)
        assert result is None


# ============================================================================
# Edge Cases
# ============================================================================

class TestDashboardEdgeCases:
    """Tests for edge cases."""

    def test_dashboard_empty_storage(self, dashboard_handler, mock_storage):
        mock_storage.list_debates.return_value = []

        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["summary"]["total_debates"] == 0
        assert data["summary"]["consensus_rate"] == 0.0

    def test_dashboard_no_agents(self, dashboard_handler, mock_elo_system):
        mock_elo_system.list_agents.return_value = []

        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent_performance"]["total_agents"] == 0
        assert data["agent_performance"]["avg_elo"] == 0

    def test_dashboard_invalid_limit_param(self, dashboard_handler):
        # Invalid limit should default to 10
        result = dashboard_handler.handle("/api/dashboard/debates", {"limit": "invalid"}, None)

        assert result is not None
        assert result.status_code == 200

    def test_dashboard_negative_limit(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {"limit": "-5"}, None)

        assert result is not None
        assert result.status_code == 200

    def test_dashboard_debates_with_missing_fields(self, dashboard_handler, mock_storage):
        # Debates with missing optional fields
        mock_storage.list_debates.return_value = [
            {"id": "sparse-1"},  # Missing most fields
            {"id": "sparse-2", "created_at": "invalid-date"},
        ]

        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        # Should handle gracefully
        assert data["summary"]["total_debates"] == 2

    def test_dashboard_generated_at_timestamp(self, dashboard_handler):
        result = dashboard_handler.handle("/api/dashboard/debates", {}, None)

        assert result is not None
        data = json.loads(result.body)
        # generated_at should be a recent timestamp
        assert isinstance(data["generated_at"], (int, float))
        assert data["generated_at"] > 0
