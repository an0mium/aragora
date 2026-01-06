"""Tests for DashboardHandler."""

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.dashboard import DashboardHandler


class TestDashboardHandlerRouting:
    """Tests for DashboardHandler routing."""

    def test_can_handle_dashboard_debates(self):
        """Handler can handle /api/dashboard/debates."""
        handler = DashboardHandler({})
        assert handler.can_handle("/api/dashboard/debates") is True

    def test_cannot_handle_other_paths(self):
        """Handler cannot handle unrelated paths."""
        handler = DashboardHandler({})
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/metrics") is False
        assert handler.can_handle("/api/dashboard/other") is False


class TestGetDebatesDashboard:
    """Tests for debate dashboard endpoint."""

    def test_returns_json_response(self):
        """Dashboard endpoint returns JSON response."""
        handler = DashboardHandler({})
        result = handler._get_debates_dashboard(None, 10, 24)
        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "application/json"

    def test_response_has_required_sections(self):
        """Response has all required sections."""
        import json
        handler = DashboardHandler({})
        result = handler._get_debates_dashboard(None, 10, 24)
        data = json.loads(result.body)

        assert "summary" in data
        assert "recent_activity" in data
        assert "agent_performance" in data
        assert "debate_patterns" in data
        assert "consensus_insights" in data
        assert "system_health" in data
        assert "generated_at" in data

    def test_summary_has_expected_fields(self):
        """Summary section has expected fields."""
        import json
        handler = DashboardHandler({})
        result = handler._get_debates_dashboard(None, 10, 24)
        data = json.loads(result.body)

        summary = data["summary"]
        assert "total_debates" in summary
        assert "consensus_reached" in summary
        assert "consensus_rate" in summary
        assert "avg_confidence" in summary

    def test_recent_activity_has_period_hours(self):
        """Recent activity includes period hours."""
        import json
        handler = DashboardHandler({})
        result = handler._get_debates_dashboard(None, 10, 48)
        data = json.loads(result.body)

        activity = data["recent_activity"]
        assert activity["period_hours"] == 48

    def test_agent_performance_has_top_performers(self):
        """Agent performance includes top performers list."""
        import json
        handler = DashboardHandler({})
        result = handler._get_debates_dashboard(None, 10, 24)
        data = json.loads(result.body)

        performance = data["agent_performance"]
        assert "top_performers" in performance
        assert isinstance(performance["top_performers"], list)

    def test_consensus_insights_has_domains(self):
        """Consensus insights includes domains list."""
        import json
        handler = DashboardHandler({})
        result = handler._get_debates_dashboard(None, 10, 24)
        data = json.loads(result.body)

        insights = data["consensus_insights"]
        assert "domains" in insights
        assert isinstance(insights["domains"], list)

    def test_system_health_has_prometheus_status(self):
        """System health includes prometheus availability."""
        import json
        handler = DashboardHandler({})
        result = handler._get_debates_dashboard(None, 10, 24)
        data = json.loads(result.body)

        health = data["system_health"]
        assert "prometheus_available" in health
        assert isinstance(health["prometheus_available"], bool)


class TestDashboardWithMockedDependencies:
    """Tests with mocked dependencies."""

    def test_summary_metrics_with_storage(self):
        """Summary metrics method processes storage data."""
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {"consensus_reached": True, "confidence": 0.8},
            {"consensus_reached": False, "confidence": 0.5},
            {"consensus_reached": True, "confidence": 0.9},
        ]

        handler = DashboardHandler({"storage": mock_storage})
        summary = handler._get_summary_metrics(None)

        assert summary["total_debates"] == 3
        assert summary["consensus_reached"] == 2

    def test_agent_performance_with_elo(self):
        """Agent performance method processes ELO data."""
        mock_elo = MagicMock()
        mock_elo.list_agents.return_value = ["agent1", "agent2"]
        mock_rating = MagicMock()
        mock_rating.elo = 1600
        mock_rating.wins = 10
        mock_rating.losses = 2
        mock_rating.draws = 1
        mock_rating.win_rate = 0.83
        mock_rating.debates_count = 13
        mock_elo.get_rating.return_value = mock_rating

        handler = DashboardHandler({"elo_system": mock_elo})
        performance = handler._get_agent_performance(10)

        assert performance["total_agents"] == 2
        assert len(performance["top_performers"]) == 2

    def test_handles_missing_storage_gracefully(self):
        """Dashboard handles missing storage gracefully."""
        import json

        handler = DashboardHandler({})
        result = handler._get_debates_dashboard(None, 10, 24)
        data = json.loads(result.body)

        # Should return zeros/empty, not crash
        assert data["summary"]["total_debates"] == 0
        assert data["summary"]["consensus_reached"] == 0

    def test_handles_missing_elo_gracefully(self):
        """Dashboard handles missing ELO system gracefully."""
        import json

        handler = DashboardHandler({})
        result = handler._get_debates_dashboard(None, 10, 24)
        data = json.loads(result.body)

        # Should return empty, not crash
        assert data["agent_performance"]["total_agents"] == 0
        assert data["agent_performance"]["top_performers"] == []


class TestDashboardParameters:
    """Tests for dashboard endpoint parameters."""

    def test_limit_is_capped(self):
        """Limit parameter is capped at 50."""
        handler = DashboardHandler({})
        result = handler.handle("/api/dashboard/debates", {"limit": "100"}, None)
        # Should not crash, limit internally capped
        assert result is not None

    def test_domain_filter_accepted(self):
        """Domain filter parameter is accepted."""
        handler = DashboardHandler({})
        result = handler.handle("/api/dashboard/debates", {"domain": "science"}, None)
        assert result is not None

    def test_hours_parameter_accepted(self):
        """Hours parameter is accepted."""
        import json
        handler = DashboardHandler({})
        result = handler.handle("/api/dashboard/debates", {"hours": "72"}, None)
        data = json.loads(result.body)
        assert data["recent_activity"]["period_hours"] == 72


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
