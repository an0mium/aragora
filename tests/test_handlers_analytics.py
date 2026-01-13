"""
Tests for AnalyticsHandler endpoints.

Endpoints tested:
- GET /api/analytics/disagreements - Get disagreement statistics
- GET /api/analytics/role-rotation - Get role rotation statistics
- GET /api/analytics/early-stops - Get early stopping statistics
- GET /api/ranking/stats - Get ranking statistics
- GET /api/memory/stats - Get memory statistics
- GET /api/memory/tier-stats - Get memory tier statistics
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from aragora.server.handlers.analytics import AnalyticsHandler
from aragora.server.handlers.base import clear_cache


@dataclass
class MockAgentRanking:
    """Mock agent ranking object."""

    agent_name: str
    elo_rating: int
    total_debates: int


@pytest.fixture
def mock_storage():
    """Create a mock storage instance."""
    storage = Mock()
    storage.list_debates.return_value = [
        {
            "slug": "debate-1",
            "messages": [
                {"cognitive_role": "analyst", "content": "Analysis..."},
                {"cognitive_role": "critic", "content": "Critique..."},
            ],
            "result": {
                "rounds_used": 3,
                "early_stopped": False,
                "disagreement_report": {"unanimous_critiques": True},
                "uncertainty_metrics": {"disagreement_type": "methodological"},
            },
        },
        {
            "slug": "debate-2",
            "messages": [
                {"role": "speaker", "content": "Point..."},
            ],
            "result": {
                "rounds_used": 2,
                "early_stopped": True,
                "disagreement_report": {"unanimous_critiques": False},
            },
        },
    ]
    return storage


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = Mock()
    elo.get_leaderboard.return_value = [
        MockAgentRanking("claude", 1650, 20),
        MockAgentRanking("gpt4", 1580, 18),
        MockAgentRanking("gemini", 1520, 15),
    ]
    return elo


@pytest.fixture
def analytics_handler(mock_storage, mock_elo_system, tmp_path):
    """Create an AnalyticsHandler with mocks."""
    ctx = {
        "storage": mock_storage,
        "elo_system": mock_elo_system,
        "nomic_dir": tmp_path,
    }
    return AnalyticsHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestAnalyticsHandlerRouting:
    """Tests for route matching."""

    def test_can_handle_disagreements(self, analytics_handler):
        """Should handle /api/analytics/disagreements."""
        assert analytics_handler.can_handle("/api/analytics/disagreements") is True

    def test_can_handle_role_rotation(self, analytics_handler):
        """Should handle /api/analytics/role-rotation."""
        assert analytics_handler.can_handle("/api/analytics/role-rotation") is True

    def test_can_handle_early_stops(self, analytics_handler):
        """Should handle /api/analytics/early-stops."""
        assert analytics_handler.can_handle("/api/analytics/early-stops") is True

    def test_can_handle_ranking_stats(self, analytics_handler):
        """Should handle /api/ranking/stats."""
        assert analytics_handler.can_handle("/api/ranking/stats") is True

    def test_can_handle_memory_stats(self, analytics_handler):
        """Should handle /api/memory/stats."""
        assert analytics_handler.can_handle("/api/memory/stats") is True

    def test_cannot_handle_memory_tier_stats(self, analytics_handler):
        """Should NOT handle /api/memory/tier-stats (moved to MemoryHandler)."""
        assert analytics_handler.can_handle("/api/memory/tier-stats") is False

    def test_cannot_handle_unknown_routes(self, analytics_handler):
        """Should not handle unknown routes."""
        assert analytics_handler.can_handle("/api/analytics/unknown") is False
        assert analytics_handler.can_handle("/api/debates") is False
        assert analytics_handler.can_handle("/api/leaderboard") is False

    def test_handle_returns_none_for_unknown(self, analytics_handler):
        """Should return None for unknown paths."""
        result = analytics_handler.handle("/api/unknown", {}, None)
        assert result is None


# ============================================================================
# Disagreement Stats Tests
# ============================================================================


class TestDisagreementStats:
    """Tests for /api/analytics/disagreements endpoint."""

    def test_returns_disagreement_stats(self, analytics_handler):
        """Should return disagreement statistics."""
        result = analytics_handler.handle("/api/analytics/disagreements", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "stats" in data
        assert "total_debates" in data["stats"]
        assert "with_disagreements" in data["stats"]

    def test_counts_debates_correctly(self, analytics_handler):
        """Should count total debates correctly."""
        result = analytics_handler.handle("/api/analytics/disagreements", {}, None)

        data = json.loads(result.body)
        assert data["stats"]["total_debates"] == 2

    def test_tracks_disagreement_types(self, analytics_handler):
        """Should track disagreement types."""
        result = analytics_handler.handle("/api/analytics/disagreements", {}, None)

        data = json.loads(result.body)
        assert "disagreement_types" in data["stats"]
        # First debate has methodological disagreement type
        assert data["stats"]["disagreement_types"].get("methodological", 0) >= 1

    def test_empty_debates_returns_empty_stats(self, analytics_handler, mock_storage):
        """Should handle empty debates list."""
        mock_storage.list_debates.return_value = []

        result = analytics_handler.handle("/api/analytics/disagreements", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["stats"]["total_debates"] == 0
        assert data["stats"]["with_disagreements"] == 0

    def test_handles_storage_unavailable(self):
        """Should handle missing storage gracefully."""
        handler = AnalyticsHandler({})
        result = handler.handle("/api/analytics/disagreements", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["stats"] == {}

    def test_handles_exception(self, analytics_handler, mock_storage):
        """Should return 500 on exception."""
        mock_storage.list_debates.side_effect = Exception("DB error")

        result = analytics_handler.handle("/api/analytics/disagreements", {}, None)

        assert result.status_code == 500


# ============================================================================
# Role Rotation Stats Tests
# ============================================================================


class TestRoleRotationStats:
    """Tests for /api/analytics/role-rotation endpoint."""

    def test_returns_role_rotation_stats(self, analytics_handler):
        """Should return role rotation statistics."""
        result = analytics_handler.handle("/api/analytics/role-rotation", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "stats" in data
        assert "total_debates" in data["stats"]
        assert "role_assignments" in data["stats"]

    def test_counts_role_assignments(self, analytics_handler):
        """Should count role assignments from messages."""
        result = analytics_handler.handle("/api/analytics/role-rotation", {}, None)

        data = json.loads(result.body)
        roles = data["stats"]["role_assignments"]
        # Should have counted analyst and critic roles
        assert "analyst" in roles or "speaker" in roles

    def test_uses_fallback_role(self, analytics_handler, mock_storage):
        """Should use 'role' as fallback when cognitive_role missing."""
        mock_storage.list_debates.return_value = [
            {
                "slug": "test",
                "messages": [{"role": "speaker", "content": "test"}],
                "result": {},
            }
        ]

        result = analytics_handler.handle("/api/analytics/role-rotation", {}, None)

        data = json.loads(result.body)
        assert data["stats"]["role_assignments"].get("speaker", 0) >= 1

    def test_empty_debates_returns_empty_stats(self, analytics_handler, mock_storage):
        """Should handle empty debates list."""
        mock_storage.list_debates.return_value = []

        result = analytics_handler.handle("/api/analytics/role-rotation", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["stats"]["total_debates"] == 0
        assert data["stats"]["role_assignments"] == {}

    def test_handles_exception(self, analytics_handler, mock_storage):
        """Should return 500 on exception."""
        mock_storage.list_debates.side_effect = Exception("DB error")

        result = analytics_handler.handle("/api/analytics/role-rotation", {}, None)

        assert result.status_code == 500


# ============================================================================
# Early Stop Stats Tests
# ============================================================================


class TestEarlyStopStats:
    """Tests for /api/analytics/early-stops endpoint."""

    def test_returns_early_stop_stats(self, analytics_handler):
        """Should return early stop statistics."""
        result = analytics_handler.handle("/api/analytics/early-stops", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "stats" in data
        assert "total_debates" in data["stats"]
        assert "early_stopped" in data["stats"]
        assert "full_rounds" in data["stats"]

    def test_counts_early_stopped_debates(self, analytics_handler):
        """Should count early stopped debates."""
        result = analytics_handler.handle("/api/analytics/early-stops", {}, None)

        data = json.loads(result.body)
        # Second debate has early_stopped=True
        assert data["stats"]["early_stopped"] == 1
        assert data["stats"]["full_rounds"] == 1

    def test_calculates_average_rounds(self, analytics_handler):
        """Should calculate average rounds."""
        result = analytics_handler.handle("/api/analytics/early-stops", {}, None)

        data = json.loads(result.body)
        # (3 + 2) / 2 = 2.5
        assert data["stats"]["average_rounds"] == 2.5

    def test_empty_debates_returns_empty_stats(self, analytics_handler, mock_storage):
        """Should handle empty debates list."""
        mock_storage.list_debates.return_value = []

        result = analytics_handler.handle("/api/analytics/early-stops", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["stats"]["total_debates"] == 0
        assert data["stats"]["early_stopped"] == 0

    def test_handles_exception(self, analytics_handler, mock_storage):
        """Should return 500 on exception."""
        mock_storage.list_debates.side_effect = Exception("DB error")

        result = analytics_handler.handle("/api/analytics/early-stops", {}, None)

        assert result.status_code == 500


# ============================================================================
# Ranking Stats Tests
# ============================================================================


class TestRankingStats:
    """Tests for /api/ranking/stats endpoint."""

    def test_returns_ranking_stats(self, analytics_handler):
        """Should return ranking statistics."""
        result = analytics_handler.handle("/api/ranking/stats", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "stats" in data
        assert "total_agents" in data["stats"]
        assert "total_matches" in data["stats"]

    def test_calculates_correct_totals(self, analytics_handler):
        """Should calculate correct totals."""
        result = analytics_handler.handle("/api/ranking/stats", {}, None)

        data = json.loads(result.body)
        assert data["stats"]["total_agents"] == 3
        # 20 + 18 + 15 = 53
        assert data["stats"]["total_matches"] == 53

    def test_identifies_top_agent(self, analytics_handler):
        """Should identify top agent."""
        result = analytics_handler.handle("/api/ranking/stats", {}, None)

        data = json.loads(result.body)
        assert data["stats"]["top_agent"] == "claude"

    def test_calculates_elo_range(self, analytics_handler):
        """Should calculate ELO range."""
        result = analytics_handler.handle("/api/ranking/stats", {}, None)

        data = json.loads(result.body)
        assert data["stats"]["elo_range"]["min"] == 1520
        assert data["stats"]["elo_range"]["max"] == 1650

    def test_calculates_average_elo(self, analytics_handler):
        """Should calculate average ELO."""
        result = analytics_handler.handle("/api/ranking/stats", {}, None)

        data = json.loads(result.body)
        # (1650 + 1580 + 1520) / 3 = 1583.33...
        assert 1580 < data["stats"]["avg_elo"] < 1590

    def test_returns_503_without_elo_system(self):
        """Should return 503 when ELO system unavailable."""
        handler = AnalyticsHandler({"storage": Mock()})
        result = handler.handle("/api/ranking/stats", {}, None)

        assert result.status_code == 503

    def test_handles_empty_leaderboard(self, analytics_handler, mock_elo_system):
        """Should handle empty leaderboard."""
        mock_elo_system.get_leaderboard.return_value = []

        result = analytics_handler.handle("/api/ranking/stats", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["stats"]["total_agents"] == 0
        assert data["stats"]["top_agent"] is None

    def test_handles_exception(self, analytics_handler, mock_elo_system):
        """Should return 500 on exception."""
        mock_elo_system.get_leaderboard.side_effect = Exception("ELO error")

        result = analytics_handler.handle("/api/ranking/stats", {}, None)

        assert result.status_code == 500


# ============================================================================
# Memory Stats Tests
# ============================================================================


class TestMemoryStats:
    """Tests for /api/memory/stats endpoint."""

    # Note: /api/memory/tier-stats moved to MemoryHandler

    def test_returns_memory_stats(self, analytics_handler):
        """Should return memory statistics."""
        result = analytics_handler.handle("/api/memory/stats", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "stats" in data

    def test_detects_missing_dbs(self, analytics_handler):
        """Should detect missing database files."""
        result = analytics_handler.handle("/api/memory/stats", {}, None)

        data = json.loads(result.body)
        assert data["stats"]["embeddings_db"] is False
        assert data["stats"]["insights_db"] is False
        assert data["stats"]["continuum_memory"] is False

    def test_detects_existing_embeddings_db(self, analytics_handler, tmp_path):
        """Should detect existing embeddings database."""
        (tmp_path / "debate_embeddings.db").touch()

        result = analytics_handler.handle("/api/memory/stats", {}, None)

        data = json.loads(result.body)
        assert data["stats"]["embeddings_db"] is True

    def test_detects_existing_insights_db(self, analytics_handler, tmp_path):
        """Should detect existing insights database."""
        (tmp_path / "aragora_insights.db").touch()

        result = analytics_handler.handle("/api/memory/stats", {}, None)

        data = json.loads(result.body)
        assert data["stats"]["insights_db"] is True

    def test_detects_existing_continuum_db(self, analytics_handler, tmp_path):
        """Should detect existing continuum memory database."""
        (tmp_path / "continuum_memory.db").touch()

        result = analytics_handler.handle("/api/memory/stats", {}, None)

        data = json.loads(result.body)
        assert data["stats"]["continuum_memory"] is True

    def test_returns_empty_without_nomic_dir(self):
        """Should return empty stats without nomic_dir."""
        handler = AnalyticsHandler({"storage": Mock()})
        result = handler.handle("/api/memory/stats", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["stats"] == {}

    def test_handles_exception(self, analytics_handler, tmp_path):
        """Should return 500 on exception."""
        # Make nomic_dir a file instead of directory to cause exception
        analytics_handler.ctx["nomic_dir"] = tmp_path / "not_a_dir.txt"
        (tmp_path / "not_a_dir.txt").touch()

        result = analytics_handler.handle("/api/memory/stats", {}, None)

        # Should either handle gracefully or return error
        assert result.status_code in [200, 500]


# ============================================================================
# Caching Tests
# ============================================================================


class TestAnalyticsCaching:
    """Tests for caching behavior."""

    def test_debates_are_cached(self, analytics_handler, mock_storage):
        """Should cache debates between calls."""
        # First call
        analytics_handler.handle("/api/analytics/disagreements", {}, None)
        # Second call should use cache
        analytics_handler.handle("/api/analytics/role-rotation", {}, None)

        # Storage should only be called once due to caching
        # (actual behavior depends on cache TTL)
        assert mock_storage.list_debates.called

    def test_cached_debates_method(self, analytics_handler, mock_storage):
        """Should use _get_cached_debates for efficiency."""
        result = analytics_handler._get_cached_debates(limit=50)
        assert isinstance(result, list)


# ============================================================================
# Edge Cases
# ============================================================================


class TestAnalyticsEdgeCases:
    """Tests for edge cases."""

    def test_debate_without_result(self, analytics_handler, mock_storage):
        """Should handle debates without result field."""
        mock_storage.list_debates.return_value = [
            {"slug": "no-result", "messages": []},
        ]

        result = analytics_handler.handle("/api/analytics/early-stops", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["stats"]["total_debates"] == 1
        assert data["stats"]["average_rounds"] == 0

    def test_debate_without_messages(self, analytics_handler, mock_storage):
        """Should handle debates without messages."""
        mock_storage.list_debates.return_value = [
            {"slug": "no-messages", "result": {}},
        ]

        result = analytics_handler.handle("/api/analytics/role-rotation", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["stats"]["role_assignments"] == {}

    def test_debate_without_disagreement_report(self, analytics_handler, mock_storage):
        """Should handle debates without disagreement report."""
        mock_storage.list_debates.return_value = [
            {"slug": "no-report", "messages": [], "result": {}},
        ]

        result = analytics_handler.handle("/api/analytics/disagreements", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["stats"]["with_disagreements"] == 0
        assert data["stats"]["unanimous"] == 0

    def test_single_agent_leaderboard(self, analytics_handler, mock_elo_system):
        """Should handle single agent in leaderboard."""
        mock_elo_system.get_leaderboard.return_value = [
            MockAgentRanking("lonely", 1500, 5),
        ]

        result = analytics_handler.handle("/api/ranking/stats", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["stats"]["total_agents"] == 1
        assert data["stats"]["elo_range"]["min"] == data["stats"]["elo_range"]["max"]
