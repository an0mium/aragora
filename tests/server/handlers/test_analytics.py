"""
Tests for aragora.server.handlers.analytics - Analytics endpoints handler.

Tests cover:
- can_handle() route matching
- handle() route dispatching
- Rate limiting
- _get_disagreement_stats() - Disagreement statistics
- _get_role_rotation_stats() - Role rotation stats
- _get_early_stop_stats() - Early stopping stats
- _get_consensus_quality() - Consensus quality metrics
- _get_ranking_stats() - Ranking system stats
- _get_memory_stats() - Memory system stats
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, headers: dict = None, client_address: tuple = None):
        self.headers = headers or {}
        self.client_address = client_address or ("127.0.0.1", 12345)


def create_analytics_handler():
    """Create an AnalyticsHandler with empty context."""
    from aragora.server.handlers.analytics import AnalyticsHandler

    return AnalyticsHandler({})


def get_body(result) -> dict:
    """Extract body as dict from HandlerResult."""
    if hasattr(result, "body"):
        return json.loads(result.body.decode("utf-8"))
    return result


def get_status(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result


# ===========================================================================
# Test can_handle() Route Matching
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_handles_disagreements_route(self):
        """Should handle /api/analytics/disagreements."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/v1/analytics/disagreements") is True

    def test_handles_role_rotation_route(self):
        """Should handle /api/analytics/role-rotation."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/v1/analytics/role-rotation") is True

    def test_handles_early_stops_route(self):
        """Should handle /api/analytics/early-stops."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/v1/analytics/early-stops") is True

    def test_handles_consensus_quality_route(self):
        """Should handle /api/analytics/consensus-quality."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/v1/analytics/consensus-quality") is True

    def test_handles_ranking_stats_route(self):
        """Should handle /api/ranking/stats."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/v1/ranking/stats") is True

    def test_handles_memory_stats_route(self):
        """Should handle /api/memory/stats."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/v1/memory/stats") is True

    def test_rejects_unknown_routes(self):
        """Should reject unknown routes."""
        handler = create_analytics_handler()

        assert handler.can_handle("/api/v1/unknown") is False
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/analytics/unknown") is False


# ===========================================================================
# Test Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting in handle()."""

    async def test_rate_limit_exceeded_returns_429(self):
        """Should return 429 when rate limit exceeded."""
        from aragora.server.handlers.analytics import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler(client_address=("192.168.1.100", 12345))

        # Force rate limit by mocking is_allowed to return False
        with patch.object(_analytics_limiter, "is_allowed", return_value=False):
            result = await handler.handle("/api/v1/analytics/disagreements", {}, mock_http)

        assert get_status(result) == 429
        assert "Rate limit" in get_body(result)["error"]


# ===========================================================================
# Test _get_disagreement_stats()
# ===========================================================================


class TestGetDisagreementStats:
    """Tests for _get_disagreement_stats()."""

    def test_no_storage_returns_empty(self):
        """Should return empty stats when storage unavailable."""
        handler = create_analytics_handler()

        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_disagreement_stats()

        assert get_status(result) == 200
        assert get_body(result)["stats"] == {}

    def test_empty_debates_list(self):
        """Should handle empty debates list."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_disagreement_stats()

        assert get_status(result) == 200
        assert get_body(result)["stats"]["total_debates"] == 0

    def test_calculates_disagreement_stats(self):
        """Should calculate disagreement statistics from debates."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": "debate-1",
                "result": {
                    "disagreement_report": {"unanimous_critiques": True},
                    "uncertainty_metrics": {"disagreement_type": "factual"},
                },
            },
            {
                "id": "debate-2",
                "result": {
                    "disagreement_report": {"unanimous_critiques": False},
                    "uncertainty_metrics": {"disagreement_type": "value"},
                },
            },
            {
                "id": "debate-3",
                "result": {},  # No disagreement report
            },
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_disagreement_stats()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["total_debates"] == 3
        assert body["stats"]["with_disagreements"] == 1  # One with unanimous_critiques=True


# ===========================================================================
# Test _get_role_rotation_stats()
# ===========================================================================


class TestGetRoleRotationStats:
    """Tests for _get_role_rotation_stats()."""

    def test_no_storage_returns_empty(self):
        """Should return empty stats when storage unavailable."""
        handler = create_analytics_handler()

        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_role_rotation_stats()

        assert get_status(result) == 200
        assert get_body(result)["stats"] == {}

    def test_counts_role_assignments(self):
        """Should count role assignments from debate messages."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": "debate-1",
                "messages": [
                    {"cognitive_role": "advocate"},
                    {"cognitive_role": "critic"},
                    {"role": "synthesizer"},  # Fallback to 'role'
                ],
            },
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_role_rotation_stats()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["role_assignments"]["advocate"] == 1
        assert body["stats"]["role_assignments"]["critic"] == 1
        assert body["stats"]["role_assignments"]["synthesizer"] == 1


# ===========================================================================
# Test _get_early_stop_stats()
# ===========================================================================


class TestGetEarlyStopStats:
    """Tests for _get_early_stop_stats()."""

    def test_no_storage_returns_empty(self):
        """Should return empty stats when storage unavailable."""
        handler = create_analytics_handler()

        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_early_stop_stats()

        assert get_status(result) == 200
        assert get_body(result)["stats"] == {}

    def test_calculates_early_stop_stats(self):
        """Should calculate early stopping statistics."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {"id": "1", "result": {"rounds_used": 2, "early_stopped": True}},
            {"id": "2", "result": {"rounds_used": 5, "early_stopped": False}},
            {"id": "3", "result": {"rounds_used": 3, "early_stopped": True}},
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_early_stop_stats()

        body = get_body(result)
        assert get_status(result) == 200
        stats = body["stats"]
        assert stats["total_debates"] == 3
        assert stats["early_stopped"] == 2
        assert stats["full_rounds"] == 1
        assert stats["average_rounds"] == pytest.approx(10 / 3)  # (2+5+3)/3


# ===========================================================================
# Test _get_consensus_quality()
# ===========================================================================


class TestGetConsensusQuality:
    """Tests for _get_consensus_quality()."""

    def test_no_storage_returns_empty(self):
        """Should return empty stats when storage unavailable."""
        handler = create_analytics_handler()

        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["quality_score"] == 0

    def test_no_debates_returns_insufficient_data(self):
        """Should return insufficient_data when no debates."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["trend"] == "insufficient_data"
        assert body["quality_score"] == 0

    def test_calculates_quality_metrics(self):
        """Should calculate consensus quality metrics."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": "debate-1",
                "timestamp": "2024-01-01T12:00:00Z",
                "result": {"confidence": 0.9, "consensus_reached": True},
            },
            {
                "id": "debate-2",
                "timestamp": "2024-01-02T12:00:00Z",
                "result": {"confidence": 0.8, "consensus_reached": True},
            },
            {
                "id": "debate-3",
                "timestamp": "2024-01-03T12:00:00Z",
                "result": {"confidence": 0.7, "consensus_reached": False},
            },
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["total_debates"] == 3
        assert body["stats"]["consensus_reached_count"] == 2
        assert body["stats"]["consensus_rate"] == pytest.approx(2 / 3, rel=0.01)
        assert body["quality_score"] > 0
        assert body["quality_score"] <= 100

    def test_detects_declining_trend(self):
        """Should detect declining confidence trend."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        # Create declining confidence: first half high, second half low
        mock_storage.list_debates.return_value = [
            {
                "id": f"d{i}",
                "result": {"confidence": 0.9 if i < 5 else 0.5, "consensus_reached": True},
            }
            for i in range(10)
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["trend"] == "declining"

    def test_detects_improving_trend(self):
        """Should detect improving confidence trend."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        # Create improving confidence: first half low, second half high
        mock_storage.list_debates.return_value = [
            {
                "id": f"d{i}",
                "result": {"confidence": 0.5 if i < 5 else 0.9, "consensus_reached": True},
            }
            for i in range(10)
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["trend"] == "improving"

    def test_generates_critical_alert_low_quality(self):
        """Should generate critical alert for very low quality."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        # Create very low quality debates
        mock_storage.list_debates.return_value = [
            {"id": f"d{i}", "result": {"confidence": 0.1, "consensus_reached": False}}
            for i in range(10)
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["alert"] is not None
        assert body["alert"]["level"] == "critical"


# ===========================================================================
# Test _get_ranking_stats()
# ===========================================================================


class TestGetRankingStats:
    """Tests for _get_ranking_stats()."""

    def test_no_elo_returns_503(self):
        """Should return 503 when ranking system unavailable."""
        handler = create_analytics_handler()

        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._get_ranking_stats()

        assert get_status(result) == 503
        assert "not available" in get_body(result)["error"]

    def test_empty_leaderboard(self):
        """Should handle empty leaderboard."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_ranking_stats()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["total_agents"] == 0
        assert body["stats"]["top_agent"] is None

    def test_calculates_ranking_stats(self):
        """Should calculate ranking statistics from leaderboard."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()

        # Create mock agent stats
        class MockAgentStats:
            def __init__(self, name, elo, debates):
                self.agent_name = name
                self.elo = elo
                self.debates_count = debates

        mock_elo.get_leaderboard.return_value = [
            MockAgentStats("claude", 1600, 50),
            MockAgentStats("gpt", 1550, 40),
            MockAgentStats("gemini", 1500, 30),
        ]

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_ranking_stats()

        body = get_body(result)
        assert get_status(result) == 200
        stats = body["stats"]
        assert stats["total_agents"] == 3
        assert stats["total_matches"] == 120  # 50+40+30
        assert stats["top_agent"] == "claude"
        assert stats["elo_range"]["min"] == 1500
        assert stats["elo_range"]["max"] == 1600


# ===========================================================================
# Test _get_memory_stats()
# ===========================================================================


class TestGetMemoryStats:
    """Tests for _get_memory_stats()."""

    def test_no_nomic_dir_returns_empty(self):
        """Should return empty stats when nomic dir unavailable."""
        handler = create_analytics_handler()

        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._get_memory_stats()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"] == {}

    def test_detects_database_files(self):
        """Should detect presence of database files."""
        handler = create_analytics_handler()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test database files
            (tmp_path / "debate_embeddings.db").touch()
            (tmp_path / "continuum_memory.db").touch()
            # Note: insights.db not created

            with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
                result = handler._get_memory_stats()

            body = get_body(result)
            assert get_status(result) == 200
            assert body["stats"]["embeddings_db"] is True
            assert body["stats"]["continuum_memory"] is True
            assert body["stats"]["insights_db"] is False

    def test_no_database_files(self):
        """Should return False for missing database files."""
        handler = create_analytics_handler()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Empty directory - no database files

            with patch.object(handler, "get_nomic_dir", return_value=tmp_path):
                result = handler._get_memory_stats()

            body = get_body(result)
            assert get_status(result) == 200
            assert body["stats"]["embeddings_db"] is False
            assert body["stats"]["insights_db"] is False
            assert body["stats"]["continuum_memory"] is False


# ===========================================================================
# Test Route Dispatching
# ===========================================================================


class TestRouteDispatching:
    """Tests for handle() route dispatching."""

    async def test_routes_to_disagreement_stats(self):
        """Should route to _get_disagreement_stats."""
        handler = create_analytics_handler()
        mock_http = MockHandler()

        with patch.object(handler, "get_storage", return_value=None):
            result = await handler.handle("/api/v1/analytics/disagreements", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert "stats" in body

    async def test_routes_to_role_rotation_stats(self):
        """Should route to _get_role_rotation_stats."""
        handler = create_analytics_handler()
        mock_http = MockHandler()

        with patch.object(handler, "get_storage", return_value=None):
            result = await handler.handle("/api/v1/analytics/role-rotation", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert "stats" in body

    async def test_returns_none_for_unknown_route(self):
        """Should return None for unhandled routes."""
        handler = create_analytics_handler()
        mock_http = MockHandler()

        result = await handler.handle("/api/v1/unknown/route", {}, mock_http)

        assert result is None
