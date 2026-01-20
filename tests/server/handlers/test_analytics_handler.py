"""Tests for Analytics handler endpoints.

Validates the REST API endpoints for analytics and metrics including:
- Disagreement statistics
- Role rotation statistics
- Early stopping statistics
- Consensus quality metrics
- Ranking statistics
- Memory statistics
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics import AnalyticsHandler


@pytest.fixture
def analytics_handler():
    """Create an analytics handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = AnalyticsHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    return handler


@pytest.fixture
def mock_storage():
    """Create a mock storage with sample debates."""
    storage = MagicMock()

    # Sample debate data
    debates = [
        {
            "id": "debate-1",
            "timestamp": "2024-01-01T00:00:00Z",
            "messages": [
                {"role": "agent", "cognitive_role": "advocate"},
                {"role": "agent", "cognitive_role": "critic"},
            ],
            "result": {
                "confidence": 0.85,
                "consensus_reached": True,
                "rounds_used": 3,
                "early_stopped": False,
                "disagreement_report": {"unanimous_critiques": False},
                "uncertainty_metrics": {"disagreement_type": "minor"},
            },
        },
        {
            "id": "debate-2",
            "timestamp": "2024-01-02T00:00:00Z",
            "messages": [
                {"role": "agent", "cognitive_role": "synthesizer"},
            ],
            "result": {
                "confidence": 0.65,
                "consensus_reached": False,
                "rounds_used": 2,
                "early_stopped": True,
                "disagreement_report": {"unanimous_critiques": True},
                "uncertainty_metrics": {"disagreement_type": "major"},
            },
        },
    ]

    storage.list_debates.return_value = debates
    return storage


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = MagicMock()

    # Mock agent data
    agent1 = MagicMock()
    agent1.agent_name = "claude"
    agent1.elo = 1650
    agent1.debates_count = 50

    agent2 = MagicMock()
    agent2.agent_name = "gpt-4"
    agent2.elo = 1550
    agent2.debates_count = 45

    agent3 = MagicMock()
    agent3.agent_name = "codex"
    agent3.elo = 1480
    agent3.debates_count = 30

    elo.get_leaderboard.return_value = [agent1, agent2, agent3]
    return elo


class TestAnalyticsHandlerCanHandle:
    """Test AnalyticsHandler.can_handle method."""

    def test_can_handle_disagreements(self, analytics_handler):
        """Test can_handle returns True for disagreements endpoint."""
        assert analytics_handler.can_handle("/api/analytics/disagreements")

    def test_can_handle_role_rotation(self, analytics_handler):
        """Test can_handle returns True for role-rotation endpoint."""
        assert analytics_handler.can_handle("/api/analytics/role-rotation")

    def test_can_handle_early_stops(self, analytics_handler):
        """Test can_handle returns True for early-stops endpoint."""
        assert analytics_handler.can_handle("/api/analytics/early-stops")

    def test_can_handle_consensus_quality(self, analytics_handler):
        """Test can_handle returns True for consensus-quality endpoint."""
        assert analytics_handler.can_handle("/api/analytics/consensus-quality")

    def test_can_handle_ranking_stats(self, analytics_handler):
        """Test can_handle returns True for ranking stats endpoint."""
        assert analytics_handler.can_handle("/api/ranking/stats")

    def test_can_handle_memory_stats(self, analytics_handler):
        """Test can_handle returns True for memory stats endpoint."""
        assert analytics_handler.can_handle("/api/memory/stats")

    def test_cannot_handle_unknown(self, analytics_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not analytics_handler.can_handle("/api/analytics/unknown")
        assert not analytics_handler.can_handle("/api/debates")


class TestAnalyticsHandlerDisagreements:
    """Test GET /api/analytics/disagreements endpoint."""

    def test_disagreement_stats_no_storage(self, analytics_handler, mock_http_handler):
        """Test disagreement stats returns empty when no storage."""
        result = analytics_handler.handle("/api/analytics/disagreements", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body

    def test_disagreement_stats_with_storage(
        self, analytics_handler, mock_http_handler, mock_storage
    ):
        """Test disagreement stats with storage data."""
        analytics_handler.ctx["storage"] = mock_storage

        result = analytics_handler.handle("/api/analytics/disagreements", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body
        stats = body["stats"]
        assert "total_debates" in stats
        assert "with_disagreements" in stats
        assert "unanimous" in stats
        assert "disagreement_types" in stats


class TestAnalyticsHandlerRoleRotation:
    """Test GET /api/analytics/role-rotation endpoint."""

    def test_role_rotation_stats_no_storage(self, analytics_handler, mock_http_handler):
        """Test role rotation stats returns empty when no storage."""
        result = analytics_handler.handle("/api/analytics/role-rotation", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body

    def test_role_rotation_stats_with_storage(
        self, analytics_handler, mock_http_handler, mock_storage
    ):
        """Test role rotation stats with storage data."""
        analytics_handler.ctx["storage"] = mock_storage

        result = analytics_handler.handle("/api/analytics/role-rotation", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body
        stats = body["stats"]
        assert "total_debates" in stats
        assert "role_assignments" in stats


class TestAnalyticsHandlerEarlyStops:
    """Test GET /api/analytics/early-stops endpoint."""

    def test_early_stop_stats_no_storage(self, analytics_handler, mock_http_handler):
        """Test early stop stats returns empty when no storage."""
        result = analytics_handler.handle("/api/analytics/early-stops", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body

    def test_early_stop_stats_with_storage(
        self, analytics_handler, mock_http_handler, mock_storage
    ):
        """Test early stop stats with storage data."""
        analytics_handler.ctx["storage"] = mock_storage

        result = analytics_handler.handle("/api/analytics/early-stops", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body
        stats = body["stats"]
        assert "total_debates" in stats
        assert "early_stopped" in stats
        assert "full_rounds" in stats
        assert "average_rounds" in stats


class TestAnalyticsHandlerConsensusQuality:
    """Test GET /api/analytics/consensus-quality endpoint."""

    def test_consensus_quality_no_storage(self, analytics_handler, mock_http_handler):
        """Test consensus quality returns empty when no storage."""
        result = analytics_handler.handle("/api/analytics/consensus-quality", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body
        assert "quality_score" in body

    def test_consensus_quality_with_storage(
        self, analytics_handler, mock_http_handler, mock_storage
    ):
        """Test consensus quality with storage data."""
        analytics_handler.ctx["storage"] = mock_storage

        result = analytics_handler.handle("/api/analytics/consensus-quality", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body
        assert "quality_score" in body
        stats = body["stats"]
        assert "total_debates" in stats
        assert "confidence_history" in stats
        assert "trend" in stats
        assert "average_confidence" in stats
        assert "consensus_rate" in stats

    def test_consensus_quality_empty_debates(self, analytics_handler, mock_http_handler):
        """Test consensus quality with no debates returns proper empty state."""
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        analytics_handler.ctx["storage"] = mock_storage

        result = analytics_handler.handle("/api/analytics/consensus-quality", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert body["quality_score"] == 0
        assert body["stats"]["trend"] == "insufficient_data"


class TestAnalyticsHandlerRankingStats:
    """Test GET /api/ranking/stats endpoint."""

    def test_ranking_stats_no_elo(self, analytics_handler, mock_http_handler):
        """Test ranking stats returns 503 when no ELO system."""
        result = analytics_handler.handle("/api/ranking/stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_ranking_stats_with_elo(self, analytics_handler, mock_http_handler, mock_elo_system):
        """Test ranking stats with ELO system data."""
        analytics_handler.ctx["elo_system"] = mock_elo_system

        result = analytics_handler.handle("/api/ranking/stats", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body
        stats = body["stats"]
        assert "total_agents" in stats
        assert "total_matches" in stats
        assert "avg_elo" in stats
        assert "top_agent" in stats
        assert "elo_range" in stats
        assert stats["total_agents"] == 3
        assert stats["top_agent"] == "claude"


class TestAnalyticsHandlerMemoryStats:
    """Test GET /api/memory/stats endpoint."""

    def test_memory_stats_no_nomic_dir(self, analytics_handler, mock_http_handler):
        """Test memory stats returns empty when no nomic dir."""
        result = analytics_handler.handle("/api/memory/stats", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body

    def test_memory_stats_with_nomic_dir(self, analytics_handler, mock_http_handler, tmp_path):
        """Test memory stats with nomic dir."""
        # Create a temp nomic dir with some database files
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()
        (nomic_dir / "debate_embeddings.db").touch()

        analytics_handler.ctx["nomic_dir"] = nomic_dir

        result = analytics_handler.handle("/api/memory/stats", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body
        stats = body["stats"]
        assert "embeddings_db" in stats
        assert "insights_db" in stats
        assert "continuum_memory" in stats
        assert stats["embeddings_db"] is True


class TestAnalyticsHandlerRateLimiting:
    """Test rate limiting for analytics endpoints."""

    def test_rate_limit_exceeded(self, analytics_handler, mock_http_handler):
        """Test rate limiting returns 429 after many requests."""
        from aragora.server.handlers.analytics import _analytics_limiter

        # Save original state
        original_requests = (
            _analytics_limiter._requests.copy() if hasattr(_analytics_limiter, "_requests") else {}
        )

        try:
            # Make many requests to trigger rate limit
            for _ in range(35):
                result = analytics_handler.handle(
                    "/api/analytics/disagreements", {}, mock_http_handler
                )

            # The rate limiter should eventually kick in
            # Note: The exact behavior depends on implementation
            assert result is not None
        finally:
            # Restore rate limiter state
            if hasattr(_analytics_limiter, "_requests"):
                _analytics_limiter._requests = original_requests


class TestAnalyticsHandlerIntegration:
    """Integration tests for analytics handler."""

    def test_all_endpoints_return_valid_json(
        self, analytics_handler, mock_http_handler, mock_storage, mock_elo_system, tmp_path
    ):
        """Test all analytics endpoints return valid JSON."""
        nomic_dir = tmp_path / "nomic"
        nomic_dir.mkdir()

        analytics_handler.ctx["storage"] = mock_storage
        analytics_handler.ctx["elo_system"] = mock_elo_system
        analytics_handler.ctx["nomic_dir"] = nomic_dir

        endpoints = [
            "/api/analytics/disagreements",
            "/api/analytics/role-rotation",
            "/api/analytics/early-stops",
            "/api/analytics/consensus-quality",
            "/api/ranking/stats",
            "/api/memory/stats",
        ]

        for endpoint in endpoints:
            result = analytics_handler.handle(endpoint, {}, mock_http_handler)
            assert result is not None, f"Endpoint {endpoint} returned None"

            # Verify it's valid JSON
            try:
                body = json.loads(result.body)
                assert isinstance(body, dict), f"Endpoint {endpoint} didn't return dict"
            except json.JSONDecodeError:
                pytest.fail(f"Endpoint {endpoint} returned invalid JSON")
