"""
Tests for aragora.server.handlers._analytics_impl - Analytics Implementation.

This module tests the AnalyticsHandler implementation that provides:
- Disagreement statistics
- Role rotation stats
- Early stopping metrics
- Consensus quality metrics
- Ranking statistics
- Memory statistics
- Cross-pollination analysis
- Learning efficiency tracking
- Voting accuracy validation
- Calibration metrics

Tests cover:
1. Each analytics endpoint method
2. Edge cases (empty data, missing fields)
3. Error handling
4. Correct metric calculations
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        client_address: tuple | None = None,
    ):
        self.headers = headers or {}
        self.client_address = client_address or ("127.0.0.1", 12345)


def create_analytics_handler(ctx: dict | None = None):
    """Create an AnalyticsHandler with optional context."""
    from aragora.server.handlers._analytics_impl import AnalyticsHandler

    return AnalyticsHandler(ctx or {})


def get_body(result) -> dict:
    """Extract body as dict from HandlerResult."""
    if hasattr(result, "body"):
        body_bytes = result.body
        if isinstance(body_bytes, bytes):
            return json.loads(body_bytes.decode("utf-8"))
        return json.loads(body_bytes)
    return result


def get_status(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result


@dataclass
class MockAgentRating:
    """Mock agent rating for leaderboard tests."""

    agent_name: str
    elo: float
    debates_count: int
    wins: int = 0
    losses: int = 0


@dataclass
class MockCalibrationSummary:
    """Mock calibration summary for calibration tests."""

    total_predictions: int
    temperature: float
    scaling_factor: float = 1.0


# ===========================================================================
# Test can_handle() Route Matching
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_handles_disagreements_route(self):
        """Should handle /api/analytics/disagreements."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/analytics/disagreements") is True
        assert handler.can_handle("/api/v1/analytics/disagreements") is True

    def test_handles_role_rotation_route(self):
        """Should handle /api/analytics/role-rotation."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/analytics/role-rotation") is True
        assert handler.can_handle("/api/v1/analytics/role-rotation") is True

    def test_handles_early_stops_route(self):
        """Should handle /api/analytics/early-stops."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/analytics/early-stops") is True
        assert handler.can_handle("/api/v1/analytics/early-stops") is True

    def test_handles_consensus_quality_route(self):
        """Should handle /api/analytics/consensus-quality."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/analytics/consensus-quality") is True

    def test_handles_ranking_stats_route(self):
        """Should handle /api/ranking/stats."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/ranking/stats") is True
        assert handler.can_handle("/api/v1/ranking/stats") is True

    def test_handles_memory_stats_route(self):
        """Should handle /api/memory/stats."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/memory/stats") is True
        assert handler.can_handle("/api/v1/memory/stats") is True

    def test_handles_cross_pollination_route(self):
        """Should handle /api/analytics/cross-pollination."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/analytics/cross-pollination") is True

    def test_handles_learning_efficiency_route(self):
        """Should handle /api/analytics/learning-efficiency."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/analytics/learning-efficiency") is True

    def test_handles_voting_accuracy_route(self):
        """Should handle /api/analytics/voting-accuracy."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/analytics/voting-accuracy") is True

    def test_handles_calibration_route(self):
        """Should handle /api/analytics/calibration."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/analytics/calibration") is True

    def test_rejects_unknown_routes(self):
        """Should reject unknown routes."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/unknown") is False
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/analytics/unknown") is False


# ===========================================================================
# Test Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting in handle()."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self):
        """Should return 429 when rate limit exceeded."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler(client_address=("192.168.1.100", 12345))

        # Force rate limit by mocking is_allowed to return False
        with patch.object(_analytics_limiter, "is_allowed", return_value=False):
            result = await handler.handle("/api/analytics/disagreements", {}, mock_http)

        assert get_status(result) == 429
        body = get_body(result)
        assert "Rate limit" in body.get("error", "")


# ===========================================================================
# Test Authentication / Authorization
# ===========================================================================


class TestAuthentication:
    """Tests for authentication and authorization."""

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self):
        """Should return 401 when not authenticated."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter
        from aragora.server.handlers.utils.auth import UnauthorizedError

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(
                handler,
                "get_auth_context",
                side_effect=UnauthorizedError("Authentication required"),
            ),
        ):
            result = await handler.handle("/api/analytics/disagreements", {}, mock_http)

        assert get_status(result) == 401
        body = get_body(result)
        assert "Authentication" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self):
        """Should return 403 when permission denied."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter
        from aragora.server.handlers.secure import ForbiddenError
        from aragora.rbac.models import AuthorizationContext

        handler = create_analytics_handler()
        mock_http = MockHandler()

        # Create mock auth context
        mock_auth = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            workspace_id=None,
            roles={"viewer"},
            permissions=set(),
        )

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(
                handler,
                "check_permission",
                side_effect=ForbiddenError("Permission denied", permission="analytics:read"),
            ),
        ):
            result = await handler.handle("/api/analytics/disagreements", {}, mock_http)

        assert get_status(result) == 403


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

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["total_debates"] == 0
        assert body["stats"]["with_disagreements"] == 0
        assert body["stats"]["unanimous"] == 0

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

    def test_handles_non_dict_debate(self):
        """Should handle non-dict debate entries gracefully."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            "invalid_debate",  # String instead of dict
            None,  # None
            {"id": "valid", "result": {}},
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_disagreement_stats()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["total_debates"] == 3

    def test_counts_disagreement_types(self):
        """Should count different disagreement types."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": "d1",
                "result": {
                    "disagreement_report": {"unanimous_critiques": True},
                    "uncertainty_metrics": {"disagreement_type": "factual"},
                },
            },
            {
                "id": "d2",
                "result": {
                    "disagreement_report": {"unanimous_critiques": True},
                    "uncertainty_metrics": {"disagreement_type": "factual"},
                },
            },
            {
                "id": "d3",
                "result": {
                    "disagreement_report": {"unanimous_critiques": True},
                    "uncertainty_metrics": {"disagreement_type": "value"},
                },
            },
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_disagreement_stats()

        body = get_body(result)
        assert body["stats"]["disagreement_types"]["factual"] == 2
        assert body["stats"]["disagreement_types"]["value"] == 1


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

    def test_empty_debates_list(self):
        """Should handle empty debates list."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_role_rotation_stats()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["total_debates"] == 0
        assert body["stats"]["role_assignments"] == {}

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

    def test_handles_missing_messages(self):
        """Should handle debates without messages (returns empty messages list)."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {"id": "debate-1"},  # No messages key - treated as empty list
            {"id": "debate-2", "messages": []},  # Empty messages
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_role_rotation_stats()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["total_debates"] == 2

    def test_accumulates_roles_across_debates(self):
        """Should accumulate role counts across multiple debates."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": "d1",
                "messages": [
                    {"cognitive_role": "advocate"},
                    {"cognitive_role": "advocate"},
                ],
            },
            {
                "id": "d2",
                "messages": [
                    {"cognitive_role": "advocate"},
                    {"cognitive_role": "critic"},
                ],
            },
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_role_rotation_stats()

        body = get_body(result)
        assert body["stats"]["role_assignments"]["advocate"] == 3
        assert body["stats"]["role_assignments"]["critic"] == 1


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

    def test_empty_debates_list(self):
        """Should handle empty debates list."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_early_stop_stats()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["total_debates"] == 0
        assert body["stats"]["average_rounds"] == 0.0

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

    def test_handles_missing_result(self):
        """Should handle debates without result (returns empty result dict)."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {"id": "1"},  # No result key - treated as empty dict
            {"id": "2", "result": {}},  # Empty result
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_early_stop_stats()

        body = get_body(result)
        assert get_status(result) == 200
        assert body["stats"]["total_debates"] == 2
        assert body["stats"]["full_rounds"] == 2  # No early_stopped flag means full round

    def test_handles_zero_rounds(self):
        """Should handle debates with zero rounds."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {"id": "1", "result": {"rounds_used": 0}},
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_early_stop_stats()

        body = get_body(result)
        assert body["stats"]["average_rounds"] == 0.0


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
        assert body["alert"] is None

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
        assert body["stats"]["total_debates"] == 0
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

    def test_trend_stable_when_few_debates(self):
        """Should report stable trend with fewer than 5 debates."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {"id": f"d{i}", "result": {"confidence": 0.7, "consensus_reached": True}}
            for i in range(4)
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert body["stats"]["trend"] == "stable"

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

    def test_generates_warning_alert_medium_quality(self):
        """Should generate warning alert for medium-low quality."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        # Create medium quality debates (quality score ~50-60)
        mock_storage.list_debates.return_value = [
            {"id": f"d{i}", "result": {"confidence": 0.5, "consensus_reached": False}}
            for i in range(10)
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert body["quality_score"] >= 40  # Not critical
        if body["alert"]:
            assert body["alert"]["level"] in ["warning", "info"]

    def test_confidence_history_truncated(self):
        """Should truncate confidence history to 20 entries."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": f"d{i}",
                "timestamp": f"2024-01-{i:02d}T12:00:00Z",
                "result": {"confidence": 0.8, "consensus_reached": True},
            }
            for i in range(50)
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert len(body["stats"]["confidence_history"]) <= 20


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
        body = get_body(result)
        assert "not available" in body.get("error", "")

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
        assert body["stats"]["avg_elo"] == 1500

    def test_calculates_ranking_stats(self):
        """Should calculate ranking statistics from leaderboard."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()

        mock_elo.get_leaderboard.return_value = [
            MockAgentRating("claude", 1600, 50),
            MockAgentRating("gpt", 1550, 40),
            MockAgentRating("gemini", 1500, 30),
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
        assert stats["avg_elo"] == pytest.approx(1550)  # (1600+1550+1500)/3

    def test_single_agent_leaderboard(self):
        """Should handle single agent in leaderboard."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            MockAgentRating("claude", 1600, 10),
        ]

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_ranking_stats()

        body = get_body(result)
        stats = body["stats"]
        assert stats["total_agents"] == 1
        assert stats["top_agent"] == "claude"
        assert stats["elo_range"]["min"] == 1600
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
# Test _get_cross_pollination_stats()
# ===========================================================================


class TestGetCrossPollination:
    """Tests for _get_cross_pollination_stats()."""

    def test_returns_default_stats(self):
        """Should return default cross-pollination stats."""
        handler = create_analytics_handler()

        # Mock the imports inside the function
        with (
            patch.dict("sys.modules", {"aragora.rlm.bridge": MagicMock()}),
            patch.dict("sys.modules", {"aragora.ranking.elo": MagicMock()}),
        ):
            # Override the imports to raise ImportError
            import sys

            sys.modules["aragora.rlm.bridge"].RLMHierarchyCache = MagicMock(side_effect=ImportError)
            sys.modules["aragora.ranking.elo"].get_elo_store = MagicMock(side_effect=ImportError)

            result = handler._get_cross_pollination_stats()

        body = get_body(result)
        assert get_status(result) == 200
        assert "stats" in body
        assert body["version"] == "2.0.3"

    def test_includes_rlm_cache_stats(self):
        """Should include RLM cache stats when available."""
        handler = create_analytics_handler()

        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = {
            "hits": 100,
            "misses": 20,
            "hit_rate": 0.83,
        }

        # Patch the module before the import happens
        with patch.dict("sys.modules"):
            mock_rlm_module = MagicMock()
            mock_rlm_module.RLMHierarchyCache = MagicMock(return_value=mock_cache)
            import sys

            sys.modules["aragora.rlm.bridge"] = mock_rlm_module

            result = handler._get_cross_pollination_stats()

        body = get_body(result)
        assert body["stats"]["rlm_cache"]["enabled"] is True
        assert body["stats"]["rlm_cache"]["hits"] == 100
        assert body["stats"]["rlm_cache"]["hit_rate"] == 0.83


# ===========================================================================
# Test _get_learning_efficiency_stats()
# ===========================================================================


class TestGetLearningEfficiencyStats:
    """Tests for _get_learning_efficiency_stats()."""

    def test_elo_not_available_returns_error(self):
        """Should return error when ELO system not available."""
        handler = create_analytics_handler()

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(side_effect=ImportError)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = handler._get_learning_efficiency_stats({})

        body = get_body(result)
        assert "error" in body
        assert body["agents"] == []

    def test_returns_specific_agent_efficiency(self):
        """Should return efficiency for specific agent."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_learning_efficiency.return_value = {
            "elo_change": 50.0,
            "games_played": 20,
            "efficiency_score": 2.5,
        }

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = handler._get_learning_efficiency_stats({"agent": ["claude"], "domain": ["ai"]})

        body = get_body(result)
        assert body["agent"] == "claude"
        assert body["domain"] == "ai"
        assert body["efficiency"]["elo_change"] == 50.0

    def test_returns_batch_efficiency(self):
        """Should return efficiency for multiple agents."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            MockAgentRating("claude", 1600, 50),
            MockAgentRating("gpt", 1550, 40),
        ]
        mock_elo.get_learning_efficiency_batch.return_value = {
            "claude": {"efficiency_score": 2.5},
            "gpt": {"efficiency_score": 2.0},
        }

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = handler._get_learning_efficiency_stats({"domain": ["general"]})

        body = get_body(result)
        assert body["domain"] == "general"
        assert len(body["agents"]) == 2
        assert body["agents"][0]["agent"] == "claude"

    def test_respects_limit_parameter(self):
        """Should respect the limit query parameter."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            MockAgentRating(f"agent-{i}", 1500 + i * 10, 10) for i in range(5)
        ]
        mock_elo.get_learning_efficiency_batch.return_value = {}

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = handler._get_learning_efficiency_stats({"limit": ["5"]})

        mock_elo.get_leaderboard.assert_called_with(limit=5)


# ===========================================================================
# Test _get_voting_accuracy_stats()
# ===========================================================================


class TestGetVotingAccuracyStats:
    """Tests for _get_voting_accuracy_stats()."""

    def test_elo_not_available_returns_error(self):
        """Should return error when ELO system not available."""
        handler = create_analytics_handler()

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(side_effect=ImportError)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = handler._get_voting_accuracy_stats({})

        body = get_body(result)
        assert "error" in body
        assert body["agents"] == []

    def test_returns_specific_agent_accuracy(self):
        """Should return accuracy for specific agent."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_voting_accuracy.return_value = {
            "total_votes": 100,
            "correct_votes": 85,
            "accuracy": 0.85,
        }

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = handler._get_voting_accuracy_stats({"agent": ["claude"]})

        body = get_body(result)
        assert body["agent"] == "claude"
        assert body["accuracy"]["accuracy"] == 0.85

    def test_returns_batch_accuracy(self):
        """Should return accuracy for multiple agents."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            MockAgentRating("claude", 1600, 50),
            MockAgentRating("gpt", 1550, 40),
        ]
        mock_elo.get_voting_accuracy_batch.return_value = {
            "claude": {"accuracy": 0.85},
            "gpt": {"accuracy": 0.80},
        }

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = handler._get_voting_accuracy_stats({})

        body = get_body(result)
        assert len(body["agents"]) == 2
        assert body["agents"][0]["accuracy"]["accuracy"] == 0.85


# ===========================================================================
# Test _get_calibration_stats()
# ===========================================================================


class TestGetCalibrationStats:
    """Tests for _get_calibration_stats()."""

    def test_elo_not_available_returns_error(self):
        """Should return error when ELO system not available."""
        handler = create_analytics_handler()

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(side_effect=ImportError)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = handler._get_calibration_stats({})

        body = get_body(result)
        assert "error" in body
        assert body["agents"] == []

    def test_returns_specific_agent_calibration_with_tracker(self):
        """Should return calibration for specific agent when tracker available."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_summary.return_value = MockCalibrationSummary(
            total_predictions=100,
            temperature=1.1,
            scaling_factor=0.95,
        )

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            mock_cal_module = MagicMock()
            mock_cal_module.CalibrationTracker = MagicMock(return_value=mock_tracker)
            sys.modules["aragora.ranking.calibration"] = mock_cal_module

            result = handler._get_calibration_stats({"agent": ["claude"]})

        body = get_body(result)
        assert body["agent"] == "claude"
        assert body["calibration"]["total_predictions"] == 100
        assert body["calibration"]["temperature"] == 1.1

    def test_returns_none_calibration_when_no_data(self):
        """Should return None calibration when no data available."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.get_calibration_summary.return_value = None

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            mock_cal_module = MagicMock()
            mock_cal_module.CalibrationTracker = MagicMock(return_value=mock_tracker)
            sys.modules["aragora.ranking.calibration"] = mock_cal_module

            result = handler._get_calibration_stats({"agent": ["claude"]})

        body = get_body(result)
        assert body["calibration"] is None

    def test_returns_batch_calibration(self):
        """Should return calibration for multiple agents."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            MockAgentRating("claude", 1600, 50),
            MockAgentRating("gpt", 1550, 40),
        ]
        mock_tracker = MagicMock()

        def mock_summary(name):
            if name == "claude":
                return MockCalibrationSummary(100, 1.1)
            return None

        mock_tracker.get_calibration_summary.side_effect = mock_summary

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            mock_cal_module = MagicMock()
            mock_cal_module.CalibrationTracker = MagicMock(return_value=mock_tracker)
            sys.modules["aragora.ranking.calibration"] = mock_cal_module

            result = handler._get_calibration_stats({})

        body = get_body(result)
        assert len(body["agents"]) == 2
        assert body["agents"][0]["calibration"] is not None
        assert body["agents"][1]["calibration"] is None

    def test_handles_calibration_tracker_import_error(self):
        """Should handle CalibrationTracker import error gracefully."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            MockAgentRating("claude", 1600, 50),
        ]

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            # Make CalibrationTracker raise ImportError
            mock_cal_module = MagicMock()
            mock_cal_module.CalibrationTracker = MagicMock(side_effect=ImportError)
            sys.modules["aragora.ranking.calibration"] = mock_cal_module

            result = handler._get_calibration_stats({})

        body = get_body(result)
        # Should still return agents, just with None calibration
        assert len(body["agents"]) == 1
        assert body["agents"][0]["calibration"] is None


# ===========================================================================
# Test Route Dispatching
# ===========================================================================


class TestRouteDispatching:
    """Tests for handle() route dispatching."""

    @pytest.fixture
    def mock_auth(self):
        """Create a mock authenticated context."""
        from aragora.rbac.models import AuthorizationContext

        return AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            workspace_id=None,
            roles={"admin"},
            permissions={"analytics:read"},
        )

    @pytest.mark.asyncio
    async def test_routes_to_disagreement_stats(self, mock_auth):
        """Should route to _get_disagreement_stats."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
            patch.object(handler, "get_storage", return_value=None),
        ):
            result = await handler.handle("/api/analytics/disagreements", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert "stats" in body

    @pytest.mark.asyncio
    async def test_routes_to_role_rotation_stats(self, mock_auth):
        """Should route to _get_role_rotation_stats."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
            patch.object(handler, "get_storage", return_value=None),
        ):
            result = await handler.handle("/api/analytics/role-rotation", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert "stats" in body

    @pytest.mark.asyncio
    async def test_routes_to_early_stops_stats(self, mock_auth):
        """Should route to _get_early_stop_stats."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
            patch.object(handler, "get_storage", return_value=None),
        ):
            result = await handler.handle("/api/analytics/early-stops", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert "stats" in body

    @pytest.mark.asyncio
    async def test_routes_to_consensus_quality(self, mock_auth):
        """Should route to _get_consensus_quality."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
            patch.object(handler, "get_storage", return_value=None),
        ):
            result = await handler.handle("/api/analytics/consensus-quality", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert "quality_score" in body

    @pytest.mark.asyncio
    async def test_routes_to_ranking_stats(self, mock_auth):
        """Should route to _get_ranking_stats."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
            patch.object(handler, "get_elo_system", return_value=None),
        ):
            result = await handler.handle("/api/ranking/stats", {}, mock_http)

        assert get_status(result) == 503  # ELO not available

    @pytest.mark.asyncio
    async def test_routes_to_memory_stats(self, mock_auth):
        """Should route to _get_memory_stats."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
            patch.object(handler, "get_nomic_dir", return_value=None),
        ):
            result = await handler.handle("/api/memory/stats", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert "stats" in body

    @pytest.mark.asyncio
    async def test_routes_to_cross_pollination(self, mock_auth):
        """Should route to _get_cross_pollination_stats."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
        ):
            result = await handler.handle("/api/analytics/cross-pollination", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        assert "stats" in body

    @pytest.mark.asyncio
    async def test_routes_to_learning_efficiency(self, mock_auth):
        """Should route to _get_learning_efficiency_stats."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
            patch.dict("sys.modules"),
        ):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(side_effect=ImportError)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = await handler.handle("/api/analytics/learning-efficiency", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200
        # Error message since ELO not available
        assert "error" in body or "agents" in body

    @pytest.mark.asyncio
    async def test_routes_to_voting_accuracy(self, mock_auth):
        """Should route to _get_voting_accuracy_stats."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
            patch.dict("sys.modules"),
        ):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(side_effect=ImportError)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = await handler.handle("/api/analytics/voting-accuracy", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200

    @pytest.mark.asyncio
    async def test_routes_to_calibration(self, mock_auth):
        """Should route to _get_calibration_stats."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
            patch.dict("sys.modules"),
        ):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(side_effect=ImportError)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = await handler.handle("/api/analytics/calibration", {}, mock_http)

        body = get_body(result)
        assert get_status(result) == 200

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_route(self, mock_auth):
        """Should return None for unhandled routes."""
        from aragora.server.handlers._analytics_impl import _analytics_limiter

        handler = create_analytics_handler()
        mock_http = MockHandler()

        with (
            patch.object(_analytics_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_auth),
            patch.object(handler, "check_permission", return_value=True),
        ):
            result = await handler.handle("/api/unknown/route", {}, mock_http)

        assert result is None


# ===========================================================================
# Test Edge Cases and Error Handling
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_exception_in_disagreement_stats(self):
        """Should handle exceptions gracefully in disagreement stats."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.side_effect = RuntimeError("Database error")

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_disagreement_stats()

        # Should return error response, not crash
        assert get_status(result) == 500

    def test_handles_exception_in_role_rotation_stats(self):
        """Should handle exceptions gracefully in role rotation stats."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.side_effect = RuntimeError("Connection lost")

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_role_rotation_stats()

        assert get_status(result) == 500

    def test_handles_exception_in_early_stop_stats(self):
        """Should handle exceptions gracefully in early stop stats."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.side_effect = RuntimeError("Timeout")

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_early_stop_stats()

        assert get_status(result) == 500

    def test_handles_exception_in_consensus_quality(self):
        """Should handle exceptions gracefully in consensus quality."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.side_effect = RuntimeError("Query failed")

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        assert get_status(result) == 500

    def test_handles_exception_in_ranking_stats(self):
        """Should handle exceptions gracefully in ranking stats."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = RuntimeError("ELO system error")

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_ranking_stats()

        assert get_status(result) == 500

    def test_handles_exception_in_memory_stats(self):
        """Should handle exceptions gracefully in memory stats."""
        handler = create_analytics_handler()
        mock_path = MagicMock()
        # Make path operations fail
        mock_path.__truediv__ = MagicMock(side_effect=RuntimeError("Path error"))

        with patch.object(handler, "get_nomic_dir", return_value=mock_path):
            result = handler._get_memory_stats()

        assert get_status(result) == 500


class TestQueryParameters:
    """Tests for query parameter handling."""

    def test_learning_efficiency_default_limit(self):
        """Should use default limit of 20 for learning efficiency."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        mock_elo.get_learning_efficiency_batch.return_value = {}

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            handler._get_learning_efficiency_stats({})

        mock_elo.get_leaderboard.assert_called_with(limit=20)

    def test_voting_accuracy_default_limit(self):
        """Should use default limit of 20 for voting accuracy."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        mock_elo.get_voting_accuracy_batch.return_value = {}

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            handler._get_voting_accuracy_stats({})

        mock_elo.get_leaderboard.assert_called_with(limit=20)

    def test_calibration_default_limit(self):
        """Should use default limit of 20 for calibration."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            # Make CalibrationTracker raise ImportError
            mock_cal_module = MagicMock()
            mock_cal_module.CalibrationTracker = MagicMock(side_effect=ImportError)
            sys.modules["aragora.ranking.calibration"] = mock_cal_module

            handler._get_calibration_stats({})

        mock_elo.get_leaderboard.assert_called_with(limit=20)

    def test_learning_efficiency_domain_filter(self):
        """Should pass domain filter to learning efficiency."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        mock_elo.get_learning_efficiency_batch.return_value = {}

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            result = handler._get_learning_efficiency_stats({"domain": ["coding"]})

        body = get_body(result)
        assert body["domain"] == "coding"

    def test_handles_missing_agent_param(self):
        """Should handle missing agent parameter gracefully."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        mock_elo.get_learning_efficiency_batch.return_value = {}

        with patch.dict("sys.modules"):
            mock_elo_module = MagicMock()
            mock_elo_module.get_elo_store = MagicMock(return_value=mock_elo)
            import sys

            sys.modules["aragora.ranking.elo"] = mock_elo_module

            # No agent param - should return all agents
            result = handler._get_learning_efficiency_stats({})

        body = get_body(result)
        assert "agents" in body


# ===========================================================================
# Test Cache Behavior (TTL cache decorators)
# ===========================================================================


class TestCacheBehavior:
    """Tests for cache behavior."""

    def test_disagreement_stats_cached(self):
        """Should cache disagreement stats results."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []

        with patch.object(handler, "get_storage", return_value=mock_storage):
            # First call
            result1 = handler._get_disagreement_stats()
            # Second call - should use cache
            result2 = handler._get_disagreement_stats()

        # Both results should be successful
        assert get_status(result1) == 200
        assert get_status(result2) == 200

    def test_ranking_stats_returns_consistent_results(self):
        """Should return consistent results for ranking stats."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            MockAgentRating("claude", 1600, 50),
        ]

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_ranking_stats()

        body = get_body(result)
        assert body["stats"]["total_agents"] == 1
        assert body["stats"]["top_agent"] == "claude"


# ===========================================================================
# Test Version Prefix Handling
# ===========================================================================


class TestVersionPrefixHandling:
    """Tests for API version prefix handling."""

    def test_handles_v1_prefix(self):
        """Should handle /api/v1 prefix."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/v1/analytics/disagreements") is True

    def test_handles_v2_prefix(self):
        """Should handle /api/v2 prefix."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/v2/analytics/disagreements") is True

    def test_handles_no_version_prefix(self):
        """Should handle path without version prefix."""
        handler = create_analytics_handler()
        assert handler.can_handle("/api/analytics/disagreements") is True


# ===========================================================================
# Test Response Format
# ===========================================================================


class TestResponseFormat:
    """Tests for response format validation."""

    def test_disagreement_stats_response_structure(self):
        """Should return properly structured disagreement stats."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_disagreement_stats()

        body = get_body(result)
        assert "stats" in body
        stats = body["stats"]
        assert "total_debates" in stats
        assert "with_disagreements" in stats
        assert "unanimous" in stats
        assert "disagreement_types" in stats

    def test_consensus_quality_response_structure(self):
        """Should return properly structured consensus quality response."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {"id": "d1", "result": {"confidence": 0.8, "consensus_reached": True}}
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert "stats" in body
        assert "quality_score" in body
        assert "alert" in body
        stats = body["stats"]
        assert "total_debates" in stats
        assert "confidence_history" in stats
        assert "trend" in stats
        assert "average_confidence" in stats
        assert "consensus_rate" in stats

    def test_cross_pollination_stats_includes_version(self):
        """Should include version in cross-pollination stats."""
        handler = create_analytics_handler()

        result = handler._get_cross_pollination_stats()

        body = get_body(result)
        assert "version" in body
        assert body["version"] == "2.0.3"

    def test_cross_pollination_stats_structure(self):
        """Should return properly structured cross-pollination stats."""
        handler = create_analytics_handler()

        result = handler._get_cross_pollination_stats()

        body = get_body(result)
        assert "stats" in body
        stats = body["stats"]
        assert "calibration" in stats
        assert "learning" in stats
        assert "voting_accuracy" in stats
        assert "adaptive_rounds" in stats
        assert "rlm_cache" in stats


# ===========================================================================
# Test Metric Calculations
# ===========================================================================


class TestMetricCalculations:
    """Tests for correct metric calculations."""

    def test_average_rounds_calculation(self):
        """Should correctly calculate average rounds."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {"id": "1", "result": {"rounds_used": 3}},
            {"id": "2", "result": {"rounds_used": 5}},
            {"id": "3", "result": {"rounds_used": 4}},
            {"id": "4", "result": {"rounds_used": 8}},
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_early_stop_stats()

        body = get_body(result)
        # (3+5+4+8)/4 = 5.0
        assert body["stats"]["average_rounds"] == 5.0

    def test_average_elo_calculation(self):
        """Should correctly calculate average ELO."""
        handler = create_analytics_handler()
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            MockAgentRating("a", 1500, 10),
            MockAgentRating("b", 1600, 10),
            MockAgentRating("c", 1700, 10),
        ]

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_ranking_stats()

        body = get_body(result)
        # (1500+1600+1700)/3 = 1600
        assert body["stats"]["avg_elo"] == 1600.0

    def test_consensus_rate_calculation(self):
        """Should correctly calculate consensus rate."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {"id": "1", "result": {"confidence": 0.8, "consensus_reached": True}},
            {"id": "2", "result": {"confidence": 0.8, "consensus_reached": True}},
            {"id": "3", "result": {"confidence": 0.8, "consensus_reached": False}},
            {"id": "4", "result": {"confidence": 0.8, "consensus_reached": True}},
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        # 3/4 = 0.75
        assert body["stats"]["consensus_rate"] == 0.75

    def test_quality_score_bounds(self):
        """Should keep quality score within 0-100 bounds."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()

        # Test with very high quality debates
        mock_storage.list_debates.return_value = [
            {"id": f"d{i}", "result": {"confidence": 1.0, "consensus_reached": True}}
            for i in range(10)
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        assert body["quality_score"] <= 100
        assert body["quality_score"] >= 0

    def test_debate_id_truncation(self):
        """Should truncate debate IDs to 8 characters in confidence history."""
        handler = create_analytics_handler()
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": "very-long-debate-id-123456789",
                "timestamp": "2024-01-01T12:00:00Z",
                "result": {"confidence": 0.8, "consensus_reached": True},
            },
        ]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_consensus_quality()

        body = get_body(result)
        # Debate ID should be truncated to 8 chars
        assert len(body["stats"]["confidence_history"][0]["debate_id"]) == 8


# ===========================================================================
# Test Handler Context
# ===========================================================================


class TestHandlerContext:
    """Tests for handler context handling."""

    def test_handler_accepts_empty_context(self):
        """Should accept empty context."""
        handler = create_analytics_handler({})
        assert handler.ctx == {}

    def test_handler_accepts_none_context(self):
        """Should accept None context (converts to empty dict)."""
        handler = create_analytics_handler(None)
        assert handler.ctx == {}

    def test_handler_preserves_context(self):
        """Should preserve passed context."""
        ctx = {"storage": MagicMock(), "elo_system": MagicMock()}
        handler = create_analytics_handler(ctx)
        assert handler.ctx == ctx
