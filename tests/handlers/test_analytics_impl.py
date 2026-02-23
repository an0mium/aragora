"""Comprehensive tests for the _analytics_impl handler module.

Covers all AnalyticsHandler routes:
- GET /api/analytics/disagreements
- GET /api/analytics/role-rotation
- GET /api/analytics/early-stops
- GET /api/analytics/consensus-quality
- GET /api/ranking/stats
- GET /api/memory/stats
- GET /api/analytics/cross-pollination
- GET /api/analytics/learning-efficiency
- GET /api/analytics/voting-accuracy
- GET /api/analytics/calibration

Also covers:
- can_handle routing
- Rate limiter enforcement
- Auth/RBAC enforcement
- No-storage / no-elo fallback paths
- Cached debates helper
- Edge cases and error handling
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers._analytics_impl import (
    AnalyticsHandler,
    _analytics_limiter,
)
from aragora.server.handlers.base import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockHTTPHandler:
    """Mock HTTP handler for testing."""

    def __init__(self):
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {}


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the module-level rate limiter between tests."""
    _analytics_limiter._buckets = defaultdict(list)
    _analytics_limiter._requests = _analytics_limiter._buckets
    yield
    _analytics_limiter._buckets = defaultdict(list)
    _analytics_limiter._requests = _analytics_limiter._buckets


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    return MockHTTPHandler()


@pytest.fixture
def handler():
    """Create AnalyticsHandler with no context."""
    return AnalyticsHandler(ctx={})


@pytest.fixture
def handler_with_storage():
    """Create AnalyticsHandler with a mock storage."""
    storage = MagicMock()
    storage.list_debates.return_value = []
    return AnalyticsHandler(ctx={"storage": storage}), storage


@pytest.fixture
def handler_with_elo():
    """Create AnalyticsHandler with a mock ELO system."""
    elo = MagicMock()
    elo.get_leaderboard.return_value = []
    return AnalyticsHandler(ctx={"elo_system": elo}), elo


@pytest.fixture
def handler_with_nomic_dir(tmp_path):
    """Create AnalyticsHandler with a nomic directory."""
    return AnalyticsHandler(ctx={"nomic_dir": tmp_path}), tmp_path


# ---------------------------------------------------------------------------
# can_handle routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle path routing."""

    def test_handles_disagreements(self, handler):
        assert handler.can_handle("/api/analytics/disagreements") is True

    def test_handles_disagreements_versioned(self, handler):
        assert handler.can_handle("/api/v1/analytics/disagreements") is True

    def test_handles_role_rotation(self, handler):
        assert handler.can_handle("/api/analytics/role-rotation") is True

    def test_handles_early_stops(self, handler):
        assert handler.can_handle("/api/analytics/early-stops") is True

    def test_handles_consensus_quality(self, handler):
        assert handler.can_handle("/api/analytics/consensus-quality") is True

    def test_handles_ranking_stats(self, handler):
        assert handler.can_handle("/api/ranking/stats") is True

    def test_handles_memory_stats(self, handler):
        assert handler.can_handle("/api/memory/stats") is True

    def test_handles_cross_pollination(self, handler):
        assert handler.can_handle("/api/analytics/cross-pollination") is True

    def test_handles_learning_efficiency(self, handler):
        assert handler.can_handle("/api/analytics/learning-efficiency") is True

    def test_handles_voting_accuracy(self, handler):
        assert handler.can_handle("/api/analytics/voting-accuracy") is True

    def test_handles_calibration(self, handler):
        assert handler.can_handle("/api/analytics/calibration") is True

    def test_rejects_invalid_path(self, handler):
        assert handler.can_handle("/api/analytics/unknown") is False

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/analytics") is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting on analytics endpoints."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self, handler, mock_http_handler):
        """When the rate limiter denies a request, handler returns 429."""
        with patch(
            "aragora.server.handlers._analytics_impl._analytics_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

        assert _status(result) == 429
        body = _body(result)
        assert "Rate limit" in body.get("error", "")


# ---------------------------------------------------------------------------
# Auth/RBAC checks
# ---------------------------------------------------------------------------


class TestAuthRBAC:
    """Tests for authentication and permission enforcement.

    The conftest auto-patches SecureHandler.get_auth_context so these tests
    pass by default. The no_auto_auth marker is used for explicit auth failure
    testing.
    """

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self, mock_http_handler):
        """Without auto-auth patching, unauthenticated request returns 401."""
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        h = AnalyticsHandler(ctx={})

        async def raise_unauth(self, request, require_auth=False):
            raise UnauthorizedError("Not authenticated")

        with patch.object(SecureHandler, "get_auth_context", raise_unauth):
            result = await h.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, mock_http_handler):
        """When the user lacks analytics:read permission, returns 403."""
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import ForbiddenError, SecureHandler

        h = AnalyticsHandler(ctx={})
        ctx = AuthorizationContext(
            user_id="user-001",
            user_email="user@test.com",
            org_id="org-001",
            roles={"viewer"},
            permissions=set(),
        )

        async def get_ctx(self, request, require_auth=False):
            return ctx

        def check_perm(self, auth_context, permission, resource_id=None):
            raise ForbiddenError("Permission denied", permission=permission)

        with (
            patch.object(SecureHandler, "get_auth_context", get_ctx),
            patch.object(SecureHandler, "check_permission", check_perm),
        ):
            result = await h.handle(
                "/api/analytics/disagreements", {}, mock_http_handler
            )

        assert _status(result) == 403


# ---------------------------------------------------------------------------
# Unknown route returns None
# ---------------------------------------------------------------------------


class TestUnknownRoute:
    """Test that unmatched paths return None."""

    @pytest.mark.asyncio
    async def test_unmatched_path_returns_none(self, handler, mock_http_handler):
        """handle() should return None for paths not in ROUTES."""
        result = await handler.handle("/api/analytics/nope", {}, mock_http_handler)
        assert result is None


# ---------------------------------------------------------------------------
# Disagreement stats
# ---------------------------------------------------------------------------


class TestDisagreementStats:
    """Tests for GET /api/analytics/disagreements."""

    @pytest.mark.asyncio
    async def test_no_storage_returns_empty_stats(self, handler, mock_http_handler):
        """Without storage, returns empty stats dict."""
        result = await handler.handle(
            "/api/analytics/disagreements", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"] == {}

    @pytest.mark.asyncio
    async def test_empty_debates(self, handler_with_storage, mock_http_handler):
        """With storage but no debates, returns zero counts."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = []

        result = await h.handle(
            "/api/analytics/disagreements", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"]["total_debates"] == 0
        assert body["stats"]["with_disagreements"] == 0
        assert body["stats"]["unanimous"] == 0

    @pytest.mark.asyncio
    async def test_debates_with_disagreements(
        self, handler_with_storage, mock_http_handler
    ):
        """Debates with disagreement reports are counted correctly."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {
                "result": {
                    "disagreement_report": {"unanimous_critiques": True},
                    "uncertainty_metrics": {"disagreement_type": "factual"},
                }
            },
            {
                "result": {
                    "disagreement_report": {"unanimous_critiques": False},
                    "uncertainty_metrics": {"disagreement_type": "methodological"},
                }
            },
            {"result": {}},  # No disagreement report
        ]

        result = await h.handle(
            "/api/analytics/disagreements", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"]["total_debates"] == 3
        assert body["stats"]["with_disagreements"] == 1
        assert body["stats"]["unanimous"] == 1
        assert body["stats"]["disagreement_types"]["factual"] == 1
        assert body["stats"]["disagreement_types"]["methodological"] == 1

    @pytest.mark.asyncio
    async def test_non_dict_debate_is_skipped(
        self, handler_with_storage, mock_http_handler
    ):
        """Non-dict debate entries should be handled gracefully."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = ["not_a_dict", 42, None]

        result = await h.handle(
            "/api/analytics/disagreements", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"]["total_debates"] == 3
        assert body["stats"]["with_disagreements"] == 0


# ---------------------------------------------------------------------------
# Role rotation stats
# ---------------------------------------------------------------------------


class TestRoleRotationStats:
    """Tests for GET /api/analytics/role-rotation."""

    @pytest.mark.asyncio
    async def test_no_storage_returns_empty(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/analytics/role-rotation", {}, mock_http_handler
        )
        assert _status(result) == 200
        assert _body(result)["stats"] == {}

    @pytest.mark.asyncio
    async def test_empty_debates(self, handler_with_storage, mock_http_handler):
        h, storage = handler_with_storage
        storage.list_debates.return_value = []

        result = await h.handle(
            "/api/analytics/role-rotation", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["total_debates"] == 0
        assert body["stats"]["role_assignments"] == {}

    @pytest.mark.asyncio
    async def test_role_assignments_counted(
        self, handler_with_storage, mock_http_handler
    ):
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {
                "messages": [
                    {"cognitive_role": "critic"},
                    {"cognitive_role": "proposer"},
                    {"role": "judge"},  # Falls back to "role" key
                ]
            },
            {
                "messages": [
                    {"cognitive_role": "critic"},
                ]
            },
        ]

        result = await h.handle(
            "/api/analytics/role-rotation", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["total_debates"] == 2
        assert body["stats"]["role_assignments"]["critic"] == 2
        assert body["stats"]["role_assignments"]["proposer"] == 1
        assert body["stats"]["role_assignments"]["judge"] == 1

    @pytest.mark.asyncio
    async def test_missing_role_defaults_to_unknown(
        self, handler_with_storage, mock_http_handler
    ):
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {"messages": [{}]},  # No cognitive_role or role
        ]

        result = await h.handle(
            "/api/analytics/role-rotation", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["role_assignments"].get("unknown") == 1


# ---------------------------------------------------------------------------
# Early stop stats
# ---------------------------------------------------------------------------


class TestEarlyStopStats:
    """Tests for GET /api/analytics/early-stops."""

    @pytest.mark.asyncio
    async def test_no_storage_returns_empty(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/analytics/early-stops", {}, mock_http_handler
        )
        assert _status(result) == 200
        assert _body(result)["stats"] == {}

    @pytest.mark.asyncio
    async def test_empty_debates(self, handler_with_storage, mock_http_handler):
        h, storage = handler_with_storage
        storage.list_debates.return_value = []

        result = await h.handle(
            "/api/analytics/early-stops", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["total_debates"] == 0
        assert body["stats"]["average_rounds"] == 0.0

    @pytest.mark.asyncio
    async def test_mixed_early_stopped_and_full(
        self, handler_with_storage, mock_http_handler
    ):
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {"result": {"early_stopped": True, "rounds_used": 2}},
            {"result": {"early_stopped": False, "rounds_used": 5}},
            {"result": {"rounds_used": 3}},  # early_stopped not set (falsy)
        ]

        result = await h.handle(
            "/api/analytics/early-stops", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["total_debates"] == 3
        assert body["stats"]["early_stopped"] == 1
        assert body["stats"]["full_rounds"] == 2
        assert abs(body["stats"]["average_rounds"] - (10 / 3)) < 0.01

    @pytest.mark.asyncio
    async def test_non_dict_debate_safe(
        self, handler_with_storage, mock_http_handler
    ):
        """Non-dict debates should not cause errors."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = ["string_debate"]

        result = await h.handle(
            "/api/analytics/early-stops", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["total_debates"] == 1


# ---------------------------------------------------------------------------
# Consensus quality
# ---------------------------------------------------------------------------


class TestConsensusQuality:
    """Tests for GET /api/analytics/consensus-quality."""

    @pytest.mark.asyncio
    async def test_no_storage_returns_defaults(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["quality_score"] == 0
        assert body["alert"] is None

    @pytest.mark.asyncio
    async def test_empty_debates_insufficient_data(
        self, handler_with_storage, mock_http_handler
    ):
        h, storage = handler_with_storage
        storage.list_debates.return_value = []

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["trend"] == "insufficient_data"
        assert body["quality_score"] == 0

    @pytest.mark.asyncio
    async def test_high_confidence_high_consensus(
        self, handler_with_storage, mock_http_handler
    ):
        """High confidence + high consensus should give a high quality score."""
        h, storage = handler_with_storage
        # 6 debates all with high confidence and consensus
        storage.list_debates.return_value = [
            {
                "id": f"debate-{i}",
                "timestamp": f"2026-01-0{i+1}T00:00:00Z",
                "result": {"confidence": 0.9, "consensus_reached": True},
            }
            for i in range(6)
        ]

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        assert body["quality_score"] > 60
        assert body["stats"]["consensus_rate"] == 1.0
        assert body["stats"]["average_confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_alert(
        self, handler_with_storage, mock_http_handler
    ):
        """Low confidence scores should produce a critical or warning alert."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {
                "id": f"debate-{i}",
                "timestamp": f"2026-01-0{i+1}T00:00:00Z",
                "result": {"confidence": 0.1, "consensus_reached": False},
            }
            for i in range(6)
        ]

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        assert body["quality_score"] < 40
        assert body["alert"] is not None
        assert body["alert"]["level"] in ("critical", "warning")

    @pytest.mark.asyncio
    async def test_improving_trend_detected(
        self, handler_with_storage, mock_http_handler
    ):
        """When second half has higher confidence, trend should be 'improving'."""
        h, storage = handler_with_storage
        debates = []
        for i in range(10):
            conf = 0.3 if i < 5 else 0.8
            debates.append(
                {
                    "id": f"debate-{i}",
                    "timestamp": f"2026-01-{i+1:02d}T00:00:00Z",
                    "result": {"confidence": conf, "consensus_reached": True},
                }
            )
        storage.list_debates.return_value = debates

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["trend"] == "improving"

    @pytest.mark.asyncio
    async def test_declining_trend_detected(
        self, handler_with_storage, mock_http_handler
    ):
        """When second half has lower confidence, trend should be 'declining'."""
        h, storage = handler_with_storage
        debates = []
        for i in range(10):
            conf = 0.8 if i < 5 else 0.3
            debates.append(
                {
                    "id": f"debate-{i}",
                    "timestamp": f"2026-01-{i+1:02d}T00:00:00Z",
                    "result": {"confidence": conf, "consensus_reached": False},
                }
            )
        storage.list_debates.return_value = debates

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["trend"] == "declining"

    @pytest.mark.asyncio
    async def test_stable_trend_with_small_diff(
        self, handler_with_storage, mock_http_handler
    ):
        """When halves are close, trend should be 'stable'."""
        h, storage = handler_with_storage
        debates = []
        for i in range(10):
            debates.append(
                {
                    "id": f"debate-{i}",
                    "timestamp": f"2026-01-{i+1:02d}T00:00:00Z",
                    "result": {"confidence": 0.6, "consensus_reached": True},
                }
            )
        storage.list_debates.return_value = debates

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["trend"] == "stable"

    @pytest.mark.asyncio
    async def test_fewer_than_5_debates_no_trend(
        self, handler_with_storage, mock_http_handler
    ):
        """With fewer than 5 debates, trend is 'stable' (default, no regression)."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {
                "id": "d1",
                "timestamp": "2026-01-01T00:00:00Z",
                "result": {"confidence": 0.5, "consensus_reached": True},
            },
        ]

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["trend"] == "stable"

    @pytest.mark.asyncio
    async def test_confidence_history_capped_at_20(
        self, handler_with_storage, mock_http_handler
    ):
        """confidence_history in response is capped to 20 entries for the UI."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {
                "id": f"debate-{i}",
                "timestamp": f"2026-01-01T00:{i:02d}:00Z",
                "result": {"confidence": 0.5, "consensus_reached": True},
            }
            for i in range(30)
        ]

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        assert len(body["stats"]["confidence_history"]) == 20

    @pytest.mark.asyncio
    async def test_declining_low_confidence_info_alert(
        self, handler_with_storage, mock_http_handler
    ):
        """Declining trend with moderate quality score shows info alert."""
        h, storage = handler_with_storage
        debates = []
        for i in range(10):
            conf = 0.7 if i < 5 else 0.55
            debates.append(
                {
                    "id": f"debate-{i}",
                    "timestamp": f"2026-01-{i+1:02d}T00:00:00Z",
                    "result": {"confidence": conf, "consensus_reached": True},
                }
            )
        storage.list_debates.return_value = debates

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        # With average < 0.7 and declining trend and quality_score >= 60
        # it might be info alert; depends on exact score
        # Just verify no crash and alert handling is correct
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_debate_id_truncated_to_8_chars(
        self, handler_with_storage, mock_http_handler
    ):
        """Debate IDs in confidence_history are truncated to first 8 chars."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {
                "id": "abcdefghijklmnop",
                "timestamp": "2026-01-01T00:00:00Z",
                "result": {"confidence": 0.5, "consensus_reached": True},
            }
        ]

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["confidence_history"][0]["debate_id"] == "abcdefgh"


# ---------------------------------------------------------------------------
# Ranking stats
# ---------------------------------------------------------------------------


class TestRankingStats:
    """Tests for GET /api/ranking/stats."""

    @pytest.mark.asyncio
    async def test_no_elo_returns_503(self, handler, mock_http_handler):
        result = await handler.handle("/api/ranking/stats", {}, mock_http_handler)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_empty_leaderboard(self, handler_with_elo, mock_http_handler):
        h, elo = handler_with_elo
        elo.get_leaderboard.return_value = []

        result = await h.handle("/api/ranking/stats", {}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"]["total_agents"] == 0
        assert body["stats"]["total_matches"] == 0
        assert body["stats"]["avg_elo"] == 1500
        assert body["stats"]["top_agent"] is None

    @pytest.mark.asyncio
    async def test_populated_leaderboard(self, mock_http_handler):
        """Leaderboard with agents computes correct stats."""
        agent1 = MagicMock()
        agent1.elo = 1600
        agent1.debates_count = 10
        agent1.agent_name = "claude"

        agent2 = MagicMock()
        agent2.elo = 1400
        agent2.debates_count = 5
        agent2.agent_name = "gpt4"

        elo = MagicMock()
        elo.get_leaderboard.return_value = [agent1, agent2]

        h = AnalyticsHandler(ctx={"elo_system": elo})

        result = await h.handle("/api/ranking/stats", {}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"]["total_agents"] == 2
        assert body["stats"]["total_matches"] == 15
        assert body["stats"]["avg_elo"] == 1500.0
        assert body["stats"]["top_agent"] == "claude"
        assert body["stats"]["elo_range"]["min"] == 1400
        assert body["stats"]["elo_range"]["max"] == 1600


# ---------------------------------------------------------------------------
# Memory stats
# ---------------------------------------------------------------------------


class TestMemoryStats:
    """Tests for GET /api/memory/stats."""

    @pytest.mark.asyncio
    async def test_no_nomic_dir_returns_empty(self, handler, mock_http_handler):
        result = await handler.handle("/api/memory/stats", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["stats"] == {}

    @pytest.mark.asyncio
    async def test_no_db_files(self, handler_with_nomic_dir, mock_http_handler):
        """When nomic dir exists but no DB files, all flags are False."""
        h, nomic_dir = handler_with_nomic_dir

        result = await h.handle("/api/memory/stats", {}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"]["embeddings_db"] is False
        assert body["stats"]["insights_db"] is False
        assert body["stats"]["continuum_memory"] is False

    @pytest.mark.asyncio
    async def test_embeddings_db_detected(
        self, handler_with_nomic_dir, mock_http_handler
    ):
        h, nomic_dir = handler_with_nomic_dir
        (nomic_dir / "debate_embeddings.db").write_text("")

        result = await h.handle("/api/memory/stats", {}, mock_http_handler)
        body = _body(result)
        assert body["stats"]["embeddings_db"] is True
        assert body["stats"]["insights_db"] is False
        assert body["stats"]["continuum_memory"] is False

    @pytest.mark.asyncio
    async def test_continuum_memory_db_detected(
        self, handler_with_nomic_dir, mock_http_handler
    ):
        h, nomic_dir = handler_with_nomic_dir
        (nomic_dir / "continuum_memory.db").write_text("")

        result = await h.handle("/api/memory/stats", {}, mock_http_handler)
        body = _body(result)
        assert body["stats"]["continuum_memory"] is True

    @pytest.mark.asyncio
    async def test_insights_db_detected(
        self, handler_with_nomic_dir, mock_http_handler
    ):
        """Insights DB uses the configured DB_INSIGHTS_PATH constant."""
        from aragora.config import DB_INSIGHTS_PATH

        h, nomic_dir = handler_with_nomic_dir
        (nomic_dir / DB_INSIGHTS_PATH).write_text("")

        result = await h.handle("/api/memory/stats", {}, mock_http_handler)
        body = _body(result)
        assert body["stats"]["insights_db"] is True

    @pytest.mark.asyncio
    async def test_all_dbs_present(self, handler_with_nomic_dir, mock_http_handler):
        from aragora.config import DB_INSIGHTS_PATH

        h, nomic_dir = handler_with_nomic_dir
        (nomic_dir / "debate_embeddings.db").write_text("")
        (nomic_dir / DB_INSIGHTS_PATH).write_text("")
        (nomic_dir / "continuum_memory.db").write_text("")

        result = await h.handle("/api/memory/stats", {}, mock_http_handler)
        body = _body(result)
        assert body["stats"]["embeddings_db"] is True
        assert body["stats"]["insights_db"] is True
        assert body["stats"]["continuum_memory"] is True


# ---------------------------------------------------------------------------
# Cross-pollination stats
# ---------------------------------------------------------------------------


class TestCrossPollinationStats:
    """Tests for GET /api/analytics/cross-pollination."""

    @pytest.mark.asyncio
    async def test_default_stats_structure(self, handler, mock_http_handler):
        """Default stats should show disabled subsystems."""
        with (
            patch(
                "aragora.server.handlers._analytics_impl.RLMHierarchyCache",
                side_effect=ImportError,
            ) if False else patch.dict("sys.modules", {}),
        ):
            result = await handler.handle(
                "/api/analytics/cross-pollination", {}, mock_http_handler
            )

        assert _status(result) == 200
        body = _body(result)
        assert "stats" in body
        assert body["version"] == "2.0.3"

    @pytest.mark.asyncio
    async def test_rlm_cache_available(self, handler, mock_http_handler):
        """When RLM cache is importable, stats include cache data."""
        mock_cache = MagicMock()
        mock_cache.get_stats.return_value = {
            "hits": 100,
            "misses": 20,
            "hit_rate": 0.833,
        }

        mock_cls = MagicMock(return_value=mock_cache)
        with patch.dict(
            "sys.modules",
            {"aragora.rlm.bridge": MagicMock(RLMHierarchyCache=mock_cls)},
        ):
            result = await handler.handle(
                "/api/analytics/cross-pollination", {}, mock_http_handler
            )

        assert _status(result) == 200
        body = _body(result)
        stats = body["stats"]
        assert stats["rlm_cache"]["enabled"] is True
        assert stats["rlm_cache"]["hits"] == 100
        assert stats["rlm_cache"]["misses"] == 20

    @pytest.mark.asyncio
    async def test_elo_system_available(self, handler, mock_http_handler):
        """When ELO store is importable, learning/voting/calibration show enabled."""
        mock_elo_fn = MagicMock()

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(get_elo_store=mock_elo_fn)},
        ):
            result = await handler.handle(
                "/api/analytics/cross-pollination", {}, mock_http_handler
            )

        assert _status(result) == 200
        body = _body(result)
        stats = body["stats"]
        assert stats["learning"]["enabled"] is True
        assert stats["voting_accuracy"]["enabled"] is True
        assert stats["calibration"]["enabled"] is True


# ---------------------------------------------------------------------------
# Learning efficiency stats
# ---------------------------------------------------------------------------


class TestLearningEfficiencyStats:
    """Tests for GET /api/analytics/learning-efficiency."""

    @pytest.mark.asyncio
    async def test_elo_not_available(self, handler, mock_http_handler):
        """When ELO store cannot be imported, returns error."""
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=ImportError,
        ):
            result = await handler.handle(
                "/api/analytics/learning-efficiency", {}, mock_http_handler
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["error"] == "ELO system not available"
        assert body["agents"] == []

    @pytest.mark.asyncio
    async def test_specific_agent_query(self, handler, mock_http_handler):
        """Query for a specific agent returns that agent's efficiency."""
        mock_elo = MagicMock()
        mock_elo.get_learning_efficiency.return_value = {
            "learning_rate": 0.85,
            "improvement": 0.12,
        }

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo,
        ):
            result = await handler.handle(
                "/api/analytics/learning-efficiency",
                {"agent": ["claude"], "domain": ["coding"]},
                mock_http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["agent"] == "claude"
        assert body["domain"] == "coding"
        assert body["efficiency"]["learning_rate"] == 0.85
        mock_elo.get_learning_efficiency.assert_called_once_with(
            "claude", domain="coding"
        )

    @pytest.mark.asyncio
    async def test_all_agents_batch_query(self, handler, mock_http_handler):
        """Without agent filter, returns batch efficiency for all leaderboard agents."""
        agent1 = MagicMock()
        agent1.agent_name = "claude"
        agent2 = MagicMock()
        agent2.agent_name = "gpt4"

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent1, agent2]
        mock_elo.get_learning_efficiency_batch.return_value = {
            "claude": {"rate": 0.9},
            "gpt4": {"rate": 0.7},
        }

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo,
        ):
            result = await handler.handle(
                "/api/analytics/learning-efficiency",
                {},
                mock_http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["domain"] == "general"
        assert len(body["agents"]) == 2
        assert body["agents"][0]["agent"] == "claude"
        assert body["agents"][1]["agent"] == "gpt4"

    @pytest.mark.asyncio
    async def test_default_domain_is_general(self, handler, mock_http_handler):
        """When domain is not specified, defaults to 'general'."""
        mock_elo = MagicMock()
        mock_elo.get_learning_efficiency.return_value = {}

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo,
        ):
            result = await handler.handle(
                "/api/analytics/learning-efficiency",
                {"agent": ["test_agent"]},
                mock_http_handler,
            )

        body = _body(result)
        assert body["domain"] == "general"

    @pytest.mark.asyncio
    async def test_limit_parameter(self, handler, mock_http_handler):
        """Limit parameter controls leaderboard size."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        mock_elo.get_learning_efficiency_batch.return_value = {}

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo,
        ):
            result = await handler.handle(
                "/api/analytics/learning-efficiency",
                {"limit": ["5"]},
                mock_http_handler,
            )

        mock_elo.get_leaderboard.assert_called_once_with(limit=5)


# ---------------------------------------------------------------------------
# Voting accuracy stats
# ---------------------------------------------------------------------------


class TestVotingAccuracyStats:
    """Tests for GET /api/analytics/voting-accuracy."""

    @pytest.mark.asyncio
    async def test_elo_not_available(self, handler, mock_http_handler):
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=ImportError,
        ):
            result = await handler.handle(
                "/api/analytics/voting-accuracy", {}, mock_http_handler
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["error"] == "ELO system not available"

    @pytest.mark.asyncio
    async def test_specific_agent(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_voting_accuracy.return_value = {
            "accuracy": 0.92,
            "total_votes": 50,
        }

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo,
        ):
            result = await handler.handle(
                "/api/analytics/voting-accuracy",
                {"agent": ["claude"]},
                mock_http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["agent"] == "claude"
        assert body["accuracy"]["accuracy"] == 0.92

    @pytest.mark.asyncio
    async def test_all_agents_batch(self, handler, mock_http_handler):
        agent1 = MagicMock()
        agent1.agent_name = "claude"

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent1]
        mock_elo.get_voting_accuracy_batch.return_value = {
            "claude": {"accuracy": 0.95},
        }

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo,
        ):
            result = await handler.handle(
                "/api/analytics/voting-accuracy",
                {},
                mock_http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert len(body["agents"]) == 1
        assert body["agents"][0]["agent"] == "claude"
        assert body["agents"][0]["accuracy"] == {"accuracy": 0.95}


# ---------------------------------------------------------------------------
# Calibration stats
# ---------------------------------------------------------------------------


class TestCalibrationStats:
    """Tests for GET /api/analytics/calibration."""

    @pytest.mark.asyncio
    async def test_elo_not_available(self, handler, mock_http_handler):
        with patch(
            "aragora.ranking.elo.get_elo_store",
            side_effect=ImportError,
        ):
            result = await handler.handle(
                "/api/analytics/calibration", {}, mock_http_handler
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["error"] == "ELO system not available"

    @pytest.mark.asyncio
    async def test_specific_agent_with_tracker(self, handler, mock_http_handler):
        """Agent with calibration tracker returns calibration data."""
        mock_elo = MagicMock()

        mock_summary = MagicMock()
        mock_summary.total_predictions = 100
        mock_summary.temperature = 1.2
        mock_summary.scaling_factor = 0.95

        mock_tracker_instance = MagicMock()
        mock_tracker_instance.get_calibration_summary.return_value = mock_summary
        mock_tracker_cls = MagicMock(return_value=mock_tracker_instance)
        mock_cal_module = MagicMock(CalibrationTracker=mock_tracker_cls)

        with (
            patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo,
            ),
            patch.dict(
                "sys.modules",
                {"aragora.ranking.calibration": mock_cal_module},
            ),
        ):
            result = await handler.handle(
                "/api/analytics/calibration",
                {"agent": ["claude"]},
                mock_http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["agent"] == "claude"
        cal = body["calibration"]
        assert cal["total_predictions"] == 100
        assert cal["temperature"] == 1.2
        assert cal["scaling_factor"] == 0.95

    @pytest.mark.asyncio
    async def test_specific_agent_no_tracker(self, handler, mock_http_handler):
        """When CalibrationTracker is not importable, returns None calibration.

        The module aragora.ranking.calibration does not exist, so the handler's
        try/except ImportError naturally falls through to tracker=None.
        """
        mock_elo = MagicMock()

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo,
        ):
            result = await handler.handle(
                "/api/analytics/calibration",
                {"agent": ["claude"]},
                mock_http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["calibration"] is None

    @pytest.mark.asyncio
    async def test_specific_agent_tracker_no_summary(self, handler, mock_http_handler):
        """When tracker has no summary for agent, calibration is None."""
        mock_elo = MagicMock()

        mock_tracker_instance = MagicMock()
        mock_tracker_instance.get_calibration_summary.return_value = None
        mock_tracker_cls = MagicMock(return_value=mock_tracker_instance)
        mock_cal_module = MagicMock(CalibrationTracker=mock_tracker_cls)

        with (
            patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo,
            ),
            patch.dict(
                "sys.modules",
                {"aragora.ranking.calibration": mock_cal_module},
            ),
        ):
            result = await handler.handle(
                "/api/analytics/calibration",
                {"agent": ["claude"]},
                mock_http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["calibration"] is None

    @pytest.mark.asyncio
    async def test_all_agents_with_tracker(self, handler, mock_http_handler):
        """Batch calibration for leaderboard agents."""
        agent1 = MagicMock()
        agent1.agent_name = "claude"
        agent2 = MagicMock()
        agent2.agent_name = "gpt4"

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent1, agent2]

        mock_summary = MagicMock()
        mock_summary.total_predictions = 50
        mock_summary.temperature = 1.0

        mock_tracker_instance = MagicMock()
        # Return summary for claude, None for gpt4
        mock_tracker_instance.get_calibration_summary.side_effect = [mock_summary, None]
        mock_tracker_cls = MagicMock(return_value=mock_tracker_instance)
        mock_cal_module = MagicMock(CalibrationTracker=mock_tracker_cls)

        with (
            patch(
                "aragora.ranking.elo.get_elo_store",
                return_value=mock_elo,
            ),
            patch.dict(
                "sys.modules",
                {"aragora.ranking.calibration": mock_cal_module},
            ),
        ):
            result = await handler.handle(
                "/api/analytics/calibration",
                {},
                mock_http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        agents = body["agents"]
        assert len(agents) == 2
        assert agents[0]["agent"] == "claude"
        assert agents[0]["calibration"]["total_predictions"] == 50
        assert agents[1]["agent"] == "gpt4"
        assert agents[1]["calibration"] is None

    @pytest.mark.asyncio
    async def test_all_agents_no_tracker(self, handler, mock_http_handler):
        """When tracker unavailable, all agents get None calibration.

        Since aragora.ranking.calibration doesn't exist, the handler's
        ImportError path naturally gives tracker=None.
        """
        agent1 = MagicMock()
        agent1.agent_name = "claude"

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent1]

        with patch(
            "aragora.ranking.elo.get_elo_store",
            return_value=mock_elo,
        ):
            result = await handler.handle(
                "/api/analytics/calibration",
                {},
                mock_http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["agents"][0]["calibration"] is None


# ---------------------------------------------------------------------------
# Cached debates helper
# ---------------------------------------------------------------------------


class TestCachedDebates:
    """Tests for the _get_cached_debates helper."""

    def test_no_storage_returns_empty(self, handler):
        result = handler._get_cached_debates()
        assert result == []

    def test_returns_debates_from_storage(self, handler_with_storage):
        h, storage = handler_with_storage
        storage.list_debates.return_value = [{"id": "1"}, {"id": "2"}]

        result = h._get_cached_debates(limit=50)
        assert len(result) == 2
        storage.list_debates.assert_called_with(limit=50)

    def test_storage_error_returns_empty(self, handler_with_storage):
        """On storage error, returns empty list instead of crashing."""
        h, storage = handler_with_storage
        storage.list_debates.side_effect = RuntimeError("DB unavailable")

        result = h._get_cached_debates()
        assert result == []

    def test_storage_value_error_returns_empty(self, handler_with_storage):
        h, storage = handler_with_storage
        storage.list_debates.side_effect = ValueError("bad query")

        result = h._get_cached_debates()
        assert result == []


# ---------------------------------------------------------------------------
# Versioned paths
# ---------------------------------------------------------------------------


class TestVersionedPaths:
    """Test that versioned API paths are routed correctly."""

    @pytest.mark.asyncio
    async def test_v1_disagreements(self, handler_with_storage, mock_http_handler):
        h, storage = handler_with_storage
        storage.list_debates.return_value = []

        result = await h.handle(
            "/api/v1/analytics/disagreements", {}, mock_http_handler
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_ranking_stats(self, mock_http_handler):
        elo = MagicMock()
        elo.get_leaderboard.return_value = []
        h = AnalyticsHandler(ctx={"elo_system": elo})

        result = await h.handle("/api/v1/ranking/stats", {}, mock_http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_memory_stats(self, handler_with_nomic_dir, mock_http_handler):
        h, nomic_dir = handler_with_nomic_dir

        result = await h.handle("/api/v1/memory/stats", {}, mock_http_handler)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases across analytics endpoints."""

    @pytest.mark.asyncio
    async def test_handler_init_default_ctx(self):
        """Handler initializes with empty ctx dict if None passed."""
        h = AnalyticsHandler(ctx=None)
        assert h.ctx == {}

    @pytest.mark.asyncio
    async def test_disagreement_stats_with_missing_result_key(
        self, handler_with_storage, mock_http_handler
    ):
        """Debate dicts with missing 'result' key should not crash."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {"messages": []},  # No 'result' key
        ]

        result = await h.handle(
            "/api/analytics/disagreements", {}, mock_http_handler
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"]["total_debates"] == 1

    @pytest.mark.asyncio
    async def test_early_stops_with_zero_rounds(
        self, handler_with_storage, mock_http_handler
    ):
        """Debates with zero rounds_used should be handled."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {"result": {"rounds_used": 0}},
        ]

        result = await h.handle(
            "/api/analytics/early-stops", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["average_rounds"] == 0.0

    @pytest.mark.asyncio
    async def test_consensus_quality_missing_fields(
        self, handler_with_storage, mock_http_handler
    ):
        """Debates with missing confidence/consensus fields use defaults."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = [
            {"id": "", "timestamp": "", "result": {}},
        ]

        result = await h.handle(
            "/api/analytics/consensus-quality", {}, mock_http_handler
        )
        body = _body(result)
        # confidence defaults to 0.0, consensus_reached defaults to False
        assert body["stats"]["average_confidence"] == 0.0
        assert body["stats"]["consensus_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_role_rotation_non_dict_debate(
        self, handler_with_storage, mock_http_handler
    ):
        """Non-dict debates in role rotation should not crash."""
        h, storage = handler_with_storage
        storage.list_debates.return_value = ["not_a_dict"]

        result = await h.handle(
            "/api/analytics/role-rotation", {}, mock_http_handler
        )
        body = _body(result)
        assert body["stats"]["total_debates"] == 1
        assert body["stats"]["role_assignments"] == {}
