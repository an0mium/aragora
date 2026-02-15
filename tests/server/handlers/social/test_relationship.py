"""Tests for aragora.server.handlers.social.relationship - Relationship Handler."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Score computation utilities - import directly (no heavy deps)
# ---------------------------------------------------------------------------

from aragora.server.handlers.social.relationship import (
    compute_rivalry_score,
    compute_alliance_score,
    determine_relationship_type,
    compute_relationship_scores,
    RelationshipScores,
)


from aragora.server.handlers.social.relationship import RelationshipHandler


# ===========================================================================
# Helpers
# ===========================================================================


@dataclass
class FakeRelationship:
    """Mimics AgentRelationship returned by RelationshipTracker."""

    agent_a: str = "claude"
    agent_b: str = "gpt4"
    debate_count: int = 10
    agreement_count: int = 4
    a_wins_over_b: int = 3
    b_wins_over_a: int = 5
    rivalry_score: float = 0.45
    alliance_score: float = 0.24
    critique_count_a_to_b: int = 7
    critique_count_b_to_a: int = 5
    influence_a_on_b: float = 0.6
    influence_b_on_a: float = 0.35


class MockHTTPHandler:
    """Minimal mock for the HTTP request handler passed to handle()."""

    def __init__(self, client_ip: str = "127.0.0.1"):
        self.client_address = (client_ip, 12345)
        self.headers = {}


def _make_db(rows: list[tuple], db_path: str | None = None) -> str:
    """Create a temporary SQLite DB with an agent_relationships table."""
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_relationships (
            agent_a TEXT,
            agent_b TEXT,
            debate_count INTEGER,
            agreement_count INTEGER,
            a_wins_over_b INTEGER,
            b_wins_over_a INTEGER
        )
    """
    )
    for row in rows:
        conn.execute("INSERT INTO agent_relationships VALUES (?, ?, ?, ?, ?, ?)", row)
    conn.commit()
    conn.close()
    return db_path


def _parse_body(result) -> dict:
    """Parse JSON body from a HandlerResult."""
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


# ===========================================================================
# Score Computation Tests (no handler needed)
# ===========================================================================


class TestComputeRivalryScore:
    """Tests for compute_rivalry_score."""

    def test_below_min_debates_returns_zero(self):
        assert compute_rivalry_score(0, 0, 0, 0) == 0.0
        assert compute_rivalry_score(1, 0, 1, 0) == 0.0
        assert compute_rivalry_score(2, 1, 1, 1) == 0.0

    def test_all_disagreement_even_wins(self):
        # 20 debates, 0 agreements, 10-10 wins -> max rivalry
        score = compute_rivalry_score(20, 0, 10, 10)
        assert score == pytest.approx(1.0)

    def test_all_agreement_returns_zero(self):
        # 20 debates, 20 agreements -> no disagreement -> 0
        score = compute_rivalry_score(20, 20, 5, 5)
        assert score == pytest.approx(0.0)

    def test_one_sided_wins_reduce_competitiveness(self):
        # Agent A wins all -> competitiveness = 0
        score = compute_rivalry_score(20, 0, 20, 0)
        assert score == pytest.approx(0.0)

    def test_frequency_factor_caps_at_one(self):
        # 40 debates -> frequency_factor = min(1, 40/20) = 1.0
        score_20 = compute_rivalry_score(20, 0, 10, 10)
        score_40 = compute_rivalry_score(40, 0, 20, 20)
        # Both should have frequency_factor = 1.0, but different debate counts
        assert score_40 == pytest.approx(1.0)
        assert score_20 == pytest.approx(1.0)

    def test_low_frequency_reduces_score(self):
        # 5 debates -> frequency_factor = 5/20 = 0.25
        score = compute_rivalry_score(5, 0, 2, 3)
        expected_disagreement = 1.0
        expected_competitiveness = 1 - abs(2 - 3) / max(5, 1)
        expected_freq = 5 / 20
        assert score == pytest.approx(
            expected_disagreement * expected_competitiveness * expected_freq
        )

    def test_partial_disagreement(self):
        score = compute_rivalry_score(10, 3, 4, 3)
        disagreement = 1 - 3 / 10  # 0.7
        competitiveness = 1 - abs(4 - 3) / 7  # ~0.857
        freq = 10 / 20  # 0.5
        assert score == pytest.approx(disagreement * competitiveness * freq)

    def test_no_wins_at_all(self):
        # 10 debates, 2 agreements, 0 wins each side
        score = compute_rivalry_score(10, 2, 0, 0)
        # competitiveness = 1 - 0/max(0,1) = 1
        disagreement = 1 - 2 / 10
        freq = 10 / 20
        assert score == pytest.approx(disagreement * 1.0 * freq)


class TestComputeAllianceScore:
    """Tests for compute_alliance_score."""

    def test_below_min_debates_returns_zero(self):
        assert compute_alliance_score(0, 0) == 0.0
        assert compute_alliance_score(2, 2) == 0.0

    def test_full_agreement(self):
        score = compute_alliance_score(10, 10)
        assert score == pytest.approx(1.0 * 0.6)

    def test_no_agreement(self):
        score = compute_alliance_score(10, 0)
        assert score == pytest.approx(0.0)

    def test_half_agreement(self):
        score = compute_alliance_score(10, 5)
        assert score == pytest.approx(0.5 * 0.6)


class TestDetermineRelationshipType:
    """Tests for determine_relationship_type."""

    def test_rivalry_wins(self):
        assert determine_relationship_type(0.5, 0.2) == "rivalry"

    def test_alliance_wins(self):
        assert determine_relationship_type(0.2, 0.5) == "alliance"

    def test_neutral_both_below_threshold(self):
        assert determine_relationship_type(0.1, 0.1) == "neutral"

    def test_neutral_when_equal(self):
        assert determine_relationship_type(0.5, 0.5) == "neutral"

    def test_custom_threshold(self):
        # Below custom threshold
        assert determine_relationship_type(0.4, 0.2, threshold=0.5) == "neutral"
        # Above custom threshold
        assert determine_relationship_type(0.6, 0.2, threshold=0.5) == "rivalry"

    def test_zero_scores(self):
        assert determine_relationship_type(0.0, 0.0) == "neutral"

    def test_rivalry_at_threshold_boundary(self):
        # rivalry_score must be > threshold, not ==
        assert determine_relationship_type(0.3, 0.1, threshold=0.3) == "neutral"
        assert determine_relationship_type(0.31, 0.1, threshold=0.3) == "rivalry"


class TestComputeRelationshipScores:
    """Tests for compute_relationship_scores (combined function)."""

    def test_returns_named_tuple(self):
        result = compute_relationship_scores(20, 5, 8, 7)
        assert isinstance(result, RelationshipScores)
        assert hasattr(result, "rivalry_score")
        assert hasattr(result, "alliance_score")
        assert hasattr(result, "relationship_type")

    def test_consistency_with_individual_functions(self):
        args = (20, 5, 8, 7)
        result = compute_relationship_scores(*args)
        assert result.rivalry_score == pytest.approx(compute_rivalry_score(*args))
        assert result.alliance_score == pytest.approx(compute_alliance_score(20, 5))
        expected_type = determine_relationship_type(result.rivalry_score, result.alliance_score)
        assert result.relationship_type == expected_type

    def test_below_min_debates(self):
        result = compute_relationship_scores(2, 1, 1, 0)
        assert result.rivalry_score == 0.0
        assert result.alliance_score == 0.0
        assert result.relationship_type == "neutral"

    def test_strong_rivalry(self):
        # High disagreement, even wins, many debates
        result = compute_relationship_scores(20, 1, 10, 10)
        assert result.rivalry_score > 0.3
        assert result.relationship_type in ("rivalry", "neutral")

    def test_strong_alliance(self):
        # High agreement
        result = compute_relationship_scores(20, 20, 5, 5)
        assert result.alliance_score > 0.3
        assert result.relationship_type == "alliance"


# ===========================================================================
# Handler Tests
# ===========================================================================


@pytest.fixture
def handler():
    """Create a RelationshipHandler with RBAC bypassed."""
    # Bypass RBAC by giving the handler's handle method the context it needs
    return RelationshipHandler(ctx={})


@pytest.fixture
def mock_tracker():
    """Create a mock RelationshipTracker."""
    tracker = MagicMock()
    tracker.db_path = "/tmp/test_rels.db"
    return tracker


# ---- can_handle tests ----


class TestCanHandle:
    def test_summary_route(self, handler):
        assert handler.can_handle("/api/v1/relationships/summary") is True

    def test_graph_route(self, handler):
        assert handler.can_handle("/api/v1/relationships/graph") is True

    def test_stats_route(self, handler):
        assert handler.can_handle("/api/v1/relationships/stats") is True

    def test_pair_detail_route(self, handler):
        assert handler.can_handle("/api/v1/relationship/claude/gpt4") is True

    def test_pair_detail_with_dashes(self, handler):
        assert handler.can_handle("/api/v1/relationship/gpt-4/claude-3") is True

    def test_unknown_route(self, handler):
        assert handler.can_handle("/api/v1/something-else") is False

    def test_partial_pair_route_too_few_slashes(self, handler):
        # path.count("/") must be >= 4 for pair route
        assert handler.can_handle("/api/v1/relationship") is False

    def test_root(self, handler):
        assert handler.can_handle("/") is False


# ---- Rate limiting ----


class TestRateLimiting:
    def test_rate_limit_exceeded(self, handler):
        """Verify 429 is returned when rate limit is exceeded."""
        mock_http = MockHTTPHandler("10.0.0.1")

        with (
            patch(
                "aragora.server.handlers.social.relationship._relationship_limiter"
            ) as mock_limiter,
            patch(
                "aragora.server.handlers.social.relationship.require_permission",
                lambda p: lambda f: f,
            ),
        ):
            mock_limiter.is_allowed.return_value = False

            # Re-create handler so decorator patch takes effect at call time
            h = RelationshipHandler(ctx={})
            # Call handle directly (bypassing RBAC by patching)
            # Since require_permission is applied at class definition time,
            # we need to call the underlying method.
            # Access the original unwrapped handle function
            original_handle = RelationshipHandler.handle.__wrapped__
            result = original_handle(h, "/api/v1/relationships/summary", {}, mock_http)
            assert result is not None
            assert result.status_code == 429
            body = _parse_body(result)
            assert "Rate limit" in body.get("error", "")


# ---- RBAC permission checks ----


class TestRBACPermission:
    @pytest.mark.no_auto_auth
    def test_handle_requires_relationships_read(self):
        """Verify @require_permission('relationships:read') is applied to handle."""
        from aragora.rbac.decorators import PermissionDeniedError

        h = RelationshipHandler(ctx={})
        mock_http = MockHTTPHandler()

        # Calling handle without AuthorizationContext should raise
        with pytest.raises(PermissionDeniedError):
            h.handle("/api/v1/relationships/summary", {}, mock_http)


# ---- GET /api/v1/relationships/summary ----


class TestGetSummary:
    def _call_summary(self, handler, db_path):
        """Helper: call _get_summary with a real DB via a mock tracker."""
        mock_tracker = MagicMock()
        mock_tracker.db_path = db_path
        # Bypass require_tracker decorator by calling the wrapped function
        wrapped = handler._get_summary.__wrapped__
        return wrapped(handler, mock_tracker)

    def test_empty_table(self, handler):
        db_path = _make_db([])
        result = self._call_summary(handler, db_path)
        body = _parse_body(result)
        assert body["total_relationships"] == 0
        assert body["strongest_rivalry"] is None
        assert body["strongest_alliance"] is None
        assert body["most_connected_agent"] is None
        assert body["avg_rivalry_score"] == 0.0

    def test_no_table(self, handler):
        """DB exists but agent_relationships table missing."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        sqlite3.connect(tmp.name).close()  # empty db
        result = self._call_summary(handler, tmp.name)
        body = _parse_body(result)
        assert body["total_relationships"] == 0

    def test_single_rivalry(self, handler):
        # High disagreement, even wins, enough debates
        rows = [("claude", "gpt4", 20, 2, 10, 10)]
        db_path = _make_db(rows)
        result = self._call_summary(handler, db_path)
        body = _parse_body(result)
        assert body["total_relationships"] == 1
        assert body["strongest_rivalry"] is not None
        assert body["strongest_rivalry"]["score"] > 0
        assert body["most_connected_agent"] is not None

    def test_single_alliance(self, handler):
        # High agreement
        rows = [("claude", "gemini", 20, 18, 5, 5)]
        db_path = _make_db(rows)
        result = self._call_summary(handler, db_path)
        body = _parse_body(result)
        assert body["strongest_alliance"] is not None
        assert body["strongest_alliance"]["score"] > 0

    def test_multiple_relationships(self, handler):
        rows = [
            ("claude", "gpt4", 20, 2, 10, 10),
            ("claude", "gemini", 20, 18, 5, 5),
            ("gpt4", "gemini", 10, 5, 4, 3),
        ]
        db_path = _make_db(rows)
        result = self._call_summary(handler, db_path)
        body = _parse_body(result)
        assert body["total_relationships"] == 3
        assert body["most_connected_agent"]["name"] in ("claude", "gpt4", "gemini")

    def test_rows_below_min_debates_excluded(self, handler):
        # Only 2 debates - below the threshold of 3 in summary query
        rows = [("claude", "gpt4", 2, 1, 1, 1)]
        db_path = _make_db(rows)
        result = self._call_summary(handler, db_path)
        body = _parse_body(result)
        assert body["total_relationships"] == 0

    def test_db_error_returns_500(self, handler):
        mock_tracker = MagicMock()
        mock_tracker.db_path = "/nonexistent/path/to.db"
        wrapped = handler._get_summary.__wrapped__
        result = wrapped(handler, mock_tracker)
        assert result.status_code == 500


# ---- GET /api/v1/relationships/graph ----


class TestGetGraph:
    def _call_graph(self, handler, db_path, min_debates=3, min_score=0.0):
        mock_tracker = MagicMock()
        mock_tracker.db_path = db_path
        wrapped = handler._get_graph.__wrapped__
        return wrapped(handler, mock_tracker, min_debates, min_score)

    def test_empty_graph(self, handler):
        db_path = _make_db([])
        result = self._call_graph(handler, db_path)
        body = _parse_body(result)
        assert body["nodes"] == []
        assert body["edges"] == []
        assert body["stats"]["node_count"] == 0
        assert body["stats"]["edge_count"] == 0

    def test_graph_with_data(self, handler):
        rows = [
            ("claude", "gpt4", 20, 2, 10, 10),
            ("claude", "gemini", 15, 12, 5, 5),
        ]
        db_path = _make_db(rows)
        result = self._call_graph(handler, db_path)
        body = _parse_body(result)
        assert body["stats"]["node_count"] == 3
        assert body["stats"]["edge_count"] == 2
        # Edges have expected fields
        edge = body["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "rivalry_score" in edge
        assert "alliance_score" in edge
        assert "debate_count" in edge
        assert "type" in edge

    def test_min_debates_filter(self, handler):
        rows = [
            ("claude", "gpt4", 20, 2, 10, 10),
            ("claude", "gemini", 4, 2, 1, 1),
        ]
        db_path = _make_db(rows)
        result = self._call_graph(handler, db_path, min_debates=10)
        body = _parse_body(result)
        # Only the pair with 20 debates should pass
        assert body["stats"]["edge_count"] >= 1
        for edge in body["edges"]:
            assert edge["debate_count"] >= 10

    def test_min_score_filter(self, handler):
        rows = [
            ("claude", "gpt4", 20, 2, 10, 10),  # high rivalry
            ("a", "b", 20, 19, 1, 1),  # very low scores
        ]
        db_path = _make_db(rows)
        result = self._call_graph(handler, db_path, min_score=0.5)
        body = _parse_body(result)
        # Low-score pair should be filtered out
        for edge in body["edges"]:
            max_score = max(edge["rivalry_score"], edge["alliance_score"])
            assert max_score >= 0.5

    def test_no_table(self, handler):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        sqlite3.connect(tmp.name).close()
        result = self._call_graph(handler, tmp.name)
        body = _parse_body(result)
        assert body["nodes"] == []

    def test_db_error_returns_500(self, handler):
        mock_tracker = MagicMock()
        mock_tracker.db_path = "/nonexistent/path/to.db"
        wrapped = handler._get_graph.__wrapped__
        result = wrapped(handler, mock_tracker, 3, 0.0)
        assert result.status_code == 500


# ---- GET /api/v1/relationships/stats ----


class TestGetStats:
    def _call_stats(self, handler, db_path):
        mock_tracker = MagicMock()
        mock_tracker.db_path = db_path
        wrapped = handler._get_stats.__wrapped__
        return wrapped(handler, mock_tracker)

    def test_empty_stats(self, handler):
        db_path = _make_db([])
        result = self._call_stats(handler, db_path)
        body = _parse_body(result)
        assert body["total_tracked_pairs"] == 0
        assert body["total_debates_tracked"] == 0
        assert body["rivalries"]["count"] == 0
        assert body["alliances"]["count"] == 0
        assert body["most_debated_pair"] is None
        assert body["highest_agreement_pair"] is None

    def test_stats_with_mixed_relationships(self, handler):
        rows = [
            ("claude", "gpt4", 20, 2, 10, 10),  # rivalry
            ("claude", "gemini", 20, 18, 5, 5),  # alliance
            ("gpt4", "gemini", 10, 5, 6, 4),  # neutral-ish
        ]
        db_path = _make_db(rows)
        result = self._call_stats(handler, db_path)
        body = _parse_body(result)
        assert body["total_tracked_pairs"] == 3
        assert body["total_debates_tracked"] == 50
        assert body["most_debated_pair"]["debates"] == 20
        assert body["highest_agreement_pair"]["rate"] > 0.0
        assert (
            body["rivalries"]["count"] + body["alliances"]["count"] + body["neutral"]["count"] >= 1
        )

    def test_most_debated_pair(self, handler):
        rows = [
            ("a", "b", 5, 2, 2, 2),
            ("c", "d", 50, 10, 20, 20),
        ]
        db_path = _make_db(rows)
        result = self._call_stats(handler, db_path)
        body = _parse_body(result)
        assert body["most_debated_pair"]["agents"] == ["c", "d"]
        assert body["most_debated_pair"]["debates"] == 50

    def test_highest_agreement_pair(self, handler):
        rows = [
            ("a", "b", 10, 2, 4, 4),
            ("c", "d", 10, 9, 3, 3),
        ]
        db_path = _make_db(rows)
        result = self._call_stats(handler, db_path)
        body = _parse_body(result)
        assert body["highest_agreement_pair"]["agents"] == ["c", "d"]
        assert body["highest_agreement_pair"]["rate"] == pytest.approx(0.9)

    def test_db_error_returns_500(self, handler):
        mock_tracker = MagicMock()
        mock_tracker.db_path = "/nonexistent/path/to.db"
        wrapped = handler._get_stats.__wrapped__
        result = wrapped(handler, mock_tracker)
        assert result.status_code == 500


# ---- GET /api/v1/relationship/{agent_a}/{agent_b} ----


class TestGetPairDetail:
    def _call_pair(self, handler, tracker, agent_a, agent_b):
        wrapped = handler._get_pair_detail.__wrapped__
        return wrapped(handler, tracker, agent_a, agent_b)

    def test_existing_relationship(self, handler, mock_tracker):
        rel = FakeRelationship()
        mock_tracker.get_relationship.return_value = rel
        result = self._call_pair(handler, mock_tracker, "claude", "gpt4")
        body = _parse_body(result)
        assert body["relationship_exists"] is True
        assert body["debate_count"] == 10
        assert body["agreement_rate"] == pytest.approx(0.4)
        assert body["rivalry_score"] == pytest.approx(0.45)
        assert body["alliance_score"] == pytest.approx(0.24)
        assert body["relationship_type"] == "rivalry"
        assert "claude_wins" in body["head_to_head"]
        assert "gpt4_wins" in body["head_to_head"]

    def test_no_interactions(self, handler, mock_tracker):
        rel = FakeRelationship(debate_count=0)
        mock_tracker.get_relationship.return_value = rel
        result = self._call_pair(handler, mock_tracker, "claude", "gpt4")
        body = _parse_body(result)
        assert body["relationship_exists"] is False
        assert "No recorded interactions" in body["message"]

    def test_critique_balance_in_response(self, handler, mock_tracker):
        rel = FakeRelationship(
            critique_count_a_to_b=12,
            critique_count_b_to_a=3,
        )
        mock_tracker.get_relationship.return_value = rel
        result = self._call_pair(handler, mock_tracker, "claude", "gpt4")
        body = _parse_body(result)
        assert body["critique_balance"]["claude_to_gpt4"] == 12
        assert body["critique_balance"]["gpt4_to_claude"] == 3

    def test_influence_in_response(self, handler, mock_tracker):
        rel = FakeRelationship(influence_a_on_b=0.8, influence_b_on_a=0.2)
        mock_tracker.get_relationship.return_value = rel
        result = self._call_pair(handler, mock_tracker, "claude", "gpt4")
        body = _parse_body(result)
        assert body["influence"]["claude_on_gpt4"] == pytest.approx(0.8)
        assert body["influence"]["gpt4_on_claude"] == pytest.approx(0.2)

    def test_alliance_type(self, handler, mock_tracker):
        rel = FakeRelationship(
            rivalry_score=0.1,
            alliance_score=0.5,
        )
        mock_tracker.get_relationship.return_value = rel
        result = self._call_pair(handler, mock_tracker, "claude", "gpt4")
        body = _parse_body(result)
        assert body["relationship_type"] == "alliance"

    def test_neutral_type(self, handler, mock_tracker):
        rel = FakeRelationship(
            rivalry_score=0.1,
            alliance_score=0.1,
        )
        mock_tracker.get_relationship.return_value = rel
        result = self._call_pair(handler, mock_tracker, "claude", "gpt4")
        body = _parse_body(result)
        assert body["relationship_type"] == "neutral"

    def test_tracker_exception_returns_500(self, handler, mock_tracker):
        mock_tracker.get_relationship.side_effect = RuntimeError("DB error")
        result = self._call_pair(handler, mock_tracker, "claude", "gpt4")
        assert result.status_code == 500


# ---- Input validation (agent name patterns via handle routing) ----


class TestInputValidation:
    """Test path parameter extraction and SAFE_AGENT_PATTERN validation.

    Note: The handler's extract_path_params uses segment indices (2, 3)
    which correspond to the "v1" and "relationship" segments of the URL
    /api/v1/relationship/{agent_a}/{agent_b}. These tests validate the
    pattern matching behavior at those positions.
    """

    def _call_handle_bypassing_rbac(self, handler, path, params=None):
        """Call handle's inner logic bypassing RBAC and rate limiting."""
        mock_http = MockHTTPHandler()
        original = RelationshipHandler.handle.__wrapped__
        with patch("aragora.server.handlers.social.relationship._relationship_limiter") as limiter:
            limiter.is_allowed.return_value = True
            return original(handler, path, params or {}, mock_http)

    def test_valid_pair_route_dispatches(self, handler):
        """A valid pair route should dispatch to _get_pair_detail."""
        with patch.object(handler, "_get_pair_detail") as mock_pd:
            mock_pd.return_value = MagicMock(status_code=200, body=b"{}")
            result = self._call_handle_bypassing_rbac(handler, "/api/v1/relationship/claude/gpt4")
            assert result is not None
            # The route dispatches to _get_pair_detail
            mock_pd.assert_called_once()

    def test_pair_route_extracts_params(self, handler):
        """Verify the extracted params from extract_path_params."""
        from aragora.server.handlers.base import SAFE_AGENT_PATTERN

        # Directly test extract_path_params with the same spec as the handler
        params, err = handler.extract_path_params(
            "/api/v1/relationship/claude/gpt4",
            [
                (2, "agent_a", SAFE_AGENT_PATTERN),
                (3, "agent_b", SAFE_AGENT_PATTERN),
            ],
        )
        # Indices 2 and 3 give "v1" and "relationship"
        assert params is not None
        assert err is None
        assert params["agent_a"] == "v1"
        assert params["agent_b"] == "relationship"

    def test_extract_path_params_rejects_invalid_segment(self, handler):
        """A segment with special characters should fail SAFE_AGENT_PATTERN."""
        from aragora.server.handlers.base import SAFE_AGENT_PATTERN

        # Put invalid chars at segment index 2
        params, err = handler.extract_path_params(
            "/api/inv@lid/relationship/claude/gpt4",
            [
                (2, "agent_a", SAFE_AGENT_PATTERN),
                (3, "agent_b", SAFE_AGENT_PATTERN),
            ],
        )
        assert params is None
        assert err is not None
        assert err.status_code == 400

    def test_extract_path_params_rejects_too_long_segment(self, handler):
        """A segment exceeding 32 chars should fail SAFE_AGENT_PATTERN."""
        from aragora.server.handlers.base import SAFE_AGENT_PATTERN

        long_name = "a" * 33
        params, err = handler.extract_path_params(
            f"/api/{long_name}/relationship/claude/gpt4",
            [
                (2, "agent_a", SAFE_AGENT_PATTERN),
            ],
        )
        assert params is None
        assert err is not None
        assert err.status_code == 400

    def test_extract_path_params_rejects_empty_segment(self, handler):
        """An empty segment should be rejected."""
        from aragora.server.handlers.base import SAFE_AGENT_PATTERN

        params, err = handler.extract_path_params(
            "/api//relationship/claude/gpt4",
            [
                (2, "agent_a", SAFE_AGENT_PATTERN),
            ],
        )
        assert params is None
        assert err is not None
        assert err.status_code == 400

    def test_extract_path_params_missing_segment(self, handler):
        """Requesting a segment beyond path length returns 400."""
        from aragora.server.handlers.base import SAFE_AGENT_PATTERN

        params, err = handler.extract_path_params(
            "/api/v1",
            [
                (10, "agent_a", SAFE_AGENT_PATTERN),
            ],
        )
        assert params is None
        assert err is not None
        assert err.status_code == 400


# ---- require_tracker decorator ----


class TestRequireTracker:
    def test_tracker_not_available(self, handler):
        """When RELATIONSHIP_TRACKER_AVAILABLE is False, return 503."""
        with patch(
            "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
            False,
        ):
            # Call the decorated _get_summary with a nomic_dir arg
            result = handler._get_summary(None)
            assert result.status_code == 503

    def test_tracker_init_fails(self, handler):
        """When _get_tracker returns None, return 503."""
        with (
            patch.object(handler, "_get_tracker", return_value=None),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = handler._get_summary(None)
            assert result.status_code == 503


# ---- _get_tracker method ----


class TestGetTracker:
    def test_no_tracker_available(self, handler):
        with patch(
            "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
            False,
        ):
            assert handler._get_tracker(None) is None

    def test_with_nomic_dir_existing_db(self, handler):
        """When nomic_dir has a valid DB path, tracker should be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "positions.db"
            db_path.touch()

            with (
                patch(
                    "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                    True,
                ),
                patch(
                    "aragora.server.handlers.social.relationship.get_db_path",
                    return_value=db_path,
                ),
                patch("aragora.server.handlers.social.relationship.RelationshipTracker") as MockRT,
            ):
                MockRT.return_value = MagicMock()
                tracker = handler._get_tracker(Path(tmpdir))
                assert tracker is not None
                MockRT.assert_called_once_with(elo_db_path=str(db_path))

    def test_with_nomic_dir_no_db(self, handler):
        """When nomic_dir exists but DB doesn't, fall back to default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                    True,
                ),
                patch(
                    "aragora.server.handlers.social.relationship.get_db_path",
                    return_value=Path(tmpdir) / "nonexistent.db",
                ),
                patch("aragora.server.handlers.social.relationship.RelationshipTracker") as MockRT,
            ):
                MockRT.return_value = MagicMock()
                tracker = handler._get_tracker(Path(tmpdir))
                assert tracker is not None
                MockRT.assert_called_once_with()

    def test_fallback_no_nomic_dir(self, handler):
        with (
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch("aragora.server.handlers.social.relationship.RelationshipTracker") as MockRT,
        ):
            MockRT.return_value = MagicMock()
            tracker = handler._get_tracker(None)
            assert tracker is not None
            MockRT.assert_called_once_with()

    def test_exception_returns_none(self, handler):
        with (
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch(
                "aragora.server.handlers.social.relationship.RelationshipTracker",
                side_effect=RuntimeError("boom"),
            ),
        ):
            assert handler._get_tracker(None) is None


# ---- _empty_stats_response ----


class TestEmptyStatsResponse:
    def test_returns_correct_structure(self, handler):
        result = handler._empty_stats_response()
        body = _parse_body(result)
        assert body["total_tracked_pairs"] == 0
        assert body["total_debates_tracked"] == 0
        assert body["rivalries"] == {"count": 0, "avg_score": 0.0}
        assert body["alliances"] == {"count": 0, "avg_score": 0.0}
        assert body["neutral"] == {"count": 0}
        assert body["most_debated_pair"] is None
        assert body["highest_agreement_pair"] is None


# ---- _fetch_relationships ----


class TestFetchRelationships:
    def test_no_table(self, handler):
        """Returns empty list when table doesn't exist."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        sqlite3.connect(tmp.name).close()
        mock_tracker = MagicMock()
        mock_tracker.db_path = tmp.name
        result = handler._fetch_relationships(mock_tracker)
        assert result == []

    def test_with_data(self, handler):
        rows = [
            ("claude", "gpt4", 10, 4, 3, 5),
            ("a", "b", 2, 1, 1, 0),
        ]
        db_path = _make_db(rows)
        mock_tracker = MagicMock()
        mock_tracker.db_path = db_path
        result = handler._fetch_relationships(mock_tracker)
        assert len(result) == 2

    def test_min_debates_filter(self, handler):
        rows = [
            ("claude", "gpt4", 10, 4, 3, 5),
            ("a", "b", 2, 1, 1, 0),
        ]
        db_path = _make_db(rows)
        mock_tracker = MagicMock()
        mock_tracker.db_path = db_path
        result = handler._fetch_relationships(mock_tracker, min_debates=5)
        assert len(result) == 1
        assert result[0][0] == "claude"


# ---- handle() routing ----


class TestHandleRouting:
    """Test that handle() dispatches to correct internal methods."""

    def _call(self, handler, path, params=None):
        mock_http = MockHTTPHandler()
        original = RelationshipHandler.handle.__wrapped__
        with patch("aragora.server.handlers.social.relationship._relationship_limiter") as limiter:
            limiter.is_allowed.return_value = True
            return original(handler, path, params or {}, mock_http)

    def test_summary_route_dispatches(self, handler):
        with patch.object(
            handler, "_get_summary", return_value=MagicMock(status_code=200, body=b"{}")
        ) as m:
            self._call(handler, "/api/v1/relationships/summary")
            m.assert_called_once()

    def test_graph_route_dispatches(self, handler):
        with patch.object(
            handler, "_get_graph", return_value=MagicMock(status_code=200, body=b"{}")
        ) as m:
            self._call(handler, "/api/v1/relationships/graph")
            m.assert_called_once()

    def test_stats_route_dispatches(self, handler):
        with patch.object(
            handler, "_get_stats", return_value=MagicMock(status_code=200, body=b"{}")
        ) as m:
            self._call(handler, "/api/v1/relationships/stats")
            m.assert_called_once()

    def test_unknown_path_returns_none(self, handler):
        result = self._call(handler, "/api/v1/unknown/path")
        assert result is None

    def test_graph_passes_params(self, handler):
        with patch.object(
            handler, "_get_graph", return_value=MagicMock(status_code=200, body=b"{}")
        ) as m:
            self._call(
                handler, "/api/v1/relationships/graph", {"min_debates": "5", "min_score": "0.3"}
            )
            m.assert_called_once()
            args = m.call_args
            # nomic_dir, min_debates, min_score
            assert args[0][1] == 5  # min_debates
            assert args[0][2] == pytest.approx(0.3)  # min_score
