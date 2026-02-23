"""
Tests for social relationship handler.

Tests cover:
- RelationshipHandler routing (can_handle, handle dispatch)
- Summary endpoint (/api/v1/relationships/summary)
- Graph endpoint (/api/v1/relationships/graph)
- Stats endpoint (/api/v1/relationships/stats)
- Pair detail endpoint (/api/v1/relationship/{agent_a}/{agent_b})
- Score computation utilities
- Rate limiting
- Tracker unavailable / init failure
- Edge cases: empty data, invalid agents, missing table
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.social.relationship import (
    RelationshipHandler,
    RelationshipScores,
    compute_alliance_score,
    compute_relationship_scores,
    compute_rivalry_score,
    determine_relationship_type,
)


# ============================================================================
# Helpers
# ============================================================================


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
        return result.get("status", result.get("status_code", 200))
    return result.status_code


def _make_mock_handler() -> MagicMock:
    """Create a mock HTTP handler with client_address for rate-limit extraction."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"Content-Type": "application/json", "Content-Length": "2"}
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = b"{}"
    return mock


def _create_db(tmp_path: Path, rows: list[tuple]) -> Path:
    """Create a SQLite DB with an agent_relationships table and seed rows.

    Each row is (agent_a, agent_b, debate_count, agreement_count,
                 a_wins_over_b, b_wins_over_a).
    """
    db_path = tmp_path / "positions.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE agent_relationships (
            agent_a TEXT NOT NULL,
            agent_b TEXT NOT NULL,
            debate_count INTEGER NOT NULL DEFAULT 0,
            agreement_count INTEGER NOT NULL DEFAULT 0,
            a_wins_over_b INTEGER NOT NULL DEFAULT 0,
            b_wins_over_a INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (agent_a, agent_b)
        )
    """)
    for row in rows:
        conn.execute("INSERT INTO agent_relationships VALUES (?, ?, ?, ?, ?, ?)", row)
    conn.commit()
    conn.close()
    return db_path


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    return _make_mock_handler()


@pytest.fixture(autouse=True)
def _bypass_rate_limiter(monkeypatch):
    """Bypass the module-level rate limiter for all tests by default."""
    import aragora.server.handlers.social.relationship as mod

    monkeypatch.setattr(mod._relationship_limiter, "is_allowed", lambda key: True)
    yield


@pytest.fixture
def handler():
    """Create a RelationshipHandler with no context."""
    return RelationshipHandler()


@pytest.fixture
def handler_with_dir(tmp_path):
    """Create a RelationshipHandler with nomic_dir set."""
    return RelationshipHandler(ctx={"nomic_dir": tmp_path})


# ============================================================================
# Score Computation: compute_rivalry_score
# ============================================================================


class TestComputeRivalryScore:
    """Tests for the compute_rivalry_score utility."""

    def test_below_min_debates(self):
        """Return 0 if fewer than 3 debates."""
        assert compute_rivalry_score(2, 0, 1, 1) == 0.0

    def test_zero_debates(self):
        assert compute_rivalry_score(0, 0, 0, 0) == 0.0

    def test_full_disagreement_equal_wins(self):
        """High rivalry: no agreements, equal wins, many debates."""
        score = compute_rivalry_score(20, 0, 10, 10)
        # disagreement_rate=1, competitiveness=1, frequency_factor=1 => 1.0
        assert score == pytest.approx(1.0)

    def test_full_agreement_drops_score(self):
        """Full agreement means disagreement_rate=0, score=0."""
        score = compute_rivalry_score(10, 10, 5, 5)
        assert score == pytest.approx(0.0)

    def test_one_sided_wins_lower_competitiveness(self):
        """One-sided wins reduce competitiveness factor."""
        score = compute_rivalry_score(20, 0, 20, 0)
        # competitiveness = 1 - 20/20 = 0
        assert score == pytest.approx(0.0)

    def test_partial_disagreement(self):
        """Partial values produce expected intermediate score."""
        score = compute_rivalry_score(10, 5, 3, 2)
        # disagreement_rate = 0.5, competitiveness = 1 - 1/5 = 0.8, freq = 0.5
        assert score == pytest.approx(0.5 * 0.8 * 0.5)

    def test_few_debates_low_frequency(self):
        """3 debates give frequency_factor = 3/20 = 0.15."""
        score = compute_rivalry_score(3, 0, 1, 2)
        assert score > 0
        assert score < 0.3  # frequency caps it low

    def test_exactly_three_debates(self):
        """Minimum debates threshold boundary."""
        score = compute_rivalry_score(3, 1, 1, 1)
        assert score > 0


class TestComputeAllianceScore:
    """Tests for the compute_alliance_score utility."""

    def test_below_min_debates(self):
        assert compute_alliance_score(2, 2) == 0.0

    def test_zero_debates(self):
        assert compute_alliance_score(0, 0) == 0.0

    def test_full_agreement(self):
        """Full agreement gives max alliance = 1.0 * 0.6 = 0.6."""
        score = compute_alliance_score(10, 10)
        assert score == pytest.approx(0.6)

    def test_no_agreement(self):
        """Zero agreements give zero alliance."""
        score = compute_alliance_score(10, 0)
        assert score == pytest.approx(0.0)

    def test_partial_agreement(self):
        score = compute_alliance_score(10, 5)
        assert score == pytest.approx(0.5 * 0.6)


class TestDetermineRelationshipType:
    """Tests for determine_relationship_type."""

    def test_rivalry_dominates(self):
        assert determine_relationship_type(0.5, 0.2) == "rivalry"

    def test_alliance_dominates(self):
        assert determine_relationship_type(0.2, 0.5) == "alliance"

    def test_both_below_threshold(self):
        assert determine_relationship_type(0.1, 0.1) == "neutral"

    def test_equal_scores_above_threshold(self):
        """Ties go to neutral because neither > the other."""
        assert determine_relationship_type(0.4, 0.4) == "neutral"

    def test_custom_threshold(self):
        assert determine_relationship_type(0.3, 0.2, threshold=0.25) == "rivalry"
        assert determine_relationship_type(0.3, 0.2, threshold=0.35) == "neutral"

    def test_zero_scores(self):
        assert determine_relationship_type(0.0, 0.0) == "neutral"


class TestComputeRelationshipScores:
    """Tests for the combined compute_relationship_scores helper."""

    def test_returns_named_tuple(self):
        result = compute_relationship_scores(20, 0, 10, 10)
        assert isinstance(result, RelationshipScores)
        assert hasattr(result, "rivalry_score")
        assert hasattr(result, "alliance_score")
        assert hasattr(result, "relationship_type")

    def test_high_rivalry(self):
        scores = compute_relationship_scores(20, 0, 10, 10)
        assert scores.rivalry_score > 0.5
        assert scores.relationship_type == "rivalry"

    def test_high_alliance(self):
        scores = compute_relationship_scores(20, 20, 0, 0)
        assert scores.alliance_score > 0
        assert scores.relationship_type == "alliance"

    def test_below_threshold(self):
        scores = compute_relationship_scores(1, 0, 0, 0)
        assert scores.rivalry_score == 0.0
        assert scores.alliance_score == 0.0
        assert scores.relationship_type == "neutral"


# ============================================================================
# can_handle
# ============================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_summary_route(self, handler):
        assert handler.can_handle("/api/v1/relationships/summary") is True

    def test_graph_route(self, handler):
        assert handler.can_handle("/api/v1/relationships/graph") is True

    def test_stats_route(self, handler):
        assert handler.can_handle("/api/v1/relationships/stats") is True

    def test_pair_route(self, handler):
        assert handler.can_handle("/api/v1/relationship/claude/gpt4") is True

    def test_pair_route_with_dashes(self, handler):
        assert handler.can_handle("/api/v1/relationship/agent-a/agent-b") is True

    def test_unrelated_route(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_partial_route(self, handler):
        assert handler.can_handle("/api/v1/relationships") is False

    def test_pair_route_single_segment(self, handler):
        """One agent name still matches can_handle (path.count('/') >= 4).

        The handler accepts this at the routing level but extract_path_params
        will fail during handle() since segment 5 is missing.
        """
        assert handler.can_handle("/api/v1/relationship/claude") is True


# ============================================================================
# Rate limiting
# ============================================================================


class TestRateLimiting:
    """Tests for rate-limit enforcement on the handle() method."""

    def test_rate_limit_exceeded(self, handler, mock_http_handler, monkeypatch):
        """When rate limiter rejects, return 429."""
        import aragora.server.handlers.social.relationship as mod

        monkeypatch.setattr(mod._relationship_limiter, "is_allowed", lambda key: False)
        result = handler.handle("/api/v1/relationships/summary", {}, mock_http_handler)
        assert _status(result) == 429
        assert "Rate limit" in _body(result).get("error", "")


# ============================================================================
# Tracker unavailable / init failure
# ============================================================================


class TestTrackerUnavailable:
    """Tests when relationship tracker module is not installed."""

    def test_summary_returns_503_when_tracker_unavailable(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
            False,
        ):
            result = handler.handle("/api/v1/relationships/summary", {}, mock_http_handler)
            assert _status(result) == 503

    def test_graph_returns_503_when_tracker_unavailable(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
            False,
        ):
            result = handler.handle("/api/v1/relationships/graph", {}, mock_http_handler)
            assert _status(result) == 503

    def test_stats_returns_503_when_tracker_unavailable(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
            False,
        ):
            result = handler.handle("/api/v1/relationships/stats", {}, mock_http_handler)
            assert _status(result) == 503

    def test_pair_returns_503_when_tracker_unavailable(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
            False,
        ):
            result = handler.handle("/api/v1/relationship/alice/bob", {}, mock_http_handler)
            assert _status(result) == 503

    def test_tracker_init_failure(self, handler, mock_http_handler):
        """When _get_tracker returns None (init error), return 503."""
        with (
            patch.object(handler, "_get_tracker", return_value=None),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = handler.handle("/api/v1/relationships/summary", {}, mock_http_handler)
            assert _status(result) == 503
            assert "initialize" in _body(result).get("error", "").lower()


# ============================================================================
# _get_tracker
# ============================================================================


class TestGetTracker:
    """Tests for the _get_tracker helper."""

    def test_returns_none_when_not_available(self, handler):
        with patch(
            "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
            False,
        ):
            assert handler._get_tracker(None) is None

    def test_fallback_default_tracker(self, handler):
        """When nomic_dir is None, falls back to default constructor."""
        mock_cls = MagicMock()
        with (
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch(
                "aragora.server.handlers.social.relationship.RelationshipTracker",
                mock_cls,
            ),
        ):
            result = handler._get_tracker(None)
            mock_cls.assert_called_once_with()
            assert result == mock_cls.return_value

    def test_uses_db_path_when_exists(self, tmp_path):
        """When nomic_dir is set and DB exists, uses it."""
        mock_cls = MagicMock()
        fake_db = tmp_path / "fake.db"
        fake_db.touch()
        h = RelationshipHandler(ctx={"nomic_dir": tmp_path})
        with (
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch(
                "aragora.server.handlers.social.relationship.RelationshipTracker",
                mock_cls,
            ),
            patch(
                "aragora.server.handlers.social.relationship.get_db_path",
                return_value=fake_db,
            ),
        ):
            result = h._get_tracker(tmp_path)
            mock_cls.assert_called_once_with(elo_db_path=str(fake_db))
            assert result == mock_cls.return_value

    def test_falls_back_when_db_missing(self, tmp_path):
        """When nomic_dir set but DB does not exist, falls back to default."""
        mock_cls = MagicMock()
        missing = tmp_path / "nonexistent.db"
        h = RelationshipHandler(ctx={"nomic_dir": tmp_path})
        with (
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch(
                "aragora.server.handlers.social.relationship.RelationshipTracker",
                mock_cls,
            ),
            patch(
                "aragora.server.handlers.social.relationship.get_db_path",
                return_value=missing,
            ),
        ):
            result = h._get_tracker(tmp_path)
            mock_cls.assert_called_once_with()

    def test_catches_os_error(self, handler):
        mock_cls = MagicMock(side_effect=OSError("disk full"))
        with (
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch(
                "aragora.server.handlers.social.relationship.RelationshipTracker",
                mock_cls,
            ),
        ):
            result = handler._get_tracker(None)
            assert result is None

    def test_catches_runtime_error(self, handler):
        mock_cls = MagicMock(side_effect=RuntimeError("bad state"))
        with (
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch(
                "aragora.server.handlers.social.relationship.RelationshipTracker",
                mock_cls,
            ),
        ):
            result = handler._get_tracker(None)
            assert result is None


# ============================================================================
# Summary endpoint
# ============================================================================


class TestSummaryEndpoint:
    """Tests for GET /api/v1/relationships/summary."""

    def _call(self, handler, mock_http, **kwargs):
        return handler.handle("/api/v1/relationships/summary", kwargs, mock_http)

    def test_empty_table(self, tmp_path, mock_http_handler):
        """Empty database returns zero summary."""
        db_path = _create_db(tmp_path, [])
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_relationships"] == 0
        assert body["strongest_rivalry"] is None
        assert body["strongest_alliance"] is None
        assert body["most_connected_agent"] is None

    def test_no_table(self, tmp_path, mock_http_handler):
        """Database without the table returns zero summary."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_relationships"] == 0

    def test_with_relationships(self, tmp_path, mock_http_handler):
        """Summary with actual data computes scores."""
        rows = [
            # rivalry pair: 20 debates, 0 agreement, equal wins
            ("claude", "gpt4", 20, 0, 10, 10),
            # alliance pair: 10 debates, 10 agreements
            ("gemini", "mistral", 10, 10, 0, 0),
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_relationships"] == 2
        assert body["strongest_rivalry"] is not None
        assert body["strongest_rivalry"]["agents"] == ["claude", "gpt4"]
        assert body["strongest_alliance"] is not None
        assert body["avg_rivalry_score"] > 0
        assert body["avg_alliance_score"] > 0

    def test_most_connected_agent(self, tmp_path, mock_http_handler):
        """The agent in most pairs should be most_connected."""
        rows = [
            ("claude", "gpt4", 5, 1, 2, 2),
            ("claude", "gemini", 5, 1, 2, 2),
            ("claude", "mistral", 5, 1, 2, 2),
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert body["most_connected_agent"]["name"] == "claude"
        assert body["most_connected_agent"]["relationship_count"] == 3

    def test_rows_below_min_debates_ignored(self, tmp_path, mock_http_handler):
        """Summary only includes rows with debate_count >= 3."""
        rows = [
            ("alice", "bob", 2, 2, 1, 1),  # below threshold
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert body["total_relationships"] == 0

    def test_summary_error_handling(self, mock_http_handler):
        """Internal errors return 500."""
        tracker = MagicMock()
        tracker.db_path = "/nonexistent/path.db"
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch(
                "aragora.server.handlers.social.relationship.get_db_connection",
                side_effect=OSError("disk failure"),
            ),
        ):
            result = self._call(h, mock_http_handler)
        assert _status(result) == 500


# ============================================================================
# Graph endpoint
# ============================================================================


class TestGraphEndpoint:
    """Tests for GET /api/v1/relationships/graph."""

    def _call(self, handler, mock_http, params=None):
        return handler.handle("/api/v1/relationships/graph", params or {}, mock_http)

    def test_empty_graph(self, tmp_path, mock_http_handler):
        """Empty DB returns empty graph."""
        db_path = _create_db(tmp_path, [])
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["nodes"] == []
        assert body["edges"] == []
        assert body["stats"]["node_count"] == 0
        assert body["stats"]["edge_count"] == 0

    def test_graph_with_data(self, tmp_path, mock_http_handler):
        """Graph returns nodes and edges."""
        rows = [
            ("claude", "gpt4", 20, 0, 10, 10),
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["stats"]["node_count"] == 2
        assert body["stats"]["edge_count"] == 1
        edge = body["edges"][0]
        assert edge["source"] == "claude"
        assert edge["target"] == "gpt4"
        assert edge["debate_count"] == 20
        assert "rivalry_score" in edge
        assert "alliance_score" in edge
        assert "type" in edge

    def test_min_debates_filter(self, tmp_path, mock_http_handler):
        """min_debates query param filters out low-count rows."""
        rows = [
            ("claude", "gpt4", 20, 0, 10, 10),
            ("alice", "bob", 2, 1, 1, 0),  # below min_debates=3
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler, params={"min_debates": "5"})
        body = _body(result)
        assert body["stats"]["node_count"] == 2
        assert body["stats"]["edge_count"] == 1

    def test_min_score_filter(self, tmp_path, mock_http_handler):
        """min_score filters out edges below the threshold."""
        rows = [
            # neutral pair: 5 debates, 3 agreements, unequal wins
            ("alice", "bob", 5, 3, 4, 0),
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler, params={"min_score": "0.99"})
        body = _body(result)
        # High min_score should filter out this edge
        assert body["stats"]["edge_count"] == 0

    def test_graph_node_counters(self, tmp_path, mock_http_handler):
        """Nodes track rivals/allies counters."""
        rows = [
            ("claude", "gpt4", 20, 0, 10, 10),  # rivalry
            ("claude", "gemini", 20, 20, 0, 0),  # alliance
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        nodes_by_id = {n["id"]: n for n in body["nodes"]}
        claude_node = nodes_by_id["claude"]
        assert claude_node["rivals"] >= 1
        assert claude_node["allies"] >= 1

    def test_graph_error_handling(self, mock_http_handler):
        """Internal errors return 500."""
        tracker = MagicMock()
        tracker.db_path = "/nonexistent/path.db"
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch.object(h, "_fetch_relationships", side_effect=OSError("disk failure")),
        ):
            result = self._call(h, mock_http_handler)
        assert _status(result) == 500


# ============================================================================
# Stats endpoint
# ============================================================================


class TestStatsEndpoint:
    """Tests for GET /api/v1/relationships/stats."""

    def _call(self, handler, mock_http, params=None):
        return handler.handle("/api/v1/relationships/stats", params or {}, mock_http)

    def test_empty_stats(self, tmp_path, mock_http_handler):
        """Empty DB returns empty stats."""
        db_path = _create_db(tmp_path, [])
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_tracked_pairs"] == 0
        assert body["total_debates_tracked"] == 0
        assert body["rivalries"]["count"] == 0
        assert body["alliances"]["count"] == 0
        assert body["neutral"]["count"] == 0
        assert body["most_debated_pair"] is None
        assert body["highest_agreement_pair"] is None

    def test_stats_with_data(self, tmp_path, mock_http_handler):
        """Stats with actual relationship data."""
        rows = [
            ("claude", "gpt4", 20, 0, 10, 10),  # rivalry
            ("gemini", "mistral", 10, 10, 0, 0),  # alliance
            ("alice", "bob", 5, 2, 2, 1),  # neutral (scores below threshold)
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total_tracked_pairs"] == 3
        assert body["total_debates_tracked"] == 35

    def test_most_debated_pair(self, tmp_path, mock_http_handler):
        """most_debated_pair tracks the pair with most debates."""
        rows = [
            ("claude", "gpt4", 50, 0, 25, 25),
            ("alice", "bob", 5, 2, 2, 1),
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert body["most_debated_pair"]["agents"] == ["claude", "gpt4"]
        assert body["most_debated_pair"]["debates"] == 50

    def test_highest_agreement_pair(self, tmp_path, mock_http_handler):
        """highest_agreement_pair tracks pair with best agreement rate."""
        rows = [
            ("claude", "gpt4", 10, 9, 0, 0),  # 90% agreement
            ("alice", "bob", 10, 5, 3, 2),  # 50% agreement
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert body["highest_agreement_pair"]["agents"] == ["claude", "gpt4"]
        assert body["highest_agreement_pair"]["rate"] == pytest.approx(0.9)

    def test_highest_agreement_requires_min_debates(self, tmp_path, mock_http_handler):
        """Pairs with < 3 debates are excluded from highest_agreement."""
        rows = [
            ("alice", "bob", 2, 2, 0, 0),  # 100% agreement but only 2 debates
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert body["highest_agreement_pair"] is None

    def test_rivalry_and_alliance_counts(self, tmp_path, mock_http_handler):
        """Verify correct classification counts."""
        rows = [
            ("claude", "gpt4", 20, 0, 10, 10),  # rivalry (high rivalry score)
            ("gemini", "mistral", 20, 20, 0, 0),  # alliance (high alliance)
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert body["rivalries"]["count"] >= 1
        assert body["alliances"]["count"] >= 1
        assert body["rivalries"]["avg_score"] > 0
        assert body["alliances"]["avg_score"] > 0

    def test_stats_error_handling(self, mock_http_handler):
        """Internal errors return 500."""
        tracker = MagicMock()
        tracker.db_path = "/nonexistent/path.db"
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch.object(h, "_fetch_relationships", side_effect=ValueError("bad data")),
        ):
            result = self._call(h, mock_http_handler)
        assert _status(result) == 500


# ============================================================================
# Pair detail endpoint
# ============================================================================


class TestPairDetailEndpoint:
    """Tests for GET /api/v1/relationship/{agent_a}/{agent_b}."""

    def _call(self, handler, mock_http, agent_a="claude", agent_b="gpt4"):
        return handler.handle(f"/api/v1/relationship/{agent_a}/{agent_b}", {}, mock_http)

    def test_no_interactions(self, mock_http_handler):
        """When debate_count is 0, returns relationship_exists=False."""
        mock_rel = MagicMock()
        mock_rel.debate_count = 0
        tracker = MagicMock()
        tracker.get_relationship.return_value = mock_rel
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["relationship_exists"] is False
        assert "No recorded interactions" in body["message"]

    def test_existing_relationship(self, mock_http_handler):
        """Full relationship detail with all metrics."""
        mock_rel = MagicMock()
        mock_rel.agent_a = "claude"
        mock_rel.agent_b = "gpt4"
        mock_rel.debate_count = 20
        mock_rel.agreement_count = 5
        mock_rel.rivalry_score = 0.8
        mock_rel.alliance_score = 0.15
        mock_rel.a_wins_over_b = 12
        mock_rel.b_wins_over_a = 8
        mock_rel.critique_count_a_to_b = 30
        mock_rel.critique_count_b_to_a = 25
        mock_rel.influence_a_on_b = 0.65
        mock_rel.influence_b_on_a = 0.45
        tracker = MagicMock()
        tracker.get_relationship.return_value = mock_rel
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["relationship_exists"] is True
        assert body["agent_a"] == "claude"
        assert body["agent_b"] == "gpt4"
        assert body["debate_count"] == 20
        assert body["agreement_count"] == 5
        assert body["agreement_rate"] == pytest.approx(0.25)
        assert body["rivalry_score"] == 0.8
        assert body["alliance_score"] == 0.15
        assert body["relationship_type"] == "rivalry"
        assert body["head_to_head"]["claude_wins"] == 12
        assert body["head_to_head"]["gpt4_wins"] == 8
        assert body["critique_balance"]["claude_to_gpt4"] == 30
        assert body["critique_balance"]["gpt4_to_claude"] == 25
        assert body["influence"]["claude_on_gpt4"] == 0.65
        assert body["influence"]["gpt4_on_claude"] == 0.45

    def test_alliance_type(self, mock_http_handler):
        """Pair with alliance scores gets type 'alliance'."""
        mock_rel = MagicMock()
        mock_rel.agent_a = "gemini"
        mock_rel.agent_b = "mistral"
        mock_rel.debate_count = 10
        mock_rel.agreement_count = 9
        mock_rel.rivalry_score = 0.1
        mock_rel.alliance_score = 0.5
        mock_rel.a_wins_over_b = 0
        mock_rel.b_wins_over_a = 0
        mock_rel.critique_count_a_to_b = 5
        mock_rel.critique_count_b_to_a = 5
        mock_rel.influence_a_on_b = 0.3
        mock_rel.influence_b_on_a = 0.3
        tracker = MagicMock()
        tracker.get_relationship.return_value = mock_rel
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler, "gemini", "mistral")
        body = _body(result)
        assert body["relationship_type"] == "alliance"

    def test_invalid_agent_name(self, mock_http_handler):
        """Agent names with special chars should be rejected (400)."""
        h = RelationshipHandler()
        with (
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch.object(h, "_get_tracker", return_value=MagicMock()),
        ):
            result = self._call(h, mock_http_handler, "claude!!", "gpt4")
        assert _status(result) == 400

    def test_agent_name_too_long(self, mock_http_handler):
        """Agent names exceeding 32 chars should be rejected."""
        h = RelationshipHandler()
        long_name = "a" * 33
        with (
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch.object(h, "_get_tracker", return_value=MagicMock()),
        ):
            result = self._call(h, mock_http_handler, long_name, "gpt4")
        assert _status(result) == 400

    def test_empty_agent_name(self, mock_http_handler):
        """Empty agent segment should fail validation."""
        h = RelationshipHandler()
        # Construct path directly
        with (
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
            patch.object(h, "_get_tracker", return_value=MagicMock()),
        ):
            result = h.handle("/api/v1/relationship//gpt4", {}, mock_http_handler)
        # May return 400 or None depending on can_handle
        if result is not None:
            assert _status(result) == 400

    def test_pair_error_handling(self, mock_http_handler):
        """Internal errors return 500."""
        tracker = MagicMock()
        tracker.get_relationship.side_effect = TypeError("unexpected type")
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        assert _status(result) == 500

    def test_pair_os_error(self, mock_http_handler):
        """OSError in pair detail returns 500."""
        tracker = MagicMock()
        tracker.get_relationship.side_effect = OSError("connection lost")
        h = RelationshipHandler()
        with (
            patch.object(h, "_get_tracker", return_value=tracker),
            patch(
                "aragora.server.handlers.social.relationship.RELATIONSHIP_TRACKER_AVAILABLE",
                True,
            ),
        ):
            result = self._call(h, mock_http_handler)
        assert _status(result) == 500


# ============================================================================
# Handle dispatch / unmatched routes
# ============================================================================


class TestHandleDispatch:
    """Tests for the handle() method routing logic."""

    def test_unmatched_route_returns_none(self, handler, mock_http_handler):
        """Routes not matching any endpoint return None."""
        result = handler.handle("/api/v1/unknown", {}, mock_http_handler)
        assert result is None

    def test_summary_dispatches(self, handler, mock_http_handler):
        """Summary path dispatches to _get_summary."""
        with (
            patch.object(
                handler, "_get_summary", return_value=MagicMock(status_code=200)
            ) as mock_summary,
            patch(
                "aragora.server.handlers.social.relationship._relationship_limiter"
            ) as mock_limiter,
        ):
            mock_limiter.is_allowed.return_value = True
            handler.handle("/api/v1/relationships/summary", {}, mock_http_handler)
            mock_summary.assert_called_once()

    def test_graph_dispatches_with_params(self, handler, mock_http_handler):
        """Graph path dispatches with min_debates and min_score."""
        with (
            patch.object(
                handler, "_get_graph", return_value=MagicMock(status_code=200)
            ) as mock_graph,
            patch(
                "aragora.server.handlers.social.relationship._relationship_limiter"
            ) as mock_limiter,
        ):
            mock_limiter.is_allowed.return_value = True
            handler.handle(
                "/api/v1/relationships/graph",
                {"min_debates": "10", "min_score": "0.5"},
                mock_http_handler,
            )
            mock_graph.assert_called_once()

    def test_stats_dispatches(self, handler, mock_http_handler):
        """Stats path dispatches to _get_stats."""
        with (
            patch.object(
                handler, "_get_stats", return_value=MagicMock(status_code=200)
            ) as mock_stats,
            patch(
                "aragora.server.handlers.social.relationship._relationship_limiter"
            ) as mock_limiter,
        ):
            mock_limiter.is_allowed.return_value = True
            handler.handle("/api/v1/relationships/stats", {}, mock_http_handler)
            mock_stats.assert_called_once()


# ============================================================================
# Handler context / constructor
# ============================================================================


class TestHandlerConstruction:
    """Tests for handler initialization."""

    def test_default_ctx(self):
        h = RelationshipHandler()
        assert h.ctx == {}

    def test_custom_ctx(self, tmp_path):
        h = RelationshipHandler(ctx={"nomic_dir": tmp_path})
        assert h.ctx["nomic_dir"] == tmp_path

    def test_routes_attribute(self, handler):
        assert len(handler.ROUTES) == 4


# ============================================================================
# _empty_stats_response
# ============================================================================


class TestEmptyStatsResponse:
    """Tests for the _empty_stats_response helper."""

    def test_structure(self, handler):
        result = handler._empty_stats_response()
        body = _body(result)
        assert body["total_tracked_pairs"] == 0
        assert body["total_debates_tracked"] == 0
        assert body["rivalries"]["count"] == 0
        assert body["rivalries"]["avg_score"] == 0.0
        assert body["alliances"]["count"] == 0
        assert body["alliances"]["avg_score"] == 0.0
        assert body["neutral"]["count"] == 0
        assert body["most_debated_pair"] is None
        assert body["highest_agreement_pair"] is None


# ============================================================================
# _fetch_relationships
# ============================================================================


class TestFetchRelationships:
    """Tests for the _fetch_relationships helper."""

    def test_empty_table(self, handler, tmp_path):
        db_path = _create_db(tmp_path, [])
        tracker = MagicMock()
        tracker.db_path = db_path
        result = handler._fetch_relationships(tracker)
        assert result == []

    def test_no_table(self, handler, tmp_path):
        """Missing table returns empty list gracefully."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()
        tracker = MagicMock()
        tracker.db_path = db_path
        result = handler._fetch_relationships(tracker)
        assert result == []

    def test_fetches_rows(self, handler, tmp_path):
        rows = [
            ("claude", "gpt4", 10, 5, 3, 2),
            ("alice", "bob", 3, 1, 1, 1),
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        result = handler._fetch_relationships(tracker)
        assert len(result) == 2

    def test_min_debates_filter(self, handler, tmp_path):
        rows = [
            ("claude", "gpt4", 10, 5, 3, 2),
            ("alice", "bob", 2, 1, 1, 0),
        ]
        db_path = _create_db(tmp_path, rows)
        tracker = MagicMock()
        tracker.db_path = db_path
        result = handler._fetch_relationships(tracker, min_debates=5)
        assert len(result) == 1
        assert result[0][0] == "claude"
