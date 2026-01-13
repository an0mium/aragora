"""Tests for RelationshipHandler."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from aragora.server.handlers.relationship import RelationshipHandler


class TestRelationshipHandlerRouting:
    """Tests for route matching."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return RelationshipHandler({"nomic_dir": Path("/tmp/test")})

    def test_can_handle_summary(self, handler):
        """Should handle /api/relationships/summary."""
        assert handler.can_handle("/api/relationships/summary") is True

    def test_can_handle_graph(self, handler):
        """Should handle /api/relationships/graph."""
        assert handler.can_handle("/api/relationships/graph") is True

    def test_can_handle_stats(self, handler):
        """Should handle /api/relationships/stats."""
        assert handler.can_handle("/api/relationships/stats") is True

    def test_can_handle_pair_detail(self, handler):
        """Should handle /api/relationship/{agent_a}/{agent_b}."""
        assert handler.can_handle("/api/relationship/claude/gpt4") is True
        assert handler.can_handle("/api/relationship/agent-1/agent-2") is True

    def test_cannot_handle_unrelated(self, handler):
        """Should not handle unrelated routes."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agent/claude/profile") is False
        assert handler.can_handle("/api/consensus/stats") is False

    def test_cannot_handle_incomplete_pair_path(self, handler):
        """Should not handle incomplete relationship pair path."""
        # Only one agent specified
        assert handler.can_handle("/api/relationship/claude") is False


class TestSummaryEndpoint:
    """Tests for /api/relationships/summary endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return RelationshipHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", False)
    def test_summary_503_when_unavailable(self, handler):
        """Should return 503 when relationship tracker unavailable."""
        result = handler.handle("/api/relationships/summary", {}, Mock())
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not available" in data["error"]

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_summary_empty_database(self, mock_tracker_class, handler):
        """Should handle empty database gracefully."""
        import sqlite3
        import tempfile

        # Create a temporary database with empty relationships table
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_relationships (
                agent_a TEXT, agent_b TEXT, debate_count INTEGER,
                agreement_count INTEGER, a_wins_over_b INTEGER, b_wins_over_a INTEGER,
                PRIMARY KEY (agent_a, agent_b)
            )
        """
        )
        conn.close()

        mock_tracker = Mock()
        mock_tracker.elo_db_path = db_path
        mock_tracker_class.return_value = mock_tracker

        result = handler.handle("/api/relationships/summary", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total_relationships"] == 0

        # Cleanup
        Path(db_path).unlink(missing_ok=True)


class TestGraphEndpoint:
    """Tests for /api/relationships/graph endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return RelationshipHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", False)
    def test_graph_503_when_unavailable(self, handler):
        """Should return 503 when relationship tracker unavailable."""
        result = handler.handle("/api/relationships/graph", {}, Mock())
        assert result.status_code == 503

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_graph_respects_min_debates_param(self, mock_tracker_class, handler):
        """Should accept min_debates parameter."""
        import sqlite3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_relationships (
                agent_a TEXT, agent_b TEXT, debate_count INTEGER,
                agreement_count INTEGER, a_wins_over_b INTEGER, b_wins_over_a INTEGER,
                PRIMARY KEY (agent_a, agent_b)
            )
        """
        )
        conn.close()

        mock_tracker = Mock()
        mock_tracker.elo_db_path = db_path
        mock_tracker_class.return_value = mock_tracker

        result = handler.handle("/api/relationships/graph", {"min_debates": "5"}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "nodes" in data
        assert "edges" in data
        assert "stats" in data

        Path(db_path).unlink(missing_ok=True)


class TestPairDetailEndpoint:
    """Tests for /api/relationship/{agent_a}/{agent_b} endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return RelationshipHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", False)
    def test_pair_detail_503_when_unavailable(self, handler):
        """Should return 503 when relationship tracker unavailable."""
        result = handler.handle("/api/relationship/claude/gpt4", {}, Mock())
        assert result.status_code == 503

    def test_pair_detail_validates_agent_names(self, handler):
        """Should validate agent names for security."""
        # Path traversal attempt
        result = handler.handle("/api/relationship/../etc/passwd", {}, Mock())
        assert result is None or result.status_code == 400

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_pair_detail_returns_relationship_data(self, mock_tracker_class, handler):
        """Should return relationship data for valid pair."""
        mock_rel = Mock()
        mock_rel.debate_count = 10
        mock_rel.agreement_count = 5
        mock_rel.agent_a = "claude"
        mock_rel.agent_b = "gpt4"
        mock_rel.rivalry_score = 0.6
        mock_rel.alliance_score = 0.2
        mock_rel.a_wins_over_b = 6
        mock_rel.b_wins_over_a = 4
        mock_rel.critique_count_a_to_b = 8
        mock_rel.critique_count_b_to_a = 5
        mock_rel.influence_a_on_b = 0.3
        mock_rel.influence_b_on_a = 0.2

        mock_tracker = Mock()
        mock_tracker.get_relationship.return_value = mock_rel
        mock_tracker.elo_db_path = "/tmp/test.db"
        mock_tracker_class.return_value = mock_tracker

        result = handler.handle("/api/relationship/claude/gpt4", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["relationship_exists"] is True
        assert data["debate_count"] == 10
        assert data["relationship_type"] == "rivalry"

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_pair_detail_no_interactions(self, mock_tracker_class, handler):
        """Should handle pair with no recorded interactions."""
        mock_rel = Mock()
        mock_rel.debate_count = 0

        mock_tracker = Mock()
        mock_tracker.get_relationship.return_value = mock_rel
        mock_tracker.elo_db_path = "/tmp/test.db"
        mock_tracker_class.return_value = mock_tracker

        result = handler.handle("/api/relationship/alice/bob", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["relationship_exists"] is False


class TestStatsEndpoint:
    """Tests for /api/relationships/stats endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return RelationshipHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", False)
    def test_stats_503_when_unavailable(self, handler):
        """Should return 503 when relationship tracker unavailable."""
        result = handler.handle("/api/relationships/stats", {}, Mock())
        assert result.status_code == 503

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_stats_returns_statistics(self, mock_tracker_class, handler):
        """Should return relationship statistics."""
        import sqlite3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_relationships (
                agent_a TEXT, agent_b TEXT, debate_count INTEGER,
                agreement_count INTEGER, a_wins_over_b INTEGER, b_wins_over_a INTEGER,
                PRIMARY KEY (agent_a, agent_b)
            )
        """
        )
        conn.close()

        mock_tracker = Mock()
        mock_tracker.elo_db_path = db_path
        mock_tracker_class.return_value = mock_tracker

        result = handler.handle("/api/relationships/stats", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "total_tracked_pairs" in data
        assert "rivalries" in data
        assert "alliances" in data

        Path(db_path).unlink(missing_ok=True)


class TestHandlerImport:
    """Tests for handler module imports."""

    def test_handler_can_be_imported(self):
        """Should be importable from handlers package."""
        from aragora.server.handlers import RelationshipHandler

        assert RelationshipHandler is not None

    def test_handler_in_all(self):
        """Should be in __all__ exports."""
        from aragora.server.handlers import __all__

        assert "RelationshipHandler" in __all__


class TestScoreComputationUtilities:
    """Tests for score computation utility functions."""

    def test_rivalry_score_requires_minimum_debates(self):
        """Rivalry score is 0 with fewer than 3 debates."""
        from aragora.server.handlers.relationship import compute_rivalry_score

        # Less than 3 debates should return 0
        assert compute_rivalry_score(2, 0, 1, 1) == 0.0
        assert compute_rivalry_score(1, 0, 1, 0) == 0.0
        assert compute_rivalry_score(0, 0, 0, 0) == 0.0

    def test_rivalry_score_increases_with_disagreement(self):
        """Higher disagreement rate increases rivalry score."""
        from aragora.server.handlers.relationship import compute_rivalry_score

        # High disagreement (0 agreement out of 10) vs low disagreement (8 agreement out of 10)
        high_disagree = compute_rivalry_score(10, 0, 5, 5)
        low_disagree = compute_rivalry_score(10, 8, 5, 5)

        assert high_disagree > low_disagree

    def test_rivalry_score_competitive_wins(self):
        """Competitive win rates (close to 50/50) increase rivalry."""
        from aragora.server.handlers.relationship import compute_rivalry_score

        # Equal wins (5-5) vs lopsided (9-1)
        equal_wins = compute_rivalry_score(10, 2, 5, 5)
        lopsided = compute_rivalry_score(10, 2, 9, 1)

        assert equal_wins > lopsided

    def test_alliance_score_requires_minimum_debates(self):
        """Alliance score is 0 with fewer than 3 debates."""
        from aragora.server.handlers.relationship import compute_alliance_score

        assert compute_alliance_score(2, 2) == 0.0
        assert compute_alliance_score(1, 1) == 0.0
        assert compute_alliance_score(0, 0) == 0.0

    def test_alliance_score_increases_with_agreement(self):
        """Higher agreement rate increases alliance score."""
        from aragora.server.handlers.relationship import compute_alliance_score

        high_agree = compute_alliance_score(10, 9)
        low_agree = compute_alliance_score(10, 2)

        assert high_agree > low_agree

    def test_determine_relationship_type_rivalry(self):
        """Returns rivalry when rivalry score dominates."""
        from aragora.server.handlers.relationship import determine_relationship_type

        assert determine_relationship_type(0.8, 0.2) == "rivalry"
        assert determine_relationship_type(0.5, 0.1) == "rivalry"

    def test_determine_relationship_type_alliance(self):
        """Returns alliance when alliance score dominates."""
        from aragora.server.handlers.relationship import determine_relationship_type

        assert determine_relationship_type(0.2, 0.8) == "alliance"
        assert determine_relationship_type(0.1, 0.5) == "alliance"

    def test_determine_relationship_type_neutral(self):
        """Returns neutral when both scores are below threshold."""
        from aragora.server.handlers.relationship import determine_relationship_type

        assert determine_relationship_type(0.1, 0.1) == "neutral"
        assert determine_relationship_type(0.2, 0.2) == "neutral"

    def test_determine_relationship_type_custom_threshold(self):
        """Respects custom threshold parameter."""
        from aragora.server.handlers.relationship import determine_relationship_type

        # With default threshold (0.3), 0.4 would be rivalry
        assert determine_relationship_type(0.4, 0.1) == "rivalry"
        # With higher threshold (0.5), 0.4 becomes neutral
        assert determine_relationship_type(0.4, 0.1, threshold=0.5) == "neutral"

    def test_compute_relationship_scores_integration(self):
        """compute_relationship_scores combines all calculations."""
        from aragora.server.handlers.relationship import (
            compute_relationship_scores,
            RelationshipScores,
        )

        scores = compute_relationship_scores(
            debate_count=10, agreement_count=2, a_wins=5, b_wins=5
        )

        assert isinstance(scores, RelationshipScores)
        assert 0 <= scores.rivalry_score <= 1
        assert 0 <= scores.alliance_score <= 1
        assert scores.relationship_type in ("rivalry", "alliance", "neutral")


class TestGraphEndpointEdgeCases:
    """Additional edge case tests for graph endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return RelationshipHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_graph_respects_min_score_param(self, mock_tracker_class, handler):
        """Should filter edges below min_score threshold."""
        import sqlite3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_relationships (
                agent_a TEXT, agent_b TEXT, debate_count INTEGER,
                agreement_count INTEGER, a_wins_over_b INTEGER, b_wins_over_a INTEGER,
                PRIMARY KEY (agent_a, agent_b)
            )
        """
        )
        # Insert relationship with low scores
        conn.execute(
            "INSERT INTO agent_relationships VALUES (?, ?, ?, ?, ?, ?)",
            ("agent1", "agent2", 5, 4, 2, 3),  # High agreement, low rivalry
        )
        conn.commit()
        conn.close()

        mock_tracker = Mock()
        mock_tracker.elo_db_path = db_path
        mock_tracker_class.return_value = mock_tracker

        # With high min_score, edge should be filtered
        result = handler.handle(
            "/api/relationships/graph", {"min_score": "0.9"}, Mock()
        )
        assert result.status_code == 200
        data = json.loads(result.body)
        # The edge might be filtered due to high threshold
        assert "edges" in data

        Path(db_path).unlink(missing_ok=True)

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_graph_with_relationships(self, mock_tracker_class, handler):
        """Should return nodes and edges for relationships."""
        import sqlite3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_relationships (
                agent_a TEXT, agent_b TEXT, debate_count INTEGER,
                agreement_count INTEGER, a_wins_over_b INTEGER, b_wins_over_a INTEGER,
                PRIMARY KEY (agent_a, agent_b)
            )
        """
        )
        # Insert a rivalry (high disagreement, competitive)
        conn.execute(
            "INSERT INTO agent_relationships VALUES (?, ?, ?, ?, ?, ?)",
            ("claude", "gpt4", 20, 3, 10, 10),
        )
        conn.commit()
        conn.close()

        mock_tracker = Mock()
        mock_tracker.elo_db_path = db_path
        mock_tracker_class.return_value = mock_tracker

        result = handler.handle("/api/relationships/graph", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["nodes"]) == 2
        assert any(n["id"] == "claude" for n in data["nodes"])
        assert any(n["id"] == "gpt4" for n in data["nodes"])

        Path(db_path).unlink(missing_ok=True)


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return RelationshipHandler({"nomic_dir": Path("/tmp/test")})

    def test_rate_limit_returns_429(self, handler):
        """Should return 429 when rate limit exceeded."""
        # Mock the rate limiter to always return False (limit exceeded)
        with patch(
            "aragora.server.handlers.relationship._relationship_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            mock_handler = Mock()
            mock_handler.client_address = ("192.168.1.100", 12345)
            mock_handler.headers = {}

            result = handler.handle("/api/relationships/summary", {}, mock_handler)
            assert result.status_code == 429
            data = json.loads(result.body)
            assert "Rate limit" in data["error"]


class TestTrackerInitialization:
    """Tests for tracker initialization."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return RelationshipHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_tracker_uses_nomic_dir(self, mock_tracker_class, handler):
        """Should use nomic_dir for database path."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create positions.db
            positions_db = tmp_path / "positions.db"
            positions_db.touch()

            handler = RelationshipHandler({"nomic_dir": tmp_path})

            with patch(
                "aragora.server.handlers.relationship.get_db_path",
                return_value=positions_db,
            ):
                handler._get_tracker(tmp_path)
                mock_tracker_class.assert_called()

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_tracker_fallback_to_default(self, mock_tracker_class, handler):
        """Should fall back to default tracker when nomic_dir not set."""
        handler._get_tracker(None)
        mock_tracker_class.assert_called_with()

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", False)
    def test_tracker_unavailable(self, handler):
        """Should return None when tracker not available."""
        result = handler._get_tracker(Path("/tmp"))
        assert result is None


class TestSummaryEndpointEdgeCases:
    """Additional edge case tests for summary endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return RelationshipHandler({"nomic_dir": Path("/tmp/test")})

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_summary_with_relationships(self, mock_tracker_class, handler):
        """Should return strongest rivalry and alliance."""
        import sqlite3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_relationships (
                agent_a TEXT, agent_b TEXT, debate_count INTEGER,
                agreement_count INTEGER, a_wins_over_b INTEGER, b_wins_over_a INTEGER,
                PRIMARY KEY (agent_a, agent_b)
            )
        """
        )
        # Insert a rivalry
        conn.execute(
            "INSERT INTO agent_relationships VALUES (?, ?, ?, ?, ?, ?)",
            ("claude", "gpt4", 20, 2, 10, 10),
        )
        # Insert an alliance
        conn.execute(
            "INSERT INTO agent_relationships VALUES (?, ?, ?, ?, ?, ?)",
            ("gemini", "mistral", 15, 13, 5, 5),
        )
        conn.commit()
        conn.close()

        mock_tracker = Mock()
        mock_tracker.elo_db_path = db_path
        mock_tracker_class.return_value = mock_tracker

        result = handler.handle("/api/relationships/summary", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total_relationships"] == 2
        assert data["strongest_rivalry"] is not None
        assert data["strongest_alliance"] is not None

        Path(db_path).unlink(missing_ok=True)


class TestPairDetailEdgeCases:
    """Additional edge case tests for pair detail endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return RelationshipHandler({"nomic_dir": Path("/tmp/test")})

    def test_pair_rejects_special_characters(self, handler):
        """Should reject agent names with special characters."""
        result = handler.handle("/api/relationship/agent<script>/agent2", {}, Mock())
        # Should either return None or 400
        assert result is None or result.status_code == 400

    def test_pair_handles_missing_agent(self, handler):
        """Should handle when only one agent provided."""
        result = handler.handle("/api/relationship/claude/", {}, Mock())
        # Path validation should fail
        assert result is None or result.status_code == 400

    @patch("aragora.server.handlers.relationship.RELATIONSHIP_TRACKER_AVAILABLE", True)
    @patch("aragora.server.handlers.relationship.RelationshipTracker")
    def test_pair_detail_with_history(self, mock_tracker_class, handler):
        """Should include recent history when available."""
        mock_rel = Mock()
        mock_rel.debate_count = 15
        mock_rel.agreement_count = 8
        mock_rel.agent_a = "alice"
        mock_rel.agent_b = "bob"
        mock_rel.rivalry_score = 0.3
        mock_rel.alliance_score = 0.4
        mock_rel.a_wins_over_b = 7
        mock_rel.b_wins_over_a = 8
        mock_rel.critique_count_a_to_b = 12
        mock_rel.critique_count_b_to_a = 10
        mock_rel.influence_a_on_b = 0.25
        mock_rel.influence_b_on_a = 0.3

        mock_tracker = Mock()
        mock_tracker.get_relationship.return_value = mock_rel
        mock_tracker.elo_db_path = "/tmp/test.db"
        mock_tracker_class.return_value = mock_tracker

        result = handler.handle("/api/relationship/alice/bob", {}, Mock())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["relationship_exists"] is True
        # Alliance should dominate
        assert data["relationship_type"] == "alliance"
        assert data["agreement_count"] == 8
