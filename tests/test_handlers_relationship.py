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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_relationships (
                agent_a TEXT, agent_b TEXT, debate_count INTEGER,
                agreement_count INTEGER, a_wins_over_b INTEGER, b_wins_over_a INTEGER,
                PRIMARY KEY (agent_a, agent_b)
            )
        """)
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_relationships (
                agent_a TEXT, agent_b TEXT, debate_count INTEGER,
                agreement_count INTEGER, a_wins_over_b INTEGER, b_wins_over_a INTEGER,
                PRIMARY KEY (agent_a, agent_b)
            )
        """)
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_relationships (
                agent_a TEXT, agent_b TEXT, debate_count INTEGER,
                agreement_count INTEGER, a_wins_over_b INTEGER, b_wins_over_a INTEGER,
                PRIMARY KEY (agent_a, agent_b)
            )
        """)
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
