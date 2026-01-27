"""
Tests for Critique Handler.

Tests the critique pattern and reputation API endpoints including:
- Critique pattern retrieval
- Archive statistics
- Agent reputation queries
- Rate limiting
- Input validation
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from aragora.server.handlers.critique import CritiqueHandler


@dataclass
class MockCritiquePattern:
    """Mock critique pattern for testing."""

    issue_type: str
    pattern_text: str
    success_rate: float
    usage_count: int


@dataclass
class MockReputation:
    """Mock reputation data for testing."""

    agent_name: str
    reputation_score: float
    vote_weight: float
    proposal_acceptance_rate: float
    critique_value: float
    debates_participated: int


class TestCritiqueHandlerRouting:
    """Test CritiqueHandler routing logic."""

    def test_routes_defined(self):
        """Handler should define expected routes."""
        # ROUTES uses normalized paths without version prefix
        assert "/api/critiques/patterns" in CritiqueHandler.ROUTES
        assert "/api/critiques/archive" in CritiqueHandler.ROUTES
        assert "/api/reputation/all" in CritiqueHandler.ROUTES

    def test_can_handle_static_routes(self):
        """Should handle static routes."""
        handler = CritiqueHandler({})
        assert handler.can_handle("/api/v1/critiques/patterns") is True
        assert handler.can_handle("/api/v1/critiques/archive") is True
        assert handler.can_handle("/api/v1/reputation/all") is True

    def test_can_handle_dynamic_agent_route(self):
        """Should handle dynamic agent reputation route."""
        handler = CritiqueHandler({})
        assert handler.can_handle("/api/v1/agent/claude/reputation") is True
        assert handler.can_handle("/api/v1/agent/gpt-4/reputation") is True

    def test_cannot_handle_unknown_routes(self):
        """Should not handle unknown routes."""
        handler = CritiqueHandler({})
        assert handler.can_handle("/api/v1/unknown") is False
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/agent/claude") is False


class TestExtractAgentName:
    """Test agent name extraction and validation."""

    def test_extract_valid_agent_name(self):
        """Should extract valid agent names."""
        handler = CritiqueHandler({})
        # _extract_agent_name expects paths with version prefix already stripped
        assert handler._extract_agent_name("/api/agent/claude/reputation") == "claude"
        assert handler._extract_agent_name("/api/agent/gpt-4/reputation") == "gpt-4"

    def test_reject_path_traversal(self):
        """Should reject path traversal attempts."""
        handler = CritiqueHandler({})
        assert handler._extract_agent_name("/api/v1/agent/../etc/passwd/reputation") is None
        assert handler._extract_agent_name("/api/v1/agent/..%2F..%2Fetc/reputation") is None

    def test_reject_invalid_path(self):
        """Should reject invalid paths."""
        handler = CritiqueHandler({})
        # Short paths don't have enough parts
        assert handler._extract_agent_name("/api/") is None
        assert handler._extract_agent_name("/api/v1/agent") is None


class TestCritiquePatterns:
    """Test critique pattern retrieval."""

    def test_patterns_no_db(self):
        """Should return empty when database doesn't exist."""
        handler = CritiqueHandler({"nomic_dir": Path("/nonexistent")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/patterns", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_patterns_with_data(self, mock_store_class, tmp_path):
        """Should return patterns when data exists."""
        # Create mock database file
        db_path = tmp_path / "debates.db"
        db_path.touch()

        # Setup mock store
        mock_store = MagicMock()
        mock_store.retrieve_patterns.return_value = [
            MockCritiquePattern(
                issue_type="logic",
                pattern_text="Check for logical fallacies",
                success_rate=0.85,
                usage_count=42,
            ),
        ]
        mock_store.get_stats.return_value = {"total": 100, "archived": 50}
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/patterns", {"limit": "10"}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        mock_store.retrieve_patterns.assert_called_once()

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False)
    def test_patterns_store_unavailable(self):
        """Should return 503 when store is unavailable."""
        handler = CritiqueHandler({"nomic_dir": Path("/tmp")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/patterns", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503


class TestArchiveStats:
    """Test archive statistics endpoint."""

    def test_archive_no_db(self):
        """Should return empty stats when database doesn't exist."""
        handler = CritiqueHandler({"nomic_dir": Path("/nonexistent")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/archive", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_archive_with_data(self, mock_store_class, tmp_path):
        """Should return archive stats when data exists."""
        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_archive_stats.return_value = {"archived": 50, "by_type": {"logic": 25}}
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/archive", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200


class TestReputationEndpoints:
    """Test reputation query endpoints."""

    def test_all_reputations_no_db(self):
        """Should return empty when database doesn't exist."""
        handler = CritiqueHandler({"nomic_dir": Path("/nonexistent")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/reputation/all", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_all_reputations_with_data(self, mock_store_class, tmp_path):
        """Should return all reputations when data exists."""
        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = [
            MockReputation(
                agent_name="claude",
                reputation_score=0.92,
                vote_weight=1.2,
                proposal_acceptance_rate=0.75,
                critique_value=0.88,
                debates_participated=150,
            ),
        ]
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/reputation/all", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_agent_reputation_found(self, mock_store_class, tmp_path):
        """Should return agent reputation when found."""
        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_reputation.return_value = MockReputation(
            agent_name="claude",
            reputation_score=0.92,
            vote_weight=1.2,
            proposal_acceptance_rate=0.75,
            critique_value=0.88,
            debates_participated=150,
        )
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/agent/claude/reputation", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        mock_store.get_reputation.assert_called_once_with("claude")

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_agent_reputation_not_found(self, mock_store_class, tmp_path):
        """Should return null reputation when agent not found."""
        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_reputation.return_value = None
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/agent/unknown-agent/reputation", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200


class TestRateLimiting:
    """Test rate limiting behavior."""

    @patch("aragora.server.handlers.critique._critique_limiter")
    def test_rate_limit_exceeded(self, mock_limiter):
        """Should return 429 when rate limit exceeded."""
        mock_limiter.is_allowed.return_value = False

        handler = CritiqueHandler({"nomic_dir": Path("/tmp")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/patterns", {}, mock_handler)

        assert result is not None
        assert result.status_code == 429


class TestParameterValidation:
    """Test query parameter validation."""

    def test_limit_clamped_to_max(self):
        """Should clamp limit parameter to maximum value."""
        handler = CritiqueHandler({"nomic_dir": Path("/nonexistent")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        # Limit should be clamped to max of 50
        result = handler.handle("/api/v1/critiques/patterns", {"limit": "1000"}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    def test_min_success_bounded(self):
        """Should bound min_success parameter to valid range."""
        handler = CritiqueHandler({"nomic_dir": Path("/nonexistent")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        # min_success should be bounded to 0.0-1.0
        result = handler.handle("/api/v1/critiques/patterns", {"min_success": "2.0"}, mock_handler)

        assert result is not None
        assert result.status_code == 200
