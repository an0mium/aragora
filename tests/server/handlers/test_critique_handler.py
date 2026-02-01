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
    @patch("aragora.server.handlers.critique.get_critique_store")
    def test_patterns_with_data(self, mock_get_store, tmp_path):
        """Should return patterns when data exists."""
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
        mock_get_store.return_value = mock_store

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
    @patch("aragora.server.handlers.critique.get_critique_store")
    def test_archive_with_data(self, mock_get_store, tmp_path):
        """Should return archive stats when data exists."""
        mock_store = MagicMock()
        mock_store.get_archive_stats.return_value = {"archived": 50, "by_type": {"logic": 25}}
        mock_get_store.return_value = mock_store

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

    def test_limit_clamped_to_min(self):
        """Should clamp limit parameter to minimum value."""
        handler = CritiqueHandler({"nomic_dir": Path("/nonexistent")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        # Limit should be clamped to min of 1
        result = handler.handle("/api/v1/critiques/patterns", {"limit": "0"}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    def test_min_success_negative_bounded(self):
        """Should bound negative min_success parameter to 0.0."""
        handler = CritiqueHandler({"nomic_dir": Path("/nonexistent")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/patterns", {"min_success": "-0.5"}, mock_handler)

        assert result is not None
        assert result.status_code == 200


class TestResponseBodyValidation:
    """Test JSON response body structure."""

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_patterns_response_structure(self, mock_store_class, tmp_path):
        """Should return correct JSON structure for patterns."""
        import json

        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.retrieve_patterns.return_value = [
            MockCritiquePattern(
                issue_type="clarity",
                pattern_text="Ensure clear terminology",
                success_rate=0.9,
                usage_count=25,
            ),
        ]
        mock_store.get_stats.return_value = {"total": 100, "active": 80}
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/patterns", {}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "patterns" in body
        assert "count" in body
        assert "stats" in body
        assert len(body["patterns"]) == 1
        assert body["patterns"][0]["issue_type"] == "clarity"
        assert body["patterns"][0]["pattern"] == "Ensure clear terminology"
        assert body["patterns"][0]["success_rate"] == 0.9
        assert body["patterns"][0]["usage_count"] == 25

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_all_reputations_response_structure(self, mock_store_class, tmp_path):
        """Should return correct JSON structure for all reputations."""
        import json

        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = [
            MockReputation(
                agent_name="gpt-4",
                reputation_score=0.88,
                vote_weight=1.1,
                proposal_acceptance_rate=0.7,
                critique_value=0.85,
                debates_participated=100,
            ),
        ]
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/reputation/all", {}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "reputations" in body
        assert "count" in body
        assert body["count"] == 1
        rep = body["reputations"][0]
        assert rep["agent"] == "gpt-4"
        assert rep["score"] == 0.88
        assert rep["vote_weight"] == 1.1
        assert rep["proposal_acceptance_rate"] == 0.7
        assert rep["critique_value"] == 0.85
        assert rep["debates_participated"] == 100

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_agent_reputation_response_structure(self, mock_store_class, tmp_path):
        """Should return correct JSON structure for agent reputation."""
        import json

        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_reputation.return_value = MockReputation(
            agent_name="gemini",
            reputation_score=0.85,
            vote_weight=1.0,
            proposal_acceptance_rate=0.65,
            critique_value=0.80,
            debates_participated=75,
        )
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/agent/gemini/reputation", {}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent"] == "gemini"
        assert "reputation" in body
        rep = body["reputation"]
        assert rep["score"] == 0.85
        assert rep["vote_weight"] == 1.0
        assert rep["proposal_acceptance_rate"] == 0.65
        assert rep["critique_value"] == 0.80
        assert rep["debates_participated"] == 75

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_agent_not_found_response_structure(self, mock_store_class, tmp_path):
        """Should return correct JSON structure when agent not found."""
        import json

        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_reputation.return_value = None
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/agent/nonexistent/reputation", {}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent"] == "nonexistent"
        assert body["reputation"] is None
        assert "message" in body


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_patterns_store_exception(self, mock_store_class, tmp_path):
        """Should return 500 when store raises exception."""
        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.retrieve_patterns.side_effect = Exception("Database connection failed")
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/patterns", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_archive_store_exception(self, mock_store_class, tmp_path):
        """Should return 500 when archive stats raises exception."""
        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_archive_stats.side_effect = Exception("IO Error")
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/archive", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_all_reputations_store_exception(self, mock_store_class, tmp_path):
        """Should return 500 when get_all_reputations raises exception."""
        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_all_reputations.side_effect = Exception("Query failed")
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/reputation/all", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_agent_reputation_store_exception(self, mock_store_class, tmp_path):
        """Should return 500 when get_reputation raises exception."""
        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_reputation.side_effect = Exception("Timeout")
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/agent/claude/reputation", {}, mock_handler)

        assert result is not None
        assert result.status_code == 500


class TestNomicDirNone:
    """Test behavior when nomic_dir is None."""

    def test_patterns_nomic_dir_none(self):
        """Should return empty patterns when nomic_dir is None."""
        import json

        handler = CritiqueHandler({"nomic_dir": None})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/patterns", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["patterns"] == []
        assert body["count"] == 0

    def test_archive_nomic_dir_none(self):
        """Should return empty archive stats when nomic_dir is None."""
        import json

        handler = CritiqueHandler({"nomic_dir": None})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/archive", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["archived"] == 0
        assert body["by_type"] == {}

    def test_all_reputations_nomic_dir_none(self):
        """Should return empty reputations when nomic_dir is None."""
        import json

        handler = CritiqueHandler({"nomic_dir": None})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/reputation/all", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["reputations"] == []
        assert body["count"] == 0

    def test_agent_reputation_nomic_dir_none(self):
        """Should return no data message when nomic_dir is None."""
        import json

        handler = CritiqueHandler({"nomic_dir": None})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/agent/claude/reputation", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agent"] == "claude"
        assert body["reputation"] is None
        assert "message" in body


class TestAgentNameValidation:
    """Test various agent name validation scenarios."""

    def test_reject_empty_agent_name(self):
        """Should reject empty agent names."""
        handler = CritiqueHandler({})
        # Path with empty agent name
        assert handler._extract_agent_name("/api/agent//reputation") is None

    def test_reject_special_characters(self):
        """Should reject agent names with path traversal characters."""
        handler = CritiqueHandler({})
        assert handler._extract_agent_name("/api/agent/./reputation") is None
        assert handler._extract_agent_name("/api/agent/../reputation") is None

    def test_accept_valid_agent_names(self):
        """Should accept various valid agent name formats."""
        handler = CritiqueHandler({})
        # Standard names
        assert handler._extract_agent_name("/api/agent/claude/reputation") == "claude"
        # Names with hyphens
        assert handler._extract_agent_name("/api/agent/gpt-4-turbo/reputation") == "gpt-4-turbo"
        # Names with underscores
        assert handler._extract_agent_name("/api/agent/claude_v2/reputation") == "claude_v2"
        # Names with numbers
        assert handler._extract_agent_name("/api/agent/mistral7b/reputation") == "mistral7b"

    def test_invalid_agent_returns_400(self):
        """Should return 400 for invalid agent name in path."""
        handler = CritiqueHandler({"nomic_dir": Path("/tmp")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        # Path traversal attempt should be rejected
        result = handler.handle("/api/v1/agent/../etc/reputation", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400


class TestIntegration:
    """Integration tests covering all routes."""

    def test_all_routes_reachable(self):
        """Test all defined routes are reachable."""
        handler = CritiqueHandler({"nomic_dir": Path("/nonexistent")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        routes_to_test = [
            ("/api/v1/critiques/patterns", {}),
            ("/api/v1/critiques/archive", {}),
            ("/api/v1/reputation/all", {}),
            ("/api/v1/agent/claude/reputation", {}),
        ]

        for path, params in routes_to_test:
            result = handler.handle(path, params, mock_handler)
            assert result is not None, f"Route {path} returned None"
            # Should return success (200) or service unavailable (503)
            assert result.status_code in [200, 503], (
                f"Route {path} returned unexpected {result.status_code}"
            )

    def test_handle_returns_none_for_unknown_route(self):
        """Handler should return None for unhandled routes."""
        handler = CritiqueHandler({"nomic_dir": Path("/tmp")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/unknown/path", {}, mock_handler)

        assert result is None

    def test_version_prefix_handling(self):
        """Should handle paths with and without version prefix."""
        handler = CritiqueHandler({})

        # With version prefix
        assert handler.can_handle("/api/v1/critiques/patterns") is True
        assert handler.can_handle("/api/v2/critiques/patterns") is True

        # Without version prefix (directly matching ROUTES)
        assert handler.can_handle("/api/critiques/patterns") is True

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_multiple_patterns_returned(self, mock_store_class, tmp_path):
        """Should handle multiple patterns in response."""
        import json

        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.retrieve_patterns.return_value = [
            MockCritiquePattern("logic", "Check fallacies", 0.9, 50),
            MockCritiquePattern("clarity", "Use clear terms", 0.85, 30),
            MockCritiquePattern("evidence", "Cite sources", 0.8, 25),
        ]
        mock_store.get_stats.return_value = {"total": 200}
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/patterns", {"limit": "5"}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 3
        assert len(body["patterns"]) == 3

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", True)
    @patch("aragora.server.handlers.critique.CritiqueStore")
    def test_multiple_reputations_returned(self, mock_store_class, tmp_path):
        """Should handle multiple reputations in response."""
        import json

        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = [
            MockReputation("claude", 0.92, 1.2, 0.75, 0.88, 150),
            MockReputation("gpt-4", 0.88, 1.1, 0.70, 0.85, 120),
            MockReputation("gemini", 0.85, 1.0, 0.68, 0.82, 100),
        ]
        mock_store_class.return_value = mock_store

        handler = CritiqueHandler({"nomic_dir": tmp_path})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/reputation/all", {}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 3
        assert len(body["reputations"]) == 3


class TestStoreUnavailable:
    """Test all endpoints when CritiqueStore is unavailable."""

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False)
    def test_archive_store_unavailable(self):
        """Should return 503 when store is unavailable for archive."""
        handler = CritiqueHandler({"nomic_dir": Path("/tmp")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/critiques/archive", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False)
    def test_all_reputations_store_unavailable(self):
        """Should return 503 when store is unavailable for all reputations."""
        handler = CritiqueHandler({"nomic_dir": Path("/tmp")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/reputation/all", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.critique.CRITIQUE_STORE_AVAILABLE", False)
    def test_agent_reputation_store_unavailable(self):
        """Should return 503 when store is unavailable for agent reputation."""
        handler = CritiqueHandler({"nomic_dir": Path("/tmp")})
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/agent/claude/reputation", {}, mock_handler)

        assert result is not None
        assert result.status_code == 503
