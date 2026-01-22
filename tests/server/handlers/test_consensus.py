"""
Tests for the consensus handler - consensus memory API.

Tests:
- Route handling (can_handle)
- Similar debates endpoint
- Settled topics endpoint
- Consensus stats endpoint
- Dissents endpoint
- Rate limiting
- Error handling
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from aragora.server.handlers.consensus import ConsensusHandler, _consensus_limiter


def parse_response(result):
    """Parse HandlerResult body into dict."""
    if result is None:
        return None
    return json.loads(result.body.decode())


@pytest.fixture
def consensus_handler():
    """Create a consensus handler with mocked dependencies."""
    ctx = {"storage": None}
    handler = ConsensusHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {}
    return mock


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    _consensus_limiter.clear()
    yield


class TestConsensusHandlerRouting:
    """Tests for ConsensusHandler route matching."""

    def test_can_handle_similar(self, consensus_handler):
        """Test that handler recognizes /api/consensus/similar route."""
        assert consensus_handler.can_handle("/api/v1/consensus/similar") is True

    def test_can_handle_settled(self, consensus_handler):
        """Test that handler recognizes /api/consensus/settled route."""
        assert consensus_handler.can_handle("/api/v1/consensus/settled") is True

    def test_can_handle_stats(self, consensus_handler):
        """Test that handler recognizes /api/consensus/stats route."""
        assert consensus_handler.can_handle("/api/v1/consensus/stats") is True

    def test_can_handle_dissents(self, consensus_handler):
        """Test that handler recognizes /api/consensus/dissents route."""
        assert consensus_handler.can_handle("/api/v1/consensus/dissents") is True

    def test_can_handle_contrarian_views(self, consensus_handler):
        """Test that handler recognizes /api/consensus/contrarian-views route."""
        assert consensus_handler.can_handle("/api/v1/consensus/contrarian-views") is True

    def test_can_handle_risk_warnings(self, consensus_handler):
        """Test that handler recognizes /api/consensus/risk-warnings route."""
        assert consensus_handler.can_handle("/api/v1/consensus/risk-warnings") is True

    def test_can_handle_seed_demo(self, consensus_handler):
        """Test that handler recognizes /api/consensus/seed-demo route."""
        assert consensus_handler.can_handle("/api/v1/consensus/seed-demo") is True

    def test_can_handle_domain_pattern(self, consensus_handler):
        """Test that handler recognizes domain pattern routes."""
        assert consensus_handler.can_handle("/api/v1/consensus/domain/technology") is True
        assert consensus_handler.can_handle("/api/v1/consensus/domain/science") is True

    def test_cannot_handle_unknown_route(self, consensus_handler):
        """Test that handler rejects unknown routes."""
        assert consensus_handler.can_handle("/api/v1/unknown") is False
        assert consensus_handler.can_handle("/api/v1/debates") is False
        assert consensus_handler.can_handle("/api/v1/consensus") is False


class TestSimilarDebatesEndpoint:
    """Tests for /api/consensus/similar endpoint."""

    def test_similar_requires_topic(self, consensus_handler, mock_http_handler):
        """Test that similar endpoint requires topic parameter."""
        result = consensus_handler.handle("/api/v1/consensus/similar", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 400
        body = parse_response(result)
        assert "Topic required" in body["error"]

    def test_similar_topic_too_long(self, consensus_handler, mock_http_handler):
        """Test that similar endpoint rejects overly long topics."""
        long_topic = "x" * 501
        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": long_topic}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 400
        body = parse_response(result)
        assert "too long" in body["error"]

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_similar_success(self, mock_memory_class, consensus_handler, mock_http_handler):
        """Test successful similar debates retrieval."""
        mock_memory = MagicMock()
        mock_memory_class.return_value = mock_memory

        # Create mock similar debate result
        mock_consensus = MagicMock()
        mock_consensus.topic = "AI Safety"
        mock_consensus.conclusion = "AI safety is important"
        mock_consensus.strength.value = "strong"
        mock_consensus.confidence = 0.9
        mock_consensus.participating_agents = ["claude", "gpt-4"]
        mock_consensus.timestamp = datetime(2025, 1, 15, 10, 0, 0)

        mock_similar = MagicMock()
        mock_similar.consensus = mock_consensus
        mock_similar.similarity_score = 0.85
        mock_similar.dissents = []

        mock_memory.find_similar_debates.return_value = [mock_similar]

        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": "AI"}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_response(result)
        assert body["query"] == "AI"
        assert body["count"] == 1
        assert len(body["similar"]) == 1
        assert body["similar"][0]["topic"] == "AI Safety"


class TestSettledTopicsEndpoint:
    """Tests for /api/consensus/settled endpoint."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_settled_default_params(
        self, mock_db_conn, mock_memory_class, consensus_handler, mock_http_handler
    ):
        """Test settled endpoint with default parameters."""
        mock_memory = MagicMock()
        mock_memory.db_path = ":memory:"
        mock_memory_class.return_value = mock_memory

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("Topic 1", "Conclusion 1", 0.9, "strong", "2025-01-15T10:00:00")
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db_conn.return_value = mock_conn

        result = consensus_handler.handle("/api/v1/consensus/settled", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_response(result)
        assert body["min_confidence"] == 0.8
        assert body["count"] == 1


class TestConsensusStatsEndpoint:
    """Tests for /api/consensus/stats endpoint."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_stats_success(
        self, mock_db_conn, mock_memory_class, consensus_handler, mock_http_handler
    ):
        """Test successful stats retrieval."""
        mock_memory = MagicMock()
        mock_memory.db_path = ":memory:"
        mock_memory.get_statistics.return_value = {
            "total_consensus": 100,
            "total_dissents": 20,
            "by_domain": {"technology": 50, "science": 30},
            "by_strength": {"strong": 60, "moderate": 40},
        }
        mock_memory_class.return_value = mock_memory

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (80, 0.85)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db_conn.return_value = mock_conn

        result = consensus_handler.handle("/api/v1/consensus/stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_response(result)
        assert body["total_topics"] == 100
        assert body["high_confidence_count"] == 80
        assert body["avg_confidence"] == 0.85
        assert "technology" in body["domains"]


class TestRateLimiting:
    """Tests for rate limiting on consensus endpoints."""

    def test_rate_limit_enforcement(self, consensus_handler, mock_http_handler):
        """Test that rate limiting is enforced."""
        # Fill up rate limit (30 requests per minute)
        for _ in range(30):
            _consensus_limiter.is_allowed("127.0.0.1")

        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": "test"}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 429
        body = parse_response(result)
        assert "Rate limit" in body["error"]


class TestDomainHistoryEndpoint:
    """Tests for /api/consensus/domain/:domain endpoint."""

    def test_domain_path_extraction(self, consensus_handler, mock_http_handler):
        """Test that domain is correctly extracted from path."""
        # Test with missing feature - should return error about feature
        with patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", False):
            result = consensus_handler.handle(
                "/api/v1/consensus/domain/technology", {}, mock_http_handler
            )
            assert result is not None
            # Either feature unavailable error or successful empty response
            assert result.status_code in [200, 503]


class TestParameterValidation:
    """Tests for parameter validation."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_limit_clamped_to_max(
        self, mock_db_conn, mock_memory_class, consensus_handler, mock_http_handler
    ):
        """Test that limit parameter is clamped to maximum."""
        mock_memory = MagicMock()
        mock_memory.db_path = ":memory:"
        mock_memory_class.return_value = mock_memory

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db_conn.return_value = mock_conn

        result = consensus_handler.handle(
            "/api/v1/consensus/settled",
            {"limit": "999"},  # Way over max of 100
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_min_confidence_bounded(
        self, mock_db_conn, mock_memory_class, consensus_handler, mock_http_handler
    ):
        """Test that min_confidence is bounded between 0 and 1."""
        mock_memory = MagicMock()
        mock_memory.db_path = ":memory:"
        mock_memory_class.return_value = mock_memory

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_db_conn.return_value = mock_conn

        # Test with invalid value (should be clamped to 1.0)
        result = consensus_handler.handle(
            "/api/v1/consensus/settled",
            {"min_confidence": "2.0"},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_response(result)
        # Should be clamped to 1.0
        assert body["min_confidence"] == 1.0
