"""Tests for consensus handler endpoints.

Tests the consensus memory API endpoints including:
- GET /api/consensus/similar - Find similar debates
- GET /api/consensus/settled - Get settled topics
- GET /api/consensus/stats - Get consensus statistics
- GET /api/consensus/dissents - Get recent dissents
- GET /api/consensus/contrarian-views - Get contrarian perspectives
- GET /api/consensus/risk-warnings - Get risk warnings
- GET /api/consensus/domain/:domain - Get domain history
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


class MockConsensusRecord:
    """Mock consensus record."""

    def __init__(
        self,
        topic: str,
        conclusion: str,
        strength: str = "strong",
        confidence: float = 0.85,
        agents: List[str] = None,
        timestamp: datetime = None,
    ):
        self.topic = topic
        self.conclusion = conclusion
        self.strength = MagicMock(value=strength)
        self.confidence = confidence
        self.participating_agents = agents or ["claude", "gpt-4"]
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "conclusion": self.conclusion,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "agents": self.participating_agents,
            "timestamp": self.timestamp.isoformat(),
        }


class MockSimilarDebate:
    """Mock similar debate result."""

    def __init__(
        self,
        consensus: MockConsensusRecord,
        similarity_score: float = 0.9,
        dissents: List[Any] = None,
    ):
        self.consensus = consensus
        self.similarity_score = similarity_score
        self.dissents = dissents or []


class MockDissentRecord:
    """Mock dissent record."""

    def __init__(
        self,
        agent_id: str,
        content: str,
        confidence: float = 0.7,
        reasoning: str = None,
        debate_id: str = "debate_1",
        dissent_type: str = "fundamental_disagreement",
        rebuttal: str = None,
        metadata: Dict = None,
        timestamp: datetime = None,
    ):
        self.agent_id = agent_id
        self.content = content
        self.confidence = confidence
        self.reasoning = reasoning
        self.debate_id = debate_id
        self.dissent_type = MagicMock(value=dissent_type)
        self.rebuttal = rebuttal
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now(timezone.utc)


class MockConsensusMemory:
    """Mock ConsensusMemory for testing."""

    def __init__(self):
        self.db_path = ":memory:"
        self._similar = []
        self._statistics = {
            "total_consensus": 10,
            "total_dissents": 5,
            "by_domain": {"security": 3, "performance": 2},
            "by_strength": {"strong": 5, "moderate": 3, "weak": 2},
        }
        self._domain_history = []

    def find_similar_debates(self, topic: str, limit: int = 5) -> List[MockSimilarDebate]:
        return self._similar[:limit]

    def get_statistics(self) -> dict:
        return self._statistics

    def get_domain_consensus_history(
        self, domain: str, limit: int = 50
    ) -> List[MockConsensusRecord]:
        return self._domain_history[:limit]


class MockDissentRetriever:
    """Mock DissentRetriever for testing."""

    def __init__(self, memory: MockConsensusMemory):
        self._memory = memory
        self._contrarian = []
        self._risk_warnings = []

    def find_contrarian_views(
        self, topic: str, domain: str = None, limit: int = 10
    ) -> List[MockDissentRecord]:
        return self._contrarian[:limit]

    def find_risk_warnings(
        self, topic: str, domain: str = None, limit: int = 10
    ) -> List[MockDissentRecord]:
        return self._risk_warnings[:limit]


class MockHandler:
    """Mock HTTP handler."""

    def __init__(self):
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {}


@pytest.fixture
def mock_handler():
    """Create mock handler."""
    return MockHandler()


@pytest.fixture
def mock_consensus_memory():
    """Create mock consensus memory."""
    return MockConsensusMemory()


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return {}


@pytest.fixture
def consensus_handler(mock_server_context):
    """Create ConsensusHandler for testing."""
    from aragora.server.handlers.consensus import ConsensusHandler

    handler = ConsensusHandler(mock_server_context)
    return handler


class TestConsensusHandlerRouting:
    """Test routing logic for consensus handler."""

    def test_can_handle_similar(self, consensus_handler):
        """Test can_handle for /api/consensus/similar."""
        assert consensus_handler.can_handle("/api/v1/consensus/similar") is True

    def test_can_handle_settled(self, consensus_handler):
        """Test can_handle for /api/consensus/settled."""
        assert consensus_handler.can_handle("/api/v1/consensus/settled") is True

    def test_can_handle_stats(self, consensus_handler):
        """Test can_handle for /api/consensus/stats."""
        assert consensus_handler.can_handle("/api/v1/consensus/stats") is True

    def test_can_handle_dissents(self, consensus_handler):
        """Test can_handle for /api/consensus/dissents."""
        assert consensus_handler.can_handle("/api/v1/consensus/dissents") is True

    def test_can_handle_contrarian(self, consensus_handler):
        """Test can_handle for /api/consensus/contrarian-views."""
        assert consensus_handler.can_handle("/api/v1/consensus/contrarian-views") is True

    def test_can_handle_risk_warnings(self, consensus_handler):
        """Test can_handle for /api/consensus/risk-warnings."""
        assert consensus_handler.can_handle("/api/v1/consensus/risk-warnings") is True

    def test_can_handle_domain(self, consensus_handler):
        """Test can_handle for /api/consensus/domain/:domain."""
        assert consensus_handler.can_handle("/api/v1/consensus/domain/security") is True
        assert consensus_handler.can_handle("/api/v1/consensus/domain/performance") is True

    def test_cannot_handle_invalid(self, consensus_handler):
        """Test can_handle rejects invalid paths."""
        assert consensus_handler.can_handle("/api/v1/other/endpoint") is False
        assert consensus_handler.can_handle("/api/v1/consensus") is False


class TestConsensusHandlerSimilar:
    """Test /api/consensus/similar endpoint."""

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_similar_debates_success(
        self, mock_memory_cls, mock_limiter, consensus_handler, mock_handler
    ):
        """Test successful similar debates retrieval."""
        mock_limiter.is_allowed.return_value = True

        mock_memory = MockConsensusMemory()
        mock_memory._similar = [
            MockSimilarDebate(
                MockConsensusRecord("caching strategies", "Use Redis for hot data"),
                similarity_score=0.92,
            ),
            MockSimilarDebate(
                MockConsensusRecord("database indexing", "Index frequently queried columns"),
                similarity_score=0.85,
            ),
        ]
        mock_memory_cls.return_value = mock_memory

        result = consensus_handler.handle(
            "/api/v1/consensus/similar",
            {"topic": "cache optimization"},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["query"] == "cache optimization"
        assert len(body["similar"]) == 2
        assert body["similar"][0]["similarity"] == 0.92

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    def test_similar_missing_topic(self, mock_limiter, consensus_handler, mock_handler):
        """Test similar endpoint requires topic parameter."""
        mock_limiter.is_allowed.return_value = True

        result = consensus_handler.handle(
            "/api/v1/consensus/similar",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 400
        body = parse_body(result)
        assert "required" in body["error"].lower()

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    def test_similar_topic_too_long(self, mock_limiter, consensus_handler, mock_handler):
        """Test similar endpoint rejects oversized topic."""
        mock_limiter.is_allowed.return_value = True

        result = consensus_handler.handle(
            "/api/v1/consensus/similar",
            {"topic": "x" * 501},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 400
        body = parse_body(result)
        assert "too long" in body["error"].lower()


class TestConsensusHandlerSettled:
    """Test /api/consensus/settled endpoint."""

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_settled_topics_success(
        self, mock_db_conn, mock_memory_cls, mock_limiter, consensus_handler, mock_handler
    ):
        """Test successful settled topics retrieval."""
        mock_limiter.is_allowed.return_value = True

        mock_memory = MockConsensusMemory()
        mock_memory_cls.return_value = mock_memory

        # Mock database connection
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("API design", "REST for public APIs", 0.95, "strong", "2024-01-15T10:00:00"),
            ("Error handling", "Use structured errors", 0.88, "strong", "2024-01-14T10:00:00"),
        ]
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        result = consensus_handler.handle(
            "/api/v1/consensus/settled",
            {"min_confidence": "0.8", "limit": "10"},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["min_confidence"] == 0.8
        assert len(body["topics"]) == 2
        assert body["topics"][0]["confidence"] == 0.95


class TestConsensusHandlerStats:
    """Test /api/consensus/stats endpoint."""

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_stats_success(
        self, mock_db_conn, mock_memory_cls, mock_limiter, consensus_handler, mock_handler
    ):
        """Test successful stats retrieval."""
        mock_limiter.is_allowed.return_value = True

        mock_memory = MockConsensusMemory()
        mock_memory_cls.return_value = mock_memory

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (8, 0.85)
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        result = consensus_handler.handle(
            "/api/v1/consensus/stats",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["total_topics"] == 10
        assert body["high_confidence_count"] == 8
        assert body["avg_confidence"] == 0.85


class TestConsensusHandlerDissents:
    """Test /api/consensus/dissents endpoint."""

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_dissents_success(
        self, mock_db_conn, mock_memory_cls, mock_limiter, consensus_handler, mock_handler
    ):
        """Test successful dissents retrieval."""
        mock_limiter.is_allowed.return_value = True

        mock_memory = MockConsensusMemory()
        mock_memory_cls.return_value = mock_memory

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []  # Empty for simplicity
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        result = consensus_handler.handle(
            "/api/v1/consensus/dissents",
            {"limit": "5"},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "dissents" in body


class TestConsensusHandlerContrarianViews:
    """Test /api/consensus/contrarian-views endpoint."""

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.DissentRetriever")
    def test_contrarian_views_with_topic(
        self, mock_retriever_cls, mock_memory_cls, mock_limiter, consensus_handler, mock_handler
    ):
        """Test contrarian views with topic filter."""
        mock_limiter.is_allowed.return_value = True

        mock_memory = MockConsensusMemory()
        mock_memory_cls.return_value = mock_memory

        mock_retriever = MockDissentRetriever(mock_memory)
        mock_retriever._contrarian = [
            MockDissentRecord(
                "gpt-4",
                "NoSQL might be better for this use case",
                confidence=0.75,
                reasoning="Better horizontal scaling",
            ),
        ]
        mock_retriever_cls.return_value = mock_retriever

        result = consensus_handler.handle(
            "/api/v1/consensus/contrarian-views",
            {"topic": "database choice", "limit": "10"},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "views" in body
        assert len(body["views"]) == 1
        assert body["views"][0]["agent"] == "gpt-4"


class TestConsensusHandlerRiskWarnings:
    """Test /api/consensus/risk-warnings endpoint."""

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.DissentRetriever")
    def test_risk_warnings_success(
        self, mock_retriever_cls, mock_memory_cls, mock_limiter, consensus_handler, mock_handler
    ):
        """Test risk warnings retrieval."""
        mock_limiter.is_allowed.return_value = True

        mock_memory = MockConsensusMemory()
        mock_memory_cls.return_value = mock_memory

        mock_retriever = MockDissentRetriever(mock_memory)
        mock_retriever._risk_warnings = [
            MockDissentRecord(
                "claude",
                "SQL injection risk if input not sanitized",
                confidence=0.9,
                dissent_type="risk_warning",
                metadata={"domain": "security"},
            ),
        ]
        mock_retriever_cls.return_value = mock_retriever

        result = consensus_handler.handle(
            "/api/v1/consensus/risk-warnings",
            {"topic": "user input handling"},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "warnings" in body


class TestConsensusHandlerDomain:
    """Test /api/consensus/domain/:domain endpoint."""

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_domain_history_success(
        self, mock_memory_cls, mock_limiter, consensus_handler, mock_handler
    ):
        """Test domain history retrieval."""
        mock_limiter.is_allowed.return_value = True

        mock_memory = MockConsensusMemory()
        mock_memory._domain_history = [
            MockConsensusRecord("auth tokens", "Use JWTs with short expiry", confidence=0.9),
            MockConsensusRecord("password storage", "Use bcrypt with cost 12", confidence=0.95),
        ]
        mock_memory_cls.return_value = mock_memory

        result = consensus_handler.handle(
            "/api/v1/consensus/domain/security",
            {"limit": "10"},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert body["domain"] == "security"
        assert body["count"] == 2


class TestConsensusHandlerRateLimiting:
    """Test rate limiting for consensus endpoints."""

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    def test_rate_limit_exceeded(self, mock_limiter, consensus_handler, mock_handler):
        """Test rate limit exceeded response."""
        mock_limiter.is_allowed.return_value = False

        result = consensus_handler.handle(
            "/api/v1/consensus/stats",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 429
        body = parse_body(result)
        assert "rate limit" in body["error"].lower()


class TestConsensusHandlerFeatureUnavailable:
    """Test behavior when consensus memory is unavailable."""

    @patch("aragora.server.handlers.consensus._consensus_limiter")
    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", False)
    def test_feature_unavailable(self, mock_limiter, consensus_handler, mock_handler):
        """Test response when consensus memory feature is unavailable."""
        mock_limiter.is_allowed.return_value = True

        result = consensus_handler.handle(
            "/api/v1/consensus/stats",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 503
        body = parse_body(result)
        assert "not available" in body["error"].lower()
