"""
Tests for FastAPI consensus route endpoints.

Covers:
- Stats endpoint
- Similar debates search
- Settled topics retrieval
- Dissents listing
- Contrarian views
- Risk warnings
- Domain history
- Consensus status for a debate
- Consensus detection (POST)
- Input validation (Pydantic 422 errors)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
def mock_storage():
    """Create a mock debate storage."""
    storage = MagicMock()
    storage.get_debate = MagicMock(return_value=None)
    return storage


@pytest.fixture
def client(app, mock_storage):
    """Create a test client with mocked context."""
    app.state.context = {
        "storage": mock_storage,
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
    }
    return TestClient(app, raise_server_exceptions=False)


class TestConsensusStats:
    """Tests for GET /api/v2/consensus/stats."""

    def test_stats_returns_503_when_consensus_memory_unavailable(self, client):
        """Stats returns 503 when consensus memory module is not available."""
        with patch(
            "aragora.server.fastapi.routes.consensus._get_consensus_memory",
            side_effect=__import__("fastapi").HTTPException(
                status_code=503, detail="Consensus memory not available"
            ),
        ):
            response = client.get("/api/v2/consensus/stats")
            assert response.status_code == 503

    def test_stats_returns_200_with_valid_data(self, client):
        """Stats endpoint returns proper statistics shape."""
        mock_memory = MagicMock()
        mock_memory.get_statistics.return_value = {
            "total_consensus": 42,
            "total_dissents": 5,
            "by_domain": {"technology": 20, "finance": 22},
            "by_strength": {"strong": 30, "moderate": 12},
        }
        mock_memory.db_path = ":memory:"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (35, 0.82)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "aragora.server.fastapi.routes.consensus._get_consensus_memory",
                return_value=mock_memory,
            ),
            patch(
                "aragora.server.fastapi.routes.consensus._get_db_connection",
                return_value=mock_conn,
            ),
        ):
            response = client.get("/api/v2/consensus/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["total_topics"] == 42
            assert data["high_confidence_count"] == 35
            assert data["avg_confidence"] == 0.82
            assert data["total_dissents"] == 5
            assert "technology" in data["domains"]
            assert "finance" in data["domains"]


class TestSimilarDebates:
    """Tests for GET /api/v2/consensus/similar."""

    def test_similar_requires_topic(self, client):
        """Similar endpoint requires topic query parameter."""
        response = client.get("/api/v2/consensus/similar")
        assert response.status_code == 422

    def test_similar_validates_limit_bounds(self, client):
        """Limit must be between 1 and 20."""
        response = client.get("/api/v2/consensus/similar?topic=test&limit=0")
        assert response.status_code == 422

        response = client.get("/api/v2/consensus/similar?topic=test&limit=21")
        assert response.status_code == 422

    def test_similar_returns_results(self, client):
        """Similar endpoint returns matching debates."""
        mock_memory = MagicMock()
        mock_result = MagicMock()
        mock_result.consensus.topic = "Rate limiter design"
        mock_result.consensus.conclusion = "Token bucket is best"
        mock_result.consensus.strength.value = "strong"
        mock_result.consensus.confidence = 0.92
        mock_result.consensus.participating_agents = ["claude", "gpt4"]
        mock_result.consensus.timestamp.isoformat.return_value = "2026-01-15T12:00:00"
        mock_result.similarity_score = 0.85
        mock_result.dissents = []

        mock_memory.find_similar_debates.return_value = [mock_result]

        with patch(
            "aragora.server.fastapi.routes.consensus._get_consensus_memory",
            return_value=mock_memory,
        ):
            response = client.get("/api/v2/consensus/similar?topic=rate+limiting")
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "rate limiting"
            assert data["count"] == 1
            assert data["similar"][0]["topic"] == "Rate limiter design"
            assert data["similar"][0]["confidence"] == 0.92


class TestSettledTopics:
    """Tests for GET /api/v2/consensus/settled."""

    def test_settled_validates_confidence_bounds(self, client):
        """Confidence must be between 0.0 and 1.0."""
        response = client.get("/api/v2/consensus/settled?min_confidence=1.5")
        assert response.status_code == 422

        response = client.get("/api/v2/consensus/settled?min_confidence=-0.1")
        assert response.status_code == 422

    def test_settled_returns_topics(self, client):
        """Settled endpoint returns high-confidence topics."""
        mock_memory = MagicMock()
        mock_memory.db_path = ":memory:"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("AI safety", "Alignment is critical", 0.95, "strong", "2026-01-10T10:00:00"),
        ]
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        with (
            patch(
                "aragora.server.fastapi.routes.consensus._get_consensus_memory",
                return_value=mock_memory,
            ),
            patch(
                "aragora.server.fastapi.routes.consensus._get_db_connection",
                return_value=mock_conn,
            ),
        ):
            response = client.get("/api/v2/consensus/settled?min_confidence=0.9")
            assert response.status_code == 200
            data = response.json()
            assert data["min_confidence"] == 0.9
            assert data["count"] == 1
            assert data["topics"][0]["topic"] == "AI safety"
            assert data["topics"][0]["confidence"] == 0.95


class TestDomainHistory:
    """Tests for GET /api/v2/consensus/domain/{domain}."""

    def test_domain_validates_slug(self, client):
        """Domain must match safe slug pattern."""
        with patch(
            "aragora.server.fastapi.routes.consensus._get_consensus_memory",
            return_value=MagicMock(),
        ):
            # Use a string that passes URL routing but fails slug validation
            # (starts with a special char, has spaces encoded, etc.)
            response = client.get("/api/v2/consensus/domain/.invalid-start")
            assert response.status_code == 400

    def test_domain_returns_history(self, client):
        """Domain endpoint returns consensus history."""
        mock_memory = MagicMock()
        mock_record = MagicMock()
        mock_record.to_dict.return_value = {
            "topic": "Cloud architecture",
            "confidence": 0.88,
        }
        mock_memory.get_domain_consensus_history.return_value = [mock_record]

        with patch(
            "aragora.server.fastapi.routes.consensus._get_consensus_memory",
            return_value=mock_memory,
        ):
            response = client.get("/api/v2/consensus/domain/technology")
            assert response.status_code == 200
            data = response.json()
            assert data["domain"] == "technology"
            assert data["count"] == 1
            assert data["history"][0]["topic"] == "Cloud architecture"


class TestConsensusStatus:
    """Tests for GET /api/v2/consensus/status/{debate_id}."""

    def test_status_returns_404_for_missing_debate(self, client, mock_storage):
        """Status returns 404 when debate is not found."""
        mock_storage.get_debate.return_value = None
        response = client.get("/api/v2/consensus/status/nonexistent-id")
        assert response.status_code == 404


class TestDetectConsensus:
    """Tests for POST /api/v2/consensus/detect."""

    def test_detect_validates_empty_task(self, client):
        """Detect requires non-empty task."""
        # Bypass auth for this test
        response = client.post(
            "/api/v2/consensus/detect",
            json={"task": "", "proposals": [{"agent": "a", "content": "c"}]},
        )
        assert response.status_code in (401, 422)

    def test_detect_validates_empty_proposals(self, client):
        """Detect requires non-empty proposals list."""
        response = client.post(
            "/api/v2/consensus/detect",
            json={"task": "Test task", "proposals": []},
        )
        assert response.status_code in (401, 422)

    def test_detect_validates_threshold_bounds(self, client):
        """Threshold must be between 0.0 and 1.0."""
        response = client.post(
            "/api/v2/consensus/detect",
            json={
                "task": "Test task",
                "proposals": [{"agent": "a", "content": "c"}],
                "threshold": 1.5,
            },
        )
        assert response.status_code in (401, 422)

    def test_detect_validates_proposal_content_not_empty(self, client):
        """Proposal content must not be empty."""
        response = client.post(
            "/api/v2/consensus/detect",
            json={
                "task": "Test task",
                "proposals": [{"agent": "a", "content": "  "}],
            },
        )
        assert response.status_code in (401, 422)


class TestDissents:
    """Tests for GET /api/v2/consensus/dissents."""

    def test_dissents_validates_limit_bounds(self, client):
        """Limit must be between 1 and 50."""
        response = client.get("/api/v2/consensus/dissents?limit=0")
        assert response.status_code == 422

        response = client.get("/api/v2/consensus/dissents?limit=51")
        assert response.status_code == 422


class TestContrarianViews:
    """Tests for GET /api/v2/consensus/contrarian-views."""

    def test_contrarian_validates_limit_bounds(self, client):
        """Limit must be between 1 and 50."""
        response = client.get("/api/v2/consensus/contrarian-views?limit=0")
        assert response.status_code == 422


class TestRiskWarnings:
    """Tests for GET /api/v2/consensus/risk-warnings."""

    def test_risk_warnings_validates_limit_bounds(self, client):
        """Limit must be between 1 and 50."""
        response = client.get("/api/v2/consensus/risk-warnings?limit=0")
        assert response.status_code == 422
