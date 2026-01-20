"""Tests for ML handler endpoints.

Validates the REST API endpoints for ML capabilities including:
- Agent routing
- Response quality scoring
- Consensus prediction
- Training data export
- Embeddings and search
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.ml import MLHandler


@pytest.fixture
def ml_handler():
    """Create an ML handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = MLHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    return handler


class TestMLHandlerCanHandle:
    """Test MLHandler.can_handle method."""

    def test_can_handle_route(self, ml_handler):
        """Test can_handle returns True for route endpoint."""
        assert ml_handler.can_handle("/api/ml/route")

    def test_can_handle_score(self, ml_handler):
        """Test can_handle returns True for score endpoint."""
        assert ml_handler.can_handle("/api/ml/score")

    def test_can_handle_score_batch(self, ml_handler):
        """Test can_handle returns True for batch score endpoint."""
        assert ml_handler.can_handle("/api/ml/score-batch")

    def test_can_handle_consensus(self, ml_handler):
        """Test can_handle returns True for consensus endpoint."""
        assert ml_handler.can_handle("/api/ml/consensus")

    def test_can_handle_export_training(self, ml_handler):
        """Test can_handle returns True for export-training endpoint."""
        assert ml_handler.can_handle("/api/ml/export-training")

    def test_can_handle_models(self, ml_handler):
        """Test can_handle returns True for models endpoint."""
        assert ml_handler.can_handle("/api/ml/models")

    def test_can_handle_stats(self, ml_handler):
        """Test can_handle returns True for stats endpoint."""
        assert ml_handler.can_handle("/api/ml/stats")

    def test_can_handle_embed(self, ml_handler):
        """Test can_handle returns True for embed endpoint."""
        assert ml_handler.can_handle("/api/ml/embed")

    def test_can_handle_search(self, ml_handler):
        """Test can_handle returns True for search endpoint."""
        assert ml_handler.can_handle("/api/ml/search")

    def test_cannot_handle_unknown(self, ml_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not ml_handler.can_handle("/api/ml/unknown")
        assert not ml_handler.can_handle("/api/debates")


class TestMLHandlerGetModels:
    """Test MLHandler models endpoint."""

    def test_get_models(self, ml_handler, mock_http_handler):
        """Test get_models returns model info."""
        result = ml_handler.handle("/api/ml/models", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "capabilities" in body
        assert "models" in body
        assert "version" in body
        # Check capabilities are booleans
        caps = body["capabilities"]
        assert "routing" in caps
        assert "scoring" in caps
        assert "consensus" in caps


class TestMLHandlerGetStats:
    """Test MLHandler stats endpoint."""

    def test_get_stats(self, ml_handler, mock_http_handler):
        """Test get_stats returns statistics."""
        result = ml_handler.handle("/api/ml/stats", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body
        assert "status" in body
        # Status should be healthy or limited
        assert body["status"] in ["healthy", "limited"]


class TestMLHandlerPostRoute:
    """Test MLHandler route endpoint."""

    def test_post_route_valid(self, ml_handler, mock_http_handler):
        """Test posting valid routing request."""
        data = {
            "task": "Implement a binary search algorithm",
            "available_agents": ["claude", "gpt-4", "codex"],
            "team_size": 2,
        }
        result = ml_handler.handle_post("/api/ml/route", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "selected_agents" in body
        assert "task_type" in body
        assert "confidence" in body
        assert len(body["selected_agents"]) <= 2

    def test_post_route_missing_task(self, ml_handler, mock_http_handler):
        """Test posting route request without task."""
        data = {
            "available_agents": ["claude", "gpt-4"],
            "team_size": 2,
        }
        result = ml_handler.handle_post("/api/ml/route", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_post_route_empty_agents(self, ml_handler, mock_http_handler):
        """Test posting route request with empty agents returns error."""
        data = {
            "task": "Test task",
            "available_agents": [],
            "team_size": 2,
        }
        result = ml_handler.handle_post("/api/ml/route", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body


class TestMLHandlerPostScore:
    """Test MLHandler score endpoint."""

    def test_post_score_valid(self, ml_handler, mock_http_handler):
        """Test posting valid score request."""
        data = {
            "text": "This is a comprehensive, well-structured response that addresses the core question.",
            "context": "Design a rate limiter",
        }
        result = ml_handler.handle_post("/api/ml/score", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "overall" in body
        assert "coherence" in body
        assert "completeness" in body
        assert 0.0 <= body["overall"] <= 1.0

    def test_post_score_missing_text(self, ml_handler, mock_http_handler):
        """Test posting score request without text."""
        data = {"context": "Some context"}
        result = ml_handler.handle_post("/api/ml/score", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_post_score_empty_text(self, ml_handler, mock_http_handler):
        """Test posting score request with empty text returns error."""
        data = {"text": "", "context": "Some context"}
        result = ml_handler.handle_post("/api/ml/score", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body


class TestMLHandlerPostScoreBatch:
    """Test MLHandler batch score endpoint."""

    def test_post_score_batch_valid(self, ml_handler, mock_http_handler):
        """Test posting valid batch score request."""
        data = {
            "texts": [
                "First response text here",
                "Second response with more detail and explanation",
            ],
            "context": "Task context",
        }
        result = ml_handler.handle_post("/api/ml/score-batch", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "scores" in body
        assert len(body["scores"]) == 2

    def test_post_score_batch_empty(self, ml_handler, mock_http_handler):
        """Test posting batch score with empty list returns error."""
        data = {"texts": []}
        result = ml_handler.handle_post("/api/ml/score-batch", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body


class TestMLHandlerPostConsensus:
    """Test MLHandler consensus endpoint."""

    def test_post_consensus_valid(self, ml_handler, mock_http_handler):
        """Test posting valid consensus request."""
        data = {
            "responses": [
                ["claude", "I recommend approach A for this problem"],
                ["gpt-4", "Approach A seems like the best solution"],
            ],
            "context": "System design question",
            "current_round": 2,
            "total_rounds": 3,
        }
        result = ml_handler.handle_post("/api/ml/consensus", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "probability" in body
        assert "confidence" in body
        assert "convergence_trend" in body
        assert 0.0 <= body["probability"] <= 1.0

    def test_post_consensus_empty_responses(self, ml_handler, mock_http_handler):
        """Test posting consensus request with empty responses returns error."""
        data = {
            "responses": [],
            "current_round": 1,
            "total_rounds": 3,
        }
        result = ml_handler.handle_post("/api/ml/consensus", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body


class TestMLHandlerPostEmbed:
    """Test MLHandler embed endpoint."""

    def test_post_embed_valid(self, ml_handler, mock_http_handler):
        """Test posting valid embed request."""
        data = {"text": "Some text to embed"}
        result = ml_handler.handle_post("/api/ml/embed", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "embeddings" in body
        assert "dimension" in body
        assert isinstance(body["embeddings"], list)

    def test_post_embed_batch(self, ml_handler, mock_http_handler):
        """Test posting batch embed request."""
        data = {
            "texts": ["First text", "Second text", "Third text"],
        }
        result = ml_handler.handle_post("/api/ml/embed", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "embeddings" in body
        assert len(body["embeddings"]) == 3


class TestMLHandlerPostSearch:
    """Test MLHandler search endpoint."""

    def test_post_search_valid(self, ml_handler, mock_http_handler):
        """Test posting valid search request."""
        data = {
            "query": "rate limiting algorithms",
            "documents": [
                "Token bucket algorithm for rate limiting",
                "Database indexing strategies",
                "Leaky bucket implementation",
            ],
            "top_k": 2,
        }
        result = ml_handler.handle_post("/api/ml/search", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "results" in body
        assert len(body["results"]) <= 2
        # Results should include text and score
        if body["results"]:
            assert "text" in body["results"][0]
            assert "score" in body["results"][0]


class TestMLHandlerPostExportTraining:
    """Test MLHandler export-training endpoint."""

    def test_post_export_training_valid(self, ml_handler, mock_http_handler):
        """Test posting valid export training request."""
        data = {
            "debates": [
                {
                    "task": "Design a caching layer",
                    "consensus": "Use Redis with LRU eviction...",
                    "rejected": [
                        "Just use a dictionary",
                        "Store everything in memory",
                    ],
                }
            ],
            "format": "json",
        }
        result = ml_handler.handle_post("/api/ml/export-training", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "examples" in body or "data" in body


class TestMLComponentLazyLoading:
    """Test ML component lazy loading."""

    def test_component_loaded_on_demand(self):
        """Test that ML components are loaded lazily."""
        from aragora.server.handlers.ml import _get_ml_component, _ml_components

        # Clear cache
        _ml_components.clear()

        # Get router component
        router = _get_ml_component("router")
        assert "router" in _ml_components

        # Get scorer component
        scorer = _get_ml_component("scorer")
        assert "scorer" in _ml_components

    def test_unknown_component_returns_none(self):
        """Test that unknown component returns None."""
        from aragora.server.handlers.ml import _get_ml_component

        result = _get_ml_component("unknown_component")
        assert result is None


class TestMLHandlerIntegration:
    """Integration tests for ML handler."""

    def test_full_workflow(self, ml_handler, mock_http_handler):
        """Test full ML workflow: route -> score -> consensus."""
        # Step 1: Route agents
        route_data = {
            "task": "Design a distributed cache",
            "available_agents": ["claude", "gpt-4", "codex", "gemini"],
            "team_size": 3,
        }
        route_result = ml_handler.handle_post("/api/ml/route", route_data, mock_http_handler)
        assert route_result is not None
        route_body = json.loads(route_result.body)
        selected = route_body.get("selected_agents", [])

        # Step 2: Score a response
        score_data = {
            "text": "Implement using Redis with consistent hashing for distribution",
            "context": "Design a distributed cache",
        }
        score_result = ml_handler.handle_post("/api/ml/score", score_data, mock_http_handler)
        assert score_result is not None
        score_body = json.loads(score_result.body)
        assert "overall" in score_body

        # Step 3: Check consensus
        consensus_data = {
            "responses": [
                ["claude", "Redis with consistent hashing is the best approach"],
                ["gpt-4", "I agree, Redis provides the reliability we need"],
            ],
            "context": "Design a distributed cache",
            "current_round": 2,
            "total_rounds": 3,
        }
        consensus_result = ml_handler.handle_post(
            "/api/ml/consensus", consensus_data, mock_http_handler
        )
        assert consensus_result is not None
        consensus_body = json.loads(consensus_result.body)
        assert "probability" in consensus_body
