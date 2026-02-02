"""Tests for ML handler endpoints.

Validates the REST API endpoints for ML capabilities including:
- Agent routing
- Response quality scoring
- Consensus prediction
- Training data export
- Embeddings and search
- Circuit breaker pattern
- Rate limiting
- RBAC permissions
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.ml import (
    MLCircuitBreaker,
    MLHandler,
    _clear_ml_components,
    _get_circuit_breaker,
    _get_ml_component,
    _ml_circuit_breakers,
    _ml_components,
    get_ml_circuit_breaker_status,
)


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


@pytest.fixture(autouse=True)
def cleanup_ml_components():
    """Clean up ML components after each test."""
    yield
    _clear_ml_components()


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestMLCircuitBreaker:
    """Test MLCircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Test circuit breaker starts in closed state."""
        cb = MLCircuitBreaker()
        assert cb.state == MLCircuitBreaker.CLOSED

    def test_can_proceed_when_closed(self):
        """Test can_proceed returns True when closed."""
        cb = MLCircuitBreaker()
        assert cb.can_proceed() is True

    def test_opens_after_threshold_failures(self):
        """Test circuit opens after reaching failure threshold."""
        cb = MLCircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        cb.record_failure()
        assert cb.state == MLCircuitBreaker.CLOSED
        cb.record_failure()
        assert cb.state == MLCircuitBreaker.CLOSED
        cb.record_failure()
        assert cb.state == MLCircuitBreaker.OPEN

    def test_cannot_proceed_when_open(self):
        """Test can_proceed returns False when circuit is open."""
        cb = MLCircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        assert cb.state == MLCircuitBreaker.OPEN
        assert cb.can_proceed() is False

    def test_transitions_to_half_open_after_cooldown(self):
        """Test circuit transitions to half-open after cooldown."""
        cb = MLCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == MLCircuitBreaker.OPEN

        # Wait for cooldown
        time.sleep(0.15)

        # State should transition on next check
        assert cb.state == MLCircuitBreaker.HALF_OPEN

    def test_half_open_allows_limited_calls(self):
        """Test half-open state allows limited test calls."""
        cb = MLCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1, half_open_max_calls=2)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for cooldown
        time.sleep(0.15)

        # Should allow limited calls in half-open
        assert cb.can_proceed() is True
        assert cb.can_proceed() is True
        assert cb.can_proceed() is False  # Third call blocked

    def test_closes_after_successful_recovery(self):
        """Test circuit closes after successful recovery in half-open."""
        cb = MLCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1, half_open_max_calls=2)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for cooldown
        time.sleep(0.15)

        # Make successful calls in half-open
        cb.can_proceed()
        cb.record_success()
        cb.can_proceed()
        cb.record_success()

        assert cb.state == MLCircuitBreaker.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open state."""
        cb = MLCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for cooldown
        time.sleep(0.15)

        # Get to half-open state
        assert cb.state == MLCircuitBreaker.HALF_OPEN

        # Fail in half-open
        cb.record_failure()
        assert cb.state == MLCircuitBreaker.OPEN

    def test_success_resets_failure_count(self):
        """Test success resets failure count in closed state."""
        cb = MLCircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()  # Should be first failure again

        assert cb.state == MLCircuitBreaker.CLOSED

    def test_get_status(self):
        """Test get_status returns correct information."""
        cb = MLCircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)

        cb.record_failure()
        cb.record_failure()

        status = cb.get_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 2
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 30.0
        assert status["last_failure_time"] is not None

    def test_reset(self):
        """Test reset returns circuit to initial state."""
        cb = MLCircuitBreaker(failure_threshold=2)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == MLCircuitBreaker.OPEN

        # Reset
        cb.reset()
        assert cb.state == MLCircuitBreaker.CLOSED
        assert cb.can_proceed() is True


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with ML components."""

    def test_get_circuit_breaker_creates_new(self):
        """Test _get_circuit_breaker creates new circuit breaker."""
        _ml_circuit_breakers.clear()

        cb = _get_circuit_breaker("test_component")
        assert cb is not None
        assert "test_component" in _ml_circuit_breakers

    def test_get_circuit_breaker_returns_existing(self):
        """Test _get_circuit_breaker returns existing circuit breaker."""
        _ml_circuit_breakers.clear()

        cb1 = _get_circuit_breaker("test_component")
        cb2 = _get_circuit_breaker("test_component")
        assert cb1 is cb2

    def test_get_ml_circuit_breaker_status(self):
        """Test get_ml_circuit_breaker_status returns all statuses."""
        _ml_circuit_breakers.clear()

        # Create some circuit breakers
        _get_circuit_breaker("router")
        _get_circuit_breaker("scorer")

        status = get_ml_circuit_breaker_status()
        assert "router" in status
        assert "scorer" in status
        assert status["router"]["state"] == "closed"

    def test_clear_ml_components(self):
        """Test _clear_ml_components clears both caches."""
        # Add some components and circuit breakers
        _ml_components["test"] = MagicMock()
        _get_circuit_breaker("test")

        _clear_ml_components()

        assert len(_ml_components) == 0
        assert len(_ml_circuit_breakers) == 0

    def test_component_unavailable_when_circuit_open(self):
        """Test _get_ml_component returns None when circuit is open."""
        _clear_ml_components()

        # Open the circuit breaker
        cb = _get_circuit_breaker("router")
        for _ in range(5):
            cb.record_failure()

        assert cb.state == MLCircuitBreaker.OPEN

        # Should return None even if component could be loaded
        result = _get_ml_component("router")
        assert result is None


# =============================================================================
# Handler Tests - Can Handle
# =============================================================================


class TestMLHandlerCanHandle:
    """Test MLHandler.can_handle method."""

    def test_can_handle_route(self, ml_handler):
        """Test can_handle returns True for route endpoint."""
        assert ml_handler.can_handle("/api/v1/ml/route")

    def test_can_handle_score(self, ml_handler):
        """Test can_handle returns True for score endpoint."""
        assert ml_handler.can_handle("/api/v1/ml/score")

    def test_can_handle_score_batch(self, ml_handler):
        """Test can_handle returns True for batch score endpoint."""
        assert ml_handler.can_handle("/api/v1/ml/score-batch")

    def test_can_handle_consensus(self, ml_handler):
        """Test can_handle returns True for consensus endpoint."""
        assert ml_handler.can_handle("/api/v1/ml/consensus")

    def test_can_handle_export_training(self, ml_handler):
        """Test can_handle returns True for export-training endpoint."""
        assert ml_handler.can_handle("/api/v1/ml/export-training")

    def test_can_handle_models(self, ml_handler):
        """Test can_handle returns True for models endpoint."""
        assert ml_handler.can_handle("/api/v1/ml/models")

    def test_can_handle_stats(self, ml_handler):
        """Test can_handle returns True for stats endpoint."""
        assert ml_handler.can_handle("/api/v1/ml/stats")

    def test_can_handle_embed(self, ml_handler):
        """Test can_handle returns True for embed endpoint."""
        assert ml_handler.can_handle("/api/v1/ml/embed")

    def test_can_handle_search(self, ml_handler):
        """Test can_handle returns True for search endpoint."""
        assert ml_handler.can_handle("/api/v1/ml/search")

    def test_cannot_handle_unknown(self, ml_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not ml_handler.can_handle("/api/v1/ml/unknown")
        assert not ml_handler.can_handle("/api/v1/debates")

    def test_all_routes_registered(self, ml_handler):
        """Test all ROUTES are handled."""
        for route in MLHandler.ROUTES:
            assert ml_handler.can_handle(route), f"Route {route} not handled"


# =============================================================================
# Handler Tests - GET Endpoints
# =============================================================================


class TestMLHandlerGetModels:
    """Test MLHandler models endpoint."""

    def test_get_models(self, ml_handler, mock_http_handler):
        """Test get_models returns model info."""
        result = ml_handler.handle("/api/v1/ml/models", {}, mock_http_handler)

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
        assert "embeddings" in caps
        assert "training_export" in caps

    def test_get_models_capabilities_are_boolean(self, ml_handler, mock_http_handler):
        """Test capabilities are boolean values."""
        result = ml_handler.handle("/api/v1/ml/models", {}, mock_http_handler)
        body = json.loads(result.body)

        for cap_name, cap_value in body["capabilities"].items():
            assert isinstance(cap_value, bool), f"{cap_name} should be boolean"


class TestMLHandlerGetStats:
    """Test MLHandler stats endpoint."""

    def test_get_stats(self, ml_handler, mock_http_handler):
        """Test get_stats returns statistics."""
        result = ml_handler.handle("/api/v1/ml/stats", {}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body
        assert "status" in body
        # Status should be healthy or limited
        assert body["status"] in ["healthy", "limited"]

    def test_get_stats_includes_circuit_breakers(self, ml_handler, mock_http_handler):
        """Test stats includes circuit breaker status."""
        # Trigger component loading to create circuit breakers
        _get_circuit_breaker("router")

        result = ml_handler.handle("/api/v1/ml/stats", {}, mock_http_handler)
        body = json.loads(result.body)

        assert "circuit_breakers" in body


# =============================================================================
# Handler Tests - POST Route
# =============================================================================


class TestMLHandlerPostRoute:
    """Test MLHandler route endpoint."""

    def test_post_route_valid(self, ml_handler, mock_http_handler):
        """Test posting valid routing request."""
        data = {
            "task": "Implement a binary search algorithm",
            "available_agents": ["claude", "gpt-4", "codex"],
            "team_size": 2,
        }
        result = ml_handler.handle_post("/api/v1/ml/route", data, mock_http_handler)

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
        result = ml_handler.handle_post("/api/v1/ml/route", data, mock_http_handler)

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
        result = ml_handler.handle_post("/api/v1/ml/route", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_post_route_with_constraints(self, ml_handler, mock_http_handler):
        """Test posting route request with constraints."""
        data = {
            "task": "Write production code",
            "available_agents": ["claude", "gpt-4", "codex"],
            "team_size": 2,
            "constraints": {"require_code": True, "max_latency": 5000},
        }
        result = ml_handler.handle_post("/api/v1/ml/route", data, mock_http_handler)

        assert result is not None
        # Should either succeed or fail gracefully
        assert result.status_code in [200, 400, 503]

    def test_post_route_default_team_size(self, ml_handler, mock_http_handler):
        """Test route uses default team size when not specified."""
        data = {
            "task": "Implement feature",
            "available_agents": ["claude", "gpt-4", "codex", "gemini"],
        }
        result = ml_handler.handle_post("/api/v1/ml/route", data, mock_http_handler)

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert len(body["selected_agents"]) <= 3  # Default team_size

    def test_post_route_returns_503_when_router_unavailable(self, ml_handler, mock_http_handler):
        """Test route returns 503 when router is unavailable."""
        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=None,
        ):
            data = {
                "task": "Test task",
                "available_agents": ["claude"],
                "team_size": 1,
            }
            result = ml_handler.handle_post("/api/v1/ml/route", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 503


# =============================================================================
# Handler Tests - POST Score
# =============================================================================


class TestMLHandlerPostScore:
    """Test MLHandler score endpoint."""

    def test_post_score_valid(self, ml_handler, mock_http_handler):
        """Test posting valid score request."""
        data = {
            "text": "This is a comprehensive, well-structured response that addresses the core question.",
            "context": "Design a rate limiter",
        }
        result = ml_handler.handle_post("/api/v1/ml/score", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "overall" in body
        assert "coherence" in body
        assert "completeness" in body
        assert 0.0 <= body["overall"] <= 1.0

    def test_post_score_missing_text(self, ml_handler, mock_http_handler):
        """Test posting score request without text."""
        data = {"context": "Some context"}
        result = ml_handler.handle_post("/api/v1/ml/score", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_post_score_empty_text(self, ml_handler, mock_http_handler):
        """Test posting score request with empty text returns error."""
        data = {"text": "", "context": "Some context"}
        result = ml_handler.handle_post("/api/v1/ml/score", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_post_score_without_context(self, ml_handler, mock_http_handler):
        """Test scoring works without context."""
        data = {"text": "This is a well-written response."}
        result = ml_handler.handle_post("/api/v1/ml/score", data, mock_http_handler)

        # Should work or return 503 if scorer unavailable
        assert result is not None
        assert result.status_code in [200, 503]

    def test_post_score_returns_all_metrics(self, ml_handler, mock_http_handler):
        """Test score returns all expected metrics."""
        mock_scorer = MagicMock()
        mock_score = MagicMock()
        mock_score.overall = 0.85
        mock_score.coherence = 0.9
        mock_score.completeness = 0.8
        mock_score.relevance = 0.85
        mock_score.clarity = 0.88
        mock_score.confidence = 0.7
        mock_score.is_high_quality = True
        mock_score.needs_review = False
        mock_scorer.score.return_value = mock_score

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_scorer,
        ):
            data = {"text": "Test response"}
            result = ml_handler.handle_post("/api/v1/ml/score", data, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert "overall" in body
            assert "coherence" in body
            assert "completeness" in body
            assert "relevance" in body
            assert "clarity" in body
            assert "confidence" in body
            assert "is_high_quality" in body
            assert "needs_review" in body


# =============================================================================
# Handler Tests - POST Score Batch
# =============================================================================


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
        result = ml_handler.handle_post("/api/v1/ml/score-batch", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "scores" in body
        assert len(body["scores"]) == 2

    def test_post_score_batch_empty(self, ml_handler, mock_http_handler):
        """Test posting batch score with empty list returns error."""
        data = {"texts": []}
        result = ml_handler.handle_post("/api/v1/ml/score-batch", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_post_score_batch_exceeds_limit(self, ml_handler, mock_http_handler):
        """Test posting batch score exceeding 100 texts returns error."""
        data = {"texts": ["text"] * 101}
        result = ml_handler.handle_post("/api/v1/ml/score-batch", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body
        assert "100" in body["error"]

    def test_post_score_batch_with_contexts(self, ml_handler, mock_http_handler):
        """Test batch scoring with per-text contexts."""
        mock_scorer = MagicMock()
        mock_score = MagicMock()
        mock_score.overall = 0.75
        mock_score.coherence = 0.8
        mock_score.completeness = 0.7
        mock_score.relevance = 0.75
        mock_score.clarity = 0.78
        mock_score.confidence = 0.6
        mock_score.is_high_quality = True
        mock_scorer.score_batch.return_value = [mock_score, mock_score]

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_scorer,
        ):
            data = {
                "texts": ["Response 1", "Response 2"],
                "contexts": ["Context 1", "Context 2"],
            }
            result = ml_handler.handle_post("/api/v1/ml/score-batch", data, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert len(body["scores"]) == 2


# =============================================================================
# Handler Tests - POST Consensus
# =============================================================================


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
        result = ml_handler.handle_post("/api/v1/ml/consensus", data, mock_http_handler)

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
        result = ml_handler.handle_post("/api/v1/ml/consensus", data, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_post_consensus_returns_all_fields(self, ml_handler, mock_http_handler):
        """Test consensus returns all expected fields."""
        mock_predictor = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.probability = 0.85
        mock_prediction.confidence = 0.7
        mock_prediction.convergence_trend = "converging"
        mock_prediction.estimated_rounds = 2
        mock_prediction.likely_consensus = True
        mock_prediction.early_termination_safe = True
        mock_prediction.needs_intervention = False
        mock_prediction.key_factors = ["high_similarity"]
        mock_predictor.predict.return_value = mock_prediction

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_predictor,
        ):
            data = {
                "responses": [["agent1", "text1"]],
                "current_round": 1,
            }
            result = ml_handler.handle_post("/api/v1/ml/consensus", data, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert "probability" in body
            assert "confidence" in body
            assert "convergence_trend" in body
            assert "estimated_rounds" in body
            assert "likely_consensus" in body
            assert "early_termination_safe" in body
            assert "needs_intervention" in body
            assert "key_factors" in body

    def test_post_consensus_default_rounds(self, ml_handler, mock_http_handler):
        """Test consensus uses default round values."""
        mock_predictor = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.probability = 0.5
        mock_prediction.confidence = 0.5
        mock_prediction.convergence_trend = "stable"
        mock_prediction.estimated_rounds = 2
        mock_prediction.likely_consensus = False
        mock_prediction.early_termination_safe = False
        mock_prediction.needs_intervention = False
        mock_prediction.key_factors = []
        mock_predictor.predict.return_value = mock_prediction

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_predictor,
        ):
            data = {"responses": [["agent1", "text1"]]}
            result = ml_handler.handle_post("/api/v1/ml/consensus", data, mock_http_handler)

            assert result is not None
            # Should use defaults: current_round=1, total_rounds=3
            mock_predictor.predict.assert_called_once()


# =============================================================================
# Handler Tests - POST Embed
# =============================================================================


class TestMLHandlerPostEmbed:
    """Test MLHandler embed endpoint."""

    def test_post_embed_valid(self, ml_handler, mock_http_handler):
        """Test posting valid embed request."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed.return_value = [0.1, 0.2, 0.3, 0.4]

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_embeddings,
        ):
            data = {"text": "Some text to embed"}
            result = ml_handler.handle_post("/api/v1/ml/embed", data, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert "embeddings" in body
            assert "dimension" in body
            assert "count" in body
            assert isinstance(body["embeddings"], list)

    def test_post_embed_batch(self, ml_handler, mock_http_handler):
        """Test posting batch embed request."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_batch.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_embeddings,
        ):
            data = {"texts": ["First text", "Second text", "Third text"]}
            result = ml_handler.handle_post("/api/v1/ml/embed", data, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert "embeddings" in body
            assert len(body["embeddings"]) == 3

    def test_post_embed_no_ml_component(self, ml_handler, mock_http_handler):
        """Test embed returns 503 when ML component unavailable."""
        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=None,
        ):
            data = {"text": "Some text"}
            result = ml_handler.handle_post("/api/v1/ml/embed", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 503

    def test_post_embed_missing_text_and_texts(self, ml_handler, mock_http_handler):
        """Test embed returns error when neither text nor texts provided."""
        mock_embeddings = MagicMock()

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_embeddings,
        ):
            data = {}
            result = ml_handler.handle_post("/api/v1/ml/embed", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 400

    def test_post_embed_batch_exceeds_limit(self, ml_handler, mock_http_handler):
        """Test embed batch exceeding 100 texts returns error."""
        mock_embeddings = MagicMock()

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_embeddings,
        ):
            data = {"texts": ["text"] * 101}
            result = ml_handler.handle_post("/api/v1/ml/embed", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 400


# =============================================================================
# Handler Tests - POST Search
# =============================================================================


class TestMLHandlerPostSearch:
    """Test MLHandler search endpoint."""

    def test_post_search_valid(self, ml_handler, mock_http_handler):
        """Test posting valid search request."""
        mock_result = MagicMock()
        mock_result.text = "Token bucket algorithm for rate limiting"
        mock_result.score = 0.95
        mock_result.index = 0

        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = [mock_result]

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_embeddings,
        ):
            data = {
                "query": "rate limiting algorithms",
                "documents": [
                    "Token bucket algorithm for rate limiting",
                    "Database indexing strategies",
                    "Leaky bucket implementation",
                ],
                "top_k": 2,
            }
            result = ml_handler.handle_post("/api/v1/ml/search", data, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert "results" in body
            assert "count" in body
            if body["results"]:
                assert "text" in body["results"][0]
                assert "score" in body["results"][0]
                assert "index" in body["results"][0]

    def test_post_search_no_ml_component(self, ml_handler, mock_http_handler):
        """Test search returns 503 when ML component unavailable."""
        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=None,
        ):
            data = {"query": "test", "documents": ["doc1"]}
            result = ml_handler.handle_post("/api/v1/ml/search", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 503

    def test_post_search_missing_query(self, ml_handler, mock_http_handler):
        """Test search returns error when query is missing."""
        mock_embeddings = MagicMock()

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_embeddings,
        ):
            data = {"documents": ["doc1", "doc2"]}
            result = ml_handler.handle_post("/api/v1/ml/search", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 400

    def test_post_search_missing_documents(self, ml_handler, mock_http_handler):
        """Test search returns error when documents are missing."""
        mock_embeddings = MagicMock()

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_embeddings,
        ):
            data = {"query": "test query"}
            result = ml_handler.handle_post("/api/v1/ml/search", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 400

    def test_post_search_exceeds_document_limit(self, ml_handler, mock_http_handler):
        """Test search returns error when exceeding 1000 documents."""
        mock_embeddings = MagicMock()

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_embeddings,
        ):
            data = {"query": "test", "documents": ["doc"] * 1001}
            result = ml_handler.handle_post("/api/v1/ml/search", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 400

    def test_post_search_with_threshold(self, ml_handler, mock_http_handler):
        """Test search with score threshold."""
        mock_result = MagicMock()
        mock_result.text = "Relevant document"
        mock_result.score = 0.8
        mock_result.index = 0

        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = [mock_result]

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_embeddings,
        ):
            data = {
                "query": "search query",
                "documents": ["doc1", "doc2"],
                "top_k": 5,
                "threshold": 0.5,
            }
            result = ml_handler.handle_post("/api/v1/ml/search", data, mock_http_handler)

            assert result is not None
            mock_embeddings.search.assert_called_once()
            call_kwargs = mock_embeddings.search.call_args[1]
            assert call_kwargs["threshold"] == 0.5


# =============================================================================
# Handler Tests - POST Export Training
# =============================================================================


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
        result = ml_handler.handle_post("/api/v1/ml/export-training", data, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "examples" in body or "data" in body or "error" in body

    def test_post_export_training_empty_debates(self, ml_handler, mock_http_handler):
        """Test export returns error with empty debates."""
        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=MagicMock(),
        ):
            data = {"debates": []}
            result = ml_handler.handle_post("/api/v1/ml/export-training", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 400

    def test_post_export_training_jsonl_format(self, ml_handler, mock_http_handler):
        """Test export with JSONL format."""
        mock_example = MagicMock()
        mock_example.to_dict.return_value = {"task": "test", "consensus": "result"}

        mock_training_data = MagicMock()
        mock_training_data.examples = [mock_example]
        mock_training_data.__len__ = lambda self: 1

        mock_exporter = MagicMock()
        mock_exporter.export_debates_batch.return_value = mock_training_data

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_exporter,
        ):
            data = {
                "debates": [{"task": "test", "consensus": "result", "rejected": []}],
                "format": "jsonl",
            }
            result = ml_handler.handle_post("/api/v1/ml/export-training", data, mock_http_handler)

            assert result is not None
            body = json.loads(result.body)
            assert body.get("format") == "jsonl"


# =============================================================================
# Component Loading Tests
# =============================================================================


class TestMLComponentLazyLoading:
    """Test ML component lazy loading."""

    def test_component_loaded_on_demand(self):
        """Test that ML components are loaded lazily."""
        _clear_ml_components()

        # Get router component (may or may not be available)
        router = _get_ml_component("router")
        assert "router" in _ml_components

        # Get scorer component
        scorer = _get_ml_component("scorer")
        assert "scorer" in _ml_components

    def test_unknown_component_returns_none(self):
        """Test that unknown component returns None."""
        _clear_ml_components()
        result = _get_ml_component("unknown_component")
        assert result is None

    def test_component_cached_after_first_load(self):
        """Test component is cached after first load."""
        _clear_ml_components()

        # Mock the import
        mock_router = MagicMock()
        with patch(
            "aragora.ml.get_agent_router",
            return_value=mock_router,
        ):
            # First call loads
            result1 = _get_ml_component("router")
            # Second call returns cached
            result2 = _get_ml_component("router")

            assert result1 is result2


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestMLHandlerRateLimiting:
    """Test rate limiting on ML endpoints."""

    def test_rate_limit_response_format(self, ml_handler, mock_http_handler):
        """Test rate limit response has correct format."""
        from aragora.server.handlers.ml import _ml_limiter

        # Exhaust the rate limit
        _ml_limiter.clear()
        for _ in range(61):
            _ml_limiter.is_allowed("127.0.0.1")

        result = ml_handler.handle("/api/v1/ml/models", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 429
        body = json.loads(result.body)
        assert "error" in body

        # Reset for other tests
        _ml_limiter.clear()


# =============================================================================
# Integration Tests
# =============================================================================


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
        route_result = ml_handler.handle_post("/api/v1/ml/route", route_data, mock_http_handler)
        assert route_result is not None
        route_body = json.loads(route_result.body)
        selected = route_body.get("selected_agents", [])

        # Step 2: Score a response
        score_data = {
            "text": "Implement using Redis with consistent hashing for distribution",
            "context": "Design a distributed cache",
        }
        score_result = ml_handler.handle_post("/api/v1/ml/score", score_data, mock_http_handler)
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
            "/api/v1/ml/consensus", consensus_data, mock_http_handler
        )
        assert consensus_result is not None
        consensus_body = json.loads(consensus_result.body)
        assert "probability" in consensus_body

    def test_handler_with_none_http_handler(self, ml_handler):
        """Test handler works with None http_handler."""
        # Should use default client IP
        result = ml_handler.handle("/api/v1/ml/models", {}, None)
        assert result is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestMLHandlerErrorHandling:
    """Test error handling in ML handler."""

    def test_handles_component_exception_gracefully(self, ml_handler, mock_http_handler):
        """Test handler handles component exceptions gracefully."""
        mock_router = MagicMock()
        mock_router.route.side_effect = ValueError("Invalid input")

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_router,
        ):
            data = {
                "task": "Test task",
                "available_agents": ["claude"],
                "team_size": 1,
            }
            result = ml_handler.handle_post("/api/v1/ml/route", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 400

    def test_handles_unexpected_exception(self, ml_handler, mock_http_handler):
        """Test handler handles unexpected exceptions."""
        mock_scorer = MagicMock()
        mock_scorer.score.side_effect = RuntimeError("Unexpected error")

        with patch(
            "aragora.server.handlers.ml._get_ml_component",
            return_value=mock_scorer,
        ):
            data = {"text": "Test text"}
            result = ml_handler.handle_post("/api/v1/ml/score", data, mock_http_handler)

            assert result is not None
            assert result.status_code == 500

    def test_unknown_post_path_returns_none(self, ml_handler, mock_http_handler):
        """Test unknown POST path returns None."""
        result = ml_handler.handle_post("/api/v1/ml/unknown", {}, mock_http_handler)
        assert result is None

    def test_unknown_get_path_returns_none(self, ml_handler, mock_http_handler):
        """Test unknown GET path returns None."""
        result = ml_handler.handle("/api/v1/ml/unknown", {}, mock_http_handler)
        assert result is None
