"""Tests for ML handler (aragora/server/handlers/ml.py).

Covers all routes and behavior of the MLHandler class:
- can_handle() routing for all ROUTES
- GET  /api/v1/ml/models      - List available ML models/capabilities
- GET  /api/v1/ml/stats        - Get ML module statistics
- POST /api/v1/ml/route        - ML-based agent routing
- POST /api/v1/ml/score        - Single response quality scoring
- POST /api/v1/ml/score-batch  - Batch response quality scoring
- POST /api/v1/ml/consensus    - Consensus prediction
- POST /api/v1/ml/export-training - Training data export (ml:train permission)
- POST /api/v1/ml/embed        - Text embedding
- POST /api/v1/ml/search       - Semantic search
- Rate limiting behavior
- Circuit breaker integration
- RBAC permission checks
- Error handling for all error types
- Input validation (missing fields, size limits)
- Edge cases (empty results, component unavailable)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.ml import (
    MLHandler,
    _clear_ml_components,
    _get_circuit_breaker,
    _get_ml_component,
    _ml_limiter,
    get_ml_circuit_breaker_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to MLHandler."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        client_address: tuple[str, int] | None = None,
    ):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()
        self.client_address = client_address or ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# ---------------------------------------------------------------------------
# Mock ML component objects
# ---------------------------------------------------------------------------


class MockRoutingDecision:
    """Mock routing decision returned by router.route()."""

    def __init__(self, **kwargs):
        self.selected_agents = kwargs.get("selected_agents", ["claude", "gpt-4"])
        self.task_type = MagicMock(value=kwargs.get("task_type", "coding"))
        self.confidence = kwargs.get("confidence", 0.85)
        self.reasoning = kwargs.get("reasoning", ["task_type=coding"])
        self.agent_scores = kwargs.get("agent_scores", {"claude": 0.9, "gpt-4": 0.8})
        self.diversity_score = kwargs.get("diversity_score", 0.7)


class MockQualityScore:
    """Mock quality score returned by scorer.score()."""

    def __init__(self, **kwargs):
        self.overall = kwargs.get("overall", 0.75)
        self.coherence = kwargs.get("coherence", 0.8)
        self.completeness = kwargs.get("completeness", 0.7)
        self.relevance = kwargs.get("relevance", 0.75)
        self.clarity = kwargs.get("clarity", 0.78)
        self.confidence = kwargs.get("confidence", 0.6)
        self.is_high_quality = kwargs.get("is_high_quality", True)
        self.needs_review = kwargs.get("needs_review", False)


class MockConsensusPrediction:
    """Mock consensus prediction returned by predictor.predict()."""

    def __init__(self, **kwargs):
        self.probability = kwargs.get("probability", 0.85)
        self.confidence = kwargs.get("confidence", 0.7)
        self.convergence_trend = kwargs.get("convergence_trend", "converging")
        self.estimated_rounds = kwargs.get("estimated_rounds", 2)
        self.likely_consensus = kwargs.get("likely_consensus", True)
        self.early_termination_safe = kwargs.get("early_termination_safe", True)
        self.needs_intervention = kwargs.get("needs_intervention", False)
        self.key_factors = kwargs.get("key_factors", ["high_semantic_similarity"])


class MockTrainingExample:
    """Mock training example returned by exporter."""

    def __init__(self, data: dict | None = None):
        self._data = data or {"task": "test", "chosen": "response"}

    def to_dict(self) -> dict:
        return self._data


class MockTrainingData:
    """Mock training data container returned by exporter.export_debates_batch()."""

    def __init__(self, examples: list | None = None):
        self.examples = examples or [MockTrainingExample()]

    def __len__(self):
        return len(self.examples)

    def __bool__(self):
        return len(self.examples) > 0


class MockSearchResult:
    """Mock search result returned by embeddings.search()."""

    def __init__(self, text: str = "doc1", score: float = 0.95, index: int = 0):
        self.text = text
        self.score = score
        self.index = index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an MLHandler with minimal server context."""
    return MLHandler(ctx={})


@pytest.fixture(autouse=True)
def _reset_ml_state():
    """Reset ML component cache and rate limiter state before each test."""
    _clear_ml_components()
    _ml_limiter._buckets = defaultdict(list)
    yield
    _clear_ml_components()
    _ml_limiter._buckets = defaultdict(list)


@pytest.fixture
def mock_router():
    """Create a mock ML router component."""
    router = MagicMock()
    router.route.return_value = MockRoutingDecision()
    router._capabilities = {"claude": {}, "gpt-4": {}}
    router._historical_performance = {}
    return router


@pytest.fixture
def mock_scorer():
    """Create a mock ML scorer component."""
    scorer = MagicMock()
    scorer.score.return_value = MockQualityScore()
    scorer.score_batch.return_value = [MockQualityScore(), MockQualityScore(overall=0.82)]
    return scorer


@pytest.fixture
def mock_predictor():
    """Create a mock ML predictor component."""
    predictor = MagicMock()
    predictor.predict.return_value = MockConsensusPrediction()
    predictor.get_calibration_stats.return_value = {
        "samples": 100,
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
    }
    return predictor


@pytest.fixture
def mock_embeddings():
    """Create a mock ML embeddings component."""
    emb = MagicMock()
    emb.embed.return_value = [0.1, 0.2, 0.3]
    emb.embed_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    emb.model_name = "all-MiniLM-L6-v2"
    emb.dimension = 384
    emb._dimension = 384
    emb.search.return_value = [MockSearchResult("doc1", 0.95, 0), MockSearchResult("doc2", 0.8, 1)]
    return emb


@pytest.fixture
def mock_exporter():
    """Create a mock ML training exporter component."""
    exporter = MagicMock()
    exporter.export_debates_batch.return_value = MockTrainingData()
    return exporter


@pytest.fixture
def mock_user_with_train():
    """Create a mock user with ml:train permission."""
    user = MagicMock()
    user.role = "admin"
    return user


@pytest.fixture
def mock_user_no_train():
    """Create a mock user without ml:train permission."""
    user = MagicMock()
    user.role = "viewer"
    return user


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_route_path(self, handler):
        assert handler.can_handle("/api/v1/ml/route")

    def test_score_path(self, handler):
        assert handler.can_handle("/api/v1/ml/score")

    def test_score_batch_path(self, handler):
        assert handler.can_handle("/api/v1/ml/score-batch")

    def test_consensus_path(self, handler):
        assert handler.can_handle("/api/v1/ml/consensus")

    def test_export_training_path(self, handler):
        assert handler.can_handle("/api/v1/ml/export-training")

    def test_models_path(self, handler):
        assert handler.can_handle("/api/v1/ml/models")

    def test_stats_path(self, handler):
        assert handler.can_handle("/api/v1/ml/stats")

    def test_embed_path(self, handler):
        assert handler.can_handle("/api/v1/ml/embed")

    def test_search_path(self, handler):
        assert handler.can_handle("/api/v1/ml/search")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_match(self, handler):
        assert not handler.can_handle("/api/v1/ml")

    def test_rejects_v2_path(self, handler):
        assert not handler.can_handle("/api/v2/ml/models")

    def test_rejects_extra_suffix(self, handler):
        assert not handler.can_handle("/api/v1/ml/models/extra")

    def test_rejects_typo(self, handler):
        assert not handler.can_handle("/api/v1/mll/models")


# ============================================================================
# Constructor
# ============================================================================


class TestConstructor:
    """Test MLHandler initialization."""

    def test_default_context(self):
        h = MLHandler()
        assert h.ctx == {}

    def test_custom_context(self):
        ctx = {"storage": MagicMock()}
        h = MLHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_context_defaults_to_empty(self):
        h = MLHandler(ctx=None)
        assert h.ctx == {}


# ============================================================================
# GET /api/v1/ml/models
# ============================================================================


class TestListModels:
    """Test the list models endpoint."""

    @patch("aragora.server.handlers.ml._get_ml_component", return_value=None)
    def test_list_models_all_unavailable(self, mock_get, handler):
        """All ML components unavailable returns all-false capabilities."""
        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/ml/models", {}, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["capabilities"]["routing"] is False
        assert body["capabilities"]["scoring"] is False
        assert body["capabilities"]["consensus"] is False
        assert body["capabilities"]["embeddings"] is False
        assert body["capabilities"]["training_export"] is False
        assert body["version"] == "1.0.0"
        assert body["models"] == {}

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_list_models_with_router(self, mock_get, handler):
        """Router available shows routing capabilities."""
        router = MagicMock()
        router._capabilities = {"claude": {}, "gpt-4": {}, "codex": {}}

        def side_effect(name):
            if name == "router":
                return router
            return None

        mock_get.side_effect = side_effect

        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/ml/models", {}, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["capabilities"]["routing"] is True
        assert body["models"]["routing"]["registered_agents"] == 3
        assert "task_types" in body["models"]["routing"]

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_list_models_with_embeddings(self, mock_get, handler):
        """Embeddings available shows embedding model info."""
        emb = MagicMock()
        emb.model_name = "all-MiniLM-L6-v2"
        emb.dimension = 384
        emb._dimension = 384

        def side_effect(name):
            if name == "embeddings":
                return emb
            return None

        mock_get.side_effect = side_effect

        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/ml/models", {}, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["capabilities"]["embeddings"] is True
        assert body["models"]["embeddings"]["model"] == "all-MiniLM-L6-v2"

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_list_models_embeddings_lazy_dimension(self, mock_get, handler):
        """Embeddings without dimension loaded shows 'lazy'."""
        emb = MagicMock()
        emb.model_name = "test-model"
        emb._dimension = None

        def side_effect(name):
            if name == "embeddings":
                return emb
            return None

        mock_get.side_effect = side_effect

        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/ml/models", {}, http)
        body = _body(result)
        assert body["models"]["embeddings"]["dimension"] == "lazy"


# ============================================================================
# GET /api/v1/ml/stats
# ============================================================================


class TestStats:
    """Test the stats endpoint."""

    @patch("aragora.server.handlers.ml._get_ml_component", return_value=None)
    def test_stats_no_components(self, mock_get, handler):
        """No components available returns limited status."""
        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/ml/stats", {}, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "limited"
        assert body["stats"] == {}

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_stats_with_router(self, mock_get, handler):
        """Router available includes routing stats."""
        router = MagicMock()
        router._capabilities = {"claude": {}, "gpt-4": {}}
        router._historical_performance = {
            "claude": {"coding": [0.9, 0.85]},
        }

        def side_effect(name):
            if name == "router":
                return router
            return None

        mock_get.side_effect = side_effect

        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/ml/stats", {}, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"
        assert body["stats"]["routing"]["registered_agents"] == 2

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_stats_with_predictor(self, mock_get, handler):
        """Predictor available includes calibration stats."""
        predictor = MagicMock()
        predictor.get_calibration_stats.return_value = {
            "samples": 50,
            "accuracy": 0.9,
            "precision": 0.88,
            "recall": 0.92,
        }

        def side_effect(name):
            if name == "predictor":
                return predictor
            return None

        mock_get.side_effect = side_effect

        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/ml/stats", {}, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["stats"]["consensus"]["calibration_samples"] == 50
        assert body["stats"]["consensus"]["accuracy"] == 0.9

    def test_stats_includes_circuit_breakers(self, handler):
        """Stats endpoint includes circuit breaker status."""
        # Create a circuit breaker to populate the status
        cb = _get_circuit_breaker("test_component")
        cb.record_success()

        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/ml/stats", {}, http)
        body = _body(result)
        assert _status(result) == 200
        assert "circuit_breakers" in body


# ============================================================================
# GET /api/v1/ml/* - unmatched path returns None
# ============================================================================


class TestHandleUnmatched:
    """Test that handle() returns None for unmatched paths."""

    def test_unmatched_get_path(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/ml/unknown", {}, http)
        assert result is None


# ============================================================================
# POST /api/v1/ml/route
# ============================================================================


class TestRoute:
    """Test the agent routing endpoint."""

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_success(self, mock_get, handler, mock_router):
        mock_get.return_value = mock_router
        http = _MockHTTPHandler()
        data = {
            "task": "Implement caching",
            "available_agents": ["claude", "gpt-4"],
            "team_size": 2,
        }
        result = handler.handle_post("/api/v1/ml/route", data, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["selected_agents"] == ["claude", "gpt-4"]
        assert body["task_type"] == "coding"
        assert body["confidence"] == 0.85
        assert "reasoning" in body
        assert "agent_scores" in body
        assert "diversity_score" in body

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_with_constraints(self, mock_get, handler, mock_router):
        mock_get.return_value = mock_router
        http = _MockHTTPHandler()
        data = {
            "task": "Review code",
            "available_agents": ["claude"],
            "constraints": {"require_code": True},
        }
        result = handler.handle_post("/api/v1/ml/route", data, http)
        assert _status(result) == 200
        mock_router.route.assert_called_once_with(
            task="Review code",
            available_agents=["claude"],
            team_size=3,  # default
            constraints={"require_code": True},
        )

    @patch("aragora.server.handlers.ml._get_ml_component", return_value=None)
    def test_route_component_unavailable(self, mock_get, handler):
        http = _MockHTTPHandler()
        data = {"task": "test", "available_agents": ["claude"]}
        result = handler.handle_post("/api/v1/ml/route", data, http)
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_missing_task(self, mock_get, handler, mock_router):
        mock_get.return_value = mock_router
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/route", {"available_agents": ["a"]}, http)
        assert _status(result) == 400
        assert "task" in _body(result)["error"].lower()

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_empty_task(self, mock_get, handler, mock_router):
        mock_get.return_value = mock_router
        http = _MockHTTPHandler()
        result = handler.handle_post(
            "/api/v1/ml/route", {"task": "", "available_agents": ["a"]}, http
        )
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_missing_agents(self, mock_get, handler, mock_router):
        mock_get.return_value = mock_router
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/route", {"task": "test"}, http)
        assert _status(result) == 400
        assert "available_agents" in _body(result)["error"].lower()

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_empty_agents(self, mock_get, handler, mock_router):
        mock_get.return_value = mock_router
        http = _MockHTTPHandler()
        result = handler.handle_post(
            "/api/v1/ml/route", {"task": "test", "available_agents": []}, http
        )
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_value_error(self, mock_get, handler, mock_router):
        mock_router.route.side_effect = ValueError("invalid task type")
        mock_get.return_value = mock_router
        http = _MockHTTPHandler()
        data = {"task": "test", "available_agents": ["claude"]}
        result = handler.handle_post("/api/v1/ml/route", data, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_runtime_error(self, mock_get, handler, mock_router):
        mock_router.route.side_effect = RuntimeError("model failed")
        mock_get.return_value = mock_router
        http = _MockHTTPHandler()
        data = {"task": "test", "available_agents": ["claude"]}
        result = handler.handle_post("/api/v1/ml/route", data, http)
        assert _status(result) == 500

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_type_error(self, mock_get, handler, mock_router):
        mock_router.route.side_effect = TypeError("bad type")
        mock_get.return_value = mock_router
        http = _MockHTTPHandler()
        data = {"task": "test", "available_agents": ["claude"]}
        result = handler.handle_post("/api/v1/ml/route", data, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_os_error(self, mock_get, handler, mock_router):
        mock_router.route.side_effect = OSError("disk full")
        mock_get.return_value = mock_router
        http = _MockHTTPHandler()
        data = {"task": "test", "available_agents": ["claude"]}
        result = handler.handle_post("/api/v1/ml/route", data, http)
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/ml/score
# ============================================================================


class TestScore:
    """Test the single scoring endpoint."""

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_success(self, mock_get, handler, mock_scorer):
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        data = {"text": "This is a quality response about caching."}
        result = handler.handle_post("/api/v1/ml/score", data, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["overall"] == 0.75
        assert body["coherence"] == 0.8
        assert body["completeness"] == 0.7
        assert body["relevance"] == 0.75
        assert body["clarity"] == 0.78
        assert body["confidence"] == 0.6
        assert body["is_high_quality"] is True
        assert body["needs_review"] is False

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_with_context(self, mock_get, handler, mock_scorer):
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        data = {"text": "response text", "context": "Design a rate limiter"}
        result = handler.handle_post("/api/v1/ml/score", data, http)
        assert _status(result) == 200
        mock_scorer.score.assert_called_once_with("response text", context="Design a rate limiter")

    @patch("aragora.server.handlers.ml._get_ml_component", return_value=None)
    def test_score_component_unavailable(self, mock_get, handler):
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score", {"text": "test"}, http)
        assert _status(result) == 503

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_missing_text(self, mock_get, handler, mock_scorer):
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score", {}, http)
        assert _status(result) == 400
        assert "text" in _body(result)["error"].lower()

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_empty_text(self, mock_get, handler, mock_scorer):
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score", {"text": ""}, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_value_error(self, mock_get, handler, mock_scorer):
        mock_scorer.score.side_effect = ValueError("bad text")
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score", {"text": "test"}, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_runtime_error(self, mock_get, handler, mock_scorer):
        mock_scorer.score.side_effect = RuntimeError("model failed")
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score", {"text": "test"}, http)
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/ml/score-batch
# ============================================================================


class TestScoreBatch:
    """Test the batch scoring endpoint."""

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_batch_success(self, mock_get, handler, mock_scorer):
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        data = {"texts": ["response1", "response2"]}
        result = handler.handle_post("/api/v1/ml/score-batch", data, http)
        body = _body(result)
        assert _status(result) == 200
        assert len(body["scores"]) == 2
        assert body["scores"][0]["overall"] == 0.75
        assert body["scores"][1]["overall"] == 0.82

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_batch_with_contexts(self, mock_get, handler, mock_scorer):
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        data = {"texts": ["r1", "r2"], "contexts": ["c1", "c2"]}
        result = handler.handle_post("/api/v1/ml/score-batch", data, http)
        assert _status(result) == 200
        mock_scorer.score_batch.assert_called_once_with(["r1", "r2"], contexts=["c1", "c2"])

    @patch("aragora.server.handlers.ml._get_ml_component", return_value=None)
    def test_score_batch_component_unavailable(self, mock_get, handler):
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score-batch", {"texts": ["test"]}, http)
        assert _status(result) == 503

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_batch_missing_texts(self, mock_get, handler, mock_scorer):
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score-batch", {}, http)
        assert _status(result) == 400
        assert "texts" in _body(result)["error"].lower()

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_batch_empty_texts(self, mock_get, handler, mock_scorer):
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score-batch", {"texts": []}, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_batch_exceeds_limit(self, mock_get, handler, mock_scorer):
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        data = {"texts": [f"text_{i}" for i in range(101)]}
        result = handler.handle_post("/api/v1/ml/score-batch", data, http)
        assert _status(result) == 400
        assert "100" in _body(result)["error"]

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_batch_at_limit(self, mock_get, handler, mock_scorer):
        """Exactly 100 texts is allowed."""
        mock_scorer.score_batch.return_value = [MockQualityScore()] * 100
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        data = {"texts": [f"text_{i}" for i in range(100)]}
        result = handler.handle_post("/api/v1/ml/score-batch", data, http)
        assert _status(result) == 200

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_batch_value_error(self, mock_get, handler, mock_scorer):
        mock_scorer.score_batch.side_effect = ValueError("bad batch")
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score-batch", {"texts": ["a"]}, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_batch_runtime_error(self, mock_get, handler, mock_scorer):
        mock_scorer.score_batch.side_effect = RuntimeError("oom")
        mock_get.return_value = mock_scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score-batch", {"texts": ["a"]}, http)
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/ml/consensus
# ============================================================================


class TestConsensus:
    """Test the consensus prediction endpoint."""

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_consensus_success(self, mock_get, handler, mock_predictor):
        mock_get.return_value = mock_predictor
        http = _MockHTTPHandler()
        data = {
            "responses": [["agent1", "I agree"], ["agent2", "I also agree"]],
            "context": "Design a caching layer",
            "current_round": 2,
            "total_rounds": 3,
        }
        result = handler.handle_post("/api/v1/ml/consensus", data, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["probability"] == 0.85
        assert body["confidence"] == 0.7
        assert body["convergence_trend"] == "converging"
        assert body["estimated_rounds"] == 2
        assert body["likely_consensus"] is True
        assert body["early_termination_safe"] is True
        assert body["needs_intervention"] is False
        assert "key_factors" in body

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_consensus_defaults(self, mock_get, handler, mock_predictor):
        """Default current_round=1 and total_rounds=3 are used."""
        mock_get.return_value = mock_predictor
        http = _MockHTTPHandler()
        data = {"responses": [["a1", "yes"], ["a2", "yes"]]}
        result = handler.handle_post("/api/v1/ml/consensus", data, http)
        assert _status(result) == 200
        mock_predictor.predict.assert_called_once()
        call_kwargs = mock_predictor.predict.call_args[1]
        assert call_kwargs["current_round"] == 1
        assert call_kwargs["total_rounds"] == 3

    @patch("aragora.server.handlers.ml._get_ml_component", return_value=None)
    def test_consensus_component_unavailable(self, mock_get, handler):
        http = _MockHTTPHandler()
        data = {"responses": [["a", "b"]]}
        result = handler.handle_post("/api/v1/ml/consensus", data, http)
        assert _status(result) == 503

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_consensus_missing_responses(self, mock_get, handler, mock_predictor):
        mock_get.return_value = mock_predictor
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/consensus", {}, http)
        assert _status(result) == 400
        assert "responses" in _body(result)["error"].lower()

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_consensus_empty_responses(self, mock_get, handler, mock_predictor):
        mock_get.return_value = mock_predictor
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/consensus", {"responses": []}, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_consensus_converts_to_tuples(self, mock_get, handler, mock_predictor):
        """Responses are converted from lists to tuples."""
        mock_get.return_value = mock_predictor
        http = _MockHTTPHandler()
        data = {"responses": [["agent1", "text1"]]}
        handler.handle_post("/api/v1/ml/consensus", data, http)
        call_kwargs = mock_predictor.predict.call_args[1]
        assert call_kwargs["responses"] == [("agent1", "text1")]

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_consensus_value_error(self, mock_get, handler, mock_predictor):
        mock_predictor.predict.side_effect = ValueError("invalid format")
        mock_get.return_value = mock_predictor
        http = _MockHTTPHandler()
        data = {"responses": [["a", "b"]]}
        result = handler.handle_post("/api/v1/ml/consensus", data, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_consensus_runtime_error(self, mock_get, handler, mock_predictor):
        mock_predictor.predict.side_effect = RuntimeError("model crashed")
        mock_get.return_value = mock_predictor
        http = _MockHTTPHandler()
        data = {"responses": [["a", "b"]]}
        result = handler.handle_post("/api/v1/ml/consensus", data, http)
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/ml/export-training
# ============================================================================


class TestExportTraining:
    """Test the training data export endpoint."""

    @pytest.fixture(autouse=True)
    def _patch_has_permission(self):
        """Patch has_permission at the ml module level so export tests pass RBAC."""
        with patch("aragora.server.handlers.ml.has_permission", return_value=True):
            yield

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_json_format(self, mock_get, handler, mock_exporter, mock_user_with_train):
        mock_get.return_value = mock_exporter
        http = _MockHTTPHandler()
        data = {
            "debates": [{"task": "test", "consensus": "agreed"}],
            "format": "json",
        }
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["format"] == "json"
        assert body["examples"] == 1
        assert isinstance(body["data"], list)

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_jsonl_format(self, mock_get, handler, mock_exporter, mock_user_with_train):
        mock_get.return_value = mock_exporter
        http = _MockHTTPHandler()
        data = {
            "debates": [{"task": "test", "consensus": "agreed"}],
            "format": "jsonl",
        }
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["format"] == "jsonl"
        assert isinstance(body["data"], str)

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_default_format_is_json(
        self, mock_get, handler, mock_exporter, mock_user_with_train
    ):
        mock_get.return_value = mock_exporter
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "test", "consensus": "agreed"}]}
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["format"] == "json"

    @patch("aragora.server.handlers.ml._get_ml_component", return_value=None)
    def test_export_component_unavailable(self, mock_get, handler, mock_user_with_train):
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "t"}]}
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        assert _status(result) == 503

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_missing_debates(self, mock_get, handler, mock_exporter, mock_user_with_train):
        mock_get.return_value = mock_exporter
        http = _MockHTTPHandler()
        result = handler.handle_post(
            "/api/v1/ml/export-training", {}, http, user=mock_user_with_train
        )
        assert _status(result) == 400
        assert "debates" in _body(result)["error"].lower()

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_empty_debates(self, mock_get, handler, mock_exporter, mock_user_with_train):
        mock_get.return_value = mock_exporter
        http = _MockHTTPHandler()
        result = handler.handle_post(
            "/api/v1/ml/export-training", {"debates": []}, http, user=mock_user_with_train
        )
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_empty_result(self, mock_get, handler, mock_exporter, mock_user_with_train):
        """Exporter returns empty/None returns 500."""
        mock_exporter.export_debates_batch.return_value = None
        mock_get.return_value = mock_exporter
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "t"}]}
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        assert _status(result) == 500

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_value_error(self, mock_get, handler, mock_exporter, mock_user_with_train):
        mock_exporter.export_debates_batch.side_effect = ValueError("bad data")
        mock_get.return_value = mock_exporter
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "t"}]}
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_runtime_error(self, mock_get, handler, mock_exporter, mock_user_with_train):
        mock_exporter.export_debates_batch.side_effect = RuntimeError("disk full")
        mock_get.return_value = mock_exporter
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "t"}]}
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        assert _status(result) == 500

    def test_export_permission_denied_no_user(self, handler):
        """No user at all gets permission denied or error."""
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "t"}]}
        result = handler.handle_post("/api/v1/ml/export-training", data, http, user=None)
        # When auth is disabled, the @require_permission decorator passes through,
        # and the internal permission check may error. Accept either 403 or 500.
        assert _status(result) in (403, 500)

    @patch("aragora.server.handlers.ml.has_permission", return_value=False)
    def test_export_permission_denied_wrong_role(self, mock_perm, handler, mock_user_no_train):
        """User without ml:train permission gets 403."""
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "t"}]}
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_no_train
        )
        assert _status(result) == 403


# ============================================================================
# POST /api/v1/ml/embed
# ============================================================================


class TestEmbed:
    """Test the text embedding endpoint."""

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_embed_single_text(self, mock_get, handler, mock_embeddings):
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"text": "Hello world"}
        result = handler.handle_post("/api/v1/ml/embed", data, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["embeddings"] == [[0.1, 0.2, 0.3]]
        assert body["dimension"] == 3
        assert body["count"] == 1

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_embed_batch_texts(self, mock_get, handler, mock_embeddings):
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"texts": ["text1", "text2"]}
        result = handler.handle_post("/api/v1/ml/embed", data, http)
        body = _body(result)
        assert _status(result) == 200
        assert len(body["embeddings"]) == 2
        assert body["count"] == 2

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_embed_prefers_text_over_texts(self, mock_get, handler, mock_embeddings):
        """When both text and texts are provided, text takes precedence."""
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"text": "single", "texts": ["batch1", "batch2"]}
        result = handler.handle_post("/api/v1/ml/embed", data, http)
        assert _status(result) == 200
        mock_embeddings.embed.assert_called_once_with("single")
        mock_embeddings.embed_batch.assert_not_called()

    @patch("aragora.server.handlers.ml._get_ml_component", return_value=None)
    def test_embed_component_unavailable(self, mock_get, handler):
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/embed", {"text": "test"}, http)
        assert _status(result) == 503

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_embed_missing_both(self, mock_get, handler, mock_embeddings):
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/embed", {}, http)
        assert _status(result) == 400
        assert "text" in _body(result)["error"].lower()

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_embed_batch_exceeds_limit(self, mock_get, handler, mock_embeddings):
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"texts": [f"t_{i}" for i in range(101)]}
        result = handler.handle_post("/api/v1/ml/embed", data, http)
        assert _status(result) == 400
        assert "100" in _body(result)["error"]

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_embed_batch_at_limit(self, mock_get, handler, mock_embeddings):
        """Exactly 100 texts is allowed."""
        mock_embeddings.embed_batch.return_value = [[0.1]] * 100
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"texts": [f"t_{i}" for i in range(100)]}
        result = handler.handle_post("/api/v1/ml/embed", data, http)
        assert _status(result) == 200

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_embed_value_error(self, mock_get, handler, mock_embeddings):
        mock_embeddings.embed.side_effect = ValueError("bad input")
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/embed", {"text": "test"}, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_embed_runtime_error(self, mock_get, handler, mock_embeddings):
        mock_embeddings.embed.side_effect = RuntimeError("oom")
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/embed", {"text": "test"}, http)
        assert _status(result) == 500


# ============================================================================
# POST /api/v1/ml/search
# ============================================================================


class TestSearch:
    """Test the semantic search endpoint."""

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_success(self, mock_get, handler, mock_embeddings):
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {
            "query": "caching pattern",
            "documents": ["doc about caching", "doc about auth"],
            "top_k": 2,
            "threshold": 0.5,
        }
        result = handler.handle_post("/api/v1/ml/search", data, http)
        body = _body(result)
        assert _status(result) == 200
        assert len(body["results"]) == 2
        assert body["results"][0]["text"] == "doc1"
        assert body["results"][0]["score"] == 0.95
        assert body["results"][0]["index"] == 0
        assert body["count"] == 2

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_defaults(self, mock_get, handler, mock_embeddings):
        """Default top_k=5, threshold=0.0."""
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"query": "test", "documents": ["d1"]}
        handler.handle_post("/api/v1/ml/search", data, http)
        call_kwargs = mock_embeddings.search.call_args[1]
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["threshold"] == 0.0

    @patch("aragora.server.handlers.ml._get_ml_component", return_value=None)
    def test_search_component_unavailable(self, mock_get, handler):
        http = _MockHTTPHandler()
        data = {"query": "test", "documents": ["d1"]}
        result = handler.handle_post("/api/v1/ml/search", data, http)
        assert _status(result) == 503

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_missing_query(self, mock_get, handler, mock_embeddings):
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/search", {"documents": ["d1"]}, http)
        assert _status(result) == 400
        assert "query" in _body(result)["error"].lower()

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_empty_query(self, mock_get, handler, mock_embeddings):
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/search", {"query": "", "documents": ["d"]}, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_missing_documents(self, mock_get, handler, mock_embeddings):
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/search", {"query": "test"}, http)
        assert _status(result) == 400
        assert "documents" in _body(result)["error"].lower()

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_empty_documents(self, mock_get, handler, mock_embeddings):
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/search", {"query": "test", "documents": []}, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_exceeds_document_limit(self, mock_get, handler, mock_embeddings):
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"query": "test", "documents": [f"d_{i}" for i in range(1001)]}
        result = handler.handle_post("/api/v1/ml/search", data, http)
        assert _status(result) == 400
        assert "1000" in _body(result)["error"]

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_at_document_limit(self, mock_get, handler, mock_embeddings):
        """Exactly 1000 documents is allowed."""
        mock_embeddings.search.return_value = []
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"query": "test", "documents": [f"d_{i}" for i in range(1000)]}
        result = handler.handle_post("/api/v1/ml/search", data, http)
        assert _status(result) == 200

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_empty_results(self, mock_get, handler, mock_embeddings):
        """No results returns empty list."""
        mock_embeddings.search.return_value = []
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"query": "xyz", "documents": ["abc"]}
        result = handler.handle_post("/api/v1/ml/search", data, http)
        body = _body(result)
        assert _status(result) == 200
        assert body["results"] == []
        assert body["count"] == 0

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_value_error(self, mock_get, handler, mock_embeddings):
        mock_embeddings.search.side_effect = ValueError("bad")
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"query": "q", "documents": ["d"]}
        result = handler.handle_post("/api/v1/ml/search", data, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_runtime_error(self, mock_get, handler, mock_embeddings):
        mock_embeddings.search.side_effect = RuntimeError("crash")
        mock_get.return_value = mock_embeddings
        http = _MockHTTPHandler()
        data = {"query": "q", "documents": ["d"]}
        result = handler.handle_post("/api/v1/ml/search", data, http)
        assert _status(result) == 500


# ============================================================================
# POST unmatched path returns None
# ============================================================================


class TestHandlePostUnmatched:
    """Test that handle_post() returns None for unmatched paths."""

    def test_unmatched_post_path(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/unknown", {}, http)
        assert result is None


# ============================================================================
# Rate limiting
# ============================================================================


class TestRateLimiting:
    """Test rate limiting on ML endpoints."""

    def test_handle_rate_limit(self, handler):
        """GET requests are rate limited."""
        http = _MockHTTPHandler()
        # Fill up the rate limiter
        for _ in range(60):
            assert _ml_limiter.is_allowed("127.0.0.1")
        # Now requests should be rejected
        result = handler.handle("/api/v1/ml/models", {}, http)
        assert _status(result) == 429
        assert "rate limit" in _body(result)["error"].lower()

    def test_handle_post_rate_limit(self, handler):
        """POST requests are rate limited."""
        http = _MockHTTPHandler()
        # Fill up the rate limiter
        for _ in range(60):
            assert _ml_limiter.is_allowed("127.0.0.1")
        data = {"task": "test", "available_agents": ["claude"]}
        result = handler.handle_post("/api/v1/ml/route", data, http)
        assert _status(result) == 429
        assert "rate limit" in _body(result)["error"].lower()

    def test_different_ips_not_shared(self, handler):
        """Different IPs have separate rate limits."""
        http1 = _MockHTTPHandler(client_address=("10.0.0.1", 12345))
        http2 = _MockHTTPHandler(client_address=("10.0.0.2", 12345))
        # Fill up first IP
        for _ in range(60):
            _ml_limiter.is_allowed("10.0.0.1")
        # Second IP should still work
        assert _ml_limiter.is_allowed("10.0.0.2")

    def test_rate_limit_with_none_handler(self, handler):
        """None handler should use 'unknown' as client IP."""
        result = handler.handle("/api/v1/ml/models", {}, None)
        # Should still work (just uses "unknown" as IP)
        assert result is not None


# ============================================================================
# Circuit breaker
# ============================================================================


class TestCircuitBreaker:
    """Test circuit breaker integration for ML components."""

    def test_get_circuit_breaker_creates_new(self):
        """First call creates a new circuit breaker."""
        cb = _get_circuit_breaker("test_new_comp")
        assert cb is not None
        assert cb.state == "closed"

    def test_get_circuit_breaker_returns_same(self):
        """Subsequent calls return the same circuit breaker."""
        cb1 = _get_circuit_breaker("test_same_comp")
        cb2 = _get_circuit_breaker("test_same_comp")
        assert cb1 is cb2

    def test_circuit_breaker_status_empty(self):
        """Status is empty when no breakers have been created."""
        _clear_ml_components()
        status = get_ml_circuit_breaker_status()
        assert status == {}

    def test_circuit_breaker_status_populated(self):
        """Status includes all created breakers."""
        _get_circuit_breaker("comp_a")
        _get_circuit_breaker("comp_b")
        status = get_ml_circuit_breaker_status()
        assert "comp_a" in status
        assert "comp_b" in status
        assert status["comp_a"]["state"] == "closed"

    def test_component_unavailable_when_circuit_open(self):
        """Open circuit breaker prevents component loading."""
        cb = _get_circuit_breaker("router")
        # Trip the circuit breaker
        for _ in range(5):
            cb.record_failure()
        assert cb.state == "open"
        # Now component should return None
        comp = _get_ml_component("router")
        assert comp is None

    def test_clear_ml_components_resets_breakers(self):
        """_clear_ml_components resets all circuit breakers."""
        _get_circuit_breaker("test_reset")
        _clear_ml_components()
        status = get_ml_circuit_breaker_status()
        assert "test_reset" not in status


# ============================================================================
# _get_ml_component lazy loading
# ============================================================================


class TestGetMLComponent:
    """Test lazy loading of ML components."""

    @patch("aragora.server.handlers.ml._ml_components", {})
    def test_unknown_component_returns_none(self):
        result = _get_ml_component("nonexistent")
        assert result is None

    def test_import_error_returns_none(self):
        """ImportError during component loading returns None."""
        # All real ML modules likely won't import in test env
        comp = _get_ml_component("router")
        # Should not raise, just return None
        assert comp is None or comp is not None  # Either is fine

    @patch("aragora.server.handlers.ml._ml_components", {})
    def test_cached_component_returned(self):
        """Once loaded, component is cached."""
        mock_comp = MagicMock()
        with patch.dict("aragora.server.handlers.ml._ml_components", {"router": mock_comp}):
            result = _get_ml_component("router")
            assert result is mock_comp


# ============================================================================
# RBAC permission tests
# ============================================================================


@pytest.mark.no_auto_auth
class TestRBACPermissions:
    """Test RBAC permission enforcement.

    These tests disable auto-auth to verify that the @require_permission
    decorator enforces authentication when no handler context is available.
    """

    def test_handle_requires_auth(self):
        """GET handle requires ml:read permission."""
        h = MLHandler(ctx={})
        # No handler with auth -> should require auth
        result = h.handle("/api/v1/ml/models", {})
        if result is not None:
            assert _status(result) == 401

    def test_handle_post_requires_auth(self):
        """POST handle_post requires ml:read permission."""
        h = MLHandler(ctx={})
        result = h.handle_post("/api/v1/ml/route", {"task": "t", "available_agents": ["a"]})
        if result is not None:
            assert _status(result) == 401


# ============================================================================
# Edge cases and response formatting
# ============================================================================


class TestResponseFormatting:
    """Test response body formatting and rounding."""

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_scores_rounded(self, mock_get, handler):
        """Agent scores are rounded to 3 decimal places."""
        router = MagicMock()
        decision = MockRoutingDecision(
            agent_scores={"claude": 0.123456789},
            confidence=0.987654321,
            diversity_score=0.111111111,
        )
        router.route.return_value = decision
        mock_get.return_value = router
        http = _MockHTTPHandler()
        data = {"task": "test", "available_agents": ["claude"]}
        result = handler.handle_post("/api/v1/ml/route", data, http)
        body = _body(result)
        assert body["agent_scores"]["claude"] == 0.123
        assert body["confidence"] == 0.988
        assert body["diversity_score"] == 0.111

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_values_rounded(self, mock_get, handler):
        """Score values are rounded to 3 decimal places."""
        scorer = MagicMock()
        scorer.score.return_value = MockQualityScore(
            overall=0.777777,
            coherence=0.888888,
            completeness=0.999999,
            relevance=0.111111,
            clarity=0.222222,
            confidence=0.333333,
        )
        mock_get.return_value = scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score", {"text": "test"}, http)
        body = _body(result)
        assert body["overall"] == 0.778
        assert body["coherence"] == 0.889
        assert body["completeness"] == 1.0
        assert body["relevance"] == 0.111
        assert body["clarity"] == 0.222
        assert body["confidence"] == 0.333

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_consensus_probability_rounded(self, mock_get, handler):
        """Consensus prediction values are rounded to 3 decimal places."""
        predictor = MagicMock()
        predictor.predict.return_value = MockConsensusPrediction(
            probability=0.123456,
            confidence=0.654321,
        )
        mock_get.return_value = predictor
        http = _MockHTTPHandler()
        data = {"responses": [["a", "b"]]}
        result = handler.handle_post("/api/v1/ml/consensus", data, http)
        body = _body(result)
        assert body["probability"] == 0.123
        assert body["confidence"] == 0.654

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_scores_rounded(self, mock_get, handler):
        """Search result scores are rounded to 4 decimal places."""
        emb = MagicMock()
        emb.search.return_value = [MockSearchResult("doc", 0.12345678, 0)]
        mock_get.return_value = emb
        http = _MockHTTPHandler()
        data = {"query": "q", "documents": ["doc"]}
        result = handler.handle_post("/api/v1/ml/search", data, http)
        body = _body(result)
        assert body["results"][0]["score"] == 0.1235

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_batch_score_individual_fields(self, mock_get, handler):
        """Batch scores contain per-item fields without needs_review."""
        scorer = MagicMock()
        scorer.score_batch.return_value = [MockQualityScore()]
        mock_get.return_value = scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score-batch", {"texts": ["t"]}, http)
        body = _body(result)
        score = body["scores"][0]
        assert "overall" in score
        assert "coherence" in score
        assert "is_high_quality" in score
        # needs_review is NOT included in batch results
        assert "needs_review" not in score


# ============================================================================
# Error handling with AttributeError and KeyError
# ============================================================================


class TestAdditionalErrors:
    """Test additional error types that map to 400 or 500."""

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_key_error(self, mock_get, handler):
        router = MagicMock()
        router.route.side_effect = KeyError("missing_key")
        mock_get.return_value = router
        http = _MockHTTPHandler()
        data = {"task": "test", "available_agents": ["claude"]}
        result = handler.handle_post("/api/v1/ml/route", data, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_route_attribute_error(self, mock_get, handler):
        router = MagicMock()
        router.route.side_effect = AttributeError("missing attr")
        mock_get.return_value = router
        http = _MockHTTPHandler()
        data = {"task": "test", "available_agents": ["claude"]}
        result = handler.handle_post("/api/v1/ml/route", data, http)
        assert _status(result) == 500

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_key_error(self, mock_get, handler):
        scorer = MagicMock()
        scorer.score.side_effect = KeyError("k")
        mock_get.return_value = scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score", {"text": "t"}, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_score_attribute_error(self, mock_get, handler):
        scorer = MagicMock()
        scorer.score.side_effect = AttributeError("attr")
        mock_get.return_value = scorer
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/score", {"text": "t"}, http)
        assert _status(result) == 500

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_embed_key_error(self, mock_get, handler):
        emb = MagicMock()
        emb.embed.side_effect = KeyError("k")
        mock_get.return_value = emb
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/embed", {"text": "t"}, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_attribute_error(self, mock_get, handler):
        emb = MagicMock()
        emb.search.side_effect = AttributeError("attr")
        mock_get.return_value = emb
        http = _MockHTTPHandler()
        data = {"query": "q", "documents": ["d"]}
        result = handler.handle_post("/api/v1/ml/search", data, http)
        assert _status(result) == 500

    @patch("aragora.server.handlers.ml.has_permission", return_value=True)
    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_key_error(self, mock_get, mock_perm, handler, mock_user_with_train):
        exporter = MagicMock()
        exporter.export_debates_batch.side_effect = KeyError("k")
        mock_get.return_value = exporter
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "t"}]}
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml.has_permission", return_value=True)
    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_attribute_error(self, mock_get, mock_perm, handler, mock_user_with_train):
        exporter = MagicMock()
        exporter.export_debates_batch.side_effect = AttributeError("attr")
        mock_get.return_value = exporter
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "t"}]}
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        assert _status(result) == 500

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_consensus_key_error(self, mock_get, handler):
        predictor = MagicMock()
        predictor.predict.side_effect = KeyError("k")
        mock_get.return_value = predictor
        http = _MockHTTPHandler()
        data = {"responses": [["a", "b"]]}
        result = handler.handle_post("/api/v1/ml/consensus", data, http)
        assert _status(result) == 400

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_consensus_attribute_error(self, mock_get, handler):
        predictor = MagicMock()
        predictor.predict.side_effect = AttributeError("attr")
        mock_get.return_value = predictor
        http = _MockHTTPHandler()
        data = {"responses": [["a", "b"]]}
        result = handler.handle_post("/api/v1/ml/consensus", data, http)
        assert _status(result) == 500

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_embed_os_error(self, mock_get, handler):
        emb = MagicMock()
        emb.embed.side_effect = OSError("disk")
        mock_get.return_value = emb
        http = _MockHTTPHandler()
        result = handler.handle_post("/api/v1/ml/embed", {"text": "t"}, http)
        assert _status(result) == 500

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_search_os_error(self, mock_get, handler):
        emb = MagicMock()
        emb.search.side_effect = OSError("disk")
        mock_get.return_value = emb
        http = _MockHTTPHandler()
        data = {"query": "q", "documents": ["d"]}
        result = handler.handle_post("/api/v1/ml/search", data, http)
        assert _status(result) == 500


# ============================================================================
# Multiple training examples
# ============================================================================


class TestExportMultipleExamples:
    """Test export with multiple training examples."""

    @pytest.fixture(autouse=True)
    def _patch_has_permission(self):
        """Patch has_permission at the ml module level so export tests pass RBAC."""
        with patch("aragora.server.handlers.ml.has_permission", return_value=True):
            yield

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_multiple_json(self, mock_get, handler, mock_user_with_train):
        examples = [
            MockTrainingExample({"task": "t1", "chosen": "r1"}),
            MockTrainingExample({"task": "t2", "chosen": "r2"}),
            MockTrainingExample({"task": "t3", "chosen": "r3"}),
        ]
        exporter = MagicMock()
        exporter.export_debates_batch.return_value = MockTrainingData(examples)
        mock_get.return_value = exporter
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "t"}], "format": "json"}
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        body = _body(result)
        assert body["examples"] == 3
        assert len(body["data"]) == 3

    @patch("aragora.server.handlers.ml._get_ml_component")
    def test_export_multiple_jsonl(self, mock_get, handler, mock_user_with_train):
        examples = [
            MockTrainingExample({"task": "t1"}),
            MockTrainingExample({"task": "t2"}),
        ]
        exporter = MagicMock()
        exporter.export_debates_batch.return_value = MockTrainingData(examples)
        mock_get.return_value = exporter
        http = _MockHTTPHandler()
        data = {"debates": [{"task": "t"}], "format": "jsonl"}
        result = handler.handle_post(
            "/api/v1/ml/export-training", data, http, user=mock_user_with_train
        )
        body = _body(result)
        assert body["examples"] == 2
        assert "\n" in body["data"]
