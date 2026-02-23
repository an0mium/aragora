"""
Tests for Evaluation Handler.

Tests cover:
- Handler routing (can_handle, ROUTES)
- GET /api/v1/evaluate/dimensions - List evaluation dimensions
- GET /api/v1/evaluate/profiles - List weight profiles
- POST /api/v1/evaluate - Evaluate a response
- POST /api/v1/evaluate/compare - Compare two responses
- Rate limiting
- Input validation (missing fields, invalid JSON, invalid dimensions)
- 503 when LLM Judge not available
- Error handling via @handle_errors
- Security tests (path traversal, injection)
"""

from __future__ import annotations

import io
import json
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.evaluation import EvaluationHandler


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


def _status(result) -> int:
    """Get status code from HandlerResult."""
    return result.status_code


def _make_http_handler(body_data: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with optional JSON body."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.command = "GET"
    mock.path = "/api/v1/evaluate"

    if body_data is not None:
        body_bytes = json.dumps(body_data).encode("utf-8")
        mock.headers = {"Content-Length": str(len(body_bytes))}
        mock.rfile = io.BytesIO(body_bytes)
    else:
        mock.headers = {"Content-Length": "0"}
        mock.rfile = io.BytesIO(b"")

    return mock


# ============================================================================
# Mock Evaluation Types
# ============================================================================


class MockEvaluationDimension(Enum):
    """Mock evaluation dimension."""

    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"


class MockRubric:
    """Mock rubric for a dimension."""

    def __init__(self, description="Test description"):
        self.description = description
        self.score_1 = "Poor"
        self.score_2 = "Below Average"
        self.score_3 = "Average"
        self.score_4 = "Good"
        self.score_5 = "Excellent"


MOCK_DEFAULT_RUBRICS = {
    MockEvaluationDimension.ACCURACY: MockRubric("Measures factual accuracy"),
    MockEvaluationDimension.RELEVANCE: MockRubric("Measures relevance to query"),
    MockEvaluationDimension.CLARITY: MockRubric("Measures clarity of expression"),
    MockEvaluationDimension.COMPLETENESS: MockRubric("Measures completeness of answer"),
}

MOCK_DEFAULT_WEIGHTS = {
    MockEvaluationDimension.ACCURACY: 0.3,
    MockEvaluationDimension.RELEVANCE: 0.3,
    MockEvaluationDimension.CLARITY: 0.2,
    MockEvaluationDimension.COMPLETENESS: 0.2,
}

MOCK_WEIGHT_PROFILES = {
    "factual_qa": {
        MockEvaluationDimension.ACCURACY: 0.5,
        MockEvaluationDimension.RELEVANCE: 0.3,
        MockEvaluationDimension.CLARITY: 0.1,
        MockEvaluationDimension.COMPLETENESS: 0.1,
    },
    "creative_writing": {
        MockEvaluationDimension.ACCURACY: 0.1,
        MockEvaluationDimension.RELEVANCE: 0.2,
        MockEvaluationDimension.CLARITY: 0.4,
        MockEvaluationDimension.COMPLETENESS: 0.3,
    },
}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def handler():
    """Create EvaluationHandler instance."""
    return EvaluationHandler({})


@pytest.fixture
def handler_with_ctx():
    """Create EvaluationHandler with custom context."""
    return EvaluationHandler({"user_store": MagicMock()})


@pytest.fixture
def mock_http_handler():
    """Create a basic mock HTTP handler."""
    return _make_http_handler()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter between tests."""
    import aragora.server.handlers.evaluation as eval_mod

    eval_mod._evaluation_limiter._buckets.clear()
    yield
    eval_mod._evaluation_limiter._buckets.clear()


# ============================================================================
# Routing Tests
# ============================================================================


class TestEvaluationHandlerRouting:
    """Tests for route matching."""

    def test_can_handle_evaluate_route(self, handler):
        """Handler matches /api/v1/evaluate."""
        assert handler.can_handle("/api/v1/evaluate")

    def test_can_handle_compare_route(self, handler):
        """Handler matches /api/v1/evaluate/compare."""
        assert handler.can_handle("/api/v1/evaluate/compare")

    def test_can_handle_dimensions_route(self, handler):
        """Handler matches /api/v1/evaluate/dimensions."""
        assert handler.can_handle("/api/v1/evaluate/dimensions")

    def test_can_handle_profiles_route(self, handler):
        """Handler matches /api/v1/evaluate/profiles."""
        assert handler.can_handle("/api/v1/evaluate/profiles")

    def test_rejects_unknown_route(self, handler):
        """Handler rejects unknown routes."""
        assert not handler.can_handle("/api/v1/evaluate/unknown")

    def test_rejects_unrelated_route(self, handler):
        """Handler rejects unrelated routes."""
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_match(self, handler):
        """Handler rejects partial route matches."""
        assert not handler.can_handle("/api/v1/eval")

    def test_rejects_extra_path_segments(self, handler):
        """Handler rejects routes with extra path segments."""
        assert not handler.can_handle("/api/v1/evaluate/dimensions/extra")

    def test_routes_list_contains_all_endpoints(self, handler):
        """ROUTES list contains all four endpoints."""
        assert len(handler.ROUTES) == 4
        assert "/api/v1/evaluate" in handler.ROUTES
        assert "/api/v1/evaluate/compare" in handler.ROUTES
        assert "/api/v1/evaluate/dimensions" in handler.ROUTES
        assert "/api/v1/evaluate/profiles" in handler.ROUTES


# ============================================================================
# Initialization Tests
# ============================================================================


class TestEvaluationHandlerInit:
    """Tests for handler initialization."""

    def test_default_context(self):
        """Handler initializes with empty context by default."""
        handler = EvaluationHandler()
        assert handler.ctx == {}

    def test_custom_context(self):
        """Handler stores provided context."""
        ctx = {"key": "value"}
        handler = EvaluationHandler(ctx)
        assert handler.ctx == ctx

    def test_none_context_defaults_to_empty(self):
        """None context defaults to empty dict."""
        handler = EvaluationHandler(None)
        assert handler.ctx == {}

    def test_extends_base_handler(self, handler):
        """Handler extends BaseHandler."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)


# ============================================================================
# GET /api/v1/evaluate/dimensions Tests
# ============================================================================


class TestListDimensions:
    """Tests for list dimensions endpoint."""

    def test_dimensions_success(self, handler, mock_http_handler):
        """Returns dimensions when judge is available."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockEvaluationDimension,
            DEFAULT_RUBRICS=MOCK_DEFAULT_RUBRICS,
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_http_handler)

        assert _status(result) == 200
        data = _body(result)
        assert "dimensions" in data
        assert len(data["dimensions"]) == 4

    def test_dimensions_contain_correct_fields(self, handler, mock_http_handler):
        """Each dimension has id, name, description, rubric."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockEvaluationDimension,
            DEFAULT_RUBRICS=MOCK_DEFAULT_RUBRICS,
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_http_handler)

        data = _body(result)
        dim = data["dimensions"][0]
        assert "id" in dim
        assert "name" in dim
        assert "description" in dim
        assert "rubric" in dim
        assert all(f"score_{i}" in dim["rubric"] for i in range(1, 6))

    def test_dimensions_name_formatting(self, handler, mock_http_handler):
        """Dimension names are title-cased with underscores replaced."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockEvaluationDimension,
            DEFAULT_RUBRICS=MOCK_DEFAULT_RUBRICS,
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_http_handler)

        data = _body(result)
        names = [d["name"] for d in data["dimensions"]]
        assert "Accuracy" in names
        assert "Relevance" in names

    def test_dimensions_rubric_values(self, handler, mock_http_handler):
        """Rubric values are correctly populated."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockEvaluationDimension,
            DEFAULT_RUBRICS=MOCK_DEFAULT_RUBRICS,
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_http_handler)

        data = _body(result)
        rubric = data["dimensions"][0]["rubric"]
        assert rubric["score_1"] == "Poor"
        assert rubric["score_5"] == "Excellent"

    def test_dimensions_without_rubric(self, handler, mock_http_handler):
        """Dimensions without rubric have empty strings."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockEvaluationDimension,
            DEFAULT_RUBRICS={},  # No rubrics
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_http_handler)

        data = _body(result)
        dim = data["dimensions"][0]
        assert dim["description"] == ""
        assert dim["rubric"]["score_1"] == ""

    def test_dimensions_judge_not_available(self, handler, mock_http_handler):
        """Returns 503 when judge is not available."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=False,
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_http_handler)

        assert _status(result) == 503

    def test_dimensions_evaluation_dimension_none(self, handler, mock_http_handler):
        """Returns 503 when EvaluationDimension is None."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=None,
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_http_handler)

        assert _status(result) == 503

    def test_dimensions_default_rubrics_none(self, handler, mock_http_handler):
        """Handles None DEFAULT_RUBRICS gracefully."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockEvaluationDimension,
            DEFAULT_RUBRICS=None,
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_http_handler)

        assert _status(result) == 200
        data = _body(result)
        dim = data["dimensions"][0]
        assert dim["description"] == ""


# ============================================================================
# GET /api/v1/evaluate/profiles Tests
# ============================================================================


class TestListProfiles:
    """Tests for list profiles endpoint."""

    def test_profiles_success(self, handler, mock_http_handler):
        """Returns profiles when judge is available."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES=MOCK_WEIGHT_PROFILES,
            DEFAULT_WEIGHTS=MOCK_DEFAULT_WEIGHTS,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        assert _status(result) == 200
        data = _body(result)
        assert "profiles" in data
        # default + factual_qa + creative_writing = 3
        assert len(data["profiles"]) == 3

    def test_profiles_includes_default(self, handler, mock_http_handler):
        """First profile is always 'default'."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES=MOCK_WEIGHT_PROFILES,
            DEFAULT_WEIGHTS=MOCK_DEFAULT_WEIGHTS,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        data = _body(result)
        default_profile = data["profiles"][0]
        assert default_profile["id"] == "default"
        assert default_profile["name"] == "Default"
        assert "Balanced" in default_profile["description"]

    def test_profiles_default_weights(self, handler, mock_http_handler):
        """Default profile has correct weights."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES=MOCK_WEIGHT_PROFILES,
            DEFAULT_WEIGHTS=MOCK_DEFAULT_WEIGHTS,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        data = _body(result)
        weights = data["profiles"][0]["weights"]
        assert weights["accuracy"] == 0.3
        assert weights["relevance"] == 0.3

    def test_profiles_named_profiles(self, handler, mock_http_handler):
        """Named profiles have correct structure."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES=MOCK_WEIGHT_PROFILES,
            DEFAULT_WEIGHTS=MOCK_DEFAULT_WEIGHTS,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        data = _body(result)
        profile_ids = [p["id"] for p in data["profiles"]]
        assert "factual_qa" in profile_ids
        assert "creative_writing" in profile_ids

    def test_profiles_name_formatting(self, handler, mock_http_handler):
        """Profile names are title-cased with underscores replaced."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES=MOCK_WEIGHT_PROFILES,
            DEFAULT_WEIGHTS=MOCK_DEFAULT_WEIGHTS,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        data = _body(result)
        names = [p["name"] for p in data["profiles"]]
        assert "Factual Qa" in names
        assert "Creative Writing" in names

    def test_profiles_descriptions(self, handler, mock_http_handler):
        """Known profiles have descriptions."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES=MOCK_WEIGHT_PROFILES,
            DEFAULT_WEIGHTS=MOCK_DEFAULT_WEIGHTS,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        data = _body(result)
        factual_qa = next(p for p in data["profiles"] if p["id"] == "factual_qa")
        assert "accuracy" in factual_qa["description"].lower()

    def test_profiles_unknown_profile_empty_description(self, handler, mock_http_handler):
        """Unknown profile IDs get empty description."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES={"custom_unknown": MOCK_DEFAULT_WEIGHTS},
            DEFAULT_WEIGHTS=MOCK_DEFAULT_WEIGHTS,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        data = _body(result)
        custom = next(p for p in data["profiles"] if p["id"] == "custom_unknown")
        assert custom["description"] == ""

    def test_profiles_judge_not_available(self, handler, mock_http_handler):
        """Returns 503 when judge is not available."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=False,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        assert _status(result) == 503

    def test_profiles_weight_profiles_none(self, handler, mock_http_handler):
        """Returns 503 when WEIGHT_PROFILES is None."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES=None,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        assert _status(result) == 503

    def test_profiles_default_weights_none(self, handler, mock_http_handler):
        """Handles None DEFAULT_WEIGHTS gracefully."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES=MOCK_WEIGHT_PROFILES,
            DEFAULT_WEIGHTS=None,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        assert _status(result) == 200
        data = _body(result)
        default_profile = data["profiles"][0]
        assert default_profile["weights"] == {}


# ============================================================================
# GET handle() Routing Tests
# ============================================================================


class TestHandleGetRouting:
    """Tests for GET request routing via handle()."""

    def test_handle_routes_to_dimensions(self, handler, mock_http_handler):
        """handle() routes dimensions path correctly."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockEvaluationDimension,
            DEFAULT_RUBRICS=MOCK_DEFAULT_RUBRICS,
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_http_handler)
        assert _status(result) == 200

    def test_handle_routes_to_profiles(self, handler, mock_http_handler):
        """handle() routes profiles path correctly."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES=MOCK_WEIGHT_PROFILES,
            DEFAULT_WEIGHTS=MOCK_DEFAULT_WEIGHTS,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)
        assert _status(result) == 200

    def test_handle_returns_none_for_unknown(self, handler, mock_http_handler):
        """handle() returns None for unmatched paths."""
        result = handler.handle("/api/v1/evaluate/unknown", {}, mock_http_handler)
        assert result is None

    def test_handle_returns_none_for_evaluate_root(self, handler, mock_http_handler):
        """handle() returns None for /api/v1/evaluate (POST-only)."""
        result = handler.handle("/api/v1/evaluate", {}, mock_http_handler)
        assert result is None

    def test_handle_returns_none_for_compare(self, handler, mock_http_handler):
        """handle() returns None for /api/v1/evaluate/compare (POST-only)."""
        result = handler.handle("/api/v1/evaluate/compare", {}, mock_http_handler)
        assert result is None


# ============================================================================
# POST /api/v1/evaluate Tests
# ============================================================================


class TestEvaluateResponse:
    """Tests for evaluate response endpoint."""

    @pytest.mark.asyncio
    async def test_evaluate_success(self, handler):
        """Successful evaluation returns 200 with result."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "overall_score": 4.2,
            "passed": True,
            "dimension_scores": {"accuracy": 4.5, "relevance": 3.9},
        }

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response": "Python is a programming language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 200
        data = _body(result)
        assert data["overall_score"] == 4.2
        assert data["passed"] is True

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self, handler):
        """Evaluation passes context to judge."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response": "A language.",
            "context": "Programming languages discussion",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 200
        mock_judge.evaluate.assert_awaited_once()
        call_kwargs = mock_judge.evaluate.call_args.kwargs
        assert call_kwargs["context"] == "Programming languages discussion"

    @pytest.mark.asyncio
    async def test_evaluate_with_reference(self, handler):
        """Evaluation passes reference answer to judge."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 3.5}

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What is 2+2?",
            "response": "4",
            "reference": "The answer is 4.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 200
        call_kwargs = mock_judge.evaluate.call_args.kwargs
        assert call_kwargs["reference"] == "The answer is 4."

    @pytest.mark.asyncio
    async def test_evaluate_with_use_case(self, handler):
        """Evaluation uses specified use case profile."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response": "A language.",
            "use_case": "factual_qa",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 200
        mock_config_cls.assert_called_once()
        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["use_case"] == "factual_qa"

    @pytest.mark.asyncio
    async def test_evaluate_with_threshold(self, handler):
        """Evaluation uses custom threshold."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response": "A language.",
            "threshold": 4.0,
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 200
        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["pass_threshold"] == 4.0

    @pytest.mark.asyncio
    async def test_evaluate_with_dimensions(self, handler):
        """Evaluation filters by specified dimensions."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response": "A language.",
            "dimensions": ["accuracy", "relevance"],
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 200
        call_kwargs = mock_config_cls.call_args.kwargs
        assert len(call_kwargs["dimensions"]) == 2

    @pytest.mark.asyncio
    async def test_evaluate_invalid_dimension(self, handler):
        """Returns 400 for invalid dimension value."""
        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response": "A language.",
            "dimensions": ["nonexistent_dimension"],
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 400
        assert "dimension" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_missing_query(self, handler):
        """Returns 400 when query is missing."""
        http_handler = _make_http_handler({
            "response": "A language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 400
        assert "query" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_missing_response(self, handler):
        """Returns 400 when response is missing."""
        http_handler = _make_http_handler({
            "query": "What is Python?",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 400
        assert "response" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_empty_query(self, handler):
        """Returns 400 when query is empty string."""
        http_handler = _make_http_handler({
            "query": "",
            "response": "A language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_evaluate_empty_response(self, handler):
        """Returns 400 when response is empty string."""
        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response": "",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_evaluate_invalid_json_body(self, handler):
        """Returns 400 for invalid JSON body."""
        http_handler = MagicMock()
        http_handler.client_address = ("127.0.0.1", 12345)
        body_bytes = b"not valid json"
        http_handler.headers = {"Content-Length": str(len(body_bytes))}
        http_handler.rfile = io.BytesIO(body_bytes)

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_evaluate_empty_body(self, handler):
        """Returns 400 for empty body (missing required fields)."""
        http_handler = _make_http_handler()

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_evaluate_judge_not_available(self, handler):
        """Returns 503 when judge is not available."""
        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response": "A language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=False,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_evaluate_llm_judge_none(self, handler):
        """Returns 503 when LLMJudge is None."""
        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response": "A language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=None,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_evaluate_judge_config_none(self, handler):
        """Returns 503 when JudgeConfig is None."""
        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response": "A language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=None,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_evaluate_default_use_case(self, handler):
        """Default use_case is 'default' when not specified."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What?",
            "response": "Answer.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockEvaluationDimension,
        ):
            await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["use_case"] == "default"

    @pytest.mark.asyncio
    async def test_evaluate_default_threshold(self, handler):
        """Default threshold is 3.5 when not specified."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What?",
            "response": "Answer.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockEvaluationDimension,
        ):
            await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["pass_threshold"] == 3.5


# ============================================================================
# POST /api/v1/evaluate/compare Tests
# ============================================================================


class TestCompareResponses:
    """Tests for compare responses endpoint."""

    @pytest.mark.asyncio
    async def test_compare_success(self, handler):
        """Successful comparison returns 200."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "winner": "A",
            "score_a": 4.2,
            "score_b": 3.8,
        }

        mock_judge = MagicMock()
        mock_judge.compare = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response_a": "Python is a programming language.",
            "response_b": "Python is a snake.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 200
        data = _body(result)
        assert data["winner"] == "A"

    @pytest.mark.asyncio
    async def test_compare_with_context(self, handler):
        """Comparison passes context to judge."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"winner": "B"}

        mock_judge = MagicMock()
        mock_judge.compare = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response_a": "A language.",
            "response_b": "A great language.",
            "context": "Programming discussion",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 200
        call_kwargs = mock_judge.compare.call_args.kwargs
        assert call_kwargs["context"] == "Programming discussion"

    @pytest.mark.asyncio
    async def test_compare_with_custom_ids(self, handler):
        """Comparison uses custom response IDs."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"winner": "model_x"}

        mock_judge = MagicMock()
        mock_judge.compare = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response_a": "A language.",
            "response_b": "A great language.",
            "response_a_id": "model_x",
            "response_b_id": "model_y",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 200
        call_kwargs = mock_judge.compare.call_args.kwargs
        assert call_kwargs["response_a_id"] == "model_x"
        assert call_kwargs["response_b_id"] == "model_y"

    @pytest.mark.asyncio
    async def test_compare_default_ids(self, handler):
        """Default response IDs are 'A' and 'B'."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"winner": "A"}

        mock_judge = MagicMock()
        mock_judge.compare = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response_a": "A language.",
            "response_b": "A great language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 200
        call_kwargs = mock_judge.compare.call_args.kwargs
        assert call_kwargs["response_a_id"] == "A"
        assert call_kwargs["response_b_id"] == "B"

    @pytest.mark.asyncio
    async def test_compare_with_use_case(self, handler):
        """Comparison uses specified use case."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"winner": "A"}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.compare = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response_a": "A language.",
            "response_b": "A great language.",
            "use_case": "code_generation",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 200
        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["use_case"] == "code_generation"

    @pytest.mark.asyncio
    async def test_compare_missing_query(self, handler):
        """Returns 400 when query is missing."""
        http_handler = _make_http_handler({
            "response_a": "A language.",
            "response_b": "A great language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 400
        assert "query" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_compare_missing_response_a(self, handler):
        """Returns 400 when response_a is missing."""
        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response_b": "A great language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 400
        assert "response_a" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_compare_missing_response_b(self, handler):
        """Returns 400 when response_b is missing."""
        http_handler = _make_http_handler({
            "query": "What is Python?",
            "response_a": "A language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 400
        assert "response_b" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_compare_empty_query(self, handler):
        """Returns 400 when query is empty."""
        http_handler = _make_http_handler({
            "query": "",
            "response_a": "A language.",
            "response_b": "A great language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_compare_empty_response_a(self, handler):
        """Returns 400 when response_a is empty."""
        http_handler = _make_http_handler({
            "query": "What?",
            "response_a": "",
            "response_b": "A great language.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_compare_empty_response_b(self, handler):
        """Returns 400 when response_b is empty."""
        http_handler = _make_http_handler({
            "query": "What?",
            "response_a": "A language.",
            "response_b": "",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_compare_invalid_json(self, handler):
        """Returns 400 for invalid JSON body."""
        http_handler = MagicMock()
        http_handler.client_address = ("127.0.0.1", 12345)
        body_bytes = b"not valid json"
        http_handler.headers = {"Content-Length": str(len(body_bytes))}
        http_handler.rfile = io.BytesIO(body_bytes)

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_compare_judge_not_available(self, handler):
        """Returns 503 when judge is not available."""
        http_handler = _make_http_handler({
            "query": "What?",
            "response_a": "A.",
            "response_b": "B.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=False,
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 503


# ============================================================================
# POST handle_post() Routing Tests
# ============================================================================


class TestHandlePostRouting:
    """Tests for POST request routing via handle_post()."""

    @pytest.mark.asyncio
    async def test_handle_post_routes_to_evaluate(self, handler):
        """handle_post() routes /evaluate path correctly."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What?",
            "response": "Answer.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_post_routes_to_compare(self, handler):
        """handle_post() routes /compare path correctly."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"winner": "A"}

        mock_judge = MagicMock()
        mock_judge.compare = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What?",
            "response_a": "A.",
            "response_b": "B.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_post_returns_none_for_unknown(self, handler):
        """handle_post() returns None for unmatched POST paths."""
        http_handler = _make_http_handler({"data": "test"})

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/unknown", {}, http_handler)

        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_returns_none_for_dimensions(self, handler):
        """handle_post() returns None for GET-only dimensions path."""
        http_handler = _make_http_handler({"data": "test"})

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/dimensions", {}, http_handler)

        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_returns_none_for_profiles(self, handler):
        """handle_post() returns None for GET-only profiles path."""
        http_handler = _make_http_handler({"data": "test"})

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/profiles", {}, http_handler)

        assert result is None


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting on evaluation endpoints."""

    def test_get_rate_limited(self, handler):
        """GET requests are rate limited."""
        import aragora.server.handlers.evaluation as eval_mod

        mock = _make_http_handler()

        with patch.object(eval_mod._evaluation_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock)

        assert _status(result) == 429
        assert "rate limit" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_post_rate_limited(self, handler):
        """POST requests are rate limited."""
        import aragora.server.handlers.evaluation as eval_mod

        mock = _make_http_handler({"query": "test", "response": "test"})

        with patch.object(eval_mod._evaluation_limiter, "is_allowed", return_value=False):
            result = await handler.handle_post("/api/v1/evaluate", {}, mock)

        assert _status(result) == 429

    def test_rate_limiter_rpm(self):
        """Rate limiter is configured with 30 RPM."""
        import aragora.server.handlers.evaluation as eval_mod

        assert eval_mod._evaluation_limiter.rpm == 30

    def test_rate_limit_allowed(self, handler, mock_http_handler):
        """Requests pass when under rate limit."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockEvaluationDimension,
            DEFAULT_RUBRICS=MOCK_DEFAULT_RUBRICS,
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_http_handler)
        assert _status(result) == 200


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling via @handle_errors decorator."""

    @pytest.mark.asyncio
    async def test_evaluate_exception_handled(self, handler):
        """Exceptions in evaluate are caught by @handle_errors."""
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(side_effect=RuntimeError("LLM API error"))

        http_handler = _make_http_handler({
            "query": "What?",
            "response": "Answer.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_compare_exception_handled(self, handler):
        """Exceptions in compare are caught by @handle_errors."""
        mock_judge = MagicMock()
        mock_judge.compare = AsyncMock(side_effect=RuntimeError("LLM API error"))

        http_handler = _make_http_handler({
            "query": "What?",
            "response_a": "A.",
            "response_b": "B.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        assert _status(result) == 500


# ============================================================================
# Security Tests
# ============================================================================


class TestSecurity:
    """Security tests for evaluation endpoints."""

    def test_path_traversal_rejected(self, handler):
        """Path traversal attempts are rejected."""
        assert not handler.can_handle("/api/v1/evaluate/../../../etc/passwd")

    def test_path_traversal_encoded_rejected(self, handler):
        """Encoded path traversal rejected."""
        assert not handler.can_handle("/api/v1/evaluate%2F..%2F..%2Fetc%2Fpasswd")

    @pytest.mark.asyncio
    async def test_script_injection_in_query(self, handler):
        """Script injection in query field is handled safely."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "<script>alert('xss')</script>",
            "response": "Answer.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        # Should process normally - input validation is downstream
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_sql_injection_in_query(self, handler):
        """SQL injection attempts in query field are handled safely."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "'; DROP TABLE evaluations; --",
            "response": "Answer.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
            EvaluationDimension=MockEvaluationDimension,
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_oversized_body_rejected(self, handler):
        """Very large body is rejected."""
        # Create handler with body larger than MAX_BODY_SIZE
        large_data = {"query": "x" * 20_000_000, "response": "y"}
        body_bytes = json.dumps(large_data).encode("utf-8")

        http_handler = MagicMock()
        http_handler.client_address = ("127.0.0.1", 12345)
        http_handler.headers = {"Content-Length": str(len(body_bytes))}
        http_handler.rfile = io.BytesIO(body_bytes)

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(),
            JudgeConfig=MagicMock(),
        ):
            result = await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        assert _status(result) == 400

    def test_null_handler_rate_limit(self, handler):
        """Rate limiter handles None handler gracefully."""
        # get_client_ip returns "unknown" for None handler
        result = handler.handle("/api/v1/evaluate/dimensions", {}, None)
        # Should still work (either rate limited or processed)
        assert result is not None


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_evaluate_no_dimensions_field(self, handler):
        """Evaluation without dimensions field uses all dimensions."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What?",
            "response": "Answer.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockEvaluationDimension,
        ):
            await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["dimensions"] is None

    @pytest.mark.asyncio
    async def test_evaluate_empty_dimensions_list(self, handler):
        """Evaluation with empty dimensions list uses all dimensions."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What?",
            "response": "Answer.",
            "dimensions": [],
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockEvaluationDimension,
        ):
            await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        call_kwargs = mock_config_cls.call_args.kwargs
        # Empty list is falsy, so dimensions should be None
        assert call_kwargs["dimensions"] is None

    @pytest.mark.asyncio
    async def test_evaluate_threshold_string_conversion(self, handler):
        """Threshold is converted from string to float."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What?",
            "response": "Answer.",
            "threshold": "4.5",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockEvaluationDimension,
        ):
            await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["pass_threshold"] == 4.5
        assert isinstance(call_kwargs["pass_threshold"], float)

    def test_profiles_empty_weight_profiles(self, handler, mock_http_handler):
        """Empty weight profiles dict is falsy, returns 503."""
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES={},
            DEFAULT_WEIGHTS=MOCK_DEFAULT_WEIGHTS,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_http_handler)

        # Empty dict {} is falsy, so `not WEIGHT_PROFILES` is True => 503
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_compare_default_use_case(self, handler):
        """Default use_case for compare is 'default'."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"winner": "A"}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.compare = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What?",
            "response_a": "A.",
            "response_b": "B.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
        ):
            await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["use_case"] == "default"

    @pytest.mark.asyncio
    async def test_evaluate_dimensions_with_evaluation_dimension_none(self, handler):
        """Dimensions parsing skipped when EvaluationDimension is None."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)
        mock_judge_cls.return_value = mock_judge

        mock_config_cls = MagicMock()

        http_handler = _make_http_handler({
            "query": "What?",
            "response": "Answer.",
            "dimensions": ["accuracy"],
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=None,
        ):
            await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        call_kwargs = mock_config_cls.call_args.kwargs
        # With EvaluationDimension=None, dimensions should be None (skipped)
        assert call_kwargs["dimensions"] is None

    @pytest.mark.asyncio
    async def test_evaluate_context_and_reference_passed_as_none(self, handler):
        """Context and reference default to None when not provided."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"overall_score": 4.0}

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What?",
            "response": "Answer.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
            EvaluationDimension=MockEvaluationDimension,
        ):
            await handler.handle_post("/api/v1/evaluate", {}, http_handler)

        call_kwargs = mock_judge.evaluate.call_args.kwargs
        assert call_kwargs["context"] is None
        assert call_kwargs["reference"] is None

    @pytest.mark.asyncio
    async def test_compare_context_passed_as_none(self, handler):
        """Context defaults to None when not provided in compare."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"winner": "A"}

        mock_judge = MagicMock()
        mock_judge.compare = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler({
            "query": "What?",
            "response_a": "A.",
            "response_b": "B.",
        })

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=MagicMock(return_value=mock_judge),
            JudgeConfig=MagicMock(),
        ):
            await handler.handle_post("/api/v1/evaluate/compare", {}, http_handler)

        call_kwargs = mock_judge.compare.call_args.kwargs
        assert call_kwargs["context"] is None
