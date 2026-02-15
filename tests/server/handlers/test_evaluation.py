"""
Tests for aragora.server.handlers.evaluation - Evaluation HTTP Handlers.

Tests cover:
- EvaluationHandler: instantiation, ROUTES, can_handle
- GET /api/v1/evaluate/dimensions: success, judge unavailable
- GET /api/v1/evaluate/profiles: success, judge unavailable
- handle routing: rate limiting, returns None for unmatched paths
- handle_post routing: returns None for unmatched paths
- POST /api/v1/evaluate: body validation (missing query, missing response)
- POST /api/v1/evaluate/compare: body validation
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.evaluation import EvaluationHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Mock Evaluation Objects
# ===========================================================================


class MockDimension(Enum):
    """Mock evaluation dimension."""

    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"


class MockRubric:
    """Mock rubric for a dimension."""

    def __init__(self, description: str = "Test rubric"):
        self.description = description
        self.score_1 = "Poor"
        self.score_2 = "Fair"
        self.score_3 = "Good"
        self.score_4 = "Very Good"
        self.score_5 = "Excellent"


class MockEvalResult:
    """Mock evaluation result."""

    def __init__(self):
        self.scores = {"accuracy": 4.2, "clarity": 3.8}
        self.overall_score = 4.0
        self.passed = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "overall_score": self.overall_score,
            "passed": self.passed,
        }


class MockCompareResult:
    """Mock comparison result."""

    def __init__(self):
        self.winner = "A"
        self.confidence = 0.85

    def to_dict(self) -> dict[str, Any]:
        return {"winner": self.winner, "confidence": self.confidence}


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    from aragora.server.handlers.evaluation import _evaluation_limiter

    _evaluation_limiter._buckets.clear()


@pytest.fixture
def handler():
    """Create an EvaluationHandler."""
    return EvaluationHandler(ctx={})


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestEvaluationHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, EvaluationHandler)

    def test_routes(self, handler):
        assert "/api/v1/evaluate" in handler.ROUTES
        assert "/api/v1/evaluate/compare" in handler.ROUTES
        assert "/api/v1/evaluate/dimensions" in handler.ROUTES
        assert "/api/v1/evaluate/profiles" in handler.ROUTES

    def test_can_handle_evaluate(self, handler):
        assert handler.can_handle("/api/v1/evaluate") is True

    def test_can_handle_compare(self, handler):
        assert handler.can_handle("/api/v1/evaluate/compare") is True

    def test_can_handle_dimensions(self, handler):
        assert handler.can_handle("/api/v1/evaluate/dimensions") is True

    def test_can_handle_profiles(self, handler):
        assert handler.can_handle("/api/v1/evaluate/profiles") is True

    def test_cannot_handle_other_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_default_context(self):
        h = EvaluationHandler()
        assert h.ctx == {}


# ===========================================================================
# Test GET /api/v1/evaluate/dimensions
# ===========================================================================


class TestListDimensions:
    """Tests for the dimensions endpoint."""

    def test_list_dimensions_success(self, handler):
        mock_rubrics = {
            MockDimension.ACCURACY: MockRubric("Accuracy rubric"),
            MockDimension.CLARITY: MockRubric("Clarity rubric"),
        }
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockDimension,
            DEFAULT_RUBRICS=mock_rubrics,
        ):
            result = handler._list_dimensions()
            assert result.status_code == 200
            data = _parse_body(result)
            assert "dimensions" in data
            assert len(data["dimensions"]) == 3  # All enum members
            # Check first dimension structure
            dim = data["dimensions"][0]
            assert "id" in dim
            assert "name" in dim
            assert "rubric" in dim

    def test_list_dimensions_judge_unavailable(self, handler):
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=False,
            EvaluationDimension=None,
        ):
            result = handler._list_dimensions()
            assert result.status_code == 503

    def test_list_dimensions_no_rubric(self, handler):
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockDimension,
            DEFAULT_RUBRICS={},
        ):
            result = handler._list_dimensions()
            assert result.status_code == 200
            data = _parse_body(result)
            dim = data["dimensions"][0]
            assert dim["description"] == ""


# ===========================================================================
# Test GET /api/v1/evaluate/profiles
# ===========================================================================


class TestListProfiles:
    """Tests for the profiles endpoint."""

    def test_list_profiles_success(self, handler):
        mock_weights = {
            MockDimension.ACCURACY: 0.5,
            MockDimension.CLARITY: 0.5,
        }
        mock_profiles = {
            "factual_qa": mock_weights,
            "creative_writing": mock_weights,
        }
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES=mock_profiles,
            DEFAULT_WEIGHTS=mock_weights,
        ):
            result = handler._list_profiles()
            assert result.status_code == 200
            data = _parse_body(result)
            assert "profiles" in data
            # Default + 2 custom profiles
            assert len(data["profiles"]) == 3
            # Check default profile
            default = data["profiles"][0]
            assert default["id"] == "default"
            assert default["name"] == "Default"

    def test_list_profiles_judge_unavailable(self, handler):
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=False,
            WEIGHT_PROFILES=None,
        ):
            result = handler._list_profiles()
            assert result.status_code == 503


# ===========================================================================
# Test handle() Routing (GET)
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    def test_handle_dimensions(self, handler):
        mock_handler = _make_mock_handler()
        mock_rubrics = {
            MockDimension.ACCURACY: MockRubric(),
        }
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            EvaluationDimension=MockDimension,
            DEFAULT_RUBRICS=mock_rubrics,
        ):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_profiles(self, handler):
        mock_handler = _make_mock_handler()
        mock_weights = {MockDimension.ACCURACY: 0.5}
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            WEIGHT_PROFILES={"test": mock_weights},
            DEFAULT_WEIGHTS=mock_weights,
        ):
            result = handler.handle("/api/v1/evaluate/profiles", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/v1/evaluate/unknown", {}, mock_handler)
        assert result is None

    def test_handle_rate_limited(self, handler):
        from aragora.server.handlers.evaluation import _evaluation_limiter

        mock_handler = _make_mock_handler()
        with patch.object(_evaluation_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/v1/evaluate/dimensions", {}, mock_handler)
            assert result.status_code == 429


# ===========================================================================
# Test POST /api/v1/evaluate (body validation)
# ===========================================================================


class TestEvaluateResponse:
    """Tests for the evaluate response endpoint."""

    @pytest.mark.asyncio
    async def test_evaluate_judge_unavailable(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=False,
            LLMJudge=None,
            JudgeConfig=None,
        ):
            result = await handler._evaluate_response(mock_handler)
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_evaluate_missing_query(self, handler):
        body = json.dumps({"response": "Some response"}).encode()
        mock_handler = _make_mock_handler("POST", body)
        mock_config_cls = MagicMock()
        mock_judge_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockDimension,
        ):
            with patch.object(
                handler, "read_json_body", return_value={"response": "Some response"}
            ):
                result = await handler._evaluate_response(mock_handler)
                assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_evaluate_missing_response(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_config_cls = MagicMock()
        mock_judge_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockDimension,
        ):
            with patch.object(handler, "read_json_body", return_value={"query": "Test question"}):
                result = await handler._evaluate_response(mock_handler)
                assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_evaluate_invalid_body(self, handler):
        mock_handler = _make_mock_handler("POST", b"not json")
        mock_config_cls = MagicMock()
        mock_judge_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
        ):
            with patch.object(handler, "read_json_body", return_value=None):
                result = await handler._evaluate_response(mock_handler)
                assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_evaluate_success(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_eval_result = MockEvalResult()
        mock_judge = AsyncMock()
        mock_judge.evaluate.return_value = mock_eval_result
        mock_judge_cls = MagicMock(return_value=mock_judge)
        mock_config_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockDimension,
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"query": "What is 2+2?", "response": "4"},
            ):
                result = await handler._evaluate_response(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["overall_score"] == 4.0
                assert data["passed"] is True


# ===========================================================================
# Test POST /api/v1/evaluate/compare (body validation)
# ===========================================================================


class TestCompareResponses:
    """Tests for the compare responses endpoint."""

    @pytest.mark.asyncio
    async def test_compare_judge_unavailable(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=False,
            LLMJudge=None,
            JudgeConfig=None,
        ):
            result = await handler._compare_responses(mock_handler)
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_compare_missing_query(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_judge_cls = MagicMock()
        mock_config_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"response_a": "A", "response_b": "B"},
            ):
                result = await handler._compare_responses(mock_handler)
                assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_compare_missing_response_a(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_judge_cls = MagicMock()
        mock_config_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"query": "Q", "response_b": "B"},
            ):
                result = await handler._compare_responses(mock_handler)
                assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_compare_success(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_compare_result = MockCompareResult()
        mock_judge = AsyncMock()
        mock_judge.compare.return_value = mock_compare_result
        mock_judge_cls = MagicMock(return_value=mock_judge)
        mock_config_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={
                    "query": "Compare these",
                    "response_a": "Answer A",
                    "response_b": "Answer B",
                },
            ):
                result = await handler._compare_responses(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["winner"] == "A"
                assert data["confidence"] == 0.85


# ===========================================================================
# Test handle_post() Routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for the handle_post() method routing."""

    @pytest.mark.asyncio
    async def test_handle_post_evaluate(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_eval_result = MockEvalResult()
        mock_judge = AsyncMock()
        mock_judge.evaluate.return_value = mock_eval_result
        mock_judge_cls = MagicMock(return_value=mock_judge)
        mock_config_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.evaluation",
            JUDGE_AVAILABLE=True,
            LLMJudge=mock_judge_cls,
            JudgeConfig=mock_config_cls,
            EvaluationDimension=MockDimension,
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"query": "Q", "response": "R"},
            ):
                result = await handler.handle_post("/api/v1/evaluate", {}, mock_handler)
                assert result is not None
                assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler("POST")
        result = await handler.handle_post("/api/v1/evaluate/unknown", {}, mock_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_rate_limited(self, handler):
        from aragora.server.handlers.evaluation import _evaluation_limiter

        mock_handler = _make_mock_handler("POST")
        with patch.object(_evaluation_limiter, "is_allowed", return_value=False):
            result = await handler.handle_post("/api/v1/evaluate", {}, mock_handler)
            assert result.status_code == 429
