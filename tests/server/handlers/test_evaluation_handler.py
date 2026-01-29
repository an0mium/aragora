"""
Tests for EvaluationHandler - LLM-as-Judge evaluation endpoints.

Tests cover:
- Route matching (can_handle)
- RBAC permission enforcement
- Input validation
- Happy path operations
- Error handling
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.evaluation import EvaluationHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockDimensionScore:
    """Mock dimension score for testing."""

    dimension: str = "relevance"
    score: float = 4.0
    confidence: float = 0.85
    feedback: str = "Good relevance to the query"
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "confidence": self.confidence,
            "feedback": self.feedback,
            "examples": self.examples,
        }


@dataclass
class MockEvaluationResult:
    """Mock evaluation result for testing."""

    id: str = "eval-001"
    response_id: str = "resp-001"
    overall_score: float = 4.2
    overall_confidence: float = 0.85
    judge_model: str = "claude-sonnet-4-20250514"
    use_case: str = "default"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str = "Good quality response overall"
    strengths: list[str] = field(default_factory=lambda: ["Clear explanation", "Accurate"])
    weaknesses: list[str] = field(default_factory=lambda: ["Could be more concise"])
    suggestions: list[str] = field(default_factory=lambda: ["Add examples"])
    passes_threshold: bool = True
    threshold_used: float = 3.5
    dimension_scores: dict[str, MockDimensionScore] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "response_id": self.response_id,
            "overall_score": self.overall_score,
            "overall_confidence": self.overall_confidence,
            "judge_model": self.judge_model,
            "use_case": self.use_case,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "passes_threshold": self.passes_threshold,
            "threshold_used": self.threshold_used,
            "dimension_scores": {k: v.to_dict() for k, v in self.dimension_scores.items()},
            "weights_used": self.weights_used,
        }


@dataclass
class MockPairwiseResult:
    """Mock pairwise comparison result for testing."""

    id: str = "compare-001"
    response_a_id: str = "A"
    response_b_id: str = "B"
    winner: str = "A"
    confidence: float = 0.75
    dimension_preferences: dict[str, str] = field(
        default_factory=lambda: {"relevance": "A", "clarity": "tie", "accuracy": "A"}
    )
    explanation: str = "Response A is more comprehensive and better addresses the query."
    judge_model: str = "claude-sonnet-4-20250514"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "response_a_id": self.response_a_id,
            "response_b_id": self.response_b_id,
            "winner": self.winner,
            "confidence": self.confidence,
            "dimension_preferences": self.dimension_preferences,
            "explanation": self.explanation,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp.isoformat(),
        }


class MockEvaluationDimension:
    """Mock evaluation dimension enum."""

    def __init__(self, value: str):
        self.value = value
        self.name = value.upper()


class MockEvaluationRubric:
    """Mock evaluation rubric for testing."""

    def __init__(self):
        self.description = "How well does the response address the query?"
        self.score_1 = "Completely off-topic"
        self.score_2 = "Partially addresses the question"
        self.score_3 = "Addresses main points"
        self.score_4 = "Addresses all major aspects"
        self.score_5 = "Perfectly addresses the query"


class MockLLMJudge:
    """Mock LLM Judge for testing."""

    def __init__(self, config=None):
        self._config = config

    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> MockEvaluationResult:
        return MockEvaluationResult()

    async def compare(
        self,
        query: str,
        response_a: str,
        response_b: str,
        context: Optional[str] = None,
        response_a_id: Optional[str] = None,
        response_b_id: Optional[str] = None,
    ) -> MockPairwiseResult:
        return MockPairwiseResult(
            response_a_id=response_a_id or "A",
            response_b_id=response_b_id or "B",
        )


class MockJudgeConfig:
    """Mock judge config for testing."""

    def __init__(
        self,
        use_case: str = "default",
        pass_threshold: float = 3.5,
        dimensions: Optional[list] = None,
    ):
        self.use_case = use_case
        self.pass_threshold = pass_threshold
        self.dimensions = dimensions


def create_mock_handler(
    method: str = "GET",
    body: Optional[dict[str, Any]] = None,
    path: str = "/api/v1/evaluate",
) -> MagicMock:
    """Create a mock HTTP handler for testing."""
    mock = MagicMock()
    mock.command = method
    mock.path = path

    if body is not None:
        body_bytes = json.dumps(body).encode()
    else:
        body_bytes = b"{}"

    mock.rfile = MagicMock()
    mock.rfile.read = MagicMock(return_value=body_bytes)

    mock.headers = {"Content-Length": str(len(body_bytes))}
    mock.client_address = ("127.0.0.1", 12345)
    mock.user_context = MagicMock()
    mock.user_context.user_id = "test_user"

    return mock


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return MagicMock()


@pytest.fixture
def handler(mock_server_context):
    """Create handler with mocked dependencies."""
    h = EvaluationHandler(mock_server_context)
    return h


@pytest.fixture
def handler_with_mocked_judge(mock_server_context):
    """Create handler with mocked LLM Judge dependencies."""
    import aragora.server.handlers.evaluation as eval_module

    # Store original values
    original_judge_available = eval_module.JUDGE_AVAILABLE
    original_llm_judge = eval_module.LLMJudge
    original_judge_config = eval_module.JudgeConfig
    original_eval_dim = eval_module.EvaluationDimension
    original_default_rubrics = eval_module.DEFAULT_RUBRICS
    original_weight_profiles = eval_module.WEIGHT_PROFILES
    original_default_weights = eval_module.DEFAULT_WEIGHTS

    # Set up mocks
    mock_dimensions = [
        MockEvaluationDimension("relevance"),
        MockEvaluationDimension("accuracy"),
        MockEvaluationDimension("completeness"),
        MockEvaluationDimension("clarity"),
    ]

    # Create a mock enum that can be iterated
    class MockEnumMeta(type):
        def __iter__(cls):
            return iter(mock_dimensions)

    class MockEvaluationDimensionEnum(metaclass=MockEnumMeta):
        RELEVANCE = MockEvaluationDimension("relevance")
        ACCURACY = MockEvaluationDimension("accuracy")

        def __init__(self, value):
            self.value = value
            self.name = value.upper()

    mock_rubrics = {
        MockEvaluationDimensionEnum.RELEVANCE: MockEvaluationRubric(),
        MockEvaluationDimensionEnum.ACCURACY: MockEvaluationRubric(),
    }

    mock_weight_profiles = {
        "factual_qa": {MockEvaluationDimensionEnum.RELEVANCE: 0.3},
        "creative_writing": {MockEvaluationDimensionEnum.RELEVANCE: 0.15},
    }

    mock_default_weights = {MockEvaluationDimensionEnum.RELEVANCE: 0.2}

    # Apply mocks
    eval_module.JUDGE_AVAILABLE = True
    eval_module.LLMJudge = MockLLMJudge
    eval_module.JudgeConfig = MockJudgeConfig
    eval_module.EvaluationDimension = MockEvaluationDimensionEnum
    eval_module.DEFAULT_RUBRICS = mock_rubrics
    eval_module.WEIGHT_PROFILES = mock_weight_profiles
    eval_module.DEFAULT_WEIGHTS = mock_default_weights

    h = EvaluationHandler(mock_server_context)

    yield h

    # Restore original values
    eval_module.JUDGE_AVAILABLE = original_judge_available
    eval_module.LLMJudge = original_llm_judge
    eval_module.JudgeConfig = original_judge_config
    eval_module.EvaluationDimension = original_eval_dim
    eval_module.DEFAULT_RUBRICS = original_default_rubrics
    eval_module.WEIGHT_PROFILES = original_weight_profiles
    eval_module.DEFAULT_WEIGHTS = original_default_weights


# ===========================================================================
# Route Matching Tests
# ===========================================================================


class TestEvaluationHandlerRouting:
    """Test request routing."""

    def test_can_handle_evaluate_path(self, handler):
        """Test that handler recognizes evaluate path."""
        assert handler.can_handle("/api/v1/evaluate")

    def test_can_handle_compare_path(self, handler):
        """Test that handler recognizes compare path."""
        assert handler.can_handle("/api/v1/evaluate/compare")

    def test_can_handle_dimensions_path(self, handler):
        """Test that handler recognizes dimensions path."""
        assert handler.can_handle("/api/v1/evaluate/dimensions")

    def test_can_handle_profiles_path(self, handler):
        """Test that handler recognizes profiles path."""
        assert handler.can_handle("/api/v1/evaluate/profiles")

    def test_cannot_handle_other_paths(self, handler):
        """Test that handler rejects non-evaluation paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/backups")
        assert not handler.can_handle("/api/v2/evaluate")
        assert not handler.can_handle("/api/evaluate")

    def test_routes_constant(self, handler):
        """Test that ROUTES contains expected endpoints."""
        expected_routes = [
            "/api/v1/evaluate",
            "/api/v1/evaluate/compare",
            "/api/v1/evaluate/dimensions",
            "/api/v1/evaluate/profiles",
        ]
        for route in expected_routes:
            assert route in handler.ROUTES


# ===========================================================================
# RBAC Permission Tests
# ===========================================================================


class TestEvaluationHandlerRBAC:
    """Test RBAC permission enforcement."""

    @pytest.mark.no_auto_auth
    def test_list_dimensions_requires_evaluation_read(self, mock_server_context):
        """Test that listing dimensions requires evaluation:read permission."""
        from aragora.rbac.decorators import PermissionDeniedError

        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            h = EvaluationHandler(mock_server_context)
            mock_handler = create_mock_handler()

            # Without proper auth context, should raise PermissionDeniedError
            with pytest.raises(PermissionDeniedError):
                h.handle("/api/v1/evaluate/dimensions", {}, mock_handler)
        finally:
            del os.environ["ARAGORA_TEST_REAL_AUTH"]

    @pytest.mark.no_auto_auth
    def test_list_profiles_requires_evaluation_read(self, mock_server_context):
        """Test that listing profiles requires evaluation:read permission."""
        from aragora.rbac.decorators import PermissionDeniedError

        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            h = EvaluationHandler(mock_server_context)
            mock_handler = create_mock_handler()

            # Without proper auth context, should raise PermissionDeniedError
            with pytest.raises(PermissionDeniedError):
                h.handle("/api/v1/evaluate/profiles", {}, mock_handler)
        finally:
            del os.environ["ARAGORA_TEST_REAL_AUTH"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_evaluate_requires_evaluation_create(self, mock_server_context):
        """Test that evaluation requires evaluation:create permission."""
        from aragora.rbac.decorators import PermissionDeniedError

        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            h = EvaluationHandler(mock_server_context)
            mock_handler = create_mock_handler(
                method="POST",
                body={"query": "Test query", "response": "Test response"},
            )

            # Without proper auth context, should raise PermissionDeniedError
            with pytest.raises(PermissionDeniedError):
                await h.handle_post("/api/v1/evaluate", {}, mock_handler)
        finally:
            del os.environ["ARAGORA_TEST_REAL_AUTH"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_compare_requires_evaluation_create(self, mock_server_context):
        """Test that comparison requires evaluation:create permission."""
        from aragora.rbac.decorators import PermissionDeniedError

        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            h = EvaluationHandler(mock_server_context)
            mock_handler = create_mock_handler(
                method="POST",
                body={
                    "query": "Test query",
                    "response_a": "Response A",
                    "response_b": "Response B",
                },
            )

            # Without proper auth context, should raise PermissionDeniedError
            with pytest.raises(PermissionDeniedError):
                await h.handle_post("/api/v1/evaluate/compare", {}, mock_handler)
        finally:
            del os.environ["ARAGORA_TEST_REAL_AUTH"]


# ===========================================================================
# Input Validation Tests
# ===========================================================================


class TestEvaluationHandlerValidation:
    """Test input validation."""

    @pytest.mark.asyncio
    async def test_evaluate_missing_query(self, handler_with_mocked_judge):
        """Test evaluating without query returns 400."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"response": "Test response"},
        )

        result = await handler_with_mocked_judge.handle_post("/api/v1/evaluate", {}, mock_handler)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "query" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_evaluate_missing_response(self, handler_with_mocked_judge):
        """Test evaluating without response returns 400."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"query": "Test query"},
        )

        result = await handler_with_mocked_judge.handle_post("/api/v1/evaluate", {}, mock_handler)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "response" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_compare_missing_query(self, handler_with_mocked_judge):
        """Test comparing without query returns 400."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"response_a": "Response A", "response_b": "Response B"},
        )

        result = await handler_with_mocked_judge.handle_post(
            "/api/v1/evaluate/compare", {}, mock_handler
        )
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "query" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_compare_missing_response_a(self, handler_with_mocked_judge):
        """Test comparing without response_a returns 400."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"query": "Test query", "response_b": "Response B"},
        )

        result = await handler_with_mocked_judge.handle_post(
            "/api/v1/evaluate/compare", {}, mock_handler
        )
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "response_a" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_compare_missing_response_b(self, handler_with_mocked_judge):
        """Test comparing without response_b returns 400."""
        mock_handler = create_mock_handler(
            method="POST",
            body={"query": "Test query", "response_a": "Response A"},
        )

        result = await handler_with_mocked_judge.handle_post(
            "/api/v1/evaluate/compare", {}, mock_handler
        )
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "response_b" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_evaluate_invalid_json(self, handler_with_mocked_judge):
        """Test evaluating with invalid JSON returns 400."""
        mock_handler = create_mock_handler(method="POST")
        mock_handler.rfile.read.return_value = b"not valid json"

        result = await handler_with_mocked_judge.handle_post("/api/v1/evaluate", {}, mock_handler)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "json" in body.get("error", "").lower()


# ===========================================================================
# Happy Path Tests
# ===========================================================================


class TestListDimensions:
    """Test list dimensions endpoint."""

    def test_list_dimensions_success(self, handler_with_mocked_judge):
        """Test listing dimensions returns correct format."""
        mock_handler = create_mock_handler()

        result = handler_with_mocked_judge.handle("/api/v1/evaluate/dimensions", {}, mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "dimensions" in body
        assert isinstance(body["dimensions"], list)

    def test_list_dimensions_contains_expected_fields(self, handler_with_mocked_judge):
        """Test that each dimension has expected fields."""
        mock_handler = create_mock_handler()

        result = handler_with_mocked_judge.handle("/api/v1/evaluate/dimensions", {}, mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)

        for dim in body["dimensions"]:
            assert "id" in dim
            assert "name" in dim
            assert "description" in dim
            assert "rubric" in dim


class TestListProfiles:
    """Test list profiles endpoint."""

    def test_list_profiles_success(self, handler_with_mocked_judge):
        """Test listing profiles returns correct format."""
        mock_handler = create_mock_handler()

        result = handler_with_mocked_judge.handle("/api/v1/evaluate/profiles", {}, mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "profiles" in body
        assert isinstance(body["profiles"], list)

    def test_list_profiles_contains_default(self, handler_with_mocked_judge):
        """Test that profiles include default profile."""
        mock_handler = create_mock_handler()

        result = handler_with_mocked_judge.handle("/api/v1/evaluate/profiles", {}, mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)

        profile_ids = [p["id"] for p in body["profiles"]]
        assert "default" in profile_ids

    def test_list_profiles_contains_expected_fields(self, handler_with_mocked_judge):
        """Test that each profile has expected fields."""
        mock_handler = create_mock_handler()

        result = handler_with_mocked_judge.handle("/api/v1/evaluate/profiles", {}, mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)

        for profile in body["profiles"]:
            assert "id" in profile
            assert "name" in profile
            assert "description" in profile
            assert "weights" in profile


class TestEvaluateResponse:
    """Test evaluate response endpoint."""

    @pytest.mark.asyncio
    async def test_evaluate_success(self, handler_with_mocked_judge):
        """Test evaluating a response succeeds."""
        mock_handler = create_mock_handler(
            method="POST",
            body={
                "query": "What is the capital of France?",
                "response": "The capital of France is Paris.",
            },
        )

        result = await handler_with_mocked_judge.handle_post("/api/v1/evaluate", {}, mock_handler)
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "overall_score" in body
        assert "passes_threshold" in body

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self, handler_with_mocked_judge):
        """Test evaluating with context succeeds."""
        mock_handler = create_mock_handler(
            method="POST",
            body={
                "query": "Summarize the document",
                "response": "The document discusses climate change.",
                "context": "This is a scientific paper about global warming.",
            },
        )

        result = await handler_with_mocked_judge.handle_post("/api/v1/evaluate", {}, mock_handler)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_evaluate_with_reference(self, handler_with_mocked_judge):
        """Test evaluating with reference answer succeeds."""
        mock_handler = create_mock_handler(
            method="POST",
            body={
                "query": "What is 2 + 2?",
                "response": "The answer is 4.",
                "reference": "2 + 2 = 4",
            },
        )

        result = await handler_with_mocked_judge.handle_post("/api/v1/evaluate", {}, mock_handler)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_evaluate_with_use_case(self, handler_with_mocked_judge):
        """Test evaluating with specific use case profile."""
        mock_handler = create_mock_handler(
            method="POST",
            body={
                "query": "Write a poem",
                "response": "Roses are red...",
                "use_case": "creative_writing",
            },
        )

        result = await handler_with_mocked_judge.handle_post("/api/v1/evaluate", {}, mock_handler)
        assert result.status_code == 200


class TestCompareResponses:
    """Test compare responses endpoint."""

    @pytest.mark.asyncio
    async def test_compare_success(self, handler_with_mocked_judge):
        """Test comparing two responses succeeds."""
        mock_handler = create_mock_handler(
            method="POST",
            body={
                "query": "Explain quantum computing",
                "response_a": "Quantum computing uses qubits...",
                "response_b": "A quantum computer is a machine...",
            },
        )

        result = await handler_with_mocked_judge.handle_post(
            "/api/v1/evaluate/compare", {}, mock_handler
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "winner" in body
        assert "confidence" in body
        assert "explanation" in body

    @pytest.mark.asyncio
    async def test_compare_with_context(self, handler_with_mocked_judge):
        """Test comparing with context succeeds."""
        mock_handler = create_mock_handler(
            method="POST",
            body={
                "query": "Answer the question",
                "response_a": "Answer A",
                "response_b": "Answer B",
                "context": "Additional context here",
            },
        )

        result = await handler_with_mocked_judge.handle_post(
            "/api/v1/evaluate/compare", {}, mock_handler
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_compare_with_custom_ids(self, handler_with_mocked_judge):
        """Test comparing with custom response IDs."""
        mock_handler = create_mock_handler(
            method="POST",
            body={
                "query": "Test query",
                "response_a": "Response A content",
                "response_b": "Response B content",
                "response_a_id": "claude-response",
                "response_b_id": "gpt-response",
            },
        )

        result = await handler_with_mocked_judge.handle_post(
            "/api/v1/evaluate/compare", {}, mock_handler
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body.get("response_a_id") == "claude-response"
        assert body.get("response_b_id") == "gpt-response"


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestEvaluationHandlerErrors:
    """Test error handling."""

    def test_judge_unavailable_dimensions(self, mock_server_context):
        """Test handling when LLM Judge is unavailable for dimensions."""
        import aragora.server.handlers.evaluation as eval_module

        original = eval_module.JUDGE_AVAILABLE
        eval_module.JUDGE_AVAILABLE = False

        try:
            h = EvaluationHandler(mock_server_context)
            mock_handler = create_mock_handler()

            result = h.handle("/api/v1/evaluate/dimensions", {}, mock_handler)
            assert result.status_code == 503
            body = json.loads(result.body)
            assert "not available" in body.get("error", "").lower()
        finally:
            eval_module.JUDGE_AVAILABLE = original

    def test_judge_unavailable_profiles(self, mock_server_context):
        """Test handling when LLM Judge is unavailable for profiles."""
        import aragora.server.handlers.evaluation as eval_module

        original = eval_module.JUDGE_AVAILABLE
        eval_module.JUDGE_AVAILABLE = False

        try:
            h = EvaluationHandler(mock_server_context)
            mock_handler = create_mock_handler()

            result = h.handle("/api/v1/evaluate/profiles", {}, mock_handler)
            assert result.status_code == 503
            body = json.loads(result.body)
            assert "not available" in body.get("error", "").lower()
        finally:
            eval_module.JUDGE_AVAILABLE = original

    @pytest.mark.asyncio
    async def test_judge_unavailable_evaluate(self, mock_server_context):
        """Test handling when LLM Judge is unavailable for evaluation."""
        import aragora.server.handlers.evaluation as eval_module

        original = eval_module.JUDGE_AVAILABLE
        eval_module.JUDGE_AVAILABLE = False

        try:
            h = EvaluationHandler(mock_server_context)
            mock_handler = create_mock_handler(
                method="POST",
                body={"query": "Test", "response": "Test"},
            )

            result = await h.handle_post("/api/v1/evaluate", {}, mock_handler)
            assert result.status_code == 503
        finally:
            eval_module.JUDGE_AVAILABLE = original

    @pytest.mark.asyncio
    async def test_judge_unavailable_compare(self, mock_server_context):
        """Test handling when LLM Judge is unavailable for comparison."""
        import aragora.server.handlers.evaluation as eval_module

        original = eval_module.JUDGE_AVAILABLE
        eval_module.JUDGE_AVAILABLE = False

        try:
            h = EvaluationHandler(mock_server_context)
            mock_handler = create_mock_handler(
                method="POST",
                body={
                    "query": "Test",
                    "response_a": "A",
                    "response_b": "B",
                },
            )

            result = await h.handle_post("/api/v1/evaluate/compare", {}, mock_handler)
            assert result.status_code == 503
        finally:
            eval_module.JUDGE_AVAILABLE = original

    def test_error_response_format(self, mock_server_context):
        """Test that error responses have correct format."""
        import aragora.server.handlers.evaluation as eval_module

        original = eval_module.JUDGE_AVAILABLE
        eval_module.JUDGE_AVAILABLE = False

        try:
            h = EvaluationHandler(mock_server_context)
            mock_handler = create_mock_handler()

            result = h.handle("/api/v1/evaluate/dimensions", {}, mock_handler)
            assert result.status_code == 503
            body = json.loads(result.body)
            assert "error" in body
            assert isinstance(body["error"], str)
        finally:
            eval_module.JUDGE_AVAILABLE = original

    def test_handle_returns_none_for_unhandled_path(self, handler):
        """Test that handle returns None for unhandled GET paths."""
        mock_handler = create_mock_handler()

        result = handler.handle("/api/v1/evaluate/unknown", {}, mock_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_returns_none_for_unhandled_path(self, handler_with_mocked_judge):
        """Test that handle_post returns None for unhandled POST paths."""
        mock_handler = create_mock_handler(method="POST", body={})

        result = await handler_with_mocked_judge.handle_post(
            "/api/v1/evaluate/unknown", {}, mock_handler
        )
        assert result is None


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestEvaluationHandlerRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_limiter_initialized(self, handler):
        """Test that rate limiter is initialized."""
        import aragora.server.handlers.evaluation as eval_module

        assert eval_module._evaluation_limiter is not None
        # 30 requests per minute for expensive LLM calls
        assert eval_module._evaluation_limiter.rpm == 30
