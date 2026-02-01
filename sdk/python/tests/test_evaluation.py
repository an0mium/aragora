"""Tests for Evaluation SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


class TestEvaluationAPI:
    """Test synchronous EvaluationAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.evaluation import EvaluationAPI

        api = EvaluationAPI(mock_client)
        assert api._client is mock_client

    def test_evaluate(self, mock_client: MagicMock) -> None:
        """Test evaluate calls correct endpoint."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {
            "overall_score": 0.85,
            "dimension_scores": {"accuracy": 0.9, "clarity": 0.8},
            "feedback": "Good response with accurate information.",
        }

        api = EvaluationAPI(mock_client)
        result = api.evaluate(response="The capital of France is Paris.")

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/evaluate")
        assert call_args[1]["json"]["response"] == "The capital of France is Paris."
        assert result["overall_score"] == 0.85

    def test_evaluate_with_options(self, mock_client: MagicMock) -> None:
        """Test evaluate with all options."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {"overall_score": 0.92}

        api = EvaluationAPI(mock_client)
        api.evaluate(
            response="Paris is the capital of France.",
            prompt="What is the capital of France?",
            context="Asking about European capitals.",
            dimensions=["accuracy", "clarity", "completeness"],
            profile="strict",
            reference="The capital of France is Paris.",
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["response"] == "Paris is the capital of France."
        assert json_body["prompt"] == "What is the capital of France?"
        assert json_body["context"] == "Asking about European capitals."
        assert json_body["dimensions"] == ["accuracy", "clarity", "completeness"]
        assert json_body["profile"] == "strict"
        assert json_body["reference"] == "The capital of France is Paris."

    def test_compare(self, mock_client: MagicMock) -> None:
        """Test compare calls correct endpoint."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {
            "winner": "A",
            "margin": 0.15,
            "response_a_score": 0.9,
            "response_b_score": 0.75,
            "reasoning": "Response A is more accurate and complete.",
        }

        api = EvaluationAPI(mock_client)
        result = api.compare(
            response_a="First answer",
            response_b="Second answer",
        )

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/evaluate/compare")
        assert call_args[1]["json"]["response_a"] == "First answer"
        assert call_args[1]["json"]["response_b"] == "Second answer"
        assert result["winner"] == "A"
        assert result["margin"] == 0.15

    def test_compare_with_options(self, mock_client: MagicMock) -> None:
        """Test compare with all options."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {"winner": "tie", "margin": 0.02}

        api = EvaluationAPI(mock_client)
        api.compare(
            response_a="Response A content",
            response_b="Response B content",
            prompt="Original question",
            context="Additional context",
            dimensions=["accuracy", "helpfulness"],
            profile="lenient",
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["prompt"] == "Original question"
        assert json_body["context"] == "Additional context"
        assert json_body["dimensions"] == ["accuracy", "helpfulness"]
        assert json_body["profile"] == "lenient"

    def test_compare_tie_result(self, mock_client: MagicMock) -> None:
        """Test compare returns tie when responses are equal."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {
            "winner": "tie",
            "margin": 0.0,
            "response_a_score": 0.85,
            "response_b_score": 0.85,
        }

        api = EvaluationAPI(mock_client)
        result = api.compare(response_a="Equal response", response_b="Equal response")

        assert result["winner"] == "tie"
        assert result["margin"] == 0.0

    def test_list_dimensions(self, mock_client: MagicMock) -> None:
        """Test list_dimensions calls correct endpoint."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {
            "dimensions": [
                {"id": "accuracy", "name": "Accuracy", "weight": 1.0},
                {"id": "clarity", "name": "Clarity", "weight": 0.8},
            ]
        }

        api = EvaluationAPI(mock_client)
        result = api.list_dimensions()

        mock_client.request.assert_called_once_with("GET", "/api/v1/evaluate/dimensions")
        assert len(result["dimensions"]) == 2
        assert result["dimensions"][0]["id"] == "accuracy"

    def test_list_profiles(self, mock_client: MagicMock) -> None:
        """Test list_profiles calls correct endpoint."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {
            "profiles": [
                {"id": "default", "name": "Default Profile", "default": True},
                {"id": "strict", "name": "Strict Profile", "default": False},
            ]
        }

        api = EvaluationAPI(mock_client)
        result = api.list_profiles()

        mock_client.request.assert_called_once_with("GET", "/api/v1/evaluate/profiles")
        assert len(result["profiles"]) == 2
        assert result["profiles"][0]["default"] is True

    def test_get_dimension(self, mock_client: MagicMock) -> None:
        """Test get_dimension convenience method."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {
            "dimensions": [
                {"id": "accuracy", "name": "Accuracy", "description": "Factual correctness"},
                {"id": "clarity", "name": "Clarity", "description": "Clear expression"},
            ]
        }

        api = EvaluationAPI(mock_client)
        result = api.get_dimension("accuracy")

        assert result["id"] == "accuracy"
        assert result["name"] == "Accuracy"

    def test_get_dimension_not_found(self, mock_client: MagicMock) -> None:
        """Test get_dimension raises ValueError when not found."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {
            "dimensions": [
                {"id": "accuracy", "name": "Accuracy"},
            ]
        }

        api = EvaluationAPI(mock_client)

        with pytest.raises(ValueError, match="Dimension not found: nonexistent"):
            api.get_dimension("nonexistent")

    def test_get_profile(self, mock_client: MagicMock) -> None:
        """Test get_profile convenience method."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {
            "profiles": [
                {"id": "default", "name": "Default Profile"},
                {"id": "strict", "name": "Strict Profile"},
            ]
        }

        api = EvaluationAPI(mock_client)
        result = api.get_profile("strict")

        assert result["id"] == "strict"
        assert result["name"] == "Strict Profile"

    def test_get_profile_not_found(self, mock_client: MagicMock) -> None:
        """Test get_profile raises ValueError when not found."""
        from aragora.namespaces.evaluation import EvaluationAPI

        mock_client.request.return_value = {
            "profiles": [
                {"id": "default", "name": "Default Profile"},
            ]
        }

        api = EvaluationAPI(mock_client)

        with pytest.raises(ValueError, match="Profile not found: nonexistent"):
            api.get_profile("nonexistent")


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    from unittest.mock import AsyncMock

    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestAsyncEvaluationAPI:
    """Test asynchronous AsyncEvaluationAPI."""

    @pytest.mark.asyncio
    async def test_evaluate(self, mock_async_client: MagicMock) -> None:
        """Test evaluate calls correct endpoint."""
        from aragora.namespaces.evaluation import AsyncEvaluationAPI

        mock_async_client.request.return_value = {
            "overall_score": 0.88,
            "feedback": "Well-structured response.",
        }

        api = AsyncEvaluationAPI(mock_async_client)
        result = await api.evaluate(response="Test response content")

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/evaluate")
        assert result["overall_score"] == 0.88

    @pytest.mark.asyncio
    async def test_evaluate_with_options(self, mock_async_client: MagicMock) -> None:
        """Test evaluate with all options."""
        from aragora.namespaces.evaluation import AsyncEvaluationAPI

        mock_async_client.request.return_value = {"overall_score": 0.95}

        api = AsyncEvaluationAPI(mock_async_client)
        await api.evaluate(
            response="Detailed response",
            prompt="Complex question",
            context="Technical context",
            dimensions=["accuracy", "depth", "examples"],
            profile="expert",
            reference="Expected answer format",
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["response"] == "Detailed response"
        assert json_body["prompt"] == "Complex question"
        assert json_body["dimensions"] == ["accuracy", "depth", "examples"]
        assert json_body["profile"] == "expert"

    @pytest.mark.asyncio
    async def test_compare(self, mock_async_client: MagicMock) -> None:
        """Test compare calls correct endpoint."""
        from aragora.namespaces.evaluation import AsyncEvaluationAPI

        mock_async_client.request.return_value = {
            "winner": "B",
            "margin": 0.1,
            "reasoning": "Response B is more comprehensive.",
        }

        api = AsyncEvaluationAPI(mock_async_client)
        result = await api.compare(
            response_a="Short answer",
            response_b="Detailed comprehensive answer",
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/evaluate/compare")
        assert result["winner"] == "B"

    @pytest.mark.asyncio
    async def test_compare_with_options(self, mock_async_client: MagicMock) -> None:
        """Test compare with all options."""
        from aragora.namespaces.evaluation import AsyncEvaluationAPI

        mock_async_client.request.return_value = {"winner": "A"}

        api = AsyncEvaluationAPI(mock_async_client)
        await api.compare(
            response_a="A content",
            response_b="B content",
            prompt="Question",
            context="Context info",
            dimensions=["correctness"],
            profile="balanced",
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["prompt"] == "Question"
        assert json_body["profile"] == "balanced"

    @pytest.mark.asyncio
    async def test_list_dimensions(self, mock_async_client: MagicMock) -> None:
        """Test list_dimensions calls correct endpoint."""
        from aragora.namespaces.evaluation import AsyncEvaluationAPI

        mock_async_client.request.return_value = {
            "dimensions": [
                {"id": "helpfulness", "name": "Helpfulness"},
            ]
        }

        api = AsyncEvaluationAPI(mock_async_client)
        result = await api.list_dimensions()

        mock_async_client.request.assert_called_once_with("GET", "/api/v1/evaluate/dimensions")
        assert len(result["dimensions"]) == 1

    @pytest.mark.asyncio
    async def test_list_profiles(self, mock_async_client: MagicMock) -> None:
        """Test list_profiles calls correct endpoint."""
        from aragora.namespaces.evaluation import AsyncEvaluationAPI

        mock_async_client.request.return_value = {
            "profiles": [
                {"id": "qa", "name": "QA Profile"},
            ]
        }

        api = AsyncEvaluationAPI(mock_async_client)
        result = await api.list_profiles()

        mock_async_client.request.assert_called_once_with("GET", "/api/v1/evaluate/profiles")
        assert result["profiles"][0]["id"] == "qa"

    @pytest.mark.asyncio
    async def test_get_dimension(self, mock_async_client: MagicMock) -> None:
        """Test get_dimension convenience method."""
        from aragora.namespaces.evaluation import AsyncEvaluationAPI

        mock_async_client.request.return_value = {
            "dimensions": [
                {"id": "relevance", "name": "Relevance"},
                {"id": "accuracy", "name": "Accuracy"},
            ]
        }

        api = AsyncEvaluationAPI(mock_async_client)
        result = await api.get_dimension("relevance")

        assert result["id"] == "relevance"

    @pytest.mark.asyncio
    async def test_get_dimension_not_found(self, mock_async_client: MagicMock) -> None:
        """Test get_dimension raises ValueError when not found."""
        from aragora.namespaces.evaluation import AsyncEvaluationAPI

        mock_async_client.request.return_value = {"dimensions": []}

        api = AsyncEvaluationAPI(mock_async_client)

        with pytest.raises(ValueError, match="Dimension not found: missing"):
            await api.get_dimension("missing")

    @pytest.mark.asyncio
    async def test_get_profile(self, mock_async_client: MagicMock) -> None:
        """Test get_profile convenience method."""
        from aragora.namespaces.evaluation import AsyncEvaluationAPI

        mock_async_client.request.return_value = {
            "profiles": [
                {"id": "creative", "name": "Creative Profile"},
            ]
        }

        api = AsyncEvaluationAPI(mock_async_client)
        result = await api.get_profile("creative")

        assert result["id"] == "creative"
        assert result["name"] == "Creative Profile"

    @pytest.mark.asyncio
    async def test_get_profile_not_found(self, mock_async_client: MagicMock) -> None:
        """Test get_profile raises ValueError when not found."""
        from aragora.namespaces.evaluation import AsyncEvaluationAPI

        mock_async_client.request.return_value = {"profiles": []}

        api = AsyncEvaluationAPI(mock_async_client)

        with pytest.raises(ValueError, match="Profile not found: missing"):
            await api.get_profile("missing")
