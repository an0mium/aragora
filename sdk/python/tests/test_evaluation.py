"""Tests for Evaluation namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestEvaluationEvaluate:
    """Tests for the evaluate method."""

    def test_evaluate_basic(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "overall_score": 0.92,
                "dimension_scores": {"accuracy": 0.95, "clarity": 0.89},
                "feedback": "Excellent response.",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.evaluation.evaluate(response="The capital of France is Paris.")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/evaluate",
                json={"response": "The capital of France is Paris."},
            )
            assert result["overall_score"] == 0.92
            client.close()

    def test_evaluate_with_all_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"overall_score": 0.88}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.evaluation.evaluate(
                response="Paris is the capital of France.",
                prompt="What is the capital of France?",
                context="Geography quiz",
                dimensions=["accuracy", "clarity", "completeness"],
                profile="strict",
                reference="The capital of France is Paris.",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/evaluate",
                json={
                    "response": "Paris is the capital of France.",
                    "prompt": "What is the capital of France?",
                    "context": "Geography quiz",
                    "dimensions": ["accuracy", "clarity", "completeness"],
                    "profile": "strict",
                    "reference": "The capital of France is Paris.",
                },
            )
            client.close()

    def test_evaluate_omits_none_fields(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"overall_score": 0.75}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.evaluation.evaluate(
                response="Some answer",
                prompt="Some question",
            )
            call_json = mock_request.call_args[1]["json"]
            assert "context" not in call_json
            assert "dimensions" not in call_json
            assert "profile" not in call_json
            assert "reference" not in call_json
            assert call_json["response"] == "Some answer"
            assert call_json["prompt"] == "Some question"
            client.close()


class TestEvaluationCompare:
    """Tests for the compare method."""

    def test_compare_basic(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "winner": "A",
                "margin": 0.15,
                "response_a_score": 0.90,
                "response_b_score": 0.75,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.evaluation.compare(
                response_a="First answer",
                response_b="Second answer",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/evaluate/compare",
                json={
                    "response_a": "First answer",
                    "response_b": "Second answer",
                },
            )
            assert result["winner"] == "A"
            assert result["margin"] == 0.15
            client.close()

    def test_compare_with_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"winner": "tie", "margin": 0.01}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.evaluation.compare(
                response_a="Answer A",
                response_b="Answer B",
                prompt="Original question",
                context="Technical review",
                dimensions=["accuracy", "depth"],
                profile="technical",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/evaluate/compare",
                json={
                    "response_a": "Answer A",
                    "response_b": "Answer B",
                    "prompt": "Original question",
                    "context": "Technical review",
                    "dimensions": ["accuracy", "depth"],
                    "profile": "technical",
                },
            )
            client.close()


class TestEvaluationDimensionsProfiles:
    """Tests for dimensions and profiles listing and retrieval."""

    def test_list_dimensions(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "dimensions": [
                    {"id": "accuracy", "name": "Accuracy", "weight": 1.0},
                    {"id": "clarity", "name": "Clarity", "weight": 0.8},
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.evaluation.list_dimensions()
            mock_request.assert_called_once_with("GET", "/api/v1/evaluate/dimensions")
            assert len(result["dimensions"]) == 2
            client.close()

    def test_list_profiles(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "profiles": [
                    {"id": "default", "name": "Default", "default": True},
                    {"id": "strict", "name": "Strict", "default": False},
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.evaluation.list_profiles()
            mock_request.assert_called_once_with("GET", "/api/v1/evaluate/profiles")
            assert len(result["profiles"]) == 2
            assert result["profiles"][0]["default"] is True
            client.close()

    def test_get_dimension_found(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "dimensions": [
                    {"id": "accuracy", "name": "Accuracy"},
                    {"id": "clarity", "name": "Clarity"},
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.evaluation.get_dimension("clarity")
            assert result["id"] == "clarity"
            assert result["name"] == "Clarity"
            client.close()

    def test_get_dimension_not_found(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"dimensions": [{"id": "accuracy", "name": "Accuracy"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            with pytest.raises(ValueError, match="Dimension not found: nonexistent"):
                client.evaluation.get_dimension("nonexistent")
            client.close()

    def test_get_profile_found(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "profiles": [
                    {"id": "default", "name": "Default"},
                    {"id": "strict", "name": "Strict"},
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.evaluation.get_profile("strict")
            assert result["id"] == "strict"
            assert result["name"] == "Strict"
            client.close()

    def test_get_profile_not_found(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"profiles": [{"id": "default", "name": "Default"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            with pytest.raises(ValueError, match="Profile not found: missing"):
                client.evaluation.get_profile("missing")
            client.close()


class TestAsyncEvaluation:
    """Tests for async evaluation methods."""

    @pytest.mark.asyncio
    async def test_evaluate(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "overall_score": 0.91,
                "feedback": "Great response.",
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.evaluation.evaluate(
                response="Paris is the capital.",
                prompt="What is the capital of France?",
                dimensions=["accuracy"],
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/evaluate",
                json={
                    "response": "Paris is the capital.",
                    "prompt": "What is the capital of France?",
                    "dimensions": ["accuracy"],
                },
            )
            assert result["overall_score"] == 0.91
            await client.close()

    @pytest.mark.asyncio
    async def test_compare(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"winner": "B", "margin": 0.2}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.evaluation.compare(
                response_a="Answer A",
                response_b="Answer B",
                prompt="A question",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/evaluate/compare",
                json={
                    "response_a": "Answer A",
                    "response_b": "Answer B",
                    "prompt": "A question",
                },
            )
            assert result["winner"] == "B"
            await client.close()

    @pytest.mark.asyncio
    async def test_list_dimensions(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"dimensions": [{"id": "accuracy"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.evaluation.list_dimensions()
            mock_request.assert_called_once_with("GET", "/api/v1/evaluate/dimensions")
            assert len(result["dimensions"]) == 1
            await client.close()

    @pytest.mark.asyncio
    async def test_get_dimension_not_found(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"dimensions": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            with pytest.raises(ValueError, match="Dimension not found: missing"):
                await client.evaluation.get_dimension("missing")
            await client.close()
