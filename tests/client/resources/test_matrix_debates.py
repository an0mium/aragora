"""Tests for MatrixDebatesAPI resource."""

from unittest.mock import patch

import pytest

from aragora.client import AragoraClient
from aragora.client.resources.matrix_debates import MatrixDebatesAPI


class TestMatrixDebatesAPI:
    """Tests for MatrixDebatesAPI resource."""

    def test_matrix_debates_api_exists(self):
        """Test that MatrixDebatesAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.matrix_debates, MatrixDebatesAPI)

    def test_matrix_debates_api_has_create_methods(self):
        """Test that MatrixDebatesAPI has create methods."""
        client = AragoraClient()
        assert hasattr(client.matrix_debates, "create")
        assert hasattr(client.matrix_debates, "create_async")
        assert callable(client.matrix_debates.create)

    def test_matrix_debates_api_has_get_methods(self):
        """Test that MatrixDebatesAPI has get methods."""
        client = AragoraClient()
        assert hasattr(client.matrix_debates, "get")
        assert hasattr(client.matrix_debates, "get_async")


class TestMatrixDebateModels:
    """Tests for MatrixDebate model classes."""

    def test_matrix_debate_create_request_import(self):
        """Test MatrixDebateCreateRequest model can be imported."""
        from aragora.client.models import MatrixDebateCreateRequest

        request = MatrixDebateCreateRequest(
            task="Compare database options",
        )
        assert request.task == "Compare database options"

    def test_matrix_debate_create_response_import(self):
        """Test MatrixDebateCreateResponse model can be imported."""
        from aragora.client.models import MatrixDebateCreateResponse

        # Model import check
        assert MatrixDebateCreateResponse is not None

    def test_matrix_scenario_import(self):
        """Test MatrixScenario model can be imported."""
        from aragora.client.models import MatrixScenario

        scenario = MatrixScenario(
            name="PostgreSQL",
            description="Open source database",
        )
        assert scenario.name == "PostgreSQL"

    def test_matrix_debate_create_request_accepts_model_combinations(self):
        """Test MatrixDebateCreateRequest accepts the model_combinations alias."""
        from aragora.client.models import MatrixDebateCreateRequest

        request = MatrixDebateCreateRequest(
            task="Compare coding model combinations",
            model_combinations=[
                {
                    "name": "Frontier",
                    "agents": [
                        {"provider": "openai-api", "model": "gpt-5.4"},
                        {"provider": "anthropic-api", "model": "claude-opus-4-6"},
                    ],
                }
            ],
        )
        assert request.model_combinations[0]["name"] == "Frontier"

    def test_matrix_debates_api_create_accepts_model_combinations(self):
        """Test MatrixDebatesAPI forwards model_combinations in the request body."""
        client = AragoraClient()
        response_payload = {
            "matrix_id": "matrix-123",
            "status": "completed",
            "scenario_count": 2,
            "combination_count": 2,
        }

        with patch.object(client, "_post", return_value=response_payload) as mock_post:
            result = client.matrix_debates.create(
                task="Compare coding model combinations",
                model_combinations=[
                    {
                        "name": "Frontier",
                        "agents": [
                            {"provider": "openai-api", "model": "gpt-5.4"},
                            {"provider": "anthropic-api", "model": "claude-opus-4-6"},
                        ],
                    }
                ],
            )

        assert result.matrix_id == "matrix-123"
        assert mock_post.call_args.args[1]["model_combinations"][0]["name"] == "Frontier"
