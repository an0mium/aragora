"""Tests for MatrixDebatesAPI resource."""

import pytest
from unittest.mock import patch

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

    def test_create_forwards_scenario_specific_agents(self):
        """Test create() preserves scenario-level agent combinations."""
        client = AragoraClient()
        response = {
            "matrix_id": "matrix-123",
            "status": "completed",
            "task": "Compare model combinations",
            "scenario_count": 2,
            "best_result": {
                "scenario_name": "Combo A",
                "agents": ["anthropic-api", "openai-api"],
            },
        }

        with patch.object(client, "_post", return_value=response) as mock_post:
            result = client.matrix_debates.create(
                task="Compare model combinations",
                scenarios=[
                    {"name": "Combo A", "agents": ["anthropic-api", "openai-api"]},
                    {"name": "Combo B", "agents": ["gemini", "grok"]},
                ],
            )

        payload = mock_post.call_args.args[1]
        assert payload["scenarios"][0]["agents"] == ["anthropic-api", "openai-api"]
        assert payload["scenarios"][1]["agents"] == ["gemini", "grok"]
        assert result.best_result["scenario_name"] == "Combo A"
