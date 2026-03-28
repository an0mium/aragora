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

    def test_create_supports_model_combinations(self):
        """Matrix debate creation should forward model combinations."""
        client = AragoraClient()
        with patch.object(
            client,
            "_post",
            return_value={"matrix_id": "matrix-123", "status": "completed"},
        ) as mock_post:
            client.matrix_debates.create(
                task="Compare rollout options",
                scenarios=[{"name": "baseline"}],
                model_combinations=[
                    {"name": "Core pair", "agents": ["claude", "openai"]},
                    ["codex", "gemini"],
                ],
            )

        _, payload = mock_post.call_args.args
        assert payload["model_combinations"][0]["name"] == "Core pair"
        assert payload["model_combinations"][0]["agents"] == ["claude", "openai"]
        assert payload["model_combinations"][1]["agents"] == ["codex", "gemini"]


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
