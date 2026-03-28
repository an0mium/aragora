"""Tests for MatrixDebatesAPI resource."""

from aragora.client import AragoraClient
from aragora.client.resources.matrix_debates import MatrixDebatesAPI


class _FakeClient:
    def __init__(self):
        self.last_path = None
        self.last_payload = None

    def _post(self, path, payload):
        self.last_path = path
        self.last_payload = payload
        return {
            "matrix_id": "matrix-123",
            "scenario_count": len(payload["scenarios"]),
            "results": payload["scenarios"],
            "best_result": {
                "scenario_name": payload["scenarios"][-1]["name"],
                "agents": payload["scenarios"][-1].get("agents", []),
            },
        }


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

    def test_create_preserves_scenario_agent_overrides(self):
        """Scenario-specific agent combinations should round-trip through the client model."""
        client = _FakeClient()
        api = MatrixDebatesAPI(client)

        result = api.create(
            task="Compare the same debate question across model pairs",
            scenarios=[
                {"name": "Pair A", "agents": ["claude", "gpt4"]},
                {"name": "Pair B", "agents": ["gemini"]},
            ],
        )

        assert client.last_path == "/api/v1/debates/matrix"
        assert client.last_payload["scenarios"][0]["agents"] == ["claude", "gpt4"]
        assert client.last_payload["scenarios"][1]["agents"] == ["gemini"]
        assert result.best_result == {"scenario_name": "Pair B", "agents": ["gemini"]}


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
            agents=["claude", "gpt4"],
        )
        assert scenario.name == "PostgreSQL"
        assert scenario.agents == ["claude", "gpt4"]
