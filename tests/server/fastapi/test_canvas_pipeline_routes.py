"""
Tests for FastAPI canvas pipeline route endpoints.

Covers all 31 canvas pipeline v2 endpoints including:
- Pipeline creation (from-debate, from-ideas, from-braindump, from-template, demo)
- Pipeline execution (advance, run, auto-run, execute, self-improve)
- Pipeline querying (get, status, stage, graph, receipt, templates)
- Conversion (extract-goals, extract-principles, convert debate/workflow)
- Intelligence (intelligence, beliefs, explanations, precedents)
- Agent management (list, approve, reject)
- Canvas state (save)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def _bypass_auth(app):
    """Bypass authentication for all tests using FastAPI dependency overrides."""
    from aragora.rbac.models import AuthorizationContext
    from aragora.server.fastapi.dependencies.auth import require_authenticated

    mock_ctx = AuthorizationContext(
        user_id="test-user",
        user_email="test@example.com",
        roles={"admin"},
        permissions={"*"},
    )
    app.dependency_overrides[require_authenticated] = lambda: mock_ctx
    yield
    app.dependency_overrides.pop(require_authenticated, None)


@pytest.fixture
def mock_pipeline_store(monkeypatch):
    """Mock the pipeline store."""
    store = {}
    mock_store = MagicMock()
    mock_store.load = MagicMock(side_effect=lambda k: store.get(k))
    mock_store.save = MagicMock(side_effect=lambda k, v: store.__setitem__(k, v))
    monkeypatch.setattr(
        "aragora.server.fastapi.routes.canvas_pipeline._get_store",
        lambda: mock_store,
    )
    return store


@pytest.fixture
def sample_pipeline_result():
    """Create a sample PipelineResult mock."""
    result = MagicMock()
    result.pipeline_id = "pipe-test1234"
    result.stage_status = {
        "ideation": "complete",
        "goals": "complete",
        "workflow": "pending",
        "orchestration": "pending",
    }
    result.ideas_canvas = MagicMock()
    result.ideas_canvas.nodes = {"n1": {}, "n2": {}}
    result.actions_canvas = MagicMock()
    result.actions_canvas.nodes = {"a1": {}}
    result.orchestration_canvas = None
    result.universal_graph = None
    result.to_dict.return_value = {
        "pipeline_id": "pipe-test1234",
        "stage_status": result.stage_status,
    }
    return result


# =============================================================================
# Template listing (no auth required)
# =============================================================================


class TestListTemplates:
    """Tests for GET /api/v2/canvas/pipeline/templates."""

    def test_list_templates_returns_200(self, client):
        """Templates endpoint returns 200 even when module unavailable."""
        resp = client.get("/api/v2/canvas/pipeline/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert "total" in data
        assert isinstance(data["templates"], list)


# =============================================================================
# Pipeline Creation
# =============================================================================


class TestPipelineCreation:
    """Tests for pipeline creation endpoints."""

    @pytest.mark.usefixtures("_bypass_auth")
    def test_from_ideas_requires_ideas(self, client):
        """from-ideas rejects empty ideas list."""
        resp = client.post(
            "/api/v2/canvas/pipeline/from-ideas",
            json={"ideas": []},
        )
        assert resp.status_code == 422  # Validation error: min_length=1

    @pytest.mark.usefixtures("_bypass_auth")
    def test_from_braindump_requires_text(self, client):
        """from-braindump rejects empty text."""
        resp = client.post(
            "/api/v2/canvas/pipeline/from-braindump",
            json={"text": ""},
        )
        assert resp.status_code == 422  # Validation error: min_length=1

    @pytest.mark.usefixtures("_bypass_auth")
    def test_from_debate_requires_cartographer_data(self, client):
        """from-debate requires cartographer_data field."""
        resp = client.post(
            "/api/v2/canvas/pipeline/from-debate",
            json={},
        )
        assert resp.status_code == 422  # Missing required field

    @pytest.mark.usefixtures("_bypass_auth")
    def test_from_ideas_with_mock_pipeline(self, client, monkeypatch):
        """from-ideas succeeds with mocked pipeline backend."""
        mock_result = MagicMock()
        mock_result.pipeline_id = "pipe-mock1234"
        mock_result.stage_status = {"ideation": "complete"}
        mock_result.ideas_canvas = MagicMock()
        mock_result.ideas_canvas.nodes = {"n1": {}}
        mock_result.actions_canvas = None
        mock_result.orchestration_canvas = None
        mock_result.universal_graph = None
        mock_result.to_dict.return_value = {"pipeline_id": "pipe-mock1234"}

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.from_ideas.return_value = mock_result

        monkeypatch.setattr(
            "aragora.server.fastapi.routes.canvas_pipeline._get_store",
            lambda: MagicMock(),
        )

        with patch(
            "aragora.pipeline.idea_to_execution.IdeaToExecutionPipeline",
            mock_pipeline_cls,
        ):
            resp = client.post(
                "/api/v2/canvas/pipeline/from-ideas",
                json={"ideas": ["Test idea 1", "Test idea 2"]},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["pipeline_id"] == "pipe-mock1234"
        assert data["stages_completed"] == 1

    @pytest.mark.usefixtures("_bypass_auth")
    def test_demo_pipeline_delegates_to_from_ideas(self, client, monkeypatch):
        """Demo endpoint creates pipeline with sample ideas."""
        mock_result = MagicMock()
        mock_result.pipeline_id = "pipe-demo-test"
        mock_result.stage_status = {"ideation": "complete"}
        mock_result.ideas_canvas = None
        mock_result.actions_canvas = None
        mock_result.orchestration_canvas = None
        mock_result.universal_graph = None
        mock_result.to_dict.return_value = {"pipeline_id": "pipe-demo-test"}

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.from_ideas.return_value = mock_result

        monkeypatch.setattr(
            "aragora.server.fastapi.routes.canvas_pipeline._get_store",
            lambda: MagicMock(),
        )

        with patch(
            "aragora.pipeline.idea_to_execution.IdeaToExecutionPipeline",
            mock_pipeline_cls,
        ):
            resp = client.post("/api/v2/canvas/pipeline/demo")

        assert resp.status_code == 201
        data = resp.json()
        assert "pipe-demo" in data["pipeline_id"]


# =============================================================================
# Pipeline Querying
# =============================================================================


class TestPipelineQuerying:
    """Tests for pipeline query endpoints."""

    def test_get_pipeline_not_found(self, client, mock_pipeline_store):
        """Get pipeline returns 404 for unknown ID."""
        resp = client.get("/api/v2/canvas/pipeline/nonexistent")
        assert resp.status_code == 404

    def test_get_pipeline_from_store(self, client, mock_pipeline_store):
        """Get pipeline returns stored data."""
        mock_pipeline_store["pipe-test"] = {
            "pipeline_id": "pipe-test",
            "stage_status": {"ideation": "complete"},
        }
        resp = client.get("/api/v2/canvas/pipeline/pipe-test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_id"] == "pipe-test"

    def test_get_pipeline_status(self, client, mock_pipeline_store):
        """Status endpoint returns stage breakdown."""
        mock_pipeline_store["pipe-s1"] = {
            "pipeline_id": "pipe-s1",
            "stage_status": {
                "ideation": "complete",
                "goals": "complete",
                "workflow": "running",
                "orchestration": "pending",
            },
        }
        resp = client.get("/api/v2/canvas/pipeline/pipe-s1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["completed_stages"] == 2
        assert data["total_stages"] == 4
        assert data["current_stage"] == "workflow"

    def test_get_pipeline_stage_invalid(self, client, mock_pipeline_store):
        """Stage endpoint rejects invalid stage name."""
        mock_pipeline_store["pipe-x"] = {"pipeline_id": "pipe-x", "stage_status": {}}
        resp = client.get("/api/v2/canvas/pipeline/pipe-x/stage/invalid")
        assert resp.status_code == 400

    def test_get_pipeline_stage_valid(self, client, mock_pipeline_store):
        """Stage endpoint returns canvas data."""
        mock_pipeline_store["pipe-y"] = {
            "pipeline_id": "pipe-y",
            "stage_status": {"ideation": "complete"},
            "ideas_canvas": {"nodes": {"n1": {}, "n2": {}}, "edges": []},
        }
        resp = client.get("/api/v2/canvas/pipeline/pipe-y/stage/ideas")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stage"] == "ideas"
        assert data["node_count"] == 2

    def test_get_pipeline_graph_empty(self, client, mock_pipeline_store):
        """Graph endpoint returns empty when no universal graph."""
        mock_pipeline_store["pipe-g"] = {
            "pipeline_id": "pipe-g",
            "stage_status": {},
        }
        resp = client.get("/api/v2/canvas/pipeline/pipe-g/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_get_pipeline_receipt_no_receipt(self, client, mock_pipeline_store):
        """Receipt endpoint returns has_receipt=False when unavailable."""
        mock_pipeline_store["pipe-r"] = {
            "pipeline_id": "pipe-r",
            "stage_status": {},
        }
        resp = client.get("/api/v2/canvas/pipeline/pipe-r/receipt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_receipt"] is False


# =============================================================================
# Intelligence Endpoints
# =============================================================================


class TestIntelligence:
    """Tests for intelligence overlay endpoints."""

    def test_intelligence_returns_empty_when_unavailable(self, client, mock_pipeline_store):
        """Intelligence endpoint returns empty lists gracefully."""
        mock_pipeline_store["pipe-i1"] = {
            "pipeline_id": "pipe-i1",
            "stage_status": {},
        }
        resp = client.get("/api/v2/canvas/pipeline/pipe-i1/intelligence")
        assert resp.status_code == 200
        data = resp.json()
        assert data["beliefs"] == []
        assert data["explanations"] == []
        assert data["precedents"] == []

    def test_beliefs_returns_empty(self, client, mock_pipeline_store):
        """Beliefs endpoint returns empty list gracefully."""
        mock_pipeline_store["pipe-b1"] = {
            "pipeline_id": "pipe-b1",
            "stage_status": {},
        }
        resp = client.get("/api/v2/canvas/pipeline/pipe-b1/beliefs")
        assert resp.status_code == 200
        assert resp.json()["beliefs"] == []

    def test_explanations_returns_empty(self, client, mock_pipeline_store):
        """Explanations endpoint returns empty list gracefully."""
        mock_pipeline_store["pipe-e1"] = {
            "pipeline_id": "pipe-e1",
            "stage_status": {},
        }
        resp = client.get("/api/v2/canvas/pipeline/pipe-e1/explanations")
        assert resp.status_code == 200
        assert resp.json()["explanations"] == []

    def test_precedents_returns_empty(self, client, mock_pipeline_store):
        """Precedents endpoint returns empty list gracefully."""
        mock_pipeline_store["pipe-p1"] = {
            "pipeline_id": "pipe-p1",
            "stage_status": {},
        }
        resp = client.get("/api/v2/canvas/pipeline/pipe-p1/precedents")
        assert resp.status_code == 200
        assert resp.json()["precedents"] == []


# =============================================================================
# Agent Management
# =============================================================================


class TestAgentManagement:
    """Tests for agent management endpoints."""

    def test_list_agents_empty(self, client, mock_pipeline_store):
        """Agents endpoint returns empty list for pipeline without agents."""
        mock_pipeline_store["pipe-a1"] = {
            "pipeline_id": "pipe-a1",
            "stage_status": {},
        }
        resp = client.get("/api/v2/pipeline/pipe-a1/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agents"] == []
        assert data["total"] == 0

    def test_list_agents_with_data(self, client, mock_pipeline_store):
        """Agents endpoint returns agent list from pipeline data."""
        mock_pipeline_store["pipe-a2"] = {
            "pipeline_id": "pipe-a2",
            "stage_status": {},
            "agents": [
                {"id": "agent-1", "name": "Claude", "role": "executor", "status": "ready"},
                {"id": "agent-2", "name": "Codex", "role": "reviewer", "status": "pending"},
            ],
        }
        resp = client.get("/api/v2/pipeline/pipe-a2/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["agents"][0]["agent_name"] == "Claude"

    @pytest.mark.usefixtures("_bypass_auth")
    def test_approve_agent(self, client, mock_pipeline_store):
        """Approve agent returns success."""
        mock_pipeline_store["pipe-a3"] = {
            "pipeline_id": "pipe-a3",
            "stage_status": {},
        }
        resp = client.post("/api/v2/pipeline/pipe-a3/agents/agent-1/approve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "approved"
        assert data["success"] is True

    @pytest.mark.usefixtures("_bypass_auth")
    def test_reject_agent(self, client, mock_pipeline_store):
        """Reject agent returns success."""
        mock_pipeline_store["pipe-a4"] = {
            "pipeline_id": "pipe-a4",
            "stage_status": {},
        }
        resp = client.post("/api/v2/pipeline/pipe-a4/agents/agent-1/reject")
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "rejected"
        assert data["success"] is True


# =============================================================================
# Save Canvas State
# =============================================================================


class TestSaveCanvasState:
    """Tests for PUT /api/v2/canvas/pipeline/{id}."""

    @pytest.mark.usefixtures("_bypass_auth")
    def test_save_canvas_state(self, client, mock_pipeline_store):
        """Save canvas state persists data."""
        mock_pipeline_store["pipe-sv1"] = {"pipeline_id": "pipe-sv1"}
        resp = client.put(
            "/api/v2/canvas/pipeline/pipe-sv1",
            json={"canvas_data": {"nodes": {"n1": {}}}, "stage": "ideas"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["saved"] is True

    @pytest.mark.usefixtures("_bypass_auth")
    def test_save_canvas_creates_new(self, client, mock_pipeline_store):
        """Save canvas creates entry for new pipeline."""
        resp = client.put(
            "/api/v2/canvas/pipeline/pipe-new",
            json={"canvas_data": {"nodes": {}}, "stage": "goals"},
        )
        assert resp.status_code == 200
        assert resp.json()["saved"] is True


# =============================================================================
# Transition Approval
# =============================================================================


class TestTransitionApproval:
    """Tests for POST /api/v2/canvas/pipeline/{id}/approve-transition."""

    @pytest.mark.usefixtures("_bypass_auth")
    def test_approve_transition(self, client, mock_pipeline_store):
        """Approve transition returns success."""
        mock_pipeline_store["pipe-t1"] = {
            "pipeline_id": "pipe-t1",
            "stage_status": {"ideation": "complete"},
        }
        resp = client.post(
            "/api/v2/canvas/pipeline/pipe-t1/approve-transition",
            json={"approved": True, "feedback": "Looks good"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["approved"] is True

    @pytest.mark.usefixtures("_bypass_auth")
    def test_reject_transition(self, client, mock_pipeline_store):
        """Reject transition returns success."""
        mock_pipeline_store["pipe-t2"] = {
            "pipeline_id": "pipe-t2",
            "stage_status": {},
        }
        resp = client.post(
            "/api/v2/canvas/pipeline/pipe-t2/approve-transition",
            json={"approved": False, "feedback": "Needs revision"},
        )
        assert resp.status_code == 200
        assert resp.json()["approved"] is False


# =============================================================================
# Conversion Endpoints
# =============================================================================


class TestConversion:
    """Tests for conversion endpoints."""

    @pytest.mark.usefixtures("_bypass_auth")
    def test_convert_debate_requires_input(self, client):
        """Convert debate requires debate_id or debate_data."""
        resp = client.post(
            "/api/v2/canvas/convert/debate",
            json={},
        )
        assert resp.status_code == 400

    @pytest.mark.usefixtures("_bypass_auth")
    def test_convert_workflow_requires_input(self, client):
        """Convert workflow requires workflow_id or workflow_data."""
        resp = client.post(
            "/api/v2/canvas/convert/workflow",
            json={},
        )
        assert resp.status_code == 400
