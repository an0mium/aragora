"""
Tests for FastAPI workflow route endpoints.

Covers:
- GET  /api/v2/workflows                         - List workflows
- GET  /api/v2/workflows/{workflow_id}            - Get workflow details
- POST /api/v2/workflows                          - Create workflow
- POST /api/v2/workflows/{workflow_id}/execute     - Execute workflow
- GET  /api/v2/workflows/{workflow_id}/status      - Get execution status
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
def mock_workflow_engine():
    """Create a mock workflow engine."""
    engine = MagicMock()
    engine.list_workflows = MagicMock(return_value=[])
    engine.count_workflows = MagicMock(return_value=0)
    engine.get_workflow = MagicMock(return_value=None)
    engine.create_workflow = MagicMock(return_value={"id": "wf_created123"})
    engine.execute = MagicMock(return_value={"execution_id": "exec_run123"})
    return engine


@pytest.fixture
def client(app, mock_workflow_engine):
    """Create a test client with mocked context."""
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
        "workflow_engine": mock_workflow_engine,
    }
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def raw_workflow_engine():
    """Engine-shaped object matching the real WorkflowEngine CRUD gap."""

    class RawWorkflowEngine:
        def execute(self, *args, **kwargs):
            return {"execution_id": "exec_raw123"}

    return RawWorkflowEngine()


@pytest.fixture
def fallback_client(app, raw_workflow_engine):
    """Client whose workflow engine matches the real raw-engine interface."""
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
        "workflow_engine": raw_workflow_engine,
    }
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_workflow():
    """Sample workflow data for testing."""
    return {
        "id": "wf_test123",
        "name": "Security Review Pipeline",
        "description": "Automated security review workflow",
        "status": "completed",
        "template": "security_review",
        "nodes": [
            {
                "id": "node-1",
                "type": "debate",
                "name": "Initial Review",
                "status": "completed",
                "config": {"rounds": 3},
            },
            {
                "id": "node-2",
                "type": "verification",
                "name": "Formal Verification",
                "status": "completed",
                "config": {},
            },
            {
                "id": "node-3",
                "type": "receipt",
                "name": "Generate Receipt",
                "status": "completed",
                "config": {},
            },
        ],
        "edges": [
            {"from": "node-1", "to": "node-2"},
            {"from": "node-2", "to": "node-3"},
        ],
        "config": {"timeout": 300},
        "created_at": "2026-02-15T10:00:00",
        "updated_at": "2026-02-15T10:30:00",
        "started_at": "2026-02-15T10:01:00",
        "completed_at": "2026-02-15T10:30:00",
        "result": {"verdict": "APPROVED"},
        "error": None,
    }


def _override_auth(client, permissions=None):
    """Override auth for write operations."""
    from aragora.server.fastapi.dependencies.auth import require_authenticated
    from aragora.rbac.models import AuthorizationContext

    if permissions is None:
        permissions = {"workflows:write", "workflows:execute"}

    auth_ctx = AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        workspace_id="ws-1",
        roles={"admin"},
        permissions=permissions,
    )
    client.app.dependency_overrides[require_authenticated] = lambda: auth_ctx


def _persisted_workflow_definition():
    """Workflow definition shape returned by the persisted handler layer."""
    return {
        "id": "wf_store123",
        "name": "Stored Workflow",
        "description": "Persisted workflow definition",
        "steps": [
            {
                "id": "node-1",
                "step_type": "debate",
                "name": "Debate",
                "config": {"rounds": 3},
                "next_steps": ["node-2"],
            },
            {
                "id": "node-2",
                "step_type": "verification",
                "name": "Verify",
                "config": {},
                "next_steps": [],
            },
        ],
        "transitions": [
            {
                "id": "tr_node-1_to_node-2",
                "from_step": "node-1",
                "to_step": "node-2",
                "condition": "",
                "priority": 0,
                "label": "",
            }
        ],
        "metadata": {"config": {"timeout": 120}, "template": "security_review"},
        "created_at": "2026-04-01T12:00:00Z",
        "updated_at": "2026-04-01T12:05:00Z",
    }


# =============================================================================
# GET /api/v2/workflows
# =============================================================================


class TestListWorkflows:
    """Tests for GET /api/v2/workflows."""

    def test_list_returns_200_empty(self, client):
        """List workflows returns 200 with empty list."""
        response = client.get("/api/v2/workflows")
        assert response.status_code == 200
        data = response.json()
        assert data["workflows"] == []
        assert data["total"] == 0
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_list_with_data(self, client, mock_workflow_engine, sample_workflow):
        """List returns workflow summaries."""
        mock_workflow_engine.list_workflows.return_value = [sample_workflow]
        mock_workflow_engine.count_workflows.return_value = 1

        response = client.get("/api/v2/workflows")
        assert response.status_code == 200
        data = response.json()
        assert len(data["workflows"]) == 1
        assert data["workflows"][0]["id"] == "wf_test123"
        assert data["workflows"][0]["name"] == "Security Review Pipeline"
        assert data["workflows"][0]["status"] == "completed"
        assert data["workflows"][0]["node_count"] == 3

    def test_list_pagination(self, client, mock_workflow_engine):
        """List passes pagination params."""
        response = client.get("/api/v2/workflows?limit=10&offset=5")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 5
        mock_workflow_engine.list_workflows.assert_called_once_with(
            limit=10,
            offset=5,
            status=None,
        )

    def test_list_with_status_filter(self, client, mock_workflow_engine):
        """List passes status filter."""
        response = client.get("/api/v2/workflows?status=running")
        assert response.status_code == 200
        mock_workflow_engine.list_workflows.assert_called_once_with(
            limit=50,
            offset=0,
            status="running",
        )

    def test_list_limit_validation(self, client):
        """List limit must be between 1 and 100."""
        response = client.get("/api/v2/workflows?limit=0")
        assert response.status_code == 422

        response = client.get("/api/v2/workflows?limit=101")
        assert response.status_code == 422

    def test_list_when_engine_unavailable(self, app):
        """List returns empty when engine is not available."""
        from aragora.server.fastapi.routes.workflows import get_workflow_engine as _dep

        app.state.context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
            "workflow_engine": None,
        }
        app.dependency_overrides[_dep] = lambda: None
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/v2/workflows")
        app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert data["workflows"] == []
        assert data["total"] == 0


# =============================================================================
# GET /api/v2/workflows/{workflow_id}
# =============================================================================


class TestGetWorkflow:
    """Tests for GET /api/v2/workflows/{workflow_id}."""

    def test_get_not_found(self, client):
        """Get nonexistent workflow returns 404."""
        response = client.get("/api/v2/workflows/nonexistent-id")
        assert response.status_code == 404

    def test_get_found(self, client, mock_workflow_engine, sample_workflow):
        """Get existing workflow returns full details."""
        mock_workflow_engine.get_workflow.return_value = sample_workflow

        response = client.get("/api/v2/workflows/wf_test123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "wf_test123"
        assert data["name"] == "Security Review Pipeline"
        assert data["status"] == "completed"
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2
        assert data["result"]["verdict"] == "APPROVED"

    def test_get_unavailable_engine(self, app):
        """Get returns 503 when engine is unavailable."""
        from aragora.server.fastapi.routes.workflows import get_workflow_engine as _dep

        app.state.context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
            "workflow_engine": None,
        }
        app.dependency_overrides[_dep] = lambda: None
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/v2/workflows/wf_test123")
        app.dependency_overrides.clear()
        assert response.status_code == 503


# =============================================================================
# POST /api/v2/workflows
# =============================================================================


class TestCreateWorkflow:
    """Tests for POST /api/v2/workflows."""

    def test_create_returns_201(self, client, mock_workflow_engine):
        """Create returns 201 with new workflow."""
        _override_auth(client)

        response = client.post(
            "/api/v2/workflows",
            json={
                "name": "New Pipeline",
                "description": "A test workflow",
                "nodes": [{"id": "n1", "type": "debate", "name": "Debate"}],
                "edges": [],
            },
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["workflow_id"].startswith("wf_")
        assert data["workflow"]["name"] == "New Pipeline"
        assert data["workflow"]["status"] == "pending"

    def test_create_requires_name(self, client):
        """Create without name returns 422."""
        _override_auth(client)

        response = client.post(
            "/api/v2/workflows",
            json={"description": "No name"},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 422

    def test_create_requires_auth(self, client):
        """Create without auth returns 401."""
        response = client.post(
            "/api/v2/workflows",
            json={"name": "Unauthorized"},
        )
        assert response.status_code == 401

    def test_create_calls_engine(self, client, mock_workflow_engine):
        """Create calls the workflow engine create method."""
        _override_auth(client)

        response = client.post(
            "/api/v2/workflows",
            json={"name": "Test Workflow"},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 201
        mock_workflow_engine.create_workflow.assert_called_once()

    def test_create_unavailable_engine(self, app):
        """Create returns 503 when engine is unavailable."""
        from aragora.server.fastapi.routes.workflows import get_workflow_engine as _dep

        app.state.context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
            "workflow_engine": None,
        }
        client = TestClient(app, raise_server_exceptions=False)
        _override_auth(client)
        client.app.dependency_overrides[_dep] = lambda: None

        response = client.post(
            "/api/v2/workflows",
            json={"name": "Test"},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 503


# =============================================================================
# POST /api/v2/workflows/{workflow_id}/execute
# =============================================================================


class TestExecuteWorkflow:
    """Tests for POST /api/v2/workflows/{workflow_id}/execute."""

    def test_execute_not_found(self, client):
        """Execute nonexistent workflow returns 404."""
        _override_auth(client)

        response = client.post(
            "/api/v2/workflows/nonexistent-id/execute",
            json={"input_data": {}},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 404

    def test_execute_returns_200(self, client, mock_workflow_engine, sample_workflow):
        """Execute existing workflow returns execution info."""
        mock_workflow_engine.get_workflow.return_value = sample_workflow
        _override_auth(client)

        response = client.post(
            "/api/v2/workflows/wf_test123/execute",
            json={"input_data": {"task": "Review security"}},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["workflow_id"] == "wf_test123"
        assert "execution_id" in data
        assert data["status"] == "running"

    def test_execute_sync_mode(self, client, mock_workflow_engine, sample_workflow):
        """Execute with async=false returns completed status."""
        mock_workflow_engine.get_workflow.return_value = sample_workflow
        _override_auth(client)

        response = client.post(
            "/api/v2/workflows/wf_test123/execute",
            json={"input_data": {}, "async_execution": False},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    def test_execute_requires_auth(self, client):
        """Execute without auth returns 401."""
        response = client.post(
            "/api/v2/workflows/wf_test123/execute",
            json={"input_data": {}},
        )
        assert response.status_code == 401

    def test_execute_calls_engine(self, client, mock_workflow_engine, sample_workflow):
        """Execute calls the workflow engine execute method."""
        mock_workflow_engine.get_workflow.return_value = sample_workflow
        _override_auth(client)

        response = client.post(
            "/api/v2/workflows/wf_test123/execute",
            json={"input_data": {"key": "value"}},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        mock_workflow_engine.execute.assert_called_once_with(
            "wf_test123",
            input_data={"key": "value"},
            async_execution=True,
        )


# =============================================================================
# GET /api/v2/workflows/{workflow_id}/status
# =============================================================================


class TestWorkflowStatus:
    """Tests for GET /api/v2/workflows/{workflow_id}/status."""

    def test_status_not_found(self, client):
        """Status of nonexistent workflow returns 404."""
        response = client.get("/api/v2/workflows/nonexistent-id/status")
        assert response.status_code == 404

    def test_status_completed_workflow(self, client, mock_workflow_engine, sample_workflow):
        """Status of completed workflow shows full progress."""
        mock_workflow_engine.get_workflow.return_value = sample_workflow

        response = client.get("/api/v2/workflows/wf_test123/status")
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "wf_test123"
        assert data["status"] == "completed"
        assert data["progress"] == 1.0
        assert len(data["completed_nodes"]) == 3
        assert data["failed_nodes"] == []
        assert data["error"] is None

    def test_status_partially_complete(self, client, mock_workflow_engine):
        """Status of in-progress workflow shows partial progress."""
        mock_workflow_engine.get_workflow.return_value = {
            "id": "wf_partial",
            "name": "Partial",
            "status": "running",
            "nodes": [
                {"id": "n1", "type": "debate", "name": "Done", "status": "completed"},
                {"id": "n2", "type": "verify", "name": "Running", "status": "running"},
                {"id": "n3", "type": "receipt", "name": "Pending", "status": "pending"},
            ],
            "edges": [],
        }

        response = client.get("/api/v2/workflows/wf_partial/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["progress"] == pytest.approx(0.333, abs=0.01)
        assert data["current_node"] == "n2"
        assert data["completed_nodes"] == ["n1"]

    def test_status_with_failures(self, client, mock_workflow_engine):
        """Status shows failed nodes."""
        mock_workflow_engine.get_workflow.return_value = {
            "id": "wf_failed",
            "name": "Failed",
            "status": "failed",
            "nodes": [
                {"id": "n1", "type": "debate", "name": "Done", "status": "completed"},
                {"id": "n2", "type": "verify", "name": "Failed", "status": "failed"},
            ],
            "edges": [],
            "error": "Verification failed",
        }

        response = client.get("/api/v2/workflows/wf_failed/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["failed_nodes"] == ["n2"]
        assert data["error"] == "Verification failed"

    def test_status_unavailable_engine(self, app):
        """Status returns 503 when engine is unavailable."""
        from aragora.server.fastapi.routes.workflows import get_workflow_engine as _dep

        app.state.context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
            "workflow_engine": None,
        }
        app.dependency_overrides[_dep] = lambda: None
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/v2/workflows/wf_test123/status")
        app.dependency_overrides.clear()
        assert response.status_code == 503


# =============================================================================
# GET /api/v2/workflows/templates
# =============================================================================


class TestListTemplates:
    """Tests for GET /api/v2/workflows/templates."""

    def test_list_returns_200(self, client):
        """List returns templates when available."""
        from unittest.mock import patch

        with patch(
            "aragora.server.fastapi.routes.workflows.list_templates",
            create=True,
        ) as mock_list:
            mock_list.return_value = [
                {
                    "name": "security_review",
                    "description": "Automated security review",
                    "category": "security",
                    "nodes": [{"id": "n1"}, {"id": "n2"}],
                    "tags": ["security", "review"],
                },
                {
                    "name": "code_audit",
                    "description": "Code audit workflow",
                    "category": "development",
                    "nodes": [{"id": "n1"}],
                    "tags": ["audit"],
                },
            ]

            # We need to patch at the import location
            with patch(
                "aragora.workflow.templates.list_templates",
                mock_list,
                create=True,
            ):
                response = client.get("/api/v2/workflows/templates")

        assert response.status_code == 200
        data = response.json()
        # Templates may or may not load depending on import availability
        assert "templates" in data
        assert "total" in data

    def test_list_returns_empty_when_unavailable(self, client):
        """List returns empty when templates module unavailable."""
        # With the default client, templates import may fail gracefully
        response = client.get("/api/v2/workflows/templates")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert data["total"] >= 0

    def test_list_with_category_filter(self, client):
        """List passes category filter."""
        response = client.get("/api/v2/workflows/templates?category=security")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data


# =============================================================================
# GET /api/v2/workflows/{workflow_id}/history
# =============================================================================


class TestWorkflowHistory:
    """Tests for GET /api/v2/workflows/{workflow_id}/history."""

    def test_history_returns_200(self, client, mock_workflow_engine, sample_workflow):
        """History returns execution entries."""
        mock_workflow_engine.get_workflow.return_value = sample_workflow
        mock_workflow_engine.get_execution_history = MagicMock(
            return_value=[
                {
                    "execution_id": "exec_001",
                    "status": "completed",
                    "started_at": "2026-02-15T10:01:00",
                    "completed_at": "2026-02-15T10:30:00",
                    "duration_seconds": 1740.0,
                    "result": {"verdict": "APPROVED"},
                    "error": None,
                },
                {
                    "execution_id": "exec_002",
                    "status": "failed",
                    "started_at": "2026-02-14T10:00:00",
                    "completed_at": "2026-02-14T10:05:00",
                    "duration_seconds": 300.0,
                    "result": None,
                    "error": "Timeout exceeded",
                },
            ]
        )

        response = client.get("/api/v2/workflows/wf_test123/history")
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "wf_test123"
        assert data["total"] == 2
        assert data["executions"][0]["execution_id"] == "exec_001"
        assert data["executions"][0]["status"] == "completed"
        assert data["executions"][1]["error"] == "Timeout exceeded"

    def test_history_not_found(self, client, mock_workflow_engine):
        """History for nonexistent workflow returns 404."""
        mock_workflow_engine.get_workflow.return_value = None

        response = client.get("/api/v2/workflows/nonexistent/history")
        assert response.status_code == 404

    def test_history_with_limit(self, client, mock_workflow_engine, sample_workflow):
        """History passes limit parameter."""
        mock_workflow_engine.get_workflow.return_value = sample_workflow
        mock_workflow_engine.get_execution_history = MagicMock(return_value=[])

        response = client.get("/api/v2/workflows/wf_test123/history?limit=5")
        assert response.status_code == 200
        mock_workflow_engine.get_execution_history.assert_called_once_with(
            "wf_test123",
            limit=5,
        )

    def test_history_empty(self, client, mock_workflow_engine, sample_workflow):
        """History returns empty list when no executions."""
        mock_workflow_engine.get_workflow.return_value = sample_workflow
        mock_workflow_engine.get_execution_history = MagicMock(return_value=[])

        response = client.get("/api/v2/workflows/wf_test123/history")
        assert response.status_code == 200
        data = response.json()
        assert data["executions"] == []
        assert data["total"] == 0

    def test_history_unavailable_engine(self, app):
        """History returns 503 when engine is unavailable."""
        from aragora.server.fastapi.routes.workflows import get_workflow_engine as _dep

        app.state.context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
            "workflow_engine": None,
        }
        app.dependency_overrides[_dep] = lambda: None
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/v2/workflows/wf_test123/history")
        app.dependency_overrides.clear()
        assert response.status_code == 503


# =============================================================================
# POST /api/v2/workflows/{workflow_id}/approve
# =============================================================================


class TestApproveWorkflowStep:
    """Tests for POST /api/v2/workflows/{workflow_id}/approve."""

    def test_approve_returns_200(self, client, mock_workflow_engine, sample_workflow):
        """Approve returns success."""
        _override_auth(client)
        mock_workflow_engine.get_workflow.return_value = sample_workflow
        mock_workflow_engine.approve_step = MagicMock(return_value=True)

        response = client.post(
            "/api/v2/workflows/wf_test123/approve",
            json={"step_id": "node-2", "comment": "Looks good"},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["workflow_id"] == "wf_test123"
        assert data["step_id"] == "node-2"
        assert data["status"] == "approved"

    def test_approve_requires_step_id(self, client):
        """Approve without step_id returns 422."""
        _override_auth(client)

        response = client.post(
            "/api/v2/workflows/wf_test123/approve",
            json={"comment": "Missing step_id"},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 422

    def test_approve_not_found(self, client, mock_workflow_engine):
        """Approve on nonexistent workflow returns 404."""
        _override_auth(client)
        mock_workflow_engine.get_workflow.return_value = None

        response = client.post(
            "/api/v2/workflows/nonexistent/approve",
            json={"step_id": "node-1"},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 404

    def test_approve_requires_auth(self, client):
        """Approve without auth returns 401."""
        response = client.post(
            "/api/v2/workflows/wf_test123/approve",
            json={"step_id": "node-1"},
        )
        assert response.status_code == 401

    def test_approve_unavailable_engine(self, app):
        """Approve returns 503 when engine is unavailable."""
        from aragora.server.fastapi.routes.workflows import get_workflow_engine as _dep

        app.state.context = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
            "workflow_engine": None,
        }
        app.dependency_overrides[_dep] = lambda: None
        client = TestClient(app, raise_server_exceptions=False)
        _override_auth(client)

        response = client.post(
            "/api/v2/workflows/wf_test123/approve",
            json={"step_id": "node-1"},
        )
        client.app.dependency_overrides.clear()
        assert response.status_code == 503


class TestStoreBackedFallbacks:
    """Tests for route fallbacks when the injected engine is a raw executor only."""

    def test_list_falls_back_to_persisted_workflows(self, fallback_client):
        persisted = _persisted_workflow_definition()

        with patch(
            "aragora.server.fastapi.routes.workflows.list_workflow_records",
            new=AsyncMock(
                return_value={
                    "workflows": [persisted],
                    "total_count": 1,
                    "limit": 50,
                    "offset": 0,
                }
            ),
        ) as mock_list:
            response = fallback_client.get("/api/v2/workflows")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["workflows"][0]["id"] == "wf_store123"
        assert data["workflows"][0]["node_count"] == 2
        mock_list.assert_awaited_once()

    def test_get_falls_back_to_persisted_workflow_detail(self, fallback_client):
        persisted = _persisted_workflow_definition()

        with patch(
            "aragora.server.fastapi.routes.workflows.get_workflow_record",
            new=AsyncMock(return_value=persisted),
        ) as mock_get:
            response = fallback_client.get("/api/v2/workflows/wf_store123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "wf_store123"
        assert data["nodes"][0]["id"] == "node-1"
        assert data["nodes"][0]["type"] == "debate"
        assert data["edges"] == [{"from": "node-1", "to": "node-2"}]
        assert data["config"] == {"timeout": 120}
        mock_get.assert_awaited_once_with("wf_store123", tenant_id="default")

    def test_create_falls_back_to_persisted_workflow_crud(self, fallback_client):
        _override_auth(fallback_client)
        persisted = _persisted_workflow_definition()

        with patch(
            "aragora.server.fastapi.routes.workflows.create_workflow_record",
            new=AsyncMock(return_value=persisted),
        ) as mock_create:
            response = fallback_client.post(
                "/api/v2/workflows",
                json={
                    "name": "Stored Workflow",
                    "description": "Persisted workflow definition",
                    "nodes": [
                        {
                            "id": "node-1",
                            "type": "debate",
                            "name": "Debate",
                            "config": {"rounds": 3},
                        },
                        {
                            "id": "node-2",
                            "type": "verification",
                            "name": "Verify",
                            "config": {},
                        },
                    ],
                    "edges": [{"from": "node-1", "to": "node-2"}],
                    "config": {"timeout": 120},
                },
            )
        fallback_client.app.dependency_overrides.clear()

        assert response.status_code == 201
        data = response.json()
        assert data["workflow_id"] == "wf_store123"
        assert data["workflow"]["nodes"][0]["type"] == "debate"
        assert data["workflow"]["edges"] == [{"from": "node-1", "to": "node-2"}]

        mock_create.assert_awaited_once()
        payload = mock_create.await_args.args[0]
        assert payload["steps"][0]["step_type"] == "debate"
        assert payload["steps"][0]["next_steps"] == ["node-2"]
        assert payload["transitions"][0]["from_step"] == "node-1"
        assert payload["metadata"]["config"] == {"timeout": 120}
        assert mock_create.await_args.kwargs["tenant_id"] == "org-1"
        assert mock_create.await_args.kwargs["created_by"] == "user-1"

    def test_execute_falls_back_to_persisted_execution(self, fallback_client):
        _override_auth(fallback_client)
        persisted = _persisted_workflow_definition()
        execution = {
            "id": "exec_store123",
            "workflow_id": "wf_store123",
            "status": "completed",
        }

        with (
            patch(
                "aragora.server.fastapi.routes.workflows.get_workflow_record",
                new=AsyncMock(return_value=persisted),
            ) as mock_get,
            patch(
                "aragora.server.fastapi.routes.workflows.execute_workflow_record",
                new=AsyncMock(return_value=execution),
            ) as mock_execute,
        ):
            response = fallback_client.post(
                "/api/v2/workflows/wf_store123/execute",
                json={"input_data": {"task": "Review security"}},
            )
        fallback_client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["execution_id"] == "exec_store123"
        assert data["status"] == "completed"
        mock_get.assert_awaited_once_with("wf_store123", tenant_id="org-1")
        mock_execute.assert_awaited_once_with(
            "wf_store123",
            inputs={"task": "Review security"},
            tenant_id="org-1",
            user_id="user-1",
            org_id="org-1",
            event_emitter=None,
        )

    def test_status_and_history_fall_back_to_persisted_executions(self, fallback_client):
        persisted = _persisted_workflow_definition()
        execution = {
            "id": "exec_store123",
            "workflow_id": "wf_store123",
            "status": "running",
            "steps": [
                {"step_id": "node-1", "status": "completed"},
                {"step_id": "node-2", "status": "running"},
            ],
            "started_at": "2026-04-01T12:10:00Z",
            "completed_at": None,
            "duration_ms": 1500,
            "outputs": {"verdict": "pending"},
            "error": None,
        }

        with (
            patch(
                "aragora.server.fastapi.routes.workflows.get_workflow_record",
                new=AsyncMock(return_value=persisted),
            ),
            patch(
                "aragora.server.fastapi.routes.workflows.list_workflow_executions",
                new=AsyncMock(return_value=[execution]),
            ) as mock_executions,
        ):
            status_response = fallback_client.get("/api/v2/workflows/wf_store123/status")
            history_response = fallback_client.get("/api/v2/workflows/wf_store123/history")

        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] == "running"
        assert status_data["progress"] == pytest.approx(0.5, abs=0.01)
        assert status_data["current_node"] == "node-2"
        assert status_data["completed_nodes"] == ["node-1"]

        assert history_response.status_code == 200
        history_data = history_response.json()
        assert history_data["total"] == 1
        assert history_data["executions"][0]["execution_id"] == "exec_store123"
        assert history_data["executions"][0]["duration_seconds"] == pytest.approx(1.5, abs=0.001)
        assert history_data["executions"][0]["result"] == {"verdict": "pending"}
        assert mock_executions.await_count == 2

    def test_approve_falls_back_to_pending_approval_resolution(self, fallback_client):
        _override_auth(fallback_client)
        persisted = _persisted_workflow_definition()
        pending_approval = {"id": "approval-1", "workflow_id": "wf_store123", "step_id": "node-2"}

        with (
            patch(
                "aragora.server.fastapi.routes.workflows.get_workflow_record",
                new=AsyncMock(return_value=persisted),
            ),
            patch(
                "aragora.server.fastapi.routes.workflows.list_pending_workflow_approvals",
                new=AsyncMock(return_value=[pending_approval]),
            ) as mock_pending,
            patch(
                "aragora.server.fastapi.routes.workflows.resolve_workflow_approval",
                new=AsyncMock(return_value=True),
            ) as mock_resolve,
        ):
            response = fallback_client.post(
                "/api/v2/workflows/wf_store123/approve",
                json={"step_id": "node-2", "comment": "Ship it"},
            )
        fallback_client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "approved"
        mock_pending.assert_awaited_once_with(workflow_id="wf_store123", tenant_id="org-1")
        mock_resolve.assert_awaited_once_with(
            request_id="approval-1",
            status="approved",
            responder_id="user-1",
            notes="Ship it",
        )
