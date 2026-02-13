"""Tests for Aragora SDK Workflows API."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora_client.workflows import (
    Workflow,
    WorkflowCheckpoint,
    WorkflowExecution,
    WorkflowsAPI,
    WorkflowStep,
    WorkflowTemplate,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock AragoraClient."""
    client = MagicMock()
    client._get = AsyncMock()
    client._post = AsyncMock()
    client._put = AsyncMock()
    client._delete = AsyncMock()
    return client


@pytest.fixture
def workflows_api(mock_client: MagicMock) -> WorkflowsAPI:
    """Create WorkflowsAPI with mock client."""
    return WorkflowsAPI(mock_client)


@pytest.fixture
def workflow_response() -> dict[str, Any]:
    """Standard workflow response."""
    return {
        "id": "workflow-123",
        "name": "Test Workflow",
        "description": "A test workflow",
        "category": "security",
        "status": "active",
        "steps": [
            {
                "id": "step-1",
                "name": "Start",
                "type": "trigger",
                "config": {"event": "debate_complete"},
            }
        ],
        "config": {"timeout": 300},
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "version": 1,
    }


@pytest.fixture
def template_response() -> dict[str, Any]:
    """Standard workflow template response."""
    return {
        "id": "template-123",
        "name": "Security Audit",
        "description": "Security review workflow",
        "category": "security",
        "industry": "technology",
        "steps": [
            {"id": "s1", "name": "Scan", "type": "action"},
            {"id": "s2", "name": "Review", "type": "approval"},
        ],
        "variables": [
            {"name": "target_url", "type": "string", "required": True},
        ],
    }


@pytest.fixture
def execution_response() -> dict[str, Any]:
    """Standard workflow execution response."""
    return {
        "id": "exec-123",
        "workflow_id": "workflow-456",
        "status": "running",
        "current_step": "step-2",
        "started_at": "2026-01-01T12:00:00Z",
        "inputs": {"document": "test.pdf"},
    }


@pytest.fixture
def checkpoint_response() -> dict[str, Any]:
    """Standard checkpoint response."""
    return {
        "id": "checkpoint-123",
        "execution_id": "exec-456",
        "step_id": "step-1",
        "state": {"progress": 50, "data": {"key": "value"}},
        "created_at": "2026-01-01T12:30:00Z",
    }


# =============================================================================
# Model Tests
# =============================================================================


class TestWorkflowStep:
    """Tests for WorkflowStep model."""

    def test_minimal_creation(self) -> None:
        """Test creating step with required fields only."""
        step = WorkflowStep(id="step-1", name="Start", type="trigger")
        assert step.id == "step-1"
        assert step.name == "Start"
        assert step.type == "trigger"
        assert step.config is None
        assert step.transitions is None

    def test_full_creation(self) -> None:
        """Test creating step with all fields."""
        step = WorkflowStep(
            id="step-2",
            name="Process",
            type="action",
            config={"action": "analyze", "timeout": 60},
            transitions=[{"on": "success", "to": "step-3"}],
        )
        assert step.config["action"] == "analyze"
        assert len(step.transitions) == 1


class TestWorkflow:
    """Tests for Workflow model."""

    def test_minimal_creation(self) -> None:
        """Test creating workflow with required fields only."""
        workflow = Workflow(id="wf-1", name="Test")
        assert workflow.id == "wf-1"
        assert workflow.name == "Test"
        assert workflow.status == "draft"
        assert workflow.version == 1

    def test_full_creation(self, workflow_response: dict[str, Any]) -> None:
        """Test creating workflow with all fields."""
        workflow = Workflow.model_validate(workflow_response)
        assert workflow.id == "workflow-123"
        assert workflow.name == "Test Workflow"
        assert workflow.category == "security"
        assert workflow.status == "active"
        assert len(workflow.steps) == 1
        assert workflow.steps[0].name == "Start"


class TestWorkflowTemplate:
    """Tests for WorkflowTemplate model."""

    def test_minimal_creation(self) -> None:
        """Test creating template with required fields only."""
        template = WorkflowTemplate(id="t-1", name="Basic")
        assert template.id == "t-1"
        assert template.name == "Basic"
        assert template.variables is None

    def test_full_creation(self, template_response: dict[str, Any]) -> None:
        """Test creating template with all fields."""
        template = WorkflowTemplate.model_validate(template_response)
        assert template.id == "template-123"
        assert template.industry == "technology"
        assert len(template.steps) == 2
        assert len(template.variables) == 1


class TestWorkflowExecution:
    """Tests for WorkflowExecution model."""

    def test_minimal_creation(self) -> None:
        """Test creating execution with required fields only."""
        execution = WorkflowExecution(id="e-1", workflow_id="wf-1")
        assert execution.id == "e-1"
        assert execution.workflow_id == "wf-1"
        assert execution.status == "pending"
        assert execution.error is None

    def test_full_creation(self, execution_response: dict[str, Any]) -> None:
        """Test creating execution with all fields."""
        execution = WorkflowExecution.model_validate(execution_response)
        assert execution.status == "running"
        assert execution.current_step == "step-2"
        assert execution.inputs["document"] == "test.pdf"


class TestWorkflowCheckpoint:
    """Tests for WorkflowCheckpoint model."""

    def test_creation(self, checkpoint_response: dict[str, Any]) -> None:
        """Test creating checkpoint."""
        checkpoint = WorkflowCheckpoint.model_validate(checkpoint_response)
        assert checkpoint.id == "checkpoint-123"
        assert checkpoint.execution_id == "exec-456"
        assert checkpoint.state["progress"] == 50


# =============================================================================
# WorkflowsAPI Tests - CRUD
# =============================================================================


class TestWorkflowsAPIList:
    """Tests for WorkflowsAPI.list()."""

    @pytest.mark.asyncio
    async def test_list_default_params(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        workflow_response: dict[str, Any],
    ) -> None:
        """Test listing workflows with default parameters."""
        mock_client._get.return_value = {"workflows": [workflow_response]}

        result = await workflows_api.list()

        mock_client._get.assert_called_once_with(
            "/api/workflows", params={"limit": 50, "offset": 0}
        )
        assert len(result) == 1
        assert isinstance(result[0], Workflow)

    @pytest.mark.asyncio
    async def test_list_with_filters(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test listing workflows with filters."""
        mock_client._get.return_value = {"workflows": []}

        await workflows_api.list(
            limit=10, offset=5, status="active", category="security"
        )

        mock_client._get.assert_called_once_with(
            "/api/workflows",
            params={
                "limit": 10,
                "offset": 5,
                "status": "active",
                "category": "security",
            },
        )

    @pytest.mark.asyncio
    async def test_list_empty(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test listing when no workflows exist."""
        mock_client._get.return_value = {}

        result = await workflows_api.list()

        assert result == []


class TestWorkflowsAPIGet:
    """Tests for WorkflowsAPI.get()."""

    @pytest.mark.asyncio
    async def test_get_workflow(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        workflow_response: dict[str, Any],
    ) -> None:
        """Test getting a workflow by ID."""
        mock_client._get.return_value = workflow_response

        result = await workflows_api.get("workflow-123")

        mock_client._get.assert_called_once_with("/api/workflows/workflow-123")
        assert isinstance(result, Workflow)
        assert result.id == "workflow-123"


class TestWorkflowsAPICreate:
    """Tests for WorkflowsAPI.create()."""

    @pytest.mark.asyncio
    async def test_create_minimal(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        workflow_response: dict[str, Any],
    ) -> None:
        """Test creating workflow with minimal parameters."""
        mock_client._post.return_value = workflow_response

        result = await workflows_api.create("New Workflow")

        mock_client._post.assert_called_once_with(
            "/api/workflows", {"name": "New Workflow"}
        )
        assert isinstance(result, Workflow)

    @pytest.mark.asyncio
    async def test_create_full(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        workflow_response: dict[str, Any],
    ) -> None:
        """Test creating workflow with all parameters."""
        mock_client._post.return_value = workflow_response
        steps = [{"id": "s1", "name": "Start", "type": "trigger"}]
        config = {"timeout": 600}

        await workflows_api.create(
            "Full Workflow",
            description="Complete workflow",
            category="automation",
            steps=steps,
            config=config,
        )

        mock_client._post.assert_called_once_with(
            "/api/workflows",
            {
                "name": "Full Workflow",
                "description": "Complete workflow",
                "category": "automation",
                "steps": steps,
                "config": config,
            },
        )


class TestWorkflowsAPIUpdate:
    """Tests for WorkflowsAPI.update()."""

    @pytest.mark.asyncio
    async def test_update_name(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        workflow_response: dict[str, Any],
    ) -> None:
        """Test updating workflow name."""
        mock_client._put.return_value = workflow_response

        result = await workflows_api.update("workflow-123", name="Updated Name")

        mock_client._put.assert_called_once_with(
            "/api/workflows/workflow-123", {"name": "Updated Name"}
        )
        assert isinstance(result, Workflow)

    @pytest.mark.asyncio
    async def test_update_multiple_fields(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        workflow_response: dict[str, Any],
    ) -> None:
        """Test updating multiple workflow fields."""
        mock_client._put.return_value = workflow_response

        await workflows_api.update(
            "workflow-123",
            name="New Name",
            description="New description",
            steps=[{"id": "s1", "name": "New Step", "type": "action"}],
            config={"new_config": True},
        )

        args = mock_client._put.call_args
        assert args[0][0] == "/api/workflows/workflow-123"
        body = args[0][1]
        assert body["name"] == "New Name"
        assert body["description"] == "New description"
        assert "steps" in body
        assert "config" in body


class TestWorkflowsAPIDelete:
    """Tests for WorkflowsAPI.delete()."""

    @pytest.mark.asyncio
    async def test_delete_workflow(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test deleting a workflow."""
        mock_client._delete.return_value = None

        await workflows_api.delete("workflow-123")

        mock_client._delete.assert_called_once_with("/api/workflows/workflow-123")


# =============================================================================
# WorkflowsAPI Tests - Execution
# =============================================================================


class TestWorkflowsAPIExecute:
    """Tests for WorkflowsAPI.execute()."""

    @pytest.mark.asyncio
    async def test_execute_minimal(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        execution_response: dict[str, Any],
    ) -> None:
        """Test executing workflow with minimal parameters."""
        mock_client._post.return_value = execution_response

        result = await workflows_api.execute("workflow-123")

        mock_client._post.assert_called_once_with(
            "/api/workflows/workflow-123/execute", {}
        )
        assert isinstance(result, WorkflowExecution)

    @pytest.mark.asyncio
    async def test_execute_with_inputs(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        execution_response: dict[str, Any],
    ) -> None:
        """Test executing workflow with inputs."""
        mock_client._post.return_value = execution_response
        inputs = {"document": "analysis.pdf", "threshold": 0.8}

        await workflows_api.execute("workflow-123", inputs=inputs)

        mock_client._post.assert_called_once_with(
            "/api/workflows/workflow-123/execute", {"inputs": inputs}
        )

    @pytest.mark.asyncio
    async def test_execute_async(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        execution_response: dict[str, Any],
    ) -> None:
        """Test async workflow execution."""
        mock_client._post.return_value = execution_response

        await workflows_api.execute("workflow-123", async_execution=True)

        mock_client._post.assert_called_once_with(
            "/api/workflows/workflow-123/execute", {"async": True}
        )


class TestWorkflowsAPIGetExecution:
    """Tests for WorkflowsAPI.get_execution()."""

    @pytest.mark.asyncio
    async def test_get_execution(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        execution_response: dict[str, Any],
    ) -> None:
        """Test getting execution status."""
        mock_client._get.return_value = execution_response

        result = await workflows_api.get_execution("exec-123")

        mock_client._get.assert_called_once_with("/api/workflow-executions/exec-123")
        assert isinstance(result, WorkflowExecution)
        assert result.status == "running"


class TestWorkflowsAPIListExecutions:
    """Tests for WorkflowsAPI.list_executions()."""

    @pytest.mark.asyncio
    async def test_list_executions_default(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        execution_response: dict[str, Any],
    ) -> None:
        """Test listing executions with default params."""
        mock_client._get.return_value = {"executions": [execution_response]}

        result = await workflows_api.list_executions()

        mock_client._get.assert_called_once_with(
            "/api/workflow-executions", params={"limit": 50, "offset": 0}
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_executions_filtered(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test listing executions with filters."""
        mock_client._get.return_value = {"executions": []}

        await workflows_api.list_executions(
            workflow_id="wf-1", status="completed", limit=10
        )

        mock_client._get.assert_called_once_with(
            "/api/workflow-executions",
            params={
                "limit": 10,
                "offset": 0,
                "workflow_id": "wf-1",
                "status": "completed",
            },
        )


class TestWorkflowsAPICancelExecution:
    """Tests for WorkflowsAPI.cancel_execution()."""

    @pytest.mark.asyncio
    async def test_cancel_execution(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test canceling an execution."""
        cancelled_response = {
            "id": "exec-123",
            "workflow_id": "wf-1",
            "status": "cancelled",
        }
        mock_client._post.return_value = cancelled_response

        result = await workflows_api.cancel_execution("exec-123")

        mock_client._post.assert_called_once_with(
            "/api/workflow-executions/exec-123/cancel", {}
        )
        assert result.status == "cancelled"


# =============================================================================
# WorkflowsAPI Tests - Templates
# =============================================================================


class TestWorkflowsAPIListTemplates:
    """Tests for WorkflowsAPI.list_templates()."""

    @pytest.mark.asyncio
    async def test_list_templates_default(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        template_response: dict[str, Any],
    ) -> None:
        """Test listing templates with defaults."""
        mock_client._get.return_value = {"templates": [template_response]}

        result = await workflows_api.list_templates()

        mock_client._get.assert_called_once_with(
            "/api/workflow-templates", params={"limit": 50, "offset": 0}
        )
        assert len(result) == 1
        assert isinstance(result[0], WorkflowTemplate)

    @pytest.mark.asyncio
    async def test_list_templates_filtered(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test listing templates with filters."""
        mock_client._get.return_value = {"templates": []}

        await workflows_api.list_templates(
            category="compliance", industry="finance", limit=20
        )

        mock_client._get.assert_called_once_with(
            "/api/workflow-templates",
            params={
                "limit": 20,
                "offset": 0,
                "category": "compliance",
                "industry": "finance",
            },
        )


class TestWorkflowsAPIGetTemplate:
    """Tests for WorkflowsAPI.get_template()."""

    @pytest.mark.asyncio
    async def test_get_template(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        template_response: dict[str, Any],
    ) -> None:
        """Test getting a template by ID."""
        mock_client._get.return_value = template_response

        result = await workflows_api.get_template("template-123")

        mock_client._get.assert_called_once_with("/api/workflow-templates/template-123")
        assert result.name == "Security Audit"


class TestWorkflowsAPIRunTemplate:
    """Tests for WorkflowsAPI.run_template()."""

    @pytest.mark.asyncio
    async def test_run_template_minimal(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        execution_response: dict[str, Any],
    ) -> None:
        """Test running a template with minimal params."""
        mock_client._post.return_value = execution_response

        result = await workflows_api.run_template("template-123")

        mock_client._post.assert_called_once_with(
            "/api/workflow-templates/template-123/run", {}
        )
        assert isinstance(result, WorkflowExecution)

    @pytest.mark.asyncio
    async def test_run_template_with_variables(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        execution_response: dict[str, Any],
    ) -> None:
        """Test running a template with variables and inputs."""
        mock_client._post.return_value = execution_response
        variables = {"target_url": "https://example.com"}
        inputs = {"depth": 3}

        await workflows_api.run_template(
            "template-123", variables=variables, inputs=inputs
        )

        mock_client._post.assert_called_once_with(
            "/api/workflow-templates/template-123/run",
            {"variables": variables, "inputs": inputs},
        )


class TestWorkflowsAPICategories:
    """Tests for WorkflowsAPI category operations."""

    @pytest.mark.asyncio
    async def test_list_categories(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test listing workflow categories."""
        mock_client._get.return_value = {
            "categories": ["security", "compliance", "automation"]
        }

        result = await workflows_api.list_categories()

        mock_client._get.assert_called_once_with("/api/workflow-categories")
        assert result == ["security", "compliance", "automation"]


class TestWorkflowsAPIPatterns:
    """Tests for WorkflowsAPI pattern operations."""

    @pytest.mark.asyncio
    async def test_list_patterns(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test listing workflow patterns."""
        mock_client._get.return_value = {
            "patterns": ["sequential", "parallel", "conditional"]
        }

        result = await workflows_api.list_patterns()

        mock_client._get.assert_called_once_with("/api/workflow-patterns")
        assert "parallel" in result


# =============================================================================
# WorkflowsAPI Tests - Checkpoints
# =============================================================================


class TestWorkflowsAPIListCheckpoints:
    """Tests for WorkflowsAPI.list_checkpoints()."""

    @pytest.mark.asyncio
    async def test_list_checkpoints(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        checkpoint_response: dict[str, Any],
    ) -> None:
        """Test listing checkpoints for an execution."""
        mock_client._get.return_value = {"checkpoints": [checkpoint_response]}

        result = await workflows_api.list_checkpoints("exec-123")

        mock_client._get.assert_called_once_with(
            "/api/workflow-executions/exec-123/checkpoints",
            params={"limit": 50, "offset": 0},
        )
        assert len(result) == 1
        assert isinstance(result[0], WorkflowCheckpoint)

    @pytest.mark.asyncio
    async def test_list_checkpoints_paginated(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
    ) -> None:
        """Test listing checkpoints with pagination."""
        mock_client._get.return_value = {"checkpoints": []}

        await workflows_api.list_checkpoints("exec-123", limit=10, offset=5)

        mock_client._get.assert_called_once_with(
            "/api/workflow-executions/exec-123/checkpoints",
            params={"limit": 10, "offset": 5},
        )


class TestWorkflowsAPIRestoreCheckpoint:
    """Tests for WorkflowsAPI.restore_checkpoint()."""

    @pytest.mark.asyncio
    async def test_restore_checkpoint(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        execution_response: dict[str, Any],
    ) -> None:
        """Test restoring execution to a checkpoint."""
        mock_client._post.return_value = execution_response

        result = await workflows_api.restore_checkpoint("exec-123", "checkpoint-456")

        mock_client._post.assert_called_once_with(
            "/api/workflow-executions/exec-123/checkpoints/checkpoint-456/restore",
            {},
        )
        assert isinstance(result, WorkflowExecution)


# =============================================================================
# WorkflowsAPI Tests - Versions
# =============================================================================


class TestWorkflowsAPIListVersions:
    """Tests for WorkflowsAPI.list_versions()."""

    @pytest.mark.asyncio
    async def test_list_versions(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        workflow_response: dict[str, Any],
    ) -> None:
        """Test listing workflow versions."""
        v1 = {**workflow_response, "version": 1}
        v2 = {**workflow_response, "version": 2}
        mock_client._get.return_value = {"versions": [v1, v2]}

        result = await workflows_api.list_versions("workflow-123")

        mock_client._get.assert_called_once_with(
            "/api/workflows/workflow-123/versions",
            params={"limit": 20, "offset": 0},
        )
        assert len(result) == 2
        assert result[0].version == 1
        assert result[1].version == 2


class TestWorkflowsAPIGetVersion:
    """Tests for WorkflowsAPI.get_version()."""

    @pytest.mark.asyncio
    async def test_get_version(
        self,
        workflows_api: WorkflowsAPI,
        mock_client: MagicMock,
        workflow_response: dict[str, Any],
    ) -> None:
        """Test getting a specific workflow version."""
        mock_client._get.return_value = {**workflow_response, "version": 3}

        result = await workflows_api.get_version("workflow-123", 3)

        mock_client._get.assert_called_once_with(
            "/api/workflows/workflow-123/versions/3"
        )
        assert result.version == 3
