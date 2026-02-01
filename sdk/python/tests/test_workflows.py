"""Tests for Workflows namespace API."""

from __future__ import annotations

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestWorkflowsList:
    """Tests for listing workflows."""

    def test_list_workflows_default(self, client: AragoraClient, mock_request) -> None:
        """List workflows with default pagination."""
        mock_request.return_value = {"workflows": [], "total": 0}

        client.workflows.list()

        mock_request.assert_called_once_with(
            "GET", "/api/v1/workflows", params={"limit": 20, "offset": 0}
        )

    def test_list_workflows_custom_pagination(self, client: AragoraClient, mock_request) -> None:
        """List workflows with custom pagination."""
        mock_request.return_value = {"workflows": [], "total": 100}

        client.workflows.list(limit=50, offset=25)

        mock_request.assert_called_once_with(
            "GET", "/api/v1/workflows", params={"limit": 50, "offset": 25}
        )


class TestWorkflowsGet:
    """Tests for getting workflow details."""

    def test_get_workflow(self, client: AragoraClient, mock_request) -> None:
        """Get a workflow by ID."""
        mock_request.return_value = {
            "workflow_id": "wf_123",
            "name": "Deploy Pipeline",
            "steps": [],
        }

        result = client.workflows.get("wf_123")

        mock_request.assert_called_once_with("GET", "/api/v1/workflows/wf_123")
        assert result["name"] == "Deploy Pipeline"


class TestWorkflowsCreate:
    """Tests for workflow creation."""

    def test_create_workflow_minimal(self, client: AragoraClient, mock_request) -> None:
        """Create a workflow with required fields only."""
        mock_request.return_value = {"workflow_id": "wf_new"}

        steps = [{"action": "debate", "config": {"task": "Review PR"}}]
        result = client.workflows.create(name="PR Review", steps=steps)

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/workflows",
            json={"name": "PR Review", "steps": steps},
        )
        assert result["workflow_id"] == "wf_new"

    def test_create_workflow_with_description(self, client: AragoraClient, mock_request) -> None:
        """Create a workflow with a description."""
        mock_request.return_value = {"workflow_id": "wf_new"}

        steps = [{"action": "analyze"}]
        client.workflows.create(
            name="Analysis Flow",
            steps=steps,
            description="Automated analysis workflow",
        )

        call_json = mock_request.call_args[1]["json"]
        assert call_json["description"] == "Automated analysis workflow"

    def test_create_workflow_with_extra_kwargs(self, client: AragoraClient, mock_request) -> None:
        """Create a workflow with additional keyword arguments."""
        mock_request.return_value = {"workflow_id": "wf_new"}

        steps = [{"action": "notify"}]
        client.workflows.create(
            name="Alert Flow",
            steps=steps,
            schedule="0 9 * * *",
            workspace_id="ws_1",
        )

        call_json = mock_request.call_args[1]["json"]
        assert call_json["schedule"] == "0 9 * * *"
        assert call_json["workspace_id"] == "ws_1"


class TestWorkflowsUpdate:
    """Tests for workflow updates."""

    def test_update_workflow(self, client: AragoraClient, mock_request) -> None:
        """Update a workflow."""
        mock_request.return_value = {"workflow_id": "wf_123", "name": "Updated"}

        client.workflows.update("wf_123", name="Updated", enabled=True)

        mock_request.assert_called_once_with(
            "PUT",
            "/api/v1/workflows/wf_123",
            json={"name": "Updated", "enabled": True},
        )


class TestWorkflowsDelete:
    """Tests for workflow deletion."""

    def test_delete_workflow(self, client: AragoraClient, mock_request) -> None:
        """Delete a workflow."""
        mock_request.return_value = {"deleted": True}

        result = client.workflows.delete("wf_123")

        mock_request.assert_called_once_with("DELETE", "/api/v1/workflows/wf_123")
        assert result["deleted"] is True


class TestWorkflowsExecution:
    """Tests for workflow execution operations."""

    def test_execute_workflow(self, client: AragoraClient, mock_request) -> None:
        """Execute a workflow."""
        mock_request.return_value = {"execution_id": "exec_1", "status": "running"}

        result = client.workflows.execute("wf_123", inputs={"param": "value"})

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/workflows/wf_123/execute",
            json={"inputs": {"param": "value"}},
        )
        assert result["execution_id"] == "exec_1"

    def test_execute_workflow_no_inputs(self, client: AragoraClient, mock_request) -> None:
        """Execute a workflow without inputs."""
        mock_request.return_value = {"execution_id": "exec_2"}

        client.workflows.execute("wf_123")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/workflows/wf_123/execute",
            json={"inputs": {}},
        )

    def test_get_execution(self, client: AragoraClient, mock_request) -> None:
        """Get execution status."""
        mock_request.return_value = {
            "execution_id": "exec_1",
            "status": "completed",
            "output": {"result": "success"},
        }

        result = client.workflows.get_execution("exec_1")

        mock_request.assert_called_once_with("GET", "/api/v1/workflow-executions/exec_1")
        assert result["status"] == "completed"

    def test_list_executions_default(self, client: AragoraClient, mock_request) -> None:
        """List executions with defaults."""
        mock_request.return_value = {"executions": [], "total": 0}

        client.workflows.list_executions()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/workflow-executions",
            params={"limit": 20, "offset": 0},
        )

    def test_list_executions_filtered(self, client: AragoraClient, mock_request) -> None:
        """List executions filtered by workflow ID."""
        mock_request.return_value = {"executions": []}

        client.workflows.list_executions(workflow_id="wf_123", limit=10)

        call_params = mock_request.call_args[1]["params"]
        assert call_params["workflow_id"] == "wf_123"
        assert call_params["limit"] == 10

    def test_cancel_execution(self, client: AragoraClient, mock_request) -> None:
        """Cancel a workflow execution."""
        mock_request.return_value = {"cancelled": True}

        result = client.workflows.cancel_execution("exec_1")

        mock_request.assert_called_once_with("DELETE", "/api/v1/workflow-executions/exec_1")
        assert result["cancelled"] is True


class TestWorkflowsTemplates:
    """Tests for workflow template operations."""

    def test_list_templates(self, client: AragoraClient, mock_request) -> None:
        """List available workflow templates."""
        mock_request.return_value = {"templates": []}

        client.workflows.list_templates()

        mock_request.assert_called_once_with("GET", "/api/v1/workflow-templates")

    def test_create_from_template(self, client: AragoraClient, mock_request) -> None:
        """Create a workflow from a template."""
        mock_request.return_value = {"workflow_id": "wf_from_tpl"}

        client.workflows.create_from_template("tpl_1", "My Workflow")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/workflow-templates/tpl_1/create",
            json={"name": "My Workflow"},
        )

    def test_create_from_template_with_overrides(self, client: AragoraClient, mock_request) -> None:
        """Create from template with overrides."""
        mock_request.return_value = {"workflow_id": "wf_custom"}

        client.workflows.create_from_template(
            "tpl_1", "Custom Flow", schedule="daily", max_retries=5
        )

        call_json = mock_request.call_args[1]["json"]
        assert call_json["name"] == "Custom Flow"
        assert call_json["schedule"] == "daily"
        assert call_json["max_retries"] == 5

    def test_list_library_templates(self, client: AragoraClient, mock_request) -> None:
        """List workflow library templates."""
        mock_request.return_value = {"templates": []}

        client.workflows.list_library_templates()

        mock_request.assert_called_once_with("GET", "/api/v1/workflow/templates", params={})

    def test_get_library_template(self, client: AragoraClient, mock_request) -> None:
        """Get a workflow library template."""
        mock_request.return_value = {"template_id": "tpl_1", "name": "PR Review"}

        result = client.workflows.get_library_template("tpl_1")

        mock_request.assert_called_once_with("GET", "/api/v1/workflow/templates/tpl_1")
        assert result["name"] == "PR Review"

    def test_run_library_template(self, client: AragoraClient, mock_request) -> None:
        """Run a workflow library template."""
        mock_request.return_value = {"execution_id": "exec_tpl"}

        client.workflows.run_library_template("tpl_1", inputs={"repo": "my-repo"})

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/workflow/templates/tpl_1/run",
            json={"inputs": {"repo": "my-repo"}},
        )

    def test_list_template_categories(self, client: AragoraClient, mock_request) -> None:
        """List workflow template categories."""
        mock_request.return_value = {"categories": ["analysis", "deployment"]}

        result = client.workflows.list_template_categories()

        mock_request.assert_called_once_with("GET", "/api/v1/workflow/categories")
        assert "analysis" in result["categories"]

    def test_list_pattern_templates(self, client: AragoraClient, mock_request) -> None:
        """List workflow pattern templates."""
        mock_request.return_value = {"templates": []}

        client.workflows.list_pattern_templates()

        mock_request.assert_called_once_with("GET", "/api/v1/workflow/pattern-templates")

    def test_get_pattern_template(self, client: AragoraClient, mock_request) -> None:
        """Get a workflow pattern template by ID."""
        mock_request.return_value = {"template_id": "pt_1"}

        client.workflows.get_pattern_template("pt_1")

        mock_request.assert_called_once_with("GET", "/api/v1/workflow/pattern-templates/pt_1")


class TestAsyncWorkflows:
    """Tests for async workflows API."""

    @pytest.mark.asyncio
    async def test_async_list_workflows(self, mock_async_request) -> None:
        """List workflows asynchronously."""
        mock_async_request.return_value = {"workflows": [], "total": 0}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.workflows.list()

            mock_async_request.assert_called_once_with(
                "GET", "/api/v1/workflows", params={"limit": 20, "offset": 0}
            )

    @pytest.mark.asyncio
    async def test_async_create_workflow(self, mock_async_request) -> None:
        """Create a workflow asynchronously."""
        mock_async_request.return_value = {"workflow_id": "wf_async"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            steps = [{"action": "debate"}]
            result = await client.workflows.create("Async Flow", steps)

            assert result["workflow_id"] == "wf_async"

    @pytest.mark.asyncio
    async def test_async_execute_workflow(self, mock_async_request) -> None:
        """Execute a workflow asynchronously."""
        mock_async_request.return_value = {"execution_id": "exec_async"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.workflows.execute("wf_123", inputs={"key": "val"})

            assert result["execution_id"] == "exec_async"

    @pytest.mark.asyncio
    async def test_async_cancel_execution(self, mock_async_request) -> None:
        """Cancel execution asynchronously."""
        mock_async_request.return_value = {"cancelled": True}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.workflows.cancel_execution("exec_1")

            mock_async_request.assert_called_once_with(
                "DELETE", "/api/v1/workflow-executions/exec_1"
            )
            assert result["cancelled"] is True
