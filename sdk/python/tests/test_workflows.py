"""Tests for Workflows namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestWorkflowsCRUD:
    """Tests for basic workflow CRUD operations."""

    def test_list_workflows_default(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"workflows": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.list()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/workflows",
                params={"limit": 20, "offset": 0},
            )
            assert result["total"] == 0
            client.close()

    def test_list_workflows_with_pagination(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"workflows": [{"id": "wf_1"}], "total": 50}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.list(limit=5, offset=10)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/workflows",
                params={"limit": 5, "offset": 10},
            )
            assert len(result["workflows"]) == 1
            client.close()

    def test_get_workflow(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "wf_abc", "name": "Deploy Pipeline"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.get("wf_abc")
            mock_request.assert_called_once_with("GET", "/api/v1/workflows/wf_abc")
            assert result["name"] == "Deploy Pipeline"
            client.close()

    def test_create_workflow_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "wf_new", "name": "Data Pipeline"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            steps = [{"action": "fetch", "config": {"url": "https://example.com"}}]
            result = client.workflows.create("Data Pipeline", steps)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflows",
                json={"name": "Data Pipeline", "steps": steps},
            )
            assert result["id"] == "wf_new"
            client.close()

    def test_create_workflow_with_description(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "wf_new", "name": "ETL Job"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            steps = [{"action": "extract"}, {"action": "transform"}, {"action": "load"}]
            client.workflows.create("ETL Job", steps, description="Daily data ETL pipeline")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflows",
                json={
                    "name": "ETL Job",
                    "steps": steps,
                    "description": "Daily data ETL pipeline",
                },
            )
            client.close()

    def test_delete_workflow(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.delete("wf_old")
            mock_request.assert_called_once_with("DELETE", "/api/v1/workflows/wf_old")
            assert result["deleted"] is True
            client.close()


class TestWorkflowExecution:
    """Tests for workflow execution operations."""

    def test_execute_workflow_with_inputs(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"execution_id": "exec_123", "status": "running"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.execute("wf_abc", inputs={"query": "test input"})
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflows/wf_abc/execute",
                json={"inputs": {"query": "test input"}},
            )
            assert result["status"] == "running"
            client.close()

    def test_execute_workflow_without_inputs(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"execution_id": "exec_456", "status": "running"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.workflows.execute("wf_abc")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflows/wf_abc/execute",
                json={"inputs": {}},
            )
            client.close()

    def test_get_execution_status(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "exec_123",
                "status": "completed",
                "output": {"result": "success"},
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.get_execution("exec_123")
            mock_request.assert_called_once_with("GET", "/api/v1/workflow-executions/exec_123")
            assert result["status"] == "completed"
            client.close()

    def test_list_executions_filtered_by_workflow(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"executions": [{"id": "exec_1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.workflows.list_executions(workflow_id="wf_abc", limit=10, offset=5)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/workflow-executions",
                params={"limit": 10, "offset": 5, "workflow_id": "wf_abc"},
            )
            client.close()

    def test_cancel_execution(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"cancelled": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.cancel_execution("exec_789")
            mock_request.assert_called_once_with("DELETE", "/api/v1/workflow-executions/exec_789")
            assert result["cancelled"] is True
            client.close()


class TestWorkflowTemplateOverrides:
    """Tests for workflow template operations."""

    def test_create_from_template_with_overrides(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "wf_from_tpl", "name": "My CI Pipeline"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.create_from_template("tpl_1", "My CI Pipeline", timeout=300)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflow-templates/tpl_1/create",
                json={"name": "My CI Pipeline", "timeout": 300},
            )
            assert result["name"] == "My CI Pipeline"
            client.close()


class TestAsyncWorkflows:
    """Tests for async workflow methods."""

    @pytest.mark.asyncio
    async def test_list_workflows(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"workflows": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.workflows.list()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/workflows",
                params={"limit": 20, "offset": 0},
            )
            assert result["total"] == 0
            await client.close()

    @pytest.mark.asyncio
    async def test_execute_workflow(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"execution_id": "exec_async", "status": "running"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.workflows.execute("wf_123", inputs={"key": "value"})
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflows/wf_123/execute",
                json={"inputs": {"key": "value"}},
            )
            assert result["execution_id"] == "exec_async"
            await client.close()

    @pytest.mark.asyncio
    async def test_get_execution(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "exec_1", "status": "completed"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.workflows.get_execution("exec_1")
            mock_request.assert_called_once_with("GET", "/api/v1/workflow-executions/exec_1")
            assert result["status"] == "completed"
            await client.close()


class TestWorkflowLibraryTemplates:
    """Tests for workflow library template operations."""

    def test_list_library_templates(self) -> None:
        """List workflow library templates."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"templates": [{"id": "tpl_lib_1"}], "total": 1}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.list_library_templates(category="automation")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/workflow/templates",
                params={"category": "automation"},
            )
            assert result["total"] == 1
            client.close()

    def test_get_library_template(self) -> None:
        """Get a specific workflow library template."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "tpl_lib_1", "name": "ETL Template"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.get_library_template("tpl_lib_1")
            mock_request.assert_called_once_with("GET", "/api/v1/workflow/templates/tpl_lib_1")
            assert result["name"] == "ETL Template"
            client.close()

    def test_get_library_package(self) -> None:
        """Get a workflow library template package."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"package": {"steps": [], "config": {}}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.get_library_package("tpl_lib_1")
            mock_request.assert_called_once_with(
                "GET", "/api/v1/workflow/templates/tpl_lib_1/package"
            )
            assert "package" in result
            client.close()

    def test_run_library_template(self) -> None:
        """Run a workflow library template."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"execution_id": "exec_tpl_1", "status": "running"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.run_library_template("tpl_lib_1", inputs={"source": "db"})
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflow/templates/tpl_lib_1/run",
                json={"inputs": {"source": "db"}},
            )
            assert result["status"] == "running"
            client.close()

    def test_run_library_template_without_inputs(self) -> None:
        """Run a workflow library template without inputs."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"execution_id": "exec_tpl_2"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.workflows.run_library_template("tpl_lib_1")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflow/templates/tpl_lib_1/run",
                json={"inputs": {}},
            )
            client.close()


class TestWorkflowCategoriesAndPatterns:
    """Tests for workflow categories and patterns operations."""

    def test_list_template_categories(self) -> None:
        """List workflow template categories."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "categories": [
                    {"id": "automation", "name": "Automation"},
                    {"id": "data", "name": "Data Processing"},
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.list_template_categories()
            mock_request.assert_called_once_with("GET", "/api/v1/workflow/categories")
            assert len(result["categories"]) == 2
            client.close()

    def test_list_template_patterns(self) -> None:
        """List workflow template patterns."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "patterns": [
                    {"id": "sequential", "name": "Sequential"},
                    {"id": "parallel", "name": "Parallel"},
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.list_template_patterns()
            mock_request.assert_called_once_with("GET", "/api/v1/workflow/patterns")
            assert len(result["patterns"]) == 2
            client.close()

    def test_list_pattern_templates(self) -> None:
        """List workflow pattern templates."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"templates": [{"id": "pt_1", "pattern": "sequential"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.list_pattern_templates()
            mock_request.assert_called_once_with("GET", "/api/v1/workflow/pattern-templates")
            assert result["templates"][0]["pattern"] == "sequential"
            client.close()

    def test_get_pattern_template(self) -> None:
        """Get a specific workflow pattern template."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "pt_1", "pattern": "parallel", "steps": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.get_pattern_template("pt_1")
            mock_request.assert_called_once_with("GET", "/api/v1/workflow/pattern-templates/pt_1")
            assert result["pattern"] == "parallel"
            client.close()


class TestAsyncWorkflowsExtended:
    """Extended tests for async workflow methods."""

    @pytest.mark.asyncio
    async def test_async_list_library_templates(self) -> None:
        """List library templates asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"templates": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.workflows.list_library_templates()
            mock_request.assert_called_once_with("GET", "/api/v1/workflow/templates", params={})
            assert result["total"] == 0
            await client.close()

    @pytest.mark.asyncio
    async def test_async_run_library_template(self) -> None:
        """Run library template asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"execution_id": "exec_async"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.workflows.run_library_template("tpl_1", {"key": "val"})
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflow/templates/tpl_1/run",
                json={"inputs": {"key": "val"}},
            )
            assert result["execution_id"] == "exec_async"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_list_template_categories(self) -> None:
        """List template categories asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"categories": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.workflows.list_template_categories()
            mock_request.assert_called_once_with("GET", "/api/v1/workflow/categories")
            assert "categories" in result
            await client.close()

    @pytest.mark.asyncio
    async def test_async_cancel_execution(self) -> None:
        """Cancel execution asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"cancelled": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.workflows.cancel_execution("exec_123")
            mock_request.assert_called_once_with("DELETE", "/api/v1/workflow-executions/exec_123")
            assert result["cancelled"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_async_create_workflow(self) -> None:
        """Create workflow asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "wf_async", "name": "Async Workflow"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.workflows.create(
                name="Async Workflow",
                steps=[{"action": "notify"}],
                description="Created asynchronously",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflows",
                json={
                    "name": "Async Workflow",
                    "steps": [{"action": "notify"}],
                    "description": "Created asynchronously",
                },
            )
            assert result["name"] == "Async Workflow"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_update_workflow(self) -> None:
        """Update workflow asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "wf_1", "name": "Updated Name"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.workflows.update("wf_1", name="Updated Name")
            mock_request.assert_called_once_with(
                "PUT",
                "/api/v1/workflows/wf_1",
                json={"name": "Updated Name"},
            )
            assert result["name"] == "Updated Name"
            await client.close()

    @pytest.mark.asyncio
    async def test_async_delete_workflow(self) -> None:
        """Delete workflow asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.workflows.delete("wf_1")
            mock_request.assert_called_once_with("DELETE", "/api/v1/workflows/wf_1")
            assert result["deleted"] is True
            await client.close()


class TestWorkflowUpdate:
    """Tests for workflow update operations."""

    def test_update_workflow_name(self) -> None:
        """Update workflow name."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "wf_1", "name": "New Name"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.update("wf_1", name="New Name")
            mock_request.assert_called_once_with(
                "PUT", "/api/v1/workflows/wf_1", json={"name": "New Name"}
            )
            assert result["name"] == "New Name"
            client.close()

    def test_update_workflow_multiple_fields(self) -> None:
        """Update multiple workflow fields."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "wf_1", "name": "Updated", "enabled": False}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.update("wf_1", name="Updated", enabled=False)
            mock_request.assert_called_once_with(
                "PUT", "/api/v1/workflows/wf_1", json={"name": "Updated", "enabled": False}
            )
            assert result["enabled"] is False
            client.close()


class TestWorkflowExecutionListing:
    """Tests for listing workflow executions."""

    def test_list_executions_without_workflow_filter(self) -> None:
        """List all executions without workflow filter."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"executions": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.list_executions()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/workflow-executions",
                params={"limit": 20, "offset": 0},
            )
            assert result["total"] == 0
            client.close()

    def test_list_executions_with_custom_pagination(self) -> None:
        """List executions with custom pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"executions": [{"id": "e1"}, {"id": "e2"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.list_executions(limit=50, offset=100)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/workflow-executions",
                params={"limit": 50, "offset": 100},
            )
            assert len(result["executions"]) == 2
            client.close()


class TestWorkflowTemplateCatalog:
    """Tests for workflow template operations."""

    def test_list_templates(self) -> None:
        """List workflow templates."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"templates": [{"id": "tpl_1"}, {"id": "tpl_2"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.list_templates()
            mock_request.assert_called_once_with("GET", "/api/v1/workflow-templates")
            assert len(result["templates"]) == 2
            client.close()

    def test_create_from_template_minimal(self) -> None:
        """Create workflow from template with minimal parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "wf_new", "name": "From Template"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.workflows.create_from_template("tpl_1", "From Template")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/workflow-templates/tpl_1/create",
                json={"name": "From Template"},
            )
            assert result["name"] == "From Template"
            client.close()
