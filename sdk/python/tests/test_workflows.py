"""Tests for Workflows namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


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


class TestWorkflowTemplates:
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
