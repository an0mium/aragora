"""Tests for MCP workflow tools execution logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.workflow import (
    cancel_workflow_tool,
    get_workflow_status_tool,
    list_workflow_templates_tool,
    run_workflow_tool,
)


class TestRunWorkflowTool:
    """Tests for run_workflow_tool."""

    @pytest.mark.asyncio
    async def test_run_invalid_json_inputs(self):
        """Test run with invalid JSON inputs."""
        result = await run_workflow_tool(template="test", inputs="not json")
        assert "error" in result
        assert "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_run_non_dict_inputs(self):
        """Test run with non-dict JSON inputs."""
        result = await run_workflow_tool(template="test", inputs='["a", "b"]')
        assert "error" in result
        assert "JSON object" in result["error"]

    @pytest.mark.asyncio
    async def test_run_template_not_found(self):
        """Test run with non-existent template."""
        with patch(
            "aragora.mcp.tools_module.workflow.get_template",
            return_value=None,
        ), patch("aragora.mcp.tools_module.workflow.WorkflowEngine"):
            result = await run_workflow_tool(template="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_sync_success(self):
        """Test successful synchronous workflow execution."""
        mock_template = MagicMock()
        mock_engine = AsyncMock()
        mock_result = MagicMock()
        mock_result.outputs = {"answer": "42"}
        mock_result.duration_seconds = 5.0
        mock_result.nodes_executed = 3
        mock_engine.run.return_value = mock_result

        with patch(
            "aragora.mcp.tools_module.workflow.get_template",
            return_value=mock_template,
        ), patch(
            "aragora.mcp.tools_module.workflow.WorkflowEngine",
            return_value=mock_engine,
        ):
            result = await run_workflow_tool(
                template="debate_and_audit",
                inputs='{"question": "test"}',
                async_execution=False,
            )

        assert result["status"] == "completed"
        assert result["outputs"]["answer"] == "42"
        assert result["duration_seconds"] == 5.0

    @pytest.mark.asyncio
    async def test_run_async_success(self):
        """Test successful asynchronous workflow execution."""
        mock_template = MagicMock()
        mock_engine = AsyncMock()
        mock_engine.start_async.return_value = "exec-abc123"

        with patch(
            "aragora.mcp.tools_module.workflow.get_template",
            return_value=mock_template,
        ), patch(
            "aragora.mcp.tools_module.workflow.WorkflowEngine",
            return_value=mock_engine,
        ):
            result = await run_workflow_tool(
                template="multi_agent_review",
                async_execution=True,
            )

        assert result["execution_id"] == "exec-abc123"
        assert result["status"] == "started"
        assert result["async"] is True

    @pytest.mark.asyncio
    async def test_run_import_error(self):
        """Test run when workflow engine unavailable."""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("Workflow engine not available"),
        ):
            result = await run_workflow_tool(template="test")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_run_empty_inputs_default(self):
        """Test run with empty inputs defaults to empty dict."""
        mock_template = MagicMock()
        mock_engine = AsyncMock()
        mock_result = MagicMock()
        mock_result.outputs = {}
        mock_result.duration_seconds = 1.0
        mock_result.nodes_executed = 1
        mock_engine.run.return_value = mock_result

        with patch(
            "aragora.mcp.tools_module.workflow.get_template",
            return_value=mock_template,
        ), patch(
            "aragora.mcp.tools_module.workflow.WorkflowEngine",
            return_value=mock_engine,
        ):
            result = await run_workflow_tool(template="test", inputs="")

        assert result["status"] == "completed"


class TestGetWorkflowStatusTool:
    """Tests for get_workflow_status_tool."""

    @pytest.mark.asyncio
    async def test_status_not_found(self):
        """Test status for non-existent execution."""
        mock_engine = AsyncMock()
        mock_engine.get_status.return_value = None

        with patch(
            "aragora.mcp.tools_module.workflow.WorkflowEngine",
            return_value=mock_engine,
        ):
            result = await get_workflow_status_tool(execution_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_status_success(self):
        """Test successful status retrieval."""
        mock_engine = AsyncMock()
        from aragora.mcp.tools_module.workflow import WorkflowExecutionStatus

        mock_status = WorkflowExecutionStatus(
            status="running",
            progress=0.5,
            current_node="audit_node",
            started_at="2025-01-01T00:00:00",
            completed_at=None,
            error=None,
        )
        mock_engine.get_status.return_value = mock_status

        with patch(
            "aragora.mcp.tools_module.workflow.WorkflowEngine",
            return_value=mock_engine,
        ):
            result = await get_workflow_status_tool(execution_id="exec-123")

        assert result["execution_id"] == "exec-123"
        assert result["status"] == "running"
        assert result["progress"] == 0.5
        assert result["current_node"] == "audit_node"


class TestListWorkflowTemplatesTool:
    """Tests for list_workflow_templates_tool."""

    @pytest.mark.asyncio
    async def test_list_import_error_returns_defaults(self):
        """Test list returns default templates when module unavailable."""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("Workflow patterns not available"),
        ):
            result = await list_workflow_templates_tool()

        assert "templates" in result
        assert result["count"] >= 1
        assert "note" in result

    @pytest.mark.asyncio
    async def test_list_success(self):
        """Test successful template listing."""
        mock_templates = [
            {
                "name": "debate_and_audit",
                "description": "Run debate then audit",
                "category": "debate",
                "inputs": {"question": "str", "documents": "list"},
                "outputs": {"decision": "str"},
            },
        ]

        with patch(
            "aragora.mcp.tools_module.workflow.list_templates",
            return_value=mock_templates,
        ), patch("aragora.mcp.tools_module.workflow.WorkflowEngine"):
            result = await list_workflow_templates_tool(category="debate")

        assert result["count"] == 1
        assert result["templates"][0]["name"] == "debate_and_audit"
        assert result["category"] == "debate"

    @pytest.mark.asyncio
    async def test_list_all_category(self):
        """Test list with 'all' category passes None to list_templates."""
        with patch(
            "aragora.mcp.tools_module.workflow.list_templates",
            return_value=[],
        ) as mock_list, patch("aragora.mcp.tools_module.workflow.WorkflowEngine"):
            result = await list_workflow_templates_tool(category="all")

        mock_list.assert_called_once_with(None)


class TestCancelWorkflowTool:
    """Tests for cancel_workflow_tool."""

    @pytest.mark.asyncio
    async def test_cancel_success(self):
        """Test successful workflow cancellation."""
        mock_engine = AsyncMock()
        mock_engine.cancel.return_value = True

        with patch(
            "aragora.mcp.tools_module.workflow.WorkflowEngine",
            return_value=mock_engine,
        ):
            result = await cancel_workflow_tool(
                execution_id="exec-123", reason="User requested"
            )

        assert result["cancelled"] is True
        assert result["execution_id"] == "exec-123"
        assert result["reason"] == "User requested"

    @pytest.mark.asyncio
    async def test_cancel_not_found(self):
        """Test cancel for non-existent workflow."""
        mock_engine = AsyncMock()
        mock_engine.cancel.return_value = False

        with patch(
            "aragora.mcp.tools_module.workflow.WorkflowEngine",
            return_value=mock_engine,
        ):
            result = await cancel_workflow_tool(execution_id="nonexistent")

        assert result["cancelled"] is False

    @pytest.mark.asyncio
    async def test_cancel_default_reason(self):
        """Test cancel with default reason."""
        mock_engine = AsyncMock()
        mock_engine.cancel.return_value = True

        with patch(
            "aragora.mcp.tools_module.workflow.WorkflowEngine",
            return_value=mock_engine,
        ):
            result = await cancel_workflow_tool(execution_id="exec-123")

        assert result["reason"] == "User requested"
