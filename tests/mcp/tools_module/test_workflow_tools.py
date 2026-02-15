"""Tests for MCP workflow tools execution logic."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.workflow import (
    WorkflowExecutionStatus,
    cancel_workflow_tool,
    get_workflow_status_tool,
    list_workflow_templates_tool,
    run_workflow_tool,
)



def _make_engine_module(mock_engine_instance):
    """Create a mock module for aragora.workflow.engine with WorkflowEngine returning mock_engine_instance."""
    mod = MagicMock()
    mod.WorkflowEngine.return_value = mock_engine_instance
    return mod


_SENTINEL = object()


def _make_templates_module(get_template_rv=_SENTINEL, list_templates_rv=_SENTINEL):
    """Create a mock module for aragora.workflow.templates."""
    mod = MagicMock()
    if get_template_rv is not _SENTINEL:
        mod.get_template.return_value = get_template_rv
    if list_templates_rv is not _SENTINEL:
        mod.list_templates.return_value = list_templates_rv
    return mod


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
        mock_engine = AsyncMock()
        mock_engine_module = _make_engine_module(mock_engine)
        mock_templates_module = _make_templates_module(get_template_rv=None)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.engine": mock_engine_module,
                "aragora.workflow.templates": mock_templates_module,
            },
        ):
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

        mock_engine_module = _make_engine_module(mock_engine)
        mock_templates_module = _make_templates_module(get_template_rv=mock_template)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.engine": mock_engine_module,
                "aragora.workflow.templates": mock_templates_module,
            },
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

        mock_engine_module = _make_engine_module(mock_engine)
        mock_templates_module = _make_templates_module(get_template_rv=mock_template)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.engine": mock_engine_module,
                "aragora.workflow.templates": mock_templates_module,
            },
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
        # Setting a sys.modules entry to None causes ImportError on import
        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.engine": None,
                "aragora.workflow.templates": None,
            },
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

        mock_engine_module = _make_engine_module(mock_engine)
        mock_templates_module = _make_templates_module(get_template_rv=mock_template)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.engine": mock_engine_module,
                "aragora.workflow.templates": mock_templates_module,
            },
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

        mock_engine_module = _make_engine_module(mock_engine)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.engine": mock_engine_module,
            },
        ):
            result = await get_workflow_status_tool(execution_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_status_success(self):
        """Test successful status retrieval."""
        mock_engine = AsyncMock()

        mock_status = WorkflowExecutionStatus(
            status="running",
            progress=0.5,
            current_node="audit_node",
            started_at="2025-01-01T00:00:00",
            completed_at=None,
            error=None,
        )
        mock_engine.get_status.return_value = mock_status

        mock_engine_module = _make_engine_module(mock_engine)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.engine": mock_engine_module,
            },
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
        # Setting a sys.modules entry to None causes ImportError on import
        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.templates": None,
            },
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

        mock_templates_module = _make_templates_module(list_templates_rv=mock_templates)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.templates": mock_templates_module,
            },
        ):
            result = await list_workflow_templates_tool(category="debate")

        assert result["count"] == 1
        assert result["templates"][0]["name"] == "debate_and_audit"
        assert result["category"] == "debate"

    @pytest.mark.asyncio
    async def test_list_all_category(self):
        """Test list with 'all' category passes None to list_templates."""
        mock_templates_module = _make_templates_module(list_templates_rv=[])

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.templates": mock_templates_module,
            },
        ):
            result = await list_workflow_templates_tool(category="all")

        mock_templates_module.list_templates.assert_called_once_with(None)


class TestCancelWorkflowTool:
    """Tests for cancel_workflow_tool."""

    @pytest.mark.asyncio
    async def test_cancel_success(self):
        """Test successful workflow cancellation."""
        mock_engine = AsyncMock()
        mock_engine.cancel.return_value = True

        mock_engine_module = _make_engine_module(mock_engine)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.engine": mock_engine_module,
            },
        ):
            result = await cancel_workflow_tool(execution_id="exec-123", reason="User requested")

        assert result["cancelled"] is True
        assert result["execution_id"] == "exec-123"
        assert result["reason"] == "User requested"

    @pytest.mark.asyncio
    async def test_cancel_not_found(self):
        """Test cancel for non-existent workflow."""
        mock_engine = AsyncMock()
        mock_engine.cancel.return_value = False

        mock_engine_module = _make_engine_module(mock_engine)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.engine": mock_engine_module,
            },
        ):
            result = await cancel_workflow_tool(execution_id="nonexistent")

        assert result["cancelled"] is False

    @pytest.mark.asyncio
    async def test_cancel_default_reason(self):
        """Test cancel with default reason."""
        mock_engine = AsyncMock()
        mock_engine.cancel.return_value = True

        mock_engine_module = _make_engine_module(mock_engine)

        with patch.dict(
            "sys.modules",
            {
                "aragora.workflow.engine": mock_engine_module,
            },
        ):
            result = await cancel_workflow_tool(execution_id="exec-123")

        assert result["reason"] == "User requested"
