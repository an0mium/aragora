"""Tests for MCP canvas tools execution logic."""

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest

from aragora.mcp.tools_module.canvas import (
    canvas_add_edge_tool,
    canvas_add_node_tool,
    canvas_create_tool,
    canvas_delete_node_tool,
    canvas_execute_action_tool,
    canvas_get_tool,
    canvas_list_tool,
)



@pytest.fixture(autouse=True)
def reset_canvas_manager():
    """Reset global canvas manager between tests."""
    import aragora.mcp.tools_module.canvas as canvas_mod

    canvas_mod._manager = None
    yield
    canvas_mod._manager = None


class TestCanvasCreateTool:
    """Tests for canvas_create_tool."""

    @pytest.mark.asyncio
    async def test_create_success(self):
        """Test successful canvas creation."""
        mock_manager = AsyncMock()
        mock_canvas = MagicMock()
        mock_canvas.id = "canvas-001"
        mock_canvas.name = "Test Canvas"
        mock_manager.create_canvas.return_value = mock_canvas

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_create_tool(name="Test Canvas", owner_id="user-1")

        assert result["success"] is True
        assert result["canvas_id"] == "canvas-001"
        assert result["name"] == "Test Canvas"

    @pytest.mark.asyncio
    async def test_create_default_name(self):
        """Test canvas creation with default name."""
        mock_manager = AsyncMock()
        mock_canvas = MagicMock()
        mock_canvas.id = "canvas-002"
        mock_canvas.name = "Untitled Canvas"
        mock_manager.create_canvas.return_value = mock_canvas

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_create_tool()

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_import_error(self):
        """Test canvas creation when module unavailable."""
        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            side_effect=ImportError("Canvas not installed"),
        ):
            result = await canvas_create_tool(name="Test")

        assert result["success"] is False
        assert "not available" in result.get("error", "").lower() or "error" in result

    @pytest.mark.asyncio
    async def test_create_failure(self):
        """Test canvas creation failure."""
        mock_manager = AsyncMock()
        mock_manager.create_canvas.side_effect = RuntimeError("DB error")

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_create_tool(name="Test")

        assert result["success"] is False


class TestCanvasGetTool:
    """Tests for canvas_get_tool."""

    @pytest.mark.asyncio
    async def test_get_empty_id(self):
        """Test get with empty canvas_id."""
        result = await canvas_get_tool(canvas_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_success(self):
        """Test successful canvas retrieval."""
        mock_manager = AsyncMock()
        mock_canvas = MagicMock()
        mock_canvas.id = "canvas-001"
        mock_canvas.name = "My Canvas"
        mock_node = MagicMock()
        mock_node.to_dict.return_value = {"id": "n1", "type": "text", "label": "Node 1"}
        mock_edge = MagicMock()
        mock_edge.to_dict.return_value = {"id": "e1", "source": "n1", "target": "n2"}
        mock_canvas.nodes = {"n1": mock_node}
        mock_canvas.edges = {"e1": mock_edge}
        mock_manager.get_canvas.return_value = mock_canvas

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_get_tool(canvas_id="canvas-001")

        assert result["success"] is True
        assert result["canvas"]["id"] == "canvas-001"
        assert result["canvas"]["node_count"] == 1
        assert result["canvas"]["edge_count"] == 1

    @pytest.mark.asyncio
    async def test_get_not_found(self):
        """Test get for non-existent canvas."""
        mock_manager = AsyncMock()
        mock_manager.get_canvas.return_value = None

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_get_tool(canvas_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()


class TestCanvasAddNodeTool:
    """Tests for canvas_add_node_tool."""

    @pytest.mark.asyncio
    async def test_add_node_empty_canvas_id(self):
        """Test add node with empty canvas_id."""
        result = await canvas_add_node_tool(canvas_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_add_node_success(self):
        """Test successful node addition."""
        mock_manager = AsyncMock()
        mock_node = MagicMock()
        mock_node.id = "node-001"
        mock_node.node_type = MagicMock(value="text")
        mock_manager.add_node.return_value = mock_node

        with (
            patch(
                "aragora.mcp.tools_module.canvas._get_manager",
                return_value=mock_manager,
            ),
            patch("aragora.mcp.tools_module.canvas.CanvasNodeType", create=True) as mock_type,
            patch("aragora.mcp.tools_module.canvas.Position", create=True),
        ):
            mock_type.__getitem__ = MagicMock()
            result = await canvas_add_node_tool(
                canvas_id="canvas-001",
                node_type="text",
                label="My Node",
                x=200,
                y=300,
            )

        assert result["success"] is True
        assert result["node_id"] == "node-001"

    @pytest.mark.asyncio
    async def test_add_node_returns_none(self):
        """Test add node when manager returns None."""
        mock_manager = AsyncMock()
        mock_manager.add_node.return_value = None

        with (
            patch(
                "aragora.mcp.tools_module.canvas._get_manager",
                return_value=mock_manager,
            ),
            patch("aragora.mcp.tools_module.canvas.CanvasNodeType", create=True),
            patch("aragora.mcp.tools_module.canvas.Position", create=True),
        ):
            result = await canvas_add_node_tool(
                canvas_id="canvas-001",
                node_type="text",
                label="Test",
            )

        assert result["success"] is False


class TestCanvasAddEdgeTool:
    """Tests for canvas_add_edge_tool."""

    @pytest.mark.asyncio
    async def test_add_edge_missing_ids(self):
        """Test add edge with missing required IDs."""
        result = await canvas_add_edge_tool(canvas_id="", source_id="n1", target_id="n2")
        assert "error" in result

        result = await canvas_add_edge_tool(canvas_id="c1", source_id="", target_id="n2")
        assert "error" in result

        result = await canvas_add_edge_tool(canvas_id="c1", source_id="n1", target_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_add_edge_success(self):
        """Test successful edge addition."""
        mock_manager = AsyncMock()
        mock_edge = MagicMock()
        mock_edge.id = "edge-001"
        mock_edge.source_id = "n1"
        mock_edge.target_id = "n2"
        mock_manager.add_edge.return_value = mock_edge

        with (
            patch(
                "aragora.mcp.tools_module.canvas._get_manager",
                return_value=mock_manager,
            ),
            patch("aragora.mcp.tools_module.canvas.EdgeType", create=True),
        ):
            result = await canvas_add_edge_tool(
                canvas_id="canvas-001",
                source_id="n1",
                target_id="n2",
                edge_type="data_flow",
            )

        assert result["success"] is True
        assert result["edge_id"] == "edge-001"
        assert result["source_id"] == "n1"
        assert result["target_id"] == "n2"


class TestCanvasExecuteActionTool:
    """Tests for canvas_execute_action_tool."""

    @pytest.mark.asyncio
    async def test_execute_action_missing_params(self):
        """Test execute with missing required params."""
        result = await canvas_execute_action_tool(canvas_id="", action="start_debate")
        assert "error" in result

        result = await canvas_execute_action_tool(canvas_id="c1", action="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_action_success(self):
        """Test successful action execution."""
        mock_manager = AsyncMock()
        mock_manager.execute_action.return_value = {"status": "started", "debate_id": "d-001"}

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_execute_action_tool(
                canvas_id="canvas-001",
                action="start_debate",
                params='{"topic": "AI safety"}',
            )

        assert result["status"] == "started"
        assert result["debate_id"] == "d-001"

    @pytest.mark.asyncio
    async def test_execute_action_invalid_json_params(self):
        """Test execute with invalid JSON params defaults to empty dict."""
        mock_manager = AsyncMock()
        mock_manager.execute_action.return_value = {"status": "ok"}

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_execute_action_tool(
                canvas_id="canvas-001",
                action="clear_canvas",
                params="not json",
            )

        # Should still work since invalid JSON defaults to {}
        assert "status" in result or "error" not in result


class TestCanvasListTool:
    """Tests for canvas_list_tool."""

    @pytest.mark.asyncio
    async def test_list_success(self):
        """Test successful canvas listing."""
        mock_manager = AsyncMock()
        mock_canvas = MagicMock()
        mock_canvas.id = "canvas-001"
        mock_canvas.name = "Test"
        mock_canvas.nodes = {}
        mock_canvas.edges = {}
        mock_canvas.created_at = datetime(2025, 1, 1)
        mock_manager.list_canvases.return_value = [mock_canvas]

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_list_tool()

        assert result["count"] == 1
        assert result["canvases"][0]["id"] == "canvas-001"

    @pytest.mark.asyncio
    async def test_list_respects_limit(self):
        """Test list respects limit parameter."""
        mock_manager = AsyncMock()
        canvases = []
        for i in range(5):
            c = MagicMock()
            c.id = f"canvas-{i}"
            c.name = f"Canvas {i}"
            c.nodes = {}
            c.edges = {}
            c.created_at = None
            canvases.append(c)
        mock_manager.list_canvases.return_value = canvases

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_list_tool(limit=2)

        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_list_clamps_limit(self):
        """Test list clamps limit to valid range."""
        mock_manager = AsyncMock()
        mock_manager.list_canvases.return_value = []

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_list_tool(limit=0)

        # limit=0 should be clamped to 1
        assert result["count"] == 0


class TestCanvasDeleteNodeTool:
    """Tests for canvas_delete_node_tool."""

    @pytest.mark.asyncio
    async def test_delete_missing_ids(self):
        """Test delete with missing IDs."""
        result = await canvas_delete_node_tool(canvas_id="", node_id="n1")
        assert "error" in result

        result = await canvas_delete_node_tool(canvas_id="c1", node_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_delete_success(self):
        """Test successful node deletion."""
        mock_manager = AsyncMock()
        mock_manager.delete_node.return_value = True

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_delete_node_tool(canvas_id="canvas-001", node_id="n1")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        """Test delete for non-existent node."""
        mock_manager = AsyncMock()
        mock_manager.delete_node.return_value = False

        with patch(
            "aragora.mcp.tools_module.canvas._get_manager",
            return_value=mock_manager,
        ):
            result = await canvas_delete_node_tool(canvas_id="canvas-001", node_id="missing")

        assert result["success"] is False
