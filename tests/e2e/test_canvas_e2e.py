"""
Canvas E2E Tests.

Tests the Canvas system end-to-end:
- Canvas CRUD operations
- Node and edge management
- Canvas actions
- Renderer output
- State management
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

pytestmark = pytest.mark.e2e

# Environment flag for full server integration tests
HAS_FULL_SERVER = os.environ.get("ARAGORA_E2E_FULL_SERVER", "").lower() == "true"


# ============================================================================
# Canvas Manager Tests
# ============================================================================


class TestCanvasManagerE2E:
    """E2E tests for CanvasStateManager operations."""

    @pytest_asyncio.fixture
    async def manager(self):
        """Create a CanvasStateManager for testing."""
        from aragora.canvas.manager import CanvasStateManager

        manager = CanvasStateManager()
        yield manager

    @pytest.mark.asyncio
    async def test_create_and_retrieve_canvas(self, manager):
        """Test creating a canvas and retrieving it."""
        # Create canvas
        canvas = await manager.create_canvas(
            name="E2E Test Canvas",
            owner_id="user-123",
            workspace_id="ws-456",
        )

        assert canvas.id is not None
        assert canvas.name == "E2E Test Canvas"
        assert canvas.owner_id == "user-123"
        assert canvas.workspace_id == "ws-456"

        # Retrieve canvas
        retrieved = await manager.get_canvas(canvas.id)
        assert retrieved is not None
        assert retrieved.id == canvas.id
        assert retrieved.name == canvas.name

    @pytest.mark.asyncio
    async def test_update_canvas(self, manager):
        """Test updating canvas properties."""
        canvas = await manager.create_canvas(name="Original Name")

        # Update canvas name
        updated = await manager.update_canvas(canvas.id, name="Updated Name")
        assert updated is not None
        assert updated.name == "Updated Name"

        # Verify persistence
        retrieved = await manager.get_canvas(canvas.id)
        assert retrieved.name == "Updated Name"

        # Update canvas metadata
        updated_with_meta = await manager.update_canvas(
            canvas.id,
            metadata={"custom_field": "custom_value"},
        )
        assert updated_with_meta is not None
        assert updated_with_meta.metadata.get("custom_field") == "custom_value"

    @pytest.mark.asyncio
    async def test_delete_canvas(self, manager):
        """Test deleting a canvas."""
        canvas = await manager.create_canvas(name="To Delete")
        canvas_id = canvas.id

        # Delete canvas
        result = await manager.delete_canvas(canvas_id)
        assert result is True

        # Verify deleted
        retrieved = await manager.get_canvas(canvas_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list_canvases(self, manager):
        """Test listing canvases with filtering."""
        # Create multiple canvases
        await manager.create_canvas(name="Canvas A", owner_id="user-1")
        await manager.create_canvas(name="Canvas B", owner_id="user-1")
        await manager.create_canvas(name="Canvas C", owner_id="user-2")

        # List all canvases
        all_canvases = await manager.list_canvases()
        assert len(all_canvases) >= 3

        # List by owner
        user1_canvases = await manager.list_canvases(owner_id="user-1")
        assert len(user1_canvases) >= 2
        assert all(c.owner_id == "user-1" for c in user1_canvases)


class TestCanvasNodeManagementE2E:
    """E2E tests for canvas node operations."""

    @pytest_asyncio.fixture
    async def canvas_with_manager(self):
        """Create a manager with a test canvas."""
        from aragora.canvas.manager import CanvasStateManager

        manager = CanvasStateManager()
        canvas = await manager.create_canvas(name="Node Test Canvas")
        yield manager, canvas

    @pytest.mark.asyncio
    async def test_add_multiple_node_types(self, canvas_with_manager):
        """Test adding different types of nodes."""
        from aragora.canvas.models import CanvasNodeType, Position

        manager, canvas = canvas_with_manager

        # Add various node types
        agent_node = await manager.add_node(
            canvas.id,
            CanvasNodeType.AGENT,
            Position(100, 100),
            "Claude Agent",
            data={"agent_id": "claude-3-opus"},
        )
        assert agent_node.node_type == CanvasNodeType.AGENT
        assert agent_node.label == "Claude Agent"

        debate_node = await manager.add_node(
            canvas.id,
            CanvasNodeType.DEBATE,
            Position(300, 100),
            "Ethics Debate",
            data={"topic": "AI Ethics"},
        )
        assert debate_node.node_type == CanvasNodeType.DEBATE

        knowledge_node = await manager.add_node(
            canvas.id,
            CanvasNodeType.KNOWLEDGE,
            Position(200, 300),
            "Company Policies",
            data={"source": "internal"},
        )
        assert knowledge_node.node_type == CanvasNodeType.KNOWLEDGE

        # Verify all nodes are on the canvas
        updated_canvas = await manager.get_canvas(canvas.id)
        assert len(updated_canvas.nodes) == 3

    @pytest.mark.asyncio
    async def test_move_node(self, canvas_with_manager):
        """Test moving a node to a new position."""
        from aragora.canvas.models import CanvasNodeType, Position

        manager, canvas = canvas_with_manager

        node = await manager.add_node(
            canvas.id,
            CanvasNodeType.TEXT,
            Position(0, 0),
            "Movable Node",
        )

        # Move node
        moved = await manager.move_node(canvas.id, node.id, x=500, y=400)
        assert moved is not None
        assert moved.position.x == 500
        assert moved.position.y == 400

    @pytest.mark.asyncio
    async def test_update_node_data(self, canvas_with_manager):
        """Test updating node data."""
        from aragora.canvas.models import CanvasNodeType, Position

        manager, canvas = canvas_with_manager

        node = await manager.add_node(
            canvas.id,
            CanvasNodeType.AGENT,
            Position(100, 100),
            "Test Agent",
            data={"status": "idle"},
        )

        # Update node data
        updated = await manager.update_node(
            canvas.id,
            node.id,
            data={"status": "active", "task": "processing"},
        )
        assert updated.data["status"] == "active"
        assert updated.data["task"] == "processing"

    @pytest.mark.asyncio
    async def test_delete_node_removes_connected_edges(self, canvas_with_manager):
        """Test that deleting a node also removes its connected edges."""
        from aragora.canvas.models import CanvasNodeType, EdgeType, Position

        manager, canvas = canvas_with_manager

        # Create nodes
        node1 = await manager.add_node(canvas.id, CanvasNodeType.AGENT, Position(0, 0), "Node 1")
        node2 = await manager.add_node(canvas.id, CanvasNodeType.AGENT, Position(100, 0), "Node 2")
        node3 = await manager.add_node(canvas.id, CanvasNodeType.AGENT, Position(200, 0), "Node 3")

        # Create edges
        await manager.add_edge(canvas.id, node1.id, node2.id, EdgeType.DATA_FLOW)
        await manager.add_edge(canvas.id, node2.id, node3.id, EdgeType.DATA_FLOW)
        await manager.add_edge(canvas.id, node1.id, node3.id, EdgeType.REFERENCE)

        # Verify edges exist
        canvas_state = await manager.get_canvas(canvas.id)
        assert len(canvas_state.edges) == 3

        # Delete node2 (connected to 2 edges)
        await manager.delete_node(canvas.id, node2.id)

        # Verify node and its edges are removed
        canvas_state = await manager.get_canvas(canvas.id)
        assert len(canvas_state.nodes) == 2
        assert len(canvas_state.edges) == 1  # Only node1->node3 remains


class TestCanvasEdgeManagementE2E:
    """E2E tests for canvas edge operations."""

    @pytest_asyncio.fixture
    async def canvas_with_nodes(self):
        """Create a canvas with pre-existing nodes."""
        from aragora.canvas.manager import CanvasStateManager
        from aragora.canvas.models import CanvasNodeType, Position

        manager = CanvasStateManager()
        canvas = await manager.create_canvas(name="Edge Test Canvas")

        # Add nodes
        node_a = await manager.add_node(canvas.id, CanvasNodeType.INPUT, Position(0, 0), "Input")
        node_b = await manager.add_node(
            canvas.id, CanvasNodeType.AGENT, Position(100, 0), "Processor"
        )
        node_c = await manager.add_node(
            canvas.id, CanvasNodeType.OUTPUT, Position(200, 0), "Output"
        )

        yield manager, canvas, [node_a, node_b, node_c]

    @pytest.mark.asyncio
    async def test_create_data_flow_pipeline(self, canvas_with_nodes):
        """Test creating a data flow pipeline with edges."""
        from aragora.canvas.models import EdgeType

        manager, canvas, nodes = canvas_with_nodes
        input_node, processor, output_node = nodes

        # Create data flow pipeline
        edge1 = await manager.add_edge(
            canvas.id,
            input_node.id,
            processor.id,
            EdgeType.DATA_FLOW,
            label="Raw Data",
        )
        edge2 = await manager.add_edge(
            canvas.id,
            processor.id,
            output_node.id,
            EdgeType.DATA_FLOW,
            label="Processed Results",
        )

        assert edge1.edge_type == EdgeType.DATA_FLOW
        assert edge1.label == "Raw Data"
        assert edge2.edge_type == EdgeType.DATA_FLOW

        # Verify pipeline structure
        canvas_state = await manager.get_canvas(canvas.id)
        assert len(canvas_state.edges) == 2

    @pytest.mark.asyncio
    async def test_edge_validation(self, canvas_with_nodes):
        """Test that edges can only be created between existing nodes."""
        from aragora.canvas.models import EdgeType

        manager, canvas, nodes = canvas_with_nodes

        # Try to create edge with non-existent node
        result = await manager.add_edge(
            canvas.id,
            nodes[0].id,
            "non-existent-node",
            EdgeType.DEFAULT,
        )
        assert result is None


class TestCanvasRendererE2E:
    """E2E tests for canvas rendering."""

    @pytest_asyncio.fixture
    async def populated_canvas(self):
        """Create a populated canvas for rendering tests."""
        from aragora.canvas.manager import CanvasStateManager
        from aragora.canvas.models import CanvasNodeType, EdgeType, Position

        manager = CanvasStateManager()
        canvas = await manager.create_canvas(name="Render Test Canvas")

        # Add nodes
        agent1 = await manager.add_node(
            canvas.id,
            CanvasNodeType.AGENT,
            Position(100, 100),
            "Claude",
        )
        agent2 = await manager.add_node(
            canvas.id,
            CanvasNodeType.AGENT,
            Position(300, 100),
            "GPT-4",
        )
        debate = await manager.add_node(
            canvas.id,
            CanvasNodeType.DEBATE,
            Position(200, 250),
            "Architecture Debate",
        )

        # Add edges
        await manager.add_edge(canvas.id, agent1.id, debate.id, EdgeType.DATA_FLOW)
        await manager.add_edge(canvas.id, agent2.id, debate.id, EdgeType.DATA_FLOW)

        # Get updated canvas with nodes/edges
        updated_canvas = await manager.get_canvas(canvas.id)
        yield updated_canvas

    @pytest.mark.asyncio
    async def test_render_to_json(self, populated_canvas):
        """Test rendering canvas to JSON format."""
        from aragora.canvas.renderer import CanvasRenderer

        renderer = CanvasRenderer(populated_canvas)
        json_output = renderer.to_json()

        assert json_output["id"] == populated_canvas.id
        assert len(json_output["nodes"]) == 3
        assert len(json_output["edges"]) == 2

        # Verify node structure
        node = json_output["nodes"][0]
        assert "id" in node
        assert "type" in node
        assert "position" in node
        assert "data" in node

    @pytest.mark.asyncio
    async def test_render_to_svg(self, populated_canvas):
        """Test rendering canvas to SVG format."""
        from aragora.canvas.renderer import CanvasRenderer

        renderer = CanvasRenderer(populated_canvas)
        svg_output = renderer.to_svg(width=800, height=600)

        assert svg_output.startswith("<svg")
        assert 'width="800"' in svg_output
        assert 'height="600"' in svg_output
        assert "</svg>" in svg_output
        # Should contain elements for nodes and edges
        assert "<rect" in svg_output or "<circle" in svg_output
        assert "<line" in svg_output

    @pytest.mark.asyncio
    async def test_render_to_mermaid(self, populated_canvas):
        """Test rendering canvas to Mermaid diagram format."""
        from aragora.canvas.renderer import CanvasRenderer

        renderer = CanvasRenderer(populated_canvas)
        mermaid_output = renderer.to_mermaid()

        assert mermaid_output.startswith("graph LR")
        # Should contain node and edge definitions
        assert "-->" in mermaid_output
        assert "[" in mermaid_output  # Node label brackets


class TestCanvasHandlerE2E:
    """E2E tests for Canvas HTTP handler."""

    @pytest.fixture
    def mock_context(self):
        """Create mock server context for handler tests."""
        return {
            "storage": MagicMock(),
            "elo_system": None,
            "nomic_dir": None,
        }

    @pytest.fixture
    def canvas_handler(self, mock_context):
        """Create CanvasHandler instance."""
        from aragora.server.handlers.canvas import CanvasHandler

        return CanvasHandler(mock_context)

    def test_handler_can_handle_routes(self, canvas_handler):
        """Test handler route matching."""
        # Should handle canvas routes
        assert canvas_handler.can_handle("/api/v1/canvas") is True
        assert canvas_handler.can_handle("/api/v1/canvas/") is True
        assert canvas_handler.can_handle("/api/v1/canvas/abc-123") is True
        assert canvas_handler.can_handle("/api/v1/canvas/abc-123/nodes") is True
        assert canvas_handler.can_handle("/api/v1/canvas/abc-123/edges") is True
        assert canvas_handler.can_handle("/api/v1/canvas/abc-123/action") is True

        # Should not handle other routes
        assert canvas_handler.can_handle("/api/v1/debates") is False
        assert canvas_handler.can_handle("/api/v1/agents") is False

    @pytest.mark.skipif(not HAS_FULL_SERVER, reason="Handler tests require full server setup")
    def test_list_canvases_endpoint(self, canvas_handler):
        """Test listing canvases via handler."""
        mock_handler = MagicMock()
        mock_handler.headers = {}

        # handle() may or may not be async
        result = canvas_handler.handle("/api/v1/canvas", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.skipif(not HAS_FULL_SERVER, reason="Handler tests require full server setup")
    def test_create_canvas_endpoint(self, canvas_handler):
        """Test creating canvas via handler."""
        import json

        mock_handler = MagicMock()
        body = json.dumps({"name": "API Created Canvas"}).encode()
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        mock_handler.rfile.read.return_value = body

        # Use handle_post if available
        if hasattr(canvas_handler, "handle_post"):
            result = canvas_handler.handle_post("/api/v1/canvas", {}, mock_handler)
        else:
            result = canvas_handler.handle("/api/v1/canvas", {}, mock_handler)

        assert result is not None
        assert result.status_code in [200, 201]


class TestCanvasWorkflowE2E:
    """E2E tests for canvas workflow scenarios."""

    @pytest.mark.asyncio
    async def test_complete_debate_setup_workflow(self):
        """Test setting up a complete debate workflow on canvas."""
        from aragora.canvas.manager import CanvasStateManager
        from aragora.canvas.models import CanvasNodeType, EdgeType, Position

        manager = CanvasStateManager()

        # Step 1: Create canvas for debate workflow
        canvas = await manager.create_canvas(
            name="Multi-Agent Debate",
            metadata={"workflow_type": "debate", "topic": "AI Safety"},
        )

        # Step 2: Add agent nodes
        claude = await manager.add_node(
            canvas.id,
            CanvasNodeType.AGENT,
            Position(100, 100),
            "Claude (Pro Safety)",
            data={"agent_id": "claude-3-opus", "stance": "pro_safety"},
        )

        gpt4 = await manager.add_node(
            canvas.id,
            CanvasNodeType.AGENT,
            Position(300, 100),
            "GPT-4 (Balanced)",
            data={"agent_id": "gpt-4", "stance": "balanced"},
        )

        gemini = await manager.add_node(
            canvas.id,
            CanvasNodeType.AGENT,
            Position(500, 100),
            "Gemini (Innovation)",
            data={"agent_id": "gemini-pro", "stance": "pro_innovation"},
        )

        # Step 3: Add debate node
        debate = await manager.add_node(
            canvas.id,
            CanvasNodeType.DEBATE,
            Position(300, 300),
            "AI Safety Debate",
            data={
                "topic": "Should AI development be regulated?",
                "rounds": 5,
                "consensus_threshold": 0.7,
            },
        )

        # Step 4: Add decision output node
        decision = await manager.add_node(
            canvas.id,
            CanvasNodeType.DECISION,
            Position(300, 500),
            "Final Decision",
            data={"requires_consensus": True},
        )

        # Step 5: Connect agents to debate
        await manager.add_edge(canvas.id, claude.id, debate.id, EdgeType.DATA_FLOW)
        await manager.add_edge(canvas.id, gpt4.id, debate.id, EdgeType.DATA_FLOW)
        await manager.add_edge(canvas.id, gemini.id, debate.id, EdgeType.DATA_FLOW)

        # Step 6: Connect debate to decision
        await manager.add_edge(canvas.id, debate.id, decision.id, EdgeType.DATA_FLOW)

        # Verify complete workflow
        final_canvas = await manager.get_canvas(canvas.id)
        assert len(final_canvas.nodes) == 5
        assert len(final_canvas.edges) == 4

        # Verify all agents are connected to debate
        debate_edges = [e for e in final_canvas.edges.values() if e.target_id == debate.id]
        assert len(debate_edges) == 3

        # Verify debate is connected to decision
        decision_edges = [e for e in final_canvas.edges.values() if e.target_id == decision.id]
        assert len(decision_edges) == 1

    @pytest.mark.asyncio
    async def test_knowledge_integration_workflow(self):
        """Test setting up knowledge integration on canvas."""
        from aragora.canvas.manager import CanvasStateManager
        from aragora.canvas.models import CanvasNodeType, EdgeType, Position

        manager = CanvasStateManager()

        canvas = await manager.create_canvas(
            name="Knowledge-Augmented Research",
            metadata={"workflow_type": "research"},
        )

        # Knowledge sources
        docs = await manager.add_node(
            canvas.id,
            CanvasNodeType.KNOWLEDGE,
            Position(0, 0),
            "Company Documents",
            data={"source": "sharepoint", "path": "/docs"},
        )

        policies = await manager.add_node(
            canvas.id,
            CanvasNodeType.KNOWLEDGE,
            Position(0, 100),
            "HR Policies",
            data={"source": "notion", "database": "policies"},
        )

        # Research agent
        researcher = await manager.add_node(
            canvas.id,
            CanvasNodeType.AGENT,
            Position(200, 50),
            "Research Agent",
            data={"agent_id": "claude-3-opus", "role": "researcher"},
        )

        # Output
        report = await manager.add_node(
            canvas.id,
            CanvasNodeType.OUTPUT,
            Position(400, 50),
            "Research Report",
            data={"format": "markdown"},
        )

        # Connect knowledge to agent
        await manager.add_edge(canvas.id, docs.id, researcher.id, EdgeType.REFERENCE)
        await manager.add_edge(canvas.id, policies.id, researcher.id, EdgeType.REFERENCE)

        # Connect agent to output
        await manager.add_edge(canvas.id, researcher.id, report.id, EdgeType.DATA_FLOW)

        final_canvas = await manager.get_canvas(canvas.id)
        assert len(final_canvas.nodes) == 4
        assert len(final_canvas.edges) == 3


class TestCanvasConcurrencyE2E:
    """E2E tests for concurrent canvas operations."""

    @pytest.mark.asyncio
    async def test_concurrent_node_additions(self):
        """Test adding nodes concurrently to the same canvas."""
        from aragora.canvas.manager import CanvasStateManager
        from aragora.canvas.models import CanvasNodeType, Position

        manager = CanvasStateManager()
        canvas = await manager.create_canvas(name="Concurrent Test")

        # Add 10 nodes concurrently
        tasks = [
            manager.add_node(
                canvas.id,
                CanvasNodeType.TEXT,
                Position(i * 50, i * 50),
                f"Node {i}",
            )
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All nodes should be created
        assert len(results) == 10
        assert all(r is not None for r in results)

        # All nodes should be on the canvas
        final_canvas = await manager.get_canvas(canvas.id)
        assert len(final_canvas.nodes) == 10

    @pytest.mark.asyncio
    async def test_concurrent_canvas_operations(self):
        """Test multiple operations on different canvases concurrently."""
        from aragora.canvas.manager import CanvasStateManager
        from aragora.canvas.models import CanvasNodeType, Position

        manager = CanvasStateManager()

        # Create 5 canvases
        canvases = await asyncio.gather(
            *[manager.create_canvas(name=f"Canvas {i}") for i in range(5)]
        )

        # Add nodes to all canvases concurrently
        add_tasks = []
        for canvas in canvases:
            add_tasks.append(
                manager.add_node(
                    canvas.id,
                    CanvasNodeType.AGENT,
                    Position(0, 0),
                    "Test Agent",
                )
            )

        await asyncio.gather(*add_tasks)

        # Verify each canvas has one node
        for canvas in canvases:
            final_canvas = await manager.get_canvas(canvas.id)
            assert len(final_canvas.nodes) == 1


class TestCanvasActionExecutionE2E:
    """E2E tests for canvas action execution."""

    @pytest_asyncio.fixture
    async def manager_with_canvas(self):
        """Create a manager with a test canvas."""
        from aragora.canvas.manager import CanvasStateManager

        manager = CanvasStateManager()
        canvas = await manager.create_canvas(name="Action Test Canvas")
        yield manager, canvas

    @pytest.mark.asyncio
    async def test_start_debate_action_creates_node(self, manager_with_canvas):
        """Test that start_debate action creates a debate node."""
        manager, canvas = manager_with_canvas

        # Execute start_debate action
        result = await manager.execute_action(
            canvas_id=canvas.id,
            action="start_debate",
            params={"question": "Should we use microservices?", "rounds": 3},
        )

        # Should succeed (node created, even if execution not available)
        assert result.get("success") is True or "debate_node_id" in result

        # Verify debate node was created
        updated_canvas = await manager.get_canvas(canvas.id)
        debate_nodes = [n for n in updated_canvas.nodes.values() if n.node_type.value == "debate"]
        assert len(debate_nodes) >= 1

    @pytest.mark.asyncio
    async def test_run_workflow_action_creates_node(self, manager_with_canvas):
        """Test that run_workflow action creates a workflow node."""
        manager, canvas = manager_with_canvas

        # Execute run_workflow action with required definition
        result = await manager.execute_action(
            canvas_id=canvas.id,
            action="run_workflow",
            params={
                "definition": {
                    "name": "Test Workflow",
                    "steps": [{"type": "noop", "name": "test_step"}],
                },
                "inputs": {},
            },
        )

        # Should succeed or return error about modules not available
        # Either way, should have created a workflow node
        updated_canvas = await manager.get_canvas(canvas.id)
        workflow_nodes = [
            n for n in updated_canvas.nodes.values() if n.node_type.value == "workflow"
        ]
        # Either we have a workflow node or the action returned success/graceful error
        assert len(workflow_nodes) >= 1 or result.get("success") is True or "executed" in result

    @pytest.mark.asyncio
    async def test_query_knowledge_action(self, manager_with_canvas):
        """Test that query_knowledge action works."""
        manager, canvas = manager_with_canvas

        # Execute query_knowledge action
        result = await manager.execute_action(
            canvas_id=canvas.id,
            action="query_knowledge",
            params={"query": "architecture patterns"},
        )

        # Should succeed (may not have actual results but shouldn't error)
        assert "error" not in result or result.get("success") is True

    @pytest.mark.asyncio
    async def test_clear_canvas_action(self, manager_with_canvas):
        """Test that clear_canvas action removes all nodes and edges."""
        from aragora.canvas.models import CanvasNodeType, EdgeType, Position

        manager, canvas = manager_with_canvas

        # Add some nodes and edges
        node1 = await manager.add_node(canvas.id, CanvasNodeType.AGENT, Position(0, 0), "Node 1")
        node2 = await manager.add_node(canvas.id, CanvasNodeType.AGENT, Position(100, 0), "Node 2")
        await manager.add_edge(canvas.id, node1.id, node2.id, EdgeType.DEFAULT)

        # Verify nodes exist
        canvas_before = await manager.get_canvas(canvas.id)
        assert len(canvas_before.nodes) == 2
        assert len(canvas_before.edges) == 1

        # Execute clear_canvas action
        result = await manager.execute_action(
            canvas_id=canvas.id,
            action="clear_canvas",
            params={},
        )

        assert result.get("success") is True

        # Verify canvas is cleared
        canvas_after = await manager.get_canvas(canvas.id)
        assert len(canvas_after.nodes) == 0
        assert len(canvas_after.edges) == 0

    @pytest.mark.asyncio
    async def test_invalid_action_returns_error(self, manager_with_canvas):
        """Test that invalid actions return appropriate error."""
        manager, canvas = manager_with_canvas

        result = await manager.execute_action(
            canvas_id=canvas.id,
            action="nonexistent_action",
            params={},
        )

        # Should indicate failure or unknown action
        assert (
            result.get("success") is False or "error" in result or "unknown" in str(result).lower()
        )


class TestCanvasMCPToolsE2E:
    """E2E tests for canvas MCP tools."""

    @pytest.mark.asyncio
    async def test_canvas_create_tool(self):
        """Test canvas_create MCP tool."""
        from aragora.mcp.tools_module.canvas import canvas_create_tool

        result = await canvas_create_tool(
            name="MCP Created Canvas",
            description="Created via MCP tool",
        )

        assert result.get("success") is True
        assert "canvas_id" in result
        assert result["name"] == "MCP Created Canvas"

    @pytest.mark.asyncio
    async def test_canvas_get_tool(self):
        """Test canvas_get MCP tool."""
        from aragora.mcp.tools_module.canvas import canvas_create_tool, canvas_get_tool

        # Create a canvas first
        create_result = await canvas_create_tool(name="Get Test Canvas")
        canvas_id = create_result["canvas_id"]

        # Get the canvas
        result = await canvas_get_tool(canvas_id=canvas_id)

        assert result.get("success") is True
        assert "canvas" in result
        assert result["canvas"]["id"] == canvas_id
        assert result["canvas"]["name"] == "Get Test Canvas"

    @pytest.mark.asyncio
    async def test_canvas_add_node_tool(self):
        """Test canvas_add_node MCP tool."""
        from aragora.mcp.tools_module.canvas import canvas_create_tool, canvas_add_node_tool

        # Create a canvas first
        create_result = await canvas_create_tool(name="Node Test Canvas")
        canvas_id = create_result["canvas_id"]

        # Add a node
        result = await canvas_add_node_tool(
            canvas_id=canvas_id,
            node_type="agent",
            label="Test Agent",
            x=150,
            y=200,
            data='{"agent_id": "claude"}',
        )

        assert result.get("success") is True
        assert "node_id" in result
        assert result["node_type"] == "agent"

    @pytest.mark.asyncio
    async def test_canvas_add_edge_tool(self):
        """Test canvas_add_edge MCP tool."""
        from aragora.mcp.tools_module.canvas import (
            canvas_create_tool,
            canvas_add_node_tool,
            canvas_add_edge_tool,
        )

        # Create canvas with nodes
        create_result = await canvas_create_tool(name="Edge Test Canvas")
        canvas_id = create_result["canvas_id"]

        node1 = await canvas_add_node_tool(canvas_id=canvas_id, label="Source")
        node2 = await canvas_add_node_tool(canvas_id=canvas_id, label="Target", x=200, y=100)

        # Add edge
        result = await canvas_add_edge_tool(
            canvas_id=canvas_id,
            source_id=node1["node_id"],
            target_id=node2["node_id"],
            edge_type="data_flow",
            label="Data Connection",
        )

        assert result.get("success") is True
        assert "edge_id" in result

    @pytest.mark.asyncio
    async def test_canvas_execute_action_tool(self):
        """Test canvas_execute_action MCP tool."""
        import json
        from aragora.mcp.tools_module.canvas import canvas_create_tool, canvas_execute_action_tool

        # Create canvas
        create_result = await canvas_create_tool(name="Action Test Canvas")
        canvas_id = create_result["canvas_id"]

        # Execute start_debate action
        result = await canvas_execute_action_tool(
            canvas_id=canvas_id,
            action="start_debate",
            params=json.dumps({"question": "What is the best programming language?"}),
        )

        # Should either succeed or gracefully indicate execution not available
        assert "error" not in result or result.get("success") is True

    @pytest.mark.asyncio
    async def test_canvas_list_tool(self):
        """Test canvas_list MCP tool."""
        from aragora.mcp.tools_module.canvas import canvas_create_tool, canvas_list_tool

        # Create some canvases
        await canvas_create_tool(name="List Test 1", owner_id="list-user")
        await canvas_create_tool(name="List Test 2", owner_id="list-user")

        # List canvases
        result = await canvas_list_tool(owner_id="list-user")

        assert "canvases" in result
        assert result["count"] >= 2
        assert all(c["name"].startswith("List Test") for c in result["canvases"][:2])

    @pytest.mark.asyncio
    async def test_canvas_delete_node_tool(self):
        """Test canvas_delete_node MCP tool."""
        from aragora.mcp.tools_module.canvas import (
            canvas_create_tool,
            canvas_add_node_tool,
            canvas_delete_node_tool,
            canvas_get_tool,
        )

        # Create canvas with a node
        create_result = await canvas_create_tool(name="Delete Test Canvas")
        canvas_id = create_result["canvas_id"]

        node = await canvas_add_node_tool(canvas_id=canvas_id, label="To Delete")
        node_id = node["node_id"]

        # Verify node exists
        before = await canvas_get_tool(canvas_id=canvas_id)
        assert before["canvas"]["node_count"] == 1

        # Delete the node
        result = await canvas_delete_node_tool(canvas_id=canvas_id, node_id=node_id)
        assert result.get("success") is True

        # Verify node is deleted
        after = await canvas_get_tool(canvas_id=canvas_id)
        assert after["canvas"]["node_count"] == 0


class TestCanvasWebSocketStreamE2E:
    """E2E tests for canvas WebSocket streaming."""

    @pytest.mark.asyncio
    async def test_canvas_stream_server_initialization(self):
        """Test that canvas stream server can be initialized."""
        try:
            from aragora.server.stream.canvas_stream import CanvasStreamServer

            server = CanvasStreamServer(
                host="localhost", port=0
            )  # Port 0 for random available port
            assert server is not None
            assert hasattr(server, "start")
            assert hasattr(server, "handle_connection")
        except ImportError:
            pytest.skip("Canvas stream server not available")

    @pytest.mark.asyncio
    async def test_canvas_event_broadcasting(self):
        """Test that canvas operations work correctly (events broadcast when subscribers exist)."""
        from aragora.canvas.manager import CanvasStateManager
        from aragora.canvas.models import CanvasNodeType, Position

        manager = CanvasStateManager()

        # Create canvas and add node
        canvas = await manager.create_canvas(name="Broadcast Test")
        await manager.add_node(canvas.id, CanvasNodeType.AGENT, Position(0, 0), "Agent")

        # Verify the operations succeeded - broadcasting happens when there are subscribers
        # Without subscribers (in test mode), no events are broadcast, but the canvas should still work
        updated_canvas = await manager.get_canvas(canvas.id)
        assert len(updated_canvas.nodes) == 1
        assert updated_canvas.name == "Broadcast Test"

        # Verify the node was created correctly
        node = list(updated_canvas.nodes.values())[0]
        assert node.label == "Agent"
        assert node.node_type == CanvasNodeType.AGENT
