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
import uuid
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

pytestmark = pytest.mark.e2e


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

        # Update canvas
        updated = await manager.update_canvas(canvas.id, name="Updated Name")
        assert updated is not None
        assert updated.name == "Updated Name"

        # Verify persistence
        retrieved = await manager.get_canvas(canvas.id)
        assert retrieved.name == "Updated Name"

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
        from aragora.canvas.models import CanvasNodeType

        manager, canvas = canvas_with_manager

        # Add various node types
        agent_node = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.AGENT,
            label="Claude Agent",
            position={"x": 100, "y": 100},
            data={"agent_id": "claude-3-opus"},
        )
        assert agent_node.node_type == CanvasNodeType.AGENT
        assert agent_node.label == "Claude Agent"

        debate_node = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.DEBATE,
            label="Ethics Debate",
            position={"x": 300, "y": 100},
            data={"topic": "AI Ethics"},
        )
        assert debate_node.node_type == CanvasNodeType.DEBATE

        knowledge_node = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.KNOWLEDGE,
            label="Company Policies",
            position={"x": 200, "y": 300},
            data={"source": "internal"},
        )
        assert knowledge_node.node_type == CanvasNodeType.KNOWLEDGE

        # Verify all nodes are on the canvas
        updated_canvas = await manager.get_canvas(canvas.id)
        assert len(updated_canvas.nodes) == 3

    @pytest.mark.asyncio
    async def test_move_node(self, canvas_with_manager):
        """Test moving a node to a new position."""
        from aragora.canvas.models import CanvasNodeType

        manager, canvas = canvas_with_manager

        node = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.TEXT,
            label="Movable Node",
            position={"x": 0, "y": 0},
        )

        # Move node
        moved = await manager.move_node(canvas.id, node.id, x=500, y=400)
        assert moved is not None
        assert moved.position.x == 500
        assert moved.position.y == 400

    @pytest.mark.asyncio
    async def test_update_node_data(self, canvas_with_manager):
        """Test updating node data."""
        from aragora.canvas.models import CanvasNodeType

        manager, canvas = canvas_with_manager

        node = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.AGENT,
            label="Test Agent",
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
        from aragora.canvas.models import CanvasNodeType, EdgeType

        manager, canvas = canvas_with_manager

        # Create nodes
        node1 = await manager.add_node(canvas.id, node_type=CanvasNodeType.AGENT, label="Node 1")
        node2 = await manager.add_node(canvas.id, node_type=CanvasNodeType.AGENT, label="Node 2")
        node3 = await manager.add_node(canvas.id, node_type=CanvasNodeType.AGENT, label="Node 3")

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
        from aragora.canvas.models import CanvasNodeType

        manager = CanvasStateManager()
        canvas = await manager.create_canvas(name="Edge Test Canvas")

        # Add nodes
        node_a = await manager.add_node(canvas.id, node_type=CanvasNodeType.INPUT, label="Input")
        node_b = await manager.add_node(
            canvas.id, node_type=CanvasNodeType.AGENT, label="Processor"
        )
        node_c = await manager.add_node(canvas.id, node_type=CanvasNodeType.OUTPUT, label="Output")

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
        from aragora.canvas.models import CanvasNodeType, EdgeType

        manager = CanvasStateManager()
        canvas = await manager.create_canvas(name="Render Test Canvas")

        # Add nodes
        agent1 = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.AGENT,
            label="Claude",
            position={"x": 100, "y": 100},
        )
        agent2 = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.AGENT,
            label="GPT-4",
            position={"x": 300, "y": 100},
        )
        debate = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.DEBATE,
            label="Architecture Debate",
            position={"x": 200, "y": 250},
        )

        # Add edges
        await manager.add_edge(canvas.id, agent1.id, debate.id, EdgeType.DATA_FLOW)
        await manager.add_edge(canvas.id, agent2.id, debate.id, EdgeType.DATA_FLOW)

        yield canvas

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

    @pytest.mark.asyncio
    async def test_list_canvases_endpoint(self, canvas_handler):
        """Test listing canvases via handler."""
        mock_handler = MagicMock()
        mock_handler.headers = {}

        result = await canvas_handler.handle("/api/v1/canvas", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_create_canvas_endpoint(self, canvas_handler):
        """Test creating canvas via handler."""
        import json

        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Type": "application/json"}
        mock_handler.rfile.read.return_value = json.dumps({"name": "API Created Canvas"}).encode()

        # Get content length
        body = json.dumps({"name": "API Created Canvas"}).encode()
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }

        result = await canvas_handler.handle_post("/api/v1/canvas", {}, mock_handler)

        assert result is not None
        assert result.status_code in [200, 201]


class TestCanvasWorkflowE2E:
    """E2E tests for canvas workflow scenarios."""

    @pytest.mark.asyncio
    async def test_complete_debate_setup_workflow(self):
        """Test setting up a complete debate workflow on canvas."""
        from aragora.canvas.manager import CanvasStateManager
        from aragora.canvas.models import CanvasNodeType, EdgeType

        manager = CanvasStateManager()

        # Step 1: Create canvas for debate workflow
        canvas = await manager.create_canvas(
            name="Multi-Agent Debate",
            metadata={"workflow_type": "debate", "topic": "AI Safety"},
        )

        # Step 2: Add agent nodes
        claude = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.AGENT,
            label="Claude (Pro Safety)",
            position={"x": 100, "y": 100},
            data={"agent_id": "claude-3-opus", "stance": "pro_safety"},
        )

        gpt4 = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.AGENT,
            label="GPT-4 (Balanced)",
            position={"x": 300, "y": 100},
            data={"agent_id": "gpt-4", "stance": "balanced"},
        )

        gemini = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.AGENT,
            label="Gemini (Innovation)",
            position={"x": 500, "y": 100},
            data={"agent_id": "gemini-pro", "stance": "pro_innovation"},
        )

        # Step 3: Add debate node
        debate = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.DEBATE,
            label="AI Safety Debate",
            position={"x": 300, "y": 300},
            data={
                "topic": "Should AI development be regulated?",
                "rounds": 5,
                "consensus_threshold": 0.7,
            },
        )

        # Step 4: Add decision output node
        decision = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.DECISION,
            label="Final Decision",
            position={"x": 300, "y": 500},
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
        from aragora.canvas.models import CanvasNodeType, EdgeType

        manager = CanvasStateManager()

        canvas = await manager.create_canvas(
            name="Knowledge-Augmented Research",
            metadata={"workflow_type": "research"},
        )

        # Knowledge sources
        docs = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.KNOWLEDGE,
            label="Company Documents",
            data={"source": "sharepoint", "path": "/docs"},
        )

        policies = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.KNOWLEDGE,
            label="HR Policies",
            data={"source": "notion", "database": "policies"},
        )

        # Research agent
        researcher = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.AGENT,
            label="Research Agent",
            data={"agent_id": "claude-3-opus", "role": "researcher"},
        )

        # Output
        report = await manager.add_node(
            canvas.id,
            node_type=CanvasNodeType.OUTPUT,
            label="Research Report",
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
        from aragora.canvas.models import CanvasNodeType

        manager = CanvasStateManager()
        canvas = await manager.create_canvas(name="Concurrent Test")

        # Add 10 nodes concurrently
        tasks = [
            manager.add_node(
                canvas.id,
                node_type=CanvasNodeType.TEXT,
                label=f"Node {i}",
                position={"x": i * 50, "y": i * 50},
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
        from aragora.canvas.models import CanvasNodeType

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
                    node_type=CanvasNodeType.AGENT,
                    label="Test Agent",
                )
            )

        await asyncio.gather(*add_tasks)

        # Verify each canvas has one node
        for canvas in canvases:
            final_canvas = await manager.get_canvas(canvas.id)
            assert len(final_canvas.nodes) == 1
