"""
Tests for Canvas data models.

Tests cover:
- Position and Size dataclasses
- CanvasNode creation and serialization
- CanvasEdge creation and serialization
- CanvasEvent creation and serialization
- Canvas container operations
"""

import pytest
from datetime import datetime, timezone
import uuid

from aragora.canvas.models import (
    Canvas,
    CanvasNode,
    CanvasEdge,
    CanvasEvent,
    CanvasNodeType,
    CanvasEventType,
    EdgeType,
    Position,
    Size,
)


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_defaults(self):
        """Test Position with default values."""
        pos = Position()
        assert pos.x == 0.0
        assert pos.y == 0.0

    def test_position_custom(self):
        """Test Position with custom values."""
        pos = Position(x=100.5, y=200.75)
        assert pos.x == 100.5
        assert pos.y == 200.75

    def test_position_to_dict(self):
        """Test Position serialization."""
        pos = Position(x=50, y=100)
        data = pos.to_dict()
        assert data == {"x": 50, "y": 100}

    def test_position_from_dict(self):
        """Test Position deserialization."""
        data = {"x": 150.5, "y": 250.5}
        pos = Position.from_dict(data)
        assert pos.x == 150.5
        assert pos.y == 250.5

    def test_position_from_dict_defaults(self):
        """Test Position deserialization with missing values."""
        pos = Position.from_dict({})
        assert pos.x == 0
        assert pos.y == 0


class TestSize:
    """Tests for Size dataclass."""

    def test_size_defaults(self):
        """Test Size with default values."""
        size = Size()
        assert size.width == 200.0
        assert size.height == 100.0

    def test_size_custom(self):
        """Test Size with custom values."""
        size = Size(width=300, height=150)
        assert size.width == 300
        assert size.height == 150

    def test_size_to_dict(self):
        """Test Size serialization."""
        size = Size(width=400, height=200)
        data = size.to_dict()
        assert data == {"width": 400, "height": 200}

    def test_size_from_dict(self):
        """Test Size deserialization."""
        data = {"width": 500, "height": 300}
        size = Size.from_dict(data)
        assert size.width == 500
        assert size.height == 300

    def test_size_from_dict_defaults(self):
        """Test Size deserialization with missing values."""
        size = Size.from_dict({})
        assert size.width == 200
        assert size.height == 100


class TestCanvasNodeType:
    """Tests for CanvasNodeType enum."""

    def test_node_types(self):
        """Test all node types exist."""
        assert CanvasNodeType.AGENT.value == "agent"
        assert CanvasNodeType.DEBATE.value == "debate"
        assert CanvasNodeType.KNOWLEDGE.value == "knowledge"
        assert CanvasNodeType.CONNECTOR.value == "connector"
        assert CanvasNodeType.BROWSER.value == "browser"
        assert CanvasNodeType.WORKFLOW.value == "workflow"
        assert CanvasNodeType.INPUT.value == "input"
        assert CanvasNodeType.OUTPUT.value == "output"
        assert CanvasNodeType.TEXT.value == "text"
        assert CanvasNodeType.IMAGE.value == "image"
        assert CanvasNodeType.GROUP.value == "group"
        assert CanvasNodeType.DECISION.value == "decision"
        assert CanvasNodeType.EVIDENCE.value == "evidence"


class TestCanvasEventType:
    """Tests for CanvasEventType enum."""

    def test_connection_events(self):
        """Test connection event types."""
        assert CanvasEventType.CONNECT.value == "canvas:connect"
        assert CanvasEventType.DISCONNECT.value == "canvas:disconnect"

    def test_state_events(self):
        """Test state event types."""
        assert CanvasEventType.STATE.value == "canvas:state"
        assert CanvasEventType.SYNC.value == "canvas:sync"

    def test_node_events(self):
        """Test node event types."""
        assert CanvasEventType.NODE_CREATE.value == "canvas:node:create"
        assert CanvasEventType.NODE_UPDATE.value == "canvas:node:update"
        assert CanvasEventType.NODE_DELETE.value == "canvas:node:delete"

    def test_debate_events(self):
        """Test debate event types."""
        assert CanvasEventType.DEBATE_START.value == "canvas:debate:start"
        assert CanvasEventType.DEBATE_ROUND.value == "canvas:debate:round"
        assert CanvasEventType.DEBATE_CONSENSUS.value == "canvas:debate:consensus"


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_edge_types(self):
        """Test all edge types exist."""
        assert EdgeType.DEFAULT.value == "default"
        assert EdgeType.DATA_FLOW.value == "data_flow"
        assert EdgeType.CONTROL_FLOW.value == "control_flow"
        assert EdgeType.REFERENCE.value == "reference"
        assert EdgeType.DEPENDENCY.value == "dependency"
        assert EdgeType.CRITIQUE.value == "critique"
        assert EdgeType.SUPPORT.value == "support"


class TestCanvasNode:
    """Tests for CanvasNode dataclass."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = CanvasNode(
            id="node-1",
            node_type=CanvasNodeType.AGENT,
            label="Test Agent",
        )
        assert node.id == "node-1"
        assert node.node_type == CanvasNodeType.AGENT
        assert node.label == "Test Agent"
        assert node.locked is False
        assert node.selected is False

    def test_node_with_position_and_size(self):
        """Test node with custom position and size."""
        node = CanvasNode(
            id="node-2",
            node_type=CanvasNodeType.DEBATE,
            position=Position(100, 200),
            size=Size(300, 150),
        )
        assert node.position.x == 100
        assert node.position.y == 200
        assert node.size.width == 300
        assert node.size.height == 150

    def test_node_with_data(self):
        """Test node with custom data."""
        node = CanvasNode(
            id="node-3",
            node_type=CanvasNodeType.KNOWLEDGE,
            data={"topic": "AI Ethics", "confidence": 0.85},
        )
        assert node.data["topic"] == "AI Ethics"
        assert node.data["confidence"] == 0.85

    def test_node_to_dict(self):
        """Test node serialization."""
        node = CanvasNode(
            id="node-4",
            node_type=CanvasNodeType.TEXT,
            position=Position(50, 75),
            label="Note",
            locked=True,
        )
        data = node.to_dict()
        assert data["id"] == "node-4"
        assert data["type"] == "text"
        assert data["position"] == {"x": 50, "y": 75}
        assert data["label"] == "Note"
        assert data["locked"] is True

    def test_node_from_dict(self):
        """Test node deserialization."""
        data = {
            "id": "node-5",
            "type": "agent",
            "position": {"x": 100, "y": 100},
            "size": {"width": 250, "height": 120},
            "label": "GPT-4",
            "data": {"model": "gpt-4"},
            "locked": False,
            "selected": True,
        }
        node = CanvasNode.from_dict(data)
        assert node.id == "node-5"
        assert node.node_type == CanvasNodeType.AGENT
        assert node.position.x == 100
        assert node.size.width == 250
        assert node.label == "GPT-4"
        assert node.data["model"] == "gpt-4"
        assert node.selected is True

    def test_node_from_dict_with_parent(self):
        """Test node deserialization with parent_id."""
        data = {
            "id": "child-node",
            "type": "text",
            "parent_id": "group-1",
        }
        node = CanvasNode.from_dict(data)
        assert node.parent_id == "group-1"

    def test_node_move(self):
        """Test node move method."""
        node = CanvasNode(
            id="node-6",
            node_type=CanvasNodeType.INPUT,
            position=Position(0, 0),
        )
        original_updated = node.updated_at
        node.move(150, 250)
        assert node.position.x == 150
        assert node.position.y == 250
        assert node.updated_at >= original_updated

    def test_node_resize(self):
        """Test node resize method."""
        node = CanvasNode(
            id="node-7",
            node_type=CanvasNodeType.OUTPUT,
            size=Size(200, 100),
        )
        node.resize(400, 200)
        assert node.size.width == 400
        assert node.size.height == 200

    def test_node_update_data(self):
        """Test node update_data method."""
        node = CanvasNode(
            id="node-8",
            node_type=CanvasNodeType.EVIDENCE,
            data={"source": "Paper A"},
        )
        node.update_data(confidence=0.9, citations=5)
        assert node.data["source"] == "Paper A"
        assert node.data["confidence"] == 0.9
        assert node.data["citations"] == 5


class TestCanvasEdge:
    """Tests for CanvasEdge dataclass."""

    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = CanvasEdge(
            id="edge-1",
            source_id="node-a",
            target_id="node-b",
        )
        assert edge.id == "edge-1"
        assert edge.source_id == "node-a"
        assert edge.target_id == "node-b"
        assert edge.edge_type == EdgeType.DEFAULT
        assert edge.animated is False

    def test_edge_with_type(self):
        """Test edge with specific type."""
        edge = CanvasEdge(
            id="edge-2",
            source_id="agent-1",
            target_id="agent-2",
            edge_type=EdgeType.CRITIQUE,
            label="Refutes",
        )
        assert edge.edge_type == EdgeType.CRITIQUE
        assert edge.label == "Refutes"

    def test_edge_animated(self):
        """Test animated edge."""
        edge = CanvasEdge(
            id="edge-3",
            source_id="input",
            target_id="processor",
            edge_type=EdgeType.DATA_FLOW,
            animated=True,
        )
        assert edge.animated is True

    def test_edge_to_dict(self):
        """Test edge serialization."""
        edge = CanvasEdge(
            id="edge-4",
            source_id="a",
            target_id="b",
            edge_type=EdgeType.SUPPORT,
            label="Supports",
            data={"weight": 0.8},
        )
        data = edge.to_dict()
        assert data["id"] == "edge-4"
        assert data["source"] == "a"
        assert data["target"] == "b"
        assert data["type"] == "support"
        assert data["label"] == "Supports"
        assert data["data"]["weight"] == 0.8

    def test_edge_from_dict(self):
        """Test edge deserialization."""
        data = {
            "id": "edge-5",
            "source": "x",
            "target": "y",
            "type": "dependency",
            "animated": True,
        }
        edge = CanvasEdge.from_dict(data)
        assert edge.id == "edge-5"
        assert edge.source_id == "x"
        assert edge.target_id == "y"
        assert edge.edge_type == EdgeType.DEPENDENCY
        assert edge.animated is True

    def test_edge_from_dict_alternate_keys(self):
        """Test edge deserialization with source_id/target_id keys."""
        data = {
            "id": "edge-6",
            "source_id": "p",
            "target_id": "q",
        }
        edge = CanvasEdge.from_dict(data)
        assert edge.source_id == "p"
        assert edge.target_id == "q"


class TestCanvasEvent:
    """Tests for CanvasEvent dataclass."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = CanvasEvent(
            event_type=CanvasEventType.NODE_CREATE,
            canvas_id="canvas-1",
            node_id="node-new",
        )
        assert event.event_type == CanvasEventType.NODE_CREATE
        assert event.canvas_id == "canvas-1"
        assert event.node_id == "node-new"

    def test_event_with_data(self):
        """Test event with data payload."""
        event = CanvasEvent(
            event_type=CanvasEventType.AGENT_MESSAGE,
            canvas_id="canvas-2",
            data={"message": "Hello", "agent": "claude"},
            user_id="user-123",
        )
        assert event.data["message"] == "Hello"
        assert event.user_id == "user-123"

    def test_event_to_dict(self):
        """Test event serialization."""
        event = CanvasEvent(
            event_type=CanvasEventType.DEBATE_START,
            canvas_id="canvas-3",
            data={"topic": "AI Safety"},
            node_id="debate-node",
        )
        data = event.to_dict()
        assert data["type"] == "canvas:debate:start"
        assert data["canvas_id"] == "canvas-3"
        assert data["data"]["topic"] == "AI Safety"
        assert data["node_id"] == "debate-node"

    def test_event_from_dict(self):
        """Test event deserialization."""
        data = {
            "type": "canvas:node:move",
            "canvas_id": "canvas-4",
            "node_id": "moved-node",
            "data": {"x": 100, "y": 200},
        }
        event = CanvasEvent.from_dict(data)
        assert event.event_type == CanvasEventType.NODE_MOVE
        assert event.canvas_id == "canvas-4"
        assert event.node_id == "moved-node"


class TestCanvas:
    """Tests for Canvas container."""

    def test_canvas_creation(self):
        """Test basic canvas creation."""
        canvas = Canvas(id="canvas-1")
        assert canvas.id == "canvas-1"
        assert canvas.name == "Untitled Canvas"
        assert len(canvas.nodes) == 0
        assert len(canvas.edges) == 0

    def test_canvas_with_name(self):
        """Test canvas with custom name."""
        canvas = Canvas(id="canvas-2", name="My Debate Canvas")
        assert canvas.name == "My Debate Canvas"

    def test_canvas_add_node(self):
        """Test adding a node to canvas."""
        canvas = Canvas(id="canvas-3")
        node = canvas.add_node(
            CanvasNodeType.AGENT,
            position=Position(100, 100),
            label="Test Agent",
        )
        assert node.id in canvas.nodes
        assert canvas.nodes[node.id].label == "Test Agent"

    def test_canvas_add_multiple_nodes(self):
        """Test adding multiple nodes."""
        canvas = Canvas(id="canvas-4")
        node1 = canvas.add_node(CanvasNodeType.INPUT)
        node2 = canvas.add_node(CanvasNodeType.OUTPUT)
        assert len(canvas.nodes) == 2
        assert node1.id != node2.id

    def test_canvas_remove_node(self):
        """Test removing a node."""
        canvas = Canvas(id="canvas-5")
        node = canvas.add_node(CanvasNodeType.TEXT)
        removed = canvas.remove_node(node.id)
        assert removed.id == node.id
        assert node.id not in canvas.nodes

    def test_canvas_remove_nonexistent_node(self):
        """Test removing a node that doesn't exist."""
        canvas = Canvas(id="canvas-6")
        removed = canvas.remove_node("nonexistent")
        assert removed is None

    def test_canvas_remove_node_removes_edges(self):
        """Test that removing a node also removes connected edges."""
        canvas = Canvas(id="canvas-7")
        node1 = canvas.add_node(CanvasNodeType.INPUT)
        node2 = canvas.add_node(CanvasNodeType.OUTPUT)
        node3 = canvas.add_node(CanvasNodeType.TEXT)
        edge1 = canvas.add_edge(node1.id, node2.id)
        edge2 = canvas.add_edge(node2.id, node3.id)

        canvas.remove_node(node2.id)

        assert edge1.id not in canvas.edges
        assert edge2.id not in canvas.edges
        assert node1.id in canvas.nodes
        assert node3.id in canvas.nodes

    def test_canvas_get_node(self):
        """Test getting a node by ID."""
        canvas = Canvas(id="canvas-8")
        node = canvas.add_node(CanvasNodeType.KNOWLEDGE, label="Test")
        retrieved = canvas.get_node(node.id)
        assert retrieved.label == "Test"

    def test_canvas_get_nonexistent_node(self):
        """Test getting a nonexistent node."""
        canvas = Canvas(id="canvas-9")
        assert canvas.get_node("nonexistent") is None

    def test_canvas_add_edge(self):
        """Test adding an edge between nodes."""
        canvas = Canvas(id="canvas-10")
        node1 = canvas.add_node(CanvasNodeType.AGENT)
        node2 = canvas.add_node(CanvasNodeType.AGENT)
        edge = canvas.add_edge(node1.id, node2.id, EdgeType.CRITIQUE)

        assert edge is not None
        assert edge.id in canvas.edges
        assert edge.edge_type == EdgeType.CRITIQUE

    def test_canvas_add_edge_invalid_nodes(self):
        """Test adding an edge with invalid node IDs."""
        canvas = Canvas(id="canvas-11")
        node = canvas.add_node(CanvasNodeType.TEXT)
        edge = canvas.add_edge(node.id, "nonexistent")
        assert edge is None

    def test_canvas_remove_edge(self):
        """Test removing an edge."""
        canvas = Canvas(id="canvas-12")
        node1 = canvas.add_node(CanvasNodeType.INPUT)
        node2 = canvas.add_node(CanvasNodeType.OUTPUT)
        edge = canvas.add_edge(node1.id, node2.id)

        removed = canvas.remove_edge(edge.id)
        assert removed.id == edge.id
        assert edge.id not in canvas.edges

    def test_canvas_get_edge(self):
        """Test getting an edge by ID."""
        canvas = Canvas(id="canvas-13")
        node1 = canvas.add_node(CanvasNodeType.AGENT)
        node2 = canvas.add_node(CanvasNodeType.DEBATE)
        edge = canvas.add_edge(node1.id, node2.id, label="participates")

        retrieved = canvas.get_edge(edge.id)
        assert retrieved.label == "participates"

    def test_canvas_get_connected_nodes(self):
        """Test getting nodes connected to a given node."""
        canvas = Canvas(id="canvas-14")
        center = canvas.add_node(CanvasNodeType.DEBATE)
        agent1 = canvas.add_node(CanvasNodeType.AGENT)
        agent2 = canvas.add_node(CanvasNodeType.AGENT)
        evidence = canvas.add_node(CanvasNodeType.EVIDENCE)

        canvas.add_edge(agent1.id, center.id)
        canvas.add_edge(center.id, agent2.id)
        canvas.add_edge(evidence.id, center.id)

        connected = canvas.get_connected_nodes(center.id)
        connected_ids = {n.id for n in connected}

        assert agent1.id in connected_ids
        assert agent2.id in connected_ids
        assert evidence.id in connected_ids
        assert center.id not in connected_ids

    def test_canvas_get_edges_for_node(self):
        """Test getting all edges for a node."""
        canvas = Canvas(id="canvas-15")
        node1 = canvas.add_node(CanvasNodeType.AGENT)
        node2 = canvas.add_node(CanvasNodeType.AGENT)
        node3 = canvas.add_node(CanvasNodeType.AGENT)

        edge1 = canvas.add_edge(node1.id, node2.id)
        edge2 = canvas.add_edge(node2.id, node3.id)
        edge3 = canvas.add_edge(node3.id, node1.id)

        edges = canvas.get_edges_for_node(node1.id)
        edge_ids = {e.id for e in edges}

        assert edge1.id in edge_ids
        assert edge3.id in edge_ids
        assert edge2.id not in edge_ids

    def test_canvas_get_nodes_by_type(self):
        """Test getting nodes by type."""
        canvas = Canvas(id="canvas-16")
        canvas.add_node(CanvasNodeType.AGENT, label="Agent 1")
        canvas.add_node(CanvasNodeType.AGENT, label="Agent 2")
        canvas.add_node(CanvasNodeType.DEBATE)
        canvas.add_node(CanvasNodeType.EVIDENCE)

        agents = canvas.get_nodes_by_type(CanvasNodeType.AGENT)
        assert len(agents) == 2
        assert all(n.node_type == CanvasNodeType.AGENT for n in agents)

    def test_canvas_clear(self):
        """Test clearing all nodes and edges."""
        canvas = Canvas(id="canvas-17")
        node1 = canvas.add_node(CanvasNodeType.INPUT)
        node2 = canvas.add_node(CanvasNodeType.OUTPUT)
        canvas.add_edge(node1.id, node2.id)

        canvas.clear()

        assert len(canvas.nodes) == 0
        assert len(canvas.edges) == 0

    def test_canvas_to_dict(self):
        """Test canvas serialization."""
        canvas = Canvas(id="canvas-18", name="Test Canvas")
        node = canvas.add_node(CanvasNodeType.TEXT, label="Note")

        data = canvas.to_dict()

        assert data["id"] == "canvas-18"
        assert data["name"] == "Test Canvas"
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["label"] == "Note"

    def test_canvas_from_dict(self):
        """Test canvas deserialization."""
        data = {
            "id": "canvas-19",
            "name": "Loaded Canvas",
            "nodes": [
                {"id": "n1", "type": "agent", "label": "Agent A"},
                {"id": "n2", "type": "debate", "label": "Main Debate"},
            ],
            "edges": [
                {"id": "e1", "source": "n1", "target": "n2"},
            ],
            "metadata": {"version": 1},
        }
        canvas = Canvas.from_dict(data)

        assert canvas.id == "canvas-19"
        assert canvas.name == "Loaded Canvas"
        assert len(canvas.nodes) == 2
        assert len(canvas.edges) == 1
        assert "n1" in canvas.nodes
        assert canvas.metadata["version"] == 1

    def test_canvas_owner_and_workspace(self):
        """Test canvas owner and workspace fields."""
        canvas = Canvas(
            id="canvas-20",
            owner_id="user-123",
            workspace_id="workspace-456",
        )
        assert canvas.owner_id == "user-123"
        assert canvas.workspace_id == "workspace-456"

        data = canvas.to_dict()
        assert data["owner_id"] == "user-123"
        assert data["workspace_id"] == "workspace-456"

    def test_canvas_timestamps(self):
        """Test canvas timestamps."""
        canvas = Canvas(id="canvas-21")
        assert canvas.created_at is not None
        assert canvas.updated_at is not None

        original_updated = canvas.updated_at
        canvas.add_node(CanvasNodeType.TEXT)
        assert canvas.updated_at >= original_updated
