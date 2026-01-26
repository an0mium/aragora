"""
Tests for CanvasStateManager.

Tests cover:
- Canvas CRUD operations
- Node management
- Edge management
- Event broadcasting and subscriptions
- History management
- User selections
- Canvas actions
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import asyncio

from aragora.canvas.manager import CanvasStateManager, get_canvas_manager
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


class TestCanvasStateManagerInit:
    """Tests for CanvasStateManager initialization."""

    def test_default_initialization(self):
        """Test default manager initialization."""
        manager = CanvasStateManager()
        assert manager._max_history == 100
        assert len(manager._canvases) == 0
        assert len(manager._subscribers) == 0

    def test_custom_max_history(self):
        """Test manager with custom max_history."""
        manager = CanvasStateManager(max_history=50)
        assert manager._max_history == 50


class TestCanvasManagement:
    """Tests for canvas CRUD operations."""

    @pytest.fixture
    def manager(self):
        """Create a fresh manager for each test."""
        return CanvasStateManager()

    @pytest.mark.asyncio
    async def test_create_canvas(self, manager):
        """Test creating a new canvas."""
        canvas = await manager.create_canvas(name="Test Canvas")

        assert canvas is not None
        assert canvas.name == "Test Canvas"
        assert canvas.id is not None
        assert canvas.id in manager._canvases

    @pytest.mark.asyncio
    async def test_create_canvas_with_id(self, manager):
        """Test creating a canvas with specific ID."""
        canvas = await manager.create_canvas(
            canvas_id="my-canvas",
            name="My Canvas",
        )

        assert canvas.id == "my-canvas"
        assert "my-canvas" in manager._canvases

    @pytest.mark.asyncio
    async def test_create_canvas_with_owner(self, manager):
        """Test creating a canvas with owner."""
        canvas = await manager.create_canvas(
            name="Owned Canvas",
            owner_id="user-123",
            workspace_id="workspace-456",
        )

        assert canvas.owner_id == "user-123"
        assert canvas.workspace_id == "workspace-456"

    @pytest.mark.asyncio
    async def test_create_canvas_with_metadata(self, manager):
        """Test creating a canvas with metadata."""
        canvas = await manager.create_canvas(
            name="Meta Canvas",
            description="A test canvas",
            version=1,
        )

        assert canvas.metadata["description"] == "A test canvas"
        assert canvas.metadata["version"] == 1

    @pytest.mark.asyncio
    async def test_get_canvas(self, manager):
        """Test getting a canvas by ID."""
        created = await manager.create_canvas(canvas_id="test-get")
        retrieved = await manager.get_canvas("test-get")

        assert retrieved is not None
        assert retrieved.id == created.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_canvas(self, manager):
        """Test getting a canvas that doesn't exist."""
        result = await manager.get_canvas("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_canvas_existing(self, manager):
        """Test get_or_create with existing canvas."""
        await manager.create_canvas(canvas_id="existing", name="Existing")
        canvas = await manager.get_or_create_canvas(
            canvas_id="existing",
            name="Different Name",
        )

        assert canvas.name == "Existing"  # Original name preserved

    @pytest.mark.asyncio
    async def test_get_or_create_canvas_new(self, manager):
        """Test get_or_create with new canvas."""
        canvas = await manager.get_or_create_canvas(
            canvas_id="new-canvas",
            name="New Canvas",
        )

        assert canvas.name == "New Canvas"
        assert canvas.id == "new-canvas"

    @pytest.mark.asyncio
    async def test_delete_canvas(self, manager):
        """Test deleting a canvas."""
        await manager.create_canvas(canvas_id="to-delete")
        result = await manager.delete_canvas("to-delete")

        assert result is True
        assert "to-delete" not in manager._canvases

    @pytest.mark.asyncio
    async def test_delete_nonexistent_canvas(self, manager):
        """Test deleting a canvas that doesn't exist."""
        result = await manager.delete_canvas("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_canvases(self, manager):
        """Test listing all canvases."""
        await manager.create_canvas(name="Canvas 1")
        await manager.create_canvas(name="Canvas 2")

        canvases = await manager.list_canvases()
        assert len(canvases) == 2

    @pytest.mark.asyncio
    async def test_list_canvases_by_owner(self, manager):
        """Test listing canvases filtered by owner."""
        await manager.create_canvas(name="Canvas 1", owner_id="user-1")
        await manager.create_canvas(name="Canvas 2", owner_id="user-2")
        await manager.create_canvas(name="Canvas 3", owner_id="user-1")

        canvases = await manager.list_canvases(owner_id="user-1")
        assert len(canvases) == 2

    @pytest.mark.asyncio
    async def test_list_canvases_by_workspace(self, manager):
        """Test listing canvases filtered by workspace."""
        await manager.create_canvas(name="Canvas 1", workspace_id="ws-a")
        await manager.create_canvas(name="Canvas 2", workspace_id="ws-b")

        canvases = await manager.list_canvases(workspace_id="ws-a")
        assert len(canvases) == 1


class TestNodeOperations:
    """Tests for node operations."""

    @pytest.fixture
    async def manager_with_canvas(self):
        """Create a manager with a canvas."""
        manager = CanvasStateManager()
        await manager.create_canvas(canvas_id="test-canvas")
        return manager

    @pytest.mark.asyncio
    async def test_add_node(self, manager_with_canvas):
        """Test adding a node."""
        manager = manager_with_canvas
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.AGENT,
            position=Position(100, 200),
            label="Test Agent",
        )

        assert node is not None
        assert node.node_type == CanvasNodeType.AGENT
        assert node.label == "Test Agent"
        assert node.position.x == 100

    @pytest.mark.asyncio
    async def test_add_node_with_data(self, manager_with_canvas):
        """Test adding a node with custom data."""
        manager = manager_with_canvas
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.KNOWLEDGE,
            position=Position(0, 0),
            data={"topic": "AI Safety"},
        )

        assert node.data["topic"] == "AI Safety"

    @pytest.mark.asyncio
    async def test_add_node_invalid_canvas(self, manager_with_canvas):
        """Test adding a node to nonexistent canvas."""
        manager = manager_with_canvas
        node = await manager.add_node(
            canvas_id="nonexistent",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
        )

        assert node is None

    @pytest.mark.asyncio
    async def test_update_node(self, manager_with_canvas):
        """Test updating a node."""
        manager = manager_with_canvas
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
            label="Original",
        )

        updated = await manager.update_node(
            canvas_id="test-canvas",
            node_id=node.id,
            label="Updated",
            data={"key": "value"},
        )

        assert updated.label == "Updated"
        assert updated.data["key"] == "value"

    @pytest.mark.asyncio
    async def test_update_node_style(self, manager_with_canvas):
        """Test updating a node's style."""
        manager = manager_with_canvas
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.INPUT,
            position=Position(0, 0),
        )

        updated = await manager.update_node(
            canvas_id="test-canvas",
            node_id=node.id,
            style={"backgroundColor": "blue"},
        )

        assert updated.style["backgroundColor"] == "blue"

    @pytest.mark.asyncio
    async def test_update_nonexistent_node(self, manager_with_canvas):
        """Test updating a nonexistent node."""
        manager = manager_with_canvas
        result = await manager.update_node(
            canvas_id="test-canvas",
            node_id="nonexistent",
            label="New Label",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_move_node(self, manager_with_canvas):
        """Test moving a node."""
        manager = manager_with_canvas
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
        )

        moved = await manager.move_node(
            canvas_id="test-canvas",
            node_id=node.id,
            x=150,
            y=250,
        )

        assert moved.position.x == 150
        assert moved.position.y == 250

    @pytest.mark.asyncio
    async def test_move_locked_node(self, manager_with_canvas):
        """Test moving a locked node fails."""
        manager = manager_with_canvas
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(100, 100),
            locked=True,
        )

        result = await manager.move_node(
            canvas_id="test-canvas",
            node_id=node.id,
            x=200,
            y=200,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_resize_node(self, manager_with_canvas):
        """Test resizing a node."""
        manager = manager_with_canvas
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.OUTPUT,
            position=Position(0, 0),
        )

        resized = await manager.resize_node(
            canvas_id="test-canvas",
            node_id=node.id,
            width=400,
            height=300,
        )

        assert resized.size.width == 400
        assert resized.size.height == 300

    @pytest.mark.asyncio
    async def test_resize_locked_node(self, manager_with_canvas):
        """Test resizing a locked node fails."""
        manager = manager_with_canvas
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
            locked=True,
        )

        result = await manager.resize_node(
            canvas_id="test-canvas",
            node_id=node.id,
            width=500,
            height=400,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_node(self, manager_with_canvas):
        """Test deleting a node."""
        manager = manager_with_canvas
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
        )

        result = await manager.delete_node(
            canvas_id="test-canvas",
            node_id=node.id,
        )

        assert result is True
        canvas = await manager.get_canvas("test-canvas")
        assert node.id not in canvas.nodes

    @pytest.mark.asyncio
    async def test_delete_nonexistent_node(self, manager_with_canvas):
        """Test deleting a nonexistent node."""
        manager = manager_with_canvas
        result = await manager.delete_node(
            canvas_id="test-canvas",
            node_id="nonexistent",
        )

        assert result is False


class TestEdgeOperations:
    """Tests for edge operations."""

    @pytest.fixture
    async def manager_with_nodes(self):
        """Create a manager with a canvas and nodes."""
        manager = CanvasStateManager()
        await manager.create_canvas(canvas_id="test-canvas")
        node1 = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.AGENT,
            position=Position(0, 0),
        )
        node2 = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.DEBATE,
            position=Position(200, 0),
        )
        return manager, node1, node2

    @pytest.mark.asyncio
    async def test_add_edge(self, manager_with_nodes):
        """Test adding an edge."""
        manager, node1, node2 = manager_with_nodes
        edge = await manager.add_edge(
            canvas_id="test-canvas",
            source_id=node1.id,
            target_id=node2.id,
        )

        assert edge is not None
        assert edge.source_id == node1.id
        assert edge.target_id == node2.id

    @pytest.mark.asyncio
    async def test_add_edge_with_type(self, manager_with_nodes):
        """Test adding an edge with specific type."""
        manager, node1, node2 = manager_with_nodes
        edge = await manager.add_edge(
            canvas_id="test-canvas",
            source_id=node1.id,
            target_id=node2.id,
            edge_type=EdgeType.CRITIQUE,
            label="Refutes",
        )

        assert edge.edge_type == EdgeType.CRITIQUE
        assert edge.label == "Refutes"

    @pytest.mark.asyncio
    async def test_add_edge_invalid_nodes(self, manager_with_nodes):
        """Test adding an edge with invalid node IDs."""
        manager, node1, node2 = manager_with_nodes
        edge = await manager.add_edge(
            canvas_id="test-canvas",
            source_id=node1.id,
            target_id="nonexistent",
        )

        assert edge is None

    @pytest.mark.asyncio
    async def test_update_edge(self, manager_with_nodes):
        """Test updating an edge."""
        manager, node1, node2 = manager_with_nodes
        edge = await manager.add_edge(
            canvas_id="test-canvas",
            source_id=node1.id,
            target_id=node2.id,
            label="Original",
        )

        updated = await manager.update_edge(
            canvas_id="test-canvas",
            edge_id=edge.id,
            label="Updated",
            animated=True,
        )

        assert updated.label == "Updated"
        assert updated.animated is True

    @pytest.mark.asyncio
    async def test_delete_edge(self, manager_with_nodes):
        """Test deleting an edge."""
        manager, node1, node2 = manager_with_nodes
        edge = await manager.add_edge(
            canvas_id="test-canvas",
            source_id=node1.id,
            target_id=node2.id,
        )

        result = await manager.delete_edge(
            canvas_id="test-canvas",
            edge_id=edge.id,
        )

        assert result is True
        canvas = await manager.get_canvas("test-canvas")
        assert edge.id not in canvas.edges


class TestSubscriptions:
    """Tests for event subscriptions."""

    @pytest.fixture
    async def manager_with_canvas(self):
        """Create a manager with a canvas."""
        manager = CanvasStateManager()
        await manager.create_canvas(canvas_id="test-canvas")
        return manager

    @pytest.mark.asyncio
    async def test_subscribe(self, manager_with_canvas):
        """Test subscribing to canvas events."""
        manager = manager_with_canvas
        callback = AsyncMock()

        result = await manager.subscribe("test-canvas", callback)

        assert result is True
        assert callback in manager._subscribers["test-canvas"]

    @pytest.mark.asyncio
    async def test_unsubscribe(self, manager_with_canvas):
        """Test unsubscribing from canvas events."""
        manager = manager_with_canvas
        callback = AsyncMock()
        await manager.subscribe("test-canvas", callback)

        result = await manager.unsubscribe("test-canvas", callback)

        assert result is True
        assert callback not in manager._subscribers["test-canvas"]

    @pytest.mark.asyncio
    async def test_event_broadcast_on_node_add(self, manager_with_canvas):
        """Test that adding a node broadcasts an event."""
        manager = manager_with_canvas
        received_events = []

        async def callback(event: CanvasEvent):
            received_events.append(event)

        await manager.subscribe("test-canvas", callback)
        await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
        )

        assert len(received_events) == 1
        assert received_events[0].event_type == CanvasEventType.NODE_CREATE

    @pytest.mark.asyncio
    async def test_event_broadcast_on_move(self, manager_with_canvas):
        """Test that moving a node broadcasts an event."""
        manager = manager_with_canvas
        received_events = []

        async def callback(event: CanvasEvent):
            received_events.append(event)

        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
        )

        await manager.subscribe("test-canvas", callback)
        await manager.move_node("test-canvas", node.id, 100, 100)

        assert len(received_events) == 1
        assert received_events[0].event_type == CanvasEventType.NODE_MOVE

    @pytest.mark.asyncio
    async def test_broadcast_handles_callback_errors(self, manager_with_canvas):
        """Test that broadcast continues if a callback raises."""
        manager = manager_with_canvas
        successful_callback = AsyncMock()

        async def failing_callback(event):
            raise Exception("Callback failed")

        await manager.subscribe("test-canvas", failing_callback)
        await manager.subscribe("test-canvas", successful_callback)

        await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
        )

        # Successful callback should still be called
        assert successful_callback.called


class TestHistory:
    """Tests for history management."""

    @pytest.fixture
    async def manager_with_canvas(self):
        """Create a manager with a canvas."""
        manager = CanvasStateManager(max_history=10)
        await manager.create_canvas(canvas_id="test-canvas")
        return manager

    @pytest.mark.asyncio
    async def test_history_records_events(self, manager_with_canvas):
        """Test that history records events."""
        manager = manager_with_canvas

        await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
        )

        history = await manager.get_history("test-canvas")
        assert len(history) == 1
        assert history[0].event_type == CanvasEventType.NODE_CREATE

    @pytest.mark.asyncio
    async def test_history_respects_max_limit(self, manager_with_canvas):
        """Test that history respects max_history limit."""
        manager = manager_with_canvas

        # Add 15 nodes (max is 10)
        for i in range(15):
            await manager.add_node(
                canvas_id="test-canvas",
                node_type=CanvasNodeType.TEXT,
                position=Position(i * 10, 0),
            )

        history = await manager.get_history("test-canvas")
        assert len(history) == 10

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self, manager_with_canvas):
        """Test getting history with custom limit."""
        manager = manager_with_canvas

        for i in range(5):
            await manager.add_node(
                canvas_id="test-canvas",
                node_type=CanvasNodeType.TEXT,
                position=Position(i * 10, 0),
            )

        history = await manager.get_history("test-canvas", limit=3)
        assert len(history) == 3


class TestUserSelections:
    """Tests for user selection tracking."""

    @pytest.fixture
    async def manager_with_nodes(self):
        """Create a manager with nodes."""
        manager = CanvasStateManager()
        await manager.create_canvas(canvas_id="test-canvas")
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
        )
        return manager, node

    @pytest.mark.asyncio
    async def test_select_node(self, manager_with_nodes):
        """Test selecting a node."""
        manager, node = manager_with_nodes

        await manager.select_node(
            canvas_id="test-canvas",
            node_id=node.id,
            user_id="user-1",
        )

        selections = manager._user_selections["test-canvas"]["user-1"]
        assert node.id in selections

    @pytest.mark.asyncio
    async def test_multi_select(self, manager_with_nodes):
        """Test multi-selecting nodes."""
        manager, node1 = manager_with_nodes
        node2 = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(100, 0),
        )

        await manager.select_node(
            canvas_id="test-canvas",
            node_id=node1.id,
            user_id="user-1",
        )
        await manager.select_node(
            canvas_id="test-canvas",
            node_id=node2.id,
            user_id="user-1",
            multi_select=True,
        )

        selections = manager._user_selections["test-canvas"]["user-1"]
        assert node1.id in selections
        assert node2.id in selections

    @pytest.mark.asyncio
    async def test_single_select_clears_previous(self, manager_with_nodes):
        """Test that single select clears previous selection."""
        manager, node1 = manager_with_nodes
        node2 = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(100, 0),
        )

        await manager.select_node(
            canvas_id="test-canvas",
            node_id=node1.id,
            user_id="user-1",
        )
        await manager.select_node(
            canvas_id="test-canvas",
            node_id=node2.id,
            user_id="user-1",
            multi_select=False,
        )

        selections = manager._user_selections["test-canvas"]["user-1"]
        assert node1.id not in selections
        assert node2.id in selections


class TestActions:
    """Tests for canvas actions."""

    @pytest.fixture
    async def manager_with_canvas(self):
        """Create a manager with a canvas."""
        manager = CanvasStateManager()
        await manager.create_canvas(canvas_id="test-canvas")
        return manager

    @pytest.mark.asyncio
    async def test_execute_start_debate(self, manager_with_canvas):
        """Test executing start_debate action."""
        manager = manager_with_canvas

        result = await manager.execute_action(
            canvas_id="test-canvas",
            action="start_debate",
            params={"question": "Should we use AI?"},
        )

        assert result["success"] is True
        assert "debate_node_id" in result

        canvas = await manager.get_canvas("test-canvas")
        debates = canvas.get_nodes_by_type(CanvasNodeType.DEBATE)
        assert len(debates) == 1

    @pytest.mark.asyncio
    async def test_execute_start_debate_no_question(self, manager_with_canvas):
        """Test start_debate without question fails."""
        manager = manager_with_canvas

        result = await manager.execute_action(
            canvas_id="test-canvas",
            action="start_debate",
            params={},
        )

        assert result["success"] is False
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_run_workflow(self, manager_with_canvas):
        """Test executing run_workflow action."""
        manager = manager_with_canvas

        result = await manager.execute_action(
            canvas_id="test-canvas",
            action="run_workflow",
            params={"workflow_id": "test-workflow"},
        )

        assert result["success"] is True
        assert "workflow_node_id" in result

    @pytest.mark.asyncio
    async def test_execute_query_knowledge(self, manager_with_canvas):
        """Test executing query_knowledge action."""
        manager = manager_with_canvas

        result = await manager.execute_action(
            canvas_id="test-canvas",
            action="query_knowledge",
            params={"query": "AI regulations"},
        )

        assert result["success"] is True
        assert "knowledge_node_id" in result

    @pytest.mark.asyncio
    async def test_execute_clear_canvas(self, manager_with_canvas):
        """Test executing clear_canvas action."""
        manager = manager_with_canvas

        # Add some nodes first
        await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
        )

        result = await manager.execute_action(
            canvas_id="test-canvas",
            action="clear_canvas",
            params={},
        )

        assert result["success"] is True
        canvas = await manager.get_canvas("test-canvas")
        assert len(canvas.nodes) == 0

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, manager_with_canvas):
        """Test executing unknown action fails."""
        manager = manager_with_canvas

        result = await manager.execute_action(
            canvas_id="test-canvas",
            action="unknown_action",
            params={},
        )

        assert result["success"] is False
        assert "Unknown action" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_action_invalid_canvas(self, manager_with_canvas):
        """Test executing action on nonexistent canvas."""
        manager = manager_with_canvas

        result = await manager.execute_action(
            canvas_id="nonexistent",
            action="start_debate",
            params={"question": "Test?"},
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestStateSynchronization:
    """Tests for state synchronization."""

    @pytest.fixture
    async def manager_with_data(self):
        """Create a manager with canvas, nodes, and selections."""
        manager = CanvasStateManager()
        await manager.create_canvas(canvas_id="test-canvas")
        node = await manager.add_node(
            canvas_id="test-canvas",
            node_type=CanvasNodeType.TEXT,
            position=Position(0, 0),
        )
        await manager.select_node(
            canvas_id="test-canvas",
            node_id=node.id,
            user_id="user-1",
        )
        return manager, node

    @pytest.mark.asyncio
    async def test_get_state(self, manager_with_data):
        """Test getting full canvas state."""
        manager, node = manager_with_data

        state = await manager.get_state("test-canvas")

        assert state is not None
        assert "canvas" in state
        assert "selections" in state
        assert len(state["canvas"]["nodes"]) == 1
        assert "user-1" in state["selections"]

    @pytest.mark.asyncio
    async def test_get_state_nonexistent_canvas(self):
        """Test getting state for nonexistent canvas."""
        manager = CanvasStateManager()
        state = await manager.get_state("nonexistent")
        assert state is None

    @pytest.mark.asyncio
    async def test_sync_state(self, manager_with_data):
        """Test sync_state broadcasts state event."""
        manager, _ = manager_with_data
        received_events = []

        async def callback(event: CanvasEvent):
            received_events.append(event)

        await manager.subscribe("test-canvas", callback)
        await manager.sync_state("test-canvas")

        assert len(received_events) == 1
        assert received_events[0].event_type == CanvasEventType.STATE


class TestGlobalManager:
    """Tests for global manager singleton."""

    def test_get_canvas_manager(self):
        """Test getting global canvas manager."""
        # Reset global state
        import aragora.canvas.manager as manager_module

        manager_module._manager = None

        manager1 = get_canvas_manager()
        manager2 = get_canvas_manager()

        assert manager1 is manager2

    def test_get_canvas_manager_creates_instance(self):
        """Test that get_canvas_manager creates instance if needed."""
        import aragora.canvas.manager as manager_module

        manager_module._manager = None

        manager = get_canvas_manager()
        assert manager is not None
        assert isinstance(manager, CanvasStateManager)
