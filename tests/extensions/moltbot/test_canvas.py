"""
Tests for Moltbot Canvas component.

Tests canvas creation, element management, layers, and collaboration.
"""

import pytest
from pathlib import Path

from aragora.extensions.moltbot import (
    Canvas,
    CanvasConfig,
    CanvasElement,
    CanvasLayer,
    CanvasManager,
    ElementType,
)


class TestCanvasManager:
    """Tests for CanvasManager."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CanvasManager:
        """Create a canvas manager for testing."""
        return CanvasManager(storage_path=tmp_path / "canvas")

    @pytest.fixture
    async def canvas(self, manager: CanvasManager) -> Canvas:
        """Create a test canvas."""
        config = CanvasConfig(name="Test Canvas", width=1920, height=1080)
        return await manager.create_canvas(config, owner_id="test-user")

    @pytest.mark.asyncio
    async def test_create_canvas(self, manager: CanvasManager):
        """Test canvas creation."""
        config = CanvasConfig(name="My Canvas", width=1280, height=720)
        canvas = await manager.create_canvas(config, owner_id="user-1")

        assert canvas.id is not None
        assert canvas.config.name == "My Canvas"
        assert canvas.config.width == 1280
        assert canvas.config.height == 720
        assert canvas.owner_id == "user-1"
        assert canvas.status == "active"
        assert len(canvas.layers) == 1  # Default layer

    @pytest.mark.asyncio
    async def test_get_canvas(self, manager: CanvasManager, canvas: Canvas):
        """Test getting canvas by ID."""
        retrieved = await manager.get_canvas(canvas.id)

        assert retrieved is not None
        assert retrieved.id == canvas.id
        assert retrieved.config.name == canvas.config.name

    @pytest.mark.asyncio
    async def test_get_nonexistent_canvas(self, manager: CanvasManager):
        """Test getting nonexistent canvas."""
        result = await manager.get_canvas("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_canvases(self, manager: CanvasManager):
        """Test listing canvases."""
        # Create multiple canvases
        config1 = CanvasConfig(name="Canvas 1")
        config2 = CanvasConfig(name="Canvas 2")
        await manager.create_canvas(config1, owner_id="user-1")
        await manager.create_canvas(config2, owner_id="user-2")

        all_canvases = await manager.list_canvases()
        assert len(all_canvases) == 2

        user1_canvases = await manager.list_canvases(owner_id="user-1")
        assert len(user1_canvases) == 1
        assert user1_canvases[0].config.name == "Canvas 1"

    @pytest.mark.asyncio
    async def test_delete_canvas(self, manager: CanvasManager, canvas: Canvas):
        """Test canvas deletion."""
        result = await manager.delete_canvas(canvas.id)
        assert result is True

        retrieved = await manager.get_canvas(canvas.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_canvas(self, manager: CanvasManager):
        """Test deleting nonexistent canvas."""
        result = await manager.delete_canvas("nonexistent")
        assert result is False


class TestCanvasLayers:
    """Tests for canvas layer management."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CanvasManager:
        """Create a canvas manager for testing."""
        return CanvasManager(storage_path=tmp_path / "canvas")

    @pytest.fixture
    async def canvas(self, manager: CanvasManager) -> Canvas:
        """Create a test canvas."""
        config = CanvasConfig(name="Test Canvas")
        return await manager.create_canvas(config, owner_id="test-user")

    @pytest.mark.asyncio
    async def test_canvas_has_default_layer(self, canvas: Canvas, manager: CanvasManager):
        """Test canvas has default layer."""
        assert len(canvas.layers) == 1
        assert canvas.active_layer is not None

        layer = await manager.get_layer(canvas.active_layer)
        assert layer is not None
        assert layer.name == "Background"

    @pytest.mark.asyncio
    async def test_add_layer(self, manager: CanvasManager, canvas: Canvas):
        """Test adding a layer."""
        layer = await manager.add_layer(canvas.id, "Foreground")

        assert layer is not None
        assert layer.name == "Foreground"
        assert layer.z_index == 1  # Above background

        updated_canvas = await manager.get_canvas(canvas.id)
        assert len(updated_canvas.layers) == 2

    @pytest.mark.asyncio
    async def test_add_layer_to_nonexistent_canvas(self, manager: CanvasManager):
        """Test adding layer to nonexistent canvas."""
        result = await manager.add_layer("nonexistent", "Test")
        assert result is None


class TestCanvasElements:
    """Tests for canvas element management."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CanvasManager:
        """Create a canvas manager for testing."""
        return CanvasManager(storage_path=tmp_path / "canvas")

    @pytest.fixture
    async def canvas(self, manager: CanvasManager) -> Canvas:
        """Create a test canvas."""
        config = CanvasConfig(name="Test Canvas")
        return await manager.create_canvas(config, owner_id="test-user")

    @pytest.mark.asyncio
    async def test_add_element(self, manager: CanvasManager, canvas: Canvas):
        """Test adding an element."""
        element = await manager.add_element(
            canvas_id=canvas.id,
            element_type=ElementType.TEXT,
            x=100,
            y=100,
            width=200,
            height=50,
            content={"text": "Hello World"},
            created_by="user-1",
        )

        assert element is not None
        assert element.type == ElementType.TEXT
        assert element.x == 100
        assert element.y == 100
        assert element.content["text"] == "Hello World"

    @pytest.mark.asyncio
    async def test_add_element_increments_version(self, manager: CanvasManager, canvas: Canvas):
        """Test adding element increments canvas version."""
        initial_version = canvas.version

        await manager.add_element(
            canvas_id=canvas.id,
            element_type=ElementType.SHAPE,
            x=0,
            y=0,
            width=100,
            height=100,
        )

        updated_canvas = await manager.get_canvas(canvas.id)
        assert updated_canvas.version == initial_version + 1

    @pytest.mark.asyncio
    async def test_update_element(self, manager: CanvasManager, canvas: Canvas):
        """Test updating an element."""
        element = await manager.add_element(
            canvas_id=canvas.id,
            element_type=ElementType.SHAPE,
            x=0,
            y=0,
            width=100,
            height=100,
        )

        updated = await manager.update_element(
            canvas_id=canvas.id,
            element_id=element.id,
            updates={"x": 50, "y": 50, "rotation": 45},
        )

        assert updated is not None
        assert updated.x == 50
        assert updated.y == 50
        assert updated.rotation == 45

    @pytest.mark.asyncio
    async def test_update_locked_element_fails(self, manager: CanvasManager, canvas: Canvas):
        """Test updating locked element fails."""
        element = await manager.add_element(
            canvas_id=canvas.id,
            element_type=ElementType.TEXT,
            x=0,
            y=0,
            width=100,
            height=50,
        )

        # Lock the element
        await manager.update_element(
            canvas_id=canvas.id,
            element_id=element.id,
            updates={"locked": True},
        )

        # Try to update locked element
        with pytest.raises(ValueError, match="locked"):
            await manager.update_element(
                canvas_id=canvas.id,
                element_id=element.id,
                updates={"x": 100},
            )

    @pytest.mark.asyncio
    async def test_delete_element(self, manager: CanvasManager, canvas: Canvas):
        """Test deleting an element."""
        element = await manager.add_element(
            canvas_id=canvas.id,
            element_type=ElementType.IMAGE,
            x=0,
            y=0,
            width=300,
            height=200,
        )

        result = await manager.delete_element(canvas.id, element.id)
        assert result is True

        deleted = await manager.get_element(element.id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_list_elements(self, manager: CanvasManager, canvas: Canvas):
        """Test listing elements."""
        for i in range(5):
            await manager.add_element(
                canvas_id=canvas.id,
                element_type=ElementType.TEXT,
                x=i * 50,
                y=0,
                width=40,
                height=20,
            )

        elements = await manager.list_elements(canvas.id)
        assert len(elements) == 5

    @pytest.mark.asyncio
    async def test_max_elements_limit(self, manager: CanvasManager):
        """Test max elements limit."""
        config = CanvasConfig(name="Limited", max_elements=3)
        canvas = await manager.create_canvas(config, owner_id="user-1")

        # Add elements up to limit
        for i in range(3):
            await manager.add_element(
                canvas_id=canvas.id,
                element_type=ElementType.SHAPE,
                x=0,
                y=0,
                width=10,
                height=10,
            )

        # Adding one more should fail
        with pytest.raises(ValueError, match="max elements"):
            await manager.add_element(
                canvas_id=canvas.id,
                element_type=ElementType.SHAPE,
                x=0,
                y=0,
                width=10,
                height=10,
            )


class TestCanvasCollaboration:
    """Tests for canvas collaboration features."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CanvasManager:
        """Create a canvas manager for testing."""
        return CanvasManager(storage_path=tmp_path / "canvas")

    @pytest.fixture
    async def canvas(self, manager: CanvasManager) -> Canvas:
        """Create a test canvas."""
        config = CanvasConfig(name="Collab Canvas", max_collaborators=3)
        return await manager.create_canvas(config, owner_id="test-user")

    @pytest.mark.asyncio
    async def test_join_canvas(self, manager: CanvasManager, canvas: Canvas):
        """Test joining a canvas."""
        result = await manager.join_canvas(canvas.id, "user-1", cursor_color="#ff0000")
        assert result is True

        updated = await manager.get_canvas(canvas.id)
        assert "user-1" in updated.active_users
        assert updated.active_users["user-1"]["cursor_color"] == "#ff0000"
        assert "user-1" in updated.collaborators

    @pytest.mark.asyncio
    async def test_leave_canvas(self, manager: CanvasManager, canvas: Canvas):
        """Test leaving a canvas."""
        await manager.join_canvas(canvas.id, "user-1")
        result = await manager.leave_canvas(canvas.id, "user-1")
        assert result is True

        updated = await manager.get_canvas(canvas.id)
        assert "user-1" not in updated.active_users

    @pytest.mark.asyncio
    async def test_update_cursor(self, manager: CanvasManager, canvas: Canvas):
        """Test cursor position update."""
        await manager.join_canvas(canvas.id, "user-1")

        result = await manager.update_cursor(canvas.id, "user-1", x=150, y=200)
        assert result is True

        updated = await manager.get_canvas(canvas.id)
        assert updated.active_users["user-1"]["cursor_x"] == 150
        assert updated.active_users["user-1"]["cursor_y"] == 200

    @pytest.mark.asyncio
    async def test_max_collaborators_limit(self, manager: CanvasManager, canvas: Canvas):
        """Test max collaborators limit."""
        await manager.join_canvas(canvas.id, "user-1")
        await manager.join_canvas(canvas.id, "user-2")
        await manager.join_canvas(canvas.id, "user-3")

        with pytest.raises(ValueError, match="max collaborators"):
            await manager.join_canvas(canvas.id, "user-4")


class TestCanvasAIGeneration:
    """Tests for AI element generation."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CanvasManager:
        """Create a canvas manager for testing."""
        return CanvasManager(storage_path=tmp_path / "canvas")

    @pytest.fixture
    async def canvas(self, manager: CanvasManager) -> Canvas:
        """Create a test canvas."""
        config = CanvasConfig(name="AI Canvas")
        return await manager.create_canvas(config, owner_id="test-user")

    @pytest.mark.asyncio
    async def test_generate_element(self, manager: CanvasManager, canvas: Canvas):
        """Test AI element generation."""
        element = await manager.generate_element(
            canvas_id=canvas.id,
            prompt="A futuristic city skyline",
            x=100,
            y=100,
            created_by="user-1",
        )

        assert element is not None
        assert element.type == ElementType.AI_GENERATED
        assert element.content["prompt"] == "A futuristic city skyline"
        assert element.content["generated"] is True


class TestCanvasStats:
    """Tests for canvas statistics."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CanvasManager:
        """Create a canvas manager for testing."""
        return CanvasManager(storage_path=tmp_path / "canvas")

    @pytest.mark.asyncio
    async def test_get_stats(self, manager: CanvasManager):
        """Test getting canvas stats."""
        # Create some canvases and elements
        config = CanvasConfig(name="Test")
        canvas = await manager.create_canvas(config, owner_id="user-1")

        for _ in range(3):
            await manager.add_element(
                canvas_id=canvas.id,
                element_type=ElementType.TEXT,
                x=0,
                y=0,
                width=100,
                height=50,
            )

        stats = await manager.get_stats()

        assert stats["canvases_total"] == 1
        assert stats["canvases_active"] == 1
        assert stats["elements_total"] == 3
        assert stats["elements_by_type"]["text"] == 3
