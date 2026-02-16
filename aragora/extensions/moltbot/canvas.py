"""
Canvas - Real-time Visual Collaboration Surface.

Provides a shared visual canvas for multi-user real-time collaboration
with support for drawing, annotations, media embedding, and AI-generated content.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ElementType(Enum):
    """Types of canvas elements."""

    TEXT = "text"
    SHAPE = "shape"
    IMAGE = "image"
    VIDEO = "video"
    DRAWING = "drawing"
    ANNOTATION = "annotation"
    AI_GENERATED = "ai_generated"
    EMBED = "embed"
    STICKY_NOTE = "sticky_note"
    CONNECTOR = "connector"


@dataclass
class CanvasElement:
    """An element on the canvas."""

    id: str
    type: ElementType
    x: float
    y: float
    width: float
    height: float
    content: dict[str, Any] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)
    created_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    locked: bool = False
    z_index: int = 0
    rotation: float = 0.0
    opacity: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanvasLayer:
    """A layer on the canvas for organizing elements."""

    id: str
    name: str
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    z_index: int = 0
    elements: list[str] = field(default_factory=list)  # Element IDs
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CanvasConfig:
    """Configuration for a canvas."""

    name: str
    width: int = 1920
    height: int = 1080
    background_color: str = "#ffffff"
    grid_enabled: bool = True
    grid_size: int = 20
    snap_to_grid: bool = False
    max_elements: int = 10000
    max_collaborators: int = 50
    allow_anonymous: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Canvas:
    """A collaborative canvas instance."""

    id: str
    config: CanvasConfig
    owner_id: str
    tenant_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"  # active, archived, locked

    # Layers and elements
    layers: list[str] = field(default_factory=list)  # Layer IDs
    active_layer: str | None = None

    # Collaboration
    collaborators: list[str] = field(default_factory=list)
    active_users: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # user_id -> cursor/selection

    # History
    version: int = 0
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    redo_stack: list[dict[str, Any]] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


class CanvasManager:
    """
    Manages collaborative canvases with real-time synchronization.

    Provides CRUD operations for canvases and elements, with support
    for real-time collaboration via WebSocket event broadcasting.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        """
        Initialize the canvas manager.

        Args:
            storage_path: Path for canvas state storage
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._canvases: dict[str, Canvas] = {}
        self._layers: dict[str, CanvasLayer] = {}
        self._elements: dict[str, CanvasElement] = {}
        self._subscribers: dict[str, list[Callable]] = {}  # canvas_id -> callbacks
        self._lock = asyncio.Lock()

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)

    # ========== Canvas CRUD ==========

    async def create_canvas(
        self,
        config: CanvasConfig,
        owner_id: str,
        tenant_id: str | None = None,
    ) -> Canvas:
        """
        Create a new canvas.

        Args:
            config: Canvas configuration
            owner_id: Owner user ID
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            Created canvas
        """
        async with self._lock:
            canvas_id = str(uuid.uuid4())

            canvas = Canvas(
                id=canvas_id,
                config=config,
                owner_id=owner_id,
                tenant_id=tenant_id,
            )

            # Create default layer
            default_layer = await self._create_layer_internal(canvas_id, "Background", z_index=0)
            canvas.layers.append(default_layer.id)
            canvas.active_layer = default_layer.id

            self._canvases[canvas_id] = canvas
            logger.info(f"Created canvas {config.name} ({canvas_id})")

            return canvas

    async def get_canvas(self, canvas_id: str) -> Canvas | None:
        """Get a canvas by ID."""
        return self._canvases.get(canvas_id)

    async def list_canvases(
        self,
        owner_id: str | None = None,
        tenant_id: str | None = None,
        status: str | None = None,
    ) -> list[Canvas]:
        """List canvases with optional filters."""
        canvases = list(self._canvases.values())

        if owner_id:
            canvases = [c for c in canvases if c.owner_id == owner_id]
        if tenant_id:
            canvases = [c for c in canvases if c.tenant_id == tenant_id]
        if status:
            canvases = [c for c in canvases if c.status == status]

        return canvases

    async def delete_canvas(self, canvas_id: str) -> bool:
        """Delete a canvas and all its elements."""
        async with self._lock:
            canvas = self._canvases.get(canvas_id)
            if not canvas:
                return False

            # Delete all layers and elements
            for layer_id in canvas.layers:
                layer = self._layers.get(layer_id)
                if layer:
                    for element_id in layer.elements:
                        self._elements.pop(element_id, None)
                    del self._layers[layer_id]

            del self._canvases[canvas_id]
            self._subscribers.pop(canvas_id, None)

            logger.info(f"Deleted canvas {canvas_id}")
            return True

    # ========== Layer Management ==========

    async def _create_layer_internal(
        self,
        canvas_id: str,
        name: str,
        z_index: int = 0,
    ) -> CanvasLayer:
        """Internal method to create a layer."""
        layer_id = str(uuid.uuid4())
        layer = CanvasLayer(
            id=layer_id,
            name=name,
            z_index=z_index,
        )
        self._layers[layer_id] = layer
        return layer

    async def add_layer(
        self,
        canvas_id: str,
        name: str,
    ) -> CanvasLayer | None:
        """Add a new layer to a canvas."""
        async with self._lock:
            canvas = self._canvases.get(canvas_id)
            if not canvas:
                return None

            # Determine z-index (above all existing layers)
            max_z = max(
                (self._layers[lid].z_index for lid in canvas.layers if lid in self._layers),
                default=0,
            )

            layer = await self._create_layer_internal(canvas_id, name, z_index=max_z + 1)
            canvas.layers.append(layer.id)
            canvas.updated_at = datetime.now(timezone.utc)
            canvas.version += 1

            await self._broadcast(canvas_id, "layer_added", {"layer": layer})
            return layer

    async def get_layer(self, layer_id: str) -> CanvasLayer | None:
        """Get a layer by ID."""
        return self._layers.get(layer_id)

    # ========== Element Management ==========

    async def add_element(
        self,
        canvas_id: str,
        element_type: ElementType,
        x: float,
        y: float,
        width: float,
        height: float,
        content: dict[str, Any] | None = None,
        style: dict[str, Any] | None = None,
        layer_id: str | None = None,
        created_by: str = "",
    ) -> CanvasElement | None:
        """
        Add an element to a canvas.

        Args:
            canvas_id: Target canvas
            element_type: Type of element
            x, y: Position
            width, height: Dimensions
            content: Element content
            style: Visual style
            layer_id: Target layer (uses active layer if None)
            created_by: Creator user ID

        Returns:
            Created element
        """
        async with self._lock:
            canvas = self._canvases.get(canvas_id)
            if not canvas:
                return None

            # Check element limit
            total_elements = sum(
                len(self._layers[lid].elements) for lid in canvas.layers if lid in self._layers
            )
            if total_elements >= canvas.config.max_elements:
                raise ValueError(f"Canvas at max elements ({canvas.config.max_elements})")

            # Determine target layer
            target_layer_id = layer_id or canvas.active_layer
            if not target_layer_id or target_layer_id not in self._layers:
                return None

            layer = self._layers[target_layer_id]
            if layer.locked:
                raise ValueError("Cannot add elements to locked layer")

            # Create element
            element_id = str(uuid.uuid4())
            element = CanvasElement(
                id=element_id,
                type=element_type,
                x=x,
                y=y,
                width=width,
                height=height,
                content=content or {},
                style=style or {},
                created_by=created_by,
                z_index=len(layer.elements),
            )

            self._elements[element_id] = element
            layer.elements.append(element_id)
            canvas.updated_at = datetime.now(timezone.utc)
            canvas.version += 1

            # Record for undo
            canvas.undo_stack.append(
                {
                    "action": "add_element",
                    "element_id": element_id,
                    "layer_id": target_layer_id,
                }
            )
            canvas.redo_stack.clear()

            await self._broadcast(canvas_id, "element_added", {"element": element})
            return element

    async def update_element(
        self,
        canvas_id: str,
        element_id: str,
        updates: dict[str, Any],
    ) -> CanvasElement | None:
        """
        Update an element's properties.

        Args:
            canvas_id: Canvas containing the element
            element_id: Element to update
            updates: Property updates

        Returns:
            Updated element
        """
        async with self._lock:
            canvas = self._canvases.get(canvas_id)
            element = self._elements.get(element_id)
            if not canvas or not element:
                return None

            if element.locked:
                raise ValueError("Cannot update locked element")

            # Store previous state for undo
            prev_state = {
                "x": element.x,
                "y": element.y,
                "width": element.width,
                "height": element.height,
                "content": element.content.copy(),
                "style": element.style.copy(),
            }

            # Apply updates
            if "x" in updates:
                element.x = updates["x"]
            if "y" in updates:
                element.y = updates["y"]
            if "width" in updates:
                element.width = updates["width"]
            if "height" in updates:
                element.height = updates["height"]
            if "content" in updates:
                element.content.update(updates["content"])
            if "style" in updates:
                element.style.update(updates["style"])
            if "rotation" in updates:
                element.rotation = updates["rotation"]
            if "opacity" in updates:
                element.opacity = updates["opacity"]
            if "locked" in updates:
                element.locked = updates["locked"]

            element.updated_at = datetime.now(timezone.utc)
            canvas.updated_at = datetime.now(timezone.utc)
            canvas.version += 1

            # Record for undo
            canvas.undo_stack.append(
                {
                    "action": "update_element",
                    "element_id": element_id,
                    "prev_state": prev_state,
                }
            )
            canvas.redo_stack.clear()

            await self._broadcast(
                canvas_id,
                "element_updated",
                {
                    "element_id": element_id,
                    "updates": updates,
                },
            )
            return element

    async def delete_element(
        self,
        canvas_id: str,
        element_id: str,
    ) -> bool:
        """Delete an element from a canvas."""
        async with self._lock:
            canvas = self._canvases.get(canvas_id)
            element = self._elements.get(element_id)
            if not canvas or not element:
                return False

            # Find and remove from layer
            for layer_id in canvas.layers:
                layer = self._layers.get(layer_id)
                if layer and element_id in layer.elements:
                    layer.elements.remove(element_id)
                    break

            # Record for undo before deleting
            canvas.undo_stack.append(
                {
                    "action": "delete_element",
                    "element": element,
                    "layer_id": layer_id,
                }
            )
            canvas.redo_stack.clear()

            del self._elements[element_id]
            canvas.updated_at = datetime.now(timezone.utc)
            canvas.version += 1

            await self._broadcast(canvas_id, "element_deleted", {"element_id": element_id})
            return True

    async def get_element(self, element_id: str) -> CanvasElement | None:
        """Get an element by ID."""
        return self._elements.get(element_id)

    async def list_elements(
        self,
        canvas_id: str,
        layer_id: str | None = None,
    ) -> list[CanvasElement]:
        """List elements in a canvas or layer."""
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return []

        layer_ids = [layer_id] if layer_id else canvas.layers
        elements = []

        for lid in layer_ids:
            layer = self._layers.get(lid)
            if layer:
                for eid in layer.elements:
                    element = self._elements.get(eid)
                    if element:
                        elements.append(element)

        return elements

    # ========== Collaboration ==========

    async def join_canvas(
        self,
        canvas_id: str,
        user_id: str,
        cursor_color: str = "#000000",
    ) -> bool:
        """
        Join a canvas as a collaborator.

        Args:
            canvas_id: Canvas to join
            user_id: User joining
            cursor_color: Color for user's cursor

        Returns:
            True if joined successfully
        """
        async with self._lock:
            canvas = self._canvases.get(canvas_id)
            if not canvas:
                return False

            if len(canvas.active_users) >= canvas.config.max_collaborators:
                raise ValueError("Canvas at max collaborators")

            canvas.active_users[user_id] = {
                "cursor_x": 0,
                "cursor_y": 0,
                "cursor_color": cursor_color,
                "selection": [],
                "joined_at": datetime.now(timezone.utc).isoformat(),
            }

            if user_id not in canvas.collaborators:
                canvas.collaborators.append(user_id)

            await self._broadcast(canvas_id, "user_joined", {"user_id": user_id})
            return True

    async def leave_canvas(self, canvas_id: str, user_id: str) -> bool:
        """Leave a canvas session."""
        async with self._lock:
            canvas = self._canvases.get(canvas_id)
            if not canvas:
                return False

            canvas.active_users.pop(user_id, None)
            await self._broadcast(canvas_id, "user_left", {"user_id": user_id})
            return True

    async def update_cursor(
        self,
        canvas_id: str,
        user_id: str,
        x: float,
        y: float,
        selection: list[str] | None = None,
    ) -> bool:
        """Update a user's cursor position."""
        canvas = self._canvases.get(canvas_id)
        if not canvas or user_id not in canvas.active_users:
            return False

        canvas.active_users[user_id]["cursor_x"] = x
        canvas.active_users[user_id]["cursor_y"] = y
        if selection is not None:
            canvas.active_users[user_id]["selection"] = selection

        await self._broadcast(
            canvas_id,
            "cursor_moved",
            {
                "user_id": user_id,
                "x": x,
                "y": y,
                "selection": selection,
            },
        )
        return True

    # ========== AI Generation ==========

    async def generate_element(
        self,
        canvas_id: str,
        prompt: str,
        x: float,
        y: float,
        element_type: ElementType = ElementType.AI_GENERATED,
        created_by: str = "",
    ) -> CanvasElement | None:
        """
        Generate an AI element based on a prompt.

        Args:
            canvas_id: Target canvas
            prompt: Generation prompt
            x, y: Position for generated element
            element_type: Type of element to generate
            created_by: Creator user ID

        Returns:
            Generated element
        """
        # In a real implementation, this would call an AI service
        # For now, create a placeholder element

        return await self.add_element(
            canvas_id=canvas_id,
            element_type=element_type,
            x=x,
            y=y,
            width=200,
            height=200,
            content={
                "prompt": prompt,
                "generated": True,
                "status": "pending",
            },
            style={
                "border": "2px dashed #9333ea",
                "background": "#f3e8ff",
            },
            created_by=created_by,
        )

    # ========== Event Broadcasting ==========

    def subscribe(self, canvas_id: str, callback: Callable) -> None:
        """Subscribe to canvas events."""
        if canvas_id not in self._subscribers:
            self._subscribers[canvas_id] = []
        self._subscribers[canvas_id].append(callback)

    def unsubscribe(self, canvas_id: str, callback: Callable) -> None:
        """Unsubscribe from canvas events."""
        if canvas_id in self._subscribers:
            try:
                self._subscribers[canvas_id].remove(callback)
            except ValueError as e:
                logger.debug("unsubscribe encountered an error: %s", e)

    async def _broadcast(
        self,
        canvas_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast an event to all subscribers."""
        callbacks = self._subscribers.get(canvas_id, [])
        event = {
            "type": event_type,
            "canvas_id": canvas_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except (RuntimeError, ValueError, AttributeError) as e:  # user-supplied callback
                logger.error(f"Canvas event broadcast error: {e}")

    # ========== Statistics ==========

    async def get_stats(self) -> dict[str, Any]:
        """Get canvas manager statistics."""
        async with self._lock:
            total_elements = len(self._elements)
            total_layers = len(self._layers)
            active_users = sum(len(c.active_users) for c in self._canvases.values())

            by_type: dict[str, int] = {}
            for element in self._elements.values():
                t = element.type.value
                by_type[t] = by_type.get(t, 0) + 1

            return {
                "canvases_total": len(self._canvases),
                "canvases_active": sum(1 for c in self._canvases.values() if c.status == "active"),
                "layers_total": total_layers,
                "elements_total": total_elements,
                "elements_by_type": by_type,
                "active_users": active_users,
            }
