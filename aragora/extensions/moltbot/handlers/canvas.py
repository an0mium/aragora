"""
Moltbot Canvas Handler - Canvas Collaboration REST API.

Endpoints:
- GET  /api/v1/moltbot/canvas                     - List canvases
- POST /api/v1/moltbot/canvas                     - Create canvas
- GET  /api/v1/moltbot/canvas/{id}                - Get canvas
- DELETE /api/v1/moltbot/canvas/{id}              - Delete canvas
- GET  /api/v1/moltbot/canvas/{id}/elements       - List elements
- POST /api/v1/moltbot/canvas/{id}/elements       - Add element
- PUT  /api/v1/moltbot/canvas/{id}/elements/{eid} - Update element
- DELETE /api/v1/moltbot/canvas/{id}/elements/{eid} - Remove element
- POST /api/v1/moltbot/canvas/{id}/collaborators  - Add collaborator
- DELETE /api/v1/moltbot/canvas/{id}/collaborators/{uid} - Remove collaborator
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)

from .types import serialize_datetime, serialize_enum

if TYPE_CHECKING:
    from aragora.extensions.moltbot.canvas import Canvas, CanvasElement, CanvasManager

logger = logging.getLogger(__name__)

# Global manager instance
_manager: Optional["CanvasManager"] = None


def get_canvas_manager() -> "CanvasManager":
    """Get or create the canvas manager instance."""
    global _manager
    if _manager is None:
        from aragora.extensions.moltbot.canvas import CanvasManager

        _manager = CanvasManager()
    return _manager


class MoltbotCanvasHandler(BaseHandler):
    """HTTP handler for Moltbot canvas collaboration."""

    routes = [
        ("GET", "/api/v1/moltbot/canvas"),
        ("POST", "/api/v1/moltbot/canvas"),
        ("GET", "/api/v1/moltbot/canvas/"),
        ("DELETE", "/api/v1/moltbot/canvas/"),
        ("PUT", "/api/v1/moltbot/canvas/"),
        ("GET", "/api/v1/moltbot/canvas/*/elements"),
        ("POST", "/api/v1/moltbot/canvas/*/elements"),
        ("PUT", "/api/v1/moltbot/canvas/*/elements/*"),
        ("DELETE", "/api/v1/moltbot/canvas/*/elements/*"),
        ("POST", "/api/v1/moltbot/canvas/*/collaborators"),
        ("DELETE", "/api/v1/moltbot/canvas/*/collaborators/*"),
        ("GET", "/api/v1/moltbot/canvas/*/export"),
    ]

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests."""
        if path == "/api/v1/moltbot/canvas":
            return await self._handle_list_canvases(query_params, handler)
        elif path.startswith("/api/v1/moltbot/canvas/"):
            parts = path.split("/")
            if len(parts) >= 5:
                canvas_id = parts[4]

                # Export endpoint
                if len(parts) > 5 and parts[5] == "export":
                    return await self._handle_export_canvas(canvas_id, query_params, handler)

                # Elements list
                if len(parts) > 5 and parts[5] == "elements":
                    return await self._handle_list_elements(canvas_id, query_params, handler)

                # Get single canvas
                return await self._handle_get_canvas(canvas_id, handler)
        return None

    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if path == "/api/v1/moltbot/canvas":
            return await self._handle_create_canvas(handler)
        elif path.startswith("/api/v1/moltbot/canvas/"):
            parts = path.split("/")
            if len(parts) >= 5:
                canvas_id = parts[4]

                # Add element
                if len(parts) > 5 and parts[5] == "elements":
                    return await self._handle_add_element(canvas_id, handler)

                # Add collaborator
                if len(parts) > 5 and parts[5] == "collaborators":
                    return await self._handle_add_collaborator(canvas_id, handler)
        return None

    async def handle_put(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle PUT requests."""
        if path.startswith("/api/v1/moltbot/canvas/"):
            parts = path.split("/")
            if len(parts) >= 7 and parts[5] == "elements":
                canvas_id = parts[4]
                element_id = parts[6]
                return await self._handle_update_element(canvas_id, element_id, handler)
        return None

    async def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle DELETE requests."""
        if path.startswith("/api/v1/moltbot/canvas/"):
            parts = path.split("/")
            if len(parts) >= 5:
                canvas_id = parts[4]

                # Remove element
                if len(parts) >= 7 and parts[5] == "elements":
                    element_id = parts[6]
                    return await self._handle_remove_element(
                        canvas_id, element_id, query_params, handler
                    )

                # Remove collaborator
                if len(parts) >= 7 and parts[5] == "collaborators":
                    user_id = parts[6]
                    return await self._handle_remove_collaborator(
                        canvas_id, user_id, query_params, handler
                    )

                # Delete canvas
                return await self._handle_delete_canvas(canvas_id, query_params, handler)
        return None

    # ========== Handler Methods ==========

    def _serialize_canvas(self, canvas: "Canvas") -> dict[str, Any]:
        """Serialize canvas to JSON-safe dict."""
        return {
            "id": canvas.id,
            "name": canvas.config.name,
            "owner_id": canvas.owner_id,
            "width": canvas.config.width,
            "height": canvas.config.height,
            "background": canvas.config.background,
            "created_at": serialize_datetime(canvas.created_at),
            "updated_at": serialize_datetime(canvas.updated_at),
            "element_count": len(canvas.elements),
            "layer_count": len(canvas.layers),
        }

    def _serialize_element(self, element: "CanvasElement") -> dict[str, Any]:
        """Serialize element to JSON-safe dict."""
        return {
            "id": element.id,
            "type": serialize_enum(element.element_type),
            "x": element.x,
            "y": element.y,
            "width": element.width,
            "height": element.height,
            "rotation": element.rotation,
            "z_index": element.z_index,
            "properties": element.properties,
            "layer_id": element.layer_id,
        }

    async def _handle_list_canvases(
        self, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """List canvases for a user."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        owner_id = query_params.get("owner_id")
        tenant_id = query_params.get("tenant_id")

        manager = get_canvas_manager()
        canvases = await manager.list_canvases(
            owner_id=owner_id,
            tenant_id=tenant_id,
        )

        return json_response(
            {
                "canvases": [self._serialize_canvas(c) for c in canvases],
                "total": len(canvases),
            }
        )

    async def _handle_create_canvas(self, handler: Any) -> HandlerResult:
        """Create a new canvas."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        name = body.get("name")
        if not name:
            return error_response("name is required", 400)

        from aragora.extensions.moltbot.canvas import CanvasConfig

        config = CanvasConfig(
            name=name,
            width=body.get("width", 1920),
            height=body.get("height", 1080),
            background=body.get("background", "#ffffff"),
        )

        manager = get_canvas_manager()
        canvas = await manager.create_canvas(
            config=config,
            owner_id=body.get("owner_id", user.user_id),
            tenant_id=body.get("tenant_id"),
        )

        return json_response(
            {"success": True, "canvas": self._serialize_canvas(canvas)},
            status=201,
        )

    async def _handle_get_canvas(self, canvas_id: str, handler: Any) -> HandlerResult:
        """Get canvas details."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        manager = get_canvas_manager()
        canvas = await manager.get_canvas(canvas_id)

        if not canvas:
            return error_response("Canvas not found", 404)

        result = self._serialize_canvas(canvas)
        result["elements"] = [self._serialize_element(e) for e in canvas.elements]
        result["layers"] = [
            {"id": layer.id, "name": layer.name, "visible": layer.visible, "locked": layer.locked}
            for layer in canvas.layers
        ]

        return json_response({"canvas": result})

    async def _handle_delete_canvas(
        self, canvas_id: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Delete a canvas."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        manager = get_canvas_manager()
        canvas = await manager.get_canvas(canvas_id)

        if not canvas:
            return error_response("Canvas not found", 404)

        # Check ownership
        if canvas.owner_id != user.user_id:
            return error_response("Only owner can delete canvas", 403)

        success = await manager.delete_canvas(canvas_id)
        if not success:
            return error_response("Failed to delete canvas", 500)

        return json_response({"success": True, "deleted": canvas_id})

    async def _handle_list_elements(
        self, canvas_id: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """List elements in a canvas."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        manager = get_canvas_manager()
        canvas = await manager.get_canvas(canvas_id)

        if not canvas:
            return error_response("Canvas not found", 404)

        element_type = query_params.get("type")
        layer_id = query_params.get("layer_id")

        elements = canvas.elements
        if element_type:
            elements = [
                e
                for e in elements
                if str(e.element_type) == element_type
                or (hasattr(e.element_type, "value") and e.element_type.value == element_type)
            ]
        if layer_id:
            elements = [e for e in elements if e.layer_id == layer_id]

        return json_response(
            {
                "elements": [self._serialize_element(e) for e in elements],
                "total": len(elements),
            }
        )

    async def _handle_add_element(self, canvas_id: str, handler: Any) -> HandlerResult:
        """Add element to canvas."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        element_type = body.get("type")
        if not element_type:
            return error_response("type is required", 400)

        manager = get_canvas_manager()
        canvas = await manager.get_canvas(canvas_id)

        if not canvas:
            return error_response("Canvas not found", 404)

        from aragora.extensions.moltbot.canvas import ElementType

        try:
            elem_type = ElementType(element_type)
        except ValueError:
            return error_response(f"Invalid element type: {element_type}", 400)

        element = await manager.add_element(
            canvas_id=canvas_id,
            element_type=elem_type,
            x=body.get("x", 0),
            y=body.get("y", 0),
            width=body.get("width", 100),
            height=body.get("height", 100),
            properties=body.get("properties", {}),
            layer_id=body.get("layer_id"),
        )

        return json_response(
            {"success": True, "element": self._serialize_element(element)},
            status=201,
        )

    async def _handle_update_element(
        self, canvas_id: str, element_id: str, handler: Any
    ) -> HandlerResult:
        """Update element in canvas."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        manager = get_canvas_manager()
        element = await manager.update_element(
            canvas_id=canvas_id,
            element_id=element_id,
            x=body.get("x"),
            y=body.get("y"),
            width=body.get("width"),
            height=body.get("height"),
            rotation=body.get("rotation"),
            z_index=body.get("z_index"),
            properties=body.get("properties"),
        )

        if not element:
            return error_response("Element not found", 404)

        return json_response({"success": True, "element": self._serialize_element(element)})

    async def _handle_remove_element(
        self, canvas_id: str, element_id: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Remove element from canvas."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        manager = get_canvas_manager()
        success = await manager.remove_element(canvas_id, element_id)

        if not success:
            return error_response("Element not found", 404)

        return json_response({"success": True, "deleted": element_id})

    async def _handle_add_collaborator(self, canvas_id: str, handler: Any) -> HandlerResult:
        """Add collaborator to canvas."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        user_id = body.get("user_id")
        permission = body.get("permission", "view")

        if not user_id:
            return error_response("user_id is required", 400)

        manager = get_canvas_manager()
        canvas = await manager.get_canvas(canvas_id)

        if not canvas:
            return error_response("Canvas not found", 404)

        # Only owner can add collaborators
        if canvas.owner_id != user.user_id:
            return error_response("Only owner can add collaborators", 403)

        success = await manager.add_collaborator(
            canvas_id=canvas_id,
            user_id=user_id,
            permission=permission,
        )

        if not success:
            return error_response("Failed to add collaborator", 500)

        return json_response(
            {"success": True, "canvas_id": canvas_id, "user_id": user_id, "permission": permission},
            status=201,
        )

    async def _handle_remove_collaborator(
        self, canvas_id: str, user_id: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Remove collaborator from canvas."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        manager = get_canvas_manager()
        canvas = await manager.get_canvas(canvas_id)

        if not canvas:
            return error_response("Canvas not found", 404)

        # Only owner can remove collaborators
        if canvas.owner_id != user.user_id:
            return error_response("Only owner can remove collaborators", 403)

        success = await manager.remove_collaborator(canvas_id, user_id)

        if not success:
            return error_response("Collaborator not found", 404)

        return json_response({"success": True, "deleted": user_id})

    async def _handle_export_canvas(
        self, canvas_id: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Export canvas data."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        format_type = query_params.get("format", "json")

        manager = get_canvas_manager()
        canvas = await manager.get_canvas(canvas_id)

        if not canvas:
            return error_response("Canvas not found", 404)

        if format_type == "json":
            export_data = await manager.export_canvas(canvas_id, format="json")
            return json_response({"export": export_data})
        elif format_type == "svg":
            export_data = await manager.export_canvas(canvas_id, format="svg")
            return json_response({"svg": export_data, "format": "svg"})
        else:
            return error_response(f"Unsupported format: {format_type}", 400)
