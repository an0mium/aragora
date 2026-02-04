"""
Canvas Namespace API

Provides methods for visual collaboration canvas:
- Create and manage canvases
- Real-time collaboration
- Export and sharing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CanvasAPI:
    """Synchronous Canvas API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, workspace_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List canvases."""
        params: dict[str, Any] = {"limit": limit}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/canvas", params=params)

    def create(
        self, name: str, workspace_id: str | None = None, template: str | None = None
    ) -> dict[str, Any]:
        """Create a canvas."""
        data: dict[str, Any] = {"name": name}
        if workspace_id:
            data["workspace_id"] = workspace_id
        if template:
            data["template"] = template
        return self._client.request("POST", "/api/v1/canvas", json=data)

    def get(self, canvas_id: str) -> dict[str, Any]:
        """Get canvas by ID."""
        return self._client.request("GET", f"/api/v1/canvas/{canvas_id}")

    def update(self, canvas_id: str, **updates: Any) -> dict[str, Any]:
        """Update canvas."""
        return self._client.request("PATCH", f"/api/v1/canvas/{canvas_id}", json=updates)

    def delete(self, canvas_id: str) -> dict[str, Any]:
        """Delete canvas."""
        return self._client.request("DELETE", f"/api/v1/canvas/{canvas_id}")

    def add_element(
        self, canvas_id: str, element_type: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Add element to canvas."""
        return self._client.request(
            "POST",
            f"/api/v1/canvas/{canvas_id}/elements",
            json={
                "type": element_type,
                "data": data,
            },
        )

    def update_element(self, canvas_id: str, element_id: str, **updates: Any) -> dict[str, Any]:
        """Update canvas element."""
        return self._client.request(
            "PATCH", f"/api/v1/canvas/{canvas_id}/elements/{element_id}", json=updates
        )

    def delete_element(self, canvas_id: str, element_id: str) -> dict[str, Any]:
        """Delete canvas element."""
        return self._client.request("DELETE", f"/api/v1/canvas/{canvas_id}/elements/{element_id}")

    def export(self, canvas_id: str, format: str = "png") -> dict[str, Any]:
        """Export canvas."""
        return self._client.request(
            "GET", f"/api/v1/canvas/{canvas_id}/export", params={"format": format}
        )

    def share(self, canvas_id: str, access: str = "view") -> dict[str, Any]:
        """Get share link for canvas."""
        return self._client.request(
            "POST", f"/api/v1/canvas/{canvas_id}/share", json={"access": access}
        )


class AsyncCanvasAPI:
    """Asynchronous Canvas API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self, workspace_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List canvases."""
        params: dict[str, Any] = {"limit": limit}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/canvas", params=params)

    async def create(
        self, name: str, workspace_id: str | None = None, template: str | None = None
    ) -> dict[str, Any]:
        """Create a canvas."""
        data: dict[str, Any] = {"name": name}
        if workspace_id:
            data["workspace_id"] = workspace_id
        if template:
            data["template"] = template
        return await self._client.request("POST", "/api/v1/canvas", json=data)

    async def get(self, canvas_id: str) -> dict[str, Any]:
        """Get canvas by ID."""
        return await self._client.request("GET", f"/api/v1/canvas/{canvas_id}")

    async def update(self, canvas_id: str, **updates: Any) -> dict[str, Any]:
        """Update canvas."""
        return await self._client.request("PATCH", f"/api/v1/canvas/{canvas_id}", json=updates)

    async def delete(self, canvas_id: str) -> dict[str, Any]:
        """Delete canvas."""
        return await self._client.request("DELETE", f"/api/v1/canvas/{canvas_id}")

    async def add_element(
        self, canvas_id: str, element_type: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Add element to canvas."""
        return await self._client.request(
            "POST",
            f"/api/v1/canvas/{canvas_id}/elements",
            json={
                "type": element_type,
                "data": data,
            },
        )

    async def update_element(
        self, canvas_id: str, element_id: str, **updates: Any
    ) -> dict[str, Any]:
        """Update canvas element."""
        return await self._client.request(
            "PATCH", f"/api/v1/canvas/{canvas_id}/elements/{element_id}", json=updates
        )

    async def delete_element(self, canvas_id: str, element_id: str) -> dict[str, Any]:
        """Delete canvas element."""
        return await self._client.request(
            "DELETE", f"/api/v1/canvas/{canvas_id}/elements/{element_id}"
        )

    async def export(self, canvas_id: str, format: str = "png") -> dict[str, Any]:
        """Export canvas."""
        return await self._client.request(
            "GET", f"/api/v1/canvas/{canvas_id}/export", params={"format": format}
        )

    async def share(self, canvas_id: str, access: str = "view") -> dict[str, Any]:
        """Get share link for canvas."""
        return await self._client.request(
            "POST", f"/api/v1/canvas/{canvas_id}/share", json={"access": access}
        )
