"""
Orchestration Canvas Namespace API

Provides methods for the Orchestration Canvas (Stage 4 of the Idea-to-Execution Pipeline):
- Canvas CRUD (list, create, get, update, delete)
- Node CRUD (add, update, delete)
- Edge CRUD (add, delete)
- Export to React Flow JSON
- Execute pipeline
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class OrchestrationCanvasAPI:
    """Synchronous Orchestration Canvas API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # -------------------------------------------------------------------
    # Canvas CRUD
    # -------------------------------------------------------------------

    def list_canvases(
        self,
        workspace_id: str | None = None,
        owner_id: str | None = None,
        source_canvas_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List orchestration canvases with optional filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id is not None:
            params["workspace_id"] = workspace_id
        if owner_id is not None:
            params["owner_id"] = owner_id
        if source_canvas_id is not None:
            params["source_canvas_id"] = source_canvas_id
        return self._client.request("GET", "/api/v1/orchestration/canvas", params=params)

    def create_canvas(
        self,
        name: str,
        description: str = "",
        source_canvas_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new orchestration canvas."""
        body: dict[str, Any] = {"name": name, "description": description}
        if source_canvas_id is not None:
            body["source_canvas_id"] = source_canvas_id
        if metadata is not None:
            body["metadata"] = metadata
        return self._client.request("POST", "/api/v1/orchestration/canvas", json=body)

    def get_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Get a canvas by ID, including its nodes and edges."""
        return self._client.request("GET", f"/api/v1/orchestration/canvas/{canvas_id}")

    def update_canvas(
        self,
        canvas_id: str,
        name: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update canvas metadata."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = metadata
        return self._client.request("PUT", f"/api/v1/orchestration/canvas/{canvas_id}", json=body)

    def delete_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Delete a canvas and all its nodes and edges."""
        return self._client.request("DELETE", f"/api/v1/orchestration/canvas/{canvas_id}")

    # -------------------------------------------------------------------
    # Node CRUD
    # -------------------------------------------------------------------

    def add_node(
        self,
        canvas_id: str,
        label: str,
        orchestration_type: str = "agent_task",
        position: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a node to a canvas."""
        body: dict[str, Any] = {"label": label, "orchestration_type": orchestration_type}
        if position is not None:
            body["position"] = position
        if data is not None:
            body["data"] = data
        return self._client.request(
            "POST", f"/api/v1/orchestration/canvas/{canvas_id}/nodes", json=body
        )

    def update_node(
        self,
        canvas_id: str,
        node_id: str,
        label: str | None = None,
        position: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a node on a canvas."""
        body: dict[str, Any] = {}
        if label is not None:
            body["label"] = label
        if position is not None:
            body["position"] = position
        if data is not None:
            body["data"] = data
        return self._client.request(
            "PUT", f"/api/v1/orchestration/canvas/{canvas_id}/nodes/{node_id}", json=body
        )

    def delete_node(self, canvas_id: str, node_id: str) -> dict[str, Any]:
        """Delete a node from a canvas."""
        return self._client.request(
            "DELETE", f"/api/v1/orchestration/canvas/{canvas_id}/nodes/{node_id}"
        )

    # -------------------------------------------------------------------
    # Edge CRUD
    # -------------------------------------------------------------------

    def add_edge(
        self,
        canvas_id: str,
        source_id: str,
        target_id: str,
        edge_type: str = "default",
        label: str = "",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add an edge connecting two nodes on a canvas."""
        body: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "label": label,
        }
        if data is not None:
            body["data"] = data
        return self._client.request(
            "POST", f"/api/v1/orchestration/canvas/{canvas_id}/edges", json=body
        )

    def delete_edge(self, canvas_id: str, edge_id: str) -> dict[str, Any]:
        """Delete an edge from a canvas."""
        return self._client.request(
            "DELETE", f"/api/v1/orchestration/canvas/{canvas_id}/edges/{edge_id}"
        )

    # -------------------------------------------------------------------
    # Export & Execute
    # -------------------------------------------------------------------

    def export_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Export a canvas as React Flow JSON."""
        return self._client.request("GET", f"/api/v1/orchestration/canvas/{canvas_id}/export")

    def execute_pipeline(self, canvas_id: str) -> dict[str, Any]:
        """Execute the orchestration pipeline defined by this canvas."""
        return self._client.request(
            "POST",
            f"/api/v1/orchestration/canvas/{canvas_id}/execute",
            json={},
        )


class AsyncOrchestrationCanvasAPI:
    """Asynchronous Orchestration Canvas API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # -------------------------------------------------------------------
    # Canvas CRUD
    # -------------------------------------------------------------------

    async def list_canvases(
        self,
        workspace_id: str | None = None,
        owner_id: str | None = None,
        source_canvas_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List orchestration canvases with optional filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id is not None:
            params["workspace_id"] = workspace_id
        if owner_id is not None:
            params["owner_id"] = owner_id
        if source_canvas_id is not None:
            params["source_canvas_id"] = source_canvas_id
        return await self._client.request("GET", "/api/v1/orchestration/canvas", params=params)

    async def create_canvas(
        self,
        name: str,
        description: str = "",
        source_canvas_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new orchestration canvas."""
        body: dict[str, Any] = {"name": name, "description": description}
        if source_canvas_id is not None:
            body["source_canvas_id"] = source_canvas_id
        if metadata is not None:
            body["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/orchestration/canvas", json=body)

    async def get_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Get a canvas by ID, including its nodes and edges."""
        return await self._client.request("GET", f"/api/v1/orchestration/canvas/{canvas_id}")

    async def update_canvas(
        self,
        canvas_id: str,
        name: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update canvas metadata."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = metadata
        return await self._client.request(
            "PUT", f"/api/v1/orchestration/canvas/{canvas_id}", json=body
        )

    async def delete_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Delete a canvas and all its nodes and edges."""
        return await self._client.request("DELETE", f"/api/v1/orchestration/canvas/{canvas_id}")

    # -------------------------------------------------------------------
    # Node CRUD
    # -------------------------------------------------------------------

    async def add_node(
        self,
        canvas_id: str,
        label: str,
        orchestration_type: str = "agent_task",
        position: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a node to a canvas."""
        body: dict[str, Any] = {"label": label, "orchestration_type": orchestration_type}
        if position is not None:
            body["position"] = position
        if data is not None:
            body["data"] = data
        return await self._client.request(
            "POST", f"/api/v1/orchestration/canvas/{canvas_id}/nodes", json=body
        )

    async def update_node(
        self,
        canvas_id: str,
        node_id: str,
        label: str | None = None,
        position: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a node on a canvas."""
        body: dict[str, Any] = {}
        if label is not None:
            body["label"] = label
        if position is not None:
            body["position"] = position
        if data is not None:
            body["data"] = data
        return await self._client.request(
            "PUT", f"/api/v1/orchestration/canvas/{canvas_id}/nodes/{node_id}", json=body
        )

    async def delete_node(self, canvas_id: str, node_id: str) -> dict[str, Any]:
        """Delete a node from a canvas."""
        return await self._client.request(
            "DELETE", f"/api/v1/orchestration/canvas/{canvas_id}/nodes/{node_id}"
        )

    # -------------------------------------------------------------------
    # Edge CRUD
    # -------------------------------------------------------------------

    async def add_edge(
        self,
        canvas_id: str,
        source_id: str,
        target_id: str,
        edge_type: str = "default",
        label: str = "",
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add an edge connecting two nodes on a canvas."""
        body: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "label": label,
        }
        if data is not None:
            body["data"] = data
        return await self._client.request(
            "POST", f"/api/v1/orchestration/canvas/{canvas_id}/edges", json=body
        )

    async def delete_edge(self, canvas_id: str, edge_id: str) -> dict[str, Any]:
        """Delete an edge from a canvas."""
        return await self._client.request(
            "DELETE", f"/api/v1/orchestration/canvas/{canvas_id}/edges/{edge_id}"
        )

    # -------------------------------------------------------------------
    # Export & Execute
    # -------------------------------------------------------------------

    async def export_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Export a canvas as React Flow JSON."""
        return await self._client.request("GET", f"/api/v1/orchestration/canvas/{canvas_id}/export")

    async def execute_pipeline(self, canvas_id: str) -> dict[str, Any]:
        """Execute the orchestration pipeline defined by this canvas."""
        return await self._client.request(
            "POST",
            f"/api/v1/orchestration/canvas/{canvas_id}/execute",
            json={},
        )
