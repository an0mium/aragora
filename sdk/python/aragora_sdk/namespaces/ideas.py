"""
Ideas Namespace API

Provides methods for the Idea Canvas (Stage 1 of the Idea-to-Execution Pipeline):
- Canvas CRUD (list, create, get, update, delete)
- Node CRUD (add, update, delete)
- Edge CRUD (add, delete)
- Export to React Flow JSON
- Promote nodes to goals
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class IdeasAPI:
    """Synchronous Ideas Canvas API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # -------------------------------------------------------------------
    # Canvas CRUD
    # -------------------------------------------------------------------

    def list_canvases(
        self,
        workspace_id: str | None = None,
        owner_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List idea canvases with optional filtering.

        Args:
            workspace_id: Filter by workspace.
            owner_id: Filter by owner.
            limit: Maximum number of canvases to return.
            offset: Number of canvases to skip for pagination.

        Returns:
            Dictionary with ``canvases`` list and pagination metadata.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id is not None:
            params["workspace_id"] = workspace_id
        if owner_id is not None:
            params["owner_id"] = owner_id
        return self._client._request("GET", "/api/v1/ideas", params=params)

    def create_canvas(
        self,
        name: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new idea canvas.

        Args:
            name: Canvas display name.
            description: Optional description of the canvas purpose.
            metadata: Arbitrary key-value metadata.

        Returns:
            The newly created canvas object.
        """
        body: dict[str, Any] = {"name": name, "description": description}
        if metadata is not None:
            body["metadata"] = metadata
        return self._client._request("POST", "/api/v1/ideas", json=body)

    def get_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Get a canvas by ID, including its nodes and edges.

        Args:
            canvas_id: Unique canvas identifier.

        Returns:
            Full canvas object with nodes, edges, and metadata.
        """
        return self._client._request("GET", f"/api/v1/ideas/{canvas_id}")

    def update_canvas(
        self,
        canvas_id: str,
        name: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update canvas metadata.

        Args:
            canvas_id: Canvas to update.
            name: New display name (if provided).
            description: New description (if provided).
            metadata: New metadata (if provided).

        Returns:
            Updated canvas object.
        """
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = metadata
        return self._client._request("PUT", f"/api/v1/ideas/{canvas_id}", json=body)

    def delete_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Delete a canvas and all its nodes and edges.

        Args:
            canvas_id: Canvas to delete.

        Returns:
            Deletion confirmation with ``success`` flag.
        """
        return self._client._request("DELETE", f"/api/v1/ideas/{canvas_id}")

    # -------------------------------------------------------------------
    # Node CRUD
    # -------------------------------------------------------------------

    def add_node(
        self,
        canvas_id: str,
        label: str,
        idea_type: str = "concept",
        position: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a node to a canvas.

        Args:
            canvas_id: Target canvas.
            label: Node display label.
            idea_type: Node type (e.g. ``concept``, ``question``, ``evidence``).
            position: Optional ``{x, y}`` position on the canvas.
            data: Arbitrary node data payload.

        Returns:
            The newly created node object.
        """
        body: dict[str, Any] = {"label": label, "idea_type": idea_type}
        if position is not None:
            body["position"] = position
        if data is not None:
            body["data"] = data
        return self._client._request("POST", f"/api/v1/ideas/{canvas_id}/nodes", json=body)

    def update_node(
        self,
        canvas_id: str,
        node_id: str,
        label: str | None = None,
        position: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a node on a canvas.

        Args:
            canvas_id: Canvas containing the node.
            node_id: Node to update.
            label: New label (if provided).
            position: New position (if provided).
            data: New data payload (if provided).

        Returns:
            Updated node object.
        """
        body: dict[str, Any] = {}
        if label is not None:
            body["label"] = label
        if position is not None:
            body["position"] = position
        if data is not None:
            body["data"] = data
        return self._client._request("PUT", f"/api/v1/ideas/{canvas_id}/nodes/{node_id}", json=body)

    def delete_node(self, canvas_id: str, node_id: str) -> dict[str, Any]:
        """Delete a node from a canvas.

        Args:
            canvas_id: Canvas containing the node.
            node_id: Node to delete.

        Returns:
            Deletion confirmation.
        """
        return self._client._request("DELETE", f"/api/v1/ideas/{canvas_id}/nodes/{node_id}")

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
        """Add an edge connecting two nodes on a canvas.

        Args:
            canvas_id: Target canvas.
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Edge classification (default ``"default"``).
            label: Optional edge label.
            data: Arbitrary edge data payload.

        Returns:
            The newly created edge object.
        """
        body: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "label": label,
        }
        if data is not None:
            body["data"] = data
        return self._client._request("POST", f"/api/v1/ideas/{canvas_id}/edges", json=body)

    def delete_edge(self, canvas_id: str, edge_id: str) -> dict[str, Any]:
        """Delete an edge from a canvas.

        Args:
            canvas_id: Canvas containing the edge.
            edge_id: Edge to delete.

        Returns:
            Deletion confirmation.
        """
        return self._client._request("DELETE", f"/api/v1/ideas/{canvas_id}/edges/{edge_id}")

    # -------------------------------------------------------------------
    # Export & Promote
    # -------------------------------------------------------------------

    def export_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Export a canvas as React Flow JSON.

        Args:
            canvas_id: Canvas to export.

        Returns:
            React Flow format with ``nodes`` and ``edges`` arrays.
        """
        return self._client._request("GET", f"/api/v1/ideas/{canvas_id}/export")

    def promote_nodes(self, canvas_id: str, node_ids: list[str]) -> dict[str, Any]:
        """Promote selected nodes from the idea canvas to the goals stage.

        Args:
            canvas_id: Canvas containing the nodes.
            node_ids: List of node IDs to promote.

        Returns:
            Promotion result with ``goals_canvas``, ``provenance``,
            and ``promoted_count``.
        """
        return self._client._request(
            "POST",
            f"/api/v1/ideas/{canvas_id}/promote",
            json={"node_ids": node_ids},
        )


class AsyncIdeasAPI:
    """Asynchronous Ideas Canvas API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # -------------------------------------------------------------------
    # Canvas CRUD
    # -------------------------------------------------------------------

    async def list_canvases(
        self,
        workspace_id: str | None = None,
        owner_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List idea canvases with optional filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id is not None:
            params["workspace_id"] = workspace_id
        if owner_id is not None:
            params["owner_id"] = owner_id
        return await self._client._request("GET", "/api/v1/ideas", params=params)

    async def create_canvas(
        self,
        name: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new idea canvas."""
        body: dict[str, Any] = {"name": name, "description": description}
        if metadata is not None:
            body["metadata"] = metadata
        return await self._client._request("POST", "/api/v1/ideas", json=body)

    async def get_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Get a canvas by ID, including its nodes and edges."""
        return await self._client._request("GET", f"/api/v1/ideas/{canvas_id}")

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
        return await self._client._request("PUT", f"/api/v1/ideas/{canvas_id}", json=body)

    async def delete_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Delete a canvas and all its nodes and edges."""
        return await self._client._request("DELETE", f"/api/v1/ideas/{canvas_id}")

    # -------------------------------------------------------------------
    # Node CRUD
    # -------------------------------------------------------------------

    async def add_node(
        self,
        canvas_id: str,
        label: str,
        idea_type: str = "concept",
        position: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a node to a canvas."""
        body: dict[str, Any] = {"label": label, "idea_type": idea_type}
        if position is not None:
            body["position"] = position
        if data is not None:
            body["data"] = data
        return await self._client._request("POST", f"/api/v1/ideas/{canvas_id}/nodes", json=body)

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
        return await self._client._request(
            "PUT", f"/api/v1/ideas/{canvas_id}/nodes/{node_id}", json=body
        )

    async def delete_node(self, canvas_id: str, node_id: str) -> dict[str, Any]:
        """Delete a node from a canvas."""
        return await self._client._request("DELETE", f"/api/v1/ideas/{canvas_id}/nodes/{node_id}")

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
        return await self._client._request("POST", f"/api/v1/ideas/{canvas_id}/edges", json=body)

    async def delete_edge(self, canvas_id: str, edge_id: str) -> dict[str, Any]:
        """Delete an edge from a canvas."""
        return await self._client._request("DELETE", f"/api/v1/ideas/{canvas_id}/edges/{edge_id}")

    # -------------------------------------------------------------------
    # Export & Promote
    # -------------------------------------------------------------------

    async def export_canvas(self, canvas_id: str) -> dict[str, Any]:
        """Export a canvas as React Flow JSON."""
        return await self._client._request("GET", f"/api/v1/ideas/{canvas_id}/export")

    async def promote_nodes(self, canvas_id: str, node_ids: list[str]) -> dict[str, Any]:
        """Promote selected nodes from the idea canvas to the goals stage."""
        return await self._client._request(
            "POST",
            f"/api/v1/ideas/{canvas_id}/promote",
            json={"node_ids": node_ids},
        )
