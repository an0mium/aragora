"""Connectors API for external system integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class ConnectorsAPI:
    """API for connector operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def list(
        self,
        *,
        status: str | None = None,
        connector_type: str | None = None,
        category: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List all connectors.

        Args:
            status: Filter by status (active, paused, error)
            connector_type: Filter by type (slack, github, jira, etc.)
            category: Filter by category (communication, code, ticketing)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of connectors with metadata
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if connector_type:
            params["type"] = connector_type
        if category:
            params["category"] = category
        return await self._client._get("/api/v1/connectors", params=params)

    async def create(
        self,
        connector_type: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new connector.

        Args:
            connector_type: Connector type (slack, github, jira, etc.)
            name: Optional display name
            config: Connector-specific configuration

        Returns:
            Created connector details
        """
        body: dict[str, Any] = {"type": connector_type}
        if name:
            body["name"] = name
        if config:
            body["config"] = config
        return await self._client._post("/api/v1/connectors", body)

    async def get(self, connector_id: str) -> dict[str, Any]:
        """Get a connector by ID.

        Args:
            connector_id: Connector ID

        Returns:
            Connector details
        """
        return await self._client._get(f"/api/v1/connectors/{connector_id}")

    async def update(
        self,
        connector_id: str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a connector.

        Args:
            connector_id: Connector ID
            name: New display name
            config: Updated configuration

        Returns:
            Updated connector details
        """
        body: dict[str, Any] = {}
        if name:
            body["name"] = name
        if config:
            body["config"] = config
        return await self._client._patch(f"/api/v1/connectors/{connector_id}", body)

    async def delete(self, connector_id: str) -> dict[str, Any]:
        """Delete a connector.

        Args:
            connector_id: Connector ID

        Returns:
            Deletion confirmation
        """
        response = await self._client._request(
            "DELETE", f"/api/v1/connectors/{connector_id}"
        )
        return response.json()

    # =========================================================================
    # Actions
    # =========================================================================

    async def sync(self, connector_id: str) -> dict[str, Any]:
        """Trigger a sync for a connector.

        Args:
            connector_id: Connector ID

        Returns:
            Sync job details
        """
        return await self._client._post(f"/api/v1/connectors/{connector_id}/sync", {})

    async def cancel_sync(self, sync_id: str) -> dict[str, Any]:
        """Cancel an ongoing sync.

        Args:
            sync_id: Sync job ID

        Returns:
            Cancellation confirmation
        """
        return await self._client._post(
            f"/api/v1/connectors/syncs/{sync_id}/cancel", {}
        )

    async def test(
        self,
        *,
        connector_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Test connector connectivity.

        Args:
            connector_id: Existing connector ID to test
            config: Configuration to test (for new connectors)

        Returns:
            Test results with connectivity status
        """
        if connector_id:
            return await self._client._post(
                f"/api/v1/connectors/{connector_id}/test", {}
            )
        return await self._client._post(
            "/api/v1/connectors/test",
            {"config": config or {}},
        )

    # =========================================================================
    # Monitoring
    # =========================================================================

    async def get_sync_history(
        self,
        *,
        connector_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get sync history.

        Args:
            connector_id: Filter by connector
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of sync jobs with status
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if connector_id:
            return await self._client._get(
                f"/api/v1/connectors/{connector_id}/syncs", params=params
            )
        return await self._client._get("/api/v1/connectors/syncs", params=params)

    async def get_stats(self) -> dict[str, Any]:
        """Get connector statistics.

        Returns:
            Aggregate stats across all connectors
        """
        return await self._client._get("/api/v1/connectors/stats")

    async def get_health(self) -> dict[str, Any]:
        """Get connector health status.

        Returns:
            Health status for all connectors
        """
        return await self._client._get("/api/v1/connectors/health")

    async def list_types(self) -> dict[str, Any]:
        """List available connector types.

        Returns:
            Available types with configuration schemas
        """
        return await self._client._get("/api/v1/connectors/types")
