"""
Connectors Namespace API

Provides methods for managing enterprise data source connectors,
sync operations, and scheduler configuration.

Features:
- Connector management (GitHub Enterprise, S3, PostgreSQL, MongoDB, FHIR)
- Sync scheduling and monitoring
- Connector health checks
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


ConnectorType = Literal["github_enterprise", "s3", "postgresql", "mongodb", "fhir"]
SyncFrequency = Literal["hourly", "daily", "weekly", "manual"]


class ConnectorsAPI:
    """
    Synchronous Connectors API.

    Provides methods for managing enterprise data source connectors:
    - Connector CRUD operations
    - Sync scheduling
    - Health monitoring

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> connectors = client.connectors.list()
        >>> client.connectors.trigger_sync(connector_id="abc123")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Connector Management
    # ===========================================================================

    def list(
        self,
        connector_type: ConnectorType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List configured connectors.

        Args:
            connector_type: Filter by connector type
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            Dict with connectors array and pagination info
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if connector_type:
            params["type"] = connector_type

        return self._client.request("GET", "/api/v1/connectors", params=params)

    def get(self, connector_id: str) -> dict[str, Any]:
        """
        Get a connector by ID.

        Args:
            connector_id: Connector ID

        Returns:
            Dict with connector details
        """
        return self._client.request("GET", f"/api/v1/connectors/{connector_id}")

    def create(
        self,
        name: str,
        connector_type: ConnectorType,
        config: dict[str, Any],
        schedule: SyncFrequency = "daily",
    ) -> dict[str, Any]:
        """
        Create a new connector.

        Args:
            name: Connector name
            connector_type: Type of connector
            config: Connector-specific configuration
            schedule: Sync frequency

        Returns:
            Dict with created connector details
        """
        return self._client.request(
            "POST",
            "/api/v1/connectors",
            json={
                "name": name,
                "type": connector_type,
                "config": config,
                "schedule": schedule,
            },
        )

    def update(
        self,
        connector_id: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        schedule: SyncFrequency | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """
        Update a connector.

        Args:
            connector_id: Connector ID
            name: New name
            config: New configuration
            schedule: New sync frequency
            enabled: Enable/disable connector

        Returns:
            Dict with updated connector details
        """
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if config is not None:
            data["config"] = config
        if schedule is not None:
            data["schedule"] = schedule
        if enabled is not None:
            data["enabled"] = enabled

        return self._client.request("PATCH", f"/api/v1/connectors/{connector_id}", json=data)

    def delete(self, connector_id: str) -> dict[str, Any]:
        """
        Delete a connector.

        Args:
            connector_id: Connector ID

        Returns:
            Dict with success status
        """
        return self._client.request("DELETE", f"/api/v1/connectors/{connector_id}")

    # ===========================================================================
    # Sync Operations
    # ===========================================================================

    def trigger_sync(
        self,
        connector_id: str,
        full_sync: bool = False,
    ) -> dict[str, Any]:
        """
        Trigger a sync for a connector.

        Args:
            connector_id: Connector ID
            full_sync: Whether to do a full sync (vs incremental)

        Returns:
            Dict with sync_id and status
        """
        return self._client.request(
            "POST",
            f"/api/v1/connectors/{connector_id}/sync",
            json={"full_sync": full_sync},
        )

    def get_sync_status(self, connector_id: str, sync_id: str) -> dict[str, Any]:
        """
        Get status of a sync operation.

        Args:
            connector_id: Connector ID
            sync_id: Sync operation ID

        Returns:
            Dict with sync status and progress
        """
        return self._client.request("GET", f"/api/v1/connectors/{connector_id}/syncs/{sync_id}")

    def list_syncs(
        self,
        connector_id: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        List recent sync operations for a connector.

        Args:
            connector_id: Connector ID
            limit: Maximum number of results

        Returns:
            Dict with syncs array
        """
        return self._client.request(
            "GET",
            f"/api/v1/connectors/{connector_id}/syncs",
            params={"limit": limit},
        )

    def cancel_sync(self, connector_id: str, sync_id: str) -> dict[str, Any]:
        """
        Cancel a running sync operation.

        Args:
            connector_id: Connector ID
            sync_id: Sync operation ID

        Returns:
            Dict with success status
        """
        return self._client.request(
            "POST", f"/api/v1/connectors/{connector_id}/syncs/{sync_id}/cancel"
        )

    # ===========================================================================
    # Health and Monitoring
    # ===========================================================================

    def test_connection(self, connector_id: str) -> dict[str, Any]:
        """
        Test connectivity for a connector.

        Args:
            connector_id: Connector ID

        Returns:
            Dict with connection_ok and latency_ms
        """
        return self._client.request("POST", f"/api/v1/connectors/{connector_id}/test")

    def get_health(self, connector_id: str) -> dict[str, Any]:
        """
        Get health status of a connector.

        Args:
            connector_id: Connector ID

        Returns:
            Dict with health status, last_sync, and error info
        """
        return self._client.request("GET", f"/api/v1/connectors/{connector_id}/health")


class AsyncConnectorsAPI:
    """
    Asynchronous Connectors API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     connectors = await client.connectors.list()
        ...     await client.connectors.trigger_sync(connector_id="abc123")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Connector Management
    # ===========================================================================

    async def list(
        self,
        connector_type: ConnectorType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List configured connectors."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if connector_type:
            params["type"] = connector_type

        return await self._client.request("GET", "/api/v1/connectors", params=params)

    async def get(self, connector_id: str) -> dict[str, Any]:
        """Get a connector by ID."""
        return await self._client.request("GET", f"/api/v1/connectors/{connector_id}")

    async def create(
        self,
        name: str,
        connector_type: ConnectorType,
        config: dict[str, Any],
        schedule: SyncFrequency = "daily",
    ) -> dict[str, Any]:
        """Create a new connector."""
        return await self._client.request(
            "POST",
            "/api/v1/connectors",
            json={
                "name": name,
                "type": connector_type,
                "config": config,
                "schedule": schedule,
            },
        )

    async def update(
        self,
        connector_id: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        schedule: SyncFrequency | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Update a connector."""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if config is not None:
            data["config"] = config
        if schedule is not None:
            data["schedule"] = schedule
        if enabled is not None:
            data["enabled"] = enabled

        return await self._client.request("PATCH", f"/api/v1/connectors/{connector_id}", json=data)

    async def delete(self, connector_id: str) -> dict[str, Any]:
        """Delete a connector."""
        return await self._client.request("DELETE", f"/api/v1/connectors/{connector_id}")

    # ===========================================================================
    # Sync Operations
    # ===========================================================================

    async def trigger_sync(
        self,
        connector_id: str,
        full_sync: bool = False,
    ) -> dict[str, Any]:
        """Trigger a sync for a connector."""
        return await self._client.request(
            "POST",
            f"/api/v1/connectors/{connector_id}/sync",
            json={"full_sync": full_sync},
        )

    async def get_sync_status(self, connector_id: str, sync_id: str) -> dict[str, Any]:
        """Get status of a sync operation."""
        return await self._client.request(
            "GET", f"/api/v1/connectors/{connector_id}/syncs/{sync_id}"
        )

    async def list_syncs(
        self,
        connector_id: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List recent sync operations for a connector."""
        return await self._client.request(
            "GET",
            f"/api/v1/connectors/{connector_id}/syncs",
            params={"limit": limit},
        )

    async def cancel_sync(self, connector_id: str, sync_id: str) -> dict[str, Any]:
        """Cancel a running sync operation."""
        return await self._client.request(
            "POST", f"/api/v1/connectors/{connector_id}/syncs/{sync_id}/cancel"
        )

    # ===========================================================================
    # Health and Monitoring
    # ===========================================================================

    async def test_connection(self, connector_id: str) -> dict[str, Any]:
        """Test connectivity for a connector."""
        return await self._client.request("POST", f"/api/v1/connectors/{connector_id}/test")

    async def get_health(self, connector_id: str) -> dict[str, Any]:
        """Get health status of a connector."""
        return await self._client.request("GET", f"/api/v1/connectors/{connector_id}/health")
