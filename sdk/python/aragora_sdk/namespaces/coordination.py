"""
Coordination Namespace API

Provides methods for cross-workspace federation:
- Register, list, and unregister workspaces
- Create and list federation policies
- Execute cross-workspace operations
- Manage data sharing consent
- View coordination stats and health
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CoordinationAPI:
    """
    Synchronous Coordination API.

    Provides cross-workspace federation and coordination capabilities.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> workspaces = client.coordination.list_workspaces()
        >>> client.coordination.register_workspace(id="ws-1", name="Primary")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Workspaces
    # =========================================================================

    def register_workspace(self, **kwargs: Any) -> dict[str, Any]:
        """Register a workspace for federation."""
        return self._client.request("POST", "/api/v1/coordination/workspaces", json=kwargs)

    def list_workspaces(self) -> dict[str, Any]:
        """List registered workspaces."""
        return self._client.request("GET", "/api/v1/coordination/workspaces")

    def unregister_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Unregister a workspace."""
        return self._client.request("DELETE", f"/api/v1/coordination/workspaces/{workspace_id}")

    # =========================================================================
    # Federation Policies
    # =========================================================================

    def create_policy(self, **kwargs: Any) -> dict[str, Any]:
        """Create a federation policy."""
        return self._client.request("POST", "/api/v1/coordination/federation", json=kwargs)

    def list_policies(self) -> dict[str, Any]:
        """List federation policies."""
        return self._client.request("GET", "/api/v1/coordination/federation")

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a cross-workspace operation."""
        return self._client.request("POST", "/api/v1/coordination/execute", json=kwargs)

    def list_executions(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List pending executions."""
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/coordination/executions", params=params)

    def approve_request(self, request_id: str, **kwargs: Any) -> dict[str, Any]:
        """Approve a pending execution request."""
        return self._client.request(
            "POST", f"/api/v1/coordination/approve/{request_id}", json=kwargs
        )

    # =========================================================================
    # Consent
    # =========================================================================

    def grant_consent(self, **kwargs: Any) -> dict[str, Any]:
        """Grant data sharing consent."""
        return self._client.request("POST", "/api/v1/coordination/consent", json=kwargs)

    def revoke_consent(self, consent_id: str) -> dict[str, Any]:
        """Revoke a data sharing consent."""
        return self._client.request("DELETE", f"/api/v1/coordination/consent/{consent_id}")

    def list_consents(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List data sharing consents."""
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/coordination/consent", params=params)

    # =========================================================================
    # Stats and Health
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get coordination statistics."""
        return self._client.request("GET", "/api/v1/coordination/stats")

    def get_health(self) -> dict[str, Any]:
        """Get coordination health status."""
        return self._client.request("GET", "/api/v1/coordination/health")


class AsyncCoordinationAPI:
    """
    Asynchronous Coordination API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     workspaces = await client.coordination.list_workspaces()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def register_workspace(self, **kwargs: Any) -> dict[str, Any]:
        """Register a workspace for federation."""
        return await self._client.request("POST", "/api/v1/coordination/workspaces", json=kwargs)

    async def list_workspaces(self) -> dict[str, Any]:
        """List registered workspaces."""
        return await self._client.request("GET", "/api/v1/coordination/workspaces")

    async def unregister_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Unregister a workspace."""
        return await self._client.request(
            "DELETE", f"/api/v1/coordination/workspaces/{workspace_id}"
        )

    async def create_policy(self, **kwargs: Any) -> dict[str, Any]:
        """Create a federation policy."""
        return await self._client.request("POST", "/api/v1/coordination/federation", json=kwargs)

    async def list_policies(self) -> dict[str, Any]:
        """List federation policies."""
        return await self._client.request("GET", "/api/v1/coordination/federation")

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a cross-workspace operation."""
        return await self._client.request("POST", "/api/v1/coordination/execute", json=kwargs)

    async def list_executions(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List pending executions."""
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/v1/coordination/executions", params=params
        )

    async def approve_request(self, request_id: str, **kwargs: Any) -> dict[str, Any]:
        """Approve a pending execution request."""
        return await self._client.request(
            "POST", f"/api/v1/coordination/approve/{request_id}", json=kwargs
        )

    async def grant_consent(self, **kwargs: Any) -> dict[str, Any]:
        """Grant data sharing consent."""
        return await self._client.request("POST", "/api/v1/coordination/consent", json=kwargs)

    async def revoke_consent(self, consent_id: str) -> dict[str, Any]:
        """Revoke a data sharing consent."""
        return await self._client.request(
            "DELETE", f"/api/v1/coordination/consent/{consent_id}"
        )

    async def list_consents(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List data sharing consents."""
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/v1/coordination/consent", params=params
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get coordination statistics."""
        return await self._client.request("GET", "/api/v1/coordination/stats")

    async def get_health(self) -> dict[str, Any]:
        """Get coordination health status."""
        return await self._client.request("GET", "/api/v1/coordination/health")
