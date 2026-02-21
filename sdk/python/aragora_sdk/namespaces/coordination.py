"""
Coordination Namespace API

Provides methods for cross-workspace federation:
- Register, list, and unregister workspaces
- Create and list federation policies
- Execute cross-workspace operations
- Manage data sharing consent
- Approve pending execution requests
- View coordination stats and health
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CoordinationAPI:
    """
    Synchronous Coordination API.

    Provides cross-workspace federation and coordination capabilities
    including workspace registration, federation policy management,
    cross-workspace execution, and data sharing consent.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> workspaces = client.coordination.list_workspaces()
        >>> client.coordination.register_workspace(id="ws-1", name="Primary")
        >>> client.coordination.create_federation_policy(name="default", mode="readonly")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Workspaces
    # =========================================================================

    def register_workspace(
        self,
        *,
        id: str,
        name: str = "",
        org_id: str = "",
        federation_mode: str = "readonly",
        endpoint_url: str | None = None,
        supports_agent_execution: bool = True,
        supports_workflow_execution: bool = True,
        supports_knowledge_query: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Register a workspace for federation.

        Args:
            id: Unique workspace identifier.
            name: Human-readable workspace name.
            org_id: Organization identifier.
            federation_mode: Federation mode (readonly, readwrite, isolated).
            endpoint_url: URL for cross-workspace communication.
            supports_agent_execution: Whether workspace supports agent execution.
            supports_workflow_execution: Whether workspace supports workflow execution.
            supports_knowledge_query: Whether workspace supports knowledge queries.

        Returns:
            Dict with the registered workspace data.
        """
        body: dict[str, Any] = {
            "id": id,
            "name": name,
            "org_id": org_id,
            "federation_mode": federation_mode,
            "supports_agent_execution": supports_agent_execution,
            "supports_workflow_execution": supports_workflow_execution,
            "supports_knowledge_query": supports_knowledge_query,
            **kwargs,
        }
        if endpoint_url is not None:
            body["endpoint_url"] = endpoint_url
        return self._client.request("POST", "/api/v1/coordination/workspaces", json=body)

    def list_workspaces(self) -> dict[str, Any]:
        """
        List all registered workspaces.

        Returns:
            Dict with ``workspaces`` list and ``total`` count.
        """
        return self._client.request("GET", "/api/v1/coordination/workspaces")

    def unregister_workspace(self, workspace_id: str) -> dict[str, Any]:
        """
        Unregister a workspace from federation.

        Args:
            workspace_id: ID of the workspace to unregister.

        Returns:
            Dict with ``unregistered: True`` on success.
        """
        return self._client.request(
            "DELETE", f"/api/v1/coordination/workspaces/{workspace_id}"
        )

    # =========================================================================
    # Federation Policies
    # =========================================================================

    def create_federation_policy(
        self,
        *,
        name: str,
        description: str = "",
        mode: str = "isolated",
        sharing_scope: str = "none",
        allowed_operations: list[str] | None = None,
        max_requests_per_hour: int = 100,
        require_approval: bool = False,
        audit_all_requests: bool = True,
        workspace_id: str | None = None,
        source_workspace_id: str | None = None,
        target_workspace_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Create a federation policy.

        Args:
            name: Policy name (required).
            description: Policy description.
            mode: Federation mode (isolated, readonly, readwrite).
            sharing_scope: Sharing scope (none, metadata, full).
            allowed_operations: List of allowed operation types.
            max_requests_per_hour: Rate limit for cross-workspace requests.
            require_approval: Whether requests need manual approval.
            audit_all_requests: Whether to log all requests.
            workspace_id: Apply policy to a specific workspace.
            source_workspace_id: Source workspace for pair policy.
            target_workspace_id: Target workspace for pair policy.

        Returns:
            Dict with the created policy data.
        """
        body: dict[str, Any] = {
            "name": name,
            "description": description,
            "mode": mode,
            "sharing_scope": sharing_scope,
            "max_requests_per_hour": max_requests_per_hour,
            "require_approval": require_approval,
            "audit_all_requests": audit_all_requests,
            **kwargs,
        }
        if allowed_operations is not None:
            body["allowed_operations"] = allowed_operations
        if workspace_id is not None:
            body["workspace_id"] = workspace_id
        if source_workspace_id is not None:
            body["source_workspace_id"] = source_workspace_id
        if target_workspace_id is not None:
            body["target_workspace_id"] = target_workspace_id
        return self._client.request("POST", "/api/v1/coordination/federation", json=body)

    def list_federation_policies(self) -> dict[str, Any]:
        """
        List all federation policies.

        Returns:
            Dict with ``policies`` list and ``total`` count.
        """
        return self._client.request("GET", "/api/v1/coordination/federation")

    # Backward-compatible aliases
    create_policy = create_federation_policy
    list_policies = list_federation_policies

    # =========================================================================
    # Execution
    # =========================================================================

    def execute_cross_workspace(
        self,
        *,
        operation: str,
        source_workspace_id: str,
        target_workspace_id: str,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float = 30.0,
        requester_id: str = "",
        consent_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute a cross-workspace operation.

        Args:
            operation: Operation type (e.g., agent_execution, knowledge_query).
            source_workspace_id: ID of the source workspace.
            target_workspace_id: ID of the target workspace.
            payload: Operation-specific payload data.
            timeout_seconds: Request timeout in seconds.
            requester_id: ID of the user or agent making the request.
            consent_id: ID of the consent authorizing this operation.

        Returns:
            Dict with execution result including ``success`` and ``data``.
        """
        body: dict[str, Any] = {
            "operation": operation,
            "source_workspace_id": source_workspace_id,
            "target_workspace_id": target_workspace_id,
            "timeout_seconds": timeout_seconds,
            "requester_id": requester_id,
            **kwargs,
        }
        if payload is not None:
            body["payload"] = payload
        if consent_id is not None:
            body["consent_id"] = consent_id
        return self._client.request("POST", "/api/v1/coordination/execute", json=body)

    # Backward-compatible alias
    execute = execute_cross_workspace

    def list_executions(self, workspace_id: str | None = None) -> dict[str, Any]:
        """
        List pending cross-workspace execution requests.

        Args:
            workspace_id: Optional workspace ID to filter by.

        Returns:
            Dict with ``executions`` list and ``total`` count.
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request(
            "GET", "/api/v1/coordination/executions", params=params
        )

    # =========================================================================
    # Consent
    # =========================================================================

    def grant_consent(
        self,
        *,
        source_workspace_id: str,
        target_workspace_id: str,
        scope: str = "metadata",
        data_types: list[str] | None = None,
        operations: list[str] | None = None,
        granted_by: str = "",
        expires_in_days: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Grant data sharing consent between workspaces.

        Args:
            source_workspace_id: ID of the workspace sharing data.
            target_workspace_id: ID of the workspace receiving data.
            scope: Sharing scope (none, metadata, full).
            data_types: Types of data allowed for sharing.
            operations: Operation types covered by this consent.
            granted_by: ID of the user granting consent.
            expires_in_days: Number of days until consent expires.

        Returns:
            Dict with the created consent data.
        """
        body: dict[str, Any] = {
            "source_workspace_id": source_workspace_id,
            "target_workspace_id": target_workspace_id,
            "scope": scope,
            "granted_by": granted_by,
            **kwargs,
        }
        if data_types is not None:
            body["data_types"] = data_types
        if operations is not None:
            body["operations"] = operations
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        return self._client.request(
            "POST", "/api/v1/coordination/consent", json=body
        )

    def revoke_consent(self, consent_id: str, **kwargs: Any) -> dict[str, Any]:
        """
        Revoke a data sharing consent.

        Args:
            consent_id: ID of the consent to revoke.

        Returns:
            Dict with ``revoked: True`` on success.
        """
        return self._client.request(
            "DELETE", f"/api/v1/coordination/consent/{consent_id}"
        )

    def list_consents(self, workspace_id: str | None = None) -> dict[str, Any]:
        """
        List data sharing consents.

        Args:
            workspace_id: Optional workspace ID to filter by.

        Returns:
            Dict with ``consents`` list and ``total`` count.
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request(
            "GET", "/api/v1/coordination/consent", params=params
        )

    # =========================================================================
    # Approval
    # =========================================================================

    def approve_request(
        self, request_id: str, *, approved_by: str = "api", **kwargs: Any
    ) -> dict[str, Any]:
        """
        Approve a pending cross-workspace execution request.

        Args:
            request_id: ID of the execution request to approve.
            approved_by: ID of the user approving the request.

        Returns:
            Dict with ``approved: True`` on success.
        """
        body: dict[str, Any] = {"approved_by": approved_by, **kwargs}
        return self._client.request(
            "POST", f"/api/v1/coordination/approve/{request_id}", json=body
        )

    # =========================================================================
    # Stats and Health
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get coordination statistics.

        Returns:
            Dict with stats including total_workspaces, pending_requests, etc.
        """
        return self._client.request("GET", "/api/v1/coordination/stats")

    def get_health(self) -> dict[str, Any]:
        """
        Get coordination health status.

        Returns:
            Dict with ``status`` (healthy/degraded/unavailable) and summary info.
        """
        return self._client.request("GET", "/api/v1/coordination/health")


class AsyncCoordinationAPI:
    """
    Asynchronous Coordination API.

    Provides async cross-workspace federation and coordination capabilities.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     workspaces = await client.coordination.list_workspaces()
        ...     await client.coordination.create_federation_policy(name="default")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Workspaces
    # =========================================================================

    async def register_workspace(
        self,
        *,
        id: str,
        name: str = "",
        org_id: str = "",
        federation_mode: str = "readonly",
        endpoint_url: str | None = None,
        supports_agent_execution: bool = True,
        supports_workflow_execution: bool = True,
        supports_knowledge_query: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Register a workspace for federation."""
        body: dict[str, Any] = {
            "id": id,
            "name": name,
            "org_id": org_id,
            "federation_mode": federation_mode,
            "supports_agent_execution": supports_agent_execution,
            "supports_workflow_execution": supports_workflow_execution,
            "supports_knowledge_query": supports_knowledge_query,
            **kwargs,
        }
        if endpoint_url is not None:
            body["endpoint_url"] = endpoint_url
        return await self._client.request(
            "POST", "/api/v1/coordination/workspaces", json=body
        )

    async def list_workspaces(self) -> dict[str, Any]:
        """List all registered workspaces."""
        return await self._client.request("GET", "/api/v1/coordination/workspaces")

    async def unregister_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Unregister a workspace from federation."""
        return await self._client.request(
            "DELETE", f"/api/v1/coordination/workspaces/{workspace_id}"
        )

    # =========================================================================
    # Federation Policies
    # =========================================================================

    async def create_federation_policy(
        self,
        *,
        name: str,
        description: str = "",
        mode: str = "isolated",
        sharing_scope: str = "none",
        allowed_operations: list[str] | None = None,
        max_requests_per_hour: int = 100,
        require_approval: bool = False,
        audit_all_requests: bool = True,
        workspace_id: str | None = None,
        source_workspace_id: str | None = None,
        target_workspace_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a federation policy."""
        body: dict[str, Any] = {
            "name": name,
            "description": description,
            "mode": mode,
            "sharing_scope": sharing_scope,
            "max_requests_per_hour": max_requests_per_hour,
            "require_approval": require_approval,
            "audit_all_requests": audit_all_requests,
            **kwargs,
        }
        if allowed_operations is not None:
            body["allowed_operations"] = allowed_operations
        if workspace_id is not None:
            body["workspace_id"] = workspace_id
        if source_workspace_id is not None:
            body["source_workspace_id"] = source_workspace_id
        if target_workspace_id is not None:
            body["target_workspace_id"] = target_workspace_id
        return await self._client.request(
            "POST", "/api/v1/coordination/federation", json=body
        )

    async def list_federation_policies(self) -> dict[str, Any]:
        """List all federation policies."""
        return await self._client.request("GET", "/api/v1/coordination/federation")

    # Backward-compatible aliases
    create_policy = create_federation_policy
    list_policies = list_federation_policies

    # =========================================================================
    # Execution
    # =========================================================================

    async def execute_cross_workspace(
        self,
        *,
        operation: str,
        source_workspace_id: str,
        target_workspace_id: str,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float = 30.0,
        requester_id: str = "",
        consent_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a cross-workspace operation."""
        body: dict[str, Any] = {
            "operation": operation,
            "source_workspace_id": source_workspace_id,
            "target_workspace_id": target_workspace_id,
            "timeout_seconds": timeout_seconds,
            "requester_id": requester_id,
            **kwargs,
        }
        if payload is not None:
            body["payload"] = payload
        if consent_id is not None:
            body["consent_id"] = consent_id
        return await self._client.request(
            "POST", "/api/v1/coordination/execute", json=body
        )

    # Backward-compatible alias
    execute = execute_cross_workspace

    async def list_executions(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List pending cross-workspace execution requests."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/v1/coordination/executions", params=params
        )

    # =========================================================================
    # Consent
    # =========================================================================

    async def grant_consent(
        self,
        *,
        source_workspace_id: str,
        target_workspace_id: str,
        scope: str = "metadata",
        data_types: list[str] | None = None,
        operations: list[str] | None = None,
        granted_by: str = "",
        expires_in_days: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Grant data sharing consent between workspaces."""
        body: dict[str, Any] = {
            "source_workspace_id": source_workspace_id,
            "target_workspace_id": target_workspace_id,
            "scope": scope,
            "granted_by": granted_by,
            **kwargs,
        }
        if data_types is not None:
            body["data_types"] = data_types
        if operations is not None:
            body["operations"] = operations
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        return await self._client.request(
            "POST", "/api/v1/coordination/consent", json=body
        )

    async def revoke_consent(self, consent_id: str, **kwargs: Any) -> dict[str, Any]:
        """Revoke a data sharing consent."""
        return await self._client.request(
            "DELETE", f"/api/v1/coordination/consent/{consent_id}"
        )

    async def list_consents(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List data sharing consents."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/v1/coordination/consent", params=params
        )

    # =========================================================================
    # Approval
    # =========================================================================

    async def approve_request(
        self, request_id: str, *, approved_by: str = "api", **kwargs: Any
    ) -> dict[str, Any]:
        """Approve a pending cross-workspace execution request."""
        body: dict[str, Any] = {"approved_by": approved_by, **kwargs}
        return await self._client.request(
            "POST", f"/api/v1/coordination/approve/{request_id}", json=body
        )

    # =========================================================================
    # Stats and Health
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get coordination statistics."""
        return await self._client.request("GET", "/api/v1/coordination/stats")

    async def get_health(self) -> dict[str, Any]:
        """Get coordination health status."""
        return await self._client.request("GET", "/api/v1/coordination/health")
