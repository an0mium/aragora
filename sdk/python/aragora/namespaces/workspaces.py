"""
Workspaces Namespace API

Provides methods for workspace management, data isolation, and privacy controls.

Features:
- Workspace creation and management
- Member access control
- Retention policy management
- Content sensitivity classification
- Privacy audit logging
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class WorkspacesAPI:
    """
    Synchronous Workspaces API.

    Provides methods for workspace and privacy management:
    - Create and manage workspaces
    - Manage workspace members
    - Configure retention policies
    - Classify content sensitivity
    - Query audit logs

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> workspaces = client.workspaces.list()
        >>> workspace = client.workspaces.create(name="Engineering", tenant_id="t-123")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Workspace Management
    # ===========================================================================

    def list(
        self,
        tenant_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List workspaces.

        Args:
            tenant_id: Optional tenant filter
            limit: Maximum results (default: 20)
            offset: Pagination offset

        Returns:
            Dict with workspaces array and count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if tenant_id:
            params["tenant_id"] = tenant_id
        return self._client.request("GET", "/api/v1/workspaces", params=params)

    def create(
        self,
        name: str,
        tenant_id: str | None = None,
        description: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new workspace.

        Args:
            name: Workspace name
            tenant_id: Optional tenant ID
            description: Optional description
            settings: Optional workspace settings

        Returns:
            Created workspace info
        """
        data: dict[str, Any] = {"name": name}
        if tenant_id:
            data["tenant_id"] = tenant_id
        if description:
            data["description"] = description
        if settings:
            data["settings"] = settings

        return self._client.request("POST", "/api/v1/workspaces", json=data)

    def get(self, workspace_id: str) -> dict[str, Any]:
        """
        Get workspace details.

        Args:
            workspace_id: Workspace ID

        Returns:
            Workspace details
        """
        return self._client.request("GET", f"/api/v1/workspaces/{workspace_id}")

    def delete(self, workspace_id: str) -> dict[str, Any]:
        """
        Delete a workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            Dict with success status
        """
        return self._client.request("DELETE", f"/api/v1/workspaces/{workspace_id}")

    # ===========================================================================
    # Member Management
    # ===========================================================================

    def add_member(
        self,
        workspace_id: str,
        user_id: str,
        role: str = "member",
    ) -> dict[str, Any]:
        """
        Add a member to a workspace.

        Args:
            workspace_id: Workspace ID
            user_id: User ID to add
            role: Member role (default: "member")

        Returns:
            Dict with membership info
        """
        return self._client.request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/members",
            json={"user_id": user_id, "role": role},
        )

    def remove_member(self, workspace_id: str, user_id: str) -> dict[str, Any]:
        """
        Remove a member from a workspace.

        Args:
            workspace_id: Workspace ID
            user_id: User ID to remove

        Returns:
            Dict with success status
        """
        return self._client.request(
            "DELETE", f"/api/v1/workspaces/{workspace_id}/members/{user_id}"
        )

    # ===========================================================================
    # Retention Policies
    # ===========================================================================

    def list_retention_policies(self) -> dict[str, Any]:
        """
        List retention policies.

        Returns:
            Dict with policies array
        """
        return self._client.request("GET", "/api/v1/retention/policies")

    def create_retention_policy(
        self,
        name: str,
        retention_days: int,
        data_types: list[str] | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a retention policy.

        Args:
            name: Policy name
            retention_days: Data retention period in days
            data_types: Optional list of data types to apply policy to
            workspace_id: Optional workspace to scope policy to

        Returns:
            Created policy info
        """
        data: dict[str, Any] = {"name": name, "retention_days": retention_days}
        if data_types:
            data["data_types"] = data_types
        if workspace_id:
            data["workspace_id"] = workspace_id

        return self._client.request("POST", "/api/v1/retention/policies", json=data)

    def update_retention_policy(
        self,
        policy_id: str,
        name: str | None = None,
        retention_days: int | None = None,
        data_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Update a retention policy.

        Args:
            policy_id: Policy ID
            name: New policy name
            retention_days: New retention period
            data_types: New data types list

        Returns:
            Updated policy info
        """
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if retention_days is not None:
            data["retention_days"] = retention_days
        if data_types is not None:
            data["data_types"] = data_types

        return self._client.request("PUT", f"/api/v1/retention/policies/{policy_id}", json=data)

    def delete_retention_policy(self, policy_id: str) -> dict[str, Any]:
        """
        Delete a retention policy.

        Args:
            policy_id: Policy ID

        Returns:
            Dict with success status
        """
        return self._client.request("DELETE", f"/api/v1/retention/policies/{policy_id}")

    def execute_retention_policy(self, policy_id: str) -> dict[str, Any]:
        """
        Execute a retention policy (apply data cleanup).

        Args:
            policy_id: Policy ID

        Returns:
            Dict with execution results
        """
        return self._client.request("POST", f"/api/v1/retention/policies/{policy_id}/execute")

    def get_expiring_items(self, days: int = 30) -> dict[str, Any]:
        """
        Get items expiring within the specified days.

        Args:
            days: Days until expiration (default: 30)

        Returns:
            Dict with expiring items list
        """
        return self._client.request("GET", "/api/v1/retention/expiring", params={"days": days})

    # ===========================================================================
    # Content Classification
    # ===========================================================================

    def classify_content(
        self,
        content: str,
        content_type: str = "text",
    ) -> dict[str, Any]:
        """
        Classify content sensitivity level.

        Args:
            content: Content to classify
            content_type: Type of content (default: "text")

        Returns:
            Dict with sensitivity level and confidence
        """
        return self._client.request(
            "POST",
            "/api/v1/classify",
            json={"content": content, "content_type": content_type},
        )

    def get_classification_policy(self, level: str) -> dict[str, Any]:
        """
        Get policy for a sensitivity level.

        Args:
            level: Sensitivity level (public, internal, confidential, restricted)

        Returns:
            Dict with policy details
        """
        return self._client.request("GET", f"/api/v1/classify/policy/{level}")

    # ===========================================================================
    # Audit
    # ===========================================================================

    def query_audit_entries(
        self,
        workspace_id: str | None = None,
        action: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Query audit log entries.

        Args:
            workspace_id: Filter by workspace
            action: Filter by action type
            start_time: Start time (ISO 8601)
            end_time: End time (ISO 8601)
            limit: Maximum results (default: 100)

        Returns:
            Dict with audit entries
        """
        params: dict[str, Any] = {"limit": limit}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if action:
            params["action"] = action
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        return self._client.request("GET", "/api/v1/audit/entries", params=params)

    def generate_audit_report(
        self,
        report_type: str = "compliance",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a compliance audit report.

        Args:
            report_type: Type of report (compliance, access, data)
            start_date: Report start date (ISO 8601)
            end_date: Report end date (ISO 8601)

        Returns:
            Dict with report data
        """
        params: dict[str, Any] = {"report_type": report_type}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self._client.request("GET", "/api/v1/audit/report", params=params)

    def verify_audit_integrity(self) -> dict[str, Any]:
        """
        Verify audit log integrity.

        Returns:
            Dict with verification results
        """
        return self._client.request("GET", "/api/v1/audit/verify")


class AsyncWorkspacesAPI:
    """
    Asynchronous Workspaces API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     workspaces = await client.workspaces.list()
        ...     workspace = await client.workspaces.create(name="Engineering")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Workspace Management
    # ===========================================================================

    async def list(
        self,
        tenant_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List workspaces."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if tenant_id:
            params["tenant_id"] = tenant_id
        return await self._client.request("GET", "/api/v1/workspaces", params=params)

    async def create(
        self,
        name: str,
        tenant_id: str | None = None,
        description: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new workspace."""
        data: dict[str, Any] = {"name": name}
        if tenant_id:
            data["tenant_id"] = tenant_id
        if description:
            data["description"] = description
        if settings:
            data["settings"] = settings

        return await self._client.request("POST", "/api/v1/workspaces", json=data)

    async def get(self, workspace_id: str) -> dict[str, Any]:
        """Get workspace details."""
        return await self._client.request("GET", f"/api/v1/workspaces/{workspace_id}")

    async def delete(self, workspace_id: str) -> dict[str, Any]:
        """Delete a workspace."""
        return await self._client.request("DELETE", f"/api/v1/workspaces/{workspace_id}")

    # ===========================================================================
    # Member Management
    # ===========================================================================

    async def add_member(
        self,
        workspace_id: str,
        user_id: str,
        role: str = "member",
    ) -> dict[str, Any]:
        """Add a member to a workspace."""
        return await self._client.request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/members",
            json={"user_id": user_id, "role": role},
        )

    async def remove_member(self, workspace_id: str, user_id: str) -> dict[str, Any]:
        """Remove a member from a workspace."""
        return await self._client.request(
            "DELETE", f"/api/v1/workspaces/{workspace_id}/members/{user_id}"
        )

    # ===========================================================================
    # Retention Policies
    # ===========================================================================

    async def list_retention_policies(self) -> dict[str, Any]:
        """List retention policies."""
        return await self._client.request("GET", "/api/v1/retention/policies")

    async def create_retention_policy(
        self,
        name: str,
        retention_days: int,
        data_types: list[str] | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a retention policy."""
        data: dict[str, Any] = {"name": name, "retention_days": retention_days}
        if data_types:
            data["data_types"] = data_types
        if workspace_id:
            data["workspace_id"] = workspace_id

        return await self._client.request("POST", "/api/v1/retention/policies", json=data)

    async def update_retention_policy(
        self,
        policy_id: str,
        name: str | None = None,
        retention_days: int | None = None,
        data_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update a retention policy."""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if retention_days is not None:
            data["retention_days"] = retention_days
        if data_types is not None:
            data["data_types"] = data_types

        return await self._client.request(
            "PUT", f"/api/v1/retention/policies/{policy_id}", json=data
        )

    async def delete_retention_policy(self, policy_id: str) -> dict[str, Any]:
        """Delete a retention policy."""
        return await self._client.request("DELETE", f"/api/v1/retention/policies/{policy_id}")

    async def execute_retention_policy(self, policy_id: str) -> dict[str, Any]:
        """Execute a retention policy."""
        return await self._client.request("POST", f"/api/v1/retention/policies/{policy_id}/execute")

    async def get_expiring_items(self, days: int = 30) -> dict[str, Any]:
        """Get items expiring within the specified days."""
        return await self._client.request(
            "GET", "/api/v1/retention/expiring", params={"days": days}
        )

    # ===========================================================================
    # Content Classification
    # ===========================================================================

    async def classify_content(
        self,
        content: str,
        content_type: str = "text",
    ) -> dict[str, Any]:
        """Classify content sensitivity level."""
        return await self._client.request(
            "POST",
            "/api/v1/classify",
            json={"content": content, "content_type": content_type},
        )

    async def get_classification_policy(self, level: str) -> dict[str, Any]:
        """Get policy for a sensitivity level."""
        return await self._client.request("GET", f"/api/v1/classify/policy/{level}")

    # ===========================================================================
    # Audit
    # ===========================================================================

    async def query_audit_entries(
        self,
        workspace_id: str | None = None,
        action: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Query audit log entries."""
        params: dict[str, Any] = {"limit": limit}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if action:
            params["action"] = action
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        return await self._client.request("GET", "/api/v1/audit/entries", params=params)

    async def generate_audit_report(
        self,
        report_type: str = "compliance",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Generate a compliance audit report."""
        params: dict[str, Any] = {"report_type": report_type}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return await self._client.request("GET", "/api/v1/audit/report", params=params)

    async def verify_audit_integrity(self) -> dict[str, Any]:
        """Verify audit log integrity."""
        return await self._client.request("GET", "/api/v1/audit/verify")
