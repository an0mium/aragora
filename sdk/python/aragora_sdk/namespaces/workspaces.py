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


_List = list  # Preserve builtin list for type annotations


class WorkspacesAPI:
    """
    Synchronous Workspaces API.

    Provides methods for workspace and privacy management:
    - Create and manage workspaces
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
        data_types: _List[str] | None = None,
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
        data_types: _List[str] | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a retention policy."""
        data: dict[str, Any] = {"name": name, "retention_days": retention_days}
        if data_types:
            data["data_types"] = data_types
        if workspace_id:
            data["workspace_id"] = workspace_id

        return await self._client.request("POST", "/api/v1/retention/policies", json=data)

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
