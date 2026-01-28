"""
Policies Namespace API

Provides methods for compliance policy management:
- Policy CRUD operations
- Violation tracking
- Compliance checking
- Statistics and reporting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class PoliciesAPI:
    """
    Synchronous Policies API.

    Provides methods for compliance policy management:
    - Create, read, update, delete policies
    - Track and manage violations
    - Run compliance checks
    - Get compliance statistics

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> policies = client.policies.list()
        >>> stats = client.policies.get_stats()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Policy CRUD
    # ===========================================================================

    def list(
        self,
        workspace_id: str | None = None,
        vertical_id: str | None = None,
        framework_id: str | None = None,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List policies with optional filters.

        Args:
            workspace_id: Filter by workspace
            vertical_id: Filter by vertical
            framework_id: Filter by framework
            enabled_only: Only return enabled policies
            limit: Maximum results (default: 100)
            offset: Pagination offset

        Returns:
            Dict with policies array and total count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if vertical_id:
            params["vertical_id"] = vertical_id
        if framework_id:
            params["framework_id"] = framework_id
        if enabled_only:
            params["enabled_only"] = "true"
        return self._client.request("GET", "/api/v1/policies", params=params)

    def get(self, policy_id: str) -> dict[str, Any]:
        """
        Get a specific policy.

        Args:
            policy_id: Policy ID

        Returns:
            Dict with policy details
        """
        return self._client.request("GET", f"/api/v1/policies/{policy_id}")

    def create(
        self,
        name: str,
        framework_id: str,
        vertical_id: str,
        description: str | None = None,
        workspace_id: str = "default",
        level: str = "recommended",
        enabled: bool = True,
        rules: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new policy.

        Args:
            name: Policy name
            framework_id: Compliance framework ID
            vertical_id: Vertical/industry ID
            description: Policy description
            workspace_id: Workspace ID (default: "default")
            level: Policy level (required, recommended, optional)
            enabled: Whether policy is enabled (default: True)
            rules: List of policy rules
            metadata: Additional metadata

        Returns:
            Dict with created policy
        """
        data: dict[str, Any] = {
            "name": name,
            "framework_id": framework_id,
            "vertical_id": vertical_id,
            "workspace_id": workspace_id,
            "level": level,
            "enabled": enabled,
            "rules": rules or [],
            "metadata": metadata or {},
        }
        if description:
            data["description"] = description
        return self._client.request("POST", "/api/v1/policies", json=data)

    def update(
        self,
        policy_id: str,
        name: str | None = None,
        description: str | None = None,
        level: str | None = None,
        enabled: bool | None = None,
        rules: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update a policy.

        Args:
            policy_id: Policy ID
            name: New name
            description: New description
            level: New level
            enabled: New enabled status
            rules: New rules list
            metadata: New metadata

        Returns:
            Dict with updated policy
        """
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if level is not None:
            data["level"] = level
        if enabled is not None:
            data["enabled"] = enabled
        if rules is not None:
            data["rules"] = rules
        if metadata is not None:
            data["metadata"] = metadata
        return self._client.request("PATCH", f"/api/v1/policies/{policy_id}", json=data)

    def delete(self, policy_id: str) -> dict[str, Any]:
        """
        Delete a policy.

        Args:
            policy_id: Policy ID

        Returns:
            Dict with deletion confirmation
        """
        return self._client.request("DELETE", f"/api/v1/policies/{policy_id}")

    def toggle(self, policy_id: str, enabled: bool | None = None) -> dict[str, Any]:
        """
        Toggle a policy's enabled status.

        Args:
            policy_id: Policy ID
            enabled: New enabled status (if None, toggles current state)

        Returns:
            Dict with toggle result
        """
        data: dict[str, Any] = {}
        if enabled is not None:
            data["enabled"] = enabled
        return self._client.request("POST", f"/api/v1/policies/{policy_id}/toggle", json=data)

    def get_policy_violations(
        self,
        policy_id: str,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get violations for a specific policy.

        Args:
            policy_id: Policy ID
            status: Filter by status (open, investigating, resolved, false_positive)
            severity: Filter by severity (critical, high, medium, low)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Dict with violations array
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if severity:
            params["severity"] = severity
        return self._client.request(
            "GET", f"/api/v1/policies/{policy_id}/violations", params=params
        )

    # ===========================================================================
    # Violations
    # ===========================================================================

    def list_violations(
        self,
        workspace_id: str | None = None,
        vertical_id: str | None = None,
        framework_id: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List all violations with optional filters.

        Args:
            workspace_id: Filter by workspace
            vertical_id: Filter by vertical
            framework_id: Filter by framework
            status: Filter by status
            severity: Filter by severity
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Dict with violations array
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if vertical_id:
            params["vertical_id"] = vertical_id
        if framework_id:
            params["framework_id"] = framework_id
        if status:
            params["status"] = status
        if severity:
            params["severity"] = severity
        return self._client.request("GET", "/api/v1/compliance/violations", params=params)

    def get_violation(self, violation_id: str) -> dict[str, Any]:
        """
        Get a specific violation.

        Args:
            violation_id: Violation ID

        Returns:
            Dict with violation details
        """
        return self._client.request("GET", f"/api/v1/compliance/violations/{violation_id}")

    def update_violation(
        self,
        violation_id: str,
        status: str,
        resolution_notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Update a violation's status.

        Args:
            violation_id: Violation ID
            status: New status (open, investigating, resolved, false_positive)
            resolution_notes: Optional resolution notes

        Returns:
            Dict with updated violation
        """
        data: dict[str, Any] = {"status": status}
        if resolution_notes:
            data["resolution_notes"] = resolution_notes
        return self._client.request(
            "PATCH", f"/api/v1/compliance/violations/{violation_id}", json=data
        )

    # ===========================================================================
    # Compliance Check
    # ===========================================================================

    def check_compliance(
        self,
        content: str,
        frameworks: list[str] | None = None,
        min_severity: str = "low",
        store_violations: bool = False,
        workspace_id: str = "default",
        source: str = "manual_check",
    ) -> dict[str, Any]:
        """
        Run compliance check on content.

        Args:
            content: Content to check
            frameworks: List of framework IDs to check against
            min_severity: Minimum severity to report (critical, high, medium, low, info)
            store_violations: Whether to store found violations
            workspace_id: Workspace ID for storing violations
            source: Source identifier for violations

        Returns:
            Dict with compliance result (compliant, score, issues)
        """
        data: dict[str, Any] = {
            "content": content,
            "min_severity": min_severity,
            "store_violations": store_violations,
            "workspace_id": workspace_id,
            "source": source,
        }
        if frameworks:
            data["frameworks"] = frameworks
        return self._client.request("POST", "/api/v1/compliance/check", json=data)

    # ===========================================================================
    # Statistics
    # ===========================================================================

    def get_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        """
        Get compliance statistics.

        Args:
            workspace_id: Filter by workspace

        Returns:
            Dict with policy and violation statistics
        """
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/compliance/stats", params=params)


class AsyncPoliciesAPI:
    """
    Asynchronous Policies API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     policies = await client.policies.list()
        ...     stats = await client.policies.get_stats()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Policy CRUD
    # ===========================================================================

    async def list(
        self,
        workspace_id: str | None = None,
        vertical_id: str | None = None,
        framework_id: str | None = None,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List policies."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if vertical_id:
            params["vertical_id"] = vertical_id
        if framework_id:
            params["framework_id"] = framework_id
        if enabled_only:
            params["enabled_only"] = "true"
        return await self._client.request("GET", "/api/v1/policies", params=params)

    async def get(self, policy_id: str) -> dict[str, Any]:
        """Get a policy."""
        return await self._client.request("GET", f"/api/v1/policies/{policy_id}")

    async def create(
        self,
        name: str,
        framework_id: str,
        vertical_id: str,
        description: str | None = None,
        workspace_id: str = "default",
        level: str = "recommended",
        enabled: bool = True,
        rules: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a policy."""
        data: dict[str, Any] = {
            "name": name,
            "framework_id": framework_id,
            "vertical_id": vertical_id,
            "workspace_id": workspace_id,
            "level": level,
            "enabled": enabled,
            "rules": rules or [],
            "metadata": metadata or {},
        }
        if description:
            data["description"] = description
        return await self._client.request("POST", "/api/v1/policies", json=data)

    async def update(
        self,
        policy_id: str,
        name: str | None = None,
        description: str | None = None,
        level: str | None = None,
        enabled: bool | None = None,
        rules: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a policy."""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if level is not None:
            data["level"] = level
        if enabled is not None:
            data["enabled"] = enabled
        if rules is not None:
            data["rules"] = rules
        if metadata is not None:
            data["metadata"] = metadata
        return await self._client.request("PATCH", f"/api/v1/policies/{policy_id}", json=data)

    async def delete(self, policy_id: str) -> dict[str, Any]:
        """Delete a policy."""
        return await self._client.request("DELETE", f"/api/v1/policies/{policy_id}")

    async def toggle(self, policy_id: str, enabled: bool | None = None) -> dict[str, Any]:
        """Toggle policy enabled status."""
        data: dict[str, Any] = {}
        if enabled is not None:
            data["enabled"] = enabled
        return await self._client.request("POST", f"/api/v1/policies/{policy_id}/toggle", json=data)

    async def get_policy_violations(
        self,
        policy_id: str,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get violations for a policy."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if severity:
            params["severity"] = severity
        return await self._client.request(
            "GET", f"/api/v1/policies/{policy_id}/violations", params=params
        )

    # ===========================================================================
    # Violations
    # ===========================================================================

    async def list_violations(
        self,
        workspace_id: str | None = None,
        vertical_id: str | None = None,
        framework_id: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List violations."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if vertical_id:
            params["vertical_id"] = vertical_id
        if framework_id:
            params["framework_id"] = framework_id
        if status:
            params["status"] = status
        if severity:
            params["severity"] = severity
        return await self._client.request("GET", "/api/v1/compliance/violations", params=params)

    async def get_violation(self, violation_id: str) -> dict[str, Any]:
        """Get a violation."""
        return await self._client.request("GET", f"/api/v1/compliance/violations/{violation_id}")

    async def update_violation(
        self,
        violation_id: str,
        status: str,
        resolution_notes: str | None = None,
    ) -> dict[str, Any]:
        """Update a violation's status."""
        data: dict[str, Any] = {"status": status}
        if resolution_notes:
            data["resolution_notes"] = resolution_notes
        return await self._client.request(
            "PATCH", f"/api/v1/compliance/violations/{violation_id}", json=data
        )

    # ===========================================================================
    # Compliance Check
    # ===========================================================================

    async def check_compliance(
        self,
        content: str,
        frameworks: list[str] | None = None,
        min_severity: str = "low",
        store_violations: bool = False,
        workspace_id: str = "default",
        source: str = "manual_check",
    ) -> dict[str, Any]:
        """Run compliance check."""
        data: dict[str, Any] = {
            "content": content,
            "min_severity": min_severity,
            "store_violations": store_violations,
            "workspace_id": workspace_id,
            "source": source,
        }
        if frameworks:
            data["frameworks"] = frameworks
        return await self._client.request("POST", "/api/v1/compliance/check", json=data)

    # ===========================================================================
    # Statistics
    # ===========================================================================

    async def get_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Get compliance statistics."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/compliance/stats", params=params)
