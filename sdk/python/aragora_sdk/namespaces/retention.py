"""
Retention Namespace API

Provides access to data retention policies and management.
Essential for compliance and data lifecycle management.

Features:
- List retention policies
- Execute policies manually
- View expiring data
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

RetentionAction = Literal["archive", "delete", "anonymize"]


class RetentionAPI:
    """
    Synchronous Retention API.

    Provides methods for retention policy management:
    - List retention policies
    - Execute policies manually
    - View expiring data

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> policies = client.retention.list_policies()
        >>> expiring = client.retention.get_expiring(days=7)
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_policies(self) -> dict[str, Any]:
        """
        List retention policies.

        Returns:
            Dict with policies list containing:
            - id: Policy ID
            - name: Policy name
            - description: Policy description
            - retention_days: Days to retain
            - data_types: Affected data types
            - action: Action (archive/delete/anonymize)
            - enabled: Whether enabled
            - schedule: Execution schedule
        """
        return self._client.request("GET", "/api/v1/retention/policies")

    def get_expiring(
        self,
        days: int | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Get data items that are expiring soon.

        Args:
            days: Number of days to look ahead
            limit: Maximum items to return

        Returns:
            Dict with:
            - items: List of expiring items
            - total: Total count
        """
        params: dict[str, Any] = {}
        if days:
            params["days"] = days
        if limit:
            params["limit"] = limit
        return self._client.request(
            "GET", "/api/v1/retention/expiring", params=params if params else None
        )


class AsyncRetentionAPI:
    """
    Asynchronous Retention API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     policies = await client.retention.list_policies()
        ...     result = await client.retention.execute_policy("policy-123", dry_run=True)
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_policies(self) -> dict[str, Any]:
        """List retention policies."""
        return await self._client.request("GET", "/api/v1/retention/policies")

    async def get_expiring(
        self,
        days: int | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Get data items that are expiring soon."""
        params: dict[str, Any] = {}
        if days:
            params["days"] = days
        if limit:
            params["limit"] = limit
        return await self._client.request(
            "GET", "/api/v1/retention/expiring", params=params if params else None
        )
