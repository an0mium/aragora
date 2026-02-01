"""
SCIM 2.0 Namespace API.

Provides methods for SCIM 2.0 user and group provisioning (RFC 7643/7644):
- User CRUD operations
- Group CRUD operations
- Filtering and pagination

Endpoints:
    GET    /scim/v2/Users           - List users
    POST   /scim/v2/Users           - Create user
    GET    /scim/v2/Users/:id       - Get user
    PUT    /scim/v2/Users/:id       - Replace user
    PATCH  /scim/v2/Users/:id       - Patch user
    DELETE /scim/v2/Users/:id       - Delete user
    GET    /scim/v2/Groups          - List groups
    POST   /scim/v2/Groups          - Create group
    GET    /scim/v2/Groups/:id      - Get group
    PUT    /scim/v2/Groups/:id      - Replace group
    PATCH  /scim/v2/Groups/:id      - Patch group
    DELETE /scim/v2/Groups/:id      - Delete group
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SCIMAPI:
    """
    Synchronous SCIM 2.0 API.

    Provides RFC 7643/7644 compliant user and group provisioning.
    Typically used by IdP (Identity Provider) integrations like
    Okta, Azure AD, OneLogin, etc.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # List users with filtering
        >>> users = client.scim.list_users(filter='userName eq "john@example.com"')
        >>> # Create a user
        >>> user = client.scim.create_user({
        ...     "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
        ...     "userName": "john@example.com",
        ...     "name": {"givenName": "John", "familyName": "Doe"},
        ...     "emails": [{"value": "john@example.com", "primary": True}],
        ...     "active": True,
        ... })
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Users
    # =========================================================================

    def list_users(
        self,
        start_index: int = 1,
        count: int = 100,
        filter: str | None = None,
    ) -> dict[str, Any]:
        """
        List users with optional filtering and pagination.

        Args:
            start_index: 1-based starting index.
            count: Maximum number of results.
            filter: SCIM filter expression (e.g., 'userName eq "john@example.com"').

        Returns:
            SCIM ListResponse with Resources array.
        """
        params: dict[str, Any] = {
            "startIndex": start_index,
            "count": count,
        }
        if filter:
            params["filter"] = filter

        return self._client.request("GET", "/scim/v2/Users", params=params)

    def get_user(self, user_id: str) -> dict[str, Any]:
        """
        Get a user by ID.

        Args:
            user_id: SCIM user ID.

        Returns:
            SCIM User resource.
        """
        return self._client.request("GET", f"/scim/v2/Users/{user_id}")

    def create_user(self, user: dict[str, Any]) -> dict[str, Any]:
        """
        Create a new user.

        Args:
            user: SCIM User resource with schemas, userName, etc.

        Returns:
            Created SCIM User resource.
        """
        return self._client.request("POST", "/scim/v2/Users", json=user)

    def replace_user(self, user_id: str, user: dict[str, Any]) -> dict[str, Any]:
        """
        Replace a user (full update).

        Args:
            user_id: SCIM user ID.
            user: Complete SCIM User resource.

        Returns:
            Updated SCIM User resource.
        """
        return self._client.request("PUT", f"/scim/v2/Users/{user_id}", json=user)

    def patch_user(self, user_id: str, operations: dict[str, Any]) -> dict[str, Any]:
        """
        Partially update a user.

        Args:
            user_id: SCIM user ID.
            operations: SCIM PatchOp with Operations array.

        Returns:
            Updated SCIM User resource.
        """
        return self._client.request("PATCH", f"/scim/v2/Users/{user_id}", json=operations)

    def delete_user(self, user_id: str) -> dict[str, Any] | None:
        """
        Delete a user.

        Args:
            user_id: SCIM user ID.

        Returns:
            None (204 No Content on success).
        """
        return self._client.request("DELETE", f"/scim/v2/Users/{user_id}")

    # =========================================================================
    # Groups
    # =========================================================================

    def list_groups(
        self,
        start_index: int = 1,
        count: int = 100,
        filter: str | None = None,
    ) -> dict[str, Any]:
        """
        List groups with optional filtering and pagination.

        Args:
            start_index: 1-based starting index.
            count: Maximum number of results.
            filter: SCIM filter expression.

        Returns:
            SCIM ListResponse with Resources array.
        """
        params: dict[str, Any] = {
            "startIndex": start_index,
            "count": count,
        }
        if filter:
            params["filter"] = filter

        return self._client.request("GET", "/scim/v2/Groups", params=params)

    def get_group(self, group_id: str) -> dict[str, Any]:
        """
        Get a group by ID.

        Args:
            group_id: SCIM group ID.

        Returns:
            SCIM Group resource.
        """
        return self._client.request("GET", f"/scim/v2/Groups/{group_id}")

    def create_group(self, group: dict[str, Any]) -> dict[str, Any]:
        """
        Create a new group.

        Args:
            group: SCIM Group resource with schemas, displayName, members, etc.

        Returns:
            Created SCIM Group resource.
        """
        return self._client.request("POST", "/scim/v2/Groups", json=group)

    def replace_group(self, group_id: str, group: dict[str, Any]) -> dict[str, Any]:
        """
        Replace a group (full update).

        Args:
            group_id: SCIM group ID.
            group: Complete SCIM Group resource.

        Returns:
            Updated SCIM Group resource.
        """
        return self._client.request("PUT", f"/scim/v2/Groups/{group_id}", json=group)

    def patch_group(self, group_id: str, operations: dict[str, Any]) -> dict[str, Any]:
        """
        Partially update a group.

        Args:
            group_id: SCIM group ID.
            operations: SCIM PatchOp with Operations array.

        Returns:
            Updated SCIM Group resource.
        """
        return self._client.request("PATCH", f"/scim/v2/Groups/{group_id}", json=operations)

    def delete_group(self, group_id: str) -> dict[str, Any] | None:
        """
        Delete a group.

        Args:
            group_id: SCIM group ID.

        Returns:
            None (204 No Content on success).
        """
        return self._client.request("DELETE", f"/scim/v2/Groups/{group_id}")


class AsyncSCIMAPI:
    """Asynchronous SCIM 2.0 API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Users
    # =========================================================================

    async def list_users(
        self,
        start_index: int = 1,
        count: int = 100,
        filter: str | None = None,
    ) -> dict[str, Any]:
        """List users with optional filtering and pagination."""
        params: dict[str, Any] = {
            "startIndex": start_index,
            "count": count,
        }
        if filter:
            params["filter"] = filter

        return await self._client.request("GET", "/scim/v2/Users", params=params)

    async def get_user(self, user_id: str) -> dict[str, Any]:
        """Get a user by ID."""
        return await self._client.request("GET", f"/scim/v2/Users/{user_id}")

    async def create_user(self, user: dict[str, Any]) -> dict[str, Any]:
        """Create a new user."""
        return await self._client.request("POST", "/scim/v2/Users", json=user)

    async def replace_user(self, user_id: str, user: dict[str, Any]) -> dict[str, Any]:
        """Replace a user (full update)."""
        return await self._client.request("PUT", f"/scim/v2/Users/{user_id}", json=user)

    async def patch_user(self, user_id: str, operations: dict[str, Any]) -> dict[str, Any]:
        """Partially update a user."""
        return await self._client.request("PATCH", f"/scim/v2/Users/{user_id}", json=operations)

    async def delete_user(self, user_id: str) -> dict[str, Any] | None:
        """Delete a user."""
        return await self._client.request("DELETE", f"/scim/v2/Users/{user_id}")

    # =========================================================================
    # Groups
    # =========================================================================

    async def list_groups(
        self,
        start_index: int = 1,
        count: int = 100,
        filter: str | None = None,
    ) -> dict[str, Any]:
        """List groups with optional filtering and pagination."""
        params: dict[str, Any] = {
            "startIndex": start_index,
            "count": count,
        }
        if filter:
            params["filter"] = filter

        return await self._client.request("GET", "/scim/v2/Groups", params=params)

    async def get_group(self, group_id: str) -> dict[str, Any]:
        """Get a group by ID."""
        return await self._client.request("GET", f"/scim/v2/Groups/{group_id}")

    async def create_group(self, group: dict[str, Any]) -> dict[str, Any]:
        """Create a new group."""
        return await self._client.request("POST", "/scim/v2/Groups", json=group)

    async def replace_group(self, group_id: str, group: dict[str, Any]) -> dict[str, Any]:
        """Replace a group (full update)."""
        return await self._client.request("PUT", f"/scim/v2/Groups/{group_id}", json=group)

    async def patch_group(self, group_id: str, operations: dict[str, Any]) -> dict[str, Any]:
        """Partially update a group."""
        return await self._client.request("PATCH", f"/scim/v2/Groups/{group_id}", json=operations)

    async def delete_group(self, group_id: str) -> dict[str, Any] | None:
        """Delete a group."""
        return await self._client.request("DELETE", f"/scim/v2/Groups/{group_id}")
