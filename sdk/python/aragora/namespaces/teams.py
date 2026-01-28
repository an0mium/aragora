"""
Teams namespace for team management.

Provides API access to manage teams, team memberships,
and team-level permissions.
"""

from __future__ import annotations

from typing import Any


class TeamsAPI:
    """Synchronous teams API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        organization_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List teams.

        Args:
            limit: Maximum number of teams to return
            offset: Number of teams to skip
            organization_id: Filter by organization

        Returns:
            List of team records
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if organization_id:
            params["organization_id"] = organization_id

        return self._client._request("GET", "/api/v1/teams", params=params)

    def get(self, team_id: str) -> dict[str, Any]:
        """
        Get team details.

        Args:
            team_id: Team identifier

        Returns:
            Team details
        """
        return self._client._request("GET", f"/api/v1/teams/{team_id}")

    def create(
        self,
        name: str,
        description: str | None = None,
        organization_id: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new team.

        Args:
            name: Team name
            description: Team description
            organization_id: Parent organization
            settings: Team settings

        Returns:
            Created team record
        """
        data: dict[str, Any] = {"name": name}
        if description:
            data["description"] = description
        if organization_id:
            data["organization_id"] = organization_id
        if settings:
            data["settings"] = settings

        return self._client._request("POST", "/api/v1/teams", json=data)

    def update(
        self,
        team_id: str,
        name: str | None = None,
        description: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update a team.

        Args:
            team_id: Team identifier
            name: New team name
            description: New description
            settings: Updated settings

        Returns:
            Updated team record
        """
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if settings is not None:
            data["settings"] = settings

        return self._client._request("PATCH", f"/api/v1/teams/{team_id}", json=data)

    def delete(self, team_id: str) -> dict[str, Any]:
        """
        Delete a team.

        Args:
            team_id: Team identifier

        Returns:
            Deletion confirmation
        """
        return self._client._request("DELETE", f"/api/v1/teams/{team_id}")

    def list_members(self, team_id: str) -> list[dict[str, Any]]:
        """
        List team members.

        Args:
            team_id: Team identifier

        Returns:
            List of team members
        """
        return self._client._request("GET", f"/api/v1/teams/{team_id}/members")

    def add_member(self, team_id: str, user_id: str, role: str = "member") -> dict[str, Any]:
        """
        Add a member to the team.

        Args:
            team_id: Team identifier
            user_id: User to add
            role: Member role (admin, member)

        Returns:
            Membership record
        """
        return self._client._request(
            "POST",
            f"/api/v1/teams/{team_id}/members",
            json={"user_id": user_id, "role": role},
        )

    def update_member(self, team_id: str, user_id: str, role: str) -> dict[str, Any]:
        """
        Update a team member's role.

        Args:
            team_id: Team identifier
            user_id: User to update
            role: New role

        Returns:
            Updated membership record
        """
        return self._client._request(
            "PATCH",
            f"/api/v1/teams/{team_id}/members/{user_id}",
            json={"role": role},
        )

    def remove_member(self, team_id: str, user_id: str) -> dict[str, Any]:
        """
        Remove a member from the team.

        Args:
            team_id: Team identifier
            user_id: User to remove

        Returns:
            Removal confirmation
        """
        return self._client._request("DELETE", f"/api/v1/teams/{team_id}/members/{user_id}")

    def get_stats(self, team_id: str) -> dict[str, Any]:
        """
        Get team statistics.

        Args:
            team_id: Team identifier

        Returns:
            Team statistics
        """
        return self._client._request("GET", f"/api/v1/teams/{team_id}/stats")


class AsyncTeamsAPI:
    """Asynchronous teams API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        organization_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List teams."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if organization_id:
            params["organization_id"] = organization_id

        return await self._client._request("GET", "/api/v1/teams", params=params)

    async def get(self, team_id: str) -> dict[str, Any]:
        """Get team details."""
        return await self._client._request("GET", f"/api/v1/teams/{team_id}")

    async def create(
        self,
        name: str,
        description: str | None = None,
        organization_id: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new team."""
        data: dict[str, Any] = {"name": name}
        if description:
            data["description"] = description
        if organization_id:
            data["organization_id"] = organization_id
        if settings:
            data["settings"] = settings

        return await self._client._request("POST", "/api/v1/teams", json=data)

    async def update(
        self,
        team_id: str,
        name: str | None = None,
        description: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a team."""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if settings is not None:
            data["settings"] = settings

        return await self._client._request("PATCH", f"/api/v1/teams/{team_id}", json=data)

    async def delete(self, team_id: str) -> dict[str, Any]:
        """Delete a team."""
        return await self._client._request("DELETE", f"/api/v1/teams/{team_id}")

    async def list_members(self, team_id: str) -> list[dict[str, Any]]:
        """List team members."""
        return await self._client._request("GET", f"/api/v1/teams/{team_id}/members")

    async def add_member(self, team_id: str, user_id: str, role: str = "member") -> dict[str, Any]:
        """Add a member to the team."""
        return await self._client._request(
            "POST",
            f"/api/v1/teams/{team_id}/members",
            json={"user_id": user_id, "role": role},
        )

    async def update_member(self, team_id: str, user_id: str, role: str) -> dict[str, Any]:
        """Update a team member's role."""
        return await self._client._request(
            "PATCH",
            f"/api/v1/teams/{team_id}/members/{user_id}",
            json={"role": role},
        )

    async def remove_member(self, team_id: str, user_id: str) -> dict[str, Any]:
        """Remove a member from the team."""
        return await self._client._request("DELETE", f"/api/v1/teams/{team_id}/members/{user_id}")

    async def get_stats(self, team_id: str) -> dict[str, Any]:
        """Get team statistics."""
        return await self._client._request("GET", f"/api/v1/teams/{team_id}/stats")
