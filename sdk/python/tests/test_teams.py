"""Tests for Teams namespace API."""

from __future__ import annotations

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestTeamsList:
    """Tests for listing teams."""

    def test_list_teams_default(self, client: AragoraClient, mock_request) -> None:
        """List teams with default parameters."""
        mock_request.return_value = [{"team_id": "team_1", "name": "Alpha Team"}]

        result = client.teams.list()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/teams",
            params={"limit": 50, "offset": 0},
        )
        assert result[0]["team_id"] == "team_1"

    def test_list_teams_with_pagination(self, client: AragoraClient, mock_request) -> None:
        """List teams with custom pagination."""
        mock_request.return_value = []

        client.teams.list(limit=10, offset=20)

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["params"]["limit"] == 10
        assert call_kwargs["params"]["offset"] == 20

    def test_list_teams_filtered_by_organization(self, client: AragoraClient, mock_request) -> None:
        """List teams filtered by organization."""
        mock_request.return_value = []

        client.teams.list(organization_id="org_123")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["params"]["organization_id"] == "org_123"


class TestTeamsGet:
    """Tests for getting team details."""

    def test_get_team(self, client: AragoraClient, mock_request) -> None:
        """Get team details."""
        mock_request.return_value = {
            "team_id": "team_123",
            "name": "Engineering",
            "member_count": 15,
        }

        result = client.teams.get("team_123")

        mock_request.assert_called_once_with("GET", "/api/v1/teams/team_123")
        assert result["name"] == "Engineering"
        assert result["member_count"] == 15


class TestTeamsCreate:
    """Tests for team creation."""

    def test_create_team_minimal(self, client: AragoraClient, mock_request) -> None:
        """Create a team with only required fields."""
        mock_request.return_value = {"team_id": "team_new", "name": "New Team"}

        result = client.teams.create("New Team")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/teams",
            json={"name": "New Team"},
        )
        assert result["team_id"] == "team_new"

    def test_create_team_with_description(self, client: AragoraClient, mock_request) -> None:
        """Create a team with a description."""
        mock_request.return_value = {"team_id": "team_desc"}

        client.teams.create("Dev Team", description="Development team")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["name"] == "Dev Team"
        assert call_kwargs["json"]["description"] == "Development team"

    def test_create_team_full_options(self, client: AragoraClient, mock_request) -> None:
        """Create a team with all options."""
        mock_request.return_value = {"team_id": "team_full"}

        client.teams.create(
            name="Full Team",
            description="A fully configured team",
            organization_id="org_456",
            settings={"private": True, "approval_required": True},
        )

        call_kwargs = mock_request.call_args[1]
        call_json = call_kwargs["json"]
        assert call_json["name"] == "Full Team"
        assert call_json["description"] == "A fully configured team"
        assert call_json["organization_id"] == "org_456"
        assert call_json["settings"]["private"] is True


class TestTeamsUpdate:
    """Tests for team updates."""

    def test_update_team_name(self, client: AragoraClient, mock_request) -> None:
        """Update team name."""
        mock_request.return_value = {"team_id": "team_123", "name": "Updated Name"}

        result = client.teams.update("team_123", name="Updated Name")

        mock_request.assert_called_once_with(
            "PATCH",
            "/api/v1/teams/team_123",
            json={"name": "Updated Name"},
        )
        assert result["name"] == "Updated Name"

    def test_update_team_multiple_fields(self, client: AragoraClient, mock_request) -> None:
        """Update multiple team fields."""
        mock_request.return_value = {"team_id": "team_123"}

        client.teams.update(
            "team_123",
            name="New Name",
            description="New description",
            settings={"notifications": False},
        )

        call_kwargs = mock_request.call_args[1]
        call_json = call_kwargs["json"]
        assert call_json["name"] == "New Name"
        assert call_json["description"] == "New description"
        assert call_json["settings"]["notifications"] is False


class TestTeamsDelete:
    """Tests for team deletion."""

    def test_delete_team(self, client: AragoraClient, mock_request) -> None:
        """Delete a team."""
        mock_request.return_value = {"deleted": True}

        result = client.teams.delete("team_123")

        mock_request.assert_called_once_with("DELETE", "/api/v1/teams/team_123")
        assert result["deleted"] is True


class TestTeamsMembers:
    """Tests for team member management."""

    def test_list_members(self, client: AragoraClient, mock_request) -> None:
        """List team members."""
        mock_request.return_value = [
            {"user_id": "user_1", "role": "admin"},
            {"user_id": "user_2", "role": "member"},
        ]

        result = client.teams.list_members("team_123")

        mock_request.assert_called_once_with("GET", "/api/v1/teams/team_123/members")
        assert len(result) == 2
        assert result[0]["role"] == "admin"

    def test_add_member_default_role(self, client: AragoraClient, mock_request) -> None:
        """Add a member with default role."""
        mock_request.return_value = {"user_id": "user_new", "role": "member"}

        result = client.teams.add_member("team_123", "user_new")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/teams/team_123/members",
            json={"user_id": "user_new", "role": "member"},
        )
        assert result["role"] == "member"

    def test_add_member_as_admin(self, client: AragoraClient, mock_request) -> None:
        """Add a member with admin role."""
        mock_request.return_value = {"user_id": "user_admin", "role": "admin"}

        result = client.teams.add_member("team_123", "user_admin", role="admin")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["role"] == "admin"
        assert result["role"] == "admin"

    def test_update_member_role(self, client: AragoraClient, mock_request) -> None:
        """Update a member's role."""
        mock_request.return_value = {"user_id": "user_123", "role": "admin"}

        result = client.teams.update_member("team_123", "user_123", "admin")

        mock_request.assert_called_once_with(
            "PATCH",
            "/api/v1/teams/team_123/members/user_123",
            json={"role": "admin"},
        )
        assert result["role"] == "admin"

    def test_remove_member(self, client: AragoraClient, mock_request) -> None:
        """Remove a member from the team."""
        mock_request.return_value = {"removed": True}

        result = client.teams.remove_member("team_123", "user_456")

        mock_request.assert_called_once_with(
            "DELETE",
            "/api/v1/teams/team_123/members/user_456",
        )
        assert result["removed"] is True


class TestTeamsStats:
    """Tests for team statistics."""

    def test_get_stats(self, client: AragoraClient, mock_request) -> None:
        """Get team statistics."""
        mock_request.return_value = {
            "team_id": "team_123",
            "member_count": 10,
            "debates_count": 42,
            "active_projects": 5,
        }

        result = client.teams.get_stats("team_123")

        mock_request.assert_called_once_with("GET", "/api/v1/teams/team_123/stats")
        assert result["member_count"] == 10
        assert result["debates_count"] == 42


class TestAsyncTeams:
    """Tests for async teams API."""

    @pytest.mark.asyncio
    async def test_async_list_teams(self, mock_async_request) -> None:
        """List teams asynchronously."""
        mock_async_request.return_value = [{"team_id": "team_1", "name": "Team Alpha"}]

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.teams.list()

            assert result[0]["team_id"] == "team_1"

    @pytest.mark.asyncio
    async def test_async_get_team(self, mock_async_request) -> None:
        """Get team details asynchronously."""
        mock_async_request.return_value = {"team_id": "team_123", "name": "Engineering"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.teams.get("team_123")

            assert result["name"] == "Engineering"

    @pytest.mark.asyncio
    async def test_async_create_team(self, mock_async_request) -> None:
        """Create a team asynchronously."""
        mock_async_request.return_value = {"team_id": "team_async"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.teams.create(
                name="Async Team",
                description="Created asynchronously",
            )

            call_kwargs = mock_async_request.call_args[1]
            assert call_kwargs["json"]["name"] == "Async Team"
            assert result["team_id"] == "team_async"

    @pytest.mark.asyncio
    async def test_async_update_team(self, mock_async_request) -> None:
        """Update a team asynchronously."""
        mock_async_request.return_value = {"team_id": "team_123", "name": "Updated"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.teams.update("team_123", name="Updated")

            assert result["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_async_delete_team(self, mock_async_request) -> None:
        """Delete a team asynchronously."""
        mock_async_request.return_value = {"deleted": True}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.teams.delete("team_123")

            assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_async_list_members(self, mock_async_request) -> None:
        """List team members asynchronously."""
        mock_async_request.return_value = [{"user_id": "u1"}, {"user_id": "u2"}]

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.teams.list_members("team_123")

            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_async_add_member(self, mock_async_request) -> None:
        """Add a team member asynchronously."""
        mock_async_request.return_value = {"user_id": "user_new", "role": "member"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.teams.add_member("team_123", "user_new", role="admin")

            assert result["user_id"] == "user_new"
            call_kwargs = mock_async_request.call_args[1]
            assert call_kwargs["json"]["role"] == "admin"

    @pytest.mark.asyncio
    async def test_async_update_member(self, mock_async_request) -> None:
        """Update a team member asynchronously."""
        mock_async_request.return_value = {"user_id": "user_123", "role": "admin"}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.teams.update_member("team_123", "user_123", "admin")

            assert result["role"] == "admin"

    @pytest.mark.asyncio
    async def test_async_remove_member(self, mock_async_request) -> None:
        """Remove a team member asynchronously."""
        mock_async_request.return_value = {"removed": True}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.teams.remove_member("team_123", "user_456")

            assert result["removed"] is True

    @pytest.mark.asyncio
    async def test_async_get_stats(self, mock_async_request) -> None:
        """Get team stats asynchronously."""
        mock_async_request.return_value = {"member_count": 5, "debates_count": 20}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.teams.get_stats("team_123")

            assert result["member_count"] == 5
