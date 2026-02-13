"""Tests for the RBAC API."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aragora_client.rbac import Role


class TestRBACAPI:
    """Tests for RBACAPI methods."""

    @pytest.mark.asyncio
    async def test_list_roles(self, mock_client, mock_response):
        """Test listing roles."""
        response_data = {
            "roles": [
                {
                    "id": "role-admin",
                    "name": "Admin",
                    "description": "Full access",
                    "permissions": ["admin:*"],
                    "is_system": True,
                },
                {
                    "id": "role-user",
                    "name": "User",
                    "description": "Standard user",
                    "permissions": ["debate:read"],
                    "is_system": True,
                },
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.rbac.list_roles()

        assert len(result) == 2
        assert result[0].name == "Admin"
        assert result[1].name == "User"

    @pytest.mark.asyncio
    async def test_get_role(self, mock_client, mock_response, role_response):
        """Test getting a role by ID."""
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, role_response)
        )

        result = await mock_client.rbac.get_role("role-123")

        assert isinstance(result, Role)
        assert result.id == "role-123"
        assert result.name == "Admin"
        assert "admin:*" in result.permissions

    @pytest.mark.asyncio
    async def test_create_role(self, mock_client, mock_response):
        """Test creating a custom role."""
        response_data = {
            "id": "role-custom",
            "name": "Custom Role",
            "description": "A custom role",
            "permissions": ["debate:read", "debate:write"],
            "is_system": False,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.rbac.create_role(
            name="Custom Role",
            permissions=["debate:read", "debate:write"],
            description="A custom role",
        )

        assert result.id == "role-custom"
        assert result.name == "Custom Role"
        assert result.is_system is False

    @pytest.mark.asyncio
    async def test_update_role(self, mock_client, mock_response):
        """Test updating a role."""
        response_data = {
            "id": "role-123",
            "name": "Updated Role",
            "description": "Updated description",
            "permissions": ["debate:*"],
            "is_system": False,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.rbac.update_role(
            "role-123", name="Updated Role", permissions=["debate:*"]
        )

        assert result.name == "Updated Role"

    @pytest.mark.asyncio
    async def test_delete_role(self, mock_client, mock_response):
        """Test deleting a role."""
        mock_client._client.request = AsyncMock(return_value=mock_response(204, {}))

        # Should not raise
        await mock_client.rbac.delete_role("role-123")

    @pytest.mark.asyncio
    async def test_list_permissions(self, mock_client, mock_response):
        """Test listing available permissions."""
        response_data = {
            "permissions": [
                {
                    "id": "debate:read",
                    "name": "Read Debates",
                    "category": "debate",
                },
                {
                    "id": "debate:write",
                    "name": "Write Debates",
                    "category": "debate",
                },
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.rbac.list_permissions()

        assert len(result) == 2
        assert result[0].id == "debate:read"

    @pytest.mark.asyncio
    async def test_get_user_roles(self, mock_client, mock_response):
        """Test getting roles for a user."""
        response_data = {
            "roles": [
                {
                    "id": "role-admin",
                    "name": "Admin",
                    "description": "Full access",
                    "permissions": ["admin:*"],
                    "is_system": True,
                }
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.rbac.get_user_roles("user-123")

        assert len(result) == 1
        assert result[0].name == "Admin"
