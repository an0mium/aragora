"""Tests for the Tenancy API."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aragora_client.tenancy import Tenant


class TestTenancyAPI:
    """Tests for TenancyAPI methods."""

    @pytest.mark.asyncio
    async def test_list_tenants(self, mock_client, mock_response, tenant_response):
        """Test listing tenants."""
        response_data = {"tenants": [tenant_response]}
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.tenants.list()

        assert len(result) == 1
        assert result[0].id == "tenant-123"
        assert result[0].name == "Test Tenant"

    @pytest.mark.asyncio
    async def test_get_tenant(self, mock_client, mock_response, tenant_response):
        """Test getting a tenant by ID."""
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, tenant_response)
        )

        result = await mock_client.tenants.get("tenant-123")

        assert isinstance(result, Tenant)
        assert result.id == "tenant-123"
        assert result.slug == "test-tenant"

    @pytest.mark.asyncio
    async def test_create_tenant(self, mock_client, mock_response, tenant_response):
        """Test creating a tenant."""
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, tenant_response)
        )

        result = await mock_client.tenants.create(
            name="Test Tenant", slug="test-tenant"
        )

        assert result.id == "tenant-123"
        assert result.name == "Test Tenant"

    @pytest.mark.asyncio
    async def test_update_tenant(self, mock_client, mock_response):
        """Test updating a tenant."""
        updated_response = {
            "id": "tenant-123",
            "name": "Updated Tenant",
            "slug": "updated-tenant",
            "owner_id": "user-123",
            "created_at": "2026-01-01T00:00:00Z",
            "status": "active",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, updated_response)
        )

        result = await mock_client.tenants.update("tenant-123", name="Updated Tenant")

        assert result.name == "Updated Tenant"

    @pytest.mark.asyncio
    async def test_delete_tenant(self, mock_client, mock_response):
        """Test deleting a tenant."""
        mock_client._client.request = AsyncMock(return_value=mock_response(204, {}))

        # Should not raise
        await mock_client.tenants.delete("tenant-123")

    @pytest.mark.asyncio
    async def test_list_members(self, mock_client, mock_response):
        """Test listing tenant members."""
        response_data = {
            "members": [
                {
                    "id": "member-1",
                    "user_id": "user-123",
                    "tenant_id": "tenant-123",
                    "role": "admin",
                    "joined_at": "2026-01-01T00:00:00Z",
                }
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.tenants.list_members("tenant-123")

        assert len(result) == 1
        assert result[0].user_id == "user-123"
        assert result[0].role == "admin"

    @pytest.mark.asyncio
    async def test_add_member(self, mock_client, mock_response):
        """Test adding a member to a tenant."""
        response_data = {
            "id": "member-new",
            "user_id": "user-456",
            "tenant_id": "tenant-123",
            "role": "member",
            "joined_at": "2026-01-01T00:00:00Z",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.tenants.add_member(
            "tenant-123", user_id="user-456", role="member"
        )

        assert result.user_id == "user-456"
        assert result.role == "member"

    @pytest.mark.asyncio
    async def test_remove_member(self, mock_client, mock_response):
        """Test removing a member from a tenant."""
        mock_client._client.request = AsyncMock(return_value=mock_response(204, {}))

        # Should not raise
        await mock_client.tenants.remove_member("tenant-123", "user-456")
