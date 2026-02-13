"""Tests for the Auth API."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aragora_client.auth import AuthToken, User


class TestAuthAPI:
    """Tests for AuthAPI methods."""

    @pytest.mark.asyncio
    async def test_login_success(self, mock_client, mock_response, auth_token_response):
        """Test successful login."""
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, auth_token_response)
        )

        result = await mock_client.auth.login("test@example.com", "password123")

        assert isinstance(result, AuthToken)
        assert result.access_token == "test-access-token-123"
        assert result.refresh_token == "test-refresh-token-456"
        assert result.expires_in == 3600

    @pytest.mark.asyncio
    async def test_register_success(self, mock_client, mock_response):
        """Test successful registration."""
        response_data = {"id": "user-123", "email": "new@example.com"}
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.auth.register(
            "new@example.com", "password123", name="New User"
        )

        assert result["id"] == "user-123"
        assert result["email"] == "new@example.com"

    @pytest.mark.asyncio
    async def test_get_current_user(self, mock_client, mock_response, user_response):
        """Test getting current user profile."""
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, user_response)
        )

        result = await mock_client.auth.get_current_user()

        assert isinstance(result, User)
        assert result.id == "user-123"
        assert result.email == "test@example.com"
        assert result.email_verified is True

    @pytest.mark.asyncio
    async def test_refresh_token(self, mock_client, mock_response, auth_token_response):
        """Test token refresh."""
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, auth_token_response)
        )

        result = await mock_client.auth.refresh_token("old-refresh-token")

        assert isinstance(result, AuthToken)
        assert result.access_token == "test-access-token-123"

    @pytest.mark.asyncio
    async def test_logout(self, mock_client, mock_response):
        """Test logout."""
        mock_client._client.request = AsyncMock(return_value=mock_response(200, {}))

        # Should not raise
        await mock_client.auth.logout()

    @pytest.mark.asyncio
    async def test_logout_all(self, mock_client, mock_response):
        """Test logout from all sessions."""
        response_data = {"sessions_invalidated": 5}
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.auth.logout_all()

        assert result["sessions_invalidated"] == 5

    @pytest.mark.asyncio
    async def test_verify_email(self, mock_client, mock_response):
        """Test email verification."""
        response_data = {"verified": True}
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.auth.verify_email("verification-token")

        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_change_password(self, mock_client, mock_response):
        """Test password change."""
        mock_client._client.request = AsyncMock(return_value=mock_response(200, {}))

        # Should not raise (returns None)
        await mock_client.auth.change_password("old-pass", "new-pass")

    @pytest.mark.asyncio
    async def test_request_password_reset(self, mock_client, mock_response):
        """Test forgot password request."""
        mock_client._client.request = AsyncMock(return_value=mock_response(200, {}))

        # Should not raise (returns None)
        await mock_client.auth.request_password_reset("test@example.com")

    @pytest.mark.asyncio
    async def test_list_api_keys(self, mock_client, mock_response):
        """Test listing API keys."""
        response_data = {
            "api_keys": [
                {
                    "id": "key-1",
                    "name": "Dev Key",
                    "key_prefix": "sk-dev",
                    "created_at": "2026-01-01T00:00:00Z",
                    "scopes": ["debate:read"],
                }
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.auth.list_api_keys()

        assert len(result) == 1
        assert result[0].id == "key-1"
        assert result[0].name == "Dev Key"

    @pytest.mark.asyncio
    async def test_create_api_key(self, mock_client, mock_response):
        """Test creating an API key."""
        response_data = {
            "key": "sk-test-full-key-123",
            "api_key": {
                "id": "key-new",
                "name": "New Key",
                "key_prefix": "sk-test",
                "created_at": "2026-01-01T00:00:00Z",
                "scopes": ["debate:*"],
            },
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.auth.create_api_key("New Key", scopes=["debate:*"])

        assert result.key == "sk-test-full-key-123"
        assert result.api_key.name == "New Key"

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, mock_client, mock_response):
        """Test revoking an API key."""
        mock_client._client.request = AsyncMock(return_value=mock_response(200, {}))

        # Should not raise
        await mock_client.auth.revoke_api_key("key-123")

    @pytest.mark.asyncio
    async def test_list_sessions(self, mock_client, mock_response):
        """Test listing user sessions."""
        response_data = {
            "sessions": [
                {
                    "id": "session-1",
                    "user_agent": "Mozilla/5.0",
                    "ip_address": "127.0.0.1",
                    "created_at": "2026-01-01T00:00:00Z",
                    "last_active_at": "2026-01-01T12:00:00Z",
                    "is_current": True,
                }
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.auth.list_sessions()

        assert len(result) == 1
        assert result[0].id == "session-1"
        assert result[0].is_current is True

    @pytest.mark.asyncio
    async def test_revoke_session(self, mock_client, mock_response):
        """Test revoking a session."""
        mock_client._client.request = AsyncMock(return_value=mock_response(200, {}))

        # Should not raise
        await mock_client.auth.revoke_session("session-123")
