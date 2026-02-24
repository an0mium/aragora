"""Tests for Auth namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestAuthLoginLogout:
    """Tests for login, logout, and registration."""

    def test_login(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "access_token": "tok_abc",
                "refresh_token": "ref_xyz",
                "expires_in": 3600,
                "token_type": "Bearer",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.auth.login(email="user@example.com", password="secret123")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/auth/login",
                json={"email": "user@example.com", "password": "secret123"},
            )
            assert result["access_token"] == "tok_abc"
            assert result["token_type"] == "Bearer"
            client.close()

    def test_login_with_mfa(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"access_token": "tok_mfa"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.auth.login(email="user@example.com", password="secret123", mfa_code="123456")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/auth/login",
                json={
                    "email": "user@example.com",
                    "password": "secret123",
                    "mfa_code": "123456",
                },
            )
            client.close()

    def test_register(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "usr_1", "email": "new@example.com"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.auth.register(
                email="new@example.com", password="strongpass", name="New User"
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/auth/register",
                json={
                    "email": "new@example.com",
                    "password": "strongpass",
                    "name": "New User",
                },
            )
            assert result["email"] == "new@example.com"
            client.close()

    def test_logout(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.auth.logout()
            mock_request.assert_called_once_with("POST", "/api/v1/auth/logout")
            assert result["success"] is True
            client.close()


class TestAuthTokens:
    """Tests for token refresh and validation."""

    def test_refresh_token(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "access_token": "tok_new",
                "refresh_token": "ref_new",
                "expires_in": 3600,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.auth.refresh_token(refresh_token="ref_old")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/auth/refresh",
                json={"refresh_token": "ref_old"},
            )
            assert result["access_token"] == "tok_new"
            client.close()

    def test_get_current_user(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "usr_1",
                "email": "user@example.com",
                "name": "Test User",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.auth.get_current_user()
            mock_request.assert_called_once_with("GET", "/api/v1/auth/me")
            assert result["id"] == "usr_1"
            assert result["name"] == "Test User"
            client.close()


class TestAuthAPIKeys:
    """Tests for API key management."""

    def test_list_api_keys(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"keys": [{"id": "key_1", "name": "CI Key"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.auth.list_api_keys()
            mock_request.assert_called_once_with("GET", "/api/v1/auth/api-keys")
            assert len(result["keys"]) == 1
            assert result["keys"][0]["name"] == "CI Key"
            client.close()

    def test_create_api_key(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "key_2",
                "name": "Deploy Key",
                "secret": "sk_live_abc123",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.auth.create_api_key(
                name="Deploy Key",
                scopes=["read", "write"],
                expires_at="2026-12-31T00:00:00Z",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/auth/api-keys",
                json={
                    "name": "Deploy Key",
                    "scopes": ["read", "write"],
                    "expires_at": "2026-12-31T00:00:00Z",
                },
            )
            assert result["secret"] == "sk_live_abc123"
            client.close()

    def test_revoke_api_key(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.auth.revoke_api_key("key_2")
            mock_request.assert_called_once_with("DELETE", "/api/v1/auth/api-keys/key_2")
            assert result["success"] is True
            client.close()


class TestAuthSessions:
    """Tests for session management."""

    def test_list_sessions(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"sessions": [{"id": "sess_1", "ip": "10.0.0.1"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.auth.list_sessions()
            mock_request.assert_called_once_with("GET", "/api/v1/auth/sessions")
            assert result["sessions"][0]["id"] == "sess_1"
            client.close()

    def test_revoke_session(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.auth.revoke_session("sess_1")
            mock_request.assert_called_once_with("DELETE", "/api/v1/auth/sessions/sess_1")
            assert result["success"] is True
            client.close()


class TestAsyncAuth:
    """Tests for async auth methods."""

    @pytest.mark.asyncio
    async def test_login(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "access_token": "tok_async",
                "refresh_token": "ref_async",
                "expires_in": 3600,
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.auth.login(email="user@example.com", password="pass")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/auth/login",
                json={"email": "user@example.com", "password": "pass"},
            )
            assert result["access_token"] == "tok_async"
            await client.close()

    @pytest.mark.asyncio
    async def test_refresh_token(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"access_token": "tok_refreshed"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.auth.refresh_token(refresh_token="ref_old")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/auth/refresh",
                json={"refresh_token": "ref_old"},
            )
            assert result["access_token"] == "tok_refreshed"
            await client.close()

    @pytest.mark.asyncio
    async def test_create_api_key(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "key_a", "name": "Async Key"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.auth.create_api_key(name="Async Key")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/auth/api-keys",
                json={"name": "Async Key"},
            )
            assert result["name"] == "Async Key"
            await client.close()

    @pytest.mark.asyncio
    async def test_list_sessions(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"sessions": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.auth.list_sessions()
            mock_request.assert_called_once_with("GET", "/api/v1/auth/sessions")
            assert result["sessions"] == []
            await client.close()

    @pytest.mark.asyncio
    async def test_logout(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"success": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.auth.logout()
            mock_request.assert_called_once_with("POST", "/api/v1/auth/logout")
            assert result["success"] is True
            await client.close()
