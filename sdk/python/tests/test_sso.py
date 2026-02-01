"""Tests for SSO (Single Sign-On) SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


class TestSSOAPI:
    """Test synchronous SSOAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.sso import SSOAPI

        api = SSOAPI(mock_client)
        assert api._client is mock_client

    def test_get_status(self, mock_client: MagicMock) -> None:
        """Test get_status calls correct endpoint."""
        from aragora.namespaces.sso import SSOAPI

        mock_client.request.return_value = {
            "enabled": True,
            "configured": True,
            "provider_type": "oidc",
            "idp_url": "https://login.example.com",
            "entity_id": "aragora-sp",
        }

        api = SSOAPI(mock_client)
        result = api.get_status()

        mock_client.request.assert_called_once_with("GET", "/api/v2/sso/status")
        assert result["enabled"] is True
        assert result["provider_type"] == "oidc"

    def test_login(self, mock_client: MagicMock) -> None:
        """Test login calls correct endpoint."""
        from aragora.namespaces.sso import SSOAPI

        mock_client.request.return_value = {
            "redirect_url": "https://login.okta.com/oauth/authorize?...",
            "state": "abc123",
            "nonce": "xyz789",
            "provider": "okta",
        }

        api = SSOAPI(mock_client)
        result = api.login(provider="okta", redirect_uri="https://app.example.com/callback")

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v2/sso/login",
            json={
                "provider": "okta",
                "redirect_uri": "https://app.example.com/callback",
            },
        )
        assert result["redirect_url"].startswith("https://login.okta.com")
        assert result["state"] == "abc123"
        assert result["provider"] == "okta"

    def test_callback(self, mock_client: MagicMock) -> None:
        """Test callback calls correct endpoint."""
        from aragora.namespaces.sso import SSOAPI

        mock_client.request.return_value = {
            "success": True,
            "user": {
                "id": "user-123",
                "email": "user@example.com",
                "name": "Test User",
                "groups": ["admins", "developers"],
                "roles": ["admin"],
            },
            "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
            "refresh_token": "refresh-token-abc",
            "expires_in": 3600,
        }

        api = SSOAPI(mock_client)
        result = api.callback(provider="okta", code="auth-code-123", state="abc123")

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v2/sso/callback",
            json={
                "provider": "okta",
                "code": "auth-code-123",
                "state": "abc123",
            },
        )
        assert result["success"] is True
        assert result["user"]["email"] == "user@example.com"
        assert result["expires_in"] == 3600

    def test_callback_error(self, mock_client: MagicMock) -> None:
        """Test callback with authentication error."""
        from aragora.namespaces.sso import SSOAPI

        mock_client.request.return_value = {
            "success": False,
            "error": "invalid_grant",
            "error_description": "Authorization code expired",
        }

        api = SSOAPI(mock_client)
        result = api.callback(provider="okta", code="expired-code", state="abc123")

        assert result["success"] is False
        assert result["error"] == "invalid_grant"

    def test_logout(self, mock_client: MagicMock) -> None:
        """Test logout calls correct endpoint."""
        from aragora.namespaces.sso import SSOAPI

        mock_client.request.return_value = {
            "success": True,
            "redirect_url": "https://login.okta.com/logout",
            "message": "Session terminated",
        }

        api = SSOAPI(mock_client)
        result = api.logout(provider="okta")

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v2/sso/logout",
            json={"provider": "okta"},
        )
        assert result["success"] is True
        assert "logout" in result["redirect_url"]

    def test_get_metadata(self, mock_client: MagicMock) -> None:
        """Test get_metadata calls correct endpoint."""
        from aragora.namespaces.sso import SSOAPI

        mock_client.request.return_value = {
            "entity_id": "https://api.aragora.ai/saml/sp",
            "acs_url": "https://api.aragora.ai/saml/acs",
            "slo_url": "https://api.aragora.ai/saml/slo",
            "certificate": "-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----",
            "metadata_xml": '<?xml version="1.0"?><EntityDescriptor>...</EntityDescriptor>',
        }

        api = SSOAPI(mock_client)
        result = api.get_metadata("okta")

        mock_client.request.assert_called_once_with("GET", "/api/v2/sso/okta/metadata")
        assert result["entity_id"] == "https://api.aragora.ai/saml/sp"
        assert result["acs_url"] == "https://api.aragora.ai/saml/acs"
        assert "CERTIFICATE" in result["certificate"]

    def test_get_metadata_different_providers(self, mock_client: MagicMock) -> None:
        """Test get_metadata with different provider names."""
        from aragora.namespaces.sso import SSOAPI

        mock_client.request.return_value = {"entity_id": "test"}

        api = SSOAPI(mock_client)

        # Test with azure-ad
        api.get_metadata("azure-ad")
        mock_client.request.assert_called_with("GET", "/api/v2/sso/azure-ad/metadata")

        # Test with google
        api.get_metadata("google")
        mock_client.request.assert_called_with("GET", "/api/v2/sso/google/metadata")


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    from unittest.mock import AsyncMock

    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestAsyncSSOAPI:
    """Test asynchronous AsyncSSOAPI."""

    @pytest.mark.asyncio
    async def test_init(self, mock_async_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.sso import AsyncSSOAPI

        api = AsyncSSOAPI(mock_async_client)
        assert api._client is mock_async_client

    @pytest.mark.asyncio
    async def test_get_status(self, mock_async_client: MagicMock) -> None:
        """Test get_status calls correct endpoint."""
        from aragora.namespaces.sso import AsyncSSOAPI

        mock_async_client.request.return_value = {
            "enabled": True,
            "configured": True,
            "provider_type": "saml",
        }

        api = AsyncSSOAPI(mock_async_client)
        result = await api.get_status()

        mock_async_client.request.assert_called_once_with("GET", "/api/v2/sso/status")
        assert result["enabled"] is True
        assert result["provider_type"] == "saml"

    @pytest.mark.asyncio
    async def test_login(self, mock_async_client: MagicMock) -> None:
        """Test login calls correct endpoint."""
        from aragora.namespaces.sso import AsyncSSOAPI

        mock_async_client.request.return_value = {
            "redirect_url": "https://login.azure.com/oauth/authorize",
            "state": "state-token",
            "nonce": "nonce-value",
            "provider": "azure-ad",
        }

        api = AsyncSSOAPI(mock_async_client)
        result = await api.login(
            provider="azure-ad", redirect_uri="https://app.example.com/auth/callback"
        )

        mock_async_client.request.assert_called_once_with(
            "POST",
            "/api/v2/sso/login",
            json={
                "provider": "azure-ad",
                "redirect_uri": "https://app.example.com/auth/callback",
            },
        )
        assert result["provider"] == "azure-ad"
        assert result["state"] == "state-token"

    @pytest.mark.asyncio
    async def test_callback(self, mock_async_client: MagicMock) -> None:
        """Test callback calls correct endpoint."""
        from aragora.namespaces.sso import AsyncSSOAPI

        mock_async_client.request.return_value = {
            "success": True,
            "user": {"id": "user-456", "email": "test@company.com"},
            "token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 7200,
        }

        api = AsyncSSOAPI(mock_async_client)
        result = await api.callback(provider="azure-ad", code="auth-code-xyz", state="state-token")

        mock_async_client.request.assert_called_once_with(
            "POST",
            "/api/v2/sso/callback",
            json={
                "provider": "azure-ad",
                "code": "auth-code-xyz",
                "state": "state-token",
            },
        )
        assert result["success"] is True
        assert result["user"]["id"] == "user-456"
        assert result["expires_in"] == 7200

    @pytest.mark.asyncio
    async def test_logout(self, mock_async_client: MagicMock) -> None:
        """Test logout calls correct endpoint."""
        from aragora.namespaces.sso import AsyncSSOAPI

        mock_async_client.request.return_value = {
            "success": True,
            "message": "Logged out successfully",
        }

        api = AsyncSSOAPI(mock_async_client)
        result = await api.logout(provider="google")

        mock_async_client.request.assert_called_once_with(
            "POST",
            "/api/v2/sso/logout",
            json={"provider": "google"},
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_get_metadata(self, mock_async_client: MagicMock) -> None:
        """Test get_metadata calls correct endpoint."""
        from aragora.namespaces.sso import AsyncSSOAPI

        mock_async_client.request.return_value = {
            "entity_id": "https://api.aragora.ai/saml/sp",
            "acs_url": "https://api.aragora.ai/saml/acs",
            "slo_url": "https://api.aragora.ai/saml/slo",
            "certificate": "-----BEGIN CERTIFICATE-----...",
        }

        api = AsyncSSOAPI(mock_async_client)
        result = await api.get_metadata("onelogin")

        mock_async_client.request.assert_called_once_with("GET", "/api/v2/sso/onelogin/metadata")
        assert result["entity_id"] == "https://api.aragora.ai/saml/sp"
        assert result["slo_url"] == "https://api.aragora.ai/saml/slo"
