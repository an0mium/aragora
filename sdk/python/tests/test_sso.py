"""Tests for SSO namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestSSOStatus:
    """Tests for SSO status retrieval."""

    def test_get_status(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "enabled": True,
                "configured": True,
                "provider_type": "oidc",
                "idp_url": "https://login.example.com",
                "entity_id": "aragora-sp",
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.sso.get_status()
            mock_request.assert_called_once_with("GET", "/api/v2/sso/status")
            assert result["enabled"] is True
            assert result["provider_type"] == "oidc"
            client.close()


class TestSSOLogin:
    """Tests for SSO login flow."""

    def test_login(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "redirect_url": "https://login.okta.com/oauth/authorize?client_id=abc",
                "state": "csrf-token-123",
                "nonce": "nonce-456",
                "provider": "okta",
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.sso.login(
                provider="okta",
                redirect_uri="https://app.example.com/callback",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/sso/login",
                json={
                    "provider": "okta",
                    "redirect_uri": "https://app.example.com/callback",
                },
            )
            assert result["redirect_url"].startswith("https://login.okta.com")
            assert result["state"] == "csrf-token-123"
            client.close()

    def test_login_azure_ad(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "redirect_url": "https://login.microsoftonline.com/authorize",
                "state": "state-abc",
                "provider": "azure-ad",
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.sso.login(
                provider="azure-ad",
                redirect_uri="https://myapp.com/auth/callback",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/sso/login",
                json={
                    "provider": "azure-ad",
                    "redirect_uri": "https://myapp.com/auth/callback",
                },
            )
            assert result["provider"] == "azure-ad"
            client.close()


class TestSSOCallback:
    """Tests for SSO callback handling."""

    def test_callback(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "success": True,
                "user": {
                    "id": "user-001",
                    "email": "alice@example.com",
                    "name": "Alice",
                    "groups": ["engineering"],
                    "roles": ["admin"],
                },
                "token": "eyJhbGciOiJSUzI1NiJ9...",
                "refresh_token": "rt-xyz-789",
                "expires_in": 3600,
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.sso.callback(
                provider="okta",
                code="auth-code-abc",
                state="csrf-token-123",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/sso/callback",
                json={
                    "provider": "okta",
                    "code": "auth-code-abc",
                    "state": "csrf-token-123",
                },
            )
            assert result["success"] is True
            assert result["user"]["email"] == "alice@example.com"
            assert result["expires_in"] == 3600
            client.close()

    def test_callback_error(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "success": False,
                "error": "invalid_grant",
                "error_description": "Authorization code expired",
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.sso.callback(
                provider="okta",
                code="expired-code",
                state="abc123",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/sso/callback",
                json={
                    "provider": "okta",
                    "code": "expired-code",
                    "state": "abc123",
                },
            )
            assert result["success"] is False
            assert result["error"] == "invalid_grant"
            client.close()


class TestSSOLogout:
    """Tests for SSO logout."""

    def test_logout(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "success": True,
                "redirect_url": "https://idp.example.com/logout",
                "message": "Session terminated",
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.sso.logout(provider="okta")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/sso/logout",
                json={"provider": "okta"},
            )
            assert result["success"] is True
            assert result["message"] == "Session terminated"
            client.close()


class TestSSOMetadata:
    """Tests for SSO provider metadata retrieval."""

    def test_get_metadata(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "entity_id": "https://api.aragora.ai/saml/sp",
                "acs_url": "https://api.aragora.ai/saml/acs",
                "slo_url": "https://api.aragora.ai/saml/slo",
                "certificate": "-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----",
                "metadata_xml": '<?xml version="1.0"?><EntityDescriptor>...</EntityDescriptor>',
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.sso.get_metadata(provider="okta")
            mock_request.assert_called_once_with("GET", "/api/v2/sso/okta/metadata")
            assert result["entity_id"] == "https://api.aragora.ai/saml/sp"
            assert "CERTIFICATE" in result["certificate"]
            client.close()

    def test_get_metadata_different_provider(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entity_id": "test"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            client.sso.get_metadata(provider="azure-ad")
            mock_request.assert_called_once_with("GET", "/api/v2/sso/azure-ad/metadata")
            client.close()


class TestAsyncSSO:
    """Tests for async SSO methods."""

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"enabled": True, "configured": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.sso.get_status()
            mock_request.assert_called_once_with("GET", "/api/v2/sso/status")
            assert result["enabled"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_login(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "redirect_url": "https://idp.example.com/authorize",
                "state": "state-xyz",
                "provider": "google",
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.sso.login(
                provider="google",
                redirect_uri="https://app.example.com/callback",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/sso/login",
                json={
                    "provider": "google",
                    "redirect_uri": "https://app.example.com/callback",
                },
            )
            assert result["provider"] == "google"
            await client.close()

    @pytest.mark.asyncio
    async def test_callback(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "success": True,
                "user": {"id": "user-456", "email": "bob@example.com"},
                "token": "access-token",
                "expires_in": 7200,
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.sso.callback(
                provider="azure-ad",
                code="code-123",
                state="state-456",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/sso/callback",
                json={
                    "provider": "azure-ad",
                    "code": "code-123",
                    "state": "state-456",
                },
            )
            assert result["success"] is True
            assert result["user"]["email"] == "bob@example.com"
            await client.close()
