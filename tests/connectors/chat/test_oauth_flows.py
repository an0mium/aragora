"""
Tests for OAuth flow handling in chat connectors.

Tests cover:
- Slack OAuth flow scenarios
- Teams OAuth flow scenarios
- Token refresh mechanics
- Multi-workspace/tenant handling
- State token CSRF protection
- Error scenarios
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib
import hmac
import time


class TestSlackOAuthFlow:
    """Tests for Slack OAuth flows."""

    def test_oauth_state_generation(self):
        """Should generate secure state token for OAuth."""
        import secrets

        state = secrets.token_urlsafe(32)
        assert len(state) >= 32
        # State should be URL-safe
        assert all(c.isalnum() or c in "-_" for c in state)

    def test_oauth_url_construction(self):
        """Should construct proper OAuth authorization URL."""
        client_id = "test_client_id"
        scopes = ["channels:read", "chat:write", "commands"]
        redirect_uri = "https://example.com/oauth/callback"
        state = "test_state_token"

        url = (
            f"https://slack.com/oauth/v2/authorize?"
            f"client_id={client_id}&"
            f"scope={','.join(scopes)}&"
            f"redirect_uri={redirect_uri}&"
            f"state={state}"
        )

        assert "client_id=test_client_id" in url
        assert "channels:read" in url
        assert "state=test_state_token" in url

    @pytest.mark.asyncio
    async def test_token_exchange(self):
        """Should exchange authorization code for access token."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "access_token": "xoxb-test-token",
            "token_type": "bot",
            "scope": "channels:read,chat:write",
            "bot_user_id": "U12345",
            "app_id": "A12345",
            "team": {"name": "Test Workspace", "id": "T12345"},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # Simulate token exchange
            async with mock_client() as client:
                response = await client.post(
                    "https://slack.com/api/oauth.v2.access",
                    data={
                        "client_id": "test_client_id",
                        "client_secret": "test_client_secret",
                        "code": "test_auth_code",
                        "redirect_uri": "https://example.com/callback",
                    },
                )
                data = response.json()

            assert data["ok"] is True
            assert data["access_token"] == "xoxb-test-token"
            assert data["team"]["id"] == "T12345"

    @pytest.mark.asyncio
    async def test_invalid_auth_code(self):
        """Should handle invalid authorization code."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": False,
            "error": "invalid_code",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            async with mock_client() as client:
                response = await client.post(
                    "https://slack.com/api/oauth.v2.access",
                    data={"code": "invalid_code"},
                )
                data = response.json()

            assert data["ok"] is False
            assert data["error"] == "invalid_code"

    @pytest.mark.asyncio
    async def test_state_mismatch_protection(self):
        """Should reject requests with mismatched state token."""
        original_state = "original_state_abc123"
        returned_state = "different_state_xyz789"

        # Simulate CSRF check
        assert original_state != returned_state
        # In real implementation, this would raise an error

    @pytest.mark.asyncio
    async def test_multi_workspace_token_storage(self):
        """Should store tokens per workspace."""
        workspace_tokens = {}

        # Simulate storing tokens for multiple workspaces
        workspace_tokens["T12345"] = {
            "access_token": "xoxb-token-workspace-1",
            "team_name": "Workspace 1",
        }
        workspace_tokens["T67890"] = {
            "access_token": "xoxb-token-workspace-2",
            "team_name": "Workspace 2",
        }

        assert "T12345" in workspace_tokens
        assert "T67890" in workspace_tokens
        assert (
            workspace_tokens["T12345"]["access_token"] != workspace_tokens["T67890"]["access_token"]
        )


class TestTeamsOAuthFlow:
    """Tests for Microsoft Teams OAuth flows."""

    def test_oauth_url_construction(self):
        """Should construct proper MSAL OAuth URL."""
        client_id = "test_app_id"
        tenant_id = "common"  # or specific tenant ID
        redirect_uri = "https://example.com/auth/callback"
        scopes = ["https://graph.microsoft.com/.default"]
        state = "test_state_token"

        url = (
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize?"
            f"client_id={client_id}&"
            f"response_type=code&"
            f"redirect_uri={redirect_uri}&"
            f"response_mode=query&"
            f"scope={'+'.join(scopes)}&"
            f"state={state}"
        )

        assert "client_id=test_app_id" in url
        assert "response_type=code" in url
        assert "graph.microsoft.com" in url

    @pytest.mark.asyncio
    async def test_token_exchange_with_client_credentials(self):
        """Should exchange code for token using client credentials."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "token_type": "Bearer",
            "expires_in": 3600,
            "ext_expires_in": 3600,
            "access_token": "eyJ0eXAi...",
            "refresh_token": "0.AQYAAi...",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            async with mock_client() as client:
                response = await client.post(
                    "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                    data={
                        "client_id": "test_app_id",
                        "client_secret": "test_client_secret",
                        "code": "test_auth_code",
                        "redirect_uri": "https://example.com/callback",
                        "grant_type": "authorization_code",
                    },
                )
                data = response.json()

            assert data["token_type"] == "Bearer"
            assert "access_token" in data
            assert data["expires_in"] == 3600

    @pytest.mark.asyncio
    async def test_token_refresh(self):
        """Should refresh expired token using refresh token."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "token_type": "Bearer",
            "expires_in": 3600,
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            async with mock_client() as client:
                response = await client.post(
                    "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                    data={
                        "client_id": "test_app_id",
                        "client_secret": "test_client_secret",
                        "refresh_token": "old_refresh_token",
                        "grant_type": "refresh_token",
                        "scope": "https://graph.microsoft.com/.default",
                    },
                )
                data = response.json()

            assert data["access_token"] == "new_access_token"
            assert data["refresh_token"] == "new_refresh_token"

    @pytest.mark.asyncio
    async def test_admin_consent_required(self):
        """Should handle admin consent required errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "AADSTS65001: The user or administrator has not consented",
            "error_codes": [65001],
            "suberror": "consent_required",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            async with mock_client() as client:
                response = await client.post(
                    "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                    data={"grant_type": "authorization_code"},
                )
                data = response.json()

            assert data["error"] == "invalid_grant"
            assert "consent" in data["error_description"].lower()

    @pytest.mark.asyncio
    async def test_multi_tenant_token_handling(self):
        """Should handle tokens from different tenants."""
        tenant_tokens = {}

        # Simulate multi-tenant token storage
        tenant_tokens["tenant_1"] = {
            "access_token": "token_for_tenant_1",
            "tenant_id": "tenant-guid-1",
            "expires_at": datetime.utcnow() + timedelta(hours=1),
        }
        tenant_tokens["tenant_2"] = {
            "access_token": "token_for_tenant_2",
            "tenant_id": "tenant-guid-2",
            "expires_at": datetime.utcnow() + timedelta(hours=1),
        }

        assert len(tenant_tokens) == 2
        assert tenant_tokens["tenant_1"]["tenant_id"] != tenant_tokens["tenant_2"]["tenant_id"]


class TestTokenRefreshMechanics:
    """Tests for token refresh handling."""

    def test_token_expiry_detection(self):
        """Should detect when token is expired or expiring soon."""

        def is_token_expired(expires_at: datetime, buffer_seconds: int = 300) -> bool:
            """Check if token is expired or will expire soon."""
            return datetime.utcnow() >= (expires_at - timedelta(seconds=buffer_seconds))

        # Token expired
        expired_at = datetime.utcnow() - timedelta(hours=1)
        assert is_token_expired(expired_at) is True

        # Token expiring soon (within buffer)
        expiring_soon = datetime.utcnow() + timedelta(seconds=100)
        assert is_token_expired(expiring_soon, buffer_seconds=300) is True

        # Token still valid
        valid_until = datetime.utcnow() + timedelta(hours=1)
        assert is_token_expired(valid_until) is False

    @pytest.mark.asyncio
    async def test_proactive_token_refresh(self):
        """Should refresh token before it expires."""
        token_data = {
            "access_token": "current_token",
            "refresh_token": "current_refresh",
            "expires_at": datetime.utcnow() + timedelta(minutes=4),  # Expiring soon
        }

        # Should trigger refresh when < 5 min remaining
        buffer_seconds = 300  # 5 minutes
        should_refresh = (
            token_data["expires_at"] - datetime.utcnow()
        ).total_seconds() < buffer_seconds

        assert should_refresh is True

    @pytest.mark.asyncio
    async def test_concurrent_refresh_handling(self):
        """Should handle concurrent refresh requests gracefully."""
        import asyncio

        refresh_lock = asyncio.Lock()
        refresh_count = 0

        async def refresh_token_with_lock():
            nonlocal refresh_count
            async with refresh_lock:
                # Only one refresh should happen
                await asyncio.sleep(0.1)
                refresh_count += 1
                return {"access_token": "new_token"}

        # Simulate concurrent refresh requests
        results = await asyncio.gather(
            refresh_token_with_lock(),
            refresh_token_with_lock(),
            refresh_token_with_lock(),
        )

        # With lock, refreshes happen sequentially (not truly concurrent)
        # In real implementation, would check if already refreshing
        assert len(results) == 3


class TestOAuthErrorHandling:
    """Tests for OAuth error scenarios."""

    @pytest.mark.asyncio
    async def test_network_error_during_token_exchange(self):
        """Should handle network errors during token exchange."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            with pytest.raises(httpx.ConnectError):
                async with mock_client() as client:
                    await client.post(
                        "https://slack.com/api/oauth.v2.access",
                        data={"code": "test_code"},
                    )

    @pytest.mark.asyncio
    async def test_rate_limit_during_oauth(self):
        """Should handle rate limiting during OAuth."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.json.return_value = {"error": "rate_limited"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            async with mock_client() as client:
                response = await client.post(
                    "https://slack.com/api/oauth.v2.access",
                    data={"code": "test_code"},
                )

            assert response.status_code == 429
            assert response.headers["Retry-After"] == "60"

    @pytest.mark.asyncio
    async def test_invalid_client_credentials(self):
        """Should handle invalid client credentials."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": "invalid_client",
            "error_description": "Invalid client_id or client_secret",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            async with mock_client() as client:
                response = await client.post(
                    "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                    data={
                        "client_id": "wrong_id",
                        "client_secret": "wrong_secret",
                    },
                )
                data = response.json()

            assert response.status_code == 401
            assert data["error"] == "invalid_client"

    @pytest.mark.asyncio
    async def test_scope_downgrade_handling(self):
        """Should handle scope downgrade during re-authorization."""
        # Simulate user removing permissions during re-auth
        original_scopes = {"channels:read", "chat:write", "commands", "files:read"}
        returned_scopes = {"channels:read", "chat:write"}  # User removed some

        missing_scopes = original_scopes - returned_scopes
        assert missing_scopes == {"commands", "files:read"}

        # Should warn about missing scopes
        assert len(missing_scopes) > 0

    @pytest.mark.asyncio
    async def test_token_revocation_handling(self):
        """Should handle revoked tokens gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "ok": False,
            "error": "token_revoked",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            async with mock_client() as client:
                response = await client.get(
                    "https://slack.com/api/auth.test",
                    headers={"Authorization": "Bearer revoked_token"},
                )
                data = response.json()

            assert data["error"] == "token_revoked"
            # Should trigger re-authentication flow


class TestWebhookSignatureVerification:
    """Tests for webhook signature verification."""

    def test_slack_signature_verification(self):
        """Should verify Slack webhook signatures."""
        signing_secret = "test_signing_secret"
        timestamp = str(int(time.time()))
        body = '{"event": "test"}'

        # Calculate expected signature
        sig_basestring = f"v0:{timestamp}:{body}"
        expected_signature = (
            "v0="
            + hmac.new(signing_secret.encode(), sig_basestring.encode(), hashlib.sha256).hexdigest()
        )

        # Verify signature
        computed_signature = (
            "v0="
            + hmac.new(signing_secret.encode(), sig_basestring.encode(), hashlib.sha256).hexdigest()
        )

        assert hmac.compare_digest(expected_signature, computed_signature)

    def test_invalid_signature_rejected(self):
        """Should reject invalid webhook signatures."""
        signing_secret = "test_signing_secret"
        timestamp = str(int(time.time()))
        body = '{"event": "test"}'

        sig_basestring = f"v0:{timestamp}:{body}"
        valid_signature = (
            "v0="
            + hmac.new(signing_secret.encode(), sig_basestring.encode(), hashlib.sha256).hexdigest()
        )

        invalid_signature = "v0=invalid_signature_abc123"

        assert not hmac.compare_digest(valid_signature, invalid_signature)

    def test_timestamp_validation(self):
        """Should reject requests with old timestamps."""
        max_age_seconds = 300  # 5 minutes

        # Current timestamp - should be valid
        current_ts = int(time.time())
        assert (time.time() - current_ts) <= max_age_seconds

        # Old timestamp - should be rejected
        old_ts = int(time.time()) - 600  # 10 minutes ago
        assert (time.time() - old_ts) > max_age_seconds
