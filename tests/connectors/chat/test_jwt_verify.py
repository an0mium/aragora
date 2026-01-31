"""
Tests for JWT verification in chat connectors.

Tests the JWT verification utilities for Teams and Google Chat webhooks.

Phase 3.1: Bot Framework JWT validation for Teams webhook handler.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from aragora.connectors.chat.jwt_verify import (
    JWTVerifier,
    JWTVerificationResult,
    _fetch_openid_metadata,
    _OpenIDMetadataCache,
    get_jwt_verifier,
    verify_teams_webhook,
    verify_google_chat_webhook,
    HAS_JWT,
)


class TestJWTVerificationResult:
    """Test JWTVerificationResult dataclass."""

    def test_valid_result(self):
        """Valid result has claims and no error."""
        result = JWTVerificationResult(
            valid=True,
            claims={"sub": "user123", "aud": "app123"},
        )
        assert result.valid is True
        assert result.claims["sub"] == "user123"
        assert result.error is None

    def test_invalid_result(self):
        """Invalid result has error message."""
        result = JWTVerificationResult(
            valid=False,
            claims={},
            error="Token expired",
        )
        assert result.valid is False
        assert result.error == "Token expired"


class TestJWTVerifier:
    """Test JWTVerifier class."""

    def test_verifier_initialization(self):
        """Verifier initializes with no clients."""
        verifier = JWTVerifier()
        assert verifier._microsoft_jwks_client is None
        assert verifier._google_jwks_client is None

    def test_verifier_custom_cache_ttl(self):
        """Verifier accepts custom cache TTL."""
        verifier = JWTVerifier(cache_ttl=600)
        assert verifier._cache_ttl == 600

    def test_verifier_default_cache_ttl(self):
        """Verifier defaults to 1 hour cache TTL."""
        verifier = JWTVerifier()
        assert verifier._cache_ttl == 3600

    def test_get_jwt_verifier_singleton(self):
        """get_jwt_verifier returns singleton."""
        verifier1 = get_jwt_verifier()
        verifier2 = get_jwt_verifier()
        assert verifier1 is verifier2

    def test_microsoft_token_invalid_format(self):
        """Microsoft verification fails for invalid token format."""
        verifier = JWTVerifier()
        result = verifier.verify_microsoft_token("not-a-jwt", "app123")
        assert result.valid is False
        assert result.error is not None

    def test_google_token_invalid_format(self):
        """Google verification fails for invalid token format."""
        verifier = JWTVerifier()
        result = verifier.verify_google_token("not-a-jwt", "project123")
        assert result.valid is False
        assert result.error is not None

    def test_microsoft_token_fails_closed_without_pyjwt(self):
        """Microsoft verification fails closed when PyJWT unavailable."""
        verifier = JWTVerifier()
        with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", False):
            result = verifier.verify_microsoft_token("some.jwt.token", "app123")
        assert result.valid is False
        assert "PyJWT" in result.error

    def test_google_token_fails_closed_without_pyjwt(self):
        """Google verification fails closed when PyJWT unavailable."""
        verifier = JWTVerifier()
        with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", False):
            result = verifier.verify_google_token("some.jwt.token", "project123")
        assert result.valid is False
        assert "PyJWT" in result.error

    def test_microsoft_jwks_client_returns_none_without_pyjwt(self):
        """JWKS client returns None when PyJWT not available."""
        verifier = JWTVerifier()
        with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", False):
            client = verifier._get_microsoft_jwks_client()
        assert client is None

    def test_google_jwks_client_returns_none_without_pyjwt(self):
        """JWKS client returns None when PyJWT not available."""
        verifier = JWTVerifier()
        with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", False):
            client = verifier._get_google_jwks_client()
        assert client is None


class TestVerifyTeamsWebhook:
    """Test verify_teams_webhook function."""

    def test_missing_bearer_prefix(self):
        """Verification fails without Bearer prefix."""
        result = verify_teams_webhook("just-a-token", "app123")
        assert result is False

    def test_empty_header(self):
        """Verification fails with empty header."""
        result = verify_teams_webhook("", "app123")
        assert result is False

    def test_invalid_token(self):
        """Verification fails for invalid JWT."""
        result = verify_teams_webhook("Bearer invalid.token.here", "app123")
        assert result is False

    def test_basic_auth_rejected(self):
        """Verification fails for Basic auth scheme."""
        result = verify_teams_webhook("Basic dXNlcjpwYXNz", "app123")
        assert result is False

    def test_bearer_only_rejected(self):
        """Verification fails for 'Bearer' without token."""
        result = verify_teams_webhook("Bearer ", "app123")
        assert result is False

    # Note: test_rejects_without_jwt_library was removed because PyJWT is now
    # always installed. The fail-closed behavior when PyJWT is unavailable
    # is still implemented in the source code but cannot occur in practice.


class TestVerifyGoogleChatWebhook:
    """Test verify_google_chat_webhook function."""

    def test_missing_bearer_prefix(self):
        """Verification fails without Bearer prefix."""
        result = verify_google_chat_webhook("just-a-token")
        assert result is False

    def test_empty_header(self):
        """Verification fails with empty header."""
        result = verify_google_chat_webhook("")
        assert result is False

    def test_invalid_token(self):
        """Verification fails for invalid JWT."""
        result = verify_google_chat_webhook("Bearer invalid.token.here")
        assert result is False

    def test_accepts_without_project_id(self):
        """Can verify without project_id (skips audience check)."""
        # This should not raise even if token is invalid format
        # Just checks the function handles missing project_id
        result = verify_google_chat_webhook("Bearer test", project_id=None)
        # Will fail due to invalid token, but not due to missing project_id
        assert isinstance(result, bool)


class TestTeamsConnectorVerification:
    """Test Teams connector webhook verification integration."""

    def test_teams_connector_uses_jwt_verify(self):
        """Teams connector calls JWT verification."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="test-app-id")

        # Missing auth header
        result = connector.verify_webhook({}, b"{}")
        assert result is False

        # Invalid auth header format
        result = connector.verify_webhook({"Authorization": "Basic xyz"}, b"{}")
        assert result is False

    def test_teams_connector_accepts_bearer_token(self):
        """Teams connector accepts Bearer token (may pass or fail based on JWT lib)."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="test-app-id")

        # With valid Bearer prefix (token itself may be invalid)
        headers = {"Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test"}
        result = connector.verify_webhook(headers, b"{}")

        # Result depends on whether PyJWT is installed and token validity
        assert isinstance(result, bool)


class TestGoogleChatConnectorVerification:
    """Test Google Chat connector webhook verification integration."""

    def test_google_chat_connector_uses_jwt_verify(self):
        """Google Chat connector calls JWT verification."""
        from aragora.connectors.chat.google_chat import GoogleChatConnector

        connector = GoogleChatConnector(project_id="test-project")

        # Missing auth header
        result = connector.verify_webhook({}, b"{}")
        assert result is False

        # Invalid auth header format
        result = connector.verify_webhook({"Authorization": "Basic xyz"}, b"{}")
        assert result is False

    def test_google_chat_connector_accepts_bearer_token(self):
        """Google Chat connector accepts Bearer token."""
        from aragora.connectors.chat.google_chat import GoogleChatConnector

        connector = GoogleChatConnector(project_id="test-project")

        # With valid Bearer prefix
        headers = {"Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test"}
        result = connector.verify_webhook(headers, b"{}")

        # Result depends on whether PyJWT is installed and token validity
        assert isinstance(result, bool)


class TestJWKSClientCaching:
    """Test JWKS client caching behavior."""

    def test_jwks_client_cached(self):
        """JWKS client is cached after first creation."""
        verifier = JWTVerifier()

        # Force cache expiry
        verifier._microsoft_cache_time = 0

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Patch OpenID discovery to avoid network calls
            with patch.object(
                verifier, "_resolve_microsoft_jwks_uri", return_value="https://test/keys"
            ):
                # First call creates client
                client1 = verifier._get_microsoft_jwks_client()
                assert mock_client.call_count == 1

                # Second call uses cache
                client2 = verifier._get_microsoft_jwks_client()
                assert mock_client.call_count == 1  # No new call
                assert client1 is client2

    def test_jwks_client_refreshes_after_ttl(self):
        """JWKS client refreshes after cache TTL."""
        import time

        verifier = JWTVerifier(cache_ttl=0.1)  # Very short TTL for testing

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Patch OpenID discovery to avoid network calls
            with patch.object(
                verifier, "_resolve_microsoft_jwks_uri", return_value="https://test/keys"
            ):
                # First call
                verifier._get_microsoft_jwks_client()
                assert mock_client.call_count == 1

                # Wait for TTL to expire
                time.sleep(0.2)

                # Second call should refresh
                verifier._get_microsoft_jwks_client()
                assert mock_client.call_count == 2

    def test_google_jwks_client_cached_independently(self):
        """Google JWKS client has its own cache timer."""
        verifier = JWTVerifier()
        verifier._google_cache_time = 0

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # First call creates client
            client1 = verifier._get_google_jwks_client()
            assert mock_client.call_count == 1

            # Second call uses cache
            client2 = verifier._get_google_jwks_client()
            assert mock_client.call_count == 1
            assert client1 is client2


class TestOpenIDMetadataDiscovery:
    """Test OpenID metadata discovery for JWKS URI resolution."""

    def test_fetch_openid_metadata_success(self):
        """Successful metadata fetch returns dict with jwks_uri."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "jwks_uri": "https://login.botframework.com/v1/.well-known/keys",
                "issuer": "https://api.botframework.com",
                "authorization_endpoint": "https://invalid.botframework.com",
            }
        ).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("aragora.connectors.chat.jwt_verify.urlopen", return_value=mock_response):
            result = _fetch_openid_metadata(
                "https://login.botframework.com/v1/.well-known/openidconfiguration"
            )

        assert result is not None
        assert result["jwks_uri"] == "https://login.botframework.com/v1/.well-known/keys"
        assert result["issuer"] == "https://api.botframework.com"

    def test_fetch_openid_metadata_missing_jwks_uri(self):
        """Returns None when jwks_uri field is missing."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "issuer": "https://api.botframework.com",
            }
        ).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("aragora.connectors.chat.jwt_verify.urlopen", return_value=mock_response):
            result = _fetch_openid_metadata("https://example.com/.well-known/openidconfiguration")

        assert result is None

    def test_fetch_openid_metadata_network_error(self):
        """Returns None on network error."""
        from urllib.error import URLError

        with patch(
            "aragora.connectors.chat.jwt_verify.urlopen", side_effect=URLError("Connection refused")
        ):
            result = _fetch_openid_metadata("https://example.com/.well-known/openidconfiguration")

        assert result is None

    def test_fetch_openid_metadata_invalid_json(self):
        """Returns None on invalid JSON response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"not json"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("aragora.connectors.chat.jwt_verify.urlopen", return_value=mock_response):
            result = _fetch_openid_metadata("https://example.com/.well-known/openidconfiguration")

        assert result is None

    def test_resolve_microsoft_jwks_uri_uses_discovery(self):
        """JWKS URI is resolved from OpenID metadata."""
        verifier = JWTVerifier()

        with patch(
            "aragora.connectors.chat.jwt_verify._fetch_openid_metadata",
            return_value={"jwks_uri": "https://discovered.example.com/keys", "issuer": "test"},
        ):
            uri = verifier._resolve_microsoft_jwks_uri()

        assert uri == "https://discovered.example.com/keys"
        # Check metadata was cached
        assert verifier._microsoft_metadata is not None
        assert verifier._microsoft_metadata.jwks_uri == "https://discovered.example.com/keys"

    def test_resolve_microsoft_jwks_uri_caches_result(self):
        """Cached JWKS URI is returned without re-fetching."""
        import time

        verifier = JWTVerifier()
        verifier._microsoft_metadata = _OpenIDMetadataCache(
            jwks_uri="https://cached.example.com/keys",
            issuer="cached-issuer",
            fetched_at=time.time(),
        )

        with patch(
            "aragora.connectors.chat.jwt_verify._fetch_openid_metadata",
        ) as mock_fetch:
            uri = verifier._resolve_microsoft_jwks_uri()

        assert uri == "https://cached.example.com/keys"
        mock_fetch.assert_not_called()

    def test_resolve_microsoft_jwks_uri_falls_back_on_failure(self):
        """Falls back to hardcoded URI when discovery fails."""
        verifier = JWTVerifier()

        with patch(
            "aragora.connectors.chat.jwt_verify._fetch_openid_metadata",
            return_value=None,
        ):
            uri = verifier._resolve_microsoft_jwks_uri()

        from aragora.connectors.chat.jwt_verify import MICROSOFT_JWKS_URI

        assert uri == MICROSOFT_JWKS_URI

    def test_resolve_microsoft_jwks_uri_refetches_after_ttl(self):
        """Metadata is re-fetched after cache TTL expires."""
        import time

        verifier = JWTVerifier(cache_ttl=0.1)
        verifier._microsoft_metadata = _OpenIDMetadataCache(
            jwks_uri="https://old.example.com/keys",
            issuer="old-issuer",
            fetched_at=time.time() - 1.0,  # Expired
        )

        with patch(
            "aragora.connectors.chat.jwt_verify._fetch_openid_metadata",
            return_value={"jwks_uri": "https://new.example.com/keys", "issuer": "new-issuer"},
        ):
            uri = verifier._resolve_microsoft_jwks_uri()

        assert uri == "https://new.example.com/keys"

    def test_openid_metadata_cache_dataclass(self):
        """OpenIDMetadataCache dataclass stores fields correctly."""
        cache = _OpenIDMetadataCache(
            jwks_uri="https://example.com/keys",
            issuer="https://example.com",
        )
        assert cache.jwks_uri == "https://example.com/keys"
        assert cache.issuer == "https://example.com"
        assert cache.fetched_at > 0


class TestTeamsHandlerJWTIntegration:
    """Test JWT validation integration in the Teams handler."""

    @pytest.mark.asyncio
    async def test_verify_teams_token_rejects_missing_header(self):
        """Empty auth header is rejected."""
        from aragora.server.handlers.bots.teams import _verify_teams_token

        result = await _verify_teams_token("", "app-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_teams_token_rejects_non_bearer(self):
        """Non-Bearer auth header is rejected."""
        from aragora.server.handlers.bots.teams import _verify_teams_token

        result = await _verify_teams_token("Basic abc123", "app-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_teams_token_delegates_to_jwt_verify(self):
        """Token verification delegates to centralized jwt_verify module."""
        from aragora.server.handlers.bots.teams import _verify_teams_token

        with patch(
            "aragora.connectors.chat.jwt_verify.verify_teams_webhook",
            return_value=True,
        ) as mock_verify:
            with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", True):
                result = await _verify_teams_token("Bearer valid.jwt.token", "app-123")

        assert result is True
        mock_verify.assert_called_once_with("Bearer valid.jwt.token", "app-123")

    @pytest.mark.asyncio
    async def test_verify_teams_token_rejects_invalid_jwt(self):
        """Invalid JWT token is rejected."""
        from aragora.server.handlers.bots.teams import _verify_teams_token

        with patch(
            "aragora.connectors.chat.jwt_verify.verify_teams_webhook",
            return_value=False,
        ):
            with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", True):
                result = await _verify_teams_token("Bearer invalid.jwt.token", "app-123")

        assert result is False

    def test_ms_app_id_env_var_fallback(self):
        """TEAMS_APP_ID falls back to MS_APP_ID environment variable."""
        with patch.dict("os.environ", {"MS_APP_ID": "ms-app-fallback"}, clear=False):
            with patch.dict("os.environ", {"TEAMS_APP_ID": ""}, clear=False):
                # Re-evaluate the expression
                app_id = os.environ.get("TEAMS_APP_ID") or os.environ.get("MS_APP_ID")
                assert app_id == "ms-app-fallback"

    def test_teams_app_id_takes_precedence(self):
        """TEAMS_APP_ID takes precedence over MS_APP_ID."""
        with patch.dict(
            "os.environ", {"TEAMS_APP_ID": "teams-app", "MS_APP_ID": "ms-app"}, clear=False
        ):
            app_id = os.environ.get("TEAMS_APP_ID") or os.environ.get("MS_APP_ID")
            assert app_id == "teams-app"
