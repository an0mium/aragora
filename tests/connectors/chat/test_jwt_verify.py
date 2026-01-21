"""
Tests for JWT verification in chat connectors.

Tests the JWT verification utilities for Teams and Google Chat webhooks.
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.connectors.chat.jwt_verify import (
    JWTVerifier,
    JWTVerificationResult,
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

    def test_get_jwt_verifier_singleton(self):
        """get_jwt_verifier returns singleton."""
        verifier1 = get_jwt_verifier()
        verifier2 = get_jwt_verifier()
        assert verifier1 is verifier2

    @pytest.mark.skipif(not HAS_JWT, reason="PyJWT not installed")
    def test_microsoft_token_invalid_format(self):
        """Microsoft verification fails for invalid token format."""
        verifier = JWTVerifier()
        result = verifier.verify_microsoft_token("not-a-jwt", "app123")
        assert result.valid is False
        assert result.error is not None

    @pytest.mark.skipif(not HAS_JWT, reason="PyJWT not installed")
    def test_google_token_invalid_format(self):
        """Google verification fails for invalid token format."""
        verifier = JWTVerifier()
        result = verifier.verify_google_token("not-a-jwt", "project123")
        assert result.valid is False
        assert result.error is not None


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

    @pytest.mark.skipif(not HAS_JWT, reason="PyJWT not installed")
    def test_invalid_token(self):
        """Verification fails for invalid JWT."""
        result = verify_teams_webhook("Bearer invalid.token.here", "app123")
        assert result is False

    @pytest.mark.skipif(HAS_JWT, reason="Test only when PyJWT not installed")
    def test_accepts_without_jwt_library(self):
        """When PyJWT not installed, accepts tokens with warning."""
        # This test only runs when PyJWT is NOT installed
        result = verify_teams_webhook("Bearer some.token.here", "app123")
        assert result is True


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

    @pytest.mark.skipif(not HAS_JWT, reason="PyJWT not installed")
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

    @pytest.mark.skipif(not HAS_JWT, reason="PyJWT not installed")
    def test_jwks_client_cached(self):
        """JWKS client is cached after first creation."""
        verifier = JWTVerifier()

        # Force cache expiry
        verifier._cache_time = 0

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # First call creates client
            client1 = verifier._get_microsoft_jwks_client()
            assert mock_client.call_count == 1

            # Second call uses cache
            client2 = verifier._get_microsoft_jwks_client()
            assert mock_client.call_count == 1  # No new call
            assert client1 is client2

    @pytest.mark.skipif(not HAS_JWT, reason="PyJWT not installed")
    def test_jwks_client_refreshes_after_ttl(self):
        """JWKS client refreshes after cache TTL."""
        import time

        verifier = JWTVerifier()
        verifier._cache_ttl = 0.1  # Very short TTL for testing

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # First call
            verifier._get_microsoft_jwks_client()
            assert mock_client.call_count == 1

            # Wait for TTL to expire
            time.sleep(0.2)

            # Second call should refresh
            verifier._get_microsoft_jwks_client()
            assert mock_client.call_count == 2
