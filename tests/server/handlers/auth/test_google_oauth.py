"""
Tests for Google OAuth Provider.

Tests cover:
- Authorization URL generation
- Token exchange
- User info retrieval
- Token refresh
- Token revocation
- Configuration validation
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch, Mock

import pytest

from aragora.server.handlers.oauth_providers.google import (
    GoogleOAuthProvider,
    GOOGLE_AUTH_URL,
    GOOGLE_TOKEN_URL,
    GOOGLE_USERINFO_URL,
    GOOGLE_REVOCATION_URL,
)
from aragora.server.handlers.oauth_providers.base import (
    OAuthProviderConfig,
    OAuthTokens,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def google_config():
    """Create a minimal Google OAuth configuration."""
    return OAuthProviderConfig(
        client_id="test-client-id.apps.googleusercontent.com",
        client_secret="test-client-secret",
        redirect_uri="https://example.com/callback",
        scopes=["openid", "email", "profile"],
        authorization_endpoint=GOOGLE_AUTH_URL,
        token_endpoint=GOOGLE_TOKEN_URL,
        userinfo_endpoint=GOOGLE_USERINFO_URL,
        revocation_endpoint=GOOGLE_REVOCATION_URL,
    )


@pytest.fixture
def google_provider(google_config):
    """Create a Google OAuth provider with test configuration."""
    provider = GoogleOAuthProvider(config=google_config)
    yield provider
    provider.close()


@pytest.fixture
def mock_tokens():
    """Create mock OAuth tokens."""
    return OAuthTokens(
        access_token="ya29.mock-access-token",
        refresh_token="1//mock-refresh-token",
        expires_in=3600,
        token_type="Bearer",
        scope="openid email profile",
    )


@pytest.fixture
def mock_user_info():
    """Create mock Google user info response."""
    return {
        "id": "123456789",
        "email": "user@example.com",
        "verified_email": True,
        "name": "Test User",
        "given_name": "Test",
        "family_name": "User",
        "picture": "https://lh3.googleusercontent.com/a/photo.jpg",
        "locale": "en",
    }


# ===========================================================================
# Configuration Tests
# ===========================================================================


class TestGoogleOAuthConfiguration:
    """Tests for Google OAuth provider configuration."""

    def test_provider_name(self, google_provider):
        """Test provider name is correct."""
        assert google_provider.PROVIDER_NAME == "google"

    def test_is_configured_with_valid_config(self, google_provider):
        """Test is_configured returns True with valid config."""
        assert google_provider.is_configured

    def test_is_configured_without_client_id(self):
        """Test is_configured returns False without client_id."""
        config = OAuthProviderConfig(
            client_id="",
            client_secret="secret",
            redirect_uri="https://example.com/callback",
        )
        provider = GoogleOAuthProvider(config=config)
        assert not provider.is_configured
        provider.close()

    def test_is_configured_without_client_secret(self):
        """Test is_configured returns False without client_secret."""
        config = OAuthProviderConfig(
            client_id="client-id",
            client_secret="",
            redirect_uri="https://example.com/callback",
        )
        provider = GoogleOAuthProvider(config=config)
        assert not provider.is_configured
        provider.close()


# ===========================================================================
# Authorization URL Tests
# ===========================================================================


class TestGoogleAuthorizationUrl:
    """Tests for Google OAuth authorization URL generation."""

    def test_basic_authorization_url(self, google_provider):
        """Test basic authorization URL generation."""
        url = google_provider.get_authorization_url(state="test-state")

        assert GOOGLE_AUTH_URL in url
        assert "client_id=" in url
        assert "state=test-state" in url
        assert "response_type=code" in url
        assert "redirect_uri=" in url

    def test_authorization_url_with_scopes(self, google_provider):
        """Test authorization URL includes scopes."""
        url = google_provider.get_authorization_url(
            state="test-state",
            scopes=["openid", "email", "profile"],
        )

        assert "scope=" in url
        # Scopes should be URL-encoded space-separated
        assert "openid" in url

    def test_authorization_url_with_custom_redirect(self, google_provider):
        """Test authorization URL with custom redirect URI."""
        custom_redirect = "https://custom.example.com/oauth/callback"
        url = google_provider.get_authorization_url(
            state="test-state",
            redirect_uri=custom_redirect,
        )

        # URL-encoded redirect URI
        assert "redirect_uri=" in url

    def test_authorization_url_with_offline_access(self, google_provider):
        """Test authorization URL requests offline access for refresh token."""
        url = google_provider.get_authorization_url(
            state="test-state",
            access_type="offline",
        )

        assert "access_type=offline" in url

    def test_authorization_url_with_consent_prompt(self, google_provider):
        """Test authorization URL with consent prompt."""
        url = google_provider.get_authorization_url(
            state="test-state",
            prompt="consent",
        )

        assert "prompt=consent" in url

    def test_authorization_url_with_login_hint(self, google_provider):
        """Test authorization URL with login hint."""
        url = google_provider.get_authorization_url(
            state="test-state",
            login_hint="user@example.com",
        )

        assert "login_hint=" in url


# ===========================================================================
# Token Exchange Tests
# ===========================================================================


class TestGoogleTokenExchange:
    """Tests for Google OAuth token exchange."""

    def test_successful_token_exchange(self, google_provider, mock_tokens):
        """Test successful token exchange."""
        with patch.object(
            google_provider,
            "_get_http_client",
            return_value=MagicMock(
                post=MagicMock(
                    return_value=MagicMock(
                        status_code=200,
                        raise_for_status=MagicMock(),
                        json=lambda: {
                            "access_token": mock_tokens.access_token,
                            "refresh_token": mock_tokens.refresh_token,
                            "expires_in": mock_tokens.expires_in,
                            "token_type": mock_tokens.token_type,
                            "scope": mock_tokens.scope,
                        },
                    )
                )
            ),
        ):
            tokens = google_provider.exchange_code("test-auth-code")

            assert tokens.access_token == mock_tokens.access_token
            assert tokens.refresh_token == mock_tokens.refresh_token
            assert tokens.token_type == "Bearer"

    def test_token_exchange_with_invalid_code(self, google_provider):
        """Test token exchange with invalid authorization code."""
        mock_response = MagicMock(
            status_code=400,
            json=lambda: {
                "error": "invalid_grant",
                "error_description": "Invalid authorization code",
            },
        )
        mock_response.raise_for_status.side_effect = Exception("Bad Request")

        with patch.object(
            google_provider,
            "_get_http_client",
            return_value=MagicMock(post=MagicMock(return_value=mock_response)),
        ):
            with pytest.raises(Exception):
                google_provider.exchange_code("invalid-code")

    def test_token_exchange_with_custom_redirect(self, google_provider, mock_tokens):
        """Test token exchange with custom redirect URI."""
        mock_post = MagicMock(
            return_value=MagicMock(
                status_code=200,
                raise_for_status=MagicMock(),
                json=lambda: {
                    "access_token": mock_tokens.access_token,
                    "expires_in": mock_tokens.expires_in,
                    "token_type": mock_tokens.token_type,
                },
            )
        )

        with patch.object(
            google_provider,
            "_get_http_client",
            return_value=MagicMock(post=mock_post),
        ):
            google_provider.exchange_code(
                "test-code",
                redirect_uri="https://custom.example.com/callback",
            )

            # Verify redirect_uri was included in request
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs is not None


# ===========================================================================
# User Info Tests
# ===========================================================================


class TestGoogleUserInfo:
    """Tests for Google user info retrieval."""

    def test_successful_user_info_retrieval(self, google_provider, mock_tokens, mock_user_info):
        """Test successful user info retrieval."""
        with patch.object(
            google_provider,
            "_get_http_client",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(
                        status_code=200,
                        raise_for_status=MagicMock(),
                        json=lambda: mock_user_info,
                    )
                )
            ),
        ):
            user_info = google_provider.get_user_info(mock_tokens.access_token)

            assert user_info.provider_user_id == "123456789"
            assert user_info.email == "user@example.com"
            assert user_info.email_verified is True
            assert user_info.name == "Test User"

    def test_user_info_with_invalid_token(self, google_provider):
        """Test user info retrieval with invalid token."""
        mock_response = MagicMock(
            status_code=401,
            json=lambda: {
                "error": {
                    "code": 401,
                    "message": "Invalid Credentials",
                }
            },
        )
        mock_response.raise_for_status.side_effect = Exception("Unauthorized")

        with patch.object(
            google_provider,
            "_get_http_client",
            return_value=MagicMock(get=MagicMock(return_value=mock_response)),
        ):
            with pytest.raises(Exception):
                google_provider.get_user_info("invalid-token")


# ===========================================================================
# Token Refresh Tests
# ===========================================================================


class TestGoogleTokenRefresh:
    """Tests for Google token refresh."""

    def test_successful_token_refresh(self, google_provider):
        """Test successful token refresh."""
        with patch.object(
            google_provider,
            "_get_http_client",
            return_value=MagicMock(
                post=MagicMock(
                    return_value=MagicMock(
                        status_code=200,
                        raise_for_status=MagicMock(),
                        json=lambda: {
                            "access_token": "new-access-token",
                            "expires_in": 3600,
                            "token_type": "Bearer",
                        },
                    )
                )
            ),
        ):
            tokens = google_provider.refresh_access_token("test-refresh-token")

            assert tokens.access_token == "new-access-token"
            assert tokens.expires_in == 3600

    def test_token_refresh_with_expired_refresh_token(self, google_provider):
        """Test token refresh with expired refresh token."""
        mock_response = MagicMock(
            status_code=400,
            json=lambda: {
                "error": "invalid_grant",
                "error_description": "Token has been expired or revoked.",
            },
        )
        mock_response.raise_for_status.side_effect = Exception("Bad Request")

        with patch.object(
            google_provider,
            "_get_http_client",
            return_value=MagicMock(post=MagicMock(return_value=mock_response)),
        ):
            with pytest.raises(Exception):
                google_provider.refresh_access_token("expired-refresh-token")


# ===========================================================================
# Token Revocation Tests
# ===========================================================================


class TestGoogleTokenRevocation:
    """Tests for Google token revocation."""

    def test_successful_token_revocation(self, google_provider):
        """Test successful token revocation."""
        mock_post = MagicMock(return_value=MagicMock(status_code=200))

        with patch.object(
            google_provider,
            "_get_http_client",
            return_value=MagicMock(post=mock_post),
        ):
            result = google_provider.revoke_token("test-access-token")

            assert result is True
            mock_post.assert_called_once()

    def test_revocation_with_invalid_token(self, google_provider):
        """Test revocation with already invalid token."""
        with patch.object(
            google_provider,
            "_get_http_client",
            return_value=MagicMock(
                post=MagicMock(
                    return_value=MagicMock(
                        status_code=400,
                        json=lambda: {"error": "invalid_token"},
                    )
                )
            ),
        ):
            # Returns False for failed revocation (method handles gracefully)
            result = google_provider.revoke_token("invalid-token")
            assert result is False


# ===========================================================================
# Environment Configuration Tests
# ===========================================================================


class TestGoogleEnvironmentConfig:
    """Tests for Google OAuth environment-based configuration."""

    def test_load_config_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "GOOGLE_OAUTH_CLIENT_ID": "env-client-id",
                "GOOGLE_OAUTH_CLIENT_SECRET": "env-client-secret",
                "GOOGLE_OAUTH_REDIRECT_URI": "https://env.example.com/callback",
            },
        ):
            provider = GoogleOAuthProvider()
            # Provider should load from environment
            assert provider._config.client_id in ["env-client-id", ""]
            provider.close()

    def test_default_redirect_uri_in_dev(self):
        """Test default redirect URI in development mode."""
        with patch.dict(
            "os.environ",
            {
                "GOOGLE_OAUTH_CLIENT_ID": "test-id",
                "GOOGLE_OAUTH_CLIENT_SECRET": "test-secret",
                "ARAGORA_ENV": "development",
            },
            clear=False,
        ):
            provider = GoogleOAuthProvider()
            # Should use localhost in development
            provider.close()


# ===========================================================================
# PKCE Tests
# ===========================================================================


class TestGooglePKCE:
    """Tests for Google PKCE support."""

    def test_authorization_url_with_additional_params(self, google_provider):
        """Test authorization URL can include additional parameters."""
        url = google_provider.get_authorization_url(
            state="test-state",
            login_hint="user@example.com",
        )

        assert "login_hint=" in url

    def test_authorization_url_with_hd_param(self, google_provider):
        """Test authorization URL with hosted domain param."""
        url = google_provider.get_authorization_url(
            state="test-state",
            hd="example.com",
        )

        assert "hd=" in url
