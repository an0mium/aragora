"""
Tests for Microsoft OAuth Provider (Azure AD).

Tests cover:
- Authorization URL generation (single and multi-tenant)
- Token exchange
- User info retrieval from Microsoft Graph
- Token refresh
- Tenant configuration
- Configuration validation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.oauth_providers.microsoft import (
    MicrosoftOAuthProvider,
    MICROSOFT_AUTH_URL_TEMPLATE,
    MICROSOFT_TOKEN_URL_TEMPLATE,
    MICROSOFT_USERINFO_URL,
)
from aragora.server.handlers.oauth_providers.base import (
    OAuthProviderConfig,
    OAuthTokens,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def microsoft_config():
    """Create a minimal Microsoft OAuth configuration."""
    return OAuthProviderConfig(
        client_id="test-app-id-guid",
        client_secret="test-client-secret",
        redirect_uri="https://example.com/callback",
        scopes=["openid", "email", "profile", "User.Read"],
        authorization_endpoint=MICROSOFT_AUTH_URL_TEMPLATE.format(tenant="common"),
        token_endpoint=MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant="common"),
        userinfo_endpoint=MICROSOFT_USERINFO_URL,
        tenant="common",
    )


@pytest.fixture
def single_tenant_config():
    """Create a single-tenant Microsoft OAuth configuration."""
    tenant_id = "12345678-1234-1234-1234-123456789abc"
    return OAuthProviderConfig(
        client_id="test-app-id-guid",
        client_secret="test-client-secret",
        redirect_uri="https://example.com/callback",
        scopes=["openid", "email", "profile", "User.Read"],
        authorization_endpoint=MICROSOFT_AUTH_URL_TEMPLATE.format(tenant=tenant_id),
        token_endpoint=MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant=tenant_id),
        userinfo_endpoint=MICROSOFT_USERINFO_URL,
        tenant=tenant_id,
    )


@pytest.fixture
def microsoft_provider(microsoft_config):
    """Create a Microsoft OAuth provider with test configuration."""
    provider = MicrosoftOAuthProvider(config=microsoft_config)
    yield provider
    provider.close()


@pytest.fixture
def single_tenant_provider(single_tenant_config):
    """Create a single-tenant Microsoft OAuth provider."""
    provider = MicrosoftOAuthProvider(config=single_tenant_config)
    yield provider
    provider.close()


@pytest.fixture
def mock_tokens():
    """Create mock OAuth tokens."""
    return OAuthTokens(
        access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.mock-access-token",
        refresh_token="M.R3_BAY.mock-refresh-token",
        expires_in=3600,
        token_type="Bearer",
        scope="openid email profile User.Read",
    )


@pytest.fixture
def mock_graph_user():
    """Create mock Microsoft Graph user response."""
    return {
        "id": "abc12345-def6-7890-ghij-klmnopqrstuv",
        "displayName": "Test User",
        "givenName": "Test",
        "surname": "User",
        "mail": "user@example.com",
        "userPrincipalName": "user@example.onmicrosoft.com",
        "jobTitle": "Developer",
        "officeLocation": "Building A",
        "preferredLanguage": "en-US",
    }


# ===========================================================================
# Configuration Tests
# ===========================================================================


class TestMicrosoftOAuthConfiguration:
    """Tests for Microsoft OAuth provider configuration."""

    def test_provider_name(self, microsoft_provider):
        """Test provider name is correct."""
        assert microsoft_provider.PROVIDER_NAME == "microsoft"

    def test_is_configured_with_valid_config(self, microsoft_provider):
        """Test is_configured returns True with valid config."""
        assert microsoft_provider.is_configured

    def test_is_configured_without_client_id(self):
        """Test is_configured returns False without client_id."""
        config = OAuthProviderConfig(
            client_id="",
            client_secret="secret",
            redirect_uri="https://example.com/callback",
        )
        provider = MicrosoftOAuthProvider(config=config)
        assert not provider.is_configured
        provider.close()

    def test_default_tenant_is_common(self, microsoft_provider):
        """Test default tenant is 'common' for multi-tenant apps."""
        assert microsoft_provider.tenant == "common"

    def test_single_tenant_configuration(self, single_tenant_provider):
        """Test single-tenant configuration."""
        assert single_tenant_provider.tenant == "12345678-1234-1234-1234-123456789abc"


# ===========================================================================
# Authorization URL Tests
# ===========================================================================


class TestMicrosoftAuthorizationUrl:
    """Tests for Microsoft OAuth authorization URL generation."""

    def test_basic_authorization_url(self, microsoft_provider):
        """Test basic authorization URL generation."""
        url = microsoft_provider.get_authorization_url(state="test-state")

        assert "login.microsoftonline.com" in url
        assert "/common/" in url
        assert "client_id=" in url
        assert "state=test-state" in url
        assert "response_type=code" in url

    def test_authorization_url_with_scopes(self, microsoft_provider):
        """Test authorization URL includes scopes."""
        url = microsoft_provider.get_authorization_url(
            state="test-state",
            scopes=["openid", "email", "profile", "User.Read"],
        )

        assert "scope=" in url
        assert "User.Read" in url or "User.Read".lower() in url.lower()

    def test_authorization_url_with_custom_redirect(self, microsoft_provider):
        """Test authorization URL with custom redirect URI."""
        custom_redirect = "https://custom.example.com/oauth/callback"
        url = microsoft_provider.get_authorization_url(
            state="test-state",
            redirect_uri=custom_redirect,
        )

        assert "redirect_uri=" in url

    def test_single_tenant_authorization_url(self, single_tenant_provider):
        """Test authorization URL uses specific tenant."""
        url = single_tenant_provider.get_authorization_url(state="test-state")

        assert "12345678-1234-1234-1234-123456789abc" in url
        assert "/common/" not in url

    def test_authorization_url_with_domain_hint(self, microsoft_provider):
        """Test authorization URL with domain hint."""
        url = microsoft_provider.get_authorization_url(
            state="test-state",
            domain_hint="example.com",
        )

        assert "domain_hint=" in url

    def test_authorization_url_with_login_hint(self, microsoft_provider):
        """Test authorization URL with login hint."""
        url = microsoft_provider.get_authorization_url(
            state="test-state",
            login_hint="user@example.com",
        )

        assert "login_hint=" in url

    def test_authorization_url_with_prompt(self, microsoft_provider):
        """Test authorization URL with prompt parameter."""
        url = microsoft_provider.get_authorization_url(
            state="test-state",
            prompt="select_account",
        )

        assert "prompt=select_account" in url


# ===========================================================================
# Token Exchange Tests
# ===========================================================================


class TestMicrosoftTokenExchange:
    """Tests for Microsoft OAuth token exchange."""

    def test_successful_token_exchange(self, microsoft_provider, mock_tokens):
        """Test successful token exchange."""
        with patch.object(
            microsoft_provider,
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
            tokens = microsoft_provider.exchange_code("test-auth-code")

            assert tokens.access_token == mock_tokens.access_token
            assert tokens.refresh_token == mock_tokens.refresh_token
            assert tokens.token_type == "Bearer"

    def test_token_exchange_with_invalid_code(self, microsoft_provider):
        """Test token exchange with invalid authorization code."""
        mock_response = MagicMock(
            status_code=400,
            json=lambda: {
                "error": "invalid_grant",
                "error_description": "AADSTS70000: The provided authorization code is invalid.",
                "error_codes": [70000],
            },
        )
        mock_response.raise_for_status.side_effect = Exception("Bad Request")

        with patch.object(
            microsoft_provider,
            "_get_http_client",
            return_value=MagicMock(post=MagicMock(return_value=mock_response)),
        ):
            with pytest.raises(Exception):
                microsoft_provider.exchange_code("invalid-code")

    def test_token_exchange_with_custom_redirect(self, microsoft_provider, mock_tokens):
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
            microsoft_provider,
            "_get_http_client",
            return_value=MagicMock(post=mock_post),
        ):
            microsoft_provider.exchange_code(
                "test-code",
                redirect_uri="https://custom.example.com/callback",
            )

            mock_post.assert_called_once()


# ===========================================================================
# User Info Tests
# ===========================================================================


class TestMicrosoftUserInfo:
    """Tests for Microsoft Graph user info retrieval."""

    def test_successful_user_info_retrieval(self, microsoft_provider, mock_tokens, mock_graph_user):
        """Test successful user info retrieval from Microsoft Graph."""
        with patch.object(
            microsoft_provider,
            "_get_http_client",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(
                        status_code=200,
                        raise_for_status=MagicMock(),
                        json=lambda: mock_graph_user,
                    )
                )
            ),
        ):
            user_info = microsoft_provider.get_user_info(mock_tokens.access_token)

            assert user_info.provider_user_id == "abc12345-def6-7890-ghij-klmnopqrstuv"
            assert user_info.email == "user@example.com"
            assert user_info.name == "Test User"

    def test_user_info_with_invalid_token(self, microsoft_provider):
        """Test user info retrieval with invalid token."""
        mock_response = MagicMock(
            status_code=401,
            json=lambda: {
                "error": {
                    "code": "InvalidAuthenticationToken",
                    "message": "Access token is empty.",
                }
            },
        )
        mock_response.raise_for_status.side_effect = Exception("Unauthorized")

        with patch.object(
            microsoft_provider,
            "_get_http_client",
            return_value=MagicMock(get=MagicMock(return_value=mock_response)),
        ):
            with pytest.raises(Exception):
                microsoft_provider.get_user_info("invalid-token")

    def test_user_info_uses_upn_when_mail_missing(self, microsoft_provider, mock_tokens):
        """Test fallback to userPrincipalName when mail is missing."""
        user_without_mail = {
            "id": "user-id",
            "displayName": "Test User",
            "userPrincipalName": "user@example.onmicrosoft.com",
            # No 'mail' field
        }
        with patch.object(
            microsoft_provider,
            "_get_http_client",
            return_value=MagicMock(
                get=MagicMock(
                    return_value=MagicMock(
                        status_code=200,
                        raise_for_status=MagicMock(),
                        json=lambda: user_without_mail,
                    )
                )
            ),
        ):
            user_info = microsoft_provider.get_user_info(mock_tokens.access_token)

            # Should fall back to userPrincipalName
            assert user_info.email == "user@example.onmicrosoft.com"


# ===========================================================================
# Token Refresh Tests
# ===========================================================================


class TestMicrosoftTokenRefresh:
    """Tests for Microsoft token refresh."""

    def test_successful_token_refresh(self, microsoft_provider):
        """Test successful token refresh."""
        with patch.object(
            microsoft_provider,
            "_get_http_client",
            return_value=MagicMock(
                post=MagicMock(
                    return_value=MagicMock(
                        status_code=200,
                        raise_for_status=MagicMock(),
                        json=lambda: {
                            "access_token": "new-access-token",
                            "refresh_token": "new-refresh-token",
                            "expires_in": 3600,
                            "token_type": "Bearer",
                        },
                    )
                )
            ),
        ):
            tokens = microsoft_provider.refresh_access_token("test-refresh-token")

            assert tokens.access_token == "new-access-token"
            assert tokens.refresh_token == "new-refresh-token"

    def test_token_refresh_with_expired_refresh_token(self, microsoft_provider):
        """Test token refresh with expired refresh token."""
        mock_response = MagicMock(
            status_code=400,
            json=lambda: {
                "error": "invalid_grant",
                "error_description": "AADSTS700082: The refresh token has expired.",
                "error_codes": [700082],
            },
        )
        mock_response.raise_for_status.side_effect = Exception("Bad Request")

        with patch.object(
            microsoft_provider,
            "_get_http_client",
            return_value=MagicMock(post=MagicMock(return_value=mock_response)),
        ):
            with pytest.raises(Exception):
                microsoft_provider.refresh_access_token("expired-refresh-token")


# ===========================================================================
# Multi-Tenant Tests
# ===========================================================================


class TestMicrosoftMultiTenant:
    """Tests for Microsoft multi-tenant scenarios."""

    def test_common_tenant_allows_all_accounts(self, microsoft_provider):
        """Test 'common' tenant allows all Microsoft accounts."""
        url = microsoft_provider.get_authorization_url(state="test-state")
        assert "/common/" in url

    def test_organizations_tenant_allows_work_accounts(self):
        """Test 'organizations' tenant for work accounts only."""
        config = OAuthProviderConfig(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="https://example.com/callback",
            authorization_endpoint=MICROSOFT_AUTH_URL_TEMPLATE.format(tenant="organizations"),
            token_endpoint=MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant="organizations"),
            tenant="organizations",
        )
        provider = MicrosoftOAuthProvider(config=config)
        url = provider.get_authorization_url(state="test-state")

        assert "/organizations/" in url
        provider.close()

    def test_consumers_tenant_allows_personal_accounts(self):
        """Test 'consumers' tenant for personal accounts only."""
        config = OAuthProviderConfig(
            client_id="test-id",
            client_secret="test-secret",
            redirect_uri="https://example.com/callback",
            authorization_endpoint=MICROSOFT_AUTH_URL_TEMPLATE.format(tenant="consumers"),
            token_endpoint=MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant="consumers"),
            tenant="consumers",
        )
        provider = MicrosoftOAuthProvider(config=config)
        url = provider.get_authorization_url(state="test-state")

        assert "/consumers/" in url
        provider.close()


# ===========================================================================
# Environment Configuration Tests
# ===========================================================================


class TestMicrosoftEnvironmentConfig:
    """Tests for Microsoft OAuth environment-based configuration."""

    def test_load_config_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "MICROSOFT_OAUTH_CLIENT_ID": "env-client-id",
                "MICROSOFT_OAUTH_CLIENT_SECRET": "env-client-secret",
                "MICROSOFT_OAUTH_TENANT": "env-tenant-id",
                "MICROSOFT_OAUTH_REDIRECT_URI": "https://env.example.com/callback",
            },
        ):
            provider = MicrosoftOAuthProvider()
            # Provider should attempt to load from environment
            provider.close()

    def test_default_tenant_is_common(self):
        """Test default tenant is 'common' when not specified."""
        with patch.dict(
            "os.environ",
            {
                "MICROSOFT_OAUTH_CLIENT_ID": "test-id",
                "MICROSOFT_OAUTH_CLIENT_SECRET": "test-secret",
            },
            clear=False,
        ):
            provider = MicrosoftOAuthProvider()
            assert provider.tenant == "common"
            provider.close()


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestMicrosoftErrorHandling:
    """Tests for Microsoft OAuth error handling."""

    def test_interaction_required_error(self, microsoft_provider):
        """Test handling of interaction_required error."""
        mock_response = MagicMock(
            status_code=400,
            json=lambda: {
                "error": "interaction_required",
                "error_description": "AADSTS50076: User needs MFA.",
                "error_codes": [50076],
            },
        )
        mock_response.raise_for_status.side_effect = Exception("Bad Request")

        with patch.object(
            microsoft_provider,
            "_get_http_client",
            return_value=MagicMock(post=MagicMock(return_value=mock_response)),
        ):
            with pytest.raises(Exception):
                microsoft_provider.refresh_access_token("test-token")

    def test_consent_required_error(self, microsoft_provider):
        """Test handling of consent_required error."""
        mock_response = MagicMock(
            status_code=400,
            json=lambda: {
                "error": "consent_required",
                "error_description": "AADSTS65001: User hasn't consented.",
                "error_codes": [65001],
            },
        )
        mock_response.raise_for_status.side_effect = Exception("Bad Request")

        with patch.object(
            microsoft_provider,
            "_get_http_client",
            return_value=MagicMock(post=MagicMock(return_value=mock_response)),
        ):
            with pytest.raises(Exception):
                microsoft_provider.exchange_code("test-code")
