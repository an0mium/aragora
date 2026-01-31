"""
Tests for OAuth providers - base, Apple, OIDC, and GitHub.

Tests:
- Base OAuthProvider configuration and abstract methods
- AppleOAuthProvider JWT generation, ID token verification, JWKS caching
- OIDCProvider discovery document handling and user info retrieval
- GitHubOAuthProvider authorization and email fallback
- Error handling across all providers
"""

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from aragora.server.handlers.oauth_providers.base import (
    OAuthProvider,
    OAuthProviderConfig,
    OAuthTokens,
    OAuthUserInfo,
)
from aragora.server.handlers.oauth_providers.apple import AppleOAuthProvider
from aragora.server.handlers.oauth_providers.github import GitHubOAuthProvider
from aragora.server.handlers.oauth_providers.oidc import OIDCProvider


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def base_config() -> OAuthProviderConfig:
    """Create a basic OAuth provider configuration."""
    return OAuthProviderConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",
        redirect_uri="http://localhost:8080/callback",
        scopes=["openid", "email", "profile"],
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        userinfo_endpoint="https://auth.example.com/userinfo",
    )


@pytest.fixture
def apple_config() -> OAuthProviderConfig:
    """Create Apple OAuth provider configuration with test keys."""
    # ==========================================================================
    # SECURITY NOTICE: TEST-ONLY PRIVATE KEY
    # ==========================================================================
    # This is a deliberately generated test key for unit testing Apple OAuth
    # JWT signing functionality. It is NOT a real Apple Developer key and has
    # no access to any production systems.
    #
    # Generated via: openssl ecparam -name prime256v1 -genkey -noout
    # This key is committed intentionally for reproducible tests.
    # ==========================================================================
    test_private_key = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg5kjhE+0CG0iSuMAC
rpDuM/ER5LtkvGtyxDg31WjQ762hRANCAARISNqgPPI84MqzPNFjTjGrnoV3JcYJ
h2fXcddHRS6rlFfgICDzpYnitlrneZW7J1V0p0/8NZgQoelIb80RY6b5
-----END PRIVATE KEY-----"""

    return OAuthProviderConfig(
        client_id="com.example.app.web",
        client_secret="",  # Apple uses JWT
        redirect_uri="http://localhost:8080/api/auth/oauth/apple/callback",
        scopes=["name", "email"],
        team_id="TEAMID1234",
        key_id="KEYID12345",
        private_key=test_private_key,
    )


@pytest.fixture
def github_config() -> OAuthProviderConfig:
    """Create GitHub OAuth provider configuration."""
    return OAuthProviderConfig(
        client_id="github_client_id",
        client_secret="github_client_secret",
        redirect_uri="http://localhost:8080/api/auth/oauth/github/callback",
        scopes=["read:user", "user:email"],
    )


@pytest.fixture
def oidc_config() -> OAuthProviderConfig:
    """Create OIDC provider configuration."""
    return OAuthProviderConfig(
        client_id="oidc_client_id",
        client_secret="oidc_client_secret",
        redirect_uri="http://localhost:8080/api/auth/oauth/oidc/callback",
        scopes=["openid", "email", "profile"],
    )


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client."""
    return MagicMock(spec=httpx.Client)


def create_mock_jwt(claims: dict[str, Any], header: Optional[dict[str, str]] = None) -> str:
    """Create a mock JWT token for testing."""
    if header is None:
        header = {"alg": "RS256", "kid": "test_key_id"}

    # Encode header
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")

    # Encode payload
    payload_b64 = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")

    # Mock signature
    signature_b64 = base64.urlsafe_b64encode(b"mock_signature").decode().rstrip("=")

    return f"{header_b64}.{payload_b64}.{signature_b64}"


# =============================================================================
# Base OAuthProvider Tests
# =============================================================================


class TestOAuthProviderConfig:
    """Tests for OAuthProviderConfig dataclass."""

    def test_config_initialization_with_required_fields(self):
        """Should initialize with required fields."""
        config = OAuthProviderConfig(
            client_id="client_id",
            client_secret="client_secret",
            redirect_uri="http://localhost/callback",
        )
        assert config.client_id == "client_id"
        assert config.client_secret == "client_secret"
        assert config.redirect_uri == "http://localhost/callback"

    def test_config_default_values(self):
        """Should have correct default values."""
        config = OAuthProviderConfig(
            client_id="client_id",
            client_secret="client_secret",
            redirect_uri="http://localhost/callback",
        )
        assert config.scopes == []
        assert config.authorization_endpoint == ""
        assert config.token_endpoint == ""
        assert config.userinfo_endpoint == ""
        assert config.tenant is None
        assert config.team_id is None
        assert config.key_id is None
        assert config.private_key is None

    def test_config_with_all_fields(self, base_config):
        """Should correctly store all configuration fields."""
        assert base_config.scopes == ["openid", "email", "profile"]
        assert base_config.authorization_endpoint == "https://auth.example.com/authorize"
        assert base_config.token_endpoint == "https://auth.example.com/token"


class TestOAuthTokens:
    """Tests for OAuthTokens dataclass."""

    def test_tokens_from_dict_minimal(self):
        """Should create tokens from minimal dict."""
        data = {"access_token": "test_access_token"}
        tokens = OAuthTokens.from_dict(data)
        assert tokens.access_token == "test_access_token"
        assert tokens.token_type == "Bearer"
        assert tokens.expires_in is None
        assert tokens.refresh_token is None

    def test_tokens_from_dict_full(self):
        """Should create tokens from full dict."""
        data = {
            "access_token": "test_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "test_refresh_token",
            "scope": "openid email",
            "id_token": "test_id_token",
        }
        tokens = OAuthTokens.from_dict(data)
        assert tokens.access_token == "test_access_token"
        assert tokens.token_type == "Bearer"
        assert tokens.expires_in == 3600
        assert tokens.refresh_token == "test_refresh_token"
        assert tokens.scope == "openid email"
        assert tokens.id_token == "test_id_token"

    def test_tokens_from_dict_custom_token_type(self):
        """Should preserve custom token type."""
        data = {"access_token": "token", "token_type": "MAC"}
        tokens = OAuthTokens.from_dict(data)
        assert tokens.token_type == "MAC"


class TestOAuthUserInfo:
    """Tests for OAuthUserInfo dataclass."""

    def test_user_info_minimal(self):
        """Should create user info with minimal data."""
        info = OAuthUserInfo(
            provider="test",
            provider_user_id="user123",
        )
        assert info.provider == "test"
        assert info.provider_user_id == "user123"
        assert info.email is None
        assert info.email_verified is False

    def test_user_info_full(self):
        """Should create user info with all fields."""
        info = OAuthUserInfo(
            provider="google",
            provider_user_id="123456",
            email="user@example.com",
            email_verified=True,
            name="Test User",
            given_name="Test",
            family_name="User",
            picture="https://example.com/avatar.jpg",
            locale="en-US",
            raw_data={"custom": "data"},
        )
        assert info.email == "user@example.com"
        assert info.email_verified is True
        assert info.name == "Test User"
        assert info.raw_data == {"custom": "data"}


class TestOAuthProviderBase:
    """Tests for abstract OAuthProvider base class."""

    def test_config_property(self, base_config):
        """Should expose config via property."""

        class ConcreteProvider(OAuthProvider):
            PROVIDER_NAME = "test"

            def _load_config_from_env(self):
                return base_config

            def get_authorization_url(self, state, redirect_uri=None, scopes=None, **kwargs):
                return ""

            def exchange_code(self, code, redirect_uri=None):
                return OAuthTokens(access_token="test")

            def get_user_info(self, access_token):
                return OAuthUserInfo(provider="test", provider_user_id="123")

        provider = ConcreteProvider(base_config)
        assert provider.config == base_config

    def test_is_configured_true(self, base_config):
        """Should return True when client_id and client_secret are set."""

        class ConcreteProvider(OAuthProvider):
            PROVIDER_NAME = "test"

            def _load_config_from_env(self):
                return base_config

            def get_authorization_url(self, state, redirect_uri=None, scopes=None, **kwargs):
                return ""

            def exchange_code(self, code, redirect_uri=None):
                return OAuthTokens(access_token="test")

            def get_user_info(self, access_token):
                return OAuthUserInfo(provider="test", provider_user_id="123")

        provider = ConcreteProvider(base_config)
        assert provider.is_configured is True

    def test_is_configured_false_missing_client_id(self):
        """Should return False when client_id is missing."""
        config = OAuthProviderConfig(
            client_id="",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
        )

        class ConcreteProvider(OAuthProvider):
            PROVIDER_NAME = "test"

            def _load_config_from_env(self):
                return config

            def get_authorization_url(self, state, redirect_uri=None, scopes=None, **kwargs):
                return ""

            def exchange_code(self, code, redirect_uri=None):
                return OAuthTokens(access_token="test")

            def get_user_info(self, access_token):
                return OAuthUserInfo(provider="test", provider_user_id="123")

        provider = ConcreteProvider(config)
        assert provider.is_configured is False

    def test_context_manager(self, base_config):
        """Should support context manager protocol."""

        class ConcreteProvider(OAuthProvider):
            PROVIDER_NAME = "test"

            def _load_config_from_env(self):
                return base_config

            def get_authorization_url(self, state, redirect_uri=None, scopes=None, **kwargs):
                return ""

            def exchange_code(self, code, redirect_uri=None):
                return OAuthTokens(access_token="test")

            def get_user_info(self, access_token):
                return OAuthUserInfo(provider="test", provider_user_id="123")

        with ConcreteProvider(base_config) as provider:
            assert provider.PROVIDER_NAME == "test"

    def test_build_authorization_url(self, base_config):
        """Should correctly build authorization URL with query params."""

        class ConcreteProvider(OAuthProvider):
            PROVIDER_NAME = "test"

            def _load_config_from_env(self):
                return base_config

            def get_authorization_url(self, state, redirect_uri=None, scopes=None, **kwargs):
                return self._build_authorization_url(
                    base_config.authorization_endpoint,
                    {"client_id": base_config.client_id, "state": state},
                )

            def exchange_code(self, code, redirect_uri=None):
                return OAuthTokens(access_token="test")

            def get_user_info(self, access_token):
                return OAuthUserInfo(provider="test", provider_user_id="123")

        provider = ConcreteProvider(base_config)
        url = provider.get_authorization_url("test_state")
        assert "https://auth.example.com/authorize?" in url
        assert "client_id=test_client_id" in url
        assert "state=test_state" in url

    def test_refresh_access_token_not_implemented(self, base_config):
        """Should raise NotImplementedError for refresh by default."""

        class ConcreteProvider(OAuthProvider):
            PROVIDER_NAME = "test"

            def _load_config_from_env(self):
                return base_config

            def get_authorization_url(self, state, redirect_uri=None, scopes=None, **kwargs):
                return ""

            def exchange_code(self, code, redirect_uri=None):
                return OAuthTokens(access_token="test")

            def get_user_info(self, access_token):
                return OAuthUserInfo(provider="test", provider_user_id="123")

        provider = ConcreteProvider(base_config)
        with pytest.raises(NotImplementedError) as exc_info:
            provider.refresh_access_token("refresh_token")
        assert "test does not support token refresh" in str(exc_info.value)


# =============================================================================
# AppleOAuthProvider Tests
# =============================================================================


class TestAppleOAuthProviderInit:
    """Tests for AppleOAuthProvider initialization."""

    def test_provider_name(self, apple_config):
        """Should have correct provider name."""
        provider = AppleOAuthProvider(apple_config)
        assert provider.PROVIDER_NAME == "apple"

    def test_is_configured_true(self, apple_config):
        """Should be configured when all Apple fields are present."""
        provider = AppleOAuthProvider(apple_config)
        assert provider.is_configured is True

    def test_is_configured_false_missing_team_id(self, apple_config):
        """Should not be configured without team_id."""
        apple_config.team_id = ""
        provider = AppleOAuthProvider(apple_config)
        assert provider.is_configured is False

    def test_is_configured_false_missing_key_id(self, apple_config):
        """Should not be configured without key_id."""
        apple_config.key_id = ""
        provider = AppleOAuthProvider(apple_config)
        assert provider.is_configured is False

    def test_is_configured_false_missing_private_key(self, apple_config):
        """Should not be configured without private_key."""
        apple_config.private_key = ""
        provider = AppleOAuthProvider(apple_config)
        assert provider.is_configured is False


class TestAppleClientSecretGeneration:
    """Tests for Apple JWT client secret generation."""

    def test_generate_client_secret_returns_jwt(self, apple_config):
        """Should generate a valid JWT client secret."""
        provider = AppleOAuthProvider(apple_config)
        secret = provider._generate_client_secret()

        # JWT should have 3 parts
        parts = secret.split(".")
        assert len(parts) == 3

    def test_generate_client_secret_header(self, apple_config):
        """Should have correct JWT header with ES256 algorithm."""
        provider = AppleOAuthProvider(apple_config)
        secret = provider._generate_client_secret()

        # Decode header
        header_b64 = secret.split(".")[0]
        # Add padding
        padding = 4 - len(header_b64) % 4
        if padding != 4:
            header_b64 += "=" * padding
        header = json.loads(base64.urlsafe_b64decode(header_b64))

        assert header["alg"] == "ES256"
        assert header["kid"] == apple_config.key_id

    def test_generate_client_secret_payload(self, apple_config):
        """Should have correct JWT payload claims."""
        provider = AppleOAuthProvider(apple_config)
        secret = provider._generate_client_secret()

        # Decode payload
        payload_b64 = secret.split(".")[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        assert payload["iss"] == apple_config.team_id
        assert payload["sub"] == apple_config.client_id
        assert payload["aud"] == "https://appleid.apple.com"
        assert "iat" in payload
        assert "exp" in payload
        # Expiry should be ~180 days in the future
        assert payload["exp"] - payload["iat"] == 86400 * 180


class TestAppleAuthorizationUrl:
    """Tests for Apple authorization URL generation."""

    def test_authorization_url_basic(self, apple_config):
        """Should generate correct authorization URL."""
        provider = AppleOAuthProvider(apple_config)
        url = provider.get_authorization_url("test_state")

        assert url.startswith("https://appleid.apple.com/auth/authorize?")
        assert "client_id=com.example.app.web" in url
        assert "state=test_state" in url
        assert "response_type=code" in url
        assert "scope=name+email" in url or "scope=name%20email" in url
        assert "response_mode=form_post" in url

    def test_authorization_url_custom_redirect(self, apple_config):
        """Should use custom redirect URI."""
        provider = AppleOAuthProvider(apple_config)
        custom_uri = "https://custom.example.com/callback"
        url = provider.get_authorization_url("state", redirect_uri=custom_uri)

        assert "redirect_uri=https%3A%2F%2Fcustom.example.com%2Fcallback" in url

    def test_authorization_url_custom_scopes(self, apple_config):
        """Should use custom scopes."""
        provider = AppleOAuthProvider(apple_config)
        url = provider.get_authorization_url("state", scopes=["email"])

        assert "scope=email" in url

    def test_authorization_url_with_nonce(self, apple_config):
        """Should include nonce when provided."""
        provider = AppleOAuthProvider(apple_config)
        url = provider.get_authorization_url("state", nonce="test_nonce_123")

        assert "nonce=test_nonce_123" in url

    def test_authorization_url_custom_response_mode(self, apple_config):
        """Should support custom response mode."""
        provider = AppleOAuthProvider(apple_config)
        url = provider.get_authorization_url("state", response_mode="query")

        assert "response_mode=query" in url


class TestAppleCodeExchange:
    """Tests for Apple code exchange."""

    def test_exchange_code_calls_token_endpoint(self, apple_config, mock_http_client):
        """Should call Apple token endpoint with correct data."""
        provider = AppleOAuthProvider(apple_config)
        provider._http_client = mock_http_client

        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "apple_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "apple_refresh_token",
            "id_token": "apple_id_token",
        }
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        tokens = provider.exchange_code("auth_code_123")

        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "https://appleid.apple.com/auth/token"
        assert call_args[1]["data"]["code"] == "auth_code_123"
        assert call_args[1]["data"]["grant_type"] == "authorization_code"
        assert "client_secret" in call_args[1]["data"]

        assert tokens.access_token == "apple_access_token"
        assert tokens.id_token == "apple_id_token"


class TestAppleIdTokenVerification:
    """Tests for Apple ID token verification with mock JWKS."""

    def test_decode_id_token_unverified(self, apple_config):
        """Should decode ID token without verification."""
        claims = {
            "sub": "001234.abc123",
            "email": "user@privaterelay.appleid.com",
            "email_verified": True,
            "iss": "https://appleid.apple.com",
            "aud": "com.example.app.web",
        }
        id_token = create_mock_jwt(claims)

        provider = AppleOAuthProvider(apple_config)
        decoded = provider._decode_id_token(id_token, verify=False)

        assert decoded["sub"] == "001234.abc123"
        assert decoded["email"] == "user@privaterelay.appleid.com"

    def test_decode_id_token_invalid_format(self, apple_config):
        """Should raise ValueError for invalid token format."""
        provider = AppleOAuthProvider(apple_config)

        with pytest.raises(ValueError) as exc_info:
            provider._decode_id_token("not.a.valid.jwt.token", verify=False)
        assert "Invalid ID token format" in str(exc_info.value)

    def test_get_user_info_raises_value_error_for_opaque_token(self, apple_config):
        """Should raise ValueError for opaque access tokens (not JWTs).

        Apple doesn't have a userinfo endpoint. The method accepts ID tokens (JWTs)
        but rejects opaque access tokens with a helpful error message.
        """
        provider = AppleOAuthProvider(apple_config)

        # Opaque token (not a JWT - doesn't have 3 dot-separated parts)
        with pytest.raises(ValueError) as exc_info:
            provider.get_user_info("access_token")
        assert "userinfo endpoint" in str(exc_info.value)


class TestAppleUserInfoExtraction:
    """Tests for Apple user info extraction from callback."""

    def test_get_user_info_from_id_token_basic(self, apple_config):
        """Should extract user info from ID token."""
        claims = {
            "sub": "001234.abc123",
            "email": "user@example.com",
            "email_verified": True,
        }
        id_token = create_mock_jwt(claims)

        provider = AppleOAuthProvider(apple_config)
        with patch.object(provider, "_decode_id_token", return_value=claims):
            user_info = provider.get_user_info_from_id_token(id_token)

        assert user_info.provider == "apple"
        assert user_info.provider_user_id == "001234.abc123"
        assert user_info.email == "user@example.com"
        assert user_info.email_verified is True

    def test_get_user_info_from_id_token_with_user_data(self, apple_config):
        """Should include user data from first authorization."""
        claims = {"sub": "001234.abc123", "email": "user@example.com"}
        user_data = {"name": {"firstName": "John", "lastName": "Doe"}}

        provider = AppleOAuthProvider(apple_config)
        with patch.object(provider, "_decode_id_token", return_value=claims):
            user_info = provider.get_user_info_from_id_token("mock_id_token", user_data=user_data)

        assert user_info.given_name == "John"
        assert user_info.family_name == "Doe"
        assert user_info.name == "John Doe"

    def test_get_user_info_from_id_token_missing_sub(self, apple_config):
        """Should raise ValueError when sub is missing."""
        claims = {"email": "user@example.com"}

        provider = AppleOAuthProvider(apple_config)
        with patch.object(provider, "_decode_id_token", return_value=claims):
            with pytest.raises(ValueError) as exc_info:
                provider.get_user_info_from_id_token("mock_id_token")
            assert "No subject" in str(exc_info.value)

    def test_get_user_info_from_callback(self, apple_config):
        """Should extract user info from callback with tokens and user JSON."""
        claims = {"sub": "001234", "email": "user@example.com", "email_verified": "true"}
        tokens = OAuthTokens(access_token="access", id_token=create_mock_jwt(claims))
        user_json = '{"name": {"firstName": "Jane", "lastName": "Smith"}}'

        provider = AppleOAuthProvider(apple_config)
        with patch.object(provider, "_decode_id_token", return_value=claims):
            user_info = provider.get_user_info_from_callback(tokens, user_json)

        assert user_info.provider_user_id == "001234"
        assert user_info.given_name == "Jane"

    def test_get_user_info_from_callback_no_id_token(self, apple_config):
        """Should raise ValueError when no ID token in response."""
        tokens = OAuthTokens(access_token="access", id_token=None)

        provider = AppleOAuthProvider(apple_config)
        with pytest.raises(ValueError) as exc_info:
            provider.get_user_info_from_callback(tokens)
        assert "No ID token" in str(exc_info.value)


class TestAppleJwksCaching:
    """Tests for Apple JWKS caching behavior."""

    def test_jwks_cache_hit(self, apple_config, mock_http_client):
        """Should return cached JWKS when not expired."""
        # Set up cache
        AppleOAuthProvider._jwks_cache = {"keys": [{"kid": "test_key"}]}
        AppleOAuthProvider._jwks_cache_expiry = time.time() + 3600

        provider = AppleOAuthProvider(apple_config)
        provider._http_client = mock_http_client

        jwks = provider._fetch_apple_jwks()

        # Should not call HTTP client
        mock_http_client.get.assert_not_called()
        assert jwks["keys"][0]["kid"] == "test_key"

        # Clean up
        AppleOAuthProvider._jwks_cache = None
        AppleOAuthProvider._jwks_cache_expiry = 0

    def test_jwks_cache_miss(self, apple_config, mock_http_client):
        """Should fetch JWKS when cache expired."""
        # Expire cache
        AppleOAuthProvider._jwks_cache = None
        AppleOAuthProvider._jwks_cache_expiry = 0

        mock_response = Mock()
        mock_response.json.return_value = {"keys": [{"kid": "new_key"}]}
        mock_response.raise_for_status = Mock()
        mock_http_client.get.return_value = mock_response

        provider = AppleOAuthProvider(apple_config)
        provider._http_client = mock_http_client

        jwks = provider._fetch_apple_jwks()

        mock_http_client.get.assert_called_once_with("https://appleid.apple.com/auth/keys")
        assert jwks["keys"][0]["kid"] == "new_key"

        # Clean up
        AppleOAuthProvider._jwks_cache = None
        AppleOAuthProvider._jwks_cache_expiry = 0

    def test_jwks_fetch_failure_uses_expired_cache(self, apple_config, mock_http_client):
        """Should use expired cache as fallback on fetch failure."""
        # Set expired cache
        AppleOAuthProvider._jwks_cache = {"keys": [{"kid": "old_key"}]}
        AppleOAuthProvider._jwks_cache_expiry = time.time() - 3600

        mock_http_client.get.side_effect = httpx.ConnectError("Network error")

        provider = AppleOAuthProvider(apple_config)
        provider._http_client = mock_http_client

        jwks = provider._fetch_apple_jwks()

        assert jwks["keys"][0]["kid"] == "old_key"

        # Clean up
        AppleOAuthProvider._jwks_cache = None
        AppleOAuthProvider._jwks_cache_expiry = 0


class TestAppleTokenRefresh:
    """Tests for Apple token refresh."""

    def test_refresh_access_token(self, apple_config, mock_http_client):
        """Should refresh access token with correct parameters."""
        provider = AppleOAuthProvider(apple_config)
        provider._http_client = mock_http_client

        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        tokens = provider.refresh_access_token("refresh_token_123")

        call_args = mock_http_client.post.call_args
        assert call_args[1]["data"]["refresh_token"] == "refresh_token_123"
        assert call_args[1]["data"]["grant_type"] == "refresh_token"
        assert tokens.access_token == "new_access_token"


# =============================================================================
# OIDCProvider Tests
# =============================================================================


class TestOIDCProviderInit:
    """Tests for OIDCProvider initialization."""

    def test_provider_name(self, oidc_config):
        """Should have correct provider name."""
        provider = OIDCProvider(oidc_config)
        assert provider.PROVIDER_NAME == "oidc"

    def test_discovery_not_cached_initially(self, oidc_config):
        """Should not have discovery cached on init."""
        provider = OIDCProvider(oidc_config)
        assert provider._discovery is None


class TestOIDCDiscoveryDocument:
    """Tests for OIDC discovery document fetch and caching."""

    def test_discovery_fetch(self, oidc_config, mock_http_client):
        """Should fetch discovery document from issuer."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "authorization_endpoint": "https://idp.example.com/authorize",
            "token_endpoint": "https://idp.example.com/token",
            "userinfo_endpoint": "https://idp.example.com/userinfo",
        }
        mock_response.raise_for_status = Mock()
        mock_http_client.get.return_value = mock_response

        provider = OIDCProvider(oidc_config)
        provider._http_client = mock_http_client

        with patch(
            "aragora.server.handlers.oauth_providers.oidc._get_secret",
            return_value="https://idp.example.com",
        ):
            discovery = provider._get_discovery()

        mock_http_client.get.assert_called_once_with(
            "https://idp.example.com/.well-known/openid-configuration"
        )
        assert discovery["authorization_endpoint"] == "https://idp.example.com/authorize"

    def test_discovery_cached(self, oidc_config, mock_http_client):
        """Should cache discovery document."""
        provider = OIDCProvider(oidc_config)
        provider._discovery = {"cached": True}
        provider._http_client = mock_http_client

        discovery = provider._get_discovery()

        mock_http_client.get.assert_not_called()
        assert discovery["cached"] is True

    def test_discovery_no_issuer(self, oidc_config):
        """Should raise ValueError when issuer not configured."""
        provider = OIDCProvider(oidc_config)

        with patch(
            "aragora.server.handlers.oauth_providers.oidc._get_secret",
            return_value="",
        ):
            with pytest.raises(ValueError) as exc_info:
                provider._get_discovery()
            assert "OIDC_ISSUER not configured" in str(exc_info.value)

    def test_discovery_fetch_failure(self, oidc_config, mock_http_client):
        """Should raise ValueError on fetch failure."""
        mock_http_client.get.side_effect = httpx.ConnectError("Network error")

        provider = OIDCProvider(oidc_config)
        provider._http_client = mock_http_client

        with patch(
            "aragora.server.handlers.oauth_providers.oidc._get_secret",
            return_value="https://idp.example.com",
        ):
            with pytest.raises(ValueError) as exc_info:
                provider._get_discovery()
            assert "Failed to fetch OIDC discovery" in str(exc_info.value)


class TestOIDCAuthorizationUrl:
    """Tests for OIDC authorization URL generation from discovery."""

    def test_authorization_url_from_discovery(self, oidc_config):
        """Should generate URL using discovery endpoint."""
        provider = OIDCProvider(oidc_config)
        provider._discovery = {"authorization_endpoint": "https://idp.example.com/authorize"}

        url = provider.get_authorization_url("test_state")

        assert url.startswith("https://idp.example.com/authorize?")
        assert "client_id=oidc_client_id" in url
        assert "state=test_state" in url

    def test_authorization_url_no_endpoint_in_discovery(self, oidc_config):
        """Should raise ValueError when discovery lacks authorization_endpoint."""
        provider = OIDCProvider(oidc_config)
        provider._discovery = {}

        with pytest.raises(ValueError) as exc_info:
            provider.get_authorization_url("state")
        assert "No authorization_endpoint" in str(exc_info.value)

    def test_authorization_url_with_optional_params(self, oidc_config):
        """Should include optional OIDC parameters."""
        provider = OIDCProvider(oidc_config)
        provider._discovery = {"authorization_endpoint": "https://idp.example.com/authorize"}

        url = provider.get_authorization_url(
            "state",
            nonce="test_nonce",
            prompt="login",
            login_hint="user@example.com",
        )

        assert "nonce=test_nonce" in url
        assert "prompt=login" in url
        assert "login_hint=user%40example.com" in url


class TestOIDCCodeExchange:
    """Tests for OIDC code exchange."""

    def test_exchange_code_from_discovery(self, oidc_config, mock_http_client):
        """Should use token endpoint from discovery."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "oidc_access_token",
            "id_token": "oidc_id_token",
        }
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        provider = OIDCProvider(oidc_config)
        provider._discovery = {"token_endpoint": "https://idp.example.com/token"}
        provider._http_client = mock_http_client

        tokens = provider.exchange_code("auth_code")

        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "https://idp.example.com/token"
        assert tokens.access_token == "oidc_access_token"

    def test_exchange_code_no_endpoint(self, oidc_config):
        """Should raise ValueError when discovery lacks token_endpoint."""
        provider = OIDCProvider(oidc_config)
        provider._discovery = {}

        with pytest.raises(ValueError) as exc_info:
            provider.exchange_code("code")
        assert "No token_endpoint" in str(exc_info.value)


class TestOIDCUserInfo:
    """Tests for OIDC user info retrieval."""

    def test_get_user_info_from_endpoint(self, oidc_config, mock_http_client):
        """Should fetch user info from userinfo endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "sub": "user123",
            "email": "user@example.com",
            "email_verified": True,
            "name": "Test User",
        }
        mock_http_client.get.return_value = mock_response

        provider = OIDCProvider(oidc_config)
        provider._discovery = {"userinfo_endpoint": "https://idp.example.com/userinfo"}
        provider._http_client = mock_http_client

        user_info = provider.get_user_info("access_token")

        assert user_info.provider_user_id == "user123"
        assert user_info.email == "user@example.com"
        assert user_info.name == "Test User"

    def test_get_user_info_no_endpoint(self, oidc_config):
        """Should raise ValueError when no userinfo endpoint."""
        provider = OIDCProvider(oidc_config)
        provider._discovery = {}

        with pytest.raises(ValueError) as exc_info:
            provider.get_user_info("access_token")
        assert "No userinfo_endpoint" in str(exc_info.value)

    def test_get_user_info_from_id_token(self, oidc_config):
        """Should extract user info from ID token."""
        claims = {
            "sub": "id_token_user",
            "email": "idtoken@example.com",
            "email_verified": True,
            "given_name": "ID",
            "family_name": "Token",
        }
        id_token = create_mock_jwt(claims)

        provider = OIDCProvider(oidc_config)
        user_info = provider.get_user_info_from_id_token(id_token)

        assert user_info.provider_user_id == "id_token_user"
        assert user_info.email == "idtoken@example.com"
        assert user_info.given_name == "ID"

    def test_get_user_info_missing_sub(self, oidc_config):
        """Should raise ValueError when sub is missing."""
        claims = {"email": "user@example.com"}
        id_token = create_mock_jwt(claims)

        provider = OIDCProvider(oidc_config)
        with pytest.raises(ValueError) as exc_info:
            provider.get_user_info_from_id_token(id_token)
        assert "No subject" in str(exc_info.value)

    def test_get_user_info_missing_email(self, oidc_config):
        """Should raise ValueError when email is missing."""
        claims = {"sub": "user123"}
        id_token = create_mock_jwt(claims)

        provider = OIDCProvider(oidc_config)
        with pytest.raises(ValueError) as exc_info:
            provider.get_user_info_from_id_token(id_token)
        assert "No email" in str(exc_info.value)


class TestOIDCTokenRefresh:
    """Tests for OIDC token refresh using discovery."""

    def test_refresh_access_token(self, oidc_config, mock_http_client):
        """Should refresh token using discovery token endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        provider = OIDCProvider(oidc_config)
        provider._discovery = {"token_endpoint": "https://idp.example.com/token"}
        provider._http_client = mock_http_client

        tokens = provider.refresh_access_token("refresh_token")

        call_args = mock_http_client.post.call_args
        assert call_args[1]["data"]["grant_type"] == "refresh_token"
        assert tokens.access_token == "new_access_token"


class TestOIDCEndSession:
    """Tests for OIDC end session URL."""

    def test_get_end_session_url(self, oidc_config):
        """Should generate end session URL with parameters."""
        provider = OIDCProvider(oidc_config)
        provider._discovery = {"end_session_endpoint": "https://idp.example.com/logout"}

        url = provider.get_end_session_url(
            id_token_hint="test_token",
            post_logout_redirect_uri="https://app.example.com",
            state="logout_state",
        )

        assert "https://idp.example.com/logout?" in url
        assert "id_token_hint=test_token" in url
        assert "state=logout_state" in url

    def test_get_end_session_url_not_available(self, oidc_config):
        """Should return None when end_session_endpoint not in discovery."""
        provider = OIDCProvider(oidc_config)
        provider._discovery = {}

        url = provider.get_end_session_url()

        assert url is None


# =============================================================================
# GitHubOAuthProvider Tests
# =============================================================================


class TestGitHubOAuthProviderInit:
    """Tests for GitHubOAuthProvider initialization."""

    def test_provider_name(self, github_config):
        """Should have correct provider name."""
        provider = GitHubOAuthProvider(github_config)
        assert provider.PROVIDER_NAME == "github"

    def test_is_configured(self, github_config):
        """Should be configured with client_id and secret."""
        provider = GitHubOAuthProvider(github_config)
        assert provider.is_configured is True


class TestGitHubAuthorizationUrl:
    """Tests for GitHub authorization URL generation."""

    def test_authorization_url_basic(self, github_config):
        """Should generate correct authorization URL."""
        provider = GitHubOAuthProvider(github_config)
        url = provider.get_authorization_url("test_state")

        assert url.startswith("https://github.com/login/oauth/authorize?")
        assert "client_id=github_client_id" in url
        assert "state=test_state" in url
        assert "scope=read%3Auser" in url or "scope=read:user" in url

    def test_authorization_url_allow_signup_true(self, github_config):
        """Should include allow_signup=true parameter."""
        provider = GitHubOAuthProvider(github_config)
        url = provider.get_authorization_url("state", allow_signup=True)

        assert "allow_signup=true" in url

    def test_authorization_url_allow_signup_false(self, github_config):
        """Should include allow_signup=false parameter."""
        provider = GitHubOAuthProvider(github_config)
        url = provider.get_authorization_url("state", allow_signup=False)

        assert "allow_signup=false" in url

    def test_authorization_url_with_login_hint(self, github_config):
        """Should include login hint for username prefill."""
        provider = GitHubOAuthProvider(github_config)
        url = provider.get_authorization_url("state", login="octocat")

        assert "login=octocat" in url


class TestGitHubCodeExchange:
    """Tests for GitHub code exchange."""

    def test_exchange_code_with_json_accept_header(self, github_config, mock_http_client):
        """Should include Accept: application/json header."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "github_access_token",
            "token_type": "bearer",
            "scope": "read:user,user:email",
        }
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        provider = GitHubOAuthProvider(github_config)
        provider._http_client = mock_http_client

        tokens = provider.exchange_code("auth_code")

        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "https://github.com/login/oauth/access_token"
        assert call_args[1]["headers"]["Accept"] == "application/json"
        assert tokens.access_token == "github_access_token"

    def test_exchange_code_sends_credentials(self, github_config, mock_http_client):
        """Should send client credentials in request body."""
        mock_response = Mock()
        mock_response.json.return_value = {"access_token": "token"}
        mock_response.raise_for_status = Mock()
        mock_http_client.post.return_value = mock_response

        provider = GitHubOAuthProvider(github_config)
        provider._http_client = mock_http_client

        provider.exchange_code("code")

        call_args = mock_http_client.post.call_args
        assert call_args[1]["data"]["client_id"] == "github_client_id"
        assert call_args[1]["data"]["client_secret"] == "github_client_secret"


class TestGitHubUserInfo:
    """Tests for GitHub user info retrieval."""

    def test_get_user_info_with_public_email(self, github_config, mock_http_client):
        """Should get user info when email is public."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": 12345,
            "login": "octocat",
            "name": "The Octocat",
            "email": "octocat@github.com",
            "avatar_url": "https://avatars.githubusercontent.com/u/12345",
        }
        mock_http_client.get.return_value = mock_response

        provider = GitHubOAuthProvider(github_config)
        provider._http_client = mock_http_client

        user_info = provider.get_user_info("access_token")

        assert user_info.provider == "github"
        assert user_info.provider_user_id == "12345"
        assert user_info.email == "octocat@github.com"
        assert user_info.name == "The Octocat"

    def test_get_user_info_uses_login_as_name_fallback(self, github_config, mock_http_client):
        """Should use login as name when name is not set."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": 12345,
            "login": "octocat",
            "name": None,
            "email": "octocat@github.com",
        }
        mock_http_client.get.return_value = mock_response

        provider = GitHubOAuthProvider(github_config)
        provider._http_client = mock_http_client

        user_info = provider.get_user_info("access_token")

        assert user_info.name == "octocat"


class TestGitHubPrivateEmailFallback:
    """Tests for GitHub private email fallback to /user/emails."""

    def test_fetch_primary_email(self, github_config, mock_http_client):
        """Should fetch primary verified email from emails endpoint."""
        # First call for user info (no public email)
        user_response = Mock()
        user_response.json.return_value = {
            "id": 12345,
            "login": "octocat",
            "email": None,
        }
        user_response.raise_for_status = Mock()

        # Second call for emails
        emails_response = Mock()
        emails_response.json.return_value = [
            {"email": "secondary@example.com", "primary": False, "verified": True},
            {"email": "primary@example.com", "primary": True, "verified": True},
        ]
        emails_response.raise_for_status = Mock()

        mock_http_client.get.side_effect = [user_response, emails_response]

        provider = GitHubOAuthProvider(github_config)
        provider._http_client = mock_http_client

        user_info = provider.get_user_info("access_token")

        assert user_info.email == "primary@example.com"
        assert user_info.email_verified is True

    def test_fetch_verified_email_fallback(self, github_config, mock_http_client):
        """Should fall back to any verified email when no primary."""
        user_response = Mock()
        user_response.json.return_value = {"id": 12345, "login": "user", "email": None}
        user_response.raise_for_status = Mock()

        emails_response = Mock()
        emails_response.json.return_value = [
            {"email": "verified@example.com", "primary": False, "verified": True},
        ]
        emails_response.raise_for_status = Mock()

        mock_http_client.get.side_effect = [user_response, emails_response]

        provider = GitHubOAuthProvider(github_config)
        provider._http_client = mock_http_client

        user_info = provider.get_user_info("access_token")

        assert user_info.email == "verified@example.com"

    def test_fetch_unverified_email_last_resort(self, github_config, mock_http_client):
        """Should use unverified email as last resort."""
        user_response = Mock()
        user_response.json.return_value = {"id": 12345, "login": "user", "email": None}
        user_response.raise_for_status = Mock()

        emails_response = Mock()
        emails_response.json.return_value = [
            {"email": "unverified@example.com", "primary": False, "verified": False},
        ]
        emails_response.raise_for_status = Mock()

        mock_http_client.get.side_effect = [user_response, emails_response]

        provider = GitHubOAuthProvider(github_config)
        provider._http_client = mock_http_client

        user_info = provider.get_user_info("access_token")

        assert user_info.email == "unverified@example.com"
        assert user_info.email_verified is False

    def test_no_email_raises_error(self, github_config, mock_http_client):
        """Should raise ValueError when no email can be retrieved."""
        user_response = Mock()
        user_response.json.return_value = {"id": 12345, "login": "user", "email": None}
        user_response.raise_for_status = Mock()

        emails_response = Mock()
        emails_response.json.return_value = []
        emails_response.raise_for_status = Mock()

        mock_http_client.get.side_effect = [user_response, emails_response]

        provider = GitHubOAuthProvider(github_config)
        provider._http_client = mock_http_client

        with pytest.raises(ValueError) as exc_info:
            provider.get_user_info("access_token")
        assert "Could not retrieve email from GitHub" in str(exc_info.value)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestInvalidTokenSignature:
    """Tests for invalid token signature handling."""

    def test_apple_invalid_token_format(self, apple_config):
        """Should raise ValueError for malformed token."""
        provider = AppleOAuthProvider(apple_config)

        with pytest.raises(ValueError) as exc_info:
            provider._decode_id_token("not-a-jwt", verify=False)
        assert "Invalid ID token format" in str(exc_info.value)

    def test_oidc_invalid_token_format(self, oidc_config):
        """Should raise ValueError for malformed OIDC token."""
        provider = OIDCProvider(oidc_config)

        with pytest.raises(ValueError) as exc_info:
            provider._decode_id_token("invalid.token")
        assert "Invalid ID token format" in str(exc_info.value)


class TestExpiredTokens:
    """Tests for expired token handling."""

    def test_apple_expired_token(self, apple_config, mock_http_client):
        """Should raise ValueError for expired Apple ID token."""
        import jwt

        # Create a mock signing key
        mock_key = MagicMock()
        mock_key.key = MagicMock()

        provider = AppleOAuthProvider(apple_config)
        provider._http_client = mock_http_client

        # Set up JWKS cache
        AppleOAuthProvider._jwks_cache = {"keys": [{"kid": "test_key", "kty": "RSA", "use": "sig"}]}
        AppleOAuthProvider._jwks_cache_expiry = time.time() + 3600

        # Mock the _get_signing_key to return a mock key
        with patch.object(provider, "_get_signing_key", return_value=(mock_key.key, "RS256")):
            # Mock jwt.decode to raise ExpiredSignatureError - patch on jwt module
            with patch.object(jwt, "decode") as mock_decode:
                mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")
                with pytest.raises(ValueError) as exc_info:
                    provider._verify_id_token("expired.token.here")
                assert "expired" in str(exc_info.value).lower()

        # Clean up
        AppleOAuthProvider._jwks_cache = None
        AppleOAuthProvider._jwks_cache_expiry = 0


class TestMissingRequiredClaims:
    """Tests for missing required claims."""

    def test_apple_missing_subject(self, apple_config):
        """Should raise ValueError when Apple ID token lacks sub."""
        provider = AppleOAuthProvider(apple_config)

        with patch.object(provider, "_decode_id_token", return_value={"email": "test@example.com"}):
            with pytest.raises(ValueError) as exc_info:
                provider.get_user_info_from_id_token("mock_token")
            assert "No subject" in str(exc_info.value)

    def test_oidc_missing_subject(self, oidc_config):
        """Should raise ValueError when OIDC response lacks sub."""
        claims = {"email": "test@example.com"}
        id_token = create_mock_jwt(claims)

        provider = OIDCProvider(oidc_config)
        with pytest.raises(ValueError) as exc_info:
            provider.get_user_info_from_id_token(id_token)
        assert "No subject" in str(exc_info.value)

    def test_oidc_missing_email(self, oidc_config):
        """Should raise ValueError when OIDC response lacks email."""
        claims = {"sub": "user123"}
        id_token = create_mock_jwt(claims)

        provider = OIDCProvider(oidc_config)
        with pytest.raises(ValueError) as exc_info:
            provider.get_user_info_from_id_token(id_token)
        assert "No email" in str(exc_info.value)


class TestNetworkFailures:
    """Tests for network failure handling."""

    def test_apple_jwks_network_failure_no_cache(self, apple_config, mock_http_client):
        """Should raise RuntimeError when JWKS fetch fails without cache."""
        AppleOAuthProvider._jwks_cache = None
        AppleOAuthProvider._jwks_cache_expiry = 0

        mock_http_client.get.side_effect = httpx.ConnectError("Network error")

        provider = AppleOAuthProvider(apple_config)
        provider._http_client = mock_http_client

        with pytest.raises(RuntimeError) as exc_info:
            provider._fetch_apple_jwks()
        assert "Unable to fetch Apple JWKS" in str(exc_info.value)

    def test_oidc_discovery_network_failure(self, oidc_config, mock_http_client):
        """Should raise ValueError when discovery fetch fails."""
        mock_http_client.get.side_effect = httpx.ConnectError("Network error")

        provider = OIDCProvider(oidc_config)
        provider._http_client = mock_http_client

        with patch(
            "aragora.server.handlers.oauth_providers.oidc._get_secret",
            return_value="https://idp.example.com",
        ):
            with pytest.raises(ValueError) as exc_info:
                provider._get_discovery()
            assert "Failed to fetch OIDC discovery" in str(exc_info.value)

    def test_github_emails_network_failure(self, github_config, mock_http_client):
        """Should handle network failure gracefully for emails endpoint."""
        user_response = Mock()
        user_response.json.return_value = {"id": 12345, "login": "user", "email": None}
        user_response.raise_for_status = Mock()

        mock_http_client.get.side_effect = [
            user_response,
            httpx.ConnectError("Network error"),
        ]

        provider = GitHubOAuthProvider(github_config)
        provider._http_client = mock_http_client

        with pytest.raises(ValueError) as exc_info:
            provider.get_user_info("access_token")
        assert "Could not retrieve email" in str(exc_info.value)

    def test_token_exchange_network_failure(self, github_config, mock_http_client):
        """Should propagate network errors on token exchange."""
        mock_http_client.post.side_effect = httpx.ConnectError("Network error")

        provider = GitHubOAuthProvider(github_config)
        provider._http_client = mock_http_client

        with pytest.raises(httpx.ConnectError):
            provider.exchange_code("code")


class TestTokenRevocation:
    """Tests for token revocation."""

    def test_revoke_token_success(self, github_config, mock_http_client):
        """Should return True on successful revocation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_http_client.post.return_value = mock_response

        config = OAuthProviderConfig(
            client_id="test",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
            revocation_endpoint="https://auth.example.com/revoke",
        )

        class TestProvider(OAuthProvider):
            PROVIDER_NAME = "test"

            def _load_config_from_env(self):
                return config

            def get_authorization_url(self, state, redirect_uri=None, scopes=None, **kwargs):
                return ""

            def exchange_code(self, code, redirect_uri=None):
                return OAuthTokens(access_token="test")

            def get_user_info(self, access_token):
                return OAuthUserInfo(provider="test", provider_user_id="123")

        provider = TestProvider(config)
        provider._http_client = mock_http_client

        result = provider.revoke_token("token_to_revoke")

        assert result is True
        mock_http_client.post.assert_called_once()

    def test_revoke_token_no_endpoint(self, github_config):
        """Should return False when no revocation endpoint."""
        config = OAuthProviderConfig(
            client_id="test",
            client_secret="secret",
            redirect_uri="http://localhost/callback",
            revocation_endpoint="",
        )

        class TestProvider(OAuthProvider):
            PROVIDER_NAME = "test"

            def _load_config_from_env(self):
                return config

            def get_authorization_url(self, state, redirect_uri=None, scopes=None, **kwargs):
                return ""

            def exchange_code(self, code, redirect_uri=None):
                return OAuthTokens(access_token="test")

            def get_user_info(self, access_token):
                return OAuthUserInfo(provider="test", provider_user_id="123")

        provider = TestProvider(config)
        result = provider.revoke_token("token")

        assert result is False

    def test_apple_revoke_token(self, apple_config, mock_http_client):
        """Should revoke Apple token with correct parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_http_client.post.return_value = mock_response

        provider = AppleOAuthProvider(apple_config)
        provider._http_client = mock_http_client

        result = provider.revoke_token("token_to_revoke", "refresh_token")

        assert result is True
        call_args = mock_http_client.post.call_args
        assert call_args[1]["data"]["token_type_hint"] == "refresh_token"
