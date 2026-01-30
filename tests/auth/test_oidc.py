"""
Tests for OpenID Connect (OIDC) authentication provider.

Tests cover:
- OIDCConfig validation
- OIDCProvider initialization
- PKCE generation
- Authorization URL generation
- Token exchange (mocked)
- Discovery endpoint caching
- Error handling
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.auth.oidc import (
    OIDCConfig,
    OIDCError,
    OIDCProvider,
    PROVIDER_CONFIGS,
)
from aragora.auth.sso import (
    SSOAuthenticationError,
    SSOConfigurationError,
    SSOProviderType,
)


def make_oidc_config(**kwargs) -> OIDCConfig:
    """Helper to create OIDCConfig with provider_type default."""
    defaults = {"provider_type": SSOProviderType.OIDC}
    defaults.update(kwargs)
    return OIDCConfig(**defaults)


# ============================================================================
# OIDCConfig Tests
# ============================================================================


class TestOIDCConfig:
    """Tests for OIDCConfig dataclass."""

    def test_config_with_required_fields(self):
        """Test creating config with all required fields."""
        config = make_oidc_config(
            client_id="test-client-id",
            client_secret="test-client-secret",
            issuer_url="https://login.example.com",
            callback_url="https://app.example.com/callback",
        )

        assert config.client_id == "test-client-id"
        assert config.client_secret == "test-client-secret"
        assert config.issuer_url == "https://login.example.com"
        assert config.callback_url == "https://app.example.com/callback"

    def test_config_default_scopes(self):
        """Test that default scopes are set."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
        )

        assert "openid" in config.scopes
        assert "email" in config.scopes
        assert "profile" in config.scopes

    def test_config_default_pkce_enabled(self):
        """Test that PKCE is enabled by default."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
        )

        assert config.use_pkce is True

    def test_config_custom_scopes(self):
        """Test config with custom scopes."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
            scopes=["openid", "email", "groups"],
        )

        assert config.scopes == ["openid", "email", "groups"]

    def test_config_validate_missing_client_id(self):
        """Test validation fails without client_id."""
        config = make_oidc_config(
            client_id="",
            client_secret="secret",
            issuer_url="https://example.com",
        )

        errors = config.validate()
        assert any("client_id" in e for e in errors)

    def test_config_validate_missing_client_secret(self):
        """Test validation fails without client_secret."""
        config = make_oidc_config(
            client_id="test",
            client_secret="",
            issuer_url="https://example.com",
        )

        errors = config.validate()
        assert any("client_secret" in e for e in errors)

    def test_config_validate_missing_endpoints(self):
        """Test validation fails without issuer or explicit endpoints."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="",  # No issuer
            authorization_endpoint="",  # No explicit endpoints
            token_endpoint="",
        )

        errors = config.validate()
        assert any("issuer_url" in e or "endpoints" in e for e in errors)

    def test_config_validate_with_explicit_endpoints(self):
        """Test validation passes with explicit endpoints instead of issuer."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="",
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
        )

        errors = config.validate()
        # Should not have endpoint-related errors
        assert not any("issuer_url" in e or "endpoints" in e for e in errors)

    def test_config_claim_mapping_defaults(self):
        """Test default claim mapping."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
        )

        assert config.claim_mapping["sub"] == "id"
        assert config.claim_mapping["email"] == "email"
        assert config.claim_mapping["name"] == "name"

    def test_config_custom_claim_mapping(self):
        """Test custom claim mapping."""
        custom_mapping = {
            "sub": "user_id",
            "email": "email_address",
            "custom_claim": "custom_field",
        }
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
            claim_mapping=custom_mapping,
        )

        assert config.claim_mapping["sub"] == "user_id"
        assert config.claim_mapping["email"] == "email_address"

    def test_config_provider_type_default(self):
        """Test that provider type defaults to OIDC."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
        )

        assert config.provider_type == SSOProviderType.OIDC


# ============================================================================
# OIDCProvider Tests
# ============================================================================


class TestOIDCProviderInitialization:
    """Tests for OIDCProvider initialization."""

    def test_provider_initialization(self):
        """Test provider initializes with valid config."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://login.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )

        provider = OIDCProvider(config)

        assert provider.config == config
        assert provider._pkce_store == {}

    def test_provider_initialization_invalid_config(self):
        """Test provider raises error with invalid config."""
        config = make_oidc_config(
            client_id="",  # Invalid - empty
            client_secret="secret",
            issuer_url="https://example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )

        with pytest.raises(SSOConfigurationError) as exc_info:
            OIDCProvider(config)

        assert "client_id" in str(exc_info.value)


# ============================================================================
# PKCE Tests
# ============================================================================


class TestPKCE:
    """Tests for PKCE (Proof Key for Code Exchange) generation."""

    def test_pkce_generation_format(self):
        """Test PKCE code verifier and challenge format."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
            use_pkce=True,
        )
        provider = OIDCProvider(config)

        verifier, challenge = provider._generate_pkce()

        # Verifier should be URL-safe base64
        assert len(verifier) >= 43  # Minimum length per spec
        assert all(
            c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            for c in verifier
        )

        # Challenge should also be URL-safe base64
        assert all(
            c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            for c in challenge
        )

    def test_pkce_challenge_derivation(self):
        """Test that challenge is correctly derived from verifier."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        provider = OIDCProvider(config)

        verifier, challenge = provider._generate_pkce()

        # Manually compute expected challenge
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

        assert challenge == expected_challenge

    def test_pkce_unique_per_call(self):
        """Test that each PKCE generation produces unique values."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        provider = OIDCProvider(config)

        verifier1, challenge1 = provider._generate_pkce()
        verifier2, challenge2 = provider._generate_pkce()

        assert verifier1 != verifier2
        assert challenge1 != challenge2


# ============================================================================
# Authorization URL Tests
# ============================================================================


class TestAuthorizationURL:
    """Tests for authorization URL generation."""

    @pytest.fixture
    def provider(self):
        """Create a provider with manual endpoints (no discovery)."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="",  # No discovery
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        return OIDCProvider(config)

    @pytest.mark.asyncio
    async def test_authorization_url_basic(self, provider):
        """Test basic authorization URL generation."""
        url = await provider.get_authorization_url(state="test-state")

        assert url.startswith("https://example.com/authorize?")
        assert "client_id=test-client" in url
        assert "response_type=code" in url
        assert "state=test-state" in url
        assert "redirect_uri=" in url

    @pytest.mark.asyncio
    async def test_authorization_url_includes_scopes(self, provider):
        """Test that scopes are included in URL."""
        url = await provider.get_authorization_url(state="test-state")

        # Default scopes should be included
        assert "scope=" in url
        assert "openid" in url

    @pytest.mark.asyncio
    async def test_authorization_url_with_pkce(self, provider):
        """Test PKCE parameters are included when enabled."""
        url = await provider.get_authorization_url(state="test-state")

        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url
        # Verifier should be stored
        assert "test-state" in provider._pkce_store

    @pytest.mark.asyncio
    async def test_authorization_url_without_pkce(self):
        """Test URL generation without PKCE."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
            use_pkce=False,
        )
        provider = OIDCProvider(config)

        url = await provider.get_authorization_url(state="test-state")

        assert "code_challenge" not in url
        assert "code_challenge_method" not in url

    @pytest.mark.asyncio
    async def test_authorization_url_custom_redirect(self, provider):
        """Test custom redirect URI."""
        url = await provider.get_authorization_url(
            state="test-state",
            redirect_uri="https://custom.example.com/callback",
        )

        assert "redirect_uri=https%3A%2F%2Fcustom.example.com%2Fcallback" in url

    @pytest.mark.asyncio
    async def test_authorization_url_custom_scopes(self, provider):
        """Test custom scopes override."""
        url = await provider.get_authorization_url(
            state="test-state",
            scopes=["openid", "offline_access"],
        )

        assert "scope=openid+offline_access" in url or "scope=openid%20offline_access" in url

    @pytest.mark.asyncio
    async def test_authorization_url_nonce_included(self, provider):
        """Test that nonce is included in URL."""
        url = await provider.get_authorization_url(state="test-state")

        assert "nonce=" in url

    @pytest.mark.asyncio
    async def test_authorization_url_generates_state_if_missing(self, provider):
        """Test that state is auto-generated if not provided."""
        url = await provider.get_authorization_url()

        assert "state=" in url


# ============================================================================
# Discovery Tests
# ============================================================================


class TestDiscovery:
    """Tests for OIDC discovery endpoint."""

    @pytest.fixture
    def provider(self):
        """Create provider with issuer URL for discovery."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://login.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        return OIDCProvider(config)

    @pytest.mark.asyncio
    async def test_discovery_fetches_endpoints(self, provider):
        """Test that discovery fetches OpenID configuration."""
        discovery_doc = {
            "authorization_endpoint": "https://login.example.com/authorize",
            "token_endpoint": "https://login.example.com/token",
            "userinfo_endpoint": "https://login.example.com/userinfo",
            "jwks_uri": "https://login.example.com/.well-known/jwks.json",
        }

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps(discovery_doc).encode()
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                result = await provider._discover_endpoints()

                assert result["authorization_endpoint"] == "https://login.example.com/authorize"
                assert result["token_endpoint"] == "https://login.example.com/token"

    @pytest.mark.asyncio
    async def test_discovery_caching(self, provider):
        """Test that discovery results are cached."""
        discovery_doc = {"authorization_endpoint": "https://example.com/auth"}

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps(discovery_doc).encode()
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                # First call
                await provider._discover_endpoints()
                # Second call
                await provider._discover_endpoints()

                # Should only fetch once
                assert mock_urlopen.call_count == 1

    @pytest.mark.asyncio
    async def test_discovery_failure_returns_empty(self, provider):
        """Test that discovery failure returns empty dict."""
        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                # Use OSError which is what urllib raises on network errors
                mock_urlopen.side_effect = OSError("Network error")

                result = await provider._discover_endpoints()

                assert result == {}

    @pytest.mark.asyncio
    async def test_get_endpoint_prefers_config(self, provider):
        """Test that explicit config takes precedence over discovery."""
        provider.config.authorization_endpoint = "https://config.example.com/auth"

        endpoint = await provider._get_endpoint("authorization_endpoint")

        assert endpoint == "https://config.example.com/auth"


# ============================================================================
# Provider Presets Tests
# ============================================================================


class TestProviderPresets:
    """Tests for well-known provider configurations."""

    def test_azure_ad_preset_exists(self):
        """Test Azure AD preset configuration exists."""
        assert "azure_ad" in PROVIDER_CONFIGS
        assert "authorization_endpoint" in PROVIDER_CONFIGS["azure_ad"]
        assert "token_endpoint" in PROVIDER_CONFIGS["azure_ad"]

    def test_okta_preset_exists(self):
        """Test Okta preset configuration exists."""
        assert "okta" in PROVIDER_CONFIGS
        assert "authorization_endpoint" in PROVIDER_CONFIGS["okta"]

    def test_google_preset_exists(self):
        """Test Google preset configuration exists."""
        assert "google" in PROVIDER_CONFIGS
        assert "authorization_endpoint" in PROVIDER_CONFIGS["google"]
        assert "accounts.google.com" in PROVIDER_CONFIGS["google"]["authorization_endpoint"]

    def test_github_preset_exists(self):
        """Test GitHub preset configuration exists."""
        assert "github" in PROVIDER_CONFIGS
        assert "github.com" in PROVIDER_CONFIGS["github"]["authorization_endpoint"]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestOIDCErrors:
    """Tests for OIDC error handling."""

    def test_oidc_error_creation(self):
        """Test OIDCError creation."""
        error = OIDCError("Test error", {"key": "value"})

        assert str(error) == "Test error"
        assert error.details == {"key": "value"}

    def test_oidc_error_without_details(self):
        """Test OIDCError without details."""
        error = OIDCError("Simple error")

        assert str(error) == "Simple error"
        # Parent class may set default empty dict when None
        assert error.details == {} or error.details is None

    @pytest.mark.asyncio
    async def test_missing_authorization_endpoint_error(self):
        """Test error when discovery doesn't provide authorization endpoint."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",  # Has issuer for discovery
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        provider = OIDCProvider(config)

        # Mock discovery to return config without authorization_endpoint
        async def mock_discovery():
            return {
                "token_endpoint": "https://example.com/token",
                "jwks_uri": "https://example.com/jwks",
                # Missing authorization_endpoint
            }

        with patch.object(provider, "_discover_endpoints", mock_discovery):
            with pytest.raises(SSOConfigurationError) as exc_info:
                await provider.get_authorization_url()

        assert "authorization_endpoint" in str(exc_info.value).lower()


# ============================================================================
# Token Validation Tests (Mocked)
# ============================================================================


class TestTokenValidation:
    """Tests for token validation (with mocked JWT library)."""

    @pytest.fixture
    def provider_with_validation(self):
        """Create provider with token validation enabled."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://login.example.com",
            authorization_endpoint="https://login.example.com/authorize",
            token_endpoint="https://login.example.com/token",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
            validate_tokens=True,
        )
        return OIDCProvider(config)

    def test_config_validation_enabled(self, provider_with_validation):
        """Test that token validation is configurable."""
        assert provider_with_validation.config.validate_tokens is True

    def test_config_allowed_audiences(self):
        """Test allowed audiences configuration."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://login.example.com",
            allowed_audiences=["aud1", "aud2"],
        )

        assert config.allowed_audiences == ["aud1", "aud2"]


# ============================================================================
# State Management Tests
# ============================================================================


class TestStateManagement:
    """Tests for CSRF state management."""

    @pytest.fixture
    def provider(self):
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            authorization_endpoint="https://example.com/auth",
            token_endpoint="https://example.com/token",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        return OIDCProvider(config)

    def test_generate_state(self, provider):
        """Test state generation."""
        state = provider.generate_state()

        assert state is not None
        assert len(state) > 20  # Should be reasonably long

    def test_state_stored_on_generation(self, provider):
        """Test that generated state is stored."""
        state = provider.generate_state()

        assert state in provider._state_store

    @pytest.mark.asyncio
    async def test_state_stored_when_provided(self, provider):
        """Test that provided state is stored."""
        custom_state = "my-custom-state-123"
        await provider.get_authorization_url(state=custom_state)

        assert custom_state in provider._state_store

    def test_validate_state_success(self, provider):
        """Test successful state validation."""
        state = provider.generate_state()

        result = provider.validate_state(state)

        assert result is True

    def test_validate_state_invalid(self, provider):
        """Test validation of unknown state fails."""
        result = provider.validate_state("unknown-state")

        assert result is False

    def test_validate_state_expired(self, provider):
        """Test that expired state fails validation."""
        state = provider.generate_state()
        # Manually expire the state (simulate 11 minutes passing)
        provider._state_store[state] = time.time() - 660

        result = provider.validate_state(state)

        assert result is False

    def test_validate_state_consumed_only_once(self, provider):
        """Test that state can only be validated once (consumed on use)."""
        state = provider.generate_state()

        # First validation succeeds
        assert provider.validate_state(state) is True
        # Second validation fails (state was consumed)
        assert provider.validate_state(state) is False

    def test_cleanup_expired_states(self, provider):
        """Test cleanup of expired states."""
        # Generate some states
        state1 = provider.generate_state()
        state2 = provider.generate_state()
        state3 = provider.generate_state()

        # Expire state1 and state2
        provider._state_store[state1] = time.time() - 700
        provider._state_store[state2] = time.time() - 800

        # Clean up
        removed = provider.cleanup_expired_states()

        assert removed == 2
        assert state1 not in provider._state_store
        assert state2 not in provider._state_store
        assert state3 in provider._state_store


# ============================================================================
# Token Exchange Tests
# ============================================================================


class TestTokenExchange:
    """Tests for authorization code to token exchange."""

    @pytest.fixture
    def provider(self):
        """Create provider with manual endpoints."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="",
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        return OIDCProvider(config)

    @pytest.mark.asyncio
    async def test_exchange_code_success(self, provider):
        """Test successful authorization code exchange."""
        mock_tokens = {
            "access_token": "test-access-token",
            "id_token": "test-id-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps(mock_tokens).encode()
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                tokens = await provider._exchange_code("auth-code", None)

                assert tokens["access_token"] == "test-access-token"
                assert tokens["refresh_token"] == "test-refresh-token"

    @pytest.mark.asyncio
    async def test_exchange_code_with_pkce_verifier(self, provider):
        """Test code exchange includes PKCE code verifier."""
        mock_tokens = {"access_token": "token", "expires_in": 3600}

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps(mock_tokens).encode()
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                await provider._exchange_code("auth-code", "test-verifier")

                # Verify the request was made with code_verifier
                call_args = mock_urlopen.call_args
                request = call_args[0][0]
                assert b"code_verifier=test-verifier" in request.data

    @pytest.mark.asyncio
    async def test_exchange_code_network_error(self, provider):
        """Test code exchange handles network errors."""
        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = OSError("Connection refused")

                with pytest.raises(SSOAuthenticationError) as exc_info:
                    await provider._exchange_code("auth-code", None)

                assert "network error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_exchange_code_invalid_json_response(self, provider):
        """Test code exchange handles invalid JSON response."""
        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = b"not valid json"
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                with pytest.raises(SSOAuthenticationError) as exc_info:
                    await provider._exchange_code("auth-code", None)

                assert "invalid response" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_exchange_code_no_token_endpoint(self):
        """Test error when token endpoint is not configured and no discovery."""
        # When token_endpoint is empty and no issuer_url for discovery,
        # _exchange_code should raise SSOConfigurationError
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://example.com",  # Has issuer for discovery
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="",  # No explicit token endpoint
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        provider = OIDCProvider(config)

        # Mock discovery to return empty (no token_endpoint found)
        async def mock_discover():
            return {}

        with patch.object(provider, "_discover_endpoints", mock_discover):
            with pytest.raises(SSOConfigurationError) as exc_info:
                await provider._exchange_code("auth-code", None)

            assert "token_endpoint" in str(exc_info.value).lower()


# ============================================================================
# Authentication Flow Tests
# ============================================================================


class TestAuthentication:
    """Tests for the full authentication flow."""

    @pytest.fixture
    def provider(self):
        """Create provider with all endpoints configured."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://login.example.com",
            authorization_endpoint="https://login.example.com/authorize",
            token_endpoint="https://login.example.com/token",
            userinfo_endpoint="https://login.example.com/userinfo",
            jwks_uri="https://login.example.com/.well-known/jwks.json",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
            use_pkce=False,  # Disable PKCE for simpler testing
        )
        return OIDCProvider(config)

    @pytest.mark.asyncio
    async def test_authenticate_requires_code(self, provider):
        """Test authentication fails without authorization code."""
        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(code=None, state="test-state")

        assert "authorization code" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_authenticate_validates_state(self, provider):
        """Test authentication validates state parameter."""
        # Don't generate state first - use an invalid one
        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(code="auth-code", state="invalid-state")

        assert "state" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_authenticate_success_flow(self, provider):
        """Test successful authentication flow."""
        # Generate a valid state
        state = provider.generate_state()

        mock_tokens = {
            "access_token": "access-token-123",
            "id_token": None,  # Skip ID token validation
            "expires_in": 3600,
        }

        mock_userinfo = {
            "sub": "user-123",
            "email": "test@example.com",
            "name": "Test User",
            "given_name": "Test",
            "family_name": "User",
        }

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                # First call: token exchange, Second call: userinfo
                mock_token_response = MagicMock()
                mock_token_response.read.return_value = json.dumps(mock_tokens).encode()
                mock_token_response.__enter__ = MagicMock(return_value=mock_token_response)
                mock_token_response.__exit__ = MagicMock(return_value=False)

                mock_userinfo_response = MagicMock()
                mock_userinfo_response.read.return_value = json.dumps(mock_userinfo).encode()
                mock_userinfo_response.__enter__ = MagicMock(return_value=mock_userinfo_response)
                mock_userinfo_response.__exit__ = MagicMock(return_value=False)

                mock_urlopen.side_effect = [mock_token_response, mock_userinfo_response]

                user = await provider.authenticate(code="auth-code", state=state)

                assert user.id == "user-123"
                assert user.email == "test@example.com"
                assert user.name == "Test User"
                assert user.first_name == "Test"
                assert user.last_name == "User"

    @pytest.mark.asyncio
    async def test_authenticate_domain_restriction(self, provider):
        """Test authentication fails for disallowed email domain."""
        provider.config.allowed_domains = ["allowed.com"]

        state = provider.generate_state()

        mock_tokens = {"access_token": "token", "expires_in": 3600}
        mock_userinfo = {
            "sub": "user-123",
            "email": "test@notallowed.com",
            "name": "Test",
        }

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_token_response = MagicMock()
                mock_token_response.read.return_value = json.dumps(mock_tokens).encode()
                mock_token_response.__enter__ = MagicMock(return_value=mock_token_response)
                mock_token_response.__exit__ = MagicMock(return_value=False)

                mock_userinfo_response = MagicMock()
                mock_userinfo_response.read.return_value = json.dumps(mock_userinfo).encode()
                mock_userinfo_response.__enter__ = MagicMock(return_value=mock_userinfo_response)
                mock_userinfo_response.__exit__ = MagicMock(return_value=False)

                mock_urlopen.side_effect = [mock_token_response, mock_userinfo_response]

                with pytest.raises(SSOAuthenticationError) as exc_info:
                    await provider.authenticate(code="auth-code", state=state)

                assert "domain not allowed" in str(exc_info.value).lower()


# ============================================================================
# User Info Retrieval Tests
# ============================================================================


class TestUserInfoRetrieval:
    """Tests for user info fetching and claim mapping."""

    @pytest.fixture
    def provider(self):
        """Create provider with userinfo endpoint."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="",
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            userinfo_endpoint="https://example.com/userinfo",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        return OIDCProvider(config)

    @pytest.mark.asyncio
    async def test_fetch_userinfo_success(self, provider):
        """Test successful userinfo fetch."""
        mock_userinfo = {
            "sub": "user-456",
            "email": "user@example.com",
            "name": "John Doe",
        }

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps(mock_userinfo).encode()
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                result = await provider._fetch_userinfo("access-token")

                assert result["sub"] == "user-456"
                assert result["email"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_fetch_userinfo_network_error(self, provider):
        """Test userinfo fetch handles network errors gracefully."""
        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = OSError("Network error")

                result = await provider._fetch_userinfo("access-token")

                # Should return empty dict on error, not raise
                assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_userinfo_invalid_json(self, provider):
        """Test userinfo fetch handles invalid JSON gracefully."""
        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = b"invalid json"
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                result = await provider._fetch_userinfo("access-token")

                assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_userinfo_no_endpoint(self):
        """Test userinfo fetch returns empty when endpoint not configured."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            userinfo_endpoint="",  # No userinfo endpoint
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        provider = OIDCProvider(config)

        result = await provider._fetch_userinfo("access-token")

        assert result == {}

    def test_claims_to_user_basic(self):
        """Test basic claim to user mapping."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        provider = OIDCProvider(config)

        claims = {
            "sub": "user-id-123",
            "email": "user@example.com",
            "name": "Test User",
            "given_name": "Test",
            "family_name": "User",
            "preferred_username": "testuser",
        }
        tokens = {
            "access_token": "access",
            "refresh_token": "refresh",
            "id_token": "id",
            "expires_in": 3600,
        }

        user = provider._claims_to_user(claims, tokens)

        assert user.id == "user-id-123"
        assert user.email == "user@example.com"
        assert user.name == "Test User"
        assert user.first_name == "Test"
        assert user.last_name == "User"
        assert user.username == "testuser"
        assert user.access_token == "access"
        assert user.refresh_token == "refresh"

    def test_claims_to_user_with_roles_and_groups(self):
        """Test claim mapping with roles and groups."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        provider = OIDCProvider(config)

        claims = {
            "sub": "user-id",
            "email": "user@example.com",
            "roles": ["admin", "developer"],
            "groups": ["engineering", "platform"],
        }
        tokens = {"access_token": "token", "expires_in": 3600}

        user = provider._claims_to_user(claims, tokens)

        assert "admin" in user.roles
        assert "developer" in user.roles
        assert "engineering" in user.groups
        assert "platform" in user.groups

    def test_claims_to_user_azure_wids(self):
        """Test Azure AD wids (role IDs) are included in roles."""
        config = make_oidc_config(
            client_id="test",
            client_secret="secret",
            issuer_url="https://example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        provider = OIDCProvider(config)

        claims = {
            "sub": "user-id",
            "email": "user@example.com",
            "wids": ["role-guid-1", "role-guid-2"],
        }
        tokens = {"access_token": "token", "expires_in": 3600}

        user = provider._claims_to_user(claims, tokens)

        assert "role-guid-1" in user.roles
        assert "role-guid-2" in user.roles


# ============================================================================
# Token Refresh Tests
# ============================================================================


class TestTokenRefresh:
    """Tests for token refresh functionality."""

    @pytest.fixture
    def provider(self):
        """Create provider with token endpoint."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="",
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        return OIDCProvider(config)

    @pytest.mark.asyncio
    async def test_refresh_token_success(self, provider):
        """Test successful token refresh."""
        from aragora.auth.sso import SSOUser

        original_user = SSOUser(
            id="user-123",
            email="user@example.com",
            name="Test User",
            refresh_token="original-refresh-token",
            provider_type="oidc",
        )

        mock_tokens = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "id_token": "new-id-token",
            "expires_in": 7200,
        }

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps(mock_tokens).encode()
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                refreshed_user = await provider.refresh_token(original_user)

                assert refreshed_user is not None
                assert refreshed_user.access_token == "new-access-token"
                assert refreshed_user.refresh_token == "new-refresh-token"
                # User info should be preserved
                assert refreshed_user.id == "user-123"
                assert refreshed_user.email == "user@example.com"

    @pytest.mark.asyncio
    async def test_refresh_token_no_refresh_token(self, provider):
        """Test refresh returns None when user has no refresh token."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user-123",
            email="user@example.com",
            refresh_token=None,  # No refresh token
        )

        result = await provider.refresh_token(user)

        assert result is None

    @pytest.mark.asyncio
    async def test_refresh_token_network_error(self, provider):
        """Test refresh handles network errors."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user-123",
            email="user@example.com",
            refresh_token="some-refresh-token",
        )

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = OSError("Network error")

                result = await provider.refresh_token(user)

                assert result is None

    @pytest.mark.asyncio
    async def test_refresh_token_invalid_response(self, provider):
        """Test refresh handles invalid JSON response."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user-123",
            email="user@example.com",
            refresh_token="some-refresh-token",
        )

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = b"not json"
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                result = await provider.refresh_token(user)

                assert result is None

    @pytest.mark.asyncio
    async def test_refresh_token_preserves_original_refresh_token(self, provider):
        """Test refresh preserves original refresh token if new one not provided."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user-123",
            email="user@example.com",
            refresh_token="original-refresh-token",
        )

        mock_tokens = {
            "access_token": "new-access-token",
            # No refresh_token in response
            "expires_in": 3600,
        }

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps(mock_tokens).encode()
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                refreshed_user = await provider.refresh_token(user)

                assert refreshed_user.refresh_token == "original-refresh-token"


# ============================================================================
# Logout Tests
# ============================================================================


class TestLogout:
    """Tests for logout functionality."""

    @pytest.fixture
    def provider(self):
        """Create provider with end session endpoint."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="",
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            end_session_endpoint="https://example.com/logout",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
            post_logout_redirect_url="https://app.example.com/logged-out",
        )
        return OIDCProvider(config)

    @pytest.mark.asyncio
    async def test_logout_generates_url(self, provider):
        """Test logout generates proper URL."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user-123",
            email="user@example.com",
            id_token="some-id-token",
        )

        logout_url = await provider.logout(user)

        assert logout_url is not None
        assert "https://example.com/logout" in logout_url
        assert "id_token_hint=some-id-token" in logout_url
        assert "post_logout_redirect_uri=" in logout_url

    @pytest.mark.asyncio
    async def test_logout_without_id_token(self, provider):
        """Test logout works without ID token."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user-123",
            email="user@example.com",
            id_token=None,
        )

        logout_url = await provider.logout(user)

        assert logout_url is not None
        assert "id_token_hint" not in logout_url

    @pytest.mark.asyncio
    async def test_logout_fallback_to_config_url(self):
        """Test logout falls back to config logout_url if no end_session_endpoint."""
        config = make_oidc_config(
            client_id="test-client",
            client_secret="test-secret",
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
            end_session_endpoint="",  # No end session endpoint
            logout_url="https://example.com/fallback-logout",
            callback_url="https://app.example.com/callback",
            entity_id="test-entity",
        )
        provider = OIDCProvider(config)

        from aragora.auth.sso import SSOUser

        user = SSOUser(id="user-123", email="user@example.com")

        logout_url = await provider.logout(user)

        assert logout_url == "https://example.com/fallback-logout"


# ============================================================================
# Provider-Specific Configuration Tests
# ============================================================================


class TestProviderSpecificConfigs:
    """Tests for provider-specific configuration helpers."""

    def test_azure_ad_config_factory(self):
        """Test Azure AD config factory method."""
        config = OIDCConfig.for_azure_ad(
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
            callback_url="https://app.example.com/callback",
        )

        assert config.provider_type == SSOProviderType.AZURE_AD
        assert config.client_id == "test-client-id"
        assert config.tenant_id == "test-tenant-id"
        assert "login.microsoftonline.com" in config.issuer_url
        assert "test-tenant-id" in config.issuer_url
        assert "User.Read" in config.scopes

    def test_okta_config_factory(self):
        """Test Okta config factory method."""
        config = OIDCConfig.for_okta(
            org_url="https://myorg.okta.com",
            client_id="test-client-id",
            client_secret="test-client-secret",
            callback_url="https://app.example.com/callback",
        )

        assert config.provider_type == SSOProviderType.OKTA
        assert config.client_id == "test-client-id"
        assert config.issuer_url == "https://myorg.okta.com"
        assert "groups" in config.scopes

    def test_okta_config_normalizes_url(self):
        """Test Okta config normalizes trailing slash from URL."""
        config = OIDCConfig.for_okta(
            org_url="https://myorg.okta.com/",  # With trailing slash
            client_id="test",
            client_secret="secret",
            callback_url="https://app.example.com/callback",
        )

        assert config.issuer_url == "https://myorg.okta.com"

    def test_google_config_factory(self):
        """Test Google config factory method."""
        config = OIDCConfig.for_google(
            client_id="test-client-id",
            client_secret="test-client-secret",
            callback_url="https://app.example.com/callback",
        )

        assert config.provider_type == SSOProviderType.GOOGLE
        assert config.client_id == "test-client-id"
        assert config.issuer_url == "https://accounts.google.com"

    def test_google_config_with_hosted_domain(self):
        """Test Google config with hosted domain restriction."""
        config = OIDCConfig.for_google(
            client_id="test-client-id",
            client_secret="test-client-secret",
            callback_url="https://app.example.com/callback",
            hd="example.com",
        )

        assert config.hd == "example.com"
