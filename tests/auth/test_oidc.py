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
                mock_urlopen.side_effect = Exception("Network error")

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
