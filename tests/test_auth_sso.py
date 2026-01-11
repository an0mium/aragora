"""Tests for SSO/SAML/OIDC authentication."""

import base64
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch


class TestSSOUser:
    """Test SSOUser dataclass."""

    def test_user_creation(self):
        """Test basic user creation."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user-123",
            email="test@example.com",
            name="Test User",
            first_name="Test",
            last_name="User",
        )

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"

    def test_user_full_name_fallback(self):
        """Test full name generation from first/last name."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user-123",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
        )

        assert user.full_name == "John Doe"

    def test_user_is_admin(self):
        """Test admin role detection."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user-123",
            email="admin@example.com",
            roles=["admin", "user"],
        )

        assert user.is_admin is True

        regular_user = SSOUser(
            id="user-456",
            email="user@example.com",
            roles=["user"],
        )

        assert regular_user.is_admin is False

    def test_user_to_dict(self):
        """Test user serialization."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user-123",
            email="test@example.com",
            name="Test User",
            roles=["admin"],
            groups=["engineering"],
            provider_type="oidc",
        )

        data = user.to_dict()

        assert data["id"] == "user-123"
        assert data["email"] == "test@example.com"
        assert data["is_admin"] is True
        assert data["roles"] == ["admin"]
        assert data["provider_type"] == "oidc"


class TestSSOConfig:
    """Test SSOConfig validation."""

    def test_config_validation(self):
        """Test config validation errors."""
        from aragora.auth.sso import SSOConfig, SSOProviderType

        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            # Missing required fields
        )

        errors = config.validate()

        assert "entity_id is required" in errors
        assert "callback_url is required" in errors

    def test_valid_config(self):
        """Test valid config passes validation."""
        from aragora.auth.sso import SSOConfig, SSOProviderType

        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/auth/callback",
        )

        errors = config.validate()

        assert len(errors) == 0


class TestSSOProvider:
    """Test base SSOProvider functionality."""

    def test_generate_state(self):
        """Test state generation."""
        from aragora.auth.sso import SSOConfig, SSOProviderType
        from aragora.auth.oidc import OIDCProvider, OIDCConfig

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
        )

        provider = OIDCProvider(config)
        state = provider.generate_state()

        assert len(state) > 20  # Base64 encoded
        assert provider.validate_state(state) is True
        assert provider.validate_state(state) is False  # Already used

    def test_validate_expired_state(self):
        """Test expired state validation."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
        )

        provider = OIDCProvider(config)
        state = provider.generate_state()

        # Simulate expired state
        provider._state_store[state] = time.time() - 700  # > 10 minutes

        assert provider.validate_state(state) is False

    def test_domain_allowed(self):
        """Test domain restriction checking."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            allowed_domains=["example.com", "company.org"],
        )

        provider = OIDCProvider(config)

        assert provider.is_domain_allowed("user@example.com") is True
        assert provider.is_domain_allowed("user@company.org") is True
        assert provider.is_domain_allowed("user@other.com") is False

    def test_domain_allowed_no_restriction(self):
        """Test domain allowed when no restrictions."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            allowed_domains=[],  # No restrictions
        )

        provider = OIDCProvider(config)

        assert provider.is_domain_allowed("user@any.domain") is True

    def test_role_mapping(self):
        """Test role mapping."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            role_mapping={
                "GlobalAdmin": "admin",
                "TeamMember": "user",
            },
        )

        provider = OIDCProvider(config)

        mapped = provider.map_roles(["GlobalAdmin", "TeamMember", "Viewer"])

        assert "admin" in mapped
        assert "user" in mapped
        assert "Viewer" in mapped  # Unmapped roles pass through


class TestOIDCProvider:
    """Test OIDC provider implementation."""

    def test_oidc_config_validation(self):
        """Test OIDC config validation."""
        from aragora.auth.oidc import OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            # Missing client_id and client_secret
        )

        errors = config.validate()

        assert "client_id is required" in errors
        assert "client_secret is required" in errors

    @pytest.mark.asyncio
    async def test_get_authorization_url(self):
        """Test OIDC authorization URL generation."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            client_id="test-client",
            client_secret="test-secret",
            authorization_endpoint="https://idp.example.com/authorize",
            token_endpoint="https://idp.example.com/token",
        )

        provider = OIDCProvider(config)
        url = await provider.get_authorization_url(state="test-state")

        assert "https://idp.example.com/authorize" in url
        assert "client_id=test-client" in url
        assert "state=test-state" in url
        assert "response_type=code" in url
        assert "scope=" in url

    @pytest.mark.asyncio
    async def test_pkce_generation(self):
        """Test PKCE code challenge generation."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            client_id="test-client",
            client_secret="test-secret",
            authorization_endpoint="https://idp.example.com/authorize",
            token_endpoint="https://idp.example.com/token",
            use_pkce=True,
        )

        provider = OIDCProvider(config)
        url = await provider.get_authorization_url(state="test-state")

        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url
        assert "test-state" in provider._pkce_store


class TestSAMLProvider:
    """Test SAML provider implementation."""

    def test_saml_config_validation(self):
        """Test SAML config validation."""
        from aragora.auth.saml import SAMLConfig
        from aragora.auth.sso import SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            enabled=True,
            entity_id="https://aragora.example.com/saml/metadata",
            callback_url="https://aragora.example.com/saml/acs",
            # Missing IdP settings
        )

        errors = config.validate()

        assert "idp_entity_id is required" in errors
        assert "idp_sso_url is required" in errors
        assert any("idp_certificate" in e for e in errors)

    @pytest.mark.asyncio
    async def test_get_authorization_url(self):
        """Test SAML AuthnRequest URL generation."""
        from aragora.auth.saml import SAMLProvider, SAMLConfig
        from aragora.auth.sso import SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            enabled=True,
            entity_id="https://aragora.example.com/saml/metadata",
            callback_url="https://aragora.example.com/saml/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----",
        )

        provider = SAMLProvider(config)
        url = await provider.get_authorization_url(state="test-state")

        assert "https://idp.example.com/sso" in url
        assert "SAMLRequest=" in url
        assert "RelayState=test-state" in url

    @pytest.mark.asyncio
    async def test_get_metadata(self):
        """Test SAML metadata generation."""
        from aragora.auth.saml import SAMLProvider, SAMLConfig
        from aragora.auth.sso import SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            enabled=True,
            entity_id="https://aragora.example.com/saml/metadata",
            callback_url="https://aragora.example.com/saml/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----",
        )

        provider = SAMLProvider(config)
        metadata = await provider.get_metadata()

        assert "EntityDescriptor" in metadata
        assert "https://aragora.example.com/saml/metadata" in metadata
        assert "AssertionConsumerService" in metadata


class TestSSOSettings:
    """Test SSO settings configuration."""

    def test_sso_settings_defaults(self):
        """Test SSO settings default values."""
        from aragora.config.settings import get_settings, reset_settings

        reset_settings()
        settings = get_settings()

        assert settings.sso.enabled is False
        assert settings.sso.provider_type == "oidc"
        assert settings.sso.auto_provision is True
        assert settings.sso.session_duration == 28800

    def test_sso_settings_from_env(self):
        """Test SSO settings from environment variables."""
        import os
        from aragora.config.settings import reset_settings, SSOSettings

        os.environ["ARAGORA_SSO_ENABLED"] = "true"
        os.environ["ARAGORA_SSO_PROVIDER_TYPE"] = "azure_ad"
        os.environ["ARAGORA_SSO_CLIENT_ID"] = "test-client-id"
        os.environ["ARAGORA_SSO_ALLOWED_DOMAINS"] = "example.com,company.org"

        try:
            settings = SSOSettings()

            assert settings.enabled is True
            assert settings.provider_type == "azure_ad"
            assert settings.client_id == "test-client-id"
            assert settings.allowed_domains == ["example.com", "company.org"]

        finally:
            # Cleanup
            os.environ.pop("ARAGORA_SSO_ENABLED", None)
            os.environ.pop("ARAGORA_SSO_PROVIDER_TYPE", None)
            os.environ.pop("ARAGORA_SSO_CLIENT_ID", None)
            os.environ.pop("ARAGORA_SSO_ALLOWED_DOMAINS", None)
            reset_settings()

    def test_sso_provider_type_validation(self):
        """Test SSO provider type validation."""
        import pytest
        from pydantic import ValidationError
        from aragora.config.settings import SSOSettings

        with pytest.raises(ValidationError):
            # Invalid provider type via direct instantiation
            import os
            os.environ["ARAGORA_SSO_PROVIDER_TYPE"] = "invalid_provider"
            try:
                SSOSettings()
            finally:
                os.environ.pop("ARAGORA_SSO_PROVIDER_TYPE", None)


class TestSSOHandler:
    """Test SSO handler endpoints."""

    def _get_status(self, result) -> int:
        """Extract status from handler result (dict or HandlerResult)."""
        if hasattr(result, "status_code"):
            return result.status_code
        if hasattr(result, "status"):
            return result.status
        if isinstance(result, dict):
            return result.get("status", 200)
        return 200

    def _get_body(self, result) -> dict:
        """Extract body from handler result."""
        import json

        if hasattr(result, "body"):
            body = result.body
        elif isinstance(result, dict):
            body = result.get("body", {})
        else:
            return {}

        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        if isinstance(body, str):
            return json.loads(body)
        return body

    @pytest.mark.asyncio
    async def test_status_not_configured(self):
        """Test SSO status when not configured."""
        from aragora.server.handlers.sso import SSOHandler

        handler = SSOHandler()
        handler._initialized = True
        handler._provider = None

        result = await handler.handle_status(MagicMock(), {})

        assert self._get_status(result) == 200
        body = self._get_body(result)
        assert body["enabled"] is False
        assert body["configured"] is False

    @pytest.mark.asyncio
    async def test_login_not_configured(self):
        """Test login when SSO not configured."""
        from aragora.server.handlers.sso import SSOHandler

        handler = SSOHandler()
        handler._initialized = True
        handler._provider = None

        result = await handler.handle_login(MagicMock(), {})

        assert self._get_status(result) == 501
        body = self._get_body(result)
        assert "SSO_NOT_CONFIGURED" in str(body)

    @pytest.mark.asyncio
    async def test_callback_no_code(self):
        """Test callback without authorization code."""
        from aragora.server.handlers.sso import SSOHandler
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        handler = SSOHandler()

        # Create mock provider
        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
        )
        provider = OIDCProvider(config)
        handler._initialized = True
        handler._provider = provider

        result = await handler.handle_callback(MagicMock(), {})

        assert self._get_status(result) == 401


class TestGetSSOProvider:
    """Test SSO provider factory function."""

    def test_get_provider_not_configured(self):
        """Test get_sso_provider when not configured."""
        from aragora.auth.sso import get_sso_provider, reset_sso_provider
        from aragora.config.settings import reset_settings

        reset_settings()
        reset_sso_provider()

        provider = get_sso_provider()

        assert provider is None

    def test_reset_provider(self):
        """Test reset_sso_provider."""
        from aragora.auth.sso import _sso_initialized, reset_sso_provider

        reset_sso_provider()

        # Module-level check
        import aragora.auth.sso as sso_module
        assert sso_module._sso_initialized is False
        assert sso_module._sso_provider is None
