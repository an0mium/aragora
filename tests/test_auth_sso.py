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


class TestSAMLResponseParsing:
    """Test SAML response parsing scenarios."""

    def _create_saml_response(
        self,
        name_id: str = "user@example.com",
        status: str = "urn:oasis:names:tc:SAML:2.0:status:Success",
    ) -> str:
        """Create a test SAML response XML."""
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="_response123">
    <samlp:Status>
        <samlp:StatusCode Value="{status}"/>
    </samlp:Status>
    <saml:Assertion>
        <saml:Subject>
            <saml:NameID>{name_id}</saml:NameID>
        </saml:Subject>
        <saml:AttributeStatement>
            <saml:Attribute Name="email">
                <saml:AttributeValue>{name_id}</saml:AttributeValue>
            </saml:Attribute>
            <saml:Attribute Name="name">
                <saml:AttributeValue>Test User</saml:AttributeValue>
            </saml:Attribute>
        </saml:AttributeStatement>
    </saml:Assertion>
</samlp:Response>"""
        return base64.b64encode(xml.encode("utf-8")).decode("ascii")

    @pytest.mark.asyncio
    async def test_parse_valid_saml_response(self):
        """Test parsing a valid SAML response."""
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
        saml_response = self._create_saml_response()

        user = await provider._authenticate_simple(saml_response, None)

        assert user.email == "user@example.com"
        assert user.id == "user@example.com"
        assert user.provider_type == "saml"

    @pytest.mark.asyncio
    async def test_parse_invalid_xml_response(self):
        """Test parsing invalid XML response."""
        from aragora.auth.saml import SAMLProvider, SAMLConfig
        from aragora.auth.sso import SSOProviderType, SSOAuthenticationError

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
        invalid_response = base64.b64encode(b"<invalid xml").decode("ascii")

        with pytest.raises(SSOAuthenticationError) as exc:
            await provider._authenticate_simple(invalid_response, None)

        assert "Invalid SAML response XML" in str(exc.value)

    @pytest.mark.asyncio
    async def test_parse_failed_status_response(self):
        """Test parsing a failed status SAML response."""
        from aragora.auth.saml import SAMLProvider, SAMLConfig
        from aragora.auth.sso import SSOProviderType, SSOAuthenticationError

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
        saml_response = self._create_saml_response(
            status="urn:oasis:names:tc:SAML:2.0:status:Requester"
        )

        with pytest.raises(SSOAuthenticationError) as exc:
            await provider._authenticate_simple(saml_response, None)

        assert "SAML authentication failed" in str(exc.value)

    @pytest.mark.asyncio
    async def test_domain_restriction_in_saml(self):
        """Test domain restriction enforcement in SAML."""
        from aragora.auth.saml import SAMLProvider, SAMLConfig
        from aragora.auth.sso import SSOProviderType, SSOAuthenticationError

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            enabled=True,
            entity_id="https://aragora.example.com/saml/metadata",
            callback_url="https://aragora.example.com/saml/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----",
            allowed_domains=["allowed.com"],
        )

        provider = SAMLProvider(config)
        saml_response = self._create_saml_response("user@blocked.com")

        with pytest.raises(SSOAuthenticationError) as exc:
            await provider._authenticate_simple(saml_response, None)

        assert "domain not allowed" in str(exc.value).lower()


class TestSAMLSecurityWarnings:
    """Test SAML security warnings for production mode."""

    def test_warning_logged_without_saml_lib(self):
        """Test that a warning is logged when python3-saml is not available."""
        import logging
        from aragora.auth.saml import SAMLProvider, SAMLConfig, HAS_SAML_LIB
        from aragora.auth.sso import SSOProviderType

        if HAS_SAML_LIB:
            pytest.skip("python3-saml is installed, skipping warning test")

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            enabled=True,
            entity_id="https://aragora.example.com/saml/metadata",
            callback_url="https://aragora.example.com/saml/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----",
        )

        with patch("aragora.auth.saml.logger") as mock_logger:
            SAMLProvider(config)
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "SECURITY WARNING" in warning_msg

    def test_production_mode_requires_saml_lib(self):
        """Test that production mode requires python3-saml."""
        import os
        from aragora.auth.saml import SAMLProvider, SAMLConfig, HAS_SAML_LIB
        from aragora.auth.sso import SSOProviderType, SSOConfigurationError

        if HAS_SAML_LIB:
            pytest.skip("python3-saml is installed, skipping production check")

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            enabled=True,
            entity_id="https://aragora.example.com/saml/metadata",
            callback_url="https://aragora.example.com/saml/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----",
        )

        os.environ["ARAGORA_ENV"] = "production"
        try:
            with pytest.raises(SSOConfigurationError) as exc:
                SAMLProvider(config)
            assert "python3-saml required for production" in str(exc.value)
        finally:
            os.environ.pop("ARAGORA_ENV", None)


class TestHTTPSEnforcement:
    """Test HTTPS enforcement for SSO callbacks."""

    @pytest.mark.asyncio
    async def test_https_enforced_in_production(self):
        """Test that HTTPS is enforced in production mode."""
        import os
        from aragora.server.handlers.sso import SSOHandler
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        handler = SSOHandler()

        # Create mock provider with HTTP callback (insecure)
        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="http://aragora.example.com/callback",  # HTTP!
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
        )
        provider = OIDCProvider(config)
        handler._initialized = True
        handler._provider = provider

        os.environ["ARAGORA_ENV"] = "production"
        try:
            result = await handler.handle_callback(MagicMock(), {"code": "auth-code"})

            # Should fail with insecure callback error
            status = (
                result.get("status")
                if isinstance(result, dict)
                else getattr(result, "status_code", 200)
            )
            assert status == 400
            body = (
                result.get("body", {}) if isinstance(result, dict) else getattr(result, "body", {})
            )
            if isinstance(body, bytes):
                import json

                body = json.loads(body.decode())
            assert "INSECURE_CALLBACK_URL" in str(body)
        finally:
            os.environ.pop("ARAGORA_ENV", None)

    @pytest.mark.asyncio
    async def test_https_not_enforced_in_development(self):
        """Test that HTTPS is not enforced in development mode."""
        import os
        from aragora.server.handlers.sso import SSOHandler
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        handler = SSOHandler()

        # Create mock provider with HTTP callback
        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="http://localhost:8080/callback",  # HTTP allowed in dev
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
        )
        provider = OIDCProvider(config)
        handler._initialized = True
        handler._provider = provider

        # Ensure not in production mode
        os.environ.pop("ARAGORA_ENV", None)

        result = await handler.handle_callback(MagicMock(), {"code": "auth-code"})

        # Should proceed (though it may fail for other reasons like no IdP)
        status = (
            result.get("status")
            if isinstance(result, dict)
            else getattr(result, "status_code", 200)
        )
        # Not 400 for INSECURE_CALLBACK_URL
        if status == 400:
            body = (
                result.get("body", {}) if isinstance(result, dict) else getattr(result, "body", {})
            )
            if isinstance(body, bytes):
                import json

                body = json.loads(body.decode())
            assert "INSECURE_CALLBACK_URL" not in str(body)


class TestPEMCertificateValidation:
    """Test PEM certificate format validation."""

    def test_valid_pem_certificate(self):
        """Test valid PEM certificate passes validation."""
        from aragora.config.settings import SSOSettings
        import os

        os.environ["ARAGORA_SSO_IDP_CERTIFICATE"] = (
            "-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----"
        )
        try:
            settings = SSOSettings()
            assert settings.idp_certificate.startswith("-----BEGIN")
        finally:
            os.environ.pop("ARAGORA_SSO_IDP_CERTIFICATE", None)

    def test_invalid_pem_certificate_rejected(self):
        """Test invalid PEM certificate is rejected."""
        from pydantic import ValidationError
        from aragora.config.settings import SSOSettings
        import os

        os.environ["ARAGORA_SSO_IDP_CERTIFICATE"] = "not-a-valid-certificate"
        try:
            with pytest.raises(ValidationError) as exc:
                SSOSettings()
            assert "PEM format" in str(exc.value)
        finally:
            os.environ.pop("ARAGORA_SSO_IDP_CERTIFICATE", None)

    def test_empty_certificate_allowed(self):
        """Test empty certificate is allowed."""
        from aragora.config.settings import SSOSettings
        import os

        # Remove any certificate env var
        os.environ.pop("ARAGORA_SSO_IDP_CERTIFICATE", None)

        settings = SSOSettings()
        assert settings.idp_certificate is None or settings.idp_certificate == ""


class TestSSOStateManagement:
    """Test SSO state parameter management."""

    def test_state_creation(self):
        """Test state is created and stored."""
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

        assert len(state) > 20
        assert state in provider._state_store

    def test_state_validation_and_cleanup(self):
        """Test state is consumed after validation."""
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

        # First validation should succeed and consume the state
        assert provider.validate_state(state) is True

        # Second validation should fail (state already used)
        assert provider.validate_state(state) is False

    def test_state_expiration(self):
        """Test expired state is rejected."""
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

        # Simulate expired state (10+ minutes old)
        provider._state_store[state] = time.time() - 700

        assert provider.validate_state(state) is False

    def test_unknown_state_rejected(self):
        """Test unknown state is rejected."""
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

        assert provider.validate_state("unknown-state") is False


class TestSSOSecurityEdgeCases:
    """Test security edge cases in SSO."""

    @pytest.mark.asyncio
    async def test_idp_error_handling(self):
        """Test IdP error response is handled."""
        from aragora.server.handlers.sso import SSOHandler
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        handler = SSOHandler()

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

        # Simulate IdP error response
        result = await handler.handle_callback(
            MagicMock(), {"error": "access_denied", "error_description": "User denied access"}
        )

        status = (
            result.get("status")
            if isinstance(result, dict)
            else getattr(result, "status_code", 200)
        )
        assert status == 401

        body = result.get("body", {}) if isinstance(result, dict) else getattr(result, "body", {})
        if isinstance(body, bytes):
            import json

            body = json.loads(body.decode())
        assert "SSO_IDP_ERROR" in str(body)

    @pytest.mark.asyncio
    async def test_session_expired_handling(self):
        """Test expired session handling."""
        from aragora.server.handlers.sso import SSOHandler
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType, SSOAuthenticationError

        handler = SSOHandler()

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

        # Mock authenticate to raise INVALID_STATE error
        async def mock_authenticate(*args, **kwargs):
            raise SSOAuthenticationError("INVALID_STATE: Session expired")

        provider.authenticate = mock_authenticate

        result = await handler.handle_callback(
            MagicMock(), {"code": "auth-code", "state": "expired-state"}
        )

        status = (
            result.get("status")
            if isinstance(result, dict)
            else getattr(result, "status_code", 200)
        )
        assert status == 401

        body = result.get("body", {}) if isinstance(result, dict) else getattr(result, "body", {})
        if isinstance(body, bytes):
            import json

            body = json.loads(body.decode())
        assert "SSO_SESSION_EXPIRED" in str(body)

    def test_logout_revokes_token(self):
        """Test logout revokes the session token."""
        from aragora.server.handlers.sso import SSOHandler

        handler = SSOHandler()
        handler._initialized = True
        handler._provider = None

        # Should not error even without provider
        import asyncio

        result = asyncio.run(handler.handle_logout(MagicMock(), {}))

        body = result.get("body", {}) if isinstance(result, dict) else getattr(result, "body", {})
        if isinstance(body, bytes):
            import json

            body = json.loads(body.decode())
        assert body.get("success") is True


class TestOIDCDiscovery:
    """Test OIDC discovery behavior."""

    @pytest.mark.asyncio
    async def test_oidc_uses_explicit_endpoints(self):
        """Test OIDC uses explicitly configured endpoints."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            client_id="test-client",
            client_secret="test-secret",
            authorization_endpoint="https://custom-idp.example.com/authorize",
            token_endpoint="https://custom-idp.example.com/token",
        )

        provider = OIDCProvider(config)
        url = await provider.get_authorization_url(state="test")

        assert "custom-idp.example.com/authorize" in url

    def test_oidc_scopes_configuration(self):
        """Test OIDC scopes configuration."""
        from aragora.auth.oidc import OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            enabled=True,
            entity_id="https://aragora.example.com",
            callback_url="https://aragora.example.com/callback",
            client_id="test-client",
            client_secret="test-secret",
            scopes=["openid", "email", "profile", "groups"],
        )

        assert "groups" in config.scopes
        assert "openid" in config.scopes


class TestRoleAndGroupMapping:
    """Test role and group mapping functionality."""

    def test_role_mapping_with_default(self):
        """Test role mapping with default role."""
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
                "Admin": "admin",
                "Member": "user",
            },
            default_role="guest",
        )

        provider = OIDCProvider(config)

        # Mapped roles
        mapped = provider.map_roles(["Admin"])
        assert "admin" in mapped

        # With unmapped roles - pass through
        mapped = provider.map_roles(["UnknownRole"])
        assert "UnknownRole" in mapped

    def test_group_mapping(self):
        """Test group mapping."""
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
            group_mapping={
                "CN=Engineering,OU=Groups,DC=example,DC=com": "engineering",
                "CN=Sales,OU=Groups,DC=example,DC=com": "sales",
            },
        )

        provider = OIDCProvider(config)

        # Mapped groups
        mapped = provider.map_groups(["CN=Engineering,OU=Groups,DC=example,DC=com"])
        assert "engineering" in mapped

        # Unmapped groups pass through
        mapped = provider.map_groups(["unmapped-group"])
        assert "unmapped-group" in mapped
