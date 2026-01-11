"""
Tests for the Aragora SSO Authentication System.

Covers:
- SSO base classes (SSOConfig, SSOUser, SSOProvider)
- SAML provider configuration and authentication
- OIDC provider configuration and authentication
- State management and CSRF protection
- Domain restrictions and role mapping
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Optional, List
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


# =============================================================================
# SSO Base Classes Tests
# =============================================================================


class TestSSOProviderType:
    """Test SSOProviderType enum."""

    def test_all_provider_types_exist(self):
        """Test that expected provider types exist."""
        from aragora.auth.sso import SSOProviderType

        expected = ["SAML", "OIDC", "AZURE_AD", "OKTA", "GOOGLE", "GITHUB"]
        for ptype in expected:
            assert hasattr(SSOProviderType, ptype)

    def test_provider_type_values(self):
        """Test provider type values."""
        from aragora.auth.sso import SSOProviderType

        assert SSOProviderType.SAML.value == "saml"
        assert SSOProviderType.OIDC.value == "oidc"
        assert SSOProviderType.AZURE_AD.value == "azure_ad"


class TestSSOError:
    """Test SSO error classes."""

    def test_sso_error_creation(self):
        """Test SSO error with message and code."""
        from aragora.auth.sso import SSOError

        error = SSOError("Test error", "TEST_CODE", {"key": "value"})

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.code == "TEST_CODE"
        assert error.details == {"key": "value"}

    def test_sso_authentication_error(self):
        """Test authentication error."""
        from aragora.auth.sso import SSOAuthenticationError

        error = SSOAuthenticationError("Auth failed", {"reason": "bad_token"})

        assert error.code == "SSO_AUTH_FAILED"
        assert error.details["reason"] == "bad_token"

    def test_sso_configuration_error(self):
        """Test configuration error."""
        from aragora.auth.sso import SSOConfigurationError

        error = SSOConfigurationError("Bad config")

        assert error.code == "SSO_CONFIG_ERROR"


class TestSSOUser:
    """Test SSOUser dataclass."""

    def test_minimal_user(self):
        """Test creating user with minimal fields."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(id="user123", email="user@example.com")

        assert user.id == "user123"
        assert user.email == "user@example.com"
        assert user.name == ""
        assert user.roles == []
        assert user.groups == []

    def test_full_user(self):
        """Test creating user with all fields."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user123",
            email="user@example.com",
            name="Test User",
            first_name="Test",
            last_name="User",
            roles=["admin", "user"],
            groups=["engineering"],
            provider_type="saml",
            organization_id="org123",
        )

        assert user.name == "Test User"
        assert user.first_name == "Test"
        assert user.last_name == "User"
        assert "admin" in user.roles
        assert "engineering" in user.groups

    def test_is_admin_property(self):
        """Test is_admin property detection."""
        from aragora.auth.sso import SSOUser

        admin_user = SSOUser(id="1", email="admin@test.com", roles=["admin"])
        owner_user = SSOUser(id="2", email="owner@test.com", roles=["owner"])
        regular_user = SSOUser(id="3", email="user@test.com", roles=["user"])

        assert admin_user.is_admin is True
        assert owner_user.is_admin is True
        assert regular_user.is_admin is False

    def test_is_admin_case_insensitive(self):
        """Test is_admin checks are case-insensitive."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(id="1", email="test@test.com", roles=["ADMIN"])
        assert user.is_admin is True

        user2 = SSOUser(id="2", email="test@test.com", roles=["Administrator"])
        assert user2.is_admin is True

    def test_full_name_property(self):
        """Test full_name property."""
        from aragora.auth.sso import SSOUser

        # With first and last name
        user1 = SSOUser(id="1", email="test@test.com", first_name="John", last_name="Doe")
        assert user1.full_name == "John Doe"

        # With name only
        user2 = SSOUser(id="2", email="test@test.com", name="Jane Smith")
        assert user2.full_name == "Jane Smith"

        # With display_name
        user3 = SSOUser(id="3", email="test@test.com", display_name="Bob")
        assert user3.full_name == "Bob"

        # Fallback to email prefix
        user4 = SSOUser(id="4", email="charlie@test.com")
        assert user4.full_name == "charlie"

    def test_to_dict(self):
        """Test converting user to dictionary."""
        from aragora.auth.sso import SSOUser

        user = SSOUser(
            id="user123",
            email="user@example.com",
            name="Test User",
            roles=["admin"],
            groups=["dev"],
            provider_type="saml",
        )

        data = user.to_dict()

        assert data["id"] == "user123"
        assert data["email"] == "user@example.com"
        assert data["roles"] == ["admin"]
        assert data["groups"] == ["dev"]
        assert data["provider_type"] == "saml"
        assert data["is_admin"] is True

    def test_authenticated_at_auto_set(self):
        """Test authenticated_at is automatically set."""
        from aragora.auth.sso import SSOUser

        before = time.time()
        user = SSOUser(id="1", email="test@test.com")
        after = time.time()

        assert before <= user.authenticated_at <= after


class TestSSOConfig:
    """Test SSOConfig dataclass."""

    def test_minimal_config(self):
        """Test creating config with minimal fields."""
        from aragora.auth.sso import SSOConfig, SSOProviderType

        config = SSOConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://example.com",
            callback_url="https://example.com/callback",
        )

        assert config.provider_type == SSOProviderType.SAML
        assert config.enabled is False
        assert config.auto_provision is True

    def test_config_validation_passes(self):
        """Test config validation passes for valid config."""
        from aragora.auth.sso import SSOConfig, SSOProviderType

        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="my-client-id",
            callback_url="https://example.com/callback",
        )

        errors = config.validate()
        assert errors == []

    def test_config_validation_missing_entity_id(self):
        """Test config validation fails for missing entity_id."""
        from aragora.auth.sso import SSOConfig, SSOProviderType

        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="",
            callback_url="https://example.com/callback",
        )

        errors = config.validate()
        assert "entity_id is required" in errors

    def test_config_validation_missing_callback_url(self):
        """Test config validation fails for missing callback_url."""
        from aragora.auth.sso import SSOConfig, SSOProviderType

        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="my-client-id",
            callback_url="",
        )

        errors = config.validate()
        assert "callback_url is required" in errors


# =============================================================================
# SSOProvider Base Tests
# =============================================================================


class TestSSOProviderBase:
    """Test SSOProvider base class functionality."""

    def test_generate_state(self):
        """Test generating state for CSRF protection."""
        from aragora.auth.sso import SSOProviderType
        from aragora.auth.oidc import OIDCProvider, OIDCConfig

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        provider = OIDCProvider(config)
        state1 = provider.generate_state()
        state2 = provider.generate_state()

        assert len(state1) > 16  # Should be a secure random string
        assert state1 != state2  # Should be unique

    def test_validate_state_success(self):
        """Test validating a valid state."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        provider = OIDCProvider(config)
        state = provider.generate_state()

        is_valid = provider.validate_state(state)
        assert is_valid is True

    def test_validate_state_expired(self):
        """Test validating an expired state."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        provider = OIDCProvider(config)

        # Generate state in the past
        with patch("time.time", return_value=1000.0):
            state = provider.generate_state()

        # Validate 11 minutes later (expired)
        with patch("time.time", return_value=1660.0):
            is_valid = provider.validate_state(state)
            assert is_valid is False

    def test_validate_state_unknown(self):
        """Test validating an unknown state."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        provider = OIDCProvider(config)
        is_valid = provider.validate_state("unknown-state")
        assert is_valid is False

    def test_cleanup_expired_states(self):
        """Test cleaning up expired states."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        provider = OIDCProvider(config)

        # Generate states in the past
        with patch("time.time", return_value=1000.0):
            provider.generate_state()
            provider.generate_state()
            provider.generate_state()

        # Clean up 11 minutes later
        with patch("time.time", return_value=1660.0):
            removed = provider.cleanup_expired_states()
            assert removed == 3

    def test_map_roles(self):
        """Test role mapping."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test",
            client_secret="secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test",
            role_mapping={
                "idp_admin": "admin",
                "idp_user": "user",
            },
        )

        provider = OIDCProvider(config)
        mapped = provider.map_roles(["idp_admin", "idp_user", "other_role"])

        assert "admin" in mapped
        assert "user" in mapped
        assert "other_role" in mapped  # Unmapped roles pass through

    def test_map_roles_default(self):
        """Test default role when no roles mapped."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test",
            client_secret="secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test",
            default_role="viewer",
        )

        provider = OIDCProvider(config)
        mapped = provider.map_roles([])

        assert "viewer" in mapped

    def test_is_domain_allowed_no_restrictions(self):
        """Test domain check with no restrictions."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test",
            client_secret="secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test",
        )

        provider = OIDCProvider(config)

        assert provider.is_domain_allowed("user@any-domain.com") is True

    def test_is_domain_allowed_with_restrictions(self):
        """Test domain check with restrictions."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test",
            client_secret="secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test",
            allowed_domains=["example.com", "corp.example.com"],
        )

        provider = OIDCProvider(config)

        assert provider.is_domain_allowed("user@example.com") is True
        assert provider.is_domain_allowed("user@corp.example.com") is True
        assert provider.is_domain_allowed("user@other.com") is False


# =============================================================================
# SAML Provider Tests
# =============================================================================


class TestSAMLConfig:
    """Test SAMLConfig dataclass."""

    def test_saml_config_defaults(self):
        """Test SAML config default values."""
        from aragora.auth.saml import SAMLConfig
        from aragora.auth.sso import SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com/saml/metadata",
            callback_url="https://sp.example.com/saml/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----...",
        )

        assert config.provider_type == SSOProviderType.SAML
        assert config.want_assertions_signed is True
        assert config.authn_request_signed is False

    def test_saml_config_validation_passes(self):
        """Test SAML config validation passes."""
        from aragora.auth.saml import SAMLConfig
        from aragora.auth.sso import SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/callback",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----",
        )

        errors = config.validate()
        assert errors == []

    def test_saml_config_validation_missing_idp_entity_id(self):
        """Test SAML config validation fails for missing IdP entity ID."""
        from aragora.auth.saml import SAMLConfig
        from aragora.auth.sso import SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/callback",
            idp_entity_id="",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="cert",
        )

        errors = config.validate()
        assert "idp_entity_id is required" in errors

    def test_saml_config_validation_missing_idp_sso_url(self):
        """Test SAML config validation fails for missing IdP SSO URL."""
        from aragora.auth.saml import SAMLConfig
        from aragora.auth.sso import SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/callback",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="",
            idp_certificate="cert",
        )

        errors = config.validate()
        assert "idp_sso_url is required" in errors

    def test_saml_config_validation_missing_certificate(self):
        """Test SAML config validation fails for missing certificate."""
        from aragora.auth.saml import SAMLConfig
        from aragora.auth.sso import SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/callback",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="",
        )

        errors = config.validate()
        assert any("idp_certificate is required" in e for e in errors)

    def test_saml_config_signed_requests_require_key(self):
        """Test signed requests require SP private key."""
        from aragora.auth.saml import SAMLConfig
        from aragora.auth.sso import SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/callback",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="cert",
            authn_request_signed=True,
            sp_private_key="",
        )

        errors = config.validate()
        assert "sp_private_key required" in errors[0]


class TestSAMLProvider:
    """Test SAMLProvider class."""

    @pytest.fixture
    def saml_config(self):
        """Create a valid SAML config for testing."""
        from aragora.auth.saml import SAMLConfig
        from aragora.auth.sso import SSOProviderType

        return SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/saml/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\nTEST\n-----END CERTIFICATE-----",
        )

    def test_saml_provider_creation(self, saml_config):
        """Test SAML provider creation."""
        from aragora.auth.saml import SAMLProvider
        from aragora.auth.sso import SSOProviderType

        # Mock to prevent production warning
        with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
            provider = SAMLProvider(saml_config)

        assert provider.provider_type == SSOProviderType.SAML

    def test_saml_provider_invalid_config_raises(self):
        """Test SAML provider raises for invalid config."""
        from aragora.auth.saml import SAMLProvider, SAMLConfig
        from aragora.auth.sso import SSOConfigurationError, SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="",  # Invalid
            callback_url="https://sp.example.com",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="cert",
        )

        with pytest.raises(SSOConfigurationError):
            SAMLProvider(config)

    @pytest.mark.asyncio
    async def test_saml_get_authorization_url(self, saml_config):
        """Test generating SAML authorization URL."""
        from aragora.auth.saml import SAMLProvider

        with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
            provider = SAMLProvider(saml_config)

        url = await provider.get_authorization_url(state="test-state")

        assert url.startswith(saml_config.idp_sso_url)
        assert "SAMLRequest=" in url
        assert "RelayState=test-state" in url

    @pytest.mark.asyncio
    async def test_saml_get_metadata(self, saml_config):
        """Test generating SP metadata."""
        from aragora.auth.saml import SAMLProvider

        with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
            provider = SAMLProvider(saml_config)

        metadata = await provider.get_metadata()

        assert "EntityDescriptor" in metadata
        assert saml_config.entity_id in metadata
        assert "AssertionConsumerService" in metadata


class TestSAMLAuthentication:
    """Test SAML authentication flow."""

    @pytest.fixture
    def saml_provider(self):
        """Create SAML provider for testing."""
        from aragora.auth.saml import SAMLProvider, SAMLConfig
        from aragora.auth.sso import SSOProviderType

        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/saml/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\nTEST\n-----END CERTIFICATE-----",
        )

        with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
            return SAMLProvider(config)

    @pytest.mark.asyncio
    async def test_authenticate_no_response_raises(self, saml_provider):
        """Test authentication without response raises error."""
        from aragora.auth.sso import SSOAuthenticationError

        with pytest.raises(SSOAuthenticationError, match="No SAML response"):
            await saml_provider.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_parses_valid_response(self, saml_provider):
        """Test authentication parses valid SAML response."""
        # Create a minimal valid SAML response
        saml_response_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
                        xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">
            <samlp:Status>
                <samlp:StatusCode Value="urn:oasis:names:tc:SAML:2.0:status:Success"/>
            </samlp:Status>
            <saml:Assertion>
                <saml:Subject>
                    <saml:NameID>user@example.com</saml:NameID>
                </saml:Subject>
                <saml:AttributeStatement>
                    <saml:Attribute Name="email">
                        <saml:AttributeValue>user@example.com</saml:AttributeValue>
                    </saml:Attribute>
                    <saml:Attribute Name="name">
                        <saml:AttributeValue>Test User</saml:AttributeValue>
                    </saml:Attribute>
                </saml:AttributeStatement>
            </saml:Assertion>
        </samlp:Response>"""

        encoded_response = base64.b64encode(saml_response_xml.encode()).decode()

        user = await saml_provider.authenticate(saml_response=encoded_response)

        assert user.id == "user@example.com"
        assert user.email == "user@example.com"
        assert user.provider_type == "saml"


# =============================================================================
# OIDC Provider Tests
# =============================================================================


class TestOIDCConfig:
    """Test OIDCConfig dataclass."""

    def test_oidc_config_defaults(self):
        """Test OIDC config default values."""
        from aragora.auth.oidc import OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        assert config.provider_type == SSOProviderType.OIDC
        assert config.use_pkce is True
        assert config.validate_tokens is True
        assert "openid" in config.scopes

    def test_oidc_config_validation_passes(self):
        """Test OIDC config validation passes."""
        from aragora.auth.oidc import OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        errors = config.validate()
        assert errors == []

    def test_oidc_config_validation_missing_client_id(self):
        """Test OIDC config validation fails for missing client_id."""
        from aragora.auth.oidc import OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test",
        )

        errors = config.validate()
        assert "client_id is required" in errors

    def test_oidc_config_validation_missing_client_secret(self):
        """Test OIDC config validation fails for missing client_secret."""
        from aragora.auth.oidc import OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test",
        )

        errors = config.validate()
        assert "client_secret is required" in errors

    def test_oidc_config_requires_issuer_or_endpoints(self):
        """Test OIDC config requires issuer_url or explicit endpoints."""
        from aragora.auth.oidc import OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="",
            callback_url="https://app.example.com/callback",
            entity_id="test",
        )

        errors = config.validate()
        assert any("issuer_url or explicit endpoints" in e for e in errors)


class TestOIDCProvider:
    """Test OIDCProvider class."""

    @pytest.fixture
    def oidc_config(self):
        """Create a valid OIDC config for testing."""
        from aragora.auth.oidc import OIDCConfig
        from aragora.auth.sso import SSOProviderType

        return OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

    def test_oidc_provider_creation(self, oidc_config):
        """Test OIDC provider creation."""
        from aragora.auth.oidc import OIDCProvider
        from aragora.auth.sso import SSOProviderType

        provider = OIDCProvider(oidc_config)

        assert provider.provider_type == SSOProviderType.OIDC

    def test_oidc_provider_invalid_config_raises(self):
        """Test OIDC provider raises for invalid config."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOConfigurationError, SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="",  # Invalid
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="",
        )

        with pytest.raises(SSOConfigurationError):
            OIDCProvider(config)

    @pytest.mark.asyncio
    async def test_oidc_get_authorization_url(self, oidc_config):
        """Test generating OIDC authorization URL."""
        from aragora.auth.oidc import OIDCProvider

        provider = OIDCProvider(oidc_config)

        # Mock discovery
        with patch.object(provider, "_discover_endpoints", return_value={
            "authorization_endpoint": "https://idp.example.com/authorize"
        }):
            url = await provider.get_authorization_url(state="test-state")

        assert "https://idp.example.com/authorize" in url
        assert "client_id=test-client" in url
        assert "state=test-state" in url
        assert "response_type=code" in url

    @pytest.mark.asyncio
    async def test_oidc_get_authorization_url_with_pkce(self, oidc_config):
        """Test OIDC authorization URL includes PKCE challenge."""
        from aragora.auth.oidc import OIDCProvider

        oidc_config.use_pkce = True
        provider = OIDCProvider(oidc_config)

        with patch.object(provider, "_discover_endpoints", return_value={
            "authorization_endpoint": "https://idp.example.com/authorize"
        }):
            url = await provider.get_authorization_url(state="test-state")

        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url


class TestOIDCAuthentication:
    """Test OIDC authentication flow."""

    @pytest.fixture
    def oidc_provider(self):
        """Create OIDC provider for testing."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
            use_pkce=False,  # Simplify testing
        )

        return OIDCProvider(config)

    @pytest.mark.asyncio
    async def test_authenticate_no_code_raises(self, oidc_provider):
        """Test authentication without code raises error."""
        from aragora.auth.sso import SSOAuthenticationError

        with pytest.raises(SSOAuthenticationError, match="No authorization code"):
            await oidc_provider.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_invalid_state_raises(self, oidc_provider):
        """Test authentication with invalid state raises error."""
        from aragora.auth.sso import SSOAuthenticationError

        with pytest.raises(SSOAuthenticationError, match="Invalid or expired state"):
            await oidc_provider.authenticate(code="test-code", state="invalid-state")

    @pytest.mark.asyncio
    async def test_authenticate_exchanges_code(self, oidc_provider):
        """Test authentication exchanges code for tokens."""
        # Generate valid state
        state = oidc_provider.generate_state()

        # Mock token exchange and userinfo
        mock_tokens = {
            "access_token": "test-access-token",
            "id_token": "test-id-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
        }

        mock_userinfo = {
            "sub": "user123",
            "email": "user@example.com",
            "name": "Test User",
        }

        with patch.object(oidc_provider, "_exchange_code", return_value=mock_tokens):
            with patch.object(oidc_provider, "_fetch_userinfo", return_value=mock_userinfo):
                user = await oidc_provider.authenticate(code="test-code", state=state)

        assert user.id == "user123"
        assert user.email == "user@example.com"
        assert user.access_token == "test-access-token"

    @pytest.mark.asyncio
    async def test_authenticate_domain_restriction(self, oidc_provider):
        """Test authentication respects domain restrictions."""
        from aragora.auth.sso import SSOAuthenticationError

        # Set domain restriction
        oidc_provider.config.allowed_domains = ["allowed.com"]

        state = oidc_provider.generate_state()

        mock_tokens = {"access_token": "test", "expires_in": 3600}
        mock_userinfo = {
            "sub": "user123",
            "email": "user@notallowed.com",
            "name": "Test User",
        }

        with patch.object(oidc_provider, "_exchange_code", return_value=mock_tokens):
            with patch.object(oidc_provider, "_fetch_userinfo", return_value=mock_userinfo):
                with pytest.raises(SSOAuthenticationError, match="domain not allowed"):
                    await oidc_provider.authenticate(code="test-code", state=state)


class TestOIDCTokenRefresh:
    """Test OIDC token refresh functionality."""

    @pytest.mark.asyncio
    async def test_refresh_token_success(self):
        """Test successful token refresh."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOUser, SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        provider = OIDCProvider(config)

        user = SSOUser(
            id="user123",
            email="user@example.com",
            refresh_token="old-refresh-token",
        )

        with patch.object(provider, "_get_endpoint", return_value="https://idp.example.com/token"):
            with patch("aragora.auth.oidc.HAS_HTTPX", False):
                with patch("urllib.request.urlopen") as mock_urlopen:
                    mock_response = MagicMock()
                    mock_response.read.return_value = b'{"access_token": "new-access-token", "refresh_token": "new-refresh-token", "expires_in": 3600}'
                    mock_response.__enter__ = MagicMock(return_value=mock_response)
                    mock_response.__exit__ = MagicMock(return_value=False)
                    mock_urlopen.return_value = mock_response

                    refreshed_user = await provider.refresh_token(user)

        assert refreshed_user is not None
        assert refreshed_user.access_token == "new-access-token"

    @pytest.mark.asyncio
    async def test_refresh_token_no_refresh_token(self):
        """Test refresh fails without refresh token."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOUser, SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        provider = OIDCProvider(config)

        user = SSOUser(
            id="user123",
            email="user@example.com",
            refresh_token=None,  # No refresh token
        )

        result = await provider.refresh_token(user)
        assert result is None


# =============================================================================
# Provider Discovery Tests
# =============================================================================


class TestOIDCDiscovery:
    """Test OIDC discovery functionality."""

    @pytest.mark.asyncio
    async def test_discover_endpoints_caches_result(self):
        """Test discovery caches results."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        provider = OIDCProvider(config)

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = b'{"authorization_endpoint": "https://idp.example.com/authorize", "token_endpoint": "https://idp.example.com/token"}'
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                # First call should fetch
                result1 = await provider._discover_endpoints()

                # Second call should use cache
                result2 = await provider._discover_endpoints()

                # urlopen should only be called once
                assert mock_urlopen.call_count == 1

    @pytest.mark.asyncio
    async def test_discover_endpoints_handles_failure(self):
        """Test discovery handles failures gracefully."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test-client",
        )

        provider = OIDCProvider(config)

        with patch("aragora.auth.oidc.HAS_HTTPX", False):
            with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
                result = await provider._discover_endpoints()

        assert result == {}  # Returns empty dict on failure


# =============================================================================
# Global Provider Tests
# =============================================================================


class TestGlobalSSOProvider:
    """Test global SSO provider management."""

    def test_reset_sso_provider(self):
        """Test resetting the global SSO provider."""
        from aragora.auth.sso import reset_sso_provider

        reset_sso_provider()

        # Import after reset to check state
        from aragora.auth import sso
        assert sso._sso_initialized is False
        assert sso._sso_provider is None


class TestProviderConfigs:
    """Test well-known provider configurations."""

    def test_provider_configs_exist(self):
        """Test well-known provider configs exist."""
        from aragora.auth.oidc import PROVIDER_CONFIGS

        assert "azure_ad" in PROVIDER_CONFIGS
        assert "okta" in PROVIDER_CONFIGS
        assert "google" in PROVIDER_CONFIGS
        assert "github" in PROVIDER_CONFIGS

    def test_azure_ad_config_has_required_endpoints(self):
        """Test Azure AD config has required endpoints."""
        from aragora.auth.oidc import PROVIDER_CONFIGS

        azure_config = PROVIDER_CONFIGS["azure_ad"]

        assert "authorization_endpoint" in azure_config
        assert "token_endpoint" in azure_config
        assert "userinfo_endpoint" in azure_config
        assert "jwks_uri" in azure_config

    def test_google_config_has_required_endpoints(self):
        """Test Google config has required endpoints."""
        from aragora.auth.oidc import PROVIDER_CONFIGS

        google_config = PROVIDER_CONFIGS["google"]

        assert "authorization_endpoint" in google_config
        assert "token_endpoint" in google_config
        assert "userinfo_endpoint" in google_config


class TestPKCEGeneration:
    """Test PKCE code verifier and challenge generation."""

    def test_generate_pkce_produces_valid_verifier(self):
        """Test PKCE verifier is valid length."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test",
            client_secret="secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test",
        )

        provider = OIDCProvider(config)
        verifier, challenge = provider._generate_pkce()

        # Verifier should be between 43-128 characters
        assert len(verifier) >= 43
        assert len(verifier) <= 128

    def test_generate_pkce_challenge_is_base64url(self):
        """Test PKCE challenge is base64url encoded."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test",
            client_secret="secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test",
        )

        provider = OIDCProvider(config)
        verifier, challenge = provider._generate_pkce()

        # Challenge should be base64url without padding
        assert "=" not in challenge
        assert "+" not in challenge
        assert "/" not in challenge

    def test_generate_pkce_unique_each_time(self):
        """Test PKCE generates unique values each time."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            client_id="test",
            client_secret="secret",
            issuer_url="https://idp.example.com",
            callback_url="https://app.example.com/callback",
            entity_id="test",
        )

        provider = OIDCProvider(config)

        v1, c1 = provider._generate_pkce()
        v2, c2 = provider._generate_pkce()

        assert v1 != v2
        assert c1 != c2
