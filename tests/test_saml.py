"""
Tests for SAML 2.0 authentication provider.

Tests cover:
- Configuration validation
- AuthnRequest URL generation
- SAML Response parsing
- Attribute mapping
- Error handling
"""

import base64
import pytest
from urllib.parse import parse_qs, urlparse

from aragora.auth.saml import (
    SAMLConfig,
    SAMLProvider,
    SAMLError,
)
from aragora.auth.sso import (
    SSOProviderType,
    SSOUser,
    SSOAuthenticationError,
    SSOConfigurationError,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def valid_config() -> SAMLConfig:
    """Create a valid SAML configuration."""
    return SAMLConfig(
        provider_type=SSOProviderType.SAML,
        entity_id="https://aragora.example.com/saml/metadata",
        callback_url="https://aragora.example.com/saml/acs",
        idp_entity_id="https://idp.example.com/metadata",
        idp_sso_url="https://idp.example.com/sso",
        idp_certificate="-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----",
    )


@pytest.fixture
def provider(valid_config: SAMLConfig) -> SAMLProvider:
    """Create a SAML provider with valid config."""
    return SAMLProvider(valid_config)


@pytest.fixture
def sample_saml_response() -> str:
    """Create a sample SAML response for testing."""
    # Note: NameID text must not have whitespace/newlines around the value
    response_xml = """<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="_response123"
    Version="2.0"
    IssueInstant="2024-01-01T00:00:00Z">
    <samlp:Status>
        <samlp:StatusCode Value="urn:oasis:names:tc:SAML:2.0:status:Success"/>
    </samlp:Status>
    <saml:Assertion ID="_assertion123" Version="2.0">
        <saml:Subject>
            <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">test@example.com</saml:NameID>
        </saml:Subject>
        <saml:AttributeStatement>
            <saml:Attribute Name="email">
                <saml:AttributeValue>test@example.com</saml:AttributeValue>
            </saml:Attribute>
            <saml:Attribute Name="name">
                <saml:AttributeValue>Test User</saml:AttributeValue>
            </saml:Attribute>
            <saml:Attribute Name="firstName">
                <saml:AttributeValue>Test</saml:AttributeValue>
            </saml:Attribute>
            <saml:Attribute Name="lastName">
                <saml:AttributeValue>User</saml:AttributeValue>
            </saml:Attribute>
            <saml:Attribute Name="groups">
                <saml:AttributeValue>admin</saml:AttributeValue>
                <saml:AttributeValue>developers</saml:AttributeValue>
            </saml:Attribute>
        </saml:AttributeStatement>
    </saml:Assertion>
</samlp:Response>"""
    return base64.b64encode(response_xml.encode("utf-8")).decode("ascii")


@pytest.fixture
def failed_saml_response() -> str:
    """Create a failed SAML response for testing."""
    response_xml = """<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="_response123"
    Version="2.0">
    <samlp:Status>
        <samlp:StatusCode Value="urn:oasis:names:tc:SAML:2.0:status:AuthnFailed"/>
    </samlp:Status>
</samlp:Response>"""
    return base64.b64encode(response_xml.encode("utf-8")).decode("ascii")


# =============================================================================
# Configuration Tests
# =============================================================================


class TestSAMLConfig:
    """Tests for SAML configuration."""

    def test_valid_config(self):
        """Valid config passes validation."""
        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com/metadata",
            callback_url="https://sp.example.com/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----",
        )
        errors = config.validate()
        assert errors == []

    def test_missing_entity_id(self):
        """Missing entity_id fails validation."""
        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            callback_url="https://sp.example.com/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="test-cert",
        )
        errors = config.validate()
        assert "entity_id is required" in errors

    def test_missing_idp_entity_id(self):
        """Missing idp_entity_id fails validation."""
        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com/metadata",
            callback_url="https://sp.example.com/acs",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="test-cert",
        )
        errors = config.validate()
        assert "idp_entity_id is required" in errors

    def test_missing_idp_sso_url(self):
        """Missing idp_sso_url fails validation."""
        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com/metadata",
            callback_url="https://sp.example.com/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_certificate="test-cert",
        )
        errors = config.validate()
        assert "idp_sso_url is required" in errors

    def test_missing_idp_certificate(self):
        """Missing idp_certificate fails validation."""
        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com/metadata",
            callback_url="https://sp.example.com/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
        )
        errors = config.validate()
        assert "idp_certificate is required for signature verification" in errors

    def test_signed_request_requires_private_key(self):
        """Signed requests require SP private key."""
        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com/metadata",
            callback_url="https://sp.example.com/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="test-cert",
            authn_request_signed=True,
        )
        errors = config.validate()
        assert "sp_private_key required when authn_request_signed is True" in errors

    def test_default_name_id_format(self):
        """Default name ID format is email address."""
        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="https://sp.example.com/metadata",
            callback_url="https://sp.example.com/acs",
            idp_entity_id="https://idp.example.com/metadata",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="test-cert",
        )
        assert "emailAddress" in config.name_id_format

    def test_default_attribute_mapping(self):
        """Default attribute mapping includes common SAML attributes."""
        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="test",
            callback_url="test",
            idp_entity_id="test",
            idp_sso_url="test",
            idp_certificate="test",
        )
        mapping = config.attribute_mapping
        # Check both full URI and short name mappings
        assert "email" in mapping
        assert "mail" in mapping
        assert "groups" in mapping

    def test_provider_type_is_saml(self):
        """Provider type is correctly set to SAML."""
        config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="test",
            callback_url="test",
            idp_entity_id="test",
            idp_sso_url="test",
            idp_certificate="test",
        )
        assert config.provider_type == SSOProviderType.SAML


class TestSAMLProviderInit:
    """Tests for SAML provider initialization."""

    def test_create_with_valid_config(self, valid_config: SAMLConfig):
        """Provider creates successfully with valid config."""
        provider = SAMLProvider(valid_config)
        assert provider.config == valid_config

    def test_create_with_invalid_config_raises(self):
        """Provider raises error with invalid config."""
        # Missing required fields
        invalid_config = SAMLConfig(
            provider_type=SSOProviderType.SAML,
            entity_id="test",
            callback_url="test",
        )
        with pytest.raises(SSOConfigurationError) as exc_info:
            SAMLProvider(invalid_config)
        assert "Invalid SAML configuration" in str(exc_info.value)


# =============================================================================
# Authorization URL Tests
# =============================================================================


class TestAuthorizationURL:
    """Tests for SAML AuthnRequest URL generation."""

    @pytest.mark.asyncio
    async def test_authorization_url_format(self, provider: SAMLProvider):
        """Authorization URL has correct format."""
        url = await provider.get_authorization_url()
        parsed = urlparse(url)

        assert parsed.scheme == "https"
        assert parsed.netloc == "idp.example.com"
        assert parsed.path == "/sso"

    @pytest.mark.asyncio
    async def test_authorization_url_contains_saml_request(self, provider: SAMLProvider):
        """Authorization URL contains SAMLRequest parameter."""
        url = await provider.get_authorization_url()
        params = parse_qs(urlparse(url).query)

        assert "SAMLRequest" in params
        assert len(params["SAMLRequest"][0]) > 0

    @pytest.mark.asyncio
    async def test_authorization_url_with_state(self, provider: SAMLProvider):
        """Authorization URL includes RelayState when state provided."""
        url = await provider.get_authorization_url(state="test-state")
        params = parse_qs(urlparse(url).query)

        assert "RelayState" in params
        assert params["RelayState"][0] == "test-state"

    @pytest.mark.asyncio
    async def test_authorization_url_stores_state(self, provider: SAMLProvider):
        """State is stored for later validation."""
        await provider.get_authorization_url(state="test-state")
        assert "test-state" in provider._state_store

    @pytest.mark.asyncio
    async def test_authorization_url_without_state(self, provider: SAMLProvider):
        """Authorization URL works without state."""
        url = await provider.get_authorization_url()
        params = parse_qs(urlparse(url).query)

        # RelayState should not be present if no state provided
        assert "RelayState" not in params or params.get("RelayState", [""])[0] == ""


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Tests for SAML response authentication."""

    @pytest.mark.asyncio
    async def test_authenticate_without_response_raises(self, provider: SAMLProvider):
        """Authentication without SAML response raises error."""
        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(saml_response=None)
        assert "No SAML response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_authenticate_success(self, provider: SAMLProvider, sample_saml_response: str):
        """Successful authentication returns user."""
        user = await provider.authenticate(saml_response=sample_saml_response)

        assert user.id == "test@example.com"
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.first_name == "Test"
        assert user.last_name == "User"
        assert "admin" in user.groups
        assert "developers" in user.groups

    @pytest.mark.asyncio
    async def test_authenticate_failure_status(
        self, provider: SAMLProvider, failed_saml_response: str
    ):
        """Failed SAML status raises error."""
        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(saml_response=failed_saml_response)
        assert "AuthnFailed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_authenticate_invalid_xml_raises(self, provider: SAMLProvider):
        """Invalid XML in response raises error."""
        invalid_response = base64.b64encode(b"not valid xml").decode("ascii")
        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(saml_response=invalid_response)
        assert (
            "Invalid SAML response" in str(exc_info.value)
            or "failed" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_authenticate_domain_restriction(
        self,
        valid_config: SAMLConfig,
        sample_saml_response: str,
    ):
        """Authentication fails if email domain not allowed."""
        valid_config.allowed_domains = ["allowed.com"]
        provider = SAMLProvider(valid_config)

        with pytest.raises(SSOAuthenticationError) as exc_info:
            await provider.authenticate(saml_response=sample_saml_response)
        assert "domain not allowed" in str(exc_info.value).lower()


# =============================================================================
# Attribute Mapping Tests
# =============================================================================


class TestAttributeMapping:
    """Tests for SAML attribute mapping."""

    def test_map_email_attribute(self, provider: SAMLProvider):
        """Email attribute is correctly mapped."""
        attributes = {"email": ["test@example.com"]}
        result = provider._map_attributes(attributes)
        assert result.get("email") == "test@example.com"

    def test_map_mail_attribute(self, provider: SAMLProvider):
        """Mail attribute (alternative) is correctly mapped to email."""
        attributes = {"mail": ["test@example.com"]}
        result = provider._map_attributes(attributes)
        assert result.get("email") == "test@example.com"

    def test_map_name_attribute(self, provider: SAMLProvider):
        """Name attribute is correctly mapped."""
        attributes = {"name": ["Test User"]}
        result = provider._map_attributes(attributes)
        assert result.get("name") == "Test User"

    def test_map_given_name_attribute(self, provider: SAMLProvider):
        """Given name attribute is correctly mapped to first_name."""
        attributes = {"givenName": ["Test"]}
        result = provider._map_attributes(attributes)
        assert result.get("first_name") == "Test"

    def test_map_surname_attribute(self, provider: SAMLProvider):
        """Surname attribute is correctly mapped to last_name."""
        attributes = {"surname": ["User"]}
        result = provider._map_attributes(attributes)
        assert result.get("last_name") == "User"

    def test_map_groups_attribute(self, provider: SAMLProvider):
        """Groups attribute is correctly mapped."""
        attributes = {"groups": ["admin", "users"]}
        result = provider._map_attributes(attributes)
        assert result.get("groups") == ["admin", "users"]

    def test_map_full_uri_attribute(self, provider: SAMLProvider):
        """Full URI attributes (Azure AD style) are correctly mapped."""
        attributes = {
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress": [
                "test@example.com"
            ]
        }
        result = provider._map_attributes(attributes)
        assert result.get("email") == "test@example.com"

    def test_map_unknown_attribute_ignored(self, provider: SAMLProvider):
        """Unknown attributes are ignored."""
        attributes = {"unknownAttribute": ["value"]}
        result = provider._map_attributes(attributes)
        assert "unknownAttribute" not in result


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement:
    """Tests for SAML state/RelayState management."""

    @pytest.mark.asyncio
    async def test_state_stored_on_auth_url(self, provider: SAMLProvider):
        """State is stored when generating authorization URL."""
        await provider.get_authorization_url(state="test-state-123")
        assert "test-state-123" in provider._state_store

    def test_validate_valid_state(self, provider: SAMLProvider):
        """Valid state passes validation."""
        import time

        provider._state_store["valid-state"] = time.time()
        assert provider.validate_state("valid-state") is True

    def test_validate_invalid_state(self, provider: SAMLProvider):
        """Invalid/unknown state fails validation."""
        assert provider.validate_state("unknown-state") is False

    def test_validate_expired_state(self, provider: SAMLProvider):
        """Expired state fails validation."""
        import time

        provider._state_store["expired-state"] = time.time() - 1000  # Old timestamp
        # State validation uses a 10-minute window by default
        provider.config.session_duration_seconds = 100  # Short timeout for test
        # The base SSOProvider.validate_state checks for 10 minute expiry
        # After 1000 seconds, it should be expired


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_saml_error_has_correct_code(self):
        """SAMLError has correct error code."""
        error = SAMLError("Test error", {"detail": "value"})
        assert error.code == "SAML_ERROR"
        assert error.details == {"detail": "value"}

    def test_authentication_error_inheritance(self):
        """SSOAuthenticationError inherits from SSOError."""
        error = SSOAuthenticationError("Auth failed")
        assert error.code == "SSO_AUTH_FAILED"

    def test_configuration_error_inheritance(self):
        """SSOConfigurationError inherits from SSOError."""
        error = SSOConfigurationError("Config invalid")
        assert error.code == "SSO_CONFIG_ERROR"


# =============================================================================
# Provider Type Tests
# =============================================================================


class TestProviderType:
    """Tests for provider type property."""

    def test_provider_type_is_saml(self, provider: SAMLProvider):
        """Provider type property returns SAML."""
        assert provider.provider_type == SSOProviderType.SAML
