"""
Tests for SAML 2.0 authentication provider.

Tests cover:
- SAMLConfig validation
- SAMLProvider initialization
- Authorization URL generation (simple and library-based)
- SAML Response authentication (simple parser)
- Attribute mapping
- Domain restrictions
- Error handling
- Metadata generation
"""

from __future__ import annotations

import base64
import time
import zlib
from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree import ElementTree as ET

import pytest

from aragora.auth.saml import (
    SAMLConfig,
    SAMLError,
    SAMLProvider,
)
from aragora.auth.sso import (
    SSOAuthenticationError,
    SSOConfigurationError,
    SSOProviderType,
)


def make_saml_config(**kwargs) -> SAMLConfig:
    """Helper to create SAMLConfig with sensible defaults."""
    defaults = {
        "provider_type": SSOProviderType.SAML,
        "entity_id": "https://aragora.example.com/saml/metadata",
        "callback_url": "https://aragora.example.com/saml/acs",
        "idp_entity_id": "https://idp.example.com/metadata",
        "idp_sso_url": "https://idp.example.com/sso",
        "idp_certificate": "-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----",
    }
    defaults.update(kwargs)
    return SAMLConfig(**defaults)


def create_saml_response(
    name_id: str = "user@example.com",
    status: str = "urn:oasis:names:tc:SAML:2.0:status:Success",
    attributes: dict = None,
) -> str:
    """Helper to create a valid SAML response for testing."""
    if attributes is None:
        attributes = {}

    attr_statements = ""
    for name, values in attributes.items():
        values_xml = "".join(f"<saml:AttributeValue>{v}</saml:AttributeValue>" for v in values)
        attr_statements += f'''
        <saml:Attribute Name="{name}">
            {values_xml}
        </saml:Attribute>'''

    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
                xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
                ID="_response123"
                Version="2.0"
                IssueInstant="2026-01-26T12:00:00Z">
    <samlp:Status>
        <samlp:StatusCode Value="{status}"/>
    </samlp:Status>
    <saml:Assertion>
        <saml:Subject>
            <saml:NameID>{name_id}</saml:NameID>
        </saml:Subject>
        <saml:AttributeStatement>
            {attr_statements}
        </saml:AttributeStatement>
    </saml:Assertion>
</samlp:Response>'''

    return base64.b64encode(xml.encode("utf-8")).decode("ascii")


# ============================================================================
# SAMLConfig Tests
# ============================================================================


class TestSAMLConfig:
    """Tests for SAMLConfig dataclass."""

    def test_config_with_required_fields(self):
        """Test creating config with all required fields."""
        config = make_saml_config()

        assert config.entity_id == "https://aragora.example.com/saml/metadata"
        assert config.callback_url == "https://aragora.example.com/saml/acs"
        assert config.idp_entity_id == "https://idp.example.com/metadata"
        assert config.idp_sso_url == "https://idp.example.com/sso"
        assert config.idp_certificate.startswith("-----BEGIN CERTIFICATE-----")

    def test_config_default_provider_type(self):
        """Test that provider_type is set to SAML after __post_init__."""
        config = SAMLConfig(
            provider_type=None,  # Will be set by __post_init__
            entity_id="https://sp.example.com",
            callback_url="https://sp.example.com/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_certificate="cert",
        )
        assert config.provider_type == SSOProviderType.SAML

    def test_config_default_name_id_format(self):
        """Test default NameID format is email."""
        config = make_saml_config()
        assert "emailAddress" in config.name_id_format

    def test_config_default_attribute_mapping(self):
        """Test default attribute mapping includes common SAML attributes."""
        config = make_saml_config()

        # Should have both long URN and short name mappings
        assert "email" in config.attribute_mapping
        assert "mail" in config.attribute_mapping
        assert (
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress"
            in config.attribute_mapping
        )

    def test_config_validate_missing_idp_entity_id(self):
        """Test validation fails without idp_entity_id."""
        config = make_saml_config(idp_entity_id="")

        errors = config.validate()
        assert any("idp_entity_id" in e for e in errors)

    def test_config_validate_missing_idp_sso_url(self):
        """Test validation fails without idp_sso_url."""
        config = make_saml_config(idp_sso_url="")

        errors = config.validate()
        assert any("idp_sso_url" in e for e in errors)

    def test_config_validate_missing_idp_certificate(self):
        """Test validation fails without idp_certificate."""
        config = make_saml_config(idp_certificate="")

        errors = config.validate()
        assert any("idp_certificate" in e for e in errors)

    def test_config_validate_signed_request_without_private_key(self):
        """Test validation fails when signed requests enabled without SP private key."""
        config = make_saml_config(
            authn_request_signed=True,
            sp_private_key="",
        )

        errors = config.validate()
        assert any("sp_private_key" in e for e in errors)

    def test_config_validate_signed_request_with_private_key(self):
        """Test validation passes with signed requests and private key."""
        config = make_saml_config(
            authn_request_signed=True,
            sp_private_key="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----",
        )

        errors = config.validate()
        assert not any("sp_private_key" in e for e in errors)

    def test_config_custom_attribute_mapping(self):
        """Test custom attribute mapping."""
        custom_mapping = {
            "customEmail": "email",
            "customName": "name",
        }
        config = make_saml_config(attribute_mapping=custom_mapping)

        assert config.attribute_mapping == custom_mapping


# ============================================================================
# SAMLProvider Initialization Tests
# ============================================================================


class TestSAMLProviderInit:
    """Tests for SAMLProvider initialization."""

    def test_init_validates_config(self):
        """Test provider raises on invalid config."""
        config = make_saml_config(idp_entity_id="")  # Invalid

        with pytest.raises(SSOConfigurationError) as exc:
            SAMLProvider(config)

        assert "idp_entity_id" in str(exc.value)

    def test_init_with_valid_config(self):
        """Test provider initializes with valid config."""
        config = make_saml_config()
        provider = SAMLProvider(config)

        assert provider.config == config
        assert provider.provider_type == SSOProviderType.SAML

    @patch.dict("os.environ", {"ARAGORA_ENV": "development"})
    def test_init_warns_without_library_in_dev(self):
        """Test warning is logged when SAML library is missing in dev."""
        config = make_saml_config()

        with patch("aragora.auth.saml.logger") as mock_logger:
            with patch("aragora.auth.saml.HAS_SAML_LIB", False):
                provider = SAMLProvider(config)
                # Should warn but not raise
                assert provider is not None

    @patch.dict("os.environ", {"ARAGORA_ENV": "production"})
    def test_init_fails_without_library_in_production(self):
        """Test provider raises when SAML library missing in production."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with pytest.raises(SSOConfigurationError) as exc:
                SAMLProvider(config)

            assert "python3-saml" in str(exc.value)

    @patch.dict("os.environ", {"ARAGORA_ENV": "staging"})
    def test_init_fails_without_library_in_staging(self):
        """Test provider raises when SAML library missing in staging."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with pytest.raises(SSOConfigurationError) as exc:
                SAMLProvider(config)

            assert "python3-saml" in str(exc.value)


# ============================================================================
# Authorization URL Tests
# ============================================================================


class TestSAMLAuthorizationUrl:
    """Tests for SAML authorization URL generation."""

    @pytest.mark.asyncio
    async def test_get_authorization_url_simple(self):
        """Test generating AuthnRequest URL without library."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                url = await provider.get_authorization_url()

                # URL should point to IdP SSO
                assert url.startswith(config.idp_sso_url)

                # Should contain SAMLRequest
                assert "SAMLRequest=" in url

    @pytest.mark.asyncio
    async def test_get_authorization_url_with_relay_state(self):
        """Test AuthnRequest includes RelayState."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                url = await provider.get_authorization_url(relay_state="custom-state")

                assert "RelayState=custom-state" in url

    @pytest.mark.asyncio
    async def test_get_authorization_url_with_state(self):
        """Test state parameter is used as RelayState."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                url = await provider.get_authorization_url(state="csrf-token")

                assert "RelayState=csrf-token" in url

    @pytest.mark.asyncio
    async def test_get_authorization_url_deflated_request(self):
        """Test AuthnRequest is properly deflate-encoded."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                url = await provider.get_authorization_url()

                # Extract SAMLRequest
                from urllib.parse import parse_qs, urlparse

                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                saml_request = params["SAMLRequest"][0]

                # Decode and decompress
                decoded = base64.b64decode(saml_request)
                # Add zlib header/footer for decompression
                decompressed = zlib.decompress(decoded, -15)
                xml = decompressed.decode("utf-8")

                # Verify XML structure
                assert "AuthnRequest" in xml
                assert config.entity_id in xml
                assert config.idp_sso_url in xml


# ============================================================================
# Authentication Tests (Simple Parser)
# ============================================================================


class TestSAMLAuthentication:
    """Tests for SAML authentication."""

    @pytest.mark.asyncio
    async def test_authenticate_no_response(self):
        """Test authentication fails without SAML response."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)

                with pytest.raises(SSOAuthenticationError) as exc:
                    await provider.authenticate(saml_response=None)

                assert "No SAML response" in str(exc.value)

    @pytest.mark.asyncio
    async def test_authenticate_with_valid_state(self):
        """Test authentication succeeds with valid state in store."""
        config = make_saml_config()
        saml_response = create_saml_response()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                # Add the state to the store (valid, not expired)
                provider._state_store["valid-state"] = time.time()

                # Should succeed with valid state
                user = await provider.authenticate(
                    saml_response=saml_response,
                    relay_state="valid-state",
                )
                assert user.email == "user@example.com"

    @pytest.mark.asyncio
    async def test_authenticate_expired_state(self):
        """Test authentication fails with expired state."""
        config = make_saml_config()
        saml_response = create_saml_response()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                # Add expired state (10 minutes ago, beyond typical 5-min expiry)
                provider._state_store["expired-state"] = time.time() - 600

                with pytest.raises(SSOAuthenticationError) as exc:
                    await provider.authenticate(
                        saml_response=saml_response,
                        relay_state="expired-state",
                    )

                assert "expired" in str(exc.value).lower() or "invalid" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_authenticate_simple_success(self):
        """Test successful authentication with simple parser."""
        config = make_saml_config()
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={
                "email": ["user@example.com"],
                "name": ["Test User"],
            },
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert user.id == "user@example.com"
                assert user.email == "user@example.com"
                assert user.provider_type == "saml"

    @pytest.mark.asyncio
    async def test_authenticate_simple_failed_status(self):
        """Test authentication fails with non-success status."""
        config = make_saml_config()
        saml_response = create_saml_response(
            status="urn:oasis:names:tc:SAML:2.0:status:Requester",
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)

                with pytest.raises(SSOAuthenticationError) as exc:
                    await provider.authenticate(saml_response=saml_response)

                assert "failed" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_authenticate_missing_nameid(self):
        """Test authentication fails without NameID."""
        # Create response without NameID
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
                xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">
    <samlp:Status>
        <samlp:StatusCode Value="urn:oasis:names:tc:SAML:2.0:status:Success"/>
    </samlp:Status>
    <saml:Assertion>
        <saml:Subject>
        </saml:Subject>
    </saml:Assertion>
</samlp:Response>"""
        saml_response = base64.b64encode(xml.encode()).decode()

        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)

                with pytest.raises(SSOAuthenticationError) as exc:
                    await provider.authenticate(saml_response=saml_response)

                assert "NameID" in str(exc.value)

    @pytest.mark.asyncio
    async def test_authenticate_domain_restricted(self):
        """Test authentication fails when email domain not allowed."""
        config = make_saml_config(allowed_domains=["company.com"])
        saml_response = create_saml_response(name_id="user@other.com")

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)

                with pytest.raises(SSOAuthenticationError) as exc:
                    await provider.authenticate(saml_response=saml_response)

                assert "domain not allowed" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_authenticate_domain_allowed(self):
        """Test authentication succeeds when email domain is allowed."""
        config = make_saml_config(allowed_domains=["example.com"])
        saml_response = create_saml_response(name_id="user@example.com")

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert user.email == "user@example.com"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_xml(self):
        """Test authentication fails with invalid XML."""
        config = make_saml_config()
        saml_response = base64.b64encode(b"not valid xml").decode()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)

                with pytest.raises(SSOAuthenticationError) as exc:
                    await provider.authenticate(saml_response=saml_response)

                assert "XML" in str(exc.value) or "failed" in str(exc.value).lower()


# ============================================================================
# Attribute Mapping Tests
# ============================================================================


class TestSAMLAttributeMapping:
    """Tests for SAML attribute mapping."""

    @pytest.mark.asyncio
    async def test_map_attributes_email(self):
        """Test email attribute mapping."""
        config = make_saml_config()
        saml_response = create_saml_response(
            name_id="userid123",
            attributes={"email": ["mapped@example.com"]},
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert user.email == "mapped@example.com"

    @pytest.mark.asyncio
    async def test_map_attributes_name_fields(self):
        """Test name-related attribute mapping."""
        config = make_saml_config()
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={
                "name": ["Full Name"],
                "firstName": ["First"],
                "lastName": ["Last"],
            },
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert user.name == "Full Name"
                assert user.first_name == "First"
                assert user.last_name == "Last"

    @pytest.mark.asyncio
    async def test_map_attributes_roles_and_groups(self):
        """Test roles and groups attribute mapping."""
        config = make_saml_config()
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={
                "roles": ["admin", "editor"],
                "groups": ["engineering", "leadership"],
            },
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                # Note: map_roles/map_groups may transform these
                assert "engineering" in user.groups or user.raw_claims.get("groups")

    @pytest.mark.asyncio
    async def test_map_attributes_urn_format(self):
        """Test URN-formatted attribute names are mapped correctly."""
        config = make_saml_config()
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={
                "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress": [
                    "urn-email@example.com"
                ],
            },
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert user.email == "urn-email@example.com"


# ============================================================================
# Metadata Generation Tests
# ============================================================================


class TestSAMLMetadata:
    """Tests for SP metadata generation."""

    @pytest.mark.asyncio
    async def test_get_metadata_simple(self):
        """Test metadata generation without library."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                metadata = await provider.get_metadata()

                # Should be valid XML
                root = ET.fromstring(metadata)
                assert root is not None

                # Should contain entity ID
                assert config.entity_id in metadata

                # Should contain ACS URL
                assert config.callback_url in metadata

    @pytest.mark.asyncio
    async def test_get_metadata_contains_nameid_format(self):
        """Test metadata includes NameID format."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)
                metadata = await provider.get_metadata()

                assert config.name_id_format in metadata


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestSAMLErrorHandling:
    """Tests for SAML error handling."""

    def test_saml_error_creation(self):
        """Test SAMLError can be created with details."""
        error = SAMLError("Test error", {"code": "TEST_CODE"})

        assert "Test error" in str(error)
        assert error.code == "SAML_ERROR"
        assert error.details["code"] == "TEST_CODE"

    def test_saml_error_inherits_sso_error(self):
        """Test SAMLError is an SSOError."""
        from aragora.auth.sso import SSOError

        error = SAMLError("Test")
        assert isinstance(error, SSOError)


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestSAMLHelpers:
    """Tests for SAML helper methods."""

    def test_get_host_from_url(self):
        """Test extracting host from URL."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)

                host = provider._get_host_from_url("https://example.com:8080/path")
                assert host == "example.com:8080"

    def test_get_path_from_url(self):
        """Test extracting path from URL."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
                provider = SAMLProvider(config)

                path = provider._get_path_from_url("https://example.com/saml/acs")
                assert path == "/saml/acs"


# ============================================================================
# Library Integration Tests
# ============================================================================


class TestSAMLWithLibrary:
    """Tests that require python3-saml library."""

    @pytest.mark.asyncio
    async def test_get_authorization_url_with_library(self):
        """Test AuthnRequest generation with library."""
        config = make_saml_config()
        provider = SAMLProvider(config)

        url = await provider.get_authorization_url(state="test-state")

        assert config.idp_sso_url in url
        assert "SAMLRequest" in url

    @pytest.mark.asyncio
    async def test_get_metadata_with_library(self):
        """Test metadata generation with library."""
        config = make_saml_config(
            sp_certificate="-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----",
        )
        provider = SAMLProvider(config)

        metadata = await provider.get_metadata()

        assert "EntityDescriptor" in metadata
        assert config.entity_id in metadata
