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
    HAS_SAML_LIB,
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


@pytest.fixture(autouse=True)
def enable_unsafe_saml_fallback_without_library(monkeypatch):
    """Keep test behavior deterministic when python3-saml is unavailable."""
    if not HAS_SAML_LIB:
        monkeypatch.setenv("ARAGORA_ALLOW_UNSAFE_SAML", "true")
        monkeypatch.setenv("ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED", "true")


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

    @patch.dict(
        "os.environ",
        {
            "ARAGORA_ENV": "development",
            "ARAGORA_ALLOW_UNSAFE_SAML": "",
            "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "",
        },
    )
    def test_init_fails_without_library_in_dev_by_default(self):
        """Test provider raises when SAML library missing in dev without explicit opt-in."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with pytest.raises(SSOConfigurationError) as exc:
                SAMLProvider(config)

            assert "python3-saml" in str(exc.value)
            assert "ARAGORA_ALLOW_UNSAFE_SAML" in str(exc.value)

    @patch.dict(
        "os.environ",
        {
            "ARAGORA_ENV": "development",
            "ARAGORA_ALLOW_UNSAFE_SAML": "true",
            "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
        },
    )
    def test_init_warns_without_library_in_dev_with_explicit_opt_in(self):
        """Test warning is logged when SAML library is missing in dev with explicit opt-in."""
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                url = await provider.get_authorization_url(relay_state="custom-state")

                assert "RelayState=custom-state" in url

    @pytest.mark.asyncio
    async def test_get_authorization_url_with_state(self):
        """Test state parameter is used as RelayState."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                url = await provider.get_authorization_url(state="csrf-token")

                assert "RelayState=csrf-token" in url

    @pytest.mark.asyncio
    async def test_get_authorization_url_deflated_request(self):
        """Test AuthnRequest is properly deflate-encoded."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert user.email == "user@example.com"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_xml(self):
        """Test authentication fails with invalid XML."""
        config = make_saml_config()
        saml_response = base64.b64encode(b"not valid xml").decode()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)

                host = provider._get_host_from_url("https://example.com:8080/path")
                assert host == "example.com:8080"

    def test_get_path_from_url(self):
        """Test extracting path from URL."""
        config = make_saml_config()

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
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


# ============================================================================
# Session Management Tests
# ============================================================================


class TestSAMLSessionManagement:
    """Tests for SAML session management integration."""

    @pytest.mark.asyncio
    async def test_session_creation_after_authentication(self):
        """Test that sessions can be created from authenticated SAML users."""
        from aragora.auth.sso import SSOSessionManager

        config = make_saml_config()
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={
                "email": ["user@example.com"],
                "name": ["Test User"],
            },
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                # Create session from authenticated user
                session_manager = SSOSessionManager()
                session = await session_manager.create_session(user)

                assert session.user_id == user.id
                assert session.email == user.email
                assert session.session_id is not None

    @pytest.mark.asyncio
    async def test_session_retrieval(self):
        """Test session retrieval by ID."""
        from aragora.auth.sso import SSOSessionManager, SSOUser

        session_manager = SSOSessionManager()
        user = SSOUser(
            id="user123",
            email="user@example.com",
            name="Test User",
            provider_type="saml",
        )

        session = await session_manager.create_session(user)
        retrieved = await session_manager.get_session(session.session_id)

        assert retrieved.user_id == user.id
        assert retrieved.email == user.email

    @pytest.mark.asyncio
    async def test_session_expiry(self):
        """Test that expired sessions are rejected."""
        from aragora.auth.sso import SSOSessionManager, SSOUser

        session_manager = SSOSessionManager(session_duration=1)  # 1 second
        user = SSOUser(
            id="user123",
            email="user@example.com",
            provider_type="saml",
        )

        session = await session_manager.create_session(user)

        # Wait for expiry
        import asyncio

        await asyncio.sleep(1.1)

        with pytest.raises(KeyError) as exc:
            await session_manager.get_session(session.session_id)

        assert "expired" in str(exc.value).lower() or session.session_id in str(exc.value)

    @pytest.mark.asyncio
    async def test_session_logout(self):
        """Test session logout removes the session."""
        from aragora.auth.sso import SSOSessionManager, SSOUser

        session_manager = SSOSessionManager()
        user = SSOUser(
            id="user123",
            email="user@example.com",
            provider_type="saml",
        )

        session = await session_manager.create_session(user)
        await session_manager.logout(session.session_id)

        with pytest.raises(KeyError):
            await session_manager.get_session(session.session_id)

    @pytest.mark.asyncio
    async def test_session_refresh(self):
        """Test session refresh extends expiry."""
        from aragora.auth.sso import SSOSessionManager, SSOUser

        session_manager = SSOSessionManager(session_duration=3600)
        user = SSOUser(
            id="user123",
            email="user@example.com",
            provider_type="saml",
        )

        session = await session_manager.create_session(user)
        original_expiry = session.expires_at

        # Small delay to ensure time difference
        import asyncio

        await asyncio.sleep(0.1)

        refreshed = await session_manager.refresh_session(session.session_id)

        assert refreshed.expires_at > original_expiry


# ============================================================================
# Multiple IdP Support Tests
# ============================================================================


class TestMultipleIdPSupport:
    """Tests for multiple Identity Provider configurations."""

    def test_multiple_provider_instances(self):
        """Test creating multiple SAML providers for different IdPs."""
        config_azure = make_saml_config(
            entity_id="https://aragora.example.com/saml/azure",
            idp_entity_id="https://login.microsoftonline.com/tenant/saml",
            idp_sso_url="https://login.microsoftonline.com/tenant/saml2",
        )

        config_okta = make_saml_config(
            entity_id="https://aragora.example.com/saml/okta",
            idp_entity_id="https://company.okta.com/app/metadata",
            idp_sso_url="https://company.okta.com/app/sso",
        )

        provider_azure = SAMLProvider(config_azure)
        provider_okta = SAMLProvider(config_okta)

        assert provider_azure.config.idp_entity_id != provider_okta.config.idp_entity_id
        assert provider_azure.provider_type == SSOProviderType.SAML
        assert provider_okta.provider_type == SSOProviderType.SAML

    @pytest.mark.asyncio
    async def test_different_idp_authorization_urls(self):
        """Test that different IdPs generate different auth URLs."""
        config_azure = make_saml_config(
            idp_sso_url="https://login.microsoftonline.com/tenant/saml2",
        )
        config_okta = make_saml_config(
            idp_sso_url="https://company.okta.com/app/sso",
        )

        provider_azure = SAMLProvider(config_azure)
        provider_okta = SAMLProvider(config_okta)

        url_azure = await provider_azure.get_authorization_url()
        url_okta = await provider_okta.get_authorization_url()

        assert "microsoftonline.com" in url_azure
        assert "okta.com" in url_okta

    def test_provider_specific_attribute_mapping(self):
        """Test IdP-specific attribute mappings."""
        # Azure AD uses specific claim URIs
        azure_mapping = {
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress": "email",
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name": "name",
            "http://schemas.microsoft.com/ws/2008/06/identity/claims/groups": "groups",
        }

        # Okta uses simpler attribute names
        okta_mapping = {
            "email": "email",
            "displayName": "name",
            "groups": "groups",
        }

        config_azure = make_saml_config(attribute_mapping=azure_mapping)
        config_okta = make_saml_config(attribute_mapping=okta_mapping)

        assert config_azure.attribute_mapping != config_okta.attribute_mapping

    @pytest.mark.asyncio
    async def test_idp_specific_role_mapping(self):
        """Test IdP-specific role mapping."""
        config = make_saml_config(
            role_mapping={
                "Aragora-Admins": "admin",
                "Aragora-Users": "user",
                "Engineering": "developer",
            },
        )
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={"roles": ["Aragora-Admins", "Engineering"]},
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                # Roles should be mapped
                assert "admin" in user.roles or "Aragora-Admins" in user.roles
                assert "developer" in user.roles or "Engineering" in user.roles

    def test_provider_isolation(self):
        """Test that providers maintain isolated state."""
        config1 = make_saml_config(
            entity_id="https://sp1.example.com",
            idp_sso_url="https://idp1.example.com/sso",
        )
        config2 = make_saml_config(
            entity_id="https://sp2.example.com",
            idp_sso_url="https://idp2.example.com/sso",
        )

        provider1 = SAMLProvider(config1)
        provider2 = SAMLProvider(config2)

        # Add state to provider1
        provider1._state_store["state1"] = time.time()

        # Provider2 should not have this state
        assert "state1" not in provider2._state_store


# ============================================================================
# Signature Verification Tests
# ============================================================================


class TestSignatureVerification:
    """Tests for SAML assertion signature verification."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_SAML_LIB, reason="python3-saml not installed")
    async def test_library_validates_signatures(self):
        """Test that library-based authentication validates signatures."""
        config = make_saml_config(
            want_assertions_signed=True,
        )
        provider = SAMLProvider(config)

        # Create an unsigned/invalid response
        saml_response = create_saml_response(name_id="user@example.com")

        # With the library, this should fail validation
        # (the response isn't properly signed)
        with pytest.raises(SSOAuthenticationError):
            await provider._authenticate_with_library(saml_response, None)

    def test_config_signature_requirements(self):
        """Test signature requirement configuration."""
        config_signed = make_saml_config(
            want_assertions_signed=True,
            authn_request_signed=True,
            sp_private_key="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----",
        )

        config_unsigned = make_saml_config(
            want_assertions_signed=False,
            authn_request_signed=False,
        )

        assert config_signed.want_assertions_signed is True
        assert config_signed.authn_request_signed is True
        assert config_unsigned.want_assertions_signed is False
        assert config_unsigned.authn_request_signed is False

    def test_onelogin_settings_include_security(self):
        """Test that OneLogin settings include security configuration."""
        config = make_saml_config(
            want_assertions_signed=True,
            want_assertions_encrypted=True,
            authn_request_signed=True,
            sp_private_key="-----BEGIN PRIVATE KEY-----\nkey\n-----END PRIVATE KEY-----",
        )
        provider = SAMLProvider(config)

        settings = provider._get_onelogin_settings()

        assert settings["security"]["wantAssertionsSigned"] is True
        assert settings["security"]["wantAssertionsEncrypted"] is True
        assert settings["security"]["authnRequestsSigned"] is True


# ============================================================================
# Assertion Expiry Tests
# ============================================================================


class TestAssertionExpiry:
    """Tests for SAML assertion time-based validation."""

    @pytest.mark.asyncio
    async def test_state_expiry_validation(self):
        """Test that expired state is rejected."""
        config = make_saml_config()
        saml_response = create_saml_response(name_id="user@example.com")

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                # Add state that is 11 minutes old (beyond 10 minute window)
                provider._state_store["old-state"] = time.time() - 660

                with pytest.raises(SSOAuthenticationError) as exc:
                    await provider.authenticate(
                        saml_response=saml_response,
                        relay_state="old-state",
                    )

                assert "invalid" in str(exc.value).lower() or "expired" in str(exc.value).lower()

    def test_state_cleanup(self):
        """Test expired state cleanup."""
        config = make_saml_config()
        provider = SAMLProvider(config)

        # Add mix of fresh and expired states
        provider._state_store["fresh"] = time.time()
        provider._state_store["expired1"] = time.time() - 700  # 11+ minutes old
        provider._state_store["expired2"] = time.time() - 800  # 13+ minutes old

        cleaned = provider.cleanup_expired_states()

        assert cleaned == 2
        assert "fresh" in provider._state_store
        assert "expired1" not in provider._state_store
        assert "expired2" not in provider._state_store


# ============================================================================
# Role and Group Mapping Edge Cases
# ============================================================================


class TestRoleGroupMappingEdgeCases:
    """Tests for edge cases in role and group mapping."""

    @pytest.mark.asyncio
    async def test_empty_roles_gets_default(self):
        """Test that users with no roles get default role."""
        config = make_saml_config(
            default_role="viewer",
        )
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={},  # No roles
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert "viewer" in user.roles

    @pytest.mark.asyncio
    async def test_role_deduplication(self):
        """Test that duplicate roles are deduplicated."""
        config = make_saml_config(
            role_mapping={
                "AdminGroup1": "admin",
                "AdminGroup2": "admin",  # Maps to same role
            },
        )
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={"roles": ["AdminGroup1", "AdminGroup2"]},
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                # Should have "admin" only once
                admin_count = user.roles.count("admin")
                assert admin_count <= 1

    @pytest.mark.asyncio
    async def test_unmapped_roles_passed_through(self):
        """Test that unmapped roles are passed through unchanged."""
        config = make_saml_config(
            role_mapping={
                "KnownRole": "mapped_role",
            },
        )
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={"roles": ["KnownRole", "UnknownRole"]},
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                # KnownRole should be mapped, UnknownRole passed through
                assert "mapped_role" in user.roles or "KnownRole" in user.roles
                assert "UnknownRole" in user.roles

    @pytest.mark.asyncio
    async def test_group_mapping(self):
        """Test group mapping works similarly to roles."""
        config = make_saml_config(
            group_mapping={
                "IdP-Engineering": "engineering",
                "IdP-Marketing": "marketing",
            },
        )
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={"groups": ["IdP-Engineering", "IdP-Sales"]},
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                # engineering should be mapped, IdP-Sales passed through
                assert "engineering" in user.groups or "IdP-Engineering" in user.groups


# ============================================================================
# Advanced Provider Configuration Tests
# ============================================================================


class TestAdvancedConfiguration:
    """Tests for advanced SAML provider configuration."""

    def test_sp_certificate_configuration(self):
        """Test SP certificate configuration for signed requests."""
        config = make_saml_config(
            sp_certificate="-----BEGIN CERTIFICATE-----\nSP_CERT\n-----END CERTIFICATE-----",
            sp_private_key="-----BEGIN PRIVATE KEY-----\nSP_KEY\n-----END PRIVATE KEY-----",
            authn_request_signed=True,
        )

        errors = config.validate()
        assert not any("sp_private_key" in e for e in errors)

    def test_single_logout_configuration(self):
        """Test Single Logout URL configuration."""
        config = make_saml_config(
            idp_slo_url="https://idp.example.com/slo",
            logout_url="https://aragora.example.com/saml/slo",
        )

        provider = SAMLProvider(config)
        settings = provider._get_onelogin_settings()

        assert settings["idp"]["singleLogoutService"]["url"] == "https://idp.example.com/slo"

    def test_custom_name_id_format(self):
        """Test custom NameID format configuration."""
        config = make_saml_config(
            name_id_format="urn:oasis:names:tc:SAML:2.0:nameid-format:persistent",
        )

        provider = SAMLProvider(config)
        settings = provider._get_onelogin_settings()

        assert (
            settings["sp"]["NameIDFormat"] == "urn:oasis:names:tc:SAML:2.0:nameid-format:persistent"
        )

    def test_encrypted_assertions_configuration(self):
        """Test encrypted assertions configuration."""
        config = make_saml_config(
            want_assertions_encrypted=True,
            sp_certificate="-----BEGIN CERTIFICATE-----\nCERT\n-----END CERTIFICATE-----",
            sp_private_key="-----BEGIN PRIVATE KEY-----\nKEY\n-----END PRIVATE KEY-----",
        )

        provider = SAMLProvider(config)
        settings = provider._get_onelogin_settings()

        assert settings["security"]["wantAssertionsEncrypted"] is True

    @pytest.mark.asyncio
    async def test_custom_acs_url_in_auth_request(self):
        """Test custom ACS URL overrides callback_url."""
        config = make_saml_config(
            callback_url="https://default.example.com/acs",
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                url = await provider.get_authorization_url(
                    redirect_uri="https://custom.example.com/acs"
                )

                # The SAMLRequest should be encoded, but we can check via decoding
                from urllib.parse import parse_qs, urlparse

                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                saml_request = params["SAMLRequest"][0]

                # Decode and decompress
                decoded = base64.b64decode(saml_request)
                decompressed = zlib.decompress(decoded, -15)
                xml = decompressed.decode("utf-8")

                assert "https://custom.example.com/acs" in xml


# ============================================================================
# User Data Extraction Tests
# ============================================================================


class TestUserDataExtraction:
    """Tests for extracting user data from SAML assertions."""

    @pytest.mark.asyncio
    async def test_user_raw_claims_preserved(self):
        """Test that raw SAML claims are preserved in user object."""
        config = make_saml_config()
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={
                "email": ["user@example.com"],
                "customAttr": ["value1", "value2"],
            },
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert "customAttr" in user.raw_claims
                assert user.raw_claims["customAttr"] == ["value1", "value2"]

    @pytest.mark.asyncio
    async def test_user_provider_info(self):
        """Test that provider info is correctly set."""
        config = make_saml_config(
            entity_id="https://aragora.example.com/saml",
        )
        saml_response = create_saml_response(name_id="user@example.com")

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert user.provider_type == "saml"
                assert user.provider_id == config.entity_id

    @pytest.mark.asyncio
    async def test_user_full_name_property(self):
        """Test SSOUser full_name property."""
        config = make_saml_config()
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={
                "firstName": ["John"],
                "lastName": ["Doe"],
            },
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert user.full_name == "John Doe"

    @pytest.mark.asyncio
    async def test_user_is_admin_property(self):
        """Test SSOUser is_admin property."""
        config = make_saml_config()
        saml_response = create_saml_response(
            name_id="admin@example.com",
            attributes={
                "roles": ["admin", "user"],
            },
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                assert user.is_admin is True

    @pytest.mark.asyncio
    async def test_user_to_dict(self):
        """Test SSOUser to_dict serialization."""
        config = make_saml_config()
        saml_response = create_saml_response(
            name_id="user@example.com",
            attributes={
                "email": ["user@example.com"],
                "name": ["Test User"],
            },
        )

        with patch("aragora.auth.saml.HAS_SAML_LIB", False):
            with patch.dict(
                "os.environ",
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_UNSAFE_SAML": "true",
                    "ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED": "true",
                },
            ):
                provider = SAMLProvider(config)
                user = await provider.authenticate(saml_response=saml_response)

                user_dict = user.to_dict()

                assert user_dict["id"] == user.id
                assert user_dict["email"] == user.email
                assert user_dict["provider_type"] == "saml"
                assert "authenticated_at" in user_dict
