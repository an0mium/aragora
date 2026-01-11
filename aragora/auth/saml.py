"""
SAML 2.0 Authentication Provider for Aragora.

Implements SAML 2.0 Service Provider (SP) functionality for enterprise SSO.
Supports common IdPs: Azure AD, Okta, OneLogin, PingFederate, etc.

Requirements:
    pip install python3-saml  # Or: pip install pysaml2

Usage:
    from aragora.auth.saml import SAMLProvider, SAMLConfig

    config = SAMLConfig(
        entity_id="https://aragora.example.com/saml/metadata",
        callback_url="https://aragora.example.com/saml/acs",
        idp_entity_id="https://idp.example.com/metadata",
        idp_sso_url="https://idp.example.com/sso",
        idp_certificate="-----BEGIN CERTIFICATE-----...",
    )

    provider = SAMLProvider(config)
    auth_url = await provider.get_authorization_url()
    user = await provider.authenticate(saml_response="...")
"""

from __future__ import annotations

import base64
import logging
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, quote
import xml.etree.ElementTree as ET

from .sso import (
    SSOProvider,
    SSOProviderType,
    SSOConfig,
    SSOUser,
    SSOError,
    SSOAuthenticationError,
    SSOConfigurationError,
)

logger = logging.getLogger(__name__)

# Optional: python3-saml for full SAML support
try:
    from onelogin.saml2.auth import OneLogin_Saml2_Auth
    from onelogin.saml2.utils import OneLogin_Saml2_Utils
    HAS_SAML_LIB = True
except ImportError:
    HAS_SAML_LIB = False
    OneLogin_Saml2_Auth = None
    OneLogin_Saml2_Utils = None


class SAMLError(SSOError):
    """SAML-specific error."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "SAML_ERROR", details)


@dataclass
class SAMLConfig(SSOConfig):
    """
    SAML 2.0 SP configuration.

    Extends base SSOConfig with SAML-specific settings.
    """

    # IdP metadata
    idp_entity_id: str = ""
    idp_sso_url: str = ""  # Single Sign-On URL (HTTP-Redirect or HTTP-POST)
    idp_slo_url: str = ""  # Single Logout URL (optional)
    idp_certificate: str = ""  # IdP's X.509 certificate (PEM format)

    # SP certificates (for signed requests/encrypted assertions)
    sp_private_key: str = ""  # SP's private key (PEM format)
    sp_certificate: str = ""  # SP's X.509 certificate (PEM format)

    # SAML settings
    name_id_format: str = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
    authn_request_signed: bool = False
    want_assertions_signed: bool = True
    want_assertions_encrypted: bool = False

    # Attribute mapping (SAML attribute -> user field)
    attribute_mapping: Dict[str, str] = field(default_factory=lambda: {
        # Common SAML attributes
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress": "email",
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name": "name",
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname": "first_name",
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname": "last_name",
        "http://schemas.microsoft.com/ws/2008/06/identity/claims/groups": "groups",
        "http://schemas.microsoft.com/ws/2008/06/identity/claims/role": "roles",
        # Short names (fallback)
        "email": "email",
        "mail": "email",
        "name": "name",
        "displayName": "display_name",
        "firstName": "first_name",
        "givenName": "first_name",
        "lastName": "last_name",
        "surname": "last_name",
        "groups": "groups",
        "roles": "roles",
        "memberOf": "groups",
    })

    def __post_init__(self):
        if not self.provider_type:
            self.provider_type = SSOProviderType.SAML

    def validate(self) -> List[str]:
        """Validate SAML configuration."""
        errors = super().validate()

        if not self.idp_entity_id:
            errors.append("idp_entity_id is required")

        if not self.idp_sso_url:
            errors.append("idp_sso_url is required")

        if not self.idp_certificate:
            errors.append("idp_certificate is required for signature verification")

        if self.authn_request_signed and not self.sp_private_key:
            errors.append("sp_private_key required when authn_request_signed is True")

        return errors


class SAMLProvider(SSOProvider):
    """
    SAML 2.0 Service Provider implementation.

    Supports HTTP-Redirect binding for AuthnRequest and
    HTTP-POST binding for Response (ACS).
    """

    def __init__(self, config: SAMLConfig):
        super().__init__(config)
        self.config: SAMLConfig = config

        # Validate config
        errors = config.validate()
        if errors:
            raise SSOConfigurationError(
                f"Invalid SAML configuration: {', '.join(errors)}",
                {"errors": errors}
            )

    @property
    def provider_type(self) -> SSOProviderType:
        return SSOProviderType.SAML

    async def get_authorization_url(
        self,
        state: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        relay_state: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate SAML AuthnRequest URL.

        Uses HTTP-Redirect binding with deflate encoding.

        Args:
            state: CSRF state (stored as RelayState)
            redirect_uri: Override ACS URL
            relay_state: SAML RelayState parameter

        Returns:
            IdP SSO URL with encoded AuthnRequest
        """
        if HAS_SAML_LIB:
            return await self._get_url_with_library(state, redirect_uri, relay_state)
        else:
            return await self._get_url_simple(state, redirect_uri, relay_state)

    async def _get_url_simple(
        self,
        state: Optional[str],
        redirect_uri: Optional[str],
        relay_state: Optional[str],
    ) -> str:
        """Generate AuthnRequest without external library."""
        # Generate request ID
        import secrets
        request_id = f"_aragora_{secrets.token_hex(16)}"

        # Build AuthnRequest XML
        issue_instant = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        acs_url = redirect_uri or self.config.callback_url

        authn_request = f"""<?xml version="1.0" encoding="UTF-8"?>
<samlp:AuthnRequest
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="{request_id}"
    Version="2.0"
    IssueInstant="{issue_instant}"
    Destination="{self.config.idp_sso_url}"
    AssertionConsumerServiceURL="{acs_url}"
    ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">
    <saml:Issuer>{self.config.entity_id}</saml:Issuer>
    <samlp:NameIDPolicy
        Format="{self.config.name_id_format}"
        AllowCreate="true"/>
</samlp:AuthnRequest>"""

        # Deflate and base64 encode
        compressed = zlib.compress(authn_request.encode("utf-8"))[2:-4]  # Remove zlib header/footer
        encoded = base64.b64encode(compressed).decode("ascii")

        # Build URL
        params = {"SAMLRequest": encoded}

        # RelayState (use state if provided, or relay_state)
        rs = relay_state or state
        if rs:
            params["RelayState"] = rs
            # Store state for validation
            if state:
                self._state_store[state] = time.time()

        url = f"{self.config.idp_sso_url}?{urlencode(params, quote_via=quote)}"
        return url

    async def _get_url_with_library(
        self,
        state: Optional[str],
        redirect_uri: Optional[str],
        relay_state: Optional[str],
    ) -> str:
        """Generate AuthnRequest using python3-saml library."""
        settings = self._get_onelogin_settings(redirect_uri)

        # Create mock request object
        request_data = {
            "https": "on",
            "http_host": self._get_host_from_url(self.config.callback_url),
            "script_name": self._get_path_from_url(self.config.callback_url),
        }

        auth = OneLogin_Saml2_Auth(request_data, settings)
        rs = relay_state or state or ""

        if state:
            self._state_store[state] = time.time()

        return auth.login(return_to=rs)

    async def authenticate(
        self,
        code: Optional[str] = None,
        saml_response: Optional[str] = None,
        relay_state: Optional[str] = None,
        **kwargs,
    ) -> SSOUser:
        """
        Authenticate user from SAML Response.

        Args:
            saml_response: Base64-encoded SAML Response from IdP
            relay_state: RelayState parameter (for state validation)

        Returns:
            Authenticated user

        Raises:
            SSOAuthenticationError: If authentication fails
        """
        if not saml_response:
            raise SSOAuthenticationError("No SAML response provided")

        # Validate RelayState if we have state tracking
        if relay_state and relay_state in self._state_store:
            if not self.validate_state(relay_state):
                raise SSOAuthenticationError(
                    "Invalid or expired state parameter",
                    {"code": "INVALID_STATE"}
                )

        if HAS_SAML_LIB:
            return await self._authenticate_with_library(saml_response, relay_state)
        else:
            return await self._authenticate_simple(saml_response, relay_state)

    async def _authenticate_simple(
        self,
        saml_response: str,
        relay_state: Optional[str],
    ) -> SSOUser:
        """
        Parse SAML response without external library.

        WARNING: This is a simplified parser for development/testing.
        For production, use python3-saml for proper signature validation.
        """
        try:
            # Decode response
            decoded = base64.b64decode(saml_response)
            xml_str = decoded.decode("utf-8")

            # Parse XML
            root = ET.fromstring(xml_str)

            # Define namespaces
            namespaces = {
                "samlp": "urn:oasis:names:tc:SAML:2.0:protocol",
                "saml": "urn:oasis:names:tc:SAML:2.0:assertion",
            }

            # Check status
            status = root.find(".//samlp:StatusCode", namespaces)
            if status is not None:
                status_value = status.get("Value", "")
                if "Success" not in status_value:
                    raise SSOAuthenticationError(
                        f"SAML authentication failed: {status_value}",
                        {"status": status_value}
                    )

            # Extract NameID
            name_id = root.find(".//saml:NameID", namespaces)
            if name_id is None or not name_id.text:
                raise SSOAuthenticationError("No NameID in SAML response")

            user_id = name_id.text

            # Extract attributes
            attributes: Dict[str, List[str]] = {}
            attr_statements = root.findall(".//saml:AttributeStatement/saml:Attribute", namespaces)

            for attr in attr_statements:
                attr_name = attr.get("Name", "")
                values = []
                for val in attr.findall("saml:AttributeValue", namespaces):
                    if val.text:
                        values.append(val.text)
                if attr_name and values:
                    attributes[attr_name] = values

            # Map attributes to user fields
            user_data = self._map_attributes(attributes)

            # Create user
            user = SSOUser(
                id=user_id,
                email=user_data.get("email", user_id),
                name=user_data.get("name", ""),
                first_name=user_data.get("first_name", ""),
                last_name=user_data.get("last_name", ""),
                display_name=user_data.get("display_name", ""),
                roles=self.map_roles(user_data.get("roles", [])),
                groups=self.map_groups(user_data.get("groups", [])),
                provider_type="saml",
                provider_id=self.config.entity_id,
                raw_claims=attributes,
            )

            # Check domain restriction
            if not self.is_domain_allowed(user.email):
                raise SSOAuthenticationError(
                    f"Email domain not allowed: {user.email.split('@')[-1]}",
                    {"code": "DOMAIN_NOT_ALLOWED"}
                )

            logger.info(f"SAML authentication successful for {user.email}")
            return user

        except ET.ParseError as e:
            raise SSOAuthenticationError(f"Invalid SAML response XML: {e}")
        except SSOAuthenticationError:
            raise
        except Exception as e:
            logger.error(f"SAML authentication error: {e}")
            raise SSOAuthenticationError(f"SAML authentication failed: {e}")

    async def _authenticate_with_library(
        self,
        saml_response: str,
        relay_state: Optional[str],
    ) -> SSOUser:
        """Authenticate using python3-saml library."""
        settings = self._get_onelogin_settings()

        # Create mock request object
        request_data = {
            "https": "on",
            "http_host": self._get_host_from_url(self.config.callback_url),
            "script_name": self._get_path_from_url(self.config.callback_url),
            "post_data": {
                "SAMLResponse": saml_response,
                "RelayState": relay_state or "",
            },
        }

        auth = OneLogin_Saml2_Auth(request_data, settings)
        auth.process_response()

        errors = auth.get_errors()
        if errors:
            raise SSOAuthenticationError(
                f"SAML validation failed: {', '.join(errors)}",
                {"errors": errors, "reason": auth.get_last_error_reason()}
            )

        if not auth.is_authenticated():
            raise SSOAuthenticationError("User not authenticated")

        # Get user data
        name_id = auth.get_nameid()
        attributes = auth.get_attributes()

        # Map attributes
        user_data = self._map_attributes(attributes)

        user = SSOUser(
            id=name_id,
            email=user_data.get("email", name_id),
            name=user_data.get("name", ""),
            first_name=user_data.get("first_name", ""),
            last_name=user_data.get("last_name", ""),
            display_name=user_data.get("display_name", ""),
            roles=self.map_roles(user_data.get("roles", [])),
            groups=self.map_groups(user_data.get("groups", [])),
            provider_type="saml",
            provider_id=self.config.entity_id,
            raw_claims=attributes,
        )

        if not self.is_domain_allowed(user.email):
            raise SSOAuthenticationError(
                f"Email domain not allowed: {user.email.split('@')[-1]}",
                {"code": "DOMAIN_NOT_ALLOWED"}
            )

        return user

    def _map_attributes(self, attributes: Dict[str, List[str]]) -> Dict[str, Any]:
        """Map SAML attributes to user fields."""
        result: Dict[str, Any] = {}

        for saml_attr, user_field in self.config.attribute_mapping.items():
            if saml_attr in attributes:
                values = attributes[saml_attr]
                if user_field in ("roles", "groups"):
                    # Multi-valued
                    result[user_field] = values
                else:
                    # Single-valued
                    result[user_field] = values[0] if values else ""

        return result

    def _get_onelogin_settings(self, acs_url: Optional[str] = None) -> Dict[str, Any]:
        """Get settings dict for python3-saml."""
        return {
            "strict": True,
            "debug": False,
            "sp": {
                "entityId": self.config.entity_id,
                "assertionConsumerService": {
                    "url": acs_url or self.config.callback_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                },
                "singleLogoutService": {
                    "url": self.config.logout_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                } if self.config.logout_url else {},
                "NameIDFormat": self.config.name_id_format,
                "x509cert": self.config.sp_certificate,
                "privateKey": self.config.sp_private_key,
            },
            "idp": {
                "entityId": self.config.idp_entity_id,
                "singleSignOnService": {
                    "url": self.config.idp_sso_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "singleLogoutService": {
                    "url": self.config.idp_slo_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                } if self.config.idp_slo_url else {},
                "x509cert": self.config.idp_certificate,
            },
            "security": {
                "authnRequestsSigned": self.config.authn_request_signed,
                "wantAssertionsSigned": self.config.want_assertions_signed,
                "wantAssertionsEncrypted": self.config.want_assertions_encrypted,
            },
        }

    def _get_host_from_url(self, url: str) -> str:
        """Extract host from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc

    def _get_path_from_url(self, url: str) -> str:
        """Extract path from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.path

    async def get_metadata(self) -> str:
        """
        Generate SP metadata XML.

        Returns:
            XML string for SP metadata
        """
        if HAS_SAML_LIB:
            settings = self._get_onelogin_settings()
            from onelogin.saml2.metadata import OneLogin_Saml2_Metadata
            return OneLogin_Saml2_Metadata.builder(settings)

        # Simple metadata without library
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
    entityID="{self.config.entity_id}">
    <md:SPSSODescriptor
        AuthnRequestsSigned="{str(self.config.authn_request_signed).lower()}"
        WantAssertionsSigned="{str(self.config.want_assertions_signed).lower()}"
        protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
        <md:NameIDFormat>{self.config.name_id_format}</md:NameIDFormat>
        <md:AssertionConsumerService
            Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            Location="{self.config.callback_url}"
            index="0"/>
    </md:SPSSODescriptor>
</md:EntityDescriptor>"""


__all__ = [
    "SAMLError",
    "SAMLConfig",
    "SAMLProvider",
    "HAS_SAML_LIB",
]
