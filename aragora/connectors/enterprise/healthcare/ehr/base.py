"""
Base EHR Adapter.

Common functionality for EHR vendor adapters including:
- SMART on FHIR authentication
- Vendor detection and routing
- Connection configuration
- Capability discovery
"""

from __future__ import annotations

import asyncio
import base64
import logging
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class EHRVendor(str, Enum):
    """Supported EHR vendors."""

    EPIC = "epic"
    CERNER = "cerner"
    ALLSCRIPTS = "allscripts"
    MEDITECH = "meditech"
    ATHENAHEALTH = "athenahealth"
    NEXTGEN = "nextgen"
    UNKNOWN = "unknown"


class EHRCapability(str, Enum):
    """EHR system capabilities."""

    # Authentication
    SMART_ON_FHIR = "smart_on_fhir"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
    OAUTH2_AUTHORIZATION_CODE = "oauth2_authorization_code"
    BACKEND_SERVICES = "backend_services"
    JWT_BEARER = "jwt_bearer"

    # FHIR Operations
    FHIR_R4 = "fhir_r4"
    FHIR_STU3 = "fhir_stu3"
    FHIR_DSTU2 = "fhir_dstu2"
    BULK_DATA_EXPORT = "bulk_data_export"
    BATCH_OPERATIONS = "batch_operations"

    # Clinical Features
    CDS_HOOKS = "cds_hooks"
    SUBSCRIPTIONS = "subscriptions"
    DOCUMENT_REFERENCES = "document_references"

    # Vendor Extensions
    EPIC_MYCHART = "epic_mychart"
    EPIC_CARE_EVERYWHERE = "epic_care_everywhere"
    CERNER_MILLENNIUM = "cerner_millennium"
    CERNER_POWERCHART = "cerner_powerchart"


@dataclass
class EHRConnectionConfig:
    """Configuration for EHR connection."""

    vendor: EHRVendor
    base_url: str
    organization_id: str

    # OAuth2 / SMART on FHIR
    client_id: str
    client_secret: str | None = None  # Not used for public apps
    redirect_uri: str | None = None  # For authorization code flow

    # Backend Services (JWT Bearer)
    private_key: str | None = None  # RSA/EC private key for JWT signing
    key_id: str | None = None  # JWK key ID

    # Token endpoints (auto-discovered if not provided)
    token_endpoint: str | None = None
    authorize_endpoint: str | None = None
    introspect_endpoint: str | None = None

    # Scopes
    scopes: list[str] = field(
        default_factory=lambda: [
            "patient/*.read",
            "system/*.read",
            "launch/patient",
        ]
    )

    # Timeouts and limits
    timeout_seconds: int = 30
    max_retries: int = 3
    rate_limit_requests_per_second: float = 10.0

    # PHI handling
    enable_phi_redaction: bool = True
    audit_all_access: bool = True


@dataclass
class SMARTConfiguration:
    """SMART on FHIR configuration discovered from well-known endpoint."""

    authorization_endpoint: str
    token_endpoint: str
    introspection_endpoint: str | None = None
    revocation_endpoint: str | None = None
    registration_endpoint: str | None = None
    management_endpoint: str | None = None
    scopes_supported: list[str] = field(default_factory=list)
    response_types_supported: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    code_challenge_methods_supported: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SMARTConfiguration:
        """Parse from SMART discovery document."""
        return cls(
            authorization_endpoint=data.get("authorization_endpoint", ""),
            token_endpoint=data.get("token_endpoint", ""),
            introspection_endpoint=data.get("introspection_endpoint"),
            revocation_endpoint=data.get("revocation_endpoint"),
            registration_endpoint=data.get("registration_endpoint"),
            management_endpoint=data.get("management_endpoint"),
            scopes_supported=data.get("scopes_supported", []),
            response_types_supported=data.get("response_types_supported", []),
            capabilities=data.get("capabilities", []),
            code_challenge_methods_supported=data.get("code_challenge_methods_supported", []),
        )


@dataclass
class TokenResponse:
    """OAuth2 token response."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: str | None = None
    scope: str | None = None
    patient: str | None = None  # SMART launch context
    id_token: str | None = None  # OpenID Connect

    # Internal tracking
    obtained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_expired(self) -> bool:
        """Check if token is expired (with 60-second buffer)."""
        expiry = self.obtained_at + timedelta(seconds=self.expires_in - 60)
        return datetime.now(timezone.utc) >= expiry

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenResponse:
        """Parse from OAuth2 token response."""
        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", 3600),
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
            patient=data.get("patient"),
            id_token=data.get("id_token"),
        )


class EHRAdapter(ABC):
    """
    Base class for EHR vendor adapters.

    Provides common functionality for:
    - SMART on FHIR discovery and authentication
    - Token management and refresh
    - Rate limiting and circuit breaking
    - Audit logging

    Subclasses implement vendor-specific:
    - Custom endpoints and operations
    - Data transformations
    - Extended resource types
    """

    vendor: EHRVendor = EHRVendor.UNKNOWN
    capabilities: set[EHRCapability] = set()

    def __init__(self, config: EHRConnectionConfig):
        self.config = config
        self._http_client: httpx.AsyncClient | None = None
        self._smart_config: SMARTConfiguration | None = None
        self._token: TokenResponse | None = None
        self._token_lock = asyncio.Lock()

        # Statistics
        self._requests_made = 0
        self._errors_count = 0
        self._last_request_at: datetime | None = None

    async def __aenter__(self) -> EHRAdapter:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Establish connection and authenticate."""
        import httpx

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            follow_redirects=True,
        )

        # Discover SMART configuration
        await self._discover_smart_config()

        # Authenticate
        await self._authenticate()

        logger.info(f"Connected to {self.vendor.value} EHR at {self.config.base_url}")

    async def disconnect(self) -> None:
        """Close connection."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._token = None
        logger.info(f"Disconnected from {self.vendor.value} EHR")

    async def _discover_smart_config(self) -> None:
        """Discover SMART on FHIR configuration from well-known endpoint."""
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")

        # Try standard SMART discovery endpoint
        discovery_url = f"{self.config.base_url}/.well-known/smart-configuration"

        try:
            response = await self._http_client.get(discovery_url)
            if response.status_code == 200:
                self._smart_config = SMARTConfiguration.from_dict(response.json())
                logger.debug(f"Discovered SMART configuration from {discovery_url}")
                return
        except (OSError, ValueError, TypeError, KeyError) as e:
            logger.debug(f"SMART discovery failed: {e}")

        # Fallback: try FHIR metadata endpoint
        metadata_url = f"{self.config.base_url}/metadata"
        try:
            response = await self._http_client.get(
                metadata_url,
                headers={"Accept": "application/fhir+json"},
            )
            if response.status_code == 200:
                metadata = response.json()
                security = self._extract_security_from_metadata(metadata)
                if security:
                    self._smart_config = security
                    logger.debug("Extracted SMART config from FHIR metadata")
                    return
        except (OSError, ValueError, TypeError, KeyError) as e:
            logger.debug(f"FHIR metadata fetch failed: {e}")

        # Use configured endpoints as fallback
        if self.config.token_endpoint:
            self._smart_config = SMARTConfiguration(
                authorization_endpoint=self.config.authorize_endpoint or "",
                token_endpoint=self.config.token_endpoint,
            )
        else:
            raise RuntimeError(f"Could not discover SMART configuration for {self.config.base_url}")

    def _extract_security_from_metadata(
        self, metadata: dict[str, Any]
    ) -> SMARTConfiguration | None:
        """Extract OAuth2 endpoints from FHIR CapabilityStatement."""
        try:
            rest = metadata.get("rest", [{}])[0]
            security = rest.get("security", {})

            for ext in security.get("extension", []):
                if (
                    ext.get("url")
                    == "http://fhir-registry.smarthealthit.org/StructureDefinition/oauth-uris"
                ):
                    extensions = {e.get("url"): e.get("valueUri") for e in ext.get("extension", [])}
                    return SMARTConfiguration(
                        authorization_endpoint=extensions.get("authorize", ""),
                        token_endpoint=extensions.get("token", ""),
                    )
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to parse SMART configuration from capability statement: {e}")
        return None

    async def _authenticate(self) -> None:
        """Authenticate with the EHR system."""
        if EHRCapability.BACKEND_SERVICES in self.capabilities and self.config.private_key:
            await self._authenticate_backend_services()
        elif self.config.client_secret:
            await self._authenticate_client_credentials()
        else:
            logger.warning(
                "No authentication credentials provided. "
                "Some operations may require authentication."
            )

    async def _authenticate_client_credentials(self) -> None:
        """Authenticate using OAuth2 client credentials flow."""
        if not self._smart_config or not self._http_client:
            raise RuntimeError("SMART configuration not available")

        token_endpoint = self._smart_config.token_endpoint

        # Build request
        data = {
            "grant_type": "client_credentials",
            "scope": " ".join(self.config.scopes),
        }

        # Add client authentication
        auth = base64.b64encode(
            f"{self.config.client_id}:{self.config.client_secret}".encode()
        ).decode()

        response = await self._http_client.post(
            token_endpoint,
            data=data,
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"Authentication failed: {response.status_code} {response.text}")

        self._token = TokenResponse.from_dict(response.json())
        logger.debug(f"Obtained access token (expires in {self._token.expires_in}s)")

    async def _authenticate_backend_services(self) -> None:
        """Authenticate using SMART Backend Services (JWT Bearer)."""
        import jwt
        import time

        if not self._smart_config or not self._http_client:
            raise RuntimeError("SMART configuration not available")

        if not self.config.private_key:
            raise RuntimeError("Private key required for backend services auth")

        token_endpoint = self._smart_config.token_endpoint

        # Build JWT assertion
        now = int(time.time())
        claims = {
            "iss": self.config.client_id,
            "sub": self.config.client_id,
            "aud": token_endpoint,
            "exp": now + 300,  # 5 minutes
            "jti": secrets.token_hex(16),
        }

        # Sign JWT
        headers = {"alg": "RS384"}
        if self.config.key_id:
            headers["kid"] = self.config.key_id

        client_assertion = jwt.encode(
            claims,
            self.config.private_key,
            algorithm="RS384",
            headers=headers,
        )

        # Request token
        data = {
            "grant_type": "client_credentials",
            "scope": " ".join(self.config.scopes),
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": client_assertion,
        }

        response = await self._http_client.post(
            token_endpoint,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Backend services auth failed: {response.status_code} {response.text}"
            )

        self._token = TokenResponse.from_dict(response.json())
        logger.debug(f"Obtained backend services token (expires in {self._token.expires_in}s)")

    async def _ensure_authenticated(self) -> str:
        """Ensure we have a valid access token, refreshing if needed."""
        async with self._token_lock:
            if self._token is None:
                await self._authenticate()

            if self._token and self._token.is_expired:
                if self._token.refresh_token:
                    await self._refresh_token()
                else:
                    await self._authenticate()

            if not self._token:
                raise RuntimeError("Failed to obtain access token")

            return self._token.access_token

    async def _refresh_token(self) -> None:
        """Refresh the access token using refresh token."""
        if not self._smart_config or not self._http_client or not self._token:
            raise RuntimeError("Cannot refresh: missing configuration or token")

        if not self._token.refresh_token:
            raise RuntimeError("No refresh token available")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._token.refresh_token,
        }

        # Add client authentication if available
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if self.config.client_secret:
            auth = base64.b64encode(
                f"{self.config.client_id}:{self.config.client_secret}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {auth}"
        else:
            data["client_id"] = self.config.client_id

        response = await self._http_client.post(
            self._smart_config.token_endpoint,
            data=data,
            headers=headers,
        )

        if response.status_code != 200:
            # Refresh failed, try full re-authentication
            await self._authenticate()
            return

        self._token = TokenResponse.from_dict(response.json())
        logger.debug(f"Refreshed access token (expires in {self._token.expires_in}s)")

    def _get_headers(self, access_token: str) -> dict[str, str]:
        """Get standard headers for FHIR requests."""
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make authenticated request to EHR."""
        if not self._http_client:
            raise RuntimeError("Not connected")

        access_token = await self._ensure_authenticated()
        headers = self._get_headers(access_token)
        headers.update(kwargs.pop("headers", {}))

        url = f"{self.config.base_url}{path}"

        self._requests_made += 1
        self._last_request_at = datetime.now(timezone.utc)

        try:
            response = await self._http_client.request(
                method,
                url,
                headers=headers,
                **kwargs,
            )

            if response.status_code == 401:
                # Token might be invalid, retry with fresh token
                self._token = None
                access_token = await self._ensure_authenticated()
                headers["Authorization"] = f"Bearer {access_token}"
                response = await self._http_client.request(
                    method,
                    url,
                    headers=headers,
                    **kwargs,
                )

            response.raise_for_status()
            return response.json() if response.content else {}

        except (OSError, ValueError, RuntimeError) as e:
            self._errors_count += 1
            logger.error(f"EHR request failed: {method} {path}: {e}")
            raise

    # Abstract methods for vendor-specific implementations
    @abstractmethod
    async def get_patient(self, patient_id: str) -> dict[str, Any]:
        """Get patient resource by ID."""
        ...

    @abstractmethod
    async def search_patients(
        self,
        family: str | None = None,
        given: str | None = None,
        birthdate: str | None = None,
        identifier: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search for patients."""
        ...

    @abstractmethod
    async def get_patient_records(
        self,
        patient_id: str,
        resource_types: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Get all clinical records for a patient."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            "vendor": self.vendor.value,
            "base_url": self.config.base_url,
            "organization_id": self.config.organization_id,
            "requests_made": self._requests_made,
            "errors_count": self._errors_count,
            "last_request_at": (
                self._last_request_at.isoformat() if self._last_request_at else None
            ),
            "is_authenticated": self._token is not None and not self._token.is_expired,
            "capabilities": [c.value for c in self.capabilities],
        }


def detect_vendor(base_url: str, metadata: dict[str, Any] | None = None) -> EHRVendor:
    """
    Detect EHR vendor from URL patterns or FHIR metadata.

    Args:
        base_url: FHIR server base URL
        metadata: Optional FHIR CapabilityStatement

    Returns:
        Detected EHRVendor
    """
    url_lower = base_url.lower()

    # URL-based detection
    if "epic" in url_lower or "mychart" in url_lower:
        return EHRVendor.EPIC
    if "cerner" in url_lower or "cernercentral" in url_lower or "millennium" in url_lower:
        return EHRVendor.CERNER
    if "allscripts" in url_lower:
        return EHRVendor.ALLSCRIPTS
    if "meditech" in url_lower:
        return EHRVendor.MEDITECH
    if "athenahealth" in url_lower or "athena" in url_lower:
        return EHRVendor.ATHENAHEALTH
    if "nextgen" in url_lower:
        return EHRVendor.NEXTGEN

    # Metadata-based detection
    if metadata:
        software = metadata.get("software", {})
        name = software.get("name", "").lower()

        if "epic" in name:
            return EHRVendor.EPIC
        if "cerner" in name or "millennium" in name:
            return EHRVendor.CERNER
        if "allscripts" in name:
            return EHRVendor.ALLSCRIPTS
        if "meditech" in name:
            return EHRVendor.MEDITECH

        # Check for vendor-specific extensions
        for rest in metadata.get("rest", []):
            for resource in rest.get("resource", []):
                for ext in resource.get("extension", []):
                    ext_url = ext.get("url", "")
                    if "epic" in ext_url.lower():
                        return EHRVendor.EPIC
                    if "cerner" in ext_url.lower():
                        return EHRVendor.CERNER

    return EHRVendor.UNKNOWN


def create_adapter(config: EHRConnectionConfig) -> EHRAdapter:
    """
    Factory function to create appropriate EHR adapter.

    Args:
        config: EHR connection configuration

    Returns:
        Vendor-specific EHR adapter

    Raises:
        ValueError: If vendor is not supported
    """
    from aragora.connectors.enterprise.healthcare.ehr.epic import EpicAdapter
    from aragora.connectors.enterprise.healthcare.ehr.cerner import CernerAdapter

    if config.vendor == EHRVendor.EPIC:
        return EpicAdapter(config)
    elif config.vendor == EHRVendor.CERNER:
        return CernerAdapter(config)
    else:
        raise ValueError(f"Unsupported EHR vendor: {config.vendor}")
