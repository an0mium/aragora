"""
CRM Platform API Handlers.

Stability: STABLE

Unified API for Customer Relationship Management platforms:
- HubSpot (contacts, companies, deals, marketing)
- Salesforce - planned
- Pipedrive - planned

Usage:
    GET    /api/v1/crm/platforms                  - List connected platforms
    POST   /api/v1/crm/connect                    - Connect a platform
    DELETE /api/v1/crm/{platform}                 - Disconnect platform

    GET    /api/v1/crm/contacts                   - List contacts (cross-platform)
    GET    /api/v1/crm/{platform}/contacts        - Platform contacts
    POST   /api/v1/crm/{platform}/contacts        - Create contact
    PUT    /api/v1/crm/{platform}/contacts/{id}   - Update contact

    GET    /api/v1/crm/companies                  - List companies
    GET    /api/v1/crm/deals                      - List deals/opportunities
    GET    /api/v1/crm/pipeline                   - Get sales pipeline

    POST   /api/v1/crm/sync-lead                  - Sync lead from external source
    POST   /api/v1/crm/enrich                     - Enrich contact data

Features:
- Circuit breaker pattern for CRM platform API resilience
- Rate limiting (60 requests/minute)
- RBAC permission checks (crm:read, crm:write, crm:configure)
- Comprehensive input validation with safe ID patterns
- Error isolation (platform failures handled gracefully)
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


from aragora.server.handlers.base import HandlerResult, json_response
from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils import parse_json_body
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.handlers.utils.responses import error_response
from aragora.server.validation.query_params import safe_query_int

logger = logging.getLogger(__name__)


# =============================================================================
# Input Validation Constants
# =============================================================================

# Platform ID validation: alphanumeric and underscores only
SAFE_PLATFORM_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,49}$")

# Contact/Company/Deal ID validation: alphanumeric, hyphens, underscores
SAFE_RESOURCE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,127}$")

# Email validation pattern (basic but covers most cases)
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# Max lengths for input validation
MAX_EMAIL_LENGTH = 254
MAX_NAME_LENGTH = 128
MAX_PHONE_LENGTH = 32
MAX_COMPANY_NAME_LENGTH = 256
MAX_JOB_TITLE_LENGTH = 128
MAX_DOMAIN_LENGTH = 253
MAX_DEAL_NAME_LENGTH = 256
MAX_STAGE_LENGTH = 64
MAX_PIPELINE_LENGTH = 64
MAX_CREDENTIAL_VALUE_LENGTH = 1024
MAX_SEARCH_QUERY_LENGTH = 256


def _validate_platform_id(platform: str) -> tuple[bool, str | None]:
    """Validate a platform ID.

    Args:
        platform: Platform identifier to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not platform:
        return False, "Platform is required"
    if len(platform) > 50:
        return False, "Platform name too long (max 50 characters)"
    if not SAFE_PLATFORM_PATTERN.match(platform):
        return False, "Invalid platform format (alphanumeric and underscores only)"
    return True, None


def _validate_resource_id(resource_id: str, resource_type: str = "ID") -> tuple[bool, str | None]:
    """Validate a resource ID (contact, company, deal, etc.).

    Args:
        resource_id: Resource identifier to validate
        resource_type: Type name for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not resource_id:
        return False, f"{resource_type} is required"
    if len(resource_id) > 128:
        return False, f"{resource_type} too long (max 128 characters)"
    if not SAFE_RESOURCE_ID_PATTERN.match(resource_id):
        return False, f"Invalid {resource_type.lower()} format"
    return True, None


def _validate_email(email: str | None, required: bool = False) -> tuple[bool, str | None]:
    """Validate an email address.

    Args:
        email: Email to validate
        required: Whether email is required

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email:
        if required:
            return False, "Email is required"
        return True, None
    if len(email) > MAX_EMAIL_LENGTH:
        return False, f"Email too long (max {MAX_EMAIL_LENGTH} characters)"
    if not EMAIL_PATTERN.match(email):
        return False, "Invalid email format"
    return True, None


def _validate_string_field(
    value: str | None,
    field_name: str,
    max_length: int,
    required: bool = False,
) -> tuple[bool, str | None]:
    """Validate a string field with length constraints.

    Args:
        value: Value to validate
        field_name: Field name for error messages
        max_length: Maximum allowed length
        required: Whether field is required

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not value:
        if required:
            return False, f"{field_name} is required"
        return True, None
    if len(value) > max_length:
        return False, f"{field_name} too long (max {max_length} characters)"
    return True, None


def _validate_amount(amount: Any) -> tuple[bool, str | None, float | None]:
    """Validate a monetary amount.

    Args:
        amount: Amount value to validate

    Returns:
        Tuple of (is_valid, error_message, parsed_value)
    """
    if amount is None:
        return True, None, None
    try:
        amt = float(amount)
        if amt < 0:
            return False, "Amount cannot be negative", None
        if amt > 1_000_000_000_000:  # 1 trillion max
            return False, "Amount too large", None
        return True, None, amt
    except (ValueError, TypeError):
        return False, "Invalid amount format", None


def _validate_probability(probability: Any) -> tuple[bool, str | None, float | None]:
    """Validate a probability value (0-100).

    Args:
        probability: Probability value to validate

    Returns:
        Tuple of (is_valid, error_message, parsed_value)
    """
    if probability is None:
        return True, None, None
    try:
        prob = float(probability)
        if prob < 0 or prob > 100:
            return False, "Probability must be between 0 and 100", None
        return True, None, prob
    except (ValueError, TypeError):
        return False, "Invalid probability format", None


# =============================================================================
# Circuit Breaker for CRM Platform Access
# =============================================================================


class CRMCircuitBreaker:
    """Circuit breaker for CRM platform API access.

    Prevents cascading failures when external platform APIs are unavailable.
    Uses a simple state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
    """

    # State constants
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 2,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            cooldown_seconds: Time to wait before allowing test calls
            half_open_max_calls: Number of test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._check_state()

    def _check_state(self) -> str:
        """Check and potentially transition state (must hold lock)."""
        if self._state == self.OPEN:
            # Check if cooldown has elapsed
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.cooldown_seconds
            ):
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info("CRM circuit breaker transitioning to HALF_OPEN")
        return self._state

    def can_proceed(self) -> bool:
        """Check if a call can proceed.

        Returns:
            True if call is allowed, False if circuit is open
        """
        with self._lock:
            state = self._check_state()
            if state == self.CLOSED:
                return True
            elif state == self.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("CRM circuit breaker closed after successful recovery")
            elif self._state == self.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                self._state = self.OPEN
                self._success_count = 0
                logger.warning("CRM circuit breaker reopened after failure in HALF_OPEN")
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(
                        f"CRM circuit breaker opened after {self._failure_count} failures"
                    )

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            return {
                "state": self._check_state(),
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# Global circuit breaker instance for CRM platform access
_circuit_breaker = CRMCircuitBreaker()
_circuit_breaker_lock = threading.Lock()


def get_crm_circuit_breaker() -> CRMCircuitBreaker:
    """Get the global circuit breaker for CRM platform access."""
    return _circuit_breaker


def reset_crm_circuit_breaker() -> None:
    """Reset the global circuit breaker (for testing)."""
    with _circuit_breaker_lock:
        _circuit_breaker.reset()


# Platform credentials storage
_platform_credentials: dict[str, dict[str, Any]] = {}
_platform_connectors: dict[str, Any] = {}


SUPPORTED_PLATFORMS: dict[str, dict[str, Any]] = {
    "hubspot": {
        "name": "HubSpot",
        "description": "All-in-one CRM with marketing, sales, and service hubs",
        "features": ["contacts", "companies", "deals", "tickets", "marketing"],
    },
    "salesforce": {
        "name": "Salesforce",
        "description": "Enterprise CRM platform",
        "features": ["contacts", "accounts", "opportunities", "leads", "campaigns"],
        "coming_soon": True,
    },
    "pipedrive": {
        "name": "Pipedrive",
        "description": "Sales-focused CRM",
        "features": ["contacts", "organizations", "deals", "pipelines"],
        "coming_soon": True,
    },
}


@dataclass
class UnifiedContact:
    """Unified contact representation across CRM platforms."""

    id: str
    platform: str
    email: str | None
    first_name: str | None
    last_name: str | None
    phone: str | None
    company: str | None
    job_title: str | None
    lifecycle_stage: str | None
    lead_status: str | None
    owner_id: str | None
    created_at: datetime | None
    updated_at: datetime | None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": f"{self.first_name or ''} {self.last_name or ''}".strip() or None,
            "phone": self.phone,
            "company": self.company,
            "job_title": self.job_title,
            "lifecycle_stage": self.lifecycle_stage,
            "lead_status": self.lead_status,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "properties": self.properties,
        }


@dataclass
class UnifiedCompany:
    """Unified company representation."""

    id: str
    platform: str
    name: str
    domain: str | None
    industry: str | None
    employee_count: int | None
    annual_revenue: float | None
    owner_id: str | None
    created_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "name": self.name,
            "domain": self.domain,
            "industry": self.industry,
            "employee_count": self.employee_count,
            "annual_revenue": self.annual_revenue,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class UnifiedDeal:
    """Unified deal/opportunity representation."""

    id: str
    platform: str
    name: str
    amount: float | None
    stage: str
    pipeline: str | None
    close_date: datetime | None
    probability: float | None
    contact_ids: list[str] = field(default_factory=list)
    company_id: str | None = None
    owner_id: str | None = None
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "name": self.name,
            "amount": self.amount,
            "stage": self.stage,
            "pipeline": self.pipeline,
            "close_date": self.close_date.isoformat() if self.close_date else None,
            "probability": self.probability,
            "contact_ids": self.contact_ids,
            "company_id": self.company_id,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CRMHandler(SecureHandler):
    """Handler for CRM platform API endpoints.

    Features:
    - Circuit breaker pattern for CRM platform API resilience
    - Rate limiting (60 requests/minute)
    - RBAC permission checks (crm:read, crm:write, crm:configure)
    - Comprehensive input validation
    """

    def __init__(self, ctx: dict | None = None, server_context: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = server_context or ctx or {}

    RESOURCE_TYPE = "crm"

    ROUTES = [
        "/api/v1/crm/platforms",
        "/api/v1/crm/connect",
        "/api/v1/crm/status",  # Circuit breaker status endpoint
        "/api/v1/crm/{platform}",
        "/api/v1/crm/contacts",
        "/api/v1/crm/{platform}/contacts",
        "/api/v1/crm/{platform}/contacts/{contact_id}",
        "/api/v1/crm/companies",
        "/api/v1/crm/{platform}/companies",
        "/api/v1/crm/{platform}/companies/{company_id}",
        "/api/v1/crm/deals",
        "/api/v1/crm/{platform}/deals",
        "/api/v1/crm/{platform}/deals/{deal_id}",
        "/api/v1/crm/pipeline",
        "/api/v1/crm/sync-lead",
        "/api/v1/crm/enrich",
        "/api/v1/crm/search",
    ]

    async def _check_permission(self, request: Any, permission: str) -> HandlerResult | None:
        """Check if user has the required permission using RBAC system."""
        try:
            auth_context = await self.get_auth_context(request, require_auth=True)
            self.check_permission(auth_context, permission)
            return None
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/crm/")

    def _check_circuit_breaker(self) -> HandlerResult | None:
        """Check if the circuit breaker allows the request to proceed.

        Returns:
            Error response if circuit is open, None if request can proceed
        """
        cb = get_crm_circuit_breaker()
        if not cb.can_proceed():
            logger.warning("CRM circuit breaker is open, rejecting request")
            return error_response(
                "CRM service temporarily unavailable (circuit breaker open)",
                503,
            )
        return None

    @rate_limit(requests_per_minute=60)
    async def handle_request(self, request: Any) -> HandlerResult:
        """Route request to appropriate handler.

        Rate limited to 60 requests per minute.
        """
        method = request.method
        path = str(request.path)

        # Parse path components
        platform = None
        resource_id = None

        parts = path.replace("/api/v1/crm/", "").split("/")
        if parts and parts[0] in SUPPORTED_PLATFORMS:
            platform = parts[0]
            if len(parts) > 2:
                resource_id = parts[2]

        # Route to handlers
        # Status endpoint (no circuit breaker check for status itself)
        if path.endswith("/status") and method == "GET":
            return await self._get_status(request)

        if path.endswith("/platforms") and method == "GET":
            return await self._list_platforms(request)

        elif path.endswith("/connect") and method == "POST":
            if err := await self._check_permission(request, "crm:configure"):
                return err
            return await self._connect_platform(request)

        elif platform and path.endswith(f"/{platform}") and method == "DELETE":
            if err := await self._check_permission(request, "crm:configure"):
                return err
            return await self._disconnect_platform(request, platform)

        # Contacts
        elif path.endswith("/contacts") and not platform and method == "GET":
            if err := await self._check_permission(request, "crm:read"):
                return err
            return await self._list_all_contacts(request)

        elif platform and "contacts" in path:
            if method == "GET" and not resource_id:
                if err := await self._check_permission(request, "crm:read"):
                    return err
                return await self._list_platform_contacts(request, platform)
            elif method == "POST" and not resource_id:
                if err := await self._check_permission(request, "crm:write"):
                    return err
                return await self._create_contact(request, platform)
            elif method == "PUT" and resource_id:
                if err := await self._check_permission(request, "crm:write"):
                    return err
                return await self._update_contact(request, platform, resource_id)
            elif method == "GET" and resource_id:
                if err := await self._check_permission(request, "crm:read"):
                    return err
                return await self._get_contact(request, platform, resource_id)

        # Companies
        elif path.endswith("/companies") and not platform and method == "GET":
            if err := await self._check_permission(request, "crm:read"):
                return err
            return await self._list_all_companies(request)

        elif platform and "companies" in path:
            if method == "GET" and not resource_id:
                if err := await self._check_permission(request, "crm:read"):
                    return err
                return await self._list_platform_companies(request, platform)
            elif method == "POST" and not resource_id:
                if err := await self._check_permission(request, "crm:write"):
                    return err
                return await self._create_company(request, platform)
            elif method == "GET" and resource_id:
                if err := await self._check_permission(request, "crm:read"):
                    return err
                return await self._get_company(request, platform, resource_id)

        # Deals
        elif path.endswith("/deals") and not platform and method == "GET":
            if err := await self._check_permission(request, "crm:read"):
                return err
            return await self._list_all_deals(request)

        elif platform and "deals" in path:
            if method == "GET" and not resource_id:
                if err := await self._check_permission(request, "crm:read"):
                    return err
                return await self._list_platform_deals(request, platform)
            elif method == "POST" and not resource_id:
                if err := await self._check_permission(request, "crm:write"):
                    return err
                return await self._create_deal(request, platform)
            elif method == "GET" and resource_id:
                if err := await self._check_permission(request, "crm:read"):
                    return err
                return await self._get_deal(request, platform, resource_id)

        # Pipeline
        elif path.endswith("/pipeline") and method == "GET":
            if err := await self._check_permission(request, "crm:read"):
                return err
            return await self._get_pipeline(request)

        # Lead sync
        elif path.endswith("/sync-lead") and method == "POST":
            if err := await self._check_permission(request, "crm:write"):
                return err
            return await self._sync_lead(request)

        # Enrichment
        elif path.endswith("/enrich") and method == "POST":
            if err := await self._check_permission(request, "crm:write"):
                return err
            return await self._enrich_contact(request)

        # Search
        elif path.endswith("/search") and method == "POST":
            if err := await self._check_permission(request, "crm:read"):
                return err
            return await self._search_crm(request)

        return self._error_response(404, "Endpoint not found")

    async def _get_status(self, request: Any) -> HandlerResult:
        """Get CRM handler status including circuit breaker state."""
        cb = get_crm_circuit_breaker()
        cb_status = cb.get_status()

        # Get connected platforms count
        connected_platforms = list(_platform_credentials.keys())

        return self._json_response(
            200,
            {
                "status": "healthy" if cb_status["state"] == "closed" else "degraded",
                "circuit_breaker": cb_status,
                "connected_platforms": connected_platforms,
                "connected_count": len(connected_platforms),
                "supported_platforms": list(SUPPORTED_PLATFORMS.keys()),
            },
        )

    async def _list_platforms(self, request: Any) -> HandlerResult:
        """List all supported CRM platforms and connection status."""
        platforms = []
        for platform_id, meta in SUPPORTED_PLATFORMS.items():
            connected = platform_id in _platform_credentials
            platforms.append(
                {
                    "id": platform_id,
                    "name": meta["name"],
                    "description": meta["description"],
                    "features": meta["features"],
                    "connected": connected,
                    "coming_soon": meta.get("coming_soon", False),
                    "connected_at": _platform_credentials.get(platform_id, {}).get("connected_at"),
                }
            )

        return self._json_response(
            200,
            {
                "platforms": platforms,
                "connected_count": sum(1 for p in platforms if p["connected"]),
            },
        )

    async def _connect_platform(self, request: Any) -> HandlerResult:
        """Connect a CRM platform with credentials."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM connect_platform: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        platform = body.get("platform")
        # Validate platform ID format
        valid, err = (
            _validate_platform_id(platform) if platform else (False, "Platform is required")
        )
        if not valid:
            return self._error_response(400, err)

        if platform not in SUPPORTED_PLATFORMS:
            return self._error_response(400, f"Unsupported platform: {platform}")

        if SUPPORTED_PLATFORMS[platform].get("coming_soon"):
            return self._error_response(400, f"{platform} integration is coming soon")

        credentials = body.get("credentials", {})
        if not credentials:
            return self._error_response(400, "Credentials are required")

        # Validate credential values don't exceed max length
        for key, value in credentials.items():
            if isinstance(value, str) and len(value) > MAX_CREDENTIAL_VALUE_LENGTH:
                return self._error_response(
                    400,
                    f"Credential '{key}' too long (max {MAX_CREDENTIAL_VALUE_LENGTH} characters)",
                )

        # Validate required credentials
        required_fields = self._get_required_credentials(platform)
        missing = [f for f in required_fields if f not in credentials]
        if missing:
            return self._error_response(400, f"Missing required credentials: {', '.join(missing)}")

        # Store credentials
        _platform_credentials[platform] = {
            "credentials": credentials,
            "connected_at": datetime.now(timezone.utc).isoformat(),
        }

        # Initialize connector
        try:
            connector = await self._get_connector(platform)
            if connector:
                _platform_connectors[platform] = connector
        except Exception as e:
            logger.warning(f"Could not initialize {platform} connector: {e}")

        logger.info(f"Connected CRM platform: {platform}")

        return self._json_response(
            200,
            {
                "message": f"Successfully connected to {SUPPORTED_PLATFORMS[platform]['name']}",
                "platform": platform,
                "connected_at": _platform_credentials[platform]["connected_at"],
            },
        )

    async def _disconnect_platform(self, request: Any, platform: str) -> HandlerResult:
        """Disconnect a CRM platform."""
        # Validate platform ID format
        valid, err = _validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        if platform in _platform_connectors:
            connector = _platform_connectors[platform]
            if hasattr(connector, "close"):
                await connector.close()
            del _platform_connectors[platform]

        del _platform_credentials[platform]

        logger.info(f"Disconnected CRM platform: {platform}")

        return self._json_response(
            200,
            {
                "message": f"Disconnected from {SUPPORTED_PLATFORMS[platform]['name']}",
                "platform": platform,
            },
        )

    # Contact operations

    async def _list_all_contacts(self, request: Any) -> HandlerResult:
        """List contacts from all connected platforms."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        email = request.query.get("email")

        # Validate email format if provided
        if email:
            valid, err = _validate_email(email)
            if not valid:
                return self._error_response(400, err)

        all_contacts: list[dict[str, Any]] = []
        cb = get_crm_circuit_breaker()

        tasks = []
        for platform in _platform_credentials.keys():
            if not SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                tasks.append(self._fetch_platform_contacts(platform, limit, email))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        has_failure = False
        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error(f"Error fetching contacts from {platform}: {result}")
                has_failure = True
                continue
            all_contacts.extend(result)

        # Record circuit breaker status
        if has_failure:
            cb.record_failure()
        else:
            cb.record_success()

        return self._json_response(
            200,
            {
                "contacts": all_contacts[:limit],
                "total": len(all_contacts),
                "platforms_queried": list(_platform_credentials.keys()),
            },
        )

    async def _fetch_platform_contacts(
        self,
        platform: str,
        limit: int = 100,
        email: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch contacts from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return []

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                if email:
                    contact = await connector.get_contact_by_email(email)
                    cb.record_success()
                    if contact:
                        return [self._normalize_hubspot_contact(contact)]
                    return []
                else:
                    contacts = await connector.get_contacts(limit=limit)
                    cb.record_success()
                    return [self._normalize_hubspot_contact(c) for c in contacts]

        except Exception as e:
            logger.error(f"Error fetching {platform} contacts: {e}")
            cb.record_failure()

        return []

    async def _list_platform_contacts(self, request: Any, platform: str) -> HandlerResult:
        """List contacts from a specific platform."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform ID format
        valid, err = _validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        email = request.query.get("email")

        # Validate email format if provided
        if email:
            valid, err = _validate_email(email)
            if not valid:
                return self._error_response(400, err)

        contacts = await self._fetch_platform_contacts(platform, limit, email)

        return self._json_response(
            200,
            {
                "contacts": contacts,
                "total": len(contacts),
                "platform": platform,
            },
        )

    async def _get_contact(self, request: Any, platform: str, contact_id: str) -> HandlerResult:
        """Get a specific contact."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform ID format
        valid, err = _validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        # Validate contact ID format
        valid, err = _validate_resource_id(contact_id, "Contact ID")
        if not valid:
            return self._error_response(400, err)

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                contact = await connector.get_contact(contact_id)
                cb.record_success()
                return self._json_response(200, self._normalize_hubspot_contact(contact))

        except Exception as e:
            logger.warning("CRM get_contact failed for %s/%s: %s", platform, contact_id, e)
            cb.record_failure()
            return self._error_response(404, f"Contact not found: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _create_contact(self, request: Any, platform: str) -> HandlerResult:
        """Create a new contact."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform ID format
        valid, err = _validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM create_contact: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        # Validate contact fields
        email = body.get("email")
        valid, err = _validate_email(email)
        if not valid:
            return self._error_response(400, err)

        first_name = body.get("first_name")
        valid, err = _validate_string_field(first_name, "First name", MAX_NAME_LENGTH)
        if not valid:
            return self._error_response(400, err)

        last_name = body.get("last_name")
        valid, err = _validate_string_field(last_name, "Last name", MAX_NAME_LENGTH)
        if not valid:
            return self._error_response(400, err)

        phone = body.get("phone")
        valid, err = _validate_string_field(phone, "Phone", MAX_PHONE_LENGTH)
        if not valid:
            return self._error_response(400, err)

        company = body.get("company")
        valid, err = _validate_string_field(company, "Company", MAX_COMPANY_NAME_LENGTH)
        if not valid:
            return self._error_response(400, err)

        job_title = body.get("job_title")
        valid, err = _validate_string_field(job_title, "Job title", MAX_JOB_TITLE_LENGTH)
        if not valid:
            return self._error_response(400, err)

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                properties = {
                    "email": email,
                    "firstname": first_name,
                    "lastname": last_name,
                    "phone": phone,
                    "company": company,
                    "jobtitle": job_title,
                }
                # Remove None values
                properties = {k: v for k, v in properties.items() if v is not None}

                contact = await connector.create_contact(properties)
                cb.record_success()
                return self._json_response(201, self._normalize_hubspot_contact(contact))

        except Exception as e:
            logger.error("CRM create_contact failed for %s: %s", platform, e, exc_info=True)
            cb.record_failure()
            return self._error_response(500, f"Failed to create contact: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _update_contact(
        self,
        request: Any,
        platform: str,
        contact_id: str,
    ) -> HandlerResult:
        """Update an existing contact."""
        # Check circuit breaker
        if err := self._check_circuit_breaker():
            return err

        # Validate platform ID format
        valid, err = _validate_platform_id(platform)
        if not valid:
            return self._error_response(400, err)

        # Validate contact ID format
        valid, err = _validate_resource_id(contact_id, "Contact ID")
        if not valid:
            return self._error_response(400, err)

        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM update_contact: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        # Validate contact fields if provided
        if "email" in body:
            valid, err = _validate_email(body["email"])
            if not valid:
                return self._error_response(400, err)

        if "first_name" in body:
            valid, err = _validate_string_field(body["first_name"], "First name", MAX_NAME_LENGTH)
            if not valid:
                return self._error_response(400, err)

        if "last_name" in body:
            valid, err = _validate_string_field(body["last_name"], "Last name", MAX_NAME_LENGTH)
            if not valid:
                return self._error_response(400, err)

        if "phone" in body:
            valid, err = _validate_string_field(body["phone"], "Phone", MAX_PHONE_LENGTH)
            if not valid:
                return self._error_response(400, err)

        if "company" in body:
            valid, err = _validate_string_field(body["company"], "Company", MAX_COMPANY_NAME_LENGTH)
            if not valid:
                return self._error_response(400, err)

        if "job_title" in body:
            valid, err = _validate_string_field(
                body["job_title"], "Job title", MAX_JOB_TITLE_LENGTH
            )
            if not valid:
                return self._error_response(400, err)

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        cb = get_crm_circuit_breaker()

        try:
            if platform == "hubspot":
                properties = {}
                field_mapping = {
                    "email": "email",
                    "first_name": "firstname",
                    "last_name": "lastname",
                    "phone": "phone",
                    "company": "company",
                    "job_title": "jobtitle",
                    "lifecycle_stage": "lifecyclestage",
                    "lead_status": "hs_lead_status",
                }
                for api_field, hubspot_field in field_mapping.items():
                    if api_field in body:
                        properties[hubspot_field] = body[api_field]

                contact = await connector.update_contact(contact_id, properties)
                cb.record_success()
                return self._json_response(200, self._normalize_hubspot_contact(contact))

        except Exception as e:
            logger.error(
                "CRM update_contact failed for %s/%s: %s", platform, contact_id, e, exc_info=True
            )
            cb.record_failure()
            return self._error_response(500, f"Failed to update contact: {e}")

        return self._error_response(400, "Unsupported platform")

    # Company operations

    async def _list_all_companies(self, request: Any) -> HandlerResult:
        """List companies from all connected platforms."""
        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)

        all_companies: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            if not SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                tasks.append(self._fetch_platform_companies(platform, limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error(f"Error fetching companies from {platform}: {result}")
                continue
            all_companies.extend(result)

        return self._json_response(
            200,
            {
                "companies": all_companies[:limit],
                "total": len(all_companies),
            },
        )

    async def _fetch_platform_companies(
        self,
        platform: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch companies from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return []

        try:
            if platform == "hubspot":
                companies = await connector.get_companies(limit=limit)
                return [self._normalize_hubspot_company(c) for c in companies]

        except Exception as e:
            logger.error(f"Error fetching {platform} companies: {e}")

        return []

    async def _list_platform_companies(self, request: Any, platform: str) -> HandlerResult:
        """List companies from a specific platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        companies = await self._fetch_platform_companies(platform, limit)

        return self._json_response(
            200,
            {
                "companies": companies,
                "total": len(companies),
                "platform": platform,
            },
        )

    async def _get_company(self, request: Any, platform: str, company_id: str) -> HandlerResult:
        """Get a specific company."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "hubspot":
                company = await connector.get_company(company_id)
                return self._json_response(200, self._normalize_hubspot_company(company))

        except Exception as e:
            logger.warning("CRM get_company failed for %s/%s: %s", platform, company_id, e)
            return self._error_response(404, f"Company not found: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _create_company(self, request: Any, platform: str) -> HandlerResult:
        """Create a new company."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM create_company: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "hubspot":
                properties = {
                    "name": body.get("name"),
                    "domain": body.get("domain"),
                    "industry": body.get("industry"),
                    "numberofemployees": body.get("employee_count"),
                    "annualrevenue": body.get("annual_revenue"),
                }
                properties = {k: v for k, v in properties.items() if v is not None}

                company = await connector.create_company(properties)
                return self._json_response(201, self._normalize_hubspot_company(company))

        except Exception as e:
            logger.error("CRM create_company failed for %s: %s", platform, e, exc_info=True)
            return self._error_response(500, f"Failed to create company: {e}")

        return self._error_response(400, "Unsupported platform")

    # Deal operations

    async def _list_all_deals(self, request: Any) -> HandlerResult:
        """List deals from all connected platforms."""
        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        stage = request.query.get("stage")

        all_deals: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            if not SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                tasks.append(self._fetch_platform_deals(platform, limit, stage))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error(f"Error fetching deals from {platform}: {result}")
                continue
            all_deals.extend(result)

        return self._json_response(
            200,
            {
                "deals": all_deals[:limit],
                "total": len(all_deals),
            },
        )

    async def _fetch_platform_deals(
        self,
        platform: str,
        limit: int = 100,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch deals from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return []

        try:
            if platform == "hubspot":
                deals = await connector.get_deals(limit=limit)
                normalized = [self._normalize_hubspot_deal(d) for d in deals]
                if stage:
                    normalized = [d for d in normalized if d.get("stage") == stage]
                return normalized

        except Exception as e:
            logger.error(f"Error fetching {platform} deals: {e}")

        return []

    async def _list_platform_deals(self, request: Any, platform: str) -> HandlerResult:
        """List deals from a specific platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        limit = safe_query_int(request.query, "limit", default=100, max_val=1000)
        stage = request.query.get("stage")
        deals = await self._fetch_platform_deals(platform, limit, stage)

        return self._json_response(
            200,
            {
                "deals": deals,
                "total": len(deals),
                "platform": platform,
            },
        )

    async def _get_deal(self, request: Any, platform: str, deal_id: str) -> HandlerResult:
        """Get a specific deal."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "hubspot":
                deal = await connector.get_deal(deal_id)
                return self._json_response(200, self._normalize_hubspot_deal(deal))

        except Exception as e:
            logger.warning("CRM get_deal failed for %s/%s: %s", platform, deal_id, e)
            return self._error_response(404, f"Deal not found: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _create_deal(self, request: Any, platform: str) -> HandlerResult:
        """Create a new deal."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM create_deal: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "hubspot":
                properties = {
                    "dealname": body.get("name"),
                    "amount": body.get("amount"),
                    "dealstage": body.get("stage"),
                    "pipeline": body.get("pipeline", "default"),
                    "closedate": body.get("close_date"),
                }
                properties = {k: v for k, v in properties.items() if v is not None}

                deal = await connector.create_deal(properties)
                return self._json_response(201, self._normalize_hubspot_deal(deal))

        except Exception as e:
            logger.error("CRM create_deal failed for %s: %s", platform, e, exc_info=True)
            return self._error_response(500, f"Failed to create deal: {e}")

        return self._error_response(400, "Unsupported platform")

    # Pipeline

    async def _get_pipeline(self, request: Any) -> HandlerResult:
        """Get sales pipeline summary."""
        platform = request.query.get("platform")

        if platform and platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        pipelines: list[dict[str, Any]] = []

        platforms_to_query = [platform] if platform else list(_platform_credentials.keys())

        for p in platforms_to_query:
            if SUPPORTED_PLATFORMS.get(p, {}).get("coming_soon"):
                continue

            connector = await self._get_connector(p)
            if not connector:
                continue

            try:
                if p == "hubspot":
                    pipeline_data = await connector.get_pipelines()
                    for pipe in pipeline_data:
                        pipelines.append(
                            {
                                "id": pipe.id,
                                "platform": p,
                                "name": pipe.label,
                                "stages": [
                                    {
                                        "id": s.id,
                                        "name": s.label,
                                        "display_order": s.display_order,
                                        "probability": (
                                            s.metadata.get("probability")
                                            if hasattr(s, "metadata")
                                            else None
                                        ),
                                    }
                                    for s in (pipe.stages if hasattr(pipe, "stages") else [])
                                ],
                            }
                        )

            except Exception as e:
                logger.error(f"Error fetching {p} pipelines: {e}")

        # Get deal summary by stage
        all_deals = await self._list_all_deals(request)
        deals = all_deals.get("body", {}).get("deals", []) if isinstance(all_deals, dict) else []

        stage_summary: dict[str, dict[str, Any]] = {}
        for deal in deals:
            stage = deal.get("stage", "unknown")
            if stage not in stage_summary:
                stage_summary[stage] = {"count": 0, "total_value": 0}
            stage_summary[stage]["count"] += 1
            stage_summary[stage]["total_value"] += deal.get("amount") or 0

        return self._json_response(
            200,
            {
                "pipelines": pipelines,
                "stage_summary": stage_summary,
                "total_deals": len(deals),
                "total_pipeline_value": sum(s["total_value"] for s in stage_summary.values()),
            },
        )

    # Lead sync

    async def _sync_lead(self, request: Any) -> HandlerResult:
        """Sync a lead from an external source (e.g., LinkedIn Ads, form submission)."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM sync_lead: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        target_platform = body.get("platform", "hubspot")
        if target_platform not in _platform_credentials:
            return self._error_response(404, f"Platform {target_platform} is not connected")

        source = body.get("source", "api")
        lead_data = body.get("lead", {})

        if not lead_data.get("email"):
            return self._error_response(400, "Lead email is required")

        connector = await self._get_connector(target_platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {target_platform} connector")

        try:
            # Check if contact exists
            existing = None
            try:
                existing = await connector.get_contact_by_email(lead_data["email"])
            except (ConnectionError, TimeoutError, OSError) as e:
                # Network errors - log and proceed with create (may cause duplicate)
                logger.warning(f"Contact lookup failed, proceeding with create: {e}")

            if existing:
                # Update existing contact
                properties = self._map_lead_to_hubspot(lead_data, source)
                contact = await connector.update_contact(existing.id, properties)
                action = "updated"
            else:
                # Create new contact
                properties = self._map_lead_to_hubspot(lead_data, source)
                contact = await connector.create_contact(properties)
                action = "created"

            return self._json_response(
                200,
                {
                    "action": action,
                    "contact": self._normalize_hubspot_contact(contact),
                    "source": source,
                },
            )

        except Exception as e:
            logger.error("CRM sync_lead failed: %s", e, exc_info=True)
            return self._error_response(500, f"Failed to sync lead: {e}")

    # Enrichment

    async def _enrich_contact(self, request: Any) -> HandlerResult:
        """Enrich contact data using available sources."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM enrich_contact: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        email = body.get("email")
        if not email:
            return self._error_response(400, "Email is required for enrichment")

        # Placeholder for enrichment logic
        # In production, this would integrate with services like Clearbit, ZoomInfo, etc.
        enriched_data = {
            "email": email,
            "enriched": False,
            "message": "Enrichment service integration pending",
            "available_providers": ["clearbit", "zoominfo", "apollo"],
        }

        return self._json_response(200, enriched_data)

    # Search

    async def _search_crm(self, request: Any) -> HandlerResult:
        """Search across CRM data."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("CRM search: invalid JSON body: %s", e)
            return self._error_response(400, f"Invalid JSON body: {e}")

        query = body.get("query", "")
        object_types = body.get("types", ["contacts", "companies", "deals"])
        limit = body.get("limit", 20)

        results: dict[str, list[dict[str, Any]]] = {}

        for platform in _platform_credentials.keys():
            if SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                continue

            connector = await self._get_connector(platform)
            if not connector:
                continue

            try:
                if platform == "hubspot":
                    if "contacts" in object_types:
                        contacts = await connector.search_contacts(query, limit=limit)
                        results.setdefault("contacts", []).extend(
                            [self._normalize_hubspot_contact(c) for c in contacts]
                        )

                    if "companies" in object_types:
                        companies = await connector.search_companies(query, limit=limit)
                        results.setdefault("companies", []).extend(
                            [self._normalize_hubspot_company(c) for c in companies]
                        )

                    if "deals" in object_types:
                        deals = await connector.search_deals(query, limit=limit)
                        results.setdefault("deals", []).extend(
                            [self._normalize_hubspot_deal(d) for d in deals]
                        )

            except Exception as e:
                logger.error(f"Error searching {platform}: {e}")

        return self._json_response(
            200,
            {
                "query": query,
                "results": results,
                "total": sum(len(v) for v in results.values()),
            },
        )

    # Helper methods

    def _get_required_credentials(self, platform: str) -> list[str]:
        """Get required credential fields for a platform."""
        requirements = {
            "hubspot": ["access_token"],
            "salesforce": ["client_id", "client_secret", "refresh_token", "instance_url"],
            "pipedrive": ["api_token"],
        }
        return requirements.get(platform, [])

    async def _get_connector(self, platform: str) -> Any | None:
        """Get or create a connector for a platform."""
        if platform in _platform_connectors:
            return _platform_connectors[platform]

        if platform not in _platform_credentials:
            return None

        creds = _platform_credentials[platform]["credentials"]

        try:
            if platform == "hubspot":
                from aragora.connectors.crm.hubspot import (
                    HubSpotConnector,
                    HubSpotCredentials,
                )

                connector = HubSpotConnector(HubSpotCredentials(**creds))
                _platform_connectors[platform] = connector
                return connector

        except Exception as e:
            logger.error(f"Failed to create {platform} connector: {e}")
            return None

        return None

    def _normalize_hubspot_contact(self, contact: Any) -> dict[str, Any]:
        """Normalize HubSpot contact to unified format."""
        props = contact.properties if hasattr(contact, "properties") else {}
        return {
            "id": contact.id,
            "platform": "hubspot",
            "email": props.get("email"),
            "first_name": props.get("firstname"),
            "last_name": props.get("lastname"),
            "full_name": f"{props.get('firstname', '')} {props.get('lastname', '')}".strip()
            or None,
            "phone": props.get("phone"),
            "company": props.get("company"),
            "job_title": props.get("jobtitle"),
            "lifecycle_stage": props.get("lifecyclestage"),
            "lead_status": props.get("hs_lead_status"),
            "owner_id": props.get("hubspot_owner_id"),
            "created_at": (
                contact.created_at.isoformat()
                if hasattr(contact, "created_at") and contact.created_at
                else None
            ),
            "updated_at": (
                contact.updated_at.isoformat()
                if hasattr(contact, "updated_at") and contact.updated_at
                else None
            ),
        }

    def _normalize_hubspot_company(self, company: Any) -> dict[str, Any]:
        """Normalize HubSpot company to unified format."""
        props = company.properties if hasattr(company, "properties") else {}
        # Safely parse employee_count
        employee_count = None
        if props.get("numberofemployees"):
            try:
                employee_count = int(props.get("numberofemployees"))
            except (ValueError, TypeError):
                employee_count = None
        # Safely parse annual_revenue
        annual_revenue = None
        if props.get("annualrevenue"):
            try:
                annual_revenue = float(props.get("annualrevenue"))
            except (ValueError, TypeError):
                annual_revenue = None
        return {
            "id": company.id,
            "platform": "hubspot",
            "name": props.get("name"),
            "domain": props.get("domain"),
            "industry": props.get("industry"),
            "employee_count": employee_count,
            "annual_revenue": annual_revenue,
            "owner_id": props.get("hubspot_owner_id"),
            "created_at": (
                company.created_at.isoformat()
                if hasattr(company, "created_at") and company.created_at
                else None
            ),
        }

    def _normalize_hubspot_deal(self, deal: Any) -> dict[str, Any]:
        """Normalize HubSpot deal to unified format."""
        props = deal.properties if hasattr(deal, "properties") else {}
        # Safely parse amount
        amount = None
        if props.get("amount"):
            try:
                amount = float(props.get("amount"))
            except (ValueError, TypeError):
                amount = None
        return {
            "id": deal.id,
            "platform": "hubspot",
            "name": props.get("dealname"),
            "amount": amount,
            "stage": props.get("dealstage"),
            "pipeline": props.get("pipeline"),
            "close_date": props.get("closedate"),
            "owner_id": props.get("hubspot_owner_id"),
            "created_at": (
                deal.created_at.isoformat()
                if hasattr(deal, "created_at") and deal.created_at
                else None
            ),
        }

    def _map_lead_to_hubspot(self, lead: dict[str, Any], source: str) -> dict[str, Any]:
        """Map lead data to HubSpot contact properties."""
        return {
            "email": lead.get("email"),
            "firstname": lead.get("first_name"),
            "lastname": lead.get("last_name"),
            "phone": lead.get("phone"),
            "company": lead.get("company"),
            "jobtitle": lead.get("job_title"),
            "lifecyclestage": "lead",
            "hs_lead_status": "NEW",
            "hs_analytics_source": source,
        }

    async def _get_json_body(self, request: Any) -> dict[str, Any]:
        """Parse JSON body from request.

        Wraps parse_json_body and returns just the dict, raising on error.
        """
        body, _err = await parse_json_body(request, context="crm")
        return body if body is not None else {}

    def _json_response(self, status: int, data: Any) -> HandlerResult:
        """Create a JSON response."""
        return json_response(data, status=status)

    def _error_response(self, status: int, message: str) -> HandlerResult:
        """Create an error response."""
        return error_response(message, status=status)


__all__ = ["CRMHandler"]
