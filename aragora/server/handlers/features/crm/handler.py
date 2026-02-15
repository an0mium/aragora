# mypy: ignore-errors
"""
CRM Handler - Main Handler Class with Request Routing.

This module provides the main CRMHandler class that:
- Routes requests to appropriate operations
- Manages platform connections
- Provides utility methods for connectors and normalization
- Integrates with RBAC for permission checks

Stability: STABLE
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils import parse_json_body
from aragora.server.handlers.utils.rate_limit import rate_limit

from .circuit_breaker import get_crm_circuit_breaker
from .contacts import ContactOperationsMixin
from .companies import CompanyOperationsMixin
from .deals import DealOperationsMixin
from .pipeline import PipelineOperationsMixin
from .models import SUPPORTED_PLATFORMS
from .validation import validate_platform_id, MAX_CREDENTIAL_VALUE_LENGTH

logger = logging.getLogger(__name__)

# Platform credentials storage (module-level for sharing across mixins)
_platform_credentials: dict[str, dict[str, Any]] = {}
_platform_connectors: dict[str, Any] = {}


class CRMHandler(
    ContactOperationsMixin,
    CompanyOperationsMixin,
    DealOperationsMixin,
    PipelineOperationsMixin,
    SecureHandler,
):
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
            return self._error_response(401, "Authentication required")
        except ForbiddenError as e:
            return self._error_response(403, str(e))

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
            return self._error_response(
                503,
                "CRM service temporarily unavailable (circuit breaker open)",
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

    # ==========================================================================
    # Platform Management
    # ==========================================================================

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
            return self._error_response(400, "Invalid request body")

        platform = body.get("platform")
        # Validate platform ID format
        valid, err = validate_platform_id(platform) if platform else (False, "Platform is required")
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
        valid, err = validate_platform_id(platform)
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

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

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

    def _json_response(self, status: int, data: Any) -> dict[str, Any]:
        """Create a JSON response."""
        return {
            "status_code": status,
            "headers": {"Content-Type": "application/json"},
            "body": data,
        }

    def _error_response(self, status: int, message: str) -> dict[str, Any]:
        """Create an error response."""
        return self._json_response(status, {"error": message})


__all__ = [
    "CRMHandler",
    "_platform_credentials",
    "_platform_connectors",
]
