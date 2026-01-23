"""
CRM Platform API Handlers.

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
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aragora.server.handlers.base import HandlerResult, error_response, json_response
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.decorators import has_permission

logger = logging.getLogger(__name__)


# Platform credentials storage
_platform_credentials: dict[str, dict[str, Any]] = {}
_platform_connectors: dict[str, Any] = {}


SUPPORTED_PLATFORMS = {
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
    """Handler for CRM platform API endpoints."""

    RESOURCE_TYPE = "crm"

    ROUTES = [
        "/api/v1/crm/platforms",
        "/api/v1/crm/connect",
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

    def _check_permission(self, request: Any, permission: str) -> HandlerResult | None:
        """Check if user has the required permission."""
        user = self.get_current_user(request)
        if user:
            user_role = user.role if hasattr(user, "role") else None
            if not has_permission(user_role, permission):
                return self._error_response(403, f"Permission denied: {permission} required")
        return None

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/crm/")

    async def handle_request(self, request: Any) -> Any:
        """Route request to appropriate handler."""
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
        if path.endswith("/platforms") and method == "GET":
            return await self._list_platforms(request)

        elif path.endswith("/connect") and method == "POST":
            if err := self._check_permission(request, "crm:configure"):
                return err
            return await self._connect_platform(request)

        elif platform and path.endswith(f"/{platform}") and method == "DELETE":
            if err := self._check_permission(request, "crm:configure"):
                return err
            return await self._disconnect_platform(request, platform)

        # Contacts
        elif path.endswith("/contacts") and not platform and method == "GET":
            if err := self._check_permission(request, "crm:read"):
                return err
            return await self._list_all_contacts(request)

        elif platform and "contacts" in path:
            if method == "GET" and not resource_id:
                if err := self._check_permission(request, "crm:read"):
                    return err
                return await self._list_platform_contacts(request, platform)
            elif method == "POST" and not resource_id:
                if err := self._check_permission(request, "crm:write"):
                    return err
                return await self._create_contact(request, platform)
            elif method == "PUT" and resource_id:
                if err := self._check_permission(request, "crm:write"):
                    return err
                return await self._update_contact(request, platform, resource_id)
            elif method == "GET" and resource_id:
                if err := self._check_permission(request, "crm:read"):
                    return err
                return await self._get_contact(request, platform, resource_id)

        # Companies
        elif path.endswith("/companies") and not platform and method == "GET":
            if err := self._check_permission(request, "crm:read"):
                return err
            return await self._list_all_companies(request)

        elif platform and "companies" in path:
            if method == "GET" and not resource_id:
                if err := self._check_permission(request, "crm:read"):
                    return err
                return await self._list_platform_companies(request, platform)
            elif method == "POST" and not resource_id:
                if err := self._check_permission(request, "crm:write"):
                    return err
                return await self._create_company(request, platform)
            elif method == "GET" and resource_id:
                if err := self._check_permission(request, "crm:read"):
                    return err
                return await self._get_company(request, platform, resource_id)

        # Deals
        elif path.endswith("/deals") and not platform and method == "GET":
            if err := self._check_permission(request, "crm:read"):
                return err
            return await self._list_all_deals(request)

        elif platform and "deals" in path:
            if method == "GET" and not resource_id:
                if err := self._check_permission(request, "crm:read"):
                    return err
                return await self._list_platform_deals(request, platform)
            elif method == "POST" and not resource_id:
                if err := self._check_permission(request, "crm:write"):
                    return err
                return await self._create_deal(request, platform)
            elif method == "GET" and resource_id:
                if err := self._check_permission(request, "crm:read"):
                    return err
                return await self._get_deal(request, platform, resource_id)

        # Pipeline
        elif path.endswith("/pipeline") and method == "GET":
            if err := self._check_permission(request, "crm:read"):
                return err
            return await self._get_pipeline(request)

        # Lead sync
        elif path.endswith("/sync-lead") and method == "POST":
            if err := self._check_permission(request, "crm:write"):
                return err
            return await self._sync_lead(request)

        # Enrichment
        elif path.endswith("/enrich") and method == "POST":
            if err := self._check_permission(request, "crm:write"):
                return err
            return await self._enrich_contact(request)

        # Search
        elif path.endswith("/search") and method == "POST":
            if err := self._check_permission(request, "crm:read"):
                return err
            return await self._search_crm(request)

        return self._error_response(404, "Endpoint not found")

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
            return self._error_response(400, f"Invalid JSON body: {e}")

        platform = body.get("platform")
        if not platform:
            return self._error_response(400, "Platform is required")

        if platform not in SUPPORTED_PLATFORMS:
            return self._error_response(400, f"Unsupported platform: {platform}")

        if SUPPORTED_PLATFORMS[platform].get("coming_soon"):
            return self._error_response(400, f"{platform} integration is coming soon")

        credentials = body.get("credentials", {})
        if not credentials:
            return self._error_response(400, "Credentials are required")

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
        limit = int(request.query.get("limit", 100))
        email = request.query.get("email")

        all_contacts: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            if not SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                tasks.append(self._fetch_platform_contacts(platform, limit, email))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching contacts from {platform}: {result}")
                continue
            all_contacts.extend(result)

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

        try:
            if platform == "hubspot":
                if email:
                    contact = await connector.get_contact_by_email(email)
                    if contact:
                        return [self._normalize_hubspot_contact(contact)]
                    return []
                else:
                    contacts = await connector.get_contacts(limit=limit)
                    return [self._normalize_hubspot_contact(c) for c in contacts]

        except Exception as e:
            logger.error(f"Error fetching {platform} contacts: {e}")

        return []

    async def _list_platform_contacts(self, request: Any, platform: str) -> HandlerResult:
        """List contacts from a specific platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        limit = int(request.query.get("limit", 100))
        email = request.query.get("email")

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
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "hubspot":
                contact = await connector.get_contact(contact_id)
                return self._json_response(200, self._normalize_hubspot_contact(contact))

        except Exception as e:
            return self._error_response(404, f"Contact not found: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _create_contact(self, request: Any, platform: str) -> HandlerResult:
        """Create a new contact."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "hubspot":
                properties = {
                    "email": body.get("email"),
                    "firstname": body.get("first_name"),
                    "lastname": body.get("last_name"),
                    "phone": body.get("phone"),
                    "company": body.get("company"),
                    "jobtitle": body.get("job_title"),
                }
                # Remove None values
                properties = {k: v for k, v in properties.items() if v is not None}

                contact = await connector.create_contact(properties)
                return self._json_response(201, self._normalize_hubspot_contact(contact))

        except Exception as e:
            return self._error_response(500, f"Failed to create contact: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _update_contact(
        self,
        request: Any,
        platform: str,
        contact_id: str,
    ) -> HandlerResult:
        """Update an existing contact."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            return self._error_response(400, f"Invalid JSON body: {e}")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

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
                return self._json_response(200, self._normalize_hubspot_contact(contact))

        except Exception as e:
            return self._error_response(500, f"Failed to update contact: {e}")

        return self._error_response(400, "Unsupported platform")

    # Company operations

    async def _list_all_companies(self, request: Any) -> HandlerResult:
        """List companies from all connected platforms."""
        limit = int(request.query.get("limit", 100))

        all_companies: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            if not SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                tasks.append(self._fetch_platform_companies(platform, limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, Exception):
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

        limit = int(request.query.get("limit", 100))
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
            return self._error_response(404, f"Company not found: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _create_company(self, request: Any, platform: str) -> HandlerResult:
        """Create a new company."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
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
            return self._error_response(500, f"Failed to create company: {e}")

        return self._error_response(400, "Unsupported platform")

    # Deal operations

    async def _list_all_deals(self, request: Any) -> HandlerResult:
        """List deals from all connected platforms."""
        limit = int(request.query.get("limit", 100))
        stage = request.query.get("stage")

        all_deals: list[dict[str, Any]] = []

        tasks = []
        for platform in _platform_credentials.keys():
            if not SUPPORTED_PLATFORMS.get(platform, {}).get("coming_soon"):
                tasks.append(self._fetch_platform_deals(platform, limit, stage))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, Exception):
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

        limit = int(request.query.get("limit", 100))
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
            return self._error_response(404, f"Deal not found: {e}")

        return self._error_response(400, "Unsupported platform")

    async def _create_deal(self, request: Any, platform: str) -> HandlerResult:
        """Create a new deal."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
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
                                        "probability": s.metadata.get("probability")
                                        if hasattr(s, "metadata")
                                        else None,
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
            except Exception:
                pass

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
            return self._error_response(500, f"Failed to sync lead: {e}")

    # Enrichment

    async def _enrich_contact(self, request: Any) -> HandlerResult:
        """Enrich contact data using available sources."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
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
            "created_at": contact.created_at.isoformat()
            if hasattr(contact, "created_at") and contact.created_at
            else None,
            "updated_at": contact.updated_at.isoformat()
            if hasattr(contact, "updated_at") and contact.updated_at
            else None,
        }

    def _normalize_hubspot_company(self, company: Any) -> dict[str, Any]:
        """Normalize HubSpot company to unified format."""
        props = company.properties if hasattr(company, "properties") else {}
        return {
            "id": company.id,
            "platform": "hubspot",
            "name": props.get("name"),
            "domain": props.get("domain"),
            "industry": props.get("industry"),
            "employee_count": int(props.get("numberofemployees"))
            if props.get("numberofemployees")
            else None,
            "annual_revenue": float(props.get("annualrevenue"))
            if props.get("annualrevenue")
            else None,
            "owner_id": props.get("hubspot_owner_id"),
            "created_at": company.created_at.isoformat()
            if hasattr(company, "created_at") and company.created_at
            else None,
        }

    def _normalize_hubspot_deal(self, deal: Any) -> dict[str, Any]:
        """Normalize HubSpot deal to unified format."""
        props = deal.properties if hasattr(deal, "properties") else {}
        return {
            "id": deal.id,
            "platform": "hubspot",
            "name": props.get("dealname"),
            "amount": float(props.get("amount")) if props.get("amount") else None,
            "stage": props.get("dealstage"),
            "pipeline": props.get("pipeline"),
            "close_date": props.get("closedate"),
            "owner_id": props.get("hubspot_owner_id"),
            "created_at": deal.created_at.isoformat()
            if hasattr(deal, "created_at") and deal.created_at
            else None,
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
        """Parse JSON body from request."""
        body = await request.json()
        return body if isinstance(body, dict) else {}

    def _json_response(self, status: int, data: Any) -> HandlerResult:
        """Create a JSON response."""
        return json_response(data, status=status)

    def _error_response(self, status: int, message: str) -> HandlerResult:
        """Create an error response."""
        return error_response(message, status=status)


__all__ = ["CRMHandler"]
