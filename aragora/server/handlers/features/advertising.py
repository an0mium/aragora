"""
Advertising Platform API Handlers.

Stability: STABLE (graduated 2026-02)

Unified API for managing advertising campaigns across platforms:
- Google Ads (Search, Display, Shopping, YouTube)
- Meta Ads (Facebook, Instagram)
- LinkedIn Ads (B2B)
- Microsoft Ads (Bing)

Usage:
    GET    /api/v1/advertising/platforms           - List connected platforms
    POST   /api/v1/advertising/connect             - Connect a platform
    DELETE /api/v1/advertising/{platform}          - Disconnect platform

    GET    /api/v1/advertising/campaigns           - List all campaigns (cross-platform)
    GET    /api/v1/advertising/{platform}/campaigns - List platform campaigns
    POST   /api/v1/advertising/{platform}/campaigns - Create campaign
    PUT    /api/v1/advertising/{platform}/campaigns/{id} - Update campaign

    GET    /api/v1/advertising/performance         - Cross-platform performance
    GET    /api/v1/advertising/{platform}/performance - Platform performance

    POST   /api/v1/advertising/analyze             - Multi-agent performance analysis
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from aragora.resilience import CircuitBreaker
from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils import parse_json_body
from aragora.server.handlers.utils.params import get_clamped_int_param
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.handlers.utils.responses import error_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input validation constants
# ---------------------------------------------------------------------------

# Budget constraints
MAX_DAILY_BUDGET = 1_000_000.0  # $1M daily cap
MAX_TOTAL_BUDGET = 100_000_000.0  # $100M lifetime cap
MIN_BUDGET = 0.0  # Non-negative

# Campaign name constraints
MAX_CAMPAIGN_NAME_LENGTH = 256
MIN_CAMPAIGN_NAME_LENGTH = 1
# Allow alphanumeric, spaces, hyphens, underscores, periods, parentheses, ampersands
_CAMPAIGN_NAME_PATTERN = re.compile(r"^[\w\s\-.()/&:,!'+#@]+$", re.UNICODE)

# Allowed campaign types per platform
ALLOWED_CAMPAIGN_TYPES: dict[str, set[str]] = {
    "google_ads": {"SEARCH", "DISPLAY", "SHOPPING", "VIDEO", "APP", "SMART", "PERFORMANCE_MAX"},
    "meta_ads": {
        "OUTCOME_TRAFFIC",
        "OUTCOME_ENGAGEMENT",
        "OUTCOME_LEADS",
        "OUTCOME_SALES",
        "OUTCOME_AWARENESS",
        "OUTCOME_APP_PROMOTION",
    },
    "linkedin_ads": {
        "SPONSORED_UPDATES",
        "SPONSORED_INMAILS",
        "TEXT_ADS",
        "DYNAMIC_ADS",
        "VIDEO_ADS",
    },
    "microsoft_ads": {"Search", "Shopping", "Audience", "DynamicSearchAds"},
}

# Allowed objectives per platform
ALLOWED_OBJECTIVES: dict[str, set[str]] = {
    "google_ads": {"SEARCH", "DISPLAY", "SHOPPING", "VIDEO", "APP", "SMART", "PERFORMANCE_MAX"},
    "meta_ads": {
        "OUTCOME_TRAFFIC",
        "OUTCOME_ENGAGEMENT",
        "OUTCOME_LEADS",
        "OUTCOME_SALES",
        "OUTCOME_AWARENESS",
        "OUTCOME_APP_PROMOTION",
    },
    "linkedin_ads": {
        "WEBSITE_VISITS",
        "ENGAGEMENT",
        "VIDEO_VIEWS",
        "LEAD_GENERATION",
        "BRAND_AWARENESS",
        "JOB_APPLICANTS",
        "WEBSITE_CONVERSIONS",
    },
    "microsoft_ads": {"Search", "Shopping", "Audience", "DynamicSearchAds"},
}

# Allowed campaign statuses
ALLOWED_CAMPAIGN_STATUSES = {"ENABLED", "PAUSED", "REMOVED", "ACTIVE", "ARCHIVED", "DELETED"}

# Allowed analysis types
ALLOWED_ANALYSIS_TYPES = {
    "performance_review",
    "budget_optimization",
    "audience_analysis",
    "creative_review",
}

# Allowed budget recommendation objectives
ALLOWED_BUDGET_OBJECTIVES = {"balanced", "awareness", "conversions"}

# Maximum date range for analysis (2 years)
MAX_ANALYSIS_DAYS = 730


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_budget(
    value: Any, field_name: str, max_limit: float
) -> tuple[float | None, str | None]:
    """Validate a budget value. Returns (parsed_value, error_message)."""
    if value is None:
        return None, None
    try:
        budget = float(value)
    except (TypeError, ValueError):
        return None, f"{field_name} must be a number"
    if budget < MIN_BUDGET:
        return None, f"{field_name} must be non-negative"
    if budget > max_limit:
        return None, f"{field_name} exceeds maximum allowed value of {max_limit:,.2f}"
    return budget, None


def _validate_campaign_name(name: Any) -> tuple[str | None, str | None]:
    """Validate and sanitize a campaign name. Returns (sanitized_name, error_message)."""
    if not name or not isinstance(name, str):
        return None, "Campaign name is required and must be a string"
    name = name.strip()
    if len(name) < MIN_CAMPAIGN_NAME_LENGTH:
        return None, "Campaign name must not be empty"
    if len(name) > MAX_CAMPAIGN_NAME_LENGTH:
        return None, f"Campaign name must not exceed {MAX_CAMPAIGN_NAME_LENGTH} characters"
    if not _CAMPAIGN_NAME_PATTERN.match(name):
        return None, "Campaign name contains invalid characters"
    return name, None


def _validate_date_iso(value: Any, field_name: str) -> tuple[date | None, str | None]:
    """Validate an ISO date string. Returns (parsed_date, error_message)."""
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, f"{field_name} must be an ISO format date string (YYYY-MM-DD)"
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None, f"{field_name} must be a valid ISO format date (YYYY-MM-DD)"
    return parsed, None


def _validate_date_range(
    start_str: Any, end_str: Any
) -> tuple[tuple[date, date] | None, str | None]:
    """Validate a date range pair. Returns ((start, end), error_message)."""
    start, err = _validate_date_iso(start_str, "start_date")
    if err:
        return None, err
    end, err = _validate_date_iso(end_str, "end_date")
    if err:
        return None, err
    if start and end and start > end:
        return None, "start_date must be before or equal to end_date"
    return (start, end) if start and end else None, None


def _validate_against_allowlist(value: Any, allowlist: set[str], field_name: str) -> str | None:
    """Validate a value is in the allowlist. Returns error message or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        return f"{field_name} must be a string"
    if value not in allowlist:
        return f"Invalid {field_name}: '{value}'. Allowed values: {sorted(allowlist)}"
    return None


# Circuit breaker for advertising platform operations
_advertising_circuit_breaker = CircuitBreaker(
    name="advertising_handler",
    failure_threshold=5,
    cooldown_seconds=60,
    half_open_max_calls=3,
)


def get_advertising_circuit_breaker() -> CircuitBreaker:
    """Get the circuit breaker for advertising service."""
    return _advertising_circuit_breaker


# Platform credentials storage (in production, use encrypted store)
_platform_credentials: dict[str, dict[str, Any]] = {}
_platform_connectors: dict[str, Any] = {}


SUPPORTED_PLATFORMS = {
    "google_ads": {
        "name": "Google Ads",
        "description": "Search, Display, Shopping, and YouTube advertising",
        "features": ["campaigns", "ad_groups", "keywords", "performance", "conversions"],
    },
    "meta_ads": {
        "name": "Meta Ads",
        "description": "Facebook and Instagram advertising",
        "features": ["campaigns", "ad_sets", "creatives", "audiences", "insights"],
    },
    "linkedin_ads": {
        "name": "LinkedIn Ads",
        "description": "B2B advertising with professional targeting",
        "features": ["campaigns", "creatives", "lead_gen", "audiences", "analytics"],
    },
    "microsoft_ads": {
        "name": "Microsoft Ads",
        "description": "Bing search advertising",
        "features": ["campaigns", "ad_groups", "keywords", "audiences", "reporting"],
    },
}


@dataclass
class UnifiedCampaign:
    """Unified campaign representation across platforms."""

    id: str
    platform: str
    name: str
    status: str
    objective: str | None
    daily_budget: float | None
    total_budget: float | None
    start_date: date | None
    end_date: date | None
    created_at: datetime | None
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "platform": self.platform,
            "name": self.name,
            "status": self.status,
            "objective": self.objective,
            "daily_budget": self.daily_budget,
            "total_budget": self.total_budget,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metrics": self.metrics,
        }


@dataclass
class UnifiedPerformance:
    """Unified performance metrics across platforms."""

    platform: str
    campaign_id: str | None
    campaign_name: str | None
    date_range: tuple[date, date]
    impressions: int
    clicks: int
    cost: float
    conversions: int
    conversion_value: float
    ctr: float
    cpc: float
    cpm: float
    roas: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform,
            "campaign_id": self.campaign_id,
            "campaign_name": self.campaign_name,
            "date_range": {
                "start": self.date_range[0].isoformat(),
                "end": self.date_range[1].isoformat(),
            },
            "impressions": self.impressions,
            "clicks": self.clicks,
            "cost": round(self.cost, 2),
            "conversions": self.conversions,
            "conversion_value": round(self.conversion_value, 2),
            "ctr": round(self.ctr, 4),
            "cpc": round(self.cpc, 2),
            "cpm": round(self.cpm, 2),
            "roas": round(self.roas, 2),
        }


class AdvertisingHandler(SecureHandler):
    """Handler for advertising platform API endpoints."""

    def __init__(self, ctx: dict | None = None, server_context: dict | None = None):
        """Initialize handler with optional context."""
        if server_context is not None:
            self.ctx = server_context
        else:
            self.ctx = server_context or ctx or {}

    RESOURCE_TYPE = "advertising"

    ROUTES = [
        "/api/v1/advertising/platforms",
        "/api/v1/advertising/connect",
        "/api/v1/advertising/{platform}",
        "/api/v1/advertising/campaigns",
        "/api/v1/advertising/{platform}/campaigns",
        "/api/v1/advertising/{platform}/campaigns/{campaign_id}",
        "/api/v1/advertising/performance",
        "/api/v1/advertising/{platform}/performance",
        "/api/v1/advertising/analyze",
        "/api/v1/advertising/budget-recommendations",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/advertising/")

    async def _check_permission(self, request: Any, permission: str) -> Any:
        """Check if user has the required permission using RBAC system."""
        try:
            auth_context = await self.get_auth_context(request, require_auth=True)
            self.check_permission(auth_context, permission)
            return None
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning("Handler error: %s", e)
            return error_response("Permission denied", 403)

    @rate_limit(requests_per_minute=60)
    async def handle_request(self, request: Any) -> dict[str, Any]:
        """Route request to appropriate handler."""
        method = request.method
        path = str(request.path)

        # Parse path components
        platform = None
        campaign_id = None

        parts = path.replace("/api/v1/advertising/", "").split("/")
        if parts and parts[0] in SUPPORTED_PLATFORMS:
            platform = parts[0]
            if len(parts) > 2 and parts[1] == "campaigns":
                campaign_id = parts[2]

        # Route to handlers
        if path.endswith("/platforms") and method == "GET":
            return await self._list_platforms(request)

        elif path.endswith("/connect") and method == "POST":
            if err := await self._check_permission(request, "advertising:configure"):
                return err
            return await self._connect_platform(request)

        elif platform and path.endswith(f"/{platform}") and method == "DELETE":
            if err := await self._check_permission(request, "advertising:configure"):
                return err
            return await self._disconnect_platform(request, platform)

        elif path.endswith("/campaigns") and not platform and method == "GET":
            if err := await self._check_permission(request, "advertising:read"):
                return err
            return await self._list_all_campaigns(request)

        elif platform and "campaigns" in path:
            if method == "GET" and not campaign_id:
                if err := await self._check_permission(request, "advertising:read"):
                    return err
                return await self._list_platform_campaigns(request, platform)
            elif method == "POST" and not campaign_id:
                if err := await self._check_permission(request, "advertising:write"):
                    return err
                return await self._create_campaign(request, platform)
            elif method == "PUT" and campaign_id:
                if err := await self._check_permission(request, "advertising:write"):
                    return err
                return await self._update_campaign(request, platform, campaign_id)
            elif method == "GET" and campaign_id:
                if err := await self._check_permission(request, "advertising:read"):
                    return err
                return await self._get_campaign(request, platform, campaign_id)

        elif path.endswith("/performance") and not platform and method == "GET":
            if err := await self._check_permission(request, "advertising:read"):
                return err
            return await self._get_cross_platform_performance(request)

        elif platform and path.endswith("/performance") and method == "GET":
            if err := await self._check_permission(request, "advertising:read"):
                return err
            return await self._get_platform_performance(request, platform)

        elif path.endswith("/analyze") and method == "POST":
            if err := await self._check_permission(request, "advertising:analyze"):
                return err
            return await self._analyze_performance(request)

        elif path.endswith("/budget-recommendations") and method == "GET":
            if err := await self._check_permission(request, "advertising:read"):
                return err
            return await self._get_budget_recommendations(request)

        return self._error_response(404, "Endpoint not found")

    async def _list_platforms(self, request: Any) -> dict[str, Any]:
        """List all supported advertising platforms and connection status."""
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

    async def _connect_platform(self, request: Any) -> dict[str, Any]:
        """Connect an advertising platform with credentials."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("Handler error: %s", e)
            return self._error_response(400, "Invalid request body")

        platform = body.get("platform")
        if not platform:
            return self._error_response(400, "Platform is required")

        if platform not in SUPPORTED_PLATFORMS:
            return self._error_response(400, f"Unsupported platform: {platform}")

        credentials = body.get("credentials", {})
        if not credentials:
            return self._error_response(400, "Missing required credentials")

        # Validate required credentials per platform
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

        logger.info(f"Connected advertising platform: {platform}")

        return self._json_response(
            200,
            {
                "message": f"Successfully connected to {SUPPORTED_PLATFORMS[platform]['name']}",
                "platform": platform,
                "connected_at": _platform_credentials[platform]["connected_at"],
            },
        )

    async def _disconnect_platform(self, request: Any, platform: str) -> dict[str, Any]:
        """Disconnect an advertising platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        # Close connector if exists
        if platform in _platform_connectors:
            connector = _platform_connectors[platform]
            if hasattr(connector, "close"):
                await connector.close()
            del _platform_connectors[platform]

        del _platform_credentials[platform]

        logger.info(f"Disconnected advertising platform: {platform}")

        return self._json_response(
            200,
            {
                "message": f"Disconnected from {SUPPORTED_PLATFORMS[platform]['name']}",
                "platform": platform,
            },
        )

    async def _list_all_campaigns(self, request: Any) -> dict[str, Any]:
        """List campaigns from all connected platforms."""
        status_filter = request.query.get("status")
        # Use safe parameter parsing with bounds
        limit = get_clamped_int_param(dict(request.query), "limit", 100, 1, 1000)

        all_campaigns: list[dict[str, Any]] = []

        # Gather campaigns from all connected platforms in parallel
        tasks = []
        for platform in _platform_credentials.keys():
            tasks.append(self._fetch_platform_campaigns(platform, status_filter))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error(f"Error fetching campaigns from {platform}: {result}")
                continue
            all_campaigns.extend(result)

        # Sort by name and apply limit
        all_campaigns.sort(key=lambda c: c.get("name", ""))
        all_campaigns = all_campaigns[:limit]

        return self._json_response(
            200,
            {
                "campaigns": all_campaigns,
                "total": len(all_campaigns),
                "platforms_queried": list(_platform_credentials.keys()),
            },
        )

    async def _fetch_platform_campaigns(
        self,
        platform: str,
        status_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch campaigns from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return []

        try:
            if platform == "google_ads":
                campaigns = await connector.get_campaigns()
                return [self._normalize_google_campaign(c) for c in campaigns]

            elif platform == "meta_ads":
                campaigns = await connector.get_campaigns()
                return [self._normalize_meta_campaign(c) for c in campaigns]

            elif platform == "linkedin_ads":
                campaigns = await connector.get_campaigns()
                return [self._normalize_linkedin_campaign(c) for c in campaigns]

            elif platform == "microsoft_ads":
                campaigns = await connector.get_campaigns()
                return [self._normalize_microsoft_campaign(c) for c in campaigns]

        except Exception as e:
            logger.error(f"Error fetching {platform} campaigns: {e}")

        return []

    async def _list_platform_campaigns(self, request: Any, platform: str) -> dict[str, Any]:
        """List campaigns from a specific platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        status_filter = request.query.get("status")
        campaigns = await self._fetch_platform_campaigns(platform, status_filter)

        return self._json_response(
            200,
            {
                "campaigns": campaigns,
                "total": len(campaigns),
                "platform": platform,
            },
        )

    async def _get_campaign(self, request: Any, platform: str, campaign_id: str) -> dict[str, Any]:
        """Get a specific campaign by ID."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "google_ads":
                campaign = await connector.get_campaign(campaign_id)
                return self._json_response(200, self._normalize_google_campaign(campaign))

            elif platform == "meta_ads":
                campaign = await connector.get_campaign(campaign_id)
                return self._json_response(200, self._normalize_meta_campaign(campaign))

            elif platform == "linkedin_ads":
                campaign = await connector.get_campaign(campaign_id)
                return self._json_response(200, self._normalize_linkedin_campaign(campaign))

            elif platform == "microsoft_ads":
                campaign = await connector.get_campaign(campaign_id)
                return self._json_response(200, self._normalize_microsoft_campaign(campaign))

        except Exception as e:
            logger.warning("Handler error: %s", e)
            return self._error_response(404, "Campaign not found")

        return self._error_response(400, "Unsupported platform")

    async def _create_campaign(self, request: Any, platform: str) -> dict[str, Any]:
        """Create a new campaign on a platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("Handler error: %s", e)
            return self._error_response(400, "Invalid request body")

        # --- Validate campaign name ---
        name, err = _validate_campaign_name(body.get("name"))
        if err:
            return self._error_response(400, err)

        # --- Validate daily budget ---
        daily_budget_raw = body.get("daily_budget")
        daily_budget, err = _validate_budget(daily_budget_raw, "daily_budget", MAX_DAILY_BUDGET)
        if err:
            return self._error_response(400, err)

        # --- Validate total/lifetime budget if provided ---
        total_budget_raw = body.get("total_budget") or body.get("lifetime_budget")
        total_budget, err = _validate_budget(total_budget_raw, "total_budget", MAX_TOTAL_BUDGET)
        if err:
            return self._error_response(400, err)

        # --- Validate campaign type against platform allowlist ---
        campaign_type = body.get("campaign_type")
        if campaign_type is not None:
            err = _validate_against_allowlist(
                campaign_type, ALLOWED_CAMPAIGN_TYPES.get(platform, set()), "campaign_type"
            )
            if err:
                return self._error_response(400, err)

        # --- Validate objective against platform allowlist ---
        objective = body.get("objective")
        if objective is not None:
            err = _validate_against_allowlist(
                objective, ALLOWED_OBJECTIVES.get(platform, set()), "objective"
            )
            if err:
                return self._error_response(400, err)

        # --- Validate date fields ---
        start_date_str = body.get("start_date")
        end_date_str = body.get("end_date")
        if start_date_str or end_date_str:
            _, err = _validate_date_range(start_date_str, end_date_str)
            if err:
                return self._error_response(400, err)

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        try:
            if platform == "google_ads":
                campaign_id = await connector.create_campaign(
                    name=name,
                    budget_micros=int(
                        (daily_budget if daily_budget is not None else 10) * 1_000_000
                    ),
                    campaign_type=campaign_type or "SEARCH",
                )
                return self._json_response(
                    201,
                    {
                        "campaign_id": campaign_id,
                        "platform": platform,
                        "name": name,
                    },
                )

            elif platform == "meta_ads":
                campaign = await connector.create_campaign(
                    name=name,
                    objective=objective or "OUTCOME_TRAFFIC",
                )
                return self._json_response(201, self._normalize_meta_campaign(campaign))

            elif platform == "linkedin_ads":
                # LinkedIn requires campaign group first
                campaign_group_id = body.get("campaign_group_id")
                if not campaign_group_id:
                    return self._error_response(400, "campaign_group_id is required for LinkedIn")

                campaign = await connector.create_campaign(
                    name=name,
                    campaign_group_id=campaign_group_id,
                    campaign_type=campaign_type or "SPONSORED_UPDATES",
                    objective_type=objective or "WEBSITE_VISITS",
                    daily_budget=daily_budget if daily_budget is not None else 50,
                )
                return self._json_response(201, self._normalize_linkedin_campaign(campaign))

            elif platform == "microsoft_ads":
                campaign_id = await connector.create_campaign(
                    name=name,
                    campaign_type=campaign_type or "Search",
                    daily_budget=daily_budget if daily_budget is not None else 50,
                )
                return self._json_response(
                    201,
                    {
                        "campaign_id": campaign_id,
                        "platform": platform,
                        "name": name,
                    },
                )

        except Exception as e:
            logger.warning("Handler error: %s", e)
            return self._error_response(500, "Failed to create campaign")

        return self._error_response(400, "Unsupported platform")

    async def _update_campaign(
        self,
        request: Any,
        platform: str,
        campaign_id: str,
    ) -> dict[str, Any]:
        """Update a campaign."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("Handler error: %s", e)
            return self._error_response(400, "Invalid request body")

        connector = await self._get_connector(platform)
        if not connector:
            return self._error_response(500, f"Could not initialize {platform} connector")

        # --- Validate name if provided ---
        if "name" in body:
            _, err = _validate_campaign_name(body["name"])
            if err:
                return self._error_response(400, err)

        # --- Validate status against allowlist ---
        if "status" in body:
            err = _validate_against_allowlist(body["status"], ALLOWED_CAMPAIGN_STATUSES, "status")
            if err:
                return self._error_response(400, err)

        # --- Validate budget ---
        if "daily_budget" in body:
            budget, err = _validate_budget(body["daily_budget"], "daily_budget", MAX_DAILY_BUDGET)
            if err:
                return self._error_response(400, err)
        else:
            budget = None

        try:
            # Handle status updates
            if "status" in body:
                status = body["status"]
                if platform == "google_ads":
                    await connector.update_campaign_status(campaign_id, status)
                elif platform == "meta_ads":
                    await connector.update_campaign(campaign_id, status=status)
                elif platform == "linkedin_ads":
                    await connector.update_campaign_status(campaign_id, status)
                elif platform == "microsoft_ads":
                    await connector.update_campaign_status(campaign_id, status)

            # Handle budget updates
            if budget is not None:
                if platform == "google_ads":
                    await connector.update_campaign_budget(campaign_id, int(budget * 1_000_000))
                elif platform == "microsoft_ads":
                    await connector.update_campaign_budget(campaign_id, budget)

            return self._json_response(
                200,
                {
                    "message": "Campaign updated",
                    "campaign_id": campaign_id,
                    "platform": platform,
                },
            )

        except Exception as e:
            logger.warning("Handler error: %s", e)
            return self._error_response(500, "Failed to update campaign")

    async def _get_cross_platform_performance(self, request: Any) -> dict[str, Any]:
        """Get performance metrics across all connected platforms."""
        # Parse date range from query params with bounds (1-365 days)
        days = get_clamped_int_param(dict(request.query), "days", 30, 1, 365)
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Gather performance from all platforms in parallel
        tasks = []
        for platform in _platform_credentials.keys():
            tasks.append(self._fetch_platform_performance(platform, start_date, end_date))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        platform_metrics: list[dict[str, Any]] = []
        totals = {
            "impressions": 0,
            "clicks": 0,
            "cost": 0.0,
            "conversions": 0,
            "conversion_value": 0.0,
        }

        for platform, result in zip(_platform_credentials.keys(), results):
            if isinstance(result, BaseException):
                logger.error(f"Error fetching performance from {platform}: {result}")
                continue

            perf_result: dict[str, Any] = result
            platform_metrics.append(perf_result)
            totals["impressions"] += perf_result.get("impressions", 0)
            totals["clicks"] += perf_result.get("clicks", 0)
            totals["cost"] += perf_result.get("cost", 0)
            totals["conversions"] += perf_result.get("conversions", 0)
            totals["conversion_value"] += perf_result.get("conversion_value", 0)

        # Calculate totals
        totals["ctr"] = (
            (totals["clicks"] / totals["impressions"] * 100) if totals["impressions"] > 0 else 0
        )
        totals["cpc"] = (totals["cost"] / totals["clicks"]) if totals["clicks"] > 0 else 0
        totals["cpm"] = (
            (totals["cost"] / totals["impressions"] * 1000) if totals["impressions"] > 0 else 0
        )
        totals["roas"] = (totals["conversion_value"] / totals["cost"]) if totals["cost"] > 0 else 0

        return self._json_response(
            200,
            {
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "platforms": platform_metrics,
                "totals": {
                    k: round(v, 2) if isinstance(v, float) else v for k, v in totals.items()
                },
            },
        )

    async def _fetch_platform_performance(
        self,
        platform: str,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        """Fetch performance metrics from a specific platform."""
        connector = await self._get_connector(platform)
        if not connector:
            return {"platform": platform, "error": "Connector not available"}

        try:
            if platform == "google_ads":
                metrics = await connector.get_campaign_performance(start_date, end_date)
                return self._aggregate_google_metrics(platform, metrics, start_date, end_date)

            elif platform == "meta_ads":
                insights = await connector.get_insights(
                    level="account",
                    date_preset="last_30d",
                )
                return self._aggregate_meta_insights(platform, insights, start_date, end_date)

            elif platform == "linkedin_ads":
                analytics = await connector.get_account_analytics(start_date, end_date)
                return self._normalize_linkedin_analytics(platform, analytics, start_date, end_date)

            elif platform == "microsoft_ads":
                # Microsoft Ads requires async report generation
                return {
                    "platform": platform,
                    "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                    "note": "Microsoft Ads reporting requires async report generation",
                }

        except Exception as e:
            logger.error(f"Error fetching {platform} performance: {e}")
            return {"platform": platform, "error": "Failed to fetch platform performance"}

        return {"platform": platform}

    async def _get_platform_performance(self, request: Any, platform: str) -> dict[str, Any]:
        """Get performance metrics for a specific platform."""
        if platform not in _platform_credentials:
            return self._error_response(404, f"Platform {platform} is not connected")

        # Parse date range from query params with bounds (1-365 days)
        days = get_clamped_int_param(dict(request.query), "days", 30, 1, 365)
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        metrics = await self._fetch_platform_performance(platform, start_date, end_date)

        return self._json_response(200, metrics)

    async def _analyze_performance(self, request: Any) -> dict[str, Any]:
        """Run multi-agent analysis on advertising performance."""
        try:
            body = await self._get_json_body(request)
        except Exception as e:
            logger.warning("Handler error: %s", e)
            return self._error_response(400, "Invalid request body")

        # --- Validate platforms list ---
        platforms = body.get("platforms", list(_platform_credentials.keys()))
        if not isinstance(platforms, list):
            return self._error_response(400, "platforms must be a list")
        for p in platforms:
            if p not in SUPPORTED_PLATFORMS:
                return self._error_response(400, f"Unsupported platform: '{p}'")

        # --- Validate analysis type ---
        analysis_type = body.get("type", "performance_review")
        err = _validate_against_allowlist(analysis_type, ALLOWED_ANALYSIS_TYPES, "type")
        if err:
            return self._error_response(400, err)

        # --- Validate days ---
        days = body.get("days", 30)
        try:
            days = int(days)
        except (TypeError, ValueError):
            return self._error_response(400, "days must be an integer")
        if days < 1 or days > MAX_ANALYSIS_DAYS:
            return self._error_response(400, f"days must be between 1 and {MAX_ANALYSIS_DAYS}")

        # Gather performance data
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        performance_data = {}
        for platform in platforms:
            if platform in _platform_credentials:
                perf = await self._fetch_platform_performance(platform, start_date, end_date)
                performance_data[platform] = perf

        # Create analysis task
        analysis_id = str(uuid4())

        # In production, this would trigger a multi-agent debate
        # For now, return a structured analysis template
        analysis = {
            "analysis_id": analysis_id,
            "type": analysis_type,
            "status": "completed",
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "platforms_analyzed": platforms,
            "summary": self._generate_performance_summary(performance_data),
            "recommendations": self._generate_recommendations(performance_data),
            "insights": self._generate_insights(performance_data),
        }

        return self._json_response(200, analysis)

    async def _get_budget_recommendations(self, request: Any) -> dict[str, Any]:
        """Get budget allocation recommendations across platforms."""
        # --- Validate budget parameter ---
        budget_raw = request.query.get("budget", 10000)
        total_budget, err = _validate_budget(budget_raw, "budget", MAX_TOTAL_BUDGET)
        if err:
            return self._error_response(400, err)
        if total_budget is None:
            total_budget = 10000.0

        # --- Validate objective ---
        objective = request.query.get("objective", "balanced")
        err = _validate_against_allowlist(objective, ALLOWED_BUDGET_OBJECTIVES, "objective")
        if err:
            return self._error_response(400, err)

        # Gather current performance
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        performance_data = {}
        for platform in _platform_credentials.keys():
            perf = await self._fetch_platform_performance(platform, start_date, end_date)
            performance_data[platform] = perf

        # Generate recommendations based on performance
        recommendations = self._calculate_budget_recommendations(
            performance_data, total_budget, objective
        )

        return self._json_response(
            200,
            {
                "total_budget": total_budget,
                "objective": objective,
                "recommendations": recommendations,
                "rationale": self._generate_budget_rationale(performance_data, objective),
            },
        )

    # Helper methods

    def _get_required_credentials(self, platform: str) -> list[str]:
        """Get required credential fields for a platform."""
        requirements = {
            "google_ads": [
                "developer_token",
                "client_id",
                "client_secret",
                "refresh_token",
                "customer_id",
            ],
            "meta_ads": ["access_token", "ad_account_id"],
            "linkedin_ads": ["access_token", "ad_account_id"],
            "microsoft_ads": [
                "developer_token",
                "client_id",
                "client_secret",
                "refresh_token",
                "account_id",
                "customer_id",
            ],
        }
        return requirements.get(platform, [])

    async def _get_connector(self, platform: str) -> Any | None:
        """Get or create a connector for a platform.

        Uses circuit breaker to prevent cascading failures when platform APIs
        are unavailable.
        """
        if platform in _platform_connectors:
            return _platform_connectors[platform]

        if platform not in _platform_credentials:
            return None

        # Check circuit breaker state before attempting connection
        cb = get_advertising_circuit_breaker()
        if not cb.can_execute():
            logger.warning(f"Circuit breaker open for advertising service, skipping {platform}")
            return None

        creds = _platform_credentials[platform]["credentials"]
        connector: Any = None

        try:
            if platform == "google_ads":
                from aragora.connectors.advertising.google_ads import (
                    GoogleAdsConnector,
                    GoogleAdsCredentials,
                )

                connector = GoogleAdsConnector(GoogleAdsCredentials(**creds))

            elif platform == "meta_ads":
                from aragora.connectors.advertising.meta_ads import (
                    MetaAdsConnector,
                    MetaAdsCredentials,
                )

                connector = MetaAdsConnector(MetaAdsCredentials(**creds))

            elif platform == "linkedin_ads":
                from aragora.connectors.advertising.linkedin_ads import (
                    LinkedInAdsConnector,
                    LinkedInAdsCredentials,
                )

                connector = LinkedInAdsConnector(LinkedInAdsCredentials(**creds))

            elif platform == "microsoft_ads":
                from aragora.connectors.advertising.microsoft_ads import (
                    MicrosoftAdsConnector,
                    MicrosoftAdsCredentials,
                )

                connector = MicrosoftAdsConnector(MicrosoftAdsCredentials(**creds))

            else:
                return None

            _platform_connectors[platform] = connector
            cb.record_success()
            return connector

        except Exception as e:
            logger.error(f"Failed to create {platform} connector: {e}")
            cb.record_failure()
            return None

    def _normalize_google_campaign(self, campaign: Any) -> dict[str, Any]:
        """Normalize Google Ads campaign to unified format."""
        return {
            "id": campaign.id,
            "platform": "google_ads",
            "name": campaign.name,
            "status": (
                campaign.status.value if hasattr(campaign.status, "value") else campaign.status
            ),
            "objective": (
                campaign.campaign_type.value
                if hasattr(campaign.campaign_type, "value")
                else campaign.campaign_type
            ),
            "daily_budget": campaign.budget_micros / 1_000_000 if campaign.budget_micros else None,
            "total_budget": None,
            "start_date": campaign.start_date.isoformat() if campaign.start_date else None,
            "end_date": campaign.end_date.isoformat() if campaign.end_date else None,
            "bidding_strategy": (
                campaign.bidding_strategy.value
                if hasattr(campaign.bidding_strategy, "value")
                else campaign.bidding_strategy
            ),
        }

    def _normalize_meta_campaign(self, campaign: Any) -> dict[str, Any]:
        """Normalize Meta Ads campaign to unified format."""
        return {
            "id": campaign.id,
            "platform": "meta_ads",
            "name": campaign.name,
            "status": (
                campaign.status.value if hasattr(campaign.status, "value") else campaign.status
            ),
            "objective": (
                campaign.objective.value
                if hasattr(campaign.objective, "value")
                else campaign.objective
            ),
            "daily_budget": campaign.daily_budget,
            "total_budget": campaign.lifetime_budget,
            "start_date": campaign.start_time.date().isoformat() if campaign.start_time else None,
            "end_date": campaign.stop_time.date().isoformat() if campaign.stop_time else None,
            "spend_cap": campaign.spend_cap,
        }

    def _normalize_linkedin_campaign(self, campaign: Any) -> dict[str, Any]:
        """Normalize LinkedIn Ads campaign to unified format."""
        return {
            "id": campaign.id,
            "platform": "linkedin_ads",
            "name": campaign.name,
            "status": (
                campaign.status.value if hasattr(campaign.status, "value") else campaign.status
            ),
            "objective": (
                campaign.objective_type.value
                if hasattr(campaign.objective_type, "value")
                else str(campaign.objective_type)
            ),
            "daily_budget": campaign.daily_budget,
            "total_budget": campaign.total_budget,
            "start_date": (
                campaign.run_schedule_start.date().isoformat()
                if campaign.run_schedule_start
                else None
            ),
            "end_date": (
                campaign.run_schedule_end.date().isoformat() if campaign.run_schedule_end else None
            ),
            "campaign_type": (
                campaign.campaign_type.value
                if hasattr(campaign.campaign_type, "value")
                else campaign.campaign_type
            ),
        }

    def _normalize_microsoft_campaign(self, campaign: Any) -> dict[str, Any]:
        """Normalize Microsoft Ads campaign to unified format."""
        return {
            "id": campaign.id,
            "platform": "microsoft_ads",
            "name": campaign.name,
            "status": (
                campaign.status.value if hasattr(campaign.status, "value") else campaign.status
            ),
            "objective": (
                campaign.campaign_type.value
                if hasattr(campaign.campaign_type, "value")
                else campaign.campaign_type
            ),
            "daily_budget": campaign.daily_budget,
            "total_budget": None,
            "start_date": campaign.start_date.isoformat() if campaign.start_date else None,
            "end_date": campaign.end_date.isoformat() if campaign.end_date else None,
            "bidding_scheme": (
                campaign.bidding_scheme.value
                if hasattr(campaign.bidding_scheme, "value")
                else campaign.bidding_scheme
            ),
        }

    def _aggregate_google_metrics(
        self,
        platform: str,
        metrics: list[Any],
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        """Aggregate Google Ads metrics."""
        totals = {
            "impressions": sum(m.impressions for m in metrics),
            "clicks": sum(m.clicks for m in metrics),
            "cost": sum(m.cost_micros / 1_000_000 for m in metrics),
            "conversions": sum(m.conversions for m in metrics),
            "conversion_value": sum(m.conversion_value for m in metrics),
        }

        totals["ctr"] = (
            (totals["clicks"] / totals["impressions"] * 100) if totals["impressions"] > 0 else 0
        )
        totals["cpc"] = (totals["cost"] / totals["clicks"]) if totals["clicks"] > 0 else 0
        totals["cpm"] = (
            (totals["cost"] / totals["impressions"] * 1000) if totals["impressions"] > 0 else 0
        )
        totals["roas"] = (totals["conversion_value"] / totals["cost"]) if totals["cost"] > 0 else 0

        return {
            "platform": platform,
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            **{k: round(v, 2) if isinstance(v, float) else v for k, v in totals.items()},
        }

    def _aggregate_meta_insights(
        self,
        platform: str,
        insights: list[Any],
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        """Aggregate Meta Ads insights."""
        totals = {
            "impressions": sum(i.impressions for i in insights),
            "clicks": sum(i.clicks for i in insights),
            "cost": sum(i.spend for i in insights),
            "conversions": sum(i.conversions for i in insights),
            "conversion_value": sum(i.conversion_value for i in insights),
        }

        totals["ctr"] = (
            (totals["clicks"] / totals["impressions"] * 100) if totals["impressions"] > 0 else 0
        )
        totals["cpc"] = (totals["cost"] / totals["clicks"]) if totals["clicks"] > 0 else 0
        totals["cpm"] = (
            (totals["cost"] / totals["impressions"] * 1000) if totals["impressions"] > 0 else 0
        )
        totals["roas"] = (totals["conversion_value"] / totals["cost"]) if totals["cost"] > 0 else 0

        return {
            "platform": platform,
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            **{k: round(v, 2) if isinstance(v, float) else v for k, v in totals.items()},
        }

    def _normalize_linkedin_analytics(
        self,
        platform: str,
        analytics: Any,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        """Normalize LinkedIn Ads analytics."""
        return {
            "platform": platform,
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "impressions": analytics.impressions,
            "clicks": analytics.clicks,
            "cost": round(analytics.cost, 2),
            "conversions": analytics.conversions,
            "leads": analytics.leads,
            "ctr": round(analytics.ctr, 4),
            "cpc": round(analytics.cpc, 2),
            "cpm": round(analytics.cpm, 2),
        }

    def _generate_performance_summary(self, performance_data: dict[str, Any]) -> dict[str, Any]:
        """Generate a performance summary across platforms."""
        total_spend = sum(p.get("cost", 0) for p in performance_data.values())
        total_conversions = sum(p.get("conversions", 0) for p in performance_data.values())

        best_roas_platform: tuple[str | None, dict[str, Any]] = max(
            performance_data.items(), key=lambda x: x[1].get("roas", 0), default=(None, {})
        )

        return {
            "total_spend": round(total_spend, 2),
            "total_conversions": total_conversions,
            "best_performing_platform": best_roas_platform[0],
            "platforms_analyzed": len(performance_data),
        }

    def _generate_recommendations(self, performance_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []

        for platform, data in performance_data.items():
            roas = data.get("roas", 0)
            cpc = data.get("cpc", 0)

            if roas > 3:
                recommendations.append(
                    {
                        "platform": platform,
                        "type": "increase_budget",
                        "priority": "high",
                        "message": f"Strong ROAS of {roas:.2f}x. Consider increasing budget allocation.",
                    }
                )
            elif roas < 1:
                recommendations.append(
                    {
                        "platform": platform,
                        "type": "optimize",
                        "priority": "high",
                        "message": f"ROAS below 1x ({roas:.2f}x). Review targeting and creative performance.",
                    }
                )

            if cpc > 5:
                recommendations.append(
                    {
                        "platform": platform,
                        "type": "reduce_cpc",
                        "priority": "medium",
                        "message": f"High CPC of ${cpc:.2f}. Consider bid adjustments or audience refinement.",
                    }
                )

        return recommendations

    def _generate_insights(self, performance_data: dict[str, Any]) -> list[str]:
        """Generate performance insights."""
        insights = []

        total_spend = sum(p.get("cost", 0) for p in performance_data.values())
        if total_spend > 0:
            for platform, data in performance_data.items():
                share = data.get("cost", 0) / total_spend * 100
                insights.append(f"{platform} accounts for {share:.1f}% of total ad spend")

        return insights

    def _calculate_budget_recommendations(
        self,
        performance_data: dict[str, Any],
        total_budget: float,
        objective: str,
    ) -> list[dict[str, Any]]:
        """Calculate budget allocation recommendations."""
        recommendations = []

        # Simple allocation based on ROAS
        total_roas = sum(p.get("roas", 1) for p in performance_data.values())

        for platform, data in performance_data.items():
            roas = data.get("roas", 1)
            if objective == "conversions":
                # Allocate more to high-ROAS platforms
                share = roas / total_roas if total_roas > 0 else 1 / len(performance_data)
            elif objective == "awareness":
                # Allocate based on CPM efficiency
                share = 1 / len(performance_data)
            else:
                # Balanced
                share = roas / total_roas if total_roas > 0 else 1 / len(performance_data)

            recommendations.append(
                {
                    "platform": platform,
                    "recommended_budget": round(total_budget * share, 2),
                    "share_percentage": round(share * 100, 1),
                    "expected_roas": roas,
                }
            )

        return recommendations

    def _generate_budget_rationale(
        self,
        performance_data: dict[str, Any],
        objective: str,
    ) -> str:
        """Generate rationale for budget recommendations."""
        if objective == "conversions":
            return "Budget allocated based on historical ROAS performance. Platforms with higher ROAS receive proportionally larger budgets to maximize conversions."
        elif objective == "awareness":
            return "Budget distributed evenly to maximize reach across platforms. Consider CPM efficiency for awareness campaigns."
        else:
            return "Balanced allocation considering both reach and conversion performance across platforms."

    async def _get_json_body(self, request: Any) -> dict[str, Any]:
        """Parse JSON body from request.

        Wraps parse_json_body and returns just the dict, raising on error.
        """
        body, _err = await parse_json_body(request, context="advertising")
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


__all__ = ["AdvertisingHandler"]
