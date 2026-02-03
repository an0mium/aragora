"""
Inbox Command Center API Handler.

Provides unified API endpoints for the inbox command center including:
- Prioritized email fetching with cross-channel context
- Quick actions (archive, snooze, reply, forward)
- Bulk operations
- Daily digest statistics
- Sender profile lookups

Endpoints:
- GET /api/inbox/command - Fetch prioritized inbox
- POST /api/inbox/actions - Execute quick action
- POST /api/inbox/bulk-actions - Execute bulk action
- GET /api/inbox/sender-profile - Get sender profile
- GET /api/inbox/daily-digest - Get daily digest
- POST /api/inbox/reprioritize - Trigger AI re-prioritization
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

from aiohttp import web

from aragora.rbac.checker import get_permission_checker
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.utils.auth import get_auth_context, UnauthorizedError
from aragora.server.handlers.utils import parse_json_body
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.services import (
    ServiceRegistry,
    EmailPrioritizer,
    EmailPrioritizationConfig,
    SenderHistoryService,
)
from aragora.cache import HybridTTLCache, register_cache
from aragora.utils.redis_cache import RedisTTLCache
from aragora.server.validation.query_params import safe_query_int

# ---------------------------------------------------------------------------
# Security constants: allowlists and input bounds
# ---------------------------------------------------------------------------

# Explicit allowlist of valid quick/bulk actions. Any action not in this set
# is rejected before reaching handler logic, preventing command injection.
ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {
        "archive",
        "snooze",
        "reply",
        "forward",
        "spam",
        "mark_important",
        "mark_vip",
        "block",
        "delete",
    }
)

# Explicit allowlist of valid bulk-action filter types.
ALLOWED_BULK_FILTERS: frozenset[str] = frozenset(
    {
        "low",
        "deferred",
        "spam",
        "read",
        "all",
    }
)

# Explicit allowlist of valid priority filter values for GET /inbox/command.
ALLOWED_PRIORITY_FILTERS: frozenset[str] = frozenset(
    {
        "critical",
        "high",
        "medium",
        "low",
        "defer",
    }
)

# Explicit allowlist of valid force_tier values for reprioritization.
ALLOWED_FORCE_TIERS: frozenset[str] = frozenset(
    {
        "tier_1_rules",
        "tier_2_lightweight",
        "tier_3_debate",
    }
)

# Explicit allowlist of valid snooze duration values.
ALLOWED_SNOOZE_DURATIONS: frozenset[str] = frozenset(
    {
        "1h",
        "3h",
        "1d",
        "3d",
        "1w",
    }
)

# Input length bounds
MAX_EMAIL_ID_LENGTH = 256
MAX_EMAIL_IDS_PER_REQUEST = 200
MAX_EMAIL_ADDRESS_LENGTH = 320  # RFC 5321 maximum
MAX_REPLY_BODY_LENGTH = 100_000  # 100 KB
MAX_FORWARD_TO_LENGTH = 320
MAX_SENDER_PARAM_LENGTH = 320
MAX_PARAMS_KEYS = 20

# Pattern for validating email IDs (alphanumeric, hyphens, underscores, dots)
_EMAIL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
# RFC 5322 simplified email validation
_EMAIL_ADDRESS_PATTERN = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~\-]+@[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$"
)


def _validate_email_id(email_id: Any) -> str | None:
    """Validate and sanitize an email ID.

    Returns the sanitized ID string, or None if invalid.
    """
    if not isinstance(email_id, str):
        return None
    email_id = email_id.strip()
    if not email_id or len(email_id) > MAX_EMAIL_ID_LENGTH:
        return None
    if not _EMAIL_ID_PATTERN.match(email_id):
        return None
    return email_id


def _validate_email_address(address: Any) -> str | None:
    """Validate and sanitize an email address.

    Returns the sanitized address string, or None if invalid.
    """
    if not isinstance(address, str):
        return None
    address = address.strip()
    if not address or len(address) > MAX_EMAIL_ADDRESS_LENGTH:
        return None
    if not _EMAIL_ADDRESS_PATTERN.match(address):
        return None
    return address


def _sanitize_string_param(value: Any, max_length: int) -> str:
    """Sanitize a generic string parameter.

    Returns the stripped and length-bounded string, or empty string if invalid.
    """
    if not isinstance(value, str):
        return ""
    return value.strip()[:max_length]


def _validate_params(params: Any) -> dict[str, Any] | None:
    """Validate the params dict from request body.

    Returns sanitized params dict, or None if invalid.
    """
    if params is None:
        return {}
    if not isinstance(params, dict):
        return None
    if len(params) > MAX_PARAMS_KEYS:
        return None
    return params


if TYPE_CHECKING:
    from aragora.connectors.enterprise.communication.gmail import GmailConnector

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=dict[str, Any])


class IterableTTLCache(Generic[T]):
    """
    TTL cache wrapper that supports iteration for inbox operations.

    Wraps HybridTTLCache to provide dict-like iteration while maintaining
    Redis persistence for multi-instance deployments.
    """

    def __init__(self, name: str, maxsize: int, ttl_seconds: float) -> None:
        self._cache: RedisTTLCache[T] = HybridTTLCache(
            prefix=name,
            maxsize=maxsize,
            ttl_seconds=ttl_seconds,
        )
        self._keys: set[str] = set()  # Track keys for iteration
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        """Get value from cache."""
        return self._cache.get(key)

    def set(self, key: str, value: T) -> None:
        """Store value in cache."""
        self._cache.set(key, value)
        with self._lock:
            self._keys.add(key)

    def __setitem__(self, key: str, value: T) -> None:
        """Dict-style assignment."""
        self.set(key, value)

    def __getitem__(self, key: str) -> T:
        """Dict-style access."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    def items(self) -> list[tuple[str, T]]:
        """Return list of (key, value) pairs."""
        result: list[tuple[str, T]] = []
        with self._lock:
            for key in list(self._keys):
                value = self.get(key)
                if value is not None:
                    result.append((key, value))
                else:
                    self._keys.discard(key)
        return result

    def values(self) -> list[T]:
        """Return list of values."""
        return [v for _, v in self.items()]

    def invalidate(self, key: str) -> bool:
        """Remove key from cache."""
        with self._lock:
            self._keys.discard(key)
        return self._cache.invalidate(key)

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.stats


# Production-ready cache for prioritized emails (Redis when available, fallback to in-memory)
_email_cache: IterableTTLCache = IterableTTLCache(
    name="inbox_email_cache",
    maxsize=10000,
    ttl_seconds=3600,  # 1 hour TTL
)
_priority_results: IterableTTLCache = IterableTTLCache(
    name="inbox_priority_results",
    maxsize=1000,
    ttl_seconds=1800,  # 30 min TTL
)

# Register underlying caches for monitoring
register_cache("inbox_email", _email_cache._cache)
register_cache("inbox_priority", _priority_results._cache)


@dataclass
class InboxCommandHandler:
    """Handler for inbox command center API endpoints."""

    gmail_connector: Optional["GmailConnector"] = None
    prioritizer: EmailPrioritizer | None = None
    sender_history: SenderHistoryService | None = None
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize services from registry if not provided."""
        self._ensure_services()

    def _ensure_services(self) -> None:
        """Lazily initialize services from the registry."""
        if self._initialized:
            return

        registry = ServiceRegistry.get()

        # Try to get GmailConnector from registry
        if self.gmail_connector is None:
            try:
                from aragora.connectors.enterprise.communication.gmail import GmailConnector

                if registry.has(GmailConnector):
                    self.gmail_connector = registry.resolve(GmailConnector)
                    logger.debug("Resolved GmailConnector from registry")
            except ImportError as e:
                logger.debug("GmailConnector module not available: %s", e)
            except (TypeError, RuntimeError) as e:
                # TypeError: registry misconfiguration; RuntimeError: dependency issues
                logger.debug("GmailConnector not available: %s", e)

        # Try to get or create EmailPrioritizer
        if self.prioritizer is None:
            if registry.has(EmailPrioritizer):
                self.prioritizer = registry.resolve(EmailPrioritizer)
                logger.debug("Resolved EmailPrioritizer from registry")
            elif self.gmail_connector is not None:
                # Create a prioritizer with the connector
                self.prioritizer = EmailPrioritizer(
                    gmail_connector=self.gmail_connector,
                    config=EmailPrioritizationConfig(),
                )
                registry.register(EmailPrioritizer, self.prioritizer)
                logger.info("Created and registered EmailPrioritizer")

        # Try to get SenderHistoryService
        if self.sender_history is None:
            if registry.has(SenderHistoryService):
                self.sender_history = registry.resolve(SenderHistoryService)
                logger.debug("Resolved SenderHistoryService from registry")

        self._initialized = True

    async def _check_permission(self, request: web.Request, permission: str) -> None:
        """Check if the request has the required permission.

        SECURITY: Uses JWT-only authentication. X-User-ID headers are NOT trusted
        to prevent user impersonation attacks.

        Raises:
            web.HTTPForbidden: If permission check fails
            web.HTTPUnauthorized: If no valid authentication
        """
        try:
            # SECURITY: Only trust JWT tokens, never trust X-User-ID headers
            context = await get_auth_context(request, require_auth=False)
        except UnauthorizedError as e:
            raise web.HTTPUnauthorized(
                text=str(e),
                content_type="application/json",
            )
        except (ValueError, KeyError, AttributeError) as e:
            # Auth extraction failed due to malformed token or missing fields
            logger.warning("Auth extraction failed: %s", e)
            context = AuthorizationContext(
                user_id="anonymous",
                org_id=None,
                roles=set(),
            )

        if not context.user_id or context.user_id == "anonymous":
            # Require authentication for all inbox operations
            raise web.HTTPUnauthorized(
                text="Authentication required for inbox access",
                content_type="application/json",
            )

        checker = get_permission_checker()
        decision = checker.check_permission(context, permission)

        if not decision.allowed:
            logger.warning(
                "Permission denied: %s for user %s - %s",
                permission,
                context.user_id,
                decision.reason,
            )
            raise web.HTTPForbidden(
                text=f"Permission denied: {decision.reason}",
                content_type="application/json",
            )

    @rate_limit(requests_per_minute=60, limiter_name="inbox_read")
    async def handle_get_inbox(self, request: web.Request) -> web.Response:
        """
        GET /api/inbox/command

        Fetch prioritized inbox with stats.

        Query params:
            - limit: Max emails to return (default 50)
            - offset: Pagination offset (default 0)
            - priority: Filter by priority level (critical, high, medium, low, defer)
            - unread_only: Only return unread emails (default false)
        """
        try:
            await self._check_permission(request, "inbox:read")
            self._ensure_services()

            limit = safe_query_int(request.query, "limit", default=50, max_val=1000)
            offset = safe_query_int(request.query, "offset", default=0, max_val=100000)
            priority_filter = request.query.get("priority")
            unread_only = request.query.get("unread_only", "false").lower() == "true"

            # Validate priority filter against allowlist
            if priority_filter is not None:
                priority_filter = priority_filter.strip().lower()
                if priority_filter not in ALLOWED_PRIORITY_FILTERS:
                    return web.json_response(
                        {
                            "success": False,
                            "error": f"Invalid priority filter. Allowed values: {', '.join(sorted(ALLOWED_PRIORITY_FILTERS))}",
                        },
                        status=400,
                    )

            # Get emails from service
            emails = await self._fetch_prioritized_emails(
                limit=limit,
                offset=offset,
                priority_filter=priority_filter,
                unread_only=unread_only,
            )

            # Calculate stats
            stats = await self._calculate_inbox_stats(emails)

            return web.json_response(
                {
                    "success": True,
                    "emails": emails,
                    "total": stats["total"],
                    "stats": stats,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        except (web.HTTPUnauthorized, web.HTTPForbidden):
            raise
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.exception("Failed to fetch inbox: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    @rate_limit(requests_per_minute=30, limiter_name="inbox_write")
    async def handle_quick_action(self, request: web.Request) -> web.Response:
        """
        POST /api/inbox/actions

        Execute quick action on email(s).

        Body:
            - action: Action to perform (archive, snooze, reply, forward, spam, etc.)
            - emailIds: List of email IDs to act on
            - params: Optional action-specific parameters
        """
        try:
            await self._check_permission(request, "inbox:write")
            self._ensure_services()

            body, err = await parse_json_body(request, context="inbox_quick_action")
            if err:
                return err
            action = body.get("action")
            raw_email_ids = body.get("emailIds")
            raw_params = body.get("params", {})

            if not action or not isinstance(action, str):
                return web.json_response(
                    {"success": False, "error": "action is required"},
                    status=400,
                )

            # Validate action against allowlist
            action = action.strip().lower()
            if action not in ALLOWED_ACTIONS:
                return web.json_response(
                    {
                        "success": False,
                        "error": f"Invalid action '{action}'. Allowed actions: {', '.join(sorted(ALLOWED_ACTIONS))}",
                    },
                    status=400,
                )

            # Validate emailIds is a list with bounded length
            if raw_email_ids is None:
                return web.json_response(
                    {"success": False, "error": "emailIds is required"},
                    status=400,
                )
            if not isinstance(raw_email_ids, list) or not raw_email_ids:
                return web.json_response(
                    {"success": False, "error": "emailIds must be a non-empty list"},
                    status=400,
                )
            if len(raw_email_ids) > MAX_EMAIL_IDS_PER_REQUEST:
                return web.json_response(
                    {
                        "success": False,
                        "error": f"emailIds exceeds maximum of {MAX_EMAIL_IDS_PER_REQUEST}",
                    },
                    status=400,
                )

            # Validate and sanitize each email ID
            email_ids: list[str] = []
            for raw_id in raw_email_ids:
                validated = _validate_email_id(raw_id)
                if validated is None:
                    return web.json_response(
                        {
                            "success": False,
                            "error": f"Invalid email ID: must be alphanumeric (max {MAX_EMAIL_ID_LENGTH} chars)",
                        },
                        status=400,
                    )
                email_ids.append(validated)

            # Validate params
            params = _validate_params(raw_params)
            if params is None:
                return web.json_response(
                    {"success": False, "error": "Invalid params object"},
                    status=400,
                )

            # Sanitize action-specific params
            params = self._sanitize_action_params(action, params)

            # Execute action
            results = await self._execute_action(action, email_ids, params)

            return web.json_response(
                {
                    "success": True,
                    "action": action,
                    "processed": len(email_ids),
                    "results": results,
                }
            )
        except (web.HTTPUnauthorized, web.HTTPForbidden):
            raise
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.exception("Failed to execute action: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    @rate_limit(requests_per_minute=10, limiter_name="inbox_bulk_write")
    async def handle_bulk_action(self, request: web.Request) -> web.Response:
        """
        POST /api/inbox/bulk-actions

        Execute bulk action based on filter.

        Body:
            - action: Action to perform
            - filter: Filter to apply (low, deferred, spam, read, all)
            - params: Optional action-specific parameters
        """
        try:
            await self._check_permission(request, "inbox:write")
            self._ensure_services()

            body, err = await parse_json_body(request, context="inbox_bulk_action")
            if err:
                return err
            action = body.get("action")
            filter_type = body.get("filter")
            raw_params = body.get("params", {})

            if (
                not action
                or not isinstance(action, str)
                or not filter_type
                or not isinstance(filter_type, str)
            ):
                return web.json_response(
                    {"success": False, "error": "action and filter are required"},
                    status=400,
                )

            # Validate action against allowlist
            action = action.strip().lower()
            if action not in ALLOWED_ACTIONS:
                return web.json_response(
                    {
                        "success": False,
                        "error": f"Invalid action '{action}'. Allowed actions: {', '.join(sorted(ALLOWED_ACTIONS))}",
                    },
                    status=400,
                )

            # Validate filter against allowlist
            filter_type = filter_type.strip().lower()
            if filter_type not in ALLOWED_BULK_FILTERS:
                return web.json_response(
                    {
                        "success": False,
                        "error": f"Invalid filter '{filter_type}'. Allowed filters: {', '.join(sorted(ALLOWED_BULK_FILTERS))}",
                    },
                    status=400,
                )

            # Validate params
            params = _validate_params(raw_params)
            if params is None:
                return web.json_response(
                    {"success": False, "error": "Invalid params object"},
                    status=400,
                )

            # Sanitize action-specific params
            params = self._sanitize_action_params(action, params)

            # Get matching email IDs
            email_ids = await self._get_emails_by_filter(filter_type)

            if not email_ids:
                return web.json_response(
                    {
                        "success": True,
                        "action": action,
                        "processed": 0,
                        "message": "No emails matched the filter",
                    }
                )

            # Execute action
            results = await self._execute_action(action, email_ids, params)

            return web.json_response(
                {
                    "success": True,
                    "action": action,
                    "filter": filter_type,
                    "processed": len(email_ids),
                    "results": results,
                }
            )
        except (web.HTTPUnauthorized, web.HTTPForbidden):
            raise
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.exception("Failed to execute bulk action: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    @rate_limit(requests_per_minute=60, limiter_name="inbox_read")
    async def handle_get_sender_profile(self, request: web.Request) -> web.Response:
        """
        GET /api/inbox/sender-profile

        Get profile information for a sender.

        Query params:
            - email: Sender email address
        """
        try:
            await self._check_permission(request, "inbox:read")
            self._ensure_services()

            raw_email = request.query.get("email")
            if not raw_email:
                return web.json_response(
                    {"success": False, "error": "email parameter is required"},
                    status=400,
                )

            email = _validate_email_address(raw_email)
            if email is None:
                return web.json_response(
                    {"success": False, "error": "Invalid email address format"},
                    status=400,
                )

            profile = await self._get_sender_profile(email)
            return web.json_response({"success": True, "profile": profile})
        except (web.HTTPUnauthorized, web.HTTPForbidden):
            raise
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.exception("Failed to get sender profile: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    @rate_limit(requests_per_minute=30, limiter_name="inbox_read")
    async def handle_get_daily_digest(self, request: web.Request) -> web.Response:
        """
        GET /api/inbox/daily-digest

        Get daily digest statistics.
        """
        try:
            await self._check_permission(request, "inbox:read")
            self._ensure_services()

            digest = await self._calculate_daily_digest()
            return web.json_response({"success": True, "digest": digest})
        except (web.HTTPUnauthorized, web.HTTPForbidden):
            raise
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.exception("Failed to get daily digest: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    @rate_limit(requests_per_minute=10, limiter_name="inbox_reprioritize")
    async def handle_reprioritize(self, request: web.Request) -> web.Response:
        """
        POST /api/inbox/reprioritize

        Trigger AI re-prioritization of inbox.

        Body:
            - emailIds: Optional list of specific email IDs to reprioritize
            - force_tier: Optional tier to force (tier_1_rules, tier_2_lightweight, tier_3_debate)
        """
        try:
            await self._check_permission(request, "inbox:write")
            self._ensure_services()

            body, err = await parse_json_body(request, context="inbox_reprioritize")
            if err:
                return err
            raw_email_ids = body.get("emailIds")
            force_tier = body.get("force_tier")

            # Validate and sanitize emailIds if provided
            email_ids: list[str] | None = None
            if raw_email_ids is not None:
                if not isinstance(raw_email_ids, list):
                    return web.json_response(
                        {"success": False, "error": "emailIds must be a list"},
                        status=400,
                    )
                if len(raw_email_ids) > MAX_EMAIL_IDS_PER_REQUEST:
                    return web.json_response(
                        {
                            "success": False,
                            "error": f"emailIds exceeds maximum of {MAX_EMAIL_IDS_PER_REQUEST}",
                        },
                        status=400,
                    )
                email_ids = []
                for raw_id in raw_email_ids:
                    validated = _validate_email_id(raw_id)
                    if validated is None:
                        return web.json_response(
                            {
                                "success": False,
                                "error": f"Invalid email ID: must be alphanumeric (max {MAX_EMAIL_ID_LENGTH} chars)",
                            },
                            status=400,
                        )
                    email_ids.append(validated)

            # Validate force_tier against allowlist
            if force_tier is not None:
                if not isinstance(force_tier, str):
                    return web.json_response(
                        {"success": False, "error": "force_tier must be a string"},
                        status=400,
                    )
                force_tier = force_tier.strip().lower()
                if force_tier not in ALLOWED_FORCE_TIERS:
                    return web.json_response(
                        {
                            "success": False,
                            "error": f"Invalid force_tier. Allowed values: {', '.join(sorted(ALLOWED_FORCE_TIERS))}",
                        },
                        status=400,
                    )

            # Trigger reprioritization
            result = await self._reprioritize_emails(email_ids, force_tier)

            return web.json_response(
                {
                    "success": True,
                    "reprioritized": result["count"],
                    "changes": result["changes"],
                    "tier_used": result.get("tier_used"),
                }
            )
        except (web.HTTPUnauthorized, web.HTTPForbidden):
            raise
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.exception("Failed to reprioritize: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    # =========================================================================
    # Private helper methods - Service Integration
    # =========================================================================

    async def _fetch_prioritized_emails(
        self,
        limit: int,
        offset: int,
        priority_filter: str | None,
        unread_only: bool,
    ) -> list[dict[str, Any]]:
        """Fetch and prioritize emails using the EmailPrioritizer service."""
        # If no Gmail connector, return demo data
        if self.gmail_connector is None:
            return self._get_demo_emails(limit, offset, priority_filter)

        try:
            # Fetch emails from Gmail
            emails: list[Any] = []
            fetch_limit = limit + offset + 50  # Fetch extra for filtering

            # Use list_messages instead of sync_items for inbox fetching
            messages = await self.gmail_connector.list_messages(max_results=fetch_limit)
            for msg in messages:
                if len(emails) >= fetch_limit:
                    break
                emails.append(msg)

            if not emails:
                return []

            # Prioritize emails
            if self.prioritizer:
                results = await self.prioritizer.rank_inbox(emails, limit=fetch_limit)

                # Convert to response format
                prioritized = []
                for result in results:
                    # Find the original email
                    email_data = next(
                        (e for e in emails if getattr(e, "id", None) == result.email_id),
                        None,
                    )

                    entry = {
                        "id": result.email_id,
                        "from": (
                            getattr(email_data, "from_address", "unknown")
                            if email_data
                            else "unknown"
                        ),
                        "subject": (
                            getattr(email_data, "subject", "No subject")
                            if email_data
                            else "No subject"
                        ),
                        "snippet": getattr(email_data, "snippet", "")[:200] if email_data else "",
                        "priority": result.priority.name.lower(),
                        "confidence": result.confidence,
                        "reasoning": result.rationale,
                        "tier_used": result.tier_used.value,
                        "scores": {
                            "sender": result.sender_score,
                            "urgency": result.content_urgency_score,
                            "context": result.context_relevance_score,
                            "time_sensitivity": result.time_sensitivity_score,
                        },
                        "suggested_labels": result.suggested_labels,
                        "auto_archive": result.auto_archive,
                        "timestamp": (
                            getattr(email_data, "date", datetime.utcnow()).isoformat()
                            if email_data
                            else datetime.utcnow().isoformat()
                        ),
                        "unread": getattr(email_data, "unread", True) if email_data else True,
                    }

                    # Apply filters
                    if priority_filter and entry["priority"] != priority_filter.lower():
                        continue
                    if unread_only and not entry["unread"]:
                        continue

                    prioritized.append(entry)

                    # Cache for bulk operations
                    _email_cache[result.email_id] = entry

                # Apply pagination
                return prioritized[offset : offset + limit]
            else:
                # No prioritizer - return basic list
                return [
                    {
                        "id": getattr(e, "id", f"email_{i}"),
                        "from": getattr(e, "from_address", "unknown"),
                        "subject": getattr(e, "subject", "No subject"),
                        "snippet": getattr(e, "snippet", "")[:200],
                        "priority": "medium",
                        "confidence": 0.5,
                        "timestamp": getattr(e, "date", datetime.utcnow()).isoformat(),
                        "unread": getattr(e, "unread", True),
                    }
                    for i, e in enumerate(emails[offset : offset + limit])
                ]

        except (OSError, ConnectionError, RuntimeError, ValueError, AttributeError) as e:
            logger.warning("Failed to fetch from Gmail, using demo data: %s", e)
            return self._get_demo_emails(limit, offset, priority_filter)

    def _get_demo_emails(
        self,
        limit: int,
        offset: int,
        priority_filter: str | None,
    ) -> list[dict[str, Any]]:
        """Return demo email data when services aren't available."""
        demo_emails = [
            {
                "id": "demo_1",
                "from": "ceo@company.com",
                "subject": "Q4 Strategy Review - Urgent Response Needed",
                "snippet": "Please review the attached strategy document and provide your feedback by EOD...",
                "priority": "critical",
                "confidence": 0.95,
                "reasoning": "VIP sender; deadline detected; reply expected",
                "tier_used": "tier_1_rules",
                "category": "Work",
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "unread": True,
            },
            {
                "id": "demo_2",
                "from": "client@bigcorp.com",
                "subject": "Contract renewal discussion",
                "snippet": "Following up on our conversation about the contract renewal. Can we schedule a call?",
                "priority": "high",
                "confidence": 0.85,
                "reasoning": "Client sender; reply expected",
                "tier_used": "tier_1_rules",
                "category": "Work",
                "timestamp": (datetime.utcnow() - timedelta(hours=5)).isoformat(),
                "unread": True,
            },
            {
                "id": "demo_3",
                "from": "notifications@github.com",
                "subject": "[aragora] PR #142: Fix memory leak in debate engine",
                "snippet": "A new pull request has been opened by contributor...",
                "priority": "medium",
                "confidence": 0.75,
                "reasoning": "Automated notification; work-related",
                "tier_used": "tier_1_rules",
                "category": "Updates",
                "timestamp": (datetime.utcnow() - timedelta(hours=8)).isoformat(),
                "unread": True,
            },
            {
                "id": "demo_4",
                "from": "newsletter@techblog.com",
                "subject": "This week in AI: Top stories and insights",
                "snippet": "Unsubscribe | View in browser. This week's top AI stories include...",
                "priority": "defer",
                "confidence": 0.92,
                "reasoning": "Newsletter detected; auto-archive candidate",
                "tier_used": "tier_1_rules",
                "category": "Newsletter",
                "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "unread": False,
                "auto_archive": True,
            },
            {
                "id": "demo_5",
                "from": "team@company.com",
                "subject": "Weekly standup notes",
                "snippet": "Here are the notes from this week's standup meeting...",
                "priority": "low",
                "confidence": 0.8,
                "reasoning": "Internal sender; informational",
                "tier_used": "tier_1_rules",
                "category": "Work",
                "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "unread": False,
            },
        ]

        # Populate cache for bulk actions to work in demo mode
        for email in demo_emails:
            _email_cache[str(email["id"])] = email

        # Apply filters
        if priority_filter:
            demo_emails = [e for e in demo_emails if e["priority"] == priority_filter.lower()]

        return demo_emails[offset : offset + limit]

    async def _calculate_inbox_stats(
        self,
        emails: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate inbox statistics from prioritized emails."""
        total = len(emails)

        # Count by priority
        priority_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "defer": 0,
        }

        for email in emails:
            priority = email.get("priority", "medium").lower()
            if priority in priority_counts:
                priority_counts[priority] += 1

        # Count unread
        unread_count = sum(1 for e in emails if e.get("unread", False))

        return {
            "total": total,
            "unread": unread_count,
            "critical": priority_counts["critical"],
            "high": priority_counts["high"],
            "medium": priority_counts["medium"],
            "low": priority_counts["low"],
            "deferred": priority_counts["defer"],
            "actionRequired": priority_counts["critical"] + priority_counts["high"],
        }

    async def _execute_action(
        self,
        action: str,
        email_ids: list[str],
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute action on emails using Gmail connector."""
        results = []
        for email_id in email_ids:
            try:
                result = await self._perform_action(action, email_id, params)
                results.append(
                    {
                        "emailId": email_id,
                        "success": True,
                        "result": result,
                    }
                )

                # Record action for learning
                # Note: We pass email=None since we only have a dict representation,
                # not an EmailMessage object. The method handles None gracefully.
                if self.prioritizer:
                    await self.prioritizer.record_user_action(
                        email_id=email_id,
                        action=action,
                        email=None,
                    )

            except (
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                RuntimeError,
                OSError,
                ConnectionError,
            ) as e:
                logger.warning("Action %s failed for %s: %s", action, email_id, e)
                results.append(
                    {
                        "emailId": email_id,
                        "success": False,
                        "error": str(e),
                    }
                )
        return results

    def _sanitize_action_params(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Sanitize action-specific parameters based on the action type.

        Enforces length bounds and format validation on parameters that will
        be passed to downstream services (Gmail API, etc.).
        """
        sanitized: dict[str, Any] = {}

        if action == "snooze":
            duration = params.get("duration", "1d")
            if isinstance(duration, str) and duration.strip() in ALLOWED_SNOOZE_DURATIONS:
                sanitized["duration"] = duration.strip()
            else:
                sanitized["duration"] = "1d"  # Safe default

        elif action == "reply":
            body = params.get("body", "")
            sanitized["body"] = _sanitize_string_param(body, MAX_REPLY_BODY_LENGTH)

        elif action == "forward":
            to = params.get("to", "")
            validated_to = _validate_email_address(to)
            sanitized["to"] = validated_to if validated_to else ""

        elif action in ("mark_vip", "block"):
            sender = params.get("sender", "")
            if sender:
                validated_sender = _validate_email_address(sender)
                if validated_sender:
                    sanitized["sender"] = validated_sender

        # For actions without specific params (archive, spam, mark_important, delete),
        # return empty dict to avoid passing through unvalidated data
        return sanitized

    async def _perform_action(
        self,
        action: str,
        email_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform a single action on an email."""
        action_handlers = {
            "archive": self._archive_email,
            "snooze": self._snooze_email,
            "reply": self._create_reply_draft,
            "forward": self._create_forward_draft,
            "spam": self._mark_spam,
            "mark_important": self._mark_important,
            "mark_vip": self._mark_sender_vip,
            "block": self._block_sender,
            "delete": self._delete_email,
        }

        handler = action_handlers.get(action)
        if not handler:
            raise ValueError(f"Unknown action: {action}")

        return await handler(email_id, params)

    async def _archive_email(self, email_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Archive an email via Gmail API."""
        if self.gmail_connector and hasattr(self.gmail_connector, "archive_message"):
            try:
                # hasattr check above confirms archive_message exists at runtime
                archive_fn: Callable[[str], Any] = self.gmail_connector.archive_message
                await archive_fn(email_id)
                logger.info("Archived email %s", email_id)
                return {"archived": True}
            except (OSError, ConnectionError, RuntimeError, AttributeError) as e:
                logger.warning("Gmail archive failed: %s", e)

        # Fallback to demo mode
        logger.info("[Demo] Archiving email %s", email_id)
        return {"archived": True, "demo": True}

    async def _snooze_email(self, email_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Snooze an email."""
        duration = params.get("duration", "1d")
        # Parse duration to snooze until time
        duration_map = {
            "1h": timedelta(hours=1),
            "3h": timedelta(hours=3),
            "1d": timedelta(days=1),
            "3d": timedelta(days=3),
            "1w": timedelta(weeks=1),
        }
        delta = duration_map.get(duration, timedelta(days=1))
        snooze_until = datetime.utcnow() + delta

        if self.gmail_connector and hasattr(self.gmail_connector, "snooze_message"):
            try:
                await self.gmail_connector.snooze_message(email_id, snooze_until)
                logger.info("Snoozed email %s until %s", email_id, snooze_until)
                return {"snoozed": True, "until": snooze_until.isoformat()}
            except (OSError, ConnectionError, RuntimeError, AttributeError) as e:
                logger.warning("Gmail snooze failed: %s", e)

        logger.info("[Demo] Snoozing email %s for %s", email_id, duration)
        return {"snoozed": True, "until": snooze_until.isoformat(), "demo": True}

    async def _create_reply_draft(self, email_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Create a reply draft."""
        body = params.get("body", "")

        if self.gmail_connector and hasattr(self.gmail_connector, "create_draft"):
            try:
                draft_id = await self.gmail_connector.create_draft(
                    in_reply_to=email_id,
                    body=body,
                )
                logger.info("Created reply draft for %s", email_id)
                return {"draftId": draft_id}
            except (OSError, ConnectionError, RuntimeError, AttributeError) as e:
                logger.warning("Gmail draft creation failed: %s", e)

        logger.info("[Demo] Creating reply draft for %s", email_id)
        return {"draftId": f"draft_{email_id}", "demo": True}

    async def _create_forward_draft(self, email_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Create a forward draft."""
        to = params.get("to", "")

        if self.gmail_connector and hasattr(self.gmail_connector, "create_forward_draft"):
            try:
                draft_id = await self.gmail_connector.create_forward_draft(
                    message_id=email_id,
                    to=to,
                )
                logger.info("Created forward draft for %s", email_id)
                return {"draftId": draft_id}
            except (OSError, ConnectionError, RuntimeError, AttributeError) as e:
                logger.warning("Gmail forward draft failed: %s", e)

        logger.info("[Demo] Creating forward draft for %s", email_id)
        return {"draftId": f"draft_fwd_{email_id}", "demo": True}

    async def _mark_spam(self, email_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Mark email as spam."""
        if self.gmail_connector and hasattr(self.gmail_connector, "mark_spam"):
            try:
                await self.gmail_connector.mark_spam(email_id)
                logger.info("Marked %s as spam", email_id)
                return {"spam": True}
            except (OSError, ConnectionError, RuntimeError, AttributeError) as e:
                logger.warning("Gmail mark spam failed: %s", e)

        logger.info("[Demo] Marking %s as spam", email_id)
        return {"spam": True, "demo": True}

    async def _mark_important(self, email_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Mark email as important."""
        if self.gmail_connector and hasattr(self.gmail_connector, "modify_labels"):
            try:
                await self.gmail_connector.modify_labels(
                    email_id,
                    add_labels=["IMPORTANT"],
                )
                logger.info("Marked %s as important", email_id)
                return {"important": True}
            except (OSError, ConnectionError, RuntimeError, AttributeError) as e:
                logger.warning("Gmail modify labels failed: %s", e)

        logger.info("[Demo] Marking %s as important", email_id)
        return {"important": True, "demo": True}

    async def _mark_sender_vip(self, email_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Mark sender as VIP."""
        email_data = _email_cache.get(email_id)
        sender = email_data.get("from") if email_data else params.get("sender")

        if sender and self.prioritizer:
            # Add to VIP list in config
            self.prioritizer.config.vip_addresses.add(sender)
            logger.info("Marked sender %s as VIP", sender)
            return {"vip": True, "sender": sender}

        logger.info("[Demo] Marking sender of %s as VIP", email_id)
        return {"vip": True, "demo": True}

    async def _block_sender(self, email_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Block sender."""
        email_data = _email_cache.get(email_id)
        sender = email_data.get("from") if email_data else params.get("sender")

        if sender and self.prioritizer:
            # Add to auto-archive list
            self.prioritizer.config.auto_archive_senders.add(sender)
            logger.info("Blocked sender %s", sender)
            return {"blocked": True, "sender": sender}

        logger.info("[Demo] Blocking sender of %s", email_id)
        return {"blocked": True, "demo": True}

    async def _delete_email(self, email_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Delete email."""
        if self.gmail_connector and hasattr(self.gmail_connector, "trash_message"):
            try:
                await self.gmail_connector.trash_message(email_id)
                logger.info("Deleted email %s", email_id)
                return {"deleted": True}
            except (OSError, ConnectionError, RuntimeError, AttributeError) as e:
                logger.warning("Gmail delete failed: %s", e)

        logger.info("[Demo] Deleting email %s", email_id)
        return {"deleted": True, "demo": True}

    async def _get_emails_by_filter(self, filter_type: str) -> list[str]:
        """Get email IDs matching filter from cache.

        The filter_type is expected to have been validated against
        ALLOWED_BULK_FILTERS before calling this method.
        """
        if filter_type not in ALLOWED_BULK_FILTERS:
            logger.warning("Unexpected filter_type in _get_emails_by_filter: %s", filter_type)
            return []

        filter_map: dict[str, list[str] | Callable[[dict[str, Any]], bool] | None] = {
            "low": ["low", "defer"],
            "deferred": ["defer"],
            "spam": ["spam"],
            "read": lambda e: not e.get("unread", True),
            "all": None,
        }

        matching_ids = []
        filter_value = filter_map.get(filter_type)

        for email_id, email_data in _email_cache.items():
            if filter_value is None:
                matching_ids.append(email_id)
            elif callable(filter_value):
                if filter_value(email_data):
                    matching_ids.append(email_id)
            elif isinstance(filter_value, list):
                if email_data.get("priority") in filter_value:
                    matching_ids.append(email_id)

        return matching_ids

    async def _get_sender_profile(self, email: str) -> dict[str, Any]:
        """Get sender profile information from SenderHistoryService."""
        if self.sender_history:
            try:
                stats = await self.sender_history.get_sender_stats(
                    user_id="default",
                    sender_email=email,
                )
                if stats:
                    avg_hours = (
                        stats.avg_response_time_minutes / 60.0
                        if stats.avg_response_time_minutes
                        else None
                    )
                    return {
                        "email": email,
                        "name": email.split("@")[0],
                        "isVip": stats.is_vip,
                        "isInternal": False,
                        "responseRate": stats.reply_rate,
                        "avgResponseTime": f"{avg_hours:.1f}h" if avg_hours else "N/A",
                        "totalEmails": stats.total_emails,
                        "lastContact": (
                            stats.last_email_date.strftime("%Y-%m-%d")
                            if stats.last_email_date
                            else "Never"
                        ),
                    }
            except (OSError, ConnectionError, RuntimeError, ValueError, AttributeError) as e:
                logger.warning("Failed to get sender stats: %s", e)

        # Check prioritizer config for VIP status
        is_vip = False
        if self.prioritizer:
            is_vip = email.lower() in {a.lower() for a in self.prioritizer.config.vip_addresses}
            domain = email.split("@")[-1] if "@" in email else ""
            is_vip = is_vip or domain.lower() in {
                d.lower() for d in self.prioritizer.config.vip_domains
            }

        # Return basic profile
        return {
            "email": email,
            "name": email.split("@")[0],
            "isVip": is_vip,
            "isInternal": False,
            "responseRate": 0.0,
            "avgResponseTime": "N/A",
            "totalEmails": 0,
            "lastContact": "Unknown",
        }

    async def _calculate_daily_digest(self) -> dict[str, Any]:
        """Calculate daily digest statistics."""
        # Try to get real stats from sender history (if method exists)
        if self.sender_history and hasattr(self.sender_history, "get_daily_summary"):
            try:
                # hasattr check above confirms get_daily_summary exists at runtime
                get_summary_fn: Callable[..., Any] = getattr(
                    self.sender_history, "get_daily_summary"
                )
                today_stats: dict[str, Any] | None = await get_summary_fn(user_id="default")
                if today_stats:
                    return today_stats
            except (OSError, ConnectionError, RuntimeError, ValueError, AttributeError) as e:
                logger.debug("Daily summary not available: %s", e)

        # Use cached data stats
        emails_in_cache = list(_email_cache.values())
        critical_count = sum(1 for e in emails_in_cache if e.get("priority") == "critical")

        # Compute top senders from cache
        sender_counts: dict[str, int] = {}
        for email in emails_in_cache:
            sender = email.get("from", "unknown")
            sender_counts[sender] = sender_counts.get(sender, 0) + 1

        sender_list: list[dict[str, Any]] = [
            {"name": k, "count": v} for k, v in sender_counts.items()
        ]
        top_senders = sorted(
            sender_list,
            key=lambda x: x["count"],
            reverse=True,
        )[:5]

        # Compute category breakdown
        category_counts: dict[str, int] = {}
        for email in emails_in_cache:
            category = email.get("category", "General")
            category_counts[category] = category_counts.get(category, 0) + 1

        total = len(emails_in_cache) or 1  # Avoid division by zero
        category_breakdown = [
            {
                "category": cat,
                "count": count,
                "percentage": round(count / total * 100),
            }
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])
        ]

        return {
            "emailsReceived": len(emails_in_cache),
            "emailsProcessed": len(emails_in_cache),
            "criticalHandled": critical_count,
            "timeSaved": f"{len(emails_in_cache) * 2} min",  # Estimate 2 min saved per email
            "topSenders": top_senders
            or [
                {"name": "team@company.com", "count": 0},
            ],
            "categoryBreakdown": category_breakdown
            or [
                {"category": "General", "count": 0, "percentage": 100},
            ],
        }

    async def _reprioritize_emails(
        self,
        email_ids: Optional[list[str]],
        force_tier: str | None = None,
    ) -> dict[str, Any]:
        """
        Reprioritize emails using AI.

        Uses batch operations to avoid N+1 query patterns:
        1. Batch fetch emails using gmail_connector.get_messages()
        2. Batch score using prioritizer.score_emails() (loads sender profiles once)
        """
        from aragora.services.email_prioritization import ScoringTier

        if not self.prioritizer:
            return {
                "count": 0,
                "changes": [],
                "error": "Prioritizer not available",
            }

        # Get emails to reprioritize
        if email_ids:
            emails_to_process = [
                (eid, _email_cache.get(eid)) for eid in email_ids if eid in _email_cache
            ]
        else:
            emails_to_process = list(_email_cache.items())

        if not emails_to_process:
            return {"count": 0, "changes": []}

        # Map force_tier string to ScoringTier enum
        tier_map = {
            "tier_1_rules": ScoringTier.TIER_1_RULES,
            "tier_2_lightweight": ScoringTier.TIER_2_LIGHTWEIGHT,
            "tier_3_debate": ScoringTier.TIER_3_DEBATE,
        }
        scoring_tier = tier_map.get(force_tier) if force_tier else None

        changes: list[dict[str, Any]] = []
        processed_count = 0

        # Build map of email_id -> cached_email for quick lookup
        cache_map = {eid: cached for eid, cached in emails_to_process if cached}
        email_ids_to_fetch = list(cache_map.keys())

        if not email_ids_to_fetch:
            return {"count": 0, "changes": []}

        # Batch fetch emails from Gmail if connector available
        # This reduces N individual get_message() calls to 1 batch call
        email_messages = []
        if self.gmail_connector:
            try:
                email_messages = await self.gmail_connector.get_messages(email_ids_to_fetch)
            except (OSError, ConnectionError, RuntimeError, AttributeError) as e:
                logger.warning("Batch email fetch failed: %s", e)
                # Fall back to individual fetches on batch failure
                for eid in email_ids_to_fetch:
                    try:
                        msg = await self.gmail_connector.get_message(eid)
                        if msg:
                            email_messages.append(msg)
                    except (OSError, ConnectionError, RuntimeError, AttributeError) as fetch_err:
                        logger.debug("Could not fetch email %s: %s", eid, fetch_err)

        if not email_messages:
            # No Gmail connector or all fetches failed
            return {
                "count": len(email_ids_to_fetch),
                "changes": [],
                "tier_used": force_tier or "auto",
            }

        # Batch score all emails at once
        # This loads sender profiles in bulk (1-2 queries instead of N)
        try:
            results = await self.prioritizer.score_emails(email_messages, force_tier=scoring_tier)
        except (ValueError, RuntimeError, OSError, ConnectionError, AttributeError) as e:
            logger.warning("Batch scoring failed: %s", e)
            return {
                "count": 0,
                "changes": [],
                "error": f"Batch scoring failed: {e}",
            }

        # Process results and update cache
        for email_msg, result in zip(email_messages, results):
            email_id = email_msg.id
            cached_email = cache_map.get(email_id)
            if not cached_email:
                processed_count += 1
                continue

            old_priority = cached_email.get("priority", "medium")
            old_confidence = cached_email.get("confidence", 0.5)

            new_priority = result.priority.name.lower()
            new_confidence = result.confidence

            # Update cache (get, modify, set pattern for TTL cache)
            updated_email = dict(cached_email)
            updated_email.update(
                {
                    "priority": new_priority,
                    "confidence": new_confidence,
                    "reasoning": result.rationale,
                    "tier_used": result.tier_used.value,
                    "scores": {
                        "sender": result.sender_score,
                        "urgency": result.content_urgency_score,
                        "context": result.context_relevance_score,
                        "time_sensitivity": result.time_sensitivity_score,
                    },
                    "suggested_labels": result.suggested_labels,
                    "auto_archive": result.auto_archive,
                }
            )
            _email_cache.set(email_id, updated_email)

            # Track if priority changed
            if old_priority != new_priority:
                changes.append(
                    {
                        "email_id": email_id,
                        "old_priority": old_priority,
                        "new_priority": new_priority,
                        "old_confidence": old_confidence,
                        "new_confidence": new_confidence,
                        "tier_used": result.tier_used.value,
                    }
                )

            processed_count += 1

        return {
            "count": processed_count,
            "changes": changes,
            "tier_used": force_tier or "auto",
        }


def register_routes(app: web.Application) -> None:
    """Register inbox command center routes."""
    handler = InboxCommandHandler()

    # Main inbox endpoints
    app.router.add_get("/api/inbox/command", handler.handle_get_inbox)
    app.router.add_post("/api/inbox/actions", handler.handle_quick_action)
    app.router.add_post("/api/inbox/bulk-actions", handler.handle_bulk_action)
    app.router.add_get("/api/inbox/sender-profile", handler.handle_get_sender_profile)
    app.router.add_get("/api/inbox/daily-digest", handler.handle_get_daily_digest)
    app.router.add_post("/api/inbox/reprioritize", handler.handle_reprioritize)

    # API v1 endpoints
    app.router.add_get("/api/v1/inbox/command", handler.handle_get_inbox)
    app.router.add_post("/api/v1/inbox/actions", handler.handle_quick_action)
    app.router.add_post("/api/v1/inbox/bulk-actions", handler.handle_bulk_action)
    app.router.add_get("/api/v1/inbox/sender-profile", handler.handle_get_sender_profile)
    app.router.add_get("/api/v1/inbox/daily-digest", handler.handle_get_daily_digest)
    app.router.add_post("/api/v1/inbox/reprioritize", handler.handle_reprioritize)

    # Aliases for backward compatibility
    app.router.add_get("/api/email/daily-digest", handler.handle_get_daily_digest)
    app.router.add_get("/api/email/sender-profile", handler.handle_get_sender_profile)

    logger.info("Registered inbox command center routes")
