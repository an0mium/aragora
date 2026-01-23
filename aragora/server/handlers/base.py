"""
Base handler utilities for modular endpoint handlers.

Provides common response formatting, error handling, and utilities
shared across all endpoint modules.

Note: Some utilities have been extracted to handlers/utils/ for better
organization. They are re-exported here for backwards compatibility.

Authentication Requirements
---------------------------
Aragora uses header-based authentication (Bearer tokens in Authorization header).
This approach is inherently immune to CSRF attacks because:

1. Tokens are sent via Authorization header, not cookies
2. JavaScript must explicitly set the header (not automatic like cookies)
3. Cross-origin requests cannot access the header

Endpoint Authentication Patterns:
- Public endpoints: No authentication required (e.g., /api/plans, /api/health)
- Protected endpoints: Require valid Bearer token
- Write operations: Always require authentication

Handler classes should use `extract_user_from_request()` for authentication:

    from aragora.billing.jwt_auth import extract_user_from_request

    def _handle_protected(self, handler) -> HandlerResult:
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)
        # ... proceed with authenticated logic

Rate limiting is applied via the @rate_limit decorator from utils.rate_limit.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
from functools import wraps
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, TypedDict, Union

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.billing.auth.context import UserAuthContext
from aragora.protocols import AgentRating, HTTPRequestHandler

if TYPE_CHECKING:
    from pathlib import Path

    from aragora.debate.calibration import CalibrationTracker
    from aragora.debate.embeddings import DebateEmbeddingsDatabase
    from aragora.evidence.collector import EvidenceCollector
    from aragora.evidence.store import EvidenceStore
    from aragora.insights.moment_detector import MomentDetector
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.store import CritiqueStore
    from aragora.ranking.elo import EloSystem
    from aragora.server.storage import DebateStorage
    from aragora.server.stream.ws_manager import WebSocketManager
    from aragora.storage.documents import DocumentStore
    from aragora.storage.webhooks import WebhookStore
    from aragora.users.store import UserStore
    from aragora.billing.usage import UsageTracker


class ServerContext(TypedDict, total=False):
    """Type definition for server context passed to handlers.

    All fields are optional (total=False) since not all handlers need all resources.
    Handlers should use ctx.get("key") to safely access optional fields.

    Core Resources:
        storage: Main debate storage
        user_store: User authentication and profile storage
        elo_system: Agent ELO rating system
        nomic_dir: Path to Nomic session directory

    Memory Systems:
        continuum_memory: Cross-debate memory system
        critique_store: Critique persistence

    Analytics & Monitoring:
        calibration_tracker: Prediction calibration tracking
        moment_detector: Significant moment detection
        usage_tracker: API usage tracking

    Feature Stores:
        document_store: Document persistence
        evidence_store: Evidence snippet storage
        evidence_collector: Evidence collection service
        webhook_store: Webhook configuration storage
        audio_store: Audio file storage

    Event & Communication:
        event_emitter: Event emission for pub/sub
        ws_manager: WebSocket connection manager
        connectors: External service connectors

    Database Paths:
        analytics_db: Path to analytics database
        debate_embeddings: Debate embedding database
    """

    # Core Resources
    storage: "DebateStorage"
    user_store: "UserStore"
    elo_system: "EloSystem"
    nomic_dir: "Path"

    # Memory Systems
    continuum_memory: "ContinuumMemory"
    critique_store: "CritiqueStore"

    # Analytics & Monitoring
    calibration_tracker: "CalibrationTracker"
    moment_detector: "MomentDetector"
    usage_tracker: "UsageTracker"

    # Feature Stores
    document_store: "DocumentStore"
    evidence_store: "EvidenceStore"
    evidence_collector: "EvidenceCollector"
    webhook_store: "WebhookStore"
    audio_store: Any  # AudioStore type if available

    # Event & Communication
    event_emitter: Any  # EventEmitter type if available
    ws_manager: "WebSocketManager"
    connectors: dict[str, Any]  # Service connectors

    # Database Paths
    analytics_db: str
    debate_embeddings: "DebateEmbeddingsDatabase"


# Import from extracted utility modules (re-exported for backwards compatibility)
from aragora.server.errors import safe_error_message
from aragora.server.handlers.admin.cache import (
    CACHE_INVALIDATION_MAP,
    BoundedTTLCache,
    _cache,
    async_ttl_cache,
    clear_cache,
    get_cache_stats,
    get_handler_cache,
    invalidate_agent_cache,
    invalidate_cache,
    invalidate_debate_cache,
    invalidate_leaderboard_cache,
    invalidate_on_event,
    ttl_cache,
)
from aragora.server.handlers.utils.database import (
    get_db_connection,
    table_exists,
)
from aragora.server.handlers.utils.decorators import (
    PERMISSION_MATRIX,
    auto_error_response,
    deprecated_endpoint,
    generate_trace_id,
    handle_errors,
    has_permission,
    log_request,
    require_auth,
    require_feature,
    require_permission,
    require_storage,
    require_user_auth,
    safe_fetch,
    validate_params,
    with_error_recovery,
)
from aragora.server.handlers.utils.params import (
    get_bool_param,
    get_bounded_float_param,
    get_bounded_string_param,
    get_clamped_int_param,
    get_float_param,
    get_int_param,
    get_string_param,
    parse_query_params,
)
from aragora.server.handlers.utils.routing import (
    PathMatcher,
    RouteDispatcher,
)
from aragora.server.handlers.utils.safe_data import (
    safe_get,
    safe_get_nested,
    safe_json_parse,
)
from aragora.server.validation import (
    SAFE_AGENT_PATTERN,
    SAFE_ID_PATTERN,
    SAFE_SLUG_PATTERN,
    validate_agent_name,
    validate_debate_id,
    validate_path_segment,
    validate_string,
)

# Rate limiting is available from aragora.server.middleware.rate_limit
# or aragora.server.rate_limit for backward compatibility

# Re-export DB_TIMEOUT_SECONDS for backwards compatibility
__all__ = [
    "DB_TIMEOUT_SECONDS",
    "require_auth",
    "require_user_auth",
    "require_quota",
    "require_storage",
    "require_feature",
    "require_permission",
    "api_endpoint",
    "rate_limit",
    "validate_body",
    "has_permission",
    "PERMISSION_MATRIX",
    "deprecated_endpoint",
    "error_response",
    "json_response",
    "success_response",
    "HandlerResult",
    "handle_errors",
    "auto_error_response",
    "log_request",
    "ttl_cache",
    "safe_error_message",
    "safe_error_response",
    "async_ttl_cache",
    "BoundedTTLCache",
    "_cache",
    "clear_cache",
    "get_cache_stats",
    "CACHE_INVALIDATION_MAP",
    "invalidate_cache",
    "invalidate_on_event",
    "invalidate_leaderboard_cache",
    "invalidate_agent_cache",
    "invalidate_debate_cache",
    "PathMatcher",
    "RouteDispatcher",
    "safe_fetch",
    "with_error_recovery",
    "get_db_connection",
    "table_exists",
    "safe_get",
    "safe_get_nested",
    "safe_json_parse",
    "get_host_header",
    "get_agent_name",
    "agent_to_dict",
    "validate_params",
    "SAFE_ID_PATTERN",
    "SAFE_SLUG_PATTERN",
    "SAFE_AGENT_PATTERN",
    "validate_agent_name",
    "validate_debate_id",
    "validate_string",
    "feature_unavailable_response",
    # Parameter extraction helpers
    "get_int_param",
    "get_float_param",
    "get_bool_param",
    "get_string_param",
    "get_clamped_int_param",
    "get_bounded_float_param",
    "get_bounded_string_param",
    "parse_query_params",
    # Handler mixins
    "PaginatedHandlerMixin",
    "CachedHandlerMixin",
    "AuthenticatedHandlerMixin",
    "BaseHandler",
    # Note: validate_json_content_type and read_json_body_validated are BaseHandler methods
]


def feature_unavailable_response(
    feature_id: str,
    message: Optional[str] = None,
) -> "HandlerResult":
    """
    Create a standardized response for unavailable features.

    This should be used by all handlers when an optional feature is missing.
    Provides consistent error format with helpful installation hints.

    Args:
        feature_id: The feature identifier (e.g., "pulse", "genesis")
        message: Optional custom message (defaults to feature description)

    Returns:
        HandlerResult with 503 status and helpful information

    Example:
        if not self._pulse_manager:
            return feature_unavailable_response("pulse")
    """
    # Import here to avoid circular imports
    from aragora.server.handlers.features import (
        feature_unavailable_response as _feature_unavailable,
    )

    return _feature_unavailable(feature_id, message)


logger = logging.getLogger(__name__)


# =============================================================================
# Database Connection Helper (imported from utils/database.py)
# =============================================================================
# get_db_connection and table_exists are now in aragora.server.handlers.utils.database
# and re-exported above for backwards compatibility


# =============================================================================
# Dict Access Helpers (imported from utils/safe_data.py)
# =============================================================================
# safe_get, safe_get_nested, safe_json_parse are now in
# aragora.server.handlers.utils.safe_data and re-exported above


# Default host from environment (used when Host header is missing)
_DEFAULT_HOST = os.environ.get("ARAGORA_DEFAULT_HOST", "localhost:8080")


def get_host_header(handler: Optional[HTTPRequestHandler], default: str | None = None) -> str:
    """Extract Host header from request handler.

    Args:
        handler: HTTP request handler with headers attribute
        default: Default value if handler is None or Host header missing.
                 If None, uses ARAGORA_DEFAULT_HOST env var or 'localhost:8080'.

    Returns:
        Host header value or default

    Example:
        # Before (repeated 5+ times):
        host = handler.headers.get('Host', 'localhost:8080') if handler else 'localhost:8080'

        # After:
        host = get_host_header(handler)
    """
    if default is None:
        default = _DEFAULT_HOST
    if handler is None:
        return default
    return handler.headers.get("Host", default) if hasattr(handler, "headers") else default


def get_agent_name(agent: Union[dict, AgentRating, Any, None]) -> Optional[str]:
    """Extract agent name from dict or object.

    Handles the common pattern where agent data might be either
    a dict with 'name'/'agent_name' key or an object with name attribute.

    Args:
        agent: Dict or AgentRating-like object containing agent name

    Returns:
        Agent name string or None if not found

    Example:
        # Before (repeated 4+ times):
        name = agent.get("name") if isinstance(agent, dict) else getattr(agent, "name", None)

        # After:
        name = get_agent_name(agent)
    """
    if agent is None:
        return None
    if isinstance(agent, dict):
        return agent.get("agent_name") or agent.get("name")
    return getattr(agent, "agent_name", None) or getattr(agent, "name", None)


def agent_to_dict(agent: Union[dict, AgentRating, Any, None], include_name: bool = True) -> dict:
    """Convert agent object or dict to standardized dict with ELO fields.

    Handles the common pattern where agent data might be either a dict
    or an AgentRating object, extracting standard fields with safe defaults.

    Args:
        agent: Dict or object containing agent data
        include_name: Whether to include name/agent_name fields (default: True)

    Returns:
        Dict with standardized ELO-related fields

    Example:
        # Before (repeated 40+ times across handlers):
        agent_dict = {
            "name": getattr(agent, "name", "unknown"),
            "elo": getattr(agent, "elo", 1500),
            "wins": getattr(agent, "wins", 0),
            "losses": getattr(agent, "losses", 0),
            ...
        }

        # After:
        agent_dict = agent_to_dict(agent)
    """
    if agent is None:
        return {}

    if isinstance(agent, dict):
        return agent.copy()

    # Extract standard ELO fields from object
    name = get_agent_name(agent) or "unknown"
    result = {
        "elo": getattr(agent, "elo", 1500),
        "wins": getattr(agent, "wins", 0),
        "losses": getattr(agent, "losses", 0),
        "draws": getattr(agent, "draws", 0),
        "win_rate": getattr(agent, "win_rate", 0.0),
        "games": getattr(agent, "games_played", getattr(agent, "games", 0)),
        "matches": getattr(agent, "matches", 0),
    }

    if include_name:
        result["name"] = name
        result["agent_name"] = name

    return result


# =============================================================================
# Response Builders (imported from utils/responses.py)
# =============================================================================
# Core response utilities are now in aragora.server.handlers.utils.responses
# and re-exported here for backwards compatibility

from aragora.server.handlers.utils.responses import (
    HandlerResult,
    error_response,
    json_response,
    success_response,
)


def safe_error_response(
    exception: Exception,
    context: str,
    status: int = 500,
    handler: Optional[HTTPRequestHandler] = None,
) -> HandlerResult:
    """Create an error response with sanitized message.

    Logs full exception details server-side for debugging, but returns only
    a generic, safe message to the client. This prevents information disclosure
    of internal paths, stack traces, and sensitive configuration.

    Args:
        exception: The exception that occurred
        context: Context for logging (e.g., "debate creation", "agent lookup")
        status: HTTP status code (default: 500)
        handler: Optional request handler for extracting trace_id

    Returns:
        HandlerResult with sanitized error message

    Example:
        try:
            result = do_something()
        except Exception as e:
            return safe_error_response(e, "debate creation", 500, handler)
    """
    from aragora.server.errors import ErrorFormatter

    # Generate or extract trace ID
    trace_id = None
    if handler is not None:
        # Try to get existing trace_id from handler
        if hasattr(handler, "trace_id"):
            trace_id = handler.trace_id
        elif hasattr(handler, "headers") and handler.headers:
            trace_id = handler.headers.get("X-Request-ID") or handler.headers.get("X-Trace-ID")
    if not trace_id:
        trace_id = generate_trace_id()

    # Format error with sanitization (logs full details server-side)
    error_dict = ErrorFormatter.format_server_error(exception, context=context, trace_id=trace_id)

    return json_response(error_dict, status=status)


# Note: Exception handling, tracing, decorators, and RBAC are all imported from
# utils/decorators.py. See that module for: generate_trace_id, map_exception_to_status,
# validate_params, handle_errors, auto_error_response, log_request, PERMISSION_MATRIX,
# has_permission, require_permission, require_user_auth, require_auth, require_storage,
# require_feature, safe_fetch, with_error_recovery


def require_quota(debate_count: int = 1) -> Callable[[Callable], Callable]:
    """
    Decorator that enforces organization debate quota limits.

    Checks if the user's organization has remaining debate quota before allowing
    the operation. If at limit, returns 429 with upgrade information.
    On successful operation, increments the organization's usage counter.

    This decorator requires authentication - use after @require_user_auth or
    with handlers that already have user context.

    Args:
        debate_count: Number of debates this operation will create (default: 1).
                     For batch operations, pass the batch size.

    Returns:
        Decorator that enforces quota and increments usage on success.

    Usage:
        @require_quota()
        def _create_debate(self, handler, user: UserAuthContext) -> HandlerResult:
            # User's org quota is verified before this runs
            ...

        @require_quota(debate_count=10)
        def _submit_batch(self, handler, user: UserAuthContext, batch_size: int) -> HandlerResult:
            # Checks if org has capacity for 10 debates
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from aragora.billing.jwt_auth import extract_user_from_request

            # Extract handler from kwargs or args
            handler = kwargs.get("handler")
            if handler is None and args:
                for arg in args:
                    if hasattr(arg, "headers"):
                        handler = arg
                        break

            # Get user context - may already be in kwargs from @require_user_auth
            user_ctx = kwargs.get("user")

            if user_ctx is None:
                # Authenticate if not already done
                if handler is None:
                    logger.warning("require_quota: No handler provided")
                    return error_response("Authentication required", 401)

                user_store = None
                if hasattr(handler, "user_store"):
                    user_store = handler.user_store
                elif hasattr(handler.__class__, "user_store"):
                    user_store = handler.__class__.user_store

                user_ctx = extract_user_from_request(handler, user_store)

                if not user_ctx.is_authenticated:
                    error_msg = user_ctx.error_reason or "Authentication required"
                    return error_response(error_msg, 401)

                kwargs["user"] = user_ctx

            # Check organization quota
            if user_ctx.org_id:
                try:
                    # Get organization from user store
                    user_store = None
                    if handler and hasattr(handler, "user_store"):
                        user_store = handler.user_store
                    elif handler and hasattr(handler.__class__, "user_store"):
                        user_store = handler.__class__.user_store

                    if user_store and hasattr(user_store, "get_organization_by_id"):
                        org = user_store.get_organization_by_id(user_ctx.org_id)
                        if org:
                            # Check if at limit
                            if org.is_at_limit:
                                logger.info(
                                    f"Quota exceeded for org {user_ctx.org_id}: "
                                    f"{org.debates_used_this_month}/{org.limits.debates_per_month}"
                                )
                                return json_response(
                                    {
                                        "error": "Monthly debate quota exceeded",
                                        "code": "quota_exceeded",
                                        "limit": org.limits.debates_per_month,
                                        "used": org.debates_used_this_month,
                                        "remaining": 0,
                                        "tier": org.tier.value,
                                        "upgrade_url": "/pricing",
                                        "message": f"Your {org.tier.value} plan allows {org.limits.debates_per_month} debates per month. Upgrade to increase your limit.",
                                    },
                                    status=429,
                                )

                            # Check if this operation would exceed quota
                            if (
                                org.debates_used_this_month + debate_count
                                > org.limits.debates_per_month
                            ):
                                remaining = (
                                    org.limits.debates_per_month - org.debates_used_this_month
                                )
                                logger.info(
                                    f"Quota insufficient for org {user_ctx.org_id}: "
                                    f"requested {debate_count}, remaining {remaining}"
                                )
                                return json_response(
                                    {
                                        "error": f"Insufficient quota: requested {debate_count} debates but only {remaining} remaining",
                                        "code": "quota_insufficient",
                                        "limit": org.limits.debates_per_month,
                                        "used": org.debates_used_this_month,
                                        "remaining": remaining,
                                        "requested": debate_count,
                                        "tier": org.tier.value,
                                        "upgrade_url": "/pricing",
                                    },
                                    status=429,
                                )

                except Exception as e:
                    # Log but don't block on quota check failure
                    logger.warning(f"Quota check failed for org {user_ctx.org_id}: {e}")

            # Execute the handler
            result = func(*args, **kwargs)

            # Increment usage on success (status < 400)
            if user_ctx.org_id:
                status_code = getattr(result, "status_code", 200) if result else 200
                if status_code < 400:
                    try:
                        user_store = None
                        if handler and hasattr(handler, "user_store"):
                            user_store = handler.user_store
                        elif handler and hasattr(handler.__class__, "user_store"):
                            user_store = handler.__class__.user_store

                        if user_store and hasattr(user_store, "increment_usage"):
                            user_store.increment_usage(user_ctx.org_id, debate_count)
                            logger.debug(
                                f"Incremented usage for org {user_ctx.org_id} by {debate_count}"
                            )
                    except Exception as e:
                        logger.warning(f"Usage increment failed for org {user_ctx.org_id}: {e}")

            return result

        return wrapper

    return decorator


def api_endpoint(
    *,
    method: str,
    path: str,
    summary: str = "",
    description: str = "",
) -> Callable[[Callable], Callable]:
    """Attach API metadata to a handler method."""

    def decorator(func: Callable) -> Callable:
        setattr(
            func,
            "_api_metadata",
            {
                "method": method,
                "path": path,
                "summary": summary,
                "description": description,
            },
        )
        return func

    return decorator


def rate_limit(*args, **kwargs) -> Callable[[Callable], Callable]:
    """Async-friendly wrapper around middleware rate limiting."""
    from aragora.server.middleware.rate_limit.decorators import rate_limit as _rate_limit

    decorator = _rate_limit(*args, **kwargs)

    def wrapper(func: Callable) -> Callable:
        decorated = decorator(func)
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*wrapper_args, **wrapper_kwargs):
                result = decorated(*wrapper_args, **wrapper_kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

            return async_wrapper
        return decorated

    return wrapper


def validate_body(required_fields: list[str]) -> Callable[[Callable], Callable]:
    """Validate JSON request body has required fields for async handlers."""

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(self, request, *args, **kwargs):
                try:
                    body = await request.json()
                except Exception:
                    if hasattr(self, "error_response"):
                        return self.error_response("Invalid JSON body", status=400)
                    return error_response("Invalid JSON body", status=400)

                missing = [field for field in required_fields if field not in body]
                if missing:
                    message = f"Missing required fields: {', '.join(missing)}"
                    if hasattr(self, "error_response"):
                        return self.error_response(message, status=400)
                    return error_response(message, status=400)

                return await func(self, request, *args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(self, request, *args, **kwargs):
            try:
                body = request.json() if callable(getattr(request, "json", None)) else None
            except Exception:
                if hasattr(self, "error_response"):
                    return self.error_response("Invalid JSON body", status=400)
                return error_response("Invalid JSON body", status=400)

            missing = [field for field in required_fields if field not in (body or {})]
            if missing:
                message = f"Missing required fields: {', '.join(missing)}"
                if hasattr(self, "error_response"):
                    return self.error_response(message, status=400)
                return error_response(message, status=400)

            return func(self, request, *args, **kwargs)

        return sync_wrapper

    return decorator


# =============================================================================
# Handler Mixins
# =============================================================================
# These mixins provide reusable patterns for common handler operations.
# Handlers can inherit from these in addition to BaseHandler to get
# standardized implementations of common operations.


class PaginatedHandlerMixin:
    """Mixin for standardized pagination handling.

    Provides consistent limit/offset extraction with validation and defaults.

    Usage:
        class MyHandler(BaseHandler, PaginatedHandlerMixin):
            def handle(self, path, query_params, handler):
                limit, offset = self.get_pagination(query_params)
                results = self.get_data(limit=limit, offset=offset)
                return self.paginated_response(results, total=100, limit=limit, offset=offset)
    """

    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100
    DEFAULT_OFFSET = 0

    def get_pagination(
        self,
        query_params: dict,
        default_limit: int | None = None,
        max_limit: int | None = None,
    ) -> tuple[int, int]:
        """Extract and validate pagination parameters.

        Args:
            query_params: Query parameters dict
            default_limit: Override default limit (default: DEFAULT_LIMIT)
            max_limit: Override max limit (default: MAX_LIMIT)

        Returns:
            Tuple of (limit, offset) with validated bounds
        """
        default_limit = default_limit or self.DEFAULT_LIMIT
        max_limit = max_limit or self.MAX_LIMIT

        limit = get_int_param(query_params, "limit", default_limit)
        offset = get_int_param(query_params, "offset", self.DEFAULT_OFFSET)

        # Clamp values
        limit = max(1, min(limit, max_limit))
        offset = max(0, offset)

        return limit, offset

    def paginated_response(
        self,
        items: list,
        total: int,
        limit: int,
        offset: int,
        items_key: str = "items",
    ) -> "HandlerResult":
        """Create a standardized paginated response.

        Args:
            items: List of items for this page
            total: Total count of all items
            limit: Page size used
            offset: Starting offset
            items_key: Key name for items in response (default: "items")

        Returns:
            JSON response with pagination metadata
        """
        return json_response(
            {
                items_key: items,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(items) < total,
            }
        )


class CachedHandlerMixin:
    """Mixin for cached response generation.

    Provides a simple interface for caching handler responses with TTL.

    Usage:
        class MyHandler(BaseHandler, CachedHandlerMixin):
            def _get_data(self, key: str):
                return self.cached_response(
                    cache_key=f"mydata:{key}",
                    ttl_seconds=300,
                    generator=lambda: expensive_computation(key),
                )
    """

    def cached_response(
        self,
        cache_key: str,
        ttl_seconds: float,
        generator: Callable[[], Any],
    ) -> Any:
        """Get or generate a cached response.

        Args:
            cache_key: Unique key for this cached item
            ttl_seconds: How long to cache the result
            generator: Callable that generates the value if not cached

        Returns:
            Cached or freshly generated value
        """
        cache = get_handler_cache()
        hit, cached_value = cache.get(cache_key, ttl_seconds)

        if hit:
            return cached_value

        value = generator()
        cache.set(cache_key, value)
        return value

    async def async_cached_response(
        self,
        cache_key: str,
        ttl_seconds: float,
        generator: Callable[[], Any],
    ) -> Any:
        """Async version of cached_response.

        Args:
            cache_key: Unique key for this cached item
            ttl_seconds: How long to cache the result
            generator: Async callable that generates the value if not cached

        Returns:
            Cached or freshly generated value
        """
        cache = get_handler_cache()
        hit, cached_value = cache.get(cache_key, ttl_seconds)

        if hit:
            return cached_value

        value = await generator()
        cache.set(cache_key, value)
        return value


class AuthenticatedHandlerMixin:
    """Mixin for requiring authenticated access.

    Provides standardized authentication extraction and error handling.

    Usage:
        class MyHandler(BaseHandler, AuthenticatedHandlerMixin):
            def handle_post(self, path, query_params, handler):
                user = self.require_auth(handler)
                if isinstance(user, tuple):  # Error response
                    return user
                # user is now the authenticated context
                return json_response({"user_id": user.user_id})
    """

    def require_auth(self, handler) -> Any:
        """Require authentication and return user context or error.

        Args:
            handler: HTTP request handler with headers

        Returns:
            UserAuthContext if authenticated,
            or HandlerResult with 401 error if not
        """
        # This method is typically overridden or uses BaseHandler's method
        # When used with BaseHandler, call require_auth_or_error instead
        if hasattr(self, "require_auth_or_error"):
            user, err = self.require_auth_or_error(handler)
            if err:
                return err
            return user

        # Fallback implementation
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = getattr(self, "ctx", {}).get("user_store")
        if hasattr(self.__class__, "user_store"):
            user_store = self.__class__.user_store

        user_ctx = extract_user_from_request(handler, user_store)
        if not user_ctx.is_authenticated:
            return error_response("Authentication required", 401)
        return user_ctx


class BaseHandler:
    """
    Base class for endpoint handlers.

    Subclasses implement specific endpoint groups and register
    their routes via the `routes` class attribute.
    """

    ctx: ServerContext
    _current_handler: Any = None
    _current_query_params: dict[str, Any] = None  # type: ignore[assignment]

    def __init__(self, server_context: ServerContext):
        """
        Initialize with server context.

        Args:
            server_context: ServerContext containing shared server resources like
                           storage, elo_system, debate_embeddings, etc.
                           See ServerContext TypedDict for available fields.
        """
        self.ctx = server_context
        self._current_handler = None
        self._current_query_params = {}

    def set_request_context(
        self, handler: Any, query_params: Optional[dict[str, Any]] = None
    ) -> None:
        """Set the current request context for helper methods.

        Call this at the start of request handling to enable helper methods
        like get_query_param(), get_json_body(), json_response(), json_error().

        Args:
            handler: HTTP request handler
            query_params: Parsed query parameters dict
        """
        self._current_handler = handler
        self._current_query_params = query_params or {}

    def get_query_param(
        self,
        name_or_handler: Any,
        name_or_default: Optional[str] = None,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Get a query parameter from the current request.

        Supports two calling patterns for backwards compatibility:
        1. get_query_param(name, default) - uses stored request context
        2. get_query_param(handler, name, default) - extracts from handler

        Args:
            name_or_handler: Either parameter name (str) or HTTP handler
            name_or_default: Either default value or parameter name
            default: Default value (only used in handler pattern)

        Returns:
            Parameter value or default
        """
        # Detect calling pattern
        if isinstance(name_or_handler, str):
            # Pattern 1: get_query_param(name, default)
            name = name_or_handler
            default_value = name_or_default
            if self._current_query_params is None:
                return default_value
            value = self._current_query_params.get(name)
        else:
            # Pattern 2: get_query_param(handler, name, default)
            handler = name_or_handler
            name = name_or_default or ""
            default_value = default
            # Try to extract query params from handler
            query_params = {}
            if hasattr(handler, "path") and "?" in handler.path:
                from urllib.parse import parse_qs, urlparse

                parsed = urlparse(handler.path)
                query_params = parse_qs(parsed.query)
            value = query_params.get(name)

        if isinstance(value, list):
            return value[0] if value else default_value
        return value if value is not None else default_value

    def get_json_body(self) -> Optional[dict[str, Any]]:
        """Get JSON body from the current request.

        Returns:
            Parsed JSON dict, empty dict if no body, None on error
        """
        if self._current_handler is None:
            return None
        return self.read_json_body(self._current_handler)

    def json_response(
        self,
        data: Any,
        status: Union[int, HTTPStatus] = HTTPStatus.OK,
    ) -> HandlerResult:
        """Create a JSON response.

        Args:
            data: Data to serialize as JSON
            status: HTTP status code

        Returns:
            HandlerResult with JSON response
        """
        status_code = status.value if isinstance(status, HTTPStatus) else status
        return json_response(data, status=status_code)

    def success_response(
        self,
        data: Any,
        status: Union[int, HTTPStatus] = HTTPStatus.OK,
    ) -> HandlerResult:
        """Create a standard success response."""
        status_code = status.value if isinstance(status, HTTPStatus) else status
        return json_response(data, status=status_code)

    def error_response(
        self,
        message: str,
        status: Union[int, HTTPStatus] = HTTPStatus.BAD_REQUEST,
    ) -> HandlerResult:
        """Create a standard error response."""
        status_code = status.value if isinstance(status, HTTPStatus) else status
        return error_response(message, status=status_code)

    def json_error(
        self,
        message: str,
        status: Union[int, HTTPStatus] = HTTPStatus.BAD_REQUEST,
    ) -> HandlerResult:
        """Create a JSON error response.

        Args:
            message: Error message
            status: HTTP status code

        Returns:
            HandlerResult with error response
        """
        status_code = status.value if isinstance(status, HTTPStatus) else status
        return error_response(message, status_code)

    def extract_path_param(
        self,
        path: str,
        segment_index: int,
        param_name: str,
        pattern: re.Pattern = None,
    ) -> Tuple[Optional[str], Optional[HandlerResult]]:
        """Extract and validate a path segment parameter.

        Consolidates the common pattern of:
        1. Split path into parts
        2. Check segment exists at index
        3. Validate against pattern
        4. Return error response if invalid

        Args:
            path: URL path to extract from
            segment_index: Index of segment to extract (0-based)
            param_name: Human-readable name for error messages
            pattern: Regex pattern to validate against (default: SAFE_ID_PATTERN)

        Returns:
            Tuple of (value, error_response):
            - (value, None) on success
            - (None, HandlerResult) on failure

        Example:
            # Before:
            parts = path.split("/")
            if len(parts) < 4:
                return error_response("Invalid path", 400)
            domain = parts[3]
            is_valid, err = validate_path_segment(domain, "domain", SAFE_ID_PATTERN)
            if not is_valid:
                return error_response(err, 400)

            # After:
            domain, err = self.extract_path_param(path, 3, "domain")
            if err:
                return err
        """
        pattern = pattern or SAFE_ID_PATTERN
        parts = path.strip("/").split("/")

        if segment_index >= len(parts):
            return None, error_response(f"Missing {param_name} in path", 400)

        value = parts[segment_index]
        if not value:
            return None, error_response(f"Empty {param_name}", 400)

        is_valid, err_msg = validate_path_segment(value, param_name, pattern)
        if not is_valid:
            return None, error_response(err_msg, 400)

        return value, None

    def extract_path_params(
        self,
        path: str,
        param_specs: list[Tuple[int, str, Optional[re.Pattern]]],
    ) -> Tuple[Optional[dict], Optional[HandlerResult]]:
        """Extract and validate multiple path parameters at once.

        Args:
            path: URL path to extract from
            param_specs: List of (segment_index, param_name, pattern) tuples.
                        If pattern is None, SAFE_ID_PATTERN is used.

        Returns:
            Tuple of (params_dict, error_response):
            - ({"name": value, ...}, None) on success
            - (None, HandlerResult) on first failure

        Example:
            # Extract agent_a and agent_b from /api/agents/compare/claude/gpt4
            params, err = self.extract_path_params(path, [
                (3, "agent_a", SAFE_AGENT_PATTERN),
                (4, "agent_b", SAFE_AGENT_PATTERN),
            ])
            if err:
                return err
            # params = {"agent_a": "claude", "agent_b": "gpt4"}
        """
        result = {}
        for segment_index, param_name, pattern in param_specs:
            value, err = self.extract_path_param(path, segment_index, param_name, pattern)
            if err:
                return None, err
            result[param_name] = value
        return result, None

    def get_storage(self) -> Optional["DebateStorage"]:
        """Get debate storage instance."""
        return self.ctx.get("storage")

    def get_elo_system(self) -> Optional["EloSystem"]:
        """Get ELO system instance."""
        # Check class attribute first (set by unified_server), then ctx
        if hasattr(self.__class__, "elo_system") and self.__class__.elo_system is not None:
            return self.__class__.elo_system
        return self.ctx.get("elo_system")

    def get_debate_embeddings(self) -> Optional["DebateEmbeddingsDatabase"]:
        """Get debate embeddings database."""
        return self.ctx.get("debate_embeddings")

    def get_critique_store(self) -> Optional["CritiqueStore"]:
        """Get critique store instance."""
        return self.ctx.get("critique_store")

    def get_nomic_dir(self) -> Optional["Path"]:
        """Get nomic directory path."""
        return self.ctx.get("nomic_dir")

    def get_current_user(self, handler: HTTPRequestHandler) -> Optional[UserAuthContext]:
        """Get authenticated user from request, if any.

        Unlike @require_user_auth decorator which requires authentication,
        this method allows optional authentication - returning None if
        no valid auth is provided. Useful for endpoints that work for
        anonymous users but have enhanced features when authenticated.

        Args:
            handler: HTTP request handler with headers

        Returns:
            UserAuthContext if authenticated, None otherwise

        Example:
            def handle(self, path, query_params, handler):
                user = self.get_current_user(handler)
                if user:
                    # Show personalized content
                    return json_response({"user": user.email, "debates": ...})
                else:
                    # Show public content
                    return json_response({"debates": ...})
        """
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = None
        if hasattr(handler, "user_store"):
            user_store = handler.user_store
        elif hasattr(self.__class__, "user_store"):
            user_store = self.__class__.user_store

        user_ctx = extract_user_from_request(handler, user_store)
        return user_ctx if user_ctx.is_authenticated else None

    def require_auth_or_error(
        self, handler: HTTPRequestHandler
    ) -> Tuple[Optional[UserAuthContext], Optional["HandlerResult"]]:
        """Require authentication and return user or error response.

        Alternative to @require_user_auth decorator for cases where you need
        the user context inline without using a decorator.

        Args:
            handler: HTTP request handler with headers

        Returns:
            Tuple of (UserAuthContext, None) if authenticated,
            or (None, HandlerResult) with 401 error if not

        Example:
            def handle_post(self, path, query_params, handler):
                user, err = self.require_auth_or_error(handler)
                if err:
                    return err
                # user is now guaranteed to be authenticated
                return json_response({"created_by": user.user_id})
        """
        user = self.get_current_user(handler)
        if user is None:
            return None, error_response("Authentication required", 401)
        return user, None

    def require_admin_or_error(
        self, handler: HTTPRequestHandler
    ) -> Tuple[Optional[UserAuthContext], Optional["HandlerResult"]]:
        """Require admin authentication and return user or error response.

        Checks that the user is authenticated and has admin privileges
        (either 'admin' role or 'admin' permission).

        Args:
            handler: HTTP request handler with headers

        Returns:
            Tuple of (UserAuthContext, None) if authenticated as admin,
            or (None, HandlerResult) with 401/403 error if not

        Example:
            def handle_post(self, path, query_params, handler):
                user, err = self.require_admin_or_error(handler)
                if err:
                    return err
                # user is now guaranteed to be an admin
                return json_response({"admin_action": "completed"})
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return None, err

        # Check for admin role or permission
        roles = getattr(user, "roles", []) or []
        permissions = getattr(user, "permissions", []) or []

        is_admin = "admin" in roles or "admin" in permissions or getattr(user, "is_admin", False)

        if not is_admin:
            return None, error_response("Admin access required", 403)

        return user, None

    # === POST Body Parsing Support ===

    # Maximum request body size (10MB default)
    MAX_BODY_SIZE = 10 * 1024 * 1024

    def read_json_body(self, handler, max_size: int = None) -> Optional[dict]:
        """Read and parse JSON body from request handler.

        Args:
            handler: The HTTP request handler with headers and rfile
            max_size: Maximum body size to accept (default: MAX_BODY_SIZE)

        Returns:
            Parsed JSON dict, empty dict for no content, or None for parse errors
        """
        max_size = max_size or self.MAX_BODY_SIZE
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length <= 0:
                return {}
            if content_length > max_size:
                return None  # Body too large
            body = handler.rfile.read(content_length)
            return json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            return None

    def validate_content_length(self, handler, max_size: int = None) -> Optional[int]:
        """Validate Content-Length header.

        Args:
            handler: The HTTP request handler
            max_size: Maximum allowed size (default: MAX_BODY_SIZE)

        Returns:
            Content length if valid, None if invalid
        """
        max_size = max_size or self.MAX_BODY_SIZE
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            return None

        if content_length < 0 or content_length > max_size:
            return None

        return content_length

    def validate_json_content_type(self, handler) -> Optional[HandlerResult]:
        """Validate that Content-Type is application/json for JSON endpoints.

        Args:
            handler: The HTTP request handler with headers

        Returns:
            None if valid, HandlerResult with 415 error if Content-Type is invalid
        """
        if not hasattr(handler, "headers"):
            return error_response("Missing Content-Type header", 415)

        content_type = handler.headers.get("Content-Type", "")
        # Accept application/json with or without charset
        if not content_type:
            # Allow empty Content-Type for backwards compatibility with empty bodies
            content_length = handler.headers.get("Content-Length", "0")
            if content_length == "0" or content_length == 0:
                return None
            return error_response("Content-Type header required for POST with body", 415)

        # Parse media type (ignore parameters like charset)
        media_type = content_type.split(";")[0].strip().lower()
        if media_type not in ("application/json", "text/json"):
            return error_response(
                f"Unsupported Content-Type: {content_type}. Expected application/json", 415
            )

        return None

    def read_json_body_validated(
        self, handler, max_size: int = None
    ) -> Tuple[Optional[dict], Optional[HandlerResult]]:
        """Read and parse JSON body with Content-Type validation.

        Combines Content-Type validation and body parsing into a single call.

        Args:
            handler: The HTTP request handler with headers and rfile
            max_size: Maximum body size to accept (default: MAX_BODY_SIZE)

        Returns:
            Tuple of (parsed_dict, None) on success,
            or (None, HandlerResult) with error response on failure
        """
        # Validate Content-Type
        content_type_error = self.validate_json_content_type(handler)
        if content_type_error:
            return None, content_type_error

        # Read and parse body
        body = self.read_json_body(handler, max_size)
        if body is None:
            return None, error_response("Invalid or too large JSON body", 400)

        return body, None

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """
        Handle a GET request. Override in subclasses.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """
        Handle a POST request. Override in subclasses that support POST.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """
        Handle a DELETE request. Override in subclasses that support DELETE.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None

    def handle_patch(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """
        Handle a PATCH request. Override in subclasses that support PATCH.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None

    def handle_put(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """
        Handle a PUT request. Override in subclasses that support PUT.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None
