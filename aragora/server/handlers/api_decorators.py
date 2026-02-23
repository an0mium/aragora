"""
API-specific decorators for HTTP handlers.

This module provides decorators for API endpoint definition and validation:
- api_endpoint: Attach API metadata to handler methods
- rate_limit: Async-friendly rate limiting wrapper
- validate_body: JSON request body validation
- require_quota: Organization debate quota enforcement

These decorators complement the auth/feature decorators in utils/decorators.py.

Usage:
    from aragora.server.handlers.api_decorators import (
        api_endpoint,
        rate_limit,
        validate_body,
        require_quota,
    )

    class MyHandler(BaseHandler):
        @api_endpoint(method="POST", path="/api/debates", summary="Create debate")
        @rate_limit(requests_per_minute=30)
        @validate_body(["task", "agents"])
        @require_quota()
        def _create_debate(self, handler, user):
            ...
"""

from __future__ import annotations

import inspect
import json
import logging
from functools import wraps
from typing import Any
from collections.abc import Callable

from aragora.server.handlers.utils.responses import error_response, json_response
from aragora.server.middleware.rate_limit.decorators import rate_limit as _rate_limit

logger = logging.getLogger(__name__)


def extract_user_from_request(handler, user_store):
    """Proxy extract_user_from_request for patching in tests without caching import."""
    from aragora.billing import jwt_auth

    return jwt_auth.extract_user_from_request(handler, user_store)


def api_endpoint(
    *,
    method: str,
    path: str,
    summary: str = "",
    description: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach API metadata to a handler method.

    This decorator attaches metadata that can be used for:
    - OpenAPI documentation generation
    - Route registration
    - API discovery

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        path: API path pattern (e.g., "/api/debates/{id}")
        summary: Short summary for documentation
        description: Detailed description for documentation

    Returns:
        Decorator that attaches _api_metadata attribute to the function

    Example:
        @api_endpoint(
            method="POST",
            path="/api/debates",
            summary="Create a new debate",
            description="Creates a new multi-agent debate with the specified task and agents."
        )
        def _create_debate(self, handler, user):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
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


def rate_limit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Async-friendly wrapper around middleware rate limiting.

    This decorator wraps the middleware rate_limit decorator to properly
    handle both sync and async handler methods.

    Args:
        *args: Positional arguments passed to middleware rate_limit
        **kwargs: Keyword arguments passed to middleware rate_limit
            - requests_per_minute: Max requests per minute
            - requests_per_second: Max requests per second
            - burst_size: Allow bursts up to this size

    Returns:
        Decorator that applies rate limiting

    Example:
        @rate_limit(requests_per_minute=30)
        def _create_debate(self, handler, user):
            ...

        @rate_limit(requests_per_second=5, burst_size=10)
        async def _stream_results(self, handler):
            ...
    """
    decorator = _rate_limit(*args, **kwargs)

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        decorated: Callable[..., Any] = decorator(func)
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> Any:
                result = decorated(*wrapper_args, **wrapper_kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

            return async_wrapper
        return decorated

    return wrapper


def validate_body(required_fields: list[str]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Validate JSON request body has required fields for async handlers.

    This decorator ensures the request body is valid JSON and contains
    all required fields before the handler method is called.

    Args:
        required_fields: List of field names that must be present in the body

    Returns:
        Decorator that validates request body

    Example:
        @validate_body(["task", "agents"])
        async def _create_debate(self, request, user):
            body = await request.json()
            # body is guaranteed to have 'task' and 'agents'
            ...

        @validate_body(["name"])
        def _create_agent(self, request, user):
            body = request.json()
            # body is guaranteed to have 'name'
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    body = await request.json()
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.debug("JSON parse error in request body: %s", e)
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
        def sync_wrapper(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                body = request.json() if callable(getattr(request, "json", None)) else None
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug("JSON parse error in request body: %s", e)
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


def require_quota(debate_count: int = 1) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
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

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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
                                    "Quota exceeded for org %s: %s/%s",
                                    user_ctx.org_id,
                                    org.debates_used_this_month,
                                    org.limits.debates_per_month,
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
                                    "Quota insufficient for org %s: requested %s, remaining %s",
                                    user_ctx.org_id,
                                    debate_count,
                                    remaining,
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

                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    OSError,
                    RuntimeError,
                ) as e:
                    # Log but don't block on quota check failure
                    logger.warning("Quota check failed for org %s: %s", user_ctx.org_id, e)

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
                                "Incremented usage for org %s by %s", user_ctx.org_id, debate_count
                            )
                    except (
                        ValueError,
                        TypeError,
                        KeyError,
                        AttributeError,
                        OSError,
                        RuntimeError,
                    ) as e:
                        logger.warning("Usage increment failed for org %s: %s", user_ctx.org_id, e)

            return result

        return wrapper

    return decorator


__all__ = [
    "api_endpoint",
    "rate_limit",
    "validate_body",
    "require_quota",
]
