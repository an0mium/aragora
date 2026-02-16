"""
Slack integration configuration and lazy singletons.

Module-level constants, environment variables, and lazy-initialized
singleton accessors for audit logging, rate limiting, and workspace management.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from re import Pattern
from collections.abc import Callable

logger = logging.getLogger(__name__)

# --- Lazy import for audit logger (avoid circular imports) ---

_slack_audit: Any = None


def _get_audit_logger() -> Any:
    """Get or create Slack audit logger (lazy initialization)."""
    global _slack_audit
    if _slack_audit is None:
        try:
            from aragora.audit.slack_audit import get_slack_audit_logger

            _slack_audit = get_slack_audit_logger()
        except (ImportError, RuntimeError, OSError) as e:
            logger.debug(f"Slack audit logger not available: {e}")
            _slack_audit = None
    return _slack_audit


# --- Lazy import for user rate limiter ---

_slack_user_limiter: Any = None


def _get_user_rate_limiter() -> Any:
    """Get or create user rate limiter for per-user rate limiting."""
    global _slack_user_limiter
    if _slack_user_limiter is None:
        try:
            from aragora.server.middleware.rate_limit.user_limiter import (
                get_user_rate_limiter,
            )

            _slack_user_limiter = get_user_rate_limiter()
        except (ImportError, RuntimeError, OSError) as e:
            logger.debug(f"User rate limiter not available: {e}")
            _slack_user_limiter = None
    return _slack_user_limiter


# --- Lazy import for workspace rate limiter ---

_slack_workspace_limiter: Any = None

# Configurable workspace rate limit (requests per minute)
SLACK_WORKSPACE_RATE_LIMIT_RPM = int(os.environ.get("SLACK_WORKSPACE_RATE_LIMIT_RPM", "30"))


def _get_workspace_rate_limiter() -> Any:
    """Get or create workspace rate limiter for per-workspace rate limiting."""
    global _slack_workspace_limiter
    if _slack_workspace_limiter is None:
        try:
            from aragora.server.middleware.rate_limit.user_limiter import (
                get_user_rate_limiter,
            )

            # Use the same limiter infrastructure but with workspace-specific configuration
            _slack_workspace_limiter = get_user_rate_limiter()
            if _slack_workspace_limiter:
                _slack_workspace_limiter.action_limits["slack_workspace_command"] = (
                    SLACK_WORKSPACE_RATE_LIMIT_RPM
                )
        except (ImportError, RuntimeError, OSError) as e:
            logger.debug(f"Workspace rate limiter not available: {e}")
            _slack_workspace_limiter = None
    return _slack_workspace_limiter


# --- URL validation ---

# Allowed domains for Slack response URLs (SSRF protection)
SLACK_ALLOWED_DOMAINS = frozenset({"hooks.slack.com", "api.slack.com"})

# Base URL for internal API calls (configurable for production)
ARAGORA_API_BASE_URL = os.environ.get("ARAGORA_API_BASE_URL", "http://localhost:8080")


def _validate_slack_url(url: str) -> bool:
    """Validate that a URL is a legitimate Slack endpoint.

    This prevents SSRF attacks by ensuring we only POST to Slack's servers.

    Args:
        url: The URL to validate

    Returns:
        True if the URL is a valid Slack endpoint, False otherwise
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        # Must be HTTPS
        if parsed.scheme != "https":
            return False
        # Must be a Slack domain
        if parsed.netloc not in SLACK_ALLOWED_DOMAINS:
            return False
        return True
    except (ValueError, TypeError) as e:
        logger.debug(f"URL validation failed for slack: {e}")
        return False


# --- Task tracking ---

import asyncio
import threading
from collections.abc import Coroutine


def _handle_task_exception(task: asyncio.Task[Any], task_name: str) -> None:
    """Handle exceptions from fire-and-forget async tasks."""
    if task.cancelled():
        logger.debug(f"Task {task_name} was cancelled")
    elif task.exception():
        exc = task.exception()
        logger.error(f"Task {task_name} failed with exception: {exc}", exc_info=exc)


def create_tracked_task(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
    """Create an async task with exception logging.

    Use this instead of raw asyncio.create_task() for fire-and-forget tasks
    to ensure exceptions are logged rather than silently swallowed.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        task = loop.create_task(coro, name=name)
        task.add_done_callback(lambda t: _handle_task_exception(t, name))
        return task

    def _run_in_thread() -> None:
        try:
            asyncio.run(coro)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError, ConnectionError) as exc:  # pragma: no cover - safety net
            logger.error(f"Task {name} failed with exception: {exc}", exc_info=exc)

    thread = threading.Thread(target=_run_in_thread, name=f"slack-task-{name}", daemon=True)
    thread.start()

    class _BackgroundTask:
        def add_done_callback(self, _cb):
            return None

    return _BackgroundTask()  # type: ignore[return-value]


# --- Handler imports ---

# RBAC permission for integration status endpoints
BOTS_READ_PERMISSION = "bots.read"

# Environment variables for Slack integration (fallback for single-workspace mode)
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

# Log at debug level for unconfigured optional integrations
if not SLACK_SIGNING_SECRET:
    logger.debug("SLACK_SIGNING_SECRET not configured - signature verification disabled")
if not SLACK_BOT_TOKEN:
    logger.debug("SLACK_BOT_TOKEN not configured - Slack API calls disabled")

# --- Multi-workspace support ---

_workspace_store: Any = None


def get_workspace_store() -> Any:
    """Get the Slack workspace store for multi-workspace support."""
    global _workspace_store
    if _workspace_store is None:
        try:
            from aragora.storage.slack_workspace_store import get_slack_workspace_store

            _workspace_store = get_slack_workspace_store()
        except ImportError:
            logger.debug("Slack workspace store not available")
    return _workspace_store


def resolve_workspace(team_id: str) -> Any:
    """Resolve a workspace by team_id.

    Returns workspace object if found, None otherwise.
    Falls back to environment variable configuration if no store configured.
    """
    if not team_id:
        return None

    store = get_workspace_store()
    if store:
        try:
            return store.get(team_id)
        except (KeyError, OSError, RuntimeError) as e:
            logger.debug(f"Failed to get workspace {team_id}: {e}")

    return None


# --- Command parsing patterns ---

COMMAND_PATTERN: Pattern[str] = re.compile(r"^/aragora\s+(\w+)(?:\s+(.*))?$")
TOPIC_PATTERN: Pattern[str] = re.compile(r'^["\']?(.+?)["\']?$')

# --- Slack integration singleton ---

_slack_integration: Any | None = None


def get_slack_integration() -> Any | None:
    """Get or create the Slack integration singleton."""
    global _slack_integration
    if _slack_integration is None:
        if not SLACK_WEBHOOK_URL:
            logger.debug("Slack integration disabled (no SLACK_WEBHOOK_URL)")
            return None
        try:
            from aragora.integrations.slack import SlackConfig, SlackIntegration

            config = SlackConfig(webhook_url=SLACK_WEBHOOK_URL)
            _slack_integration = SlackIntegration(config)
            logger.info("Slack integration initialized")
        except ImportError as e:
            logger.warning(f"Slack integration module not available: {e}")
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Invalid Slack configuration: {e}")
            return None
        except (RuntimeError, OSError, AttributeError) as e:
            logger.exception(f"Unexpected error initializing Slack integration: {e}")
            return None
    return _slack_integration


# --- Re-export common handler utilities for backward compatibility ---
# Other modules import these from config.py

# These are typed as Any since we have fallback stubs
HandlerResult: Any
SecureHandler: Any
ForbiddenError: type[Exception]
UnauthorizedError: type[Exception]
error_response: Callable[..., Any]
json_response: Callable[..., Any]
auto_error_response: Callable[[str], Callable[[Any], Any]]
rate_limit: Callable[..., Callable[[Any], Any]]

try:
    from aragora.server.handlers.base import (
        HandlerResult as _HandlerResult,
        error_response as _error_response,
        json_response as _json_response,
    )
    from aragora.server.handlers.secure import SecureHandler as _SecureHandler
    from aragora.server.handlers.utils.auth import (
        ForbiddenError as _ForbiddenError,
        UnauthorizedError as _UnauthorizedError,
    )
    from aragora.server.handlers.utils.decorators import (
        auto_error_response as _auto_error_response,
    )
    from aragora.server.handlers.utils.rate_limit import rate_limit as _rate_limit

    HandlerResult = _HandlerResult
    SecureHandler = _SecureHandler
    ForbiddenError = _ForbiddenError
    UnauthorizedError = _UnauthorizedError
    error_response = _error_response
    json_response = _json_response
    auto_error_response = _auto_error_response
    rate_limit = _rate_limit
except ImportError as e:
    logger.warning(f"Failed to import handler utilities: {e}")
    # Define stubs to prevent import errors
    # Note: These are typed as Any at module level (lines 243-244), so None is valid
    HandlerResult = None
    SecureHandler = None
    ForbiddenError = Exception
    UnauthorizedError = Exception

    def error_response(*args: Any, **kwargs: Any) -> Any:
        return None

    def json_response(*args: Any, **kwargs: Any) -> Any:
        return None

    def auto_error_response(operation: str) -> Callable[[Any], Any]:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    def rate_limit(**kwargs: Any) -> Callable[[Any], Any]:
        def decorator(func: Any) -> Any:
            return func

        return decorator
