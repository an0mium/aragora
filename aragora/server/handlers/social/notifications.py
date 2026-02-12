"""
Notifications handler for Email and Telegram integrations.

Provides endpoints for configuring and managing notification channels:
- Email notifications via SMTP
- Telegram bot notifications
- Test notification delivery
- Status and configuration management

Multi-Tenancy:
- All configurations are scoped to org_id for tenant isolation
- Per-org email/telegram configs stored in NotificationConfigStore
- Backward compatible with env var configuration (used as system default)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from aragora.billing.auth.context import UserAuthContext
from aragora.integrations.email import EmailConfig, EmailIntegration, EmailRecipient
from aragora.integrations.telegram import TelegramConfig, TelegramIntegration
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip
from aragora.server.validation.schema import (
    EMAIL_CONFIG_SCHEMA,
    NOTIFICATION_SEND_SCHEMA,
    TELEGRAM_CONFIG_SCHEMA,
    validate_against_schema,
)
from aragora.storage.notification_config_store import (
    StoredEmailConfig,
    StoredTelegramConfig,
    StoredEmailRecipient,
    get_notification_config_store,
)

logger = logging.getLogger(__name__)

# Rate limiter for notification endpoints (30 requests per minute - can trigger external calls)
_notifications_limiter = RateLimiter(requests_per_minute=30)


def _run_async_in_thread(coro: Any) -> Any:
    """Run an async coroutine in a thread-safe manner.

    Creates a new event loop for the thread to avoid RuntimeError when
    asyncio.run() is called from within a ThreadPoolExecutor.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =============================================================================
# Per-Organization Integration Factory with TTL Caching
# =============================================================================

import time as _time

# Cache settings
_CACHE_TTL_SECONDS = 3600  # 1 hour - integrations rarely change
_CACHE_MAX_SIZE = 50  # Max cached orgs before eviction


class _TTLCache:
    """Simple TTL cache with size limits for integration objects."""

    def __init__(self, max_size: int = _CACHE_MAX_SIZE, ttl: float = _CACHE_TTL_SECONDS):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None
        value, cached_at = self._cache[key]
        if _time.time() - cached_at > self._ttl:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (value, _time.time())

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()


# Cache of active integrations per org (with TTL and size limits)
_org_email_cache = _TTLCache()
_org_telegram_cache = _TTLCache()

# Legacy references for backward compatibility - these wrapper dicts delegate to cache
_org_email_integrations: dict[str, EmailIntegration] = {}  # Unused, kept for compat
_org_telegram_integrations: dict[str, TelegramIntegration] = {}  # Unused, kept for compat

# System-wide fallback from environment (for backward compatibility)
_system_email_integration: EmailIntegration | None = None
_system_telegram_integration: TelegramIntegration | None = None


async def get_email_integration_for_org(org_id: str | None = None) -> EmailIntegration | None:
    """Get email integration for an organization.

    Priority:
    1. Per-org config from NotificationConfigStore
    2. System-wide config from environment variables (fallback)
    """
    # Backward-compat: check legacy in-memory cache first
    if org_id and org_id in _org_email_integrations:
        return _org_email_integrations[org_id]

    # Check org-specific TTL cache
    if org_id:
        cached = _org_email_cache.get(org_id)
        if cached is not None:
            return cached

    # Try to load from store
    if org_id:
        store = get_notification_config_store()
        stored_config = await store.get_email_config(org_id)
        if stored_config and stored_config.smtp_host:
            try:
                config = EmailConfig(
                    provider=stored_config.provider,
                    smtp_host=stored_config.smtp_host,
                    smtp_port=stored_config.smtp_port,
                    smtp_username=stored_config.smtp_username,
                    smtp_password=stored_config.smtp_password,
                    use_tls=stored_config.use_tls,
                    use_ssl=stored_config.use_ssl,
                    sendgrid_api_key=stored_config.sendgrid_api_key,
                    ses_region=stored_config.ses_region,
                    ses_access_key_id=stored_config.ses_access_key_id,
                    ses_secret_access_key=stored_config.ses_secret_access_key,
                    from_email=stored_config.from_email,
                    from_name=stored_config.from_name,
                    notify_on_consensus=stored_config.notify_on_consensus,
                    notify_on_debate_end=stored_config.notify_on_debate_end,
                    notify_on_error=stored_config.notify_on_error,
                    enable_digest=stored_config.enable_digest,
                    digest_frequency=stored_config.digest_frequency,
                    min_consensus_confidence=stored_config.min_consensus_confidence,
                    max_emails_per_hour=stored_config.max_emails_per_hour,
                )
                integration = EmailIntegration(config)

                # Load recipients for this org
                recipients = await store.get_recipients(org_id)
                for r in recipients:
                    integration.add_recipient(
                        EmailRecipient(email=r.email, name=r.name, preferences=r.preferences)
                    )

                _org_email_cache.set(org_id, integration)
                logger.info("Email integration loaded for org %s", org_id)
                return integration
            except Exception as e:
                logger.warning("Failed to create email integration for org %s: %s", org_id, e)

    # Fall back to system-wide config from environment
    return _get_system_email_integration()


def _get_system_email_integration() -> EmailIntegration | None:
    """Get system-wide email integration from environment (backward compatibility)."""
    global _system_email_integration
    if _system_email_integration is not None:
        return _system_email_integration

    smtp_host = os.getenv("SMTP_HOST")
    if smtp_host:
        try:
            config = EmailConfig(
                smtp_host=smtp_host,
                smtp_port=int(os.getenv("SMTP_PORT", "587")),
                smtp_username=os.getenv("SMTP_USERNAME", ""),
                smtp_password=os.getenv("SMTP_PASSWORD", ""),
                use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
                use_ssl=os.getenv("SMTP_USE_SSL", "false").lower() == "true",
                from_email=os.getenv("SMTP_FROM_EMAIL", "debates@aragora.ai"),
                from_name=os.getenv("SMTP_FROM_NAME", "Aragora Debates"),
            )
            _system_email_integration = EmailIntegration(config)
            logger.info("System email integration initialized with host: %s", smtp_host)
        except Exception as e:
            logger.warning("Failed to initialize system email integration: %s", e)

    return _system_email_integration


async def get_telegram_integration_for_org(
    org_id: str | None = None,
) -> TelegramIntegration | None:
    """Get telegram integration for an organization.

    Priority:
    1. Per-org config from NotificationConfigStore
    2. System-wide config from environment variables (fallback)
    """
    # Backward-compat: check legacy in-memory cache first
    if org_id and org_id in _org_telegram_integrations:
        return _org_telegram_integrations[org_id]

    # Check org-specific TTL cache
    if org_id:
        cached = _org_telegram_cache.get(org_id)
        if cached is not None:
            return cached

    # Try to load from store
    if org_id:
        store = get_notification_config_store()
        stored_config = await store.get_telegram_config(org_id)
        if stored_config and stored_config.bot_token and stored_config.chat_id:
            try:
                config = TelegramConfig(
                    bot_token=stored_config.bot_token,
                    chat_id=stored_config.chat_id,
                    notify_on_consensus=stored_config.notify_on_consensus,
                    notify_on_debate_end=stored_config.notify_on_debate_end,
                    notify_on_error=stored_config.notify_on_error,
                    min_consensus_confidence=stored_config.min_consensus_confidence,
                    max_messages_per_minute=stored_config.max_messages_per_minute,
                )
                integration = TelegramIntegration(config)
                _org_telegram_cache.set(org_id, integration)
                logger.info("Telegram integration loaded for org %s", org_id)
                return integration
            except Exception as e:
                logger.warning("Failed to create telegram integration for org %s: %s", org_id, e)

    # Fall back to system-wide config from environment
    return _get_system_telegram_integration()


def _get_system_telegram_integration() -> TelegramIntegration | None:
    """Get system-wide telegram integration from environment (backward compatibility)."""
    global _system_telegram_integration
    if _system_telegram_integration is not None:
        return _system_telegram_integration

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if bot_token and chat_id:
        try:
            config = TelegramConfig(bot_token=bot_token, chat_id=chat_id)
            _system_telegram_integration = TelegramIntegration(config)
            logger.info("System telegram integration initialized")
        except Exception as e:
            logger.warning("Failed to initialize system telegram integration: %s", e)

    return _system_telegram_integration


def invalidate_org_integration_cache(org_id: str) -> None:
    """Invalidate cached integrations when config changes."""
    _org_email_cache.invalidate(org_id)
    _org_telegram_cache.invalidate(org_id)
    _org_email_integrations.pop(org_id, None)
    _org_telegram_integrations.pop(org_id, None)
    logger.debug("Invalidated integration cache for org %s", org_id)


# =============================================================================
# Backward Compatibility Functions (for utility functions and other modules)
# =============================================================================


def get_email_integration() -> EmailIntegration | None:
    """Get system-wide email integration (backward compatibility)."""
    return _get_system_email_integration()


def get_telegram_integration() -> TelegramIntegration | None:
    """Get system-wide telegram integration (backward compatibility)."""
    return _get_system_telegram_integration()


def configure_email_integration(config: EmailConfig) -> EmailIntegration:
    """Configure system-wide email integration (backward compatibility)."""
    global _system_email_integration
    _system_email_integration = EmailIntegration(config)
    logger.info("System email integration configured with host: %s", config.smtp_host)
    return _system_email_integration


def configure_telegram_integration(config: TelegramConfig) -> TelegramIntegration:
    """Configure system-wide telegram integration (backward compatibility)."""
    global _system_telegram_integration
    _system_telegram_integration = TelegramIntegration(config)
    logger.info("System telegram integration configured")
    return _system_telegram_integration


class NotificationsHandler(SecureHandler):
    """Handler for notification-related endpoints.

    Extends SecureHandler for JWT-based authentication, RBAC permission
    enforcement, and security audit logging.

    SECURITY: All endpoints require authentication. Recipients and status
    are scoped to the authenticated user/organization.

    Endpoints:
        GET  /api/notifications/status - Get integration status
        POST /api/notifications/email/config - Configure email settings
        POST /api/notifications/telegram/config - Configure Telegram settings
        POST /api/notifications/email/recipient - Add email recipient
        DELETE /api/notifications/email/recipient - Remove email recipient
        POST /api/notifications/test - Send test notification
        POST /api/notifications/send - Send a notification
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    RESOURCE_TYPE = "notification"

    ROUTES = [
        "/api/v1/notifications/status",
        "/api/v1/notifications/email/recipients",
        "/api/v1/notifications/email/config",
        "/api/v1/notifications/telegram/config",
        "/api/v1/notifications/email/recipient",
        "/api/v1/notifications/test",
        "/api/v1/notifications/send",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/notifications")

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Handle GET requests.

        SECURITY: All GET endpoints require authentication and RBAC permissions.
        Access is logged for audit trails.
        """
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _notifications_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for notifications endpoint: %s", client_ip)
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # SECURITY: Use SecureHandler's auth method for consistent RBAC
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # SECURITY: Require 'notifications.read' permission for all GET endpoints
        user_ctx: UserAuthContext | None
        perm_err: HandlerResult | None
        user_ctx, perm_err = self.require_permission_or_error(handler, "read")
        if perm_err:
            return perm_err

        if path == "/api/v1/notifications/status":
            # SECURITY: Log access with org context for audit trail
            logger.info(
                "Notifications status accessed by user %s in org %s", user.user_id, user.org_id
            )
            return self._get_status(user.org_id)

        if path == "/api/v1/notifications/email/recipients":
            # SECURITY: Log recipient list access with org context for audit trail
            logger.info("Email recipients accessed by user %s in org %s", user.user_id, user.org_id)
            return self._get_email_recipients(user.org_id)

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests.

        SECURITY: All POST endpoints require authentication and RBAC permissions.
        Configuration changes require 'write' permission.
        """
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _notifications_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for notifications endpoint: %s", client_ip)
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Require authentication for all POST endpoints
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # SECURITY: Require 'notifications.write' permission for configuration changes
        user_ctx: UserAuthContext | None
        perm_err: HandlerResult | None
        user_ctx, perm_err = self.require_permission_or_error(handler, "write")
        if perm_err:
            return perm_err

        if path == "/api/v1/notifications/email/config":
            logger.info("Email config modified by user %s in org %s", user.user_id, user.org_id)
            return self._configure_email(handler, user.org_id)

        if path == "/api/v1/notifications/telegram/config":
            logger.info("Telegram config modified by user %s in org %s", user.user_id, user.org_id)
            return self._configure_telegram(handler, user.org_id)

        if path == "/api/v1/notifications/email/recipient":
            logger.info("Email recipient added by user %s in org %s", user.user_id, user.org_id)
            return self._add_email_recipient(handler, user.org_id)

        if path == "/api/v1/notifications/test":
            logger.info("Test notification sent by user %s", user.user_id)
            return self._send_test_notification(handler)

        if path == "/api/v1/notifications/send":
            logger.info("Notification sent by user %s", user.user_id)
            return self._send_notification(handler)

        return None

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle DELETE requests.

        SECURITY: All DELETE endpoints require authentication and RBAC permissions.
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # SECURITY: Require 'notifications.delete' permission
        user_ctx: UserAuthContext | None
        perm_err: HandlerResult | None
        user_ctx, perm_err = self.require_permission_or_error(handler, "delete")
        if perm_err:
            return perm_err

        if path == "/api/v1/notifications/email/recipient":
            logger.info("Email recipient removed by user %s in org %s", user.user_id, user.org_id)
            return self._remove_email_recipient(handler, query_params, user.org_id)

        return None

    def _get_status(self, org_id: str | None = None) -> HandlerResult:
        """Get status of notification integrations scoped to organization.

        SECURITY: org_id is used for tenant scoping. Only configuration
        and recipients belonging to the authenticated user's organization
        are returned.
        """
        import asyncio

        # Get org-specific integrations (async)
        async def get_integrations() -> tuple[EmailIntegration | None, TelegramIntegration | None]:
            email = await get_email_integration_for_org(org_id)
            telegram = await get_telegram_integration_for_org(org_id)
            return email, telegram

        try:
            try:
                loop = asyncio.get_running_loop()
                # Loop is running, execute in thread pool
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    email, telegram = pool.submit(_run_async_in_thread, get_integrations()).result()
            except RuntimeError:
                # No running loop, create a new one
                loop = asyncio.new_event_loop()
                try:
                    email, telegram = loop.run_until_complete(get_integrations())
                finally:
                    loop.close()
        except Exception as e:
            logger.warning("Failed to get integrations for org %s: %s", org_id, e)
            email, telegram = None, None

        # Log org context for debugging tenant isolation issues
        if org_id:
            logger.debug("Getting notification status for org: %s", org_id)

        return json_response(
            {
                "email": {
                    "configured": email is not None,
                    "host": email.config.smtp_host if email else None,
                    "recipients_count": len(email.recipients) if email else 0,
                    "settings": (
                        {
                            "notify_on_consensus": (
                                email.config.notify_on_consensus if email else False
                            ),
                            "notify_on_debate_end": (
                                email.config.notify_on_debate_end if email else False
                            ),
                            "notify_on_error": email.config.notify_on_error if email else False,
                            "enable_digest": email.config.enable_digest if email else False,
                            "digest_frequency": email.config.digest_frequency if email else "daily",
                        }
                        if email
                        else None
                    ),
                },
                "telegram": {
                    "configured": telegram is not None,
                    "chat_id": telegram.config.chat_id[:8] + "..." if telegram else None,
                    "settings": (
                        {
                            "notify_on_consensus": (
                                telegram.config.notify_on_consensus if telegram else False
                            ),
                            "notify_on_debate_end": (
                                telegram.config.notify_on_debate_end if telegram else False
                            ),
                            "notify_on_error": (
                                telegram.config.notify_on_error if telegram else False
                            ),
                        }
                        if telegram
                        else None
                    ),
                },
            }
        )

    def _get_email_recipients(self, org_id: str | None = None) -> HandlerResult:
        """Get list of email recipients scoped to organization.

        SECURITY: org_id is used for tenant scoping. Only recipients
        belonging to the authenticated user's organization are returned.
        """
        import asyncio

        if not org_id:
            # No org context - use system integration
            email = get_email_integration()
            if not email:
                return json_response({"recipients": [], "error": "Email not configured"})
            return json_response(
                {
                    "recipients": [{"email": r.email, "name": r.name} for r in email.recipients],
                    "count": len(email.recipients),
                }
            )

        # Get org-specific recipients from store
        async def get_org_recipients() -> list[StoredEmailRecipient]:
            store = get_notification_config_store()
            return await store.get_recipients(org_id)

        try:
            try:
                loop = asyncio.get_running_loop()
                # Loop is running, execute in thread pool
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    recipients = pool.submit(_run_async_in_thread, get_org_recipients()).result()
            except RuntimeError:
                # No running loop, create a new one
                loop = asyncio.new_event_loop()
                try:
                    recipients = loop.run_until_complete(get_org_recipients())
                finally:
                    loop.close()
        except Exception as e:
            logger.warning("Failed to get recipients for org %s: %s", org_id, e)
            return json_response({"recipients": [], "error": str(e)})

        logger.debug("Getting email recipients for org: %s", org_id)

        return json_response(
            {
                "recipients": [{"email": r.email, "name": r.name} for r in recipients],
                "count": len(recipients),
                "org_id": org_id,
            }
        )

    def _configure_email(self, handler: Any, org_id: str | None = None) -> HandlerResult:
        """Configure email integration settings for an organization."""
        import asyncio

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, EMAIL_CONFIG_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        try:
            # Save to per-org store if org_id provided
            if org_id:
                stored_config = StoredEmailConfig(
                    org_id=org_id,
                    provider=body.get("provider", "smtp"),
                    smtp_host=body.get("smtp_host", ""),
                    smtp_port=body.get("smtp_port", 587),
                    smtp_username=body.get("smtp_username", ""),
                    smtp_password=body.get("smtp_password", ""),
                    use_tls=body.get("use_tls", True),
                    use_ssl=body.get("use_ssl", False),
                    sendgrid_api_key=body.get("sendgrid_api_key", ""),
                    ses_region=body.get("ses_region", "us-east-1"),
                    ses_access_key_id=body.get("ses_access_key_id", ""),
                    ses_secret_access_key=body.get("ses_secret_access_key", ""),
                    from_email=body.get("from_email", "debates@aragora.ai"),
                    from_name=body.get("from_name", "Aragora Debates"),
                    notify_on_consensus=body.get("notify_on_consensus", True),
                    notify_on_debate_end=body.get("notify_on_debate_end", True),
                    notify_on_error=body.get("notify_on_error", True),
                    enable_digest=body.get("enable_digest", True),
                    digest_frequency=body.get("digest_frequency", "daily"),
                    min_consensus_confidence=body.get("min_consensus_confidence", 0.7),
                    max_emails_per_hour=body.get("max_emails_per_hour", 50),
                )

                async def save_config() -> None:
                    store = get_notification_config_store()
                    await store.save_email_config(stored_config)
                    invalidate_org_integration_cache(org_id)

                try:
                    try:
                        loop = asyncio.get_running_loop()
                        # Loop is running, execute in thread pool
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            pool.submit(_run_async_in_thread, save_config()).result()
                    except RuntimeError:
                        # No running loop, create a new one
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(save_config())
                        finally:
                            loop.close()
                except Exception as e:
                    logger.error("Failed to save email config for org %s: %s", org_id, e)
                    return error_response(f"Failed to save configuration: {e}", 500)

                return json_response(
                    {
                        "success": True,
                        "message": f"Email configured for org {org_id} with host: {stored_config.smtp_host}",
                        "org_id": org_id,
                    }
                )

            # No org_id - configure system-wide (backward compatibility)
            config = EmailConfig(
                smtp_host=body.get("smtp_host", ""),
                smtp_port=body.get("smtp_port", 587),
                smtp_username=body.get("smtp_username", ""),
                smtp_password=body.get("smtp_password", ""),
                use_tls=body.get("use_tls", True),
                use_ssl=body.get("use_ssl", False),
                from_email=body.get("from_email", "debates@aragora.ai"),
                from_name=body.get("from_name", "Aragora Debates"),
                notify_on_consensus=body.get("notify_on_consensus", True),
                notify_on_debate_end=body.get("notify_on_debate_end", True),
                notify_on_error=body.get("notify_on_error", True),
                enable_digest=body.get("enable_digest", True),
                digest_frequency=body.get("digest_frequency", "daily"),
                min_consensus_confidence=body.get("min_consensus_confidence", 0.7),
                max_emails_per_hour=body.get("max_emails_per_hour", 50),
            )
            configure_email_integration(config)
            return json_response(
                {
                    "success": True,
                    "message": f"Email configured with host: {config.smtp_host}",
                }
            )
        except ValueError as e:
            return error_response(f"Invalid configuration: {e}", 400)
        except Exception as e:
            logger.error("Failed to configure email: %s", e)
            return error_response("Failed to configure email", 500)

    def _configure_telegram(self, handler: Any, org_id: str | None = None) -> HandlerResult:
        """Configure Telegram integration settings for an organization."""
        import asyncio

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, TELEGRAM_CONFIG_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        bot_token = body.get("bot_token", "")
        chat_id = body.get("chat_id", "")

        try:
            # Save to per-org store if org_id provided
            if org_id:
                stored_config = StoredTelegramConfig(
                    org_id=org_id,
                    bot_token=bot_token,
                    chat_id=chat_id,
                    notify_on_consensus=body.get("notify_on_consensus", True),
                    notify_on_debate_end=body.get("notify_on_debate_end", True),
                    notify_on_error=body.get("notify_on_error", True),
                    min_consensus_confidence=body.get("min_consensus_confidence", 0.7),
                    max_messages_per_minute=body.get("max_messages_per_minute", 20),
                )

                async def save_config() -> None:
                    store = get_notification_config_store()
                    await store.save_telegram_config(stored_config)
                    invalidate_org_integration_cache(org_id)

                try:
                    try:
                        loop = asyncio.get_running_loop()
                        # Loop is running, execute in thread pool
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            pool.submit(_run_async_in_thread, save_config()).result()
                    except RuntimeError:
                        # No running loop, create a new one
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(save_config())
                        finally:
                            loop.close()
                except Exception as e:
                    logger.error("Failed to save telegram config for org %s: %s", org_id, e)
                    return error_response(f"Failed to save configuration: {e}", 500)

                return json_response(
                    {
                        "success": True,
                        "message": f"Telegram configured for org {org_id}",
                        "org_id": org_id,
                    }
                )

            # No org_id - configure system-wide (backward compatibility)
            config = TelegramConfig(
                bot_token=bot_token,
                chat_id=chat_id,
                notify_on_consensus=body.get("notify_on_consensus", True),
                notify_on_debate_end=body.get("notify_on_debate_end", True),
                notify_on_error=body.get("notify_on_error", True),
                min_consensus_confidence=body.get("min_consensus_confidence", 0.7),
                max_messages_per_minute=body.get("max_messages_per_minute", 20),
            )
            configure_telegram_integration(config)
            return json_response(
                {
                    "success": True,
                    "message": "Telegram configured successfully",
                }
            )
        except ValueError as e:
            return error_response(f"Invalid configuration: {e}", 400)
        except Exception as e:
            logger.error("Failed to configure telegram: %s", e)
            return error_response("Failed to configure telegram", 500)

    def _add_email_recipient(self, handler: Any, org_id: str | None = None) -> HandlerResult:
        """Add an email recipient for an organization."""
        import asyncio

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        recipient_email = body.get("email", "")
        if not recipient_email or "@" not in recipient_email:
            return error_response("Valid email address required", 400)

        # Save to per-org store if org_id provided
        if org_id:
            stored_recipient = StoredEmailRecipient(
                org_id=org_id,
                email=recipient_email,
                name=body.get("name"),
                preferences=body.get("preferences", {}),
            )

            async def save_recipient() -> list[StoredEmailRecipient]:
                store = get_notification_config_store()
                await store.add_recipient(stored_recipient)
                # Also add to cached integration if it exists
                if org_id in _org_email_integrations:
                    _org_email_integrations[org_id].add_recipient(
                        EmailRecipient(
                            email=recipient_email,
                            name=body.get("name"),
                            preferences=body.get("preferences", {}),
                        )
                    )
                return await store.get_recipients(org_id)

            try:
                try:
                    loop = asyncio.get_running_loop()
                    # Loop is running, execute in thread pool
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        recipients = pool.submit(_run_async_in_thread, save_recipient()).result()
                except RuntimeError:
                    # No running loop, create a new one
                    loop = asyncio.new_event_loop()
                    try:
                        recipients = loop.run_until_complete(save_recipient())
                    finally:
                        loop.close()
            except Exception as e:
                logger.error("Failed to add recipient for org %s: %s", org_id, e)
                return error_response(f"Failed to add recipient: {e}", 500)

            return json_response(
                {
                    "success": True,
                    "message": f"Recipient added: {recipient_email}",
                    "recipients_count": len(recipients),
                    "org_id": org_id,
                }
            )

        # No org_id - use system integration (backward compatibility)
        email = get_email_integration()
        if not email:
            return error_response("Email integration not configured", 503)

        recipient = EmailRecipient(
            email=recipient_email,
            name=body.get("name"),
            preferences=body.get("preferences", {}),
        )
        email.add_recipient(recipient)

        return json_response(
            {
                "success": True,
                "message": f"Recipient added: {recipient_email}",
                "recipients_count": len(email.recipients),
            }
        )

    def _remove_email_recipient(
        self, handler: Any, query_params: dict, org_id: str | None = None
    ) -> HandlerResult:
        """Remove an email recipient for an organization."""
        import asyncio

        recipient_email = query_params.get("email", "")
        if not recipient_email:
            return error_response("email parameter required", 400)

        # Remove from per-org store if org_id provided
        if org_id:

            async def remove_recipient() -> tuple[bool, list[StoredEmailRecipient]]:
                store = get_notification_config_store()
                removed = await store.remove_recipient(org_id, recipient_email)
                # Also remove from cached integration if it exists
                if org_id in _org_email_integrations:
                    _org_email_integrations[org_id].remove_recipient(recipient_email)
                return removed, await store.get_recipients(org_id)

            try:
                try:
                    loop = asyncio.get_running_loop()
                    # Loop is running, execute in thread pool
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        removed, recipients = pool.submit(
                            _run_async_in_thread, remove_recipient()
                        ).result()
                except RuntimeError:
                    # No running loop, create a new one
                    loop = asyncio.new_event_loop()
                    try:
                        removed, recipients = loop.run_until_complete(remove_recipient())
                    finally:
                        loop.close()
            except Exception as e:
                logger.error("Failed to remove recipient for org %s: %s", org_id, e)
                return error_response(f"Failed to remove recipient: {e}", 500)

            if removed:
                return json_response(
                    {
                        "success": True,
                        "message": f"Recipient removed: {recipient_email}",
                        "recipients_count": len(recipients),
                        "org_id": org_id,
                    }
                )
            else:
                return error_response(f"Recipient not found: {recipient_email}", 404)

        # No org_id - use system integration (backward compatibility)
        email = get_email_integration()
        if not email:
            return error_response("Email integration not configured", 503)

        removed = email.remove_recipient(recipient_email)
        if removed:
            return json_response(
                {
                    "success": True,
                    "message": f"Recipient removed: {recipient_email}",
                    "recipients_count": len(email.recipients),
                }
            )
        else:
            return error_response(f"Recipient not found: {recipient_email}", 404)

    def _send_test_notification(self, handler: Any) -> HandlerResult:
        """Send a test notification."""
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        notification_type = body.get("type", "all")
        results = {}

        # Test email
        if notification_type in ("all", "email"):
            email = get_email_integration()
            if email:
                if email.recipients:
                    # Import asyncio for running async in sync context
                    import asyncio

                    async def send_test_email() -> bool:
                        return await email._send_email(
                            email.recipients[0],
                            "Aragora Test Notification",
                            "<h1>Test Notification</h1><p>Your email integration is working correctly!</p>",
                            "Test Notification - Your email integration is working correctly!",
                        )

                    try:
                        try:
                            loop = asyncio.get_running_loop()
                            # Already in async context, run in thread with new loop
                            import concurrent.futures

                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                success = pool.submit(
                                    _run_async_in_thread, send_test_email()
                                ).result()
                        except RuntimeError:
                            # No running loop, create a new one
                            loop = asyncio.new_event_loop()
                            try:
                                success = loop.run_until_complete(send_test_email())
                            finally:
                                loop.close()
                        results["email"] = {
                            "success": success,
                            "recipient": email.recipients[0].email,
                        }
                    except Exception as e:
                        results["email"] = {"success": False, "error": str(e)}
                else:
                    results["email"] = {"success": False, "error": "No recipients configured"}
            else:
                results["email"] = {"success": False, "error": "Email not configured"}

        # Test telegram
        if notification_type in ("all", "telegram"):
            telegram = get_telegram_integration()
            if telegram:
                import asyncio

                from aragora.integrations.telegram import TelegramMessage

                async def send_test_telegram() -> bool:
                    msg = TelegramMessage(
                        text="<b>Test Notification</b>\n\nYour Telegram integration is working correctly! 🎉",
                    )
                    return await telegram._send_message(msg)

                try:
                    try:
                        loop = asyncio.get_running_loop()
                        # Loop is running, execute in thread pool
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            success = pool.submit(
                                _run_async_in_thread, send_test_telegram()
                            ).result()
                    except RuntimeError:
                        # No running loop, create a new one
                        loop = asyncio.new_event_loop()
                        try:
                            success = loop.run_until_complete(send_test_telegram())
                        finally:
                            loop.close()
                    results["telegram"] = {"success": success}
                except Exception as e:
                    results["telegram"] = {"success": False, "error": str(e)}
            else:
                results["telegram"] = {"success": False, "error": "Telegram not configured"}

        all_success = all(r.get("success", False) for r in results.values())
        return json_response(
            {
                "success": all_success,
                "results": results,
            }
        )

    def _send_notification(self, handler: Any) -> HandlerResult:
        """Send a notification with custom content."""
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, NOTIFICATION_SEND_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        notification_type = body.get("type", "all")
        subject = body.get("subject", "Aragora Notification")
        message = body.get("message", "")
        html_message = body.get("html_message", f"<p>{message}</p>")

        results = {}
        import asyncio

        # Send email
        if notification_type in ("all", "email"):
            email = get_email_integration()
            if email and email.recipients:

                async def send_emails() -> int:
                    sent = 0
                    for recipient in email.recipients:
                        success = await email._send_email(recipient, subject, html_message, message)
                        if success:
                            sent += 1
                    return sent

                try:
                    try:
                        loop = asyncio.get_running_loop()
                        # Loop is running, execute in thread pool
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            sent = pool.submit(_run_async_in_thread, send_emails()).result()
                    except RuntimeError:
                        # No running loop, create a new one
                        loop = asyncio.new_event_loop()
                        try:
                            sent = loop.run_until_complete(send_emails())
                        finally:
                            loop.close()
                    results["email"] = {
                        "success": sent > 0,
                        "sent": sent,
                        "total": len(email.recipients),
                    }
                except Exception as e:
                    results["email"] = {"success": False, "error": str(e)}
            else:
                results["email"] = {
                    "success": False,
                    "error": "Email not configured or no recipients",
                }

        # Send telegram
        if notification_type in ("all", "telegram"):
            telegram = get_telegram_integration()
            if telegram:
                from aragora.integrations.telegram import TelegramMessage

                # Convert to Telegram HTML format
                telegram_text = f"<b>{subject}</b>\n\n{message}"

                async def send_telegram() -> bool:
                    msg = TelegramMessage(text=telegram_text)
                    return await telegram._send_message(msg)

                try:
                    try:
                        loop = asyncio.get_running_loop()
                        # Loop is running, execute in thread pool
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            success = pool.submit(_run_async_in_thread, send_telegram()).result()
                    except RuntimeError:
                        # No running loop, create a new one
                        loop = asyncio.new_event_loop()
                        try:
                            success = loop.run_until_complete(send_telegram())
                        finally:
                            loop.close()
                    results["telegram"] = {"success": success}
                except Exception as e:
                    results["telegram"] = {"success": False, "error": str(e)}
            else:
                results["telegram"] = {"success": False, "error": "Telegram not configured"}

        all_success = all(r.get("success", False) for r in results.values())
        return json_response(
            {
                "success": all_success,
                "results": results,
            }
        )


# Utility functions for use by other handlers/orchestrator
async def notify_debate_completed(result: Any) -> dict[str, bool]:
    """Notify all configured channels about a completed debate.

    This function is designed to be called from the debate orchestrator
    after a debate completes.

    Args:
        result: DebateResult object

    Returns:
        Dict with success status for each channel
    """
    results = {}

    email = get_email_integration()
    if email:
        try:
            sent = await email.send_debate_summary(result)
            results["email"] = sent > 0
        except Exception as e:
            logger.error("Failed to send email notification: %s", e)
            results["email"] = False

    telegram = get_telegram_integration()
    if telegram:
        try:
            success = await telegram.post_debate_summary(result)
            results["telegram"] = success
        except Exception as e:
            logger.error("Failed to send telegram notification: %s", e)
            results["telegram"] = False

    return results


async def notify_consensus_reached(
    debate_id: str,
    confidence: float,
    winner: str | None = None,
    task: str | None = None,
) -> dict[str, bool]:
    """Notify all configured channels about consensus being reached.

    Args:
        debate_id: ID of the debate
        confidence: Consensus confidence score
        winner: Winning agent name
        task: Task description

    Returns:
        Dict with success status for each channel
    """
    results = {}

    email = get_email_integration()
    if email:
        try:
            sent = await email.send_consensus_alert(debate_id, confidence, winner, task)
            results["email"] = sent > 0
        except Exception as e:
            logger.error("Failed to send email consensus alert: %s", e)
            results["email"] = False

    telegram = get_telegram_integration()
    if telegram:
        try:
            success = await telegram.send_consensus_alert(debate_id, confidence, winner, task)
            results["telegram"] = success
        except Exception as e:
            logger.error("Failed to send telegram consensus alert: %s", e)
            results["telegram"] = False

    return results


async def notify_error(
    error_type: str,
    error_message: str,
    debate_id: str | None = None,
    severity: str = "warning",
) -> dict[str, bool]:
    """Notify configured channels about an error.

    Args:
        error_type: Type of error
        error_message: Error details
        debate_id: Optional debate ID
        severity: One of "info", "warning", "error", "critical"

    Returns:
        Dict with success status for each channel
    """
    results = {}

    telegram = get_telegram_integration()
    if telegram:
        try:
            success = await telegram.send_error_alert(
                error_type, error_message, debate_id, severity
            )
            results["telegram"] = success
        except Exception as e:
            logger.error("Failed to send telegram error alert: %s", e)
            results["telegram"] = False

    return results


__all__ = [
    "NotificationsHandler",
    "get_email_integration",
    "get_telegram_integration",
    "configure_email_integration",
    "configure_telegram_integration",
    "notify_debate_completed",
    "notify_consensus_reached",
    "notify_error",
]
