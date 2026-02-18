"""
Storage and initialization utilities for email handlers.

Provides thread-safe lazy initialization of:
- Email persistent store
- Gmail connector
- Email prioritizer
- Cross-channel context service
- User config cache
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from aragora.server.handlers.utils.lazy_stores import LazyStoreFactory

logger = logging.getLogger(__name__)

# RBAC imports (optional - graceful degradation if not available)
try:
    from aragora.rbac import check_permission

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False

from aragora.server.handlers.utils.rbac_guard import rbac_fail_closed


def _check_email_permission(auth_context: Any | None, permission_key: str) -> dict[str, Any] | None:
    """
    Check RBAC permission for email operations.

    Args:
        auth_context: Optional AuthorizationContext from request
        permission_key: Permission like "email:read" or "email:write"

    Returns:
        None if allowed, error dict with success=False if denied
    """
    # Fail closed for all operations in production if RBAC is unavailable.
    if not RBAC_AVAILABLE:
        if rbac_fail_closed():
            return {
                "success": False,
                "error": "Service unavailable: access control module not loaded",
            }
        # Dev/test: fail closed for write-sensitive operations only
        if permission_key in {"email:write", "email:update", "email:oauth"}:
            return {
                "success": False,
                "error": "Permission denied",
            }
        return None  # Read-only paths degrade gracefully in dev/test

    if auth_context is None:
        if permission_key in {"email:write", "email:update", "email:oauth"}:
            return {
                "success": False,
                "error": "Permission denied",
            }
        return None

    try:
        decision = check_permission(auth_context, permission_key)
        if not decision.allowed:
            logger.warning(
                f"RBAC denied: permission={permission_key} "
                f"user={getattr(auth_context, 'user_id', 'unknown')} "
                f"reason={decision.reason}"
            )
            return {
                "success": False,
                "error": "Permission denied",
            }
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        logger.warning(f"RBAC check failed for {permission_key}: {e}")
        return None  # Fail open

    return None


# =============================================================================
# Persistent Storage
# =============================================================================

_email_store = LazyStoreFactory(
    store_name="email_store",
    import_path="aragora.storage.email_store",
    factory_name="get_email_store",
    logger_context="EmailHandler",
)

# Public alias so other modules can import get_email_store
get_email_store = _email_store.get


def _load_config_from_store(user_id: str, workspace_id: str = "default") -> dict[str, Any]:
    """Load config from persistent store into memory cache."""
    if _email_store is None:
        return {}
    store = _email_store.get()
    if store:
        try:
            config = store.get_user_config(user_id, workspace_id)
            if config:
                return config
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning(f"[EmailHandler] Failed to load config from store: {e}")
    return {}


def _save_config_to_store(
    user_id: str, config: dict[str, Any], workspace_id: str = "default"
) -> None:
    """Save config to persistent store."""
    if _email_store is None:
        return
    store = _email_store.get()
    if store:
        try:
            store.save_user_config(user_id, workspace_id, config)
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning(f"[EmailHandler] Failed to save config to store: {e}")


# Global instances (initialized lazily) with thread-safe access
_gmail_connector: Any | None = None
_gmail_connector_lock = threading.Lock()
_prioritizer: Any | None = None
_prioritizer_lock = threading.Lock()
_context_service: Any | None = None
_context_service_lock = threading.Lock()
_user_configs: dict[str, dict[str, Any]] = {}
_user_configs_lock = threading.Lock()


def get_gmail_connector(user_id: str = "default"):
    """Get or create Gmail connector for a user (thread-safe)."""
    global _gmail_connector
    if _gmail_connector is not None:
        return _gmail_connector

    with _gmail_connector_lock:
        # Double-check after acquiring lock
        if _gmail_connector is None:
            from aragora.connectors.enterprise.communication.gmail import GmailConnector

            _gmail_connector = GmailConnector()
        return _gmail_connector


def get_prioritizer(user_id: str = "default"):
    """Get or create email prioritizer for a user (thread-safe)."""
    global _prioritizer
    if _prioritizer is not None:
        return _prioritizer

    with _prioritizer_lock:
        # Double-check after acquiring lock
        if _prioritizer is None:
            from aragora.services.email_prioritization import (
                EmailPrioritizer,
                EmailPrioritizationConfig,
            )

            # Load user config if available (thread-safe access)
            with _user_configs_lock:
                config_data = _user_configs.get(user_id, {}).copy()

            config = EmailPrioritizationConfig(
                vip_domains=set(config_data.get("vip_domains", [])),
                vip_addresses=set(config_data.get("vip_addresses", [])),
                internal_domains=set(config_data.get("internal_domains", [])),
                auto_archive_senders=set(config_data.get("auto_archive_senders", [])),
            )

            _prioritizer = EmailPrioritizer(
                gmail_connector=get_gmail_connector(user_id),
                config=config,
            )
        return _prioritizer


def get_context_service():
    """Get or create cross-channel context service (thread-safe)."""
    global _context_service
    if _context_service is not None:
        return _context_service

    with _context_service_lock:
        # Double-check after acquiring lock
        if _context_service is None:
            from aragora.services.cross_channel_context import CrossChannelContextService

            _context_service = CrossChannelContextService()
        return _context_service


def get_user_config(user_id: str) -> dict[str, Any]:
    """Get user config from cache (thread-safe)."""
    with _user_configs_lock:
        return _user_configs.get(user_id, {}).copy()


def set_user_config(user_id: str, config: dict[str, Any]) -> None:
    """Set user config in cache (thread-safe)."""
    with _user_configs_lock:
        _user_configs[user_id] = config.copy()
