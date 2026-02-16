"""
Storage utilities for shared inbox handlers.

Provides lazy-initialized stores and activity logging.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from aragora.server.handlers.utils.lazy_stores import LazyStoreFactory

from .models import RoutingRule, SharedInbox, SharedInboxMessage

logger = logging.getLogger(__name__)

# =============================================================================
# Persistent Storage Access (using LazyStoreFactory)
# =============================================================================

_email_store = LazyStoreFactory(
    store_name="email_store",
    import_path="aragora.storage.email_store",
    factory_name="get_email_store",
    logger_context="SharedInbox",
)

_rules_store = LazyStoreFactory(
    store_name="rules_store",
    import_path="aragora.services.rules_store",
    factory_name="get_rules_store",
    logger_context="SharedInbox",
)

_activity_store = LazyStoreFactory(
    store_name="activity_store",
    import_path="aragora.storage.inbox_activity_store",
    factory_name="get_inbox_activity_store",
    logger_context="SharedInbox",
)


def _get_email_store():
    """Get the email store (lazy init, thread-safe)."""
    return _email_store.get()


def _get_rules_store():
    """Get the rules store (lazy init, thread-safe)."""
    return _rules_store.get()


def _get_activity_store():
    """Get the activity store (lazy init, thread-safe)."""
    return _activity_store.get()


def _log_activity(
    inbox_id: str,
    org_id: str,
    actor_id: str,
    action: str,
    target_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log an inbox activity (non-blocking helper)."""
    store = _get_activity_store()
    if store:
        try:
            from aragora.storage.inbox_activity_store import InboxActivity

            activity = InboxActivity(
                inbox_id=inbox_id,
                org_id=org_id,
                actor_id=actor_id,
                action=action,
                target_id=target_id,
                metadata=metadata or {},
            )
            store.log_activity(activity)
        except (ImportError, ValueError, TypeError, KeyError, AttributeError, OSError, RuntimeError) as e:
            logger.debug(f"[SharedInbox] Failed to log activity: {e}")


# Storage configuration
USE_PERSISTENT_STORAGE = True  # Set to False for in-memory only (testing)

# =============================================================================
# In-Memory Storage (fallback when USE_PERSISTENT_STORAGE=False)
# =============================================================================

_shared_inboxes: dict[str, SharedInbox] = {}
_inbox_messages: dict[str, dict[str, SharedInboxMessage]] = {}  # inbox_id -> {msg_id -> message}
_routing_rules: dict[str, RoutingRule] = {}
_storage_lock = threading.Lock()


def _get_store():
    """Get the persistent storage instance if enabled."""
    if not USE_PERSISTENT_STORAGE:
        return None
    return _get_email_store()
