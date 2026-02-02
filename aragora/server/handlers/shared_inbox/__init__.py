"""
Shared Inbox Handler Package.

This package contains modular components for shared inbox management:
- models.py: Data models (MessageStatus, RoutingRule, SharedInbox, etc.)
- _shared_inbox_handler.py: Main SharedInboxHandler (parent directory)

All public APIs are re-exported from this package for backward compatibility.
"""

from __future__ import annotations

from typing import Any

# Import models from the dedicated models module (no circular import risk)
from aragora.server.handlers.shared_inbox.models import (
    MessageStatus,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionField,
    RuleConditionOperator,
    RoutingRule,
    SharedInbox,
    SharedInboxMessage,
)


def __getattr__(name: str) -> Any:
    """Lazy import handler to avoid circular import with _shared_inbox_handler."""
    # Handler class and functions
    handler_exports = {
        "SharedInboxHandler",
        "handle_create_shared_inbox",
        "handle_list_shared_inboxes",
        "handle_get_shared_inbox",
        "handle_get_inbox_messages",
        "handle_assign_message",
        "handle_update_message_status",
        "handle_add_message_tag",
        "handle_add_message_to_inbox",
        "handle_create_routing_rule",
        "handle_list_routing_rules",
        "handle_update_routing_rule",
        "handle_delete_routing_rule",
        "handle_test_routing_rule",
        "apply_routing_rules_to_message",
        "get_matching_rules_for_email",
        "_shared_inboxes",
        "_inbox_messages",
        "_routing_rules",
        "_storage_lock",
        "_get_store",
        "_get_rules_store",
        "_get_activity_store",
        "_log_activity",
    }

    if name in handler_exports:
        from aragora.server.handlers import _shared_inbox_handler

        return getattr(_shared_inbox_handler, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Classes
    "SharedInboxHandler",
    "MessageStatus",
    "RuleConditionField",
    "RuleConditionOperator",
    "RuleActionType",
    "RuleCondition",
    "RuleAction",
    "RoutingRule",
    "SharedInboxMessage",
    "SharedInbox",
    # Handler functions
    "handle_create_shared_inbox",
    "handle_list_shared_inboxes",
    "handle_get_shared_inbox",
    "handle_get_inbox_messages",
    "handle_assign_message",
    "handle_update_message_status",
    "handle_add_message_tag",
    "handle_add_message_to_inbox",
    "handle_create_routing_rule",
    "handle_list_routing_rules",
    "handle_update_routing_rule",
    "handle_delete_routing_rule",
    "handle_test_routing_rule",
    "apply_routing_rules_to_message",
    "get_matching_rules_for_email",
    # Internal state (for testing)
    "_shared_inboxes",
    "_inbox_messages",
    "_routing_rules",
    "_storage_lock",
    "_get_store",
    "_get_rules_store",
    "_get_activity_store",
    "_log_activity",
]
