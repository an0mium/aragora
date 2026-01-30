"""
Shared Inbox Handler Package.

This package contains modular components for shared inbox management:
- models.py: Data models (MessageStatus, RoutingRule, SharedInbox, etc.)
- _shared_inbox_handler.py: Main SharedInboxHandler (parent directory)

All public APIs are re-exported from this package for backward compatibility.
"""

# Re-export everything from the handler module for backward compatibility
# This allows existing code using `from aragora.server.handlers.shared_inbox import X`
# to continue working after the module was split into a package.
from aragora.server.handlers._shared_inbox_handler import (
    # Classes
    SharedInboxHandler,
    MessageStatus,
    RuleConditionField,
    RuleConditionOperator,
    RuleActionType,
    RuleCondition,
    RuleAction,
    RoutingRule,
    SharedInboxMessage,
    SharedInbox,
    # Handler functions
    handle_create_shared_inbox,
    handle_list_shared_inboxes,
    handle_get_shared_inbox,
    handle_get_inbox_messages,
    handle_assign_message,
    handle_update_message_status,
    handle_add_message_tag,
    handle_add_message_to_inbox,
    handle_create_routing_rule,
    handle_list_routing_rules,
    handle_update_routing_rule,
    handle_delete_routing_rule,
    handle_test_routing_rule,
    apply_routing_rules_to_message,
    get_matching_rules_for_email,
    # Internal state (for testing)
    _shared_inboxes,
    _inbox_messages,
    _routing_rules,
    _storage_lock,
    _get_store,
    _get_rules_store,
    _get_activity_store,
    _log_activity,
)

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
