"""
Shared Inbox Handler Package.

This package contains modular components for shared inbox management:
- models.py: Data models (MessageStatus, RoutingRule, SharedInbox, etc.)

The main handler remains in shared_inbox.py (parent directory) for backward
compatibility. Future refactoring will move handlers into this package.
"""

from .models import (
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

__all__ = [
    "MessageStatus",
    "RuleAction",
    "RuleActionType",
    "RuleCondition",
    "RuleConditionField",
    "RuleConditionOperator",
    "RoutingRule",
    "SharedInbox",
    "SharedInboxMessage",
]
