"""
Routing rules engine for Shared Inbox.

Contains rule evaluation and matching logic for email routing.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Protocol

from aragora.server.validation.security import (
    execute_regex_with_timeout,
    REGEX_TIMEOUT_SECONDS,
)

from .models import (
    MessageStatus,
    RuleConditionField,
    RuleConditionOperator,
    RoutingRule,
    SharedInboxMessage,
)
from .storage import (
    _get_rules_store,
    _routing_rules,
    _shared_inboxes,
    _inbox_messages,
    _storage_lock,
)

logger = logging.getLogger(__name__)


class MessageLike(Protocol):
    """Protocol for message-like objects that can be evaluated by routing rules."""

    from_address: str
    to_addresses: list[str]
    subject: str
    priority: str | None


def _evaluate_rule(rule: RoutingRule, message: MessageLike) -> bool:
    """
    Evaluate if a routing rule matches a message.

    Uses timeout-protected regex execution to prevent ReDoS attacks
    even if a malicious pattern bypasses validation.
    """
    results = []

    for condition in rule.conditions:
        value = ""
        if condition.field == RuleConditionField.FROM:
            value = message.from_address.lower()
        elif condition.field == RuleConditionField.TO:
            value = " ".join(message.to_addresses).lower()
        elif condition.field == RuleConditionField.SUBJECT:
            value = message.subject.lower()
        elif condition.field == RuleConditionField.SENDER_DOMAIN:
            value = (
                message.from_address.split("@")[-1].lower() if "@" in message.from_address else ""
            )
        elif condition.field == RuleConditionField.PRIORITY:
            value = message.priority or ""

        condition_value = condition.value.lower()
        matched = False

        if condition.operator == RuleConditionOperator.CONTAINS:
            matched = condition_value in value
        elif condition.operator == RuleConditionOperator.EQUALS:
            matched = value == condition_value
        elif condition.operator == RuleConditionOperator.STARTS_WITH:
            matched = value.startswith(condition_value)
        elif condition.operator == RuleConditionOperator.ENDS_WITH:
            matched = value.endswith(condition_value)
        elif condition.operator == RuleConditionOperator.MATCHES:
            # Use timeout-protected regex execution to prevent ReDoS
            # This is a defense-in-depth measure even though we validate
            # patterns at creation time
            match = execute_regex_with_timeout(
                pattern=condition_value,
                text=value,
                timeout=REGEX_TIMEOUT_SECONDS,
                flags=re.IGNORECASE,
            )
            matched = match is not None
            if match is None:
                # Log if regex timed out (potential ReDoS attempt)
                logger.warning(
                    f"[SharedInbox] Regex evaluation timed out or failed for rule "
                    f"condition pattern: {condition_value[:50]}..."
                )
        results.append(matched)

    if rule.condition_logic == "AND":
        return all(results) if results else False
    else:  # OR
        return any(results) if results else False


async def get_matching_rules_for_email(
    inbox_id: str,
    email_data: dict[str, Any],
    workspace_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Get all matching routing rules for an email message.

    This function queries the RulesStore to find enabled rules that match
    the given email data, sorted by priority.

    Args:
        inbox_id: The inbox ID to get rules for
        email_data: Email data dictionary with keys:
            - from_address: Sender email address
            - to_addresses: List of recipient addresses
            - subject: Email subject line
            - snippet: Email body preview
            - priority: Email priority level (optional)
        workspace_id: Optional workspace filter

    Returns:
        List of matching rule dictionaries sorted by priority (ascending)
    """
    matching_rules = []
    store_rules: list[dict[str, Any]] = []

    # Try RulesStore first (primary persistent storage)
    rules_store = _get_rules_store()
    if rules_store:
        try:
            store_rules = rules_store.get_matching_rules(
                inbox_id=inbox_id,
                email_data=email_data,
                workspace_id=workspace_id,
            )
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            logger.warning(f"[SharedInbox] Failed to get matching rules from RulesStore: {e}")

    # Fallback to in-memory evaluation
    with _storage_lock:
        # Get all enabled rules for this workspace/inbox
        rules = [
            rule
            for rule in _routing_rules.values()
            if rule.enabled and (workspace_id is None or rule.workspace_id == workspace_id)
        ]
        # Sort by priority
        rules.sort(key=lambda r: r.priority)

        # Create a message-like object for evaluation
        class _EmailMessage:
            def __init__(self, data: dict[str, Any]):
                self.from_address = data.get("from_address", "")
                self.to_addresses = data.get("to_addresses", [])
                self.subject = data.get("subject", "")
                self.snippet = data.get("snippet", "")
                self.priority = data.get("priority")

        msg = _EmailMessage(email_data)

        for rule in rules:
            if _evaluate_rule(rule, msg):
                matching_rules.append(rule.to_dict())

    if store_rules:
        if not matching_rules:
            return store_rules
        combined: dict[str, dict[str, Any]] = {r.get("id", ""): r for r in store_rules}
        for rule in matching_rules:
            rule_id = rule.get("id", "")
            combined[rule_id] = rule
        merged = list(combined.values())
        merged.sort(key=lambda r: r.get("priority", 0))
        return merged

    return matching_rules


async def apply_routing_rules_to_message(
    inbox_id: str,
    message: SharedInboxMessage,
    workspace_id: str,
) -> dict[str, Any]:
    """
    Apply matching routing rules to a message.

    Args:
        inbox_id: The inbox containing the message
        message: The message to apply rules to
        workspace_id: The workspace ID

    Returns:
        Dictionary with applied actions and any changes made
    """
    # Build email data from message
    email_data = {
        "from_address": message.from_address,
        "to_addresses": message.to_addresses,
        "subject": message.subject,
        "snippet": message.snippet,
        "priority": message.priority,
    }

    # Get matching rules
    matching_rules = await get_matching_rules_for_email(
        inbox_id=inbox_id,
        email_data=email_data,
        workspace_id=workspace_id,
    )

    if not matching_rules:
        return {"applied": False, "rules_matched": 0, "actions": []}

    applied_actions = []
    changes_made = {}

    for rule in matching_rules:
        actions = rule.get("actions", [])
        for action in actions:
            action_type = action.get("type")
            target = action.get("target")
            _ = action.get("params", {})  # Reserved for future action params

            if action_type == "assign" and target:
                message.assigned_to = target
                message.assigned_at = datetime.now(timezone.utc)
                if message.status == MessageStatus.OPEN:
                    message.status = MessageStatus.ASSIGNED
                changes_made["assigned_to"] = target
                applied_actions.append({"type": "assign", "target": target})

            elif action_type == "label" and target:
                if target not in message.tags:
                    message.tags.append(target)
                applied_actions.append({"type": "label", "target": target})

            elif action_type == "escalate":
                message.priority = "high"
                changes_made["priority"] = "high"
                applied_actions.append({"type": "escalate"})

            elif action_type == "archive":
                message.status = MessageStatus.CLOSED
                changes_made["status"] = "closed"
                applied_actions.append({"type": "archive"})

        # Update rule stats
        rules_store = _get_rules_store()
        if rules_store:
            try:
                rules_store.increment_rule_stats(rule["id"], matched=0, applied=1)
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.debug(f"Failed to increment rule stats for {rule['id']}: {e}")

    return {
        "applied": bool(applied_actions),
        "rules_matched": len(matching_rules),
        "actions": applied_actions,
        "changes": changes_made,
    }


def evaluate_rule_for_test(
    rule: RoutingRule,
    workspace_id: str,
) -> int:
    """
    Evaluate a routing rule against existing messages for testing.

    Args:
        rule: The routing rule to test
        workspace_id: The workspace ID

    Returns:
        Count of matching messages
    """
    match_count = 0
    with _storage_lock:
        for inbox_id, messages in _inbox_messages.items():
            inbox = _shared_inboxes.get(inbox_id)
            if inbox and inbox.workspace_id == workspace_id:
                for message in messages.values():
                    if _evaluate_rule(rule, message):
                        match_count += 1
    return match_count
