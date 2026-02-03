"""
Input validation for Shared Inbox.

Contains validation functions and constants for routing rules, inboxes, and messages.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from aragora.server.validation.security import (
    is_safe_regex_pattern,
    sanitize_user_input,
)

from .models import (
    RuleActionType,
    RuleConditionField,
    RuleConditionOperator,
    RoutingRule,
)

# =============================================================================
# Input Validation Constants
# =============================================================================

# Allowed email header fields for routing rule conditions (whitelist)
# These are standard email fields that are safe to use for routing
ALLOWED_RULE_CONDITION_FIELDS = {
    RuleConditionField.FROM,
    RuleConditionField.TO,
    RuleConditionField.SUBJECT,
    RuleConditionField.BODY,
    RuleConditionField.LABELS,
    RuleConditionField.PRIORITY,
    RuleConditionField.SENDER_DOMAIN,
}

# Operators that require regex validation
REGEX_OPERATORS = {
    RuleConditionOperator.MATCHES,
}

# Maximum lengths for input validation
MAX_RULE_NAME_LENGTH = 200
MAX_RULE_DESCRIPTION_LENGTH = 1000
MAX_CONDITION_VALUE_LENGTH = 500
MAX_REGEX_PATTERN_LENGTH = 200
MAX_TAG_LENGTH = 100
MAX_INBOX_NAME_LENGTH = 200
MAX_INBOX_DESCRIPTION_LENGTH = 1000
MAX_CONDITIONS_PER_RULE = 20
MAX_ACTIONS_PER_RULE = 10
MAX_RULES_PER_WORKSPACE = 500

# Rate limiting configuration for rule creation
RULE_RATE_LIMIT_WINDOW_SECONDS = 60  # 1 minute window
RULE_RATE_LIMIT_MAX_REQUESTS = 10  # Max 10 rule creations per minute per workspace


# =============================================================================
# Rate Limiting for Rule Creation
# =============================================================================


@dataclass
class RateLimitEntry:
    """Entry for rate limiting tracking."""

    timestamps: list[float]


class RuleRateLimiter:
    """
    Rate limiter for routing rule creation to prevent abuse.

    Implements a sliding window rate limiter with configurable
    window size and max requests.
    """

    def __init__(
        self,
        window_seconds: float = RULE_RATE_LIMIT_WINDOW_SECONDS,
        max_requests: int = RULE_RATE_LIMIT_MAX_REQUESTS,
    ):
        self._window_seconds = window_seconds
        self._max_requests = max_requests
        self._entries: dict[str, RateLimitEntry] = defaultdict(
            lambda: RateLimitEntry(timestamps=[])
        )
        self._lock = threading.Lock()

    def is_allowed(self, workspace_id: str) -> tuple[bool, int]:
        """
        Check if a rule creation request is allowed.

        Args:
            workspace_id: The workspace ID making the request

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        now = time.time()
        cutoff = now - self._window_seconds

        with self._lock:
            entry = self._entries[workspace_id]
            # Remove old timestamps outside the window
            entry.timestamps = [ts for ts in entry.timestamps if ts > cutoff]

            remaining = self._max_requests - len(entry.timestamps)
            if remaining <= 0:
                return False, 0

            return True, remaining

    def record_request(self, workspace_id: str) -> None:
        """Record a rule creation request."""
        now = time.time()
        with self._lock:
            self._entries[workspace_id].timestamps.append(now)

    def get_retry_after(self, workspace_id: str) -> float:
        """Get seconds until next request is allowed."""
        now = time.time()
        cutoff = now - self._window_seconds

        with self._lock:
            entry = self._entries[workspace_id]
            valid_timestamps = [ts for ts in entry.timestamps if ts > cutoff]

            if len(valid_timestamps) < self._max_requests:
                return 0.0

            # Return time until oldest timestamp expires
            oldest = min(valid_timestamps)
            return max(0.0, (oldest + self._window_seconds) - now)


# Global rate limiter instance
_rule_rate_limiter = RuleRateLimiter()


def get_rule_rate_limiter() -> RuleRateLimiter:
    """Get the global rule rate limiter instance."""
    return _rule_rate_limiter


# =============================================================================
# Routing Rule Validation
# =============================================================================


@dataclass
class RuleValidationResult:
    """Result of routing rule validation."""

    is_valid: bool
    error: str | None = None
    sanitized_conditions: list[dict[str, Any]] | None = None
    sanitized_actions: list[dict[str, Any]] | None = None


def validate_safe_regex(
    pattern: str, max_length: int = MAX_REGEX_PATTERN_LENGTH
) -> tuple[bool, str | None]:
    """
    Validate that a regex pattern is safe from ReDoS attacks.

    Args:
        pattern: The regex pattern to validate
        max_length: Maximum allowed pattern length

    Returns:
        Tuple of (is_safe, error_message)
    """
    if not pattern:
        return False, "Empty regex pattern"

    if len(pattern) > max_length:
        return False, f"Pattern exceeds maximum length of {max_length} characters"

    # Use the centralized security validation
    is_safe, error = is_safe_regex_pattern(pattern)
    if not is_safe:
        return False, error

    # Additional check: try to compile the pattern
    try:
        re.compile(pattern)
    except re.error as e:
        return False, f"Invalid regex syntax: {e}"

    return True, None


def validate_rule_condition_field(field_value: str) -> tuple[bool, str | None]:
    """
    Validate that a rule condition field is in the allowed whitelist.

    Args:
        field_value: The field value to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        field = RuleConditionField(field_value)
        if field not in ALLOWED_RULE_CONDITION_FIELDS:
            return False, f"Field '{field_value}' is not allowed for routing rules"
        return True, None
    except ValueError:
        allowed = ", ".join(f.value for f in ALLOWED_RULE_CONDITION_FIELDS)
        return False, f"Invalid field '{field_value}'. Allowed fields: {allowed}"


def validate_rule_condition(
    condition: dict[str, Any],
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """
    Validate a single routing rule condition.

    Performs comprehensive validation including:
    - Field whitelist check
    - Operator validation
    - Regex safety check for MATCHES operator
    - Value length and sanitization

    Args:
        condition: The condition dictionary to validate

    Returns:
        Tuple of (is_valid, error_message, sanitized_condition)
    """
    # Check required fields
    if not isinstance(condition, dict):
        return False, "Condition must be a dictionary", None

    field = condition.get("field")
    operator = condition.get("operator")
    value = condition.get("value")

    if not field:
        return False, "Condition missing 'field'", None
    if not operator:
        return False, "Condition missing 'operator'", None
    if value is None:
        return False, "Condition missing 'value'", None

    # Validate field against whitelist
    is_valid, error = validate_rule_condition_field(field)
    if not is_valid:
        return False, error, None

    # Validate operator
    try:
        op = RuleConditionOperator(operator)
    except ValueError:
        allowed_ops = ", ".join(o.value for o in RuleConditionOperator)
        return False, f"Invalid operator '{operator}'. Allowed: {allowed_ops}", None

    # Validate and sanitize value
    if not isinstance(value, str):
        return False, "Condition value must be a string", None

    if len(value) > MAX_CONDITION_VALUE_LENGTH:
        return (
            False,
            f"Condition value exceeds maximum length of {MAX_CONDITION_VALUE_LENGTH}",
            None,
        )

    # For regex operators, validate the pattern
    if op in REGEX_OPERATORS:
        is_safe, regex_error = validate_safe_regex(value)
        if not is_safe:
            return False, f"Unsafe regex pattern: {regex_error}", None

    # Sanitize the value (remove control characters, normalize whitespace)
    sanitized_value = sanitize_user_input(value, max_length=MAX_CONDITION_VALUE_LENGTH)

    return (
        True,
        None,
        {
            "field": field,
            "operator": operator,
            "value": sanitized_value,
        },
    )


def validate_rule_action(action: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any] | None]:
    """
    Validate a single routing rule action.

    Args:
        action: The action dictionary to validate

    Returns:
        Tuple of (is_valid, error_message, sanitized_action)
    """
    if not isinstance(action, dict):
        return False, "Action must be a dictionary", None

    action_type = action.get("type")
    target = action.get("target")
    params = action.get("params", {})

    if not action_type:
        return False, "Action missing 'type'", None

    # Validate action type
    try:
        RuleActionType(action_type)
    except ValueError:
        allowed_types = ", ".join(t.value for t in RuleActionType)
        return False, f"Invalid action type '{action_type}'. Allowed: {allowed_types}", None

    # Validate target if provided
    sanitized_target = None
    if target is not None:
        if not isinstance(target, str):
            return False, "Action target must be a string", None
        if len(target) > 200:
            return False, "Action target exceeds maximum length of 200", None
        sanitized_target = sanitize_user_input(target, max_length=200)

    # Validate params if provided
    if not isinstance(params, dict):
        return False, "Action params must be a dictionary", None

    return (
        True,
        None,
        {
            "type": action_type,
            "target": sanitized_target,
            "params": params,
        },
    )


def detect_circular_routing(
    new_rule_actions: list[dict[str, Any]],
    existing_rules: list[RoutingRule],
    workspace_id: str,
) -> tuple[bool, str | None]:
    """
    Detect potential circular routing in rules.

    Circular routing can occur when:
    - Rule A forwards to inbox B
    - Rule B (in inbox B) forwards back to inbox A

    This creates an infinite loop of message routing.

    Args:
        new_rule_actions: Actions in the new/updated rule
        existing_rules: All existing rules in the workspace
        workspace_id: The workspace ID

    Returns:
        Tuple of (has_circular_routing, error_message)
    """
    # Build a graph of forward actions
    # forward_graph[source_inbox] = set of target_inboxes
    forward_graph: dict[str, set[str]] = defaultdict(set)

    # Add existing forward actions to the graph
    for rule in existing_rules:
        if rule.workspace_id != workspace_id or not rule.enabled:
            continue

        # Get the source inbox (if rule is inbox-specific) or "global"
        source = getattr(rule, "inbox_id", None) or "global"

        for action in rule.actions:
            if action.type.value == "forward" and action.target:
                forward_graph[source].add(action.target)

    # Check new rule's forward actions for cycles
    for action in new_rule_actions:
        if action.get("type") == "forward" and action.get("target"):
            target = action["target"]

            # Check if this creates a cycle using BFS
            visited = set()
            queue = [target]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                # Check if we can reach any inbox that forwards to us
                # (simplified: check if target eventually routes back)
                for next_target in forward_graph.get(current, set()):
                    if next_target == target or next_target == "global":
                        return (
                            True,
                            f"Circular routing detected: forwarding to '{target}' may create a routing loop",
                        )
                    queue.append(next_target)

    return False, None


def validate_routing_rule(
    name: str,
    conditions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    workspace_id: str,
    description: str | None = None,
    existing_rules: list[RoutingRule] | None = None,
    check_circular: bool = True,
) -> RuleValidationResult:
    """
    Comprehensive validation for routing rules.

    Validates:
    - Rule name length and format
    - Description length
    - Conditions (field whitelist, operator, regex safety)
    - Actions (type, target)
    - Circular routing detection
    - Resource limits (max conditions/actions per rule)

    Args:
        name: Rule name
        conditions: List of condition dictionaries
        actions: List of action dictionaries
        workspace_id: Workspace ID for circular routing check
        description: Optional rule description
        existing_rules: Existing rules for circular routing detection
        check_circular: Whether to check for circular routing

    Returns:
        RuleValidationResult with validation status and sanitized data
    """
    # Validate name
    if not name:
        return RuleValidationResult(is_valid=False, error="Rule name is required")

    if len(name) > MAX_RULE_NAME_LENGTH:
        return RuleValidationResult(
            is_valid=False, error=f"Rule name exceeds maximum length of {MAX_RULE_NAME_LENGTH}"
        )

    sanitized_name = sanitize_user_input(name, max_length=MAX_RULE_NAME_LENGTH)
    if not sanitized_name:
        return RuleValidationResult(
            is_valid=False, error="Rule name cannot be empty after sanitization"
        )

    # Validate description
    if description and len(description) > MAX_RULE_DESCRIPTION_LENGTH:
        return RuleValidationResult(
            is_valid=False,
            error=f"Description exceeds maximum length of {MAX_RULE_DESCRIPTION_LENGTH}",
        )

    # Validate conditions count
    if not conditions:
        return RuleValidationResult(is_valid=False, error="At least one condition is required")

    if len(conditions) > MAX_CONDITIONS_PER_RULE:
        return RuleValidationResult(
            is_valid=False,
            error=f"Number of conditions ({len(conditions)}) exceeds maximum of {MAX_CONDITIONS_PER_RULE}",
        )

    # Validate each condition
    sanitized_conditions = []
    for i, condition in enumerate(conditions):
        is_valid, error, sanitized = validate_rule_condition(condition)
        if not is_valid:
            return RuleValidationResult(is_valid=False, error=f"Condition {i + 1}: {error}")
        sanitized_conditions.append(sanitized)

    # Validate actions count
    if not actions:
        return RuleValidationResult(is_valid=False, error="At least one action is required")

    if len(actions) > MAX_ACTIONS_PER_RULE:
        return RuleValidationResult(
            is_valid=False,
            error=f"Number of actions ({len(actions)}) exceeds maximum of {MAX_ACTIONS_PER_RULE}",
        )

    # Validate each action
    sanitized_actions = []
    for i, action in enumerate(actions):
        is_valid, error, sanitized = validate_rule_action(action)
        if not is_valid:
            return RuleValidationResult(is_valid=False, error=f"Action {i + 1}: {error}")
        sanitized_actions.append(sanitized)

    # Check for circular routing
    if check_circular and existing_rules is not None:
        has_circular, circular_error = detect_circular_routing(
            sanitized_actions, existing_rules, workspace_id
        )
        if has_circular:
            return RuleValidationResult(is_valid=False, error=circular_error)

    return RuleValidationResult(
        is_valid=True,
        sanitized_conditions=sanitized_conditions,
        sanitized_actions=sanitized_actions,
    )


def validate_inbox_input(
    name: str,
    description: str | None = None,
    email_address: str | None = None,
) -> tuple[bool, str | None]:
    """
    Validate shared inbox creation/update inputs.

    Args:
        name: Inbox name
        description: Optional description
        email_address: Optional email address

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Inbox name is required"

    if len(name) > MAX_INBOX_NAME_LENGTH:
        return False, f"Inbox name exceeds maximum length of {MAX_INBOX_NAME_LENGTH}"

    if description and len(description) > MAX_INBOX_DESCRIPTION_LENGTH:
        return False, f"Description exceeds maximum length of {MAX_INBOX_DESCRIPTION_LENGTH}"

    # Basic email validation if provided
    if email_address:
        # Simple check - must contain @ and have parts before and after
        if "@" not in email_address or len(email_address.split("@")) != 2:
            return False, "Invalid email address format"
        local, domain = email_address.split("@")
        if not local or not domain or "." not in domain:
            return False, "Invalid email address format"

    return True, None


def validate_tag(tag: str) -> tuple[bool, str | None]:
    """
    Validate a message tag.

    Args:
        tag: The tag to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not tag:
        return False, "Tag cannot be empty"

    if len(tag) > MAX_TAG_LENGTH:
        return False, f"Tag exceeds maximum length of {MAX_TAG_LENGTH}"

    # Tags should only contain alphanumeric, hyphen, underscore
    if not re.match(r"^[\w\-]+$", tag):
        return False, "Tag can only contain letters, numbers, hyphens, and underscores"

    return True, None
