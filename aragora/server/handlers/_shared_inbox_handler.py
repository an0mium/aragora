"""
HTTP API Handlers for Shared Inbox Management.

Provides REST APIs for collaborative email inbox management:
- Create and manage shared inboxes
- Message assignment and ownership
- Team-based inbox views
- Status tracking (open, assigned, resolved)
- Tagging and labeling

Endpoints:
- POST /api/v1/inbox/shared - Create shared inbox
- GET /api/v1/inbox/shared - List shared inboxes
- GET /api/v1/inbox/shared/:id - Get shared inbox details
- GET /api/v1/inbox/shared/:id/messages - Get messages in inbox
- POST /api/v1/inbox/shared/:id/messages/:msg_id/assign - Assign message
- POST /api/v1/inbox/shared/:id/messages/:msg_id/status - Update status
- POST /api/v1/inbox/shared/:id/messages/:msg_id/tag - Add tag
- POST /api/v1/inbox/routing/rules - Create routing rule
- GET /api/v1/inbox/routing/rules - List routing rules
- PATCH /api/v1/inbox/routing/rules/:id - Update routing rule
- DELETE /api/v1/inbox/routing/rules/:id - Delete routing rule
- POST /api/v1/inbox/routing/rules/:id/test - Test routing rule
"""
# mypy: disable-error-code="assignment,attr-defined,index"
# RuleAction/RuleConditionOperator type handling is dynamic

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)
from aragora.server.handlers.utils.lazy_stores import LazyStoreFactory
from aragora.observability.metrics import track_handler
from aragora.rbac.decorators import require_permission
from aragora.server.validation.query_params import safe_query_int
from aragora.server.validation.security import (
    is_safe_regex_pattern,
    execute_regex_with_timeout,
    sanitize_user_input,
    REGEX_TIMEOUT_SECONDS,
)

# Import models from the models module to avoid duplication
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

logger = logging.getLogger(__name__)

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


def _get_email_store() -> Any | None:
    """Get the email store (lazy init, thread-safe)."""
    return _email_store.get()


def _get_rules_store() -> Any | None:
    """Get the rules store (lazy init, thread-safe)."""
    return _rules_store.get()


def _get_activity_store() -> Any | None:
    """Get the activity store (lazy init, thread-safe)."""
    return _activity_store.get()


def _log_activity(
    inbox_id: str,
    org_id: str,
    actor_id: str,
    action: str,
    target_id: str | None = None,
    metadata: Optional[dict[str, Any]] = None,
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
        except Exception as e:
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


def _get_store() -> Any | None:
    """Get the persistent storage instance if enabled."""
    if not USE_PERSISTENT_STORAGE:
        return None
    return _get_email_store()


# =============================================================================
# Shared Inbox Handlers
# =============================================================================


@track_handler("inbox/shared/create")
async def handle_create_shared_inbox(
    workspace_id: str,
    name: str,
    description: str | None = None,
    email_address: str | None = None,
    connector_type: str | None = None,
    team_members: Optional[list[str]] = None,
    admins: Optional[list[str]] = None,
    settings: Optional[dict[str, Any]] = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    """
    Create a new shared inbox.

    POST /api/v1/inbox/shared
    {
        "workspace_id": "ws_123",
        "name": "Support Inbox",
        "description": "Customer support emails",
        "email_address": "support@company.com",
        "connector_type": "gmail",
        "team_members": ["user1", "user2"],
        "admins": ["admin1"]
    }
    """
    try:
        inbox_id = f"inbox_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        inbox = SharedInbox(
            id=inbox_id,
            workspace_id=workspace_id,
            name=name,
            description=description,
            email_address=email_address,
            connector_type=connector_type,
            team_members=team_members or [],
            admins=admins or [],
            settings=settings or {},
            created_at=now,
            updated_at=now,
            created_by=created_by,
        )

        # Persist to store if available
        store = _get_store()
        if store:
            try:
                store.create_shared_inbox(
                    inbox_id=inbox_id,
                    workspace_id=workspace_id,
                    name=name,
                    description=description,
                    email_address=email_address,
                    connector_type=connector_type,
                    team_members=team_members or [],
                    admins=admins or [],
                    settings=settings or {},
                    created_by=created_by,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to persist inbox to store: {e}")

        # Always keep in-memory cache for fast reads
        with _storage_lock:
            _shared_inboxes[inbox_id] = inbox
            _inbox_messages[inbox_id] = {}

        logger.info(f"[SharedInbox] Created inbox {inbox_id}: {name}")

        return {
            "success": True,
            "inbox": inbox.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to create shared inbox: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@track_handler("inbox/shared/list", method="GET")
async def handle_list_shared_inboxes(
    workspace_id: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    List shared inboxes the user has access to.

    GET /api/v1/inbox/shared?workspace_id=ws_123
    """
    try:
        # Try persistent store first
        store = _get_store()
        if store:
            try:
                stored_inboxes = store.list_shared_inboxes(workspace_id, user_id)
                if stored_inboxes:
                    # Update in-memory cache
                    for inbox_data in stored_inboxes:
                        inbox_id = inbox_data.get("id")
                        if inbox_id and inbox_id not in _shared_inboxes:
                            # Reconstruct SharedInbox object for cache
                            with _storage_lock:
                                if inbox_id not in _inbox_messages:
                                    _inbox_messages[inbox_id] = {}
                    return {
                        "success": True,
                        "inboxes": stored_inboxes,
                        "total": len(stored_inboxes),
                    }
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to load from store, using cache: {e}")

        # Fallback to in-memory
        with _storage_lock:
            inboxes = [
                inbox.to_dict()
                for inbox in _shared_inboxes.values()
                if inbox.workspace_id == workspace_id
                and (user_id is None or user_id in inbox.team_members or user_id in inbox.admins)
            ]

        return {
            "success": True,
            "inboxes": inboxes,
            "total": len(inboxes),
        }

    except Exception as e:
        logger.exception(f"Failed to list shared inboxes: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_shared_inbox(
    inbox_id: str,
) -> dict[str, Any]:
    """
    Get shared inbox details.

    GET /api/v1/inbox/shared/:id
    """
    try:
        # Try persistent store first
        store = _get_store()
        if store:
            try:
                inbox_data = store.get_shared_inbox(inbox_id)
                if inbox_data:
                    return {
                        "success": True,
                        "inbox": inbox_data,
                    }
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to load from store: {e}")

        # Fallback to in-memory
        with _storage_lock:
            inbox = _shared_inboxes.get(inbox_id)
            if not inbox:
                return {"success": False, "error": "Inbox not found"}

            # Update counts
            messages = _inbox_messages.get(inbox_id, {})
            inbox.message_count = len(messages)
            inbox.unread_count = sum(1 for m in messages.values() if m.status == MessageStatus.OPEN)

            return {
                "success": True,
                "inbox": inbox.to_dict(),
            }

    except Exception as e:
        logger.exception(f"Failed to get shared inbox: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@track_handler("inbox/shared/messages", method="GET")
async def handle_get_inbox_messages(
    inbox_id: str,
    status: str | None = None,
    assigned_to: str | None = None,
    tag: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Get messages in a shared inbox.

    GET /api/v1/inbox/shared/:id/messages
    Query params: status, assigned_to, tag, limit, offset
    """
    try:
        # Try persistent store first
        store = _get_store()
        if store:
            try:
                messages_data = store.get_inbox_messages(
                    inbox_id=inbox_id,
                    status=status,
                    assigned_to=assigned_to,
                    limit=limit,
                    offset=offset,
                )
                if messages_data is not None:
                    # Apply tag filter (not in store query)
                    if tag:
                        messages_data = [m for m in messages_data if tag in m.get("tags", [])]

                    # Get total count by querying with no limit
                    # This is more accurate than returning len(messages_data)
                    total_count = len(messages_data)
                    if len(messages_data) == limit or offset > 0:
                        # There might be more messages - query for total
                        try:
                            all_messages = store.get_inbox_messages(
                                inbox_id=inbox_id,
                                status=status,
                                assigned_to=assigned_to,
                                limit=10000,  # Large limit to get all
                                offset=0,
                            )
                            if all_messages is not None:
                                if tag:
                                    all_messages = [
                                        m for m in all_messages if tag in m.get("tags", [])
                                    ]
                                total_count = len(all_messages)
                        except (OSError, RuntimeError, ValueError, KeyError) as e:
                            # Fall back to page count + offset
                            logger.debug(f"Error counting messages, using fallback: {e}")
                            total_count = offset + len(messages_data)

                    return {
                        "success": True,
                        "messages": messages_data,
                        "total": total_count,
                        "limit": limit,
                        "offset": offset,
                    }
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to load messages from store: {e}")

        # Fallback to in-memory
        with _storage_lock:
            if inbox_id not in _inbox_messages:
                return {"success": False, "error": "Inbox not found"}

            messages = list(_inbox_messages[inbox_id].values())

        # Filter
        if status:
            messages = [m for m in messages if m.status.value == status]
        if assigned_to:
            messages = [m for m in messages if m.assigned_to == assigned_to]
        if tag:
            messages = [m for m in messages if tag in m.tags]

        # Sort by received_at descending
        messages.sort(key=lambda m: m.received_at, reverse=True)

        # Paginate
        total = len(messages)
        messages = messages[offset : offset + limit]

        return {
            "success": True,
            "messages": [m.to_dict() for m in messages],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.exception(f"Failed to get inbox messages: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@track_handler("inbox/shared/messages/assign")
async def handle_assign_message(
    inbox_id: str,
    message_id: str,
    assigned_to: str,
    assigned_by: str | None = None,
    org_id: str | None = None,
) -> dict[str, Any]:
    """
    Assign a message to a team member.

    POST /api/v1/inbox/shared/:id/messages/:msg_id/assign
    {
        "assigned_to": "user_123"
    }
    """
    try:
        now = datetime.now(timezone.utc)
        new_status = None
        previous_assignee = None

        with _storage_lock:
            messages = _inbox_messages.get(inbox_id, {})
            message = messages.get(message_id)

            if not message:
                return {"success": False, "error": "Message not found"}

            previous_assignee = message.assigned_to
            message.assigned_to = assigned_to
            message.assigned_at = now
            if message.status == MessageStatus.OPEN:
                message.status = MessageStatus.ASSIGNED
                new_status = MessageStatus.ASSIGNED.value

        # Persist to store if available
        store = _get_store()
        if store:
            try:
                updates = {
                    "assigned_to": assigned_to,
                    "assigned_at": now.isoformat(),
                }
                if new_status:
                    updates["status"] = new_status
                store.update_message(message_id, updates)
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to persist assignment to store: {e}")

        logger.info(f"[SharedInbox] Assigned message {message_id} to {assigned_to}")

        # Log activity
        if org_id:
            action = "reassigned" if previous_assignee else "assigned"
            _log_activity(
                inbox_id=inbox_id,
                org_id=org_id,
                actor_id=assigned_by or "system",
                action=action,
                target_id=message_id,
                metadata={
                    "assignee_id": assigned_to,
                    "previous_assignee": previous_assignee,
                },
            )

        return {
            "success": True,
            "message": message.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to assign message: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_update_message_status(
    inbox_id: str,
    message_id: str,
    status: str,
    updated_by: str | None = None,
    org_id: str | None = None,
) -> dict[str, Any]:
    """
    Update message status.

    POST /api/v1/inbox/shared/:id/messages/:msg_id/status
    {
        "status": "resolved"
    }
    """
    try:
        now = datetime.now(timezone.utc)
        is_resolved = False
        previous_status = None

        with _storage_lock:
            messages = _inbox_messages.get(inbox_id, {})
            message = messages.get(message_id)

            if not message:
                return {"success": False, "error": "Message not found"}

            previous_status = message.status.value
            message.status = MessageStatus(status)

            if message.status == MessageStatus.RESOLVED:
                message.resolved_at = now
                message.resolved_by = updated_by
                is_resolved = True

        # Persist to store if available
        store = _get_store()
        if store:
            try:
                updates = {"status": status}
                if is_resolved:
                    updates["resolved_at"] = now.isoformat()
                    updates["resolved_by"] = updated_by
                store.update_message(message_id, updates)
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to persist status to store: {e}")

        logger.info(f"[SharedInbox] Updated message {message_id} status to {status}")

        # Log activity
        if org_id:
            _log_activity(
                inbox_id=inbox_id,
                org_id=org_id,
                actor_id=updated_by or "system",
                action="status_changed",
                target_id=message_id,
                metadata={
                    "from_status": previous_status,
                    "to_status": status,
                },
            )

        return {
            "success": True,
            "message": message.to_dict(),
        }

    except ValueError:
        return {"success": False, "error": f"Invalid status: {status}"}
    except Exception as e:
        logger.exception(f"Failed to update message status: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_add_message_tag(
    inbox_id: str,
    message_id: str,
    tag: str,
    added_by: str | None = None,
    org_id: str | None = None,
) -> dict[str, Any]:
    """
    Add a tag to a message.

    POST /api/v1/inbox/shared/:id/messages/:msg_id/tag
    {
        "tag": "urgent"
    }
    """
    try:
        tag_added = False
        with _storage_lock:
            messages = _inbox_messages.get(inbox_id, {})
            message = messages.get(message_id)

            if not message:
                return {"success": False, "error": "Message not found"}

            if tag not in message.tags:
                message.tags.append(tag)
                tag_added = True

        # Log activity if tag was actually added
        if tag_added and org_id:
            _log_activity(
                inbox_id=inbox_id,
                org_id=org_id,
                actor_id=added_by or "system",
                action="tag_added",
                target_id=message_id,
                metadata={"tag": tag},
            )

        return {
            "success": True,
            "message": message.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to add message tag: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_add_message_to_inbox(
    inbox_id: str,
    email_id: str,
    subject: str,
    from_address: str,
    to_addresses: list[str],
    snippet: str,
    received_at: datetime | None = None,
    thread_id: str | None = None,
    priority: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """
    Add a message to a shared inbox (used by sync/routing).

    Internal API - not directly exposed to users.
    """
    try:
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        actual_received_at = received_at or datetime.now(timezone.utc)

        message = SharedInboxMessage(
            id=message_id,
            inbox_id=inbox_id,
            email_id=email_id,
            subject=subject,
            from_address=from_address,
            to_addresses=to_addresses,
            snippet=snippet,
            received_at=actual_received_at,
            thread_id=thread_id,
            priority=priority,
        )

        # Persist to store if available
        store = _get_store()
        if store:
            try:
                # Get workspace_id from inbox if not provided
                if not workspace_id:
                    inbox_data = store.get_shared_inbox(inbox_id)
                    workspace_id = (
                        inbox_data.get("workspace_id", "default") if inbox_data else "default"
                    )

                store.save_message(
                    message_id=message_id,
                    inbox_id=inbox_id,
                    workspace_id=workspace_id,
                    email_id=email_id,
                    subject=subject,
                    from_address=from_address,
                    to_addresses=to_addresses,
                    snippet=snippet,
                    received_at=actual_received_at,
                    thread_id=thread_id,
                    priority=priority,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to persist message to store: {e}")

        # Keep in-memory cache
        with _storage_lock:
            if inbox_id not in _inbox_messages:
                _inbox_messages[inbox_id] = {}

            _inbox_messages[inbox_id][message_id] = message

        return {
            "success": True,
            "message": message.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to add message to inbox: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Routing Rule Handlers
# =============================================================================


async def handle_create_routing_rule(
    workspace_id: str,
    name: str,
    conditions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    condition_logic: str = "AND",
    priority: int = 5,
    enabled: bool = True,
    description: str | None = None,
    created_by: str | None = None,
    inbox_id: str | None = None,
) -> dict[str, Any]:
    """
    Create a routing rule with comprehensive input validation.

    POST /api/v1/inbox/routing/rules
    {
        "workspace_id": "ws_123",
        "name": "Urgent Customer Issues",
        "conditions": [
            {"field": "subject", "operator": "contains", "value": "urgent"}
        ],
        "condition_logic": "AND",
        "actions": [
            {"type": "assign", "target": "support-team"},
            {"type": "label", "target": "urgent"}
        ],
        "priority": 1
    }

    Security validations performed:
    - Rate limiting per workspace (10 rules/minute)
    - Rule name and description length limits
    - Condition field whitelist validation
    - ReDoS protection for regex patterns in MATCHES operator
    - Circular routing detection for forward actions
    - Maximum conditions/actions per rule limits
    """
    try:
        # Rate limiting check
        is_allowed, remaining = _rule_rate_limiter.is_allowed(workspace_id)
        if not is_allowed:
            retry_after = _rule_rate_limiter.get_retry_after(workspace_id)
            logger.warning(
                f"[SharedInbox] Rate limit exceeded for workspace {workspace_id}. "
                f"Retry after {retry_after:.1f}s"
            )
            return {
                "success": False,
                "error": f"Rate limit exceeded. Try again in {int(retry_after) + 1} seconds.",
                "retry_after": int(retry_after) + 1,
            }

        # Check workspace rule count limit
        with _storage_lock:
            workspace_rule_count = sum(
                1 for r in _routing_rules.values() if r.workspace_id == workspace_id
            )
        if workspace_rule_count >= MAX_RULES_PER_WORKSPACE:
            return {
                "success": False,
                "error": f"Maximum number of rules ({MAX_RULES_PER_WORKSPACE}) reached for this workspace",
            }

        # Validate condition_logic
        if condition_logic not in ("AND", "OR"):
            return {
                "success": False,
                "error": "condition_logic must be 'AND' or 'OR'",
            }

        # Validate priority
        if not isinstance(priority, int) or priority < 0 or priority > 100:
            return {
                "success": False,
                "error": "priority must be an integer between 0 and 100",
            }

        # Get existing rules for circular routing detection
        with _storage_lock:
            existing_rules = list(_routing_rules.values())

        # Comprehensive input validation
        validation_result = validate_routing_rule(
            name=name,
            conditions=conditions,
            actions=actions,
            workspace_id=workspace_id,
            description=description,
            existing_rules=existing_rules,
            check_circular=True,
        )

        if not validation_result.is_valid:
            logger.warning(
                f"[SharedInbox] Rule validation failed for workspace {workspace_id}: "
                f"{validation_result.error}"
            )
            return {
                "success": False,
                "error": validation_result.error,
            }

        # Use sanitized conditions and actions
        validated_conditions = validation_result.sanitized_conditions
        validated_actions = validation_result.sanitized_actions

        # Record the rate limit request (after validation passes)
        _rule_rate_limiter.record_request(workspace_id)

        rule_id = f"rule_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Sanitize name and description
        sanitized_name = sanitize_user_input(name, max_length=MAX_RULE_NAME_LENGTH)
        sanitized_description = None
        if description:
            sanitized_description = sanitize_user_input(
                description, max_length=MAX_RULE_DESCRIPTION_LENGTH
            )

        # Prepare rule data for persistent storage
        rule_data = {
            "id": rule_id,
            "name": sanitized_name,
            "workspace_id": workspace_id,
            "inbox_id": inbox_id,
            "conditions": validated_conditions,
            "condition_logic": condition_logic,
            "actions": validated_actions,
            "priority": priority,
            "enabled": enabled,
            "description": sanitized_description,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "created_by": created_by,
            "stats": {"total_matches": 0, "matched": 0, "applied": 0},
        }

        # Use RulesStore for persistent storage (primary)
        rules_store = _get_rules_store()
        if rules_store:
            try:
                rules_store.create_rule(rule_data)
                logger.info(
                    f"[SharedInbox] Created routing rule {rule_id}: {sanitized_name} (persistent)"
                )
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to persist rule to RulesStore: {e}")
                # Fall through to in-memory storage

        # Also persist to email store for backward compatibility
        email_store = _get_store()
        if email_store:
            try:
                email_store.create_routing_rule(
                    rule_id=rule_id,
                    workspace_id=workspace_id,
                    name=sanitized_name,
                    conditions=validated_conditions,
                    actions=validated_actions,
                    priority=priority,
                    enabled=enabled,
                    description=sanitized_description,
                    inbox_id=inbox_id,
                )
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to persist rule to email store: {e}")

        # Build in-memory RoutingRule object for cache
        rule = RoutingRule(
            id=rule_id,
            workspace_id=workspace_id,
            name=sanitized_name,
            conditions=[RuleCondition.from_dict(c) for c in validated_conditions],
            condition_logic=condition_logic,
            actions=[RuleAction.from_dict(a) for a in validated_actions],
            priority=priority,
            enabled=enabled,
            description=sanitized_description,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            stats={"total_matches": 0},
        )

        # Keep in-memory cache for fast reads
        with _storage_lock:
            _routing_rules[rule_id] = rule

        logger.info(
            f"[SharedInbox] Created routing rule {rule_id} for workspace {workspace_id} "
            f"(remaining rate limit: {remaining - 1})"
        )

        return {
            "success": True,
            "rule": rule.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to create routing rule: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_list_routing_rules(
    workspace_id: str,
    enabled_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    inbox_id: str | None = None,
) -> dict[str, Any]:
    """
    List routing rules for a workspace.

    GET /api/v1/inbox/routing/rules?workspace_id=ws_123
    """
    try:
        # Try RulesStore first (primary persistent storage)
        rules_store = _get_rules_store()
        if rules_store:
            try:
                rules = rules_store.list_rules(
                    workspace_id=workspace_id,
                    inbox_id=inbox_id,
                    enabled_only=enabled_only,
                    limit=limit,
                    offset=offset,
                )
                total = rules_store.count_rules(
                    workspace_id=workspace_id,
                    inbox_id=inbox_id,
                    enabled_only=enabled_only,
                )
                return {
                    "success": True,
                    "rules": rules,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to load rules from RulesStore: {e}")

        # Try email store as fallback
        email_store = _get_store()
        if email_store:
            try:
                rules = email_store.list_routing_rules(
                    workspace_id=workspace_id,
                    inbox_id=inbox_id,
                    enabled_only=enabled_only,
                )
                if rules is not None:
                    total = len(rules)
                    rules = rules[offset : offset + limit]
                    return {
                        "success": True,
                        "rules": rules,
                        "total": total,
                        "limit": limit,
                        "offset": offset,
                    }
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to load rules from email store: {e}")

        # Fallback to in-memory
        with _storage_lock:
            all_rules = [
                rule.to_dict()
                for rule in sorted(_routing_rules.values(), key=lambda r: r.priority)
                if rule.workspace_id == workspace_id
            ]
            if inbox_id:
                all_rules = [
                    r
                    for r in all_rules
                    if r.get("inbox_id") == inbox_id or r.get("inbox_id") is None
                ]
            if enabled_only:
                all_rules = [r for r in all_rules if r.get("enabled", True)]
            total = len(all_rules)
            rules = all_rules[offset : offset + limit]

        return {
            "success": True,
            "rules": rules,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.exception(f"Failed to list routing rules: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_update_routing_rule(
    rule_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """
    Update a routing rule with input validation.

    PATCH /api/v1/inbox/routing/rules/:id
    {
        "enabled": false,
        "priority": 2
    }

    Security validations performed:
    - Name and description length limits
    - Condition field whitelist validation
    - ReDoS protection for regex patterns
    - Circular routing detection for forward actions
    - Priority range validation
    - Condition logic validation
    """
    try:
        # Get existing rule to validate updates against
        existing_rule = None
        workspace_id = None

        with _storage_lock:
            existing_rule = _routing_rules.get(rule_id)
            if existing_rule:
                workspace_id = existing_rule.workspace_id

        # If not in memory, try persistent stores
        if not existing_rule:
            rules_store = _get_rules_store()
            if rules_store:
                try:
                    rule_data = rules_store.get_rule(rule_id)
                    if rule_data:
                        existing_rule = RoutingRule.from_dict(rule_data)
                        workspace_id = existing_rule.workspace_id
                except (OSError, RuntimeError, ValueError, KeyError) as e:
                    logger.debug(f"Failed to get rule {rule_id} from store: {e}")

        if not existing_rule:
            return {"success": False, "error": "Rule not found"}

        # Validate name if being updated
        if "name" in updates:
            name = updates["name"]
            if not name:
                return {"success": False, "error": "Rule name cannot be empty"}
            if len(name) > MAX_RULE_NAME_LENGTH:
                return {
                    "success": False,
                    "error": f"Rule name exceeds maximum length of {MAX_RULE_NAME_LENGTH}",
                }
            updates["name"] = sanitize_user_input(name, max_length=MAX_RULE_NAME_LENGTH)

        # Validate description if being updated
        if "description" in updates:
            description = updates["description"]
            if description and len(description) > MAX_RULE_DESCRIPTION_LENGTH:
                return {
                    "success": False,
                    "error": f"Description exceeds maximum length of {MAX_RULE_DESCRIPTION_LENGTH}",
                }
            if description:
                updates["description"] = sanitize_user_input(
                    description, max_length=MAX_RULE_DESCRIPTION_LENGTH
                )

        # Validate condition_logic if being updated
        if "condition_logic" in updates:
            if updates["condition_logic"] not in ("AND", "OR"):
                return {
                    "success": False,
                    "error": "condition_logic must be 'AND' or 'OR'",
                }

        # Validate priority if being updated
        if "priority" in updates:
            priority = updates["priority"]
            if not isinstance(priority, int) or priority < 0 or priority > 100:
                return {
                    "success": False,
                    "error": "priority must be an integer between 0 and 100",
                }

        # Validate conditions if being updated
        validated_conditions = None
        if "conditions" in updates:
            conditions = updates["conditions"]
            if not conditions:
                return {"success": False, "error": "At least one condition is required"}

            if len(conditions) > MAX_CONDITIONS_PER_RULE:
                return {
                    "success": False,
                    "error": f"Number of conditions ({len(conditions)}) exceeds maximum of {MAX_CONDITIONS_PER_RULE}",
                }

            validated_conditions = []
            for i, condition in enumerate(conditions):
                is_valid, error, sanitized = validate_rule_condition(condition)
                if not is_valid:
                    return {
                        "success": False,
                        "error": f"Condition {i + 1}: {error}",
                    }
                validated_conditions.append(sanitized)
            updates["conditions"] = validated_conditions

        # Validate actions if being updated
        validated_actions = None
        if "actions" in updates:
            actions = updates["actions"]
            if not actions:
                return {"success": False, "error": "At least one action is required"}

            if len(actions) > MAX_ACTIONS_PER_RULE:
                return {
                    "success": False,
                    "error": f"Number of actions ({len(actions)}) exceeds maximum of {MAX_ACTIONS_PER_RULE}",
                }

            validated_actions = []
            for i, action in enumerate(actions):
                is_valid, error, sanitized = validate_rule_action(action)
                if not is_valid:
                    return {
                        "success": False,
                        "error": f"Action {i + 1}: {error}",
                    }
                validated_actions.append(sanitized)

            # Check for circular routing with the new actions
            with _storage_lock:
                # Exclude the current rule from existing rules for circular check
                existing_rules = [r for r in _routing_rules.values() if r.id != rule_id]

            has_circular, circular_error = detect_circular_routing(
                validated_actions,
                existing_rules,
                workspace_id,
            )
            if has_circular:
                return {"success": False, "error": circular_error}

            updates["actions"] = validated_actions

        updated_rule_data = None

        # Update in RulesStore first (primary persistent storage)
        rules_store = _get_rules_store()
        if rules_store:
            try:
                updated_rule_data = rules_store.update_rule(rule_id, updates)
                if updated_rule_data:
                    logger.info(f"[SharedInbox] Updated routing rule {rule_id} in RulesStore")
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to update rule in RulesStore: {e}")

        # Also update in email store for backward compatibility
        email_store = _get_store()
        if email_store:
            try:
                email_store.update_routing_rule(rule_id, **updates)
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to update rule in email store: {e}")

        # Update in-memory cache
        with _storage_lock:
            rule = _routing_rules.get(rule_id)
            if rule:
                # Update fields with validated data
                if "name" in updates:
                    rule.name = updates["name"]
                if "description" in updates:
                    rule.description = updates["description"]
                if "conditions" in updates:
                    rule.conditions = [RuleCondition.from_dict(c) for c in updates["conditions"]]
                if "condition_logic" in updates:
                    rule.condition_logic = updates["condition_logic"]
                if "actions" in updates:
                    rule.actions = [RuleAction.from_dict(a) for a in updates["actions"]]
                if "priority" in updates:
                    rule.priority = updates["priority"]
                if "enabled" in updates:
                    rule.enabled = updates["enabled"]

                rule.updated_at = datetime.now(timezone.utc)

        # Return data from persistent storage if available
        if updated_rule_data:
            return {
                "success": True,
                "rule": updated_rule_data,
            }

        # Fallback: try to get from RulesStore
        if rules_store:
            try:
                rule_data = rules_store.get_rule(rule_id)
                if rule_data:
                    return {
                        "success": True,
                        "rule": rule_data,
                    }
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.debug(f"Failed to get rule {rule_id} from store: {e}")

        # Return from in-memory cache if available
        with _storage_lock:
            rule = _routing_rules.get(rule_id)
            if rule:
                return {
                    "success": True,
                    "rule": rule.to_dict(),
                }

        return {"success": False, "error": "Rule not found"}

    except Exception as e:
        logger.exception(f"Failed to update routing rule: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_delete_routing_rule(
    rule_id: str,
) -> dict[str, Any]:
    """
    Delete a routing rule.

    DELETE /api/v1/inbox/routing/rules/:id
    """
    try:
        deleted = False

        # Delete from RulesStore (primary persistent storage)
        rules_store = _get_rules_store()
        if rules_store:
            try:
                deleted = rules_store.delete_rule(rule_id)
                if deleted:
                    logger.info(f"[SharedInbox] Deleted routing rule {rule_id} from RulesStore")
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to delete rule from RulesStore: {e}")

        # Also delete from email store for backward compatibility
        email_store = _get_store()
        if email_store:
            try:
                email_store.delete_routing_rule(rule_id)
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to delete rule from email store: {e}")

        # Delete from in-memory cache
        with _storage_lock:
            if rule_id in _routing_rules:
                del _routing_rules[rule_id]
                deleted = True

        if deleted:
            return {
                "success": True,
                "deleted": rule_id,
            }
        else:
            return {
                "success": False,
                "error": "Rule not found",
            }

    except Exception as e:
        logger.exception(f"Failed to delete routing rule: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_test_routing_rule(
    rule_id: str,
    workspace_id: str,
) -> dict[str, Any]:
    """
    Test a routing rule against existing messages.

    POST /api/v1/inbox/routing/rules/:id/test
    {
        "workspace_id": "ws_123"
    }
    """
    try:
        rule = None
        rule_data = None

        # Try RulesStore first (primary persistent storage)
        rules_store = _get_rules_store()
        if rules_store:
            try:
                rule_data = rules_store.get_rule(rule_id)
                if rule_data:
                    rule = RoutingRule.from_dict(rule_data)
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"[SharedInbox] Failed to load rule from RulesStore: {e}")

        # Try email store as fallback
        if not rule:
            email_store = _get_store()
            if email_store:
                try:
                    rule_data = email_store.get_routing_rule(rule_id)
                    if rule_data:
                        rule = RoutingRule.from_dict(rule_data)
                except (OSError, RuntimeError, ValueError, KeyError) as e:
                    logger.warning(f"[SharedInbox] Failed to load rule from email store: {e}")

        # Fallback to in-memory
        if not rule:
            with _storage_lock:
                rule = _routing_rules.get(rule_id)
                if not rule:
                    return {"success": False, "error": "Rule not found"}

        # Count matching messages across all inboxes in workspace
        match_count = 0
        with _storage_lock:
            for inbox_id, messages in _inbox_messages.items():
                inbox = _shared_inboxes.get(inbox_id)
                if inbox and inbox.workspace_id == workspace_id:
                    for message in messages.values():
                        if _evaluate_rule(rule, message):
                            match_count += 1

        return {
            "success": True,
            "rule_id": rule_id,
            "match_count": match_count,
            "rule": rule.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to test routing rule: {e}")
        return {
            "success": False,
            "error": str(e),
        }


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
        elif condition.operator == RuleConditionOperator.NOT_MATCHES:
            # NOT_MATCHES: true if regex does NOT match
            match = execute_regex_with_timeout(
                pattern=condition_value,
                text=value,
                timeout=REGEX_TIMEOUT_SECONDS,
                flags=re.IGNORECASE,
            )
            matched = match is None
            # Note: For NOT_MATCHES, a timeout should be treated as NOT matched
            # (fail-safe: if we can't evaluate, don't match)

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

    # Try RulesStore first (primary persistent storage)
    rules_store = _get_rules_store()
    if rules_store:
        try:
            matching_rules = rules_store.get_matching_rules(
                inbox_id=inbox_id,
                email_data=email_data,
                workspace_id=workspace_id,
            )
            return matching_rules
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


# =============================================================================
# Handler Class
# =============================================================================


class SharedInboxHandler(BaseHandler):
    """
    HTTP handler for shared inbox endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/inbox/shared",
        "/api/v1/inbox/routing/rules",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/inbox/shared/",
        "/api/v1/inbox/routing/rules/",
    ]

    def __init__(self, ctx: dict[str, Any]) -> None:
        """Initialize with server context."""
        super().__init__(ctx)

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route shared inbox endpoint requests."""
        return None

    @require_permission("inbox:create")
    async def handle_post_shared_inbox(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/v1/inbox/shared"""
        workspace_id = data.get("workspace_id")
        name = data.get("name")

        if not workspace_id or not name:
            return error_response("workspace_id and name required", 400)

        # Validate inbox inputs
        is_valid, error = validate_inbox_input(
            name=name,
            description=data.get("description"),
            email_address=data.get("email_address"),
        )
        if not is_valid:
            return error_response(error, 400)

        # Sanitize name and description
        sanitized_name = sanitize_user_input(name, max_length=MAX_INBOX_NAME_LENGTH)
        sanitized_description = None
        if data.get("description"):
            sanitized_description = sanitize_user_input(
                data["description"], max_length=MAX_INBOX_DESCRIPTION_LENGTH
            )

        result = await handle_create_shared_inbox(
            workspace_id=workspace_id,
            name=sanitized_name,
            description=sanitized_description,
            email_address=data.get("email_address"),
            connector_type=data.get("connector_type"),
            team_members=data.get("team_members"),
            admins=data.get("admins"),
            settings=data.get("settings"),
            created_by=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:read")
    async def handle_get_shared_inboxes(self, params: dict[str, Any]) -> HandlerResult:
        """GET /api/v1/inbox/shared"""
        workspace_id = params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id required", 400)

        result = await handle_list_shared_inboxes(
            workspace_id=workspace_id,
            user_id=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:read")
    async def handle_get_shared_inbox(self, params: dict[str, Any], inbox_id: str) -> HandlerResult:
        """GET /api/v1/inbox/shared/:id"""
        result = await handle_get_shared_inbox(inbox_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    @require_permission("inbox:read")
    async def handle_get_inbox_messages(
        self, params: dict[str, Any], inbox_id: str
    ) -> HandlerResult:
        """GET /api/v1/inbox/shared/:id/messages"""
        result = await handle_get_inbox_messages(
            inbox_id=inbox_id,
            status=params.get("status"),
            assigned_to=params.get("assigned_to"),
            tag=params.get("tag"),
            limit=safe_query_int(params, "limit", default=50, min_val=1, max_val=1000),
            offset=safe_query_int(params, "offset", default=0, min_val=0, max_val=100000),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:manage")
    async def handle_post_assign_message(
        self, data: dict[str, Any], inbox_id: str, message_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/shared/:id/messages/:msg_id/assign"""
        assigned_to = data.get("assigned_to")
        if not assigned_to:
            return error_response("assigned_to required", 400)

        # Validate and sanitize assigned_to (user ID)
        if not isinstance(assigned_to, str):
            return error_response("assigned_to must be a string", 400)

        if len(assigned_to) > 200:
            return error_response("assigned_to exceeds maximum length of 200", 400)

        # Sanitize user ID
        sanitized_assigned_to = sanitize_user_input(assigned_to, max_length=200)
        if not sanitized_assigned_to:
            return error_response("assigned_to cannot be empty", 400)

        result = await handle_assign_message(
            inbox_id=inbox_id,
            message_id=message_id,
            assigned_to=sanitized_assigned_to,
            assigned_by=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:manage")
    async def handle_post_update_status(
        self, data: dict[str, Any], inbox_id: str, message_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/shared/:id/messages/:msg_id/status"""
        status = data.get("status")
        if not status:
            return error_response("status required", 400)

        # Validate status is a valid MessageStatus
        try:
            MessageStatus(status)
        except ValueError:
            valid_statuses = ", ".join(s.value for s in MessageStatus)
            return error_response(
                f"Invalid status '{status}'. Valid statuses: {valid_statuses}", 400
            )

        result = await handle_update_message_status(
            inbox_id=inbox_id,
            message_id=message_id,
            status=status,
            updated_by=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:manage")
    async def handle_post_add_tag(
        self, data: dict[str, Any], inbox_id: str, message_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/shared/:id/messages/:msg_id/tag"""
        tag = data.get("tag")
        if not tag:
            return error_response("tag required", 400)

        # Validate tag format and length
        is_valid, error = validate_tag(tag)
        if not is_valid:
            return error_response(error, 400)

        result = await handle_add_message_tag(
            inbox_id=inbox_id,
            message_id=message_id,
            tag=tag,
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:admin")
    async def handle_post_routing_rule(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/v1/inbox/routing/rules"""
        workspace_id = data.get("workspace_id")
        name = data.get("name")
        conditions = data.get("conditions", [])
        actions = data.get("actions", [])

        if not workspace_id or not name or not conditions or not actions:
            return error_response("workspace_id, name, conditions, and actions required", 400)

        result = await handle_create_routing_rule(
            workspace_id=workspace_id,
            name=name,
            conditions=conditions,
            actions=actions,
            condition_logic=data.get("condition_logic", "AND"),
            priority=data.get("priority", 5),
            enabled=data.get("enabled", True),
            description=data.get("description"),
            created_by=self._get_user_id(),
            inbox_id=data.get("inbox_id"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:read")
    async def handle_get_routing_rules(self, params: dict[str, Any]) -> HandlerResult:
        """GET /api/v1/inbox/routing/rules"""
        workspace_id = params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id required", 400)

        result = await handle_list_routing_rules(
            workspace_id=workspace_id,
            enabled_only=params.get("enabled_only", "false").lower() == "true",
            limit=safe_query_int(params, "limit", default=100, min_val=1, max_val=1000),
            offset=safe_query_int(params, "offset", default=0, min_val=0, max_val=100000),
            inbox_id=params.get("inbox_id"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:admin")
    async def handle_patch_routing_rule(self, data: dict[str, Any], rule_id: str) -> HandlerResult:
        """PATCH /api/v1/inbox/routing/rules/:id"""
        result = await handle_update_routing_rule(rule_id, data)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:admin")
    async def handle_delete_routing_rule(self, rule_id: str) -> HandlerResult:
        """DELETE /api/v1/inbox/routing/rules/:id"""
        result = await handle_delete_routing_rule(rule_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:admin")
    async def handle_post_test_routing_rule(
        self, data: dict[str, Any], rule_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/routing/rules/:id/test"""
        workspace_id = data.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id required", 400)

        result = await handle_test_routing_rule(rule_id, workspace_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"
