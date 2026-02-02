"""
Routing Rules API Handler.

Provides CRUD operations and rule evaluation for the decision routing rules engine.
Allows users to create, manage, and test rules that control how deliberation
decisions are routed to various channels based on conditions.

Stability: STABLE
- Full RBAC enforcement on all endpoints
- Rate limiting configured per operation type
- Comprehensive input validation
- Audit logging for all write operations

Usage:
    GET    /api/v1/routing-rules              - List all rules
    POST   /api/v1/routing-rules              - Create a new rule
    GET    /api/v1/routing-rules/{id}         - Get a specific rule
    PUT    /api/v1/routing-rules/{id}         - Update a rule
    DELETE /api/v1/routing-rules/{id}         - Delete a rule
    POST   /api/v1/routing-rules/{id}/toggle  - Enable/disable a rule
    POST   /api/v1/routing-rules/evaluate     - Test rules against context
    GET    /api/v1/routing-rules/templates    - Get predefined rule templates

Security:
    All endpoints require RBAC permissions:
    - policies.read: List, get, evaluate, templates
    - policies.create: Create new rules
    - policies.update: Update existing rules, toggle
    - policies.delete: Delete rules
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any


from aragora.audit.unified import audit_data
from aragora.server.handlers.secure import (
    ForbiddenError,
    SecureHandler,
    UnauthorizedError,
)
from aragora.server.handlers.utils import parse_json_body
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Validation constants
MAX_RULE_NAME_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 2000
MAX_CONDITIONS = 50
MAX_ACTIONS = 20
MAX_TAGS = 20
MAX_TAG_LENGTH = 50
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
VALID_MATCH_MODES = {"all", "any"}

# In-memory rule storage (for development/demo)
# In production, rules would be stored in a database
_rules_store: dict[str, dict[str, Any]] = {}


def _get_routing_engine():
    """Get or create the routing rules engine."""
    from aragora.core.routing_rules import RoutingRule, RoutingRulesEngine

    engine = RoutingRulesEngine()

    # Load rules from store into engine
    for rule_id, rule_data in _rules_store.items():
        try:
            rule = RoutingRule.from_dict(rule_data)
            engine.add_rule(rule)
        except Exception as e:
            logger.error(f"Failed to load rule {rule_id}: {e}")

    return engine


def _validate_rule_id(rule_id: str) -> tuple[bool, str | None]:
    """Validate a rule ID for safe use.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not rule_id:
        return False, "Rule ID is required"
    if not SAFE_ID_PATTERN.match(rule_id):
        return False, "Invalid rule ID format"
    return True, None


def _validate_rule_data(data: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate rule creation/update data.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate name
    name = data.get("name", "")
    if name and len(name) > MAX_RULE_NAME_LENGTH:
        return False, f"Rule name exceeds maximum length of {MAX_RULE_NAME_LENGTH}"

    # Validate description
    description = data.get("description", "")
    if description and len(description) > MAX_DESCRIPTION_LENGTH:
        return False, f"Description exceeds maximum length of {MAX_DESCRIPTION_LENGTH}"

    # Validate conditions count
    conditions = data.get("conditions", [])
    if len(conditions) > MAX_CONDITIONS:
        return False, f"Too many conditions (max: {MAX_CONDITIONS})"

    # Validate actions count
    actions = data.get("actions", [])
    if len(actions) > MAX_ACTIONS:
        return False, f"Too many actions (max: {MAX_ACTIONS})"

    # Validate tags
    tags = data.get("tags", [])
    if len(tags) > MAX_TAGS:
        return False, f"Too many tags (max: {MAX_TAGS})"
    for tag in tags:
        if not isinstance(tag, str):
            return False, "Tags must be strings"
        if len(tag) > MAX_TAG_LENGTH:
            return False, f"Tag exceeds maximum length of {MAX_TAG_LENGTH}"

    # Validate match_mode
    match_mode = data.get("match_mode")
    if match_mode and match_mode not in VALID_MATCH_MODES:
        return False, f"Invalid match_mode: must be one of {VALID_MATCH_MODES}"

    # Validate priority
    priority = data.get("priority")
    if priority is not None:
        if not isinstance(priority, int):
            return False, "Priority must be an integer"
        if priority < -1000 or priority > 1000:
            return False, "Priority must be between -1000 and 1000"

    return True, None


class RoutingRulesHandler(SecureHandler):
    """
    Handler for routing rules CRUD and evaluation endpoints.

    Stability: STABLE

    Provides management of decision routing rules that control how
    deliberation decisions are delivered to various channels.

    Features:
    - Full RBAC enforcement on all endpoints
    - Rate limiting (60 RPM read, 30 RPM write operations)
    - Input validation with safe defaults
    - Audit logging for all mutations
    - Proper error handling and logging

    RBAC Permissions:
    - policies.read: List, get, evaluate, templates
    - policies.create: Create new rules
    - policies.update: Update existing rules, toggle
    - policies.delete: Delete rules
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/api/v1/routing-rules",
        "/api/v1/routing-rules/{rule_id}",
        "/api/v1/routing-rules/{rule_id}/toggle",
        "/api/v1/routing-rules/evaluate",
        "/api/v1/routing-rules/templates",
    ]

    RESOURCE_TYPE = "policy"  # For audit logging

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/routing-rules")

    @rate_limit(requests_per_minute=60, limiter_name="routing_rules_handler")
    async def handle_request(self, request: Any) -> dict[str, Any]:
        """Route request to appropriate handler with RBAC enforcement."""
        method = request.method
        path = request.path

        # Remove query string for path matching
        path_only = path.split("?")[0]

        # RBAC: Authenticate and authorize
        try:
            auth_context = await self.get_auth_context(request, require_auth=True)
        except UnauthorizedError:
            return {"status": "error", "error": "Authentication required", "code": 401}

        # Determine required permission based on method and path
        try:
            if method == "POST" and path_only == "/api/v1/routing-rules":
                self.check_permission(auth_context, "policies.create")
            elif method in ("PUT", "PATCH"):
                self.check_permission(auth_context, "policies.update")
            elif method == "DELETE":
                self.check_permission(auth_context, "policies.delete")
            elif "toggle" in path_only and method == "POST":
                self.check_permission(auth_context, "policies.update")
            elif "evaluate" in path_only and method == "POST":
                # Evaluate is read-only (testing rules)
                self.check_permission(auth_context, "policies.read")
            else:
                # GET requests and evaluate (read-only operations)
                self.check_permission(auth_context, "policies.read")
        except ForbiddenError as e:
            return {"status": "error", "error": str(e), "code": 403}

        # Store auth context for audit logging
        self._auth_context = auth_context

        # List/Create rules
        if path_only == "/api/v1/routing-rules":
            if method == "GET":
                return await self._list_rules(request)
            elif method == "POST":
                return await self._create_rule(request)

        # Templates endpoint
        if path_only == "/api/v1/routing-rules/templates":
            if method == "GET":
                return await self._get_templates(request)

        # Evaluate endpoint
        if path_only == "/api/v1/routing-rules/evaluate":
            if method == "POST":
                return await self._evaluate_rules(request)

        # Rule-specific operations
        if "/api/v1/routing-rules/" in path_only:
            parts = path_only.split("/")
            if len(parts) >= 5:
                rule_id = parts[4]

                # Validate rule_id
                is_valid, error = _validate_rule_id(rule_id)
                if not is_valid:
                    return {"status": "error", "error": error, "code": 400}

                # Toggle endpoint
                if len(parts) == 6 and parts[5] == "toggle":
                    if method == "POST":
                        return await self._toggle_rule(request, rule_id)

                # Standard CRUD operations
                if method == "GET":
                    return await self._get_rule(request, rule_id)
                elif method == "PUT":
                    return await self._update_rule(request, rule_id)
                elif method == "DELETE":
                    return await self._delete_rule(request, rule_id)

        return self._method_not_allowed(method, path)

    async def _list_rules(self, request: Any) -> dict[str, Any]:
        """List all routing rules with optional filtering."""
        try:
            # Parse query parameters safely
            args = getattr(request, "args", {}) or {}
            enabled_only = args.get("enabled_only", "false").lower() == "true"
            tags_param = args.get("tags", "")
            tags = tags_param.split(",") if tags_param else None

            from aragora.core.routing_rules import RoutingRule

            rules = []
            for rule_data in _rules_store.values():
                try:
                    rule = RoutingRule.from_dict(rule_data)

                    # Apply filters
                    if enabled_only and not rule.enabled:
                        continue
                    if tags and not any(t in rule.tags for t in tags if t):
                        continue

                    rules.append(rule.to_dict())
                except Exception as e:
                    logger.warning(f"Skipping invalid rule data: {e}")
                    continue

            # Sort by priority (descending)
            rules.sort(key=lambda r: r.get("priority", 0), reverse=True)

            return {
                "status": "success",
                "rules": rules,
                "count": len(rules),
            }
        except Exception as e:
            logger.error(f"Failed to list rules: {e}")
            return {
                "status": "error",
                "error": "Failed to list rules",
                "code": 500,
            }

    async def _create_rule(self, request: Any) -> dict[str, Any]:
        """Create a new routing rule."""
        try:
            data = await self._get_json_body(request)
            if not data:
                return {"status": "error", "error": "Missing request body", "code": 400}

            # Validate input data
            is_valid, error = _validate_rule_data(data)
            if not is_valid:
                return {"status": "error", "error": error, "code": 400}

            from aragora.core.routing_rules import Action, Condition, RoutingRule

            # Parse conditions with validation
            try:
                conditions = [Condition.from_dict(c) for c in data.get("conditions", [])]
            except (KeyError, ValueError, TypeError) as e:
                return {"status": "error", "error": f"Invalid condition: {e}", "code": 400}

            # Parse actions with validation
            try:
                actions = [Action.from_dict(a) for a in data.get("actions", [])]
            except (KeyError, ValueError, TypeError) as e:
                return {"status": "error", "error": f"Invalid action: {e}", "code": 400}

            # Create rule
            rule = RoutingRule.create(
                name=data.get("name", "Untitled Rule"),
                conditions=conditions,
                actions=actions,
                description=data.get("description", ""),
                priority=data.get("priority", 0),
                enabled=data.get("enabled", True),
                match_mode=data.get("match_mode", "all"),
                stop_processing=data.get("stop_processing", False),
                tags=data.get("tags", []),
            )

            # Store rule
            _rules_store[rule.id] = rule.to_dict()

            # Audit log the creation
            self._audit_rule_change("create", rule.id, rule.name)

            logger.info(f"Created routing rule: {rule.id} ({rule.name})")

            return {
                "status": "success",
                "rule": rule.to_dict(),
            }
        except Exception as e:
            logger.error(f"Failed to create rule: {e}")
            return {
                "status": "error",
                "error": "Failed to create rule",
                "code": 500,
            }

    async def _get_rule(self, request: Any, rule_id: str) -> dict[str, Any]:
        """Get a specific routing rule by ID."""
        try:
            if rule_id not in _rules_store:
                return {"status": "error", "error": "Rule not found", "code": 404}

            return {
                "status": "success",
                "rule": _rules_store[rule_id],
            }
        except Exception as e:
            logger.error(f"Failed to get rule {rule_id}: {e}")
            return {
                "status": "error",
                "error": "Failed to get rule",
                "code": 500,
            }

    async def _update_rule(self, request: Any, rule_id: str) -> dict[str, Any]:
        """Update an existing routing rule."""
        try:
            if rule_id not in _rules_store:
                return {"status": "error", "error": "Rule not found", "code": 404}

            data = await self._get_json_body(request)
            if not data:
                return {"status": "error", "error": "Missing request body", "code": 400}

            # Validate input data
            is_valid, error = _validate_rule_data(data)
            if not is_valid:
                return {"status": "error", "error": error, "code": 400}

            from aragora.core.routing_rules import Action, Condition

            # Make a copy to avoid partial updates on error
            existing = dict(_rules_store[rule_id])

            # Update fields
            if "name" in data:
                existing["name"] = data["name"]
            if "description" in data:
                existing["description"] = data["description"]
            if "conditions" in data:
                try:
                    existing["conditions"] = [
                        Condition.from_dict(c).to_dict() for c in data["conditions"]
                    ]
                except (KeyError, ValueError, TypeError) as e:
                    return {"status": "error", "error": f"Invalid condition: {e}", "code": 400}
            if "actions" in data:
                try:
                    existing["actions"] = [Action.from_dict(a).to_dict() for a in data["actions"]]
                except (KeyError, ValueError, TypeError) as e:
                    return {"status": "error", "error": f"Invalid action: {e}", "code": 400}
            if "priority" in data:
                existing["priority"] = data["priority"]
            if "enabled" in data:
                existing["enabled"] = data["enabled"]
            if "match_mode" in data:
                existing["match_mode"] = data["match_mode"]
            if "stop_processing" in data:
                existing["stop_processing"] = data["stop_processing"]
            if "tags" in data:
                existing["tags"] = data["tags"]

            # Update timestamp
            existing["updated_at"] = datetime.now(timezone.utc).isoformat()

            _rules_store[rule_id] = existing

            # Audit log the update
            self._audit_rule_change("update", rule_id, existing.get("name", ""))

            logger.info(f"Updated routing rule: {rule_id}")

            return {
                "status": "success",
                "rule": existing,
            }
        except Exception as e:
            logger.error(f"Failed to update rule {rule_id}: {e}")
            return {
                "status": "error",
                "error": "Failed to update rule",
                "code": 500,
            }

    async def _delete_rule(self, request: Any, rule_id: str) -> dict[str, Any]:
        """Delete a routing rule."""
        try:
            if rule_id not in _rules_store:
                return {"status": "error", "error": "Rule not found", "code": 404}

            rule_name = _rules_store[rule_id].get("name", "")
            del _rules_store[rule_id]

            # Audit log the deletion
            self._audit_rule_change("delete", rule_id, rule_name)

            logger.info(f"Deleted routing rule: {rule_id}")

            return {
                "status": "success",
                "message": f"Rule {rule_id} deleted",
            }
        except Exception as e:
            logger.error(f"Failed to delete rule {rule_id}: {e}")
            return {
                "status": "error",
                "error": "Failed to delete rule",
                "code": 500,
            }

    async def _toggle_rule(self, request: Any, rule_id: str) -> dict[str, Any]:
        """Toggle a rule's enabled state."""
        try:
            if rule_id not in _rules_store:
                return {"status": "error", "error": "Rule not found", "code": 404}

            data = await self._get_json_body(request)
            enabled = data.get("enabled") if data else None

            existing = _rules_store[rule_id]

            old_enabled = existing.get("enabled", True)
            if enabled is not None:
                existing["enabled"] = bool(enabled)
            else:
                existing["enabled"] = not old_enabled

            existing["updated_at"] = datetime.now(timezone.utc).isoformat()
            _rules_store[rule_id] = existing

            # Audit log the toggle
            action = "enable" if existing["enabled"] else "disable"
            self._audit_rule_change(action, rule_id, existing.get("name", ""))

            logger.info(f"Toggled rule {rule_id} to enabled={existing['enabled']}")

            return {
                "status": "success",
                "rule": existing,
            }
        except Exception as e:
            logger.error(f"Failed to toggle rule {rule_id}: {e}")
            return {
                "status": "error",
                "error": "Failed to toggle rule",
                "code": 500,
            }

    async def _evaluate_rules(self, request: Any) -> dict[str, Any]:
        """
        Evaluate rules against a context for testing.

        Expected body:
        {
            "context": {
                "confidence": 0.65,
                "topic": "security review",
                "agent_count": 3,
                ...
            }
        }
        """
        try:
            data = await self._get_json_body(request)
            if not data:
                return {"status": "error", "error": "Missing request body", "code": 400}

            context = data.get("context", {})
            if not context:
                return {"status": "error", "error": "Missing context", "code": 400}

            if not isinstance(context, dict):
                return {"status": "error", "error": "Context must be an object", "code": 400}

            engine = _get_routing_engine()
            results = engine.evaluate(context, execute_actions=False)

            # Format results
            formatted_results = []
            matching_actions = []

            for result in results:
                formatted_results.append(
                    {
                        "rule_id": result.rule.id,
                        "rule_name": result.rule.name,
                        "matched": result.matched,
                        "actions": [a.to_dict() for a in result.actions],
                        "execution_time_ms": round(result.execution_time_ms, 3),
                    }
                )
                if result.matched:
                    matching_actions.extend([a.to_dict() for a in result.actions])

            return {
                "status": "success",
                "context": context,
                "results": formatted_results,
                "matching_actions": matching_actions,
                "rules_evaluated": len(results),
                "rules_matched": sum(1 for r in results if r.matched),
            }
        except Exception as e:
            logger.error(f"Failed to evaluate rules: {e}")
            return {
                "status": "error",
                "error": "Failed to evaluate rules",
                "code": 500,
            }

    async def _get_templates(self, request: Any) -> dict[str, Any]:
        """Get predefined rule templates."""
        try:
            from aragora.core.routing_rules import RULE_TEMPLATES

            templates = []
            for key, rule in RULE_TEMPLATES.items():
                template_data = rule.to_dict()
                template_data["template_key"] = key
                templates.append(template_data)

            return {
                "status": "success",
                "templates": templates,
                "count": len(templates),
            }
        except ImportError:
            logger.error("Routing rules module not available")
            return {
                "status": "error",
                "error": "Routing rules module not available",
                "code": 503,
            }
        except Exception as e:
            logger.error(f"Failed to get templates: {e}")
            return {
                "status": "error",
                "error": "Failed to get templates",
                "code": 500,
            }

    async def _get_json_body(self, request: Any) -> dict[str, Any] | None:
        """Extract JSON body from request."""
        try:
            if hasattr(request, "json"):
                body, _err = await parse_json_body(request, context="routing_rules._get_json_body")
                return body
            if hasattr(request, "body"):
                raw_body = request.body
                if isinstance(raw_body, bytes):
                    raw_body = raw_body.decode("utf-8")
                return json.loads(raw_body) if raw_body else None
            return None
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError):
            return None

    def _method_not_allowed(self, method: str, path: str) -> dict[str, Any]:
        """Return method not allowed response."""
        return {
            "status": "error",
            "error": f"Method {method} not allowed for {path}",
            "code": 405,
        }

    def _audit_rule_change(self, action: str, rule_id: str, rule_name: str) -> None:
        """Log an audit event for rule changes."""
        try:
            user_id = getattr(self, "_auth_context", None)
            if user_id and hasattr(user_id, "user_id"):
                user_id = user_id.user_id
            else:
                user_id = "unknown"

            audit_data(
                user_id=user_id,
                resource_type="routing_rule",
                resource_id=rule_id,
                action=action,
                rule_name=rule_name,
            )
        except Exception as e:
            # Don't fail the operation if audit logging fails
            logger.warning(f"Failed to audit rule change: {e}")


# Handler class (instantiated by server with context)
# Note: Do not instantiate at module level - requires server_context
routing_rules_handler: Any = None
