"""
Routing Rules API Handler.

Provides CRUD operations and rule evaluation for the decision routing rules engine.
Allows users to create, manage, and test rules that control how deliberation
decisions are routed to various channels based on conditions.

Usage:
    GET    /api/v1/routing-rules              - List all rules
    POST   /api/v1/routing-rules              - Create a new rule
    GET    /api/v1/routing-rules/{id}         - Get a specific rule
    PUT    /api/v1/routing-rules/{id}         - Update a rule
    DELETE /api/v1/routing-rules/{id}         - Delete a rule
    POST   /api/v1/routing-rules/{id}/toggle  - Enable/disable a rule
    POST   /api/v1/routing-rules/evaluate     - Test rules against context
    GET    /api/v1/routing-rules/templates    - Get predefined rule templates
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from aragora.server.handlers.base import BaseHandler

logger = logging.getLogger(__name__)

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


class RoutingRulesHandler(BaseHandler):
    """
    Handler for routing rules CRUD and evaluation endpoints.

    Provides management of decision routing rules that control how
    deliberation decisions are delivered to various channels.
    """

    ROUTES = [
        "/api/v1/routing-rules",
        "/api/v1/routing-rules/{rule_id}",
        "/api/v1/routing-rules/{rule_id}/toggle",
        "/api/v1/routing-rules/evaluate",
        "/api/v1/routing-rules/templates",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/routing-rules/")

    async def handle_request(self, request: Any) -> dict[str, Any]:
        """Route request to appropriate handler."""
        method = request.method
        path = request.path

        # Remove query string for path matching
        path_only = path.split("?")[0]

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
            # Parse query parameters
            enabled_only = request.args.get("enabled_only", "false").lower() == "true"
            tags = request.args.get("tags", "").split(",") if request.args.get("tags") else None

            from aragora.core.routing_rules import RoutingRule

            rules = []
            for rule_data in _rules_store.values():
                rule = RoutingRule.from_dict(rule_data)

                # Apply filters
                if enabled_only and not rule.enabled:
                    continue
                if tags and not any(t in rule.tags for t in tags if t):
                    continue

                rules.append(rule.to_dict())

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
                "error": str(e),
            }

    async def _create_rule(self, request: Any) -> dict[str, Any]:
        """Create a new routing rule."""
        try:
            data = await self._get_json_body(request)
            if not data:
                return {"status": "error", "error": "Missing request body"}

            from aragora.core.routing_rules import Action, Condition, RoutingRule

            # Parse conditions
            conditions = [Condition.from_dict(c) for c in data.get("conditions", [])]

            # Parse actions
            actions = [Action.from_dict(a) for a in data.get("actions", [])]

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

            logger.info(f"Created routing rule: {rule.id} ({rule.name})")

            return {
                "status": "success",
                "rule": rule.to_dict(),
            }
        except Exception as e:
            logger.error(f"Failed to create rule: {e}")
            return {
                "status": "error",
                "error": str(e),
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
                "error": str(e),
            }

    async def _update_rule(self, request: Any, rule_id: str) -> dict[str, Any]:
        """Update an existing routing rule."""
        try:
            if rule_id not in _rules_store:
                return {"status": "error", "error": "Rule not found", "code": 404}

            data = await self._get_json_body(request)
            if not data:
                return {"status": "error", "error": "Missing request body"}

            from aragora.core.routing_rules import Action, Condition

            existing = _rules_store[rule_id]

            # Update fields
            if "name" in data:
                existing["name"] = data["name"]
            if "description" in data:
                existing["description"] = data["description"]
            if "conditions" in data:
                existing["conditions"] = [
                    Condition.from_dict(c).to_dict() for c in data["conditions"]
                ]
            if "actions" in data:
                existing["actions"] = [Action.from_dict(a).to_dict() for a in data["actions"]]
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

            logger.info(f"Updated routing rule: {rule_id}")

            return {
                "status": "success",
                "rule": existing,
            }
        except Exception as e:
            logger.error(f"Failed to update rule {rule_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def _delete_rule(self, request: Any, rule_id: str) -> dict[str, Any]:
        """Delete a routing rule."""
        try:
            if rule_id not in _rules_store:
                return {"status": "error", "error": "Rule not found", "code": 404}

            del _rules_store[rule_id]

            logger.info(f"Deleted routing rule: {rule_id}")

            return {
                "status": "success",
                "message": f"Rule {rule_id} deleted",
            }
        except Exception as e:
            logger.error(f"Failed to delete rule {rule_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def _toggle_rule(self, request: Any, rule_id: str) -> dict[str, Any]:
        """Toggle a rule's enabled state."""
        try:
            if rule_id not in _rules_store:
                return {"status": "error", "error": "Rule not found", "code": 404}

            data = await self._get_json_body(request)
            enabled = data.get("enabled") if data else None

            existing = _rules_store[rule_id]

            if enabled is not None:
                existing["enabled"] = enabled
            else:
                existing["enabled"] = not existing.get("enabled", True)

            existing["updated_at"] = datetime.now(timezone.utc).isoformat()
            _rules_store[rule_id] = existing

            logger.info(f"Toggled rule {rule_id} to enabled={existing['enabled']}")

            return {
                "status": "success",
                "rule": existing,
            }
        except Exception as e:
            logger.error(f"Failed to toggle rule {rule_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
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
                return {"status": "error", "error": "Missing request body"}

            context = data.get("context", {})
            if not context:
                return {"status": "error", "error": "Missing context"}

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
                "error": str(e),
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
        except Exception as e:
            logger.error(f"Failed to get templates: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def _get_json_body(self, request: Any) -> Optional[dict[str, Any]]:
        """Extract JSON body from request."""
        try:
            if hasattr(request, "json"):
                return await request.json()
            if hasattr(request, "body"):
                body = request.body
                if isinstance(body, bytes):
                    body = body.decode("utf-8")
                return json.loads(body) if body else None
            return None
        except Exception:
            return None

    def _method_not_allowed(self, method: str, path: str) -> dict[str, Any]:
        """Return method not allowed response."""
        return {
            "status": "error",
            "error": f"Method {method} not allowed for {path}",
            "code": 405,
        }


# Handler class (instantiated by server with context)
# Note: Do not instantiate at module level - requires server_context
routing_rules_handler = None  # type: ignore
