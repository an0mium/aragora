"""
Debate decision-bridge handler.

Routes:
    POST /api/v1/debates/{id}/bridge

Bridges the latest DecisionPlan for a debate to a requested external target.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    handle_errors,
    json_response,
)

from ..openapi_decorator import api_endpoint

logger = logging.getLogger(__name__)

SUPPORTED_BRIDGE_TARGETS = frozenset({"jira", "linear", "n8n"})


def _get_plan_store():
    """Lazy import to avoid circular imports at module import time."""
    from aragora.pipeline.plan_store import get_plan_store

    return get_plan_store()


def _bridge_succeeded(target: str, result: Any) -> bool:
    """Return whether the requested target produced a concrete result."""
    if target == "jira":
        return bool(getattr(result, "jira_issues", []))
    if target == "linear":
        return bool(getattr(result, "linear_issues", []))
    return bool(getattr(result, "n8n_triggered", False))


class DebateDecisionBridgeHandler(BaseHandler):
    """POST handler for bridging a debate's latest DecisionPlan to external tools."""

    ROUTES = ["/api/v1/debates/{id}/bridge"]

    def __init__(self, ctx: dict[str, Any] | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        parts = path.split("/")
        return (
            len(parts) == 6
            and parts[1] == "api"
            and parts[2] == "v1"
            and parts[3] == "debates"
            and bool(parts[4])
            and parts[5] == "bridge"
        )

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """This route is POST-only."""
        return None

    def _extract_debate_id(self, path: str) -> str | None:
        parts = path.split("/")
        if len(parts) >= 5 and parts[4]:
            return parts[4]
        return None

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{id}/bridge",
        summary="Bridge a debate plan to external tools",
        description=(
            "Resolve the latest DecisionPlan for the debate and route it to one "
            "configured external target such as Jira, Linear, or n8n."
        ),
        tags=["Debates"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {"description": "Bridge dispatched successfully"},
            "400": {"description": "Invalid request body or unsupported target"},
            "404": {"description": "No DecisionPlan found for the debate"},
            "409": {"description": "DecisionPlan has no implementation tasks for this target"},
            "502": {
                "description": "Bridge target was reached but no external artifact was produced"
            },
        },
    )
    @handle_errors("debate decision bridge")
    @require_permission("debates:write")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        self.set_request_context(handler, query_params)

        debate_id = self._extract_debate_id(path)
        if not debate_id:
            return json_response({"error": "Missing debate ID"}, status=400)

        body = self.get_json_body()
        if body is None:
            return json_response({"error": "Invalid JSON body"}, status=400)

        target = str(body.get("target", "")).strip().lower()
        if not target:
            return json_response({"error": "target is required"}, status=400)
        if target not in SUPPORTED_BRIDGE_TARGETS:
            supported = ", ".join(sorted(SUPPORTED_BRIDGE_TARGETS))
            return json_response(
                {"error": f"Unsupported target: {target}", "supported_targets": supported},
                status=400,
            )

        store = _get_plan_store()
        plans = store.list(debate_id=debate_id, limit=1, offset=0)
        if not plans:
            return json_response(
                {"error": f"No decision plan found for debate: {debate_id}"},
                status=404,
            )

        plan = copy.deepcopy(plans[0])
        metadata = dict(getattr(plan, "metadata", {}) or {})
        metadata["integrations"] = [target]
        plan.metadata = metadata

        implement_plan = getattr(plan, "implement_plan", None)
        tasks = list(getattr(implement_plan, "tasks", []) or [])
        if target in {"jira", "linear"} and not tasks:
            return json_response(
                {
                    "error": f"Decision plan {plan.id} has no implementation tasks for {target}",
                    "debate_id": debate_id,
                    "plan_id": plan.id,
                    "target": target,
                },
                status=409,
            )

        from aragora.integrations.decision_bridge import DecisionBridge

        bridge = DecisionBridge(default_targets=[target])
        result = asyncio.run(bridge.handle_decision_plan(plan))
        payload = result.to_dict()
        response = {
            "debate_id": debate_id,
            "plan_id": getattr(plan, "id", ""),
            "target": target,
            "result": payload,
            "success": _bridge_succeeded(target, result),
        }

        if response["success"]:
            return json_response(response)

        errors = list(getattr(result, "errors", []) or [])
        if errors:
            response["error"] = errors[0]
        else:
            response["error"] = f"Bridge target {target} did not produce any external artifacts"
        return json_response(response, status=502)
