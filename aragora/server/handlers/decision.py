"""
Decision Router HTTP Handler.

Provides REST API endpoints for unified decision-making capabilities:
- POST /api/v1/decisions - Create a new decision request (debate, workflow, gauntlet)
- GET  /api/v1/decisions/:id - Get decision result by ID
- GET  /api/v1/decisions/:id/status - Get decision status for polling

Usage:
    # In unified_server.py
    from aragora.server.handlers.decision import DecisionHandler

    handlers.append(DecisionHandler(ctx))
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.rbac.decorators import require_permission

logger = logging.getLogger(__name__)

# Lazy-loaded router instance
_decision_router = None


def _get_decision_router():
    """Get or create the decision router singleton."""
    global _decision_router
    if _decision_router is None:
        try:
            from aragora.core.decision import DecisionRouter

            _decision_router = DecisionRouter()
        except Exception as e:
            logger.warning(f"DecisionRouter not available: {e}")
    return _decision_router


# Lazy-loaded result store instance
_decision_result_store = None


def _get_result_store():
    """Get the decision result store for persistence."""
    global _decision_result_store
    if _decision_result_store is None:
        try:
            from aragora.storage.decision_result_store import get_decision_result_store

            _decision_result_store = get_decision_result_store()
        except Exception as e:
            logger.warning(f"DecisionResultStore not available, using in-memory: {e}")
    return _decision_result_store


# Fallback in-memory cache (used only if persistent store fails)
_decision_results_fallback: Dict[str, Dict[str, Any]] = {}


def _save_result(request_id: str, data: Dict[str, Any]) -> None:
    """Save a decision result to persistent store with fallback."""
    store = _get_result_store()
    if store:
        try:
            store.save(request_id, data)
            return
        except Exception as e:
            logger.warning(f"Failed to persist result, using fallback: {e}")
    # Fallback to in-memory
    _decision_results_fallback[request_id] = data


def _get_result(request_id: str) -> Optional[Dict[str, Any]]:
    """Get a decision result from persistent store with fallback."""
    store = _get_result_store()
    if store:
        try:
            result = store.get(request_id)
            if result:
                return result
        except Exception as e:
            logger.warning(f"Failed to retrieve from store: {e}")
    # Fallback to in-memory
    return _decision_results_fallback.get(request_id)


class DecisionHandler(BaseHandler):
    """
    Handler for unified decision-making API endpoints.

    Provides a single entry point for debates, workflows, gauntlets,
    and quick decisions via the DecisionRouter.
    """

    ROUTES = [
        "/api/v1/decisions",
        "/api/v1/decisions/*",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the request."""
        if path == "/api/v1/decisions":
            return True
        if path.startswith("/api/v1/decisions/"):
            return True
        return False

    @require_permission("decisions:read")
    def handle(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle GET requests."""
        if path == "/api/v1/decisions":
            # List recent decisions (optional)
            return self._list_decisions(query_params)

        if path.startswith("/api/v1/decisions/"):
            parts = path.split("/")
            # parts = ['', 'api', 'v1', 'decisions', '<request_id>', ...]
            if len(parts) >= 5:
                request_id = parts[4]
                if len(parts) == 6 and parts[5] == "status":
                    return self._get_decision_status(request_id)
                return self._get_decision(request_id)

        return None

    @require_permission("decisions:create")
    def handle_post(self, path: str, query_params: dict, handler=None) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if path == "/api/v1/decisions":
            return self._create_decision(handler)
        return None

    def _create_decision(self, handler) -> HandlerResult:
        """
        Create a new decision request.

        Expected body:
        {
            "content": "Question or topic",
            "decision_type": "debate|workflow|gauntlet|quick|auto",
            "config": {
                "agents": ["anthropic-api", "openai-api"],
                "rounds": 3,
                "consensus": "majority",
                "timeout_seconds": 300
            },
            "context": {
                "user_id": "user-123",
                "workspace_id": "ws-456"
            },
            "priority": "high|normal|low",
            "response_channels": [
                {"platform": "http_api"}
            ]
        }
        """
        # Parse body
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body.get("content"):
            return error_response("Missing required field: content", 400)

        # Get authentication context
        from aragora.billing.auth import extract_user_from_request

        auth_ctx = extract_user_from_request(handler)

        # Build decision request
        try:
            from aragora.core.decision import DecisionRequest

            # Get headers for correlation ID
            headers = {}
            if hasattr(handler, "headers"):
                headers = dict(handler.headers)

            request = DecisionRequest.from_http(body, headers)

            # Set user context from auth if not provided
            if auth_ctx.authenticated:
                if not request.context.user_id:
                    request.context.user_id = auth_ctx.user_id
                if not request.context.workspace_id:
                    request.context.workspace_id = auth_ctx.org_id

        except ValueError as e:
            return error_response(f"Invalid request: {e}", 400)
        except Exception as e:
            logger.warning(f"Failed to parse decision request: {e}")
            return error_response(f"Failed to parse request: {e}", 400)

        # Get router
        router = _get_decision_router()
        if not router:
            return error_response("Decision router not available", 503)

        # Check RBAC if user is authenticated
        if auth_ctx.authenticated:
            try:
                from aragora.rbac import (
                    RBACEnforcer,
                    ResourceType,
                    Action,
                    IsolationContext,
                )

                enforcer = RBACEnforcer()
                ctx = IsolationContext(  # type: ignore[call-arg]
                    workspace_id=request.context.workspace_id,
                    user_id=request.context.user_id,
                )
                enforcer.require(  # type: ignore[unused-coroutine]
                    auth_ctx.user_id,
                    (
                        ResourceType.DECISION  # type: ignore[attr-defined]
                        if hasattr(ResourceType, "DECISION")
                        else ResourceType.DEBATE
                    ),
                    Action.CREATE,
                    ctx,
                )
            except Exception as e:
                logger.error(f"RBAC authorization check failed: {e}")
                return error_response("Authorization service unavailable", 503)

        # Route the decision (run synchronously for now)
        import asyncio

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(router.route(request))
            finally:
                loop.close()

            # Cache result for polling (persistent)
            _save_result(
                request.request_id,
                {
                    "request_id": request.request_id,
                    "status": "completed" if result.success else "failed",
                    "result": result.to_dict(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            return json_response(
                {
                    "request_id": request.request_id,
                    "status": "completed" if result.success else "failed",
                    "decision_type": result.decision_type.value,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "consensus_reached": result.consensus_reached,
                    "reasoning": result.reasoning,
                    "evidence_used": result.evidence_used,
                    "duration_seconds": result.duration_seconds,
                    "error": result.error,
                }
            )

        except asyncio.TimeoutError:
            # Save as pending for async polling
            _save_result(
                request.request_id,
                {
                    "request_id": request.request_id,
                    "status": "timeout",
                    "error": "Decision timed out",
                },
            )
            return error_response("Decision request timed out", 408)

        except Exception as e:
            logger.exception(f"Decision routing failed: {e}")
            _save_result(
                request.request_id,
                {
                    "request_id": request.request_id,
                    "status": "failed",
                    "error": str(e),
                },
            )
            return error_response(f"Decision failed: {e}", 500)

    def _get_decision(self, request_id: str) -> HandlerResult:
        """Get a decision result by ID."""
        result = _get_result(request_id)
        if result:
            return json_response(result)
        return error_response("Decision not found", 404)

    def _get_decision_status(self, request_id: str) -> HandlerResult:
        """Get decision status for polling."""
        store = _get_result_store()
        if store:
            try:
                return json_response(store.get_status(request_id))
            except Exception as e:
                logger.warning(f"Failed to get status from store: {e}")

        # Fallback to in-memory
        if request_id in _decision_results_fallback:
            result = _decision_results_fallback[request_id]
            return json_response(
                {
                    "request_id": request_id,
                    "status": result.get("status", "unknown"),
                    "completed_at": result.get("completed_at"),
                }
            )
        return json_response(
            {
                "request_id": request_id,
                "status": "not_found",
            }
        )

    def _list_decisions(self, query_params: dict) -> HandlerResult:
        """List recent decisions."""
        limit = int(query_params.get("limit", 20))

        store = _get_result_store()
        if store:
            try:
                decisions = store.list_recent(limit)
                total = store.count()
                return json_response(
                    {
                        "decisions": decisions,
                        "total": total,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to list from store: {e}")

        # Fallback to in-memory
        decisions = list(_decision_results_fallback.values())[-limit:]
        return json_response(
            {
                "decisions": [
                    {
                        "request_id": d["request_id"],
                        "status": d.get("status"),
                        "completed_at": d.get("completed_at"),
                    }
                    for d in decisions
                ],
                "total": len(_decision_results_fallback),
            }
        )


__all__ = ["DecisionHandler"]
