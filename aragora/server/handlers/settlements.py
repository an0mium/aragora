"""
Settlement HTTP handler.

Endpoints for managing debate claim settlements:
- GET  /api/v1/settlements          - List pending settlements
- GET  /api/v1/settlements/history  - Get settlement history
- GET  /api/v1/settlements/summary  - Get settlement summary stats
- GET  /api/v1/settlements/{id}     - Get a specific settlement
- POST /api/v1/settlements/{id}/settle - Settle a claim with an outcome
- POST /api/v1/settlements/batch    - Settle multiple claims at once
- GET  /api/v1/settlements/agent/{agent}/accuracy - Agent accuracy stats

Follows the BaseHandler pattern from base.py with HandlerResult.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from .utils.decorators import require_permission

logger = logging.getLogger(__name__)


class SettlementHandler(BaseHandler):
    """Handler for settlement API endpoints.

    Provides REST APIs for listing pending settlements, submitting
    outcomes, and viewing settlement history and agent accuracy.
    """

    ROUTES = [
        "/api/settlements",
    ]

    ROUTE_PREFIXES = [
        "/api/settlements",
        "/api/settlements/",
        "/api/v1/settlements",
        "/api/v1/settlements/",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None) -> None:
        """Initialize handler with server context."""
        self.ctx = ctx or {}

    def _get_tracker(self) -> Any:
        """Get or create the SettlementTracker from server context."""
        tracker = self.ctx.get("settlement_tracker")
        if tracker is not None:
            return tracker

        # Lazy-create a tracker from available context
        from aragora.debate.settlement import SettlementTracker

        elo_system = self.ctx.get("elo_system")
        calibration_tracker = self.ctx.get("calibration_tracker")
        knowledge_mound = self.ctx.get("knowledge_mound")

        tracker = SettlementTracker(
            elo_system=elo_system,
            calibration_tracker=calibration_tracker,
            knowledge_mound=knowledge_mound,
        )
        self.ctx["settlement_tracker"] = tracker
        return tracker

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given request."""
        normalized = strip_version_prefix(path)
        return normalized == "/api/settlements" or normalized.startswith("/api/settlements/")

    @require_permission("settlements:read")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route GET requests."""
        normalized = strip_version_prefix(path)
        path_clean = normalized.rstrip("/")

        if path_clean == "/api/settlements":
            return self._list_pending(query_params)
        if path_clean == "/api/settlements/history":
            return self._get_history(query_params)
        if path_clean == "/api/settlements/summary":
            return self._get_summary()

        # GET /api/settlements/agent/{agent}/accuracy
        if "/settlements/agent/" in path_clean and path_clean.endswith("/accuracy"):
            segments = path_clean.split("/")
            # /api/settlements/agent/{agent}/accuracy
            agent_idx = segments.index("agent") + 1 if "agent" in segments else -1
            if agent_idx > 0 and agent_idx < len(segments):
                return self._get_agent_accuracy(segments[agent_idx])

        # GET /api/settlements/{id}
        if "/settlements/" in normalized and not path_clean.endswith(("/settle", "/batch")):
            settlement_id = path_clean.split("/")[-1]
            if settlement_id and settlement_id not in ("history", "summary", "agent"):
                return self._get_settlement(settlement_id)

        return None

    @handle_errors("settle claim")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route POST requests."""
        normalized = strip_version_prefix(path)
        path_clean = normalized.rstrip("/")

        # POST /api/settlements/batch
        if path_clean == "/api/settlements/batch":
            body = self.read_json_body(handler)
            if body is None:
                return error_response("Invalid JSON body", 400)
            return self._settle_batch(body)

        # POST /api/settlements/{id}/settle
        if path_clean.endswith("/settle"):
            body = self.read_json_body(handler)
            if body is None:
                return error_response("Invalid JSON body", 400)
            # Extract settlement_id: /api/settlements/{id}/settle
            segments = path_clean.split("/")
            settle_idx = segments.index("settle") if "settle" in segments else -1
            if settle_idx > 0:
                settlement_id = segments[settle_idx - 1]
                return self._settle_single(settlement_id, body)

        return None

    # ------------------------------------------------------------------
    # GET /api/v1/settlements
    # ------------------------------------------------------------------

    @handle_errors("list pending settlements")
    def _list_pending(self, query_params: dict[str, Any]) -> HandlerResult:
        """List pending (unsettled) settlements.

        Query params:
            debate_id: Filter by debate ID
            domain: Filter by domain
            limit: Max results (default 100)
        """
        tracker = self._get_tracker()

        debate_id = query_params.get("debate_id")
        domain = query_params.get("domain")
        limit = int(query_params.get("limit", "100"))

        pending = tracker.get_pending(
            debate_id=debate_id,
            domain=domain,
            limit=limit,
        )

        return json_response({
            "data": {
                "settlements": [r.to_dict() for r in pending],
                "count": len(pending),
                "status": "pending",
            }
        })

    # ------------------------------------------------------------------
    # GET /api/v1/settlements/history
    # ------------------------------------------------------------------

    @handle_errors("get settlement history")
    def _get_history(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get settled (resolved) settlements.

        Query params:
            debate_id: Filter by debate ID
            author: Filter by claim author
            limit: Max results (default 100)
        """
        tracker = self._get_tracker()

        debate_id = query_params.get("debate_id")
        author = query_params.get("author")
        limit = int(query_params.get("limit", "100"))

        history = tracker.get_history(
            debate_id=debate_id,
            author=author,
            limit=limit,
        )

        return json_response({
            "data": {
                "settlements": [r.to_dict() for r in history],
                "count": len(history),
                "status": "settled",
            }
        })

    # ------------------------------------------------------------------
    # GET /api/v1/settlements/summary
    # ------------------------------------------------------------------

    @handle_errors("get settlement summary")
    def _get_summary(self) -> HandlerResult:
        """Get overall settlement summary statistics."""
        tracker = self._get_tracker()
        summary = tracker.get_summary()
        return json_response({"data": summary})

    # ------------------------------------------------------------------
    # GET /api/v1/settlements/{id}
    # ------------------------------------------------------------------

    @handle_errors("get settlement")
    def _get_settlement(self, settlement_id: str) -> HandlerResult:
        """Get a specific settlement by ID."""
        tracker = self._get_tracker()
        record = tracker.get_settlement(settlement_id)

        if record is None:
            return error_response(f"Settlement not found: {settlement_id}", 404)

        return json_response({"data": record.to_dict()})

    # ------------------------------------------------------------------
    # GET /api/v1/settlements/agent/{agent}/accuracy
    # ------------------------------------------------------------------

    @handle_errors("get agent accuracy")
    def _get_agent_accuracy(self, agent: str) -> HandlerResult:
        """Get accuracy statistics for a specific agent."""
        tracker = self._get_tracker()
        accuracy = tracker.get_agent_accuracy(agent)
        return json_response({"data": accuracy})

    # ------------------------------------------------------------------
    # POST /api/v1/settlements/{id}/settle
    # ------------------------------------------------------------------

    @handle_errors("settle single claim")
    def _settle_single(self, settlement_id: str, body: dict[str, Any]) -> HandlerResult:
        """Settle a single claim with an outcome.

        Body:
            outcome: str  -- "correct", "incorrect", or "partial" (required)
            evidence: str -- Supporting evidence for the outcome
            settled_by: str -- Who/what resolved the settlement
        """
        outcome = body.get("outcome", "")
        if not outcome:
            return error_response("outcome is required", 400)

        if outcome not in ("correct", "incorrect", "partial"):
            return error_response(
                "outcome must be 'correct', 'incorrect', or 'partial'", 400
            )

        evidence = body.get("evidence", "")
        settled_by = body.get("settled_by", "api")

        tracker = self._get_tracker()

        try:
            result = tracker.settle(
                settlement_id=settlement_id,
                outcome=outcome,
                evidence=evidence,
                settled_by=settled_by,
            )
            return json_response({"data": result.to_dict()})
        except KeyError:
            return error_response(f"Settlement not found: {settlement_id}", 404)
        except ValueError as e:
            return error_response(str(e), 409)

    # ------------------------------------------------------------------
    # POST /api/v1/settlements/batch
    # ------------------------------------------------------------------

    @handle_errors("settle batch")
    def _settle_batch(self, body: dict[str, Any]) -> HandlerResult:
        """Settle multiple claims at once.

        Body:
            settlements: list[dict]  -- List of {settlement_id, outcome, evidence}
            settled_by: str -- Who/what resolved the settlements
        """
        settlements = body.get("settlements", [])
        if not settlements:
            return error_response("settlements list is required", 400)

        settled_by = body.get("settled_by", "api")

        tracker = self._get_tracker()
        results = tracker.settle_batch(
            settlements=settlements,
            settled_by=settled_by,
        )

        return json_response({
            "data": {
                "results": [r.to_dict() for r in results],
                "count": len(results),
            }
        })
