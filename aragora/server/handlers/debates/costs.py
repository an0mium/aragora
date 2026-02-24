"""
Per-debate cost breakdown handler mixin.

Provides the costs endpoint for individual debates:
- GET /api/v1/debates/{id}/costs - Get per-debate cost breakdown
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from ..openapi_decorator import api_endpoint

try:
    from aragora.rbac.decorators import require_permission
except ImportError:  # pragma: no cover

    def require_permission(*_a, **_kw):  # type: ignore[misc]
        def _noop(fn):  # type: ignore[no-untyped-def]
            return fn

        return _noop


logger = logging.getLogger(__name__)


class _DebatesHandlerProtocol(Protocol):
    """Protocol defining the interface expected by CostsMixin."""

    ctx: dict[str, Any]

    def get_storage(self) -> Any | None:
        """Get debate storage instance."""
        ...


class CostsMixin:
    """Mixin providing per-debate cost breakdown for DebatesHandler.

    Returns cost data from the DebateCostTracker singleton, with
    fallback to the global CostTracker for debates not tracked by
    the per-debate tracker.
    """

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{id}/costs",
        summary="Get per-debate cost breakdown",
        description=(
            "Get a detailed cost breakdown for a specific debate including "
            "total cost, per-agent costs, per-round costs, and model usage."
        ),
        tags=["Debates", "Costs"],
        parameters=[
            {
                "name": "id",
                "in": "path",
                "required": True,
                "schema": {"type": "string"},
                "description": "The debate ID",
            },
        ],
        responses={
            "200": {
                "description": "Cost breakdown returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "object",
                                    "properties": {
                                        "debate_id": {"type": "string"},
                                        "total_cost_usd": {"type": "string"},
                                        "total_tokens_in": {"type": "integer"},
                                        "total_tokens_out": {"type": "integer"},
                                        "total_calls": {"type": "integer"},
                                        "per_agent": {"type": "object"},
                                        "per_round": {"type": "object"},
                                        "model_usage": {"type": "object"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "404": {"description": "Debate not found"},
            "503": {"description": "Storage not available"},
        },
    )
    @handle_errors("debate costs")
    def _get_debate_costs(self: _DebatesHandlerProtocol, debate_id: str) -> HandlerResult:
        """Get per-debate cost breakdown.

        Queries the DebateCostTracker for granular per-agent, per-round,
        and per-model cost data. Falls back to the global CostTracker
        if the debate has no per-call records.

        Args:
            debate_id: The ID of the debate to get costs for.

        Returns:
            HandlerResult with cost breakdown JSON wrapped in {"data": ...}.
        """
        # Verify debate exists
        storage = self.get_storage()
        if not storage:
            return error_response("Storage not available", 503)

        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        # Try the per-debate cost tracker first
        try:
            from aragora.billing.debate_costs import get_debate_cost_tracker

            tracker = get_debate_cost_tracker()
            summary = tracker.get_debate_cost(debate_id)

            if summary.total_calls > 0:
                return json_response({"data": summary.to_dict()})
        except (ImportError, RuntimeError, TypeError) as exc:
            logger.debug("DebateCostTracker unavailable: %s", exc)

        # Fallback: query the global CostTracker for basic debate cost
        try:
            from aragora.billing.cost_tracker import get_cost_tracker

            global_tracker = get_cost_tracker()
            # get_debate_cost is async in CostTracker; run sync for handler
            from aragora.server.http_utils import run_async

            cost_data = run_async(global_tracker.get_debate_cost(debate_id))

            return json_response({"data": cost_data})
        except (ImportError, RuntimeError, TypeError) as exc:
            logger.debug("Global CostTracker unavailable: %s", exc)

        # No cost data available - return empty summary
        return json_response(
            {
                "data": {
                    "debate_id": debate_id,
                    "total_cost_usd": "0",
                    "total_tokens_in": 0,
                    "total_tokens_out": 0,
                    "total_calls": 0,
                    "per_agent": {},
                    "per_round": {},
                    "model_usage": {},
                }
            }
        )


__all__ = ["CostsMixin"]
