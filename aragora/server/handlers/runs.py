"""Run Ledger API handler.

Endpoints:
- GET /api/runs          - List runs with pagination and filters
- GET /api/runs/{run_id} - Get run details
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.pipeline.backbone_contracts import RunLedger
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    handle_errors,
)
from aragora.server.handlers.utils.routing import RouteDispatcher
from aragora.server.validation.query_params import safe_query_int

logger = logging.getLogger(__name__)

_ROUTES = ["/api/runs"]
_RUN_PREFIX = "/api/runs/"


def _get_plan_store():
    """Lazy import to avoid circular imports at module load time."""
    from aragora.pipeline.plan_store import get_plan_store

    return get_plan_store()


def _format_run_summary(run: RunLedger) -> dict[str, Any]:
    """Format a RunLedger into summary fields for list responses."""
    return {
        "run_id": run.run_id,
        "status": run.status,
        "stages": [
            {
                "stage": evt.stage,
                "status": evt.status,
                "created_at": evt.created_at,
            }
            for evt in run.stage_events
        ],
        "execution_id": run.execution_id,
        "receipt_id": run.receipt_id,
        "safety_mode": run.metadata.get("safety_mode", ""),
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def _format_run_detail(run: RunLedger) -> dict[str, Any]:
    """Format a RunLedger into full detail for single-run responses."""
    summary = _format_run_summary(run)
    summary.update(
        {
            "entrypoint": run.entrypoint,
            "plan_id": run.plan_id,
            "debate_id": run.debate_id,
            "taint_flags": list(run.taint_flags),
            "metadata": dict(run.metadata),
        }
    )
    return summary


class RunsHandler(BaseHandler):
    """Handler for run ledger read operations."""

    def __init__(self, ctx: dict[str, Any]) -> None:
        super().__init__(ctx)
        self._get_dispatcher = RouteDispatcher()
        self._get_dispatcher.add_route("/api/runs", self._list_runs)
        self._get_dispatcher.add_route("/api/runs/{run_id}", self._get_run)

    def can_handle(self, path: str) -> bool:
        if path in _ROUTES:
            return True
        if path.startswith(_RUN_PREFIX):
            return True
        return False

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Handle GET requests (public read-only)."""
        self.set_request_context(handler, query_params)
        result = self._get_dispatcher.dispatch(path, query_params)
        if result is not None:
            return result
        return self._try_get_by_id(path)

    # -------------------------------------------------------------------------
    # GET /api/runs
    # -------------------------------------------------------------------------

    @handle_errors("runs list")
    def _list_runs(self, query_params: dict[str, Any]) -> HandlerResult:
        """List runs with optional filters and pagination."""
        store = _get_plan_store()

        status = query_params.get("status")
        plan_id = query_params.get("plan_id")
        debate_id = query_params.get("debate_id")
        execution_id = query_params.get("execution_id")
        limit = safe_query_int(query_params, "limit", default=50, min_val=1, max_val=200)
        offset = safe_query_int(query_params, "offset", default=0, min_val=0, max_val=100000)

        runs = store.list_runs(
            status=status,
            plan_id=plan_id,
            debate_id=debate_id,
            execution_id=execution_id,
            limit=limit,
            offset=offset,
        )

        return json_response(
            {
                "runs": [_format_run_summary(r) for r in runs],
                "limit": limit,
                "offset": offset,
            }
        )

    # -------------------------------------------------------------------------
    # GET /api/runs/{run_id}
    # -------------------------------------------------------------------------

    @handle_errors("run detail")
    def _get_run(self, query_params: dict[str, Any], *, run_id: str = "") -> HandlerResult:
        """Get a single run by ID."""
        if not run_id:
            return error_response("Missing run_id", 400)

        store = _get_plan_store()
        run = store.get_run(run_id)
        if run is None:
            return error_response(f"Run {run_id} not found", 404)

        return json_response(_format_run_detail(run))

    # -------------------------------------------------------------------------
    # Fallback path-param extraction
    # -------------------------------------------------------------------------

    def _try_get_by_id(self, path: str) -> HandlerResult | None:
        """Extract run_id from path and delegate to _get_run."""
        if not path.startswith(_RUN_PREFIX):
            return None
        run_id = path[len(_RUN_PREFIX) :].strip("/")
        if not run_id or "/" in run_id:
            return None
        return self._get_run({}, run_id=run_id)
