"""
Run ledger endpoint handlers.

Exposes backbone RunLedger data for monitoring pipeline runs.

Endpoints:
- GET /api/runs       - List runs with optional filters
- GET /api/runs/:id   - Get run details by ID
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.pipeline.plan_store import PlanStore
from aragora.server.validation import validate_path_segment

from .base import (
    HandlerResult,
    error_response,
    json_response,
    handle_errors,
)

logger = logging.getLogger(__name__)


def _run_summary(run_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract the subset of fields specified by the task contract."""
    return {
        "run_id": run_dict.get("run_id", ""),
        "status": run_dict.get("status", ""),
        "stages": [evt.get("stage", "") for evt in run_dict.get("stage_events", [])],
        "execution_id": run_dict.get("execution_id", ""),
        "receipt_id": run_dict.get("receipt_id", ""),
        "safety_mode": (run_dict.get("metadata") or {}).get("safety_mode", ""),
    }


@handle_errors
async def get_runs(request: Any) -> HandlerResult:
    """List runs with optional query filters.

    Query params: status, plan_id, execution_id, limit, offset.
    """
    query = getattr(request, "query", {}) if request is not None else {}

    status = query.get("status")
    plan_id = query.get("plan_id")
    execution_id = query.get("execution_id")

    try:
        limit = int(query.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    try:
        offset = int(query.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0

    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    store = PlanStore()
    runs = store.list_runs(
        status=status,
        plan_id=plan_id,
        execution_id=execution_id,
        limit=limit,
        offset=offset,
    )
    return json_response({"runs": [_run_summary(r.to_dict()) for r in runs], "total": len(runs)})


@handle_errors
async def get_run_by_id(request: Any) -> HandlerResult:
    """Fetch a single run by its run_id."""
    run_id: str = ""
    match_info = getattr(request, "match_info", None)
    if match_info is not None:
        run_id = match_info.get("run_id", "")
    if not run_id:
        return error_response("Missing run_id", 400)

    is_valid, validation_error = validate_path_segment(run_id, "run_id")
    if not is_valid:
        return error_response(str(validation_error), 400)

    store = PlanStore()
    run = store.get_run(run_id)
    if run is None:
        return error_response("Run not found", 404)
    return json_response(_run_summary(run.to_dict()))
