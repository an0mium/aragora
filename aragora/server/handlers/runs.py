"""Handlers for GET /api/runs and GET /api/runs/{run_id}.

Delegates to :class:`~aragora.pipeline.plan_store.PlanStore` for
:class:`~aragora.pipeline.backbone_contracts.RunLedger` data.
"""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)


def _get_plan_store(request: web.Request) -> Any:
    """Resolve a PlanStore from the request's app context."""
    ctx = request.app.get("context") or {}
    store = ctx.get("plan_store")
    if store is None:
        raise web.HTTPServiceUnavailable(
            text='{"error":"plan_store not configured"}',
            content_type="application/json",
        )
    return store


def _ledger_summary(ledger: Any) -> dict[str, Any]:
    """Return the subset of RunLedger fields required by the API."""
    stages = [evt.to_dict() for evt in getattr(ledger, "stage_events", [])]
    metadata = getattr(ledger, "metadata", {}) or {}
    return {
        "run_id": ledger.run_id,
        "status": ledger.status,
        "stages": stages,
        "execution_id": ledger.execution_id,
        "receipt_id": ledger.receipt_id,
        "safety_mode": metadata.get("safety_mode", "standard"),
    }


async def handle_list_runs(request: web.Request) -> web.Response:
    """GET /api/runs — list backbone runs with optional filters."""
    store = _get_plan_store(request)

    status = request.query.get("status")
    limit = int(request.query.get("limit", "50"))
    offset = int(request.query.get("offset", "0"))

    runs = store.list_runs(status=status, limit=limit, offset=offset)
    return web.json_response(
        {"runs": [_ledger_summary(r) for r in runs], "total": len(runs)},
    )


async def handle_get_run(request: web.Request) -> web.Response:
    """GET /api/runs/{run_id} — fetch a single backbone run."""
    store = _get_plan_store(request)
    run_id = request.match_info["run_id"]

    ledger = store.get_run(run_id)
    if ledger is None:
        raise web.HTTPNotFound(
            text=f'{{"error":"run {run_id} not found"}}',
            content_type="application/json",
        )
    return web.json_response(_ledger_summary(ledger))
