"""Backbone run ledger handlers.

Endpoints:
- GET /api/runs          - List persisted backbone runs
- GET /api/runs/{run_id} - Fetch one persisted backbone run
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from aragora.pipeline.backbone_contracts import RunLedger
from aragora.server.handlers.base import HandlerResult, error_response, json_response


def _get_plan_store() -> Any:
    """Lazy import to avoid handler import cycles."""
    from aragora.pipeline.plan_store import get_plan_store

    return get_plan_store()


def _coerce_int(
    value: Any,
    *,
    default: int,
    min_value: int,
    max_value: int | None = None,
) -> int:
    """Parse a bounded integer from query params."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default

    if parsed < min_value:
        return min_value
    if max_value is not None and parsed > max_value:
        return max_value
    return parsed


def _event_value(event: Any, field: str) -> str:
    """Read one string field from a RunStageEvent or serialized dict."""
    if isinstance(event, dict):
        value = event.get(field, "")
    else:
        value = getattr(event, field, "")
    return str(value or "").strip()


def _stage_payload(stage_events: Iterable[Any]) -> list[dict[str, str]]:
    """Collapse stage events into one latest-status record per stage."""
    stage_index: dict[str, dict[str, str]] = {}
    stage_order: list[str] = []

    for event in stage_events:
        stage = _event_value(event, "stage")
        if not stage:
            continue

        if stage not in stage_index:
            stage_order.append(stage)

        stage_index[stage] = {
            "stage": stage,
            "status": _event_value(event, "status") or "unknown",
        }

    return [stage_index[stage] for stage in stage_order]


def _run_payload(run: RunLedger) -> dict[str, Any]:
    """Serialize the compact run payload used by the runs endpoints."""
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    safety_mode = str(metadata.get("safety_mode", "") or "").strip() or None

    return {
        "run_id": run.run_id,
        "status": run.status,
        "stages": _stage_payload(run.stage_events),
        "execution_id": run.execution_id or None,
        "receipt_id": run.receipt_id or None,
        "safety_mode": safety_mode,
    }


def _get_backbone_run(store: Any, run_id: str) -> RunLedger | None:
    """Read a single run, preferring the explicit backbone accessor when available."""
    getter = getattr(store, "get_backbone_run", None)
    if callable(getter):
        return cast(RunLedger | None, getter(run_id))

    getter = getattr(store, "get_run", None)
    if callable(getter):
        return cast(RunLedger | None, getter(run_id))

    return None


def _list_backbone_runs(
    store: Any,
    *,
    status: str | None,
    limit: int,
    offset: int,
) -> list[RunLedger]:
    """List runs, preferring the explicit backbone accessor when available."""
    lister = getattr(store, "list_backbone_runs", None)
    if callable(lister):
        return cast(list[RunLedger], lister(status=status, limit=limit, offset=offset))

    lister = getattr(store, "list_runs", None)
    if callable(lister):
        return cast(list[RunLedger], lister(status=status, limit=limit, offset=offset))

    return []


def handle_runs_list(
    query_params: dict[str, Any] | None = None,
    *,
    store: Any | None = None,
) -> HandlerResult:
    """Handle GET /api/runs."""
    params = query_params or {}
    run_store = store or _get_plan_store()

    status = str(params.get("status", "") or "").strip() or None
    limit = _coerce_int(params.get("limit", 50), default=50, min_value=1, max_value=100)
    offset = _coerce_int(params.get("offset", 0), default=0, min_value=0)

    runs = _list_backbone_runs(run_store, status=status, limit=limit, offset=offset)
    return json_response({"runs": [_run_payload(run) for run in runs]})


def handle_run_detail(
    run_id: str,
    *,
    store: Any | None = None,
) -> HandlerResult:
    """Handle GET /api/runs/{run_id}."""
    normalized_run_id = str(run_id or "").strip()
    if not normalized_run_id:
        return error_response("run_id is required", 400)

    run_store = store or _get_plan_store()
    run = _get_backbone_run(run_store, normalized_run_id)
    if run is None:
        return error_response("Run not found", 404)

    return json_response({"run": _run_payload(run)})


__all__ = [
    "handle_run_detail",
    "handle_runs_list",
]
