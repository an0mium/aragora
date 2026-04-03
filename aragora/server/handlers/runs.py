"""Handlers for /api/runs endpoints — exposes RunLedger data."""

from __future__ import annotations

from typing import Any

from aragora.pipeline.plan_store import PlanStore
from aragora.pipeline.backbone_contracts import RunLedger


def _run_summary(run: RunLedger) -> dict[str, Any]:
    """Extract the subset of RunLedger fields required by the API."""
    return {
        "run_id": run.run_id,
        "status": run.status,
        "stages": [evt.to_dict() for evt in run.stage_events],
        "execution_id": run.execution_id,
        "receipt_id": run.receipt_id,
        "safety_mode": run.metadata.get("safety_mode", "standard"),
    }


def get_backbone_run(store: PlanStore, run_id: str) -> RunLedger | None:
    """Convenience wrapper surfacing PlanStore.get_run as backbone accessor."""
    return store.get_run(run_id)


def get_runs(
    store: PlanStore,
    *,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Return a paginated list of run summaries.

    Returns ``{"runs": [...], "total": <int>}`` where each entry contains
    the fields: run_id, status, stages, execution_id, receipt_id, safety_mode.
    """
    runs = store.list_runs(status=status, limit=limit, offset=offset)
    return {
        "runs": [_run_summary(r) for r in runs],
        "total": len(runs),
    }


def get_run_by_id(store: PlanStore, run_id: str) -> dict[str, Any] | None:
    """Return a single run summary or ``None`` when not found.

    Delegates to :func:`get_backbone_run` and projects the standard
    run_id / status / stages / execution_id / receipt_id / safety_mode shape.
    """
    run = get_backbone_run(store, run_id)
    if run is None:
        return None
    return _run_summary(run)
