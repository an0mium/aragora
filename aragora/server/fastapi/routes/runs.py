"""FastAPI routes for /api/runs — thin routing layer over runs handler."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from aragora.pipeline.plan_store import PlanStore
from aragora.server.handlers.runs import get_run_by_id, get_runs

router = APIRouter(prefix="/api/runs", tags=["runs"])

_store: PlanStore | None = None


def _get_store() -> PlanStore:
    global _store  # noqa: PLW0603
    if _store is None:
        _store = PlanStore()
    return _store


@router.get("")
@router.get("/")
def list_runs(
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """List backbone runs with optional status filter."""
    return get_runs(_get_store(), status=status, limit=limit, offset=offset)


@router.get("/{run_id}")
def read_run(run_id: str) -> dict[str, Any]:
    """Get a single backbone run by ID."""
    result = get_run_by_id(_get_store(), run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return result
