"""FastAPI v2 routes for backbone runs.

GET /api/v2/runs           — list runs
GET /api/v2/runs/{run_id}  — get single run
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Runs"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class StageEventResponse(BaseModel):
    stage: str
    status: str
    event_id: str = ""
    artifact_ref: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class RunResponse(BaseModel):
    run_id: str
    status: str
    stages: list[StageEventResponse] = Field(default_factory=list)
    execution_id: str = ""
    receipt_id: str = ""
    safety_mode: str = "standard"


class RunListResponse(BaseModel):
    runs: list[RunResponse]
    total: int


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def _get_plan_store(request: Request) -> Any:
    ctx = getattr(request.app.state, "context", None) or {}
    store = ctx.get("plan_store")
    if store is None:
        raise HTTPException(status_code=503, detail="plan_store not configured")
    return store


def _ledger_to_response(ledger: Any) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    request: Request,
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    store: Any = Depends(_get_plan_store),
) -> RunListResponse:
    """List backbone pipeline runs."""
    runs = store.list_runs(status=status, limit=limit, offset=offset)
    return RunListResponse(
        runs=[RunResponse(**_ledger_to_response(r)) for r in runs],
        total=len(runs),
    )


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    request: Request,
    run_id: str,
    store: Any = Depends(_get_plan_store),
) -> RunResponse:
    """Get a single backbone pipeline run."""
    ledger = store.get_run(run_id)
    if ledger is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return RunResponse(**_ledger_to_response(ledger))
