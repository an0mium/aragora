"""
Run Ledger Endpoints (FastAPI v2).

Provides endpoints for listing and retrieving backbone run ledgers:
- GET /api/v2/runs            - List runs with optional filters
- GET /api/v2/runs/{run_id}   - Get run details by ID
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from aragora.pipeline.plan_store import PlanStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Runs"])


# -- Pydantic models ----------------------------------------------------------


class RunSummary(BaseModel):
    """Summary representation of a backbone run."""

    run_id: str
    status: str = ""
    stages: list[str] = Field(default_factory=list)
    execution_id: str = ""
    receipt_id: str = ""
    safety_mode: str = ""


class RunListResponse(BaseModel):
    """Paginated list of runs."""

    runs: list[RunSummary]
    total: int


# -- Helpers -------------------------------------------------------------------


def _to_summary(run_dict: dict[str, Any]) -> RunSummary:
    return RunSummary(
        run_id=run_dict.get("run_id", ""),
        status=run_dict.get("status", ""),
        stages=[evt.get("stage", "") for evt in run_dict.get("stage_events", [])],
        execution_id=run_dict.get("execution_id", ""),
        receipt_id=run_dict.get("receipt_id", ""),
        safety_mode=(run_dict.get("metadata") or {}).get("safety_mode", ""),
    )


# -- Routes --------------------------------------------------------------------


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    status: str | None = Query(None),
    plan_id: str | None = Query(None),
    execution_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> RunListResponse:
    """List backbone runs with optional filters."""
    store = PlanStore()
    runs = store.list_runs(
        status=status,
        plan_id=plan_id,
        execution_id=execution_id,
        limit=limit,
        offset=offset,
    )
    summaries = [_to_summary(r.to_dict()) for r in runs]
    return RunListResponse(runs=summaries, total=len(summaries))


@router.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(run_id: str) -> RunSummary:
    """Fetch a single run by ID."""
    store = PlanStore()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _to_summary(run.to_dict())
