"""
Run Ledger Endpoints (FastAPI v2).

Provides async run ledger endpoints:
- List runs with pagination and filters
- Get run by ID
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Runs"])


# =============================================================================
# Pydantic Models
# =============================================================================


class RunStageResponse(BaseModel):
    """A single stage event in a run."""

    stage: str
    status: str
    created_at: str


class RunSummary(BaseModel):
    """Summary of a run for list views."""

    run_id: str
    status: str
    stages: list[RunStageResponse] = Field(default_factory=list)
    execution_id: str = ""
    receipt_id: str = ""
    safety_mode: str = ""
    created_at: str = ""
    updated_at: str = ""


class RunDetail(RunSummary):
    """Full detail of a single run."""

    entrypoint: str = ""
    plan_id: str = ""
    debate_id: str = ""
    taint_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunListResponse(BaseModel):
    """Paginated list of runs."""

    runs: list[RunSummary]
    limit: int
    offset: int


# =============================================================================
# Helpers
# =============================================================================


def _get_plan_store():
    from aragora.pipeline.plan_store import get_plan_store

    return get_plan_store()


def _format_summary(run) -> dict[str, Any]:
    from aragora.server.handlers.runs import _format_run_summary

    return _format_run_summary(run)


def _format_detail(run) -> dict[str, Any]:
    from aragora.server.handlers.runs import _format_run_detail

    return _format_run_detail(run)


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    request: Request,
    status: str | None = Query(None, description="Filter by status"),
    plan_id: str | None = Query(None, description="Filter by plan ID"),
    debate_id: str | None = Query(None, description="Filter by debate ID"),
    execution_id: str | None = Query(None, description="Filter by execution ID"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, le=100000, description="Offset for pagination"),
):
    """List run ledgers with optional filters and pagination."""
    store = _get_plan_store()
    runs = store.list_runs(
        status=status,
        plan_id=plan_id,
        debate_id=debate_id,
        execution_id=execution_id,
        limit=limit,
        offset=offset,
    )
    return {
        "runs": [_format_summary(r) for r in runs],
        "limit": limit,
        "offset": offset,
    }


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(request: Request, run_id: str):
    """Get a single run ledger by ID."""
    from fastapi import HTTPException

    store = _get_plan_store()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return _format_detail(run)
