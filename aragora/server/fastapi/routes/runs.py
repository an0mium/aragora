"""FastAPI routes for backbone run ledger reads."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aragora.server.handlers.runs import (
    _count_backbone_runs,
    _get_backbone_run,
    _list_backbone_runs,
    serialize_run_detail,
    serialize_run_summary,
)

router = APIRouter(tags=["Runs"])


class RunStageSummary(BaseModel):
    stage: str
    status: str


class RunSummaryResponse(BaseModel):
    run_id: str
    status: str
    stages: list[RunStageSummary] = Field(default_factory=list)
    execution_id: str = ""
    receipt_id: str = ""
    safety_mode: str = ""


class RunDetailResponse(RunSummaryResponse):
    entrypoint: str = ""
    plan_id: str = ""
    debate_id: str = ""
    created_at: str = ""
    updated_at: str = ""


class RunListResponse(BaseModel):
    runs: list[RunSummaryResponse]
    total: int
    limit: int
    offset: int


def _resolve_plan_store(request: Request) -> Any:
    ctx = getattr(request.app.state, "context", None)
    if ctx and ctx.get("plan_store") is not None:
        return ctx["plan_store"]

    from aragora.pipeline.plan_store import get_plan_store

    return get_plan_store()


@router.get("/api/runs", response_model=RunListResponse)
async def list_runs(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
    execution_id: str | None = Query(None),
    store: Any = Depends(_resolve_plan_store),
) -> RunListResponse:
    del request
    runs = _list_backbone_runs(
        store,
        status=status,
        execution_id=execution_id,
        limit=limit,
        offset=offset,
    )
    total = _count_backbone_runs(
        store,
        status=status,
        execution_id=execution_id,
        fallback_count=len(runs),
    )
    return RunListResponse(
        runs=[RunSummaryResponse(**serialize_run_summary(run)) for run in runs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/api/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: str, store: Any = Depends(_resolve_plan_store)) -> RunDetailResponse:
    run = _get_backbone_run(store, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return RunDetailResponse(**serialize_run_detail(run))


__all__ = ["router"]
