"""
Gauntlet Endpoints (FastAPI v2).

Provides async gauntlet stress-test management endpoints:
- Start a gauntlet run
- Get run status
- Get run findings
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext
from aragora.server.fastapi.dependencies.auth import require_permission

from ..middleware.error_handling import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Gauntlet"])


# =============================================================================
# Pydantic Models
# =============================================================================


class StartGauntletRequest(BaseModel):
    """Request to start a gauntlet stress-test."""

    input_content: str = Field(..., description="Content to stress-test")
    input_type: str = Field("spec", description="Type of input: spec, architecture, policy, code, strategy, contract")
    persona: str | None = Field(None, description="Regulatory persona to use (e.g., 'gdpr', 'hipaa')")
    agents: list[str] = Field(
        default_factory=lambda: ["anthropic-api"],
        description="Agent types for the gauntlet",
    )
    profile: str = Field("default", description="Gauntlet profile to use")

    model_config = {  # type: ignore[assignment,dict-item]
        "json_schema_extra": {
            "examples": [
                {
                    "input_content": "Design a user authentication system with OAuth2",
                    "input_type": "spec",
                    "persona": "security",
                    "agents": ["anthropic-api"],
                    "profile": "default",
                }
            ]
        }
    }


class StartGauntletResponse(BaseModel):
    """Response after starting a gauntlet."""

    gauntlet_id: str
    status: str
    message: str


class GauntletStatusResponse(BaseModel):
    """Response for gauntlet status."""

    gauntlet_id: str
    status: str
    input_type: str = ""
    input_summary: str = ""
    persona: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    model_config = {"extra": "allow"}


class FindingSummary(BaseModel):
    """Summary of a gauntlet finding."""

    id: str = ""
    category: str = ""
    severity: str = ""
    severity_level: str = ""
    title: str = ""
    description: str = ""

    model_config = {"extra": "allow"}


class FindingsResponse(BaseModel):
    """Response for gauntlet findings."""

    gauntlet_id: str
    findings: list[FindingSummary]
    total: int
    verdict: str | None = None
    confidence: float | None = None


# =============================================================================
# Dependencies
# =============================================================================


async def get_gauntlet_storage(request: Request):
    """Dependency to get gauntlet storage.

    Tries app context first, then falls back to the gauntlet storage module.
    """
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        store = ctx.get("gauntlet_storage")
        if store:
            return store

    # Fall back to gauntlet storage module
    try:
        from aragora.gauntlet.storage import GauntletStorage

        return GauntletStorage()
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.warning(f"Gauntlet storage not available: {e}")
        raise HTTPException(status_code=503, detail="Gauntlet storage not available")


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/gauntlet/run", response_model=StartGauntletResponse, status_code=202)
async def start_gauntlet(
    body: StartGauntletRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("gauntlet:run")),
) -> StartGauntletResponse:
    """
    Start a new gauntlet stress-test.

    Returns immediately with a gauntlet ID. The gauntlet runs in the background.
    Use GET /gauntlet/{run_id}/status to poll for status.
    """
    try:
        gauntlet_id = f"gauntlet-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        input_hash = hashlib.sha256(body.input_content.encode()).hexdigest()
        input_summary = (
            body.input_content[:200] + "..."
            if len(body.input_content) > 200
            else body.input_content
        )

        # Store initial state in gauntlet runs (in-memory)
        try:
            from aragora.server.handlers.gauntlet.storage import get_gauntlet_runs

            gauntlet_runs = get_gauntlet_runs()
            gauntlet_runs[gauntlet_id] = {
                "gauntlet_id": gauntlet_id,
                "status": "pending",
                "input_type": body.input_type,
                "input_summary": input_summary,
                "input_hash": input_hash,
                "persona": body.persona,
                "profile": body.profile,
                "created_at": datetime.now().isoformat(),
                "result": None,
            }
        except (ImportError, RuntimeError) as e:
            logger.debug(f"Could not store gauntlet run in memory: {e}")

        # Persist to storage if available
        ctx = getattr(request.app.state, "context", None)
        if ctx:
            store = ctx.get("gauntlet_storage")
            if store and hasattr(store, "save_inflight"):
                try:
                    store.save_inflight(
                        gauntlet_id=gauntlet_id,
                        status="pending",
                        input_type=body.input_type,
                        input_summary=input_summary,
                        input_hash=input_hash,
                        persona=body.persona,
                        profile=body.profile,
                        agents=body.agents,
                    )
                except (OSError, RuntimeError, ValueError) as e:
                    logger.warning(f"Failed to persist gauntlet run: {e}")

        return StartGauntletResponse(
            gauntlet_id=gauntlet_id,
            status="pending",
            message="Gauntlet stress-test started",
        )

    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.exception(f"Failed to start gauntlet: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start gauntlet: {e}")


@router.get("/gauntlet/{run_id}/status", response_model=GauntletStatusResponse)
async def get_gauntlet_status(
    run_id: str,
    store=Depends(get_gauntlet_storage),
) -> GauntletStatusResponse:
    """
    Get the status of a gauntlet run.

    Returns current status, input info, and results if completed.
    """
    try:
        # Check in-memory runs first
        try:
            from aragora.server.handlers.gauntlet.storage import get_gauntlet_runs

            gauntlet_runs = get_gauntlet_runs()
            if run_id in gauntlet_runs:
                run = gauntlet_runs[run_id]
                return GauntletStatusResponse(
                    gauntlet_id=run.get("gauntlet_id", run_id),
                    status=run.get("status", "unknown"),
                    input_type=run.get("input_type", ""),
                    input_summary=run.get("input_summary", ""),
                    persona=run.get("persona"),
                    created_at=run.get("created_at"),
                    completed_at=run.get("completed_at"),
                    result=run.get("result"),
                    error=run.get("error"),
                )
        except (ImportError, RuntimeError):
            pass

        # Check persistent storage - inflight
        if hasattr(store, "get_inflight"):
            inflight = store.get_inflight(run_id)
            if inflight:
                inflight_dict = inflight.to_dict() if hasattr(inflight, "to_dict") else inflight
                return GauntletStatusResponse(
                    gauntlet_id=run_id,
                    status=inflight_dict.get("status", "unknown") if isinstance(inflight_dict, dict) else "unknown",
                    input_type=inflight_dict.get("input_type", "") if isinstance(inflight_dict, dict) else "",
                    input_summary=inflight_dict.get("input_summary", "") if isinstance(inflight_dict, dict) else "",
                    persona=inflight_dict.get("persona") if isinstance(inflight_dict, dict) else None,
                    created_at=inflight_dict.get("created_at") if isinstance(inflight_dict, dict) else None,
                )

        # Check completed results
        if hasattr(store, "get"):
            stored = store.get(run_id)
            if stored:
                return GauntletStatusResponse(
                    gauntlet_id=run_id,
                    status="completed",
                    result=stored if isinstance(stored, dict) else (stored.to_dict() if hasattr(stored, "to_dict") else None),
                )

        raise NotFoundError(f"Gauntlet run {run_id} not found")

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception(f"Error getting gauntlet status {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get gauntlet status: {e}")


@router.get("/gauntlet/{run_id}/findings", response_model=FindingsResponse)
async def get_gauntlet_findings(
    run_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max findings to return"),
    offset: int = Query(0, ge=0, description="Number of findings to skip"),
    severity: str | None = Query(None, description="Filter by severity: CRITICAL, HIGH, MEDIUM, LOW"),
    store=Depends(get_gauntlet_storage),
) -> FindingsResponse:
    """
    Get findings from a gauntlet run.

    Returns paginated list of findings from the gauntlet stress-test.
    """
    try:
        result_data = None
        verdict = None
        confidence = None

        # Check in-memory runs first
        try:
            from aragora.server.handlers.gauntlet.storage import get_gauntlet_runs

            gauntlet_runs = get_gauntlet_runs()
            if run_id in gauntlet_runs:
                run = gauntlet_runs[run_id]
                if run.get("status") != "completed":
                    return FindingsResponse(
                        gauntlet_id=run_id,
                        findings=[],
                        total=0,
                        verdict=None,
                        confidence=None,
                    )
                result_data = run.get("result", {})
        except (ImportError, RuntimeError):
            pass

        # Check persistent storage
        if result_data is None:
            if hasattr(store, "get"):
                stored = store.get(run_id)
                if stored:
                    result_data = stored if isinstance(stored, dict) else (stored.to_dict() if hasattr(stored, "to_dict") else {})

        if result_data is None:
            raise NotFoundError(f"Gauntlet run {run_id} not found")

        # Extract findings
        if isinstance(result_data, dict):
            raw_findings = result_data.get("findings", [])
            verdict = result_data.get("verdict")
            confidence = result_data.get("confidence")
        else:
            raw_findings = getattr(result_data, "findings", [])
            verdict = getattr(result_data, "verdict", None)
            confidence = getattr(result_data, "confidence", None)

        # Convert findings to summary objects
        findings = []
        for f in raw_findings:
            if isinstance(f, dict):
                finding = FindingSummary(
                    id=f.get("id", f.get("finding_id", "")),
                    category=f.get("category", ""),
                    severity=f.get("severity", ""),
                    severity_level=f.get("severity_level", f.get("severity", "")),
                    title=f.get("title", ""),
                    description=f.get("description", "")[:500],
                )
            else:
                finding = FindingSummary(
                    id=getattr(f, "finding_id", getattr(f, "id", "")),
                    category=getattr(f, "category", ""),
                    severity=getattr(f, "severity", ""),
                    severity_level=getattr(f, "severity_level", getattr(f, "severity", "")),
                    title=getattr(f, "title", ""),
                    description=str(getattr(f, "description", ""))[:500],
                )
            findings.append(finding)

        # Filter by severity
        if severity:
            severity_upper = severity.upper()
            findings = [
                f for f in findings
                if f.severity.upper() == severity_upper or f.severity_level.upper() == severity_upper
            ]

        total = len(findings)
        findings = findings[offset:offset + limit]

        return FindingsResponse(
            gauntlet_id=run_id,
            findings=findings,
            total=total,
            verdict=verdict,
            confidence=confidence,
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception(f"Error getting gauntlet findings {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get gauntlet findings: {e}")
