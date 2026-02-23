"""
Decision Endpoints (FastAPI v2).

Provides async decision orchestration endpoints:
- Start a new debate (returns immediately)
- Get debate status
- Cancel a running debate
- Subscribe to debate events via SSE
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aragora.config import DEFAULT_CONSENSUS, DEFAULT_ROUNDS
from aragora.config.settings import get_settings
from aragora.rbac.models import AuthorizationContext
from aragora.server.fastapi.dependencies.auth import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Decisions"])

# =============================================================================
# Pydantic Models
# =============================================================================


class StartDebateRequest(BaseModel):
    """Request to start a new debate."""

    task: str = Field(..., description="The topic or question to debate")
    agents: list[str] | None = Field(
        None, description="List of agent names (defaults to configured agents)"
    )
    rounds: int = Field(DEFAULT_ROUNDS, ge=1, le=20, description="Number of debate rounds")
    consensus: str = Field(
        DEFAULT_CONSENSUS,
        description="Consensus mechanism: majority, unanimous, supermajority, etc.",
    )
    timeout: float = Field(600.0, ge=30, le=3600, description="Max execution time in seconds")
    priority: int = Field(0, ge=0, le=10, description="Priority (higher = more urgent)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")

    # Feature flags
    enable_streaming: bool = Field(True, description="Enable event streaming")
    enable_checkpointing: bool = Field(True, description="Enable debate checkpointing")
    enable_memory: bool = Field(True, description="Enable memory system")

    model_config = {  # type: ignore[assignment,dict-item]
        "json_schema_extra": {
            "examples": [
                {
                    "task": "What is the best caching strategy for a high-traffic API?",
                    "agents": get_settings().agent.default_agent_list[:3],  # type: ignore[dict-item,list-item]
                    "rounds": DEFAULT_ROUNDS,
                    "consensus": DEFAULT_CONSENSUS,
                }
            ]
        }
    }


class DebateResponse(BaseModel):
    """Response for debate operations."""

    id: str
    task: str
    status: str
    progress: float = 0.0
    current_round: int = 0
    total_rounds: int = DEFAULT_ROUNDS
    agents: list[str] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class StartDebateResponse(BaseModel):
    """Response after starting a debate."""

    id: str
    status: str
    message: str
    events_url: str


class CancelResponse(BaseModel):
    """Response after cancelling a debate."""

    id: str
    cancelled: bool
    message: str


# =============================================================================
# Dependencies
# =============================================================================


async def get_decision_service(request: Request):
    """Dependency to get the decision service from app state."""
    from aragora.debate.decision_service import get_decision_service as get_service

    # Check if we have a custom service in app state
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        service = ctx.get("decision_service")
        if service:
            return service

    # Fall back to global service
    return get_service()


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/decisions", response_model=StartDebateResponse, status_code=202)
async def start_decision(
    body: StartDebateRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("debates:create")),
    service=Depends(get_decision_service),
) -> StartDebateResponse:
    """
    Start a new debate decision.

    Returns immediately with a debate ID. The debate runs in the background.
    Use GET /decisions/{id} to poll for status or subscribe to events via SSE.
    """
    from aragora.debate.decision_service import DebateRequest

    try:
        # Build the debate request
        consensus_type = cast(
            Literal[
                "majority",
                "unanimous",
                "judge",
                "none",
                "weighted",
                "supermajority",
                "any",
                "byzantine",
            ],
            body.consensus,
        )
        debate_request = DebateRequest(
            task=body.task,
            agents=body.agents,
            rounds=body.rounds,
            consensus=consensus_type,
            timeout=body.timeout,
            priority=body.priority,
            metadata=body.metadata,
            enable_streaming=body.enable_streaming,
            enable_checkpointing=body.enable_checkpointing,
            enable_memory=body.enable_memory,
        )

        # Start the debate
        debate_id = await service.start_debate(debate_request)

        return StartDebateResponse(
            id=debate_id,
            status="pending",
            message="Debate started successfully",
            events_url=f"/api/v2/decisions/{debate_id}/events",
        )

    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.exception("Failed to start debate: %s", e)
        raise HTTPException(status_code=500, detail="Failed to start debate")


@router.get("/decisions/{debate_id}", response_model=DebateResponse)
async def get_decision(
    debate_id: str,
    service=Depends(get_decision_service),
) -> DebateResponse:
    """
    Get current status of a debate decision.

    Returns the current state including progress, messages, and result if complete.
    """
    try:
        state = await service.get_debate(debate_id)

        if not state:
            raise HTTPException(status_code=404, detail=f"Debate {debate_id} not found")

        return DebateResponse(
            id=state.id,
            task=state.task,
            status=state.status.value,
            progress=state.progress,
            current_round=state.current_round,
            total_rounds=state.total_rounds,
            agents=state.agents,
            result=state.result.to_dict() if state.result else None,
            error=state.error,
            created_at=state.created_at.isoformat() if state.created_at else None,
            updated_at=state.updated_at.isoformat() if state.updated_at else None,
            completed_at=state.completed_at.isoformat() if state.completed_at else None,
            metadata=state.metadata,
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Failed to get debate %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to get debate")


@router.delete("/decisions/{debate_id}", response_model=CancelResponse)
async def cancel_decision(
    debate_id: str,
    auth: AuthorizationContext = Depends(require_permission("debates:delete")),
    service=Depends(get_decision_service),
) -> CancelResponse:
    """
    Cancel a running debate.

    Only works on pending or running debates.
    """
    try:
        cancelled = await service.cancel_debate(debate_id)

        if cancelled:
            return CancelResponse(
                id=debate_id,
                cancelled=True,
                message="Debate cancelled successfully",
            )
        else:
            return CancelResponse(
                id=debate_id,
                cancelled=False,
                message="Debate not found or already completed",
            )

    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.exception("Failed to cancel debate %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to cancel debate")


@router.get("/decisions/{debate_id}/events")
async def stream_events(
    debate_id: str,
    service=Depends(get_decision_service),
) -> StreamingResponse:
    """
    Subscribe to real-time debate events via Server-Sent Events (SSE).

    Events include:
    - debate_started
    - round_started / round_completed
    - agent_message
    - consensus_reached
    - debate_completed / debate_failed
    - progress_update (heartbeat every 60s)
    """
    import json

    # Verify debate exists
    state = await service.get_debate(debate_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Debate {debate_id} not found")

    async def event_generator():
        """Generate SSE events from debate subscription."""
        try:
            async for event in service.subscribe_events(debate_id):
                data = json.dumps(event.to_dict())
                yield f"event: {event.type.value}\ndata: {data}\n\n"

        except asyncio.CancelledError:
            # Client disconnected
            pass
        except (RuntimeError, ValueError, TypeError, OSError, StopAsyncIteration) as e:
            logger.exception("Error streaming events for debate %s: %s", debate_id, e)
            error_data = json.dumps({"error": "Event streaming error"})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/decisions", response_model=list[DebateResponse])
async def list_decisions(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    service=Depends(get_decision_service),
) -> list[DebateResponse]:
    """
    List debate decisions.

    Returns a list of debates, optionally filtered by status.
    """
    from aragora.debate.decision_service import DebateStatus

    try:
        # Convert status string to enum if provided
        status_filter = None
        if status:
            try:
                status_filter = DebateStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid values: {[s.value for s in DebateStatus]}",
                )

        states = await service.list_debates(status=status_filter, limit=limit)

        return [
            DebateResponse(
                id=s.id,
                task=s.task,
                status=s.status.value,
                progress=s.progress,
                current_round=s.current_round,
                total_rounds=s.total_rounds,
                agents=s.agents,
                result=s.result.to_dict() if s.result else None,
                error=s.error,
                created_at=s.created_at.isoformat() if s.created_at else None,
                updated_at=s.updated_at.isoformat() if s.updated_at else None,
                completed_at=s.completed_at.isoformat() if s.completed_at else None,
                metadata=s.metadata,
            )
            for s in states
        ]

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Failed to list debates: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list debates")
