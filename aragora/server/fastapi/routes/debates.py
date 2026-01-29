"""
Debate Endpoints (FastAPI v2).

Provides async debate management endpoints:
- List debates with pagination
- Get debate by ID
- Get debate messages
- Get debate convergence status
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query, Request, HTTPException
from pydantic import BaseModel, Field

from ..middleware.error_handling import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Debates"])


# =============================================================================
# Pydantic Models
# =============================================================================


class DebateSummary(BaseModel):
    """Summary of a debate for list views."""

    id: str
    task: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    round_count: int = 0
    agent_count: int = 0
    has_consensus: bool = False

    class Config:
        extra = "allow"


class DebateListResponse(BaseModel):
    """Response for debate listing."""

    debates: list[DebateSummary]
    total: int
    limit: int
    offset: int


class DebateDetail(BaseModel):
    """Full debate details."""

    id: str
    task: str
    status: str
    protocol: dict[str, Any] = Field(default_factory=dict)
    agents: list[str] = Field(default_factory=list)
    rounds: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: Optional[str] = None
    consensus: Optional[dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class MessageResponse(BaseModel):
    """Response for debate messages."""

    debate_id: str
    messages: list[dict[str, Any]]
    total: int
    has_more: bool


class ConvergenceResponse(BaseModel):
    """Response for convergence status."""

    debate_id: str
    converged: bool
    confidence: float = 0.0
    rounds_to_convergence: Optional[int] = None
    similarity_scores: list[float] = Field(default_factory=list)


# =============================================================================
# Dependencies
# =============================================================================


async def get_storage(request: Request):
    """Dependency to get storage from app state."""
    ctx = getattr(request.app.state, "context", None)
    if not ctx:
        raise HTTPException(status_code=503, detail="Server not initialized")

    storage = ctx.get("storage")
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")

    return storage


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/debates", response_model=DebateListResponse)
async def list_debates(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    status: Optional[str] = Query(None, description="Filter by status"),
    storage=Depends(get_storage),
) -> DebateListResponse:
    """
    List all debates with pagination.

    Returns a paginated list of debate summaries.
    """
    try:
        # Get debates from storage
        if hasattr(storage, "list_debates"):
            debates_raw = storage.list_debates(
                limit=limit,
                offset=offset,
                status=status,
            )
        else:
            # Fallback for simpler storage implementations
            all_debates = list(storage.debates.values()) if hasattr(storage, "debates") else []
            debates_raw = all_debates[offset : offset + limit]

        # Get total count
        if hasattr(storage, "count_debates"):
            total = storage.count_debates(status=status)
        else:
            total = len(storage.debates) if hasattr(storage, "debates") else 0

        # Convert to summaries
        debates = []
        for d in debates_raw:
            if isinstance(d, dict):
                summary = DebateSummary(
                    id=d.get("id", ""),
                    task=d.get("task", d.get("environment", {}).get("task", "")),
                    status=d.get("status", "unknown"),
                    created_at=d.get("created_at"),
                    updated_at=d.get("updated_at"),
                    round_count=len(d.get("rounds", [])),
                    agent_count=len(d.get("agents", [])),
                    has_consensus=d.get("consensus") is not None,
                )
            else:
                # Handle dataclass/object
                summary = DebateSummary(
                    id=getattr(d, "id", ""),
                    task=getattr(d, "task", getattr(getattr(d, "environment", None), "task", "")),
                    status=getattr(d, "status", "unknown"),
                    created_at=str(getattr(d, "created_at", "")) if hasattr(d, "created_at") else None,
                    updated_at=str(getattr(d, "updated_at", "")) if hasattr(d, "updated_at") else None,
                    round_count=len(getattr(d, "rounds", [])),
                    agent_count=len(getattr(d, "agents", [])),
                    has_consensus=getattr(d, "consensus", None) is not None,
                )
            debates.append(summary)

        return DebateListResponse(
            debates=debates,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.exception(f"Error listing debates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list debates: {e}")


@router.get("/debates/{debate_id}", response_model=DebateDetail)
async def get_debate(
    debate_id: str,
    storage=Depends(get_storage),
) -> DebateDetail:
    """
    Get debate by ID.

    Returns full debate details including rounds and consensus.
    """
    try:
        # Get debate from storage
        if hasattr(storage, "get_debate"):
            debate = storage.get_debate(debate_id)
        elif hasattr(storage, "debates"):
            debate = storage.debates.get(debate_id)
        else:
            debate = None

        if not debate:
            raise NotFoundError(f"Debate {debate_id} not found")

        # Convert to response model
        if isinstance(debate, dict):
            return DebateDetail(
                id=debate.get("id", debate_id),
                task=debate.get("task", debate.get("environment", {}).get("task", "")),
                status=debate.get("status", "unknown"),
                protocol=debate.get("protocol", {}),
                agents=debate.get("agents", []),
                rounds=debate.get("rounds", []),
                final_answer=debate.get("final_answer"),
                consensus=debate.get("consensus"),
                created_at=debate.get("created_at"),
                updated_at=debate.get("updated_at"),
                metadata=debate.get("metadata", {}),
            )
        else:
            # Handle dataclass/object
            return DebateDetail(
                id=getattr(debate, "id", debate_id),
                task=getattr(debate, "task", getattr(getattr(debate, "environment", None), "task", "")),
                status=getattr(debate, "status", "unknown"),
                protocol=getattr(debate, "protocol", {}).__dict__ if hasattr(getattr(debate, "protocol", None), "__dict__") else {},
                agents=[str(a) for a in getattr(debate, "agents", [])],
                rounds=[r if isinstance(r, dict) else r.__dict__ for r in getattr(debate, "rounds", [])],
                final_answer=getattr(debate, "final_answer", None),
                consensus=getattr(debate, "consensus", None),
                created_at=str(getattr(debate, "created_at", "")) if hasattr(debate, "created_at") else None,
                updated_at=str(getattr(debate, "updated_at", "")) if hasattr(debate, "updated_at") else None,
                metadata=getattr(debate, "metadata", {}),
            )

    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"Error getting debate {debate_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get debate: {e}")


@router.get("/debates/{debate_id}/messages", response_model=MessageResponse)
async def get_debate_messages(
    debate_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    storage=Depends(get_storage),
) -> MessageResponse:
    """
    Get messages from a debate.

    Returns paginated list of debate messages/rounds.
    """
    try:
        # Get debate
        if hasattr(storage, "get_debate"):
            debate = storage.get_debate(debate_id)
        elif hasattr(storage, "debates"):
            debate = storage.debates.get(debate_id)
        else:
            debate = None

        if not debate:
            raise NotFoundError(f"Debate {debate_id} not found")

        # Extract messages from rounds
        messages: list[dict[str, Any]] = []

        rounds = debate.get("rounds", []) if isinstance(debate, dict) else getattr(debate, "rounds", [])

        for round_data in rounds:
            if isinstance(round_data, dict):
                round_messages = round_data.get("messages", [])
            else:
                round_messages = getattr(round_data, "messages", [])

            for msg in round_messages:
                if isinstance(msg, dict):
                    messages.append(msg)
                else:
                    messages.append(msg.__dict__ if hasattr(msg, "__dict__") else {"content": str(msg)})

        # Paginate
        total = len(messages)
        messages = messages[offset : offset + limit]

        return MessageResponse(
            debate_id=debate_id,
            messages=messages,
            total=total,
            has_more=(offset + limit) < total,
        )

    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"Error getting messages for debate {debate_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {e}")


@router.get("/debates/{debate_id}/convergence", response_model=ConvergenceResponse)
async def get_debate_convergence(
    debate_id: str,
    storage=Depends(get_storage),
) -> ConvergenceResponse:
    """
    Get convergence status for a debate.

    Returns whether the debate has converged and related metrics.
    """
    try:
        # Get debate
        if hasattr(storage, "get_debate"):
            debate = storage.get_debate(debate_id)
        elif hasattr(storage, "debates"):
            debate = storage.debates.get(debate_id)
        else:
            debate = None

        if not debate:
            raise NotFoundError(f"Debate {debate_id} not found")

        # Extract convergence info
        if isinstance(debate, dict):
            consensus = debate.get("consensus")
            converged = consensus is not None
            confidence = consensus.get("confidence", 0.0) if consensus else 0.0
        else:
            consensus = getattr(debate, "consensus", None)
            converged = consensus is not None
            confidence = getattr(consensus, "confidence", 0.0) if consensus else 0.0

        # Get similarity scores if available
        similarity_scores: list[float] = []
        if isinstance(debate, dict):
            metrics = debate.get("metrics", {})
            similarity_scores = metrics.get("similarity_scores", [])
        else:
            metrics = getattr(debate, "metrics", None)
            if metrics:
                similarity_scores = getattr(metrics, "similarity_scores", [])

        return ConvergenceResponse(
            debate_id=debate_id,
            converged=converged,
            confidence=confidence,
            rounds_to_convergence=len(debate.get("rounds", [])) if converged and isinstance(debate, dict) else None,
            similarity_scores=similarity_scores,
        )

    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"Error getting convergence for debate {debate_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get convergence: {e}")
