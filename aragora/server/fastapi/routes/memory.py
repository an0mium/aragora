"""
Memory Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/memory/ (legacy handler)

Provides async memory management endpoints:
- GET  /api/v2/memory/search - Search memories across tiers
- POST /api/v2/memory/store  - Store a new memory entry
- GET  /api/v2/memory/recall - Recall memories by query

Migration Notes:
    Delegates to existing ContinuumMemory for search and recall.
    The store endpoint writes to the configured memory backend.
    RBAC is enforced via FastAPI dependency injection.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/memory", tags=["Memory"])

# =============================================================================
# Pydantic Models
# =============================================================================


class MemoryEntry(BaseModel):
    """A single memory entry."""

    id: str | None = None
    tier: str | None = None
    content: str = ""
    preview: str = ""
    importance: float | None = None
    surprise_score: float | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_estimate: int = 0

    model_config = {"extra": "allow"}


class MemorySearchResponse(BaseModel):
    """Response for GET /memory/search."""

    query: str
    results: list[MemoryEntry]
    total: int
    tiers_searched: list[str]
    sort_by: str = "relevance"


class StoreMemoryRequest(BaseModel):
    """Request body for POST /memory/store."""

    content: str = Field(..., min_length=1, max_length=50000, description="Memory content to store")
    tier: str = Field("medium", description="Memory tier (fast, medium, slow, glacial)")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    tags: list[str] = Field(default_factory=list, description="Optional tags")


class StoreMemoryResponse(BaseModel):
    """Response for POST /memory/store."""

    success: bool
    memory_id: str | None = None
    tier: str
    message: str


class MemoryRecallResponse(BaseModel):
    """Response for GET /memory/recall."""

    query: str
    memories: list[MemoryEntry]
    total: int
    tiers: list[str]


# =============================================================================
# Dependencies
# =============================================================================


async def get_continuum_memory(request: Request):
    """Dependency to get ContinuumMemory from app state."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        memory = ctx.get("continuum_memory")
        if memory:
            return memory

    # Fall back to global instance
    try:
        from aragora.memory.continuum import get_continuum_memory as _get_mem

        return _get_mem()
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.warning("ContinuumMemory not available: %s", e)
        raise HTTPException(status_code=503, detail="Memory system not available")


def _estimate_tokens(text: str) -> int:
    """Estimate token count for text (approx 4 chars per token)."""
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / 4)))


def _format_memory_entry(entry: Any, preview_chars: int = 300) -> MemoryEntry:
    """Convert a memory entry object/dict to a MemoryEntry model."""
    if isinstance(entry, dict):
        content = entry.get("content", "")
        tier_value = entry.get("tier")
        tier_name = str(tier_value).lower() if tier_value else None
        # Handle enum-like tier values
        if hasattr(tier_value, "name"):
            tier_name = tier_value.name.lower()
        importance = entry.get("importance")
        surprise = entry.get("surprise_score", entry.get("surprise"))
        return MemoryEntry(
            id=entry.get("id") or entry.get("memory_id"),
            tier=tier_name,
            content=content,
            preview=content[:preview_chars] + "..." if len(content) > preview_chars else content,
            importance=round(float(importance), 3) if importance is not None else None,
            surprise_score=round(float(surprise), 3) if surprise is not None else None,
            created_at=str(entry.get("created_at")) if entry.get("created_at") else None,
            updated_at=str(entry.get("updated_at")) if entry.get("updated_at") else None,
            metadata=entry.get("metadata", {}),
            token_estimate=_estimate_tokens(content),
        )
    else:
        content = getattr(entry, "content", "") or ""
        tier_value = getattr(entry, "tier", None)
        tier_name = None
        if hasattr(tier_value, "name"):
            tier_name = tier_value.name.lower()
        elif tier_value is not None:
            tier_name = str(tier_value).lower()
        importance = getattr(entry, "importance", None)
        surprise = getattr(entry, "surprise_score", getattr(entry, "surprise", None))
        created_at = getattr(entry, "created_at", None)
        updated_at = getattr(entry, "updated_at", None)
        return MemoryEntry(
            id=getattr(entry, "id", None) or getattr(entry, "memory_id", None),
            tier=tier_name,
            content=content,
            preview=content[:preview_chars] + "..." if len(content) > preview_chars else content,
            importance=round(float(importance), 3) if importance is not None else None,
            surprise_score=round(float(surprise), 3) if surprise is not None else None,
            created_at=str(created_at) if created_at is not None else None,
            updated_at=str(updated_at) if updated_at is not None else None,
            metadata=getattr(entry, "metadata", {}) or {},
            token_estimate=_estimate_tokens(content),
        )


# =============================================================================
# Endpoints
# =============================================================================


_VALID_TIERS = {"fast", "medium", "slow", "glacial"}
_VALID_SORT = {"relevance", "importance", "recency"}


@router.get("/search", response_model=MemorySearchResponse)
async def search_memories(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    tier: str | None = Query(None, description="Filter by tier (comma-separated)"),
    min_importance: float = Query(0.0, ge=0.0, le=1.0, description="Minimum importance"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    sort: str = Query("relevance", description="Sort by: relevance, importance, recency"),
    auth: AuthorizationContext = Depends(require_permission("memory:read")),
    continuum=Depends(get_continuum_memory),
) -> MemorySearchResponse:
    """
    Search memories across all tiers.

    Returns matching memories with content previews and metadata.
    Requires `memory:read` permission.
    """
    if sort not in _VALID_SORT:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort. Must be one of: {', '.join(sorted(_VALID_SORT))}",
        )

    try:
        # Parse tiers
        tiers_list: list[Any] = []
        tiers_searched: list[str] = []

        memory_tier_cls: Any = None
        try:
            from aragora.memory.continuum import MemoryTier as _MemoryTier

            memory_tier_cls = _MemoryTier
        except ImportError:
            pass

        if tier and memory_tier_cls:
            for tier_name in tier.split(","):
                tier_name = tier_name.strip().lower()
                if tier_name in _VALID_TIERS:
                    try:
                        tiers_list.append(memory_tier_cls[tier_name.upper()])
                        tiers_searched.append(tier_name)
                    except KeyError:
                        continue
        elif memory_tier_cls:
            tiers_list = list(memory_tier_cls)
            tiers_searched = [t.name.lower() for t in memory_tier_cls]
        else:
            tiers_searched = list(_VALID_TIERS)

        # Execute search
        memories = continuum.retrieve(
            query=q,
            tiers=tiers_list if tiers_list else None,
            limit=limit,
            min_importance=min_importance,
        )

        # Sort results
        if sort == "importance":
            memories.sort(key=lambda m: getattr(m, "importance", 0), reverse=True)
        elif sort == "recency":
            memories.sort(key=lambda m: getattr(m, "updated_at", 0), reverse=True)

        # Convert to response entries
        results = [_format_memory_entry(m) for m in memories]

        return MemorySearchResponse(
            query=q,
            results=results,
            total=len(results),
            tiers_searched=tiers_searched,
            sort_by=sort,
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error searching memories: %s", e)
        raise HTTPException(status_code=500, detail="Failed to search memories")


@router.post("/store", response_model=StoreMemoryResponse)
async def store_memory(
    body: StoreMemoryRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("memory:write")),
    continuum=Depends(get_continuum_memory),
) -> StoreMemoryResponse:
    """
    Store a new memory entry.

    Writes content to the specified memory tier.
    Requires `memory:write` permission.
    """
    tier_name = body.tier.lower()
    if tier_name not in _VALID_TIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier: {body.tier}. Must be one of: {', '.join(sorted(_VALID_TIERS))}",
        )

    try:
        # Resolve the tier enum
        tier_enum = None
        try:
            from aragora.memory.continuum import MemoryTier

            tier_enum = MemoryTier[tier_name.upper()]
        except (ImportError, KeyError):
            pass

        # Store the memory
        memory_id = None
        if hasattr(continuum, "store"):
            result = continuum.store(
                content=body.content,
                tier=tier_enum or tier_name,
                importance=body.importance,
                metadata={
                    **(body.metadata or {}),
                    "tags": body.tags,
                    "user_id": auth.user_id,
                },
            )
            memory_id = getattr(result, "id", None) or getattr(result, "memory_id", None)
            if isinstance(result, str):
                memory_id = result
        elif hasattr(continuum, "add"):
            result = continuum.add(
                content=body.content,
                tier=tier_enum or tier_name,
                importance=body.importance,
                metadata={
                    **(body.metadata or {}),
                    "tags": body.tags,
                    "user_id": auth.user_id,
                },
            )
            memory_id = getattr(result, "id", None) or str(result) if result else None
        else:
            raise HTTPException(status_code=503, detail="Memory storage method not available")

        logger.info("Memory stored: id=%s, tier=%s", memory_id, tier_name)

        return StoreMemoryResponse(
            success=True,
            memory_id=memory_id,
            tier=tier_name,
            message="Memory stored successfully",
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error storing memory: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store memory")


@router.get("/recall", response_model=MemoryRecallResponse)
async def recall_memories(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500, description="Recall query"),
    tier: str | None = Query(None, description="Filter by tier (comma-separated)"),
    limit: int = Query(10, ge=1, le=100, description="Max results"),
    auth: AuthorizationContext = Depends(require_permission("memory:read")),
    continuum=Depends(get_continuum_memory),
) -> MemoryRecallResponse:
    """
    Recall memories by query.

    Similar to search but optimized for context retrieval. Returns the most
    relevant memories for injection into debate or agent context.
    Requires `memory:read` permission.
    """
    try:
        # Parse tiers
        tiers_list: list[Any] = []
        tiers_searched: list[str] = []

        memory_tier_cls: Any = None
        try:
            from aragora.memory.continuum import MemoryTier as _MemoryTier

            memory_tier_cls = _MemoryTier
        except ImportError:
            pass

        if tier and memory_tier_cls:
            for tier_name in tier.split(","):
                tier_name = tier_name.strip().lower()
                if tier_name in _VALID_TIERS:
                    try:
                        tiers_list.append(memory_tier_cls[tier_name.upper()])
                        tiers_searched.append(tier_name)
                    except KeyError:
                        continue
        elif memory_tier_cls:
            tiers_list = list(memory_tier_cls)
            tiers_searched = [t.name.lower() for t in memory_tier_cls]
        else:
            tiers_searched = list(_VALID_TIERS)

        # Retrieve memories (recall is optimized for context injection)
        memories = continuum.retrieve(
            query=q,
            tiers=tiers_list if tiers_list else None,
            limit=limit,
        )

        # Convert to response entries (shorter preview for recall)
        results = [_format_memory_entry(m, preview_chars=200) for m in memories]

        return MemoryRecallResponse(
            query=q,
            memories=results,
            total=len(results),
            tiers=tiers_searched,
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error recalling memories: %s", e)
        raise HTTPException(status_code=500, detail="Failed to recall memories")
