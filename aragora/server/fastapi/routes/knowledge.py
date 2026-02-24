"""
Knowledge Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/knowledge/ (aiohttp handler)

Provides async knowledge mound management endpoints:
- GET  /api/v2/knowledge/search          - Search knowledge mound
- GET  /api/v2/knowledge/items/{item_id} - Get knowledge item by ID
- POST /api/v2/knowledge/items           - Ingest a new knowledge item
- GET  /api/v2/knowledge/stats           - Knowledge mound statistics

Migration Notes:
    This module replaces the legacy knowledge handler endpoints with native
    FastAPI routes. Key improvements:
    - Pydantic request/response models with automatic validation
    - FastAPI dependency injection for auth and storage
    - Proper HTTP status codes (422 for validation, 404 for not found)
    - OpenAPI schema auto-generation
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_permission
from ..middleware.error_handling import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Knowledge"])


# =============================================================================
# Pydantic Models
# =============================================================================


class KnowledgeItemSummary(BaseModel):
    """Summary of a knowledge item for list/search views."""

    id: str
    title: str = ""
    content_type: str = "text"
    source: str = ""
    confidence: float = 0.0
    created_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0

    model_config = {"extra": "allow"}


class KnowledgeSearchResponse(BaseModel):
    """Response for knowledge search."""

    items: list[KnowledgeItemSummary]
    total: int
    query: str


class KnowledgeItemDetail(BaseModel):
    """Full knowledge item details."""

    id: str
    title: str = ""
    content: str = ""
    content_type: str = "text"
    source: str = ""
    confidence: float = 0.0
    created_at: str | None = None
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    debate_id: str | None = None
    adapter: str | None = None

    model_config = {"extra": "allow"}


class IngestKnowledgeRequest(BaseModel):
    """Request body for POST /knowledge/items."""

    title: str = Field(..., min_length=1, max_length=500, description="Item title")
    content: str = Field(..., min_length=1, description="Item content")
    content_type: str = Field("text", description="Content type (text, url, document)")
    source: str = Field("api", description="Source identifier")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IngestKnowledgeResponse(BaseModel):
    """Response for POST /knowledge/items."""

    success: bool
    item_id: str
    item: KnowledgeItemDetail


class KnowledgeStatsResponse(BaseModel):
    """Response for knowledge mound statistics."""

    total_items: int = 0
    adapters: list[str] = Field(default_factory=list)
    items_by_type: dict[str, int] = Field(default_factory=dict)
    items_by_source: dict[str, int] = Field(default_factory=dict)
    last_ingested_at: str | None = None
    storage_backend: str = "unknown"


# =============================================================================
# Dependencies
# =============================================================================


async def get_knowledge_mound(request: Request):
    """Dependency to get the Knowledge Mound from app state."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        km = ctx.get("knowledge_mound")
        if km:
            return km

    # Fall back to global knowledge mound
    try:
        from aragora.knowledge.mound import get_knowledge_mound as _get_km

        return _get_km()
    except (ImportError, RuntimeError, OSError, ValueError) as e:
        logger.warning("Knowledge Mound not available: %s", e)
        return None


# =============================================================================
# Helpers
# =============================================================================


def _item_to_summary(item: Any, relevance: float = 0.0) -> KnowledgeItemSummary:
    """Convert a knowledge item to a summary."""
    if isinstance(item, dict):
        return KnowledgeItemSummary(
            id=item.get("id", item.get("item_id", "")),
            title=item.get("title", ""),
            content_type=item.get("content_type", "text"),
            source=item.get("source", ""),
            confidence=item.get("confidence", 0.0),
            created_at=item.get("created_at"),
            tags=item.get("tags", []),
            relevance_score=item.get("relevance_score", relevance),
        )
    return KnowledgeItemSummary(
        id=getattr(item, "id", getattr(item, "item_id", "")),
        title=getattr(item, "title", ""),
        content_type=getattr(item, "content_type", "text"),
        source=getattr(item, "source", ""),
        confidence=getattr(item, "confidence", 0.0),
        created_at=str(getattr(item, "created_at", "")) if hasattr(item, "created_at") else None,
        tags=getattr(item, "tags", []),
        relevance_score=getattr(item, "relevance_score", relevance),
    )


def _item_to_detail(item: Any) -> KnowledgeItemDetail:
    """Convert a knowledge item to full detail."""
    if isinstance(item, dict):
        return KnowledgeItemDetail(
            id=item.get("id", item.get("item_id", "")),
            title=item.get("title", ""),
            content=item.get("content", ""),
            content_type=item.get("content_type", "text"),
            source=item.get("source", ""),
            confidence=item.get("confidence", 0.0),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
            tags=item.get("tags", []),
            metadata=item.get("metadata", {}),
            debate_id=item.get("debate_id"),
            adapter=item.get("adapter"),
        )
    return KnowledgeItemDetail(
        id=getattr(item, "id", getattr(item, "item_id", "")),
        title=getattr(item, "title", ""),
        content=getattr(item, "content", ""),
        content_type=getattr(item, "content_type", "text"),
        source=getattr(item, "source", ""),
        confidence=getattr(item, "confidence", 0.0),
        created_at=str(getattr(item, "created_at", "")) if hasattr(item, "created_at") else None,
        updated_at=str(getattr(item, "updated_at", "")) if hasattr(item, "updated_at") else None,
        tags=getattr(item, "tags", []),
        metadata=getattr(item, "metadata", {}),
        debate_id=getattr(item, "debate_id", None),
        adapter=getattr(item, "adapter", None),
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/knowledge/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    request: Request,
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results to return"),
    content_type: str | None = Query(None, description="Filter by content type"),
    source: str | None = Query(None, description="Filter by source"),
    km=Depends(get_knowledge_mound),
) -> KnowledgeSearchResponse:
    """
    Search the knowledge mound.

    Returns knowledge items matching the query with relevance scoring.
    Supports filtering by content type and source.
    """
    if not km:
        raise HTTPException(status_code=503, detail="Knowledge Mound not available")

    try:
        items: list[KnowledgeItemSummary] = []

        # Build search kwargs
        search_kwargs: dict[str, Any] = {"limit": limit}
        if content_type:
            search_kwargs["content_type"] = content_type
        if source:
            search_kwargs["source"] = source

        # Try semantic search first, fall back to simple search
        if hasattr(km, "search"):
            results = km.search(query, **search_kwargs)
        elif hasattr(km, "query"):
            results = km.query(query, **search_kwargs)
        else:
            results = []

        # Convert results - handle (item, score) tuples or plain items
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                item, score = result
                items.append(_item_to_summary(item, relevance=float(score)))
            else:
                items.append(_item_to_summary(result))

        return KnowledgeSearchResponse(
            items=items[:limit],
            total=len(items),
            query=query,
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error searching knowledge: %s", e)
        raise HTTPException(status_code=500, detail="Failed to search knowledge")


@router.get("/knowledge/stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats(
    request: Request,
    km=Depends(get_knowledge_mound),
) -> KnowledgeStatsResponse:
    """
    Get knowledge mound statistics.

    Returns aggregate statistics about the knowledge mound including
    item counts, adapter info, and storage details.
    """
    if not km:
        return KnowledgeStatsResponse(storage_backend="not_initialized")

    try:
        stats: dict[str, Any] = {}

        if hasattr(km, "get_stats"):
            raw_stats = km.get_stats()
            if isinstance(raw_stats, dict):
                stats = raw_stats
        elif hasattr(km, "stats"):
            raw_stats = km.stats()
            if isinstance(raw_stats, dict):
                stats = raw_stats

        # Get adapter list
        adapters: list[str] = []
        if hasattr(km, "list_adapters"):
            try:
                adapters = [str(a) for a in km.list_adapters()]
            except (RuntimeError, TypeError, AttributeError):
                pass

        return KnowledgeStatsResponse(
            total_items=stats.get("total_items", stats.get("count", 0)),
            adapters=adapters,
            items_by_type=stats.get("items_by_type", {}),
            items_by_source=stats.get("items_by_source", {}),
            last_ingested_at=stats.get("last_ingested_at"),
            storage_backend=stats.get("storage_backend", "sqlite"),
        )

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting knowledge stats: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get knowledge stats")


@router.get("/knowledge/items/{item_id}", response_model=KnowledgeItemDetail)
async def get_knowledge_item(
    item_id: str,
    km=Depends(get_knowledge_mound),
) -> KnowledgeItemDetail:
    """
    Get knowledge item by ID.

    Returns full details of a specific knowledge item including content and metadata.
    """
    if not km:
        raise HTTPException(status_code=503, detail="Knowledge Mound not available")

    try:
        item = None

        if hasattr(km, "get"):
            item = km.get(item_id)
        elif hasattr(km, "get_item"):
            item = km.get_item(item_id)
        elif hasattr(km, "get_by_id"):
            item = km.get_by_id(item_id)

        if not item:
            raise NotFoundError(f"Knowledge item {item_id} not found")

        return _item_to_detail(item)

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting knowledge item %s: %s", item_id, e)
        raise HTTPException(status_code=500, detail="Failed to get knowledge item")


@router.post("/knowledge/items", response_model=IngestKnowledgeResponse, status_code=201)
async def ingest_knowledge_item(
    body: IngestKnowledgeRequest,
    auth: AuthorizationContext = Depends(require_permission("knowledge:write")),
    km=Depends(get_knowledge_mound),
) -> IngestKnowledgeResponse:
    """
    Ingest a new knowledge item.

    Adds a knowledge item to the mound for future debate context.
    Requires `knowledge:write` permission.
    """
    if not km:
        raise HTTPException(status_code=503, detail="Knowledge Mound not available")

    try:
        import uuid

        item_id = f"ki_{uuid.uuid4().hex[:12]}"

        item_data: dict[str, Any] = {
            "id": item_id,
            "title": body.title,
            "content": body.content,
            "content_type": body.content_type,
            "source": body.source,
            "tags": body.tags,
            "metadata": body.metadata,
        }

        # Try to ingest via the knowledge mound API
        if hasattr(km, "ingest"):
            km.ingest(item_data)
        elif hasattr(km, "add"):
            km.add(item_data)
        elif hasattr(km, "store"):
            km.store(item_data)
        else:
            logger.warning("Knowledge Mound has no ingest/add/store method")

        logger.info("Ingested knowledge item: %s (source=%s)", item_id, body.source)

        return IngestKnowledgeResponse(
            success=True,
            item_id=item_id,
            item=KnowledgeItemDetail(
                id=item_id,
                title=body.title,
                content=body.content,
                content_type=body.content_type,
                source=body.source,
                tags=body.tags,
                metadata=body.metadata,
            ),
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error ingesting knowledge item: %s", e)
        raise HTTPException(status_code=500, detail="Failed to ingest knowledge item")
