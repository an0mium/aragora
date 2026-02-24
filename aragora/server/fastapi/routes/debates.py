"""
Debate Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/debates/ (aiohttp handler)

Provides async debate management endpoints:
- GET    /api/v2/debates                              - List debates with pagination
- POST   /api/v2/debates                              - Create a new debate
- GET    /api/v2/debates/{debate_id}                  - Get debate by ID
- GET    /api/v2/debates/{debate_id}/messages         - Get debate messages
- GET    /api/v2/debates/{debate_id}/convergence      - Get convergence status
- GET    /api/v2/debates/{debate_id}/export/{format}  - Export debate in format
- GET    /api/v2/debates/{debate_id}/argument-graph   - Get argument graph
- GET    /api/v2/debates/{debate_id}/stats            - Get graph statistics
- PATCH  /api/v2/debates/{debate_id}                  - Update debate metadata
- DELETE /api/v2/debates/{debate_id}                  - Delete a debate

Migration Notes:
    This module replaces the CrudOperationsMixin in the legacy debates handler
    with native FastAPI routes. Key improvements:
    - Pydantic request/response models with automatic validation
    - FastAPI dependency injection for auth and storage
    - Proper HTTP status codes (422 for validation, 404 for not found)
    - OpenAPI schema auto-generation
"""

from __future__ import annotations

import json as json_mod
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_permission
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
    created_at: str | None = None
    updated_at: str | None = None
    round_count: int = 0
    agent_count: int = 0
    has_consensus: bool = False

    model_config = {"extra": "allow"}


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
    final_answer: str | None = None
    consensus: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


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
    rounds_to_convergence: int | None = None
    similarity_scores: list[float] = Field(default_factory=list)


class UpdateDebateRequest(BaseModel):
    """Request body for PATCH /debates/{debate_id}.

    All fields are optional. Only provided fields are updated.
    """

    title: str | None = Field(None, max_length=500, description="Update debate title")
    tags: list[str] | None = Field(None, max_length=50, description="Update tags")
    status: str | None = Field(None, description="Update status (active, paused, concluded, archived)")
    metadata: dict[str, Any] | None = Field(None, description="Update custom metadata")


class UpdateDebateResponse(BaseModel):
    """Response for PATCH /debates/{debate_id}."""

    success: bool
    debate_id: str
    updated_fields: list[str]
    debate: DebateSummary


class DeleteDebateResponse(BaseModel):
    """Response for DELETE /debates/{debate_id}."""

    deleted: bool
    id: str


class CreateDebateRequest(BaseModel):
    """Request body for POST /debates.

    Mirrors the legacy /api/v1/debates POST body.
    """

    question: str = Field(..., min_length=1, max_length=5000, description="Topic/question to debate")
    agents: str | None = Field(None, description="Comma-separated agent list")
    rounds: int = Field(3, ge=1, le=20, description="Number of debate rounds")
    consensus: str = Field("majority", description="Consensus method")
    auto_select: bool = Field(False, description="Auto-select agents")
    context: str | None = Field(None, max_length=10000, description="Additional context")
    metadata: dict[str, Any] | None = Field(None, description="Custom metadata")


class CreateDebateResponse(BaseModel):
    """Response for POST /debates."""

    debate_id: str
    status: str
    message: str | None = None

    model_config = {"extra": "allow"}


class ExportResponse(BaseModel):
    """Response for debate export (JSON format)."""

    debate_id: str
    format: str
    content: Any


class ArgumentGraphResponse(BaseModel):
    """Response for argument graph."""

    debate_id: str
    format: str
    graph: Any


class GraphStatsResponse(BaseModel):
    """Response for graph statistics."""

    node_count: int = 0
    edge_count: int = 0
    depth: int = 0
    clusters: int = 0
    avg_branching_factor: float = 0.0
    avg_path_length: float = 0.0

    model_config = {"extra": "allow"}


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


def get_nomic_dir() -> Path | None:
    """Get the nomic directory from environment."""
    nomic_dir_str = os.environ.get("ARAGORA_NOMIC_DIR", ".")
    nomic_dir = Path(nomic_dir_str)
    if nomic_dir.exists():
        return nomic_dir
    return None


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/debates", response_model=DebateListResponse)
async def list_debates(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    status: str | None = Query(None, description="Filter by status"),
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
                    created_at=str(getattr(d, "created_at", ""))
                    if hasattr(d, "created_at")
                    else None,
                    updated_at=str(getattr(d, "updated_at", ""))
                    if hasattr(d, "updated_at")
                    else None,
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

    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing debates: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list debates")


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
                task=getattr(
                    debate, "task", getattr(getattr(debate, "environment", None), "task", "")
                ),
                status=getattr(debate, "status", "unknown"),
                protocol=getattr(debate, "protocol", {}).__dict__
                if hasattr(getattr(debate, "protocol", None), "__dict__")
                else {},
                agents=[str(a) for a in getattr(debate, "agents", [])],
                rounds=[
                    r if isinstance(r, dict) else r.__dict__ for r in getattr(debate, "rounds", [])
                ],
                final_answer=getattr(debate, "final_answer", None),
                consensus=getattr(debate, "consensus", None),
                created_at=str(getattr(debate, "created_at", ""))
                if hasattr(debate, "created_at")
                else None,
                updated_at=str(getattr(debate, "updated_at", ""))
                if hasattr(debate, "updated_at")
                else None,
                metadata=getattr(debate, "metadata", {}),
            )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting debate %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to get debate")


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

        rounds = (
            debate.get("rounds", []) if isinstance(debate, dict) else getattr(debate, "rounds", [])
        )

        for round_data in rounds:
            if isinstance(round_data, dict):
                round_messages = round_data.get("messages", [])
            else:
                round_messages = getattr(round_data, "messages", [])

            for msg in round_messages:
                if isinstance(msg, dict):
                    messages.append(msg)
                else:
                    messages.append(
                        msg.__dict__ if hasattr(msg, "__dict__") else {"content": str(msg)}
                    )

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
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting messages for debate %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to get messages")


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
            rounds_to_convergence=len(debate.get("rounds", []))
            if converged and isinstance(debate, dict)
            else None,
            similarity_scores=similarity_scores,
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting convergence for debate %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to get convergence")


# Valid status values (internal + SDK)
_VALID_STATUSES = {
    "active", "paused", "concluded", "archived",  # internal
    "pending", "running", "completed", "failed", "cancelled",  # SDK
}


@router.patch("/debates/{debate_id}", response_model=UpdateDebateResponse)
async def update_debate(
    debate_id: str,
    body: UpdateDebateRequest,
    auth: AuthorizationContext = Depends(require_permission("debates:write")),
    storage=Depends(get_storage),
) -> UpdateDebateResponse:
    """
    Update debate metadata.

    Allows updating title, tags, status, and custom metadata.
    Requires `debates:write` permission.
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

        # Build updates from non-None fields
        updates: dict[str, Any] = {}
        if body.title is not None:
            updates["title"] = body.title
        if body.tags is not None:
            updates["tags"] = body.tags
        if body.status is not None:
            if body.status not in _VALID_STATUSES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of: {', '.join(sorted(_VALID_STATUSES))}",
                )
            updates["status"] = body.status
        if body.metadata is not None:
            updates["metadata"] = body.metadata

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Apply updates
        if isinstance(debate, dict):
            for key, value in updates.items():
                debate[key] = value
        else:
            for key, value in updates.items():
                setattr(debate, key, value)

        # Save updated debate
        if hasattr(storage, "save_debate"):
            storage.save_debate(debate_id, debate)

        logger.info("Debate %s updated: %s", debate_id, list(updates.keys()))

        # Build response
        if isinstance(debate, dict):
            summary = DebateSummary(
                id=debate.get("id", debate_id),
                task=debate.get("task", debate.get("title", "")),
                status=debate.get("status", "unknown"),
                created_at=debate.get("created_at"),
                updated_at=debate.get("updated_at"),
                round_count=len(debate.get("rounds", [])),
                agent_count=len(debate.get("agents", [])),
                has_consensus=debate.get("consensus") is not None,
            )
        else:
            summary = DebateSummary(
                id=getattr(debate, "id", debate_id),
                task=getattr(debate, "task", ""),
                status=getattr(debate, "status", "unknown"),
                round_count=len(getattr(debate, "rounds", [])),
                agent_count=len(getattr(debate, "agents", [])),
                has_consensus=getattr(debate, "consensus", None) is not None,
            )

        return UpdateDebateResponse(
            success=True,
            debate_id=debate_id,
            updated_fields=list(updates.keys()),
            debate=summary,
        )

    except NotFoundError:
        raise
    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error updating debate %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to update debate")


@router.delete("/debates/{debate_id}", response_model=DeleteDebateResponse)
async def delete_debate(
    debate_id: str,
    auth: AuthorizationContext = Depends(require_permission("debates:delete")),
    storage=Depends(get_storage),
) -> DeleteDebateResponse:
    """
    Delete a debate.

    Permanently deletes a debate and cascades to associated data.
    For soft-delete, use PATCH with status='archived' instead.
    Requires `debates:delete` permission.
    """
    try:
        # Check debate exists
        if hasattr(storage, "get_debate"):
            debate = storage.get_debate(debate_id)
        elif hasattr(storage, "debates"):
            debate = storage.debates.get(debate_id)
        else:
            debate = None

        if not debate:
            raise NotFoundError(f"Debate {debate_id} not found")

        # Perform deletion
        deleted = False
        if hasattr(storage, "delete_debate"):
            deleted = storage.delete_debate(debate_id, cascade_critiques=True)
        elif hasattr(storage, "debates") and debate_id in storage.debates:
            del storage.debates[debate_id]
            deleted = True

        if not deleted:
            raise NotFoundError(f"Debate {debate_id} not found")

        logger.info("Debate %s permanently deleted", debate_id)
        return DeleteDebateResponse(deleted=True, id=debate_id)

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error deleting debate %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to delete debate")


# =============================================================================
# New Endpoints (Issue #258)
# =============================================================================


@router.post("/debates", response_model=CreateDebateResponse, status_code=200)
async def create_debate(
    body: CreateDebateRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("debates:create")),
    storage=Depends(get_storage),
) -> CreateDebateResponse:
    """
    Create a new debate.

    Validates the request and delegates to the debate controller for
    actual debate orchestration. Returns the debate_id for polling/streaming.
    Requires `debates:create` permission.
    """
    try:
        # Build debate body in legacy format for controller compatibility
        debate_body: dict[str, Any] = {
            "question": body.question,
            "rounds": body.rounds,
            "consensus": body.consensus,
            "auto_select": body.auto_select,
        }
        if body.agents:
            debate_body["agents"] = body.agents
        if body.context:
            debate_body["context"] = body.context
        if body.metadata:
            debate_body["metadata"] = body.metadata

        # Delegate to the debate controller
        try:
            from aragora.server.debate_controller import DebateRequest

            debate_request = DebateRequest.from_dict(debate_body)
        except (ImportError, ValueError) as e:
            logger.warning("Invalid debate request: %s", e)
            raise HTTPException(status_code=400, detail="Invalid debate request")

        try:
            import aragora.server.debate_controller as debate_controller_mod

            controller_getter = getattr(debate_controller_mod, "get_debate_controller", None)
            if not callable(controller_getter):
                raise HTTPException(status_code=503, detail="Debate orchestrator not available")
            controller = controller_getter()
            response = controller.start_debate(debate_request)
        except ImportError:
            logger.warning("Debate controller not available")
            raise HTTPException(status_code=503, detail="Debate orchestrator not available")
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError, OSError) as e:
            logger.exception("Failed to start debate: %s", e)
            raise HTTPException(status_code=500, detail="Failed to start debate")

        logger.info("Debate created: %s", response.debate_id)

        # Return the response from the controller
        response_data = response.to_dict() if hasattr(response, "to_dict") else {}
        return CreateDebateResponse(
            debate_id=response_data.get("debate_id", response.debate_id),
            status=response_data.get("status", "started"),
            message=response_data.get("message"),
            **{k: v for k, v in response_data.items() if k not in ("debate_id", "status", "message")},
        )

    except HTTPException:
        raise
    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error creating debate: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create debate")


_VALID_EXPORT_FORMATS = {"json", "csv", "html", "txt", "md"}


@router.get("/debates/{debate_id}/export/{export_format}")
async def export_debate(
    debate_id: str,
    export_format: str,
    table: str = Query("summary", description="Table type for CSV export"),
    auth: AuthorizationContext = Depends(require_permission("export:read")),
    storage=Depends(get_storage),
) -> Response:
    """
    Export a debate in the specified format.

    Supports JSON, CSV, HTML, TXT, and Markdown formats.
    Requires `export:read` permission.
    """
    if export_format not in _VALID_EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format: {export_format}. Valid: {', '.join(sorted(_VALID_EXPORT_FORMATS))}",
        )

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

        # JSON export returns directly
        if export_format == "json":
            debate_data = debate if isinstance(debate, dict) else (
                debate.__dict__ if hasattr(debate, "__dict__") else {}
            )
            return Response(
                content=json_mod.dumps(debate_data, indent=2, default=str),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="debate_{debate_id}.json"'},
            )

        # Delegate to the export formatters for other formats
        try:
            from aragora.server.debate_export import (
                format_debate_csv,
                format_debate_html,
                format_debate_md,
                format_debate_txt,
            )
        except ImportError:
            raise HTTPException(status_code=503, detail="Export module not available")

        debate_dict = debate if isinstance(debate, dict) else (
            debate.__dict__ if hasattr(debate, "__dict__") else {}
        )

        if export_format == "csv":
            result = format_debate_csv(debate_dict, table)
        elif export_format == "html":
            result = format_debate_html(debate_dict)
        elif export_format == "txt":
            result = format_debate_txt(debate_dict)
        elif export_format == "md":
            result = format_debate_md(debate_dict)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {export_format}")

        raw_content = result.content
        content = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content

        return Response(
            content=content,
            media_type=getattr(result, "content_type", "text/plain"),
            headers={
                "Content-Disposition": f'attachment; filename="{getattr(result, "filename", f"debate_{debate_id}.{export_format}")}"'
            },
        )

    except NotFoundError:
        raise
    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error exporting debate %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to export debate")


@router.get("/debates/{debate_id}/argument-graph", response_model=ArgumentGraphResponse)
async def get_argument_graph(
    debate_id: str,
    output_format: str = Query("json", alias="format", description="Output format: json or mermaid"),
    auth: AuthorizationContext = Depends(require_permission("analysis:read")),
) -> ArgumentGraphResponse:
    """
    Get the argument graph for a debate.

    Reconstructs the graph from stored debate traces via ArgumentCartographer.
    Supports JSON (default) and Mermaid diagram output.
    Requires `analysis:read` permission.
    """
    try:
        from aragora.debate.traces import DebateTrace
        from aragora.visualization.mapper import ArgumentCartographer
    except ImportError:
        raise HTTPException(status_code=503, detail="Graph analysis module not available")

    nomic_dir = get_nomic_dir()
    if not nomic_dir:
        raise HTTPException(status_code=503, detail="Nomic directory not configured")

    try:
        trace_path = nomic_dir / "traces" / f"{debate_id}.json"
        if not trace_path.exists():
            raise NotFoundError(f"Debate {debate_id} not found")

        trace = DebateTrace.load(trace_path)
        result = trace.to_debate_result()

        cartographer = ArgumentCartographer()
        cartographer.set_debate_context(debate_id, result.task or "")

        for msg in result.messages:
            cartographer.update_from_message(
                agent=msg.agent,
                content=msg.content,
                role=msg.role,
                round_num=msg.round,
            )

        for critique in result.critiques:
            cartographer.update_from_critique(
                critic_agent=critique.agent,
                target_agent=critique.target or "",
                severity=critique.severity,
                round_num=getattr(critique, "round", 1),
                critique_text=critique.reasoning,
            )

        if output_format == "mermaid":
            mermaid_code = cartographer.export_mermaid()
            return ArgumentGraphResponse(
                debate_id=debate_id,
                format="mermaid",
                graph=mermaid_code,
            )

        # Default: JSON graph
        graph_json = json_mod.loads(cartographer.export_json())
        return ArgumentGraphResponse(
            debate_id=debate_id,
            format="json",
            graph=graph_json,
        )

    except NotFoundError:
        raise
    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting argument graph for debate %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to get argument graph")


@router.get("/debates/{debate_id}/stats", response_model=GraphStatsResponse)
async def get_debate_stats(
    debate_id: str,
    auth: AuthorizationContext = Depends(require_permission("analysis:read")),
) -> GraphStatsResponse:
    """
    Get argument graph statistics for a debate.

    Returns node counts, edge counts, depth, branching factor, and complexity.
    Requires `analysis:read` permission.
    """
    try:
        from aragora.debate.traces import DebateTrace
        from aragora.visualization.mapper import ArgumentCartographer
    except ImportError:
        raise HTTPException(status_code=503, detail="Graph analysis module not available")

    nomic_dir = get_nomic_dir()
    if not nomic_dir:
        raise HTTPException(status_code=503, detail="Nomic directory not configured")

    try:
        trace_path = nomic_dir / "traces" / f"{debate_id}.json"

        if not trace_path.exists():
            # Try replays directory as fallback
            replay_path = nomic_dir / "replays" / debate_id / "events.jsonl"
            if replay_path.exists():
                return await _build_stats_from_replay(debate_id, replay_path)
            raise NotFoundError(f"Debate {debate_id} not found")

        # Load from trace file
        trace = DebateTrace.load(trace_path)
        result = trace.to_debate_result()

        # Build cartographer from debate result
        cartographer = ArgumentCartographer()
        cartographer.set_debate_context(debate_id, result.task or "")

        for msg in result.messages:
            cartographer.update_from_message(
                agent=msg.agent,
                content=msg.content,
                role=msg.role,
                round_num=msg.round,
            )

        for critique in result.critiques:
            cartographer.update_from_critique(
                critic_agent=critique.agent,
                target_agent=critique.target or "",
                severity=critique.severity,
                round_num=getattr(critique, "round", 1),
                critique_text=critique.reasoning,
            )

        stats = cartographer.get_statistics()
        return GraphStatsResponse(**stats)

    except NotFoundError:
        raise
    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting stats for debate %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to get debate stats")


async def _build_stats_from_replay(debate_id: str, replay_path: Path) -> GraphStatsResponse:
    """Build graph stats from replay events file."""
    try:
        from aragora.visualization.mapper import ArgumentCartographer
    except ImportError:
        raise HTTPException(status_code=503, detail="Graph analysis module not available")

    try:
        cartographer = ArgumentCartographer()
        cartographer.set_debate_context(debate_id, "")

        with replay_path.open() as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        event = json_mod.loads(line)
                    except json_mod.JSONDecodeError:
                        logger.warning("Skipping malformed JSONL line %s", line_num)
                        continue

                    if event.get("type") == "agent_message":
                        cartographer.update_from_message(
                            agent=event.get("agent", "unknown"),
                            content=event.get("data", {}).get("content", ""),
                            role=event.get("data", {}).get("role", "proposer"),
                            round_num=event.get("round", 1),
                        )
                    elif event.get("type") == "critique":
                        cartographer.update_from_critique(
                            critic_agent=event.get("agent", "unknown"),
                            target_agent=event.get("data", {}).get("target", "unknown"),
                            severity=event.get("data", {}).get("severity", 0.5),
                            round_num=event.get("round", 1),
                            critique_text=event.get("data", {}).get("content", ""),
                        )

        stats = cartographer.get_statistics()
        return GraphStatsResponse(**stats)

    except FileNotFoundError:
        raise NotFoundError(f"Replay file not found: {debate_id}")
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error building stats from replay %s: %s", debate_id, e)
        raise HTTPException(status_code=500, detail="Failed to build stats from replay")
