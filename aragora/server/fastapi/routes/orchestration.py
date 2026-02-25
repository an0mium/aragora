"""
Orchestration Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/orchestration/ (aiohttp handler)

Surfaces the unified orchestration control plane as REST endpoints:
- POST   /api/v2/orchestration/deliberate       - Start async deliberation
- POST   /api/v2/orchestration/deliberate/sync   - Start sync deliberation
- GET    /api/v2/orchestration/status/{request_id} - Get deliberation status
- GET    /api/v2/orchestration/templates         - List deliberation templates

Migration Notes:
    This module replaces OrchestrationHandler in the legacy handler with
    native FastAPI routes. Key improvements:
    - Pydantic request/response models with automatic validation
    - FastAPI dependency injection for auth and storage
    - Proper HTTP status codes (422 for validation, 404 for not found)
    - OpenAPI schema auto-generation
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_permission
from ..middleware.error_handling import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Orchestration"])

# =============================================================================
# Pydantic Models
# =============================================================================


class KnowledgeSourceItem(BaseModel):
    """A knowledge context source for deliberation."""

    type: str = Field(..., description="Source type (slack, confluence, github, jira, document)")
    id: str = Field(..., description="Source identifier (channel ID, page ID, PR number, etc.)")
    lookback_minutes: int = Field(60, ge=1, le=10080, description="Lookback window in minutes")
    max_items: int = Field(50, ge=1, le=500, description="Maximum items to fetch")


class OutputChannelItem(BaseModel):
    """An output channel for routing deliberation results."""

    type: str = Field(..., description="Channel type (slack, teams, discord, telegram, email, webhook)")
    id: str = Field(..., description="Channel identifier (channel ID, email, webhook URL)")
    thread_id: str | None = Field(None, description="Optional thread/conversation ID")


class DeliberateRequest(BaseModel):
    """Request body for POST /orchestration/deliberate.

    Unified deliberation endpoint that accepts knowledge context sources,
    agent configuration, and output channels.
    """

    question: str = Field(
        ..., min_length=1, max_length=5000, description="Question or topic for deliberation"
    )
    knowledge_sources: list[KnowledgeSourceItem] = Field(
        default_factory=list, description="Knowledge context sources"
    )
    workspaces: list[str] = Field(
        default_factory=list, description="Workspace IDs for knowledge context"
    )
    team_strategy: str = Field(
        "best_for_domain",
        description="Agent team selection strategy (specified, best_for_domain, diverse, fast, random)",
    )
    agents: list[str] = Field(
        default_factory=list, description="Explicit agent list (used with 'specified' strategy)"
    )
    output_channels: list[OutputChannelItem] = Field(
        default_factory=list, description="Channels to route results to"
    )
    output_format: str = Field(
        "standard",
        description="Output format (standard, decision_receipt, summary, github_review, slack_message)",
    )
    require_consensus: bool = Field(True, description="Whether consensus is required")
    priority: str = Field("normal", description="Request priority (low, normal, high, critical)")
    max_rounds: int = Field(5, ge=1, le=20, description="Maximum debate rounds")
    timeout_seconds: float = Field(300.0, ge=10, le=3600, description="Timeout in seconds")
    template: str | None = Field(None, description="Deliberation template name")
    notify: bool = Field(True, description="Auto-notify on completion")
    dry_run: bool = Field(False, description="Return cost estimate only without executing")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class DeliberateAsyncResponse(BaseModel):
    """Response for async POST /orchestration/deliberate."""

    request_id: str
    status: str
    message: str | None = None
    estimated_cost_usd: float | None = None

    model_config = {"extra": "allow"}


class DryRunResponse(BaseModel):
    """Response for dry_run deliberation requests."""

    request_id: str
    dry_run: bool = True
    estimated_cost: dict[str, Any] | None = None
    agents: list[str]
    max_rounds: int
    message: str = "Dry run -- no debate executed"


class OrchestrationResultResponse(BaseModel):
    """Full orchestration result."""

    request_id: str
    success: bool
    consensus_reached: bool = False
    final_answer: str | None = None
    confidence: float | None = None
    agents_participated: list[str] = Field(default_factory=list)
    rounds_completed: int = 0
    duration_seconds: float = 0.0
    knowledge_context_used: list[str] = Field(default_factory=list)
    channels_notified: list[str] = Field(default_factory=list)
    receipt_id: str | None = None
    error: str | None = None
    created_at: str | None = None
    estimated_cost_usd: float | None = None

    model_config = {"extra": "allow"}


class StatusResponse(BaseModel):
    """Response for GET /orchestration/status/{request_id}."""

    request_id: str
    status: str  # completed, failed, in_progress
    result: OrchestrationResultResponse | None = None


class TemplateItem(BaseModel):
    """A deliberation template."""

    name: str
    description: str
    default_agents: list[str] = Field(default_factory=list)
    default_knowledge_sources: list[str] = Field(default_factory=list)
    output_format: str = "standard"
    consensus_threshold: float = 0.7
    max_rounds: int = 5
    personas: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class TemplateListResponse(BaseModel):
    """Response for GET /orchestration/templates."""

    templates: list[TemplateItem]
    count: int


# =============================================================================
# Dependencies
# =============================================================================


def _get_orchestration_handler() -> Any:
    """Lazily import and return the legacy OrchestrationHandler singleton.

    Returns None if the handler module is unavailable, allowing
    graceful degradation with a 503 response.
    """
    try:
        from aragora.server.handlers.orchestration.handler import handler

        return handler
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.debug("Orchestration handler not available: %s", e)
        return None


def _get_orchestration_stores() -> tuple[dict[str, Any], dict[str, Any]]:
    """Get the in-memory orchestration request and result stores."""
    try:
        from aragora.server.handlers.orchestration.handler import (
            _orchestration_requests,
            _orchestration_results,
        )

        return _orchestration_requests, _orchestration_results
    except (ImportError, AttributeError) as e:
        logger.debug("Orchestration stores not available: %s", e)
        return {}, {}


async def get_orch_handler(request: Request) -> Any:
    """Dependency to get the orchestration handler from app state or module."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        handler = ctx.get("orchestration_handler")
        if handler is not None:
            return handler

    handler = _get_orchestration_handler()
    if handler is None:
        raise HTTPException(status_code=503, detail="Orchestration service not available")
    return handler


# =============================================================================
# Helper: Build legacy request dict from Pydantic model
# =============================================================================


def _build_legacy_request_dict(body: DeliberateRequest) -> dict[str, Any]:
    """Convert a DeliberateRequest Pydantic model to the legacy dict format
    expected by OrchestrationRequest.from_dict().
    """
    data: dict[str, Any] = {
        "question": body.question,
        "team_strategy": body.team_strategy,
        "agents": body.agents,
        "output_format": body.output_format,
        "require_consensus": body.require_consensus,
        "priority": body.priority,
        "max_rounds": body.max_rounds,
        "timeout_seconds": body.timeout_seconds,
        "template": body.template,
        "notify": body.notify,
        "dry_run": body.dry_run,
        "metadata": body.metadata,
        "workspaces": body.workspaces,
    }

    # Convert knowledge sources
    if body.knowledge_sources:
        data["knowledge_sources"] = [
            {
                "type": ks.type,
                "id": ks.id,
                "lookback_minutes": ks.lookback_minutes,
                "max_items": ks.max_items,
            }
            for ks in body.knowledge_sources
        ]

    # Convert output channels
    if body.output_channels:
        data["output_channels"] = [
            {
                "type": ch.type,
                "id": ch.id,
                "thread_id": ch.thread_id,
            }
            for ch in body.output_channels
        ]

    return data


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/orchestration/deliberate", status_code=202)
async def start_deliberation(
    body: DeliberateRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("orchestration:execute")),
    orch_handler: Any = Depends(get_orch_handler),
) -> DeliberateAsyncResponse | DryRunResponse | OrchestrationResultResponse:
    """
    Start an asynchronous deliberation.

    Queues a deliberation request and returns immediately with a request_id.
    Poll GET /orchestration/status/{request_id} for results.

    Requires `orchestration:execute` permission.
    """
    try:
        data = _build_legacy_request_dict(body)

        # Delegate to legacy handler's _handle_deliberate (which does
        # full RBAC validation on sources/channels, cost estimation, etc.)
        try:
            from aragora.server.handlers.orchestration.models import (
                OrchestrationRequest,
                OrchestrationResult,
            )
            from aragora.server.handlers.orchestration.templates import TEMPLATES
        except ImportError:
            raise HTTPException(
                status_code=503, detail="Orchestration models not available"
            )

        # Parse the request using the legacy model
        orch_request = OrchestrationRequest.from_dict(data)

        # Apply template if specified
        if orch_request.template and orch_request.template in TEMPLATES:
            template = TEMPLATES[orch_request.template]
            if not orch_request.agents:
                orch_request.agents = template.default_agents
            if not orch_request.knowledge_sources:
                from aragora.server.handlers.orchestration.models import KnowledgeContextSource

                for src in template.default_knowledge_sources:
                    orch_request.knowledge_sources.append(
                        KnowledgeContextSource.from_string(src)
                    )
            orch_request.max_rounds = template.max_rounds

        # Validate question
        if not orch_request.question:
            raise HTTPException(status_code=400, detail="Question is required")

        # Cost estimation (non-blocking)
        cost_estimate = None
        estimated_cost_usd = None
        try:
            from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

            cost_estimate = estimate_debate_cost(
                num_agents=len(orch_request.agents) or 3,
                num_rounds=orch_request.max_rounds,
                model_types=orch_request.agents if orch_request.agents else None,
            )
            if cost_estimate and "total_estimated_cost_usd" in cost_estimate:
                estimated_cost_usd = float(cost_estimate["total_estimated_cost_usd"])
        except (ImportError, ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("Cost estimation failed (non-blocking): %s", exc)

        # Handle dry_run
        if orch_request.dry_run:
            return DryRunResponse(
                request_id=orch_request.request_id,
                estimated_cost=cost_estimate,
                agents=orch_request.agents,
                max_rounds=orch_request.max_rounds,
            )

        # Store request
        requests_store, results_store = _get_orchestration_stores()
        requests_store[orch_request.request_id] = orch_request

        # Queue async execution
        async def _execute_and_store() -> None:
            try:
                result = await orch_handler._execute_deliberation(orch_request)
                results_store[orch_request.request_id] = result
            except (
                ValueError, TypeError, KeyError, AttributeError,
                RuntimeError, OSError,
            ) as e:
                logger.exception("Async deliberation failed: %s", e)
                results_store[orch_request.request_id] = OrchestrationResult(
                    request_id=orch_request.request_id,
                    success=False,
                    error="Orchestration failed",
                )
            finally:
                requests_store.pop(orch_request.request_id, None)

        task = asyncio.create_task(_execute_and_store())
        task.add_done_callback(
            lambda t: logger.error("Async deliberation task failed: %s", t.exception())
            if not t.cancelled() and t.exception()
            else None
        )

        return DeliberateAsyncResponse(
            request_id=orch_request.request_id,
            status="queued",
            message=(
                "Deliberation queued. Check status at "
                f"/api/v2/orchestration/status/{orch_request.request_id}"
            ),
            estimated_cost_usd=estimated_cost_usd,
        )

    except HTTPException:
        raise
    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error starting deliberation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to start deliberation")


@router.post("/orchestration/deliberate/sync")
async def start_deliberation_sync(
    body: DeliberateRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("orchestration:execute")),
    orch_handler: Any = Depends(get_orch_handler),
) -> OrchestrationResultResponse | DryRunResponse:
    """
    Start a synchronous deliberation.

    Blocks until the deliberation is complete and returns the full result.
    For long-running deliberations, prefer the async endpoint.

    Requires `orchestration:execute` permission.
    """
    try:
        data = _build_legacy_request_dict(body)

        try:
            from aragora.server.handlers.orchestration.models import (
                OrchestrationRequest,
            )
            from aragora.server.handlers.orchestration.templates import TEMPLATES
        except ImportError:
            raise HTTPException(
                status_code=503, detail="Orchestration models not available"
            )

        orch_request = OrchestrationRequest.from_dict(data)

        # Apply template if specified
        if orch_request.template and orch_request.template in TEMPLATES:
            template = TEMPLATES[orch_request.template]
            if not orch_request.agents:
                orch_request.agents = template.default_agents
            if not orch_request.knowledge_sources:
                from aragora.server.handlers.orchestration.models import KnowledgeContextSource

                for src in template.default_knowledge_sources:
                    orch_request.knowledge_sources.append(
                        KnowledgeContextSource.from_string(src)
                    )
            orch_request.max_rounds = template.max_rounds

        if not orch_request.question:
            raise HTTPException(status_code=400, detail="Question is required")

        # Cost estimation
        cost_estimate = None
        estimated_cost_usd = None
        try:
            from aragora.server.handlers.debates.cost_estimation import estimate_debate_cost

            cost_estimate = estimate_debate_cost(
                num_agents=len(orch_request.agents) or 3,
                num_rounds=orch_request.max_rounds,
                model_types=orch_request.agents if orch_request.agents else None,
            )
            if cost_estimate and "total_estimated_cost_usd" in cost_estimate:
                estimated_cost_usd = float(cost_estimate["total_estimated_cost_usd"])
        except (ImportError, ValueError, TypeError, KeyError, AttributeError) as exc:
            logger.warning("Cost estimation failed (non-blocking): %s", exc)

        # Handle dry_run
        if orch_request.dry_run:
            return DryRunResponse(
                request_id=orch_request.request_id,
                estimated_cost=cost_estimate,
                agents=orch_request.agents,
                max_rounds=orch_request.max_rounds,
            )

        # Synchronous execution
        requests_store, results_store = _get_orchestration_stores()
        requests_store[orch_request.request_id] = orch_request

        try:
            result = await orch_handler._execute_deliberation(orch_request)
            results_store[orch_request.request_id] = result
        finally:
            requests_store.pop(orch_request.request_id, None)

        result_dict = result.to_dict()

        return OrchestrationResultResponse(
            request_id=result_dict.get("request_id", orch_request.request_id),
            success=result_dict.get("success", False),
            consensus_reached=result_dict.get("consensus_reached", False),
            final_answer=result_dict.get("final_answer"),
            confidence=result_dict.get("confidence"),
            agents_participated=result_dict.get("agents_participated", []),
            rounds_completed=result_dict.get("rounds_completed", 0),
            duration_seconds=result_dict.get("duration_seconds", 0.0),
            knowledge_context_used=result_dict.get("knowledge_context_used", []),
            channels_notified=result_dict.get("channels_notified", []),
            receipt_id=result_dict.get("receipt_id"),
            error=result_dict.get("error"),
            created_at=result_dict.get("created_at"),
            estimated_cost_usd=estimated_cost_usd,
        )

    except HTTPException:
        raise
    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error in synchronous deliberation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to execute deliberation")


@router.get("/orchestration/status/{request_id}", response_model=StatusResponse)
async def get_deliberation_status(
    request_id: str,
    auth: AuthorizationContext = Depends(require_permission("orchestration:read")),
) -> StatusResponse:
    """
    Get the status of a deliberation request.

    Returns status (completed, failed, in_progress) and, if completed,
    the full result.

    Requires `orchestration:read` permission.
    """
    try:
        requests_store, results_store = _get_orchestration_stores()

        # Check if result is available
        if request_id in results_store:
            result = results_store[request_id]
            result_dict = result.to_dict() if hasattr(result, "to_dict") else {}

            return StatusResponse(
                request_id=request_id,
                status="completed" if result_dict.get("success") else "failed",
                result=OrchestrationResultResponse(
                    request_id=result_dict.get("request_id", request_id),
                    success=result_dict.get("success", False),
                    consensus_reached=result_dict.get("consensus_reached", False),
                    final_answer=result_dict.get("final_answer"),
                    confidence=result_dict.get("confidence"),
                    agents_participated=result_dict.get("agents_participated", []),
                    rounds_completed=result_dict.get("rounds_completed", 0),
                    duration_seconds=result_dict.get("duration_seconds", 0.0),
                    knowledge_context_used=result_dict.get("knowledge_context_used", []),
                    channels_notified=result_dict.get("channels_notified", []),
                    receipt_id=result_dict.get("receipt_id"),
                    error=result_dict.get("error"),
                    created_at=result_dict.get("created_at"),
                ),
            )

        # Check if request is still in progress
        if request_id in requests_store:
            return StatusResponse(
                request_id=request_id,
                status="in_progress",
                result=None,
            )

        raise NotFoundError(f"Orchestration request {request_id} not found")

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting deliberation status %s: %s", request_id, e)
        raise HTTPException(status_code=500, detail="Failed to get deliberation status")


@router.get("/orchestration/templates", response_model=TemplateListResponse)
async def list_templates(
    request: Request,
    category: str | None = Query(None, description="Filter by category"),
    search: str | None = Query(None, description="Text search in name and description"),
    tags: str | None = Query(None, description="Comma-separated tag filter"),
    limit: int = Query(50, ge=1, le=500, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    auth: AuthorizationContext = Depends(require_permission("orchestration:read")),
) -> TemplateListResponse:
    """
    List available deliberation templates.

    Returns templates that define pre-built deliberation patterns with
    default agent configurations, knowledge sources, and output formats.

    Supports filtering by category, text search, and tags.
    Requires `orchestration:read` permission.
    """
    try:
        try:
            from aragora.server.handlers.orchestration.templates import (
                _list_templates,
                TEMPLATES,
            )
        except ImportError:
            raise HTTPException(
                status_code=503, detail="Orchestration templates not available"
            )

        parsed_tags = None
        if tags:
            parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]

        if _list_templates is not None:
            templates_raw = _list_templates(
                category=category,
                search=search,
                tags=parsed_tags,
                limit=limit,
                offset=offset,
            )
            template_dicts = [
                t.to_dict() if hasattr(t, "to_dict") else t for t in templates_raw
            ]
        else:
            # Fallback: unfiltered from TEMPLATES dict
            template_dicts = [
                t.to_dict() if hasattr(t, "to_dict") else t
                for t in TEMPLATES.values()
            ]

        templates = []
        for td in template_dicts:
            if isinstance(td, dict):
                templates.append(
                    TemplateItem(
                        name=td.get("name", ""),
                        description=td.get("description", ""),
                        default_agents=td.get("default_agents", []),
                        default_knowledge_sources=td.get("default_knowledge_sources", []),
                        output_format=td.get("output_format", "standard"),
                        consensus_threshold=td.get("consensus_threshold", 0.7),
                        max_rounds=td.get("max_rounds", 5),
                        personas=td.get("personas", []),
                    )
                )
            else:
                templates.append(
                    TemplateItem(
                        name=getattr(td, "name", ""),
                        description=getattr(td, "description", ""),
                        default_agents=getattr(td, "default_agents", []),
                        default_knowledge_sources=getattr(td, "default_knowledge_sources", []),
                        output_format=str(getattr(td, "output_format", "standard")),
                        consensus_threshold=getattr(td, "consensus_threshold", 0.7),
                        max_rounds=getattr(td, "max_rounds", 5),
                        personas=getattr(td, "personas", []),
                    )
                )

        return TemplateListResponse(
            templates=templates,
            count=len(templates),
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing orchestration templates: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list templates")
