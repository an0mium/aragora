"""
Marketplace Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/marketplace.py (aiohttp handler)

Provides async marketplace template endpoints:
- GET    /api/v2/marketplace/templates              - List/search templates
- GET    /api/v2/marketplace/templates/{template_id} - Get template details
- POST   /api/v2/marketplace/templates              - Create a template
- DELETE /api/v2/marketplace/templates/{template_id} - Delete a template
- POST   /api/v2/marketplace/templates/{template_id}/ratings - Rate a template
- GET    /api/v2/marketplace/templates/{template_id}/ratings - Get template ratings
- POST   /api/v2/marketplace/templates/{template_id}/star    - Star a template
- GET    /api/v2/marketplace/categories             - List categories
- GET    /api/v2/marketplace/templates/{template_id}/export  - Export template
- POST   /api/v2/marketplace/templates/import       - Import a template
- GET    /api/v2/marketplace/status                 - Health and circuit breaker status

Migration Notes:
    This module replaces the legacy MarketplaceHandler class with native FastAPI
    routes. Key improvements:
    - Pydantic request/response models with automatic validation
    - FastAPI dependency injection for auth and storage
    - Proper HTTP status codes (422 for validation, 404 for not found)
    - OpenAPI schema auto-generation
    - Circuit breaker pattern preserved for registry access resilience
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_permission
from ..middleware.error_handling import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/marketplace", tags=["Marketplace"])


# =============================================================================
# Constants for Input Validation
# =============================================================================

SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")

MAX_TEMPLATE_ID_LENGTH = 128
MAX_QUERY_LENGTH = 500
MAX_TAGS_LENGTH = 1000
MAX_REVIEW_LENGTH = 2000
MIN_RATING = 1
MAX_RATING = 5
DEFAULT_LIMIT = 50
MIN_LIMIT = 1
MAX_LIMIT = 200
MAX_OFFSET = 10000


# =============================================================================
# Pydantic Models
# =============================================================================


class TemplateSummary(BaseModel):
    """Summary of a marketplace template."""

    id: str = ""
    name: str = ""
    description: str = ""
    category: str = ""
    template_type: str = ""
    tags: list[str] = Field(default_factory=list)
    downloads: int = 0
    stars: int = 0
    average_rating: float = 0.0

    model_config = {"extra": "allow"}


class TemplateListResponse(BaseModel):
    """Response for template listing."""

    templates: list[dict[str, Any]]
    count: int
    limit: int
    offset: int


class TemplateDetailResponse(BaseModel):
    """Full template details."""

    model_config = {"extra": "allow"}


class CreateTemplateRequest(BaseModel):
    """Request body for POST /templates."""

    id: str | None = Field(None, max_length=128, description="Template ID")
    name: str = Field(..., min_length=1, max_length=200, description="Template name")
    description: str = Field("", max_length=2000, description="Template description")
    category: str = Field("", description="Template category")
    template_type: str = Field("", description="Template type")
    tags: list[str] = Field(default_factory=list, description="Template tags")
    config: dict[str, Any] = Field(default_factory=dict, description="Template configuration")

    model_config = {"extra": "allow"}


class CreateTemplateResponse(BaseModel):
    """Response for POST /templates."""

    id: str
    success: bool


class DeleteTemplateResponse(BaseModel):
    """Response for DELETE /templates/{id}."""

    success: bool
    deleted: str


class RateTemplateRequest(BaseModel):
    """Request body for POST /templates/{id}/ratings."""

    score: int = Field(..., ge=MIN_RATING, le=MAX_RATING, description="Rating score (1-5)")
    review: str | None = Field(
        None, max_length=MAX_REVIEW_LENGTH, description="Optional review text"
    )


class RateTemplateResponse(BaseModel):
    """Response for POST /templates/{id}/ratings."""

    success: bool
    average_rating: float


class RatingEntry(BaseModel):
    """A single rating entry."""

    user_id: str = ""
    score: int = 0
    review: str | None = None
    created_at: str = ""

    model_config = {"extra": "allow"}


class RatingsResponse(BaseModel):
    """Response for GET /templates/{id}/ratings."""

    ratings: list[RatingEntry]
    average: float
    count: int


class StarTemplateResponse(BaseModel):
    """Response for POST /templates/{id}/star."""

    success: bool
    stars: int


class CategoriesResponse(BaseModel):
    """Response for GET /categories."""

    categories: list[str]


class MarketplaceStatusResponse(BaseModel):
    """Response for GET /status."""

    status: str
    circuit_breaker: dict[str, Any]


# =============================================================================
# Circuit Breaker
# =============================================================================

_circuit_breaker_instance: Any = None
_circuit_breaker_lock = threading.Lock()


def _get_circuit_breaker() -> Any:
    """Get or create the marketplace circuit breaker."""
    global _circuit_breaker_instance
    with _circuit_breaker_lock:
        if _circuit_breaker_instance is None:
            try:
                from aragora.resilience.simple_circuit_breaker import (
                    SimpleCircuitBreaker,
                )

                _circuit_breaker_instance = SimpleCircuitBreaker()
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning("Circuit breaker not available: %s", e)
                return None
        return _circuit_breaker_instance


def _check_circuit_breaker() -> None:
    """Raise 503 if the circuit breaker is open."""
    cb = _get_circuit_breaker()
    if cb and not cb.can_proceed():
        raise HTTPException(
            status_code=503,
            detail="Marketplace temporarily unavailable. Please try again later.",
        )


def _record_success() -> None:
    """Record a successful operation on the circuit breaker."""
    cb = _get_circuit_breaker()
    if cb:
        cb.record_success()


def _record_failure() -> None:
    """Record a failed operation on the circuit breaker."""
    cb = _get_circuit_breaker()
    if cb:
        cb.record_failure()


# =============================================================================
# Dependencies
# =============================================================================


async def get_registry(request: Request) -> Any:
    """Dependency to get the TemplateRegistry from app state or create one."""
    ctx = getattr(request.app.state, "context", None)
    if ctx and "marketplace_registry" in ctx:
        return ctx.get("marketplace_registry")

    # Fall back to creating a new registry
    try:
        from aragora.marketplace import TemplateRegistry

        return TemplateRegistry()
    except (ImportError, RuntimeError, ValueError) as e:
        logger.warning("Marketplace registry not available: %s", e)
        return None


# =============================================================================
# Helpers
# =============================================================================


def _validate_template_id(template_id: str) -> None:
    """Validate a template ID string; raises HTTPException on failure."""
    if not template_id or not isinstance(template_id, str):
        raise HTTPException(status_code=400, detail="Template ID is required")
    if len(template_id) > MAX_TEMPLATE_ID_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Template ID must be at most {MAX_TEMPLATE_ID_LENGTH} characters",
        )
    if not SAFE_ID_PATTERN.match(template_id):
        raise HTTPException(
            status_code=400,
            detail="Template ID contains invalid characters",
        )


def _require_registry(registry: Any) -> Any:
    """Ensure the registry is available; raises 503 if not."""
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Marketplace registry is not available",
        )
    return registry


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/templates", response_model=TemplateListResponse)
async def list_templates(
    request: Request,
    q: str | None = Query(None, max_length=MAX_QUERY_LENGTH, description="Search query"),
    category: str | None = Query(None, description="Filter by category"),
    type: str | None = Query(None, description="Filter by template type"),
    tags: str | None = Query(None, max_length=MAX_TAGS_LENGTH, description="Comma-separated tags"),
    limit: int = Query(DEFAULT_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT, description="Max results"),
    offset: int = Query(0, ge=0, le=MAX_OFFSET, description="Pagination offset"),
    registry=Depends(get_registry),
) -> TemplateListResponse:
    """
    List or search marketplace templates.

    Returns a paginated list of templates with optional filtering by query,
    category, type, and tags.
    """
    _check_circuit_breaker()
    registry = _require_registry(registry)

    try:
        # Parse category enum if provided
        category_enum = None
        if category:
            try:
                from aragora.marketplace import TemplateCategory

                category_enum = TemplateCategory(category)
            except (ValueError, ImportError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category: {category}",
                )

        # Parse tags
        parsed_tags: list[str] | None = None
        if tags:
            parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]

        templates = registry.search(
            query=q if q else None,
            category=category_enum,
            template_type=type,
            tags=parsed_tags,
            limit=limit,
            offset=offset,
        )

        _record_success()

        return TemplateListResponse(
            templates=[t.to_dict() for t in templates],
            count=len(templates),
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise
    except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
        _record_failure()
        logger.exception("Error listing templates: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list templates")


@router.get("/categories", response_model=CategoriesResponse)
async def list_categories(
    request: Request,
    registry=Depends(get_registry),
) -> CategoriesResponse:
    """
    List available template categories.

    Returns all marketplace category names.
    """
    _check_circuit_breaker()
    registry = _require_registry(registry)

    try:
        categories = registry.list_categories()
        _record_success()
        return CategoriesResponse(categories=categories)

    except (KeyError, ValueError, TypeError, OSError) as e:
        _record_failure()
        logger.exception("Error listing categories: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list categories")


@router.get("/status", response_model=MarketplaceStatusResponse)
async def get_marketplace_status(
    request: Request,
) -> MarketplaceStatusResponse:
    """
    Get marketplace health and circuit breaker status.

    Returns the overall health status and circuit breaker state.
    """
    cb = _get_circuit_breaker()
    if cb:
        cb_status = cb.get_status()
    else:
        cb_status = {"state": "unknown", "message": "Circuit breaker not available"}

    status = "healthy" if cb_status.get("state") == "closed" else "degraded"
    return MarketplaceStatusResponse(status=status, circuit_breaker=cb_status)


@router.post("/templates/import", response_model=CreateTemplateResponse, status_code=201)
async def import_template(
    request: Request,
    body: CreateTemplateRequest,
    auth: AuthorizationContext = Depends(require_permission("marketplace:write")),
    registry=Depends(get_registry),
) -> CreateTemplateResponse:
    """
    Import a template.

    Alias for template creation. Accepts the same body format.
    """
    return await create_template(request=request, body=body, auth=auth, registry=registry)


@router.get("/templates/{template_id}/ratings", response_model=RatingsResponse)
async def get_template_ratings(
    template_id: str,
    request: Request,
    registry=Depends(get_registry),
) -> RatingsResponse:
    """
    Get ratings for a template.

    Returns all ratings and the average score for the specified template.
    """
    _validate_template_id(template_id)
    _check_circuit_breaker()
    registry = _require_registry(registry)

    try:
        ratings = registry.get_ratings(template_id)
        avg = registry.get_average_rating(template_id)

        _record_success()

        return RatingsResponse(
            ratings=[
                RatingEntry(
                    user_id=r.user_id,
                    score=r.score,
                    review=r.review,
                    created_at=r.created_at.isoformat()
                    if hasattr(r.created_at, "isoformat")
                    else str(r.created_at),
                )
                for r in ratings
            ],
            average=avg,
            count=len(ratings),
        )

    except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
        _record_failure()
        logger.exception("Error getting ratings for %s: %s", template_id, e)
        raise HTTPException(status_code=500, detail="Failed to get template ratings")


@router.get("/templates/{template_id}/export")
async def export_template(
    template_id: str,
    request: Request,
    registry=Depends(get_registry),
) -> Response:
    """
    Export a template as a JSON file download.

    Returns the template data as an application/json attachment.
    """
    _validate_template_id(template_id)
    _check_circuit_breaker()
    registry = _require_registry(registry)

    try:
        json_str = registry.export_template(template_id)

        if json_str is None:
            raise NotFoundError(f"Template not found: {template_id}")

        _record_success()

        return Response(
            content=json_str.encode("utf-8"),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{template_id}.json"'},
        )

    except NotFoundError:
        raise
    except (KeyError, ValueError, TypeError, OSError, UnicodeEncodeError) as e:
        _record_failure()
        logger.exception("Error exporting template %s: %s", template_id, e)
        raise HTTPException(status_code=500, detail="Failed to export template")


@router.get("/templates/{template_id}", response_model=TemplateSummary)
async def get_template(
    template_id: str,
    request: Request,
    registry=Depends(get_registry),
) -> dict[str, Any]:
    """
    Get template details by ID.

    Returns detailed information about a specific template. Also increments
    the download counter.
    """
    _validate_template_id(template_id)
    _check_circuit_breaker()
    registry = _require_registry(registry)

    try:
        template = registry.get(template_id)

        if template is None:
            raise NotFoundError(f"Template not found: {template_id}")

        # Increment download count
        registry.increment_downloads(template_id)

        _record_success()

        return template.to_dict()

    except NotFoundError:
        raise
    except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
        _record_failure()
        logger.exception("Error getting template %s: %s", template_id, e)
        raise HTTPException(status_code=500, detail="Failed to get template")


@router.post("/templates", response_model=CreateTemplateResponse, status_code=201)
async def create_template(
    request: Request,
    body: CreateTemplateRequest,
    auth: AuthorizationContext = Depends(require_permission("marketplace:write")),
    registry=Depends(get_registry),
) -> CreateTemplateResponse:
    """
    Create a new marketplace template.

    Requires `marketplace:write` permission. Validates the template ID if
    provided, then imports the template into the registry.
    """
    _check_circuit_breaker()
    registry = _require_registry(registry)

    try:
        # Validate template ID if explicitly provided
        if body.id:
            _validate_template_id(body.id)

        template_data = body.model_dump(exclude_none=True)
        template_id = registry.import_template(json.dumps(template_data))

        _record_success()

        return CreateTemplateResponse(id=template_id, success=True)

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid template data: %s", e)
        raise HTTPException(status_code=400, detail="Invalid template data")
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as e:
        _record_failure()
        logger.exception("Error creating template: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create template")


@router.delete("/templates/{template_id}", response_model=DeleteTemplateResponse)
async def delete_template(
    template_id: str,
    auth: AuthorizationContext = Depends(require_permission("marketplace:delete")),
    registry=Depends(get_registry),
) -> DeleteTemplateResponse:
    """
    Delete a marketplace template.

    Requires `marketplace:delete` permission. Built-in templates cannot be
    deleted and will return 403.
    """
    _validate_template_id(template_id)
    _check_circuit_breaker()
    registry = _require_registry(registry)

    try:
        result = registry.delete(template_id)

        if not result:
            raise HTTPException(
                status_code=403,
                detail=f"Cannot delete template: {template_id} (may be built-in)",
            )

        _record_success()

        return DeleteTemplateResponse(success=True, deleted=template_id)

    except HTTPException:
        raise
    except (KeyError, ValueError, TypeError, OSError) as e:
        _record_failure()
        logger.exception("Error deleting template %s: %s", template_id, e)
        raise HTTPException(status_code=500, detail="Failed to delete template")


@router.post("/templates/{template_id}/ratings", response_model=RateTemplateResponse)
async def rate_template(
    template_id: str,
    body: RateTemplateRequest,
    auth: AuthorizationContext = Depends(require_permission("marketplace:write")),
    registry=Depends(get_registry),
) -> RateTemplateResponse:
    """
    Rate a marketplace template.

    Requires `marketplace:write` permission. Accepts a score (1-5) and an
    optional review text.
    """
    _validate_template_id(template_id)
    _check_circuit_breaker()
    registry = _require_registry(registry)

    try:
        from aragora.marketplace import TemplateRating

        rating = TemplateRating(
            user_id=auth.user_id,
            template_id=template_id,
            score=body.score,
            review=body.review,
        )
        registry.rate(rating)

        _record_success()

        return RateTemplateResponse(
            success=True,
            average_rating=registry.get_average_rating(template_id),
        )

    except (ImportError, RuntimeError) as e:
        logger.warning("Marketplace rating infrastructure not available: %s", e)
        raise HTTPException(status_code=503, detail="Marketplace rating not available")
    except ValueError as e:
        logger.warning("Invalid rating data for template %s: %s", template_id, e)
        raise HTTPException(status_code=400, detail="Invalid rating data")
    except (KeyError, TypeError, AttributeError, OSError) as e:
        _record_failure()
        logger.exception("Error rating template %s: %s", template_id, e)
        raise HTTPException(status_code=500, detail="Failed to rate template")


@router.post("/templates/{template_id}/star", response_model=StarTemplateResponse)
async def star_template(
    template_id: str,
    auth: AuthorizationContext = Depends(require_permission("marketplace:write")),
    registry=Depends(get_registry),
) -> StarTemplateResponse:
    """
    Star a marketplace template.

    Requires `marketplace:write` permission. Increments the star count
    and returns the new total.
    """
    _validate_template_id(template_id)
    _check_circuit_breaker()
    registry = _require_registry(registry)

    try:
        registry.star(template_id)

        template = registry.get(template_id)
        stars = template.metadata.stars if template else 0

        _record_success()

        return StarTemplateResponse(success=True, stars=stars)

    except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
        _record_failure()
        logger.exception("Error starring template %s: %s", template_id, e)
        raise HTTPException(status_code=500, detail="Failed to star template")
