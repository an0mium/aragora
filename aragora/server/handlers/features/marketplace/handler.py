"""Marketplace Handler - Template discovery and deployment.

Provides API endpoints for discovering, browsing, and deploying workflow templates
across different industry verticals.

Stability: STABLE
- Circuit breaker pattern for template loading resilience
- Rate limiting on write operations (deploy, rate)
- Comprehensive input validation (IDs, pagination, ratings, config)
- RBAC permission enforcement

Endpoints:
- GET  /api/v1/marketplace/templates         - List all available templates
- GET  /api/v1/marketplace/templates/{id}    - Get template details
- GET  /api/v1/marketplace/categories        - List template categories
- GET  /api/v1/marketplace/search            - Search templates
- POST /api/v1/marketplace/templates/{id}/deploy - Deploy a template
- GET  /api/v1/marketplace/deployments       - List deployed templates
- GET  /api/v1/marketplace/popular           - Get popular templates
- POST /api/v1/marketplace/templates/{id}/rate - Rate a template
- GET  /api/v1/marketplace/demo              - Get demo marketplace data
- GET  /api/v1/marketplace/status            - Circuit breaker and health status
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from aragora.rbac.decorators import require_permission
from aragora.server.validation.core import sanitize_string

from ...base import (
    HandlerResult,
    error_response,
    json_response,
)
from ...utils.rate_limit import rate_limit
from .circuit_breaker import get_marketplace_circuit_breaker_status
from .models import (
    CATEGORY_INFO,
    DeploymentStatus,
    TemplateCategory,
    TemplateDeployment,
    TemplateMetadata,
    TemplateRating,
)
from .store import (
    _get_full_template,
    _get_tenant_deployments,
    get_deployments,
    get_download_counts,
    get_ratings,
)
from .validation import (
    _validate_category_filter,
    _validate_config,
    _validate_deployment_name_internal,
    _validate_pagination,
    _validate_rating,
    _validate_review_internal,
    _validate_search_query,
    _validate_template_id,
)

logger = logging.getLogger(__name__)


def _load_templates_proxy() -> dict[str, TemplateMetadata]:
    """Load templates via the package namespace so tests can patch it."""
    from aragora.server.handlers.features import marketplace as marketplace_module

    return marketplace_module._load_templates()


async def _parse_json_body_proxy(
    request: Any, *, context: str
) -> tuple[dict[str, Any] | None, Any]:
    """Parse JSON via the package namespace so tests can patch it."""
    from aragora.server.handlers.features import marketplace as marketplace_module

    return await marketplace_module.parse_json_body(request, context=context)


def _get_marketplace_circuit_breaker_proxy():
    """Fetch circuit breaker via package namespace so tests can patch it."""
    from aragora.server.handlers.features import marketplace as marketplace_module

    return marketplace_module._get_marketplace_circuit_breaker()


class MarketplaceHandler:
    """Handler for marketplace API endpoints.

    Production-ready with:
    - Circuit breaker for template loading resilience
    - Rate limiting on write operations (deploy: 20/min, rate: 10/min)
    - Input validation for IDs, pagination, ratings, config, search queries
    - RBAC permission enforcement
    """

    ROUTES = [
        "/api/v1/marketplace/templates",
        "/api/v1/marketplace/templates/search",
        "/api/v1/marketplace/templates/{template_id}",
        "/api/v1/marketplace/templates/{template_id}/deploy",
        "/api/v1/marketplace/templates/{template_id}/rate",
        "/api/v1/marketplace/categories",
        "/api/v1/marketplace/search",
        "/api/v1/marketplace/deployments",
        "/api/v1/marketplace/deployments/{deployment_id}",
        "/api/v1/marketplace/my-deployments",
        "/api/v1/marketplace/popular",
        "/api/v1/marketplace/demo",
        "/api/v1/marketplace/status",
    ]

    ctx: dict[str, Any]

    def __init__(self, server_context: dict[str, Any] | None = None):
        """Initialize handler with optional server context."""
        self.ctx = server_context if server_context is not None else {}
        # Pre-load templates
        _load_templates_proxy()

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        # Check exact routes
        if path in self.ROUTES:
            return True
        # Check template-specific paths with IDs
        if path.startswith("/api/v1/marketplace/templates/"):
            return True
        if path.startswith("/api/v1/marketplace/deployments/"):
            return True
        return False

    @require_permission("marketplace:read")
    async def handle(self, request: Any, path: str, method: str) -> HandlerResult:
        """Route requests to appropriate handler methods."""
        try:
            tenant_id = self._get_tenant_id(request)

            # List templates
            if path == "/api/v1/marketplace/templates" and method == "GET":
                return await self._handle_list_templates(request, tenant_id)

            # List categories
            elif path == "/api/v1/marketplace/categories" and method == "GET":
                return await self._handle_list_categories(request, tenant_id)

            # Search templates
            elif path == "/api/v1/marketplace/search" and method == "GET":
                return await self._handle_search(request, tenant_id)

            # Popular templates
            elif path == "/api/v1/marketplace/popular" and method == "GET":
                return await self._handle_popular(request, tenant_id)

            # List deployments
            elif path == "/api/v1/marketplace/deployments" and method == "GET":
                return await self._handle_list_deployments(request, tenant_id)

            # Demo data
            elif path == "/api/v1/marketplace/demo" and method == "GET":
                return await self._handle_demo(request, tenant_id)

            # Health/status
            elif path == "/api/v1/marketplace/status" and method == "GET":
                return await self._handle_status(request, tenant_id)

            # Template-specific paths
            elif path.startswith("/api/v1/marketplace/templates/"):
                parts = path.split("/")
                if len(parts) >= 6:
                    template_id = parts[5]

                    # Validate template ID
                    valid, err = _validate_template_id(template_id)
                    if not valid:
                        return error_response(err, 400)

                    if len(parts) == 6:
                        if method == "GET":
                            return await self._handle_get_template(request, tenant_id, template_id)

                    elif len(parts) == 7:
                        action = parts[6]
                        if action == "deploy" and method == "POST":
                            return await self._handle_deploy(request, tenant_id, template_id)
                        elif action == "rate" and method == "POST":
                            return await self._handle_rate(request, tenant_id, template_id)

            # Deployment-specific paths
            elif path.startswith("/api/v1/marketplace/deployments/"):
                parts = path.split("/")
                if len(parts) >= 6:
                    deployment_id = parts[5]

                    # Validate deployment ID
                    from .validation import _validate_deployment_id

                    valid, err = _validate_deployment_id(deployment_id)
                    if not valid:
                        return error_response(err, 400)

                    if len(parts) == 6:
                        if method == "GET":
                            return await self._handle_get_deployment(
                                request, tenant_id, deployment_id
                            )
                        elif method == "DELETE":
                            return await self._handle_delete_deployment(
                                request, tenant_id, deployment_id
                            )

            return error_response("Not found", 404)

        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.exception("Error in marketplace handler: %s", e)
            return error_response("Internal server error", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        tenant_id = getattr(request, "tenant_id", None)
        if not tenant_id or not isinstance(tenant_id, str):
            return "default"
        # Sanitize tenant ID
        if len(tenant_id) > 128:
            return "default"
        return tenant_id

    async def _get_json_body(self, request: Any) -> dict[str, Any]:
        """Parse JSON body from request."""
        if hasattr(request, "json"):
            body, _err = await _parse_json_body_proxy(request, context="marketplace._get_json_body")
            return body if body is not None else {}
        return {}

    # =========================================================================
    # List Templates
    # =========================================================================

    async def _handle_list_templates(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all available templates."""
        templates = _load_templates_proxy()
        query = getattr(request, "query", {})

        # Validate category filter
        category_filter = query.get("category")
        if category_filter:
            valid, _, err = _validate_category_filter(category_filter)
            if not valid:
                return error_response(err, 400)
            templates = {k: v for k, v in templates.items() if v.category.value == category_filter}

        # Convert to list and sort by downloads
        template_list = sorted(
            templates.values(),
            key=lambda t: t.downloads,
            reverse=True,
        )

        # Validate pagination
        limit, offset, err = _validate_pagination(query)
        if err:
            return error_response(err, 400)

        total = len(template_list)
        template_list = template_list[offset : offset + limit]

        return json_response(
            {
                "templates": [t.to_dict() for t in template_list],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    # =========================================================================
    # Get Template Details
    # =========================================================================

    async def _handle_get_template(
        self, request: Any, tenant_id: str, template_id: str
    ) -> HandlerResult:
        """Get detailed template information."""
        templates = _load_templates_proxy()
        meta = templates.get(template_id)

        if not meta:
            return error_response("Template not found", 404)

        # Load full template content
        full_template = _get_full_template(template_id)

        # Get ratings
        ratings = get_ratings().get(template_id, [])
        recent_ratings = sorted(ratings, key=lambda r: r.created_at, reverse=True)[:5]

        return json_response(
            {
                "template": meta.to_dict(),
                "full_definition": full_template,
                "ratings": {
                    "average": meta.rating,
                    "count": meta.rating_count,
                    "recent": [r.to_dict() for r in recent_ratings],
                },
                "related": self._get_related_templates(meta, templates),
            }
        )

    def _get_related_templates(
        self, template: TemplateMetadata, all_templates: dict[str, TemplateMetadata]
    ) -> list[dict[str, Any]]:
        """Find related templates based on category and tags."""
        related: list[tuple[int, TemplateMetadata]] = []

        for other in all_templates.values():
            if other.id == template.id:
                continue

            # Same category
            if other.category == template.category:
                score = 2
            else:
                score = 0

            # Shared tags
            shared_tags = set(template.tags) & set(other.tags)
            score += len(shared_tags)

            if score > 0:
                related.append((score, other))

        # Sort by score and take top 5
        related.sort(key=lambda x: x[0], reverse=True)
        related_templates = [t.to_dict() for _, t in related]

        if len(related_templates) < 5:
            # Fill remaining slots with popular templates (by downloads/rating)
            related_ids = {t["id"] for t in related_templates}
            fallback = [
                t for t in all_templates.values() if t.id != template.id and t.id not in related_ids
            ]
            fallback.sort(key=lambda t: (t.downloads, t.rating), reverse=True)
            for extra in fallback:
                related_templates.append(extra.to_dict())
                if len(related_templates) >= 5:
                    break

        return related_templates[:5]

    # =========================================================================
    # List Categories
    # =========================================================================

    async def _handle_list_categories(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all template categories with counts."""
        templates = _load_templates_proxy()

        # Count templates per category
        category_counts: dict[str, int] = {}
        for template in templates.values():
            cat = template.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        categories = []
        for category in TemplateCategory:
            info = CATEGORY_INFO.get(category, {})
            categories.append(
                {
                    "id": category.value,
                    "name": info.get("name", category.value.title()),
                    "description": info.get("description", ""),
                    "icon": info.get("icon", "folder"),
                    "color": info.get("color", "#718096"),
                    "template_count": category_counts.get(category.value, 0),
                }
            )

        return json_response({"categories": categories})

    # =========================================================================
    # Search Templates
    # =========================================================================

    async def _handle_search(self, request: Any, tenant_id: str) -> HandlerResult:
        """Search templates by query."""
        templates = _load_templates_proxy()
        query = getattr(request, "query", {})

        # Validate search query
        raw_query = query.get("q", "")
        valid, search_query, err = _validate_search_query(raw_query)
        if not valid:
            return error_response(err, 400)

        # Validate category filter
        category_filter = query.get("category")
        if category_filter:
            valid, category_filter, err = _validate_category_filter(category_filter)
            if not valid:
                return error_response(err, 400)

        tags_filter = query.get("tags", "").split(",") if query.get("tags") else []
        # Sanitize tags
        tags_filter = [sanitize_string(t.strip(), 100) for t in tags_filter if t.strip()]

        has_debate = query.get("has_debate")
        has_checkpoint = query.get("has_checkpoint")

        results = []
        for template in templates.values():
            # Text search
            if search_query:
                searchable = (
                    f"{template.name} {template.description} {' '.join(template.tags)}".lower()
                )
                if search_query not in searchable:
                    continue

            # Category filter
            if category_filter and template.category.value != category_filter:
                continue

            # Tags filter
            if tags_filter:
                if not any(tag in template.tags for tag in tags_filter):
                    continue

            # Feature filters
            if has_debate == "true" and not template.has_debate:
                continue
            if has_checkpoint == "true" and not template.has_human_checkpoint:
                continue

            results.append(template)

        # Sort by relevance (downloads as proxy)
        results.sort(key=lambda t: t.downloads, reverse=True)

        # Validate pagination
        limit, offset, _ = _validate_pagination(query)
        paged_results = results[offset : offset + limit]

        return json_response(
            {
                "results": [t.to_dict() for t in paged_results],
                "total": len(results),
                "query": search_query,
            }
        )

    # =========================================================================
    # Popular Templates
    # =========================================================================

    async def _handle_popular(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get most popular templates."""
        templates = _load_templates_proxy()
        query = getattr(request, "query", {})

        limit, _, _ = _validate_pagination(query)
        if limit > 50:
            limit = 50

        # Sort by downloads and rating
        sorted_templates = sorted(
            templates.values(),
            key=lambda t: (t.downloads, t.rating),
            reverse=True,
        )[:limit]

        return json_response(
            {
                "popular": [t.to_dict() for t in sorted_templates],
            }
        )

    # =========================================================================
    # Deploy Template
    # =========================================================================

    @rate_limit(requests_per_minute=20, limiter_name="marketplace.deploy")
    async def _handle_deploy(self, request: Any, tenant_id: str, template_id: str) -> HandlerResult:
        """Deploy a template for the tenant."""
        cb = _get_marketplace_circuit_breaker_proxy()
        if not cb.is_allowed():
            return error_response("Marketplace temporarily unavailable", 503)

        try:
            templates = _load_templates_proxy()
            meta = templates.get(template_id)

            if not meta:
                return error_response("Template not found", 404)

            body = await self._get_json_body(request)

            # Validate deployment name
            valid, name, err = _validate_deployment_name_internal(body.get("name"), meta.name)
            if not valid:
                return error_response(err, 400)

            # Validate config
            valid, config, err = _validate_config(body.get("config"))
            if not valid:
                return error_response(err, 400)

            # Create deployment
            deployment_id = f"deploy_{uuid4().hex[:12]}"
            deployment = TemplateDeployment(
                id=deployment_id,
                template_id=template_id,
                tenant_id=tenant_id,
                name=name,
                status=DeploymentStatus.ACTIVE,
                config=config,
            )

            # Store deployment
            tenant_deployments = _get_tenant_deployments(tenant_id)
            tenant_deployments[deployment_id] = deployment

            # Increment download count
            download_counts = get_download_counts()
            download_counts[template_id] = download_counts.get(template_id, 0) + 1
            meta.downloads = download_counts[template_id]

            cb.record_success()

            return json_response(
                {
                    "deployment": deployment.to_dict(),
                    "template": meta.to_dict(),
                    "message": f"Successfully deployed {meta.name}",
                }
            )

        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            cb.record_failure()
            logger.exception("Error deploying template: %s", e)
            return error_response("Deployment failed", 500)

    # =========================================================================
    # List Deployments
    # =========================================================================

    async def _handle_list_deployments(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all deployments for the tenant."""
        deployments = _get_tenant_deployments(tenant_id)

        deployment_list = sorted(
            deployments.values(),
            key=lambda d: d.deployed_at,
            reverse=True,
        )

        return json_response(
            {
                "deployments": [d.to_dict() for d in deployment_list],
                "total": len(deployment_list),
            }
        )

    async def _handle_get_deployment(
        self, request: Any, tenant_id: str, deployment_id: str
    ) -> HandlerResult:
        """Get deployment details."""
        deployments = _get_tenant_deployments(tenant_id)
        deployment = deployments.get(deployment_id)

        if not deployment:
            return error_response("Deployment not found", 404)

        # Get template info
        templates = _load_templates_proxy()
        template = templates.get(deployment.template_id)

        return json_response(
            {
                "deployment": deployment.to_dict(),
                "template": template.to_dict() if template else None,
            }
        )

    async def _handle_delete_deployment(
        self, request: Any, tenant_id: str, deployment_id: str
    ) -> HandlerResult:
        """Archive a deployment."""
        deployments = _get_tenant_deployments(tenant_id)
        deployment = deployments.get(deployment_id)

        if not deployment:
            return error_response("Deployment not found", 404)

        deployment.status = DeploymentStatus.ARCHIVED

        return json_response(
            {
                "message": "Deployment archived",
                "deployment": deployment.to_dict(),
            }
        )

    # =========================================================================
    # Rate Template
    # =========================================================================

    @rate_limit(requests_per_minute=10, limiter_name="marketplace.rate")
    async def _handle_rate(self, request: Any, tenant_id: str, template_id: str) -> HandlerResult:
        """Rate a template."""
        cb = _get_marketplace_circuit_breaker_proxy()
        if not cb.is_allowed():
            return error_response("Marketplace temporarily unavailable", 503)

        try:
            templates = _load_templates_proxy()
            meta = templates.get(template_id)

            if not meta:
                return error_response("Template not found", 404)

            body = await self._get_json_body(request)

            # Validate rating
            valid, rating_value, err = _validate_rating(body.get("rating"))
            if not valid:
                return error_response(err, 400)

            # Validate review
            valid, review, err = _validate_review_internal(body.get("review"))
            if not valid:
                return error_response(err, 400)

            # Create rating
            rating = TemplateRating(
                id=f"rating_{uuid4().hex[:12]}",
                template_id=template_id,
                tenant_id=tenant_id,
                user_id=getattr(request, "user_id", "anonymous"),
                rating=rating_value,
                review=review,
            )

            # Store rating
            ratings_store = get_ratings()
            if template_id not in ratings_store:
                ratings_store[template_id] = []
            ratings_store[template_id].append(rating)

            # Update average rating
            all_ratings = ratings_store[template_id]
            meta.rating = sum(r.rating for r in all_ratings) / len(all_ratings)
            meta.rating_count = len(all_ratings)

            cb.record_success()

            return json_response(
                {
                    "rating": rating.to_dict(),
                    "template_rating": {
                        "average": meta.rating,
                        "count": meta.rating_count,
                    },
                }
            )

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            cb.record_failure()
            logger.exception("Error rating template: %s", e)
            return error_response("Rating failed", 500)

    # =========================================================================
    # Health / Status
    # =========================================================================

    async def _handle_status(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get marketplace health status including circuit breaker state."""
        templates = _load_templates_proxy()
        cb_status = get_marketplace_circuit_breaker_status()

        return json_response(
            {
                "status": "healthy" if cb_status["state"] == "closed" else "degraded",
                "templates_loaded": len(templates),
                "circuit_breaker": cb_status,
                "deployments_count": sum(len(d) for d in get_deployments().values()),
                "ratings_count": sum(len(r) for r in get_ratings().values()),
            }
        )

    # =========================================================================
    # Demo Data
    # =========================================================================

    async def _handle_demo(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get demo marketplace data for development."""
        templates = _load_templates_proxy()

        # Group by category
        by_category: dict[str, list[dict[str, Any]]] = {}
        for template in templates.values():
            cat = template.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(template.to_dict())

        # Get featured templates (highest rated or most downloads)
        featured = sorted(
            templates.values(),
            key=lambda t: (t.rating, t.downloads),
            reverse=True,
        )[:6]

        return json_response(
            {
                "featured": [t.to_dict() for t in featured],
                "by_category": by_category,
                "categories": [
                    {
                        "id": cat.value,
                        **CATEGORY_INFO.get(cat, {}),
                        "count": len(by_category.get(cat.value, [])),
                    }
                    for cat in TemplateCategory
                ],
                "total_templates": len(templates),
            }
        )


# =============================================================================
# Module-level helpers
# =============================================================================


def get_marketplace_handler() -> MarketplaceHandler:
    """Get a MarketplaceHandler instance."""
    return MarketplaceHandler()


@require_permission("debates:write")
async def handle_marketplace(request: Any, path: str, method: str) -> HandlerResult:
    """Handle a marketplace request."""
    handler = get_marketplace_handler()
    return await handler.handle(request, path, method)
