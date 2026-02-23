"""Template Registry API handlers.

Provides endpoints for the public template registry:
- Browse and search templates
- Submit new templates
- Install templates
- Rate and review templates
- Admin approve/reject
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    get_int_param,
    get_string_param,
)
from aragora.server.handlers.utils.decorators import require_permission

logger = logging.getLogger(__name__)


class TemplateRegistryHandler(BaseHandler):
    """HTTP handler for the public template registry API."""

    ROUTES = [
        "/api/v1/templates/registry",
        "/api/v1/templates/registry/*",
    ]

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/v1/templates/registry")

    def _get_registry(self):
        from aragora.workflow.templates.registry import get_template_registry

        return get_template_registry()

    def _extract_listing_id(self, path: str) -> str | None:
        """Extract listing ID from path like /api/v1/templates/registry/{id}."""
        parts = path.strip("/").split("/")
        # api/v1/templates/registry/{id} -> 5 parts minimum
        if len(parts) >= 5 and parts[3] == "registry":
            return parts[4]
        return None

    # =====================================================================
    # GET
    # =====================================================================

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        if not self.can_handle(path):
            return None

        listing_id = self._extract_listing_id(path)

        # GET /api/v1/templates/registry/{id}/analytics
        if listing_id and path.endswith("/analytics"):
            return self._handle_get_analytics(listing_id)

        # GET /api/v1/templates/registry/{id}
        if listing_id and not path.endswith("/registry"):
            # Make sure it's not a sub-action path
            action = path.split("/")[-1]
            if action in ("install", "rate", "approve", "reject", "analytics"):
                return None
            return self._handle_get(listing_id)

        # GET /api/v1/templates/registry
        return self._handle_search(query_params)

    def _handle_search(self, query_params: dict[str, Any]) -> HandlerResult:
        registry = self._get_registry()
        query = get_string_param(query_params, "query", "")
        category = get_string_param(query_params, "category", None)
        status = get_string_param(query_params, "status", None)
        limit = get_int_param(query_params, "limit", 20)
        offset = get_int_param(query_params, "offset", 0)

        tags_param = get_string_param(query_params, "tags", None)
        tags = tags_param.split(",") if tags_param else None

        results = registry.search(
            query=query or "",
            category=category,
            tags=tags,
            status=status,
            limit=limit,
            offset=offset,
        )
        return json_response(
            {
                "templates": [r.to_dict() for r in results],
                "count": len(results),
            }
        )

    def _handle_get(self, listing_id: str) -> HandlerResult:
        registry = self._get_registry()
        listing = registry.get(listing_id)
        if listing is None:
            return error_response(f"Template not found: {listing_id}", 404)
        return json_response(listing.to_dict())

    def _handle_get_analytics(self, listing_id: str) -> HandlerResult:
        registry = self._get_registry()
        listing = registry.get(listing_id)
        if listing is None:
            return error_response(f"Template not found: {listing_id}", 404)
        analytics = registry.get_analytics(listing_id)
        return json_response(analytics)

    # =====================================================================
    # POST
    # =====================================================================

    @handle_errors("workflow registry submission")
    @require_permission("workflows:write")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        if not self.can_handle(path):
            return None

        listing_id = self._extract_listing_id(path)

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        # POST /api/v1/templates/registry/{id}/install
        if listing_id and path.endswith("/install"):
            return self._handle_install(listing_id, body, query_params)

        # POST /api/v1/templates/registry/{id}/rate
        if listing_id and path.endswith("/rate"):
            return self._handle_rate(listing_id, body, query_params)

        # POST /api/v1/templates/registry/{id}/approve
        if listing_id and path.endswith("/approve"):
            return self._handle_approve(listing_id, body, query_params)

        # POST /api/v1/templates/registry/{id}/reject
        if listing_id and path.endswith("/reject"):
            return self._handle_reject(listing_id, body, query_params)

        # POST /api/v1/templates/registry (submit new template)
        if not listing_id:
            return self._handle_submit(body, query_params)

        return None

    def _handle_submit(self, body: dict[str, Any], query_params: dict[str, Any]) -> HandlerResult:
        name = body.get("name")
        if not name:
            return error_response("Missing required field: name", 400)

        description = body.get("description", "")
        category = body.get("category", "general")
        tags = body.get("tags", [])
        template_data = body.get("template_data", {})
        author_id = body.get("author_id") or get_string_param(query_params, "user_id", "anonymous")
        version = body.get("version", "1.0.0")

        registry = self._get_registry()
        listing_id = registry.submit(
            template_data=template_data,
            name=name,
            description=description,
            category=category,
            author_id=author_id,
            tags=tags,
            version=version,
        )
        return json_response({"id": listing_id, "status": "pending"}, status=201)

    def _handle_install(
        self, listing_id: str, body: dict[str, Any], query_params: dict[str, Any]
    ) -> HandlerResult:
        user_id = body.get("user_id") or get_string_param(query_params, "user_id", "")
        registry = self._get_registry()
        try:
            template_data = registry.install(listing_id, user_id=user_id)
        except ValueError as e:
            logger.warning("Template install failed: %s", e)
            return error_response("Template not found or not approved", 404)
        return json_response({"installed": True, "template_data": template_data})

    def _handle_rate(
        self, listing_id: str, body: dict[str, Any], query_params: dict[str, Any]
    ) -> HandlerResult:
        rating = body.get("rating")
        if rating is None:
            return error_response("Missing required field: rating", 400)
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            return error_response("Rating must be an integer", 400)

        review = body.get("review")
        user_id = body.get("user_id") or get_string_param(query_params, "user_id", "")

        registry = self._get_registry()
        try:
            registry.rate(listing_id, user_id=user_id, rating=rating, review=review)
        except ValueError as e:
            logger.warning("Template rating failed: %s", e)
            return error_response("Invalid rating value", 400)
        return json_response({"rated": True, "listing_id": listing_id, "rating": rating})

    def _handle_approve(
        self, listing_id: str, body: dict[str, Any], query_params: dict[str, Any]
    ) -> HandlerResult:
        approved_by = body.get("approved_by") or get_string_param(query_params, "user_id", "")
        registry = self._get_registry()
        if registry.approve(listing_id, approved_by=approved_by):
            return json_response({"approved": True, "listing_id": listing_id})
        return error_response(f"Template not found: {listing_id}", 404)

    def _handle_reject(
        self, listing_id: str, body: dict[str, Any], query_params: dict[str, Any]
    ) -> HandlerResult:
        reason = body.get("reason")
        registry = self._get_registry()
        if registry.reject(listing_id, reason=reason):
            return json_response({"rejected": True, "listing_id": listing_id})
        return error_response(f"Template not found: {listing_id}", 404)
