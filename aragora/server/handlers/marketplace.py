"""
Marketplace API Handlers.

Provides REST API endpoints for the agent template marketplace,
including template CRUD, search, ratings, and import/export.
"""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING

from aragora.server.handlers.base import BaseHandler, HandlerResult

if TYPE_CHECKING:
    from aragora.marketplace import TemplateRegistry

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_registry: TemplateRegistry | None = None


def _get_registry() -> "TemplateRegistry":
    """Get or create the template registry."""
    global _registry
    if _registry is None:
        from aragora.marketplace import TemplateRegistry

        _registry = TemplateRegistry()
    return _registry


class MarketplaceHandler(BaseHandler):
    """Handler for marketplace template operations."""

    def handle_list_templates(self) -> HandlerResult:
        """
        GET /api/v1/marketplace/templates

        Query params:
            q: Search query
            category: Filter by category
            type: Filter by template type
            tags: Comma-separated tags
            limit: Max results (default 50)
            offset: Pagination offset
        """
        try:
            registry = _get_registry()

            # Parse query parameters
            query = self.get_query_param("q")
            category_str = self.get_query_param("category")
            template_type = self.get_query_param("type")
            tags_str = self.get_query_param("tags")
            limit = int(self.get_query_param("limit") or "50")
            offset = int(self.get_query_param("offset") or "0")

            # Convert category string to enum
            category = None
            if category_str:
                from aragora.marketplace import TemplateCategory

                try:
                    category = TemplateCategory(category_str)
                except ValueError:
                    return self.json_error(
                        f"Invalid category: {category_str}",
                        HTTPStatus.BAD_REQUEST,
                    )

            # Parse tags
            tags = tags_str.split(",") if tags_str else None

            # Search templates
            templates = registry.search(
                query=query,
                category=category,
                template_type=template_type,
                tags=tags,
                limit=limit,
                offset=offset,
            )

            return self.json_response(
                {
                    "templates": [t.to_dict() for t in templates],
                    "count": len(templates),
                    "limit": limit,
                    "offset": offset,
                }
            )

        except Exception as e:
            logger.exception("Error listing templates")
            return self.json_error(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_get_template(self, template_id: str) -> HandlerResult:
        """
        GET /api/v1/marketplace/templates/{id}
        """
        try:
            registry = _get_registry()
            template = registry.get(template_id)

            if template is None:
                return self.json_error(
                    f"Template not found: {template_id}",
                    HTTPStatus.NOT_FOUND,
                )

            # Increment download count
            registry.increment_downloads(template_id)

            return self.json_response(template.to_dict())

        except Exception as e:
            logger.exception(f"Error getting template {template_id}")
            return self.json_error(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_create_template(self) -> HandlerResult:
        """
        POST /api/v1/marketplace/templates

        Body: Template JSON
        """
        try:
            user, error = self.require_auth_or_error(self._current_handler)
            if error:
                return error

            registry = _get_registry()
            body = self.get_json_body()

            if not body:
                return self.json_error("Request body required", HTTPStatus.BAD_REQUEST)

            # Import and register template
            template_id = registry.import_template(json.dumps(body))

            return self.json_response(
                {"id": template_id, "success": True},
                status=HTTPStatus.CREATED,
            )

        except ValueError as e:
            return self.json_error(str(e), HTTPStatus.BAD_REQUEST)
        except Exception as e:
            logger.exception("Error creating template")
            return self.json_error(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_delete_template(self, template_id: str) -> HandlerResult:
        """
        DELETE /api/v1/marketplace/templates/{id}
        """
        try:
            user, error = self.require_auth_or_error(self._current_handler)
            if error:
                return error

            registry = _get_registry()
            result = registry.delete(template_id)

            if not result:
                return self.json_error(
                    f"Cannot delete template: {template_id} (may be built-in)",
                    HTTPStatus.FORBIDDEN,
                )

            return self.json_response({"success": True, "deleted": template_id})

        except Exception as e:
            logger.exception(f"Error deleting template {template_id}")
            return self.json_error(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_rate_template(self, template_id: str) -> HandlerResult:
        """
        POST /api/v1/marketplace/templates/{id}/ratings

        Body: {"score": 1-5, "review": "optional review text"}
        """
        try:
            user, error = self.require_auth_or_error(self._current_handler)
            if error:
                return error

            registry = _get_registry()
            body = self.get_json_body()

            if not body or "score" not in body:
                return self.json_error("Score required", HTTPStatus.BAD_REQUEST)

            score = body["score"]
            if not isinstance(score, int) or not 1 <= score <= 5:
                return self.json_error("Score must be 1-5", HTTPStatus.BAD_REQUEST)

            from aragora.marketplace import TemplateRating

            rating = TemplateRating(
                user_id=user.id if hasattr(user, "id") else str(user),
                template_id=template_id,
                score=score,
                review=body.get("review"),
            )
            registry.rate(rating)

            return self.json_response(
                {
                    "success": True,
                    "average_rating": registry.get_average_rating(template_id),
                }
            )

        except ValueError as e:
            return self.json_error(str(e), HTTPStatus.BAD_REQUEST)
        except Exception as e:
            logger.exception(f"Error rating template {template_id}")
            return self.json_error(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_get_ratings(self, template_id: str) -> HandlerResult:
        """
        GET /api/v1/marketplace/templates/{id}/ratings
        """
        try:
            registry = _get_registry()
            ratings = registry.get_ratings(template_id)
            avg = registry.get_average_rating(template_id)

            return self.json_response(
                {
                    "ratings": [
                        {
                            "user_id": r.user_id,
                            "score": r.score,
                            "review": r.review,
                            "created_at": r.created_at.isoformat(),
                        }
                        for r in ratings
                    ],
                    "average": avg,
                    "count": len(ratings),
                }
            )

        except Exception as e:
            logger.exception(f"Error getting ratings for {template_id}")
            return self.json_error(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_star_template(self, template_id: str) -> HandlerResult:
        """
        POST /api/v1/marketplace/templates/{id}/star
        """
        try:
            user, error = self.require_auth_or_error(self._current_handler)
            if error:
                return error

            registry = _get_registry()
            registry.star(template_id)

            template = registry.get(template_id)
            stars = template.metadata.stars if template else 0

            return self.json_response({"success": True, "stars": stars})

        except Exception as e:
            logger.exception(f"Error starring template {template_id}")
            return self.json_error(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_list_categories(self) -> HandlerResult:
        """
        GET /api/v1/marketplace/categories
        """
        try:
            registry = _get_registry()
            categories = registry.list_categories()

            return self.json_response({"categories": categories})

        except Exception as e:
            logger.exception("Error listing categories")
            return self.json_error(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_export_template(self, template_id: str) -> HandlerResult:
        """
        GET /api/v1/marketplace/templates/{id}/export
        """
        try:
            registry = _get_registry()
            json_str = registry.export_template(template_id)

            if json_str is None:
                return self.json_error(
                    f"Template not found: {template_id}",
                    HTTPStatus.NOT_FOUND,
                )

            return HandlerResult(
                status_code=HTTPStatus.OK,
                content_type="application/json",
                body=json_str.encode("utf-8"),
                headers={"Content-Disposition": f'attachment; filename="{template_id}.json"'},
            )

        except Exception as e:
            logger.exception(f"Error exporting template {template_id}")
            return self.json_error(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_import_template(self) -> HandlerResult:
        """
        POST /api/v1/marketplace/templates/import

        Body: Template JSON
        """
        return self.handle_create_template()
