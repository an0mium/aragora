"""
Example: ResourceHandler usage.

This example shows how to use ResourceHandler for RESTful CRUD endpoints.
ResourceHandler provides a structured approach to handling standard
resource operations with automatic permission mapping.

Key features demonstrated:
- Automatic CRUD routing (list, get, create, update, delete)
- Resource-based permission generation
- Standard RESTful patterns
- Minimal boilerplate for CRUD handlers
"""

from __future__ import annotations

import logging
from typing import Any
from datetime import datetime, timezone

from aragora.protocols import HTTPRequestHandler

from ..base import (
    HandlerResult,
    ResourceHandler,
    json_response,
    error_response,
)

logger = logging.getLogger(__name__)


class ExampleResourceHandler(ResourceHandler):
    """
    Example RESTful resource handler.

    ResourceHandler extends PermissionHandler with:
    - Automatic routing for CRUD operations
    - Auto-generated permissions from RESOURCE_NAME
    - Standard method signatures for resource operations
    - Built-in resource ID extraction from paths

    For RESOURCE_NAME = "article", permissions are:
    - GET: article:read
    - POST: article:create
    - PUT/PATCH: article:update
    - DELETE: article:delete

    Routes follow REST conventions:
    - GET /api/v1/articles -> _list_resources()
    - GET /api/v1/articles/{id} -> _get_resource(id)
    - POST /api/v1/articles -> _create_resource()
    - PUT /api/v1/articles/{id} -> _update_resource(id)
    - PATCH /api/v1/articles/{id} -> _patch_resource(id)
    - DELETE /api/v1/articles/{id} -> _delete_resource(id)
    """

    # Define the resource name - used for permission generation
    RESOURCE_NAME = "article"

    # Define routes (plural form is convention)
    ROUTES = [
        "/api/v1/articles",
        "/api/v1/articles/*",  # Wildcard for ID-based routes
    ]

    # Simulated in-memory storage for the example
    _storage: dict[str, dict[str, Any]] = {
        "article-1": {
            "id": "article-1",
            "title": "Getting Started with Aragora",
            "content": "Introduction to the platform...",
            "author": "system",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        },
        "article-2": {
            "id": "article-2",
            "title": "Advanced Debate Configuration",
            "content": "Deep dive into debate settings...",
            "author": "system",
            "created_at": "2024-01-02T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        },
    }

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path == "/api/v1/articles" or path.startswith("/api/v1/articles/")

    # Override the resource operation methods

    def _list_resources(
        self, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult:
        """
        List all resources with optional filtering and pagination.

        Called for: GET /api/v1/articles

        Args:
            query_params: Query parameters for filtering/pagination
            handler: HTTP request handler

        Returns:
            JSON response with list of resources
        """
        # Extract pagination params
        limit = int(query_params.get("limit", 20))
        offset = int(query_params.get("offset", 0))

        # Get all articles
        articles = list(self._storage.values())
        total = len(articles)

        # Apply pagination
        paginated = articles[offset : offset + limit]

        return json_response(
            {
                "articles": paginated,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(paginated) < total,
            }
        )

    def _get_resource(self, resource_id: str, handler: HTTPRequestHandler) -> HandlerResult:
        """
        Get a single resource by ID.

        Called for: GET /api/v1/articles/{id}

        Args:
            resource_id: The article ID from the URL path
            handler: HTTP request handler

        Returns:
            JSON response with the resource, or 404 if not found
        """
        article = self._storage.get(resource_id)
        if not article:
            return error_response(f"Article not found: {resource_id}", 404)

        return json_response(article)

    def _create_resource(self, handler: HTTPRequestHandler) -> HandlerResult:
        """
        Create a new resource.

        Called for: POST /api/v1/articles

        Args:
            handler: HTTP request handler with JSON body

        Returns:
            JSON response with created resource (201 status)
        """
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Validate required fields
        title = body.get("title")
        if not title:
            return error_response("Title is required", 400)

        # Generate ID and create article
        article_id = f"article-{len(self._storage) + 1}"
        now = datetime.now(timezone.utc).isoformat() + "Z"

        article = {
            "id": article_id,
            "title": title,
            "content": body.get("content", ""),
            "author": self.current_user.user_id if self.current_user else "anonymous",
            "created_at": now,
            "updated_at": now,
        }

        self._storage[article_id] = article
        logger.info(f"Created article {article_id}")

        return json_response(article, status=201)

    def _update_resource(self, resource_id: str, handler: HTTPRequestHandler) -> HandlerResult:
        """
        Update a resource (full replacement).

        Called for: PUT /api/v1/articles/{id}

        Args:
            resource_id: The article ID from the URL path
            handler: HTTP request handler with JSON body

        Returns:
            JSON response with updated resource
        """
        if resource_id not in self._storage:
            return error_response(f"Article not found: {resource_id}", 404)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Validate required fields for full update
        title = body.get("title")
        if not title:
            return error_response("Title is required for full update", 400)

        # Update article
        existing = self._storage[resource_id]
        now = datetime.now(timezone.utc).isoformat() + "Z"

        article = {
            "id": resource_id,
            "title": title,
            "content": body.get("content", ""),
            "author": existing["author"],  # Preserve original author
            "created_at": existing["created_at"],
            "updated_at": now,
        }

        self._storage[resource_id] = article
        logger.info(f"Updated article {resource_id}")

        return json_response(article)

    def _patch_resource(self, resource_id: str, handler: HTTPRequestHandler) -> HandlerResult:
        """
        Partially update a resource.

        Called for: PATCH /api/v1/articles/{id}

        Only updates fields provided in the request body.

        Args:
            resource_id: The article ID from the URL path
            handler: HTTP request handler with JSON body

        Returns:
            JSON response with updated resource
        """
        if resource_id not in self._storage:
            return error_response(f"Article not found: {resource_id}", 404)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        if not body:
            return error_response("No fields to update", 400)

        # Partial update - only update provided fields
        article = self._storage[resource_id].copy()
        now = datetime.now(timezone.utc).isoformat() + "Z"

        # Only update allowed fields
        if "title" in body:
            article["title"] = body["title"]
        if "content" in body:
            article["content"] = body["content"]

        article["updated_at"] = now

        self._storage[resource_id] = article
        logger.info(f"Patched article {resource_id}: {list(body.keys())}")

        return json_response(article)

    def _delete_resource(self, resource_id: str, handler: HTTPRequestHandler) -> HandlerResult:
        """
        Delete a resource.

        Called for: DELETE /api/v1/articles/{id}

        Args:
            resource_id: The article ID from the URL path
            handler: HTTP request handler

        Returns:
            JSON response confirming deletion
        """
        if resource_id not in self._storage:
            return error_response(f"Article not found: {resource_id}", 404)

        del self._storage[resource_id]
        logger.info(f"Deleted article {resource_id}")

        return json_response(
            {
                "deleted": True,
                "id": resource_id,
            }
        )
