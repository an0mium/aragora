"""
Example: PermissionHandler usage.

This example shows how to use PermissionHandler for handlers that need
fine-grained RBAC permission checking beyond simple authentication.

Key features demonstrated:
- Method-level permission requirements
- Custom permission checking
- Permission-based access control
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.protocols import HTTPRequestHandler

from ..base import (
    HandlerResult,
    PermissionHandler,
    json_response,
    error_response,
)

logger = logging.getLogger(__name__)


class ExamplePermissionHandler(PermissionHandler):
    """
    Example handler with RBAC permission checking.

    PermissionHandler extends AuthenticatedHandler with:
    - REQUIRED_PERMISSIONS dict mapping HTTP methods to permission strings
    - _ensure_permission() method for automatic permission checking
    - _check_custom_permission() for endpoint-specific permissions

    Permission strings follow the format: "resource:action"
    Examples: "documents:read", "documents:write", "admin:manage"
    """

    ROUTES = [
        "/api/v1/documents",
        "/api/v1/documents/sensitive",
    ]

    # Define method -> permission mapping
    # These are checked automatically by _ensure_permission()
    REQUIRED_PERMISSIONS = {
        "GET": "documents:read",
        "POST": "documents:write",
        "PUT": "documents:write",
        "PATCH": "documents:write",
        "DELETE": "documents:delete",
    }

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES or path.startswith("/api/v1/documents/")

    def handle(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """
        Handle GET requests with permission checking.

        The _ensure_permission() method:
        1. Verifies authentication (like _ensure_authenticated)
        2. Looks up the required permission for the HTTP method
        3. Checks if the user has that permission
        4. Returns (user, None) if permission granted
        5. Returns (None, error_response) with 403 if denied

        REQUIRED_PERMISSIONS["GET"] = "documents:read" means:
        - Only users with "documents:read" permission can access GET endpoints
        """
        # Check permission for GET method
        user, err = self._ensure_permission(handler, "GET")
        if err:
            # Returns 401 if not authenticated, 403 if no permission
            return err

        if path == "/api/v1/documents":
            return self._list_documents(query_params)

        if path == "/api/v1/documents/sensitive":
            # This endpoint requires additional permission
            return self._get_sensitive_documents(handler)

        # Handle document ID paths: /api/v1/documents/{id}
        if path.startswith("/api/v1/documents/"):
            doc_id = path.split("/")[-1]
            return self._get_document(doc_id)

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle POST requests with write permission."""
        # Check permission for POST method (documents:write)
        user, err = self._ensure_permission(handler, "POST")
        if err:
            return err

        if path == "/api/v1/documents":
            return self._create_document(handler)

        return None

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle DELETE requests with delete permission."""
        # Check permission for DELETE method (documents:delete)
        user, err = self._ensure_permission(handler, "DELETE")
        if err:
            return err

        if path.startswith("/api/v1/documents/"):
            doc_id = path.split("/")[-1]
            return self._delete_document(doc_id)

        return None

    def _list_documents(self, query_params: dict[str, Any]) -> HandlerResult:
        """List all documents the user can access."""
        user = self.current_user
        limit = int(query_params.get("limit", 20))

        return json_response(
            {
                "documents": [
                    {"id": "doc1", "title": "Document 1"},
                    {"id": "doc2", "title": "Document 2"},
                ],
                "total": 2,
                "limit": limit,
                "user_id": user.user_id if user else None,
            }
        )

    def _get_document(self, doc_id: str) -> HandlerResult:
        """Get a specific document."""
        return json_response(
            {
                "id": doc_id,
                "title": f"Document {doc_id}",
                "content": "Document content...",
            }
        )

    def _get_sensitive_documents(self, handler: HTTPRequestHandler) -> HandlerResult:
        """
        Get sensitive documents - requires additional permission.

        This demonstrates _check_custom_permission() for endpoints that
        need permissions beyond the standard method mapping.
        """
        # Check for additional 'documents:sensitive' permission
        user, err = self._check_custom_permission(handler, "documents:sensitive")
        if err:
            return err

        return json_response(
            {
                "documents": [
                    {"id": "sensitive1", "title": "Confidential Report"},
                ],
                "classification": "confidential",
            }
        )

    def _create_document(self, handler: HTTPRequestHandler) -> HandlerResult:
        """Create a new document."""
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        title = body.get("title", "Untitled")

        return json_response(
            {
                "id": "new_doc_123",
                "title": title,
                "created": True,
            },
            status=201,
        )

    def _delete_document(self, doc_id: str) -> HandlerResult:
        """Delete a document."""
        user = self.current_user

        logger.info(f"User {user.user_id} deleted document {doc_id}")

        return json_response(
            {
                "deleted": True,
                "id": doc_id,
            }
        )
