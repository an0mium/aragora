"""
Example: Basic TypedHandler usage.

This example shows how to use TypedHandler for handlers that need
proper type annotations for better IDE support and static analysis.

Key features demonstrated:
- Explicit HTTPRequestHandler type for handler parameters
- Type-safe access to handler.headers, handler.path, etc.
- Proper return type annotations
- Use of helper methods with full type checking
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.protocols import HTTPRequestHandler

from ..base import (
    HandlerResult,
    TypedHandler,
    json_response,
    error_response,
)
from ..utils.decorators import require_permission

logger = logging.getLogger(__name__)


class ExampleTypedHandler(TypedHandler):
    """
    Example handler demonstrating TypedHandler usage.

    TypedHandler provides:
    - Explicit type annotations for all handler methods
    - Better IDE autocomplete (handler.headers is typed)
    - Static analysis support (mypy, pyright)
    - Dependency injection for testing

    This example implements a simple echo/info endpoint.
    """

    # Define routes this handler manages
    ROUTES = [
        "/api/v1/example",
        "/api/v1/example/info",
        "/api/v1/example/echo",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES or path.startswith("/api/v1/example/")

    def handle(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """
        Handle GET requests with proper type annotations.

        The `handler` parameter is now typed as HTTPRequestHandler, providing:
        - handler.headers: HTTPHeaders with typed get() method
        - handler.path: str
        - handler.command: str (HTTP method)
        - handler.client_address: tuple[str, int]

        Args:
            path: The request path (e.g., "/api/v1/example/info")
            query_params: Parsed query parameters as dict
            handler: HTTP request handler with typed access

        Returns:
            HandlerResult if handled, None if not handled
        """
        # Type-safe access to headers
        user_agent = handler.headers.get("User-Agent", "Unknown")
        accept = handler.headers.get("Accept", "*/*")

        if path == "/api/v1/example/info":
            return self._get_info(handler, user_agent, accept)

        if path == "/api/v1/example":
            return json_response(
                {
                    "message": "Example typed handler",
                    "endpoints": self.ROUTES,
                }
            )

        return None

    @require_permission("debates:write")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle POST requests with type safety."""
        if path == "/api/v1/example/echo":
            return self._handle_echo(handler)
        return None

    def _get_info(
        self,
        handler: HTTPRequestHandler,
        user_agent: str,
        accept: str,
    ) -> HandlerResult:
        """
        Return server info with request details.

        This method demonstrates type-safe access to handler properties.
        """
        # All these accesses are type-checked
        client_ip, client_port = handler.client_address
        request_path = handler.path
        method = handler.command

        return json_response(
            {
                "server": "Aragora API",
                "version": "1.0",
                "request": {
                    "path": request_path,
                    "method": method,
                    "client_ip": client_ip,
                    "client_port": client_port,
                    "user_agent": user_agent,
                    "accept": accept,
                },
            }
        )

    def _handle_echo(self, handler: HTTPRequestHandler) -> HandlerResult:
        """Echo back the request body."""
        # Use the type-safe read_json_body method from BaseHandler
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        return json_response(
            {
                "echoed": body,
                "method": handler.command,
                "content_type": handler.headers.get("Content-Type"),
            }
        )
