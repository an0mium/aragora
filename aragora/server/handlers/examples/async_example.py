"""
Example: AsyncTypedHandler usage.

This example shows how to use AsyncTypedHandler for handlers that need
async operations (database queries, external API calls, etc.).

Key features demonstrated:
- Async handler methods
- Proper async/await patterns
- Integration with async dependencies
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aragora.protocols import HTTPRequestHandler

from ..base import (
    AsyncTypedHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
)
from ..utils.decorators import require_permission

logger = logging.getLogger(__name__)


class ExampleAsyncHandler(AsyncTypedHandler):
    """
    Example async handler for I/O-bound operations.

    AsyncTypedHandler provides:
    - Async versions of all handle methods (handle, handle_post, etc.)
    - Proper integration with asyncio
    - Support for awaiting database queries, API calls, etc.

    Use this when your handler needs to:
    - Query async databases (aiohttp, asyncpg, etc.)
    - Make async HTTP requests
    - Wait for I/O operations
    """

    ROUTES = [
        "/api/v1/async-example",
        "/api/v1/async-example/slow",
        "/api/v1/async-example/parallel",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """
        Handle GET requests asynchronously.

        All handler methods in AsyncTypedHandler are async and can
        use await for I/O operations.
        """
        if path == "/api/v1/async-example":
            return await self._get_async_data()

        if path == "/api/v1/async-example/slow":
            return await self._get_slow_data(query_params)

        if path == "/api/v1/async-example/parallel":
            return await self._get_parallel_data()

        return None

    @handle_errors("example async creation")
    @require_permission("debates:write")
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle POST requests asynchronously."""
        if path == "/api/v1/async-example":
            return await self._create_async_data(handler)
        return None

    async def _get_async_data(self) -> HandlerResult:
        """
        Fetch data using async operations.

        In a real handler, this would query a database or external API.
        """
        # Simulate async database query
        await asyncio.sleep(0.01)  # 10ms simulated latency

        return json_response(
            {
                "message": "Data fetched asynchronously",
                "data": {
                    "items": [1, 2, 3],
                    "count": 3,
                },
            }
        )

    async def _get_slow_data(self, query_params: dict[str, Any]) -> HandlerResult:
        """
        Demonstrate handling of slow operations.

        The 'delay' query param controls simulated latency.
        """
        delay = float(query_params.get("delay", "0.5"))
        # Cap delay at 5 seconds for safety
        delay = min(delay, 5.0)

        # Simulate slow operation
        await asyncio.sleep(delay)

        return json_response(
            {
                "message": f"Completed after {delay}s delay",
                "delay_seconds": delay,
            }
        )

    async def _get_parallel_data(self) -> HandlerResult:
        """
        Demonstrate parallel async operations.

        Uses asyncio.gather() to run multiple operations concurrently.
        """

        async def fetch_source_a():
            await asyncio.sleep(0.1)
            return {"source": "A", "value": 100}

        async def fetch_source_b():
            await asyncio.sleep(0.1)
            return {"source": "B", "value": 200}

        async def fetch_source_c():
            await asyncio.sleep(0.1)
            return {"source": "C", "value": 300}

        # Run all three fetches in parallel
        # Total time ~100ms instead of 300ms sequential
        results = await asyncio.gather(
            fetch_source_a(),
            fetch_source_b(),
            fetch_source_c(),
        )

        return json_response(
            {
                "message": "Parallel fetch completed",
                "results": results,
                "total": sum(r["value"] for r in results),
            }
        )

    async def _create_async_data(self, handler: HTTPRequestHandler) -> HandlerResult:
        """Create data using async operations."""
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Simulate async database insert
        await asyncio.sleep(0.01)

        return json_response(
            {
                "created": True,
                "id": "new-item-123",
                "data": body,
            },
            status=201,
        )


class AsyncAuthenticatedExample(AsyncTypedHandler):
    """
    Example combining async with authentication.

    Shows how to use authentication checking in async handlers.
    """

    ROUTES = ["/api/v1/async-auth"]

    def can_handle(self, path: str) -> bool:
        return path in self.ROUTES

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle authenticated async requests."""
        # Note: require_auth_or_error is sync, but can be used in async context
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Now do async operations with authenticated user
        data = await self._fetch_user_data(user.user_id)

        return json_response(
            {
                "user_id": user.user_id,
                "data": data,
            }
        )

    async def _fetch_user_data(self, user_id: str) -> dict[str, Any]:
        """Fetch user-specific data asynchronously."""
        # Simulate async database query
        await asyncio.sleep(0.01)
        return {
            "preferences": {"theme": "dark"},
            "recent_activity": ["login", "view_dashboard"],
        }
