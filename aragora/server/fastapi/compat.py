"""
Compatibility Layer for Legacy Handler Migration.

Provides utilities to bridge the legacy BaseHandler (BaseHTTPRequestHandler-based)
and new FastAPI router patterns during the incremental migration.

Architecture:
    Legacy: BaseHTTPRequestHandler -> HandlerRegistryMixin -> BaseHandler.handle()
    New:    uvicorn -> FastAPI app -> APIRouter -> async endpoint functions

This module supports TWO migration strategies:

Strategy A: Wrap a legacy BaseHandler as a FastAPI router (fast path)
    Use ``legacy_handler_to_router()`` to wrap an existing handler class
    into a FastAPI router without rewriting it. The wrapper translates
    FastAPI Request objects into the handler/query_params interface that
    BaseHandler expects, and converts HandlerResult back to FastAPI responses.
    This is useful for handlers that are too large to rewrite immediately.

Strategy B: Write native FastAPI routes (clean path)
    Write new routes as shown in ``aragora/server/fastapi/routes/consensus.py``.
    This is the recommended approach for new development and for handlers
    selected for full migration (like the top-5 critical handlers).

Dual Registration:
    During migration, both legacy and FastAPI handlers remain active.
    The legacy server (ThreadingHTTPServer on port 8080) serves all
    existing ``/api/`` and ``/api/v1/`` routes via HandlerRegistryMixin.
    The FastAPI server (uvicorn on port 8081) serves ``/api/v2/`` routes.
    Clients can migrate incrementally from v1 to v2.

    Once a handler is fully migrated and v1 clients have moved, the legacy
    handler can be removed and the FastAPI route can serve both v1 and v2
    via prefix aliasing.

Usage:
    # Strategy A: Quick wrap of a legacy handler
    from aragora.server.fastapi.compat import legacy_handler_to_router

    router = legacy_handler_to_router(
        ConsensusHandler,
        prefix="/api/v2/consensus",
        tags=["Consensus"],
    )
    app.include_router(router)

    # Strategy B: Native FastAPI route (recommended)
    from fastapi import APIRouter
    router = APIRouter(prefix="/api/v2/consensus", tags=["Consensus"])

    @router.get("/stats")
    async def get_stats(): ...
"""

from __future__ import annotations

import asyncio
import json
import logging
from io import BytesIO
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class _FakeHandler:
    """Minimal shim that looks like a BaseHTTPRequestHandler to legacy handlers.

    Legacy BaseHandler methods access ``handler.headers``, ``handler.rfile``,
    ``handler.command``, and ``handler.path``. This shim translates a FastAPI
    Request into those attributes so existing handler code runs unmodified.
    """

    def __init__(self, request: Request, body: bytes | None = None) -> None:
        self.headers = dict(request.headers)
        self.command = request.method
        self.path = str(request.url.path)
        if request.url.query:
            self.path += f"?{request.url.query}"
        self._body = body or b""
        self.rfile = BytesIO(self._body)

        # Some handlers check for Content-Length via .headers.get()
        # which works since we stored as a plain dict above.
        # Provide a .get() method that falls back for case-insensitive lookup.
        self._header_dict = {k.lower(): v for k, v in request.headers.items()}

    def get_header(self, name: str, default: str | None = None) -> str | None:
        return self._header_dict.get(name.lower(), default)


def _handler_result_to_response(result: Any) -> Response:
    """Convert a legacy HandlerResult to a FastAPI Response.

    Handles both HandlerResult dataclass and plain dict returns.
    """
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Not found"},
        )

    # HandlerResult dataclass
    if hasattr(result, "status_code") and hasattr(result, "body"):
        content_type = getattr(result, "content_type", "application/json")
        headers = getattr(result, "headers", {}) or {}
        body = result.body

        if isinstance(body, bytes):
            return Response(
                content=body,
                status_code=result.status_code,
                media_type=content_type,
                headers=headers,
            )
        return Response(
            content=str(body).encode("utf-8"),
            status_code=result.status_code,
            media_type=content_type,
            headers=headers,
        )

    # Plain dict return
    if isinstance(result, dict):
        status = result.get("status", 200)
        body = result.get("body", result)
        return JSONResponse(content=body, status_code=status)

    # Fallback
    return JSONResponse(content={"data": str(result)}, status_code=200)


def legacy_handler_to_router(
    handler_class: type,
    prefix: str,
    tags: list[str] | None = None,
    ctx: dict[str, Any] | None = None,
) -> APIRouter:
    """Wrap a legacy BaseHandler class as a FastAPI APIRouter.

    Creates a catch-all router that delegates to the legacy handler's
    ``handle()``, ``handle_post()``, ``handle_delete()``, ``handle_patch()``,
    and ``handle_put()`` methods.

    Args:
        handler_class: A BaseHandler subclass (e.g., ConsensusHandler).
        prefix: The URL prefix for the router (e.g., "/api/v2/consensus").
        tags: OpenAPI tags for documentation.
        ctx: Server context dict to pass to the handler constructor.

    Returns:
        A FastAPI APIRouter that delegates to the legacy handler.

    Notes:
        - This wrapper does NOT generate Pydantic models or OpenAPI schemas.
          The auto-generated docs will show a generic JSON response.
          For proper OpenAPI docs, write native FastAPI routes (Strategy B).
        - Rate limiting and caching from the legacy handler still apply since
          the handler code runs unmodified.
        - Auth is handled by the legacy handler's own auth checks.
    """
    router = APIRouter(prefix=prefix, tags=tags or [])
    handler_instance = handler_class(ctx=ctx or {})

    async def _dispatch(request: Request, method: str) -> Response:
        """Common dispatch logic for all HTTP methods."""
        # Read body for methods that have one
        body = None
        if method in ("POST", "PUT", "PATCH"):
            body = await request.body()

        # Build fake handler
        fake = _FakeHandler(request, body)

        # Parse query params
        query_str = str(request.url.query) if request.url.query else ""
        query_params = {}
        if query_str:
            parsed = parse_qs(query_str)
            query_params = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}

        # Determine legacy path (strip the v2 prefix to match legacy handler expectations)
        path = str(request.url.path)

        # Dispatch to appropriate handler method
        result = None
        try:
            if method == "GET":
                result = handler_instance.handle(path, query_params, fake)
            elif method == "POST":
                result = handler_instance.handle_post(path, query_params, fake)
            elif method == "DELETE":
                result = handler_instance.handle_delete(path, query_params, fake)
            elif method == "PATCH":
                result = handler_instance.handle_patch(path, query_params, fake)
            elif method == "PUT":
                result = handler_instance.handle_put(path, query_params, fake)

            # Handle async results
            if asyncio.iscoroutine(result):
                result = await result

        except Exception as e:
            logger.exception("Legacy handler error: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

        return _handler_result_to_response(result)

    @router.api_route("/{path:path}", methods=["GET"], include_in_schema=False)
    async def handle_get(request: Request) -> Response:
        return await _dispatch(request, "GET")

    @router.api_route("/{path:path}", methods=["POST"], include_in_schema=False)
    async def handle_post(request: Request) -> Response:
        return await _dispatch(request, "POST")

    @router.api_route("/{path:path}", methods=["DELETE"], include_in_schema=False)
    async def handle_delete(request: Request) -> Response:
        return await _dispatch(request, "DELETE")

    @router.api_route("/{path:path}", methods=["PATCH"], include_in_schema=False)
    async def handle_patch(request: Request) -> Response:
        return await _dispatch(request, "PATCH")

    @router.api_route("/{path:path}", methods=["PUT"], include_in_schema=False)
    async def handle_put(request: Request) -> Response:
        return await _dispatch(request, "PUT")

    return router
