"""
Distributed Tracing Middleware.

Adds trace ID propagation and request timing for observability.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for distributed tracing.

    - Generates or propagates trace IDs
    - Records request timing
    - Adds trace headers to responses
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate trace ID
        trace_id = request.headers.get("X-Trace-ID")
        if not trace_id:
            trace_id = str(uuid.uuid4())[:8]

        # Store trace ID on request state
        request.state.trace_id = trace_id
        request.state.start_time = time.perf_counter()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - request.state.start_time) * 1000

        # Add trace headers to response
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response
