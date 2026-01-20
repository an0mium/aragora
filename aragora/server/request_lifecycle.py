"""
Request lifecycle management for HTTP handlers.

Centralizes the common request handling pattern:
- State reset
- Timing and tracing
- Error handling
- Metrics and logging

This eliminates repetition across do_GET, do_POST, do_PUT, etc.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from aragora.server.tracing_service import TracingService

logger = logging.getLogger(__name__)


class RequestLifecycleManager:
    """Manages the lifecycle of HTTP requests.

    Handles the common boilerplate for all HTTP methods:
    - State reset
    - Timing measurement
    - Distributed tracing
    - Error handling with proper span recording
    - Prometheus metrics recording
    - Request logging

    Usage:
        lifecycle = RequestLifecycleManager(handler, tracing)
        lifecycle.handle_request(
            method="GET",
            internal_handler=self._do_GET_internal,
            path_arg=path,
            query_arg=query,
        )
    """

    def __init__(
        self,
        handler: Any,  # UnifiedHandler instance
        tracing: Optional["TracingService"] = None,
        record_metrics_fn: Optional[Callable[[str, str, int, float], None]] = None,
        log_request_fn: Optional[Callable[[str, str, int, float], None]] = None,
        normalize_endpoint_fn: Optional[Callable[[str], str]] = None,
    ):
        """Initialize the lifecycle manager.

        Args:
            handler: The HTTP handler instance (for sending responses)
            tracing: TracingService for distributed tracing
            record_metrics_fn: Function to record Prometheus metrics
            log_request_fn: Function to log request details
            normalize_endpoint_fn: Function to normalize endpoint paths
        """
        self.handler = handler
        self.tracing = tracing
        self._record_metrics = record_metrics_fn
        self._log_request = log_request_fn
        self._normalize_endpoint = normalize_endpoint_fn

    def handle_request(
        self,
        method: str,
        internal_handler: Callable,
        *,
        with_query: bool = False,
        trace_api_only: bool = True,
        record_api_metrics_only: bool = True,
    ) -> None:
        """Execute the request lifecycle.

        Args:
            method: HTTP method (GET, POST, etc.)
            internal_handler: The _do_METHOD_internal function to call
            with_query: Whether to parse and pass query parameters
            trace_api_only: Only start tracing for /api/* paths
            record_api_metrics_only: Only record metrics for /api/* paths
        """
        # Reset per-request state
        self.handler._rate_limit_result = None
        self.handler._response_status = 200

        start_time = time.time()
        parsed = urlparse(self.handler.path)
        path = parsed.path
        query = parse_qs(parsed.query) if with_query else {}

        is_api_request = path.startswith("/api/")

        # Start tracing span for API requests
        span = None
        if self.tracing and (is_api_request or not trace_api_only):
            span = self.tracing.start_request_span(method, path, dict(self.handler.headers))

        try:
            # Call the internal handler
            if with_query:
                internal_handler(path, query)
            else:
                internal_handler(path)

        except Exception as e:
            # Top-level safety net for handlers
            self.handler._response_status = 500
            logger.exception(f"[request] Unhandled exception in {method} {path}: {e}")
            if span:
                span.set_error(e)
            self._send_error_response()

        finally:
            status_code = getattr(self.handler, "_response_status", 200)
            duration_seconds = time.time() - start_time

            # Finish tracing span
            if span and self.tracing:
                self.tracing.finish_request_span(span, status_code)

            # Record Prometheus metrics
            if self._record_metrics and (is_api_request or not record_api_metrics_only):
                endpoint = path
                if self._normalize_endpoint:
                    endpoint = self._normalize_endpoint(path)
                self._record_metrics(method, endpoint, status_code, duration_seconds)

            # Log the request
            if self._log_request:
                duration_ms = duration_seconds * 1000
                self._log_request(method, path, status_code, duration_ms)

    def _send_error_response(self) -> None:
        """Send a JSON error response if possible."""
        try:
            self.handler._send_json({"error": "Internal server error"}, status=500)
        except Exception as send_err:
            logger.debug(f"Could not send error response (already sent?): {send_err}")


def create_lifecycle_manager(handler: Any) -> RequestLifecycleManager:
    """Factory function to create a lifecycle manager from a handler.

    This convenience function extracts the necessary components from the handler
    and creates a properly configured lifecycle manager.

    Args:
        handler: UnifiedHandler instance

    Returns:
        Configured RequestLifecycleManager
    """
    from aragora.server.prometheus import record_http_request

    return RequestLifecycleManager(
        handler=handler,
        tracing=getattr(handler, "tracing", None),
        record_metrics_fn=record_http_request,
        log_request_fn=getattr(handler, "_log_request", None),
        normalize_endpoint_fn=getattr(handler, "_normalize_endpoint", None),
    )
