"""
Request logging and metrics for the unified server.

This module provides the RequestLoggingMixin class with methods for:
- Request logging with timing and status (_log_request)
- Client IP extraction with proxy support (_get_client_ip)
- Endpoint normalization for metrics (_normalize_endpoint)

These methods are extracted from UnifiedHandler to improve modularity
and allow easier testing of logging logic.
"""

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Trusted proxies for X-Forwarded-For header validation
TRUSTED_PROXIES: frozenset[str] = frozenset(
    p.strip() for p in os.getenv("ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
)


class RequestLoggingMixin:
    """Mixin providing request logging and metrics methods.

    This mixin expects the following from the parent class:
    - client_address: Tuple of (ip, port) from the client connection
    - headers: HTTP headers dict (for X-Forwarded-For)

    Logging behavior can be customized via class attributes:
    - _request_log_enabled: Enable/disable request logging (default: True)
    - _slow_request_threshold_ms: Threshold for slow request warnings (default: 1000)
    """

    # Type stubs for attributes expected from parent class
    client_address: tuple[str, int]
    headers: Any

    # Request logging configuration
    _request_log_enabled: bool = True
    _slow_request_threshold_ms: int = 1000

    def _log_request(
        self,
        method: str,
        path: str,
        status: int,
        duration_ms: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log request details for observability.

        Logs request information including method, path, status, duration,
        and client IP. Slow requests (above threshold) are logged as warnings.
        Errors (5xx) are logged as errors, client errors (4xx) as warnings.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            status: HTTP status code
            duration_ms: Request duration in milliseconds
            extra: Additional context to log (key=value pairs)
        """
        if not self._request_log_enabled:
            return

        # Determine log level based on status and duration
        if status >= 500:
            log_fn = logger.error
        elif status >= 400:
            log_fn = logger.warning
        elif duration_ms > self._slow_request_threshold_ms:
            log_fn = logger.warning
        else:
            log_fn = logger.info

        # Build log message
        client_ip = self._get_client_ip()
        msg_parts = [
            f"{method} {path}",
            f"status={status}",
            f"duration={duration_ms:.1f}ms",
            f"ip={client_ip}",
        ]

        if extra:
            for k, v in extra.items():
                msg_parts.append(f"{k}={v}")

        if duration_ms > self._slow_request_threshold_ms:
            msg_parts.append("SLOW")

        log_fn(f"[request] {' '.join(msg_parts)}")

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize API endpoint path for metrics by replacing dynamic IDs.

        Replaces UUIDs, numeric IDs, and other dynamic segments with placeholders
        to avoid high cardinality in Prometheus metrics.

        Examples:
            /api/debates/abc123/messages -> /api/debates/{id}/messages
            /api/users/550e8400-e29b-41d4-a716-446655440000 -> /api/users/{id}
            /api/items/12345 -> /api/items/{id}

        Args:
            path: Raw request path (e.g., "/api/debates/abc123/messages")

        Returns:
            Normalized path (e.g., "/api/debates/{id}/messages")
        """
        # UUID pattern (e.g., 550e8400-e29b-41d4-a716-446655440000)
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        # Short ID pattern (alphanumeric, 8-32 chars, likely an ID)
        short_id_pattern = r"/[a-zA-Z0-9]{8,32}(?=/|$)"
        # Numeric ID pattern
        numeric_pattern = r"/\d+(?=/|$)"

        normalized = path
        # Replace UUIDs first (most specific)
        normalized = re.sub(uuid_pattern, "{id}", normalized)
        # Replace numeric IDs
        normalized = re.sub(numeric_pattern, "/{id}", normalized)
        # Replace remaining short alphanumeric IDs in path segments
        # Only if they're surrounded by slashes or at end
        normalized = re.sub(short_id_pattern, "/{id}", normalized)

        return normalized

    def _get_client_ip(self) -> str:
        """Get client IP address, respecting trusted proxy headers.

        If the direct connection is from a trusted proxy (e.g., load balancer),
        extracts the real client IP from X-Forwarded-For header.

        Returns:
            Client IP address string
        """
        remote_ip = self.client_address[0] if hasattr(self, "client_address") else "unknown"
        client_ip = remote_ip
        if remote_ip in TRUSTED_PROXIES:
            forwarded = self.headers.get("X-Forwarded-For", "")
            if forwarded:
                first_ip = forwarded.split(",")[0].strip()
                if first_ip:
                    client_ip = first_ip
        return client_ip


__all__ = ["RequestLoggingMixin", "TRUSTED_PROXIES"]
