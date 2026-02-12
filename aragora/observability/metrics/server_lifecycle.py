"""
Metrics server lifecycle management.

Extracted from metrics/__init__.py for maintainability.
Provides functions to start, stop, and query the Prometheus metrics HTTP server.
"""

from __future__ import annotations

import logging

from aragora.observability.metrics.base import get_metrics_enabled

logger = logging.getLogger(__name__)

_metrics_server: int | None = None


def start_metrics_server(port: int = 9090) -> bool:
    """Start the Prometheus metrics HTTP server."""
    global _metrics_server
    if not get_metrics_enabled():
        logger.warning("Metrics not enabled, server not started")
        return False
    if _metrics_server is not None:
        logger.warning("Metrics server already running")
        return True
    try:
        from prometheus_client import start_http_server

        start_http_server(port)
        _metrics_server = port
        try:
            from aragora.observability.metrics import server as _server

            _server._metrics_server = port
        except Exception as e:
            logger.debug("Could not sync metrics server state to submodule: %s", e)
        logger.info("Metrics server started on port %s", port)
        return True
    except ImportError as e:
        logger.error(
            "Failed to start metrics server: prometheus-client not installed",
            extra={"error": str(e)},
        )
        return False
    except OSError as e:
        logger.error(
            "Failed to start metrics server due to OS error",
            extra={"port": port, "error_type": type(e).__name__, "error": str(e)},
        )
        return False
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(
            "Failed to start metrics server due to configuration error",
            extra={"port": port, "error_type": type(e).__name__, "error": str(e)},
        )
        return False


def stop_metrics_server() -> bool:
    """Stop the Prometheus metrics server (marks as stopped)."""
    global _metrics_server
    if _metrics_server is None:
        return False
    port = _metrics_server
    _metrics_server = None
    try:
        from aragora.observability.metrics import server as _server

        _server._metrics_server = None
    except Exception as e:
        logger.debug("Could not sync metrics server shutdown to submodule: %s", e)
    logger.info("Metrics server on port %s marked for shutdown", port)
    return True


def is_metrics_server_running() -> bool:
    """Check if the metrics server is running."""
    return _metrics_server is not None


def get_metrics_server_port() -> int | None:
    """Get the metrics server port if running."""
    return _metrics_server
