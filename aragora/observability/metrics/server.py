"""
Prometheus metrics server management.

Provides functionality for starting and stopping the Prometheus metrics HTTP server.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.metrics.base import get_metrics_enabled

logger = logging.getLogger(__name__)

# Server state
_metrics_server: Any = None


def start_metrics_server(port: int = 9090) -> bool:
    """Start a metrics server on the specified port.

    Args:
        port: Port to listen on (default: 9090)

    Returns:
        True if server started successfully, False otherwise
    """
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
        logger.info(f"Metrics server started on port {port}")
        return True
    except ImportError as e:
        logger.error(
            "Failed to start metrics server: prometheus-client not installed",
            extra={"error": str(e)},
        )
        return False
    except OSError as e:
        # Port already in use or permission denied
        logger.error(
            "Failed to start metrics server due to OS error",
            extra={"port": port, "error_type": type(e).__name__, "error": str(e)},
        )
        return False
    except (ValueError, TypeError, RuntimeError) as e:
        # Invalid configuration or runtime issues
        logger.error(
            "Failed to start metrics server due to configuration error",
            extra={"port": port, "error_type": type(e).__name__, "error": str(e)},
        )
        return False


def stop_metrics_server() -> bool:
    """Stop the Prometheus metrics server.

    Note: prometheus_client's start_http_server() creates a daemon thread
    that cannot be cleanly stopped. This function marks the server as
    stopped for tracking purposes; the actual thread terminates with
    process exit.

    Returns:
        True if server was marked as stopped, False if not running.
    """
    global _metrics_server

    if _metrics_server is None:
        return False

    port = _metrics_server
    _metrics_server = None
    logger.info(f"Metrics server on port {port} marked for shutdown")
    return True


def is_metrics_server_running() -> bool:
    """Check if the metrics server is running.

    Returns:
        True if server is running, False otherwise
    """
    return _metrics_server is not None


def get_metrics_server_port() -> int | None:
    """Get the port the metrics server is running on.

    Returns:
        Port number if running, None otherwise
    """
    return _metrics_server


__all__ = [
    "start_metrics_server",
    "stop_metrics_server",
    "is_metrics_server_running",
    "get_metrics_server_port",
]
