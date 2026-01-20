"""
Base utilities for metrics module.

Provides shared functionality used across all metric submodules:
- NoOpMetric class for when metrics are disabled
- Initialization helpers
- Configuration access
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.config import get_metrics_config

logger = logging.getLogger(__name__)

# Global initialization state
_initialized = False


class NoOpMetric:
    """No-op metric implementation for when Prometheus is disabled.

    Provides all standard metric interface methods (labels, inc, dec, set, observe)
    that silently do nothing.
    """

    def labels(self, *args: Any, **kwargs: Any) -> "NoOpMetric":
        """Return self for method chaining."""
        return self

    def inc(self, amount: float = 1) -> None:
        """No-op increment."""
        pass

    def dec(self, amount: float = 1) -> None:
        """No-op decrement."""
        pass

    def set(self, value: float) -> None:
        """No-op set."""
        pass

    def observe(self, value: float) -> None:
        """No-op observe."""
        pass


def get_metrics_enabled() -> bool:
    """Check if metrics are enabled.

    Returns:
        True if metrics are enabled in config
    """
    config = get_metrics_config()
    return config.enabled


def ensure_metrics_initialized() -> bool:
    """Ensure metrics are initialized.

    Returns:
        True if metrics are available (Prometheus enabled and imported)
    """
    global _initialized

    if _initialized:
        return get_metrics_enabled()

    # Import and init all submodules
    from aragora.observability.metrics.core import init_core_metrics
    from aragora.observability.metrics.bridge import init_bridge_metrics
    from aragora.observability.metrics.km import init_km_metrics
    from aragora.observability.metrics.notification import init_notification_metrics
    from aragora.observability.metrics.stores import init_store_metrics
    from aragora.observability.metrics.gauntlet import init_gauntlet_metrics

    enabled = get_metrics_enabled()
    if enabled:
        try:
            init_core_metrics()
            init_bridge_metrics()
            init_km_metrics()
            init_notification_metrics()
            init_store_metrics()
            init_gauntlet_metrics()
            logger.info("All metrics submodules initialized")
        except ImportError as e:
            logger.warning(f"prometheus-client not installed: {e}")
            enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            enabled = False

    _initialized = True
    return enabled


def normalize_endpoint(endpoint: str) -> str:
    """Normalize endpoint path to control cardinality.

    Replaces dynamic path segments (IDs, UUIDs) with placeholders.

    Args:
        endpoint: Raw endpoint path

    Returns:
        Normalized endpoint path
    """
    import re

    # Replace UUIDs
    endpoint = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        ":id",
        endpoint,
        flags=re.IGNORECASE,
    )

    # Replace numeric IDs
    endpoint = re.sub(r"/\d+", "/:id", endpoint)

    # Replace base64-like tokens
    endpoint = re.sub(r"/[A-Za-z0-9_-]{20,}", "/:token", endpoint)

    return endpoint
