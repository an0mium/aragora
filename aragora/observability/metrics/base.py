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

    def labels(self, *args: Any, **kwargs: Any) -> NoOpMetric:
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


def _get_collector_by_name(name: str) -> Any:
    """Look up an existing Prometheus collector by metric name.

    When a metric is already registered in the default REGISTRY, creating
    it again raises ``ValueError: Duplicated timeseries``.  This helper
    retrieves the existing collector so we can reuse it instead.

    Args:
        name: The Prometheus metric name (e.g. ``aragora_cache_hits_total``)

    Returns:
        The existing collector, or ``None`` if not found.
    """
    try:
        from prometheus_client import REGISTRY

        return REGISTRY._names_to_collectors.get(name)
    except (ImportError, AttributeError):
        return None


def get_or_create_counter(
    name: str, documentation: str, labelnames: list[str] | None = None
) -> Any:
    """Create a Prometheus Counter, reusing an existing one on conflict.

    Falls back to :class:`NoOpMetric` when ``prometheus_client`` is missing.
    """
    try:
        from prometheus_client import Counter

        if labelnames:
            return Counter(name, documentation, labelnames)
        return Counter(name, documentation)
    except ImportError:
        return NoOpMetric()
    except ValueError:
        existing = _get_collector_by_name(name)
        if existing is not None:
            return existing
        return NoOpMetric()


def get_or_create_gauge(name: str, documentation: str, labelnames: list[str] | None = None) -> Any:
    """Create a Prometheus Gauge, reusing an existing one on conflict.

    Falls back to :class:`NoOpMetric` when ``prometheus_client`` is missing.
    """
    try:
        from prometheus_client import Gauge

        if labelnames:
            return Gauge(name, documentation, labelnames)
        return Gauge(name, documentation)
    except ImportError:
        return NoOpMetric()
    except ValueError:
        existing = _get_collector_by_name(name)
        if existing is not None:
            return existing
        return NoOpMetric()


def get_or_create_histogram(
    name: str,
    documentation: str,
    labelnames: list[str] | None = None,
    buckets: list[float] | None = None,
) -> Any:
    """Create a Prometheus Histogram, reusing an existing one on conflict.

    Falls back to :class:`NoOpMetric` when ``prometheus_client`` is missing.
    """
    try:
        from prometheus_client import Histogram

        kwargs: dict[str, Any] = {}
        if labelnames:
            kwargs["labelnames"] = labelnames
        if buckets:
            kwargs["buckets"] = buckets
        return Histogram(name, documentation, **kwargs)
    except ImportError:
        return NoOpMetric()
    except ValueError:
        existing = _get_collector_by_name(name)
        if existing is not None:
            return existing
        return NoOpMetric()


def get_or_create_info(name: str, documentation: str) -> Any:
    """Create a Prometheus Info metric, reusing an existing one on conflict.

    Falls back to :class:`NoOpMetric` when ``prometheus_client`` is missing.
    """
    try:
        from prometheus_client import Info

        return Info(name, documentation)
    except ImportError:
        return NoOpMetric()
    except ValueError:
        existing = _get_collector_by_name(name)
        if existing is not None:
            return existing
        return NoOpMetric()


def ensure_metrics_initialized() -> bool:
    """Ensure metrics are initialized.

    Returns:
        True if metrics are available (Prometheus enabled and imported)
    """
    global _initialized

    if _initialized:
        return get_metrics_enabled()

    # Import and init all submodules
    # Use _aragora_metrics_impl which is registered by __init__.py
    from _aragora_metrics_impl import init_core_metrics
    from aragora.observability.metrics.bridge import init_bridge_metrics
    from aragora.observability.metrics.km import init_km_metrics
    from aragora.observability.metrics.notification import init_notification_metrics
    from aragora.observability.metrics.stores import init_store_metrics
    from aragora.observability.metrics.gauntlet import init_gauntlet_metrics
    from aragora.observability.metrics.slo import init_slo_metrics
    from aragora.observability.metrics.security import init_security_metrics

    enabled = get_metrics_enabled()
    if enabled:
        try:
            init_core_metrics()
            init_bridge_metrics()
            init_km_metrics()
            init_notification_metrics()
            init_store_metrics()
            init_gauntlet_metrics()
            init_slo_metrics()
            init_security_metrics()
            logger.info("All metrics submodules initialized")
        except ImportError as e:
            logger.warning("prometheus-client not installed: %s", e)
            enabled = False
        except (RuntimeError, TypeError, ValueError) as e:
            logger.error("Failed to initialize metrics: %s", e)
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
