"""
Prometheus metrics for Enterprise Connectors.

Provides metrics for monitoring connector sync operations, success rates,
and performance across all enterprise data sources.

Usage:
    from aragora.connectors.metrics import record_sync, record_sync_items

    # Record a sync operation
    record_sync(
        connector_type="sharepoint",
        connector_id="sharepoint_corp",
        status="success",
        duration_seconds=45.2,
        items_synced=1500,
    )

    # Record individual sync items
    record_sync_items(
        connector_type="confluence",
        item_type="page",
        count=250,
    )

Requirements:
    pip install prometheus-client

Environment Variables:
    METRICS_ENABLED: Set to "true" to enable metrics (default: true)

See docs/OBSERVABILITY.md for configuration guide.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Prometheus metrics - initialized lazily
_initialized = False

# Metric instances (will be set during initialization)
CONNECTOR_SYNCS: Any = None
CONNECTOR_SYNC_DURATION: Any = None
CONNECTOR_SYNC_ITEMS: Any = None
CONNECTOR_SYNC_ERRORS: Any = None
CONNECTOR_SYNC_LATENCY: Any = None
CONNECTOR_ACTIVE_SYNCS: Any = None
CONNECTOR_LAST_SYNC: Any = None
CONNECTOR_HEALTH: Any = None
CONNECTOR_RATE_LIMITS: Any = None
CONNECTOR_AUTH_FAILURES: Any = None
CONNECTOR_CACHE_HITS: Any = None
CONNECTOR_CACHE_MISSES: Any = None
CONNECTOR_RETRIES: Any = None


def _init_metrics() -> bool:
    """Initialize Prometheus metrics lazily."""
    global _initialized
    global CONNECTOR_SYNCS, CONNECTOR_SYNC_DURATION, CONNECTOR_SYNC_ITEMS
    global CONNECTOR_SYNC_ERRORS, CONNECTOR_SYNC_LATENCY, CONNECTOR_ACTIVE_SYNCS
    global CONNECTOR_LAST_SYNC, CONNECTOR_HEALTH, CONNECTOR_RATE_LIMITS
    global CONNECTOR_AUTH_FAILURES, CONNECTOR_CACHE_HITS, CONNECTOR_CACHE_MISSES
    global CONNECTOR_RETRIES

    if _initialized:
        return True

    try:
        from aragora.observability.config import get_metrics_config

        config = get_metrics_config()
        if not config.enabled:
            _init_noop_metrics()
            _initialized = True
            return False
    except ImportError:
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Sync operation metrics
        CONNECTOR_SYNCS = Counter(
            "aragora_connector_syncs_total",
            "Total connector sync operations",
            ["connector_type", "connector_id", "status"],
        )

        CONNECTOR_SYNC_DURATION = Histogram(
            "aragora_connector_sync_duration_seconds",
            "Duration of sync operations",
            ["connector_type"],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0],
        )

        CONNECTOR_SYNC_ITEMS = Counter(
            "aragora_connector_sync_items_total",
            "Total items synced",
            ["connector_type", "item_type"],
        )

        CONNECTOR_SYNC_ERRORS = Counter(
            "aragora_connector_sync_errors_total",
            "Total sync errors by error type",
            ["connector_type", "error_type"],
        )

        CONNECTOR_SYNC_LATENCY = Histogram(
            "aragora_connector_item_latency_seconds",
            "Per-item sync latency",
            ["connector_type", "item_type"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        # Active sync tracking
        CONNECTOR_ACTIVE_SYNCS = Gauge(
            "aragora_connector_active_syncs",
            "Number of currently running sync operations",
            ["connector_type"],
        )

        CONNECTOR_LAST_SYNC = Gauge(
            "aragora_connector_last_sync_timestamp",
            "Unix timestamp of last successful sync",
            ["connector_type", "connector_id"],
        )

        # Health metrics
        CONNECTOR_HEALTH = Gauge(
            "aragora_connector_health",
            "Connector health status (1=healthy, 0=unhealthy)",
            ["connector_type", "connector_id"],
        )

        CONNECTOR_RATE_LIMITS = Counter(
            "aragora_connector_rate_limits_total",
            "Number of rate limit responses received",
            ["connector_type"],
        )

        CONNECTOR_AUTH_FAILURES = Counter(
            "aragora_connector_auth_failures_total",
            "Number of authentication failures",
            ["connector_type", "connector_id"],
        )

        CONNECTOR_CACHE_HITS = Counter(
            "aragora_connector_cache_hits_total",
            "Number of cache hits",
            ["connector_type"],
        )

        CONNECTOR_CACHE_MISSES = Counter(
            "aragora_connector_cache_misses_total",
            "Number of cache misses",
            ["connector_type"],
        )

        CONNECTOR_RETRIES = Counter(
            "aragora_connector_retries_total",
            "Number of retry attempts",
            ["connector_type", "reason"],
        )

        _initialized = True
        logger.info("Connector Prometheus metrics initialized")
        return True

    except ImportError:
        logger.warning("prometheus_client not installed, connector metrics disabled")
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global CONNECTOR_SYNCS, CONNECTOR_SYNC_DURATION, CONNECTOR_SYNC_ITEMS
    global CONNECTOR_SYNC_ERRORS, CONNECTOR_SYNC_LATENCY, CONNECTOR_ACTIVE_SYNCS
    global CONNECTOR_LAST_SYNC, CONNECTOR_HEALTH, CONNECTOR_RATE_LIMITS
    global CONNECTOR_AUTH_FAILURES, CONNECTOR_CACHE_HITS, CONNECTOR_CACHE_MISSES
    global CONNECTOR_RETRIES

    class NoopMetric:
        """No-op metric that accepts any method call and returns self for chaining."""

        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: self

    noop = NoopMetric()
    CONNECTOR_SYNCS = noop
    CONNECTOR_SYNC_DURATION = noop
    CONNECTOR_SYNC_ITEMS = noop
    CONNECTOR_SYNC_ERRORS = noop
    CONNECTOR_SYNC_LATENCY = noop
    CONNECTOR_ACTIVE_SYNCS = noop
    CONNECTOR_LAST_SYNC = noop
    CONNECTOR_HEALTH = noop
    CONNECTOR_RATE_LIMITS = noop
    CONNECTOR_AUTH_FAILURES = noop
    CONNECTOR_CACHE_HITS = noop
    CONNECTOR_CACHE_MISSES = noop
    CONNECTOR_RETRIES = noop


def record_sync(
    connector_type: str,
    connector_id: str,
    status: str,
    duration_seconds: float = 0.0,
    items_synced: int = 0,
) -> None:
    """
    Record a sync operation.

    Args:
        connector_type: Type of connector (sharepoint, confluence, etc.)
        connector_id: Unique identifier for the connector instance
        status: Sync status (success, failure, partial)
        duration_seconds: Total sync duration
        items_synced: Number of items successfully synced
    """
    _init_metrics()

    CONNECTOR_SYNCS.labels(
        connector_type=connector_type,
        connector_id=connector_id,
        status=status,
    ).inc()

    if duration_seconds > 0:
        CONNECTOR_SYNC_DURATION.labels(connector_type=connector_type).observe(duration_seconds)

    if status == "success":
        import time

        CONNECTOR_LAST_SYNC.labels(
            connector_type=connector_type,
            connector_id=connector_id,
        ).set(time.time())


def record_sync_items(
    connector_type: str,
    item_type: str,
    count: int = 1,
) -> None:
    """
    Record synced items.

    Args:
        connector_type: Type of connector
        item_type: Type of item (page, document, message, etc.)
        count: Number of items synced
    """
    _init_metrics()

    CONNECTOR_SYNC_ITEMS.labels(
        connector_type=connector_type,
        item_type=item_type,
    ).inc(count)


def record_sync_error(
    connector_type: str,
    error_type: str,
) -> None:
    """
    Record a sync error.

    Args:
        connector_type: Type of connector
        error_type: Type of error (network, auth, rate_limit, parse, etc.)
    """
    _init_metrics()

    CONNECTOR_SYNC_ERRORS.labels(
        connector_type=connector_type,
        error_type=error_type,
    ).inc()


def record_item_latency(
    connector_type: str,
    item_type: str,
    latency_seconds: float,
) -> None:
    """
    Record per-item sync latency.

    Args:
        connector_type: Type of connector
        item_type: Type of item being synced
        latency_seconds: Time taken to sync the item
    """
    _init_metrics()

    CONNECTOR_SYNC_LATENCY.labels(
        connector_type=connector_type,
        item_type=item_type,
    ).observe(latency_seconds)


def record_rate_limit(connector_type: str) -> None:
    """Record a rate limit response."""
    _init_metrics()
    CONNECTOR_RATE_LIMITS.labels(connector_type=connector_type).inc()


def record_auth_failure(connector_type: str, connector_id: str) -> None:
    """Record an authentication failure."""
    _init_metrics()
    CONNECTOR_AUTH_FAILURES.labels(
        connector_type=connector_type,
        connector_id=connector_id,
    ).inc()


def set_connector_health(
    connector_type: str,
    connector_id: str,
    healthy: bool,
) -> None:
    """
    Set connector health status.

    Args:
        connector_type: Type of connector
        connector_id: Connector instance identifier
        healthy: Whether the connector is healthy
    """
    _init_metrics()
    CONNECTOR_HEALTH.labels(
        connector_type=connector_type,
        connector_id=connector_id,
    ).set(1 if healthy else 0)


def inc_active_syncs(connector_type: str) -> None:
    """Increment active sync count."""
    _init_metrics()
    CONNECTOR_ACTIVE_SYNCS.labels(connector_type=connector_type).inc()


def dec_active_syncs(connector_type: str) -> None:
    """Decrement active sync count."""
    _init_metrics()
    CONNECTOR_ACTIVE_SYNCS.labels(connector_type=connector_type).dec()


@contextmanager
def measure_sync(
    connector_type: str,
    connector_id: str,
) -> Generator[dict, None, None]:
    """
    Context manager for measuring sync operations.

    Usage:
        with measure_sync("sharepoint", "sharepoint_corp") as ctx:
            items = await connector.sync()
            ctx["items_synced"] = len(items)
            ctx["status"] = "success"

    Args:
        connector_type: Type of connector
        connector_id: Connector instance identifier

    Yields:
        Dictionary to populate with sync results
    """
    start_time = time.perf_counter()
    inc_active_syncs(connector_type)

    context: dict = {
        "status": "success",
        "items_synced": 0,
    }

    try:
        yield context
    except Exception as e:
        context["status"] = "failure"
        error_type = type(e).__name__
        record_sync_error(connector_type, error_type)
        raise
    finally:
        dec_active_syncs(connector_type)
        duration = time.perf_counter() - start_time
        record_sync(
            connector_type=connector_type,
            connector_id=connector_id,
            status=context.get("status", "unknown"),
            duration_seconds=duration,
            items_synced=context.get("items_synced", 0),
        )


@contextmanager
def measure_item_sync(
    connector_type: str,
    item_type: str,
) -> Generator[None, None, None]:
    """
    Context manager for measuring individual item sync.

    Usage:
        with measure_item_sync("confluence", "page"):
            await process_page(page)

    Args:
        connector_type: Type of connector
        item_type: Type of item being synced
    """
    start_time = time.perf_counter()

    try:
        yield
        record_sync_items(connector_type, item_type, count=1)
    finally:
        duration = time.perf_counter() - start_time
        record_item_latency(connector_type, item_type, duration)


# Convenience functions for specific connector types
def record_sharepoint_sync(
    connector_id: str,
    status: str,
    duration_seconds: float,
    documents: int = 0,
    folders: int = 0,
) -> None:
    """Record SharePoint sync operation."""
    record_sync("sharepoint", connector_id, status, duration_seconds, documents + folders)
    if documents > 0:
        record_sync_items("sharepoint", "document", documents)
    if folders > 0:
        record_sync_items("sharepoint", "folder", folders)


def record_confluence_sync(
    connector_id: str,
    status: str,
    duration_seconds: float,
    pages: int = 0,
    attachments: int = 0,
) -> None:
    """Record Confluence sync operation."""
    record_sync("confluence", connector_id, status, duration_seconds, pages + attachments)
    if pages > 0:
        record_sync_items("confluence", "page", pages)
    if attachments > 0:
        record_sync_items("confluence", "attachment", attachments)


def record_notion_sync(
    connector_id: str,
    status: str,
    duration_seconds: float,
    pages: int = 0,
    databases: int = 0,
) -> None:
    """Record Notion sync operation."""
    record_sync("notion", connector_id, status, duration_seconds, pages + databases)
    if pages > 0:
        record_sync_items("notion", "page", pages)
    if databases > 0:
        record_sync_items("notion", "database", databases)


def record_slack_sync(
    connector_id: str,
    status: str,
    duration_seconds: float,
    messages: int = 0,
    files: int = 0,
) -> None:
    """Record Slack sync operation."""
    record_sync("slack", connector_id, status, duration_seconds, messages + files)
    if messages > 0:
        record_sync_items("slack", "message", messages)
    if files > 0:
        record_sync_items("slack", "file", files)


def record_gdrive_sync(
    connector_id: str,
    status: str,
    duration_seconds: float,
    documents: int = 0,
    folders: int = 0,
) -> None:
    """Record Google Drive sync operation."""
    record_sync("gdrive", connector_id, status, duration_seconds, documents + folders)
    if documents > 0:
        record_sync_items("gdrive", "document", documents)
    if folders > 0:
        record_sync_items("gdrive", "folder", folders)


def record_cache_hit(connector_type: str) -> None:
    """Record a cache hit."""
    _init_metrics()
    CONNECTOR_CACHE_HITS.labels(connector_type=connector_type).inc()


def record_cache_miss(connector_type: str) -> None:
    """Record a cache miss."""
    _init_metrics()
    CONNECTOR_CACHE_MISSES.labels(connector_type=connector_type).inc()


def record_retry(connector_type: str, reason: str) -> None:
    """Record a retry attempt."""
    _init_metrics()
    CONNECTOR_RETRIES.labels(connector_type=connector_type, reason=reason).inc()


def get_connector_metrics() -> dict:
    """Get current connector metrics summary."""
    _init_metrics()

    # Try to get metrics from Prometheus registry
    try:
        from prometheus_client import REGISTRY

        metrics = {}

        # Collect all connector metrics
        for metric in REGISTRY.collect():
            if metric.name.startswith("aragora_connector"):
                samples = list(metric.samples)
                if samples:
                    metrics[metric.name] = {
                        "help": metric.documentation,
                        "type": metric.type,
                        "samples": [
                            {"labels": dict(s.labels), "value": s.value}
                            for s in samples
                        ],
                    }

        return metrics

    except ImportError:
        return {"error": "prometheus_client not installed"}
    except Exception as e:
        return {"error": str(e)}
