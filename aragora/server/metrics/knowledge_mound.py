"""
Knowledge Mound Metrics for Aragora.

Tracks visibility, sharing, federation, and global knowledge operations.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from .types import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# =============================================================================
# Knowledge Mound Metrics
# =============================================================================

# Visibility operations
KNOWLEDGE_VISIBILITY_CHANGES = Counter(
    name="aragora_knowledge_visibility_changes_total",
    help="Number of visibility level changes on knowledge items",
    label_names=["from_level", "to_level", "workspace_id"],
)

KNOWLEDGE_ACCESS_GRANTS = Counter(
    name="aragora_knowledge_access_grants_total",
    help="Number of access grants created/revoked",
    label_names=["action", "grantee_type", "workspace_id"],  # action: grant/revoke
)

# Sharing operations
KNOWLEDGE_SHARES = Counter(
    name="aragora_knowledge_shares_total",
    help="Number of knowledge sharing operations",
    label_names=[
        "action",
        "target_type",
    ],  # action: share/accept/decline/revoke, target_type: workspace/user
)

KNOWLEDGE_SHARED_ITEMS = Gauge(
    name="aragora_knowledge_shared_items_count",
    help="Current number of shared items pending acceptance",
    label_names=["workspace_id"],
)

# Global knowledge operations
KNOWLEDGE_GLOBAL_FACTS = Counter(
    name="aragora_knowledge_global_facts_total",
    help="Number of global/verified facts stored or promoted",
    label_names=["action"],  # action: stored/promoted/queried
)

KNOWLEDGE_GLOBAL_QUERIES = Counter(
    name="aragora_knowledge_global_queries_total",
    help="Number of queries against global knowledge",
    label_names=["has_results"],  # has_results: true/false
)

# Federation operations
KNOWLEDGE_FEDERATION_SYNCS = Counter(
    name="aragora_knowledge_federation_syncs_total",
    help="Number of federation sync operations",
    label_names=[
        "region_id",
        "direction",
        "status",
    ],  # direction: push/pull, status: success/failed
)

KNOWLEDGE_FEDERATION_NODES = Counter(
    name="aragora_knowledge_federation_nodes_total",
    help="Number of nodes synced via federation",
    label_names=["region_id", "direction"],
)

KNOWLEDGE_FEDERATION_LATENCY = Histogram(
    name="aragora_knowledge_federation_latency_seconds",
    help="Latency of federation sync operations",
    label_names=["region_id", "direction"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

KNOWLEDGE_FEDERATION_REGIONS = Gauge(
    name="aragora_knowledge_federation_regions_count",
    help="Number of federated regions by status",
    label_names=["status"],  # status: enabled/disabled/healthy/unhealthy
)


# =============================================================================
# Helpers
# =============================================================================


def track_visibility_change(
    node_id: str,
    from_level: str,
    to_level: str,
    workspace_id: str,
) -> None:
    """Track a visibility level change on a knowledge item."""
    KNOWLEDGE_VISIBILITY_CHANGES.inc(
        from_level=from_level,
        to_level=to_level,
        workspace_id=workspace_id,
    )


def track_access_grant(
    action: str,
    grantee_type: str,
    workspace_id: str,
) -> None:
    """Track an access grant create/revoke operation."""
    KNOWLEDGE_ACCESS_GRANTS.inc(
        action=action,
        grantee_type=grantee_type,
        workspace_id=workspace_id,
    )


def track_share(action: str, target_type: str) -> None:
    """Track a sharing operation (share/accept/decline/revoke)."""
    KNOWLEDGE_SHARES.inc(action=action, target_type=target_type)


def track_shared_items_count(workspace_id: str, count: int) -> None:
    """Update the count of pending shared items for a workspace."""
    KNOWLEDGE_SHARED_ITEMS.set(count, workspace_id=workspace_id)


def track_global_fact(action: str) -> None:
    """Track a global knowledge operation (stored/promoted/queried)."""
    KNOWLEDGE_GLOBAL_FACTS.inc(action=action)


def track_global_query(has_results: bool) -> None:
    """Track a query against global knowledge."""
    KNOWLEDGE_GLOBAL_QUERIES.inc(has_results=str(has_results).lower())


@contextmanager
def track_federation_sync(
    region_id: str,
    direction: str,
) -> Generator[dict, None, None]:
    """Context manager to track federation sync operations.

    Usage:
        with track_federation_sync("region-1", "push") as ctx:
            # perform sync
            ctx["nodes_synced"] = 42
            ctx["status"] = "success"

    Args:
        region_id: The federated region ID
        direction: Sync direction ("push" or "pull")

    Yields:
        Dict to populate with sync results (nodes_synced, status)
    """
    start = time.perf_counter()
    ctx: dict[str, Any] = {"status": "success", "nodes_synced": 0}
    try:
        yield ctx
    except (ValueError, TypeError, KeyError) as e:
        # Data validation and lookup errors
        ctx["status"] = "failed"
        logger.warning("Federation sync error for region %s (%s): %s", region_id, direction, e)
        raise
    except (OSError, IOError, ConnectionError, TimeoutError) as e:
        # I/O and network-related errors (common with federation)
        ctx["status"] = "failed"
        logger.warning("Federation sync I/O error for region %s (%s): %s", region_id, direction, e)
        raise
    except RuntimeError as e:
        # Runtime errors (async issues, state errors)
        ctx["status"] = "failed"
        logger.warning(
            "Federation sync runtime error for region %s (%s): %s", region_id, direction, e
        )
        raise
    finally:
        duration = time.perf_counter() - start
        KNOWLEDGE_FEDERATION_SYNCS.inc(
            region_id=region_id,
            direction=direction,
            status=str(ctx["status"]),
        )
        KNOWLEDGE_FEDERATION_NODES.inc(
            region_id=region_id,
            direction=direction,
            value=int(ctx.get("nodes_synced", 0)),
        )
        KNOWLEDGE_FEDERATION_LATENCY.observe(
            duration,
            region_id=region_id,
            direction=direction,
        )


def track_federation_regions(
    enabled: int = 0,
    disabled: int = 0,
    healthy: int = 0,
    unhealthy: int = 0,
) -> None:
    """Update federation region counts by status."""
    KNOWLEDGE_FEDERATION_REGIONS.set(enabled, status="enabled")
    KNOWLEDGE_FEDERATION_REGIONS.set(disabled, status="disabled")
    KNOWLEDGE_FEDERATION_REGIONS.set(healthy, status="healthy")
    KNOWLEDGE_FEDERATION_REGIONS.set(unhealthy, status="unhealthy")


__all__ = [
    "KNOWLEDGE_VISIBILITY_CHANGES",
    "KNOWLEDGE_ACCESS_GRANTS",
    "KNOWLEDGE_SHARES",
    "KNOWLEDGE_SHARED_ITEMS",
    "KNOWLEDGE_GLOBAL_FACTS",
    "KNOWLEDGE_GLOBAL_QUERIES",
    "KNOWLEDGE_FEDERATION_SYNCS",
    "KNOWLEDGE_FEDERATION_NODES",
    "KNOWLEDGE_FEDERATION_LATENCY",
    "KNOWLEDGE_FEDERATION_REGIONS",
    "track_visibility_change",
    "track_access_grant",
    "track_share",
    "track_shared_items_count",
    "track_global_fact",
    "track_global_query",
    "track_federation_sync",
    "track_federation_regions",
]
