"""
Dashboard health check utilities.

Extracted from dashboard.py to reduce file size. Contains:
- System health metrics
- Connector health aggregation
- Connector type mapping
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_system_health() -> dict[str, Any]:
    """Get system health metrics.

    Returns:
        Health dict with uptime_seconds, cache_entries, active_websocket_connections,
        prometheus_available.
    """
    health: dict[str, Any] = {
        "uptime_seconds": 0,
        "cache_entries": 0,
        "active_websocket_connections": 0,
        "prometheus_available": False,
    }

    try:
        from aragora.server.prometheus import is_prometheus_available

        health["prometheus_available"] = is_prometheus_available()

        # Get cache stats if available
        from aragora.server.handlers.base import _cache

        if _cache:
            health["cache_entries"] = len(_cache)

    except Exception as e:
        logger.warning("System health error: %s: %s", type(e).__name__, e)

    return health


def get_connector_type(connector: Any) -> str:
    """Extract connector type from connector instance.

    Args:
        connector: Connector instance.

    Returns:
        Connector type string (e.g., "github", "s3", "postgresql").
    """
    if not connector:
        return "unknown"
    class_name = type(connector).__name__.lower()
    type_mapping = {
        "githubenterpriseconnector": "github",
        "s3connector": "s3",
        "postgresqlconnector": "postgresql",
        "mongodbconnector": "mongodb",
        "fhirconnector": "fhir",
    }
    return type_mapping.get(class_name, class_name.replace("connector", ""))


def get_connector_health() -> dict[str, Any]:
    """Get connector health metrics for dashboard.

    Returns aggregated health from the sync scheduler including:
    - Summary stats (total, healthy, degraded, unhealthy, health_score)
    - Per-connector breakdown with status, sync metrics, and errors

    Returns:
        Dict with summary and connectors list.
    """
    result: dict[str, Any] = {
        "summary": {
            "total_connectors": 0,
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "health_score": 100,
            "scheduler_running": False,
            "running_syncs": 0,
            "success_rate": 1.0,
        },
        "connectors": [],
    }

    try:
        from aragora.server.handlers.connectors import get_scheduler

        scheduler = get_scheduler()
        stats = scheduler.get_stats()

        # Build summary from scheduler stats
        result["summary"]["total_connectors"] = stats.get("total_jobs", 0)
        result["summary"]["scheduler_running"] = scheduler._scheduler_task is not None
        result["summary"]["running_syncs"] = stats.get("running_syncs", 0)
        result["summary"]["success_rate"] = stats.get("success_rate", 1.0)

        # Build per-connector breakdown from jobs
        jobs = scheduler.list_jobs()
        healthy = degraded = unhealthy = 0

        for job in jobs:
            # Determine health: 3+ failures = unhealthy, 1-2 = degraded
            if job.consecutive_failures >= 3:
                health = "unhealthy"
                unhealthy += 1
            elif job.consecutive_failures >= 1:
                health = "degraded"
                degraded += 1
            else:
                health = "healthy"
                healthy += 1

            # Determine status
            if job.current_run_id:
                status = "syncing"
            elif job.consecutive_failures >= 3:
                status = "error"
            elif not job.schedule.enabled:
                status = "disconnected"
            else:
                status = "connected"

            # Calculate metrics from history
            history = scheduler.get_history(job_id=job.id, limit=100)
            total_syncs = len(history)
            failed = sum(1 for h in history if h.status.value == "failed")
            error_rate = (failed / total_syncs * 100) if total_syncs > 0 else 0.0
            avg_duration = (
                sum(h.duration_seconds or 0 for h in history) / total_syncs
                if total_syncs > 0
                else 0.0
            )
            total_items = sum(h.items_synced for h in history)

            connector_name = job.connector_id
            connector_type = "unknown"
            if job.connector:
                connector_name = getattr(job.connector, "name", job.connector_id)
                connector_type = get_connector_type(job.connector)

            result["connectors"].append(
                {
                    "connector_id": job.connector_id,
                    "connector_name": connector_name,
                    "connector_type": connector_type,
                    "status": status,
                    "health": health,
                    "uptime": round(100 - error_rate, 1),
                    "error_rate": round(error_rate, 1),
                    "last_sync": job.last_run.isoformat() if job.last_run else None,
                    "next_sync": job.next_run.isoformat() if job.next_run else None,
                    "items_synced": total_items,
                    "avg_sync_duration": round(avg_duration, 1),
                    "consecutive_failures": job.consecutive_failures,
                }
            )

        result["summary"]["healthy"] = healthy
        result["summary"]["degraded"] = degraded
        result["summary"]["unhealthy"] = unhealthy
        total = healthy + degraded + unhealthy
        result["summary"]["health_score"] = round((healthy / total) * 100) if total > 0 else 100

    except ImportError:
        logger.debug("Connector scheduler not available")
    except Exception as e:
        logger.warning("Connector health error: %s: %s", type(e).__name__, e)

    return result
