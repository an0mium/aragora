"""
Shared health check utilities and helper endpoints.

Provides:
- /api/health/sync - Supabase sync service status
- /api/health/circuits - Circuit breaker status
- /api/health/slow-debates - Slow-running debate detection
- /api/health/components - Component health from HealthRegistry
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)


def sync_status(handler: Any) -> HandlerResult:
    """Get Supabase sync service status.

    Returns status of the background sync service including:
    - enabled: Whether sync is enabled via SUPABASE_SYNC_ENABLED
    - running: Whether the background thread is running
    - queue_size: Number of items pending sync
    - synced_count: Total items successfully synced
    - failed_count: Total items that failed after max retries
    - last_sync_at: Timestamp of last sync attempt
    - last_error: Most recent error message (if any)

    Returns:
        JSON response with sync service status
    """
    try:
        from aragora.persistence.sync_service import get_sync_service

        sync = get_sync_service()
        status = sync.get_status()

        return json_response(
            {
                "enabled": status.enabled,
                "running": status.running,
                "queue_size": status.queue_size,
                "synced_count": status.synced_count,
                "failed_count": status.failed_count,
                "last_sync_at": (
                    status.last_sync_at.isoformat() + "Z" if status.last_sync_at else None
                ),
                "last_error": status.last_error,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )

    except ImportError:
        return json_response(
            {
                "enabled": False,
                "error": "sync_service module not available",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError) as e:
        logger.warning(f"Sync status check failed: {e}")
        return json_response(
            {
                "enabled": False,
                "error": "Health check failed",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )


def slow_debates_status(handler: Any) -> HandlerResult:
    """Get status of slow-running debates.

    Returns debates that have been running longer than the slow threshold
    (default 30 seconds per round). Useful for identifying performance
    bottlenecks and stuck debates.

    Includes:
    - current_slow: Currently active debates running slow
    - recent_slow: Historical slow debates from performance monitor

    Returns:
        JSON response with slow debate information
    """
    # Slow threshold: 30 seconds total, or configurable via env
    import os

    slow_threshold = float(os.getenv("ARAGORA_SLOW_DEBATE_THRESHOLD", "30"))

    current_slow = []
    recent_slow = []
    errors = []

    # Get currently active slow debates
    try:
        from aragora.server.stream.state_manager import (
            get_active_debates,
            get_active_debates_lock,
        )

        now = time.time()

        with get_active_debates_lock():
            for debate_id, debate_info in get_active_debates().items():
                start_time = debate_info.get("start_time", now)
                duration = now - start_time
                if duration > slow_threshold:
                    current_slow.append(
                        {
                            "debate_id": debate_id,
                            "duration_seconds": round(duration, 2),
                            "task": debate_info.get("task", "")[:100],
                            "agents": debate_info.get("agents", []),
                            "current_round": debate_info.get("current_round", 0),
                            "total_rounds": debate_info.get("total_rounds", 0),
                            "started_at": datetime.fromtimestamp(start_time).isoformat() + "Z",
                        }
                    )

        # Sort by duration descending
        current_slow.sort(key=lambda x: x["duration_seconds"], reverse=True)

    except ImportError:
        errors.append("state_manager not available")
    except (OSError, RuntimeError, ValueError) as e:
        errors.append(f"state_manager error: {type(e).__name__}")

    # Get historical slow debates from performance monitor
    try:
        from aragora.debate.performance_monitor import get_debate_monitor

        monitor = get_debate_monitor()

        # Get historical slow debates
        recent_slow = monitor.get_slow_debates(limit=20)

        # Also get currently monitored slow debates
        monitored_current = monitor.get_current_slow_debates()
        for debate in monitored_current:
            # Avoid duplicates with state manager
            if not any(d["debate_id"] == debate["debate_id"] for d in current_slow):
                current_slow.append(debate)

    except ImportError:
        errors.append("performance_monitor not available")
    except (OSError, RuntimeError, ValueError) as e:
        errors.append(f"performance_monitor error: {type(e).__name__}")

    # Determine overall status
    total_slow = len(current_slow) + len(recent_slow)
    status = "healthy" if total_slow == 0 else "degraded"
    if errors and total_slow == 0:
        status = "partial"

    return json_response(
        {
            "status": status,
            "slow_threshold_seconds": slow_threshold,
            "current_slow_count": len(current_slow),
            "recent_slow_count": len(recent_slow),
            "current_slow": current_slow[:20],
            "recent_slow": recent_slow[:20],
            "errors": errors if errors else None,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
    )


def circuit_breakers_status(handler: Any) -> HandlerResult:
    """Get detailed circuit breaker status for all registered breakers.

    Returns comprehensive metrics including:
    - summary: Counts of open, closed, half-open circuits
    - circuit_breakers: Per-breaker details with failures, cooldowns, entity tracking
    - health: Overall health status and high-failure circuit warnings

    This endpoint is useful for:
    - Monitoring dashboards
    - Debugging cascading failures
    - Tuning circuit breaker thresholds

    Returns:
        JSON response with detailed circuit breaker metrics
    """
    try:
        from aragora.resilience import get_circuit_breaker_metrics

        metrics = get_circuit_breaker_metrics()

        return json_response(
            {
                "status": metrics.get("health", {}).get("status", "unknown"),
                "summary": metrics.get("summary", {}),
                "circuit_breakers": metrics.get("circuit_breakers", {}),
                "health": metrics.get("health", {}),
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )

    except ImportError:
        return json_response(
            {
                "status": "unavailable",
                "error": "resilience module not available",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError) as e:
        logger.warning(f"Circuit breaker status check failed: {e}")
        return json_response(
            {
                "status": "error",
                "error": "Health check failed",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )


def component_health_status(handler: Any) -> HandlerResult:
    """Get health status from the resilience patterns HealthRegistry.

    Returns status of all registered health checkers including:
    - Component name and health status
    - Consecutive failures and last error
    - Latency metrics
    - Custom metadata

    This endpoint provides a unified view of component health across
    the application, separate from circuit breakers.

    Returns:
        JSON response with component health report
    """
    try:
        from aragora.resilience.health import get_global_health_registry

        registry = get_global_health_registry()
        report = registry.get_report()

        return json_response(
            {
                "status": "healthy" if report.overall_healthy else "degraded",
                "overall_healthy": report.overall_healthy,
                "summary": report.summary,
                "components": {
                    name: {
                        "healthy": status.healthy,
                        "consecutive_failures": status.consecutive_failures,
                        "last_error": status.last_error,
                        "latency_ms": status.latency_ms,
                        "last_check": status.last_check.isoformat() + "Z",
                        "metadata": status.metadata,
                    }
                    for name, status in report.components.items()
                },
                "checked_at": report.checked_at.isoformat() + "Z",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )

    except ImportError:
        return json_response(
            {
                "status": "unavailable",
                "error": "resilience_patterns module not available",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError) as e:
        logger.warning(f"Component health status check failed: {e}")
        return json_response(
            {
                "status": "error",
                "error": "Health check failed",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )
