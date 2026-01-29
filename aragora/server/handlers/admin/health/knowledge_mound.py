# mypy: ignore-errors
"""
Knowledge Mound health check implementations.

Provides comprehensive health checks for:
- /api/health/knowledge-mound - Full KM subsystem health
- /api/health/decay - Confidence decay scheduler status

The main function delegates to helper functions in knowledge_mound_utils.py
for better modularity and testability.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from ...base import HandlerResult, json_response
from .knowledge_mound_utils import (
    check_knowledge_mound_module,
    check_mound_core_initialization,
    check_storage_backend,
    check_culture_accumulator,
    check_staleness_tracker,
    check_rlm_integration,
    check_debate_integration,
    check_knowledge_mound_redis_cache,
    check_bidirectional_adapters,
    check_control_plane_adapter,
    check_km_metrics,
    check_confidence_decay_scheduler,
)

logger = logging.getLogger(__name__)


def knowledge_mound_health(handler) -> HandlerResult:
    """Comprehensive health check for Knowledge Mound subsystem.

    Returns detailed status of:
    - Core mound: Storage layer, fact storage, workspace config
    - Culture accumulator: Organizational pattern tracking
    - Staleness tracking: Fact age and refresh status
    - Storage backends: PostgreSQL/SQLite status
    - RLM integration: Context compression caching
    - Debate integration: Knowledge-debate operations
    - Redis cache: Distributed caching status
    - Bidirectional adapters: Adapter availability
    - Control Plane adapter: CP integration status
    - KM metrics: Prometheus integration
    - Confidence decay: Scheduler status

    Returns:
        JSON response with comprehensive KM health metrics
    """
    components: dict[str, dict[str, Any]] = {}
    all_healthy = True
    warnings: list[str] = []
    start_time = time.time()

    # 1. Check if Knowledge Mound module is available
    components["module"], should_abort = check_knowledge_mound_module()
    if should_abort:
        all_healthy = False
        return json_response(
            {
                "status": "unavailable",
                "error": "Knowledge Mound module not installed",
                "components": components,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
        )

    # 2. Check core mound initialization
    components["core"], mound = check_mound_core_initialization()
    if not components["core"]["healthy"]:
        all_healthy = False

    # 3-5. Check storage, culture accumulator, and staleness tracker
    components["storage"] = check_storage_backend(mound)
    components["culture_accumulator"] = check_culture_accumulator(mound)
    components["staleness_tracker"] = check_staleness_tracker(mound)

    # 6-11. Check integrations and features
    components["rlm_integration"] = check_rlm_integration()
    components["debate_integration"] = check_debate_integration()
    components["redis_cache"] = check_knowledge_mound_redis_cache()
    components["bidirectional_adapters"] = check_bidirectional_adapters()
    components["control_plane_adapter"] = check_control_plane_adapter()
    components["km_metrics"] = check_km_metrics()

    # 12. Check Confidence Decay Scheduler
    components["confidence_decay"], decay_warnings = check_confidence_decay_scheduler()
    warnings.extend(decay_warnings)

    # Calculate response time and determine overall status
    response_time_ms = round((time.time() - start_time) * 1000, 2)
    healthy_count = sum(1 for c in components.values() if c.get("healthy", False))
    active_count = sum(1 for c in components.values() if c.get("status") == "active")
    total_components = len(components)

    status = "healthy" if all_healthy else "degraded"
    if active_count == 0:
        status = "not_configured"

    return json_response(
        {
            "status": status,
            "summary": {
                "total_components": total_components,
                "healthy": healthy_count,
                "active": active_count,
            },
            "components": components,
            "warnings": warnings if warnings else None,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
    )


def decay_health(handler) -> HandlerResult:
    """Confidence decay scheduler health - dedicated endpoint for decay monitoring.

    Provides focused status for the confidence decay scheduler including:
    - Scheduler running status
    - Decay interval configuration
    - Total cycles and items processed
    - Last run timestamps per workspace
    - Alerting for stale workspaces (>48h since last decay)
    - Prometheus metrics availability
    """
    start_time = time.time()
    warnings: list[str] = []

    try:
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            get_decay_scheduler,
            DECAY_METRICS_AVAILABLE,
        )
    except ImportError:
        return json_response(
            {
                "status": "not_available",
                "message": "Confidence decay scheduler module not installed",
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
            },
            status=503,
        )

    scheduler = get_decay_scheduler()
    if not scheduler:
        return json_response(
            {
                "status": "not_configured",
                "message": "Confidence decay scheduler not initialized",
                "metrics_available": DECAY_METRICS_AVAILABLE,
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
            }
        )

    # Get scheduler statistics
    stats = scheduler.get_stats()
    is_running = scheduler.is_running

    # Build workspace status
    workspace_status: dict[str, Any] = {}
    last_runs = stats.get("last_decay_per_workspace", {})
    now = datetime.now(timezone.utc)
    stale_threshold_hours = 48

    for workspace_id, last_run_str in last_runs.items():
        try:
            last_run = datetime.fromisoformat(last_run_str.replace("Z", "+00:00"))
            hours_since = (now - last_run).total_seconds() / 3600
            is_stale = hours_since > stale_threshold_hours
            workspace_status[workspace_id] = {
                "last_decay": last_run_str,
                "hours_since_decay": round(hours_since, 1),
                "stale": is_stale,
            }
            if is_stale:
                warnings.append(
                    f"Workspace {workspace_id} has not had decay in {round(hours_since)}h (>{stale_threshold_hours}h threshold)"
                )
        except (ValueError, TypeError):
            workspace_status[workspace_id] = {
                "last_decay": last_run_str,
                "parse_error": True,
            }

    # Determine overall status
    stale_count = sum(1 for w in workspace_status.values() if w.get("stale", False))
    if not is_running:
        status = "stopped"
    elif stale_count > 0:
        status = "degraded"
    else:
        status = "healthy"

    response_time_ms = round((time.time() - start_time) * 1000, 2)

    return json_response(
        {
            "status": status,
            "scheduler": {
                "running": is_running,
                "decay_interval_hours": stats.get("decay_interval_hours", 24),
                "min_confidence_threshold": stats.get("min_confidence_threshold", 0.1),
                "decay_rate": stats.get("decay_rate", 0.95),
            },
            "statistics": {
                "total_cycles": stats.get("total_decay_cycles", 0),
                "total_items_processed": stats.get("total_items_decayed", 0),
                "total_items_expired": stats.get("total_items_expired", 0),
                "errors": stats.get("decay_errors", 0),
            },
            "workspaces": {
                "total": len(workspace_status),
                "stale_count": stale_count,
                "stale_threshold_hours": stale_threshold_hours,
                "details": workspace_status if workspace_status else None,
            },
            "metrics_available": DECAY_METRICS_AVAILABLE,
            "warnings": warnings if warnings else None,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
    )
