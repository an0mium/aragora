# mypy: ignore-errors
"""
Knowledge Mound health check implementations.

Provides comprehensive health checks for:
- /api/health/knowledge-mound - Full KM subsystem health
- /api/health/decay - Confidence decay scheduler status
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)


def knowledge_mound_health(handler) -> HandlerResult:
    """Comprehensive health check for Knowledge Mound subsystem.

    Returns detailed status of:
    - Core mound: Storage layer, fact storage, workspace config
    - Culture accumulator: Organizational pattern tracking
    - Staleness tracking: Fact age and refresh status
    - Query performance: Retrieval latency stats
    - Storage backends: PostgreSQL/SQLite status
    - RLM integration: Context compression caching

    This endpoint is useful for:
    - Monitoring Knowledge Mound operational health
    - Debugging knowledge retrieval issues
    - Verifying debate-knowledge integration
    - Tracking storage capacity and performance

    Returns:
        JSON response with comprehensive KM health metrics
    """
    components: Dict[str, Dict[str, Any]] = {}
    all_healthy = True
    warnings: list[str] = []
    start_time = time.time()

    # 1. Check if Knowledge Mound module is available
    try:
        from aragora.knowledge.mound import KnowledgeMound  # noqa: F401
        from aragora.knowledge.mound.types import MoundConfig  # noqa: F401

        components["module"] = {
            "healthy": True,
            "status": "available",
        }
    except ImportError as e:
        components["module"] = {
            "healthy": False,
            "status": "not_available",
            "error": str(e)[:100],
        }
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
    try:
        mound = KnowledgeMound(workspace_id="health_check")  # type: ignore[abstract]

        components["core"] = {
            "healthy": True,
            "status": "initialized",
            "workspace_id": mound.workspace_id,
        }

        # Check config
        if hasattr(mound, "config") and mound.config:
            components["core"]["config"] = {
                "enable_staleness_tracking": mound.config.enable_staleness_tracking,  # type: ignore[attr-defined]
                "enable_culture_accumulator": mound.config.enable_culture_accumulator,  # type: ignore[attr-defined]
                "enable_rlm_summaries": mound.config.enable_rlm_summaries,  # type: ignore[attr-defined]
                "default_staleness_hours": mound.config.default_staleness_hours,  # type: ignore[attr-defined]
            }

    except Exception as e:
        components["core"] = {
            "healthy": False,
            "status": "initialization_failed",
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }
        all_healthy = False

    # 3. Check storage backend
    try:
        import os

        database_url = os.environ.get("KNOWLEDGE_MOUND_DATABASE_URL", "")

        if "postgres" in database_url.lower():
            components["storage"] = {
                "healthy": True,
                "backend": "postgresql",
                "status": "configured",
            }
        else:
            components["storage"] = {
                "healthy": True,
                "backend": "sqlite",
                "status": "configured",
                "note": "Using local SQLite storage",
            }

        # Try to get storage stats if mound initialized
        if "mound" in locals() and hasattr(mound, "_store"):
            try:
                # Check if store is accessible
                if mound._store is not None:
                    components["storage"]["store_type"] = type(mound._store).__name__
            except AttributeError:
                pass

    except Exception as e:
        components["storage"] = {
            "healthy": True,
            "status": "unknown",
            "warning": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 4. Check culture accumulator
    try:
        if "mound" in locals():
            if hasattr(mound, "_culture_accumulator") and mound._culture_accumulator:
                accumulator = mound._culture_accumulator
                components["culture_accumulator"] = {
                    "healthy": True,
                    "status": "active",
                    "type": type(accumulator).__name__,
                }

                # Try to get pattern counts
                try:
                    if hasattr(accumulator, "_patterns"):
                        workspace_count = len(accumulator._patterns)
                        components["culture_accumulator"]["workspaces_tracked"] = workspace_count
                except (AttributeError, TypeError):
                    pass
            else:
                components["culture_accumulator"] = {
                    "healthy": True,
                    "status": "not_initialized",
                    "note": "Culture accumulator disabled or not yet created",
                }
    except Exception as e:
        components["culture_accumulator"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 5. Check staleness tracker
    try:
        if "mound" in locals():
            if hasattr(mound, "_staleness_tracker") and mound._staleness_tracker:
                components["staleness_tracker"] = {
                    "healthy": True,
                    "status": "active",
                }
            else:
                components["staleness_tracker"] = {
                    "healthy": True,
                    "status": "not_initialized",
                    "note": "Staleness tracking disabled",
                }
    except Exception as e:
        components["staleness_tracker"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 6. Check RLM integration
    try:
        from aragora.rlm import HAS_OFFICIAL_RLM

        if HAS_OFFICIAL_RLM:
            components["rlm_integration"] = {
                "healthy": True,
                "status": "active",
                "type": "official_rlm",
            }
        else:
            components["rlm_integration"] = {
                "healthy": True,
                "status": "fallback",
                "type": "compression_only",
                "note": "Using compression fallback (official RLM not installed)",
            }
    except ImportError:
        components["rlm_integration"] = {
            "healthy": True,
            "status": "not_available",
            "note": "RLM module not installed",
        }
    except Exception as e:
        components["rlm_integration"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 7. Check debate integration via knowledge_mound_ops
    try:
        from aragora.debate.knowledge_mound_ops import get_knowledge_mound_stats  # type: ignore[attr-defined]

        km_stats = get_knowledge_mound_stats()
        components["debate_integration"] = {
            "healthy": True,
            "status": "active",
            "facts_count": km_stats.get("facts_count", 0),
            "consensus_stored": km_stats.get("consensus_stored", 0),
            "retrievals_count": km_stats.get("retrievals_count", 0),
        }
    except ImportError:
        components["debate_integration"] = {
            "healthy": True,
            "status": "not_available",
            "note": "knowledge_mound_ops module not available",
        }
    except Exception as e:
        components["debate_integration"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 8. Check Redis cache for mound (if configured)
    try:
        import os

        redis_url = os.environ.get("KNOWLEDGE_MOUND_REDIS_URL") or os.environ.get("REDIS_URL")
        if redis_url:
            from aragora.knowledge.mound.redis_cache import KnowledgeMoundCache  # type: ignore[attr-defined]

            cache = KnowledgeMoundCache(redis_url=redis_url)
            components["redis_cache"] = {
                "healthy": True,
                "status": "configured",
            }
        else:
            components["redis_cache"] = {
                "healthy": True,
                "status": "not_configured",
                "note": "Knowledge Mound Redis cache not configured",
            }
    except ImportError:
        components["redis_cache"] = {
            "healthy": True,
            "status": "not_available",
            "note": "Redis cache module not installed",
        }
    except Exception as e:
        components["redis_cache"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 9. Check bidirectional adapters
    try:
        from aragora.knowledge.mound.adapters import (
            ContinuumAdapter,
            ConsensusAdapter,
            CritiqueAdapter,
            EvidenceAdapter,
            BeliefAdapter,
            InsightsAdapter,
            EloAdapter,
            PulseAdapter,
            CostAdapter,
            RankingAdapter,
            CultureAdapter,
        )

        adapter_classes = [
            ("continuum", ContinuumAdapter),
            ("consensus", ConsensusAdapter),
            ("critique", CritiqueAdapter),
            ("evidence", EvidenceAdapter),
            ("belief", BeliefAdapter),
            ("insights", InsightsAdapter),
            ("elo", EloAdapter),
            ("pulse", PulseAdapter),
            ("cost", CostAdapter),
            ("ranking", RankingAdapter),
            ("culture", CultureAdapter),
        ]

        components["bidirectional_adapters"] = {
            "healthy": True,
            "status": "available",
            "adapters_available": len(adapter_classes),
            "adapter_list": [name for name, _ in adapter_classes],
        }
    except ImportError as e:
        components["bidirectional_adapters"] = {
            "healthy": True,
            "status": "partial",
            "error": f"Some adapters not available: {str(e)[:80]}",
        }
    except Exception as e:
        components["bidirectional_adapters"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 10. Check Control Plane adapter
    try:
        from aragora.knowledge.mound.adapters.control_plane_adapter import (  # noqa: F401
            ControlPlaneAdapter,
            TaskOutcome,
            AgentCapabilityRecord,
            CrossWorkspaceInsight,
        )

        components["control_plane_adapter"] = {
            "healthy": True,
            "status": "available",
            "features": [
                "task_outcome_storage",
                "capability_records",
                "cross_workspace_insights",
                "agent_recommendations",
            ],
        }
    except ImportError as e:
        components["control_plane_adapter"] = {
            "healthy": True,
            "status": "not_available",
            "error": str(e)[:80],
        }
    except Exception as e:
        components["control_plane_adapter"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 11. Check KM metrics availability
    try:
        from aragora.observability.metrics.km import (  # noqa: F401
            init_km_metrics,
            record_km_operation,
            record_cp_task_outcome,
        )

        components["km_metrics"] = {
            "healthy": True,
            "status": "available",
            "prometheus_integration": True,
        }
    except ImportError:
        components["km_metrics"] = {
            "healthy": True,
            "status": "not_available",
            "prometheus_integration": False,
        }
    except Exception as e:
        components["km_metrics"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # 12. Check Confidence Decay Scheduler
    try:
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            get_decay_scheduler,
        )

        scheduler = get_decay_scheduler()
        if scheduler:
            stats = scheduler.get_stats()
            components["confidence_decay"] = {
                "healthy": True,
                "status": "active" if scheduler.is_running else "stopped",
                "running": scheduler.is_running,
                "decay_interval_hours": stats.get("decay_interval_hours", 24),
                "total_cycles": stats.get("total_decay_cycles", 0),
                "total_items_processed": stats.get("total_items_processed", 0),
                "last_run": stats.get("last_run", {}),
                "workspaces_monitored": stats.get("workspaces"),
            }
            # Add alerting info
            if scheduler.is_running:
                last_runs = stats.get("last_run", {})
                if last_runs:
                    # Check if any workspace hasn't been processed in >48 hours
                    now = datetime.now()
                    stale_workspaces = []
                    for ws_id, run_time_str in last_runs.items():
                        try:
                            run_time = datetime.fromisoformat(run_time_str)
                            hours_since = (now - run_time).total_seconds() / 3600
                            if hours_since > 48:
                                stale_workspaces.append(ws_id)
                        except (ValueError, TypeError):
                            pass
                    if stale_workspaces:
                        components["confidence_decay"]["alert"] = {
                            "level": "warning",
                            "message": f"Decay not run in >48h for: {', '.join(stale_workspaces)}",
                        }
                        warnings.append(
                            f"Confidence decay stale for workspaces: {stale_workspaces}"
                        )
        else:
            components["confidence_decay"] = {
                "healthy": True,
                "status": "not_configured",
                "note": "Confidence decay scheduler not initialized",
            }
    except ImportError:
        components["confidence_decay"] = {
            "healthy": True,
            "status": "not_available",
            "note": "Confidence decay module not installed",
        }
    except Exception as e:
        components["confidence_decay"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # Calculate response time
    response_time_ms = round((time.time() - start_time) * 1000, 2)

    # Determine overall status
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
    workspace_status: Dict[str, Any] = {}
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
