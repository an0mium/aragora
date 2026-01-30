"""
Knowledge Mound health check utility functions.

Provides standalone helper functions for Knowledge Mound health checks,
extracted from knowledge_mound.py for better modularity and testability.

Each function performs a specific health check and returns a dict with:
- healthy: bool - Whether the component is healthy
- status: str - Component status (e.g., "active", "not_configured", "error")
- Additional fields specific to each check

These functions are used by knowledge_mound_health() to build the complete
health response.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


def check_knowledge_mound_module() -> tuple[dict[str, Any], bool]:
    """Check if Knowledge Mound module is available.

    Returns:
        Tuple of (component_dict, should_abort):
        - component_dict: Health check result for the module
        - should_abort: True if the module is not available (caller should abort)
    """
    try:
        from aragora.knowledge.mound import KnowledgeMound  # noqa: F401
        from aragora.knowledge.mound.types import MoundConfig  # noqa: F401

        return {
            "healthy": True,
            "status": "available",
        }, False
    except ImportError as e:
        return {
            "healthy": False,
            "status": "not_available",
            "error": str(e)[:100],
        }, True


def check_mound_core_initialization() -> tuple[dict[str, Any], "KnowledgeMound | None"]:
    """Check core mound initialization.

    Returns:
        Tuple of (component_dict, mound_instance):
        - component_dict: Health check result for core initialization
        - mound_instance: The KnowledgeMound instance if successful, None otherwise
    """
    try:
        from aragora.knowledge.mound import KnowledgeMound as KnowledgeMoundClass
        from aragora.knowledge.mound.types import MoundConfig

        # KnowledgeMound is a concrete class composed of mixins but mypy sees it as abstract
        # due to how the mixin pattern is implemented. It is instantiable at runtime.
        mound: KnowledgeMound = cast("KnowledgeMound", KnowledgeMoundClass(
            workspace_id="health_check"
        ))

        result: dict[str, Any] = {
            "healthy": True,
            "status": "initialized",
            "workspace_id": mound.workspace_id,
        }

        # Check config - use getattr for optional attributes that may vary by version
        if hasattr(mound, "config") and mound.config:
            config: MoundConfig = mound.config
            result["config"] = {
                "enable_staleness_tracking": getattr(config, "enable_staleness_detection", False),
                "enable_culture_accumulator": getattr(config, "enable_culture_accumulator", False),
                "enable_rlm_summaries": getattr(config, "enable_rlm_summaries", False),
                "default_staleness_hours": getattr(config, "default_staleness_hours", None),
            }

        return result, mound

    except Exception as e:
        return {
            "healthy": False,
            "status": "initialization_failed",
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }, None


def check_storage_backend(mound: "KnowledgeMound | None" = None) -> dict[str, Any]:
    """Check storage backend configuration and status.

    Args:
        mound: Optional KnowledgeMound instance to check store type.

    Returns:
        Dict with storage backend health status.
    """
    try:
        database_url = os.environ.get("KNOWLEDGE_MOUND_DATABASE_URL", "")

        if "postgres" in database_url.lower():
            result: dict[str, Any] = {
                "healthy": True,
                "backend": "postgresql",
                "status": "configured",
            }
        else:
            result = {
                "healthy": True,
                "backend": "sqlite",
                "status": "configured",
                "note": "Using local SQLite storage",
            }

        # Try to get storage stats if mound initialized
        if mound is not None and hasattr(mound, "_store"):
            try:
                # Check if store is accessible
                if mound._store is not None:
                    result["store_type"] = type(mound._store).__name__
            except AttributeError:
                pass

        return result

    except Exception as e:
        return {
            "healthy": True,
            "status": "unknown",
            "warning": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_culture_accumulator(mound: "KnowledgeMound | None" = None) -> dict[str, Any]:
    """Check culture accumulator status.

    Args:
        mound: Optional KnowledgeMound instance to check accumulator.

    Returns:
        Dict with culture accumulator health status.
    """
    try:
        if mound is not None:
            if hasattr(mound, "_culture_accumulator") and mound._culture_accumulator:
                accumulator = mound._culture_accumulator
                result: dict[str, Any] = {
                    "healthy": True,
                    "status": "active",
                    "type": type(accumulator).__name__,
                }

                # Try to get pattern counts
                try:
                    if hasattr(accumulator, "_patterns"):
                        workspace_count = len(accumulator._patterns)
                        result["workspaces_tracked"] = workspace_count
                except (AttributeError, TypeError):
                    pass

                return result
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                    "note": "Culture accumulator disabled or not yet created",
                }

        return {
            "healthy": True,
            "status": "not_initialized",
            "note": "Mound not available for culture accumulator check",
        }

    except Exception as e:
        return {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_staleness_tracker(mound: "KnowledgeMound | None" = None) -> dict[str, Any]:
    """Check staleness tracker status.

    Args:
        mound: Optional KnowledgeMound instance to check tracker.

    Returns:
        Dict with staleness tracker health status.
    """
    try:
        if mound is not None:
            if hasattr(mound, "_staleness_tracker") and mound._staleness_tracker:
                return {
                    "healthy": True,
                    "status": "active",
                }
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                    "note": "Staleness tracking disabled",
                }

        return {
            "healthy": True,
            "status": "not_initialized",
            "note": "Mound not available for staleness tracker check",
        }

    except Exception as e:
        return {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_rlm_integration() -> dict[str, Any]:
    """Check RLM (Recursive Language Models) integration status.

    Returns:
        Dict with RLM integration health status.
    """
    try:
        from aragora.rlm import HAS_OFFICIAL_RLM

        if HAS_OFFICIAL_RLM:
            return {
                "healthy": True,
                "status": "active",
                "type": "official_rlm",
            }
        else:
            return {
                "healthy": True,
                "status": "fallback",
                "type": "compression_only",
                "note": "Using compression fallback (official RLM not installed)",
            }
    except ImportError:
        return {
            "healthy": True,
            "status": "not_available",
            "note": "RLM module not installed",
        }
    except Exception as e:
        return {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_debate_integration() -> dict[str, Any]:
    """Check debate integration via knowledge_mound_ops.

    Returns:
        Dict with debate-KM integration health status.
    """
    try:
        from aragora.debate import knowledge_mound_ops

        # Check if get_knowledge_mound_stats function exists in the module
        get_stats_fn = getattr(knowledge_mound_ops, "get_knowledge_mound_stats", None)
        if get_stats_fn is not None:
            km_stats: dict[str, Any] = get_stats_fn()
            return {
                "healthy": True,
                "status": "active",
                "facts_count": km_stats.get("facts_count", 0),
                "consensus_stored": km_stats.get("consensus_stored", 0),
                "retrievals_count": km_stats.get("retrievals_count", 0),
            }
        else:
            # Module exists but function not available
            return {
                "healthy": True,
                "status": "partial",
                "note": "KnowledgeMoundOperations class available but stats function not found",
            }
    except ImportError:
        return {
            "healthy": True,
            "status": "not_available",
            "note": "knowledge_mound_ops module not available",
        }
    except Exception as e:
        return {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_knowledge_mound_redis_cache() -> dict[str, Any]:
    """Check Redis cache configuration for Knowledge Mound.

    Returns:
        Dict with Redis cache health status.
    """
    try:
        redis_url = os.environ.get("KNOWLEDGE_MOUND_REDIS_URL") or os.environ.get("REDIS_URL")
        if redis_url:
            from aragora.knowledge.mound.redis_cache import RedisCache

            RedisCache(url=redis_url)
            return {
                "healthy": True,
                "status": "configured",
            }
        else:
            return {
                "healthy": True,
                "status": "not_configured",
                "note": "Knowledge Mound Redis cache not configured",
            }
    except ImportError:
        return {
            "healthy": True,
            "status": "not_available",
            "note": "Redis cache module not installed",
        }
    except Exception as e:
        return {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_codebase_context() -> dict[str, Any]:
    """Check codebase context availability for TRUE RLM workflows."""
    try:
        from pathlib import Path

        root = Path(
            os.environ.get("ARAGORA_CODEBASE_ROOT")
            or os.environ.get("ARAGORA_REPO_ROOT")
            or os.getcwd()
        ).resolve()
        context_dir = root / ".nomic" / "context"
        manifest_path = context_dir / "codebase_manifest.tsv"

        info: dict[str, Any] = {
            "healthy": True,
            "status": "available" if manifest_path.exists() else "missing",
            "root": str(root),
            "context_dir": str(context_dir),
            "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        }

        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    header = [handle.readline().strip() for _ in range(3)]
                for line in header:
                    if "files=" in line and "lines=" in line:
                        for part in line.split():
                            if part.startswith("files="):
                                info["files"] = int(part.split("=", 1)[1])
                            if part.startswith("lines="):
                                info["lines"] = int(part.split("=", 1)[1])
            except OSError as exc:
                info["note"] = f"manifest_read_error: {exc}"

        return info
    except Exception as e:
        return {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_bidirectional_adapters() -> dict[str, Any]:
    """Check bidirectional adapter availability.

    Returns:
        Dict with adapter availability status.
    """
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

        return {
            "healthy": True,
            "status": "available",
            "adapters_available": len(adapter_classes),
            "adapter_list": [name for name, _ in adapter_classes],
        }
    except ImportError as e:
        return {
            "healthy": True,
            "status": "partial",
            "error": f"Some adapters not available: {str(e)[:80]}",
        }
    except Exception as e:
        return {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_control_plane_adapter() -> dict[str, Any]:
    """Check Control Plane adapter availability.

    Returns:
        Dict with Control Plane adapter health status.
    """
    try:
        from aragora.knowledge.mound.adapters.control_plane_adapter import (  # noqa: F401
            ControlPlaneAdapter,
            TaskOutcome,
            AgentCapabilityRecord,
            CrossWorkspaceInsight,
        )

        return {
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
        return {
            "healthy": True,
            "status": "not_available",
            "error": str(e)[:80],
        }
    except Exception as e:
        return {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_km_metrics() -> dict[str, Any]:
    """Check Knowledge Mound metrics availability.

    Returns:
        Dict with KM metrics/Prometheus integration status.
    """
    try:
        from aragora.observability.metrics.km import (  # noqa: F401
            init_km_metrics,
            record_km_operation,
            record_cp_task_outcome,
        )

        return {
            "healthy": True,
            "status": "available",
            "prometheus_integration": True,
        }
    except ImportError:
        return {
            "healthy": True,
            "status": "not_available",
            "prometheus_integration": False,
        }
    except Exception as e:
        return {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }


def check_confidence_decay_scheduler() -> tuple[dict[str, Any], list[str]]:
    """Check Confidence Decay Scheduler status.

    Returns:
        Tuple of (component_dict, warnings_list):
        - component_dict: Health check result for confidence decay
        - warnings_list: List of warning messages to add to response
    """
    warnings: list[str] = []

    try:
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            get_decay_scheduler,
        )

        scheduler = get_decay_scheduler()
        if scheduler:
            stats = scheduler.get_stats()
            result: dict[str, Any] = {
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
                        result["alert"] = {
                            "level": "warning",
                            "message": f"Decay not run in >48h for: {', '.join(stale_workspaces)}",
                        }
                        warnings.append(
                            f"Confidence decay stale for workspaces: {stale_workspaces}"
                        )

            return result, warnings
        else:
            return {
                "healthy": True,
                "status": "not_configured",
                "note": "Confidence decay scheduler not initialized",
            }, warnings

    except ImportError:
        return {
            "healthy": True,
            "status": "not_available",
            "note": "Confidence decay module not installed",
        }, warnings
    except Exception as e:
        return {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }, warnings
