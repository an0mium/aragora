"""
Cross-pollination health check implementations.

Provides health checks for cross-pollination feature integrations:
- ELO skill weighting
- Calibration tracking
- Evidence quality scoring
- RLM hierarchy caching
- Knowledge Mound operations
- Trending topics (Pulse)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)


@runtime_checkable
class _EloSystem(Protocol):
    """Protocol for ELO system interface."""

    def get_leaderboard(self, limit: int = 10) -> list[Any]: ...


@runtime_checkable
class _HandlerWithContext(Protocol):
    """Protocol for handler with context and ELO system."""

    ctx: dict[str, Any]

    def get_elo_system(self) -> _EloSystem | None: ...


def cross_pollination_health(handler: _HandlerWithContext) -> HandlerResult:
    """Check health of cross-pollination feature integrations.

    Returns status of:
    - ELO skill weighting integration
    - Calibration tracking
    - Evidence quality scoring
    - RLM hierarchy caching
    - Knowledge Mound operations

    This endpoint is useful for:
    - Verifying cross-pollination features are operational
    - Debugging feature integration issues
    - Monitoring feature-level health

    Returns:
        JSON response with cross-pollination health metrics
    """
    features: dict[str, dict[str, Any]] = {}
    all_healthy = True

    # Check ELO system for skill weighting
    try:
        elo = handler.get_elo_system()
        if elo is not None:
            # Check if ELO has domain ratings
            leaderboard = elo.get_leaderboard(limit=1)
            features["elo_weighting"] = {
                "healthy": True,
                "status": "active",
                "agents_tracked": len(leaderboard) if leaderboard else 0,
            }
        else:
            features["elo_weighting"] = {
                "healthy": True,
                "status": "not_configured",
                "note": "ELO system not initialized",
            }
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
        logger.warning("ELO weighting health check failed: %s", e)
        features["elo_weighting"] = {
            "healthy": False,
            "status": "error",
            "error": "Health check failed",
        }
        all_healthy = False

    # Check calibration tracker
    try:
        calibration = handler.ctx.get("calibration_tracker")
        if calibration is not None:
            features["calibration"] = {
                "healthy": True,
                "status": "active",
            }
            # Try to get calibration stats
            try:
                if hasattr(calibration, "get_calibration_stats"):
                    stats = calibration.get_calibration_stats()
                    features["calibration"]["tracked_agents"] = stats.get("tracked_agents", 0)
            except (AttributeError, KeyError):
                pass
        else:
            features["calibration"] = {
                "healthy": True,
                "status": "not_configured",
                "note": "Calibration tracker not initialized",
            }
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
        logger.warning("Calibration health check failed: %s", e)
        features["calibration"] = {
            "healthy": False,
            "status": "error",
            "error": "Health check failed",
        }
        all_healthy = False

    # Check evidence quality scoring
    try:
        evidence_store = handler.ctx.get("evidence_store")
        if evidence_store is not None:
            features["evidence_quality"] = {
                "healthy": True,
                "status": "active",
            }
        else:
            features["evidence_quality"] = {
                "healthy": True,
                "status": "not_configured",
                "note": "Evidence store not initialized",
            }
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
        logger.warning("Evidence quality health check failed: %s", e)
        features["evidence_quality"] = {
            "healthy": False,
            "status": "error",
            "error": "Health check failed",
        }
        all_healthy = False

    # Check RLM hierarchy caching
    try:
        from aragora.rlm.cache import get_rlm_cache_stats

        cache_stats = get_rlm_cache_stats()
        hits = cache_stats.get("hits", 0)
        misses = cache_stats.get("misses", 0)
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0

        features["rlm_caching"] = {
            "healthy": True,
            "status": "active",
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_rate": round(hit_rate, 3),
        }
    except ImportError:
        features["rlm_caching"] = {
            "healthy": True,
            "status": "not_available",
            "note": "RLM cache module not available",
        }
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError) as e:
        logger.warning("RLM caching health check failed: %s", e)
        features["rlm_caching"] = {
            "healthy": True,
            "status": "error",
            "error": "Health check failed",
        }

    # Check Knowledge Mound
    try:
        from aragora.debate import knowledge_mound_ops

        get_stats_fn = getattr(knowledge_mound_ops, "get_knowledge_mound_stats", None)
        if get_stats_fn is not None:
            km_stats: dict[str, Any] = get_stats_fn()
            features["knowledge_mound"] = {
                "healthy": True,
                "status": "active",
                "facts_count": km_stats.get("facts_count", 0),
                "consensus_stored": km_stats.get("consensus_stored", 0),
            }
        else:
            features["knowledge_mound"] = {
                "healthy": True,
                "status": "partial",
                "note": "KnowledgeMoundOperations class available but stats function not found",
            }
    except ImportError:
        features["knowledge_mound"] = {
            "healthy": True,
            "status": "not_available",
            "note": "Knowledge Mound module not available",
        }
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError) as e:
        logger.warning("Knowledge mound health check failed: %s", e)
        features["knowledge_mound"] = {
            "healthy": True,
            "status": "error",
            "error": "Health check failed",
        }

    # Check trending topics (Pulse)
    try:
        from aragora import pulse

        get_stats_fn = getattr(pulse, "get_pulse_stats", None)
        if get_stats_fn is not None:
            pulse_stats: dict[str, Any] = get_stats_fn()
            features["trending_topics"] = {
                "healthy": True,
                "status": "active",
                "topics_tracked": pulse_stats.get("topics_count", 0),
            }
        else:
            features["trending_topics"] = {
                "healthy": True,
                "status": "partial",
                "note": "Pulse module available but stats function not found",
            }
    except ImportError:
        features["trending_topics"] = {
            "healthy": True,
            "status": "not_available",
            "note": "Pulse module not available",
        }
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, OSError) as e:
        logger.warning("Trending topics health check failed: %s", e)
        features["trending_topics"] = {
            "healthy": True,
            "status": "error",
            "error": "Health check failed",
        }

    # Count active features
    active_count = sum(1 for f in features.values() if f.get("status") == "active")
    total_features = len(features)

    status = "healthy" if all_healthy else "degraded"
    if active_count == 0:
        status = "not_configured"

    return json_response(
        {
            "status": status,
            "active_features": active_count,
            "total_features": total_features,
            "features": features,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
    )
