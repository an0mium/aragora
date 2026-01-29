# mypy: ignore-errors
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
from typing import Any

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)

def cross_pollination_health(handler) -> HandlerResult:
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
    except Exception as e:
        features["elo_weighting"] = {
            "healthy": False,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
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
    except Exception as e:
        features["calibration"] = {
            "healthy": False,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
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
    except Exception as e:
        features["evidence_quality"] = {
            "healthy": False,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
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
    except Exception as e:
        features["rlm_caching"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # Check Knowledge Mound
    try:
        from aragora.debate.knowledge_mound_ops import get_knowledge_mound_stats  # type: ignore[attr-defined]

        km_stats = get_knowledge_mound_stats()
        features["knowledge_mound"] = {
            "healthy": True,
            "status": "active",
            "facts_count": km_stats.get("facts_count", 0),
            "consensus_stored": km_stats.get("consensus_stored", 0),
        }
    except ImportError:
        features["knowledge_mound"] = {
            "healthy": True,
            "status": "not_available",
            "note": "Knowledge Mound module not available",
        }
    except Exception as e:
        features["knowledge_mound"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
        }

    # Check trending topics (Pulse)
    try:
        from aragora.pulse import get_pulse_stats  # type: ignore[attr-defined]

        pulse_stats = get_pulse_stats()
        features["trending_topics"] = {
            "healthy": True,
            "status": "active",
            "topics_tracked": pulse_stats.get("topics_count", 0),
        }
    except ImportError:
        features["trending_topics"] = {
            "healthy": True,
            "status": "not_available",
            "note": "Pulse module not available",
        }
    except Exception as e:
        features["trending_topics"] = {
            "healthy": True,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:80]}",
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
