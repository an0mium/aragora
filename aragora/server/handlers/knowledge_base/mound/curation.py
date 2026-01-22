"""
Curation operations mixin for Knowledge Mound.

Provides endpoints for auto-curation management:
- GET /api/knowledge/mound/curation/policy - Get curation policy
- POST /api/knowledge/mound/curation/policy - Set curation policy
- GET /api/knowledge/mound/curation/status - Get curation status
- POST /api/knowledge/mound/curation/run - Trigger curation run
- GET /api/knowledge/mound/curation/history - Get curation history
- GET /api/knowledge/mound/curation/scores - Get quality scores
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from aragora.server.http_utils import run_async as _run_async

from ...base import HandlerResult, error_response, json_response, require_auth

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class CurationOperationsMixin:
    """Mixin providing curation management endpoints."""

    _mound: Optional["KnowledgeMound"]

    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """Abstract method - implemented by main handler."""
        raise NotImplementedError

    def _handle_curation_routes(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Route curation-related requests."""
        if path == "/api/knowledge/mound/curation/policy":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_set_curation_policy(handler)
            return self._handle_get_curation_policy(query_params)

        if path == "/api/knowledge/mound/curation/status":
            return self._handle_curation_status(query_params)

        if path == "/api/knowledge/mound/curation/run":
            return self._handle_run_curation(handler)

        if path == "/api/knowledge/mound/curation/history":
            return self._handle_curation_history(query_params)

        if path == "/api/knowledge/mound/curation/scores":
            return self._handle_quality_scores(query_params)

        if path == "/api/knowledge/mound/curation/tiers":
            return self._handle_tier_distribution(query_params)

        return None

    def _handle_get_curation_policy(self, query_params: dict) -> HandlerResult:
        """Get current curation policy for a workspace."""
        workspace_id = query_params.get("workspace_id", "default")

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            from aragora.knowledge.mound.ops.auto_curation import CurationPolicy

            # Try to get policy from mound if method exists
            if hasattr(mound, "get_curation_policy"):
                policy = _run_async(mound.get_curation_policy(workspace_id))
                if policy:
                    return json_response(
                        {
                            "workspace_id": workspace_id,
                            "policy": {
                                "policy_id": policy.policy_id,
                                "enabled": policy.enabled,
                                "name": policy.name,
                                "quality_threshold": policy.quality_threshold,
                                "promotion_threshold": policy.promotion_threshold,
                                "demotion_threshold": policy.demotion_threshold,
                                "archive_threshold": policy.archive_threshold,
                                "refresh_staleness_threshold": policy.refresh_staleness_threshold,
                                "usage_window_days": policy.usage_window_days,
                                "min_retrievals_for_promotion": policy.min_retrievals_for_promotion,
                            },
                        }
                    )

            # Return default policy
            default_policy = CurationPolicy(workspace_id=workspace_id)
            return json_response(
                {
                    "workspace_id": workspace_id,
                    "policy": {
                        "policy_id": default_policy.policy_id,
                        "enabled": default_policy.enabled,
                        "name": default_policy.name,
                        "quality_threshold": default_policy.quality_threshold,
                        "promotion_threshold": default_policy.promotion_threshold,
                        "demotion_threshold": default_policy.demotion_threshold,
                        "archive_threshold": default_policy.archive_threshold,
                        "refresh_staleness_threshold": default_policy.refresh_staleness_threshold,
                        "usage_window_days": default_policy.usage_window_days,
                        "min_retrievals_for_promotion": default_policy.min_retrievals_for_promotion,
                    },
                    "note": "Using default policy (no custom policy set)",
                }
            )

        except ImportError:
            return error_response("Curation module not available", 501)
        except (ValueError, TypeError, KeyError) as e:
            logger.exception("Error getting curation policy: %s", e)
            return error_response("Failed to get curation policy", 500)

    @require_auth
    def _handle_set_curation_policy(self, handler: Any) -> HandlerResult:
        """Set curation policy for a workspace."""
        from ...base import read_json_body  # type: ignore[attr-defined]

        body = read_json_body(handler)
        if body is None:
            return error_response("JSON body required", 400)

        workspace_id = body.get("workspace_id", "default")

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            from aragora.knowledge.mound.ops.auto_curation import CurationPolicy

            # Build policy from request body
            policy = CurationPolicy(
                workspace_id=workspace_id,
                enabled=body.get("enabled", True),
                name=body.get("name", "custom"),
                quality_threshold=body.get("quality_threshold", 0.5),
                promotion_threshold=body.get("promotion_threshold", 0.85),
                demotion_threshold=body.get("demotion_threshold", 0.35),
                archive_threshold=body.get("archive_threshold", 0.2),
                refresh_staleness_threshold=body.get("refresh_staleness_threshold", 0.7),
                usage_window_days=body.get("usage_window_days", 30),
                min_retrievals_for_promotion=body.get("min_retrievals_for_promotion", 5),
            )

            # Set policy if method exists
            if hasattr(mound, "set_curation_policy"):
                _run_async(mound.set_curation_policy(policy))
                logger.info(
                    "Curation policy set for workspace %s: %s",
                    workspace_id,
                    policy.policy_id,
                )
                return json_response(
                    {
                        "success": True,
                        "workspace_id": workspace_id,
                        "policy_id": policy.policy_id,
                        "message": "Curation policy updated",
                    }
                )

            return json_response(
                {
                    "success": True,
                    "workspace_id": workspace_id,
                    "policy_id": policy.policy_id,
                    "note": "Policy validated but storage not available",
                }
            )

        except ImportError:
            return error_response("Curation module not available", 501)
        except (ValueError, TypeError, KeyError) as e:
            logger.exception("Error setting curation policy: %s", e)
            return error_response(f"Failed to set curation policy: {e}", 500)

    def _handle_curation_status(self, query_params: dict) -> HandlerResult:
        """Get current curation status for a workspace."""
        workspace_id = query_params.get("workspace_id", "default")

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            status = {
                "workspace_id": workspace_id,
                "last_run": None,
                "next_scheduled": None,
                "is_running": False,
                "stats": {
                    "total_items": 0,
                    "items_scored": 0,
                    "items_pending": 0,
                },
            }

            # Get status from mound if method exists
            if hasattr(mound, "get_curation_status"):
                mound_status = _run_async(mound.get_curation_status(workspace_id))
                if mound_status:
                    status.update(mound_status)

            # Get item count from mound
            if hasattr(mound, "get_node_count"):
                status["stats"]["total_items"] = _run_async(mound.get_node_count(workspace_id))

            return json_response(status)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.exception("Error getting curation status: %s", e)
            return error_response("Failed to get curation status", 500)

    @require_auth
    def _handle_run_curation(self, handler: Any) -> HandlerResult:
        """Trigger a curation run for a workspace."""
        from ...base import read_json_body  # type: ignore[attr-defined]

        body = read_json_body(handler) or {}
        workspace_id = body.get("workspace_id", "default")
        dry_run = body.get("dry_run", False)
        limit = body.get("limit", 100)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result = {
                "workspace_id": workspace_id,
                "dry_run": dry_run,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "promoted": 0,
                "demoted": 0,
                "archived": 0,
                "refreshed": 0,
                "flagged": 0,
            }

            # Run curation if method exists
            if hasattr(mound, "run_curation"):
                curation_result = _run_async(
                    mound.run_curation(
                        workspace_id=workspace_id,
                        dry_run=dry_run,
                        limit=limit,
                    )
                )
                if curation_result:
                    result.update(
                        {
                            "promoted": getattr(curation_result, "promoted_count", 0),
                            "demoted": getattr(curation_result, "demoted_count", 0),
                            "archived": getattr(curation_result, "archived_count", 0),
                            "refreshed": getattr(curation_result, "refreshed_count", 0),
                            "flagged": getattr(curation_result, "flagged_count", 0),
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    logger.info(
                        "Curation run completed for workspace %s: promoted=%d, demoted=%d",
                        workspace_id,
                        result["promoted"],
                        result["demoted"],
                    )
                    return json_response(result)

            # Curation not available, return simulated result
            result["note"] = "Curation engine not available, no actions taken"
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            return json_response(result)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.exception("Error running curation: %s", e)
            return error_response(f"Failed to run curation: {e}", 500)

    def _handle_curation_history(self, query_params: dict) -> HandlerResult:
        """Get curation history for a workspace."""
        workspace_id = query_params.get("workspace_id", "default")
        limit = int(query_params.get("limit", 20))
        limit = max(1, min(100, limit))

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            history = []

            # Get history from mound if method exists
            if hasattr(mound, "get_curation_history"):
                history = _run_async(mound.get_curation_history(workspace_id, limit=limit))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "history": history,
                    "count": len(history),
                }
            )

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.exception("Error getting curation history: %s", e)
            return error_response("Failed to get curation history", 500)

    def _handle_quality_scores(self, query_params: dict) -> HandlerResult:
        """Get quality scores for knowledge items."""
        workspace_id = query_params.get("workspace_id", "default")
        limit = int(query_params.get("limit", 50))
        limit = max(1, min(200, limit))
        min_score = float(query_params.get("min_score", 0.0))
        max_score = float(query_params.get("max_score", 1.0))

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            scores = []

            # Get scores from mound if method exists
            if hasattr(mound, "get_quality_scores"):
                raw_scores = _run_async(
                    mound.get_quality_scores(
                        workspace_id,
                        limit=limit,
                        min_score=min_score,
                        max_score=max_score,
                    )
                )
                for score in raw_scores:
                    scores.append(
                        {
                            "node_id": score.node_id,
                            "overall_score": score.overall_score,
                            "freshness_score": score.freshness_score,
                            "confidence_score": score.confidence_score,
                            "usage_score": score.usage_score,
                            "relevance_score": score.relevance_score,
                            "relationship_score": score.relationship_score,
                            "recommendation": score.recommendation.value,
                            "debate_uses": score.debate_uses,
                            "retrieval_count": score.retrieval_count,
                        }
                    )

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "scores": scores,
                    "count": len(scores),
                    "filters": {
                        "min_score": min_score,
                        "max_score": max_score,
                        "limit": limit,
                    },
                }
            )

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.exception("Error getting quality scores: %s", e)
            return error_response("Failed to get quality scores", 500)

    def _handle_tier_distribution(self, query_params: dict) -> HandlerResult:
        """Get tier distribution for knowledge items."""
        workspace_id = query_params.get("workspace_id", "default")

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            from aragora.knowledge.mound.ops.auto_curation import TierLevel

            distribution = {
                TierLevel.HOT.value: 0,
                TierLevel.WARM.value: 0,
                TierLevel.COLD.value: 0,
                TierLevel.GLACIAL.value: 0,
            }

            # Get distribution from mound if method exists
            if hasattr(mound, "get_tier_distribution"):
                distribution = _run_async(mound.get_tier_distribution(workspace_id))

            total = sum(distribution.values())

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "distribution": distribution,
                    "total": total,
                    "percentages": {
                        tier: round((count / total) * 100, 1) if total > 0 else 0
                        for tier, count in distribution.items()
                    },
                }
            )

        except ImportError:
            return error_response("Curation module not available", 501)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.exception("Error getting tier distribution: %s", e)
            return error_response("Failed to get tier distribution", 500)
