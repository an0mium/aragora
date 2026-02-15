"""Dashboard action and analytics endpoint methods (mixin).

Contains write operations (quick actions, dismiss, complete) and analytics
endpoints (search, export, quality/calibration/performance/evolution metrics).

Extracted from dashboard.py for maintainability.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from ..base import (
    HandlerResult,
    error_response,
    json_response,
    ttl_cache,
)
from ..openapi_decorator import api_endpoint

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DashboardActionsMixin:
    """Mixin providing dashboard action and analytics endpoints.

    Requires the host class to provide:
    - get_storage() -> storage instance
    - ctx: dict with calibration_tracker, performance_monitor, etc.
    - _get_summary_metrics_sql(storage, domain) -> dict
    - _get_agent_performance(limit) -> dict
    - _get_consensus_insights(domain) -> dict
    """

    if TYPE_CHECKING:

        def get_storage(self) -> Any: ...

    ctx: dict[str, Any]

    def _get_summary_metrics_sql(self, storage: Any, domain: str | None) -> dict[str, Any]: ...
    def _get_agent_performance(self, limit: int) -> dict[str, Any]: ...
    def _get_consensus_insights(self, domain: str | None) -> dict[str, Any]: ...

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/quick-actions",
        summary="Get available quick actions",
        tags=["Dashboard"],
        responses={
            "200": {"description": "List of available quick actions"},
            "401": {"description": "Unauthorized"},
        },
    )
    def _get_quick_actions(self) -> HandlerResult:
        """Return quick actions list."""
        actions = [
            {
                "id": "archive_read",
                "name": "Archive All Read",
                "description": "Archive all read emails older than 24 hours",
                "icon": "archive",
                "available": True,
            },
            {
                "id": "snooze_low",
                "name": "Snooze Low Priority",
                "description": "Snooze all low priority emails until tomorrow",
                "icon": "clock",
                "available": True,
            },
            {
                "id": "mark_spam",
                "name": "Mark Bulk as Spam",
                "description": "Mark selected promotional emails as spam",
                "icon": "slash",
                "available": True,
            },
            {
                "id": "complete_actions",
                "name": "Complete Done Actions",
                "description": "Mark action items you've completed",
                "icon": "check-circle",
                "available": True,
            },
            {
                "id": "ai_respond",
                "name": "AI Auto-Respond",
                "description": "Let AI draft responses for simple emails",
                "icon": "sparkles",
                "available": True,
            },
            {
                "id": "sync_inbox",
                "name": "Sync Inbox",
                "description": "Force sync with email provider",
                "icon": "refresh",
                "available": True,
            },
        ]
        return json_response({"actions": actions, "total": len(actions)})

    @api_endpoint(
        method="POST",
        path="/api/v1/dashboard/quick-actions/{action_id}",
        summary="Execute a quick action",
        tags=["Dashboard"],
        parameters=[
            {"name": "action_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        responses={
            "200": {"description": "Action executed"},
            "401": {"description": "Unauthorized"},
            "404": {"description": "Action not found"},
        },
    )
    def _execute_quick_action(self, action_id: str) -> HandlerResult:
        """Execute a quick action (stub)."""
        if not action_id:
            return error_response("action_id is required", 400)
        return json_response(
            {
                "success": True,
                "action_id": action_id,
                "executed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/urgent",
        summary="Get urgent items",
        tags=["Dashboard"],
        parameters=[
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
        ],
        responses={
            "200": {"description": "Urgent items requiring attention"},
            "401": {"description": "Unauthorized"},
        },
    )
    def _get_urgent_items(self, limit: int, offset: int) -> HandlerResult:
        """Return urgent items: debates with low confidence or no consensus."""
        items: list[dict[str, Any]] = []

        try:
            storage = self.get_storage()
            if storage:
                with storage.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id, domain, confidence, created_at FROM debates "
                        "WHERE consensus_reached = 0 OR confidence < 0.3 "
                        "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    )
                    for row in cursor.fetchall():
                        items.append(
                            {
                                "id": row[0],
                                "type": "low_consensus",
                                "domain": row[1],
                                "confidence": row[2],
                                "created_at": row[3],
                                "description": f"Debate in {row[1] or 'general'} needs attention",
                            }
                        )
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Urgent items error: %s: %s", type(e).__name__, e)

        return json_response({"items": items, "total": len(items)})

    @api_endpoint(
        method="POST",
        path="/api/v1/dashboard/urgent/{item_id}/dismiss",
        summary="Dismiss an urgent item",
        tags=["Dashboard"],
        parameters=[
            {"name": "item_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        responses={
            "200": {"description": "Item dismissed"},
            "401": {"description": "Unauthorized"},
        },
    )
    def _dismiss_urgent_item(self, item_id: str) -> HandlerResult:
        """Dismiss an urgent item by marking it as reviewed."""
        if not item_id:
            return error_response("item_id is required", 400)
        try:
            storage = self.get_storage()
            if storage:
                with storage.connection() as conn:
                    cursor = conn.cursor()
                    # Mark as reviewed by setting confidence to indicate human review
                    cursor.execute(
                        "UPDATE debates SET consensus_reached = 1 WHERE id = ?",
                        (item_id,),
                    )
                    conn.commit()
                    if cursor.rowcount == 0:
                        return error_response("Item not found", 404)
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Dismiss urgent item error: %s: %s", type(e).__name__, e)
            return error_response("Failed to dismiss item", 500)
        return json_response(
            {
                "success": True,
                "item_id": item_id,
                "dismissed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/pending-actions",
        summary="Get pending actions",
        tags=["Dashboard"],
        parameters=[
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
        ],
        responses={
            "200": {"description": "Actions awaiting completion"},
            "401": {"description": "Unauthorized"},
        },
    )
    def _get_pending_actions(self, limit: int, offset: int) -> HandlerResult:
        """Return pending actions: recent debates awaiting review."""
        actions: list[dict[str, Any]] = []

        try:
            storage = self.get_storage()
            if storage:
                with storage.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id, domain, created_at FROM debates "
                        "WHERE status = 'pending' OR status = 'in_progress' "
                        "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    )
                    for row in cursor.fetchall():
                        actions.append(
                            {
                                "id": row[0],
                                "type": "review_debate",
                                "domain": row[1],
                                "created_at": row[2],
                                "description": f"Review debate in {row[1] or 'general'}",
                            }
                        )
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Pending actions error: %s: %s", type(e).__name__, e)

        return json_response({"actions": actions, "total": len(actions)})

    @api_endpoint(
        method="POST",
        path="/api/v1/dashboard/pending-actions/{action_id}/complete",
        summary="Complete a pending action",
        tags=["Dashboard"],
        parameters=[
            {"name": "action_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        responses={
            "200": {"description": "Action completed"},
            "401": {"description": "Unauthorized"},
            "404": {"description": "Action not found"},
        },
    )
    def _complete_pending_action(self, action_id: str) -> HandlerResult:
        """Complete a pending action by updating its status."""
        if not action_id:
            return error_response("action_id is required", 400)
        try:
            storage = self.get_storage()
            if storage:
                with storage.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE debates SET status = 'completed' WHERE id = ? AND (status = 'pending' OR status = 'in_progress')",
                        (action_id,),
                    )
                    conn.commit()
                    if cursor.rowcount == 0:
                        return error_response("Action not found or already completed", 404)
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Complete action error: %s: %s", type(e).__name__, e)
            return error_response("Failed to complete action", 500)
        return json_response(
            {
                "success": True,
                "action_id": action_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/search",
        summary="Search dashboard",
        tags=["Dashboard"],
        parameters=[
            {"name": "q", "in": "query", "schema": {"type": "string"}, "required": True},
        ],
        responses={
            "200": {"description": "Search results"},
            "401": {"description": "Unauthorized"},
        },
    )
    def _search_dashboard(self, query: str) -> HandlerResult:
        """Search dashboard data by domain or debate ID."""
        results: list[dict[str, Any]] = []

        if not query:
            return json_response({"results": [], "total": 0})

        try:
            storage = self.get_storage()
            if storage:
                with storage.connection() as conn:
                    cursor = conn.cursor()
                    like_query = f"%{query}%"
                    cursor.execute(
                        "SELECT id, domain, consensus_reached, confidence, "
                        "created_at FROM debates "
                        "WHERE id LIKE ? OR domain LIKE ? "
                        "ORDER BY created_at DESC LIMIT 20",
                        (like_query, like_query),
                    )
                    for row in cursor.fetchall():
                        results.append(
                            {
                                "id": row[0],
                                "domain": row[1],
                                "consensus_reached": bool(row[2]),
                                "confidence": row[3],
                                "created_at": row[4],
                            }
                        )
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Dashboard search error: %s: %s", type(e).__name__, e)

        return json_response({"results": results, "total": len(results)})

    @api_endpoint(
        method="POST",
        path="/api/v1/dashboard/export",
        summary="Export dashboard data",
        tags=["Dashboard"],
        responses={
            "200": {"description": "Dashboard data exported"},
            "401": {"description": "Unauthorized"},
        },
    )
    def _export_dashboard_data(self) -> HandlerResult:
        """Export dashboard data as a JSON snapshot."""
        export: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {},
            "agent_performance": {},
            "consensus_insights": {},
        }

        try:
            storage = self.get_storage()
            if storage:
                export["summary"] = self._get_summary_metrics_sql(storage, None)
            export["agent_performance"] = self._get_agent_performance(50)
            export["consensus_insights"] = self._get_consensus_insights(None)
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Export error: %s: %s", type(e).__name__, e)

        return json_response(export)

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/quality-metrics",
        summary="Get quality metrics",
        tags=["Dashboard"],
        responses={
            "200": {"description": "Debate quality metrics"},
            "401": {"description": "Unauthorized"},
        },
    )
    @ttl_cache(ttl_seconds=60, key_prefix="quality_metrics", skip_first=True)
    def _get_quality_metrics(self) -> HandlerResult:
        """Get unified quality metrics across all subsystems."""
        result: dict[str, Any] = {
            "calibration": {},
            "performance": {},
            "evolution": {},
            "debate_quality": {},
            "generated_at": time.time(),
        }

        result["calibration"] = self._get_calibration_metrics()
        result["performance"] = self._get_performance_metrics()
        result["evolution"] = self._get_evolution_metrics()
        result["debate_quality"] = self._get_debate_quality_metrics()

        return json_response(result)

    def _get_calibration_metrics(self) -> dict[str, Any]:
        """Get comprehensive agent calibration metrics."""
        metrics: dict[str, Any] = {
            "agents": {},
            "overall_calibration": 0.0,
            "overconfident_agents": [],
            "underconfident_agents": [],
            "well_calibrated_agents": [],
            "top_by_brier": [],
            "calibration_curves": {},
            "domain_breakdown": {},
        }

        try:
            calibration_tracker = self.ctx.get("calibration_tracker")
            if not calibration_tracker:
                return metrics

            summary = calibration_tracker.get_calibration_summary()
            if summary:
                metrics["agents"] = summary.get("agents", {})
                metrics["overall_calibration"] = summary.get("overall", 0.0)

                # Categorize agents by calibration bias
                agent_brier_scores: list[tuple[str, float]] = []

                for agent, data in metrics["agents"].items():
                    bias = data.get("calibration_bias", 0)
                    brier = data.get("brier_score", 1.0)
                    agent_brier_scores.append((agent, brier))

                    if bias > 0.1:
                        metrics["overconfident_agents"].append(agent)
                    elif bias < -0.1:
                        metrics["underconfident_agents"].append(agent)
                    else:
                        metrics["well_calibrated_agents"].append(agent)

                # Get top agents by Brier score (lower is better)
                agent_brier_scores.sort(key=lambda x: x[1])
                metrics["top_by_brier"] = [
                    {"agent": agent, "brier_score": round(brier, 3)}
                    for agent, brier in agent_brier_scores[:5]
                ]

            # Get calibration curves for top 3 agents
            all_agents = calibration_tracker.get_all_agents()
            for agent in all_agents[:3]:
                try:
                    curve = calibration_tracker.get_calibration_curve(agent, num_buckets=10)
                    if curve:
                        metrics["calibration_curves"][agent] = [
                            {
                                "bucket": i,
                                "confidence_range": f"{bucket.range_start:.1f}-{bucket.range_end:.1f}",
                                "expected_accuracy": bucket.expected_accuracy,
                                "actual_accuracy": bucket.accuracy,
                                "count": bucket.total_predictions,
                            }
                            for i, bucket in enumerate(curve)
                        ]
                except (KeyError, ValueError, TypeError, AttributeError) as e:
                    logger.debug("Calibration curve error for %s: %s", agent, e)

            # Get domain breakdown for agents with sufficient data
            for agent in all_agents[:5]:
                try:
                    domain_data = calibration_tracker.get_domain_breakdown(agent)
                    if domain_data:
                        metrics["domain_breakdown"][agent] = {
                            domain: {
                                "predictions": s.total_predictions,
                                "accuracy": round(s.accuracy, 3),
                                "brier_score": round(s.brier_score, 3),
                                "ece": round(s.ece, 3) if hasattr(s, "ece") else None,
                            }
                            for domain, s in domain_data.items()
                        }
                except (KeyError, ValueError, TypeError, AttributeError) as e:
                    logger.debug("Domain breakdown error for %s: %s", agent, e)

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Calibration metrics error: %s", e)

        return metrics

    def _get_performance_metrics(self) -> dict[str, Any]:
        """Get agent performance metrics."""
        metrics: dict[str, Any] = {
            "agents": {},
            "avg_latency_ms": 0.0,
            "success_rate": 0.0,
            "total_calls": 0,
        }

        try:
            performance_monitor = self.ctx.get("performance_monitor")
            if performance_monitor:
                insights = cast(Any, performance_monitor).get_performance_insights()
                if insights:
                    metrics["agents"] = insights.get("agents", {})
                    metrics["avg_latency_ms"] = insights.get("avg_latency_ms", 0.0)
                    metrics["success_rate"] = insights.get("success_rate", 0.0)
                    metrics["total_calls"] = insights.get("total_calls", 0)
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Performance metrics error: %s", e)

        return metrics

    def _get_evolution_metrics(self) -> dict[str, Any]:
        """Get prompt evolution progress."""
        metrics: dict[str, Any] = {
            "agents": {},
            "total_versions": 0,
            "patterns_extracted": 0,
            "last_evolution": None,
        }

        try:
            prompt_evolver = self.ctx.get("prompt_evolver")
            if prompt_evolver:
                # Get version counts per agent
                for agent_name in ["claude", "gemini", "codex", "grok"]:
                    try:
                        version = cast(Any, prompt_evolver).get_prompt_version(agent_name)
                        if version:
                            metrics["agents"][agent_name] = {
                                "current_version": version.version,
                                "performance_score": version.performance_score,
                                "debates_count": version.debates_count,
                            }
                            metrics["total_versions"] += version.version
                    except (AttributeError, KeyError) as e:
                        logger.debug("Skipping agent version with missing data: %s", e)

                # Get pattern count
                patterns = cast(Any, prompt_evolver).get_top_patterns(limit=100)
                metrics["patterns_extracted"] = len(patterns) if patterns else 0
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Evolution metrics error: %s", e)

        return metrics

    def _get_debate_quality_metrics(self) -> dict[str, Any]:
        """Get debate quality scores."""
        metrics: dict[str, Any] = {
            "avg_confidence": 0.0,
            "consensus_rate": 0.0,
            "avg_rounds": 0.0,
            "evidence_quality": 0.0,
            "recent_winners": [],
        }

        try:
            # Get from ELO system
            elo_system = self.ctx.get("elo_system")
            if elo_system:
                recent = elo_system.get_recent_matches(limit=10)
                if recent:
                    winners = [m.get("winner") for m in recent if m.get("winner")]
                    metrics["recent_winners"] = winners[:5]

                    # Calculate avg confidence from matches
                    confidences = [m.get("confidence", 0) for m in recent if m.get("confidence")]
                    if confidences:
                        metrics["avg_confidence"] = sum(confidences) / len(confidences)

            # Get from storage
            storage = self.get_storage()
            if storage:
                summary = self._get_summary_metrics_sql(storage, None)
                if summary:
                    metrics["consensus_rate"] = summary.get("consensus_rate", 0.0)
                    metrics["avg_rounds"] = summary.get("avg_rounds", 0.0)

        except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
            logger.warning("Debate quality metrics error: %s", e)

        return metrics
