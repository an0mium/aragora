"""Self-improvement transparency dashboard endpoints.

Endpoints:
- GET /api/self-improve/meta-planner/goals      - MetaPlanner prioritized goals
- GET /api/self-improve/execution/timeline       - Branch execution timeline
- GET /api/self-improve/learning/insights        - Cross-cycle learning data
- GET /api/self-improve/metrics/comparison       - Before/after codebase metrics

These endpoints expose the internal state of the Nomic Loop self-improvement
system so the frontend dashboard can visualize what the system is doing,
why it chose specific goals, and how it learns over time.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    HandlerResult,
    error_response,
    json_response,
    handle_errors,
)
from .secure import SecureHandler
from .utils.auth_mixins import SecureEndpointMixin, require_permission
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


class SelfImproveDetailsHandler(SecureEndpointMixin, SecureHandler):  # type: ignore[misc]
    """Handler for self-improvement transparency dashboard.

    Exposes MetaPlanner goals, branch execution state, cross-cycle
    learning insights, and before/after metrics comparison.

    RBAC Permissions:
    - self_improve:read - View goals, timeline, insights, metrics
    """

    RESOURCE_TYPE = "self_improve"

    ROUTES = [
        "/api/self-improve/meta-planner/goals",
        "/api/self-improve/execution/timeline",
        "/api/self-improve/learning/insights",
        "/api/self-improve/metrics/comparison",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        path = strip_version_prefix(path)
        return path in self.ROUTES

    @rate_limit(requests_per_minute=30)
    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route GET requests to the appropriate endpoint."""
        path = strip_version_prefix(path)

        handlers = {
            "/api/self-improve/meta-planner/goals": self._get_meta_planner_goals,
            "/api/self-improve/execution/timeline": self._get_execution_timeline,
            "/api/self-improve/learning/insights": self._get_learning_insights,
            "/api/self-improve/metrics/comparison": self._get_metrics_comparison,
        }

        endpoint_handler = handlers.get(path)
        if endpoint_handler:
            return await endpoint_handler()

        return None

    async def _get_meta_planner_goals(self) -> HandlerResult:
        """Get MetaPlanner prioritized goals with reasoning.

        Instantiates MetaPlanner in scan_mode (no LLM calls) to get
        current prioritized goals derived from codebase signals.

        Returns:
            {"data": {"goals": [...], "signals_used": [...], "config": {...}}}
        """
        try:
            from aragora.nomic.meta_planner import (
                MetaPlanner,
                MetaPlannerConfig,
                PrioritizedGoal,
            )

            config = MetaPlannerConfig(
                scan_mode=True,
                quick_mode=False,
                max_goals=10,
                enable_metrics_collection=False,
                enable_cross_cycle_learning=False,
            )
            planner = MetaPlanner(config)

            goals: list[PrioritizedGoal] = await planner.prioritize_work(
                objective=None,
            )

            goals_data = []
            for g in goals:
                goals_data.append({
                    "id": g.id,
                    "track": g.track.value,
                    "description": g.description,
                    "rationale": g.rationale,
                    "estimated_impact": g.estimated_impact,
                    "priority": g.priority,
                    "focus_areas": g.focus_areas,
                    "file_hints": g.file_hints[:10],
                })

            # Collect signal types from goal descriptions
            signals_used = set()
            for g in goals:
                desc = g.description.lower()
                for signal in [
                    "recent_change", "untested", "regression", "test_failure",
                    "lint", "todo", "low_nps", "feedback_queue", "strategic",
                    "pipeline_goal", "feedback_goal",
                ]:
                    if signal in desc:
                        signals_used.add(signal)

            return json_response({
                "data": {
                    "goals": goals_data,
                    "signals_used": sorted(signals_used),
                    "config": {
                        "scan_mode": config.scan_mode,
                        "max_goals": config.max_goals,
                    },
                }
            })

        except ImportError:
            logger.debug("MetaPlanner not available")
            return json_response({
                "data": {
                    "goals": [],
                    "signals_used": [],
                    "config": {},
                    "error": "MetaPlanner module not available",
                }
            })
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Failed to get meta-planner goals: %s", exc)
            return json_response({
                "data": {
                    "goals": [],
                    "signals_used": [],
                    "config": {},
                    "error": "Failed to generate goals",
                }
            })

    async def _get_execution_timeline(self) -> HandlerResult:
        """Get active branches and execution status.

        Queries BranchCoordinator for worktree state and recent
        merge decisions.

        Returns:
            {"data": {"branches": [...], "merge_decisions": [...], "active_count": N}}
        """
        branches_data: list[dict[str, Any]] = []
        merge_decisions: list[dict[str, Any]] = []

        # Get worktree info from BranchCoordinator
        try:
            from aragora.nomic.branch_coordinator import BranchCoordinator

            coordinator = BranchCoordinator()
            worktrees = coordinator.list_worktrees()

            for wt in worktrees:
                branches_data.append({
                    "branch_name": wt.branch_name,
                    "worktree_path": str(wt.worktree_path),
                    "track": wt.track,
                    "created_at": wt.created_at.isoformat() if wt.created_at else None,
                    "assignment_id": wt.assignment_id,
                    "status": "active",
                })

        except ImportError:
            logger.debug("BranchCoordinator not available")
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Failed to list worktrees: %s", exc)

        # Get recent cycle records for merge decisions
        try:
            from aragora.nomic.cycle_store import get_cycle_store

            store = get_cycle_store()
            recent_cycles = store.get_recent(limit=20)

            for cycle in recent_cycles:
                cycle_data = cycle if isinstance(cycle, dict) else getattr(cycle, "__dict__", {})
                cycle_id = cycle_data.get("cycle_id", "")
                status = cycle_data.get("status", "unknown")
                started_at = cycle_data.get("started_at")
                completed_at = cycle_data.get("completed_at")

                merge_decisions.append({
                    "cycle_id": cycle_id,
                    "status": status,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "success": cycle_data.get("success", False),
                })

        except ImportError:
            logger.debug("CycleStore not available")
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            logger.debug("Failed to get cycle records: %s", exc)

        # Get run state from self-improve handler's active tasks
        try:
            from aragora.server.handlers.self_improve import _active_tasks

            for run_id, task in _active_tasks.items():
                if not task.done():
                    branches_data.append({
                        "branch_name": f"run/{run_id[:12]}",
                        "worktree_path": None,
                        "track": None,
                        "created_at": None,
                        "assignment_id": run_id,
                        "status": "running",
                    })
        except ImportError:
            pass

        active_count = sum(1 for b in branches_data if b.get("status") in ("active", "running"))

        return json_response({
            "data": {
                "branches": branches_data,
                "merge_decisions": merge_decisions,
                "active_count": active_count,
            }
        })

    async def _get_learning_insights(self) -> HandlerResult:
        """Get cross-cycle learning data.

        Queries the NomicCycleAdapter for high-ROI patterns,
        recurring failures, and historical learnings.

        Returns:
            {"data": {"insights": [...], "high_roi_patterns": [...], "recurring_failures": [...]}}
        """
        insights: list[dict[str, Any]] = []
        high_roi_patterns: list[dict[str, Any]] = []
        recurring_failures: list[dict[str, Any]] = []

        # Query Knowledge Mound for cross-cycle learning
        try:
            from aragora.knowledge.mound.adapters.nomic_cycle_adapter import (
                get_nomic_cycle_adapter,
            )

            adapter = get_nomic_cycle_adapter()

            # High-ROI goal types
            try:
                roi_data = await adapter.find_high_roi_goal_types(limit=10)
                for entry in roi_data:
                    high_roi_patterns.append({
                        "pattern": entry.get("pattern", ""),
                        "avg_improvement_score": entry.get("avg_improvement_score", 0),
                        "cycle_count": entry.get("cycle_count", 0),
                    })
            except (RuntimeError, ValueError, OSError, AttributeError) as exc:
                logger.debug("High-ROI query failed: %s", exc)

            # Recurring failures
            try:
                failure_data = await adapter.find_recurring_failures(
                    min_occurrences=2, limit=10
                )
                for entry in failure_data:
                    recurring_failures.append({
                        "pattern": entry.get("pattern", ""),
                        "occurrences": entry.get("occurrences", 0),
                        "affected_tracks": entry.get("affected_tracks", []),
                    })
            except (RuntimeError, ValueError, OSError, AttributeError) as exc:
                logger.debug("Recurring failures query failed: %s", exc)

        except ImportError:
            logger.debug("NomicCycleAdapter not available")
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Failed to query KM learning data: %s", exc)

        # Query outcome tracker for regression history as insights
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            regressions = NomicOutcomeTracker.get_regression_history(limit=10)
            for reg in regressions:
                insights.append({
                    "type": "regression",
                    "cycle_id": reg.get("cycle_id", ""),
                    "regressed_metrics": reg.get("regressed_metrics", []),
                    "recommendation": reg.get("recommendation", ""),
                })
        except ImportError:
            logger.debug("OutcomeTracker not available")
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Failed to get regression history: %s", exc)

        # Query strategic memory for recurring findings
        try:
            from aragora.nomic.strategic_memory import StrategicMemoryStore

            sm_store = StrategicMemoryStore()
            recurring = sm_store.get_recurring_findings(min_occurrences=2)
            for finding in recurring[:10]:
                insights.append({
                    "type": "strategic_finding",
                    "category": finding.category,
                    "description": finding.description[:200],
                    "occurrences": getattr(finding, "occurrences", 0),
                })
        except ImportError:
            pass
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            logger.debug("Strategic memory query failed: %s", exc)

        return json_response({
            "data": {
                "insights": insights,
                "high_roi_patterns": high_roi_patterns,
                "recurring_failures": recurring_failures,
            }
        })

    async def _get_metrics_comparison(self) -> HandlerResult:
        """Get before/after codebase metrics from recent cycles.

        Queries OutcomeTracker for the most recent outcome comparisons
        showing which metrics improved or regressed.

        Returns:
            {"data": {"comparisons": [...], "regressions": [...]}}
        """
        comparisons: list[dict[str, Any]] = []
        regressions: list[dict[str, Any]] = []

        # Get outcome comparisons from tracker
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            # Get regression history (includes metric deltas)
            reg_history = NomicOutcomeTracker.get_regression_history(limit=10)
            for reg in reg_history:
                regressions.append({
                    "cycle_id": reg.get("cycle_id", ""),
                    "regressed_metrics": reg.get("regressed_metrics", []),
                    "recommendation": reg.get("recommendation", ""),
                    "timestamp": reg.get("timestamp"),
                })
        except ImportError:
            logger.debug("OutcomeTracker not available")
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Failed to get outcome comparisons: %s", exc)

        # Get recent cycle records with metrics
        try:
            from aragora.nomic.cycle_store import get_cycle_store

            store = get_cycle_store()
            recent = store.get_recent(limit=10)

            for cycle in recent:
                cycle_data = cycle if isinstance(cycle, dict) else getattr(cycle, "__dict__", {})
                scores = cycle_data.get("evidence_quality_scores", {})
                if scores:
                    comparisons.append({
                        "cycle_id": cycle_data.get("cycle_id", ""),
                        "metrics": scores,
                        "success": cycle_data.get("success", False),
                        "timestamp": cycle_data.get("started_at"),
                    })
        except ImportError:
            logger.debug("CycleStore not available")
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            logger.debug("Failed to get cycle metrics: %s", exc)

        # Get codebase metrics snapshot if available
        try:
            from aragora.nomic.metrics_collector import MetricsCollector, MetricsCollectorConfig

            config = MetricsCollectorConfig(
                test_timeout=30,
                test_args=["--co", "-q"],  # Collect-only for speed
            )
            collector = MetricsCollector(config)
            from aragora.nomic.metrics_collector import MetricSnapshot
            import time as _time

            snapshot = MetricSnapshot(timestamp=_time.time())
            try:
                collector._collect_size_metrics(snapshot, None)
            except (OSError, ValueError):
                pass

            if snapshot.files_count > 0:
                comparisons.insert(0, {
                    "cycle_id": "current",
                    "metrics": {
                        "files_count": snapshot.files_count,
                        "total_lines": snapshot.total_lines,
                        "test_files": snapshot.test_files,
                        "lint_errors": snapshot.lint_errors,
                    },
                    "success": True,
                    "timestamp": snapshot.timestamp,
                })
        except ImportError:
            logger.debug("MetricsCollector not available")
        except (RuntimeError, ValueError, OSError) as exc:
            logger.debug("Failed to collect current metrics: %s", exc)

        return json_response({
            "data": {
                "comparisons": comparisons,
                "regressions": regressions,
            }
        })
