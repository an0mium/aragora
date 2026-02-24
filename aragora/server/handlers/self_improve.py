"""Self-improvement run management endpoints.

Endpoints:
- POST /api/self-improve/run         - Start a new self-improvement cycle
- POST /api/self-improve/start       - Start a new run (legacy alias)
- GET  /api/self-improve/status      - Get current cycle status (running/idle)
- GET  /api/self-improve/runs        - List all runs
- GET  /api/self-improve/runs/:id    - Get run status and progress
- GET  /api/self-improve/history     - Get run history (alias for /runs)
- GET  /api/self-improve/feedback    - Get feedback loop state and metrics
- POST /api/self-improve/runs/:id/cancel - Cancel a running run
- POST /api/self-improve/coordinate  - Start a hierarchical coordination cycle
- GET  /api/self-improve/worktrees   - List active worktrees
- POST /api/self-improve/worktrees/cleanup - Clean up all worktrees

These endpoints expose the SelfImprovePipeline and HierarchicalCoordinator
through a REST API, enabling web UI, API clients, and chat integrations
to trigger and monitor autonomous improvement runs.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    HandlerResult,
    error_response,
    get_int_param,
    json_response,
    handle_errors,
)
from .secure import SecureHandler
from .utils.auth_mixins import SecureEndpointMixin, require_permission
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Active run tasks for cancellation support
_active_tasks: dict[str, asyncio.Task[Any]] = {}


def _extract_run_id(path: str) -> str | None:
    """Extract run_id from /api/self-improve/runs/{run_id}[/cancel]."""
    parts = path.strip("/").split("/")
    # Expected: ["api", "self-improve", "runs", "<run_id>", ...]
    if len(parts) >= 4 and parts[2] == "runs":
        return parts[3]
    return None


class SelfImproveHandler(SecureEndpointMixin, SecureHandler):  # type: ignore[misc]
    """Handler for self-improvement run management.

    Integrates:
    - SelfImprovePipeline for goal-driven self-improvement cycles
    - HierarchicalCoordinator for planner/worker/judge coordination
    - HardenedOrchestrator for robust execution with worktree isolation

    RBAC Permissions:
    - self_improve:read  - View runs, status, and history
    - self_improve:admin - Start, cancel, and coordinate runs
    """

    RESOURCE_TYPE = "self_improve"

    ROUTES = [
        "/api/self-improve/run",
        "/api/self-improve/start",
        "/api/self-improve/status",
        "/api/self-improve/runs",
        "/api/self-improve/history",
        "/api/self-improve/coordinate",
        "/api/self-improve/worktrees",
        "/api/self-improve/worktrees/cleanup",
        "/api/self-improve/feedback",
        "/api/self-improve/feedback-summary",
        "/api/self-improve/regression-history",
        "/api/self-improve/goals",
        "/api/self-improve/metrics/summary",
    ]

    def __init__(self, server_context: dict[str, Any]) -> None:
        super().__init__(server_context)
        self._store = None

    def _get_store(self) -> Any:
        """Lazy-load the run store."""
        if self._store is None:
            try:
                from aragora.nomic.stores.run_store import SelfImproveRunStore

                self._store = SelfImproveRunStore()
            except (ImportError, OSError) as e:
                logger.warning("Failed to initialize run store: %s", type(e).__name__)
                return None
        return self._store

    def _get_stream_server(self) -> Any:
        """Lazy-load the Nomic Loop stream server for WebSocket events."""
        try:
            from aragora.server.stream.nomic_loop_stream import NomicLoopStreamServer

            if not hasattr(self, "_stream_server"):
                self._stream_server = NomicLoopStreamServer()
            return self._stream_server
        except ImportError:
            return None

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        path = strip_version_prefix(path)
        return path in self.ROUTES or path.startswith("/api/self-improve/")

    @require_permission("self_improve:read")
    @rate_limit(requests_per_minute=30)
    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route GET requests for self-improvement run data."""
        path = strip_version_prefix(path)

        if path == "/api/self-improve/feedback":
            return self._get_feedback_state()

        if path == "/api/self-improve/feedback-summary":
            return self._get_feedback_summary(query_params)

        if path == "/api/self-improve/regression-history":
            return self._get_regression_history(query_params)

        if path == "/api/self-improve/status":
            return self._get_status()

        if path in ("/api/self-improve/runs", "/api/self-improve/history"):
            return self._list_runs(query_params)

        if path == "/api/self-improve/worktrees":
            return self._list_worktrees()

        if path == "/api/self-improve/goals":
            return self._get_goal_queue(query_params)

        if path == "/api/self-improve/metrics/summary":
            return self._get_metrics_summary()

        # GET /api/self-improve/runs/:id
        run_id = _extract_run_id(path)
        if run_id and not path.endswith("/cancel"):
            return self._get_run(run_id)

        return None

    @handle_errors("self-improve write")
    @require_permission("self_improve:admin")
    @rate_limit(requests_per_minute=10)
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests for starting and cancelling runs."""
        path = strip_version_prefix(path)

        if path in ("/api/self-improve/run", "/api/self-improve/start"):
            body = self.read_json_body(handler) or {}
            return await self._start_run(body)

        if path == "/api/self-improve/coordinate":
            body = self.read_json_body(handler) or {}
            return await self._start_coordination(body)

        if path == "/api/self-improve/worktrees/cleanup":
            return self._cleanup_worktrees()

        # POST /api/self-improve/runs/:id/cancel
        if path.endswith("/cancel"):
            run_id = _extract_run_id(path)
            if run_id:
                return self._cancel_run(run_id)

        return None

    # ------------------------------------------------------------------
    # GET /api/self-improve/status
    # ------------------------------------------------------------------

    def _get_status(self) -> HandlerResult:
        """Get current self-improvement cycle status.

        Returns whether a cycle is running, idle, the current phase,
        and progress information.
        """
        active_run_ids = [rid for rid, task in _active_tasks.items() if not task.done()]

        if not active_run_ids:
            return json_response(
                {
                    "state": "idle",
                    "active_runs": 0,
                    "runs": [],
                }
            )

        store = self._get_store()
        active_runs = []
        if store:
            for run_id in active_run_ids:
                run = store.get_run(run_id)
                if run:
                    active_runs.append(run.to_dict())

        return json_response(
            {
                "state": "running",
                "active_runs": len(active_run_ids),
                "runs": active_runs,
            }
        )

    # ------------------------------------------------------------------
    # GET /api/self-improve/runs, /history
    # ------------------------------------------------------------------

    def _list_runs(self, query_params: dict[str, Any]) -> HandlerResult:
        """List self-improvement runs with pagination."""
        store = self._get_store()
        if not store:
            return error_response("Self-improvement store not available", 503)

        limit = get_int_param(query_params, "limit", 50)
        offset = get_int_param(query_params, "offset", 0)
        status = query_params.get("status")

        runs = store.list_runs(limit=limit, offset=offset, status=status)
        return json_response(
            {
                "runs": [r.to_dict() for r in runs],
                "total": len(runs),
                "limit": limit,
                "offset": offset,
            }
        )

    # ------------------------------------------------------------------
    # GET /api/self-improve/runs/:id
    # ------------------------------------------------------------------

    def _get_run(self, run_id: str) -> HandlerResult:
        """Get a specific run's status and progress."""
        store = self._get_store()
        if not store:
            return error_response("Self-improvement store not available", 503)

        run = store.get_run(run_id)
        if not run:
            return error_response(f"Run {run_id} not found", 404)

        return json_response(run.to_dict())

    # ------------------------------------------------------------------
    # GET /api/self-improve/feedback
    # ------------------------------------------------------------------

    def _get_feedback_state(self) -> HandlerResult:
        """Get feedback loop state: regressions, adjustments, metrics."""
        regression_history: list[dict[str, Any]] = []
        selection_adjustments: dict[str, Any] = {}
        feedback_metrics: dict[str, Any] = {
            "debates_processed": 0,
            "agents_tracked": 0,
            "avg_adjustment": 0.0,
        }

        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            regression_history = NomicOutcomeTracker.get_regression_history(limit=10)
        except (ImportError, RuntimeError):
            pass

        try:
            from aragora.debate.selection_feedback import SelectionFeedbackLoop

            loop = SelectionFeedbackLoop()
            # Expose internal state if available
            states = getattr(loop, "_agent_states", {})
            adjustments = getattr(loop, "_selection_adjustments", {})
            selection_adjustments = {
                name: adj for name, adj in adjustments.items()
            }
            feedback_metrics["agents_tracked"] = len(states)
            feedback_metrics["debates_processed"] = sum(
                getattr(s, "debates_participated", 0) for s in states.values()
            )
            if adjustments:
                feedback_metrics["avg_adjustment"] = round(
                    sum(adjustments.values()) / len(adjustments), 4
                )
        except (ImportError, RuntimeError):
            pass

        return json_response({
            "data": {
                "regression_history": regression_history,
                "selection_adjustments": selection_adjustments,
                "feedback_metrics": feedback_metrics,
            }
        })

    # ------------------------------------------------------------------
    # GET /api/self-improve/regression-history
    # ------------------------------------------------------------------

    def _get_regression_history(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get regression history from NomicOutcomeTracker.

        Query params:
            limit: int (default 20) — Max entries to return
        """
        limit = get_int_param(query_params, "limit", 20)
        regressions: list[dict[str, Any]] = []

        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            regressions = NomicOutcomeTracker.get_regression_history(limit=limit)
        except (ImportError, RuntimeError, TypeError, AttributeError) as e:
            logger.debug("Regression history unavailable: %s", type(e).__name__)

        return json_response({
            "data": {
                "regressions": regressions,
                "total": len(regressions),
            }
        })

    # ------------------------------------------------------------------
    # GET /api/self-improve/feedback-summary
    # ------------------------------------------------------------------

    def _get_feedback_summary(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get post-execution feedback summary for a pipeline.

        Query params:
            pipeline_id: str — Pipeline to get feedback for
        """
        pipeline_id = query_params.get("pipeline_id", "")
        elo_changes: list[dict[str, Any]] = []
        km_entries: list[dict[str, Any]] = []
        regressions: list[dict[str, Any]] = []

        # Get ELO changes from recent debates/executions
        try:
            from aragora.ranking.elo import EloTracker

            tracker = EloTracker()
            history = getattr(tracker, "get_recent_changes", lambda n: [])(10)
            for change in history:
                elo_changes.append({
                    "agent": getattr(change, "agent_name", str(change)),
                    "delta": getattr(change, "delta", 0),
                    "new_rating": getattr(change, "new_rating", 0),
                    "reason": getattr(change, "reason", ""),
                })
        except (ImportError, RuntimeError, TypeError, AttributeError):
            pass

        # Get KM entries stored from this pipeline
        try:
            from aragora.pipeline.km_bridge import PipelineKMBridge

            bridge = PipelineKMBridge()
            if bridge.available and pipeline_id:
                entries = bridge.get_entries_for_pipeline(pipeline_id)
                for entry in (entries if isinstance(entries, list) else []):
                    km_entries.append({
                        "id": getattr(entry, "id", str(entry)),
                        "type": getattr(entry, "entry_type", "unknown"),
                        "confidence": getattr(entry, "confidence", 0.5),
                    })
        except (ImportError, RuntimeError, TypeError, AttributeError):
            pass

        # Get regressions
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            regressions = NomicOutcomeTracker.get_regression_history(limit=5)
        except (ImportError, RuntimeError, TypeError, AttributeError):
            pass

        return json_response({
            "data": {
                "pipeline_id": pipeline_id,
                "elo_changes": elo_changes,
                "km_entries_stored": len(km_entries),
                "km_entries": km_entries,
                "regressions": regressions,
                "regression_count": len(regressions),
            }
        })

    # ------------------------------------------------------------------
    # GET /api/self-improve/goals
    # ------------------------------------------------------------------

    def _get_goal_queue(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get improvement goal queue from ImprovementQueue.

        Returns pending goals with source, priority, and context.
        """
        limit = get_int_param(query_params, "limit", 20)
        goals: list[dict[str, Any]] = []

        try:
            from aragora.nomic.feedback_orchestrator import ImprovementQueue

            queue = ImprovementQueue()
            pending = queue.peek(limit=limit)
            for g in pending:
                goals.append({
                    "goal": g.goal,
                    "source": g.source,
                    "priority": g.priority,
                    "context": g.context,
                    "estimated_impact": g.context.get("estimated_impact", "medium"),
                    "track": g.context.get("track", "core"),
                    "status": "pending",
                })
        except (ImportError, OSError, RuntimeError) as e:
            logger.debug("ImprovementQueue not available: %s", type(e).__name__)

        # Also include legacy in-memory goals
        try:
            from aragora.nomic.feedback_orchestrator import ImprovementQueue as IQ

            legacy_queue = IQ.load()
            for fg in legacy_queue.goals[-limit:]:
                goals.append({
                    "goal": fg.description,
                    "source": fg.source,
                    "priority": fg.priority,
                    "context": fg.metadata,
                    "estimated_impact": fg.estimated_impact,
                    "track": fg.track,
                    "status": "pending",
                })
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.debug("Legacy goal queue not available: %s", type(e).__name__)

        # Deduplicate by goal text
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for g in goals:
            key = g["goal"]
            if key not in seen:
                seen.add(key)
                deduped.append(g)

        # Sort by priority descending
        deduped.sort(key=lambda g: g.get("priority", 0), reverse=True)

        return json_response({
            "data": {
                "goals": deduped[:limit],
                "total": len(deduped),
            }
        })

    # ------------------------------------------------------------------
    # GET /api/self-improve/metrics/summary
    # ------------------------------------------------------------------

    def _get_metrics_summary(self) -> HandlerResult:
        """Get aggregated metrics summary for the Nomic Loop dashboard.

        Computes:
        - cycle_success_rate: fraction of completed runs that succeeded
        - goal_completion_rate: completed_subtasks / total_subtasks
        - total_cycles: number of runs
        - total_goals_queued: number of pending improvement goals
        - recent_activity: last 10 significant events
        """
        store = self._get_store()

        total_cycles = 0
        completed_cycles = 0
        failed_cycles = 0
        total_subtasks = 0
        completed_subtasks = 0
        recent_activity: list[dict[str, Any]] = []

        if store:
            runs = store.list_runs(limit=100, offset=0)
            total_cycles = len(runs)
            for run in runs:
                rd = run.to_dict()
                status = rd.get("status", "")
                if status == "completed":
                    completed_cycles += 1
                elif status == "failed":
                    failed_cycles += 1
                total_subtasks += rd.get("total_subtasks", 0)
                completed_subtasks += rd.get("completed_subtasks", 0)

                # Build activity feed entries from runs
                if rd.get("summary"):
                    recent_activity.append({
                        "type": "cycle_completed" if status == "completed" else "cycle_failed",
                        "message": rd["summary"],
                        "timestamp": rd.get("completed_at") or rd.get("started_at") or rd.get("created_at", ""),
                        "run_id": rd.get("run_id", ""),
                        "status": status,
                    })

        # Compute rates
        cycle_success_rate = (
            completed_cycles / total_cycles if total_cycles > 0 else 0.0
        )
        goal_completion_rate = (
            completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
        )

        # Get queued goal count
        total_goals_queued = 0
        try:
            from aragora.nomic.feedback_orchestrator import ImprovementQueue

            queue = ImprovementQueue()
            pending = queue.peek(limit=100)
            total_goals_queued = len(pending)
        except (ImportError, OSError, RuntimeError):
            pass

        # Get test pass rate from gauntlet if available
        test_pass_rate = 1.0
        try:
            from aragora.nomic.feedback_orchestrator import ImprovementQueue as IQ

            legacy_queue = IQ.load()
            gauntlet_goals = [g for g in legacy_queue.goals if g.source == "gauntlet_finding"]
            if gauntlet_goals:
                # More gauntlet findings = lower inferred test health
                test_pass_rate = max(0.0, 1.0 - len(gauntlet_goals) * 0.1)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            pass

        # Compute overall health score (weighted average)
        health_score = (
            cycle_success_rate * 0.4
            + goal_completion_rate * 0.35
            + test_pass_rate * 0.25
        )

        # Sort activity by timestamp descending, take latest 10
        recent_activity.sort(
            key=lambda a: a.get("timestamp", ""), reverse=True
        )
        recent_activity = recent_activity[:10]

        return json_response({
            "data": {
                "health_score": round(health_score, 4),
                "cycle_success_rate": round(cycle_success_rate, 4),
                "goal_completion_rate": round(goal_completion_rate, 4),
                "test_pass_rate": round(test_pass_rate, 4),
                "total_cycles": total_cycles,
                "completed_cycles": completed_cycles,
                "failed_cycles": failed_cycles,
                "total_subtasks": total_subtasks,
                "completed_subtasks": completed_subtasks,
                "total_goals_queued": total_goals_queued,
                "recent_activity": recent_activity,
            }
        })

    # ------------------------------------------------------------------
    # POST /api/self-improve/run (and /start alias)
    # ------------------------------------------------------------------

    async def _start_run(self, body: dict[str, Any]) -> HandlerResult:
        """Start a new self-improvement run.

        Accepts config overrides:
        - goal (required): The improvement objective
        - mode: "flat" (SelfImprovePipeline) or "hierarchical"
        - scan_mode: Use codebase signals only (default: true)
        - quick_mode: Skip debate, use heuristics (default: false)
        - budget_limit_usd: Total budget cap
        - max_cycles: Max execution cycles (default: 5)
        - dry_run: Preview plan without executing (default: false)
        - tracks: Focus tracks (e.g. ["qa", "developer"])
        - require_approval: Require human approval at checkpoints
        """
        store = self._get_store()
        if not store:
            return error_response("Self-improvement store not available", 503)

        goal = body.get("goal", "").strip()
        if not goal:
            return error_response("'goal' is required", 400)

        tracks = body.get("tracks")
        mode = body.get("mode", "flat")
        budget_limit = body.get("budget_limit_usd")
        max_cycles = body.get("max_cycles", 5)
        dry_run = body.get("dry_run", False)
        scan_mode = body.get("scan_mode", True)
        quick_mode = body.get("quick_mode", False)
        require_approval = body.get("require_approval", False)

        if mode not in ("flat", "hierarchical"):
            return error_response("'mode' must be 'flat' or 'hierarchical'", 400)

        run = store.create_run(
            goal=goal,
            tracks=tracks or [],
            mode=mode,
            budget_limit_usd=budget_limit,
            max_cycles=max_cycles,
            dry_run=dry_run,
        )

        if dry_run:
            # For dry runs, use SelfImprovePipeline.dry_run() for richer output
            plan = await self._generate_plan(goal, tracks, scan_mode, quick_mode)
            store.update_run(
                run.run_id,
                status="completed",
                plan=plan,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            # Emit phase_completed for dry-run planning
            try:
                stream = self._get_stream_server()
                if stream:
                    await stream.emit_phase_completed(
                        "planning",
                        cycle=1,
                        duration_sec=0.0,
                        result_summary="Dry-run plan generated",
                    )
            except (RuntimeError, OSError) as e:
                logger.debug("WebSocket emit skipped: %s", type(e).__name__)
            return json_response(
                {"run_id": run.run_id, "status": "preview", "plan": plan},
            )

        # Start async execution
        task = asyncio.create_task(
            self._execute_run(
                run.run_id,
                goal,
                tracks,
                mode,
                budget_limit,
                max_cycles,
                scan_mode=scan_mode,
                quick_mode=quick_mode,
                require_approval=require_approval,
            )
        )
        _active_tasks[run.run_id] = task

        store.update_run(
            run.run_id,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        return json_response(
            {"run_id": run.run_id, "status": "started"},
            status=202,
        )

    # ------------------------------------------------------------------
    # POST /api/self-improve/runs/:id/cancel
    # ------------------------------------------------------------------

    def _cancel_run(self, run_id: str) -> HandlerResult:
        """Cancel a running self-improvement run."""
        store = self._get_store()
        if not store:
            return error_response("Self-improvement store not available", 503)

        run = store.cancel_run(run_id)
        if not run:
            return error_response(f"Run {run_id} not found or already terminal", 404)

        # Cancel the async task if active
        task = _active_tasks.pop(run_id, None)
        if task and not task.done():
            task.cancel()

        return json_response(
            {
                "run_id": run_id,
                "status": "cancelled",
            }
        )

    # ------------------------------------------------------------------
    # POST /api/self-improve/coordinate (HierarchicalCoordinator)
    # ------------------------------------------------------------------

    async def _start_coordination(self, body: dict[str, Any]) -> HandlerResult:
        """Start a hierarchical planner/worker/judge coordination cycle.

        Uses HierarchicalCoordinator for structured goal execution with
        automatic decomposition, parallel workers, and judge review.

        Body:
            goal (required): The coordination objective
            tracks: Optional focus tracks
            max_cycles: Max plan-execute-judge cycles (default: 3)
            quality_threshold: Judge approval threshold (default: 0.6)
            max_parallel_workers: Parallel worker limit (default: 4)
        """
        store = self._get_store()
        if not store:
            return error_response("Self-improvement store not available", 503)

        goal = body.get("goal", "").strip()
        if not goal:
            return error_response("'goal' is required", 400)

        tracks = body.get("tracks")
        max_cycles = body.get("max_cycles", 3)
        quality_threshold = body.get("quality_threshold", 0.6)
        max_parallel = body.get("max_parallel_workers", 4)

        run = store.create_run(
            goal=goal,
            tracks=tracks or [],
            mode="hierarchical",
            max_cycles=max_cycles,
        )

        # Start async coordination
        task = asyncio.create_task(
            self._execute_coordination(
                run.run_id,
                goal,
                tracks,
                max_cycles=max_cycles,
                quality_threshold=quality_threshold,
                max_parallel=max_parallel,
            )
        )
        _active_tasks[run.run_id] = task

        store.update_run(
            run.run_id,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        return json_response(
            {"run_id": run.run_id, "status": "coordinating", "mode": "hierarchical"},
            status=202,
        )

    async def _execute_coordination(
        self,
        run_id: str,
        goal: str,
        tracks: list[str] | None,
        max_cycles: int = 3,
        quality_threshold: float = 0.6,
        max_parallel: int = 4,
    ) -> None:
        """Execute a hierarchical coordination cycle in the background."""
        store = self._get_store()
        if not store:
            return

        try:
            from aragora.nomic.hierarchical_coordinator import (
                CoordinatorConfig,
                HierarchicalCoordinator,
            )

            config = CoordinatorConfig(
                max_cycles=max_cycles,
                quality_threshold=quality_threshold,
                max_parallel_workers=max_parallel,
            )
            coordinator = HierarchicalCoordinator(config=config)

            result = await coordinator.coordinate(
                goal=goal,
                tracks=tracks,
            )

            store.update_run(
                run_id,
                status="completed" if result.success else "failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                total_subtasks=len(result.worker_reports),
                completed_subtasks=sum(1 for r in result.worker_reports if r.success),
                failed_subtasks=sum(1 for r in result.worker_reports if not r.success),
                summary=(
                    f"Coordination completed in {result.cycles_used} cycles"
                    if result.success
                    else "Coordination failed"
                ),
            )

        except asyncio.CancelledError:
            store.update_run(
                run_id,
                status="cancelled",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error("Coordination run %s failed: %s", run_id, type(e).__name__)
            store.update_run(
                run_id,
                status="failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error="Coordination failed",
            )
        finally:
            _active_tasks.pop(run_id, None)

    # ------------------------------------------------------------------
    # Plan Generation
    # ------------------------------------------------------------------

    async def _generate_plan(
        self,
        goal: str,
        tracks: list[str] | None,
        scan_mode: bool = True,
        quick_mode: bool = False,
    ) -> dict[str, Any]:
        """Generate a decomposition plan without executing.

        Uses SelfImprovePipeline.dry_run() when available for richer output,
        falling back to TaskDecomposer for basic decomposition.
        """
        # Try SelfImprovePipeline first for richer dry_run output
        try:
            from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline

            config = SelfImproveConfig(
                scan_mode=scan_mode,
                quick_mode=quick_mode,
            )
            pipeline = SelfImprovePipeline(config=config)
            plan = await pipeline.dry_run(goal)
            plan["tracks"] = tracks or []
            return plan
        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            logger.debug("SelfImprovePipeline dry_run unavailable: %s", type(e).__name__)

        # Fallback to TaskDecomposer
        try:
            from aragora.nomic.task_decomposer import TaskDecomposer

            decomposer = TaskDecomposer()
            result = decomposer.analyze(goal)
            return {
                "goal": goal,
                "tracks": tracks or [],
                "subtasks": [
                    {"description": st.description, "track": st.track, "priority": st.priority}
                    for st in (result.subtasks if hasattr(result, "subtasks") else [])
                ],
                "complexity": getattr(result, "complexity_score", 0),
            }
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            logger.warning("Plan generation failed: %s", type(e).__name__)
            return {
                "goal": goal,
                "tracks": tracks or [],
                "subtasks": [],
                "error": "Plan generation unavailable",
            }

    # ------------------------------------------------------------------
    # Background Execution
    # ------------------------------------------------------------------

    async def _execute_run(
        self,
        run_id: str,
        goal: str,
        tracks: list[str] | None,
        mode: str,
        budget_limit: float | None,
        max_cycles: int,
        scan_mode: bool = True,
        quick_mode: bool = False,
        require_approval: bool = False,
    ) -> None:
        """Execute a self-improvement run in the background.

        For 'flat' mode, uses SelfImprovePipeline with config overrides.
        For 'hierarchical' mode, delegates to _execute_coordination.
        Falls back to HardenedOrchestrator if SelfImprovePipeline is unavailable.
        """
        store = self._get_store()
        if not store:
            return

        if mode == "hierarchical":
            await self._execute_coordination(run_id, goal, tracks, max_cycles=max_cycles)
            return

        stream = self._get_stream_server()

        # Emit loop_started
        try:
            if stream:
                await stream.emit_loop_started(cycles=max_cycles, auto_approve=not require_approval)
        except (RuntimeError, OSError) as e:
            logger.debug("WebSocket emit skipped: %s", type(e).__name__)

        # Try SelfImprovePipeline first
        try:
            from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline

            config = SelfImproveConfig(
                scan_mode=scan_mode,
                quick_mode=quick_mode,
                budget_limit_usd=budget_limit or 10.0,
                require_approval=require_approval,
                autonomous=not require_approval,
                max_goals=max_cycles,
            )
            pipeline = SelfImprovePipeline(config=config)

            # Emit phase_started before pipeline execution
            try:
                if stream:
                    await stream.emit_phase_started("planning", cycle=1)
            except (RuntimeError, OSError) as e:
                logger.debug("WebSocket emit skipped: %s", type(e).__name__)

            result = await pipeline.run(goal)

            summary = f"Completed {result.subtasks_completed}/{result.subtasks_total} subtasks"
            store.update_run(
                run_id,
                status="completed" if result.subtasks_completed > 0 else "failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                total_subtasks=result.subtasks_total,
                completed_subtasks=result.subtasks_completed,
                failed_subtasks=result.subtasks_failed,
                summary=summary,
            )

            # Emit loop_stopped with summary
            try:
                if stream:
                    await stream.emit_loop_stopped(reason=summary)
            except (RuntimeError, OSError) as e:
                logger.debug("WebSocket emit skipped: %s", type(e).__name__)
            return

        except ImportError:
            logger.debug("SelfImprovePipeline not available, falling back to HardenedOrchestrator")
        except asyncio.CancelledError:
            store.update_run(
                run_id,
                status="cancelled",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            return
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.warning("SelfImprovePipeline failed, falling back: %s", type(e).__name__)
            # Emit error event
            try:
                if stream:
                    await stream.emit_error(
                        "SelfImprovePipeline failed, falling back to orchestrator"
                    )
            except (RuntimeError, OSError):
                pass

        # Fallback to HardenedOrchestrator
        try:
            from aragora.nomic.hardened_orchestrator import HardenedOrchestrator

            orchestrator = HardenedOrchestrator(
                require_human_approval=require_approval,
                budget_limit_usd=budget_limit,
                use_worktree_isolation=True,
            )

            result = await orchestrator.execute_goal_coordinated(
                goal=goal,
                tracks=tracks,
                max_cycles=max_cycles,
            )

            store.update_run(
                run_id,
                status="completed" if result.success else "failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                total_subtasks=result.total_subtasks,
                completed_subtasks=result.completed_subtasks,
                failed_subtasks=result.failed_subtasks,
                summary=result.summary,
                error=result.error,
            )

            # Emit loop_stopped
            try:
                if stream:
                    await stream.emit_loop_stopped(
                        reason=result.summary or "Orchestration complete"
                    )
            except (RuntimeError, OSError) as e:
                logger.debug("WebSocket emit skipped: %s", type(e).__name__)

        except asyncio.CancelledError:
            store.update_run(
                run_id,
                status="cancelled",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error("Self-improvement run %s failed: %s", run_id, type(e).__name__)
            store.update_run(
                run_id,
                status="failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error="Orchestration failed",
            )
            # Emit error event
            try:
                if stream:
                    await stream.emit_error("Orchestration failed")
            except (RuntimeError, OSError):
                pass
        finally:
            _active_tasks.pop(run_id, None)

    # ------------------------------------------------------------------
    # Worktree Management
    # ------------------------------------------------------------------

    def _list_worktrees(self) -> HandlerResult:
        """List active git worktrees managed by the branch coordinator."""
        try:
            from aragora.nomic.branch_coordinator import BranchCoordinator

            coordinator = BranchCoordinator()
            worktrees = coordinator.list_worktrees()
            return json_response(
                {
                    "worktrees": [
                        {
                            "branch_name": wt.branch_name,
                            "worktree_path": str(wt.worktree_path),
                            "track": wt.track,
                            "created_at": wt.created_at,
                            "assignment_id": wt.assignment_id,
                        }
                        for wt in worktrees
                    ],
                    "total": len(worktrees),
                }
            )
        except (ImportError, OSError, ValueError) as e:
            logger.warning("Failed to list worktrees: %s", type(e).__name__)
            return json_response(
                {"worktrees": [], "total": 0, "error": "Worktree listing unavailable"}
            )

    def _cleanup_worktrees(self) -> HandlerResult:
        """Clean up all managed worktrees."""
        try:
            from aragora.nomic.branch_coordinator import BranchCoordinator

            coordinator = BranchCoordinator()
            removed = coordinator.cleanup_all_worktrees()
            return json_response(
                {
                    "removed": removed,
                    "status": "cleaned",
                }
            )
        except (ImportError, OSError, ValueError):
            logger.warning("Worktree cleanup failed")
            return error_response("Worktree cleanup failed", 503)
