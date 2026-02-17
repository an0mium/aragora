"""Self-improvement run management endpoints.

Endpoints:
- POST /api/self-improve/start    - Start a new self-improvement run
- GET  /api/self-improve/runs     - List all runs
- GET  /api/self-improve/runs/:id - Get run status and progress
- POST /api/self-improve/runs/:id/cancel - Cancel a running run
- GET  /api/self-improve/history   - Get run history (alias for /runs)
- GET  /api/self-improve/worktrees - List active worktrees
- POST /api/self-improve/worktrees/cleanup - Clean up all worktrees

These endpoints expose the HardenedOrchestrator's self-improvement
pipeline through a REST API, enabling web UI, API clients, and
chat integrations to trigger and monitor autonomous improvement runs.
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

    RBAC Permissions:
    - self_improve:read  - View runs and history
    - self_improve:admin - Start and cancel runs
    """

    RESOURCE_TYPE = "self_improve"

    ROUTES = [
        "/api/self-improve/start",
        "/api/self-improve/runs",
        "/api/self-improve/history",
        "/api/self-improve/worktrees",
        "/api/v1/self-improve/worktrees/cleanup",
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
                logger.warning(f"Failed to initialize run store: {type(e).__name__}")
                return None
        return self._store

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

        if path in ("/api/self-improve/runs", "/api/self-improve/history"):
            return self._list_runs(query_params)

        if path == "/api/self-improve/worktrees":
            return self._list_worktrees()

        # GET /api/self-improve/runs/:id
        run_id = _extract_run_id(path)
        if run_id and not path.endswith("/cancel"):
            return self._get_run(run_id)

        return None

    @require_permission("self_improve:admin")
    @rate_limit(requests_per_minute=10)
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests for starting and cancelling runs."""
        path = strip_version_prefix(path)

        if path == "/api/self-improve/start":
            body = self.read_json_body(handler) or {}
            return await self._start_run(body)

        if path == "/api/self-improve/worktrees/cleanup":
            return self._cleanup_worktrees()

        # POST /api/self-improve/runs/:id/cancel
        if path.endswith("/cancel"):
            run_id = _extract_run_id(path)
            if run_id:
                return self._cancel_run(run_id)

        return None

    def _list_runs(self, query_params: dict[str, Any]) -> HandlerResult:
        """List self-improvement runs with pagination."""
        store = self._get_store()
        if not store:
            return error_response("Self-improvement store not available", 503)

        limit = get_int_param(query_params, "limit", 50)
        offset = get_int_param(query_params, "offset", 0)
        status = query_params.get("status")

        runs = store.list_runs(limit=limit, offset=offset, status=status)
        return json_response({
            "runs": [r.to_dict() for r in runs],
            "total": len(runs),
            "limit": limit,
            "offset": offset,
        })

    def _get_run(self, run_id: str) -> HandlerResult:
        """Get a specific run's status and progress."""
        store = self._get_store()
        if not store:
            return error_response("Self-improvement store not available", 503)

        run = store.get_run(run_id)
        if not run:
            return error_response(f"Run {run_id} not found", 404)

        return json_response(run.to_dict())

    async def _start_run(self, body: dict[str, Any]) -> HandlerResult:
        """Start a new self-improvement run."""
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
            # For dry runs, decompose the goal and return the plan
            plan = await self._generate_plan(goal, tracks)
            store.update_run(
                run.run_id,
                status="completed",
                plan=plan,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            return json_response(
                {"run_id": run.run_id, "status": "preview", "plan": plan},
            )

        # Start async execution
        task = asyncio.create_task(
            self._execute_run(run.run_id, goal, tracks, mode, budget_limit, max_cycles)
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

        return json_response({
            "run_id": run_id,
            "status": "cancelled",
        })

    async def _generate_plan(
        self, goal: str, tracks: list[str] | None
    ) -> dict[str, Any]:
        """Generate a decomposition plan without executing."""
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
            logger.warning(f"Plan generation failed: {type(e).__name__}")
            return {"goal": goal, "tracks": tracks or [], "subtasks": [], "error": "Plan generation unavailable"}

    async def _execute_run(
        self,
        run_id: str,
        goal: str,
        tracks: list[str] | None,
        mode: str,
        budget_limit: float | None,
        max_cycles: int,
    ) -> None:
        """Execute a self-improvement run in the background."""
        store = self._get_store()
        if not store:
            return

        try:
            from aragora.nomic.hardened_orchestrator import HardenedOrchestrator

            orchestrator = HardenedOrchestrator(
                require_human_approval=False,
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

        except asyncio.CancelledError:
            store.update_run(
                run_id,
                status="cancelled",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error(f"Self-improvement run {run_id} failed: {type(e).__name__}")
            store.update_run(
                run_id,
                status="failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error=f"Orchestration failed: {type(e).__name__}",
            )
        finally:
            _active_tasks.pop(run_id, None)

    def _list_worktrees(self) -> HandlerResult:
        """List active git worktrees managed by the branch coordinator."""
        try:
            from aragora.nomic.branch_coordinator import BranchCoordinator

            coordinator = BranchCoordinator()
            worktrees = coordinator.list_worktrees()
            return json_response({
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
            })
        except (ImportError, OSError, ValueError) as e:
            logger.warning(f"Failed to list worktrees: {type(e).__name__}")
            return json_response({"worktrees": [], "total": 0, "error": "Worktree listing unavailable"})

    def _cleanup_worktrees(self) -> HandlerResult:
        """Clean up all managed worktrees."""
        try:
            from aragora.nomic.branch_coordinator import BranchCoordinator

            coordinator = BranchCoordinator()
            removed = coordinator.cleanup_all_worktrees()
            return json_response({
                "removed": removed,
                "status": "cleaned",
            })
        except (ImportError, OSError, ValueError) as e:
            logger.warning(f"Failed to cleanup worktrees: {type(e).__name__}")
            return error_response(f"Worktree cleanup failed: {type(e).__name__}", 503)
