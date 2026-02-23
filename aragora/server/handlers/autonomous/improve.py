"""Autonomous self-improvement REST API endpoints.

Production-ready endpoints for triggering, monitoring, and listing
autonomous self-improvement runs via the HardenedOrchestrator and
AutonomousOrchestrator.

Endpoints:
- POST /api/v1/autonomous/improve          Start a new improvement run
- GET  /api/v1/autonomous/improve          List all improvement runs
- GET  /api/v1/autonomous/improve/:run_id  Get a specific run's status

These endpoints expose the existing SelfImprovePipeline, HardenedOrchestrator,
and AutonomousOrchestrator to the frontend through a versioned, RBAC-protected
REST API surface.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from ..base import (
    HandlerResult,
    error_response,
    get_int_param,
    json_response,
    handle_errors,
)
from ..secure import SecureHandler
from ..utils.auth_mixins import SecureEndpointMixin, require_permission
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Active run tasks for cancellation support
_active_improve_tasks: dict[str, asyncio.Task[Any]] = {}

# Budget constraints
MAX_BUDGET_USD = 100.0
MIN_BUDGET_USD = 0.01


def _extract_run_id(path: str) -> str | None:
    """Extract run_id from /api/autonomous/improve/{run_id}.

    After strip_version_prefix, the path looks like:
        /api/autonomous/improve/{run_id}
    Split: ["", "api", "autonomous", "improve", "<run_id>"]
    """
    parts = path.strip("/").split("/")
    # Expected: ["api", "autonomous", "improve", "<run_id>"]
    if len(parts) == 4 and parts[2] == "improve":
        return parts[3]
    return None


class AutonomousImproveHandler(SecureEndpointMixin, SecureHandler):  # type: ignore[misc]
    """Handler for autonomous self-improvement run management.

    Integrates:
    - SelfImprovePipeline for goal-driven self-improvement cycles
    - HardenedOrchestrator for robust execution with worktree isolation
    - AutonomousOrchestrator for multi-agent coordination

    RBAC Permissions:
    - autonomous:read    - View runs and their status
    - autonomous:improve - Start new improvement runs
    """

    RESOURCE_TYPE = "autonomous_improve"

    ROUTES = [
        "/api/autonomous/improve",
    ]

    def __init__(self, server_context: dict[str, Any]) -> None:
        super().__init__(server_context)
        self._store = None

    def _get_store(self) -> Any:
        """Lazy-load the run store to avoid heavy imports at module level."""
        if self._store is None:
            try:
                from aragora.nomic.stores.run_store import SelfImproveRunStore

                self._store = SelfImproveRunStore()
            except (ImportError, OSError) as e:
                logger.warning("Failed to initialize run store: %s", type(e).__name__)
                return None
        return self._store

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        path = strip_version_prefix(path)
        # Match /api/autonomous/improve and /api/autonomous/improve/{run_id}
        return path == "/api/autonomous/improve" or (
            path.startswith("/api/autonomous/improve/") and len(path.split("/")) == 5
        )

    @require_permission("autonomous:read")
    @rate_limit(requests_per_minute=30)
    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route GET requests for autonomous improvement data."""
        path = strip_version_prefix(path)

        # GET /api/autonomous/improve - list all runs
        if path == "/api/autonomous/improve":
            return self._list_runs(query_params)

        # GET /api/autonomous/improve/{run_id} - get specific run
        run_id = _extract_run_id(path)
        if run_id:
            return self._get_run(run_id)

        return None

    @handle_errors("autonomous improve")
    @require_permission("autonomous:improve")
    @rate_limit(requests_per_minute=10)
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST /api/v1/autonomous/improve - start a new run."""
        path = strip_version_prefix(path)

        if path == "/api/autonomous/improve":
            body = self.read_json_body(handler) or {}
            return await self._start_run(body)

        return None

    # ------------------------------------------------------------------
    # GET /api/v1/autonomous/improve - list runs
    # ------------------------------------------------------------------

    def _list_runs(self, query_params: dict[str, Any]) -> HandlerResult:
        """List all autonomous improvement runs with pagination."""
        store = self._get_store()
        if not store:
            return error_response("Self-improvement store not available", 503)

        limit = get_int_param(query_params, "limit", 50)
        offset = get_int_param(query_params, "offset", 0)
        status = query_params.get("status")

        runs = store.list_runs(limit=limit, offset=offset, status=status)
        return json_response(
            {
                "data": {
                    "runs": [r.to_dict() for r in runs],
                    "total": len(runs),
                    "limit": limit,
                    "offset": offset,
                }
            }
        )

    # ------------------------------------------------------------------
    # GET /api/v1/autonomous/improve/{run_id} - get run status
    # ------------------------------------------------------------------

    def _get_run(self, run_id: str) -> HandlerResult:
        """Get a specific run's status, progress, and result."""
        store = self._get_store()
        if not store:
            return error_response("Self-improvement store not available", 503)

        run = store.get_run(run_id)
        if not run:
            return error_response(f"Run {run_id} not found", 404)

        run_dict = run.to_dict()
        # Build progress info
        progress = {
            "total_subtasks": run_dict.get("total_subtasks", 0),
            "completed_subtasks": run_dict.get("completed_subtasks", 0),
            "failed_subtasks": run_dict.get("failed_subtasks", 0),
        }
        total = progress["total_subtasks"]
        if total > 0:
            progress["percent_complete"] = round(
                (progress["completed_subtasks"] / total) * 100, 1
            )
        else:
            progress["percent_complete"] = 0.0

        result = None
        if run_dict.get("status") in ("completed", "failed"):
            result = {
                "summary": run_dict.get("summary", ""),
                "error": run_dict.get("error"),
                "cost_usd": run_dict.get("cost_usd", 0.0),
                "completed_at": run_dict.get("completed_at"),
            }

        return json_response(
            {
                "data": {
                    "run_id": run_dict["run_id"],
                    "status": run_dict["status"],
                    "goal": run_dict.get("goal", ""),
                    "tracks": run_dict.get("tracks", []),
                    "created_at": run_dict.get("created_at", ""),
                    "started_at": run_dict.get("started_at"),
                    "progress": progress,
                    "result": result,
                }
            }
        )

    # ------------------------------------------------------------------
    # POST /api/v1/autonomous/improve - start a run
    # ------------------------------------------------------------------

    async def _start_run(self, body: dict[str, Any]) -> HandlerResult:
        """Start a new autonomous improvement run.

        Body:
            goal (required): The improvement objective
            budget_limit (optional): Maximum spend in USD
            require_approval (optional): Require human approval at checkpoints
            tracks (optional): Focus tracks (e.g. ["qa", "developer"])
        """
        store = self._get_store()
        if not store:
            return error_response("Self-improvement store not available", 503)

        # Validate goal
        goal = body.get("goal")
        if not goal or not isinstance(goal, str) or not goal.strip():
            return error_response("'goal' is required and must be a non-empty string", 400)
        goal = goal.strip()

        # Validate budget_limit
        budget_limit = body.get("budget_limit")
        if budget_limit is not None:
            try:
                budget_limit = float(budget_limit)
            except (TypeError, ValueError):
                return error_response(
                    "'budget_limit' must be a number", 400
                )
            if budget_limit < MIN_BUDGET_USD or budget_limit > MAX_BUDGET_USD:
                return error_response(
                    f"'budget_limit' must be between {MIN_BUDGET_USD} and {MAX_BUDGET_USD}",
                    400,
                )

        # Validate require_approval
        require_approval = body.get("require_approval", False)
        if not isinstance(require_approval, bool):
            return error_response("'require_approval' must be a boolean", 400)

        # Validate tracks
        tracks = body.get("tracks")
        if tracks is not None:
            if not isinstance(tracks, list) or not all(isinstance(t, str) for t in tracks):
                return error_response("'tracks' must be a list of strings", 400)

        # Create the run record
        run = store.create_run(
            goal=goal,
            tracks=tracks or [],
            mode="flat",
            budget_limit_usd=budget_limit,
        )

        # Start async execution in background
        task = asyncio.create_task(
            self._execute_run(
                run_id=run.run_id,
                goal=goal,
                tracks=tracks,
                budget_limit=budget_limit,
                require_approval=require_approval,
            )
        )
        _active_improve_tasks[run.run_id] = task

        store.update_run(
            run.run_id,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        return json_response(
            {"data": {"run_id": run.run_id, "status": "queued"}},
            status=202,
        )

    # ------------------------------------------------------------------
    # Background Execution
    # ------------------------------------------------------------------

    async def _execute_run(
        self,
        run_id: str,
        goal: str,
        tracks: list[str] | None,
        budget_limit: float | None,
        require_approval: bool = False,
    ) -> None:
        """Execute an improvement run in the background.

        Tries SelfImprovePipeline first, falls back to HardenedOrchestrator.
        """
        store = self._get_store()
        if not store:
            return

        # Try SelfImprovePipeline first
        try:
            from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline

            config = SelfImproveConfig(
                budget_limit_usd=budget_limit or 10.0,
                require_approval=require_approval,
                autonomous=not require_approval,
            )
            pipeline = SelfImprovePipeline(config=config)
            result = await pipeline.run(goal)

            store.update_run(
                run_id,
                status="completed" if result.subtasks_completed > 0 else "failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                total_subtasks=result.subtasks_total,
                completed_subtasks=result.subtasks_completed,
                failed_subtasks=result.subtasks_failed,
                summary=f"Completed {result.subtasks_completed}/{result.subtasks_total} subtasks",
            )
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
            logger.error("Autonomous improve run %s failed: %s", run_id, type(e).__name__)
            store.update_run(
                run_id,
                status="failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error="Orchestration failed",
            )
        finally:
            _active_improve_tasks.pop(run_id, None)
