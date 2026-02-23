"""Pipeline execution trigger handler.

Executes a pipeline's Stage 4 (Orchestration) nodes via the self-improvement
pipeline, bridging the visual canvas editor to autonomous code execution.

Endpoints:
- POST /api/v1/pipeline/:pipeline_id/execute  Start execution
- GET  /api/v1/pipeline/:pipeline_id/execute   Get execution status
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from ..base import (
    SAFE_ID_PATTERN,
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    validate_path_segment,
    handle_errors,
)
from ..utils.decorators import require_permission
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

_execute_limiter = RateLimiter(requests_per_minute=10)

# Active pipeline executions: pipeline_id -> execution state
_executions: dict[str, dict[str, Any]] = {}
_execution_tasks: dict[str, asyncio.Task[Any]] = {}


class PipelineExecuteHandler(BaseHandler):
    """Handler for pipeline execution via self-improvement pipeline.

    Converts orchestration nodes from a pipeline's Stage 4 into
    PrioritizedGoals and executes them through SelfImprovePipeline.
    """

    ROUTES = ["/api/v1/pipeline"]

    def __init__(self, ctx: dict[str, Any] | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        cleaned = strip_version_prefix(path)
        # Match /api/pipeline/:id/execute
        parts = cleaned.strip("/").split("/")
        return (
            len(parts) >= 4
            and parts[0] == "api"
            and parts[1] == "pipeline"
            and parts[3] == "execute"
        )

    def _extract_pipeline_id(self, path: str) -> str | None:
        """Extract pipeline_id from /api/pipeline/:id/execute."""
        cleaned = strip_version_prefix(path)
        parts = cleaned.strip("/").split("/")
        if len(parts) >= 4 and parts[1] == "pipeline" and parts[3] == "execute":
            pid = parts[2]
            if validate_path_segment(pid, SAFE_ID_PATTERN):
                return pid
        return None

    @require_permission("pipeline:read")
    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """GET /api/v1/pipeline/:pipeline_id/execute — execution status."""
        pipeline_id = self._extract_pipeline_id(path)
        if not pipeline_id:
            return error_response("Invalid pipeline ID", 400)

        execution = _executions.get(pipeline_id)
        if not execution:
            return json_response({"pipeline_id": pipeline_id, "status": "not_started"})

        return json_response(execution)

    @handle_errors("pipeline execute")
    @require_permission("pipeline:execute")
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """POST /api/v1/pipeline/:pipeline_id/execute — start execution."""
        client_ip = get_client_ip(handler)
        if not _execute_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        pipeline_id = self._extract_pipeline_id(path)
        if not pipeline_id:
            return error_response("Invalid pipeline ID", 400)

        # Check if already executing
        if pipeline_id in _execution_tasks:
            task = _execution_tasks[pipeline_id]
            if not task.done():
                return error_response("Pipeline is already executing", 409)

        body = self.read_json_body(handler) or {}
        budget_limit = body.get("budget_limit_usd")
        require_approval = body.get("require_approval", False)
        dry_run = body.get("dry_run", False)

        # Load orchestration nodes from the pipeline graph
        orch_nodes = self._load_orchestration_nodes(pipeline_id)
        if not orch_nodes:
            return error_response(
                "No orchestration nodes found in pipeline", 404
            )

        # Convert to goals
        goals = self._convert_to_goals(orch_nodes, pipeline_id)

        cycle_id = f"pipe-{uuid.uuid4().hex[:12]}"

        # Store execution state
        _executions[pipeline_id] = {
            "pipeline_id": pipeline_id,
            "cycle_id": cycle_id,
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "goal_count": len(goals),
            "dry_run": dry_run,
        }

        if dry_run:
            _executions[pipeline_id]["status"] = "preview"
            _executions[pipeline_id]["goals"] = [
                {"description": g.description, "track": g.track.value, "priority": g.priority}
                for g in goals
            ]
            return json_response(_executions[pipeline_id])

        # Start background execution
        task = asyncio.create_task(
            self._execute_pipeline(pipeline_id, cycle_id, goals, budget_limit, require_approval)
        )
        _execution_tasks[pipeline_id] = task

        return json_response(
            {"pipeline_id": pipeline_id, "cycle_id": cycle_id, "status": "started"},
            status=202,
        )

    def _load_orchestration_nodes(self, pipeline_id: str) -> list[dict[str, Any]]:
        """Load Stage 4 orchestration nodes from the pipeline graph."""
        try:
            from aragora.pipeline.graph_store import get_graph_store

            store = get_graph_store()
            graph = store.get(pipeline_id)
            if not graph:
                return []

            # Filter to orchestration stage nodes
            nodes = []
            graph_nodes = getattr(graph, "nodes", {})
            if isinstance(graph_nodes, dict):
                for node_id, node in graph_nodes.items():
                    node_data = getattr(node, "data", node) if not isinstance(node, dict) else node
                    stage = node_data.get("stage", "")
                    if stage == "orchestration":
                        nodes.append({"id": node_id, **node_data})
            return nodes
        except (ImportError, RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning("Failed to load orchestration nodes: %s", type(e).__name__)
            return []

    def _convert_to_goals(
        self, orch_nodes: list[dict[str, Any]], pipeline_id: str
    ) -> list[Any]:
        """Convert orchestration nodes to PrioritizedGoal objects."""
        from aragora.nomic.meta_planner import PrioritizedGoal, Track

        goals = []
        for i, node in enumerate(orch_nodes, start=1):
            label = node.get("label", node.get("description", f"Task {i}"))
            orch_type = node.get("orch_type", node.get("orchType", "agent_task"))
            assigned_agent = node.get("assigned_agent", node.get("assignedAgent", ""))

            # Map orch type to track
            track_map = {
                "agent_task": Track.CORE,
                "debate": Track.CORE,
                "human_gate": Track.CORE,
                "verification": Track.QA,
                "parallel_fan": Track.DEVELOPER,
                "merge": Track.DEVELOPER,
            }
            track = track_map.get(orch_type, Track.CORE)

            goals.append(PrioritizedGoal(
                id=f"pipe-goal-{pipeline_id[:8]}-{i}",
                track=track,
                description=label,
                rationale=f"From pipeline {pipeline_id} orchestration node {node.get('id', i)}",
                estimated_impact="medium",
                priority=i,
                focus_areas=[orch_type],
                file_hints=[],
            ))

        return goals

    async def _execute_pipeline(
        self,
        pipeline_id: str,
        cycle_id: str,
        goals: list[Any],
        budget_limit: float | None,
        require_approval: bool,
    ) -> None:
        """Execute pipeline goals via SelfImprovePipeline in background."""
        try:
            from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline

            # Build combined goal description from all orchestration nodes
            combined_goal = "; ".join(g.description for g in goals[:10])

            config = SelfImproveConfig(
                budget_limit_usd=budget_limit or 10.0,
                require_approval=require_approval,
                autonomous=not require_approval,
                max_goals=len(goals),
            )
            pipeline = SelfImprovePipeline(config=config)
            result = await pipeline.run(combined_goal)

            _executions[pipeline_id].update({
                "status": "completed" if result.subtasks_completed > 0 else "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "total_subtasks": result.subtasks_total,
                "completed_subtasks": result.subtasks_completed,
                "failed_subtasks": result.subtasks_failed,
            })

            # Generate provenance receipt
            try:
                from aragora.pipeline.receipt_generator import generate_pipeline_receipt

                await generate_pipeline_receipt(pipeline_id, _executions[pipeline_id])
            except (ImportError, RuntimeError, ValueError, OSError) as e:
                logger.debug("Receipt generation skipped: %s", type(e).__name__)

        except asyncio.CancelledError:
            _executions[pipeline_id].update({
                "status": "cancelled",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        except ImportError:
            logger.debug("SelfImprovePipeline not available")
            _executions[pipeline_id].update({
                "status": "failed",
                "error": "Self-improvement pipeline not available",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error("Pipeline execution failed: %s", type(e).__name__)
            _executions[pipeline_id].update({
                "status": "failed",
                "error": "Pipeline execution failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            _execution_tasks.pop(pipeline_id, None)
