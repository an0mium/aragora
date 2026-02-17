"""
Canvas Pipeline REST Handler.

Exposes the idea-to-execution pipeline via REST endpoints:

- POST /api/v1/canvas/pipeline/from-debate    → Full pipeline from debate
- POST /api/v1/canvas/pipeline/from-ideas     → Full pipeline from raw ideas
- POST /api/v1/canvas/pipeline/advance        → Advance to next stage
- GET  /api/v1/canvas/pipeline/{id}           → Get pipeline result
- GET  /api/v1/canvas/pipeline/{id}/stage/{s} → Get specific stage canvas
- POST /api/v1/canvas/convert/debate          → Convert debate to ideas canvas
- POST /api/v1/canvas/convert/workflow        → Convert workflow to actions canvas
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# In-memory pipeline result store (production would use persistence)
_pipeline_results: dict[str, Any] = {}
# Live PipelineResult objects for advance_stage()
_pipeline_objects: dict[str, Any] = {}
# Async pipeline tasks
_pipeline_tasks: dict[str, asyncio.Task[Any]] = {}
# Pipeline receipts
_pipeline_receipts: dict[str, dict[str, Any]] = {}


class CanvasPipelineHandler:
    """HTTP handler for the idea-to-execution canvas pipeline."""

    ROUTES = [
        "POST /api/v1/canvas/pipeline/from-debate",
        "POST /api/v1/canvas/pipeline/from-ideas",
        "POST /api/v1/canvas/pipeline/advance",
        "POST /api/v1/canvas/pipeline/run",
        "GET /api/v1/canvas/pipeline/{id}",
        "GET /api/v1/canvas/pipeline/{id}/status",
        "GET /api/v1/canvas/pipeline/{id}/stage/{stage}",
        "GET /api/v1/canvas/pipeline/{id}/graph",
        "GET /api/v1/canvas/pipeline/{id}/receipt",
        "POST /api/v1/canvas/convert/debate",
        "POST /api/v1/canvas/convert/workflow",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None) -> None:
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/canvas/") or path.startswith("/api/canvas/")

    async def handle_from_debate(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """POST /api/v1/canvas/pipeline/from-debate

        Run full pipeline from an ArgumentCartographer debate export.

        Body:
            cartographer_data: dict — ArgumentCartographer.to_dict() output
            auto_advance: bool (default True) — auto-generate all stages
        """
        try:
            from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

            cartographer_data = request_data.get("cartographer_data", {})
            auto_advance = request_data.get("auto_advance", True)

            if not cartographer_data:
                return {"error": "Missing required field: cartographer_data"}

            pipeline = IdeaToExecutionPipeline()
            result = pipeline.from_debate(
                cartographer_data,
                auto_advance=auto_advance,
            )

            # Store result for later retrieval
            result_dict = result.to_dict()
            _pipeline_results[result.pipeline_id] = result_dict
            _pipeline_objects[result.pipeline_id] = result

            return {
                "pipeline_id": result.pipeline_id,
                "stage_status": result.stage_status,
                "stages_completed": sum(
                    1 for s in result.stage_status.values() if s == "complete"
                ),
                "total_nodes": (
                    len(result.ideas_canvas.nodes if result.ideas_canvas else {})
                    + len(result.actions_canvas.nodes if result.actions_canvas else {})
                    + len(
                        result.orchestration_canvas.nodes
                        if result.orchestration_canvas
                        else {}
                    )
                ),
                "result": result_dict,
            }
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Pipeline from-debate failed: %s", e)
            return {"error": "Pipeline execution failed"}

    async def handle_from_ideas(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """POST /api/v1/canvas/pipeline/from-ideas

        Run full pipeline from raw idea strings.

        Body:
            ideas: list[str] — List of idea/thought strings
            auto_advance: bool (default True)
        """
        try:
            from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

            ideas = request_data.get("ideas", [])
            auto_advance = request_data.get("auto_advance", True)

            if not ideas:
                return {"error": "Missing required field: ideas"}

            pipeline = IdeaToExecutionPipeline()
            result = pipeline.from_ideas(ideas, auto_advance=auto_advance)

            result_dict = result.to_dict()
            _pipeline_results[result.pipeline_id] = result_dict
            _pipeline_objects[result.pipeline_id] = result

            return {
                "pipeline_id": result.pipeline_id,
                "stage_status": result.stage_status,
                "goals_count": len(result.goal_graph.goals) if result.goal_graph else 0,
                "result": result_dict,
            }
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Pipeline from-ideas failed: %s", e)
            return {"error": "Pipeline execution failed"}

    async def handle_advance(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """POST /api/v1/canvas/pipeline/advance

        Advance a pipeline to the next stage.

        Body:
            pipeline_id: str — ID of an existing pipeline
            target_stage: str — Stage to advance to (goals, actions, orchestration)
        """
        try:
            from aragora.canvas.stages import PipelineStage
            from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

            pipeline_id = request_data.get("pipeline_id", "")
            target_stage = request_data.get("target_stage", "")

            if not pipeline_id:
                return {"error": "Missing required field: pipeline_id"}
            if not target_stage:
                return {"error": "Missing required field: target_stage"}

            result_obj = _pipeline_objects.get(pipeline_id)
            if not result_obj:
                return {"error": f"Pipeline {pipeline_id} not found"}

            try:
                stage = PipelineStage(target_stage)
            except ValueError:
                return {"error": f"Invalid stage: {target_stage}"}

            pipeline = IdeaToExecutionPipeline()
            result_obj = pipeline.advance_stage(result_obj, stage)

            # Update both stores
            result_dict = result_obj.to_dict()
            _pipeline_results[pipeline_id] = result_dict
            _pipeline_objects[pipeline_id] = result_obj

            return {
                "pipeline_id": pipeline_id,
                "advanced_to": target_stage,
                "stage_status": result_obj.stage_status,
                "result": result_dict,
            }
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Pipeline advance failed: %s", e)
            return {"error": "Pipeline advance failed"}

    async def handle_get_pipeline(
        self, pipeline_id: str
    ) -> dict[str, Any]:
        """GET /api/v1/canvas/pipeline/{id}"""
        result = _pipeline_results.get(pipeline_id)
        if not result:
            return {"error": f"Pipeline {pipeline_id} not found"}
        return result

    async def handle_get_stage(
        self, pipeline_id: str, stage: str
    ) -> dict[str, Any]:
        """GET /api/v1/canvas/pipeline/{id}/stage/{stage}"""
        result = _pipeline_results.get(pipeline_id)
        if not result:
            return {"error": f"Pipeline {pipeline_id} not found"}

        stage_key = {
            "ideas": "ideas",
            "goals": "goals",
            "actions": "actions",
            "orchestration": "orchestration",
        }.get(stage)

        if not stage_key or stage_key not in result:
            return {"error": f"Stage {stage} not found"}

        return {"stage": stage, "data": result[stage_key]}

    async def handle_convert_debate(
        self, request_data: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /api/v1/canvas/convert/debate

        Convert a debate graph to a React Flow-compatible ideas canvas.
        """
        try:
            from aragora.canvas.converters import debate_to_ideas_canvas, to_react_flow

            cartographer_data = request_data.get("cartographer_data", {})
            if not cartographer_data:
                return {"error": "Missing required field: cartographer_data"}

            canvas = debate_to_ideas_canvas(cartographer_data)
            return to_react_flow(canvas)
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Convert debate failed: %s", e)
            return {"error": "Conversion failed"}

    async def handle_convert_workflow(
        self, request_data: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /api/v1/canvas/convert/workflow

        Convert a WorkflowDefinition to a React Flow-compatible actions canvas.
        """
        try:
            from aragora.canvas.converters import (
                to_react_flow,
                workflow_to_actions_canvas,
            )

            workflow_data = request_data.get("workflow_data", {})
            if not workflow_data:
                return {"error": "Missing required field: workflow_data"}

            canvas = workflow_to_actions_canvas(workflow_data)
            return to_react_flow(canvas)
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Convert workflow failed: %s", e)
            return {"error": "Conversion failed"}

    # =========================================================================
    # Async pipeline endpoints (run/status/graph/receipt)
    # =========================================================================

    async def handle_run(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """POST /api/v1/canvas/pipeline/run

        Start an async pipeline execution. Returns immediately with pipeline_id.

        Body:
            input_text: str — The idea/problem statement
            stages: list[str] (optional) — Stages to run
            debate_rounds: int (default 3)
            workflow_mode: str (default "quick")
            dry_run: bool (default False)
            enable_receipts: bool (default True)
        """
        try:
            from aragora.pipeline.idea_to_execution import (
                IdeaToExecutionPipeline,
                PipelineConfig,
            )

            input_text = request_data.get("input_text", "")
            if not input_text:
                return {"error": "Missing required field: input_text"}

            config = PipelineConfig(
                stages_to_run=request_data.get("stages", [
                    "ideation", "goals", "workflow", "orchestration",
                ]),
                debate_rounds=request_data.get("debate_rounds", 3),
                workflow_mode=request_data.get("workflow_mode", "quick"),
                dry_run=request_data.get("dry_run", False),
                enable_receipts=request_data.get("enable_receipts", True),
            )

            # Set up stream emitter as event callback
            try:
                from aragora.server.stream.pipeline_stream import get_pipeline_emitter
                emitter = get_pipeline_emitter()
            except ImportError:
                emitter = None

            pipeline = IdeaToExecutionPipeline()

            async def _run_pipeline() -> None:
                if emitter:
                    config.event_callback = emitter.as_event_callback(pipeline_id)
                result = await pipeline.run(input_text, config)
                result_dict = result.to_dict()
                _pipeline_results[result.pipeline_id] = result_dict
                _pipeline_objects[result.pipeline_id] = result
                if result.receipt:
                    _pipeline_receipts[result.pipeline_id] = result.receipt

            # Generate pipeline_id before launching task
            import uuid
            pipeline_id = f"pipe-{uuid.uuid4().hex[:8]}"
            # Store placeholder so status queries work immediately
            _pipeline_results[pipeline_id] = {
                "pipeline_id": pipeline_id,
                "stage_status": {"ideas": "pending", "goals": "pending", "actions": "pending", "orchestration": "pending"},
                "status": "running",
            }

            task = asyncio.create_task(_run_pipeline())
            _pipeline_tasks[pipeline_id] = task

            return {
                "pipeline_id": pipeline_id,
                "status": "running",
                "stages": config.stages_to_run,
            }
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Pipeline run failed: %s", e)
            return {"error": "Pipeline execution failed"}

    async def handle_status(self, pipeline_id: str) -> dict[str, Any]:
        """GET /api/v1/canvas/pipeline/{id}/status

        Get per-stage status for a pipeline.
        """
        result = _pipeline_results.get(pipeline_id)
        if not result:
            return {"error": f"Pipeline {pipeline_id} not found"}

        # Check if async task is still running
        task = _pipeline_tasks.get(pipeline_id)
        is_running = task is not None and not task.done()

        status_info: dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "status": "running" if is_running else "completed",
            "stage_status": result.get("stage_status", {}),
        }

        if result.get("stage_results"):
            status_info["stage_results"] = result["stage_results"]
        if result.get("duration"):
            status_info["duration"] = result["duration"]

        return status_info

    async def handle_graph(
        self, pipeline_id: str, request_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """GET /api/v1/canvas/pipeline/{id}/graph

        Get React Flow JSON for any stage of the pipeline.

        Query params (via request_data):
            stage: str (optional) — specific stage (ideas, goals, actions, orchestration)
        """
        result = _pipeline_results.get(pipeline_id)
        if not result:
            return {"error": f"Pipeline {pipeline_id} not found"}

        stage = (request_data or {}).get("stage", "")

        graphs: dict[str, Any] = {}
        if not stage or stage == "ideas":
            if result.get("ideas"):
                graphs["ideas"] = result["ideas"]
        if not stage or stage == "goals":
            if result.get("goals"):
                # Convert goals to React Flow nodes
                goals_data = result["goals"]
                rf_nodes = []
                rf_edges = []
                for i, goal in enumerate(goals_data.get("goals", [])):
                    rf_nodes.append({
                        "id": goal.get("id", f"goal-{i}"),
                        "type": "goalNode",
                        "position": {"x": 100, "y": i * 120},
                        "data": goal,
                    })
                    for dep in goal.get("dependencies", []):
                        rf_edges.append({
                            "id": f"dep-{dep}-{goal['id']}",
                            "source": dep,
                            "target": goal["id"],
                        })
                graphs["goals"] = {"nodes": rf_nodes, "edges": rf_edges}
        if not stage or stage == "actions":
            if result.get("actions"):
                graphs["actions"] = result["actions"]
        if not stage or stage == "orchestration":
            if result.get("orchestration"):
                graphs["orchestration"] = result["orchestration"]

        # If final_workflow present, add it
        if not stage or stage == "workflow":
            wf = result.get("final_workflow")
            if wf:
                rf_nodes = []
                rf_edges = []
                for i, step in enumerate(wf.get("steps", [])):
                    rf_nodes.append({
                        "id": step.get("id", f"step-{i}"),
                        "type": "workflowStep",
                        "position": {"x": 200, "y": i * 100},
                        "data": step,
                    })
                for trans in wf.get("transitions", []):
                    rf_edges.append({
                        "id": trans.get("id", ""),
                        "source": trans.get("from_step", ""),
                        "target": trans.get("to_step", ""),
                    })
                graphs["workflow"] = {"nodes": rf_nodes, "edges": rf_edges}

        return {"pipeline_id": pipeline_id, "graphs": graphs}

    async def handle_receipt(self, pipeline_id: str) -> dict[str, Any]:
        """GET /api/v1/canvas/pipeline/{id}/receipt

        Get the DecisionReceipt for a completed pipeline.
        """
        receipt = _pipeline_receipts.get(pipeline_id)
        if receipt:
            return {"pipeline_id": pipeline_id, "receipt": receipt}

        # Check if pipeline result has a receipt
        result = _pipeline_results.get(pipeline_id)
        if not result:
            return {"error": f"Pipeline {pipeline_id} not found"}

        if result.get("receipt"):
            return {"pipeline_id": pipeline_id, "receipt": result["receipt"]}

        return {"error": f"No receipt available for pipeline {pipeline_id}"}
