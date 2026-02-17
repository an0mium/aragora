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

import logging
from typing import Any

logger = logging.getLogger(__name__)

# In-memory pipeline result store (production would use persistence)
_pipeline_results: dict[str, Any] = {}
# Live PipelineResult objects for advance_stage()
_pipeline_objects: dict[str, Any] = {}


class CanvasPipelineHandler:
    """HTTP handler for the idea-to-execution canvas pipeline."""

    ROUTES = [
        "POST /api/v1/canvas/pipeline/from-debate",
        "POST /api/v1/canvas/pipeline/from-ideas",
        "POST /api/v1/canvas/pipeline/advance",
        "GET /api/v1/canvas/pipeline/{id}",
        "GET /api/v1/canvas/pipeline/{id}/stage/{stage}",
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
