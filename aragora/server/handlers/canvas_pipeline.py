"""
Canvas Pipeline REST Handler.

Exposes the idea-to-execution pipeline via REST endpoints:

- POST /api/v1/canvas/pipeline/from-debate              → Full pipeline from debate
- POST /api/v1/canvas/pipeline/from-ideas               → Full pipeline from raw ideas
- POST /api/v1/canvas/pipeline/advance                  → Advance to next stage
- POST /api/v1/canvas/pipeline/run                      → Start async pipeline
- POST /api/v1/canvas/pipeline/{id}/approve-transition  → Approve/reject stage transition
- PUT  /api/v1/canvas/pipeline/{id}                     → Save canvas state
- GET  /api/v1/canvas/pipeline/{id}                     → Get pipeline result
- GET  /api/v1/canvas/pipeline/{id}/status              → Per-stage status
- GET  /api/v1/canvas/pipeline/{id}/stage/{s}           → Get specific stage canvas
- GET  /api/v1/canvas/pipeline/{id}/graph               → React Flow JSON for any stage
- GET  /api/v1/canvas/pipeline/{id}/receipt              → DecisionReceipt
- POST /api/v1/canvas/pipeline/extract-goals            → Extract goals from ideas canvas
- POST /api/v1/canvas/convert/debate                    → Convert debate to ideas canvas
- POST /api/v1/canvas/convert/workflow                  → Convert workflow to actions canvas
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from aragora.server.handlers.base import HandlerResult, error_response, handle_errors, json_response

logger = logging.getLogger(__name__)

# Path patterns for route dispatch
_PIPELINE_ID = re.compile(r"^/api/v1/canvas/pipeline/([a-zA-Z0-9_-]+)$")
_PIPELINE_STATUS = re.compile(r"^/api/v1/canvas/pipeline/([a-zA-Z0-9_-]+)/status$")
_PIPELINE_STAGE = re.compile(r"^/api/v1/canvas/pipeline/([a-zA-Z0-9_-]+)/stage/(\w+)$")
_PIPELINE_GRAPH = re.compile(r"^/api/v1/canvas/pipeline/([a-zA-Z0-9_-]+)/graph$")
_PIPELINE_RECEIPT = re.compile(r"^/api/v1/canvas/pipeline/([a-zA-Z0-9_-]+)/receipt$")

# Live PipelineResult objects for advance_stage() (cannot be persisted)
_pipeline_objects: dict[str, Any] = {}
# Async pipeline tasks (cannot be persisted)
_pipeline_tasks: dict[str, asyncio.Task[Any]] = {}


def _get_store() -> Any:
    """Lazy-load the persistent pipeline store."""
    from aragora.storage.pipeline_store import get_pipeline_store
    return get_pipeline_store()


def _get_ai_agent() -> Any | None:
    """Try to create an AI agent for goal synthesis.

    Returns an agent with a generate() method, or None if unavailable.
    """
    try:
        from aragora.agents.api_agents.anthropic import AnthropicAPIAgent
        return AnthropicAPIAgent(model="claude-sonnet-4-5-20250929")
    except (ImportError, OSError, ValueError):
        pass
    try:
        from aragora.agents.api_agents.openai import OpenAIAPIAgent
        return OpenAIAPIAgent(model="gpt-4o-mini")
    except (ImportError, OSError, ValueError):
        pass
    return None


def _persist_universal_graph(result: Any) -> None:
    """Persist the UniversalGraph from a PipelineResult to GraphStore."""
    if result.universal_graph is None:
        return
    try:
        from aragora.pipeline.graph_store import get_graph_store
        store = get_graph_store()
        store.create(result.universal_graph)
        logger.info(
            "Persisted universal graph %s with %d nodes",
            result.universal_graph.id,
            len(result.universal_graph.nodes),
        )
    except (ImportError, OSError) as e:
        logger.debug("Could not persist universal graph: %s", e)


class CanvasPipelineHandler:
    """HTTP handler for the idea-to-execution canvas pipeline."""

    ROUTES = [
        "POST /api/v1/canvas/pipeline/from-debate",
        "POST /api/v1/canvas/pipeline/from-ideas",
        "POST /api/v1/canvas/pipeline/advance",
        "POST /api/v1/canvas/pipeline/run",
        "POST /api/v1/canvas/pipeline/{id}/approve-transition",
        "GET /api/v1/canvas/pipeline/{id}",
        "GET /api/v1/canvas/pipeline/{id}/status",
        "GET /api/v1/canvas/pipeline/{id}/stage/{stage}",
        "GET /api/v1/canvas/pipeline/{id}/graph",
        "GET /api/v1/canvas/pipeline/{id}/receipt",
        "PUT /api/v1/canvas/pipeline/{id}",
        "POST /api/v1/canvas/pipeline/extract-goals",
        "POST /api/v1/canvas/convert/debate",
        "POST /api/v1/canvas/convert/workflow",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None) -> None:
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/canvas/") or path.startswith("/api/canvas/")

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> Any:
        """Dispatch GET requests to the appropriate handler method."""
        body = self._get_request_body(handler)

        # GET /api/v1/canvas/pipeline/{id}/status
        m = _PIPELINE_STATUS.match(path)
        if m:
            return self.handle_status(m.group(1))

        # GET /api/v1/canvas/pipeline/{id}/graph
        m = _PIPELINE_GRAPH.match(path)
        if m:
            return self.handle_graph(m.group(1), query_params)

        # GET /api/v1/canvas/pipeline/{id}/receipt
        m = _PIPELINE_RECEIPT.match(path)
        if m:
            return self.handle_receipt(m.group(1))

        # GET /api/v1/canvas/pipeline/{id}/stage/{stage}
        m = _PIPELINE_STAGE.match(path)
        if m:
            return self.handle_get_stage(m.group(1), m.group(2))

        # GET /api/v1/canvas/pipeline/{id}
        m = _PIPELINE_ID.match(path)
        if m:
            return self.handle_get_pipeline(m.group(1))

        return None

    def _check_permission(self, handler: Any, permission: str) -> Any:
        """Check RBAC permission and return error response if denied."""
        try:
            from aragora.billing.jwt_auth import extract_user_from_request
            from aragora.rbac.checker import get_permission_checker
            from aragora.rbac.models import AuthorizationContext
            from aragora.server.handlers.utils.responses import error_response

            user_ctx = extract_user_from_request(handler, None)
            if not user_ctx or not user_ctx.is_authenticated:
                return error_response("Authentication required", status=401)

            auth_ctx = AuthorizationContext(
                user_id=user_ctx.user_id,
                user_email=user_ctx.email,
                org_id=user_ctx.org_id,
                workspace_id=None,
                roles={user_ctx.role} if user_ctx.role else {"member"},
            )
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, permission)
            if not decision.allowed:
                logger.warning("Permission denied: %s", permission)
                return error_response("Permission denied", status=403)
            return None
        except (ImportError, AttributeError, ValueError) as e:
            logger.debug("Permission check unavailable: %s", e)
            return None

    @handle_errors("canvas pipeline operation")
    def handle_post(self, path: str, query_params: dict[str, Any], handler: Any) -> Any:
        """Dispatch POST requests to the appropriate handler method."""
        # Match route first so unknown paths return None (letting other handlers try)
        route_map = {
            "/from-debate": self.handle_from_debate,
            "/from-ideas": self.handle_from_ideas,
            "/pipeline/advance": self.handle_advance,
            "/pipeline/run": self.handle_run,
            "/pipeline/extract-goals": self.handle_extract_goals,
            "/convert/debate": self.handle_convert_debate,
            "/convert/workflow": self.handle_convert_workflow,
        }

        # Check for transition approval: /api/v1/canvas/pipeline/{id}/approve-transition
        if "/approve-transition" in path:
            auth_error = self._check_permission(handler, "pipeline:write")
            if auth_error:
                return auth_error
            body = self._get_request_body(handler)
            m = re.match(r".*/pipeline/([a-zA-Z0-9_-]+)/approve-transition$", path)
            if m:
                return self.handle_approve_transition(m.group(1), body)
            return None

        target = None
        for suffix, method in route_map.items():
            if path.endswith(suffix):
                target = method
                break

        if target is None:
            return None

        auth_error = self._check_permission(handler, "pipeline:write")
        if auth_error:
            return auth_error

        body = self._get_request_body(handler)
        return target(body)

    def handle_put(self, path: str, query_params: dict[str, Any], handler: Any) -> Any:
        """Dispatch PUT requests — save canvas state.

        PUT /api/v1/canvas/pipeline/{id}
        """
        m = _PIPELINE_ID.match(path)
        if not m:
            return None

        auth_error = self._check_permission(handler, "pipeline:write")
        if auth_error:
            return auth_error

        body = self._get_request_body(handler)
        return self.handle_save_pipeline(m.group(1), body)

    @staticmethod
    def _get_request_body(handler: Any) -> dict[str, Any]:
        """Extract JSON body from the request handler."""
        try:
            if hasattr(handler, "request") and hasattr(handler.request, "body"):
                raw = handler.request.body
                if raw:
                    return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            pass
        return {}

    async def handle_from_debate(self, request_data: dict[str, Any]) -> HandlerResult:
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
            use_ai = request_data.get("use_ai", False)

            if not cartographer_data:
                return error_response("Missing required field: cartographer_data", 400)

            use_universal = request_data.get("use_universal", False)
            agent = _get_ai_agent() if use_ai else None
            pipeline = IdeaToExecutionPipeline(
                agent=agent, use_universal=use_universal,
            )

            # Wire stream emitter for real-time progress
            event_cb = None
            try:
                from aragora.server.stream.pipeline_stream import get_pipeline_emitter
                event_cb = get_pipeline_emitter().as_event_callback(
                    f"pipe-from-debate"  # placeholder until pipeline_id is known
                )
            except ImportError:
                pass

            result = pipeline.from_debate(
                cartographer_data,
                auto_advance=auto_advance,
                event_callback=event_cb,
            )

            # Persist result and keep live object in memory
            result_dict = result.to_dict()
            _get_store().save(result.pipeline_id, result_dict)
            _pipeline_objects[result.pipeline_id] = result

            # Persist universal graph if generated
            _persist_universal_graph(result)

            return json_response({
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
                "has_universal_graph": result.universal_graph is not None,
                "result": result_dict,
            }, 201)
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Pipeline from-debate failed: %s", e)
            return error_response("Pipeline execution failed", 500)

    async def handle_from_ideas(self, request_data: dict[str, Any]) -> HandlerResult:
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
            use_ai = request_data.get("use_ai", False)

            if not ideas:
                return error_response("Missing required field: ideas", 400)

            use_universal = request_data.get("use_universal", False)
            agent = _get_ai_agent() if use_ai else None
            pipeline = IdeaToExecutionPipeline(
                agent=agent, use_universal=use_universal,
            )

            # Wire stream emitter for real-time progress
            event_cb = None
            try:
                from aragora.server.stream.pipeline_stream import get_pipeline_emitter
                event_cb = get_pipeline_emitter().as_event_callback(
                    f"pipe-from-ideas"  # placeholder until pipeline_id is known
                )
            except ImportError:
                pass

            result = pipeline.from_ideas(
                ideas, auto_advance=auto_advance, event_callback=event_cb,
            )

            result_dict = result.to_dict()
            _get_store().save(result.pipeline_id, result_dict)
            _pipeline_objects[result.pipeline_id] = result
            _persist_universal_graph(result)

            return json_response({
                "pipeline_id": result.pipeline_id,
                "stage_status": result.stage_status,
                "goals_count": len(result.goal_graph.goals) if result.goal_graph else 0,
                "has_universal_graph": result.universal_graph is not None,
                "result": result_dict,
            }, 201)
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Pipeline from-ideas failed: %s", e)
            return error_response("Pipeline execution failed", 500)

    async def handle_advance(self, request_data: dict[str, Any]) -> HandlerResult:
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
                return error_response("Missing required field: pipeline_id", 400)
            if not target_stage:
                return error_response("Missing required field: target_stage", 400)

            result_obj = _pipeline_objects.get(pipeline_id)
            if not result_obj:
                return error_response(f"Pipeline {pipeline_id} not found", 404)

            try:
                stage = PipelineStage(target_stage)
            except ValueError:
                return error_response(f"Invalid stage: {target_stage}", 400)

            pipeline = IdeaToExecutionPipeline()
            result_obj = pipeline.advance_stage(result_obj, stage)

            # Persist updated result and keep live object
            result_dict = result_obj.to_dict()
            _get_store().save(pipeline_id, result_dict)
            _pipeline_objects[pipeline_id] = result_obj

            return json_response({
                "pipeline_id": pipeline_id,
                "advanced_to": target_stage,
                "stage_status": result_obj.stage_status,
                "result": result_dict,
            })
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Pipeline advance failed: %s", e)
            return error_response("Pipeline advance failed", 500)

    async def handle_get_pipeline(
        self, pipeline_id: str
    ) -> HandlerResult:
        """GET /api/v1/canvas/pipeline/{id}"""
        result = _get_store().get(pipeline_id)
        if not result:
            return error_response(f"Pipeline {pipeline_id} not found", 404)
        return json_response(result)

    async def handle_get_stage(
        self, pipeline_id: str, stage: str
    ) -> HandlerResult:
        """GET /api/v1/canvas/pipeline/{id}/stage/{stage}"""
        result = _get_store().get(pipeline_id)
        if not result:
            return error_response(f"Pipeline {pipeline_id} not found", 404)

        stage_key = {
            "ideas": "ideas",
            "goals": "goals",
            "actions": "actions",
            "orchestration": "orchestration",
        }.get(stage)

        if not stage_key or stage_key not in result:
            return error_response(f"Stage {stage} not found", 404)

        return json_response({"stage": stage, "data": result[stage_key]})

    async def handle_convert_debate(
        self, request_data: dict[str, Any]
    ) -> HandlerResult:
        """POST /api/v1/canvas/convert/debate

        Convert a debate graph to a React Flow-compatible ideas canvas.
        """
        try:
            from aragora.canvas.converters import debate_to_ideas_canvas, to_react_flow

            cartographer_data = request_data.get("cartographer_data", {})
            if not cartographer_data:
                return error_response("Missing required field: cartographer_data", 400)

            canvas = debate_to_ideas_canvas(cartographer_data)
            return json_response(to_react_flow(canvas))
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Convert debate failed: %s", e)
            return error_response("Conversion failed", 500)

    async def handle_convert_workflow(
        self, request_data: dict[str, Any]
    ) -> HandlerResult:
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
                return error_response("Missing required field: workflow_data", 400)

            canvas = workflow_to_actions_canvas(workflow_data)
            return json_response(to_react_flow(canvas))
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Convert workflow failed: %s", e)
            return error_response("Conversion failed", 500)

    # =========================================================================
    # Async pipeline endpoints (run/status/graph/receipt)
    # =========================================================================

    async def handle_run(self, request_data: dict[str, Any]) -> HandlerResult:
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
            use_ai = request_data.get("use_ai", False)
            if not input_text:
                return error_response("Missing required field: input_text", 400)

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

            use_universal = request_data.get("use_universal", False)
            agent = _get_ai_agent() if use_ai else None
            pipeline = IdeaToExecutionPipeline(
                agent=agent, use_universal=use_universal,
            )

            async def _run_pipeline() -> None:
                if emitter:
                    config.event_callback = emitter.as_event_callback(pipeline_id)
                result = await pipeline.run(input_text, config, pipeline_id=pipeline_id)
                result_dict = result.to_dict()
                _get_store().save(pipeline_id, result_dict)
                _pipeline_objects[pipeline_id] = result
                _persist_universal_graph(result)

            # Generate pipeline_id before launching task
            import uuid
            pipeline_id = f"pipe-{uuid.uuid4().hex[:8]}"
            # Store placeholder so status queries work immediately
            store = _get_store()
            store.save(pipeline_id, {
                "stage_status": {"ideas": "pending", "goals": "pending", "actions": "pending", "orchestration": "pending"},
            })

            task = asyncio.create_task(_run_pipeline())
            task.add_done_callback(
                lambda t: logger.error(
                    "Canvas pipeline task failed: %s", t.exception(),
                )
                if not t.cancelled() and t.exception()
                else None
            )
            _pipeline_tasks[pipeline_id] = task

            return json_response({
                "pipeline_id": pipeline_id,
                "status": "running",
                "stages": config.stages_to_run,
            }, 202)
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Pipeline run failed: %s", e)
            return error_response("Pipeline execution failed", 500)

    async def handle_status(self, pipeline_id: str) -> HandlerResult:
        """GET /api/v1/canvas/pipeline/{id}/status

        Get per-stage status for a pipeline.
        """
        result = _get_store().get(pipeline_id)
        if not result:
            return error_response(f"Pipeline {pipeline_id} not found", 404)

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

        return json_response(status_info)

    async def handle_graph(
        self, pipeline_id: str, request_data: dict[str, Any] | None = None,
    ) -> HandlerResult:
        """GET /api/v1/canvas/pipeline/{id}/graph

        Get React Flow JSON for any stage of the pipeline.

        Query params (via request_data):
            stage: str (optional) — specific stage (ideas, goals, actions, orchestration)
        """
        result = _get_store().get(pipeline_id)
        if not result:
            return error_response(f"Pipeline {pipeline_id} not found", 404)

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

        return json_response({"pipeline_id": pipeline_id, "graphs": graphs})

    async def handle_receipt(self, pipeline_id: str) -> HandlerResult:
        """GET /api/v1/canvas/pipeline/{id}/receipt

        Get the DecisionReceipt for a completed pipeline.
        """
        result = _get_store().get(pipeline_id)
        if not result:
            return error_response(f"Pipeline {pipeline_id} not found", 404)

        if result.get("receipt"):
            return json_response({"pipeline_id": pipeline_id, "receipt": result["receipt"]})

        return error_response(f"No receipt available for pipeline {pipeline_id}", 404)

    async def handle_extract_goals(
        self, request_data: dict[str, Any]
    ) -> HandlerResult:
        """POST /api/v1/canvas/pipeline/extract-goals

        Extract goals from an ideas canvas using GoalExtractor.

        Body:
            ideas_canvas_id: str — ID of the ideas canvas to extract from
            ideas_canvas_data: dict (optional) — Raw canvas data (if not using store)
            config: dict (optional) — GoalExtractionConfig overrides
        """
        try:
            from aragora.goals.extractor import GoalExtractor, GoalExtractionConfig

            canvas_data = request_data.get("ideas_canvas_data")
            canvas_id = request_data.get("ideas_canvas_id", "")

            # If no raw data provided, try loading from store
            if not canvas_data and canvas_id:
                try:
                    from aragora.canvas.idea_store import get_idea_canvas_store
                    from aragora.canvas import get_canvas_manager

                    manager = get_canvas_manager()
                    canvas = await manager.get_canvas(canvas_id)
                    if canvas:
                        canvas_data = {
                            "nodes": [n.to_dict() for n in canvas.nodes.values()],
                            "edges": [e.to_dict() for e in canvas.edges.values()],
                        }
                except (ImportError, RuntimeError, OSError) as e:
                    logger.debug("Could not load canvas from store: %s", e)

            if not canvas_data:
                return error_response("Missing ideas_canvas_data or valid ideas_canvas_id", 400)

            # Build extraction config from request
            config_data = request_data.get("config", {})
            config = GoalExtractionConfig(
                confidence_threshold=float(config_data.get("confidence_threshold", 0.6)),
                max_goals=int(config_data.get("max_goals", 10)),
                require_consensus=bool(config_data.get("require_consensus", True)),
                smart_scoring=bool(config_data.get("smart_scoring", True)),
            )

            extractor = GoalExtractor()
            goal_graph = extractor.extract_from_ideas(canvas_data)

            # Filter by confidence threshold
            if config.confidence_threshold > 0:
                goal_graph.goals = [
                    g for g in goal_graph.goals
                    if g.confidence >= config.confidence_threshold
                ]

            # Limit to max_goals
            if config.max_goals and len(goal_graph.goals) > config.max_goals:
                goal_graph.goals = goal_graph.goals[:config.max_goals]

            result = goal_graph.to_dict()
            result["source_canvas_id"] = canvas_id
            result["goals_count"] = len(goal_graph.goals)

            return json_response(result)
        except (ImportError, ValueError, TypeError) as e:
            logger.warning("Goal extraction failed: %s", e)
            return error_response("Goal extraction failed", 500)

    # =========================================================================
    # PUT: Save canvas state
    # =========================================================================

    async def handle_save_pipeline(
        self, pipeline_id: str, request_data: dict[str, Any],
    ) -> HandlerResult:
        """PUT /api/v1/canvas/pipeline/{id}

        Save the current canvas state (nodes + edges) for all stages.

        Body:
            pipeline_id: str
            stages: {
                ideas: { nodes: [...], edges: [...] },
                goals: { nodes: [...], edges: [...] },
                actions: { nodes: [...], edges: [...] },
                orchestration: { nodes: [...], edges: [...] },
            }
        """
        store = _get_store()
        existing = store.get(pipeline_id)
        if not existing:
            # Allow creating a new pipeline via PUT
            existing = {
                "pipeline_id": pipeline_id,
                "stage_status": {},
            }

        stages = request_data.get("stages", {})
        if not stages:
            return error_response("Missing required field: stages", 400)

        # Merge each stage's canvas data into the stored result
        for stage_name in ("ideas", "goals", "actions", "orchestration"):
            stage_data = stages.get(stage_name)
            if stage_data is not None:
                existing[stage_name] = {
                    "nodes": stage_data.get("nodes", []),
                    "edges": stage_data.get("edges", []),
                }
                # Mark stage as complete if it has nodes
                if stage_data.get("nodes"):
                    existing.setdefault("stage_status", {})[stage_name] = "complete"

        store.save(pipeline_id, existing)

        return json_response({
            "pipeline_id": pipeline_id,
            "saved": True,
            "stage_status": existing.get("stage_status", {}),
        })

    # =========================================================================
    # POST: Approve/reject stage transition
    # =========================================================================

    async def handle_approve_transition(
        self, pipeline_id: str, request_data: dict[str, Any],
    ) -> HandlerResult:
        """POST /api/v1/canvas/pipeline/{id}/approve-transition

        Approve or reject a pending stage transition.

        Body:
            from_stage: str — Source stage (e.g., "ideas")
            to_stage: str — Target stage (e.g., "goals")
            approved: bool — Whether to approve the transition
            comment: str (optional) — Human reviewer comment
        """
        store = _get_store()
        existing = store.get(pipeline_id)
        if not existing:
            return error_response(f"Pipeline {pipeline_id} not found", 404)

        from_stage = request_data.get("from_stage", "")
        to_stage = request_data.get("to_stage", "")
        approved = request_data.get("approved", False)
        comment = request_data.get("comment", "")

        if not from_stage or not to_stage:
            return error_response("Missing required fields: from_stage, to_stage", 400)

        # Find and update the matching transition
        transitions = existing.get("transitions", [])
        updated = False
        for transition in transitions:
            t_from = transition.get("from_stage", "")
            t_to = transition.get("to_stage", "")
            if t_from == from_stage and t_to == to_stage:
                transition["status"] = "approved" if approved else "rejected"
                transition["human_comment"] = comment
                transition["reviewed_at"] = time.time()
                updated = True
                break

        if not updated:
            # Create a new transition record if none exists
            transitions.append({
                "from_stage": from_stage,
                "to_stage": to_stage,
                "status": "approved" if approved else "rejected",
                "human_comment": comment,
                "reviewed_at": time.time(),
            })
            existing["transitions"] = transitions

        # If approved, advance the pipeline to the next stage
        if approved:
            stage_status = existing.get("stage_status", {})
            stage_status[from_stage] = "complete"
            if to_stage not in stage_status or stage_status[to_stage] == "pending":
                stage_status[to_stage] = "active"
            existing["stage_status"] = stage_status

        store.save(pipeline_id, existing)

        return json_response({
            "pipeline_id": pipeline_id,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "status": "approved" if approved else "rejected",
            "comment": comment,
        })
