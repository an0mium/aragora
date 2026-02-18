"""Visual Workflow Builder API handlers.

Provides endpoints for the visual workflow builder:
- NL-to-workflow generation
- Auto-layout
- Step type catalog
- Pattern-based creation
- Workflow validation
- Execution replay
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from aragora.server.handlers.utils.decorators import require_permission

from .core import track_handler, _run_async

logger = logging.getLogger(__name__)


class WorkflowBuilderHandler(BaseHandler):
    """HTTP handlers for visual workflow builder API."""

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/api/v1/workflows/generate",
        "/api/v1/workflows/auto-layout",
        "/api/v1/workflows/step-types",
        "/api/v1/workflows/from-pattern",
        "/api/v1/workflows/validate",
        "/api/v1/workflows/*/replay",
    ]

    def can_handle(self, path: str) -> bool:
        return (
            path == "/api/v1/workflows/generate"
            or path == "/api/v1/workflows/auto-layout"
            or path == "/api/v1/workflows/step-types"
            or path == "/api/v1/workflows/from-pattern"
            or path == "/api/v1/workflows/validate"
            or (path.startswith("/api/v1/workflows/") and path.endswith("/replay"))
        )

    # -----------------------------------------------------------------
    # GET
    # -----------------------------------------------------------------

    @track_handler("workflows/builder", method="GET")
    @require_permission("workflows:read")
    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle GET requests for builder endpoints."""
        if path == "/api/v1/workflows/step-types":
            return self._get_step_catalog(query_params)
        return None

    # -----------------------------------------------------------------
    # POST
    # -----------------------------------------------------------------

    @handle_errors("workflow generation")
    @track_handler("workflows/builder", method="POST")
    @require_permission("workflows:read")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests for builder endpoints."""
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if path == "/api/v1/workflows/generate":
            return self._generate_workflow(body)
        if path == "/api/v1/workflows/auto-layout":
            return self._auto_layout(body)
        if path == "/api/v1/workflows/from-pattern":
            return self._create_from_pattern(body)
        if path == "/api/v1/workflows/validate":
            return self._validate_workflow(body)
        if path.endswith("/replay"):
            return self._replay_workflow(path, body)

        return None

    # -----------------------------------------------------------------
    # GET /api/v1/workflows/step-types
    # -----------------------------------------------------------------

    def _get_step_catalog(self, query_params: dict[str, Any]) -> HandlerResult:
        """Return the full step type catalog for the visual builder palette."""
        try:
            from aragora.workflow.step_catalog import get_step_catalog, list_step_categories

            catalog = get_step_catalog()
            category_filter = query_params.get("category")

            items = [info.to_dict() for info in catalog.values()]
            if category_filter:
                items = [i for i in items if i["category"] == category_filter]

            return json_response({
                "step_types": items,
                "categories": list_step_categories(),
                "count": len(items),
            })
        except ImportError:
            logger.warning("step_catalog module not available")
            return error_response("Step catalog not available", 503)

    # -----------------------------------------------------------------
    # POST /api/v1/workflows/generate
    # -----------------------------------------------------------------

    def _generate_workflow(self, body: dict[str, Any]) -> HandlerResult:
        """Generate a workflow from a natural language description."""
        description = body.get("description", "")
        if not description:
            return error_response("'description' is required", 400)

        try:
            from aragora.workflow.nl_builder import NLWorkflowBuilder, NLBuildConfig

            mode = body.get("mode", "quick")
            config = NLBuildConfig(mode=mode)
            builder = NLWorkflowBuilder(config)

            if mode == "quick":
                result = builder.build_quick(
                    description,
                    category=body.get("category"),
                )
            else:
                result = _run_async(builder.build(
                    description,
                    category=body.get("category"),
                    agents=body.get("agents"),
                ))

            return json_response(result.to_dict())
        except ImportError:
            logger.warning("nl_builder module not available")
            return error_response("NL builder not available", 503)
        except (TypeError, ValueError) as exc:
            logger.warning("Workflow generation failed: %s", exc)
            return error_response("Generation failed", 400)

    # -----------------------------------------------------------------
    # POST /api/v1/workflows/auto-layout
    # -----------------------------------------------------------------

    def _auto_layout(self, body: dict[str, Any]) -> HandlerResult:
        """Compute auto-layout positions for workflow steps."""
        steps = body.get("steps", [])
        transitions = body.get("transitions", [])
        layout_type = body.get("layout", "flow")

        if not steps:
            return error_response("'steps' list is required", 400)

        try:
            from aragora.workflow.layout import flow_layout, grid_layout

            if layout_type == "grid":
                columns = body.get("columns", 3)
                positions = grid_layout(steps, columns=columns)
            else:
                positions = flow_layout(steps, transitions)

            return json_response({
                "positions": [p.to_dict() for p in positions],
                "layout": layout_type,
                "count": len(positions),
            })
        except ImportError:
            logger.warning("layout module not available")
            return error_response("Layout module not available", 503)
        except (KeyError, TypeError) as exc:
            logger.warning("Auto-layout failed: %s", exc)
            return error_response("Layout computation failed", 400)

    # -----------------------------------------------------------------
    # POST /api/v1/workflows/from-pattern
    # -----------------------------------------------------------------

    def _create_from_pattern(self, body: dict[str, Any]) -> HandlerResult:
        """Create a workflow from a named pattern."""
        pattern_name = body.get("pattern", "")
        if not pattern_name:
            return error_response("'pattern' is required", 400)

        try:
            from aragora.workflow.patterns.base import PatternType
            from aragora.workflow.patterns import create_pattern

            pattern_type = PatternType(pattern_name)
            kwargs: dict[str, Any] = {}
            if body.get("name"):
                kwargs["name"] = body["name"]
            if body.get("agents"):
                kwargs["agents"] = body["agents"]
            if body.get("task"):
                kwargs["task"] = body["task"]

            pattern = create_pattern(pattern_type, **kwargs)
            workflow = pattern.create(**kwargs)
            return json_response(workflow.to_dict(), status=201)
        except ValueError:
            return error_response(f"Unknown pattern: {pattern_name}", 400)
        except ImportError:
            logger.warning("patterns module not available")
            return error_response("Patterns module not available", 503)
        except (TypeError, AttributeError) as exc:
            logger.warning("Pattern creation failed: %s", exc)
            return error_response("Pattern creation failed", 400)

    # -----------------------------------------------------------------
    # POST /api/v1/workflows/validate
    # -----------------------------------------------------------------

    def _validate_workflow(self, body: dict[str, Any]) -> HandlerResult:
        """Validate a workflow definition."""
        try:
            from aragora.workflow.types import WorkflowDefinition
            from aragora.workflow.validation import validate_workflow

            definition = WorkflowDefinition.from_dict(body)
            result = validate_workflow(definition)
            return json_response(result.to_dict())
        except ImportError:
            logger.warning("validation module not available")
            return error_response("Validation module not available", 503)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Workflow validation failed: %s", exc)
            return error_response("Invalid workflow definition", 400)

    # -----------------------------------------------------------------
    # POST /api/v1/workflows/{id}/replay
    # -----------------------------------------------------------------

    def _replay_workflow(self, path: str, body: dict[str, Any]) -> HandlerResult:
        """Replay a workflow execution with same or new inputs."""
        # Extract workflow_id: /api/v1/workflows/{id}/replay
        parts = path.strip("/").split("/")
        # Expected: api, v1, workflows, {id}, replay
        if len(parts) < 5 or parts[4] != "replay":
            return error_response("Invalid replay path", 400)
        workflow_id = parts[3]

        try:
            from .execution import execute_workflow
            from .crud import get_workflow

            # Verify workflow exists
            workflow = _run_async(get_workflow(workflow_id))
            if not workflow:
                return error_response(f"Workflow not found: {workflow_id}", 404)

            inputs = body.get("inputs", {})
            result = _run_async(execute_workflow(workflow_id, inputs=inputs))
            return json_response(result)
        except ValueError as exc:
            logger.warning("Replay failed: %s", exc)
            return error_response("Replay failed", 404)
        except (ConnectionError, TimeoutError) as exc:
            logger.error("Connection error during replay: %s", exc)
            return error_response("Execution service unavailable", 503)
        except (KeyError, TypeError, AttributeError) as exc:
            logger.error("Data error during replay: %s", exc)
            return error_response("Internal data error", 500)


__all__ = ["WorkflowBuilderHandler"]
