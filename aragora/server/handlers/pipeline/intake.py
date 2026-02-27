"""Pipeline intake handler — accept vague prompts and start pipelines.

Endpoints:
- POST /api/v1/pipeline/start      Start a new pipeline from a prompt
- GET  /api/v1/pipeline/start/:id   Get intake status
"""

from __future__ import annotations

import asyncio
import logging
import uuid
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

_intake_limiter = RateLimiter(requests_per_minute=20)

# Track active intake sessions: pipeline_id -> IntakeResult dict
_active_intakes: dict[str, dict[str, Any]] = {}


class PipelineIntakeHandler(BaseHandler):
    """Handler for pipeline intake — prompt → ideas → pipeline.

    Accepts a free-text prompt, parses it into ideas, optionally runs
    interrogation, and kicks off the pipeline.
    """

    ROUTES = ["/api/v1/pipeline/start"]

    def __init__(self, ctx: dict[str, Any] | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        cleaned = strip_version_prefix(path)
        return cleaned.rstrip("/").startswith("/api/pipeline/start")

    @require_permission("pipeline:read")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """GET /api/v1/pipeline/start/:id — get intake status."""
        cleaned = strip_version_prefix(path)
        parts = cleaned.strip("/").split("/")

        # /api/pipeline/start/:id
        if len(parts) >= 4 and parts[2] == "start":
            pipeline_id = parts[3]
            valid, err = validate_path_segment(pipeline_id, "pipeline_id", SAFE_ID_PATTERN)
            if not valid:
                return error_response(err or "Invalid pipeline ID", 400)

            intake = _active_intakes.get(pipeline_id)
            if not intake:
                return error_response("Intake not found", 404)
            return json_response({"data": intake})

        # /api/pipeline/start — list recent intakes
        recent = sorted(
            _active_intakes.values(),
            key=lambda x: x.get("pipeline_id", ""),
            reverse=True,
        )[:20]
        return json_response({"data": recent})

    @handle_errors("pipeline intake")
    @require_permission("pipeline:write")
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """POST /api/v1/pipeline/start — start a new pipeline from a prompt."""
        client_ip = get_client_ip(handler)
        if not _intake_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        # Parse request body
        body = {}
        if hasattr(handler, "request_body"):
            body = handler.request_body or {}
        elif hasattr(handler, "json_body"):
            body = handler.json_body or {}

        prompt = body.get("prompt", "").strip()
        if not prompt:
            return error_response("Prompt is required", 400)

        if len(prompt) > 50000:
            return error_response("Prompt too long (max 50000 characters)", 400)

        autonomy_level = body.get("autonomy_level", 2)
        if not isinstance(autonomy_level, int) or not 1 <= autonomy_level <= 5:
            return error_response("autonomy_level must be 1-5", 400)

        skip_interrogation = body.get("skip_interrogation", False)
        pipeline_id = body.get("pipeline_id") or str(uuid.uuid4())

        # Import intake engine lazily
        from aragora.pipeline.intake import PipelineIntake, IntakeRequest, AutonomyLevel

        request = IntakeRequest(
            prompt=prompt,
            autonomy_level=AutonomyLevel(autonomy_level),
            pipeline_id=pipeline_id,
            skip_interrogation=bool(skip_interrogation),
        )

        intake = PipelineIntake()

        # For web requests, skip interactive interrogation
        # (interrogation requires a chat interface, handled separately via WebSocket)
        request.skip_interrogation = True

        result = await intake.process(request)
        result_dict = result.to_dict()

        # Store for status queries
        _active_intakes[pipeline_id] = result_dict

        # If ready, optionally auto-execute based on autonomy level
        if result.ready_for_pipeline and autonomy_level >= 3:
            # Fire-and-forget execution for autonomous modes
            async def _run_pipeline() -> None:
                try:
                    pipeline_result = await intake.execute(result, request)
                    _active_intakes[pipeline_id]["pipeline_status"] = "completed"
                    _active_intakes[pipeline_id]["stage_status"] = (
                        pipeline_result.stage_status
                        if hasattr(pipeline_result, "stage_status")
                        else {}
                    )
                except Exception:
                    logger.exception("Pipeline execution failed: %s", pipeline_id)
                    _active_intakes[pipeline_id]["pipeline_status"] = "failed"

            asyncio.create_task(_run_pipeline())
            result_dict["pipeline_status"] = "running"
        else:
            result_dict["pipeline_status"] = "awaiting_approval"

        return json_response({"data": result_dict}, 201)
