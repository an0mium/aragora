"""
Debate intervention endpoint handlers.

Provides HTTP endpoints for human intervention in live debates:
- POST /api/v1/debates/{id}/pause      -- Pause a running debate
- POST /api/v1/debates/{id}/resume     -- Resume a paused debate
- POST /api/v1/debates/{id}/nudge      -- Send a nudge/hint to agents
- POST /api/v1/debates/{id}/challenge  -- Inject a counterargument
- POST /api/v1/debates/{id}/inject-evidence -- Inject evidence
- GET  /api/v1/debates/{id}/intervention-log -- Retrieve intervention log

Follows existing debate handler patterns (BaseHandler mixin, @handle_errors,
strip_version_prefix, RBAC decorators).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from aragora.rbac.decorators import require_permission
from aragora.server.validation import validate_debate_id

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from ..openapi_decorator import api_endpoint

logger = logging.getLogger(__name__)

# Route patterns managed by DebateInterventionsHandler
INTERVENTION_ROUTES = [
    "/api/v1/debates/*/pause",
    "/api/v1/debates/*/resume",
    "/api/v1/debates/*/nudge",
    "/api/v1/debates/*/challenge",
    "/api/v1/debates/*/inject-evidence",
    "/api/v1/debates/*/intervention-log",
]


def _strip_version_prefix(path: str) -> str:
    """Normalize versioned paths to unversioned for consistent matching."""
    return path.replace("/api/v1/", "/api/").replace("/api/v2/", "/api/")


def _extract_debate_id_from_path(path: str) -> tuple[str | None, str | None]:
    """Extract and validate debate ID from intervention path.

    Handles paths like /api/v1/debates/{id}/pause or /api/debates/{id}/pause.

    Returns:
        (debate_id, error_message) -- error_message is set if invalid.
    """
    normalized = _strip_version_prefix(path)
    # ['', 'api', 'debates', '{id}', 'pause']
    parts = normalized.split("/")
    if len(parts) < 5:
        return None, "Invalid path"

    debate_id = parts[3]
    is_valid, err = validate_debate_id(debate_id)
    if not is_valid:
        return None, err

    return debate_id, None


class _HandlerProtocol(Protocol):
    """Protocol for type-checking the mixin composition."""

    ctx: dict[str, Any]

    def read_json_body(
        self, handler: Any, max_size: int | None = None
    ) -> dict[str, Any] | None: ...

    def get_current_user(self, handler: Any) -> Any: ...


class DebateInterventionsHandler(BaseHandler):
    """Handler for debate intervention endpoints.

    Manages pause/resume, nudges, challenges, and evidence injection
    for live debates.
    """

    ROUTES = INTERVENTION_ROUTES

    def __init__(self, ctx: dict | None = None, server_context: dict | None = None):
        if server_context is not None:
            self.ctx = server_context
        else:
            self.ctx = ctx or {}

    # ------------------------------------------------------------------
    # Route matching
    # ------------------------------------------------------------------

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        normalized = _strip_version_prefix(path)
        suffixes = (
            "/pause",
            "/resume",
            "/nudge",
            "/challenge",
            "/inject-evidence",
            "/intervention-log",
        )
        if normalized.startswith("/api/debates/"):
            return any(normalized.endswith(s) for s in suffixes)
        return False

    # ------------------------------------------------------------------
    # GET handler
    # ------------------------------------------------------------------

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route GET requests."""
        normalized = _strip_version_prefix(path)
        if normalized.endswith("/intervention-log"):
            return self._get_intervention_log(path, handler)
        return None

    # ------------------------------------------------------------------
    # POST handler
    # ------------------------------------------------------------------

    @handle_errors("debate interventions")
    @require_permission("debates:update")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route POST requests to intervention methods."""
        normalized = _strip_version_prefix(path)

        if normalized.endswith("/pause"):
            return self._pause_debate(path, handler)
        if normalized.endswith("/resume"):
            return self._resume_debate(path, handler)
        if normalized.endswith("/nudge"):
            return self._nudge_debate(path, handler)
        if normalized.endswith("/challenge"):
            return self._challenge_debate(path, handler)
        if normalized.endswith("/inject-evidence"):
            return self._inject_evidence(path, handler)

        return None

    # ------------------------------------------------------------------
    # Endpoint implementations
    # ------------------------------------------------------------------

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{id}/pause",
        summary="Pause a running debate",
        description="Pause a live debate. The debate must be in a running state.",
        tags=["Debate Interventions"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {"description": "Debate paused successfully"},
            "400": {"description": "Debate cannot be paused (wrong state)"},
            "404": {"description": "Debate not found"},
        },
    )
    @handle_errors("pause debate")
    def _pause_debate(self, path: str, handler: Any) -> HandlerResult:
        debate_id, err = _extract_debate_id_from_path(path)
        if err:
            return error_response(err, 400)

        manager = self._get_or_create_manager(debate_id, handler)
        if manager is None:
            return error_response(f"Debate not found: {debate_id}", 404)

        user_id = self._extract_user_id(handler)

        try:
            entry = manager.pause(user_id=user_id)
        except ValueError as exc:
            return error_response(str(exc), 400)

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "state": "paused",
                "intervention": entry.to_dict(),
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{id}/resume",
        summary="Resume a paused debate",
        description="Resume a debate that was previously paused.",
        tags=["Debate Interventions"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {"description": "Debate resumed successfully"},
            "400": {"description": "Debate cannot be resumed (wrong state)"},
            "404": {"description": "Debate not found"},
        },
    )
    @handle_errors("resume debate")
    def _resume_debate(self, path: str, handler: Any) -> HandlerResult:
        debate_id, err = _extract_debate_id_from_path(path)
        if err:
            return error_response(err, 400)

        manager = self._get_or_create_manager(debate_id, handler)
        if manager is None:
            return error_response(f"Debate not found: {debate_id}", 404)

        user_id = self._extract_user_id(handler)

        try:
            entry = manager.resume(user_id=user_id)
        except ValueError as exc:
            return error_response(str(exc), 400)

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "state": "running",
                "intervention": entry.to_dict(),
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{id}/nudge",
        summary="Send a nudge to the debate",
        description="Send a nudge/hint message to steer the debate. Optionally target a specific agent.",
        tags=["Debate Interventions"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {"description": "Nudge delivered successfully"},
            "400": {"description": "Invalid request or debate completed"},
            "404": {"description": "Debate not found"},
        },
    )
    @handle_errors("nudge debate")
    def _nudge_debate(self, path: str, handler: Any) -> HandlerResult:
        debate_id, err = _extract_debate_id_from_path(path)
        if err:
            return error_response(err, 400)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        message = (body.get("message") or "").strip()
        if not message:
            return error_response("Missing required field: message", 400)

        target_agent = body.get("target_agent")
        user_id = self._extract_user_id(handler)

        manager = self._get_or_create_manager(debate_id, handler)
        if manager is None:
            return error_response(f"Debate not found: {debate_id}", 404)

        try:
            entry = manager.nudge(
                message=message,
                user_id=user_id,
                target_agent=target_agent,
            )
        except ValueError as exc:
            return error_response(str(exc), 400)

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "intervention": entry.to_dict(),
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{id}/challenge",
        summary="Inject a challenge into the debate",
        description="Inject a counterargument or challenge that agents must address.",
        tags=["Debate Interventions"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {"description": "Challenge injected successfully"},
            "400": {"description": "Invalid request or debate completed"},
            "404": {"description": "Debate not found"},
        },
    )
    @handle_errors("challenge debate")
    def _challenge_debate(self, path: str, handler: Any) -> HandlerResult:
        debate_id, err = _extract_debate_id_from_path(path)
        if err:
            return error_response(err, 400)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        challenge_text = (body.get("challenge") or "").strip()
        if not challenge_text:
            return error_response("Missing required field: challenge", 400)

        user_id = self._extract_user_id(handler)

        manager = self._get_or_create_manager(debate_id, handler)
        if manager is None:
            return error_response(f"Debate not found: {debate_id}", 404)

        try:
            entry = manager.challenge(
                challenge_text=challenge_text,
                user_id=user_id,
            )
        except ValueError as exc:
            return error_response(str(exc), 400)

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "intervention": entry.to_dict(),
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/debates/{id}/inject-evidence",
        summary="Inject evidence into the debate",
        description="Inject external evidence with optional source citation into the debate.",
        tags=["Debate Interventions"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {"description": "Evidence injected successfully"},
            "400": {"description": "Invalid request or debate completed"},
            "404": {"description": "Debate not found"},
        },
    )
    @handle_errors("inject evidence")
    def _inject_evidence(self, path: str, handler: Any) -> HandlerResult:
        debate_id, err = _extract_debate_id_from_path(path)
        if err:
            return error_response(err, 400)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        evidence = (body.get("evidence") or "").strip()
        if not evidence:
            return error_response("Missing required field: evidence", 400)

        source = body.get("source")
        user_id = self._extract_user_id(handler)

        manager = self._get_or_create_manager(debate_id, handler)
        if manager is None:
            return error_response(f"Debate not found: {debate_id}", 404)

        try:
            entry = manager.inject_evidence(
                evidence=evidence,
                source=source,
                user_id=user_id,
            )
        except ValueError as exc:
            return error_response(str(exc), 400)

        return json_response(
            {
                "success": True,
                "debate_id": debate_id,
                "intervention": entry.to_dict(),
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{id}/intervention-log",
        summary="Get intervention history",
        description="Retrieve the full intervention log for a debate.",
        tags=["Debate Interventions"],
        parameters=[{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
        responses={
            "200": {"description": "Intervention log retrieved"},
            "400": {"description": "Invalid debate ID"},
        },
    )
    @handle_errors("get intervention log")
    def _get_intervention_log(self, path: str, handler: Any) -> HandlerResult:
        debate_id, err = _extract_debate_id_from_path(path)
        if err:
            return error_response(err, 400)

        from aragora.debate.intervention import get_intervention_manager

        manager = get_intervention_manager(debate_id, create=False)
        if manager is None:
            # No interventions yet -- return empty log
            return json_response(
                {
                    "debate_id": debate_id,
                    "state": "unknown",
                    "entry_count": 0,
                    "entries": [],
                }
            )

        log = manager.get_log()
        result = log.to_dict()
        result["state"] = manager.state.value
        return json_response(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_manager(self, debate_id: str, handler: Any) -> Any:
        """Get or create an InterventionManager for the given debate.

        Returns None if the debate does not exist in state or storage.
        """
        from aragora.debate.intervention import get_intervention_manager

        # Try to get the stream emitter from the handler for WebSocket events
        emitter = getattr(handler, "stream_emitter", None)
        return get_intervention_manager(debate_id, emitter=emitter, create=True)

    def _extract_user_id(self, handler: Any) -> str | None:
        """Extract user ID from the request handler, if available."""
        try:
            user = self.get_current_user(handler)
            if user:
                return getattr(user, "user_id", None)
        except (AttributeError, TypeError, ValueError):
            pass
        return None


__all__ = [
    "DebateInterventionsHandler",
    "INTERVENTION_ROUTES",
]
