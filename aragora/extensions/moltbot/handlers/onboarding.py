"""
Moltbot Onboarding Handler - Onboarding Flow Management REST API.

Endpoints:
- GET  /api/v1/moltbot/flows                  - List flows
- POST /api/v1/moltbot/flows                  - Create flow
- GET  /api/v1/moltbot/flows/{id}             - Get flow
- PUT  /api/v1/moltbot/flows/{id}             - Update flow
- DELETE /api/v1/moltbot/flows/{id}           - Delete flow
- POST /api/v1/moltbot/flows/{id}/steps       - Add step
- GET  /api/v1/moltbot/flows/{id}/sessions    - List sessions
- POST /api/v1/moltbot/flows/{id}/sessions    - Start session
- GET  /api/v1/moltbot/sessions/{id}          - Get session
- POST /api/v1/moltbot/sessions/{id}/advance  - Advance session
- POST /api/v1/moltbot/sessions/{id}/complete - Complete session
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)

from .types import serialize_datetime, serialize_enum

if TYPE_CHECKING:
    from aragora.extensions.moltbot.onboarding import OnboardingOrchestrator

logger = logging.getLogger(__name__)

# Global orchestrator instance
_orchestrator: Optional["OnboardingOrchestrator"] = None

def get_orchestrator() -> "OnboardingOrchestrator":
    """Get or create the onboarding orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        from aragora.extensions.moltbot.onboarding import OnboardingOrchestrator

        _orchestrator = OnboardingOrchestrator()
    return _orchestrator

class MoltbotOnboardingHandler(BaseHandler):
    """HTTP handler for Moltbot onboarding flows."""

    routes = [
        ("GET", "/api/v1/moltbot/flows"),
        ("POST", "/api/v1/moltbot/flows"),
        ("GET", "/api/v1/moltbot/flows/"),
        ("PUT", "/api/v1/moltbot/flows/"),
        ("DELETE", "/api/v1/moltbot/flows/"),
        ("POST", "/api/v1/moltbot/flows/*/steps"),
        ("GET", "/api/v1/moltbot/flows/*/sessions"),
        ("POST", "/api/v1/moltbot/flows/*/sessions"),
        ("GET", "/api/v1/moltbot/sessions/"),
        ("POST", "/api/v1/moltbot/sessions/*/advance"),
        ("POST", "/api/v1/moltbot/sessions/*/complete"),
        ("POST", "/api/v1/moltbot/sessions/*/skip"),
    ]

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle GET requests."""
        if path == "/api/v1/moltbot/flows":
            return await self._handle_list_flows(query_params, handler)
        elif path.startswith("/api/v1/moltbot/flows/"):
            parts = path.split("/")
            if len(parts) >= 5:
                flow_id = parts[4]

                # Sessions list
                if len(parts) > 5 and parts[5] == "sessions":
                    return await self._handle_list_sessions(flow_id, query_params, handler)

                # Get single flow
                return await self._handle_get_flow(flow_id, handler)

        elif path.startswith("/api/v1/moltbot/sessions/"):
            parts = path.split("/")
            if len(parts) >= 5:
                session_id = parts[4]
                return await self._handle_get_session(session_id, handler)
        return None

    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        if path == "/api/v1/moltbot/flows":
            return await self._handle_create_flow(handler)
        elif path.startswith("/api/v1/moltbot/flows/"):
            parts = path.split("/")
            if len(parts) >= 5:
                flow_id = parts[4]

                # Add step
                if len(parts) > 5 and parts[5] == "steps":
                    return await self._handle_add_step(flow_id, handler)

                # Start session
                if len(parts) > 5 and parts[5] == "sessions":
                    return await self._handle_start_session(flow_id, handler)

        elif path.startswith("/api/v1/moltbot/sessions/"):
            parts = path.split("/")
            if len(parts) >= 6:
                session_id = parts[4]
                action = parts[5]

                if action == "advance":
                    return await self._handle_advance_session(session_id, handler)
                elif action == "complete":
                    return await self._handle_complete_session(session_id, handler)
                elif action == "skip":
                    return await self._handle_skip_step(session_id, handler)
        return None

    async def handle_put(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle PUT requests."""
        if path.startswith("/api/v1/moltbot/flows/"):
            parts = path.split("/")
            if len(parts) >= 5:
                flow_id = parts[4]
                return await self._handle_update_flow(flow_id, handler)
        return None

    async def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle DELETE requests."""
        if path.startswith("/api/v1/moltbot/flows/"):
            parts = path.split("/")
            if len(parts) >= 5:
                flow_id = parts[4]
                return await self._handle_delete_flow(flow_id, handler)
        return None

    # ========== Handler Methods ==========

    def _serialize_flow(self, flow: Any) -> dict[str, Any]:
        """Serialize flow to JSON-safe dict."""
        return {
            "id": flow.id,
            "name": flow.name,
            "description": flow.description,
            "is_active": flow.is_active,
            "step_count": len(flow.steps) if hasattr(flow, "steps") else 0,
            "created_at": serialize_datetime(flow.created_at),
            "updated_at": serialize_datetime(flow.updated_at),
        }

    def _serialize_step(self, step: Any) -> dict[str, Any]:
        """Serialize step to JSON-safe dict."""
        return {
            "id": step.id,
            "type": serialize_enum(step.step_type),
            "title": step.title,
            "content": step.content,
            "order": step.order,
            "is_required": step.is_required,
            "timeout_seconds": step.timeout_seconds,
            "metadata": step.metadata,
        }

    def _serialize_session(self, session: Any) -> dict[str, Any]:
        """Serialize session to JSON-safe dict."""
        return {
            "id": session.id,
            "flow_id": session.flow_id,
            "user_id": session.user_id,
            "device_id": session.device_id,
            "status": serialize_enum(session.status),
            "current_step_index": session.current_step_index,
            "completed_steps": session.completed_steps,
            "skipped_steps": session.skipped_steps,
            "responses": session.responses,
            "started_at": serialize_datetime(session.started_at),
            "completed_at": serialize_datetime(session.completed_at),
        }

    async def _handle_list_flows(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """List all onboarding flows."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        status_filter = query_params.get("status")
        target_segment = query_params.get("target_segment")

        orchestrator = get_orchestrator()
        flows = await orchestrator.list_flows(
            status=status_filter,
            target_segment=target_segment,
        )

        return json_response(
            {
                "flows": [self._serialize_flow(f) for f in flows],
                "total": len(flows),
            }
        )

    async def _handle_create_flow(self, handler: Any) -> HandlerResult:
        """Create a new onboarding flow."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        name = body.get("name")
        if not name:
            return error_response("name is required", 400)

        orchestrator = get_orchestrator()
        flow = await orchestrator.create_flow(
            name=name,
            description=body.get("description", ""),
            target_segment=body.get("target_segment"),
        )

        return json_response(
            {"success": True, "flow": self._serialize_flow(flow)},
            status=201,
        )

    async def _handle_get_flow(self, flow_id: str, handler: Any) -> HandlerResult:
        """Get flow details with steps."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        orchestrator = get_orchestrator()
        flow = await orchestrator.get_flow(flow_id)

        if not flow:
            return error_response("Flow not found", 404)

        result = self._serialize_flow(flow)
        result["steps"] = [self._serialize_step(s) for s in flow.steps]

        return json_response({"flow": result})

    async def _handle_update_flow(self, flow_id: str, handler: Any) -> HandlerResult:
        """Update an onboarding flow."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        orchestrator = get_orchestrator()
        flow = await orchestrator.get_flow(flow_id)

        if not flow:
            return error_response("Flow not found", 404)

        # Update flow attributes directly
        if "name" in body:
            flow.name = body["name"]
        if "description" in body:
            flow.description = body["description"]
        if "status" in body:
            flow.status = body["status"]

        return json_response({"success": True, "flow": self._serialize_flow(flow)})

    async def _handle_delete_flow(self, flow_id: str, handler: Any) -> HandlerResult:
        """Delete an onboarding flow."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        orchestrator = get_orchestrator()
        # Archive flow instead of deleting (soft delete)
        flow = await orchestrator.archive_flow(flow_id)

        if not flow:
            return error_response("Flow not found", 404)

        return json_response({"success": True, "archived": flow_id})

    async def _handle_add_step(self, flow_id: str, handler: Any) -> HandlerResult:
        """Add step to flow."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        step_type = body.get("type")
        name = body.get("name") or body.get("title")  # Support both name and title

        if not step_type or not name:
            return error_response("type and name are required", 400)

        orchestrator = get_orchestrator()
        step = await orchestrator.add_step(
            flow_id=flow_id,
            name=name,
            step_type=step_type,
            content=body.get("content", {}),
            required=body.get("required", body.get("is_required", True)),
            validation=body.get("validation"),
            next_step=body.get("next_step"),
            branch_conditions=body.get("branch_conditions"),
        )

        if not step:
            return error_response("Flow not found", 404)

        return json_response(
            {"success": True, "step": self._serialize_step(step)},
            status=201,
        )

    async def _handle_list_sessions(
        self, flow_id: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """List sessions for a flow."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        status_filter = query_params.get("status")

        orchestrator = get_orchestrator()
        sessions = await orchestrator.list_sessions(
            flow_id=flow_id,
            status=status_filter,
        )

        return json_response(
            {
                "sessions": [self._serialize_session(s) for s in sessions],
                "total": len(sessions),
            }
        )

    async def _handle_start_session(self, flow_id: str, handler: Any) -> HandlerResult:
        """Start a new onboarding session."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            body = {}

        orchestrator = get_orchestrator()
        try:
            session = await orchestrator.start_session(
                flow_id=flow_id,
                user_id=body.get("user_id", user.user_id),
                channel_id=body.get("channel_id", "web"),
                tenant_id=body.get("tenant_id"),
                initial_data=body.get("initial_data"),
            )
        except ValueError as e:
            return error_response(str(e), 404)

        return json_response(
            {"success": True, "session": self._serialize_session(session)},
            status=201,
        )

    async def _handle_get_session(self, session_id: str, handler: Any) -> HandlerResult:
        """Get session details."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        orchestrator = get_orchestrator()
        session = await orchestrator.get_session(session_id)

        if not session:
            return error_response("Session not found", 404)

        result = self._serialize_session(session)

        # Include current step info if available
        if hasattr(session, "current_step") and session.current_step:
            result["current_step"] = self._serialize_step(session.current_step)

        return json_response({"session": result})

    async def _handle_advance_session(self, session_id: str, handler: Any) -> HandlerResult:
        """Advance session to next step."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        data = body.get("data", body.get("response", {})) if body else {}

        orchestrator = get_orchestrator()
        result = await orchestrator.submit_step(
            session_id=session_id,
            data=data,
        )

        if not result.get("success"):
            error_msg = result.get("error", "Session not found or already completed")
            return error_response(error_msg, 400 if "validation" in error_msg.lower() else 404)

        # Get updated session for response
        session = await orchestrator.get_session(session_id)
        response_data: dict[str, Any] = {"success": True, "result": result}
        if session:
            response_data["session"] = self._serialize_session(session)

        return json_response(response_data)

    async def _handle_complete_session(self, session_id: str, handler: Any) -> HandlerResult:
        """Complete an onboarding session."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        orchestrator = get_orchestrator()
        session = await orchestrator.get_session(session_id)

        if not session:
            return error_response("Session not found", 404)

        # Mark session as completed by submitting empty data for remaining steps
        # The submit_step method auto-completes when all steps are done
        result = await orchestrator.submit_step(session_id, {})
        if result.get("completed"):
            session = await orchestrator.get_session(session_id)

        return json_response(
            {"success": True, "session": self._serialize_session(session) if session else {}}
        )

    async def _handle_skip_step(self, session_id: str, handler: Any) -> HandlerResult:
        """Skip current step in session."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        orchestrator = get_orchestrator()
        session = await orchestrator.get_session(session_id)

        if not session:
            return error_response("Session not found", 404)

        # Skip by submitting empty data - validation may fail for required fields
        # This effectively allows skipping optional steps
        result = await orchestrator.submit_step(session_id, {"__skip__": True})

        if not result.get("success") and "required" in str(result.get("error", "")).lower():
            return error_response("Step is required and cannot be skipped", 400)

        session = await orchestrator.get_session(session_id)
        return json_response(
            {"success": True, "session": self._serialize_session(session) if session else {}}
        )
