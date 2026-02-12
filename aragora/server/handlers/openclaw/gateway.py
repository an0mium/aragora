"""
HTTP Handlers for OpenClaw Gateway.

Stability: STABLE

Provides REST API endpoints for the OpenClaw gateway integration:
- Session management (create, get, list, close)
- Action execution (execute, status, cancel)
- Credential management (store, list, delete, rotate)
- Admin operations (health, metrics, audit)

Endpoints:
    Session Management:
    - POST   /api/gateway/openclaw/sessions           - Create session
    - GET    /api/gateway/openclaw/sessions/:id       - Get session
    - DELETE /api/gateway/openclaw/sessions/:id       - Close session
    - GET    /api/gateway/openclaw/sessions           - List sessions

    Action Management:
    - POST   /api/gateway/openclaw/actions            - Execute action
    - GET    /api/gateway/openclaw/actions/:id        - Get action status
    - POST   /api/gateway/openclaw/actions/:id/cancel - Cancel action

    Credential Management:
    - POST   /api/gateway/openclaw/credentials            - Store credential
    - GET    /api/gateway/openclaw/credentials            - List credentials (no values)
    - DELETE /api/gateway/openclaw/credentials/:id        - Delete credential
    - POST   /api/gateway/openclaw/credentials/:id/rotate - Rotate credential

    Admin Endpoints:
    - GET    /api/gateway/openclaw/health   - Gateway health
    - GET    /api/gateway/openclaw/metrics  - Gateway metrics
    - GET    /api/gateway/openclaw/audit    - Audit log
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.observability.metrics import track_handler
from aragora.resilience import CircuitBreaker
from aragora.server.handlers.base import BaseHandler, HandlerResult
from aragora.server.handlers.utils.decorators import require_permission
from aragora.server.handlers.openclaw.credentials import CredentialHandlerMixin
from aragora.server.handlers.openclaw.orchestrator import SessionOrchestrationMixin
from aragora.server.handlers.openclaw.policies import PolicyHandlerMixin

logger = logging.getLogger(__name__)


# =============================================================================
# Resilience Configuration
# =============================================================================

# Circuit breaker for OpenClaw gateway service
_openclaw_circuit_breaker = CircuitBreaker(
    name="openclaw_gateway_handler",
    failure_threshold=5,
    cooldown_seconds=30.0,
)


def get_openclaw_circuit_breaker() -> CircuitBreaker:
    """Get the circuit breaker for OpenClaw gateway service."""
    return _openclaw_circuit_breaker


def get_openclaw_circuit_breaker_status() -> dict[str, Any]:
    """Get current status of the OpenClaw gateway circuit breaker."""
    return _openclaw_circuit_breaker.to_dict()


# =============================================================================
# Handler Implementation
# =============================================================================


class OpenClawGatewayHandler(
    SessionOrchestrationMixin,
    CredentialHandlerMixin,
    PolicyHandlerMixin,
    BaseHandler,
):
    """
    HTTP handler for OpenClaw gateway operations.

    Stability: STABLE

    Features:
    - Circuit breaker pattern for service resilience
    - Rate limiting (30-120 requests/minute depending on endpoint)
    - RBAC permission checks (gateway:sessions.*, gateway:actions.*, etc.)
    - Comprehensive input validation and audit logging

    Provides REST API access to OpenClaw gateway for:
    - Session management
    - Action execution
    - Credential management
    - Admin operations
    """

    def __init__(self, server_context: dict[str, Any]) -> None:
        """Initialize with server context."""
        super().__init__(server_context)

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return (
            path.startswith("/api/gateway/openclaw/")
            or path.startswith("/api/v1/gateway/openclaw/")
            or path.startswith("/api/v1/openclaw/")
        )

    def _normalize_path(self, path: str) -> str:
        """Normalize versioned paths to base form."""
        if path.startswith("/api/v1/gateway/openclaw/"):
            return path.replace("/api/v1/gateway/openclaw/", "/api/gateway/openclaw/", 1)
        if path.startswith("/api/v1/openclaw/"):
            return path.replace("/api/v1/openclaw/", "/api/gateway/openclaw/", 1)
        return path

    def _get_user_id(self, handler: Any) -> str:
        """Extract user ID from request handler."""
        user = self.get_current_user(handler)
        if user:
            return user.user_id
        return "anonymous"

    def _get_tenant_id(self, handler: Any) -> str | None:
        """Extract tenant ID from request handler."""
        user = self.get_current_user(handler)
        if user and hasattr(user, "org_id"):
            return user.org_id
        return None

    # =========================================================================
    # GET Handlers
    # =========================================================================

    @track_handler("gateway/openclaw", method="GET")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Handle GET requests."""
        path = self._normalize_path(path)

        # GET /api/gateway/openclaw/sessions
        if path == "/api/gateway/openclaw/sessions":
            return self._handle_list_sessions(query_params, handler)

        # GET /api/gateway/openclaw/sessions/:id
        if path.startswith("/api/gateway/openclaw/sessions/") and path.count("/") == 5:
            session_id = path.split("/")[-1]
            return self._handle_get_session(session_id, handler)

        # GET /api/gateway/openclaw/actions/:id
        if path.startswith("/api/gateway/openclaw/actions/") and path.count("/") == 5:
            action_id = path.split("/")[-1]
            return self._handle_get_action(action_id, handler)

        # GET /api/gateway/openclaw/credentials
        if path == "/api/gateway/openclaw/credentials":
            return self._handle_list_credentials(query_params, handler)

        # GET /api/gateway/openclaw/health
        if path == "/api/gateway/openclaw/health":
            return self._handle_health(handler)

        # GET /api/gateway/openclaw/metrics
        if path == "/api/gateway/openclaw/metrics":
            return self._handle_metrics(handler)

        # GET /api/gateway/openclaw/audit
        if path == "/api/gateway/openclaw/audit":
            return self._handle_audit(query_params, handler)

        # GET /api/gateway/openclaw/policy/rules
        if path == "/api/gateway/openclaw/policy/rules":
            return self._handle_get_policy_rules(query_params, handler)

        # GET /api/gateway/openclaw/approvals
        if path == "/api/gateway/openclaw/approvals":
            return self._handle_list_approvals(query_params, handler)

        # GET /api/gateway/openclaw/stats
        if path == "/api/gateway/openclaw/stats":
            return self._handle_stats(handler)

        return None

    # =========================================================================
    # POST Handlers
    # =========================================================================

    @track_handler("gateway/openclaw", method="POST")
    @require_permission("debates:write")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        path = self._normalize_path(path)

        # POST /api/gateway/openclaw/sessions
        if path == "/api/gateway/openclaw/sessions":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_create_session(body, handler)

        # POST /api/gateway/openclaw/actions
        if path == "/api/gateway/openclaw/actions":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_execute_action(body, handler)

        # POST /api/gateway/openclaw/actions/:id/cancel
        if path.endswith("/cancel") and "/actions/" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                action_id = parts[-2]
                return self._handle_cancel_action(action_id, handler)

        # POST /api/gateway/openclaw/sessions/:id/end
        if path.endswith("/end") and "/sessions/" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                session_id = parts[-2]
                return self._handle_end_session(session_id, handler)

        # POST /api/gateway/openclaw/policy/rules
        if path == "/api/gateway/openclaw/policy/rules":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_add_policy_rule(body, handler)

        # POST /api/gateway/openclaw/approvals/:id/approve
        if path.endswith("/approve") and "/approvals/" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                approval_id = parts[-2]
                body, err = self.read_json_body_validated(handler)
                if err:
                    return err
                return self._handle_approve_action(approval_id, body, handler)

        # POST /api/gateway/openclaw/approvals/:id/deny
        if path.endswith("/deny") and "/approvals/" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                approval_id = parts[-2]
                body, err = self.read_json_body_validated(handler)
                if err:
                    return err
                return self._handle_deny_action(approval_id, body, handler)

        # POST /api/gateway/openclaw/credentials
        if path == "/api/gateway/openclaw/credentials":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_store_credential(body, handler)

        # POST /api/gateway/openclaw/credentials/:id/rotate
        if path.endswith("/rotate") and "/credentials/" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                credential_id = parts[-2]
                body, err = self.read_json_body_validated(handler)
                if err:
                    return err
                return self._handle_rotate_credential(credential_id, body, handler)

        return None

    # =========================================================================
    # DELETE Handlers
    # =========================================================================

    @track_handler("gateway/openclaw", method="DELETE")
    @require_permission("debates:delete")
    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle DELETE requests."""
        path = self._normalize_path(path)

        # DELETE /api/gateway/openclaw/sessions/:id
        if path.startswith("/api/gateway/openclaw/sessions/") and path.count("/") == 5:
            session_id = path.split("/")[-1]
            return self._handle_close_session(session_id, handler)

        # DELETE /api/gateway/openclaw/policy/rules/:name
        if path.startswith("/api/gateway/openclaw/policy/rules/"):
            rule_name = path.split("/")[-1]
            return self._handle_remove_policy_rule(rule_name, handler)

        # DELETE /api/gateway/openclaw/credentials/:id
        if path.startswith("/api/gateway/openclaw/credentials/") and path.count("/") == 5:
            credential_id = path.split("/")[-1]
            return self._handle_delete_credential(credential_id, handler)

        return None


# =============================================================================
# Handler Registration
# =============================================================================


def get_openclaw_gateway_handler(
    server_context: dict[str, Any],
) -> OpenClawGatewayHandler:
    """Get an instance of the OpenClaw gateway handler."""
    return OpenClawGatewayHandler(server_context)


__all__ = [
    "OpenClawGatewayHandler",
    "get_openclaw_gateway_handler",
    # Resilience
    "get_openclaw_circuit_breaker",
    "get_openclaw_circuit_breaker_status",
]
