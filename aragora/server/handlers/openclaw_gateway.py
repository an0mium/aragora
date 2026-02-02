"""
HTTP Handlers for OpenClaw Gateway.

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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aragora.observability.metrics import track_handler
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from aragora.server.handlers.utils.decorators import (
    has_permission,
    require_permission,
)
from aragora.server.handlers.utils.rate_limit import (
    auth_rate_limit,
    rate_limit,
)
from aragora.server.validation.query_params import safe_query_int

logger = logging.getLogger(__name__)

# =============================================================================
# Data Models
# =============================================================================


class SessionStatus(Enum):
    """OpenClaw session status."""

    ACTIVE = "active"
    IDLE = "idle"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class ActionStatus(Enum):
    """OpenClaw action execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class CredentialType(Enum):
    """Credential types supported by the gateway."""

    API_KEY = "api_key"
    OAUTH_TOKEN = "oauth_token"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    SERVICE_ACCOUNT = "service_account"


@dataclass
class Session:
    """OpenClaw session data."""

    id: str
    user_id: str
    tenant_id: str | None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
            "config": self.config,
            "metadata": self.metadata,
        }


@dataclass
class Action:
    """OpenClaw action data."""

    id: str
    session_id: str
    action_type: str
    status: ActionStatus
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "action_type": self.action_type,
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


@dataclass
class Credential:
    """OpenClaw credential metadata (never includes actual secret values)."""

    id: str
    name: str
    credential_type: CredentialType
    user_id: str
    tenant_id: str | None
    created_at: datetime
    updated_at: datetime
    last_rotated_at: datetime | None
    expires_at: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization (excludes secret)."""
        return {
            "id": self.id,
            "name": self.name,
            "credential_type": self.credential_type.value,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_rotated_at": (self.last_rotated_at.isoformat() if self.last_rotated_at else None),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


@dataclass
class AuditEntry:
    """Audit log entry for OpenClaw gateway operations."""

    id: str
    timestamp: datetime
    action: str
    actor_id: str
    resource_type: str
    resource_id: str | None
    result: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "actor_id": self.actor_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "result": self.result,
            "details": self.details,
        }


# =============================================================================
# In-Memory Storage (would be replaced with persistent storage in production)
# =============================================================================


class OpenClawGatewayStore:
    """In-memory store for OpenClaw gateway data.

    In production, this would be replaced with a persistent storage backend
    (PostgreSQL, Redis, etc.).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._actions: dict[str, Action] = {}
        self._credentials: dict[str, Credential] = {}
        self._credential_secrets: dict[str, str] = {}  # Stored separately
        self._audit_log: list[AuditEntry] = []

    # Session methods
    def create_session(
        self,
        user_id: str,
        tenant_id: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session."""
        now = datetime.now(timezone.utc)
        session = Session(
            id=str(uuid.uuid4()),
            user_id=user_id,
            tenant_id=tenant_id,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            last_activity_at=now,
            config=config or {},
            metadata=metadata or {},
        )
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        user_id: str | None = None,
        tenant_id: str | None = None,
        status: SessionStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Session], int]:
        """List sessions with optional filtering."""
        sessions = list(self._sessions.values())

        # Apply filters
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if tenant_id:
            sessions = [s for s in sessions if s.tenant_id == tenant_id]
        if status:
            sessions = [s for s in sessions if s.status == status]

        # Sort by created_at descending
        sessions.sort(key=lambda s: s.created_at, reverse=True)

        total = len(sessions)
        return sessions[offset : offset + limit], total

    def update_session_status(self, session_id: str, status: SessionStatus) -> Session | None:
        """Update session status."""
        session = self._sessions.get(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.now(timezone.utc)
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    # Action methods
    def create_action(
        self,
        session_id: str,
        action_type: str,
        input_data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Action:
        """Create a new action."""
        now = datetime.now(timezone.utc)
        action = Action(
            id=str(uuid.uuid4()),
            session_id=session_id,
            action_type=action_type,
            status=ActionStatus.PENDING,
            input_data=input_data,
            output_data=None,
            error=None,
            created_at=now,
            started_at=None,
            completed_at=None,
            metadata=metadata or {},
        )
        self._actions[action.id] = action
        return action

    def get_action(self, action_id: str) -> Action | None:
        """Get action by ID."""
        return self._actions.get(action_id)

    def update_action(
        self,
        action_id: str,
        status: ActionStatus | None = None,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Action | None:
        """Update action state."""
        action = self._actions.get(action_id)
        if action:
            now = datetime.now(timezone.utc)
            if status:
                action.status = status
                if status == ActionStatus.RUNNING and not action.started_at:
                    action.started_at = now
                elif status in (
                    ActionStatus.COMPLETED,
                    ActionStatus.FAILED,
                    ActionStatus.CANCELLED,
                ):
                    action.completed_at = now
            if output_data is not None:
                action.output_data = output_data
            if error is not None:
                action.error = error
        return action

    # Credential methods
    def store_credential(
        self,
        name: str,
        credential_type: CredentialType,
        secret_value: str,
        user_id: str,
        tenant_id: str | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Credential:
        """Store a new credential."""
        now = datetime.now(timezone.utc)
        credential = Credential(
            id=str(uuid.uuid4()),
            name=name,
            credential_type=credential_type,
            user_id=user_id,
            tenant_id=tenant_id,
            created_at=now,
            updated_at=now,
            last_rotated_at=None,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        self._credentials[credential.id] = credential
        self._credential_secrets[credential.id] = secret_value
        return credential

    def get_credential(self, credential_id: str) -> Credential | None:
        """Get credential metadata by ID (not the secret)."""
        return self._credentials.get(credential_id)

    def list_credentials(
        self,
        user_id: str | None = None,
        tenant_id: str | None = None,
        credential_type: CredentialType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Credential], int]:
        """List credentials with optional filtering (no secret values)."""
        credentials = list(self._credentials.values())

        # Apply filters
        if user_id:
            credentials = [c for c in credentials if c.user_id == user_id]
        if tenant_id:
            credentials = [c for c in credentials if c.tenant_id == tenant_id]
        if credential_type:
            credentials = [c for c in credentials if c.credential_type == credential_type]

        # Sort by created_at descending
        credentials.sort(key=lambda c: c.created_at, reverse=True)

        total = len(credentials)
        return credentials[offset : offset + limit], total

    def delete_credential(self, credential_id: str) -> bool:
        """Delete a credential."""
        if credential_id in self._credentials:
            del self._credentials[credential_id]
            del self._credential_secrets[credential_id]
            return True
        return False

    def rotate_credential(self, credential_id: str, new_secret_value: str) -> Credential | None:
        """Rotate a credential's secret value."""
        credential = self._credentials.get(credential_id)
        if credential:
            now = datetime.now(timezone.utc)
            credential.last_rotated_at = now
            credential.updated_at = now
            self._credential_secrets[credential_id] = new_secret_value
        return credential

    # Audit methods
    def add_audit_entry(
        self,
        action: str,
        actor_id: str,
        resource_type: str,
        resource_id: str | None = None,
        result: str = "success",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Add an audit log entry."""
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            action=action,
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            result=result,
            details=details or {},
        )
        self._audit_log.append(entry)
        # Keep only last 10000 entries
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]
        return entry

    def get_audit_log(
        self,
        action: str | None = None,
        actor_id: str | None = None,
        resource_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AuditEntry], int]:
        """Get audit log entries with optional filtering."""
        entries = self._audit_log.copy()

        # Apply filters
        if action:
            entries = [e for e in entries if e.action == action]
        if actor_id:
            entries = [e for e in entries if e.actor_id == actor_id]
        if resource_type:
            entries = [e for e in entries if e.resource_type == resource_type]

        # Sort by timestamp descending (most recent first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        total = len(entries)
        return entries[offset : offset + limit], total

    # Metrics
    def get_metrics(self) -> dict[str, Any]:
        """Get gateway metrics."""
        active_sessions = sum(
            1 for s in self._sessions.values() if s.status == SessionStatus.ACTIVE
        )
        pending_actions = sum(1 for a in self._actions.values() if a.status == ActionStatus.PENDING)
        running_actions = sum(1 for a in self._actions.values() if a.status == ActionStatus.RUNNING)

        return {
            "sessions": {
                "total": len(self._sessions),
                "active": active_sessions,
                "by_status": {
                    status.value: sum(1 for s in self._sessions.values() if s.status == status)
                    for status in SessionStatus
                },
            },
            "actions": {
                "total": len(self._actions),
                "pending": pending_actions,
                "running": running_actions,
                "by_status": {
                    status.value: sum(1 for a in self._actions.values() if a.status == status)
                    for status in ActionStatus
                },
            },
            "credentials": {
                "total": len(self._credentials),
                "by_type": {
                    ctype.value: sum(
                        1 for c in self._credentials.values() if c.credential_type == ctype
                    )
                    for ctype in CredentialType
                },
            },
            "audit_log_entries": len(self._audit_log),
        }


# Global store instance
_store: OpenClawGatewayStore | None = None


def _get_store() -> OpenClawGatewayStore:
    """Get or create the global store instance."""
    global _store
    if _store is None:
        _store = OpenClawGatewayStore()
    return _store


# =============================================================================
# Handler Implementation
# =============================================================================


class OpenClawGatewayHandler(BaseHandler):
    """
    HTTP handler for OpenClaw gateway operations.

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
        if path.startswith("/api/gateway/openclaw/sessions/") and path.count("/") == 4:
            session_id = path.split("/")[-1]
            return self._handle_get_session(session_id, handler)

        # GET /api/gateway/openclaw/actions/:id
        if path.startswith("/api/gateway/openclaw/actions/") and path.count("/") == 4:
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

    @require_permission("gateway:sessions.read")
    @rate_limit(requests_per_minute=120, limiter_name="openclaw_gateway_list_sessions")
    def _handle_list_sessions(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """List sessions with optional filtering."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)
            tenant_id = self._get_tenant_id(handler)

            # Parse query parameters
            status_str = query_params.get("status")
            status = SessionStatus(status_str) if status_str else None
            limit = safe_query_int(query_params, "limit", default=50, max_val=500)
            offset = safe_query_int(query_params, "offset", default=0, min_val=0, max_val=100000)

            # List sessions (scoped to user/tenant for non-admin)
            sessions, total = store.list_sessions(
                user_id=user_id,
                tenant_id=tenant_id,
                status=status,
                limit=limit,
                offset=offset,
            )

            return json_response(
                {
                    "sessions": [s.to_dict() for s in sessions],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )
        except ValueError as e:
            return error_response(f"Invalid parameter: {e}", 400)
        except Exception as e:
            logger.error("Error listing sessions: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:sessions.read")
    @rate_limit(requests_per_minute=120, limiter_name="openclaw_gateway_get_session")
    def _handle_get_session(self, session_id: str, handler: Any) -> HandlerResult:
        """Get session by ID."""
        try:
            store = _get_store()
            session = store.get_session(session_id)

            if not session:
                return error_response(f"Session not found: {session_id}", 404)

            # Check access (user can only see their own sessions unless admin)
            user_id = self._get_user_id(handler)
            user = self.get_current_user(handler)
            is_admin = user and has_permission(
                user.role if hasattr(user, "role") else None, "gateway:admin"
            )

            if not is_admin and session.user_id != user_id:
                return error_response("Access denied", 403)

            return json_response(session.to_dict())
        except Exception as e:
            logger.error("Error getting session %s: %s", session_id, e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:actions.read")
    @rate_limit(requests_per_minute=120, limiter_name="openclaw_gateway_get_action")
    def _handle_get_action(self, action_id: str, handler: Any) -> HandlerResult:
        """Get action status by ID."""
        try:
            store = _get_store()
            action = store.get_action(action_id)

            if not action:
                return error_response(f"Action not found: {action_id}", 404)

            # Check access via session ownership
            session = store.get_session(action.session_id)
            if session:
                user_id = self._get_user_id(handler)
                user = self.get_current_user(handler)
                is_admin = user and has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )

                if not is_admin and session.user_id != user_id:
                    return error_response("Access denied", 403)

            return json_response(action.to_dict())
        except Exception as e:
            logger.error("Error getting action %s: %s", action_id, e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:credentials.read")
    @rate_limit(requests_per_minute=60, limiter_name="openclaw_gateway_list_creds")
    def _handle_list_credentials(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """List credentials (metadata only, no secret values)."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)
            tenant_id = self._get_tenant_id(handler)

            # Parse query parameters
            type_str = query_params.get("type")
            cred_type = CredentialType(type_str) if type_str else None
            limit = safe_query_int(query_params, "limit", default=50, max_val=500)
            offset = safe_query_int(query_params, "offset", default=0, min_val=0, max_val=100000)

            # List credentials (scoped to user/tenant)
            credentials, total = store.list_credentials(
                user_id=user_id,
                tenant_id=tenant_id,
                credential_type=cred_type,
                limit=limit,
                offset=offset,
            )

            return json_response(
                {
                    "credentials": [c.to_dict() for c in credentials],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )
        except ValueError as e:
            return error_response(f"Invalid parameter: {e}", 400)
        except Exception as e:
            logger.error("Error listing credentials: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    def _handle_health(self, handler: Any) -> HandlerResult:
        """Get gateway health status (public endpoint)."""
        try:
            store = _get_store()
            metrics = store.get_metrics()

            # Basic health check
            healthy = True
            status = "healthy"

            # Check for any critical issues
            if metrics["actions"]["running"] > 100:
                status = "degraded"
            if metrics["actions"]["pending"] > 500:
                healthy = False
                status = "unhealthy"

            return json_response(
                {
                    "status": status,
                    "healthy": healthy,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_sessions": metrics["sessions"]["active"],
                    "pending_actions": metrics["actions"]["pending"],
                    "running_actions": metrics["actions"]["running"],
                }
            )
        except Exception as e:
            logger.error("Error getting health: %s", e)
            return json_response(
                {
                    "status": "error",
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=503,
            )

    @require_permission("gateway:metrics.read")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_metrics")
    def _handle_metrics(self, handler: Any) -> HandlerResult:
        """Get gateway metrics."""
        try:
            store = _get_store()
            metrics = store.get_metrics()
            metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

            return json_response(metrics)
        except Exception as e:
            logger.error("Error getting metrics: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:audit.read")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_audit")
    def _handle_audit(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """Get audit log entries."""
        try:
            store = _get_store()

            # Parse query parameters
            action_filter = query_params.get("action")
            actor_filter = query_params.get("actor_id")
            resource_type = query_params.get("resource_type")
            limit = safe_query_int(query_params, "limit", default=100, max_val=1000)
            offset = safe_query_int(query_params, "offset", default=0, min_val=0, max_val=100000)

            entries, total = store.get_audit_log(
                action=action_filter,
                actor_id=actor_filter,
                resource_type=resource_type,
                limit=limit,
                offset=offset,
            )

            return json_response(
                {
                    "entries": [e.to_dict() for e in entries],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )
        except Exception as e:
            logger.error("Error getting audit log: %s", e)
            return error_response(safe_error_message(e, "gateway"), 500)

    # =========================================================================
    # POST Handlers
    # =========================================================================

    @track_handler("gateway/openclaw", method="POST")
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

    @require_permission("gateway:sessions.create")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_create_session")
    def _handle_create_session(self, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Create a new session."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)
            tenant_id = self._get_tenant_id(handler)

            config = body.get("config", {})
            metadata = body.get("metadata", {})

            session = store.create_session(
                user_id=user_id,
                tenant_id=tenant_id,
                config=config,
                metadata=metadata,
            )

            # Audit
            store.add_audit_entry(
                action="session.create",
                actor_id=user_id,
                resource_type="session",
                resource_id=session.id,
                result="success",
            )

            logger.info(f"Created session {session.id} for user {user_id}")
            return json_response(session.to_dict(), status=201)

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:actions.execute")
    @auth_rate_limit(
        requests_per_minute=60,
        limiter_name="openclaw_gateway_execute_action",
        endpoint_name="OpenClaw execute action",
    )
    def _handle_execute_action(self, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Execute an action."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            # Validate required fields
            session_id = body.get("session_id")
            if not session_id:
                return error_response("session_id is required", 400)

            action_type = body.get("action_type")
            if not action_type:
                return error_response("action_type is required", 400)

            # Verify session exists and is owned by user
            session = store.get_session(session_id)
            if not session:
                return error_response(f"Session not found: {session_id}", 404)

            if session.user_id != user_id:
                user = self.get_current_user(handler)
                is_admin = user and has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )
                if not is_admin:
                    return error_response("Access denied", 403)

            if session.status != SessionStatus.ACTIVE:
                return error_response(
                    f"Session is not active (status: {session.status.value})", 400
                )

            input_data = body.get("input", {})
            metadata = body.get("metadata", {})

            # Create action
            action = store.create_action(
                session_id=session_id,
                action_type=action_type,
                input_data=input_data,
                metadata=metadata,
            )

            # Update session activity
            session.last_activity_at = datetime.now(timezone.utc)

            # Audit
            store.add_audit_entry(
                action="action.execute",
                actor_id=user_id,
                resource_type="action",
                resource_id=action.id,
                result="pending",
                details={"action_type": action_type, "session_id": session_id},
            )

            # In a real implementation, this would dispatch to the OpenClaw runtime
            # For now, we just mark it as running
            store.update_action(action.id, status=ActionStatus.RUNNING)

            logger.info(f"Created action {action.id} (type: {action_type}) in session {session_id}")
            return json_response(action.to_dict(), status=202)

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:actions.cancel")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_cancel_action")
    def _handle_cancel_action(self, action_id: str, handler: Any) -> HandlerResult:
        """Cancel a running action."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            action = store.get_action(action_id)
            if not action:
                return error_response(f"Action not found: {action_id}", 404)

            # Verify access
            session = store.get_session(action.session_id)
            if session and session.user_id != user_id:
                user = self.get_current_user(handler)
                is_admin = user and has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )
                if not is_admin:
                    return error_response("Access denied", 403)

            # Check if cancellable
            if action.status not in (ActionStatus.PENDING, ActionStatus.RUNNING):
                return error_response(
                    f"Action cannot be cancelled (status: {action.status.value})", 400
                )

            # Cancel the action
            store.update_action(action.id, status=ActionStatus.CANCELLED)

            # Audit
            store.add_audit_entry(
                action="action.cancel",
                actor_id=user_id,
                resource_type="action",
                resource_id=action_id,
                result="success",
            )

            logger.info(f"Cancelled action {action_id}")
            return json_response({"cancelled": True, "action_id": action_id})

        except Exception as e:
            logger.error(f"Error cancelling action {action_id}: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:credentials.create")
    @auth_rate_limit(
        requests_per_minute=10,
        limiter_name="openclaw_gateway_store_credential",
        endpoint_name="OpenClaw store credential",
    )
    def _handle_store_credential(self, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Store a new credential."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)
            tenant_id = self._get_tenant_id(handler)

            # Validate required fields
            name = body.get("name")
            if not name:
                return error_response("name is required", 400)

            credential_type_str = body.get("type")
            if not credential_type_str:
                return error_response("type is required", 400)

            try:
                credential_type = CredentialType(credential_type_str)
            except ValueError:
                valid_types = [t.value for t in CredentialType]
                return error_response(f"Invalid credential type. Valid types: {valid_types}", 400)

            secret_value = body.get("secret")
            if not secret_value:
                return error_response("secret is required", 400)

            # Optional expiration
            expires_at = None
            if body.get("expires_at"):
                try:
                    expires_at = datetime.fromisoformat(body["expires_at"])
                except ValueError:
                    return error_response("Invalid expires_at format (use ISO 8601)", 400)

            metadata = body.get("metadata", {})

            credential = store.store_credential(
                name=name,
                credential_type=credential_type,
                secret_value=secret_value,
                user_id=user_id,
                tenant_id=tenant_id,
                expires_at=expires_at,
                metadata=metadata,
            )

            # Audit (without revealing the secret)
            store.add_audit_entry(
                action="credential.create",
                actor_id=user_id,
                resource_type="credential",
                resource_id=credential.id,
                result="success",
                details={"name": name, "type": credential_type_str},
            )

            logger.info(f"Stored credential {credential.id} ({name}) for user {user_id}")
            return json_response(credential.to_dict(), status=201)

        except Exception as e:
            logger.error(f"Error storing credential: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:credentials.rotate")
    @auth_rate_limit(
        requests_per_minute=10,
        limiter_name="openclaw_gateway_rotate_credential",
        endpoint_name="OpenClaw rotate credential",
    )
    def _handle_rotate_credential(
        self, credential_id: str, body: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Rotate a credential's secret value."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            credential = store.get_credential(credential_id)
            if not credential:
                return error_response(f"Credential not found: {credential_id}", 404)

            # Verify ownership
            if credential.user_id != user_id:
                user = self.get_current_user(handler)
                is_admin = user and has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )
                if not is_admin:
                    return error_response("Access denied", 403)

            new_secret = body.get("secret")
            if not new_secret:
                return error_response("secret is required", 400)

            # Rotate
            credential = store.rotate_credential(credential_id, new_secret)

            # Audit
            store.add_audit_entry(
                action="credential.rotate",
                actor_id=user_id,
                resource_type="credential",
                resource_id=credential_id,
                result="success",
            )

            logger.info(f"Rotated credential {credential_id}")
            return json_response(
                {
                    "rotated": True,
                    "credential_id": credential_id,
                    "rotated_at": credential.last_rotated_at.isoformat()
                    if credential.last_rotated_at
                    else None,
                }
            )

        except Exception as e:
            logger.error(f"Error rotating credential {credential_id}: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    # =========================================================================
    # Policy, Approvals, Stats, and End Session Handlers
    # =========================================================================

    @require_permission("gateway:policy.read")
    @rate_limit(requests_per_minute=60, limiter_name="openclaw_gateway_get_policy")
    def _handle_get_policy_rules(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """Get active policy rules."""
        try:
            store = _get_store()
            rules = store.get_policy_rules() if hasattr(store, "get_policy_rules") else []

            return json_response(
                {
                    "rules": [r.to_dict() if hasattr(r, "to_dict") else r for r in rules],
                    "total": len(rules),
                }
            )
        except Exception as e:
            logger.error(f"Error getting policy rules: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:policy.write")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_add_policy")
    def _handle_add_policy_rule(self, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Add a policy rule."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            name = body.get("name")
            if not name:
                return error_response("name is required", 400)

            action_types = body.get("action_types", [])
            decision = body.get("decision", "deny")
            priority = body.get("priority", 0)
            description = body.get("description", "")
            enabled = body.get("enabled", True)
            config = body.get("config", {})

            if hasattr(store, "add_policy_rule"):
                rule = store.add_policy_rule(
                    name=name,
                    action_types=action_types,
                    decision=decision,
                    priority=priority,
                    description=description,
                    enabled=enabled,
                    config=config,
                )
            else:
                rule = {
                    "name": name,
                    "action_types": action_types,
                    "decision": decision,
                    "priority": priority,
                    "description": description,
                    "enabled": enabled,
                    "config": config,
                }

            # Audit
            store.add_audit_entry(
                action="policy.rule.add",
                actor_id=user_id,
                resource_type="policy_rule",
                resource_id=name,
                result="success",
                details={"decision": decision, "action_types": action_types},
            )

            logger.info(f"Added policy rule {name}")
            result = rule.to_dict() if hasattr(rule, "to_dict") else rule
            return json_response(result, status=201)

        except Exception as e:
            logger.error(f"Error adding policy rule: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:policy.write")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_remove_policy")
    def _handle_remove_policy_rule(self, rule_name: str, handler: Any) -> HandlerResult:
        """Remove a policy rule."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            if hasattr(store, "remove_policy_rule"):
                removed = store.remove_policy_rule(rule_name)
            else:
                removed = True

            # Audit
            store.add_audit_entry(
                action="policy.rule.remove",
                actor_id=user_id,
                resource_type="policy_rule",
                resource_id=rule_name,
                result="success",
            )

            logger.info(f"Removed policy rule {rule_name}")
            return json_response({"success": removed, "name": rule_name})

        except Exception as e:
            logger.error(f"Error removing policy rule {rule_name}: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:approvals.read")
    @rate_limit(requests_per_minute=60, limiter_name="openclaw_gateway_list_approvals")
    def _handle_list_approvals(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """List pending approval requests."""
        try:
            store = _get_store()
            tenant_id = self._get_tenant_id(handler)

            limit = safe_query_int(query_params, "limit", default=50, max_val=500)
            offset = safe_query_int(query_params, "offset", default=0, min_val=0, max_val=100000)

            if hasattr(store, "list_approvals"):
                approvals, total = store.list_approvals(
                    tenant_id=tenant_id,
                    limit=limit,
                    offset=offset,
                )
            else:
                approvals, total = [], 0

            return json_response(
                {
                    "approvals": [a.to_dict() if hasattr(a, "to_dict") else a for a in approvals],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }
            )
        except Exception as e:
            logger.error(f"Error listing approvals: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:approvals.write")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_approve")
    def _handle_approve_action(
        self, approval_id: str, body: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Approve a pending action."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            approver_id = body.get("approver_id", user_id)
            reason = body.get("reason", "")

            if hasattr(store, "approve_action"):
                result = store.approve_action(
                    approval_id=approval_id,
                    approver_id=approver_id,
                    reason=reason,
                )
                success = result if isinstance(result, bool) else True
            else:
                success = True

            # Audit
            store.add_audit_entry(
                action="approval.approve",
                actor_id=user_id,
                resource_type="approval",
                resource_id=approval_id,
                result="success",
                details={"approver_id": approver_id, "reason": reason},
            )

            logger.info(f"Approved action {approval_id} by {approver_id}")
            return json_response({"success": success, "approval_id": approval_id})

        except Exception as e:
            logger.error(f"Error approving action {approval_id}: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:approvals.write")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_deny")
    def _handle_deny_action(
        self, approval_id: str, body: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Deny a pending action."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            approver_id = body.get("approver_id", user_id)
            reason = body.get("reason", "")

            if hasattr(store, "deny_action"):
                result = store.deny_action(
                    approval_id=approval_id,
                    approver_id=approver_id,
                    reason=reason,
                )
                success = result if isinstance(result, bool) else True
            else:
                success = True

            # Audit
            store.add_audit_entry(
                action="approval.deny",
                actor_id=user_id,
                resource_type="approval",
                resource_id=approval_id,
                result="success",
                details={"approver_id": approver_id, "reason": reason},
            )

            logger.info(f"Denied action {approval_id} by {approver_id}")
            return json_response({"success": success, "approval_id": approval_id})

        except Exception as e:
            logger.error(f"Error denying action {approval_id}: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:metrics.read")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_stats")
    def _handle_stats(self, handler: Any) -> HandlerResult:
        """Get proxy statistics."""
        try:
            store = _get_store()
            metrics = store.get_metrics()

            return json_response(
                {
                    "active_sessions": metrics.get("sessions", {}).get("active", 0),
                    "actions_allowed": metrics.get("actions", {}).get("completed", 0),
                    "actions_denied": metrics.get("actions", {}).get("failed", 0),
                    "pending_approvals": metrics.get("actions", {}).get("pending", 0),
                    "policy_rules": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:sessions.delete")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_end_session")
    def _handle_end_session(self, session_id: str, handler: Any) -> HandlerResult:
        """End a session via POST (SDK-compatible endpoint)."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            session = store.get_session(session_id)
            if not session:
                return error_response(f"Session not found: {session_id}", 404)

            # Verify ownership
            if session.user_id != user_id:
                user = self.get_current_user(handler)
                is_admin = user and has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )
                if not is_admin:
                    return error_response("Access denied", 403)

            # Close the session
            store.update_session_status(session_id, SessionStatus.CLOSED)

            # Audit
            store.add_audit_entry(
                action="session.end",
                actor_id=user_id,
                resource_type="session",
                resource_id=session_id,
                result="success",
            )

            logger.info(f"Ended session {session_id}")
            return json_response({"success": True, "session_id": session_id})

        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    # =========================================================================
    # DELETE Handlers
    # =========================================================================

    @track_handler("gateway/openclaw", method="DELETE")
    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle DELETE requests."""
        path = self._normalize_path(path)

        # DELETE /api/gateway/openclaw/sessions/:id
        if path.startswith("/api/gateway/openclaw/sessions/") and path.count("/") == 4:
            session_id = path.split("/")[-1]
            return self._handle_close_session(session_id, handler)

        # DELETE /api/gateway/openclaw/policy/rules/:name
        if path.startswith("/api/gateway/openclaw/policy/rules/"):
            rule_name = path.split("/")[-1]
            return self._handle_remove_policy_rule(rule_name, handler)

        # DELETE /api/gateway/openclaw/credentials/:id
        if path.startswith("/api/gateway/openclaw/credentials/") and path.count("/") == 4:
            credential_id = path.split("/")[-1]
            return self._handle_delete_credential(credential_id, handler)

        return None

    @require_permission("gateway:sessions.delete")
    @rate_limit(requests_per_minute=30, limiter_name="openclaw_gateway_close_session")
    def _handle_close_session(self, session_id: str, handler: Any) -> HandlerResult:
        """Close a session."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            session = store.get_session(session_id)
            if not session:
                return error_response(f"Session not found: {session_id}", 404)

            # Verify ownership
            if session.user_id != user_id:
                user = self.get_current_user(handler)
                is_admin = user and has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )
                if not is_admin:
                    return error_response("Access denied", 403)

            # Close the session
            store.update_session_status(session_id, SessionStatus.CLOSED)

            # Audit
            store.add_audit_entry(
                action="session.close",
                actor_id=user_id,
                resource_type="session",
                resource_id=session_id,
                result="success",
            )

            logger.info(f"Closed session {session_id}")
            return json_response({"closed": True, "session_id": session_id})

        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)

    @require_permission("gateway:credentials.delete")
    @rate_limit(requests_per_minute=20, limiter_name="openclaw_gateway_delete_cred")
    def _handle_delete_credential(self, credential_id: str, handler: Any) -> HandlerResult:
        """Delete a credential."""
        try:
            store = _get_store()
            user_id = self._get_user_id(handler)

            credential = store.get_credential(credential_id)
            if not credential:
                return error_response(f"Credential not found: {credential_id}", 404)

            # Verify ownership
            if credential.user_id != user_id:
                user = self.get_current_user(handler)
                is_admin = user and has_permission(
                    user.role if hasattr(user, "role") else None, "gateway:admin"
                )
                if not is_admin:
                    return error_response("Access denied", 403)

            # Delete
            store.delete_credential(credential_id)

            # Audit
            store.add_audit_entry(
                action="credential.delete",
                actor_id=user_id,
                resource_type="credential",
                resource_id=credential_id,
                result="success",
            )

            logger.info(f"Deleted credential {credential_id}")
            return json_response({"deleted": True, "credential_id": credential_id})

        except Exception as e:
            logger.error(f"Error deleting credential {credential_id}: {e}")
            return error_response(safe_error_message(e, "gateway"), 500)


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
    # Data models
    "Session",
    "SessionStatus",
    "Action",
    "ActionStatus",
    "Credential",
    "CredentialType",
    "AuditEntry",
    # Store (for testing)
    "OpenClawGatewayStore",
    "_get_store",
]
