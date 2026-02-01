"""
OpenClaw API resource for the Aragora client.

Provides methods for interacting with the OpenClaw Enterprise Gateway:
- Session management (create, end, list)
- Action execution (shell, file, browser)
- Policy management (get, update rules)
- Approval workflows (list, approve, deny)
- Audit trail queries
"""

from __future__ import annotations

import builtins
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class OpenClawSession:
    """An active OpenClaw proxy session."""

    session_id: str
    user_id: str
    tenant_id: str
    workspace_id: str
    roles: builtins.list[str] = field(default_factory=list)
    status: str = "active"
    action_count: int = 0
    created_at: datetime | None = None
    last_activity: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Result of an action executed through the proxy."""

    success: bool
    action_id: str
    decision: str  # allow, deny, require_approval
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    audit_id: str | None = None
    requires_approval: bool = False
    approval_id: str | None = None


@dataclass
class PolicyRule:
    """A policy rule in the gateway."""

    name: str
    action_types: builtins.list[str]
    decision: str
    priority: int = 0
    description: str = ""
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingApproval:
    """An action pending human approval."""

    approval_id: str
    session_id: str
    user_id: str
    tenant_id: str
    action_type: str
    action_params: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    created_at: datetime | None = None
    expires_at: datetime | None = None


@dataclass
class AuditRecord:
    """A record from the audit trail."""

    record_id: str
    event_type: str
    timestamp: float = 0.0
    user_id: str | None = None
    session_id: str | None = None
    tenant_id: str | None = None
    action_type: str | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProxyStats:
    """Gateway proxy statistics."""

    active_sessions: int = 0
    actions_allowed: int = 0
    actions_denied: int = 0
    pending_approvals: int = 0
    policy_rules: int = 0


# =============================================================================
# API Resource
# =============================================================================


class OpenClawAPI:
    """API interface for OpenClaw Enterprise Gateway operations."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_session(
        self,
        user_id: str,
        tenant_id: str,
        workspace_id: str = "/workspace",
        roles: builtins.list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OpenClawSession:
        """Create a new proxy session."""
        payload: dict[str, Any] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
        }
        if roles:
            payload["roles"] = roles
        if metadata:
            payload["metadata"] = metadata

        response = self._client._post("/api/v1/openclaw/sessions", json=payload)
        return self._parse_session(response)

    async def create_session_async(
        self,
        user_id: str,
        tenant_id: str,
        workspace_id: str = "/workspace",
        roles: builtins.list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OpenClawSession:
        """Create a new proxy session (async)."""
        payload: dict[str, Any] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
        }
        if roles:
            payload["roles"] = roles
        if metadata:
            payload["metadata"] = metadata

        response = await self._client._post_async("/api/v1/openclaw/sessions", json=payload)
        return self._parse_session(response)

    def end_session(self, session_id: str) -> bool:
        """End an active session."""
        response = self._client._post(f"/api/v1/openclaw/sessions/{session_id}/end")
        return response.get("success", False)

    async def end_session_async(self, session_id: str) -> bool:
        """End an active session (async)."""
        response = await self._client._post_async(f"/api/v1/openclaw/sessions/{session_id}/end")
        return response.get("success", False)

    def get_session(self, session_id: str) -> OpenClawSession:
        """Get session details."""
        response = self._client._get(f"/api/v1/openclaw/sessions/{session_id}")
        return self._parse_session(response)

    async def get_session_async(self, session_id: str) -> OpenClawSession:
        """Get session details (async)."""
        response = await self._client._get_async(f"/api/v1/openclaw/sessions/{session_id}")
        return self._parse_session(response)

    def list_sessions(
        self,
        tenant_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> builtins.list[OpenClawSession]:
        """List active sessions."""
        params: dict[str, Any] = {"limit": limit}
        if tenant_id:
            params["tenant_id"] = tenant_id
        if status:
            params["status"] = status

        response = self._client._get("/api/v1/openclaw/sessions", params=params)
        return [self._parse_session(s) for s in response.get("sessions", [])]

    async def list_sessions_async(
        self,
        tenant_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> builtins.list[OpenClawSession]:
        """List active sessions (async)."""
        params: dict[str, Any] = {"limit": limit}
        if tenant_id:
            params["tenant_id"] = tenant_id
        if status:
            params["status"] = status

        response = await self._client._get_async("/api/v1/openclaw/sessions", params=params)
        return [self._parse_session(s) for s in response.get("sessions", [])]

    # =========================================================================
    # Action Execution
    # =========================================================================

    def execute_action(
        self,
        session_id: str,
        action_type: str,
        **params: Any,
    ) -> ActionResult:
        """Execute an action through the proxy."""
        payload: dict[str, Any] = {
            "session_id": session_id,
            "action_type": action_type,
            "params": params,
        }
        response = self._client._post("/api/v1/openclaw/actions", json=payload)
        return self._parse_action_result(response)

    async def execute_action_async(
        self,
        session_id: str,
        action_type: str,
        **params: Any,
    ) -> ActionResult:
        """Execute an action through the proxy (async)."""
        payload: dict[str, Any] = {
            "session_id": session_id,
            "action_type": action_type,
            "params": params,
        }
        response = await self._client._post_async("/api/v1/openclaw/actions", json=payload)
        return self._parse_action_result(response)

    def execute_shell(self, session_id: str, command: str) -> ActionResult:
        """Execute a shell command through the proxy."""
        return self.execute_action(session_id, "shell", command=command)

    async def execute_shell_async(self, session_id: str, command: str) -> ActionResult:
        """Execute a shell command through the proxy (async)."""
        return await self.execute_action_async(session_id, "shell", command=command)

    def execute_file_read(self, session_id: str, path: str) -> ActionResult:
        """Read a file through the proxy."""
        return self.execute_action(session_id, "file_read", path=path)

    async def execute_file_read_async(self, session_id: str, path: str) -> ActionResult:
        """Read a file through the proxy (async)."""
        return await self.execute_action_async(session_id, "file_read", path=path)

    def execute_file_write(self, session_id: str, path: str, content: str) -> ActionResult:
        """Write a file through the proxy."""
        return self.execute_action(session_id, "file_write", path=path, content=content)

    async def execute_file_write_async(
        self, session_id: str, path: str, content: str
    ) -> ActionResult:
        """Write a file through the proxy (async)."""
        return await self.execute_action_async(session_id, "file_write", path=path, content=content)

    def execute_browser(self, session_id: str, url: str, action: str = "navigate") -> ActionResult:
        """Execute a browser action through the proxy."""
        return self.execute_action(session_id, "browser", url=url, action=action)

    async def execute_browser_async(
        self, session_id: str, url: str, action: str = "navigate"
    ) -> ActionResult:
        """Execute a browser action through the proxy (async)."""
        return await self.execute_action_async(session_id, "browser", url=url, action=action)

    # =========================================================================
    # Policy Management
    # =========================================================================

    def get_policy_rules(self) -> builtins.list[PolicyRule]:
        """Get active policy rules."""
        response = self._client._get("/api/v1/openclaw/policy/rules")
        return [self._parse_rule(r) for r in response.get("rules", [])]

    async def get_policy_rules_async(self) -> builtins.list[PolicyRule]:
        """Get active policy rules (async)."""
        response = await self._client._get_async("/api/v1/openclaw/policy/rules")
        return [self._parse_rule(r) for r in response.get("rules", [])]

    def add_rule(self, rule: PolicyRule) -> PolicyRule:
        """Add a policy rule."""
        payload = {
            "name": rule.name,
            "action_types": rule.action_types,
            "decision": rule.decision,
            "priority": rule.priority,
            "description": rule.description,
            "enabled": rule.enabled,
            "config": rule.config,
        }
        response = self._client._post("/api/v1/openclaw/policy/rules", json=payload)
        return self._parse_rule(response)

    async def add_rule_async(self, rule: PolicyRule) -> PolicyRule:
        """Add a policy rule (async)."""
        payload = {
            "name": rule.name,
            "action_types": rule.action_types,
            "decision": rule.decision,
            "priority": rule.priority,
            "description": rule.description,
            "enabled": rule.enabled,
            "config": rule.config,
        }
        response = await self._client._post_async("/api/v1/openclaw/policy/rules", json=payload)
        return self._parse_rule(response)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a policy rule."""
        response = self._client._delete(f"/api/v1/openclaw/policy/rules/{rule_name}")
        return response.get("success", False)

    async def remove_rule_async(self, rule_name: str) -> bool:
        """Remove a policy rule (async)."""
        response = await self._client._delete_async(f"/api/v1/openclaw/policy/rules/{rule_name}")
        return response.get("success", False)

    # =========================================================================
    # Approval Workflows
    # =========================================================================

    def list_pending_approvals(
        self,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> builtins.list[PendingApproval]:
        """List pending approval requests."""
        params: dict[str, Any] = {"limit": limit}
        if tenant_id:
            params["tenant_id"] = tenant_id

        response = self._client._get("/api/v1/openclaw/approvals", params=params)
        return [self._parse_approval(a) for a in response.get("approvals", [])]

    async def list_pending_approvals_async(
        self,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> builtins.list[PendingApproval]:
        """List pending approval requests (async)."""
        params: dict[str, Any] = {"limit": limit}
        if tenant_id:
            params["tenant_id"] = tenant_id

        response = await self._client._get_async("/api/v1/openclaw/approvals", params=params)
        return [self._parse_approval(a) for a in response.get("approvals", [])]

    def approve_action(self, approval_id: str, approver_id: str, reason: str = "") -> bool:
        """Approve a pending action."""
        payload = {"approver_id": approver_id, "reason": reason}
        response = self._client._post(
            f"/api/v1/openclaw/approvals/{approval_id}/approve", json=payload
        )
        return response.get("success", False)

    async def approve_action_async(
        self, approval_id: str, approver_id: str, reason: str = ""
    ) -> bool:
        """Approve a pending action (async)."""
        payload = {"approver_id": approver_id, "reason": reason}
        response = await self._client._post_async(
            f"/api/v1/openclaw/approvals/{approval_id}/approve", json=payload
        )
        return response.get("success", False)

    def deny_approval(self, approval_id: str, approver_id: str, reason: str = "") -> bool:
        """Deny a pending approval."""
        payload = {"approver_id": approver_id, "reason": reason}
        response = self._client._post(
            f"/api/v1/openclaw/approvals/{approval_id}/deny", json=payload
        )
        return response.get("success", False)

    async def deny_approval_async(
        self, approval_id: str, approver_id: str, reason: str = ""
    ) -> bool:
        """Deny a pending approval (async)."""
        payload = {"approver_id": approver_id, "reason": reason}
        response = await self._client._post_async(
            f"/api/v1/openclaw/approvals/{approval_id}/deny", json=payload
        )
        return response.get("success", False)

    # =========================================================================
    # Audit Trail
    # =========================================================================

    def query_audit(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        event_type: str | None = None,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> builtins.list[AuditRecord]:
        """Query the audit trail."""
        params: dict[str, Any] = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        if session_id:
            params["session_id"] = session_id
        if event_type:
            params["event_type"] = event_type
        if tenant_id:
            params["tenant_id"] = tenant_id

        response = self._client._get("/api/v1/openclaw/audit", params=params)
        return [self._parse_audit_record(r) for r in response.get("records", [])]

    async def query_audit_async(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        event_type: str | None = None,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> builtins.list[AuditRecord]:
        """Query the audit trail (async)."""
        params: dict[str, Any] = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        if session_id:
            params["session_id"] = session_id
        if event_type:
            params["event_type"] = event_type
        if tenant_id:
            params["tenant_id"] = tenant_id

        response = await self._client._get_async("/api/v1/openclaw/audit", params=params)
        return [self._parse_audit_record(r) for r in response.get("records", [])]

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> ProxyStats:
        """Get proxy statistics."""
        response = self._client._get("/api/v1/openclaw/stats")
        return ProxyStats(
            active_sessions=response.get("active_sessions", 0),
            actions_allowed=response.get("actions_allowed", 0),
            actions_denied=response.get("actions_denied", 0),
            pending_approvals=response.get("pending_approvals", 0),
            policy_rules=response.get("policy_rules", 0),
        )

    async def get_stats_async(self) -> ProxyStats:
        """Get proxy statistics (async)."""
        response = await self._client._get_async("/api/v1/openclaw/stats")
        return ProxyStats(
            active_sessions=response.get("active_sessions", 0),
            actions_allowed=response.get("actions_allowed", 0),
            actions_denied=response.get("actions_denied", 0),
            pending_approvals=response.get("pending_approvals", 0),
            policy_rules=response.get("policy_rules", 0),
        )

    # =========================================================================
    # Parsing Helpers
    # =========================================================================

    @staticmethod
    def _parse_session(data: dict[str, Any]) -> OpenClawSession:
        """Parse session data from API response."""
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(str(data["created_at"]))
            except (ValueError, TypeError):
                pass

        last_activity = None
        if data.get("last_activity"):
            try:
                last_activity = datetime.fromisoformat(str(data["last_activity"]))
            except (ValueError, TypeError):
                pass

        return OpenClawSession(
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id", ""),
            tenant_id=data.get("tenant_id", ""),
            workspace_id=data.get("workspace_id", "/workspace"),
            roles=data.get("roles", []),
            status=data.get("status", "active"),
            action_count=data.get("action_count", 0),
            created_at=created_at,
            last_activity=last_activity,
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def _parse_action_result(data: dict[str, Any]) -> ActionResult:
        """Parse action result from API response."""
        return ActionResult(
            success=data.get("success", False),
            action_id=data.get("action_id", ""),
            decision=data.get("decision", "deny"),
            result=data.get("result"),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            audit_id=data.get("audit_id"),
            requires_approval=data.get("requires_approval", False),
            approval_id=data.get("approval_id"),
        )

    @staticmethod
    def _parse_rule(data: dict[str, Any]) -> PolicyRule:
        """Parse policy rule from API response."""
        return PolicyRule(
            name=data.get("name", ""),
            action_types=data.get("action_types", []),
            decision=data.get("decision", "deny"),
            priority=data.get("priority", 0),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            config=data.get("config", {}),
        )

    @staticmethod
    def _parse_approval(data: dict[str, Any]) -> PendingApproval:
        """Parse approval from API response."""
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(str(data["created_at"]))
            except (ValueError, TypeError):
                pass

        expires_at = None
        if data.get("expires_at"):
            try:
                expires_at = datetime.fromisoformat(str(data["expires_at"]))
            except (ValueError, TypeError):
                pass

        return PendingApproval(
            approval_id=data.get("approval_id", ""),
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id", ""),
            tenant_id=data.get("tenant_id", ""),
            action_type=data.get("action_type", ""),
            action_params=data.get("action_params", {}),
            status=data.get("status", "pending"),
            created_at=created_at,
            expires_at=expires_at,
        )

    @staticmethod
    def _parse_audit_record(data: dict[str, Any]) -> AuditRecord:
        """Parse audit record from API response."""
        return AuditRecord(
            record_id=data.get("record_id", ""),
            event_type=data.get("event_type", ""),
            timestamp=data.get("timestamp", 0.0),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            tenant_id=data.get("tenant_id"),
            action_type=data.get("action_type"),
            success=data.get("success", True),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )
