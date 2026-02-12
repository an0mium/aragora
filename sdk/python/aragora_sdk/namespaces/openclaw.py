"""OpenClaw Gateway namespace API.

Provides methods for Aragora's OpenClaw gateway endpoints:
- Session orchestration
- Action execution
- Policy and approvals
- Credential lifecycle
- Health, metrics, and audit
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class OpenclawAPI:
    """Synchronous OpenClaw Gateway API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    # Session management
    def list_sessions(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List OpenClaw sessions."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/openclaw/sessions", params=params)

    def create_session(
        self,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new OpenClaw session."""
        payload: dict[str, Any] = {}
        if config is not None:
            payload["config"] = config
        if metadata is not None:
            payload["metadata"] = metadata
        return self._client.request("POST", "/api/v1/openclaw/sessions", json=payload)

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get OpenClaw session by ID."""
        return self._client.request("GET", f"/api/v1/openclaw/sessions/{session_id}")

    def end_session(self, session_id: str) -> dict[str, Any]:
        """End an active OpenClaw session."""
        return self._client.request("POST", f"/api/v1/openclaw/sessions/{session_id}/end")

    # Action management
    def execute_action(
        self,
        session_id: str,
        action_type: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an action in an OpenClaw session."""
        payload: dict[str, Any] = {
            "session_id": session_id,
            "action_type": action_type,
            "params": params or {},
        }
        return self._client.request("POST", "/api/v1/openclaw/actions", json=payload)

    def get_action(self, action_id: str) -> dict[str, Any]:
        """Get OpenClaw action status."""
        return self._client.request("GET", f"/api/v1/openclaw/actions/{action_id}")

    def cancel_action(self, action_id: str) -> dict[str, Any]:
        """Cancel an OpenClaw action."""
        return self._client.request("POST", f"/api/v1/openclaw/actions/{action_id}/cancel")

    # Policy and approvals
    def get_policy_rules(self) -> dict[str, Any]:
        """List policy rules."""
        return self._client.request("GET", "/api/v1/openclaw/policy/rules")

    def add_policy_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        """Create a policy rule."""
        return self._client.request("POST", "/api/v1/openclaw/policy/rules", json=rule)

    def remove_policy_rule(self, rule_name: str) -> dict[str, Any]:
        """Delete a policy rule."""
        return self._client.request("DELETE", f"/api/v1/openclaw/policy/rules/{rule_name}")

    def list_approvals(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List pending approvals."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/openclaw/approvals", params=params)

    def approve_action(self, approval_id: str, reason: str | None = None) -> dict[str, Any]:
        """Approve a pending action."""
        body: dict[str, Any] = {}
        if reason:
            body["reason"] = reason
        return self._client.request(
            "POST",
            f"/api/v1/openclaw/approvals/{approval_id}/approve",
            json=body,
        )

    def deny_action(self, approval_id: str, reason: str) -> dict[str, Any]:
        """Deny a pending action."""
        return self._client.request(
            "POST",
            f"/api/v1/openclaw/approvals/{approval_id}/deny",
            json={"reason": reason},
        )

    # Credential lifecycle
    def list_credentials(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List credential metadata."""
        return self._client.request(
            "GET",
            "/api/v1/openclaw/credentials",
            params={"limit": limit, "offset": offset},
        )

    def store_credential(
        self,
        name: str,
        credential_type: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a credential."""
        payload: dict[str, Any] = {
            "name": name,
            "credential_type": credential_type,
            "value": value,
        }
        if metadata is not None:
            payload["metadata"] = metadata
        return self._client.request("POST", "/api/v1/openclaw/credentials", json=payload)

    def rotate_credential(self, credential_id: str, value: str) -> dict[str, Any]:
        """Rotate a credential value."""
        return self._client.request(
            "POST",
            f"/api/v1/openclaw/credentials/{credential_id}/rotate",
            json={"value": value},
        )

    def delete_credential(self, credential_id: str) -> dict[str, Any]:
        """Delete a credential."""
        return self._client.request("DELETE", f"/api/v1/openclaw/credentials/{credential_id}")

    # Service introspection
    def health(self) -> dict[str, Any]:
        """Get OpenClaw gateway health."""
        return self._client.request("GET", "/api/v1/openclaw/health")

    def metrics(self) -> dict[str, Any]:
        """Get OpenClaw gateway metrics."""
        return self._client.request("GET", "/api/v1/openclaw/metrics")

    def audit(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Get OpenClaw audit events."""
        return self._client.request(
            "GET",
            "/api/v1/openclaw/audit",
            params={"limit": limit, "offset": offset},
        )

    def stats(self) -> dict[str, Any]:
        """Get OpenClaw aggregate stats."""
        return self._client.request("GET", "/api/v1/openclaw/stats")


class AsyncOpenclawAPI:
    """Asynchronous OpenClaw Gateway API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # Session management
    async def list_sessions(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List OpenClaw sessions."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/openclaw/sessions", params=params)

    async def create_session(
        self,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new OpenClaw session."""
        payload: dict[str, Any] = {}
        if config is not None:
            payload["config"] = config
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/openclaw/sessions", json=payload)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Get OpenClaw session by ID."""
        return await self._client.request("GET", f"/api/v1/openclaw/sessions/{session_id}")

    async def end_session(self, session_id: str) -> dict[str, Any]:
        """End an active OpenClaw session."""
        return await self._client.request("POST", f"/api/v1/openclaw/sessions/{session_id}/end")

    # Action management
    async def execute_action(
        self,
        session_id: str,
        action_type: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an action in an OpenClaw session."""
        payload: dict[str, Any] = {
            "session_id": session_id,
            "action_type": action_type,
            "params": params or {},
        }
        return await self._client.request("POST", "/api/v1/openclaw/actions", json=payload)

    async def get_action(self, action_id: str) -> dict[str, Any]:
        """Get OpenClaw action status."""
        return await self._client.request("GET", f"/api/v1/openclaw/actions/{action_id}")

    async def cancel_action(self, action_id: str) -> dict[str, Any]:
        """Cancel an OpenClaw action."""
        return await self._client.request("POST", f"/api/v1/openclaw/actions/{action_id}/cancel")

    # Policy and approvals
    async def get_policy_rules(self) -> dict[str, Any]:
        """List policy rules."""
        return await self._client.request("GET", "/api/v1/openclaw/policy/rules")

    async def add_policy_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        """Create a policy rule."""
        return await self._client.request("POST", "/api/v1/openclaw/policy/rules", json=rule)

    async def remove_policy_rule(self, rule_name: str) -> dict[str, Any]:
        """Delete a policy rule."""
        return await self._client.request("DELETE", f"/api/v1/openclaw/policy/rules/{rule_name}")

    async def list_approvals(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List pending approvals."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/openclaw/approvals", params=params)

    async def approve_action(self, approval_id: str, reason: str | None = None) -> dict[str, Any]:
        """Approve a pending action."""
        body: dict[str, Any] = {}
        if reason:
            body["reason"] = reason
        return await self._client.request(
            "POST",
            f"/api/v1/openclaw/approvals/{approval_id}/approve",
            json=body,
        )

    async def deny_action(self, approval_id: str, reason: str) -> dict[str, Any]:
        """Deny a pending action."""
        return await self._client.request(
            "POST",
            f"/api/v1/openclaw/approvals/{approval_id}/deny",
            json={"reason": reason},
        )

    # Credential lifecycle
    async def list_credentials(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List credential metadata."""
        return await self._client.request(
            "GET",
            "/api/v1/openclaw/credentials",
            params={"limit": limit, "offset": offset},
        )

    async def store_credential(
        self,
        name: str,
        credential_type: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a credential."""
        payload: dict[str, Any] = {
            "name": name,
            "credential_type": credential_type,
            "value": value,
        }
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/openclaw/credentials", json=payload)

    async def rotate_credential(self, credential_id: str, value: str) -> dict[str, Any]:
        """Rotate a credential value."""
        return await self._client.request(
            "POST",
            f"/api/v1/openclaw/credentials/{credential_id}/rotate",
            json={"value": value},
        )

    async def delete_credential(self, credential_id: str) -> dict[str, Any]:
        """Delete a credential."""
        return await self._client.request("DELETE", f"/api/v1/openclaw/credentials/{credential_id}")

    # Service introspection
    async def health(self) -> dict[str, Any]:
        """Get OpenClaw gateway health."""
        return await self._client.request("GET", "/api/v1/openclaw/health")

    async def metrics(self) -> dict[str, Any]:
        """Get OpenClaw gateway metrics."""
        return await self._client.request("GET", "/api/v1/openclaw/metrics")

    async def audit(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Get OpenClaw audit events."""
        return await self._client.request(
            "GET",
            "/api/v1/openclaw/audit",
            params={"limit": limit, "offset": offset},
        )

    async def stats(self) -> dict[str, Any]:
        """Get OpenClaw aggregate stats."""
        return await self._client.request("GET", "/api/v1/openclaw/stats")
