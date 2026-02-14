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

    # -- Session management ---------------------------------------------------

    def list_sessions(self, skip: int = 0, limit: int = 100) -> dict[str, Any]:
        """List active OpenClaw sessions."""
        return self._client.request(
            "GET", "/api/v1/openclaw/sessions", params={"skip": skip, "limit": limit}
        )

    def create_session(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new OpenClaw session."""
        return self._client.request("POST", "/api/v1/openclaw/sessions", json=kwargs)

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get an OpenClaw session by ID."""
        return self._client.request("GET", f"/api/v1/openclaw/sessions/{session_id}")

    def end_session(self, session_id: str) -> dict[str, Any]:
        """End an active OpenClaw session."""
        return self._client.request("POST", f"/api/v1/openclaw/sessions/{session_id}/end")

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete an OpenClaw session."""
        return self._client.request("DELETE", f"/api/v1/openclaw/sessions/{session_id}")

    # -- Action management ----------------------------------------------------

    def execute_action(
        self,
        action_type: str,
        input_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit a new action for execution."""
        payload: dict[str, Any] = {"action_type": action_type, **kwargs}
        if input_data is not None:
            payload["input_data"] = input_data
        return self._client.request("POST", "/api/v1/openclaw/actions", json=payload)

    def get_action(self, action_id: str) -> dict[str, Any]:
        """Get an action by ID."""
        return self._client.request("GET", f"/api/v1/openclaw/actions/{action_id}")

    def cancel_action(self, action_id: str) -> dict[str, Any]:
        """Cancel a pending action."""
        return self._client.request("POST", f"/api/v1/openclaw/actions/{action_id}/cancel")

    # -- Credential lifecycle -------------------------------------------------

    def list_credentials(self) -> dict[str, Any]:
        """List OpenClaw credentials."""
        return self._client.request("GET", "/api/v1/openclaw/credentials")

    def store_credential(self, **kwargs: Any) -> dict[str, Any]:
        """Store a new credential."""
        return self._client.request("POST", "/api/v1/openclaw/credentials", json=kwargs)

    def delete_credential(self, credential_id: str) -> dict[str, Any]:
        """Delete a credential."""
        return self._client.request("DELETE", f"/api/v1/openclaw/credentials/{credential_id}")

    def rotate_credential(
        self, credential_id: str, new_value: str | None = None
    ) -> dict[str, Any]:
        """Rotate a credential."""
        payload: dict[str, Any] = {}
        if new_value is not None:
            payload["new_value"] = new_value
        return self._client.request(
            "POST", f"/api/v1/openclaw/credentials/{credential_id}/rotate", json=payload
        )

    # -- Policy rules ---------------------------------------------------------

    def get_policy_rules(self) -> dict[str, Any]:
        """List policy rules."""
        return self._client.request("GET", "/api/v1/openclaw/policy/rules")

    def add_policy_rule(self, **kwargs: Any) -> dict[str, Any]:
        """Add a new policy rule."""
        return self._client.request("POST", "/api/v1/openclaw/policy/rules", json=kwargs)

    def remove_policy_rule(self, rule_id: str) -> dict[str, Any]:
        """Remove a policy rule."""
        return self._client.request("DELETE", f"/api/v1/openclaw/policy/rules/{rule_id}")

    # -- Approvals ------------------------------------------------------------

    def list_approvals(self) -> dict[str, Any]:
        """List pending approvals."""
        return self._client.request("GET", "/api/v1/openclaw/approvals")

    def approve_action(self, approval_id: str) -> dict[str, Any]:
        """Approve a pending approval."""
        return self._client.request("POST", f"/api/v1/openclaw/approvals/{approval_id}/approve")

    def deny_action(self, approval_id: str) -> dict[str, Any]:
        """Deny a pending approval."""
        return self._client.request("POST", f"/api/v1/openclaw/approvals/{approval_id}/deny")

    # -- Service introspection ------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Get OpenClaw gateway health status."""
        return self._client.request("GET", "/api/v1/openclaw/health")

    def metrics(self) -> dict[str, Any]:
        """Get OpenClaw gateway metrics."""
        return self._client.request("GET", "/api/v1/openclaw/metrics")

    def audit(
        self,
        event_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Get OpenClaw audit log with optional filters."""
        params: dict[str, str] = {}
        if event_type is not None:
            params["event_type"] = event_type
        if user_id is not None:
            params["user_id"] = user_id
        if session_id is not None:
            params["session_id"] = session_id
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        return self._client.request("GET", "/api/v1/openclaw/audit", params=params or None)

    def stats(self) -> dict[str, Any]:
        """Get OpenClaw gateway stats."""
        return self._client.request("GET", "/api/v1/openclaw/stats")


class AsyncOpenclawAPI:
    """Asynchronous OpenClaw Gateway API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # -- Session management ---------------------------------------------------

    async def list_sessions(self, skip: int = 0, limit: int = 100) -> dict[str, Any]:
        """List active OpenClaw sessions."""
        return await self._client.request(
            "GET", "/api/v1/openclaw/sessions", params={"skip": skip, "limit": limit}
        )

    async def create_session(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new OpenClaw session."""
        return await self._client.request("POST", "/api/v1/openclaw/sessions", json=kwargs)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Get an OpenClaw session by ID."""
        return await self._client.request("GET", f"/api/v1/openclaw/sessions/{session_id}")

    async def end_session(self, session_id: str) -> dict[str, Any]:
        """End an active OpenClaw session."""
        return await self._client.request("POST", f"/api/v1/openclaw/sessions/{session_id}/end")

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete an OpenClaw session."""
        return await self._client.request("DELETE", f"/api/v1/openclaw/sessions/{session_id}")

    # -- Action management ----------------------------------------------------

    async def execute_action(
        self,
        action_type: str,
        input_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit a new action for execution."""
        payload: dict[str, Any] = {"action_type": action_type, **kwargs}
        if input_data is not None:
            payload["input_data"] = input_data
        return await self._client.request("POST", "/api/v1/openclaw/actions", json=payload)

    async def get_action(self, action_id: str) -> dict[str, Any]:
        """Get an action by ID."""
        return await self._client.request("GET", f"/api/v1/openclaw/actions/{action_id}")

    async def cancel_action(self, action_id: str) -> dict[str, Any]:
        """Cancel a pending action."""
        return await self._client.request("POST", f"/api/v1/openclaw/actions/{action_id}/cancel")

    # -- Credential lifecycle -------------------------------------------------

    async def list_credentials(self) -> dict[str, Any]:
        """List OpenClaw credentials."""
        return await self._client.request("GET", "/api/v1/openclaw/credentials")

    async def store_credential(self, **kwargs: Any) -> dict[str, Any]:
        """Store a new credential."""
        return await self._client.request("POST", "/api/v1/openclaw/credentials", json=kwargs)

    async def delete_credential(self, credential_id: str) -> dict[str, Any]:
        """Delete a credential."""
        return await self._client.request("DELETE", f"/api/v1/openclaw/credentials/{credential_id}")

    async def rotate_credential(
        self, credential_id: str, new_value: str | None = None
    ) -> dict[str, Any]:
        """Rotate a credential."""
        payload: dict[str, Any] = {}
        if new_value is not None:
            payload["new_value"] = new_value
        return await self._client.request(
            "POST", f"/api/v1/openclaw/credentials/{credential_id}/rotate", json=payload
        )

    # -- Policy rules ---------------------------------------------------------

    async def get_policy_rules(self) -> dict[str, Any]:
        """List policy rules."""
        return await self._client.request("GET", "/api/v1/openclaw/policy/rules")

    async def add_policy_rule(self, **kwargs: Any) -> dict[str, Any]:
        """Add a new policy rule."""
        return await self._client.request("POST", "/api/v1/openclaw/policy/rules", json=kwargs)

    async def remove_policy_rule(self, rule_id: str) -> dict[str, Any]:
        """Remove a policy rule."""
        return await self._client.request("DELETE", f"/api/v1/openclaw/policy/rules/{rule_id}")

    # -- Approvals ------------------------------------------------------------

    async def list_approvals(self) -> dict[str, Any]:
        """List pending approvals."""
        return await self._client.request("GET", "/api/v1/openclaw/approvals")

    async def approve_action(self, approval_id: str) -> dict[str, Any]:
        """Approve a pending approval."""
        return await self._client.request("POST", f"/api/v1/openclaw/approvals/{approval_id}/approve")

    async def deny_action(self, approval_id: str) -> dict[str, Any]:
        """Deny a pending approval."""
        return await self._client.request("POST", f"/api/v1/openclaw/approvals/{approval_id}/deny")

    # -- Service introspection ------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Get OpenClaw gateway health status."""
        return await self._client.request("GET", "/api/v1/openclaw/health")

    async def metrics(self) -> dict[str, Any]:
        """Get OpenClaw gateway metrics."""
        return await self._client.request("GET", "/api/v1/openclaw/metrics")

    async def audit(
        self,
        event_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Get OpenClaw audit log with optional filters."""
        params: dict[str, str] = {}
        if event_type is not None:
            params["event_type"] = event_type
        if user_id is not None:
            params["user_id"] = user_id
        if session_id is not None:
            params["session_id"] = session_id
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        return await self._client.request("GET", "/api/v1/openclaw/audit", params=params or None)

    async def stats(self) -> dict[str, Any]:
        """Get OpenClaw gateway stats."""
        return await self._client.request("GET", "/api/v1/openclaw/stats")
