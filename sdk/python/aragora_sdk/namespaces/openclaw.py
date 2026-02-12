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

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete an OpenClaw session."""
        return self._client.request("DELETE", f"/api/v1/openclaw/sessions/{session_id}")

    # Action management
    def execute_action(
        self,
        session_id: str,
        action_type: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an action in an OpenClaw session."""
        payload: dict[str, Any] = {
            "session_id": session_id,
            "action_type": action_type,
        }
        if input_data is not None:
            payload["input_data"] = input_data
        if metadata is not None:
            payload["metadata"] = metadata
        return self._client.request("POST", "/api/v1/openclaw/actions", json=payload)

    def get_action(self, action_id: str) -> dict[str, Any]:
        """Get OpenClaw action status."""
        return self._client.request("GET", f"/api/v1/openclaw/actions/{action_id}")

    def cancel_action(self, action_id: str) -> dict[str, Any]:
        """Cancel an OpenClaw action."""
        return self._client.request("POST", f"/api/v1/openclaw/actions/{action_id}/cancel")

    # Policy and approvals
    def get_policy_rules(self, enabled: bool | None = None) -> dict[str, Any]:
        """List policy rules."""
        params: dict[str, Any] = {}
        if enabled is not None:
            params["enabled"] = enabled
        return self._client.request("GET", "/api/v1/openclaw/policy/rules", params=params or None)

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
    def list_credentials(
        self,
        credential_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List credential metadata."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if credential_type:
            params["credential_type"] = credential_type
        return self._client.request(
            "GET",
            "/api/v1/openclaw/credentials",
            params=params,
        )

    def store_credential(
        self,
        name: str,
        credential_type: str,
        value: str,
        expires_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a credential."""
        payload: dict[str, Any] = {
            "name": name,
            "credential_type": credential_type,
            "value": value,
        }
        if expires_at is not None:
            payload["expires_at"] = expires_at
        if metadata is not None:
            payload["metadata"] = metadata
        return self._client.request("POST", "/api/v1/openclaw/credentials", json=payload)

    def rotate_credential(self, credential_id: str, new_value: str) -> dict[str, Any]:
        """Rotate a credential with a new value."""
        return self._client.request(
            "POST",
            f"/api/v1/openclaw/credentials/{credential_id}/rotate",
            json={"new_value": new_value},
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

    def audit(
        self,
        event_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get OpenClaw audit events."""
        params: dict[str, Any] = {"limit": limit}
        if event_type:
            params["event_type"] = event_type
        if user_id:
            params["user_id"] = user_id
        if session_id:
            params["session_id"] = session_id
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        return self._client.request(
            "GET",
            "/api/v1/openclaw/audit",
            params=params,
        )

    def stats(self) -> dict[str, Any]:
        """Get OpenClaw aggregate stats."""
        return self._client.request("GET", "/api/v1/openclaw/stats")

    # Gateway task execution
    def gateway_execute(
        self,
        content: str,
        request_type: str = "task",
        capabilities: list[str] | None = None,
        plugins: list[str] | None = None,
        priority: str = "normal",
        timeout_seconds: int = 300,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a task through the OpenClaw gateway."""
        payload: dict[str, Any] = {
            "content": content,
            "request_type": request_type,
            "priority": priority,
            "timeout_seconds": timeout_seconds,
        }
        if capabilities is not None:
            payload["capabilities"] = capabilities
        if plugins is not None:
            payload["plugins"] = plugins
        if context is not None:
            payload["context"] = context
        if metadata is not None:
            payload["metadata"] = metadata
        return self._client.request(
            "POST", "/api/v1/gateway/openclaw/execute", json=payload
        )

    def gateway_status(self, task_id: str) -> dict[str, Any]:
        """Get gateway task execution status."""
        return self._client.request(
            "GET", f"/api/v1/gateway/openclaw/status/{task_id}"
        )

    def register_device(
        self,
        device_id: str,
        device_name: str,
        device_type: str,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a device with the OpenClaw gateway."""
        payload: dict[str, Any] = {
            "device_id": device_id,
            "device_name": device_name,
            "device_type": device_type,
        }
        if capabilities is not None:
            payload["capabilities"] = capabilities
        if metadata is not None:
            payload["metadata"] = metadata
        return self._client.request(
            "POST", "/api/v1/gateway/openclaw/devices/register", json=payload
        )

    def unregister_device(self, device_id: str) -> dict[str, Any]:
        """Unregister a device from the OpenClaw gateway."""
        return self._client.request(
            "POST",
            "/api/v1/gateway/openclaw/devices/unregister",
            json={"device_id": device_id},
        )

    def install_plugin(
        self,
        plugin_id: str,
        plugin_name: str,
        version: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Install a plugin on the OpenClaw gateway."""
        payload: dict[str, Any] = {
            "plugin_id": plugin_id,
            "plugin_name": plugin_name,
            "version": version,
        }
        if config is not None:
            payload["config"] = config
        return self._client.request(
            "POST", "/api/v1/gateway/openclaw/plugins/install", json=payload
        )

    def uninstall_plugin(self, plugin_id: str) -> dict[str, Any]:
        """Uninstall a plugin from the OpenClaw gateway."""
        return self._client.request(
            "POST",
            "/api/v1/gateway/openclaw/plugins/uninstall",
            json={"plugin_id": plugin_id},
        )

    def gateway_config(self) -> dict[str, Any]:
        """Get OpenClaw gateway configuration."""
        return self._client.request("GET", "/api/v1/gateway/openclaw/config")


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

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete an OpenClaw session."""
        return await self._client.request("DELETE", f"/api/v1/openclaw/sessions/{session_id}")

    # Action management
    async def execute_action(
        self,
        session_id: str,
        action_type: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an action in an OpenClaw session."""
        payload: dict[str, Any] = {
            "session_id": session_id,
            "action_type": action_type,
        }
        if input_data is not None:
            payload["input_data"] = input_data
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/openclaw/actions", json=payload)

    async def get_action(self, action_id: str) -> dict[str, Any]:
        """Get OpenClaw action status."""
        return await self._client.request("GET", f"/api/v1/openclaw/actions/{action_id}")

    async def cancel_action(self, action_id: str) -> dict[str, Any]:
        """Cancel an OpenClaw action."""
        return await self._client.request("POST", f"/api/v1/openclaw/actions/{action_id}/cancel")

    # Policy and approvals
    async def get_policy_rules(self, enabled: bool | None = None) -> dict[str, Any]:
        """List policy rules."""
        params: dict[str, Any] = {}
        if enabled is not None:
            params["enabled"] = enabled
        return await self._client.request(
            "GET", "/api/v1/openclaw/policy/rules", params=params or None
        )

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
    async def list_credentials(
        self,
        credential_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List credential metadata."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if credential_type:
            params["credential_type"] = credential_type
        return await self._client.request(
            "GET",
            "/api/v1/openclaw/credentials",
            params=params,
        )

    async def store_credential(
        self,
        name: str,
        credential_type: str,
        value: str,
        expires_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a credential."""
        payload: dict[str, Any] = {
            "name": name,
            "credential_type": credential_type,
            "value": value,
        }
        if expires_at is not None:
            payload["expires_at"] = expires_at
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/openclaw/credentials", json=payload)

    async def rotate_credential(self, credential_id: str, new_value: str) -> dict[str, Any]:
        """Rotate a credential with a new value."""
        return await self._client.request(
            "POST",
            f"/api/v1/openclaw/credentials/{credential_id}/rotate",
            json={"new_value": new_value},
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

    async def audit(
        self,
        event_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get OpenClaw audit events."""
        params: dict[str, Any] = {"limit": limit}
        if event_type:
            params["event_type"] = event_type
        if user_id:
            params["user_id"] = user_id
        if session_id:
            params["session_id"] = session_id
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        return await self._client.request(
            "GET",
            "/api/v1/openclaw/audit",
            params=params,
        )

    async def stats(self) -> dict[str, Any]:
        """Get OpenClaw aggregate stats."""
        return await self._client.request("GET", "/api/v1/openclaw/stats")
    # Gateway task execution
    async def gateway_execute(
        self,
        content: str,
        request_type: str = "task",
        capabilities: list[str] | None = None,
        plugins: list[str] | None = None,
        priority: str = "normal",
        timeout_seconds: int = 300,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a task through the OpenClaw gateway."""
        payload: dict[str, Any] = {
            "content": content,
            "request_type": request_type,
            "priority": priority,
            "timeout_seconds": timeout_seconds,
        }
        if capabilities is not None:
            payload["capabilities"] = capabilities
        if plugins is not None:
            payload["plugins"] = plugins
        if context is not None:
            payload["context"] = context
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._client.request(
            "POST", "/api/v1/gateway/openclaw/execute", json=payload
        )

    async def gateway_status(self, task_id: str) -> dict[str, Any]:
        """Get gateway task execution status."""
        return await self._client.request(
            "GET", f"/api/v1/gateway/openclaw/status/{task_id}"
        )

    async def register_device(
        self,
        device_id: str,
        device_name: str,
        device_type: str,
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a device with the OpenClaw gateway."""
        payload: dict[str, Any] = {
            "device_id": device_id,
            "device_name": device_name,
            "device_type": device_type,
        }
        if capabilities is not None:
            payload["capabilities"] = capabilities
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._client.request(
            "POST", "/api/v1/gateway/openclaw/devices/register", json=payload
        )

    async def unregister_device(self, device_id: str) -> dict[str, Any]:
        """Unregister a device from the OpenClaw gateway."""
        return await self._client.request(
            "POST",
            "/api/v1/gateway/openclaw/devices/unregister",
            json={"device_id": device_id},
        )

    async def install_plugin(
        self,
        plugin_id: str,
        plugin_name: str,
        version: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Install a plugin on the OpenClaw gateway."""
        payload: dict[str, Any] = {
            "plugin_id": plugin_id,
            "plugin_name": plugin_name,
            "version": version,
        }
        if config is not None:
            payload["config"] = config
        return await self._client.request(
            "POST", "/api/v1/gateway/openclaw/plugins/install", json=payload
        )

    async def uninstall_plugin(self, plugin_id: str) -> dict[str, Any]:
        """Uninstall a plugin from the OpenClaw gateway."""
        return await self._client.request(
            "POST",
            "/api/v1/gateway/openclaw/plugins/uninstall",
            json={"plugin_id": plugin_id},
        )

    async def gateway_config(self) -> dict[str, Any]:
        """Get OpenClaw gateway configuration."""
        return await self._client.request("GET", "/api/v1/gateway/openclaw/config")

