"""
Gateway Namespace API.

Provides methods for device gateway management:
- Device registration and management
- Channel listing
- Routing statistics and rules
- Message routing

Endpoints:
    GET    /api/v1/gateway/devices           - List devices
    POST   /api/v1/gateway/devices           - Register device
    GET    /api/v1/gateway/devices/{id}      - Get device
    DELETE /api/v1/gateway/devices/{id}      - Unregister device
    POST   /api/v1/gateway/devices/{id}/heartbeat - Device heartbeat
    GET    /api/v1/gateway/channels          - List channels
    GET    /api/v1/gateway/routing/stats     - Routing statistics
    GET    /api/v1/gateway/routing/rules     - List routing rules
    POST   /api/v1/gateway/messages/route    - Route a message
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

DeviceStatus = Literal["online", "offline", "unknown"]


class GatewayAPI:
    """
    Synchronous Gateway API.

    Provides methods for managing devices and routing messages
    through the local gateway.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # List devices
        >>> devices = client.gateway.list_devices()
        >>> # Register a device
        >>> device = client.gateway.register_device(
        ...     name="Smart Speaker",
        ...     device_type="alexa",
        ...     capabilities=["voice", "tts"],
        ... )
        >>> # Route a message
        >>> result = client.gateway.route_message(
        ...     channel="slack",
        ...     content="Hello from the gateway!",
        ... )
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Devices
    # =========================================================================

    def list_devices(
        self,
        status: DeviceStatus | None = None,
        device_type: str | None = None,
    ) -> dict[str, Any]:
        """
        List registered devices.

        Args:
            status: Filter by device status.
            device_type: Filter by device type.

        Returns:
            Dict with devices array and total count.
        """
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if device_type:
            params["type"] = device_type

        return self._client.request("GET", "/api/v1/gateway/devices", params=params)

    def get_device(self, device_id: str) -> dict[str, Any]:
        """
        Get device details.

        Args:
            device_id: Device ID.

        Returns:
            Dict with device info.
        """
        return self._client.request("GET", f"/api/v1/gateway/devices/{device_id}")

    def register_device(
        self,
        name: str,
        device_type: str = "unknown",
        device_id: str | None = None,
        capabilities: list[str] | None = None,
        allowed_channels: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Register a new device.

        Args:
            name: Device name.
            device_type: Device type (e.g., "alexa", "google_home").
            device_id: Optional device ID (auto-generated if not provided).
            capabilities: List of device capabilities.
            allowed_channels: Channels the device can access.
            metadata: Additional device metadata.

        Returns:
            Dict with device_id and success message.
        """
        data: dict[str, Any] = {"name": name, "device_type": device_type}
        if device_id:
            data["device_id"] = device_id
        if capabilities:
            data["capabilities"] = capabilities
        if allowed_channels:
            data["allowed_channels"] = allowed_channels
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/v1/gateway/devices", json=data)

    def unregister_device(self, device_id: str) -> dict[str, Any]:
        """
        Unregister a device.

        Args:
            device_id: Device ID.

        Returns:
            Dict with success message.
        """
        return self._client.request("DELETE", f"/api/v1/gateway/devices/{device_id}")

    def heartbeat(self, device_id: str) -> dict[str, Any]:
        """
        Send device heartbeat.

        Args:
            device_id: Device ID.

        Returns:
            Dict with status.
        """
        return self._client.request("POST", f"/api/v1/gateway/devices/{device_id}/heartbeat")

    # =========================================================================
    # Channels
    # =========================================================================

    def list_channels(self) -> dict[str, Any]:
        """
        List active channels.

        Returns:
            Dict with channels array and total count.
        """
        return self._client.request("GET", "/api/v1/gateway/channels")

    # =========================================================================
    # Routing
    # =========================================================================

    def get_routing_stats(self) -> dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dict with routing stats.
        """
        return self._client.request("GET", "/api/v1/gateway/routing/stats")

    def list_routing_rules(self) -> dict[str, Any]:
        """
        List routing rules.

        Returns:
            Dict with rules array and total count.
        """
        return self._client.request("GET", "/api/v1/gateway/routing/rules")

    # =========================================================================
    # Messages
    # =========================================================================

    def route_message(
        self,
        channel: str,
        content: str,
    ) -> dict[str, Any]:
        """
        Route a message through the gateway.

        Args:
            channel: Target channel.
            content: Message content.

        Returns:
            Dict with routed status, agent_id, rule_id.
        """
        return self._client.request(
            "POST",
            "/api/v1/gateway/messages/route",
            json={"channel": channel, "content": content},
        )

    # =========================================================================
    # Gateway Health
    # =========================================================================

    def health(self) -> dict[str, Any]:
        """
        Check gateway health status.

        Returns:
            Dict with gateway health status and component details.
        """
        return self._client.request("GET", "/api/v1/gateway/health")

    # =========================================================================
    # Agents
    # =========================================================================

    def list_agents(self) -> dict[str, Any]:
        """
        List registered gateway agents.

        Returns:
            Dict with agents array and total count.
        """
        return self._client.request("GET", "/api/v1/gateway/agents")

    def get_agent(self, agent_name: str) -> dict[str, Any]:
        """
        Get details for a specific gateway agent.

        Args:
            agent_name: Agent name or identifier.

        Returns:
            Dict with agent info.
        """
        return self._client.request("GET", f"/api/v1/gateway/agents/{agent_name}")

    def get_agent_health(self, agent_name: str) -> dict[str, Any]:
        """
        Check health of a specific gateway agent.

        Args:
            agent_name: Agent name or identifier.

        Returns:
            Dict with agent health status.
        """
        return self._client.request(
            "GET", f"/api/v1/gateway/agents/{agent_name}/health"
        )

    # =========================================================================
    # Credentials
    # =========================================================================

    def list_credentials(self) -> dict[str, Any]:
        """
        List stored credentials (metadata only, no secrets).

        Returns:
            Dict with credentials array and total count.
        """
        return self._client.request("GET", "/api/v1/gateway/credentials")

    def create_credential(
        self,
        name: str,
        credential_type: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a new credential in the gateway.

        Args:
            name: Credential name for identification.
            credential_type: Type of credential (api_key, oauth_token, etc.).
            value: Credential value (will be encrypted at rest).
            metadata: Optional additional metadata.

        Returns:
            Dict with credential_id and success message.
        """
        data: dict[str, Any] = {
            "name": name,
            "type": credential_type,
            "value": value,
        }
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/v1/gateway/credentials", json=data)

    def delete_credential(self, credential_id: str) -> dict[str, Any]:
        """
        Delete a stored credential.

        Args:
            credential_id: Credential ID.

        Returns:
            Dict with success message.
        """
        return self._client.request(
            "DELETE", f"/api/v1/gateway/credentials/{credential_id}"
        )

    # =========================================================================
    # Messages (extended)
    # =========================================================================

    def send_message(
        self,
        channel: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a message to the gateway for routing.

        Args:
            channel: Target channel for routing.
            content: Message content to route.
            metadata: Optional additional message metadata.

        Returns:
            Dict with routed status, agent_id, and rule_id.
        """
        data: dict[str, Any] = {"channel": channel, "content": content}
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/v1/gateway/messages", json=data)

    def get_message(self, message_id: str) -> dict[str, Any]:
        """
        Get details for a specific routed message.

        Args:
            message_id: Message ID.

        Returns:
            Dict with message details and routing info.
        """
        return self._client.request(
            "GET", f"/api/v1/gateway/messages/{message_id}"
        )

    # =========================================================================
    # Routing (extended)
    # =========================================================================

    def list_routing(self) -> dict[str, Any]:
        """
        List routing rules with statistics.

        Returns:
            Dict with rules array, total count, and stats.
        """
        return self._client.request("GET", "/api/v1/gateway/routing")

    def create_routing_rule(
        self,
        channel: str,
        pattern: str,
        agent_id: str,
        priority: int | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """
        Create a new routing rule.

        Args:
            channel: Channel to apply the rule to.
            pattern: Message matching pattern (regex supported).
            agent_id: Target agent for matched messages.
            priority: Rule evaluation priority (lower is higher).
            enabled: Whether the rule is active.

        Returns:
            Dict with created rule and success message.
        """
        data: dict[str, Any] = {
            "channel": channel,
            "pattern": pattern,
            "agent_id": agent_id,
            "enabled": enabled,
        }
        if priority is not None:
            data["priority"] = priority

        return self._client.request("POST", "/api/v1/gateway/routing", json=data)

    def get_routing_rule(self, route_id: str) -> dict[str, Any]:
        """
        Get details for a specific routing rule.

        Args:
            route_id: Routing rule ID.

        Returns:
            Dict with rule details.
        """
        return self._client.request("GET", f"/api/v1/gateway/routing/{route_id}")


class AsyncGatewayAPI:
    """Asynchronous Gateway API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Devices
    # =========================================================================

    async def list_devices(
        self,
        status: DeviceStatus | None = None,
        device_type: str | None = None,
    ) -> dict[str, Any]:
        """List registered devices."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if device_type:
            params["type"] = device_type

        return await self._client.request("GET", "/api/v1/gateway/devices", params=params)

    async def get_device(self, device_id: str) -> dict[str, Any]:
        """Get device details."""
        return await self._client.request("GET", f"/api/v1/gateway/devices/{device_id}")

    async def register_device(
        self,
        name: str,
        device_type: str = "unknown",
        device_id: str | None = None,
        capabilities: list[str] | None = None,
        allowed_channels: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a new device."""
        data: dict[str, Any] = {"name": name, "device_type": device_type}
        if device_id:
            data["device_id"] = device_id
        if capabilities:
            data["capabilities"] = capabilities
        if allowed_channels:
            data["allowed_channels"] = allowed_channels
        if metadata:
            data["metadata"] = metadata

        return await self._client.request("POST", "/api/v1/gateway/devices", json=data)

    async def unregister_device(self, device_id: str) -> dict[str, Any]:
        """Unregister a device."""
        return await self._client.request("DELETE", f"/api/v1/gateway/devices/{device_id}")

    async def heartbeat(self, device_id: str) -> dict[str, Any]:
        """Send device heartbeat."""
        return await self._client.request("POST", f"/api/v1/gateway/devices/{device_id}/heartbeat")

    # =========================================================================
    # Channels
    # =========================================================================

    async def list_channels(self) -> dict[str, Any]:
        """List active channels."""
        return await self._client.request("GET", "/api/v1/gateway/channels")

    # =========================================================================
    # Routing
    # =========================================================================

    async def get_routing_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        return await self._client.request("GET", "/api/v1/gateway/routing/stats")

    async def list_routing_rules(self) -> dict[str, Any]:
        """List routing rules."""
        return await self._client.request("GET", "/api/v1/gateway/routing/rules")

    # =========================================================================
    # Messages
    # =========================================================================

    async def route_message(
        self,
        channel: str,
        content: str,
    ) -> dict[str, Any]:
        """Route a message through the gateway."""
        return await self._client.request(
            "POST",
            "/api/v1/gateway/messages/route",
            json={"channel": channel, "content": content},
        )

    # =========================================================================
    # Gateway Health
    # =========================================================================

    async def health(self) -> dict[str, Any]:
        """Check gateway health status."""
        return await self._client.request("GET", "/api/v1/gateway/health")

    # =========================================================================
    # Agents
    # =========================================================================

    async def list_agents(self) -> dict[str, Any]:
        """List registered gateway agents."""
        return await self._client.request("GET", "/api/v1/gateway/agents")

    async def get_agent(self, agent_name: str) -> dict[str, Any]:
        """Get details for a specific gateway agent."""
        return await self._client.request(
            "GET", f"/api/v1/gateway/agents/{agent_name}"
        )

    async def get_agent_health(self, agent_name: str) -> dict[str, Any]:
        """Check health of a specific gateway agent."""
        return await self._client.request(
            "GET", f"/api/v1/gateway/agents/{agent_name}/health"
        )

    # =========================================================================
    # Credentials
    # =========================================================================

    async def list_credentials(self) -> dict[str, Any]:
        """List stored credentials (metadata only, no secrets)."""
        return await self._client.request("GET", "/api/v1/gateway/credentials")

    async def create_credential(
        self,
        name: str,
        credential_type: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a new credential in the gateway."""
        data: dict[str, Any] = {
            "name": name,
            "type": credential_type,
            "value": value,
        }
        if metadata:
            data["metadata"] = metadata

        return await self._client.request(
            "POST", "/api/v1/gateway/credentials", json=data
        )

    async def delete_credential(self, credential_id: str) -> dict[str, Any]:
        """Delete a stored credential."""
        return await self._client.request(
            "DELETE", f"/api/v1/gateway/credentials/{credential_id}"
        )

    # =========================================================================
    # Messages (extended)
    # =========================================================================

    async def send_message(
        self,
        channel: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a message to the gateway for routing."""
        data: dict[str, Any] = {"channel": channel, "content": content}
        if metadata:
            data["metadata"] = metadata

        return await self._client.request(
            "POST", "/api/v1/gateway/messages", json=data
        )

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """Get details for a specific routed message."""
        return await self._client.request(
            "GET", f"/api/v1/gateway/messages/{message_id}"
        )

    # =========================================================================
    # Routing (extended)
    # =========================================================================

    async def list_routing(self) -> dict[str, Any]:
        """List routing rules with statistics."""
        return await self._client.request("GET", "/api/v1/gateway/routing")

    async def create_routing_rule(
        self,
        channel: str,
        pattern: str,
        agent_id: str,
        priority: int | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Create a new routing rule."""
        data: dict[str, Any] = {
            "channel": channel,
            "pattern": pattern,
            "agent_id": agent_id,
            "enabled": enabled,
        }
        if priority is not None:
            data["priority"] = priority

        return await self._client.request(
            "POST", "/api/v1/gateway/routing", json=data
        )

    async def get_routing_rule(self, route_id: str) -> dict[str, Any]:
        """Get details for a specific routing rule."""
        return await self._client.request(
            "GET", f"/api/v1/gateway/routing/{route_id}"
        )
