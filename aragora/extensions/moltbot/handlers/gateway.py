"""
Moltbot Gateway Handler - Device Management REST API.

Endpoints:
- GET  /api/v1/moltbot/devices          - List devices
- POST /api/v1/moltbot/devices          - Register device
- GET  /api/v1/moltbot/devices/{id}     - Get device
- DELETE /api/v1/moltbot/devices/{id}   - Unregister device
- POST /api/v1/moltbot/devices/{id}/command  - Send command
- POST /api/v1/moltbot/devices/{id}/heartbeat - Device heartbeat
- GET  /api/v1/moltbot/gateway/stats    - Gateway statistics
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

if TYPE_CHECKING:
    from aragora.extensions.moltbot import LocalGateway

logger = logging.getLogger(__name__)

# Global gateway instance (lazily initialized)
_gateway: LocalGateway | None = None


def get_gateway() -> LocalGateway:
    """Get or create the gateway instance."""
    global _gateway
    if _gateway is None:
        from aragora.extensions.moltbot import LocalGateway

        _gateway = LocalGateway()
    return _gateway


class MoltbotGatewayHandler(BaseHandler):
    """HTTP handler for Moltbot device gateway operations."""

    routes = [
        ("GET", "/api/v1/moltbot/devices"),
        ("POST", "/api/v1/moltbot/devices"),
        ("GET", "/api/v1/moltbot/devices/"),
        ("DELETE", "/api/v1/moltbot/devices/"),
        ("POST", "/api/v1/moltbot/devices/*/command"),
        ("POST", "/api/v1/moltbot/devices/*/heartbeat"),
        ("GET", "/api/v1/moltbot/gateway/stats"),
    ]

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle GET requests."""
        if path == "/api/v1/moltbot/devices":
            return await self._handle_list_devices(query_params, handler)
        elif path == "/api/v1/moltbot/gateway/stats":
            return await self._handle_gateway_stats(handler)
        elif path.startswith("/api/v1/moltbot/devices/"):
            # Extract device ID
            parts = path.split("/")
            if len(parts) >= 5:
                device_id = parts[4]
                return await self._handle_get_device(device_id, handler)
        return None

    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        if path == "/api/v1/moltbot/devices":
            return await self._handle_register_device(handler)
        elif path.endswith("/command"):
            parts = path.split("/")
            if len(parts) >= 5:
                device_id = parts[4]
                return await self._handle_send_command(device_id, handler)
        elif path.endswith("/heartbeat"):
            parts = path.split("/")
            if len(parts) >= 5:
                device_id = parts[4]
                return await self._handle_heartbeat(device_id, handler)
        return None

    async def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle DELETE requests."""
        if path.startswith("/api/v1/moltbot/devices/"):
            parts = path.split("/")
            if len(parts) >= 5:
                device_id = parts[4]
                return await self._handle_unregister_device(device_id, handler)
        return None

    # ========== Handler Methods ==========

    async def _handle_list_devices(
        self, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """
        List devices with optional filters.

        GET /api/v1/moltbot/devices?user_id=...&device_type=...&status=...
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        gateway = get_gateway()

        # Extract filters
        user_id = query_params.get("user_id")
        device_type = query_params.get("device_type")
        status = query_params.get("status")
        tenant_id = query_params.get("tenant_id")

        devices = await gateway.list_devices(
            user_id=user_id,
            device_type=device_type,
            status=status,
            tenant_id=tenant_id,
        )

        return json_response(
            {
                "devices": [
                    {
                        "id": d.id,
                        "name": d.config.name,
                        "device_type": d.config.device_type,
                        "status": d.status,
                        "user_id": d.user_id,
                        "last_seen": d.last_seen.isoformat() if d.last_seen else None,
                        "battery_level": d.battery_level,
                        "signal_strength": d.signal_strength,
                    }
                    for d in devices
                ],
                "total": len(devices),
            }
        )

    async def _handle_register_device(self, handler: Any) -> HandlerResult:
        """
        Register a new device.

        POST /api/v1/moltbot/devices
        Body: {name, device_type, capabilities?, metadata?}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        name = body.get("name")
        device_type = body.get("device_type")

        if not name or not device_type:
            return error_response("name and device_type are required", 400)

        from aragora.extensions.moltbot import DeviceNodeConfig

        config = DeviceNodeConfig(
            name=name,
            device_type=device_type,
            capabilities=body.get("capabilities", []),
            metadata=body.get("metadata", {}),
        )

        gateway = get_gateway()
        device = await gateway.register_device(
            config=config,
            user_id=user.user_id,
            tenant_id=body.get("tenant_id"),
        )

        return json_response(
            {
                "success": True,
                "device": {
                    "id": device.id,
                    "name": device.config.name,
                    "device_type": device.config.device_type,
                    "status": device.status,
                    "gateway_id": device.gateway_id,
                },
            },
            status=201,
        )

    async def _handle_get_device(self, device_id: str, handler: Any) -> HandlerResult:
        """
        Get a device by ID.

        GET /api/v1/moltbot/devices/{device_id}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        gateway = get_gateway()
        device = await gateway.get_device(device_id)

        if not device:
            return error_response("Device not found", 404)

        return json_response(
            {
                "device": {
                    "id": device.id,
                    "name": device.config.name,
                    "device_type": device.config.device_type,
                    "status": device.status,
                    "user_id": device.user_id,
                    "gateway_id": device.gateway_id,
                    "tenant_id": device.tenant_id,
                    "state": device.state,
                    "capabilities": device.config.capabilities,
                    "last_heartbeat": device.last_heartbeat.isoformat()
                    if device.last_heartbeat
                    else None,
                    "last_seen": device.last_seen.isoformat() if device.last_seen else None,
                    "battery_level": device.battery_level,
                    "signal_strength": device.signal_strength,
                    "firmware_version": device.firmware_version,
                    "messages_sent": device.messages_sent,
                    "messages_received": device.messages_received,
                    "errors": device.errors,
                    "created_at": device.created_at.isoformat() if device.created_at else None,
                    "updated_at": device.updated_at.isoformat() if device.updated_at else None,
                },
            }
        )

    async def _handle_unregister_device(self, device_id: str, handler: Any) -> HandlerResult:
        """
        Unregister a device.

        DELETE /api/v1/moltbot/devices/{device_id}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        gateway = get_gateway()
        success = await gateway.unregister_device(device_id)

        if not success:
            return error_response("Device not found", 404)

        return json_response(
            {
                "success": True,
                "message": f"Device {device_id} unregistered",
            }
        )

    async def _handle_send_command(self, device_id: str, handler: Any) -> HandlerResult:
        """
        Send a command to a device.

        POST /api/v1/moltbot/devices/{device_id}/command
        Body: {command, payload?, timeout?}
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        command = body.get("command")
        if not command:
            return error_response("command is required", 400)

        gateway = get_gateway()
        result = await gateway.send_command(
            device_id=device_id,
            command=command,
            payload=body.get("payload"),
            timeout=body.get("timeout", 30.0),
        )

        status = 200 if result.get("success") else 400
        return json_response(result, status=status)

    async def _handle_heartbeat(self, device_id: str, handler: Any) -> HandlerResult:
        """
        Process a device heartbeat.

        POST /api/v1/moltbot/devices/{device_id}/heartbeat
        Body: {state?, metrics?}
        """
        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        gateway = get_gateway()
        device = await gateway.heartbeat(
            device_id=device_id,
            state=body.get("state") if body else None,
            metrics=body.get("metrics") if body else None,
        )

        if not device:
            return error_response("Device not found", 404)

        return json_response(
            {
                "success": True,
                "device_id": device.id,
                "status": device.status,
                "last_heartbeat": device.last_heartbeat.isoformat()
                if device.last_heartbeat
                else None,
            }
        )

    async def _handle_gateway_stats(self, handler: Any) -> HandlerResult:
        """
        Get gateway statistics.

        GET /api/v1/moltbot/gateway/stats
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        gateway = get_gateway()
        stats = await gateway.get_stats()

        return json_response(stats)
