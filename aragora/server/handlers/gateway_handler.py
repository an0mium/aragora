"""
Gateway Handler - HTTP endpoints for device gateway management.

Provides API endpoints for:
- Device registration and management
- Channel listing and configuration
- Routing statistics and rules
- Message routing

Routes:
    GET    /api/v1/gateway/devices          - List registered devices
    POST   /api/v1/gateway/devices          - Register a device
    GET    /api/v1/gateway/devices/{id}     - Get device details
    DELETE /api/v1/gateway/devices/{id}     - Unregister a device
    POST   /api/v1/gateway/devices/{id}/heartbeat - Device heartbeat
    GET    /api/v1/gateway/channels         - List active channels
    GET    /api/v1/gateway/routing/stats    - Routing statistics
    GET    /api/v1/gateway/routing/rules    - List routing rules
    POST   /api/v1/gateway/messages/route   - Route a message
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.http_utils import run_async

# RBAC imports
try:
    from aragora.rbac import AuthorizationContext, check_permission
    from aragora.billing.jwt_auth import extract_user_from_request

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False
    AuthorizationContext = None  # type: ignore[misc]

# Gateway imports
try:
    from aragora.gateway import (
        DeviceRegistry,
        DeviceNode,
        DeviceStatus,
        AgentRouter,
    )

    GATEWAY_AVAILABLE = True
except ImportError:
    GATEWAY_AVAILABLE = False
    DeviceRegistry = None  # type: ignore[misc]

logger = logging.getLogger(__name__)

class GatewayHandler(BaseHandler):
    """
    HTTP request handler for gateway API endpoints.

    Provides REST API for managing devices, channels, and message routing
    through the local gateway.
    """

    ROUTES = [
        "/api/v1/gateway/devices",
        "/api/v1/gateway/devices/*",
        "/api/v1/gateway/channels",
        "/api/v1/gateway/routing",
        "/api/v1/gateway/routing/*",
        "/api/v1/gateway/messages",
        "/api/v1/gateway/messages/*",
    ]

    def __init__(self, server_context):
        super().__init__(server_context)
        self._device_registry: DeviceRegistry | None = None
        self._agent_router: AgentRouter | None = None

    def _get_device_registry(self) -> DeviceRegistry | None:
        """Get or create device registry."""
        if not GATEWAY_AVAILABLE:
            return None
        if self._device_registry is None:
            self._device_registry = DeviceRegistry()
        return self._device_registry

    def _get_agent_router(self) -> AgentRouter | None:
        """Get or create agent router."""
        if not GATEWAY_AVAILABLE:
            return None
        if self._agent_router is None:
            self._agent_router = AgentRouter()
        return self._agent_router

    def _get_user_store(self) -> Any:
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _get_auth_context(self, handler) -> AuthorizationContext | None:
        """Build AuthorizationContext from request."""
        if not RBAC_AVAILABLE or AuthorizationContext is None:
            return None

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)

        if not auth_ctx.is_authenticated:
            return None

        user = user_store.get_user_by_id(auth_ctx.user_id) if user_store else None
        roles = set([user.role]) if user and user.role else set()

        return AuthorizationContext(
            user_id=auth_ctx.user_id,
            roles=roles,
            org_id=auth_ctx.org_id,
        )

    def _check_rbac_permission(self, handler, permission_key: str) -> HandlerResult | None:
        """Check RBAC permission. Returns None if allowed, error response if denied."""
        if not RBAC_AVAILABLE:
            return None

        rbac_ctx = self._get_auth_context(handler)
        if not rbac_ctx:
            return error_response("Not authenticated", 401)

        decision = check_permission(rbac_ctx, permission_key)
        if not decision.allowed:
            logger.warning(f"RBAC denied: user={rbac_ctx.user_id} permission={permission_key}")
            return error_response(f"Permission denied: {decision.reason}", 403)

        return None

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/gateway/")

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle GET requests."""
        if not self.can_handle(path):
            return None

        if not GATEWAY_AVAILABLE:
            return error_response("Gateway module not available", 503)

        # GET /api/v1/gateway/devices
        if path == "/api/v1/gateway/devices":
            return self._handle_list_devices(query_params, handler)

        # GET /api/v1/gateway/devices/{id}
        if path.startswith("/api/v1/gateway/devices/"):
            device_id = path.split("/")[-1]
            if device_id and device_id != "devices":
                return self._handle_get_device(device_id, handler)

        # GET /api/v1/gateway/channels
        if path == "/api/v1/gateway/channels":
            return self._handle_list_channels(query_params, handler)

        # GET /api/v1/gateway/routing/stats
        if path == "/api/v1/gateway/routing/stats":
            return self._handle_routing_stats(handler)

        # GET /api/v1/gateway/routing/rules
        if path == "/api/v1/gateway/routing/rules":
            return self._handle_list_rules(query_params, handler)

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        if not self.can_handle(path):
            return None

        if not GATEWAY_AVAILABLE:
            return error_response("Gateway module not available", 503)

        # POST /api/v1/gateway/devices
        if path == "/api/v1/gateway/devices":
            return self._handle_register_device(handler)

        # POST /api/v1/gateway/devices/{id}/heartbeat
        if "/heartbeat" in path:
            parts = path.strip("/").split("/")
            # parts = ["api", "v1", "gateway", "devices", device_id, "heartbeat"]
            if len(parts) >= 6 and parts[5] == "heartbeat":
                device_id = parts[4]
                return self._handle_heartbeat(device_id, handler)

        # POST /api/v1/gateway/messages/route
        if path == "/api/v1/gateway/messages/route":
            return self._handle_route_message(handler)

        return None

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle DELETE requests."""
        if not self.can_handle(path):
            return None

        if not GATEWAY_AVAILABLE:
            return error_response("Gateway module not available", 503)

        # DELETE /api/v1/gateway/devices/{id}
        if path.startswith("/api/v1/gateway/devices/"):
            device_id = path.split("/")[-1]
            if device_id and device_id != "devices":
                return self._handle_unregister_device(device_id, handler)

        return None

    # =========================================================================
    # Device Handlers
    # =========================================================================

    @handle_errors("list devices")
    def _handle_list_devices(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/gateway/devices."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "gateway:devices:read"):
            return error

        registry = self._get_device_registry()
        if not registry:
            return error_response("Device registry not available", 503)

        # Parse filters
        status_str = query_params.get("status")
        device_type = query_params.get("type")

        status = None
        if status_str:
            try:
                status = DeviceStatus(status_str)
            except ValueError:
                pass

        devices = run_async(registry.list_devices(status=status, device_type=device_type))

        return json_response(
            {
                "devices": [
                    {
                        "device_id": d.device_id,
                        "name": d.name,
                        "device_type": d.device_type,
                        "capabilities": d.capabilities,
                        "status": d.status.value,
                        "paired_at": d.paired_at,
                        "last_seen": d.last_seen,
                    }
                    for d in devices
                ],
                "total": len(devices),
            }
        )

    @handle_errors("get device")
    def _handle_get_device(self, device_id: str, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/gateway/devices/{id}."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "gateway:devices:read"):
            return error

        registry = self._get_device_registry()
        if not registry:
            return error_response("Device registry not available", 503)

        device = run_async(registry.get(device_id))
        if not device:
            return error_response(f"Device not found: {device_id}", 404)

        return json_response(
            {
                "device": {
                    "device_id": device.device_id,
                    "name": device.name,
                    "device_type": device.device_type,
                    "capabilities": device.capabilities,
                    "status": device.status.value,
                    "paired_at": device.paired_at,
                    "last_seen": device.last_seen,
                    "allowed_channels": device.allowed_channels,
                    "metadata": device.metadata,
                }
            }
        )

    @rate_limit(rpm=30, limiter_name="gateway_register")
    @handle_errors("register device")
    @log_request("register device")
    def _handle_register_device(self, handler: Any) -> HandlerResult:
        """Handle POST /api/v1/gateway/devices."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "gateway:devices:create"):
            return error

        registry = self._get_device_registry()
        if not registry:
            return error_response("Device registry not available", 503)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        name = body.get("name")
        if not name:
            return error_response("name is required", 400)

        device = DeviceNode(
            device_id=body.get("device_id", ""),
            name=name,
            device_type=body.get("device_type", "unknown"),
            capabilities=body.get("capabilities", []),
            allowed_channels=body.get("allowed_channels", []),
            metadata=body.get("metadata", {}),
        )

        device_id = run_async(registry.register(device))

        logger.info(f"Registered device: {device_id} ({name})")

        return json_response(
            {
                "device_id": device_id,
                "message": "Device registered successfully",
            },
            status=201,
        )

    @rate_limit(rpm=10, limiter_name="gateway_unregister")
    @handle_errors("unregister device")
    @log_request("unregister device")
    def _handle_unregister_device(self, device_id: str, handler: Any) -> HandlerResult:
        """Handle DELETE /api/v1/gateway/devices/{id}."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "gateway:devices:delete"):
            return error

        registry = self._get_device_registry()
        if not registry:
            return error_response("Device registry not available", 503)

        success = run_async(registry.unregister(device_id))
        if not success:
            return error_response(f"Device not found: {device_id}", 404)

        logger.info(f"Unregistered device: {device_id}")

        return json_response({"message": "Device unregistered successfully"})

    @handle_errors("device heartbeat")
    def _handle_heartbeat(self, device_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/v1/gateway/devices/{id}/heartbeat."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "gateway:devices:read"):
            return error

        registry = self._get_device_registry()
        if not registry:
            return error_response("Device registry not available", 503)

        success = run_async(registry.heartbeat(device_id))
        if not success:
            return error_response(f"Device not found: {device_id}", 404)

        return json_response({"status": "ok"})

    # =========================================================================
    # Channel Handlers
    # =========================================================================

    @handle_errors("list channels")
    def _handle_list_channels(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/gateway/channels."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "gateway:channels:read"):
            return error

        # Return configured channels from context or defaults
        channels = [
            {"name": "slack", "status": "available"},
            {"name": "email", "status": "available"},
            {"name": "telegram", "status": "available"},
            {"name": "whatsapp", "status": "available"},
        ]

        return json_response(
            {
                "channels": channels,
                "total": len(channels),
            }
        )

    # =========================================================================
    # Routing Handlers
    # =========================================================================

    @handle_errors("routing stats")
    def _handle_routing_stats(self, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/gateway/routing/stats."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "gateway:routing:read"):
            return error

        router = self._get_agent_router()
        if not router:
            return error_response("Agent router not available", 503)

        # Return basic stats
        return json_response(
            {
                "stats": {
                    "total_rules": 0,
                    "messages_routed": 0,
                    "routing_errors": 0,
                }
            }
        )

    @handle_errors("list rules")
    def _handle_list_rules(self, query_params: dict, handler: Any) -> HandlerResult:
        """Handle GET /api/v1/gateway/routing/rules."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "gateway:routing:read"):
            return error

        router = self._get_agent_router()
        if not router:
            return error_response("Agent router not available", 503)

        rules: list = router.list_rules() if hasattr(router, "list_rules") else []  # type: ignore[assignment]

        return json_response(
            {
                "rules": [
                    {
                        "id": getattr(r, "id", str(i)),
                        "channel": getattr(r, "channel", ""),
                        "pattern": getattr(r, "pattern", ""),
                        "agent_id": getattr(r, "agent_id", ""),
                    }
                    for i, r in enumerate(rules)
                ],
                "total": len(rules),
            }
        )

    # =========================================================================
    # Message Routing
    # =========================================================================

    @rate_limit(rpm=60, limiter_name="gateway_route")
    @handle_errors("route message")
    def _handle_route_message(self, handler: Any) -> HandlerResult:
        """Handle POST /api/v1/gateway/messages/route."""
        # RBAC check
        if error := self._check_rbac_permission(handler, "gateway:messages:route"):
            return error

        router = self._get_agent_router()
        if not router:
            return error_response("Agent router not available", 503)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        channel = body.get("channel")
        content = body.get("content")

        if not channel:
            return error_response("channel is required", 400)
        if not content:
            return error_response("content is required", 400)

        # Route the message
        result = run_async(router.route(channel=channel, content=content))  # type: ignore[call-arg]

        return json_response(
            {
                "routed": True,
                "agent_id": getattr(result, "agent_id", None),
                "rule_id": getattr(result, "rule_id", None),
            }
        )

__all__ = ["GatewayHandler"]
