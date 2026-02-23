"""
HTTP API Handlers for OpenClaw Gateway.

Provides REST APIs for secure OpenClaw integration:
- Execute tasks via OpenClaw
- Manage devices
- Install/uninstall plugins
- Configure gateway

Endpoints:
- POST /api/v1/gateway/openclaw/execute - Execute a task
- GET /api/v1/gateway/openclaw/status/:task_id - Get task status
- POST /api/v1/gateway/openclaw/devices - Register device
- DELETE /api/v1/gateway/openclaw/devices/:device_id - Unregister device
- POST /api/v1/gateway/openclaw/plugins - Install plugin
- DELETE /api/v1/gateway/openclaw/plugins/:plugin_id - Uninstall plugin
- GET /api/v1/gateway/openclaw/config - Get gateway configuration
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aragora.gateway.openclaw import (
    CapabilityFilter,
    OpenClawGatewayAdapter,
    OpenClawProtocolTranslator,
    SandboxConfig,
)
from aragora.gateway.openclaw.adapter import (
    DeviceRegistration,
    PluginInstallRequest,
)
from aragora.gateway.openclaw.protocol import (
    AragoraRequest,
    AuthorizationContext,
    TenantContext,
)
from aragora.server.handlers.utils.decorators import require_permission
from aragora.server.handlers.utils.rate_limit import auth_rate_limit
from aragora.server.handlers.utils.responses import (
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

# Singleton gateway adapter (initialized on first use)
_gateway_adapter: OpenClawGatewayAdapter | None = None


def _get_gateway_adapter() -> OpenClawGatewayAdapter:
    """Get or create the gateway adapter singleton."""
    global _gateway_adapter
    if _gateway_adapter is None:
        # Configure from environment
        endpoint = os.environ.get("OPENCLAW_ENDPOINT", "http://localhost:8081")

        # Memory bounds: 64MB to 16GB
        _raw_memory = int(os.environ.get("OPENCLAW_MAX_MEMORY_MB", "512"))
        if _raw_memory < 64 or _raw_memory > 16384:
            logger.warning(
                "OPENCLAW_MAX_MEMORY_MB=%d out of bounds [64, 16384], clamping",
                _raw_memory,
            )
        max_memory = max(64, min(_raw_memory, 16384))

        # Execution time bounds: 1 second to 1 hour
        _raw_execution = int(os.environ.get("OPENCLAW_MAX_EXECUTION_SECONDS", "300"))
        if _raw_execution < 1 or _raw_execution > 3600:
            logger.warning(
                "OPENCLAW_MAX_EXECUTION_SECONDS=%d out of bounds [1, 3600], clamping",
                _raw_execution,
            )
        max_execution = max(1, min(_raw_execution, 3600))
        plugin_allowlist = os.environ.get("OPENCLAW_PLUGIN_ALLOWLIST", "").split(",")
        plugin_allowlist = [p.strip() for p in plugin_allowlist if p.strip()]

        sandbox_config = SandboxConfig(
            max_memory_mb=max_memory,
            max_execution_seconds=max_execution,
            allowed_plugins=plugin_allowlist,
            plugin_allowlist_mode=bool(plugin_allowlist),
        )

        _gateway_adapter = OpenClawGatewayAdapter(
            openclaw_endpoint=endpoint,
            sandbox_config=sandbox_config,
            capability_filter=CapabilityFilter(),
            protocol_translator=OpenClawProtocolTranslator(),
        )

    return _gateway_adapter


VALID_ACTOR_TYPES = {"user", "service", "agent"}


def _build_auth_context(user_id: str, data: dict[str, Any]) -> AuthorizationContext:
    """Build auth context from request data.

    Note: permissions and roles are NOT taken from request data for security.
    Authorization is handled by the @require_permission decorator at the endpoint level.
    """
    actor_type = data.get("actor_type", "user")
    if actor_type not in VALID_ACTOR_TYPES:
        actor_type = "user"  # Default to user for invalid types

    return AuthorizationContext(
        actor_id=user_id,
        actor_type=actor_type,
        permissions=set(),  # Auth handled by @require_permission decorator
        roles=[],  # Roles not passed from client
        session_id=data.get("session_id"),
    )


def _build_tenant_context(data: dict[str, Any]) -> TenantContext | None:
    """Build tenant context from request data if present."""
    tenant_id = data.get("tenant_id")
    if not tenant_id:
        return None

    return TenantContext(
        tenant_id=tenant_id,
        organization_id=data.get("organization_id"),
        workspace_id=data.get("workspace_id"),
        user_id=data.get("user_id"),
        enabled_capabilities=set(data.get("enabled_capabilities", [])),
        enabled_plugins=set(data.get("enabled_plugins", [])),
    )


# =============================================================================
# Task Execution
# =============================================================================


@require_permission("gateway.execute")
@auth_rate_limit(
    requests_per_minute=60,
    limiter_name="gateway_openclaw_execute",
    endpoint_name="OpenClaw execute",
)
@require_permission("debates:read")
async def handle_openclaw_execute(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Execute a task via OpenClaw gateway.

    POST /api/v1/gateway/openclaw/execute
    Body: {
        content: str - Task content/prompt
        request_type: str (optional) - Task type (task, query, action)
        capabilities: list[str] (optional) - Required capabilities
        plugins: list[str] (optional) - Plugins to use
        priority: str (optional) - Priority (low, normal, high)
        timeout_seconds: int (optional) - Task timeout
        context: dict (optional) - Additional context
    }
    """
    try:
        content = data.get("content")
        if not content:
            return error_response("content is required", status=400)

        # Validate and clamp timeout_seconds
        adapter = _get_gateway_adapter()
        max_execution = adapter.sandbox_config.max_execution_seconds
        timeout = data.get("timeout_seconds", 300)
        if not isinstance(timeout, int) or timeout < 1:
            timeout = 300
        timeout = min(timeout, max_execution)  # Cap at configured max

        # Build request
        request = AragoraRequest(
            content=content,
            request_type=data.get("request_type", "task"),
            capabilities=data.get("capabilities", []),
            plugins=data.get("plugins", []),
            priority=data.get("priority", "normal"),
            timeout_seconds=timeout,
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
        )

        # Build contexts
        auth_context = _build_auth_context(user_id, data)
        tenant_context = _build_tenant_context(data)

        # Execute via gateway (adapter already retrieved above for timeout validation)
        result = await adapter.execute_task(
            request=request,
            auth_context=auth_context,
            tenant_context=tenant_context,
        )

        if not result.success:
            return error_response(
                result.error or "Task execution failed",
                status=400 if result.blocked_reason else 500,
            )

        return success_response(
            {
                "request_id": result.request_id,
                "status": result.response.status if result.response else "completed",
                "result": result.response.result if result.response else None,
                "execution_time_ms": result.metadata.get("execution_time_ms", 0),
            }
        )

    except (ConnectionError, TimeoutError, OSError, ValueError, TypeError, KeyError):
        logger.exception("OpenClaw execute failed")
        return error_response("Execution failed", status=500)


@require_permission("gateway.read")
@auth_rate_limit(
    requests_per_minute=120,
    limiter_name="gateway_openclaw_status",
    endpoint_name="OpenClaw status",
)
@require_permission("debates:read")
async def handle_openclaw_status(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get task status.

    GET /api/v1/gateway/openclaw/status/:task_id
    """
    try:
        task_id = data.get("task_id")
        if not task_id:
            return error_response("task_id is required", status=400)

        # For now, return a basic status (would need persistent storage for real impl)
        return success_response(
            {
                "task_id": task_id,
                "status": "unknown",  # Would query actual status from storage
                "message": "Task status lookup requires persistent storage",
            }
        )

    except (KeyError, ValueError, AttributeError, TypeError):
        logger.exception("OpenClaw status check failed")
        return error_response("Status check failed", status=500)


# =============================================================================
# Device Management
# =============================================================================


@require_permission("gateway.create")
@auth_rate_limit(
    requests_per_minute=20,
    limiter_name="gateway_openclaw_device_register",
    endpoint_name="OpenClaw device register",
)
@require_permission("debates:read")
async def handle_openclaw_device_register(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Register a device with the gateway.

    POST /api/v1/gateway/openclaw/devices
    Body: {
        device_id: str
        device_name: str
        device_type: str (desktop, mobile, server, iot)
        capabilities: list[str] (optional)
    }
    """
    try:
        device_id = data.get("device_id")
        device_name = data.get("device_name")
        device_type = data.get("device_type")

        if not device_id or not device_name or not device_type:
            return error_response(
                "device_id, device_name, and device_type are required",
                status=400,
            )

        device = DeviceRegistration(
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {}),
        )

        auth_context = _build_auth_context(user_id, data)

        adapter = _get_gateway_adapter()
        result = await adapter.register_device(device, auth_context)

        if not result.success:
            return error_response(result.error or "Device registration failed", status=400)

        return success_response(
            {
                "device_id": device_id,
                "registration_id": result.metadata.get("device_handle", {}).get("registration_id"),
                "registered_at": result.metadata.get("device_handle", {}).get("registered_at"),
            }
        )

    except (ConnectionError, TimeoutError, OSError, ValueError, TypeError):
        logger.exception("Device registration failed")
        return error_response("Registration failed", status=500)


@require_permission("gateway.delete")
@auth_rate_limit(
    requests_per_minute=20,
    limiter_name="gateway_openclaw_device_unregister",
    endpoint_name="OpenClaw device unregister",
)
@require_permission("debates:read")
async def handle_openclaw_device_unregister(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Unregister a device from the gateway.

    DELETE /api/v1/gateway/openclaw/devices/:device_id
    """
    try:
        device_id = data.get("device_id")
        if not device_id:
            return error_response("device_id is required", status=400)

        auth_context = _build_auth_context(user_id, data)

        adapter = _get_gateway_adapter()
        result = await adapter.unregister_device(device_id, auth_context)

        if not result.success:
            return error_response(result.error or "Device unregistration failed", status=400)

        return success_response(
            {
                "device_id": device_id,
                "unregistered": True,
            }
        )

    except (ConnectionError, TimeoutError, OSError, ValueError, TypeError):
        logger.exception("Device unregistration failed")
        return error_response("Unregistration failed", status=500)


# =============================================================================
# Plugin Management
# =============================================================================


@require_permission("gateway.install")
@auth_rate_limit(
    requests_per_minute=10,
    limiter_name="gateway_openclaw_plugin_install",
    endpoint_name="OpenClaw plugin install",
)
@require_permission("debates:read")
async def handle_openclaw_plugin_install(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Install a plugin via the gateway.

    POST /api/v1/gateway/openclaw/plugins
    Body: {
        plugin_id: str
        plugin_name: str
        version: str
        source: str (marketplace, local, url)
    }
    """
    try:
        plugin_id = data.get("plugin_id")
        plugin_name = data.get("plugin_name")
        version = data.get("version")
        source = data.get("source", "marketplace")

        if not plugin_id or not plugin_name or not version:
            return error_response(
                "plugin_id, plugin_name, and version are required",
                status=400,
            )

        plugin = PluginInstallRequest(
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            version=version,
            source=source,
            metadata=data.get("metadata", {}),
        )

        auth_context = _build_auth_context(user_id, data)
        tenant_context = _build_tenant_context(data)

        adapter = _get_gateway_adapter()
        result = await adapter.install_plugin(plugin, auth_context, tenant_context)

        if not result.success:
            return error_response(result.error or "Plugin installation failed", status=400)

        return success_response(
            {
                "plugin_id": plugin_id,
                "installed": True,
                "installed_at": result.metadata.get("installed_at"),
            }
        )

    except (ConnectionError, TimeoutError, OSError, ValueError, TypeError):
        logger.exception("Plugin installation failed")
        return error_response("Installation failed", status=500)


@require_permission("gateway.uninstall")
@auth_rate_limit(
    requests_per_minute=10,
    limiter_name="gateway_openclaw_plugin_uninstall",
    endpoint_name="OpenClaw plugin uninstall",
)
@require_permission("debates:read")
async def handle_openclaw_plugin_uninstall(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Uninstall a plugin from the gateway.

    DELETE /api/v1/gateway/openclaw/plugins/:plugin_id
    """
    try:
        plugin_id = data.get("plugin_id")
        if not plugin_id:
            return error_response("plugin_id is required", status=400)

        auth_context = _build_auth_context(user_id, data)

        adapter = _get_gateway_adapter()
        result = await adapter.uninstall_plugin(plugin_id, auth_context)

        if not result.success:
            return error_response(result.error or "Plugin uninstallation failed", status=400)

        return success_response(
            {
                "plugin_id": plugin_id,
                "uninstalled": True,
            }
        )

    except (ConnectionError, TimeoutError, OSError, ValueError, TypeError):
        logger.exception("Plugin uninstallation failed")
        return error_response("Uninstallation failed", status=500)


# =============================================================================
# Configuration
# =============================================================================


@require_permission("gateway.read")
@auth_rate_limit(
    requests_per_minute=30,
    limiter_name="gateway_openclaw_config",
    endpoint_name="OpenClaw config",
)
async def handle_openclaw_config(
    data: dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get gateway configuration.

    GET /api/v1/gateway/openclaw/config
    """
    try:
        adapter = _get_gateway_adapter()

        # Return public configuration (no secrets)
        config = {
            "endpoint": adapter.openclaw_endpoint,
            "sandbox": {
                "max_memory_mb": adapter.sandbox_config.max_memory_mb,
                "max_cpu_percent": adapter.sandbox_config.max_cpu_percent,
                "max_execution_seconds": adapter.sandbox_config.max_execution_seconds,
                "allow_external_network": adapter.sandbox_config.allow_external_network,
                "plugin_allowlist_mode": adapter.sandbox_config.plugin_allowlist_mode,
                "allowed_plugins_count": len(adapter.sandbox_config.allowed_plugins),
            },
            "capabilities": {
                "blocked_override_count": len(adapter.capability_filter.blocked_override),
                "tenant_enabled_count": len(adapter.capability_filter.tenant_enabled),
            },
        }

        return success_response(config)

    except (KeyError, ValueError, AttributeError, TypeError):
        logger.exception("Failed to get config")
        return error_response("Failed to retrieve configuration", status=500)


# =============================================================================
# Handler Registration
# =============================================================================


def get_openclaw_handlers() -> dict[str, Any]:
    """Get all OpenClaw handlers for registration."""
    return {
        "openclaw_execute": handle_openclaw_execute,
        "openclaw_status": handle_openclaw_status,
        "openclaw_device_register": handle_openclaw_device_register,
        "openclaw_device_unregister": handle_openclaw_device_unregister,
        "openclaw_plugin_install": handle_openclaw_plugin_install,
        "openclaw_plugin_uninstall": handle_openclaw_plugin_uninstall,
        "openclaw_config": handle_openclaw_config,
    }


__all__ = [
    "handle_openclaw_execute",
    "handle_openclaw_status",
    "handle_openclaw_device_register",
    "handle_openclaw_device_unregister",
    "handle_openclaw_plugin_install",
    "handle_openclaw_plugin_uninstall",
    "handle_openclaw_config",
    "get_openclaw_handlers",
]
