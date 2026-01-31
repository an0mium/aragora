"""
Gateway HTTP Handlers.

Provides REST API endpoints for external AI runtime gateways (OpenClaw, etc.).
"""

from __future__ import annotations

from .openclaw import (
    handle_openclaw_execute,
    handle_openclaw_status,
    handle_openclaw_device_register,
    handle_openclaw_device_unregister,
    handle_openclaw_plugin_install,
    handle_openclaw_plugin_uninstall,
    handle_openclaw_config,
    get_openclaw_handlers,
)

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
