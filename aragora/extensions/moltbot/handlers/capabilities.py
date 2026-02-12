"""
Moltbot Capabilities Handler - Device Capability Management REST API.

Endpoints:
- GET  /api/v1/moltbot/capabilities                    - List all capabilities
- GET  /api/v1/moltbot/capabilities/{name}             - Get capability details
- GET  /api/v1/moltbot/devices/{id}/capabilities       - Get device capabilities
- POST /api/v1/moltbot/devices/{id}/capabilities/check - Check device capability
- GET  /api/v1/moltbot/capabilities/matrix             - Get capability matrix
- GET  /api/v1/moltbot/capabilities/categories         - List categories
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, cast

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)

from .types import serialize_enum

if TYPE_CHECKING:
    from aragora.extensions.moltbot.protocols import CapabilityMatcherProtocol

logger = logging.getLogger(__name__)

# Global capability matcher instance
_matcher: CapabilityMatcherProtocol | None = None


def get_capability_matcher() -> CapabilityMatcherProtocol:
    """Get or create the capability matcher instance."""
    global _matcher
    if _matcher is None:
        from aragora.extensions.moltbot.capabilities import CapabilityMatcher

        _matcher = cast("CapabilityMatcherProtocol", CapabilityMatcher())
    return _matcher


class MoltbotCapabilitiesHandler(BaseHandler):
    """HTTP handler for Moltbot device capabilities."""

    routes = [
        ("GET", "/api/v1/moltbot/capabilities"),
        ("GET", "/api/v1/moltbot/capabilities/matrix"),
        ("GET", "/api/v1/moltbot/capabilities/categories"),
        ("GET", "/api/v1/moltbot/capabilities/categories/"),
        ("GET", "/api/v1/moltbot/capabilities/"),
        ("GET", "/api/v1/moltbot/devices/*/capabilities"),
        ("POST", "/api/v1/moltbot/devices/*/capabilities/check"),
    ]

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle GET requests."""
        if path == "/api/v1/moltbot/capabilities":
            return await self._handle_list_capabilities(query_params, handler)
        elif path == "/api/v1/moltbot/capabilities/matrix":
            return await self._handle_capability_matrix(query_params, handler)
        elif path == "/api/v1/moltbot/capabilities/categories":
            return await self._handle_list_categories(handler)
        elif path.startswith("/api/v1/moltbot/capabilities/categories/"):
            parts = path.split("/")
            if len(parts) >= 6:
                category = parts[5]
                return await self._handle_category_capabilities(category, handler)
        elif path.startswith("/api/v1/moltbot/capabilities/"):
            parts = path.split("/")
            if len(parts) >= 5:
                capability_name = parts[4]
                if capability_name not in ("matrix", "categories"):
                    return await self._handle_get_capability(capability_name, handler)
        elif path.startswith("/api/v1/moltbot/devices/"):
            parts = path.split("/")
            # /api/v1/moltbot/devices/{id}/capabilities
            if len(parts) >= 6 and parts[5] == "capabilities":
                device_id = parts[4]
                return await self._handle_device_capabilities(device_id, handler)
        return None

    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        if path.startswith("/api/v1/moltbot/devices/"):
            parts = path.split("/")
            # /api/v1/moltbot/devices/{id}/capabilities/check
            if len(parts) >= 7 and parts[5] == "capabilities" and parts[6] == "check":
                device_id = parts[4]
                return await self._handle_check_capability(device_id, handler)
        return None

    # ========== Handler Methods ==========

    def _serialize_capability(self, cap: Any) -> dict[str, Any]:
        """Serialize capability to JSON-safe dict."""
        return {
            "name": getattr(cap, "name", str(cap)),
            "category": serialize_enum(getattr(cap, "category", None), "unknown"),
            "description": getattr(cap, "description", ""),
            "version": getattr(cap, "version", "1.0"),
            "is_required": getattr(cap, "is_required", False),
            "metadata": getattr(cap, "metadata", {}),
        }

    def _serialize_device_capabilities(self, device_caps: Any) -> dict[str, Any]:
        """Serialize device capabilities to JSON-safe dict."""
        return {
            "device_id": device_caps.device_id if hasattr(device_caps, "device_id") else "",
            "device_type": getattr(device_caps, "device_type", "unknown"),
            "capabilities": device_caps.capabilities
            if hasattr(device_caps, "capabilities")
            else [],
            "display": self._serialize_capability_group(getattr(device_caps, "display", None)),
            "audio": self._serialize_capability_group(getattr(device_caps, "audio", None)),
            "video": self._serialize_capability_group(getattr(device_caps, "video", None)),
            "input": self._serialize_capability_group(getattr(device_caps, "input", None)),
            "network": self._serialize_capability_group(getattr(device_caps, "network", None)),
            "compute": self._serialize_capability_group(getattr(device_caps, "compute", None)),
            "sensor": self._serialize_capability_group(getattr(device_caps, "sensor", None)),
            "actuator": self._serialize_capability_group(getattr(device_caps, "actuator", None)),
        }

    def _serialize_capability_group(self, group: Any) -> dict[str, Any] | None:
        """Serialize a capability group (display, audio, etc.)."""
        if group is None:
            return None
        # Convert dataclass to dict if possible
        if hasattr(group, "__dict__"):
            return {k: v for k, v in group.__dict__.items() if not k.startswith("_")}
        return {}

    async def _handle_list_capabilities(
        self, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """List all available capabilities."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        category = query_params.get("category")

        matcher = get_capability_matcher()
        capabilities = await matcher.list_capabilities(category=category)

        return json_response(
            {
                "capabilities": [self._serialize_capability(c) for c in capabilities],
                "total": len(capabilities),
            }
        )

    async def _handle_get_capability(self, capability_name: str, handler: Any) -> HandlerResult:
        """Get capability details."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        matcher = get_capability_matcher()
        capability = await matcher.get_capability(capability_name)

        if not capability:
            return error_response("Capability not found", 404)

        result = self._serialize_capability(capability)
        # Include additional metadata
        result["dependents"] = await matcher.get_dependents(capability_name)

        return json_response({"capability": result})

    async def _handle_device_capabilities(self, device_id: str, handler: Any) -> HandlerResult:
        """Get capabilities for a specific device."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        matcher = get_capability_matcher()
        caps = await matcher.get_device_capabilities(device_id)

        if not caps:
            return error_response("Device not found or no capabilities detected", 404)

        return json_response({"device_capabilities": self._serialize_device_capabilities(caps)})

    async def _handle_check_capability(self, device_id: str, handler: Any) -> HandlerResult:
        """Check if device has specific capability."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        body, err = self.read_json_body_validated(handler)
        if err:
            return err

        if not body:
            return error_response("Request body required", 400)

        capability_name = body.get("capability")
        if not capability_name:
            return error_response("capability is required", 400)

        matcher = get_capability_matcher()
        result = await matcher.check_capability(device_id, capability_name)

        return json_response(
            {
                "device_id": device_id,
                "capability": capability_name,
                "supported": result.get("supported", False),
                "details": result.get("details"),
            }
        )

    async def _handle_capability_matrix(
        self, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Get capability matrix for devices."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        tenant_id = query_params.get("tenant_id")

        matcher = get_capability_matcher()
        matrix = await matcher.get_capability_matrix(tenant_id=tenant_id)

        return json_response(
            {
                "matrix": matrix.get("matrix", {}),
                "devices": matrix.get("devices", []),
                "capabilities": matrix.get("capabilities", []),
            }
        )

    async def _handle_list_categories(self, handler: Any) -> HandlerResult:
        """List all capability categories."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        from aragora.extensions.moltbot.capabilities import CapabilityCategory

        categories = [
            {
                "name": cat.value,
                "description": self._get_category_description(cat),
            }
            for cat in CapabilityCategory
        ]

        return json_response(
            {
                "categories": categories,
                "total": len(categories),
            }
        )

    async def _handle_category_capabilities(self, category: str, handler: Any) -> HandlerResult:
        """Get all capabilities in a category."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        matcher = get_capability_matcher()
        capabilities = await matcher.list_capabilities(category=category)

        return json_response(
            {
                "category": category,
                "capabilities": [self._serialize_capability(c) for c in capabilities],
                "total": len(capabilities),
            }
        )

    def _get_category_description(self, category: Any) -> str:
        """Get description for capability category."""
        descriptions = {
            "display": "Visual display and rendering capabilities",
            "input": "User input methods (keyboard, touch, voice)",
            "audio": "Audio playback and recording",
            "video": "Video playback and recording",
            "network": "Network connectivity and protocols",
            "storage": "Data storage and persistence",
            "sensor": "Environmental and motion sensors",
            "compute": "Processing and computation power",
            "actuator": "Physical actuators and motors",
        }
        cat_value = serialize_enum(category)
        return descriptions.get(cat_value.lower(), "Unknown category")
