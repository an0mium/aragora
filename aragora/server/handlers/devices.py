"""
Device Registration and Notification API Handlers.

Provides REST APIs for device push notification management:
- Device registration and unregistration
- Push notification delivery
- Device health monitoring
- Voice assistant webhooks (Alexa, Google Home)

Endpoints:
- POST /api/v1/devices/register - Register a device for push notifications
- DELETE /api/v1/devices/{device_id} - Unregister a device
- POST /api/v1/devices/{device_id}/notify - Send notification to a device
- POST /api/v1/devices/user/{user_id}/notify - Send to all user devices
- GET /api/v1/devices/user/{user_id} - List user's devices
- GET /api/v1/devices/health - Get device connector health
- POST /api/v1/devices/alexa/webhook - Alexa skill webhook
- POST /api/v1/devices/google/webhook - Google Actions webhook
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.secure import ForbiddenError, SecureHandler, UnauthorizedError
from aragora.server.versioning.compat import strip_version_prefix

logger = logging.getLogger(__name__)


class DeviceHandler(SecureHandler):
    """Handler for device registration and notification endpoints."""

    RESOURCE_TYPE = "devices"

    ROUTES = [
        "/api/v1/devices/register",
        "/api/v1/devices/health",
        "/api/v1/devices/alexa/webhook",
        "/api/v1/devices/google/webhook",
    ]

    # Pattern routes (handled via prefix matching)
    PATTERN_PREFIXES = [
        "/api/v1/devices/user/",
        "/api/v1/devices/",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the given path."""
        path = strip_version_prefix(path)

        # Exact matches
        if path in self.ROUTES:
            return True

        # Pattern matches
        for prefix in self.PATTERN_PREFIXES:
            if path.startswith(prefix):
                return True

        return False

    async def handle(
        self,
        path: str,
        query_params: Dict[str, Any],
        handler: Any,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
    ) -> Optional[HandlerResult]:
        """Route device requests."""
        # Require authentication for all endpoints
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        path = strip_version_prefix(path)

        # Health endpoint (read permission)
        if path == "/api/v1/devices/health":
            try:
                self.check_permission(auth_context, "devices:read")
            except ForbiddenError:
                return error_response("Permission denied: devices:read", 403)
            return await self._get_health()

        # Register device
        if path == "/api/v1/devices/register" and method == "POST":
            try:
                self.check_permission(auth_context, "devices:write")
            except ForbiddenError:
                return error_response("Permission denied: devices:write", 403)
            return await self._register_device(body or {}, auth_context)

        # Alexa webhook (no auth required - uses Alexa signature verification)
        if path == "/api/v1/devices/alexa/webhook" and method == "POST":
            return await self._handle_alexa_webhook(body or {}, handler)

        # Google webhook (no auth required - uses Google verification)
        if path == "/api/v1/devices/google/webhook" and method == "POST":
            return await self._handle_google_webhook(body or {}, handler)

        # User-specific endpoints
        if path.startswith("/api/v1/devices/user/"):
            parts = path.split("/")
            if len(parts) >= 6:
                user_id = parts[5]

                # List user devices
                if len(parts) == 6 and method == "GET":
                    try:
                        self.check_permission(auth_context, "devices:read")
                    except ForbiddenError:
                        return error_response("Permission denied: devices:read", 403)
                    return await self._list_user_devices(user_id, auth_context)

                # Notify all user devices
                if len(parts) == 7 and parts[6] == "notify" and method == "POST":
                    try:
                        self.check_permission(auth_context, "devices:notify")
                    except ForbiddenError:
                        return error_response("Permission denied: devices:notify", 403)
                    return await self._notify_user(user_id, body or {}, auth_context)

        # Device-specific endpoints
        if path.startswith("/api/v1/devices/") and not path.startswith("/api/v1/devices/user/"):
            parts = path.split("/")
            if len(parts) >= 5:
                device_id = parts[4]

                # Unregister device
                if len(parts) == 5 and method == "DELETE":
                    try:
                        self.check_permission(auth_context, "devices:write")
                    except ForbiddenError:
                        return error_response("Permission denied: devices:write", 403)
                    return await self._unregister_device(device_id, auth_context)

                # Notify single device
                if len(parts) == 6 and parts[5] == "notify" and method == "POST":
                    try:
                        self.check_permission(auth_context, "devices:notify")
                    except ForbiddenError:
                        return error_response("Permission denied: devices:notify", 403)
                    return await self._notify_device(device_id, body or {}, auth_context)

                # Get device info
                if len(parts) == 5 and method == "GET":
                    try:
                        self.check_permission(auth_context, "devices:read")
                    except ForbiddenError:
                        return error_response("Permission denied: devices:read", 403)
                    return await self._get_device(device_id, auth_context)

        return None

    async def _get_health(self) -> HandlerResult:
        """Get device connector health status."""
        try:
            from aragora.connectors.devices.registry import get_registry

            registry = get_registry()
            health = await registry.get_health()

            return json_response(health)

        except ImportError:
            return json_response(
                {
                    "status": "unavailable",
                    "error": "Device connectors not available",
                }
            )
        except Exception as e:
            logger.error(f"Error getting device health: {e}")
            return error_response(f"Error getting health: {e}", 500)

    async def _register_device(
        self,
        body: Dict[str, Any],
        auth_context: Dict[str, Any],
    ) -> HandlerResult:
        """Register a device for push notifications."""
        # Validate required fields
        required = ["device_type", "push_token"]
        missing = [f for f in required if not body.get(f)]
        if missing:
            return error_response(f"Missing required fields: {missing}", 400)

        device_type = body.get("device_type")
        push_token = body.get("push_token")

        # Get user_id from auth context or body
        user_id = body.get("user_id") or auth_context.get("user_id")
        if not user_id:
            return error_response("user_id is required", 400)

        try:
            from aragora.connectors.devices import (
                DeviceRegistration,
                DeviceType,
                get_registry,
            )

            # Parse device type
            try:
                dt = DeviceType(device_type)
            except ValueError:
                valid_types = [t.value for t in DeviceType]
                return error_response(f"Invalid device_type. Valid types: {valid_types}", 400)

            # Create registration
            registration = DeviceRegistration(
                user_id=user_id,
                device_type=dt,
                push_token=push_token,
                device_name=body.get("device_name"),
                app_version=body.get("app_version"),
                os_version=body.get("os_version"),
                device_model=body.get("device_model"),
                timezone=body.get("timezone"),
                locale=body.get("locale"),
                app_bundle_id=body.get("app_bundle_id"),
            )

            # Get appropriate connector
            registry = get_registry()

            # Map device type to connector platform
            platform_map = {
                DeviceType.ANDROID: "fcm",
                DeviceType.WEB: "web_push",
                DeviceType.IOS: "apns",
                DeviceType.ALEXA: "alexa",
                DeviceType.GOOGLE_HOME: "google_home",
            }

            platform = platform_map.get(dt)
            if not platform:
                return error_response(f"No connector for device type: {device_type}", 400)

            try:
                connector = registry.get(platform, auto_initialize=True)
            except KeyError:
                return error_response(f"Connector not available: {platform}", 503)

            # Register device
            device_token = await connector.register_device(registration)

            if device_token:
                return json_response(
                    {
                        "success": True,
                        "device_id": device_token.device_id,
                        "device_type": device_token.device_type.value,
                        "registered_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            else:
                return error_response("Failed to register device", 400)

        except ImportError:
            return error_response("Device connectors not available", 503)
        except Exception as e:
            logger.error(f"Error registering device: {e}")
            return error_response(f"Error registering device: {e}", 500)

    async def _unregister_device(
        self,
        device_id: str,
        auth_context: Dict[str, Any],
    ) -> HandlerResult:
        """Unregister a device."""
        try:
            from aragora.server.session_store import get_session_store

            store = get_session_store()

            # Get device to verify ownership
            device = store.get_device_session(device_id)
            if not device:
                return error_response("Device not found", 404)

            # Check ownership (unless admin)
            user_id = auth_context.get("user_id")
            is_admin = "admin" in auth_context.get("roles", [])
            if not is_admin and device.user_id != user_id:
                return error_response("Not authorized to delete this device", 403)

            # Delete device
            success = store.delete_device_session(device_id)

            if success:
                return json_response(
                    {
                        "success": True,
                        "device_id": device_id,
                        "deleted_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            else:
                return error_response("Failed to delete device", 500)

        except ImportError:
            return error_response("Session store not available", 503)
        except Exception as e:
            logger.error(f"Error unregistering device: {e}")
            return error_response(f"Error unregistering device: {e}", 500)

    async def _get_device(
        self,
        device_id: str,
        auth_context: Dict[str, Any],
    ) -> HandlerResult:
        """Get device information."""
        try:
            from aragora.server.session_store import get_session_store

            store = get_session_store()
            device = store.get_device_session(device_id)

            if not device:
                return error_response("Device not found", 404)

            # Check ownership (unless admin)
            user_id = auth_context.get("user_id")
            is_admin = "admin" in auth_context.get("roles", [])
            if not is_admin and device.user_id != user_id:
                return error_response("Not authorized to view this device", 403)

            return json_response(
                {
                    "device_id": device.device_id,
                    "user_id": device.user_id,
                    "device_type": device.device_type,
                    "device_name": device.device_name,
                    "app_version": device.app_version,
                    "last_active": device.last_active,
                    "notification_count": device.notification_count,
                    "created_at": device.created_at,
                }
            )

        except ImportError:
            return error_response("Session store not available", 503)
        except Exception as e:
            logger.error(f"Error getting device: {e}")
            return error_response(f"Error getting device: {e}", 500)

    async def _list_user_devices(
        self,
        user_id: str,
        auth_context: Dict[str, Any],
    ) -> HandlerResult:
        """List all devices for a user."""
        # Check ownership (unless admin)
        caller_id = auth_context.get("user_id")
        is_admin = "admin" in auth_context.get("roles", [])
        if not is_admin and user_id != caller_id:
            return error_response("Not authorized to view these devices", 403)

        try:
            from aragora.server.session_store import get_session_store

            store = get_session_store()
            devices = store.find_devices_by_user(user_id)

            return json_response(
                {
                    "user_id": user_id,
                    "device_count": len(devices),
                    "devices": [
                        {
                            "device_id": d.device_id,
                            "device_type": d.device_type,
                            "device_name": d.device_name,
                            "app_version": d.app_version,
                            "last_active": d.last_active,
                            "notification_count": d.notification_count,
                        }
                        for d in devices
                    ],
                }
            )

        except ImportError:
            return error_response("Session store not available", 503)
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
            return error_response(f"Error listing devices: {e}", 500)

    async def _notify_device(
        self,
        device_id: str,
        body: Dict[str, Any],
        auth_context: Dict[str, Any],
    ) -> HandlerResult:
        """Send notification to a specific device."""
        # Validate required fields
        if not body.get("title") or not body.get("body"):
            return error_response("title and body are required", 400)

        try:
            from aragora.connectors.devices import DeviceMessage, DeviceToken, DeviceType
            from aragora.connectors.devices.registry import get_registry
            from aragora.server.session_store import get_session_store

            store = get_session_store()
            device = store.get_device_session(device_id)

            if not device:
                return error_response("Device not found", 404)

            # Build message
            message = DeviceMessage(
                title=body["title"],
                body=body["body"],
                data=body.get("data", {}),
                image_url=body.get("image_url"),
                action_url=body.get("action_url"),
                badge=body.get("badge"),
                sound=body.get("sound", "default"),
            )

            # Get connector for device type
            registry = get_registry()
            device_type = DeviceType(device.device_type)

            platform_map = {
                DeviceType.ANDROID: "fcm",
                DeviceType.WEB: "web_push",
                DeviceType.IOS: "apns",
            }

            platform = platform_map.get(device_type)
            if not platform:
                return error_response(f"No connector for device type: {device.device_type}", 400)

            try:
                connector = registry.get(platform)
            except KeyError:
                return error_response(f"Connector not available: {platform}", 503)

            # Build DeviceToken from session
            token = DeviceToken(
                device_id=device.device_id,
                user_id=device.user_id,
                device_type=device_type,
                push_token=device.push_token,
                device_name=device.device_name,
                app_version=device.app_version,
            )

            # Send notification
            result = await connector.send_notification(token, message)

            # Update notification count
            if result.success:
                device.record_notification()
                store.set_device_session(device)

            # Handle invalid tokens
            if result.should_unregister:
                store.delete_device_session(device_id)
                logger.info(f"Removed invalid device token: {device_id}")

            return json_response(
                {
                    "success": result.success,
                    "device_id": device_id,
                    "message_id": result.message_id,
                    "status": result.status.value,
                    "error": result.error,
                }
            )

        except ImportError as e:
            return error_response(f"Required module not available: {e}", 503)
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return error_response(f"Error sending notification: {e}", 500)

    async def _notify_user(
        self,
        user_id: str,
        body: Dict[str, Any],
        auth_context: Dict[str, Any],
    ) -> HandlerResult:
        """Send notification to all devices for a user."""
        # Validate required fields
        if not body.get("title") or not body.get("body"):
            return error_response("title and body are required", 400)

        try:
            from aragora.connectors.devices import DeviceMessage, DeviceToken, DeviceType
            from aragora.connectors.devices.registry import get_registry
            from aragora.server.session_store import get_session_store

            store = get_session_store()
            devices = store.find_devices_by_user(user_id)

            if not devices:
                return json_response(
                    {
                        "success": True,
                        "user_id": user_id,
                        "devices_notified": 0,
                        "message": "No devices registered for user",
                    }
                )

            # Build message
            message = DeviceMessage(
                title=body["title"],
                body=body["body"],
                data=body.get("data", {}),
                image_url=body.get("image_url"),
                action_url=body.get("action_url"),
                badge=body.get("badge"),
                sound=body.get("sound", "default"),
            )

            registry = get_registry()
            results = []
            tokens_to_remove = []

            # Send to each device
            for device in devices:
                device_type = DeviceType(device.device_type)

                platform_map = {
                    DeviceType.ANDROID: "fcm",
                    DeviceType.WEB: "web_push",
                    DeviceType.IOS: "apns",
                }

                platform = platform_map.get(device_type)
                if not platform:
                    continue

                try:
                    connector = registry.get(platform)
                except KeyError:
                    continue

                token = DeviceToken(
                    device_id=device.device_id,
                    user_id=device.user_id,
                    device_type=device_type,
                    push_token=device.push_token,
                    device_name=device.device_name,
                    app_version=device.app_version,
                )

                result = await connector.send_notification(token, message)
                results.append(
                    {
                        "device_id": device.device_id,
                        "success": result.success,
                        "error": result.error,
                    }
                )

                if result.success:
                    device.record_notification()
                    store.set_device_session(device)

                if result.should_unregister:
                    tokens_to_remove.append(device.device_id)

            # Remove invalid tokens
            for device_id in tokens_to_remove:
                store.delete_device_session(device_id)
                logger.info(f"Removed invalid device token: {device_id}")

            success_count = sum(1 for r in results if r["success"])

            return json_response(
                {
                    "success": True,
                    "user_id": user_id,
                    "devices_notified": success_count,
                    "devices_failed": len(results) - success_count,
                    "devices_removed": len(tokens_to_remove),
                    "results": results,
                }
            )

        except ImportError as e:
            return error_response(f"Required module not available: {e}", 503)
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
            return error_response(f"Error sending notifications: {e}", 500)

    async def _handle_alexa_webhook(
        self,
        body: Dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """
        Handle incoming Alexa skill webhook requests.

        Processes voice commands and returns Alexa-formatted responses.
        """
        try:
            from aragora.connectors.devices.alexa import AlexaConnector
            from aragora.connectors.devices.registry import get_registry

            # Get or initialize Alexa connector
            registry = get_registry()
            try:
                connector = registry.get("alexa", auto_initialize=True)
            except KeyError:
                return error_response("Alexa connector not available", 503)

            if not isinstance(connector, AlexaConnector):
                return error_response("Invalid connector type", 500)

            # Verify skill ID
            if not connector.verify_skill_id(body):
                return error_response("Invalid skill ID", 403)

            # Parse request
            voice_request = connector.parse_alexa_request(body)

            # Handle the request
            voice_response = await connector.handle_voice_request(voice_request)

            # Build Alexa response format
            session_attributes = body.get("session", {}).get("attributes", {})
            alexa_response = connector.build_alexa_response(voice_response, session_attributes)

            return json_response(alexa_response)

        except ImportError:
            return error_response("Alexa connector not available", 503)
        except Exception as e:
            logger.error(f"Error handling Alexa webhook: {e}")
            return error_response(f"Error processing request: {e}", 500)

    async def _handle_google_webhook(
        self,
        body: Dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """
        Handle incoming Google Actions webhook requests.

        Processes voice commands and Smart Home intents.
        """
        try:
            from aragora.connectors.devices.google_home import GoogleHomeConnector
            from aragora.connectors.devices.registry import get_registry

            # Get or initialize Google Home connector
            registry = get_registry()
            try:
                connector = registry.get("google_home", auto_initialize=True)
            except KeyError:
                return error_response("Google Home connector not available", 503)

            if not isinstance(connector, GoogleHomeConnector):
                return error_response("Invalid connector type", 500)

            # Check for Smart Home intents
            inputs = body.get("inputs", [])
            if inputs:
                intent = inputs[0].get("intent", "")

                # Handle Smart Home SYNC
                if intent == "action.devices.SYNC":
                    request_id = body.get("requestId", "")
                    user_id = body.get("user", {}).get("userId", "")
                    response = await connector.handle_sync(request_id, user_id)
                    return json_response(response)

                # Handle Smart Home QUERY
                if intent == "action.devices.QUERY":
                    request_id = body.get("requestId", "")
                    devices = inputs[0].get("payload", {}).get("devices", [])
                    response = await connector.handle_query(request_id, devices)
                    return json_response(response)

                # Handle Smart Home EXECUTE
                if intent == "action.devices.EXECUTE":
                    request_id = body.get("requestId", "")
                    commands = inputs[0].get("payload", {}).get("commands", [])
                    response = await connector.handle_execute(request_id, commands)
                    return json_response(response)

                # Handle Smart Home DISCONNECT
                if intent == "action.devices.DISCONNECT":
                    request_id = body.get("requestId", "")
                    user_id = body.get("user", {}).get("userId", "")
                    response = await connector.handle_disconnect(request_id, user_id)
                    return json_response(response)

            # Handle Conversational Actions
            voice_request = connector.parse_google_request(body)
            voice_response = await connector.handle_voice_request(voice_request)

            # Build Google response format
            session_params = body.get("session", {}).get("params", {})
            google_response = connector.build_google_response(voice_response, session_params)

            return json_response(google_response)

        except ImportError:
            return error_response("Google Home connector not available", 503)
        except Exception as e:
            logger.error(f"Error handling Google webhook: {e}")
            return error_response(f"Error processing request: {e}", 500)
