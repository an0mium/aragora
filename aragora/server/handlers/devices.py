"""
Device Registration and Notification API Handlers.

Stability: STABLE

Provides REST APIs for device push notification management:
- Device registration and unregistration
- Push notification delivery
- Device health monitoring
- Voice assistant webhooks (Alexa, Google Home)

Endpoints:
- POST /api/devices/register - Register a device for push notifications
- DELETE /api/devices/{device_id} - Unregister a device
- POST /api/devices/{device_id}/notify - Send notification to a device
- POST /api/devices/user/{user_id}/notify - Send to all user devices
- GET /api/devices/user/{user_id} - List user's devices
- GET /api/devices/health - Get device connector health
- POST /api/devices/alexa/webhook - Alexa skill webhook
- POST /api/devices/google/webhook - Google Actions webhook

Features:
- Circuit breaker pattern for resilient connector access
- Rate limiting (30 requests/minute for notifications, 10/minute for registration)
- RBAC permission checks (devices.read, devices.write, devices.notify)
- Input validation and size limits
- Comprehensive error handling with safe error messages
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from aragora.rbac.decorators import require_permission
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.secure import ForbiddenError, SecureHandler, UnauthorizedError
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip
from aragora.server.versioning.compat import strip_version_prefix

logger = logging.getLogger(__name__)

# =============================================================================
# Rate Limiters
# =============================================================================

# Rate limiter for notification endpoints (30 requests per minute)
_notification_limiter = RateLimiter(requests_per_minute=30)

# Rate limiter for registration endpoints (10 requests per minute - less frequent)
_registration_limiter = RateLimiter(requests_per_minute=10)


# =============================================================================
# Circuit Breaker for Device Connectors
# =============================================================================


class DeviceCircuitBreaker:
    """Circuit breaker for device connector access.

    Prevents cascading failures when push notification services are unavailable.
    Uses a simple state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
    """

    # State constants
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            cooldown_seconds: Time to wait before allowing test calls
            half_open_max_calls: Number of test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._check_state()

    def _check_state(self) -> str:
        """Check and potentially transition state (must hold lock)."""
        if self._state == self.OPEN:
            # Check if cooldown has elapsed
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.cooldown_seconds
            ):
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Device circuit breaker transitioning to HALF_OPEN")
        return self._state

    def can_proceed(self) -> bool:
        """Check if a call can proceed.

        Returns:
            True if call is allowed, False if circuit is open
        """
        with self._lock:
            state = self._check_state()
            if state == self.CLOSED:
                return True
            elif state == self.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Device circuit breaker closed after successful recovery")
            elif self._state == self.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                self._state = self.OPEN
                self._success_count = 0
                logger.warning("Device circuit breaker reopened after failure in HALF_OPEN")
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(
                        f"Device circuit breaker opened after {self._failure_count} failures"
                    )

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            return {
                "state": self._check_state(),
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# Per-connector circuit breakers
_device_circuit_breakers: dict[str, DeviceCircuitBreaker] = {}
_circuit_breaker_lock = threading.Lock()


def _get_circuit_breaker(connector: str) -> DeviceCircuitBreaker:
    """Get or create a circuit breaker for a device connector."""
    with _circuit_breaker_lock:
        if connector not in _device_circuit_breakers:
            _device_circuit_breakers[connector] = DeviceCircuitBreaker()
        return _device_circuit_breakers[connector]


def get_device_circuit_breaker_status() -> dict[str, Any]:
    """Get status of all device circuit breakers.

    Returns:
        Dict mapping connector name to circuit breaker status
    """
    with _circuit_breaker_lock:
        return {name: cb.get_status() for name, cb in _device_circuit_breakers.items()}


def _clear_device_circuit_breakers() -> None:
    """Clear all device circuit breakers (for testing)."""
    with _circuit_breaker_lock:
        _device_circuit_breakers.clear()


class DeviceHandler(SecureHandler):
    """Handler for device registration and notification endpoints."""

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    RESOURCE_TYPE = "devices"

    ROUTES = [
        "/api/devices/register",
        "/api/devices/health",
        "/api/devices/alexa/webhook",
        "/api/devices/google/webhook",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        normalized = strip_version_prefix(path)
        if normalized in self.ROUTES:
            return True
        return normalized.startswith("/api/devices/")

    @require_permission("devices:read")
    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route GET requests."""
        return await self._route_request(path, "GET", query_params, handler, None)

    @require_permission("devices:write")
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route POST requests."""
        body, err = self.read_json_body_validated(handler)
        if err:
            return err
        return await self._route_request(path, "POST", query_params, handler, body)

    @require_permission("devices:delete")
    async def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route DELETE requests."""
        return await self._route_request(path, "DELETE", query_params, handler, None)

    async def _route_request(
        self,
        path: str,
        method: str,
        query_params: dict[str, Any],
        handler: Any,
        body: Optional[dict[str, Any]],
    ) -> HandlerResult | None:
        """Route device requests."""
        normalized = strip_version_prefix(path)

        # Alexa webhook (no auth required - uses Alexa signature verification)
        if normalized == "/api/devices/alexa/webhook" and method == "POST":
            return await self._handle_alexa_webhook(body or {}, handler)

        # Google webhook (no auth required - uses Google verification)
        if normalized == "/api/devices/google/webhook" and method == "POST":
            return await self._handle_google_webhook(body or {}, handler)

        # Require authentication for all other endpoints
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            return error_response(str(e), 403)

        # Get client IP for rate limiting
        client_ip = get_client_ip(handler)

        # Health endpoint (read permission)
        if normalized == "/api/devices/health" and method == "GET":
            try:
                self.check_permission(auth_context, "devices.read")
            except ForbiddenError:
                return error_response("Permission denied: devices.read", 403)
            return await self._get_health()

        # Register device (rate limited)
        if normalized == "/api/devices/register" and method == "POST":
            # Apply registration rate limit
            if not _registration_limiter.is_allowed(client_ip):
                logger.warning(f"Rate limit exceeded for device registration from {client_ip}")
                return error_response("Rate limit exceeded. Try again later.", 429)
            try:
                self.check_permission(auth_context, "devices.write")
            except ForbiddenError:
                return error_response("Permission denied: devices.write", 403)
            return await self._register_device(body or {}, auth_context, handler)

        segments = normalized.strip("/").split("/")
        if len(segments) < 2 or segments[0] != "api" or segments[1] != "devices":
            return None

        # User-specific endpoints: /api/devices/user/{user_id}[/notify]
        if len(segments) >= 4 and segments[2] == "user":
            user_id = segments[3]
            if len(segments) == 4 and method == "GET":
                try:
                    self.check_permission(auth_context, "devices.read")
                except ForbiddenError:
                    return error_response("Permission denied: devices.read", 403)
                return await self._list_user_devices(user_id, auth_context)
            if len(segments) == 5 and segments[4] == "notify" and method == "POST":
                # Apply notification rate limit
                if not _notification_limiter.is_allowed(client_ip):
                    logger.warning(f"Rate limit exceeded for user notification from {client_ip}")
                    return error_response("Rate limit exceeded. Try again later.", 429)
                try:
                    self.check_permission(auth_context, "devices.notify")
                except ForbiddenError:
                    return error_response("Permission denied: devices.notify", 403)
                return await self._notify_user(user_id, body or {}, auth_context)
            return None

        # Device-specific endpoints: /api/devices/{device_id}[/notify]
        device_id = segments[2] if len(segments) >= 3 else None
        if not device_id:
            return None

        if len(segments) == 3 and method == "GET":
            try:
                self.check_permission(auth_context, "devices.read")
            except ForbiddenError:
                return error_response("Permission denied: devices.read", 403)
            return await self._get_device(device_id, auth_context)

        if len(segments) == 3 and method == "DELETE":
            try:
                self.check_permission(auth_context, "devices.write")
            except ForbiddenError:
                return error_response("Permission denied: devices.write", 403)
            return await self._unregister_device(device_id, auth_context)

        if len(segments) == 4 and segments[3] == "notify" and method == "POST":
            # Apply notification rate limit
            if not _notification_limiter.is_allowed(client_ip):
                logger.warning(f"Rate limit exceeded for device notification from {client_ip}")
                return error_response("Rate limit exceeded. Try again later.", 429)
            try:
                self.check_permission(auth_context, "devices.notify")
            except ForbiddenError:
                return error_response("Permission denied: devices.notify", 403)
            return await self._notify_device(device_id, body or {}, auth_context)

        return None

    async def _get_health(self) -> HandlerResult:
        """Get device connector health status including circuit breaker state."""
        try:
            from aragora.connectors.devices.registry import get_registry

            registry = get_registry()
            health = await registry.get_health()

            # Add circuit breaker status
            health["circuit_breakers"] = get_device_circuit_breaker_status()

            return json_response(health)

        except ImportError:
            return json_response(
                {
                    "status": "unavailable",
                    "error": "Device connectors not available",
                    "circuit_breakers": get_device_circuit_breaker_status(),
                }
            )
        except Exception as e:
            logger.error(f"Error getting device health: {e}")
            return error_response(f"Error getting health: {e}", 500)

    async def _register_device(
        self,
        body: dict[str, Any],
        auth_context: AuthorizationContext,
        handler: Any = None,
    ) -> HandlerResult:
        """Register a device for push notifications.

        Uses circuit breaker pattern to handle connector failures gracefully.
        """
        # Validate required fields
        required = ["device_type", "push_token"]
        missing = [f for f in required if not body.get(f)]
        if missing:
            return error_response(f"Missing required fields: {missing}", 400)

        device_type = body.get("device_type")
        push_token = body.get("push_token")

        # Validate push token length (max 4KB for safety)
        if len(push_token) > 4096:
            return error_response("push_token exceeds maximum length (4096 bytes)", 400)

        # Get user_id from auth context or body
        user_id = body.get("user_id") or auth_context.user_id
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

            # Check circuit breaker before proceeding
            circuit_breaker = _get_circuit_breaker(platform)
            if not circuit_breaker.can_proceed():
                logger.warning(f"Circuit breaker open for connector: {platform}")
                return error_response(
                    f"Service temporarily unavailable for {platform}. Please retry later.",
                    503,
                )

            try:
                connector = registry.get(platform, auto_initialize=True)
            except KeyError:
                circuit_breaker.record_failure()
                return error_response(f"Connector not available: {platform}", 503)

            # Register device with circuit breaker protection
            try:
                device_token = await connector.register_device(registration)
                circuit_breaker.record_success()
            except Exception as conn_error:
                circuit_breaker.record_failure()
                logger.error(f"Connector error during registration: {conn_error}")
                raise

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
            return error_response("Error registering device. Please try again later.", 500)

    async def _unregister_device(
        self,
        device_id: str,
        auth_context: AuthorizationContext,
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
            user_id = auth_context.user_id
            is_admin = auth_context.has_role("admin") or auth_context.has_role("owner")
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
        auth_context: AuthorizationContext,
    ) -> HandlerResult:
        """Get device information."""
        try:
            from aragora.server.session_store import get_session_store

            store = get_session_store()
            device = store.get_device_session(device_id)

            if not device:
                return error_response("Device not found", 404)

            # Check ownership (unless admin)
            user_id = auth_context.user_id
            is_admin = auth_context.has_role("admin") or auth_context.has_role("owner")
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
        auth_context: AuthorizationContext,
    ) -> HandlerResult:
        """List all devices for a user."""
        # Check ownership (unless admin)
        caller_id = auth_context.user_id
        is_admin = auth_context.has_role("admin") or auth_context.has_role("owner")
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
        body: dict[str, Any],
        auth_context: AuthorizationContext,
    ) -> HandlerResult:
        """Send notification to a specific device.

        Uses circuit breaker pattern to handle connector failures gracefully.
        """
        # Validate required fields
        if not body.get("title") or not body.get("body"):
            return error_response("title and body are required", 400)

        # Validate message size limits
        title = body.get("title", "")
        msg_body = body.get("body", "")
        if len(title) > 256:
            return error_response("title exceeds maximum length (256 characters)", 400)
        if len(msg_body) > 4096:
            return error_response("body exceeds maximum length (4096 characters)", 400)

        try:
            from aragora.connectors.devices import DeviceMessage, DeviceToken, DeviceType
            from aragora.connectors.devices.registry import get_registry
            from aragora.server.session_store import get_session_store

            store = get_session_store()
            device = store.get_device_session(device_id)

            if not device:
                return error_response("Device not found", 404)

            # Check ownership (unless admin/owner)
            if (
                not (auth_context.has_role("admin") or auth_context.has_role("owner"))
                and device.user_id != auth_context.user_id
            ):
                return error_response("Not authorized to notify this device", 403)

            # Build message
            message = DeviceMessage(
                title=title,
                body=msg_body,
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

            # Check circuit breaker before proceeding
            circuit_breaker = _get_circuit_breaker(platform)
            if not circuit_breaker.can_proceed():
                logger.warning(f"Circuit breaker open for connector: {platform}")
                return error_response(
                    f"Service temporarily unavailable for {platform}. Please retry later.",
                    503,
                )

            try:
                connector = registry.get(platform)
            except KeyError:
                circuit_breaker.record_failure()
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

            # Send notification with circuit breaker protection
            try:
                result = await connector.send_notification(token, message)
                circuit_breaker.record_success()
            except Exception as conn_error:
                circuit_breaker.record_failure()
                logger.error(f"Connector error during notification: {conn_error}")
                return error_response(
                    "Failed to send notification. Please try again later.",
                    503,
                )

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
            return error_response("Error sending notification. Please try again later.", 500)

    async def _notify_user(
        self,
        user_id: str,
        body: dict[str, Any],
        auth_context: AuthorizationContext,
    ) -> HandlerResult:
        """Send notification to all devices for a user.

        Uses circuit breaker pattern to handle connector failures gracefully.
        """
        # Validate required fields
        if not body.get("title") or not body.get("body"):
            return error_response("title and body are required", 400)

        # Validate message size limits
        title = body.get("title", "")
        msg_body = body.get("body", "")
        if len(title) > 256:
            return error_response("title exceeds maximum length (256 characters)", 400)
        if len(msg_body) > 4096:
            return error_response("body exceeds maximum length (4096 characters)", 400)

        # Check ownership (unless admin/owner)
        if (
            not (auth_context.has_role("admin") or auth_context.has_role("owner"))
            and user_id != auth_context.user_id
        ):
            return error_response("Not authorized to notify these devices", 403)

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
                title=title,
                body=msg_body,
                data=body.get("data", {}),
                image_url=body.get("image_url"),
                action_url=body.get("action_url"),
                badge=body.get("badge"),
                sound=body.get("sound", "default"),
            )

            registry = get_registry()
            results = []
            tokens_to_remove = []
            circuit_open_platforms = []

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

                # Check circuit breaker before proceeding
                circuit_breaker = _get_circuit_breaker(platform)
                if not circuit_breaker.can_proceed():
                    if platform not in circuit_open_platforms:
                        circuit_open_platforms.append(platform)
                        logger.warning(f"Circuit breaker open for connector: {platform}")
                    results.append(
                        {
                            "device_id": device.device_id,
                            "success": False,
                            "error": f"Service temporarily unavailable for {platform}",
                        }
                    )
                    continue

                try:
                    connector = registry.get(platform)
                except KeyError:
                    circuit_breaker.record_failure()
                    results.append(
                        {
                            "device_id": device.device_id,
                            "success": False,
                            "error": f"Connector not available: {platform}",
                        }
                    )
                    continue

                token = DeviceToken(
                    device_id=device.device_id,
                    user_id=device.user_id,
                    device_type=device_type,
                    push_token=device.push_token,
                    device_name=device.device_name,
                    app_version=device.app_version,
                )

                # Send notification with circuit breaker protection
                try:
                    result = await connector.send_notification(token, message)
                    circuit_breaker.record_success()
                except Exception as conn_error:
                    circuit_breaker.record_failure()
                    logger.error(f"Connector error during notification: {conn_error}")
                    results.append(
                        {
                            "device_id": device.device_id,
                            "success": False,
                            "error": "Connector error",
                        }
                    )
                    continue

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
        body: dict[str, Any],
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
        body: dict[str, Any],
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


__all__ = [
    "DeviceHandler",
    "DeviceCircuitBreaker",
    "get_device_circuit_breaker_status",
    "_clear_device_circuit_breakers",
]
