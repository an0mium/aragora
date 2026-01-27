"""
Push Notification Connectors.

Implements connectors for:
- Firebase Cloud Messaging (FCM) - Android and Web
- Apple Push Notification service (APNs) - iOS
- Web Push (VAPID) - Browser notifications

These connectors provide reliable push notification delivery with:
- Retry with exponential backoff
- Circuit breaker for fault tolerance
- Batch sending support
- Token validation and cleanup
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import DeviceConnector, DeviceConnectorConfig
from .models import (
    BatchSendResult,
    DeliveryStatus,
    DeviceMessage,
    DeviceToken,
    DeviceType,
    NotificationPriority,
    SendResult,
)

logger = logging.getLogger(__name__)


class FCMConnector(DeviceConnector):
    """
    Firebase Cloud Messaging (FCM) connector.

    Supports sending push notifications to Android devices and web browsers
    using FCM HTTP v1 API.

    Required credentials:
        - FCM_PROJECT_ID: Firebase project ID
        - FCM_PRIVATE_KEY: Service account private key (JSON string)

    Or use a service account file:
        - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON

    Usage:
        connector = FCMConnector()
        await connector.initialize()

        result = await connector.send_notification(
            device=device_token,
            message=DeviceMessage(title="Hello", body="World"),
        )
    """

    # FCM HTTP v1 API endpoint
    FCM_ENDPOINT = "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"

    def __init__(self, config: Optional[DeviceConnectorConfig] = None):
        super().__init__(config)
        self._project_id: Optional[str] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0

    @property
    def platform_name(self) -> str:
        return "fcm"

    @property
    def platform_display_name(self) -> str:
        return "Firebase Cloud Messaging"

    @property
    def supported_device_types(self) -> List[DeviceType]:
        return [DeviceType.ANDROID, DeviceType.WEB]

    async def initialize(self) -> bool:
        """Initialize FCM connector with credentials."""
        self._project_id = self.config.credentials.get("project_id") or os.environ.get(
            "FCM_PROJECT_ID"
        )

        if not self._project_id:
            logger.warning("FCM_PROJECT_ID not configured")
            return False

        # Get access token
        try:
            await self._refresh_access_token()
            self._initialized = True
            logger.info(f"FCM connector initialized for project {self._project_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize FCM connector: {e}")
            return False

    async def _refresh_access_token(self) -> None:
        """Refresh the OAuth2 access token for FCM API."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account

            # Try to load from environment or file
            credentials_json = self.config.credentials.get("private_key") or os.environ.get(
                "FCM_PRIVATE_KEY"
            )
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

            if credentials_json:
                # Parse JSON string
                creds_dict = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    creds_dict,
                    scopes=["https://www.googleapis.com/auth/firebase.messaging"],
                )
            elif credentials_file:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_file,
                    scopes=["https://www.googleapis.com/auth/firebase.messaging"],
                )
            else:
                raise ValueError("No FCM credentials configured")

            # Refresh the credentials
            credentials.refresh(Request())

            self._access_token = credentials.token
            # Token typically expires in 1 hour, refresh 5 minutes before
            self._token_expires_at = time.time() + 3300

        except ImportError:
            logger.warning(
                "google-auth library not available. Install with: pip install google-auth"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to refresh FCM access token: {e}")
            raise

    async def _ensure_valid_token(self) -> str:
        """Ensure we have a valid access token."""
        if not self._access_token or time.time() >= self._token_expires_at:
            await self._refresh_access_token()
        return self._access_token or ""

    async def send_notification(
        self,
        device: DeviceToken,
        message: DeviceMessage,
        **kwargs: Any,
    ) -> SendResult:
        """Send notification via FCM HTTP v1 API."""
        if not self._initialized:
            return SendResult(
                success=False,
                device_id=device.device_id,
                status=DeliveryStatus.FAILED,
                error="FCM connector not initialized",
            )

        try:
            token = await self._ensure_valid_token()

            # Build FCM message
            fcm_message = self._build_fcm_message(device, message)

            url = self.FCM_ENDPOINT.format(project_id=self._project_id)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

            success, response, error = await self._http_request(
                method="POST",
                url=url,
                headers=headers,
                json={"message": fcm_message},
                operation="send_notification",
            )

            if success and response:
                message_id = response.get("name", "").split("/")[-1]
                return SendResult(
                    success=True,
                    device_id=device.device_id,
                    message_id=message_id,
                    status=DeliveryStatus.SENT,
                    timestamp=datetime.now(timezone.utc),
                )
            else:
                # Check for invalid token errors
                should_unregister = self._is_invalid_token_error(error or "")
                return SendResult(
                    success=False,
                    device_id=device.device_id,
                    status=DeliveryStatus.FAILED,
                    error=error,
                    should_unregister=should_unregister,
                    timestamp=datetime.now(timezone.utc),
                )

        except Exception as e:
            logger.error(f"FCM send error: {e}")
            return SendResult(
                success=False,
                device_id=device.device_id,
                status=DeliveryStatus.FAILED,
                error=str(e),
                timestamp=datetime.now(timezone.utc),
            )

    def _build_fcm_message(
        self,
        device: DeviceToken,
        message: DeviceMessage,
    ) -> Dict[str, Any]:
        """Build FCM message payload."""
        fcm_msg: Dict[str, Any] = {
            "token": device.push_token,
            "notification": {
                "title": message.title,
                "body": message.body,
            },
        }

        # Add image
        if message.image_url:
            fcm_msg["notification"]["image"] = message.image_url

        # Add data payload
        if message.data:
            fcm_msg["data"] = {k: str(v) for k, v in message.data.items()}

        # Add action URL to data
        if message.action_url:
            fcm_msg.setdefault("data", {})["action_url"] = message.action_url

        # Android-specific configuration
        if device.device_type == DeviceType.ANDROID:
            fcm_msg["android"] = {
                "priority": self._map_priority(message.priority),
                "ttl": f"{message.ttl_seconds}s",
            }

            if message.channel_id:
                fcm_msg["android"]["notification"] = {
                    "channel_id": message.channel_id,
                }

            if message.collapse_key:
                fcm_msg["android"]["collapse_key"] = message.collapse_key

        # Web push configuration
        elif device.device_type == DeviceType.WEB:
            fcm_msg["webpush"] = {
                "headers": {
                    "TTL": str(message.ttl_seconds),
                },
            }

            if message.action_url:
                fcm_msg["webpush"]["fcm_options"] = {
                    "link": message.action_url,
                }

        return fcm_msg

    def _map_priority(self, priority: NotificationPriority) -> str:
        """Map our priority to FCM priority."""
        if priority in (NotificationPriority.HIGH, NotificationPriority.URGENT):
            return "high"
        return "normal"

    def _is_invalid_token_error(self, error: str) -> bool:
        """Check if error indicates an invalid token that should be removed."""
        invalid_indicators = [
            "UNREGISTERED",
            "INVALID_ARGUMENT",
            "NOT_FOUND",
            "InvalidRegistration",
            "NotRegistered",
        ]
        return any(indicator in error for indicator in invalid_indicators)

    def validate_token(self, token: str) -> bool:
        """Validate FCM token format."""
        # FCM tokens are typically 150+ characters
        return bool(token and len(token) >= 100)

    async def send_batch(
        self,
        devices: List[DeviceToken],
        message: DeviceMessage,
        **kwargs: Any,
    ) -> BatchSendResult:
        """
        Send to multiple devices using FCM batch API.

        FCM supports up to 500 tokens per multicast request.
        """
        if not self._initialized:
            return BatchSendResult(
                total_sent=0,
                success_count=0,
                failure_count=len(devices),
                results=[
                    SendResult(
                        success=False,
                        device_id=d.device_id,
                        status=DeliveryStatus.FAILED,
                        error="FCM connector not initialized",
                    )
                    for d in devices
                ],
            )

        # Split into batches of 500
        batch_size = min(self.config.max_batch_size, 500)
        all_results = []
        tokens_to_remove = []

        for i in range(0, len(devices), batch_size):
            batch = devices[i : i + batch_size]

            # For FCM, we still need to send individual messages
            # (multicast API is legacy)
            for device in batch:
                result = await self.send_notification(device, message, **kwargs)
                all_results.append(result)

                if result.should_unregister:
                    tokens_to_remove.append(device.device_id)

        success_count = sum(1 for r in all_results if r.success)

        return BatchSendResult(
            total_sent=len(all_results),
            success_count=success_count,
            failure_count=len(all_results) - success_count,
            results=all_results,
            tokens_to_remove=tokens_to_remove,
        )


class APNsConnector(DeviceConnector):
    """
    Apple Push Notification service (APNs) connector.

    Supports sending push notifications to iOS, iPadOS, watchOS, macOS,
    and tvOS devices using APNs HTTP/2 API.

    Required credentials:
        - APNS_KEY_ID: Key ID from Apple Developer
        - APNS_TEAM_ID: Team ID from Apple Developer
        - APNS_PRIVATE_KEY: .p8 private key content
        - APNS_BUNDLE_ID: App bundle identifier

    Usage:
        connector = APNsConnector()
        await connector.initialize()

        result = await connector.send_notification(
            device=device_token,
            message=DeviceMessage(title="Hello", body="World"),
        )
    """

    # APNs endpoints
    APNS_PRODUCTION = "https://api.push.apple.com"
    APNS_SANDBOX = "https://api.sandbox.push.apple.com"

    def __init__(
        self,
        config: Optional[DeviceConnectorConfig] = None,
        use_sandbox: bool = False,
    ):
        super().__init__(config)
        self._use_sandbox = use_sandbox
        self._key_id: Optional[str] = None
        self._team_id: Optional[str] = None
        self._bundle_id: Optional[str] = None
        self._private_key: Optional[str] = None
        self._jwt_token: Optional[str] = None
        self._token_issued_at: float = 0

    @property
    def platform_name(self) -> str:
        return "apns"

    @property
    def platform_display_name(self) -> str:
        return "Apple Push Notification service"

    @property
    def supported_device_types(self) -> List[DeviceType]:
        return [DeviceType.IOS]

    @property
    def base_url(self) -> str:
        """Get the APNs endpoint URL."""
        return self.APNS_SANDBOX if self._use_sandbox else self.APNS_PRODUCTION

    async def initialize(self) -> bool:
        """Initialize APNs connector with credentials."""
        self._key_id = self.config.credentials.get("key_id") or os.environ.get("APNS_KEY_ID")
        self._team_id = self.config.credentials.get("team_id") or os.environ.get("APNS_TEAM_ID")
        self._bundle_id = self.config.credentials.get("bundle_id") or os.environ.get(
            "APNS_BUNDLE_ID"
        )
        self._private_key = self.config.credentials.get("private_key") or os.environ.get(
            "APNS_PRIVATE_KEY"
        )

        if not all([self._key_id, self._team_id, self._bundle_id, self._private_key]):
            logger.warning("APNs credentials not fully configured")
            return False

        # Verify we can create a JWT
        try:
            self._refresh_jwt_token()
            self._initialized = True
            logger.info(f"APNs connector initialized for bundle {self._bundle_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize APNs connector: {e}")
            return False

    def _refresh_jwt_token(self) -> None:
        """Generate a new JWT token for APNs authentication."""
        try:
            import jwt

            now = time.time()

            payload = {
                "iss": self._team_id,
                "iat": int(now),
            }

            headers = {
                "alg": "ES256",
                "kid": self._key_id,
            }

            self._jwt_token = jwt.encode(
                payload,
                self._private_key,
                algorithm="ES256",
                headers=headers,
            )
            self._token_issued_at = now

        except ImportError:
            logger.warning("PyJWT library not available. Install with: pip install pyjwt")
            raise
        except Exception as e:
            logger.error(f"Failed to generate APNs JWT: {e}")
            raise

    def _ensure_valid_token(self) -> str:
        """Ensure we have a valid JWT token."""
        # APNs tokens are valid for 1 hour, refresh after 50 minutes
        if not self._jwt_token or (time.time() - self._token_issued_at) >= 3000:
            self._refresh_jwt_token()
        return self._jwt_token or ""

    async def send_notification(
        self,
        device: DeviceToken,
        message: DeviceMessage,
        **kwargs: Any,
    ) -> SendResult:
        """Send notification via APNs HTTP/2 API."""
        if not self._initialized:
            return SendResult(
                success=False,
                device_id=device.device_id,
                status=DeliveryStatus.FAILED,
                error="APNs connector not initialized",
            )

        try:
            token = self._ensure_valid_token()

            # Build APNs payload
            payload = self._build_apns_payload(message)

            url = f"{self.base_url}/3/device/{device.push_token}"
            headers = {
                "authorization": f"bearer {token}",
                "apns-topic": self._bundle_id or "",
                "apns-push-type": "alert",
                "apns-priority": self._map_priority(message.priority),
                "apns-expiration": str(int(time.time()) + message.ttl_seconds),
            }

            if message.collapse_key:
                headers["apns-collapse-id"] = message.collapse_key

            success, response, error = await self._http_request(
                method="POST",
                url=url,
                headers=headers,
                json=payload,
                operation="send_notification",
            )

            if success:
                return SendResult(
                    success=True,
                    device_id=device.device_id,
                    message_id=response.get("apns-id") if response else None,
                    status=DeliveryStatus.SENT,
                    timestamp=datetime.now(timezone.utc),
                )
            else:
                # Check for invalid token errors
                should_unregister = self._is_invalid_token_error(error or "")
                return SendResult(
                    success=False,
                    device_id=device.device_id,
                    status=DeliveryStatus.FAILED,
                    error=error,
                    should_unregister=should_unregister,
                    timestamp=datetime.now(timezone.utc),
                )

        except Exception as e:
            logger.error(f"APNs send error: {e}")
            return SendResult(
                success=False,
                device_id=device.device_id,
                status=DeliveryStatus.FAILED,
                error=str(e),
                timestamp=datetime.now(timezone.utc),
            )

    def _build_apns_payload(self, message: DeviceMessage) -> Dict[str, Any]:
        """Build APNs notification payload."""
        aps: Dict[str, Any] = {
            "alert": {
                "title": message.title,
                "body": message.body,
            },
        }

        if message.badge is not None:
            aps["badge"] = message.badge

        if message.sound:
            aps["sound"] = message.sound

        if message.thread_id:
            aps["thread-id"] = message.thread_id

        if message.mutable_content:
            aps["mutable-content"] = 1

        payload: Dict[str, Any] = {"aps": aps}

        # Add custom data
        if message.data:
            payload.update(message.data)

        # Add action URL
        if message.action_url:
            payload["action_url"] = message.action_url

        return payload

    def _map_priority(self, priority: NotificationPriority) -> str:
        """Map our priority to APNs priority."""
        if priority == NotificationPriority.LOW:
            return "5"
        elif priority in (NotificationPriority.HIGH, NotificationPriority.URGENT):
            return "10"
        return "10"  # Default to high for alerts

    def _is_invalid_token_error(self, error: str) -> bool:
        """Check if error indicates an invalid token that should be removed."""
        invalid_indicators = [
            "BadDeviceToken",
            "Unregistered",
            "DeviceTokenNotForTopic",
            "TopicDisallowed",
            "ExpiredToken",
        ]
        return any(indicator in error for indicator in invalid_indicators)

    def validate_token(self, token: str) -> bool:
        """Validate APNs token format."""
        # APNs tokens are 64 hex characters
        if not token or len(token) != 64:
            return False
        try:
            int(token, 16)
            return True
        except ValueError:
            return False


class WebPushConnector(DeviceConnector):
    """
    Web Push connector using VAPID.

    Supports sending push notifications to web browsers using the
    Web Push protocol with VAPID authentication.

    Required credentials:
        - VAPID_PUBLIC_KEY: VAPID public key
        - VAPID_PRIVATE_KEY: VAPID private key
        - VAPID_SUBJECT: Contact email or URL

    Usage:
        connector = WebPushConnector()
        await connector.initialize()

        result = await connector.send_notification(
            device=device_token,  # push_token should be JSON subscription
            message=DeviceMessage(title="Hello", body="World"),
        )
    """

    def __init__(self, config: Optional[DeviceConnectorConfig] = None):
        super().__init__(config)
        self._vapid_public: Optional[str] = None
        self._vapid_private: Optional[str] = None
        self._vapid_subject: Optional[str] = None

    @property
    def platform_name(self) -> str:
        return "web_push"

    @property
    def platform_display_name(self) -> str:
        return "Web Push (VAPID)"

    @property
    def supported_device_types(self) -> List[DeviceType]:
        return [DeviceType.WEB]

    async def initialize(self) -> bool:
        """Initialize Web Push connector with VAPID credentials."""
        self._vapid_public = self.config.credentials.get("vapid_public_key") or os.environ.get(
            "VAPID_PUBLIC_KEY"
        )
        self._vapid_private = self.config.credentials.get("vapid_private_key") or os.environ.get(
            "VAPID_PRIVATE_KEY"
        )
        self._vapid_subject = self.config.credentials.get("vapid_subject") or os.environ.get(
            "VAPID_SUBJECT"
        )

        if not all([self._vapid_public, self._vapid_private, self._vapid_subject]):
            logger.warning("VAPID credentials not fully configured")
            return False

        self._initialized = True
        logger.info("Web Push connector initialized")
        return True

    async def send_notification(
        self,
        device: DeviceToken,
        message: DeviceMessage,
        **kwargs: Any,
    ) -> SendResult:
        """Send notification via Web Push."""
        if not self._initialized:
            return SendResult(
                success=False,
                device_id=device.device_id,
                status=DeliveryStatus.FAILED,
                error="Web Push connector not initialized",
            )

        try:
            from pywebpush import webpush

            # Parse subscription from push_token (JSON string)
            try:
                subscription = json.loads(device.push_token)
            except json.JSONDecodeError:
                return SendResult(
                    success=False,
                    device_id=device.device_id,
                    status=DeliveryStatus.FAILED,
                    error="Invalid subscription format",
                    should_unregister=True,
                )

            # Build payload
            payload = {
                "title": message.title,
                "body": message.body,
                "data": message.data,
            }

            if message.image_url:
                payload["image"] = message.image_url
            if message.action_url:
                payload["url"] = message.action_url

            vapid_claims = {
                "sub": self._vapid_subject,
            }

            webpush(
                subscription_info=subscription,
                data=json.dumps(payload),
                vapid_private_key=self._vapid_private,
                vapid_claims=vapid_claims,
                ttl=message.ttl_seconds,
            )

            self._record_success()
            return SendResult(
                success=True,
                device_id=device.device_id,
                status=DeliveryStatus.SENT,
                timestamp=datetime.now(timezone.utc),
            )

        except ImportError:
            logger.warning("pywebpush library not available. Install with: pip install pywebpush")
            return SendResult(
                success=False,
                device_id=device.device_id,
                status=DeliveryStatus.FAILED,
                error="pywebpush not installed",
            )

        except Exception as e:
            error_str = str(e)
            self._record_failure()

            # Check for invalid subscription
            should_unregister = "410" in error_str or "404" in error_str

            logger.error(f"Web Push send error: {e}")
            return SendResult(
                success=False,
                device_id=device.device_id,
                status=DeliveryStatus.FAILED,
                error=error_str,
                should_unregister=should_unregister,
                timestamp=datetime.now(timezone.utc),
            )

    def validate_token(self, token: str) -> bool:
        """Validate Web Push subscription format."""
        try:
            subscription = json.loads(token)
            return bool(
                subscription.get("endpoint")
                and subscription.get("keys", {}).get("p256dh")
                and subscription.get("keys", {}).get("auth")
            )
        except (json.JSONDecodeError, TypeError):
            return False
