"""
Tests for Push Notification Connectors - FCM, APNs, and Web Push.

Tests cover:
- Connector initialization and configuration
- Token validation
- Message building
- Notification sending (single and batch)
- Error handling and retry logic
- Invalid token detection
- Health status
"""

import json
import os
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# FCM Connector Tests
# =============================================================================


class TestFCMConnectorInit:
    """Tests for FCMConnector initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        from aragora.connectors.devices.push import FCMConnector

        connector = FCMConnector()

        assert connector.platform_name == "fcm"
        assert connector.platform_display_name == "Firebase Cloud Messaging"

    def test_init_with_config(self):
        """Should accept custom configuration."""
        from aragora.connectors.devices.base import DeviceConnectorConfig
        from aragora.connectors.devices.push import FCMConnector

        config = DeviceConnectorConfig(
            enable_circuit_breaker=False,
            request_timeout=45.0,
            max_batch_size=100,
        )
        connector = FCMConnector(config=config)

        assert connector.config.enable_circuit_breaker is False
        assert connector.config.request_timeout == 45.0
        assert connector.config.max_batch_size == 100

    def test_supported_device_types(self):
        """Should support Android and Web device types."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import DeviceType

        connector = FCMConnector()

        assert DeviceType.ANDROID in connector.supported_device_types
        assert DeviceType.WEB in connector.supported_device_types

    def test_fcm_endpoint_format(self):
        """Should have correct FCM endpoint format."""
        from aragora.connectors.devices.push import FCMConnector

        assert "{project_id}" in FCMConnector.FCM_ENDPOINT


class TestFCMInitialization:
    """Tests for FCM connector initialization lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_without_project_id(self):
        """Should fail initialization without project ID."""
        from aragora.connectors.devices.push import FCMConnector

        with patch.dict(os.environ, {}, clear=True):
            connector = FCMConnector()
            result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_with_project_id(self):
        """Should initialize with project ID and credentials."""
        from aragora.connectors.devices.push import FCMConnector

        with patch.dict(os.environ, {"FCM_PROJECT_ID": "test-project"}):
            connector = FCMConnector()

            with patch.object(
                connector, "_refresh_access_token", new_callable=AsyncMock
            ) as mock_refresh:
                mock_refresh.return_value = None
                result = await connector.initialize()

        assert result is True
        assert connector._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_token_failure(self):
        """Should fail if token refresh fails."""
        from aragora.connectors.devices.push import FCMConnector

        with patch.dict(os.environ, {"FCM_PROJECT_ID": "test-project"}):
            connector = FCMConnector()

            with patch.object(
                connector,
                "_refresh_access_token",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Token error"),
            ):
                result = await connector.initialize()

        assert result is False


class TestFCMTokenValidation:
    """Tests for FCM token validation."""

    def test_validate_valid_token(self):
        """Should validate correct FCM token format."""
        from aragora.connectors.devices.push import FCMConnector

        connector = FCMConnector()

        # FCM tokens are typically 150+ characters
        valid_token = "a" * 150

        assert connector.validate_token(valid_token) is True

    def test_validate_short_token(self):
        """Should reject tokens that are too short."""
        from aragora.connectors.devices.push import FCMConnector

        connector = FCMConnector()

        short_token = "a" * 50

        assert connector.validate_token(short_token) is False

    def test_validate_empty_token(self):
        """Should reject empty tokens."""
        from aragora.connectors.devices.push import FCMConnector

        connector = FCMConnector()

        assert connector.validate_token("") is False
        assert connector.validate_token(None) is False  # type: ignore


class TestFCMMessageBuilding:
    """Tests for FCM message payload building."""

    def test_build_basic_message(self):
        """Should build basic FCM message."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
        )

        connector = FCMConnector()

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.ANDROID,
            push_token="test-token",
        )
        message = DeviceMessage(title="Test", body="Test body")

        result = connector._build_fcm_message(device, message)

        assert result["token"] == "test-token"
        assert result["notification"]["title"] == "Test"
        assert result["notification"]["body"] == "Test body"

    def test_build_message_with_image(self):
        """Should include image in message."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
        )

        connector = FCMConnector()

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.ANDROID,
            push_token="token",
        )
        message = DeviceMessage(
            title="Test",
            body="Body",
            image_url="https://example.com/image.png",
        )

        result = connector._build_fcm_message(device, message)

        assert result["notification"]["image"] == "https://example.com/image.png"

    def test_build_message_with_data(self):
        """Should include data payload."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
        )

        connector = FCMConnector()

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.ANDROID,
            push_token="token",
        )
        message = DeviceMessage(
            title="Test",
            body="Body",
            data={"debate_id": "123", "type": "decision"},
        )

        result = connector._build_fcm_message(device, message)

        assert result["data"]["debate_id"] == "123"
        assert result["data"]["type"] == "decision"

    def test_build_message_with_action_url(self):
        """Should include action URL in data."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
        )

        connector = FCMConnector()

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.ANDROID,
            push_token="token",
        )
        message = DeviceMessage(
            title="Test",
            body="Body",
            action_url="https://example.com/debate/123",
        )

        result = connector._build_fcm_message(device, message)

        assert result["data"]["action_url"] == "https://example.com/debate/123"

    def test_build_android_specific_message(self):
        """Should include Android-specific configuration."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            NotificationPriority,
        )

        connector = FCMConnector()

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.ANDROID,
            push_token="token",
        )
        message = DeviceMessage(
            title="Test",
            body="Body",
            priority=NotificationPriority.HIGH,
            ttl_seconds=7200,
            channel_id="debate-notifications",
            collapse_key="debate-updates",
        )

        result = connector._build_fcm_message(device, message)

        assert result["android"]["priority"] == "high"
        assert result["android"]["ttl"] == "7200s"
        assert result["android"]["notification"]["channel_id"] == "debate-notifications"
        assert result["android"]["collapse_key"] == "debate-updates"

    def test_build_web_push_message(self):
        """Should include web push configuration."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
        )

        connector = FCMConnector()

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.WEB,
            push_token="token",
        )
        message = DeviceMessage(
            title="Test",
            body="Body",
            ttl_seconds=3600,
            action_url="https://example.com/action",
        )

        result = connector._build_fcm_message(device, message)

        assert result["webpush"]["headers"]["TTL"] == "3600"
        assert result["webpush"]["fcm_options"]["link"] == "https://example.com/action"


class TestFCMPriorityMapping:
    """Tests for priority mapping."""

    def test_map_high_priority(self):
        """Should map HIGH priority to 'high'."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import NotificationPriority

        connector = FCMConnector()

        assert connector._map_priority(NotificationPriority.HIGH) == "high"
        assert connector._map_priority(NotificationPriority.URGENT) == "high"

    def test_map_normal_priority(self):
        """Should map NORMAL and LOW priority to 'normal'."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import NotificationPriority

        connector = FCMConnector()

        assert connector._map_priority(NotificationPriority.NORMAL) == "normal"
        assert connector._map_priority(NotificationPriority.LOW) == "normal"


class TestFCMSendNotification:
    """Tests for sending notifications."""

    @pytest.mark.asyncio
    async def test_send_notification_not_initialized(self):
        """Should fail when not initialized."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = FCMConnector()
        connector._initialized = False

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.ANDROID,
            push_token="token",
        )
        message = DeviceMessage(title="Test", body="Body")

        result = await connector.send_notification(device, message)

        assert result.success is False
        assert result.status == DeliveryStatus.FAILED
        assert "not initialized" in result.error

    @pytest.mark.asyncio
    async def test_send_notification_success(self):
        """Should send notification successfully."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = FCMConnector()
        connector._initialized = True
        connector._project_id = "test-project"
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.ANDROID,
            push_token="device-token",
        )
        message = DeviceMessage(title="Test", body="Body")

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {"name": "projects/test/messages/msg123"}, None)

            result = await connector.send_notification(device, message)

        assert result.success is True
        assert result.status == DeliveryStatus.SENT
        assert result.message_id == "msg123"

    @pytest.mark.asyncio
    async def test_send_notification_failure(self):
        """Should handle notification failure."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = FCMConnector()
        connector._initialized = True
        connector._project_id = "test-project"
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.ANDROID,
            push_token="token",
        )
        message = DeviceMessage(title="Test", body="Body")

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (False, None, "API Error")

            result = await connector.send_notification(device, message)

        assert result.success is False
        assert result.status == DeliveryStatus.FAILED

    @pytest.mark.asyncio
    async def test_send_notification_invalid_token(self):
        """Should detect invalid token errors."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
        )

        connector = FCMConnector()
        connector._initialized = True
        connector._project_id = "test-project"
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.ANDROID,
            push_token="invalid-token",
        )
        message = DeviceMessage(title="Test", body="Body")

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (False, None, "UNREGISTERED")

            result = await connector.send_notification(device, message)

        assert result.should_unregister is True


class TestFCMInvalidTokenDetection:
    """Tests for invalid token error detection."""

    def test_detect_unregistered_error(self):
        """Should detect UNREGISTERED error."""
        from aragora.connectors.devices.push import FCMConnector

        connector = FCMConnector()

        assert connector._is_invalid_token_error("UNREGISTERED") is True

    def test_detect_invalid_argument_error(self):
        """Should detect INVALID_ARGUMENT error."""
        from aragora.connectors.devices.push import FCMConnector

        connector = FCMConnector()

        assert connector._is_invalid_token_error("INVALID_ARGUMENT") is True

    def test_detect_not_found_error(self):
        """Should detect NOT_FOUND error."""
        from aragora.connectors.devices.push import FCMConnector

        connector = FCMConnector()

        assert connector._is_invalid_token_error("NOT_FOUND") is True

    def test_detect_not_registered_error(self):
        """Should detect NotRegistered error."""
        from aragora.connectors.devices.push import FCMConnector

        connector = FCMConnector()

        assert connector._is_invalid_token_error("NotRegistered") is True

    def test_not_invalid_token_error(self):
        """Should not flag other errors as invalid token."""
        from aragora.connectors.devices.push import FCMConnector

        connector = FCMConnector()

        assert connector._is_invalid_token_error("rate_limited") is False
        assert connector._is_invalid_token_error("server_error") is False


class TestFCMBatchSend:
    """Tests for batch notification sending."""

    @pytest.mark.asyncio
    async def test_send_batch_not_initialized(self):
        """Should fail batch send when not initialized."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import DeviceToken, DeviceType, DeviceMessage

        connector = FCMConnector()
        connector._initialized = False

        devices = [
            DeviceToken(
                device_id="d1",
                user_id="u1",
                device_type=DeviceType.ANDROID,
                push_token="token1",
            ),
        ]
        message = DeviceMessage(title="Test", body="Body")

        result = await connector.send_batch(devices, message)

        assert result.total_sent == 0
        assert result.failure_count == 1

    @pytest.mark.asyncio
    async def test_send_batch_success(self):
        """Should send batch notifications successfully."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import DeviceToken, DeviceType, DeviceMessage

        connector = FCMConnector()
        connector._initialized = True
        connector._project_id = "test-project"
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        devices = [
            DeviceToken(
                device_id=f"d{i}",
                user_id="u1",
                device_type=DeviceType.ANDROID,
                push_token=f"token{i}",
            )
            for i in range(3)
        ]
        message = DeviceMessage(title="Test", body="Body")

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {"name": "projects/test/messages/msg"}, None)

            result = await connector.send_batch(devices, message)

        assert result.total_sent == 3
        assert result.success_count == 3
        assert result.failure_count == 0

    @pytest.mark.asyncio
    async def test_send_batch_collects_invalid_tokens(self):
        """Should collect invalid tokens for removal."""
        from aragora.connectors.devices.push import FCMConnector
        from aragora.connectors.devices.models import DeviceToken, DeviceType, DeviceMessage

        connector = FCMConnector()
        connector._initialized = True
        connector._project_id = "test-project"
        connector._access_token = "test-token"
        connector._token_expires_at = 9999999999

        devices = [
            DeviceToken(
                device_id="d1",
                user_id="u1",
                device_type=DeviceType.ANDROID,
                push_token="valid-token",
            ),
            DeviceToken(
                device_id="d2",
                user_id="u1",
                device_type=DeviceType.ANDROID,
                push_token="invalid-token",
            ),
        ]
        message = DeviceMessage(title="Test", body="Body")

        call_count = 0

        async def mock_http(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (True, {"name": "projects/test/messages/msg"}, None)
            return (False, None, "UNREGISTERED")

        with patch.object(connector, "_http_request", mock_http):
            result = await connector.send_batch(devices, message)

        assert "d2" in result.tokens_to_remove


# =============================================================================
# APNs Connector Tests
# =============================================================================


class TestAPNsConnectorInit:
    """Tests for APNsConnector initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector()

        assert connector.platform_name == "apns"
        assert connector.platform_display_name == "Apple Push Notification service"

    def test_init_with_sandbox(self):
        """Should initialize with sandbox mode."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector(use_sandbox=True)

        assert connector._use_sandbox is True
        assert connector.base_url == APNsConnector.APNS_SANDBOX

    def test_init_with_production(self):
        """Should initialize with production mode by default."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector(use_sandbox=False)

        assert connector._use_sandbox is False
        assert connector.base_url == APNsConnector.APNS_PRODUCTION

    def test_supported_device_types(self):
        """Should support iOS device type."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import DeviceType

        connector = APNsConnector()

        assert DeviceType.IOS in connector.supported_device_types


class TestAPNsInitialization:
    """Tests for APNs connector initialization lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_without_credentials(self):
        """Should fail initialization without credentials."""
        from aragora.connectors.devices.push import APNsConnector

        with patch.dict(os.environ, {}, clear=True):
            connector = APNsConnector()
            result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_with_partial_credentials(self):
        """Should fail with only partial credentials."""
        from aragora.connectors.devices.push import APNsConnector

        with patch.dict(
            os.environ,
            {
                "APNS_KEY_ID": "key123",
                # Missing other credentials
            },
            clear=True,
        ):
            connector = APNsConnector()
            result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Should initialize with valid credentials."""
        from aragora.connectors.devices.push import APNsConnector

        with patch.dict(
            os.environ,
            {
                "APNS_KEY_ID": "key123",
                "APNS_TEAM_ID": "team123",
                "APNS_BUNDLE_ID": "com.example.app",
                "APNS_PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----",
            },
        ):
            connector = APNsConnector()

            with patch.object(connector, "_refresh_jwt_token") as mock_refresh:
                mock_refresh.return_value = None
                result = await connector.initialize()

        assert result is True
        assert connector._initialized is True


class TestAPNsTokenValidation:
    """Tests for APNs token validation."""

    def test_validate_valid_token(self):
        """Should validate correct APNs token format (64 hex chars)."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector()

        # APNs tokens are 64 hex characters
        valid_token = "a" * 64

        assert connector.validate_token(valid_token) is True

    def test_validate_wrong_length_token(self):
        """Should reject tokens with wrong length."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector()

        assert connector.validate_token("a" * 63) is False
        assert connector.validate_token("a" * 65) is False

    def test_validate_non_hex_token(self):
        """Should reject non-hex tokens."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector()

        # 'g' is not a valid hex character
        invalid_token = "g" * 64

        assert connector.validate_token(invalid_token) is False

    def test_validate_empty_token(self):
        """Should reject empty tokens."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector()

        assert connector.validate_token("") is False


class TestAPNsPayloadBuilding:
    """Tests for APNs payload building."""

    def test_build_basic_payload(self):
        """Should build basic APNs payload."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import DeviceMessage

        connector = APNsConnector()

        message = DeviceMessage(title="Test", body="Test body")

        result = connector._build_apns_payload(message)

        assert result["aps"]["alert"]["title"] == "Test"
        assert result["aps"]["alert"]["body"] == "Test body"

    def test_build_payload_with_badge(self):
        """Should include badge count."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import DeviceMessage

        connector = APNsConnector()

        message = DeviceMessage(title="Test", body="Body", badge=5)

        result = connector._build_apns_payload(message)

        assert result["aps"]["badge"] == 5

    def test_build_payload_with_sound(self):
        """Should include sound."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import DeviceMessage

        connector = APNsConnector()

        message = DeviceMessage(title="Test", body="Body", sound="notification.wav")

        result = connector._build_apns_payload(message)

        assert result["aps"]["sound"] == "notification.wav"

    def test_build_payload_with_thread_id(self):
        """Should include thread ID."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import DeviceMessage

        connector = APNsConnector()

        message = DeviceMessage(title="Test", body="Body", thread_id="debate-123")

        result = connector._build_apns_payload(message)

        assert result["aps"]["thread-id"] == "debate-123"

    def test_build_payload_with_mutable_content(self):
        """Should include mutable content flag."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import DeviceMessage

        connector = APNsConnector()

        message = DeviceMessage(title="Test", body="Body", mutable_content=True)

        result = connector._build_apns_payload(message)

        assert result["aps"]["mutable-content"] == 1

    def test_build_payload_with_custom_data(self):
        """Should include custom data at top level."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import DeviceMessage

        connector = APNsConnector()

        message = DeviceMessage(
            title="Test",
            body="Body",
            data={"debate_id": "123", "type": "decision"},
        )

        result = connector._build_apns_payload(message)

        assert result["debate_id"] == "123"
        assert result["type"] == "decision"

    def test_build_payload_with_action_url(self):
        """Should include action URL."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import DeviceMessage

        connector = APNsConnector()

        message = DeviceMessage(
            title="Test",
            body="Body",
            action_url="https://example.com/debate/123",
        )

        result = connector._build_apns_payload(message)

        assert result["action_url"] == "https://example.com/debate/123"


class TestAPNsPriorityMapping:
    """Tests for APNs priority mapping."""

    def test_map_low_priority(self):
        """Should map LOW priority to '5'."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import NotificationPriority

        connector = APNsConnector()

        assert connector._map_priority(NotificationPriority.LOW) == "5"

    def test_map_high_priority(self):
        """Should map HIGH and URGENT priority to '10'."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import NotificationPriority

        connector = APNsConnector()

        assert connector._map_priority(NotificationPriority.HIGH) == "10"
        assert connector._map_priority(NotificationPriority.URGENT) == "10"

    def test_map_normal_priority(self):
        """Should map NORMAL priority to '10' (default for alerts)."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import NotificationPriority

        connector = APNsConnector()

        assert connector._map_priority(NotificationPriority.NORMAL) == "10"


class TestAPNsSendNotification:
    """Tests for APNs notification sending."""

    @pytest.mark.asyncio
    async def test_send_notification_not_initialized(self):
        """Should fail when not initialized."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = APNsConnector()
        connector._initialized = False

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.IOS,
            push_token="a" * 64,
        )
        message = DeviceMessage(title="Test", body="Body")

        result = await connector.send_notification(device, message)

        assert result.success is False
        assert result.status == DeliveryStatus.FAILED
        assert "not initialized" in result.error

    @pytest.mark.asyncio
    async def test_send_notification_success(self):
        """Should send notification successfully."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = APNsConnector()
        connector._initialized = True
        connector._bundle_id = "com.example.app"
        connector._jwt_token = "test-jwt"
        connector._token_issued_at = 9999999999

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.IOS,
            push_token="a" * 64,
        )
        message = DeviceMessage(title="Test", body="Body")

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {"apns-id": "msg123"}, None)

            result = await connector.send_notification(device, message)

        assert result.success is True
        assert result.status == DeliveryStatus.SENT

    @pytest.mark.asyncio
    async def test_send_notification_includes_correct_headers(self):
        """Should include correct APNs headers."""
        from aragora.connectors.devices.push import APNsConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            NotificationPriority,
        )

        connector = APNsConnector()
        connector._initialized = True
        connector._bundle_id = "com.example.app"
        connector._jwt_token = "test-jwt"
        connector._token_issued_at = 9999999999

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.IOS,
            push_token="a" * 64,
        )
        message = DeviceMessage(
            title="Test",
            body="Body",
            priority=NotificationPriority.HIGH,
            collapse_key="updates",
        )

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (True, {}, None)

            await connector.send_notification(device, message)

            call_kwargs = mock_http.call_args[1]
            headers = call_kwargs["headers"]

            assert headers["apns-topic"] == "com.example.app"
            assert headers["apns-push-type"] == "alert"
            assert headers["apns-priority"] == "10"
            assert headers["apns-collapse-id"] == "updates"


class TestAPNsInvalidTokenDetection:
    """Tests for APNs invalid token error detection."""

    def test_detect_bad_device_token(self):
        """Should detect BadDeviceToken error."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector()

        assert connector._is_invalid_token_error("BadDeviceToken") is True

    def test_detect_unregistered_error(self):
        """Should detect Unregistered error."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector()

        assert connector._is_invalid_token_error("Unregistered") is True

    def test_detect_expired_token(self):
        """Should detect ExpiredToken error."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector()

        assert connector._is_invalid_token_error("ExpiredToken") is True

    def test_detect_topic_error(self):
        """Should detect topic-related errors."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector()

        assert connector._is_invalid_token_error("DeviceTokenNotForTopic") is True
        assert connector._is_invalid_token_error("TopicDisallowed") is True

    def test_not_invalid_token_error(self):
        """Should not flag other errors as invalid token."""
        from aragora.connectors.devices.push import APNsConnector

        connector = APNsConnector()

        assert connector._is_invalid_token_error("TooManyRequests") is False
        assert connector._is_invalid_token_error("InternalServerError") is False


# =============================================================================
# Web Push Connector Tests
# =============================================================================


class TestWebPushConnectorInit:
    """Tests for WebPushConnector initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        from aragora.connectors.devices.push import WebPushConnector

        connector = WebPushConnector()

        assert connector.platform_name == "web_push"
        assert connector.platform_display_name == "Web Push (VAPID)"

    def test_supported_device_types(self):
        """Should support Web device type."""
        from aragora.connectors.devices.push import WebPushConnector
        from aragora.connectors.devices.models import DeviceType

        connector = WebPushConnector()

        assert DeviceType.WEB in connector.supported_device_types


class TestWebPushInitialization:
    """Tests for Web Push connector initialization."""

    @pytest.mark.asyncio
    async def test_initialize_without_credentials(self):
        """Should fail initialization without credentials."""
        from aragora.connectors.devices.push import WebPushConnector

        with patch.dict(os.environ, {}, clear=True):
            connector = WebPushConnector()
            result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_with_partial_credentials(self):
        """Should fail with only partial credentials."""
        from aragora.connectors.devices.push import WebPushConnector

        with patch.dict(
            os.environ,
            {
                "VAPID_PUBLIC_KEY": "public-key",
                # Missing private key and subject
            },
            clear=True,
        ):
            connector = WebPushConnector()
            result = await connector.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Should initialize with valid credentials."""
        from aragora.connectors.devices.push import WebPushConnector

        with patch.dict(
            os.environ,
            {
                "VAPID_PUBLIC_KEY": "public-key",
                "VAPID_PRIVATE_KEY": "private-key",
                "VAPID_SUBJECT": "mailto:test@example.com",
            },
        ):
            connector = WebPushConnector()
            result = await connector.initialize()

        assert result is True
        assert connector._initialized is True


class TestWebPushTokenValidation:
    """Tests for Web Push subscription validation."""

    def test_validate_valid_subscription(self):
        """Should validate correct Web Push subscription format."""
        from aragora.connectors.devices.push import WebPushConnector

        connector = WebPushConnector()

        valid_subscription = json.dumps(
            {
                "endpoint": "https://fcm.googleapis.com/fcm/send/...",
                "keys": {
                    "p256dh": "base64-encoded-key",
                    "auth": "base64-encoded-auth",
                },
            }
        )

        assert connector.validate_token(valid_subscription) is True

    def test_validate_missing_endpoint(self):
        """Should reject subscription without endpoint."""
        from aragora.connectors.devices.push import WebPushConnector

        connector = WebPushConnector()

        invalid_subscription = json.dumps(
            {
                "keys": {
                    "p256dh": "key",
                    "auth": "auth",
                },
            }
        )

        assert connector.validate_token(invalid_subscription) is False

    def test_validate_missing_keys(self):
        """Should reject subscription without keys."""
        from aragora.connectors.devices.push import WebPushConnector

        connector = WebPushConnector()

        invalid_subscription = json.dumps(
            {
                "endpoint": "https://example.com/push",
            }
        )

        assert connector.validate_token(invalid_subscription) is False

    def test_validate_missing_p256dh(self):
        """Should reject subscription without p256dh key."""
        from aragora.connectors.devices.push import WebPushConnector

        connector = WebPushConnector()

        invalid_subscription = json.dumps(
            {
                "endpoint": "https://example.com/push",
                "keys": {
                    "auth": "auth",
                },
            }
        )

        assert connector.validate_token(invalid_subscription) is False

    def test_validate_invalid_json(self):
        """Should reject invalid JSON."""
        from aragora.connectors.devices.push import WebPushConnector

        connector = WebPushConnector()

        assert connector.validate_token("not-valid-json") is False
        assert connector.validate_token("") is False


class TestWebPushSendNotification:
    """Tests for Web Push notification sending."""

    @pytest.mark.asyncio
    async def test_send_notification_not_initialized(self):
        """Should fail when not initialized."""
        from aragora.connectors.devices.push import WebPushConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = WebPushConnector()
        connector._initialized = False

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.WEB,
            push_token=json.dumps({"endpoint": "test", "keys": {"p256dh": "k", "auth": "a"}}),
        )
        message = DeviceMessage(title="Test", body="Body")

        result = await connector.send_notification(device, message)

        assert result.success is False
        assert result.status == DeliveryStatus.FAILED
        assert "not initialized" in result.error

    @pytest.mark.asyncio
    async def test_send_notification_invalid_subscription(self):
        """Should fail with invalid subscription format."""
        from aragora.connectors.devices.push import WebPushConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = WebPushConnector()
        connector._initialized = True
        connector._vapid_public = "public"
        connector._vapid_private = "private"
        connector._vapid_subject = "mailto:test@test.com"

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.WEB,
            push_token="not-valid-json",
        )
        message = DeviceMessage(title="Test", body="Body")

        result = await connector.send_notification(device, message)

        assert result.success is False
        assert result.should_unregister is True
        assert "Invalid subscription" in result.error

    @pytest.mark.asyncio
    async def test_send_notification_success(self):
        """Should send notification successfully."""
        from aragora.connectors.devices.push import WebPushConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = WebPushConnector()
        connector._initialized = True
        connector._vapid_public = "public"
        connector._vapid_private = "private"
        connector._vapid_subject = "mailto:test@test.com"

        subscription = json.dumps(
            {
                "endpoint": "https://fcm.googleapis.com/fcm/send/test",
                "keys": {
                    "p256dh": "base64key",
                    "auth": "base64auth",
                },
            }
        )

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.WEB,
            push_token=subscription,
        )
        message = DeviceMessage(title="Test", body="Body")

        with patch("aragora.connectors.devices.push.webpush") as mock_webpush:
            mock_webpush.return_value = None

            result = await connector.send_notification(device, message)

        assert result.success is True
        assert result.status == DeliveryStatus.SENT

    @pytest.mark.asyncio
    async def test_send_notification_with_image_and_url(self):
        """Should include image and URL in payload."""
        from aragora.connectors.devices.push import WebPushConnector
        from aragora.connectors.devices.models import DeviceToken, DeviceType, DeviceMessage

        connector = WebPushConnector()
        connector._initialized = True
        connector._vapid_public = "public"
        connector._vapid_private = "private"
        connector._vapid_subject = "mailto:test@test.com"

        subscription = json.dumps(
            {
                "endpoint": "https://example.com/push",
                "keys": {"p256dh": "key", "auth": "auth"},
            }
        )

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.WEB,
            push_token=subscription,
        )
        message = DeviceMessage(
            title="Test",
            body="Body",
            image_url="https://example.com/image.png",
            action_url="https://example.com/action",
        )

        with patch("aragora.connectors.devices.push.webpush") as mock_webpush:
            mock_webpush.return_value = None

            await connector.send_notification(device, message)

            call_kwargs = mock_webpush.call_args[1]
            payload = json.loads(call_kwargs["data"])

            assert payload["image"] == "https://example.com/image.png"
            assert payload["url"] == "https://example.com/action"

    @pytest.mark.asyncio
    async def test_send_notification_gone_subscription(self):
        """Should detect gone (410) subscriptions for removal."""
        from aragora.connectors.devices.push import WebPushConnector
        from aragora.connectors.devices.models import DeviceToken, DeviceType, DeviceMessage

        connector = WebPushConnector()
        connector._initialized = True
        connector._vapid_public = "public"
        connector._vapid_private = "private"
        connector._vapid_subject = "mailto:test@test.com"

        subscription = json.dumps(
            {
                "endpoint": "https://example.com/push",
                "keys": {"p256dh": "key", "auth": "auth"},
            }
        )

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.WEB,
            push_token=subscription,
        )
        message = DeviceMessage(title="Test", body="Body")

        with patch("aragora.connectors.devices.push.webpush") as mock_webpush:
            mock_webpush.side_effect = RuntimeError("410 Gone")

            result = await connector.send_notification(device, message)

        assert result.success is False
        assert result.should_unregister is True

    @pytest.mark.asyncio
    async def test_send_notification_not_found_subscription(self):
        """Should detect not found (404) subscriptions for removal."""
        from aragora.connectors.devices.push import WebPushConnector
        from aragora.connectors.devices.models import DeviceToken, DeviceType, DeviceMessage

        connector = WebPushConnector()
        connector._initialized = True
        connector._vapid_public = "public"
        connector._vapid_private = "private"
        connector._vapid_subject = "mailto:test@test.com"

        subscription = json.dumps(
            {
                "endpoint": "https://example.com/push",
                "keys": {"p256dh": "key", "auth": "auth"},
            }
        )

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.WEB,
            push_token=subscription,
        )
        message = DeviceMessage(title="Test", body="Body")

        with patch("aragora.connectors.devices.push.webpush") as mock_webpush:
            mock_webpush.side_effect = RuntimeError("404 Not Found")

            result = await connector.send_notification(device, message)

        assert result.success is False
        assert result.should_unregister is True

    @pytest.mark.asyncio
    async def test_send_notification_pywebpush_not_installed(self):
        """Should handle missing pywebpush library."""
        from aragora.connectors.devices.push import WebPushConnector
        from aragora.connectors.devices.models import (
            DeviceToken,
            DeviceType,
            DeviceMessage,
            DeliveryStatus,
        )

        connector = WebPushConnector()
        connector._initialized = True
        connector._vapid_public = "public"
        connector._vapid_private = "private"
        connector._vapid_subject = "mailto:test@test.com"

        subscription = json.dumps(
            {
                "endpoint": "https://example.com/push",
                "keys": {"p256dh": "key", "auth": "auth"},
            }
        )

        device = DeviceToken(
            device_id="d1",
            user_id="u1",
            device_type=DeviceType.WEB,
            push_token=subscription,
        )
        message = DeviceMessage(title="Test", body="Body")

        with patch.dict("sys.modules", {"pywebpush": None}):
            with patch(
                "aragora.connectors.devices.push.webpush",
                side_effect=ImportError("No module named 'pywebpush'"),
            ):
                result = await connector.send_notification(device, message)

        assert result.success is False
        assert result.status == DeliveryStatus.FAILED
        assert "pywebpush" in result.error
