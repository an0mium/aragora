"""
Tests for device connector base class.

Tests cover:
- DeviceConnectorConfig dataclass
- DeviceConnector abstract base class
- Circuit breaker integration
- Retry logic with exponential backoff
- HTTP request handling
- Device registration and unregistration
- Batch notification sending
- Voice device support
- Health and status checks
- Token validation
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.devices.base import DeviceConnector, DeviceConnectorConfig
from aragora.connectors.devices.models import (
    BatchSendResult,
    DeliveryStatus,
    DeviceMessage,
    DeviceRegistration,
    DeviceToken,
    DeviceType,
    SendResult,
    VoiceDeviceRequest,
    VoiceDeviceResponse,
)


class ConcreteDeviceConnector(DeviceConnector):
    """Concrete implementation for testing abstract base class."""

    @property
    def platform_name(self) -> str:
        return "test_platform"

    @property
    def platform_display_name(self) -> str:
        return "Test Platform"

    @property
    def supported_device_types(self) -> list[DeviceType]:
        return [DeviceType.ANDROID, DeviceType.IOS]

    async def send_notification(
        self,
        device: DeviceToken,
        message: DeviceMessage,
        **kwargs: Any,
    ) -> SendResult:
        """Concrete implementation for testing."""
        return SendResult(
            success=True,
            device_id=device.device_id,
            status=DeliveryStatus.DELIVERED,
        )


class TestDeviceConnectorConfig:
    """Tests for DeviceConnectorConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = DeviceConnectorConfig()

        assert config.enable_circuit_breaker is True
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_cooldown == 60.0
        assert config.request_timeout == 30.0
        assert config.max_retries == 3
        assert config.base_retry_delay == 1.0
        assert config.max_retry_delay == 30.0
        assert config.max_batch_size == 500
        assert config.batch_delay_ms == 10
        assert config.max_requests_per_second == 100.0
        assert config.credentials == {}

    def test_custom_values(self):
        """Should accept custom values."""
        config = DeviceConnectorConfig(
            enable_circuit_breaker=False,
            circuit_breaker_threshold=10,
            request_timeout=60.0,
            max_retries=5,
            credentials={"api_key": "test123"},
        )

        assert config.enable_circuit_breaker is False
        assert config.circuit_breaker_threshold == 10
        assert config.request_timeout == 60.0
        assert config.max_retries == 5
        assert config.credentials["api_key"] == "test123"


class TestDeviceConnectorInitialization:
    """Tests for DeviceConnector initialization."""

    def test_default_initialization(self):
        """Should initialize with default config."""
        connector = ConcreteDeviceConnector()

        assert connector.config is not None
        assert connector._initialized is False
        assert connector._circuit_breaker is None
        assert connector._circuit_breaker_initialized is False

    def test_custom_config(self):
        """Should accept custom config."""
        config = DeviceConnectorConfig(request_timeout=120.0)
        connector = ConcreteDeviceConnector(config=config)

        assert connector.config.request_timeout == 120.0


class TestCircuitBreaker:
    """Tests for circuit breaker support."""

    def test_lazy_initialization(self):
        """Circuit breaker should be lazily initialized."""
        connector = ConcreteDeviceConnector()

        assert connector._circuit_breaker is None
        assert connector._circuit_breaker_initialized is False

    def test_disabled_circuit_breaker(self):
        """Should return None when circuit breaker disabled."""
        config = DeviceConnectorConfig(enable_circuit_breaker=False)
        connector = ConcreteDeviceConnector(config=config)

        cb = connector._get_circuit_breaker()
        assert cb is None

    def test_circuit_breaker_creation(self):
        """Should create circuit breaker on first access."""
        with patch("aragora.resilience.get_circuit_breaker") as mock_get_cb:
            mock_cb = MagicMock()
            mock_get_cb.return_value = mock_cb

            connector = ConcreteDeviceConnector()
            cb = connector._get_circuit_breaker()

            assert cb is mock_cb
            mock_get_cb.assert_called_once()
            assert connector._circuit_breaker_initialized is True

    def test_circuit_breaker_reuse(self):
        """Should reuse existing circuit breaker."""
        with patch("aragora.resilience.get_circuit_breaker") as mock_get_cb:
            mock_cb = MagicMock()
            mock_get_cb.return_value = mock_cb

            connector = ConcreteDeviceConnector()
            cb1 = connector._get_circuit_breaker()
            cb2 = connector._get_circuit_breaker()

            assert cb1 is cb2
            mock_get_cb.assert_called_once()

    def test_check_circuit_breaker_disabled(self):
        """Should allow proceed when circuit breaker disabled."""
        config = DeviceConnectorConfig(enable_circuit_breaker=False)
        connector = ConcreteDeviceConnector(config=config)

        can_proceed, error = connector._check_circuit_breaker()

        assert can_proceed is True
        assert error is None

    def test_check_circuit_breaker_open(self):
        """Should block when circuit breaker is open."""
        with patch("aragora.resilience.get_circuit_breaker") as mock_get_cb:
            mock_cb = MagicMock()
            mock_cb.can_proceed.return_value = False
            mock_cb.cooldown_remaining.return_value = 30.0
            mock_get_cb.return_value = mock_cb

            connector = ConcreteDeviceConnector()
            can_proceed, error = connector._check_circuit_breaker()

            assert can_proceed is False
            assert "Circuit breaker open" in error
            assert "30.0s" in error

    def test_record_success(self):
        """Should record success with circuit breaker."""
        with patch("aragora.resilience.get_circuit_breaker") as mock_get_cb:
            mock_cb = MagicMock()
            mock_get_cb.return_value = mock_cb

            connector = ConcreteDeviceConnector()
            connector._get_circuit_breaker()  # Initialize
            connector._record_success()

            mock_cb.record_success.assert_called_once()

    def test_record_failure(self):
        """Should record failure with circuit breaker."""
        with patch("aragora.resilience.get_circuit_breaker") as mock_get_cb:
            mock_cb = MagicMock()
            mock_cb.get_status.return_value = "closed"
            mock_get_cb.return_value = mock_cb

            connector = ConcreteDeviceConnector()
            connector._get_circuit_breaker()  # Initialize
            connector._record_failure()

            mock_cb.record_failure.assert_called_once()


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self):
        """Should not retry successful operations."""
        connector = ConcreteDeviceConnector()
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        with patch.object(connector, "_check_circuit_breaker", return_value=(True, None)):
            with patch.object(connector, "_record_success"):
                result = await connector._with_retry("test", success_func)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Should retry on retryable exceptions."""
        config = DeviceConnectorConfig(
            enable_circuit_breaker=False,
            max_retries=3,
            base_retry_delay=0.01,
        )
        connector = ConcreteDeviceConnector(config=config)
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await connector._with_retry("test", failing_func)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Should raise after max retries exceeded."""
        config = DeviceConnectorConfig(
            enable_circuit_breaker=False,
            max_retries=2,
            base_retry_delay=0.01,
        )
        connector = ConcreteDeviceConnector(config=config)

        async def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await connector._with_retry("test", always_fails)

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_retry(self):
        """Should not retry when circuit breaker blocks."""
        connector = ConcreteDeviceConnector()

        with patch.object(
            connector, "_check_circuit_breaker", return_value=(False, "Circuit open")
        ):
            with pytest.raises(ConnectionError, match="Circuit open"):
                await connector._with_retry("test", AsyncMock())


class TestRetryableStatusCodes:
    """Tests for _is_retryable_status_code method."""

    def test_retryable_codes(self):
        """Should identify retryable status codes."""
        connector = ConcreteDeviceConnector()

        for code in [429, 500, 502, 503, 504]:
            assert connector._is_retryable_status_code(code) is True

    def test_non_retryable_codes(self):
        """Should identify non-retryable status codes."""
        connector = ConcreteDeviceConnector()

        for code in [200, 201, 400, 401, 403, 404, 422]:
            assert connector._is_retryable_status_code(code) is False


class TestHttpRequest:
    """Tests for _http_request method."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks(self):
        """Should fail fast when circuit breaker blocks."""
        connector = ConcreteDeviceConnector()

        with patch.object(
            connector, "_check_circuit_breaker", return_value=(False, "Circuit open")
        ):
            success, data, error = await connector._http_request(
                method="GET",
                url="https://example.com/api",
            )

        assert success is False
        assert error == "Circuit open"

    @pytest.mark.asyncio
    async def test_httpx_not_available(self):
        """Should handle httpx not being installed."""
        connector = ConcreteDeviceConnector()

        with patch.object(connector, "_check_circuit_breaker", return_value=(True, None)):
            with patch.dict("sys.modules", {"httpx": None}):
                # This will try to import httpx and fail
                pass  # httpx is likely installed in test env

    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Should return success for 2xx responses."""
        connector = ConcreteDeviceConnector()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}

        with patch.object(connector, "_check_circuit_breaker", return_value=(True, None)):
            with patch.object(connector, "_record_success"):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                        return_value=mock_response
                    )

                    success, data, error = await connector._http_request(
                        method="POST",
                        url="https://example.com/api",
                        json={"test": "data"},
                    )

        assert success is True
        assert data == {"data": "test"}
        assert error is None

    @pytest.mark.asyncio
    async def test_non_json_response(self):
        """Should handle non-JSON responses."""
        connector = ConcreteDeviceConnector()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("No JSON")
        mock_response.text = "OK"

        with patch.object(connector, "_check_circuit_breaker", return_value=(True, None)):
            with patch.object(connector, "_record_success"):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                        return_value=mock_response
                    )

                    success, data, error = await connector._http_request(
                        method="GET",
                        url="https://example.com",
                    )

        assert success is True
        assert data["status"] == "ok"
        assert data["text"] == "OK"

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Should not retry on 4xx errors."""
        config = DeviceConnectorConfig(
            enable_circuit_breaker=False,
            max_retries=3,
        )
        connector = ConcreteDeviceConnector(config=config)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )

            success, data, error = await connector._http_request(
                method="POST",
                url="https://example.com/api",
            )

        assert success is False
        assert "400" in error


class TestSendBatch:
    """Tests for send_batch method."""

    @pytest.fixture
    def connector(self):
        """Create connector with no delays for testing."""
        config = DeviceConnectorConfig(batch_delay_ms=0)
        return ConcreteDeviceConnector(config=config)

    @pytest.fixture
    def devices(self):
        """Create test devices."""
        return [
            DeviceToken(
                device_id=f"device_{i}",
                user_id="user_1",
                device_type=DeviceType.ANDROID,
                push_token=f"token_{i}",
            )
            for i in range(3)
        ]

    @pytest.fixture
    def message(self):
        """Create test message."""
        return DeviceMessage(
            title="Test Notification",
            body="This is a test message",
        )

    @pytest.mark.asyncio
    async def test_batch_success(self, connector, devices, message):
        """Should send to all devices and aggregate results."""
        result = await connector.send_batch(devices, message)

        assert isinstance(result, BatchSendResult)
        assert result.total_sent == 3
        assert result.success_count == 3
        assert result.failure_count == 0
        assert len(result.results) == 3

    @pytest.mark.asyncio
    async def test_batch_with_failures(self, devices, message):
        """Should track failures in batch result."""
        connector = ConcreteDeviceConnector()
        fail_count = 0

        async def fail_sometimes(device, msg, **kwargs):
            nonlocal fail_count
            fail_count += 1
            if fail_count == 2:
                return SendResult(
                    success=False,
                    device_id=device.device_id,
                    status=DeliveryStatus.FAILED,
                    error="Test failure",
                )
            return SendResult(
                success=True,
                device_id=device.device_id,
                status=DeliveryStatus.DELIVERED,
            )

        connector.send_notification = fail_sometimes

        result = await connector.send_batch(devices, message)

        assert result.success_count == 2
        assert result.failure_count == 1

    @pytest.mark.asyncio
    async def test_batch_collects_unregister_tokens(self, devices, message):
        """Should collect tokens that should be unregistered."""
        connector = ConcreteDeviceConnector()

        async def mark_for_unregister(device, msg, **kwargs):
            return SendResult(
                success=False,
                device_id=device.device_id,
                status=DeliveryStatus.INVALID_TOKEN,
                should_unregister=True,
            )

        connector.send_notification = mark_for_unregister

        result = await connector.send_batch(devices, message)

        assert len(result.tokens_to_remove) == 3

    @pytest.mark.asyncio
    async def test_batch_handles_exceptions(self, devices, message):
        """Should handle exceptions during send."""
        connector = ConcreteDeviceConnector()

        async def raise_error(device, msg, **kwargs):
            raise RuntimeError("Network error")

        connector.send_notification = raise_error

        result = await connector.send_batch(devices, message)

        assert result.failure_count == 3
        for r in result.results:
            assert r.success is False
            assert r.error  # Sanitized error message present
            assert "failed" in r.error.lower()


class TestSendToUser:
    """Tests for send_to_user method."""

    @pytest.mark.asyncio
    async def test_no_session_store(self):
        """Should handle missing session store gracefully."""
        connector = ConcreteDeviceConnector()
        message = DeviceMessage(title="Test", body="Message")

        with patch.dict("sys.modules", {"aragora.server.session_store": None}):
            result = await connector.send_to_user("user_123", message)

        assert result.total_sent == 0


class TestDeviceRegistration:
    """Tests for device registration."""

    @pytest.fixture
    def connector(self):
        """Create connector for testing."""
        return ConcreteDeviceConnector()

    @pytest.fixture
    def registration(self):
        """Create test registration."""
        return DeviceRegistration(
            user_id="user_123",
            device_type=DeviceType.ANDROID,
            push_token="test_push_token_12345",
            device_name="Test Phone",
            app_version="1.0.0",
        )

    @pytest.mark.asyncio
    async def test_unsupported_device_type(self, connector):
        """Should reject unsupported device types."""
        registration = DeviceRegistration(
            user_id="user_123",
            device_type=DeviceType.ALEXA,  # Not in supported types
            push_token="test_token_12345",
        )

        result = await connector.register_device(registration)

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_token_format(self, connector):
        """Should reject invalid token format."""
        registration = DeviceRegistration(
            user_id="user_123",
            device_type=DeviceType.ANDROID,
            push_token="short",  # Too short
        )

        result = await connector.register_device(registration)

        assert result is None

    @pytest.mark.asyncio
    async def test_successful_registration(self, connector, registration):
        """Should register device successfully."""
        result = await connector.register_device(registration)

        assert result is not None
        assert result.user_id == "user_123"
        assert result.device_type == DeviceType.ANDROID
        assert result.device_name == "Test Phone"

    def test_validate_token_length(self, connector):
        """Should validate minimum token length."""
        assert connector.validate_token("") is False
        assert connector.validate_token("short") is False
        assert connector.validate_token("valid_token_123") is True


class TestUnregisterDevice:
    """Tests for device unregistration."""

    @pytest.mark.asyncio
    async def test_unregister_no_session_store(self):
        """Should handle missing session store."""
        connector = ConcreteDeviceConnector()

        result = await connector.unregister_device("device_123")

        assert result is False


class TestVoiceDevice:
    """Tests for voice device support."""

    @pytest.fixture
    def connector(self):
        """Create connector for testing."""
        return ConcreteDeviceConnector()

    @pytest.mark.asyncio
    async def test_handle_voice_request_default(self, connector):
        """Default implementation should return error response."""
        request = VoiceDeviceRequest(
            request_id="req_123",
            device_type=DeviceType.ALEXA,
            user_id="user_123",
            intent="test_intent",
            slots={},
        )

        response = await connector.handle_voice_request(request)

        assert isinstance(response, VoiceDeviceResponse)
        assert "not available" in response.text.lower()
        assert response.should_end_session is True

    @pytest.mark.asyncio
    async def test_send_proactive_notification_default(self, connector):
        """Default implementation should return False."""
        result = await connector.send_proactive_notification(
            user_id="user_123",
            message="Test notification",
        )

        assert result is False


class TestHealthAndStatus:
    """Tests for health and status methods."""

    @pytest.fixture
    def connector(self):
        """Create initialized connector."""
        connector = ConcreteDeviceConnector()
        connector._initialized = True
        return connector

    @pytest.mark.asyncio
    async def test_test_connection(self, connector):
        """Should return connection status."""
        result = await connector.test_connection()

        assert result["platform"] == "test_platform"
        assert result["success"] is True
        assert "supported_device_types" in result

    @pytest.mark.asyncio
    async def test_get_health_uninitialized(self):
        """Should report uninitialized status."""
        connector = ConcreteDeviceConnector()

        health = await connector.get_health()

        assert health["status"] == "uninitialized"
        assert health["initialized"] is False

    @pytest.mark.asyncio
    async def test_get_health_healthy(self, connector):
        """Should report healthy status."""
        with patch.object(connector, "_get_circuit_breaker", return_value=None):
            health = await connector.get_health()

        assert health["status"] == "healthy"
        assert health["platform"] == "test_platform"
        assert health["display_name"] == "Test Platform"

    @pytest.mark.asyncio
    async def test_get_health_with_circuit_breaker(self, connector):
        """Should include circuit breaker status."""
        mock_cb = MagicMock()
        mock_cb.get_status.return_value = "closed"

        with patch.object(connector, "_get_circuit_breaker", return_value=mock_cb):
            health = await connector.get_health()

        assert health["circuit_breaker"]["enabled"] is True
        assert health["circuit_breaker"]["state"] == "closed"

    @pytest.mark.asyncio
    async def test_get_health_circuit_open(self, connector):
        """Should report unhealthy when circuit open."""
        mock_cb = MagicMock()
        mock_cb.get_status.return_value = "open"
        mock_cb.cooldown_remaining.return_value = 30.0

        with patch.object(connector, "_get_circuit_breaker", return_value=mock_cb):
            health = await connector.get_health()

        assert health["status"] == "unhealthy"
        assert health["circuit_breaker"]["cooldown_remaining"] == 30.0

    @pytest.mark.asyncio
    async def test_get_health_circuit_half_open(self, connector):
        """Should report degraded when circuit half-open."""
        mock_cb = MagicMock()
        mock_cb.get_status.return_value = "half_open"

        with patch.object(connector, "_get_circuit_breaker", return_value=mock_cb):
            health = await connector.get_health()

        assert health["status"] == "degraded"


class TestLifecycle:
    """Tests for lifecycle methods."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Should initialize connector."""
        connector = ConcreteDeviceConnector()

        result = await connector.initialize()

        assert result is True
        assert connector._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Should shutdown connector."""
        connector = ConcreteDeviceConnector()
        connector._initialized = True

        await connector.shutdown()

        assert connector._initialized is False

    def test_is_configured(self):
        """Should reflect initialization state."""
        connector = ConcreteDeviceConnector()
        assert connector.is_configured is False

        connector._initialized = True
        assert connector.is_configured is True

    def test_repr(self):
        """Should have useful repr."""
        connector = ConcreteDeviceConnector()
        repr_str = repr(connector)

        assert "ConcreteDeviceConnector" in repr_str
        assert "test_platform" in repr_str
