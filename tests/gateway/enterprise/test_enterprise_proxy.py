"""
Comprehensive tests for EnterpriseProxy and related classes.

Tests cover:
- Custom exceptions (ProxyError, CircuitOpenError, BulkheadFullError, etc.)
- Enums (HealthStatus, RetryStrategy)
- Configuration dataclasses (CircuitBreakerSettings, RetrySettings, BulkheadSettings, etc.)
- ProxyRequest and ProxyResponse types
- FrameworkCircuitBreaker state transitions
- FrameworkBulkhead concurrent request limiting
- RequestSanitizer (headers, body, validation)
- HealthCheckResult
- EnterpriseProxy initialization, lifecycle, request handling, hooks, monitoring
"""

import asyncio
import hashlib
import json
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.enterprise.proxy import (
    # Exceptions
    ProxyError,
    CircuitOpenError,
    BulkheadFullError,
    RequestTimeoutError,
    FrameworkNotConfiguredError,
    SanitizationError,
    # Enums
    HealthStatus,
    RetryStrategy,
    # Configuration
    CircuitBreakerSettings,
    RetrySettings,
    BulkheadSettings,
    SanitizationSettings,
    ExternalFrameworkConfig,
    ProxyConfig,
    # Request/Response
    ProxyRequest,
    ProxyResponse,
    # Health
    HealthCheckResult,
    # Components
    FrameworkCircuitBreaker,
    FrameworkBulkhead,
    RequestSanitizer,
    # Main class
    EnterpriseProxy,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_proxy_config():
    """Create a default proxy configuration."""
    return ProxyConfig()


@pytest.fixture
def custom_proxy_config():
    """Create a custom proxy configuration."""
    return ProxyConfig(
        default_timeout=60.0,
        default_connect_timeout=15.0,
        max_connections=200,
        max_connections_per_host=20,
        keepalive_timeout=45.0,
        enable_connection_pooling=True,
        enable_audit_logging=False,
        enable_metrics=False,
        tenant_header_name="X-Custom-Tenant",
        correlation_header_name="X-Custom-Correlation",
        user_agent="TestProxy/1.0",
    )


@pytest.fixture
def openai_framework_config():
    """Create an OpenAI framework configuration."""
    return ExternalFrameworkConfig(
        base_url="https://api.openai.com",
        timeout=60.0,
        connect_timeout=10.0,
        default_headers={"Authorization": "Bearer sk-test"},
        health_check_path="/v1/health",
    )


@pytest.fixture
def anthropic_framework_config():
    """Create an Anthropic framework configuration."""
    return ExternalFrameworkConfig(
        base_url="https://api.anthropic.com/",
        timeout=120.0,
        circuit_breaker=CircuitBreakerSettings(failure_threshold=3, cooldown_seconds=30.0),
        retry=RetrySettings(max_retries=2, base_delay=1.0),
        bulkhead=BulkheadSettings(max_concurrent=20, wait_timeout=5.0),
    )


@pytest.fixture
def disabled_framework_config():
    """Create a disabled framework configuration."""
    return ExternalFrameworkConfig(
        base_url="https://api.disabled.com",
        enabled=False,
    )


@pytest.fixture
def basic_cb_settings():
    """Create basic circuit breaker settings."""
    return CircuitBreakerSettings(
        failure_threshold=3,
        success_threshold=2,
        cooldown_seconds=10.0,
        half_open_max_calls=2,
    )


@pytest.fixture
def basic_retry_settings():
    """Create basic retry settings."""
    return RetrySettings(
        max_retries=3,
        base_delay=0.5,
        max_delay=10.0,
        strategy=RetryStrategy.EXPONENTIAL,
        jitter=False,
    )


@pytest.fixture
def basic_bulkhead_settings():
    """Create basic bulkhead settings."""
    return BulkheadSettings(max_concurrent=3, wait_timeout=2.0)


@pytest.fixture
def basic_sanitization_settings():
    """Create basic sanitization settings."""
    return SanitizationSettings()


@pytest.fixture
def circuit_breaker(basic_cb_settings):
    """Create a framework circuit breaker."""
    return FrameworkCircuitBreaker("test-framework", basic_cb_settings)


@pytest.fixture
def bulkhead(basic_bulkhead_settings):
    """Create a framework bulkhead."""
    return FrameworkBulkhead("test-framework", basic_bulkhead_settings)


@pytest.fixture
def sanitizer(basic_sanitization_settings):
    """Create a request sanitizer."""
    return RequestSanitizer(basic_sanitization_settings)


@pytest.fixture
def proxy_with_frameworks(openai_framework_config, anthropic_framework_config):
    """Create a proxy with preconfigured frameworks."""
    return EnterpriseProxy(
        config=ProxyConfig(enable_audit_logging=False),
        frameworks={
            "openai": openai_framework_config,
            "anthropic": anthropic_framework_config,
        },
    )


@pytest.fixture
def sample_proxy_request():
    """Create a sample proxy request."""
    return ProxyRequest(
        framework="openai",
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
        body=b'{"model": "gpt-4"}',
        tenant_id="tenant-123",
        correlation_id="corr-abc",
    )


@pytest.fixture
def sample_proxy_response():
    """Create a sample proxy response."""
    return ProxyResponse(
        status_code=200,
        headers={"Content-Type": "application/json"},
        body=b'{"id": "chatcmpl-123"}',
        elapsed_ms=450.0,
        framework="openai",
        correlation_id="corr-abc",
    )


def _make_mock_session(status=200, body=b'{"ok": true}', headers=None):
    """Create a mock aiohttp session with a configurable response."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.headers = headers or {"Content-Type": "application/json"}
    mock_response.read = AsyncMock(return_value=body)

    @asynccontextmanager
    async def mock_request(**kwargs):
        yield mock_response

    mock_session = AsyncMock()
    mock_session.request = MagicMock(side_effect=mock_request)
    mock_session.close = AsyncMock()
    return mock_session


# =============================================================================
# Exception Tests
# =============================================================================


class TestProxyError:
    """Tests for ProxyError and its subclasses."""

    def test_proxy_error_basic(self):
        err = ProxyError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.code == "PROXY_ERROR"
        assert err.framework is None
        assert err.details == {}

    def test_proxy_error_with_details(self):
        err = ProxyError(
            "fail",
            code="CUSTOM_CODE",
            framework="openai",
            details={"key": "value"},
        )
        assert err.code == "CUSTOM_CODE"
        assert err.framework == "openai"
        assert err.details == {"key": "value"}

    def test_proxy_error_is_exception(self):
        err = ProxyError("test")
        assert isinstance(err, Exception)

    def test_circuit_open_error(self):
        err = CircuitOpenError("openai", 15.5)
        assert isinstance(err, ProxyError)
        assert err.framework == "openai"
        assert err.cooldown_remaining == 15.5
        assert err.code == "CIRCUIT_OPEN"
        assert "15.5s" in str(err)
        assert "openai" in str(err)

    def test_circuit_open_error_with_details(self):
        err = CircuitOpenError("openai", 10.0, details={"extra": True})
        assert err.details == {"extra": True}

    def test_bulkhead_full_error(self):
        err = BulkheadFullError("anthropic", 50)
        assert isinstance(err, ProxyError)
        assert err.framework == "anthropic"
        assert err.max_concurrent == 50
        assert err.code == "BULKHEAD_FULL"
        assert "50" in str(err)

    def test_bulkhead_full_error_with_details(self):
        err = BulkheadFullError("anthropic", 50, details={"queue_depth": 10})
        assert err.details == {"queue_depth": 10}

    def test_request_timeout_error(self):
        err = RequestTimeoutError("openai", 30.0)
        assert isinstance(err, ProxyError)
        assert err.framework == "openai"
        assert err.timeout == 30.0
        assert err.code == "REQUEST_TIMEOUT"
        assert "30.0s" in str(err)

    def test_request_timeout_error_with_details(self):
        err = RequestTimeoutError("openai", 30.0, details={"attempt": 2})
        assert err.details == {"attempt": 2}

    def test_framework_not_configured_error(self):
        err = FrameworkNotConfiguredError("unknown")
        assert isinstance(err, ProxyError)
        assert err.framework == "unknown"
        assert err.code == "FRAMEWORK_NOT_CONFIGURED"
        assert "unknown" in str(err)

    def test_sanitization_error(self):
        err = SanitizationError("Injection detected", framework="openai")
        assert isinstance(err, ProxyError)
        assert err.code == "SANITIZATION_ERROR"
        assert err.framework == "openai"

    def test_sanitization_error_with_details(self):
        err = SanitizationError(
            "Bad header",
            framework="openai",
            details={"header": "X-Evil"},
        )
        assert err.details == {"header": "X-Evil"}


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for HealthStatus and RetryStrategy enums."""

    def test_health_status_values(self):
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.UNKNOWN == "unknown"

    def test_health_status_is_str(self):
        assert isinstance(HealthStatus.HEALTHY, str)

    def test_retry_strategy_values(self):
        assert RetryStrategy.EXPONENTIAL == "exponential"
        assert RetryStrategy.LINEAR == "linear"
        assert RetryStrategy.CONSTANT == "constant"

    def test_retry_strategy_is_str(self):
        assert isinstance(RetryStrategy.EXPONENTIAL, str)


# =============================================================================
# Configuration Dataclass Tests
# =============================================================================


class TestCircuitBreakerSettings:
    """Tests for CircuitBreakerSettings validation and defaults."""

    def test_defaults(self):
        settings = CircuitBreakerSettings()
        assert settings.failure_threshold == 5
        assert settings.success_threshold == 2
        assert settings.cooldown_seconds == 60.0
        assert settings.half_open_max_calls == 3

    def test_custom_values(self):
        settings = CircuitBreakerSettings(
            failure_threshold=10,
            success_threshold=5,
            cooldown_seconds=120.0,
            half_open_max_calls=5,
        )
        assert settings.failure_threshold == 10
        assert settings.success_threshold == 5

    def test_failure_threshold_validation(self):
        with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
            CircuitBreakerSettings(failure_threshold=0)

    def test_success_threshold_validation(self):
        with pytest.raises(ValueError, match="success_threshold must be at least 1"):
            CircuitBreakerSettings(success_threshold=0)

    def test_cooldown_seconds_validation(self):
        with pytest.raises(ValueError, match="cooldown_seconds must be positive"):
            CircuitBreakerSettings(cooldown_seconds=0)

    def test_half_open_max_calls_validation(self):
        with pytest.raises(ValueError, match="half_open_max_calls must be at least 1"):
            CircuitBreakerSettings(half_open_max_calls=0)


class TestRetrySettings:
    """Tests for RetrySettings validation and defaults."""

    def test_defaults(self):
        settings = RetrySettings()
        assert settings.max_retries == 3
        assert settings.base_delay == 0.5
        assert settings.max_delay == 30.0
        assert settings.strategy == RetryStrategy.EXPONENTIAL
        assert settings.jitter is True
        assert 429 in settings.retryable_status_codes
        assert 503 in settings.retryable_status_codes

    def test_custom_values(self):
        settings = RetrySettings(
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0,
            strategy=RetryStrategy.LINEAR,
            jitter=False,
            retryable_status_codes=frozenset({500}),
        )
        assert settings.max_retries == 5
        assert settings.strategy == RetryStrategy.LINEAR
        assert settings.retryable_status_codes == frozenset({500})

    def test_negative_max_retries_validation(self):
        with pytest.raises(ValueError, match="max_retries cannot be negative"):
            RetrySettings(max_retries=-1)

    def test_zero_max_retries_allowed(self):
        settings = RetrySettings(max_retries=0)
        assert settings.max_retries == 0

    def test_base_delay_validation(self):
        with pytest.raises(ValueError, match="base_delay must be positive"):
            RetrySettings(base_delay=0)

    def test_max_delay_less_than_base_delay_validation(self):
        with pytest.raises(ValueError, match="max_delay must be >= base_delay"):
            RetrySettings(base_delay=10.0, max_delay=5.0)

    def test_max_delay_equal_to_base_delay(self):
        settings = RetrySettings(base_delay=5.0, max_delay=5.0)
        assert settings.max_delay == settings.base_delay


class TestBulkheadSettings:
    """Tests for BulkheadSettings validation and defaults."""

    def test_defaults(self):
        settings = BulkheadSettings()
        assert settings.max_concurrent == 50
        assert settings.wait_timeout == 10.0

    def test_custom_values(self):
        settings = BulkheadSettings(max_concurrent=100, wait_timeout=30.0)
        assert settings.max_concurrent == 100

    def test_max_concurrent_validation(self):
        with pytest.raises(ValueError, match="max_concurrent must be at least 1"):
            BulkheadSettings(max_concurrent=0)

    def test_wait_timeout_validation(self):
        with pytest.raises(ValueError, match="wait_timeout must be positive"):
            BulkheadSettings(wait_timeout=0)


class TestSanitizationSettings:
    """Tests for SanitizationSettings defaults."""

    def test_defaults(self):
        settings = SanitizationSettings()
        assert "authorization" in settings.redact_headers
        assert "x-api-key" in settings.redact_headers
        assert "cookie" in settings.redact_headers
        assert len(settings.redact_body_patterns) == 4
        assert settings.max_body_log_size == 4096
        assert "x-forwarded-for" in settings.strip_sensitive_headers

    def test_custom_redact_headers(self):
        settings = SanitizationSettings(
            redact_headers=frozenset({"x-custom-secret"}),
        )
        assert "x-custom-secret" in settings.redact_headers
        assert "authorization" not in settings.redact_headers


class TestExternalFrameworkConfig:
    """Tests for ExternalFrameworkConfig validation and normalization."""

    def test_basic_config(self):
        config = ExternalFrameworkConfig(base_url="https://api.example.com")
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 30.0
        assert config.enabled is True

    def test_trailing_slash_normalized(self):
        config = ExternalFrameworkConfig(base_url="https://api.example.com/")
        assert config.base_url == "https://api.example.com"

    def test_multiple_trailing_slashes_normalized(self):
        config = ExternalFrameworkConfig(base_url="https://api.example.com///")
        assert not config.base_url.endswith("/")

    def test_empty_base_url_validation(self):
        with pytest.raises(ValueError, match="base_url is required"):
            ExternalFrameworkConfig(base_url="")

    def test_timeout_validation(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            ExternalFrameworkConfig(base_url="https://api.example.com", timeout=0)

    def test_connect_timeout_validation(self):
        with pytest.raises(ValueError, match="connect_timeout must be positive"):
            ExternalFrameworkConfig(base_url="https://api.example.com", connect_timeout=-1)

    def test_health_check_interval_validation(self):
        with pytest.raises(ValueError, match="health_check_interval must be positive"):
            ExternalFrameworkConfig(
                base_url="https://api.example.com",
                health_check_interval=0,
            )

    def test_default_headers(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.example.com",
            default_headers={"X-API-Key": "key123"},
        )
        assert config.default_headers["X-API-Key"] == "key123"

    def test_metadata(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.example.com",
            metadata={"provider": "openai", "tier": "enterprise"},
        )
        assert config.metadata["provider"] == "openai"


class TestProxyConfig:
    """Tests for ProxyConfig validation and defaults."""

    def test_defaults(self):
        config = ProxyConfig()
        assert config.default_timeout == 30.0
        assert config.max_connections == 100
        assert config.max_connections_per_host == 10
        assert config.enable_connection_pooling is True
        assert config.enable_audit_logging is True
        assert config.tenant_header_name == "X-Tenant-ID"
        assert config.correlation_header_name == "X-Correlation-ID"
        assert config.user_agent == "Aragora-EnterpriseProxy/1.0"

    def test_max_connections_validation(self):
        with pytest.raises(ValueError, match="max_connections must be at least 1"):
            ProxyConfig(max_connections=0)

    def test_max_connections_per_host_validation(self):
        with pytest.raises(ValueError, match="max_connections_per_host must be at least 1"):
            ProxyConfig(max_connections_per_host=0)


# =============================================================================
# ProxyRequest / ProxyResponse Tests
# =============================================================================


class TestProxyRequest:
    """Tests for ProxyRequest dataclass."""

    def test_basic_creation(self):
        req = ProxyRequest(
            framework="openai",
            method="POST",
            url="https://api.openai.com/v1/completions",
            headers={"Content-Type": "application/json"},
        )
        assert req.framework == "openai"
        assert req.method == "POST"
        assert req.body is None
        assert req.tenant_id is None
        assert req.correlation_id is None
        assert req.auth_context is None
        assert isinstance(req.metadata, dict)
        assert isinstance(req.timestamp, float)

    def test_body_hash_with_body(self):
        body = b"test body content"
        req = ProxyRequest(
            framework="test",
            method="POST",
            url="https://example.com",
            headers={},
            body=body,
        )
        expected = hashlib.sha256(body).hexdigest()
        assert req.body_hash() == expected

    def test_body_hash_without_body(self):
        req = ProxyRequest(
            framework="test",
            method="GET",
            url="https://example.com",
            headers={},
        )
        assert req.body_hash() is None

    def test_metadata_default(self):
        req = ProxyRequest(
            framework="test",
            method="GET",
            url="https://example.com",
            headers={},
        )
        assert req.metadata == {}

    def test_full_request(self):
        req = ProxyRequest(
            framework="openai",
            method="POST",
            url="https://api.openai.com/v1/chat",
            headers={"Authorization": "Bearer test"},
            body=b'{"prompt": "hello"}',
            tenant_id="t-1",
            correlation_id="c-1",
            auth_context={"user": "admin"},
            metadata={"attempt": 1},
        )
        assert req.tenant_id == "t-1"
        assert req.correlation_id == "c-1"
        assert req.auth_context == {"user": "admin"}
        assert req.metadata == {"attempt": 1}


class TestProxyResponse:
    """Tests for ProxyResponse dataclass."""

    def test_basic_creation(self):
        resp = ProxyResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"ok": true}',
            elapsed_ms=123.4,
            framework="openai",
        )
        assert resp.status_code == 200
        assert resp.elapsed_ms == 123.4
        assert resp.from_cache is False
        assert resp.correlation_id is None

    def test_is_success_2xx(self):
        for code in [200, 201, 204, 299]:
            resp = ProxyResponse(
                status_code=code, headers={}, body=b"", elapsed_ms=0.0, framework="test"
            )
            assert resp.is_success is True
            assert resp.is_client_error is False
            assert resp.is_server_error is False

    def test_is_client_error_4xx(self):
        for code in [400, 401, 403, 404, 429, 499]:
            resp = ProxyResponse(
                status_code=code, headers={}, body=b"", elapsed_ms=0.0, framework="test"
            )
            assert resp.is_client_error is True
            assert resp.is_success is False
            assert resp.is_server_error is False

    def test_is_server_error_5xx(self):
        for code in [500, 502, 503, 504, 599]:
            resp = ProxyResponse(
                status_code=code, headers={}, body=b"", elapsed_ms=0.0, framework="test"
            )
            assert resp.is_server_error is True
            assert resp.is_success is False
            assert resp.is_client_error is False

    def test_body_hash(self):
        body = b"response content"
        resp = ProxyResponse(
            status_code=200, headers={}, body=body, elapsed_ms=0.0, framework="test"
        )
        expected = hashlib.sha256(body).hexdigest()
        assert resp.body_hash() == expected

    def test_body_hash_empty_body(self):
        resp = ProxyResponse(
            status_code=200, headers={}, body=b"", elapsed_ms=0.0, framework="test"
        )
        expected = hashlib.sha256(b"").hexdigest()
        assert resp.body_hash() == expected

    def test_from_cache_flag(self):
        resp = ProxyResponse(
            status_code=200,
            headers={},
            body=b"",
            elapsed_ms=0.0,
            framework="test",
            from_cache=True,
        )
        assert resp.from_cache is True

    def test_metadata(self):
        resp = ProxyResponse(
            status_code=200,
            headers={},
            body=b"",
            elapsed_ms=0.0,
            framework="test",
            metadata={"cache_key": "abc"},
        )
        assert resp.metadata == {"cache_key": "abc"}


# =============================================================================
# FrameworkCircuitBreaker Tests
# =============================================================================


class TestFrameworkCircuitBreaker:
    """Tests for FrameworkCircuitBreaker state machine."""

    def test_initial_state_closed(self, circuit_breaker):
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.is_open is False
        assert circuit_breaker.cooldown_remaining == 0.0

    @pytest.mark.asyncio
    async def test_can_proceed_when_closed(self, circuit_breaker):
        assert await circuit_breaker.can_proceed() is True

    @pytest.mark.asyncio
    async def test_record_success_resets_failures(self, circuit_breaker):
        circuit_breaker._failures = 2
        await circuit_breaker.record_success()
        assert circuit_breaker._failures == 0

    @pytest.mark.asyncio
    async def test_single_failure_stays_closed(self, circuit_breaker):
        opened = await circuit_breaker.record_failure()
        assert opened is False
        assert circuit_breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_threshold_failures_opens_circuit(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        # failure_threshold = 3
        await cb.record_failure()
        await cb.record_failure()
        opened = await cb.record_failure()
        assert opened is True
        assert cb.state == "open"
        assert cb.is_open is True

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_requests(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        assert await cb.can_proceed() is False

    @pytest.mark.asyncio
    async def test_cooldown_remaining_when_open(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        remaining = cb.cooldown_remaining
        assert remaining > 0
        assert remaining <= basic_cb_settings.cooldown_seconds

    @pytest.mark.asyncio
    async def test_transition_to_half_open_after_cooldown(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        assert cb.state == "open"

        # Simulate cooldown elapsed
        cb._open_at = time.time() - basic_cb_settings.cooldown_seconds - 1
        assert cb.state == "half-open"

    @pytest.mark.asyncio
    async def test_half_open_allows_limited_calls(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()

        # Simulate cooldown elapsed
        cb._open_at = time.time() - basic_cb_settings.cooldown_seconds - 1

        # half_open_max_calls = 2
        assert await cb.can_proceed() is True
        assert await cb.can_proceed() is True
        assert await cb.can_proceed() is False

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()

        cb._open_at = time.time() - basic_cb_settings.cooldown_seconds - 1
        assert cb.state == "half-open"

        # success_threshold = 2
        await cb.record_success()
        assert cb.state == "half-open"  # still half-open, need 2 successes
        await cb.record_success()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_half_open_failure_resets_successes(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()

        cb._open_at = time.time() - basic_cb_settings.cooldown_seconds - 1
        await cb.record_success()
        assert cb._successes == 1

        await cb.record_failure()
        assert cb._successes == 0

    @pytest.mark.asyncio
    async def test_reset(self, circuit_breaker):
        for _ in range(5):
            await circuit_breaker.record_failure()
        await circuit_breaker.reset()

        assert circuit_breaker.state == "closed"
        assert circuit_breaker._failures == 0
        assert circuit_breaker._successes == 0
        assert circuit_breaker._open_at is None
        assert circuit_breaker._half_open_calls == 0

    def test_to_dict(self, circuit_breaker):
        result = circuit_breaker.to_dict()
        assert result["framework"] == "test-framework"
        assert result["state"] == "closed"
        assert result["failures"] == 0
        assert result["successes"] == 0
        assert result["cooldown_remaining"] == 0.0
        assert result["half_open_calls"] == 0

    @pytest.mark.asyncio
    async def test_to_dict_when_open(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        result = cb.to_dict()
        assert result["state"] == "open"
        assert result["failures"] == basic_cb_settings.failure_threshold
        assert result["cooldown_remaining"] > 0

    @pytest.mark.asyncio
    async def test_cooldown_remaining_zero_when_closed(self, circuit_breaker):
        assert circuit_breaker.cooldown_remaining == 0.0

    @pytest.mark.asyncio
    async def test_multiple_open_calls_dont_reset_open_at(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        open_at = cb._open_at

        # Additional failures should not reset open_at
        await cb.record_failure()
        assert cb._open_at == open_at

    @pytest.mark.asyncio
    async def test_success_in_closed_state_does_not_modify_open_at(self, circuit_breaker):
        await circuit_breaker.record_success()
        assert circuit_breaker._open_at is None
        assert circuit_breaker.state == "closed"


# =============================================================================
# FrameworkBulkhead Tests
# =============================================================================


class TestFrameworkBulkhead:
    """Tests for FrameworkBulkhead concurrency limiting."""

    def test_initial_state(self, bulkhead):
        assert bulkhead.active_count == 0
        assert bulkhead.available_slots == 3  # max_concurrent = 3

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, bulkhead):
        async with bulkhead.acquire():
            assert bulkhead.active_count == 1
            assert bulkhead.available_slots == 2
        assert bulkhead.active_count == 0
        assert bulkhead.available_slots == 3

    @pytest.mark.asyncio
    async def test_multiple_concurrent_acquisitions(self, bulkhead):
        acquired = []

        async def acquire_slot():
            async with bulkhead.acquire():
                acquired.append(True)
                assert bulkhead.active_count <= 3
                await asyncio.sleep(0.01)

        await asyncio.gather(acquire_slot(), acquire_slot(), acquire_slot())
        assert len(acquired) == 3
        assert bulkhead.active_count == 0

    @pytest.mark.asyncio
    async def test_bulkhead_full_raises_error(self):
        bh = FrameworkBulkhead(
            "test",
            BulkheadSettings(max_concurrent=1, wait_timeout=0.1),
        )
        acquired_event = asyncio.Event()
        release_event = asyncio.Event()

        async def hold_slot():
            async with bh.acquire():
                acquired_event.set()
                await release_event.wait()

        task = asyncio.create_task(hold_slot())
        await acquired_event.wait()

        with pytest.raises(BulkheadFullError) as exc_info:
            async with bh.acquire():
                pass

        assert exc_info.value.max_concurrent == 1
        assert exc_info.value.framework == "test"

        release_event.set()
        await task

    @pytest.mark.asyncio
    async def test_acquire_releases_on_exception(self, bulkhead):
        with pytest.raises(RuntimeError):
            async with bulkhead.acquire():
                raise RuntimeError("Test error")
        assert bulkhead.active_count == 0
        assert bulkhead.available_slots == 3

    def test_to_dict(self, bulkhead):
        result = bulkhead.to_dict()
        assert result["framework"] == "test-framework"
        assert result["active"] == 0
        assert result["max_concurrent"] == 3
        assert result["available_slots"] == 3

    @pytest.mark.asyncio
    async def test_to_dict_during_acquisition(self, bulkhead):
        async with bulkhead.acquire():
            result = bulkhead.to_dict()
            assert result["active"] == 1
            assert result["available_slots"] == 2


# =============================================================================
# HealthCheckResult Tests
# =============================================================================


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_basic_creation(self):
        result = HealthCheckResult(
            framework="openai",
            status=HealthStatus.HEALTHY,
            latency_ms=50.0,
        )
        assert result.framework == "openai"
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms == 50.0
        assert result.error is None
        assert result.consecutive_failures == 0

    def test_unhealthy_result(self):
        result = HealthCheckResult(
            framework="openai",
            status=HealthStatus.UNHEALTHY,
            error="Connection refused",
            consecutive_failures=5,
        )
        assert result.status == HealthStatus.UNHEALTHY
        assert result.error == "Connection refused"
        assert result.consecutive_failures == 5

    def test_to_dict(self):
        result = HealthCheckResult(
            framework="openai",
            status=HealthStatus.DEGRADED,
            latency_ms=1500.0,
            error="High latency",
            consecutive_failures=2,
        )
        d = result.to_dict()
        assert d["framework"] == "openai"
        assert d["status"] == "degraded"
        assert d["latency_ms"] == 1500.0
        assert d["error"] == "High latency"
        assert d["consecutive_failures"] == 2
        assert "last_check" in d

    def test_unknown_status(self):
        result = HealthCheckResult(
            framework="new-fw",
            status=HealthStatus.UNKNOWN,
        )
        assert result.status == HealthStatus.UNKNOWN


# =============================================================================
# RequestSanitizer Tests
# =============================================================================


class TestRequestSanitizer:
    """Tests for RequestSanitizer header and body sanitization."""

    def test_sanitize_headers_strips_sensitive(self, sanitizer):
        headers = {
            "Content-Type": "application/json",
            "X-Forwarded-For": "1.2.3.4",
            "X-Real-IP": "5.6.7.8",
            "Authorization": "Bearer secret",
        }
        result = sanitizer.sanitize_headers(headers)
        assert "X-Forwarded-For" not in result
        assert "X-Real-IP" not in result
        assert result["Content-Type"] == "application/json"
        assert result["Authorization"] == "Bearer secret"  # not redacted for outgoing

    def test_sanitize_headers_for_logging_redacts(self, sanitizer):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-secret-key",
            "X-API-Key": "key-123",
            "Cookie": "session=abc",
        }
        result = sanitizer.sanitize_headers(headers, for_logging=True)
        assert result["Content-Type"] == "application/json"
        assert result["Authorization"] == "[REDACTED]"
        assert result["X-API-Key"] == "[REDACTED]"
        assert result["Cookie"] == "[REDACTED]"

    def test_sanitize_headers_case_insensitive(self, sanitizer):
        headers = {
            "AUTHORIZATION": "Bearer key",
            "x-api-key": "secret",
        }
        result = sanitizer.sanitize_headers(headers, for_logging=True)
        assert result["AUTHORIZATION"] == "[REDACTED]"
        assert result["x-api-key"] == "[REDACTED]"

    def test_sanitize_body_for_logging_none(self, sanitizer):
        assert sanitizer.sanitize_body_for_logging(None) == ""

    def test_sanitize_body_for_logging_redacts_patterns(self, sanitizer):
        body = b'{"api_key": "sk-123", "model": "gpt-4", "password": "secret"}'
        result = sanitizer.sanitize_body_for_logging(body)
        assert "sk-123" not in result
        assert "secret" not in result
        assert '"[REDACTED]"' in result
        assert "gpt-4" in result

    def test_sanitize_body_for_logging_truncates_large(self, sanitizer):
        body = b"x" * 10000
        result = sanitizer.sanitize_body_for_logging(body)
        assert "truncated" in result
        assert "10000 bytes total" in result

    def test_sanitize_body_for_logging_small(self, sanitizer):
        body = b'{"model": "gpt-4"}'
        result = sanitizer.sanitize_body_for_logging(body)
        assert result == '{"model": "gpt-4"}'

    def test_sanitize_body_redacts_token(self, sanitizer):
        body = b'{"token": "tok_live_xyz", "data": "ok"}'
        result = sanitizer.sanitize_body_for_logging(body)
        assert "tok_live_xyz" not in result
        assert '"[REDACTED]"' in result

    def test_sanitize_body_redacts_secret(self, sanitizer):
        body = b'{"secret": "mysecret123"}'
        result = sanitizer.sanitize_body_for_logging(body)
        assert "mysecret123" not in result

    def test_validate_request_clean(self, sanitizer):
        req = ProxyRequest(
            framework="test",
            method="GET",
            url="https://api.example.com/v1/data",
            headers={"Content-Type": "application/json"},
        )
        sanitizer.validate_request(req)  # Should not raise

    def test_validate_request_header_injection_newline_in_key(self, sanitizer):
        req = ProxyRequest(
            framework="test",
            method="GET",
            url="https://api.example.com",
            headers={"Evil\nHeader": "value"},
        )
        with pytest.raises(SanitizationError, match="Header injection"):
            sanitizer.validate_request(req)

    def test_validate_request_header_injection_newline_in_value(self, sanitizer):
        req = ProxyRequest(
            framework="test",
            method="GET",
            url="https://api.example.com",
            headers={"Header": "value\r\nInjected: true"},
        )
        with pytest.raises(SanitizationError, match="Header injection"):
            sanitizer.validate_request(req)

    def test_validate_request_header_injection_carriage_return_in_key(self, sanitizer):
        req = ProxyRequest(
            framework="test",
            method="GET",
            url="https://api.example.com",
            headers={"Evil\rHeader": "value"},
        )
        with pytest.raises(SanitizationError, match="Header injection"):
            sanitizer.validate_request(req)

    def test_validate_request_suspicious_url_script(self, sanitizer):
        req = ProxyRequest(
            framework="test",
            method="GET",
            url="https://api.example.com/<script>alert(1)</script>",
            headers={},
        )
        with pytest.raises(SanitizationError, match="Suspicious URL"):
            sanitizer.validate_request(req)

    def test_validate_request_suspicious_url_javascript(self, sanitizer):
        req = ProxyRequest(
            framework="test",
            method="GET",
            url="javascript:alert(1)",
            headers={},
        )
        with pytest.raises(SanitizationError, match="Suspicious URL"):
            sanitizer.validate_request(req)

    def test_validate_request_suspicious_url_data(self, sanitizer):
        req = ProxyRequest(
            framework="test",
            method="GET",
            url="data:text/html,<h1>evil</h1>",
            headers={},
        )
        with pytest.raises(SanitizationError, match="Suspicious URL"):
            sanitizer.validate_request(req)

    def test_validate_request_suspicious_url_file(self, sanitizer):
        req = ProxyRequest(
            framework="test",
            method="GET",
            url="file:///etc/passwd",
            headers={},
        )
        with pytest.raises(SanitizationError, match="Suspicious URL"):
            sanitizer.validate_request(req)

    def test_custom_body_patterns(self):
        settings = SanitizationSettings(
            redact_body_patterns=[r'"credit_card"\s*:\s*"[^"]*"'],
        )
        sanitizer = RequestSanitizer(settings)
        body = b'{"credit_card": "4111111111111111"}'
        result = sanitizer.sanitize_body_for_logging(body)
        assert "4111111111111111" not in result


# =============================================================================
# EnterpriseProxy Initialization Tests
# =============================================================================


class TestEnterpriseProxyInit:
    """Tests for EnterpriseProxy initialization and configuration."""

    def test_default_initialization(self):
        proxy = EnterpriseProxy()
        assert isinstance(proxy.config, ProxyConfig)
        assert proxy.list_frameworks() == []

    def test_initialization_with_config(self, custom_proxy_config):
        proxy = EnterpriseProxy(config=custom_proxy_config)
        assert proxy.config.default_timeout == 60.0
        assert proxy.config.user_agent == "TestProxy/1.0"

    def test_initialization_with_frameworks(
        self, openai_framework_config, anthropic_framework_config
    ):
        proxy = EnterpriseProxy(
            frameworks={
                "openai": openai_framework_config,
                "anthropic": anthropic_framework_config,
            },
        )
        assert set(proxy.list_frameworks()) == {"openai", "anthropic"}

    def test_framework_components_initialized(self, openai_framework_config):
        proxy = EnterpriseProxy(frameworks={"openai": openai_framework_config})
        assert "openai" in proxy._circuit_breakers
        assert "openai" in proxy._bulkheads
        assert "openai" in proxy._sanitizers
        assert "openai" in proxy._health_results
        assert proxy._health_results["openai"].status == HealthStatus.UNKNOWN


# =============================================================================
# EnterpriseProxy Framework Management Tests
# =============================================================================


class TestEnterpriseProxyFrameworkManagement:
    """Tests for framework registration and management."""

    def test_register_framework(self):
        proxy = EnterpriseProxy()
        config = ExternalFrameworkConfig(base_url="https://api.new.com")
        proxy.register_framework("new-fw", config)

        assert "new-fw" in proxy.list_frameworks()
        assert proxy.get_framework_config("new-fw") is config

    def test_register_framework_overwrites(self, openai_framework_config):
        proxy = EnterpriseProxy(frameworks={"openai": openai_framework_config})
        new_config = ExternalFrameworkConfig(
            base_url="https://api.openai-v2.com",
            timeout=90.0,
        )
        proxy.register_framework("openai", new_config)
        assert proxy.get_framework_config("openai") is new_config

    def test_unregister_framework(self, openai_framework_config):
        proxy = EnterpriseProxy(frameworks={"openai": openai_framework_config})
        result = proxy.unregister_framework("openai")
        assert result is True
        assert "openai" not in proxy.list_frameworks()
        assert "openai" not in proxy._circuit_breakers
        assert "openai" not in proxy._bulkheads
        assert "openai" not in proxy._sanitizers
        assert "openai" not in proxy._health_results

    def test_unregister_nonexistent_framework(self):
        proxy = EnterpriseProxy()
        result = proxy.unregister_framework("nonexistent")
        assert result is False

    def test_get_framework_config_missing(self):
        proxy = EnterpriseProxy()
        assert proxy.get_framework_config("nonexistent") is None

    def test_list_frameworks_empty(self):
        proxy = EnterpriseProxy()
        assert proxy.list_frameworks() == []

    def test_list_frameworks_order(self):
        proxy = EnterpriseProxy(
            frameworks={
                "a": ExternalFrameworkConfig(base_url="https://a.com"),
                "b": ExternalFrameworkConfig(base_url="https://b.com"),
                "c": ExternalFrameworkConfig(base_url="https://c.com"),
            },
        )
        assert len(proxy.list_frameworks()) == 3


# =============================================================================
# EnterpriseProxy Hook Tests
# =============================================================================


class TestEnterpriseProxyHooks:
    """Tests for pre/post request and error hooks."""

    def test_add_pre_request_hook(self):
        proxy = EnterpriseProxy()

        async def hook(req):
            return req

        proxy.add_pre_request_hook(hook)
        assert len(proxy._pre_request_hooks) == 1

    def test_add_post_request_hook(self):
        proxy = EnterpriseProxy()

        async def hook(req, resp):
            pass

        proxy.add_post_request_hook(hook)
        assert len(proxy._post_request_hooks) == 1

    def test_add_error_hook(self):
        proxy = EnterpriseProxy()

        async def hook(req, exc):
            pass

        proxy.add_error_hook(hook)
        assert len(proxy._error_hooks) == 1

    def test_remove_pre_request_hook(self):
        proxy = EnterpriseProxy()

        async def hook(req):
            return req

        proxy.add_pre_request_hook(hook)
        assert proxy.remove_pre_request_hook(hook) is True
        assert len(proxy._pre_request_hooks) == 0

    def test_remove_nonexistent_pre_request_hook(self):
        proxy = EnterpriseProxy()

        async def hook(req):
            return req

        assert proxy.remove_pre_request_hook(hook) is False

    def test_remove_post_request_hook(self):
        proxy = EnterpriseProxy()

        async def hook(req, resp):
            pass

        proxy.add_post_request_hook(hook)
        assert proxy.remove_post_request_hook(hook) is True
        assert len(proxy._post_request_hooks) == 0

    def test_remove_nonexistent_post_request_hook(self):
        proxy = EnterpriseProxy()

        async def hook(req, resp):
            pass

        assert proxy.remove_post_request_hook(hook) is False

    def test_remove_error_hook(self):
        proxy = EnterpriseProxy()

        async def hook(req, exc):
            pass

        proxy.add_error_hook(hook)
        assert proxy.remove_error_hook(hook) is True
        assert len(proxy._error_hooks) == 0

    def test_remove_nonexistent_error_hook(self):
        proxy = EnterpriseProxy()

        async def hook(req, exc):
            pass

        assert proxy.remove_error_hook(hook) is False

    def test_multiple_hooks(self):
        proxy = EnterpriseProxy()
        hooks = []
        for _ in range(5):

            async def hook(req):
                return req

            hooks.append(hook)
            proxy.add_pre_request_hook(hook)
        assert len(proxy._pre_request_hooks) == 5


# =============================================================================
# EnterpriseProxy Lifecycle Tests
# =============================================================================


class TestEnterpriseProxyLifecycle:
    """Tests for proxy start/shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_context_manager(self, openai_framework_config):
        proxy = EnterpriseProxy(frameworks={"openai": openai_framework_config})
        with (
            patch.object(proxy, "start", new_callable=AsyncMock) as mock_start,
            patch.object(proxy, "shutdown", new_callable=AsyncMock) as mock_shutdown,
        ):
            async with proxy:
                mock_start.assert_awaited_once()
            mock_shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_health_check_task(self):
        proxy = EnterpriseProxy()

        async def noop():
            await asyncio.sleep(100)

        task = asyncio.create_task(noop())
        proxy._health_check_task = task
        proxy._session = AsyncMock()
        proxy._session.close = AsyncMock()

        await proxy.shutdown()

        assert task.cancelled()
        assert proxy._session is None

    @pytest.mark.asyncio
    async def test_shutdown_closes_session(self):
        proxy = EnterpriseProxy()
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        proxy._session = mock_session

        await proxy.shutdown()

        mock_session.close.assert_awaited_once()
        assert proxy._session is None

    @pytest.mark.asyncio
    async def test_shutdown_without_session(self):
        proxy = EnterpriseProxy()
        await proxy.shutdown()  # Should not raise

    @pytest.mark.asyncio
    async def test_ensure_session_creates_once(self):
        proxy = EnterpriseProxy()
        mock_session = MagicMock()

        with (
            patch("aiohttp.TCPConnector", return_value=MagicMock()),
            patch("aiohttp.ClientTimeout", return_value=MagicMock()),
            patch("aiohttp.ClientSession", return_value=mock_session) as mock_cls,
        ):
            session1 = await proxy._ensure_session()
            session2 = await proxy._ensure_session()

            assert session1 is session2
            mock_cls.assert_called_once()


# =============================================================================
# EnterpriseProxy Connection Pooling Tests
# =============================================================================


class TestEnterpriseProxyConnectionPooling:
    """Tests for connection pool configuration."""

    @pytest.mark.asyncio
    async def test_connector_uses_pool_settings(self):
        proxy = EnterpriseProxy(
            config=ProxyConfig(
                max_connections=200,
                max_connections_per_host=25,
                keepalive_timeout=60.0,
            )
        )

        with (
            patch("aiohttp.TCPConnector", return_value=MagicMock()) as mock_connector,
            patch("aiohttp.ClientTimeout", return_value=MagicMock()),
            patch("aiohttp.ClientSession", return_value=MagicMock()),
        ):
            await proxy._ensure_session()

            mock_connector.assert_called_once_with(
                limit=200,
                limit_per_host=25,
                keepalive_timeout=60.0,
                enable_cleanup_closed=True,
            )

    @pytest.mark.asyncio
    async def test_session_uses_user_agent(self):
        proxy = EnterpriseProxy(config=ProxyConfig(user_agent="CustomAgent/2.0"))

        mock_connector = MagicMock()
        mock_timeout = MagicMock()

        with (
            patch("aiohttp.TCPConnector", return_value=mock_connector),
            patch("aiohttp.ClientTimeout", return_value=mock_timeout),
            patch("aiohttp.ClientSession", return_value=MagicMock()) as mock_session_cls,
        ):
            await proxy._ensure_session()

            mock_session_cls.assert_called_once_with(
                connector=mock_connector,
                timeout=mock_timeout,
                headers={"User-Agent": "CustomAgent/2.0"},
            )


# =============================================================================
# EnterpriseProxy Request Tests
# =============================================================================


class TestEnterpriseProxyRequest:
    """Tests for the request method and request handling pipeline."""

    @pytest.mark.asyncio
    async def test_request_framework_not_configured(self):
        proxy = EnterpriseProxy()
        with pytest.raises(FrameworkNotConfiguredError) as exc_info:
            await proxy.request(framework="nonexistent", method="GET", path="/test")
        assert exc_info.value.framework == "nonexistent"

    @pytest.mark.asyncio
    async def test_request_framework_disabled(self, disabled_framework_config):
        proxy = EnterpriseProxy(frameworks={"disabled": disabled_framework_config})
        with pytest.raises(ProxyError) as exc_info:
            await proxy.request(framework="disabled", method="GET", path="/test")
        assert exc_info.value.code == "FRAMEWORK_DISABLED"

    @pytest.mark.asyncio
    async def test_request_builds_url(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="openai",
            method="GET",
            path="/v1/models",
        )

        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["url"] == "https://api.openai.com/v1/models"

    @pytest.mark.asyncio
    async def test_request_merges_default_headers(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="openai",
            method="GET",
            path="/v1/models",
            headers={"X-Custom": "value"},
        )

        call_kwargs = mock_session.request.call_args[1]
        # Default headers from config + custom headers
        assert "X-Custom" in call_kwargs["headers"]

    @pytest.mark.asyncio
    async def test_request_tenant_context_injection(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(
                enable_audit_logging=False,
                tenant_header_name="X-Tenant-ID",
            ),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="openai",
            method="GET",
            path="/v1/models",
            tenant_id="tenant-abc",
        )

        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["headers"]["X-Tenant-ID"] == "tenant-abc"

    @pytest.mark.asyncio
    async def test_request_correlation_id_injection(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="openai",
            method="GET",
            path="/v1/models",
            correlation_id="corr-xyz",
        )

        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["headers"]["X-Correlation-ID"] == "corr-xyz"

    @pytest.mark.asyncio
    async def test_request_json_body_serialized(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="openai",
            method="POST",
            path="/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )

        call_kwargs = mock_session.request.call_args[1]
        body_data = json.loads(call_kwargs["data"])
        assert body_data["model"] == "gpt-4"
        assert call_kwargs["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_request_raw_data_body(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="openai",
            method="POST",
            path="/v1/audio",
            data=b"raw audio data",
        )

        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["data"] == b"raw audio data"

    @pytest.mark.asyncio
    async def test_request_returns_proxy_response(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session(status=200, body=b'{"result": "ok"}')
        proxy._session = mock_session

        response = await proxy.request(
            framework="openai",
            method="GET",
            path="/v1/models",
        )

        assert isinstance(response, ProxyResponse)
        assert response.status_code == 200
        assert response.body == b'{"result": "ok"}'
        assert response.framework == "openai"
        assert response.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_request_sanitization_validation(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        with pytest.raises(SanitizationError, match="Header injection"):
            await proxy.request(
                framework="openai",
                method="GET",
                path="/v1/models",
                headers={"Evil\nHeader": "value"},
            )

    @pytest.mark.asyncio
    async def test_request_suspicious_url_blocked(self):
        config = ExternalFrameworkConfig(base_url="https://api.example.com")
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        with pytest.raises(SanitizationError, match="Suspicious URL"):
            await proxy.request(
                framework="test",
                method="GET",
                path="/<script>alert(1)</script>",
            )


# =============================================================================
# EnterpriseProxy Pre/Post Hook Integration Tests
# =============================================================================


class TestEnterpriseProxyHookIntegration:
    """Tests for hook execution during request lifecycle."""

    @pytest.mark.asyncio
    async def test_pre_request_hook_modifies_request(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        async def add_header_hook(req):
            req.headers["X-Hook-Added"] = "true"
            return req

        proxy.add_pre_request_hook(add_header_hook)

        await proxy.request(
            framework="openai",
            method="GET",
            path="/v1/models",
        )

        call_kwargs = mock_session.request.call_args[1]
        assert "X-Hook-Added" in call_kwargs["headers"]

    @pytest.mark.asyncio
    async def test_pre_request_hook_aborts_on_none(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        async def abort_hook(req):
            return None

        proxy.add_pre_request_hook(abort_hook)

        with pytest.raises(ProxyError) as exc_info:
            await proxy.request(
                framework="openai",
                method="GET",
                path="/v1/models",
            )
        assert exc_info.value.code == "REQUEST_ABORTED"

    @pytest.mark.asyncio
    async def test_pre_request_hook_exception_raises_proxy_error(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        async def failing_hook(req):
            raise ValueError("hook failure")

        proxy.add_pre_request_hook(failing_hook)

        with pytest.raises(ProxyError) as exc_info:
            await proxy.request(
                framework="openai",
                method="GET",
                path="/v1/models",
            )
        assert exc_info.value.code == "HOOK_ERROR"

    @pytest.mark.asyncio
    async def test_post_request_hook_called(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        hook_called = []

        async def audit_hook(req, resp):
            hook_called.append((req.framework, resp.status_code))

        proxy.add_post_request_hook(audit_hook)

        await proxy.request(
            framework="openai",
            method="GET",
            path="/v1/models",
        )

        assert len(hook_called) == 1
        assert hook_called[0] == ("openai", 200)

    @pytest.mark.asyncio
    async def test_post_request_hook_exception_does_not_fail_request(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        async def failing_post_hook(req, resp):
            raise RuntimeError("post hook error")

        proxy.add_post_request_hook(failing_post_hook)

        # Should not raise despite hook failure
        response = await proxy.request(
            framework="openai",
            method="GET",
            path="/v1/models",
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_error_hook_called_on_failure(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_request(**kwargs):
            raise asyncio.TimeoutError()
            yield  # unreachable but needed for generator  # noqa: E501

        mock_session.request = MagicMock(side_effect=mock_request)
        proxy._session = mock_session

        error_hook_called = []

        async def error_hook(req, exc):
            error_hook_called.append(type(exc).__name__)

        proxy.add_error_hook(error_hook)

        with pytest.raises(RequestTimeoutError):
            await proxy.request(
                framework="openai",
                method="GET",
                path="/v1/models",
                skip_retry=True,
            )

        assert len(error_hook_called) == 1
        assert error_hook_called[0] == "RequestTimeoutError"

    @pytest.mark.asyncio
    async def test_error_hook_exception_does_not_suppress_original(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        # Force a circuit open error
        cb = proxy._circuit_breakers["openai"]
        for _ in range(5):
            await cb.record_failure()

        async def failing_error_hook(req, exc):
            raise RuntimeError("error hook crash")

        proxy.add_error_hook(failing_error_hook)

        with pytest.raises(CircuitOpenError):
            await proxy.request(
                framework="openai",
                method="GET",
                path="/v1/models",
            )


# =============================================================================
# EnterpriseProxy Circuit Breaker Integration Tests
# =============================================================================


class TestEnterpriseProxyCircuitBreakerIntegration:
    """Tests for circuit breaker integration in request flow."""

    @pytest.mark.asyncio
    async def test_circuit_open_blocks_request(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        cb = proxy._circuit_breakers["openai"]
        # Default failure_threshold = 5
        for _ in range(5):
            await cb.record_failure()

        with pytest.raises(CircuitOpenError) as exc_info:
            await proxy.request(
                framework="openai",
                method="GET",
                path="/v1/models",
            )
        assert exc_info.value.cooldown_remaining > 0

    @pytest.mark.asyncio
    async def test_skip_circuit_breaker_bypasses_check(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        cb = proxy._circuit_breakers["openai"]
        for _ in range(5):
            await cb.record_failure()

        # Should not raise even though circuit is open
        response = await proxy.request(
            framework="openai",
            method="GET",
            path="/v1/models",
            skip_circuit_breaker=True,
            skip_retry=True,
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_success_records_in_circuit_breaker(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session(status=200)
        proxy._session = mock_session

        await proxy.request(
            framework="openai",
            method="GET",
            path="/v1/models",
        )

        cb = proxy._circuit_breakers["openai"]
        assert cb._failures == 0

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        cb = proxy._circuit_breakers["openai"]
        for _ in range(5):
            await cb.record_failure()
        assert cb.state == "open"

        result = await proxy.reset_circuit_breaker("openai")
        assert result is True
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker_nonexistent(self):
        proxy = EnterpriseProxy()
        result = await proxy.reset_circuit_breaker("nonexistent")
        assert result is False


# =============================================================================
# EnterpriseProxy Timeout Tests
# =============================================================================


class TestEnterpriseProxyTimeout:
    """Tests for timeout handling in requests."""

    @pytest.mark.asyncio
    async def test_timeout_raises_request_timeout_error(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_request(**kwargs):
            raise asyncio.TimeoutError()
            yield  # noqa: E501

        mock_session.request = MagicMock(side_effect=mock_request)
        proxy._session = mock_session

        with pytest.raises(RequestTimeoutError) as exc_info:
            await proxy.request(
                framework="openai",
                method="GET",
                path="/v1/models",
                timeout=5.0,
                skip_retry=True,
            )
        assert exc_info.value.framework == "openai"
        assert exc_info.value.timeout == 5.0

    @pytest.mark.asyncio
    async def test_timeout_uses_framework_default(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_request(**kwargs):
            raise asyncio.TimeoutError()
            yield  # noqa: E501

        mock_session.request = MagicMock(side_effect=mock_request)
        proxy._session = mock_session

        with pytest.raises(RequestTimeoutError) as exc_info:
            await proxy.request(
                framework="openai",
                method="GET",
                path="/v1/models",
                skip_retry=True,
            )
        assert exc_info.value.timeout == openai_framework_config.timeout


# =============================================================================
# EnterpriseProxy Retry Tests
# =============================================================================


class TestEnterpriseProxyRetry:
    """Tests for retry logic with backoff strategies."""

    def test_calculate_retry_delay_exponential(self):
        proxy = EnterpriseProxy()
        settings = RetrySettings(
            base_delay=1.0, max_delay=30.0, strategy=RetryStrategy.EXPONENTIAL, jitter=False
        )
        assert proxy._calculate_retry_delay(0, settings) == 1.0
        assert proxy._calculate_retry_delay(1, settings) == 2.0
        assert proxy._calculate_retry_delay(2, settings) == 4.0
        assert proxy._calculate_retry_delay(3, settings) == 8.0

    def test_calculate_retry_delay_linear(self):
        proxy = EnterpriseProxy()
        settings = RetrySettings(
            base_delay=1.0, max_delay=30.0, strategy=RetryStrategy.LINEAR, jitter=False
        )
        assert proxy._calculate_retry_delay(0, settings) == 1.0
        assert proxy._calculate_retry_delay(1, settings) == 2.0
        assert proxy._calculate_retry_delay(2, settings) == 3.0
        assert proxy._calculate_retry_delay(3, settings) == 4.0

    def test_calculate_retry_delay_constant(self):
        proxy = EnterpriseProxy()
        settings = RetrySettings(
            base_delay=2.0, max_delay=30.0, strategy=RetryStrategy.CONSTANT, jitter=False
        )
        assert proxy._calculate_retry_delay(0, settings) == 2.0
        assert proxy._calculate_retry_delay(1, settings) == 2.0
        assert proxy._calculate_retry_delay(5, settings) == 2.0

    def test_calculate_retry_delay_capped_at_max(self):
        proxy = EnterpriseProxy()
        settings = RetrySettings(
            base_delay=1.0, max_delay=5.0, strategy=RetryStrategy.EXPONENTIAL, jitter=False
        )
        # 2^10 = 1024, but capped at 5.0
        assert proxy._calculate_retry_delay(10, settings) == 5.0

    def test_calculate_retry_delay_with_jitter(self):
        proxy = EnterpriseProxy()
        settings = RetrySettings(
            base_delay=1.0, max_delay=30.0, strategy=RetryStrategy.CONSTANT, jitter=True
        )
        delays = [proxy._calculate_retry_delay(0, settings) for _ in range(50)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1
        # Jitter factor is 0.25, so range is [0.75, 1.25]
        for d in delays:
            assert 0.5 <= d <= 1.5

    @pytest.mark.asyncio
    async def test_retry_on_retryable_status_code(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=2, base_delay=0.01, jitter=False),
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        call_count = 0

        @asynccontextmanager
        async def mock_request(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = AsyncMock()
            mock_resp.status = 503 if call_count < 3 else 200
            mock_resp.headers = {}
            mock_resp.read = AsyncMock(return_value=b'{"ok": true}')
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.request = MagicMock(side_effect=mock_request)
        proxy._session = mock_session

        response = await proxy.request(
            framework="test",
            method="GET",
            path="/test",
        )

        assert response.status_code == 200
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_returns_last_response(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=1, base_delay=0.01, jitter=False),
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        @asynccontextmanager
        async def mock_request(**kwargs):
            mock_resp = AsyncMock()
            mock_resp.status = 503
            mock_resp.headers = {}
            mock_resp.read = AsyncMock(return_value=b"error")
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.request = MagicMock(side_effect=mock_request)
        proxy._session = mock_session

        response = await proxy.request(
            framework="test",
            method="GET",
            path="/test",
        )

        # After retries exhausted, returns last response
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_skip_retry(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=3),
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        call_count = 0

        @asynccontextmanager
        async def mock_request(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = AsyncMock()
            mock_resp.status = 503
            mock_resp.headers = {}
            mock_resp.read = AsyncMock(return_value=b"error")
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.request = MagicMock(side_effect=mock_request)
        proxy._session = mock_session

        response = await proxy.request(
            framework="test",
            method="GET",
            path="/test",
            skip_retry=True,
        )

        # Only called once, no retries
        assert call_count == 1
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=2, base_delay=0.01, jitter=False),
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        call_count = 0

        @asynccontextmanager
        async def mock_request(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.headers = {}
            mock_resp.read = AsyncMock(return_value=b'{"ok": true}')
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.request = MagicMock(side_effect=mock_request)
        proxy._session = mock_session

        response = await proxy.request(
            framework="test",
            method="GET",
            path="/test",
        )

        assert response.status_code == 200
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_records_failures_in_circuit_breaker(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=1, base_delay=0.01, jitter=False),
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        @asynccontextmanager
        async def mock_request(**kwargs):
            raise ConnectionError("Connection refused")
            yield  # noqa: E501

        mock_session = AsyncMock()
        mock_session.request = MagicMock(side_effect=mock_request)
        proxy._session = mock_session

        with pytest.raises(ConnectionError):
            await proxy.request(framework="test", method="GET", path="/test")

        cb = proxy._circuit_breakers["test"]
        assert cb._failures >= 1


# =============================================================================
# EnterpriseProxy Health Check Tests
# =============================================================================


class TestEnterpriseProxyHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_check_health_unconfigured_framework(self):
        proxy = EnterpriseProxy()
        result = await proxy.check_health("nonexistent")
        assert result.status == HealthStatus.UNKNOWN
        assert result.error == "Framework not configured"

    @pytest.mark.asyncio
    async def test_check_health_no_health_path(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            health_check_path=None,
        )
        proxy = EnterpriseProxy(frameworks={"test": config})
        result = await proxy.check_health("test")
        assert result.status == HealthStatus.UNKNOWN
        assert "No health check path" in result.error

    @pytest.mark.asyncio
    async def test_check_health_healthy(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            health_check_path="/health",
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        mock_session = _make_mock_session(status=200)
        proxy._session = mock_session

        result = await proxy.check_health("test")
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms is not None
        assert result.error is None
        assert result.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_check_health_degraded_non_2xx(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            health_check_path="/health",
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        mock_session = _make_mock_session(status=503)
        proxy._session = mock_session

        result = await proxy.check_health("test")
        assert result.status == HealthStatus.DEGRADED
        assert "503" in result.error

    @pytest.mark.asyncio
    async def test_check_health_unhealthy_on_exception(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            health_check_path="/health",
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_request(**kwargs):
            raise asyncio.TimeoutError()
            yield  # noqa: E501

        mock_session.request = MagicMock(side_effect=mock_request)
        proxy._session = mock_session

        result = await proxy.check_health("test")
        assert result.status == HealthStatus.UNHEALTHY
        assert result.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_check_health_increments_consecutive_failures(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            health_check_path="/health",
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_request(**kwargs):
            raise asyncio.TimeoutError()
            yield  # noqa: E501

        mock_session.request = MagicMock(side_effect=mock_request)
        proxy._session = mock_session

        result1 = await proxy.check_health("test")
        assert result1.consecutive_failures == 1

        result2 = await proxy.check_health("test")
        assert result2.consecutive_failures == 2

    @pytest.mark.asyncio
    async def test_check_all_health(self):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={
                "fw1": ExternalFrameworkConfig(
                    base_url="https://fw1.com",
                    health_check_path="/health",
                ),
                "fw2": ExternalFrameworkConfig(
                    base_url="https://fw2.com",
                    health_check_path=None,
                ),
            },
        )

        mock_session = _make_mock_session(status=200)
        proxy._session = mock_session

        results = await proxy.check_all_health()
        assert "fw1" in results
        assert "fw2" in results
        assert results["fw1"].status == HealthStatus.HEALTHY
        assert results["fw2"].status == HealthStatus.UNKNOWN


# =============================================================================
# EnterpriseProxy Monitoring and Statistics Tests
# =============================================================================


class TestEnterpriseProxyMonitoring:
    """Tests for monitoring endpoints and statistics."""

    def test_get_circuit_breaker_status_all(self, proxy_with_frameworks):
        status = proxy_with_frameworks.get_circuit_breaker_status()
        assert "openai" in status
        assert "anthropic" in status
        assert status["openai"]["state"] == "closed"

    def test_get_circuit_breaker_status_single(self, proxy_with_frameworks):
        status = proxy_with_frameworks.get_circuit_breaker_status("openai")
        assert status["framework"] == "openai"
        assert status["state"] == "closed"

    def test_get_circuit_breaker_status_missing(self, proxy_with_frameworks):
        status = proxy_with_frameworks.get_circuit_breaker_status("nonexistent")
        assert status == {}

    def test_get_bulkhead_status_all(self, proxy_with_frameworks):
        status = proxy_with_frameworks.get_bulkhead_status()
        assert "openai" in status
        assert "anthropic" in status

    def test_get_bulkhead_status_single(self, proxy_with_frameworks):
        status = proxy_with_frameworks.get_bulkhead_status("openai")
        assert status["framework"] == "openai"
        assert "active" in status
        assert "max_concurrent" in status

    def test_get_bulkhead_status_missing(self, proxy_with_frameworks):
        status = proxy_with_frameworks.get_bulkhead_status("nonexistent")
        assert status == {}

    def test_get_health_status_all(self, proxy_with_frameworks):
        status = proxy_with_frameworks.get_health_status()
        assert "openai" in status
        assert "anthropic" in status

    def test_get_health_status_single(self, proxy_with_frameworks):
        status = proxy_with_frameworks.get_health_status("openai")
        assert status["framework"] == "openai"
        assert status["status"] == "unknown"

    def test_get_health_status_missing(self, proxy_with_frameworks):
        status = proxy_with_frameworks.get_health_status("nonexistent")
        assert status == {}

    def test_get_stats(self, proxy_with_frameworks):
        stats = proxy_with_frameworks.get_stats()
        assert "config" in stats
        assert "frameworks" in stats
        assert "circuit_breakers" in stats
        assert "bulkheads" in stats
        assert "health" in stats
        assert "hooks" in stats

        assert stats["config"]["max_connections"] == 100
        assert "openai" in stats["frameworks"]
        assert stats["frameworks"]["openai"]["enabled"] is True
        assert stats["hooks"]["pre_request"] == 0

    def test_get_stats_with_hooks(self, proxy_with_frameworks):
        async def hook1(req):
            return req

        async def hook2(req, resp):
            pass

        async def hook3(req, exc):
            pass

        proxy_with_frameworks.add_pre_request_hook(hook1)
        proxy_with_frameworks.add_post_request_hook(hook2)
        proxy_with_frameworks.add_error_hook(hook3)

        stats = proxy_with_frameworks.get_stats()
        assert stats["hooks"]["pre_request"] == 1
        assert stats["hooks"]["post_request"] == 1
        assert stats["hooks"]["error"] == 1


# =============================================================================
# EnterpriseProxy Audit Logging Tests
# =============================================================================


class TestEnterpriseProxyAuditLogging:
    """Tests for audit logging behavior."""

    @pytest.mark.asyncio
    async def test_audit_logging_enabled(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=True),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        with patch.object(proxy, "_log_request") as mock_log:
            await proxy.request(
                framework="openai",
                method="GET",
                path="/v1/models",
                skip_retry=True,
            )
            mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_audit_logging_disabled(self, openai_framework_config):
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        with patch.object(proxy, "_log_request") as mock_log:
            await proxy.request(
                framework="openai",
                method="GET",
                path="/v1/models",
                skip_retry=True,
            )
            mock_log.assert_not_called()


# =============================================================================
# EnterpriseProxy Custom Tenant Header Tests
# =============================================================================


class TestEnterpriseProxyTenantContext:
    """Tests for tenant context injection with custom headers."""

    @pytest.mark.asyncio
    async def test_custom_tenant_header_name(self):
        config = ExternalFrameworkConfig(base_url="https://api.test.com")
        proxy = EnterpriseProxy(
            config=ProxyConfig(
                enable_audit_logging=False,
                tenant_header_name="X-Custom-Tenant",
            ),
            frameworks={"test": config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="test",
            method="GET",
            path="/test",
            tenant_id="tenant-456",
        )

        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["headers"]["X-Custom-Tenant"] == "tenant-456"

    @pytest.mark.asyncio
    async def test_custom_correlation_header_name(self):
        config = ExternalFrameworkConfig(base_url="https://api.test.com")
        proxy = EnterpriseProxy(
            config=ProxyConfig(
                enable_audit_logging=False,
                correlation_header_name="X-Request-ID",
            ),
            frameworks={"test": config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="test",
            method="GET",
            path="/test",
            correlation_id="req-789",
        )

        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["headers"]["X-Request-ID"] == "req-789"

    @pytest.mark.asyncio
    async def test_no_tenant_header_when_not_provided(self):
        config = ExternalFrameworkConfig(base_url="https://api.test.com")
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="test",
            method="GET",
            path="/test",
        )

        call_kwargs = mock_session.request.call_args[1]
        assert "X-Tenant-ID" not in call_kwargs["headers"]


# =============================================================================
# Edge Cases and Additional Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_proxy_response_boundary_status_codes(self):
        # 199 is not success, not client error, not server error
        resp = ProxyResponse(status_code=199, headers={}, body=b"", elapsed_ms=0, framework="test")
        assert resp.is_success is False
        assert resp.is_client_error is False
        assert resp.is_server_error is False

        # 300 is not success
        resp = ProxyResponse(status_code=300, headers={}, body=b"", elapsed_ms=0, framework="test")
        assert resp.is_success is False
        assert resp.is_client_error is False
        assert resp.is_server_error is False

    def test_sanitize_body_binary_data(self):
        settings = SanitizationSettings()
        sanitizer = RequestSanitizer(settings)
        # Binary data that cannot be decoded as UTF-8 within max size
        body = bytes(range(256)) * 10
        result = sanitizer.sanitize_body_for_logging(body)
        # Should handle gracefully
        assert isinstance(result, str)

    def test_circuit_breaker_settings_edge_values(self):
        settings = CircuitBreakerSettings(
            failure_threshold=1,
            success_threshold=1,
            cooldown_seconds=0.001,
            half_open_max_calls=1,
        )
        assert settings.failure_threshold == 1

    def test_retry_settings_zero_retries(self):
        settings = RetrySettings(max_retries=0)
        assert settings.max_retries == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrent_state_access(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)

        async def record_failures():
            for _ in range(10):
                await cb.record_failure()

        async def record_successes():
            for _ in range(10):
                await cb.record_success()

        # Run concurrently to test lock safety
        await asyncio.gather(record_failures(), record_successes())
        # Just verify it doesn't crash

    @pytest.mark.asyncio
    async def test_bulkhead_concurrent_acquire_release(self):
        bh = FrameworkBulkhead(
            "test",
            BulkheadSettings(max_concurrent=5, wait_timeout=5.0),
        )

        async def use_slot():
            async with bh.acquire():
                await asyncio.sleep(0.01)

        # All 10 tasks should complete since max_concurrent=5 and they release quickly
        await asyncio.gather(*[use_slot() for _ in range(10)])
        assert bh.active_count == 0

    def test_framework_config_base_url_no_trailing_slash(self):
        config = ExternalFrameworkConfig(base_url="https://api.example.com")
        assert config.base_url == "https://api.example.com"

    @pytest.mark.asyncio
    async def test_proxy_method_uppercased(self):
        config = ExternalFrameworkConfig(base_url="https://api.test.com")
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="test",
            method="post",  # lowercase
            path="/test",
        )

        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_proxy_json_sets_content_type_if_not_present(self):
        config = ExternalFrameworkConfig(base_url="https://api.test.com")
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="test",
            method="POST",
            path="/test",
            json={"key": "value"},
        )

        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_proxy_json_does_not_override_content_type(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            default_headers={"Content-Type": "application/json; charset=utf-8"},
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        mock_session = _make_mock_session()
        proxy._session = mock_session

        await proxy.request(
            framework="test",
            method="POST",
            path="/test",
            json={"key": "value"},
        )

        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["headers"]["Content-Type"] == "application/json; charset=utf-8"

    @pytest.mark.asyncio
    async def test_start_creates_health_check_task_when_paths_configured(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            health_check_path="/health",
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        with patch.object(proxy, "_ensure_session", new_callable=AsyncMock):
            with patch.object(proxy, "_health_check_loop", new_callable=AsyncMock):
                await proxy.start()
                assert proxy._health_check_task is not None

        await proxy.shutdown()

    @pytest.mark.asyncio
    async def test_start_no_health_check_task_without_paths(self):
        config = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            health_check_path=None,
        )
        proxy = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"test": config},
        )

        with patch.object(proxy, "_ensure_session", new_callable=AsyncMock):
            await proxy.start()
            assert proxy._health_check_task is None
