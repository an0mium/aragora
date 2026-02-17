"""
Comprehensive tests for aragora.gateway.enterprise.proxy module.

Tests cover:
1.  Proxy initialization and configuration
2.  Request/response lifecycle
3.  Circuit breaker state transitions (closed -> open -> half-open -> closed)
4.  Bulkhead concurrent request limiting
5.  Retry with backoff strategies (exponential, linear, constant)
6.  Request sanitization (headers, body)
7.  Health check probes
8.  Pre/post request hooks
9.  Timeout handling
10. Error handling and custom exceptions
11. Connection pooling
12. Tenant context injection
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.enterprise.proxy import (
    BulkheadFullError,
    BulkheadSettings,
    CircuitBreakerSettings,
    CircuitOpenError,
    EnterpriseProxy,
    ExternalFrameworkConfig,
    FrameworkBulkhead,
    FrameworkCircuitBreaker,
    FrameworkNotConfiguredError,
    HealthCheckResult,
    HealthStatus,
    ProxyConfig,
    ProxyError,
    ProxyRequest,
    ProxyResponse,
    RequestSanitizer,
    RequestTimeoutError,
    RetrySettings,
    RetryStrategy,
    SanitizationError,
    SanitizationSettings,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_mock_session(status=200, body=b'{"ok": true}', headers=None):
    """Return a mock aiohttp session with a configurable response."""
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


def _make_failing_mock_session(exc_class=asyncio.TimeoutError, exc_args=()):
    """Return a mock aiohttp session that always raises *exc_class*."""
    mock_session = AsyncMock()

    @asynccontextmanager
    async def mock_request(**kwargs):
        raise exc_class(*exc_args)
        yield  # noqa: E501 – unreachable but required for generator syntax

    mock_session.request = MagicMock(side_effect=mock_request)
    return mock_session


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_proxy_config():
    return ProxyConfig()


@pytest.fixture
def custom_proxy_config():
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
    return ExternalFrameworkConfig(
        base_url="https://api.openai.com",
        timeout=60.0,
        connect_timeout=10.0,
        default_headers={"Authorization": "Bearer sk-test"},
        health_check_path="/v1/health",
    )


@pytest.fixture
def anthropic_framework_config():
    return ExternalFrameworkConfig(
        base_url="https://api.anthropic.com/",
        timeout=120.0,
        circuit_breaker=CircuitBreakerSettings(failure_threshold=3, cooldown_seconds=30.0),
        retry=RetrySettings(max_retries=2, base_delay=1.0),
        bulkhead=BulkheadSettings(max_concurrent=20, wait_timeout=5.0),
    )


@pytest.fixture
def disabled_framework_config():
    return ExternalFrameworkConfig(base_url="https://api.disabled.com", enabled=False)


@pytest.fixture
def basic_cb_settings():
    return CircuitBreakerSettings(
        failure_threshold=3,
        success_threshold=2,
        cooldown_seconds=10.0,
        half_open_max_calls=2,
    )


@pytest.fixture
def basic_retry_settings():
    return RetrySettings(
        max_retries=3,
        base_delay=0.5,
        max_delay=10.0,
        strategy=RetryStrategy.EXPONENTIAL,
        jitter=False,
    )


@pytest.fixture
def basic_bulkhead_settings():
    return BulkheadSettings(max_concurrent=3, wait_timeout=2.0)


@pytest.fixture
def basic_sanitization_settings():
    return SanitizationSettings()


@pytest.fixture
def circuit_breaker(basic_cb_settings):
    return FrameworkCircuitBreaker("test-framework", basic_cb_settings)


@pytest.fixture
def bulkhead(basic_bulkhead_settings):
    return FrameworkBulkhead("test-framework", basic_bulkhead_settings)


@pytest.fixture
def sanitizer(basic_sanitization_settings):
    return RequestSanitizer(basic_sanitization_settings)


@pytest.fixture
def proxy_with_frameworks(openai_framework_config, anthropic_framework_config):
    return EnterpriseProxy(
        config=ProxyConfig(enable_audit_logging=False),
        frameworks={"openai": openai_framework_config, "anthropic": anthropic_framework_config},
    )


@pytest.fixture
def sample_proxy_request():
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
    return ProxyResponse(
        status_code=200,
        headers={"Content-Type": "application/json"},
        body=b'{"id": "chatcmpl-123"}',
        elapsed_ms=450.0,
        framework="openai",
        correlation_id="corr-abc",
    )


# ============================================================================
# 10. Error handling and custom exceptions
# ============================================================================


class TestProxyError:
    def test_proxy_error_basic(self):
        err = ProxyError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.code == "PROXY_ERROR"
        assert err.framework is None
        assert err.details == {}

    def test_proxy_error_with_all_fields(self):
        err = ProxyError("fail", code="CUSTOM_CODE", framework="openai", details={"key": "value"})
        assert err.code == "CUSTOM_CODE"
        assert err.framework == "openai"
        assert err.details == {"key": "value"}

    def test_proxy_error_is_exception(self):
        assert isinstance(ProxyError("test"), Exception)

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

    def test_circuit_open_error_default_details(self):
        assert CircuitOpenError("fw", 5.0).details == {"cooldown_remaining": 5.0}

    def test_bulkhead_full_error(self):
        err = BulkheadFullError("anthropic", 50)
        assert isinstance(err, ProxyError)
        assert err.framework == "anthropic"
        assert err.max_concurrent == 50
        assert err.code == "BULKHEAD_FULL"
        assert "50" in str(err)

    def test_bulkhead_full_error_with_details(self):
        assert BulkheadFullError("x", 50, details={"q": 10}).details == {"q": 10}

    def test_bulkhead_full_error_default_details(self):
        assert BulkheadFullError("fw", 25).details == {"max_concurrent": 25}

    def test_request_timeout_error(self):
        err = RequestTimeoutError("openai", 30.0)
        assert isinstance(err, ProxyError)
        assert err.framework == "openai"
        assert err.timeout == 30.0
        assert err.code == "REQUEST_TIMEOUT"
        assert "30.0s" in str(err)

    def test_request_timeout_error_with_details(self):
        assert RequestTimeoutError("x", 30.0, details={"attempt": 2}).details == {"attempt": 2}

    def test_request_timeout_error_default_details(self):
        assert RequestTimeoutError("fw", 10.0).details == {"timeout": 10.0}

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
        err = SanitizationError("Bad", framework="openai", details={"header": "X-Evil"})
        assert err.details == {"header": "X-Evil"}


# ============================================================================
# Enum tests
# ============================================================================


class TestEnums:
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


# ============================================================================
# 1. Proxy initialization and configuration
# ============================================================================


class TestCircuitBreakerSettings:
    def test_defaults(self):
        s = CircuitBreakerSettings()
        assert s.failure_threshold == 5
        assert s.success_threshold == 2
        assert s.cooldown_seconds == 60.0
        assert s.half_open_max_calls == 3

    def test_custom_values(self):
        s = CircuitBreakerSettings(
            failure_threshold=10, success_threshold=5, cooldown_seconds=120.0, half_open_max_calls=5
        )
        assert s.failure_threshold == 10
        assert s.success_threshold == 5

    def test_failure_threshold_validation(self):
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreakerSettings(failure_threshold=0)

    def test_success_threshold_validation(self):
        with pytest.raises(ValueError, match="success_threshold"):
            CircuitBreakerSettings(success_threshold=0)

    def test_cooldown_seconds_validation_zero(self):
        with pytest.raises(ValueError, match="cooldown_seconds"):
            CircuitBreakerSettings(cooldown_seconds=0)

    def test_cooldown_seconds_validation_negative(self):
        with pytest.raises(ValueError, match="cooldown_seconds"):
            CircuitBreakerSettings(cooldown_seconds=-5)

    def test_half_open_max_calls_validation(self):
        with pytest.raises(ValueError, match="half_open_max_calls"):
            CircuitBreakerSettings(half_open_max_calls=0)

    def test_minimum_valid_values(self):
        s = CircuitBreakerSettings(
            failure_threshold=1, success_threshold=1, cooldown_seconds=0.001, half_open_max_calls=1
        )
        assert s.failure_threshold == 1


class TestRetrySettings:
    def test_defaults(self):
        s = RetrySettings()
        assert s.max_retries == 3
        assert s.base_delay == 0.5
        assert s.max_delay == 30.0
        assert s.strategy == RetryStrategy.EXPONENTIAL
        assert s.jitter is True
        for code in (429, 500, 502, 503, 504):
            assert code in s.retryable_status_codes

    def test_custom_values(self):
        s = RetrySettings(
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0,
            strategy=RetryStrategy.LINEAR,
            jitter=False,
            retryable_status_codes=frozenset({500}),
        )
        assert s.max_retries == 5
        assert s.strategy == RetryStrategy.LINEAR
        assert s.retryable_status_codes == frozenset({500})

    def test_negative_max_retries(self):
        with pytest.raises(ValueError, match="max_retries"):
            RetrySettings(max_retries=-1)

    def test_zero_max_retries_allowed(self):
        assert RetrySettings(max_retries=0).max_retries == 0

    def test_base_delay_zero(self):
        with pytest.raises(ValueError, match="base_delay"):
            RetrySettings(base_delay=0)

    def test_base_delay_negative(self):
        with pytest.raises(ValueError, match="base_delay"):
            RetrySettings(base_delay=-1)

    def test_max_delay_less_than_base(self):
        with pytest.raises(ValueError, match="max_delay"):
            RetrySettings(base_delay=10.0, max_delay=5.0)

    def test_max_delay_equal_to_base(self):
        s = RetrySettings(base_delay=5.0, max_delay=5.0)
        assert s.max_delay == s.base_delay


class TestBulkheadSettings:
    def test_defaults(self):
        s = BulkheadSettings()
        assert s.max_concurrent == 50
        assert s.wait_timeout == 10.0

    def test_custom(self):
        assert BulkheadSettings(max_concurrent=100, wait_timeout=30.0).max_concurrent == 100

    def test_max_concurrent_validation(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            BulkheadSettings(max_concurrent=0)

    def test_wait_timeout_zero(self):
        with pytest.raises(ValueError, match="wait_timeout"):
            BulkheadSettings(wait_timeout=0)

    def test_wait_timeout_negative(self):
        with pytest.raises(ValueError, match="wait_timeout"):
            BulkheadSettings(wait_timeout=-1)


class TestSanitizationSettings:
    def test_defaults(self):
        s = SanitizationSettings()
        assert "authorization" in s.redact_headers
        assert "x-api-key" in s.redact_headers
        assert "api-key" in s.redact_headers
        assert "cookie" in s.redact_headers
        assert "set-cookie" in s.redact_headers
        assert "x-auth-token" in s.redact_headers
        assert len(s.redact_body_patterns) == 4
        assert s.max_body_log_size == 4096
        assert "x-forwarded-for" in s.strip_sensitive_headers
        assert "x-real-ip" in s.strip_sensitive_headers

    def test_custom_redact_headers(self):
        s = SanitizationSettings(redact_headers=frozenset({"x-custom-secret"}))
        assert "x-custom-secret" in s.redact_headers
        assert "authorization" not in s.redact_headers


class TestExternalFrameworkConfig:
    def test_basic(self):
        c = ExternalFrameworkConfig(base_url="https://api.example.com")
        assert c.base_url == "https://api.example.com"
        assert c.timeout == 30.0
        assert c.connect_timeout == 10.0
        assert c.enabled is True
        assert c.health_check_path is None
        assert c.health_check_interval == 30.0

    def test_trailing_slash_normalized(self):
        assert ExternalFrameworkConfig(base_url="https://x.com/").base_url == "https://x.com"

    def test_multiple_trailing_slashes(self):
        assert not ExternalFrameworkConfig(base_url="https://x.com///").base_url.endswith("/")

    def test_empty_base_url(self):
        with pytest.raises(ValueError, match="base_url"):
            ExternalFrameworkConfig(base_url="")

    def test_timeout_validation(self):
        with pytest.raises(ValueError, match="timeout"):
            ExternalFrameworkConfig(base_url="https://x.com", timeout=0)

    def test_negative_timeout(self):
        with pytest.raises(ValueError, match="timeout"):
            ExternalFrameworkConfig(base_url="https://x.com", timeout=-5)

    def test_connect_timeout_zero(self):
        with pytest.raises(ValueError, match="connect_timeout"):
            ExternalFrameworkConfig(base_url="https://x.com", connect_timeout=0)

    def test_connect_timeout_negative(self):
        with pytest.raises(ValueError, match="connect_timeout"):
            ExternalFrameworkConfig(base_url="https://x.com", connect_timeout=-1)

    def test_health_check_interval_validation(self):
        with pytest.raises(ValueError, match="health_check_interval"):
            ExternalFrameworkConfig(base_url="https://x.com", health_check_interval=0)

    def test_default_headers(self):
        c = ExternalFrameworkConfig(base_url="https://x.com", default_headers={"K": "V"})
        assert c.default_headers["K"] == "V"

    def test_metadata(self):
        c = ExternalFrameworkConfig(base_url="https://x.com", metadata={"p": "openai"})
        assert c.metadata["p"] == "openai"

    def test_disabled(self):
        assert ExternalFrameworkConfig(base_url="https://x.com", enabled=False).enabled is False

    def test_sub_dataclass_defaults(self):
        c = ExternalFrameworkConfig(base_url="https://x.com")
        assert isinstance(c.circuit_breaker, CircuitBreakerSettings)
        assert isinstance(c.retry, RetrySettings)
        assert isinstance(c.bulkhead, BulkheadSettings)
        assert isinstance(c.sanitization, SanitizationSettings)


class TestProxyConfig:
    def test_defaults(self):
        c = ProxyConfig()
        assert c.default_timeout == 30.0
        assert c.default_connect_timeout == 10.0
        assert c.max_connections == 100
        assert c.max_connections_per_host == 10
        assert c.keepalive_timeout == 30.0
        assert c.enable_connection_pooling is True
        assert c.enable_audit_logging is True
        assert c.enable_metrics is True
        assert c.tenant_header_name == "X-Tenant-ID"
        assert c.correlation_header_name == "X-Correlation-ID"
        assert c.user_agent == "Aragora-EnterpriseProxy/1.0"

    def test_max_connections_validation(self):
        with pytest.raises(ValueError, match="max_connections must be at least 1"):
            ProxyConfig(max_connections=0)

    def test_max_connections_per_host_validation(self):
        with pytest.raises(ValueError, match="max_connections_per_host must be at least 1"):
            ProxyConfig(max_connections_per_host=0)

    def test_custom(self, custom_proxy_config):
        assert custom_proxy_config.default_timeout == 60.0
        assert custom_proxy_config.user_agent == "TestProxy/1.0"
        assert custom_proxy_config.tenant_header_name == "X-Custom-Tenant"


# ============================================================================
# 2. Request / response lifecycle
# ============================================================================


class TestProxyRequest:
    def test_basic(self):
        r = ProxyRequest(
            framework="openai",
            method="POST",
            url="https://api.openai.com/v1/completions",
            headers={"Content-Type": "application/json"},
        )
        assert r.framework == "openai"
        assert r.method == "POST"
        assert r.body is None
        assert r.tenant_id is None
        assert r.correlation_id is None
        assert r.auth_context is None
        assert isinstance(r.metadata, dict)
        assert isinstance(r.timestamp, float)

    def test_body_hash_with_body(self):
        body = b"test body content"
        r = ProxyRequest(framework="t", method="POST", url="https://x.com", headers={}, body=body)
        assert r.body_hash() == hashlib.sha256(body).hexdigest()
        assert len(r.body_hash()) == 64

    def test_body_hash_without_body(self):
        r = ProxyRequest(framework="t", method="GET", url="https://x.com", headers={})
        assert r.body_hash() is None

    def test_metadata_default(self):
        assert (
            ProxyRequest(framework="t", method="GET", url="https://x.com", headers={}).metadata
            == {}
        )

    def test_full_request(self):
        r = ProxyRequest(
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
        assert r.tenant_id == "t-1"
        assert r.correlation_id == "c-1"
        assert r.auth_context == {"user": "admin"}
        assert r.metadata == {"attempt": 1}

    def test_timestamp_auto_set(self):
        before = time.time()
        r = ProxyRequest(framework="t", method="GET", url="http://x", headers={})
        assert before <= r.timestamp <= time.time()


class TestProxyResponse:
    def test_basic(self):
        r = ProxyResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"ok": true}',
            elapsed_ms=123.4,
            framework="openai",
        )
        assert r.status_code == 200
        assert r.elapsed_ms == 123.4
        assert r.from_cache is False
        assert r.correlation_id is None

    def test_is_success_2xx(self):
        for code in (200, 201, 204, 299):
            r = ProxyResponse(status_code=code, headers={}, body=b"", elapsed_ms=0, framework="t")
            assert r.is_success is True
            assert r.is_client_error is False
            assert r.is_server_error is False

    def test_is_client_error_4xx(self):
        for code in (400, 401, 403, 404, 429, 499):
            r = ProxyResponse(status_code=code, headers={}, body=b"", elapsed_ms=0, framework="t")
            assert r.is_client_error is True
            assert r.is_success is False
            assert r.is_server_error is False

    def test_is_server_error_5xx(self):
        for code in (500, 502, 503, 504, 599):
            r = ProxyResponse(status_code=code, headers={}, body=b"", elapsed_ms=0, framework="t")
            assert r.is_server_error is True
            assert r.is_success is False
            assert r.is_client_error is False

    def test_boundary_199(self):
        r = ProxyResponse(status_code=199, headers={}, body=b"", elapsed_ms=0, framework="t")
        assert not r.is_success and not r.is_client_error and not r.is_server_error

    def test_boundary_300(self):
        r = ProxyResponse(status_code=300, headers={}, body=b"", elapsed_ms=0, framework="t")
        assert not r.is_success and not r.is_client_error and not r.is_server_error

    def test_body_hash(self):
        body = b"response content"
        r = ProxyResponse(status_code=200, headers={}, body=body, elapsed_ms=0, framework="t")
        assert r.body_hash() == hashlib.sha256(body).hexdigest()

    def test_body_hash_empty(self):
        r = ProxyResponse(status_code=200, headers={}, body=b"", elapsed_ms=0, framework="t")
        assert r.body_hash() == hashlib.sha256(b"").hexdigest()

    def test_from_cache_flag(self):
        r = ProxyResponse(
            status_code=200, headers={}, body=b"", elapsed_ms=0, framework="t", from_cache=True
        )
        assert r.from_cache is True

    def test_metadata(self):
        r = ProxyResponse(
            status_code=200,
            headers={},
            body=b"",
            elapsed_ms=0,
            framework="t",
            metadata={"cache_key": "abc"},
        )
        assert r.metadata == {"cache_key": "abc"}

    def test_timestamp_auto_set(self):
        before = time.time()
        r = ProxyResponse(status_code=200, headers={}, body=b"", elapsed_ms=0, framework="t")
        assert before <= r.timestamp <= time.time()


# ============================================================================
# 3. Circuit breaker state transitions
# ============================================================================


class TestFrameworkCircuitBreaker:
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
        assert await circuit_breaker.record_failure() is False
        assert circuit_breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_threshold_opens_circuit(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        await cb.record_failure()
        await cb.record_failure()
        assert await cb.record_failure() is True
        assert cb.state == "open"
        assert cb.is_open is True

    @pytest.mark.asyncio
    async def test_open_blocks_requests(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        assert await cb.can_proceed() is False

    @pytest.mark.asyncio
    async def test_cooldown_remaining_when_open(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        r = cb.cooldown_remaining
        assert 0 < r <= basic_cb_settings.cooldown_seconds

    @pytest.mark.asyncio
    async def test_transition_to_half_open(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        cb._open_at = time.time() - basic_cb_settings.cooldown_seconds - 1
        assert cb.state == "half-open"
        assert cb.is_open is False

    @pytest.mark.asyncio
    async def test_half_open_limited_calls(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        cb._open_at = time.time() - basic_cb_settings.cooldown_seconds - 1
        assert await cb.can_proceed() is True
        assert await cb.can_proceed() is True
        assert await cb.can_proceed() is False

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        cb._open_at = time.time() - basic_cb_settings.cooldown_seconds - 1
        await cb.record_success()
        assert cb.state == "half-open"
        await cb.record_success()
        assert cb.state == "closed"
        assert cb._open_at is None
        assert cb._successes == 0
        assert cb._half_open_calls == 0

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
        d = circuit_breaker.to_dict()
        assert d["framework"] == "test-framework"
        assert d["state"] == "closed"
        assert d["failures"] == 0
        assert d["successes"] == 0
        assert d["cooldown_remaining"] == 0.0
        assert d["half_open_calls"] == 0

    @pytest.mark.asyncio
    async def test_to_dict_when_open(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        d = cb.to_dict()
        assert d["state"] == "open"
        assert d["failures"] == basic_cb_settings.failure_threshold
        assert d["cooldown_remaining"] > 0

    def test_cooldown_remaining_zero_when_closed(self, circuit_breaker):
        assert circuit_breaker.cooldown_remaining == 0.0

    @pytest.mark.asyncio
    async def test_additional_failures_dont_reset_open_at(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold):
            await cb.record_failure()
        oa = cb._open_at
        await cb.record_failure()
        assert cb._open_at == oa

    @pytest.mark.asyncio
    async def test_success_in_closed_state(self, circuit_breaker):
        await circuit_breaker.record_success()
        assert circuit_breaker._open_at is None
        assert circuit_breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_concurrent_access(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)

        async def failures():
            for _ in range(10):
                await cb.record_failure()

        async def successes():
            for _ in range(10):
                await cb.record_success()

        await asyncio.gather(failures(), successes())

    @pytest.mark.asyncio
    async def test_record_failure_below_threshold(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        assert await cb.record_failure() is False
        assert await cb.record_failure() is False

    @pytest.mark.asyncio
    async def test_record_failure_true_only_first_open(self, basic_cb_settings):
        cb = FrameworkCircuitBreaker("test", basic_cb_settings)
        for _ in range(basic_cb_settings.failure_threshold - 1):
            await cb.record_failure()
        assert await cb.record_failure() is True
        assert await cb.record_failure() is False


# ============================================================================
# 4. Bulkhead concurrent request limiting
# ============================================================================


class TestFrameworkBulkhead:
    def test_initial_state(self, bulkhead):
        assert bulkhead.active_count == 0
        assert bulkhead.available_slots == 3

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, bulkhead):
        async with bulkhead.acquire():
            assert bulkhead.active_count == 1
            assert bulkhead.available_slots == 2
        assert bulkhead.active_count == 0
        assert bulkhead.available_slots == 3

    @pytest.mark.asyncio
    async def test_multiple_concurrent(self, bulkhead):
        acquired = []

        async def go():
            async with bulkhead.acquire():
                acquired.append(True)
                assert bulkhead.active_count <= 3
                await asyncio.sleep(0.01)

        await asyncio.gather(go(), go(), go())
        assert len(acquired) == 3
        assert bulkhead.active_count == 0

    @pytest.mark.asyncio
    async def test_full_raises_error(self):
        bh = FrameworkBulkhead("test", BulkheadSettings(max_concurrent=1, wait_timeout=0.1))
        ev_acq, ev_rel = asyncio.Event(), asyncio.Event()

        async def hold():
            async with bh.acquire():
                ev_acq.set()
                await ev_rel.wait()

        task = asyncio.create_task(hold())
        await ev_acq.wait()
        with pytest.raises(BulkheadFullError) as ei:
            async with bh.acquire():
                pass
        assert ei.value.max_concurrent == 1
        assert ei.value.framework == "test"
        assert "wait_timeout" in ei.value.details
        ev_rel.set()
        await task

    @pytest.mark.asyncio
    async def test_release_on_exception(self, bulkhead):
        with pytest.raises(RuntimeError):
            async with bulkhead.acquire():
                raise RuntimeError("err")
        assert bulkhead.active_count == 0

    def test_to_dict(self, bulkhead):
        d = bulkhead.to_dict()
        assert d == {
            "framework": "test-framework",
            "active": 0,
            "max_concurrent": 3,
            "available_slots": 3,
        }

    @pytest.mark.asyncio
    async def test_to_dict_during_acquisition(self, bulkhead):
        async with bulkhead.acquire():
            d = bulkhead.to_dict()
            assert d["active"] == 1
            assert d["available_slots"] == 2

    @pytest.mark.asyncio
    async def test_many_cycles(self):
        bh = FrameworkBulkhead("test", BulkheadSettings(max_concurrent=5, wait_timeout=5.0))

        async def go():
            async with bh.acquire():
                await asyncio.sleep(0.01)

        await asyncio.gather(*[go() for _ in range(10)])
        assert bh.active_count == 0

    @pytest.mark.asyncio
    async def test_nested_acquisition(self, bulkhead):
        async with bulkhead.acquire():
            assert bulkhead.active_count == 1
            async with bulkhead.acquire():
                assert bulkhead.active_count == 2
            assert bulkhead.active_count == 1
        assert bulkhead.active_count == 0


# ============================================================================
# 7. Health check probes
# ============================================================================


class TestHealthCheckResult:
    def test_basic(self):
        r = HealthCheckResult(framework="openai", status=HealthStatus.HEALTHY, latency_ms=50.0)
        assert r.framework == "openai"
        assert r.status == HealthStatus.HEALTHY
        assert r.latency_ms == 50.0
        assert r.error is None
        assert r.consecutive_failures == 0

    def test_unhealthy(self):
        r = HealthCheckResult(
            framework="openai",
            status=HealthStatus.UNHEALTHY,
            error="Connection refused",
            consecutive_failures=5,
        )
        assert r.status == HealthStatus.UNHEALTHY
        assert r.consecutive_failures == 5

    def test_to_dict(self):
        r = HealthCheckResult(
            framework="openai",
            status=HealthStatus.DEGRADED,
            latency_ms=1500.0,
            error="High latency",
            consecutive_failures=2,
        )
        d = r.to_dict()
        assert d["framework"] == "openai"
        assert d["status"] == "degraded"
        assert d["latency_ms"] == 1500.0
        assert d["error"] == "High latency"
        assert d["consecutive_failures"] == 2
        assert "last_check" in d

    def test_unknown(self):
        assert (
            HealthCheckResult(framework="x", status=HealthStatus.UNKNOWN).status
            == HealthStatus.UNKNOWN
        )


# ============================================================================
# 6. Request sanitization
# ============================================================================


class TestRequestSanitizer:
    def test_strips_sensitive_headers(self, sanitizer):
        h = {
            "Content-Type": "application/json",
            "X-Forwarded-For": "1.2.3.4",
            "X-Real-IP": "5.6.7.8",
            "Authorization": "Bearer secret",
        }
        r = sanitizer.sanitize_headers(h)
        assert "X-Forwarded-For" not in r
        assert "X-Real-IP" not in r
        assert r["Content-Type"] == "application/json"
        assert r["Authorization"] == "Bearer secret"

    def test_logging_redacts(self, sanitizer):
        h = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk",
            "X-API-Key": "key",
            "Cookie": "session=abc",
        }
        r = sanitizer.sanitize_headers(h, for_logging=True)
        assert r["Content-Type"] == "application/json"
        assert r["Authorization"] == "[REDACTED]"
        assert r["X-API-Key"] == "[REDACTED]"
        assert r["Cookie"] == "[REDACTED]"

    def test_case_insensitive(self, sanitizer):
        h = {"AUTHORIZATION": "Bearer k", "x-api-key": "s"}
        r = sanitizer.sanitize_headers(h, for_logging=True)
        assert r["AUTHORIZATION"] == "[REDACTED]"
        assert r["x-api-key"] == "[REDACTED]"

    def test_strip_vs_redact(self, sanitizer):
        h = {"Authorization": "Bearer x", "X-Forwarded-For": "10.0.0.1"}
        r = sanitizer.sanitize_headers(h, for_logging=True)
        assert "X-Forwarded-For" not in r
        assert r["Authorization"] == "[REDACTED]"

    def test_body_none(self, sanitizer):
        assert sanitizer.sanitize_body_for_logging(None) == ""

    def test_body_redacts_api_key(self, sanitizer):
        b = b'{"api_key": "sk-123", "model": "gpt-4"}'
        r = sanitizer.sanitize_body_for_logging(b)
        assert "sk-123" not in r
        assert '"[REDACTED]"' in r
        assert "gpt-4" in r

    def test_body_redacts_password(self, sanitizer):
        assert "secret123" not in sanitizer.sanitize_body_for_logging(b'{"password": "secret123"}')

    def test_body_redacts_token(self, sanitizer):
        r = sanitizer.sanitize_body_for_logging(b'{"token": "tok_live_xyz"}')
        assert "tok_live_xyz" not in r
        assert '"[REDACTED]"' in r

    def test_body_redacts_secret(self, sanitizer):
        assert "mysecret" not in sanitizer.sanitize_body_for_logging(b'{"secret": "mysecret"}')

    def test_body_truncates_large(self, sanitizer):
        r = sanitizer.sanitize_body_for_logging(b"x" * 10000)
        assert "truncated" in r
        assert "10000 bytes total" in r

    def test_body_small(self, sanitizer):
        assert sanitizer.sanitize_body_for_logging(b'{"m": "g"}') == '{"m": "g"}'

    def test_body_binary(self, sanitizer):
        assert isinstance(sanitizer.sanitize_body_for_logging(bytes(range(256)) * 10), str)

    def test_validate_clean(self, sanitizer):
        req = ProxyRequest(
            framework="t",
            method="GET",
            url="https://api.example.com/v1/data",
            headers={"Content-Type": "application/json"},
        )
        sanitizer.validate_request(req)

    def test_validate_newline_in_key(self, sanitizer):
        req = ProxyRequest(
            framework="t", method="GET", url="https://x.com", headers={"Evil\nHeader": "v"}
        )
        with pytest.raises(SanitizationError, match="Header injection"):
            sanitizer.validate_request(req)

    def test_validate_newline_in_value(self, sanitizer):
        req = ProxyRequest(
            framework="t", method="GET", url="https://x.com", headers={"H": "v\r\nInjected: true"}
        )
        with pytest.raises(SanitizationError, match="Header injection"):
            sanitizer.validate_request(req)

    def test_validate_cr_in_key(self, sanitizer):
        req = ProxyRequest(
            framework="t", method="GET", url="https://x.com", headers={"Evil\rHeader": "v"}
        )
        with pytest.raises(SanitizationError, match="Header injection"):
            sanitizer.validate_request(req)

    def test_validate_url_script(self, sanitizer):
        req = ProxyRequest(
            framework="t", method="GET", url="https://x.com/<script>alert(1)</script>", headers={}
        )
        with pytest.raises(SanitizationError, match="Suspicious URL"):
            sanitizer.validate_request(req)

    def test_validate_url_javascript(self, sanitizer):
        req = ProxyRequest(framework="t", method="GET", url="javascript:alert(1)", headers={})
        with pytest.raises(SanitizationError, match="Suspicious URL"):
            sanitizer.validate_request(req)

    def test_validate_url_data(self, sanitizer):
        req = ProxyRequest(
            framework="t", method="GET", url="data:text/html,<h1>evil</h1>", headers={}
        )
        with pytest.raises(SanitizationError, match="Suspicious URL"):
            sanitizer.validate_request(req)

    def test_validate_url_file(self, sanitizer):
        req = ProxyRequest(framework="t", method="GET", url="file:///etc/passwd", headers={})
        with pytest.raises(SanitizationError, match="Suspicious URL"):
            sanitizer.validate_request(req)

    def test_custom_body_patterns(self):
        s = SanitizationSettings(redact_body_patterns=[r'"credit_card"\s*:\s*"[^"]*"'])
        san = RequestSanitizer(s)
        assert "4111" not in san.sanitize_body_for_logging(b'{"credit_card": "4111111111111111"}')

    def test_body_at_max_size_boundary(self):
        san = RequestSanitizer(SanitizationSettings(max_body_log_size=100))
        assert "truncated" not in san.sanitize_body_for_logging(b"a" * 100)

    def test_body_one_over_max_size(self):
        san = RequestSanitizer(SanitizationSettings(max_body_log_size=100))
        assert "truncated" in san.sanitize_body_for_logging(b"a" * 101)


# ============================================================================
# 1 (cont). EnterpriseProxy initialization
# ============================================================================


class TestEnterpriseProxyInit:
    def test_default(self):
        p = EnterpriseProxy()
        assert isinstance(p.config, ProxyConfig)
        assert p.list_frameworks() == []

    def test_with_config(self, custom_proxy_config):
        p = EnterpriseProxy(config=custom_proxy_config)
        assert p.config.default_timeout == 60.0

    def test_with_frameworks(self, openai_framework_config, anthropic_framework_config):
        p = EnterpriseProxy(
            frameworks={"openai": openai_framework_config, "anthropic": anthropic_framework_config}
        )
        assert set(p.list_frameworks()) == {"openai", "anthropic"}

    def test_components_initialized(self, openai_framework_config):
        p = EnterpriseProxy(frameworks={"openai": openai_framework_config})
        assert "openai" in p._circuit_breakers
        assert "openai" in p._bulkheads
        assert "openai" in p._sanitizers
        assert p._health_results["openai"].status == HealthStatus.UNKNOWN

    def test_no_session_on_init(self):
        assert EnterpriseProxy()._session is None

    def test_hooks_empty(self):
        p = EnterpriseProxy()
        assert p._pre_request_hooks == []
        assert p._post_request_hooks == []
        assert p._error_hooks == []


# ============================================================================
# Framework management
# ============================================================================


class TestEnterpriseProxyFrameworkManagement:
    def test_register(self):
        p = EnterpriseProxy()
        c = ExternalFrameworkConfig(base_url="https://api.new.com")
        p.register_framework("new-fw", c)
        assert "new-fw" in p.list_frameworks()
        assert p.get_framework_config("new-fw") is c
        assert "new-fw" in p._circuit_breakers
        assert "new-fw" in p._bulkheads

    def test_register_overwrites(self, openai_framework_config):
        p = EnterpriseProxy(frameworks={"openai": openai_framework_config})
        c2 = ExternalFrameworkConfig(base_url="https://api.openai-v2.com", timeout=90.0)
        p.register_framework("openai", c2)
        assert p.get_framework_config("openai") is c2

    def test_unregister(self, openai_framework_config):
        p = EnterpriseProxy(frameworks={"openai": openai_framework_config})
        assert p.unregister_framework("openai") is True
        assert "openai" not in p.list_frameworks()
        assert "openai" not in p._circuit_breakers

    def test_unregister_nonexistent(self):
        assert EnterpriseProxy().unregister_framework("x") is False

    def test_get_missing(self):
        assert EnterpriseProxy().get_framework_config("x") is None

    def test_list_empty(self):
        assert EnterpriseProxy().list_frameworks() == []

    def test_list_multiple(self):
        p = EnterpriseProxy(
            frameworks={
                "a": ExternalFrameworkConfig(base_url="https://a.com"),
                "b": ExternalFrameworkConfig(base_url="https://b.com"),
                "c": ExternalFrameworkConfig(base_url="https://c.com"),
            }
        )
        assert set(p.list_frameworks()) == {"a", "b", "c"}


# ============================================================================
# 8. Pre / post request hooks
# ============================================================================


class TestEnterpriseProxyHooks:
    def test_add_pre(self):
        p = EnterpriseProxy()

        async def h(r):
            return r

        p.add_pre_request_hook(h)
        assert len(p._pre_request_hooks) == 1

    def test_add_post(self):
        p = EnterpriseProxy()

        async def h(r, resp):
            pass

        p.add_post_request_hook(h)
        assert len(p._post_request_hooks) == 1

    def test_add_error(self):
        p = EnterpriseProxy()

        async def h(r, exc):
            pass

        p.add_error_hook(h)
        assert len(p._error_hooks) == 1

    def test_remove_pre(self):
        p = EnterpriseProxy()

        async def h(r):
            return r

        p.add_pre_request_hook(h)
        assert p.remove_pre_request_hook(h) is True
        assert len(p._pre_request_hooks) == 0

    def test_remove_pre_nonexistent(self):
        async def h(r):
            return r

        assert EnterpriseProxy().remove_pre_request_hook(h) is False

    def test_remove_post(self):
        p = EnterpriseProxy()

        async def h(r, resp):
            pass

        p.add_post_request_hook(h)
        assert p.remove_post_request_hook(h) is True

    def test_remove_post_nonexistent(self):
        async def h(r, resp):
            pass

        assert EnterpriseProxy().remove_post_request_hook(h) is False

    def test_remove_error(self):
        p = EnterpriseProxy()

        async def h(r, exc):
            pass

        p.add_error_hook(h)
        assert p.remove_error_hook(h) is True

    def test_remove_error_nonexistent(self):
        async def h(r, exc):
            pass

        assert EnterpriseProxy().remove_error_hook(h) is False

    def test_multiple_hooks(self):
        p = EnterpriseProxy()
        hooks = []
        for _ in range(5):

            async def h(r):
                return r

            hooks.append(h)
            p.add_pre_request_hook(h)
        assert len(p._pre_request_hooks) == 5


class TestEnterpriseProxyHookIntegration:
    @pytest.mark.asyncio
    async def test_pre_hook_modifies_request(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        s = _make_mock_session()
        p._session = s

        async def hook(req):
            req.headers["X-Hook-Added"] = "true"
            return req

        p.add_pre_request_hook(hook)
        await p.request(framework="openai", method="GET", path="/v1/models")
        assert "X-Hook-Added" in s.request.call_args[1]["headers"]

    @pytest.mark.asyncio
    async def test_pre_hook_aborts_on_none(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        async def hook(req):
            return None

        p.add_pre_request_hook(hook)
        with pytest.raises(ProxyError) as ei:
            await p.request(framework="openai", method="GET", path="/v1/models")
        assert ei.value.code == "REQUEST_ABORTED"

    @pytest.mark.asyncio
    async def test_pre_hook_exception_wraps(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        async def hook(req):
            raise ValueError("hook failure")

        p.add_pre_request_hook(hook)
        with pytest.raises(ProxyError) as ei:
            await p.request(framework="openai", method="GET", path="/v1/models")
        assert ei.value.code == "HOOK_ERROR"

    @pytest.mark.asyncio
    async def test_pre_hook_proxy_error_reraise(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )

        async def hook(req):
            raise ProxyError("custom", code="CUSTOM_ABORT")

        p.add_pre_request_hook(hook)
        with pytest.raises(ProxyError) as ei:
            await p.request(framework="openai", method="GET", path="/v1/models")
        assert ei.value.code == "CUSTOM_ABORT"

    @pytest.mark.asyncio
    async def test_post_hook_called(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_mock_session()
        called = []

        async def hook(req, resp):
            called.append((req.framework, resp.status_code))

        p.add_post_request_hook(hook)
        await p.request(framework="openai", method="GET", path="/v1/models")
        assert called == [("openai", 200)]

    @pytest.mark.asyncio
    async def test_post_hook_exception_ignored(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_mock_session()

        async def hook(req, resp):
            raise RuntimeError("post hook err")

        p.add_post_request_hook(hook)
        resp = await p.request(framework="openai", method="GET", path="/v1/models")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_error_hook_called(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_failing_mock_session(ConnectionError)
        called = []

        async def hook(req, exc):
            called.append(type(exc).__name__)

        p.add_error_hook(hook)
        with pytest.raises(ConnectionError):
            await p.request(framework="openai", method="GET", path="/v1/models", skip_retry=True)
        assert called == ["ConnectionError"]

    @pytest.mark.asyncio
    async def test_error_hook_exception_does_not_suppress(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        cb = p._circuit_breakers["openai"]
        for _ in range(5):
            await cb.record_failure()

        async def hook(req, exc):
            raise RuntimeError("hook crash")

        p.add_error_hook(hook)
        with pytest.raises(CircuitOpenError):
            await p.request(framework="openai", method="GET", path="/v1/models")

    @pytest.mark.asyncio
    async def test_multiple_pre_hooks_chained(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_mock_session()

        async def ha(req):
            req.metadata["a"] = True
            return req

        async def hb(req):
            req.metadata["b"] = True
            return req

        p.add_pre_request_hook(ha)
        p.add_pre_request_hook(hb)
        await p.request(framework="openai", method="GET", path="/v1/models")


# ============================================================================
# Lifecycle
# ============================================================================


class TestEnterpriseProxyLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager(self, openai_framework_config):
        p = EnterpriseProxy(frameworks={"openai": openai_framework_config})
        with (
            patch.object(p, "start", new_callable=AsyncMock) as ms,
            patch.object(p, "shutdown", new_callable=AsyncMock) as md,
        ):
            async with p:
                ms.assert_awaited_once()
            md.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_health_task(self):
        p = EnterpriseProxy()

        # Create a real task that blocks until cancelled
        async def _hang():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_hang())
        p._health_check_task = task
        p._session = AsyncMock()
        p._session.close = AsyncMock()
        await p.shutdown()
        assert task.cancelled()
        assert p._session is None

    @pytest.mark.asyncio
    async def test_shutdown_closes_session(self):
        p = EnterpriseProxy()
        s = AsyncMock()
        s.close = AsyncMock()
        p._session = s
        await p.shutdown()
        s.close.assert_awaited_once()
        assert p._session is None

    @pytest.mark.asyncio
    async def test_shutdown_without_session(self):
        await EnterpriseProxy().shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_sets_event(self):
        p = EnterpriseProxy()
        assert not p._shutdown_event.is_set()
        await p.shutdown()
        assert p._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_ensure_session_creates_once(self):
        p = EnterpriseProxy()
        ma = MagicMock()
        ma.TCPConnector.return_value = MagicMock()
        ma.ClientTimeout.return_value = MagicMock()
        ma.ClientSession.return_value = MagicMock()
        with patch.dict("sys.modules", {"aiohttp": ma}):
            s1 = await p._ensure_session()
            s2 = await p._ensure_session()
            assert s1 is s2
            ma.ClientSession.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_session_import_error(self):
        p = EnterpriseProxy()
        orig = __import__

        def fail(name, *a, **kw):
            if name == "aiohttp":
                raise ImportError("no aiohttp")
            return orig(name, *a, **kw)

        with patch("builtins.__import__", side_effect=fail):
            with pytest.raises(RuntimeError, match="aiohttp is required"):
                await p._ensure_session()


# ============================================================================
# 11. Connection pooling
# ============================================================================


class TestEnterpriseProxyConnectionPooling:
    @pytest.mark.asyncio
    async def test_connector_pool_settings(self):
        p = EnterpriseProxy(
            config=ProxyConfig(
                max_connections=200, max_connections_per_host=25, keepalive_timeout=60.0
            )
        )
        ma = MagicMock()
        ma.TCPConnector.return_value = MagicMock()
        ma.ClientTimeout.return_value = MagicMock()
        ma.ClientSession.return_value = MagicMock()
        with patch.dict("sys.modules", {"aiohttp": ma}):
            await p._ensure_session()
            ma.TCPConnector.assert_called_once_with(
                limit=200, limit_per_host=25, keepalive_timeout=60.0, enable_cleanup_closed=True
            )

    @pytest.mark.asyncio
    async def test_session_user_agent(self):
        p = EnterpriseProxy(config=ProxyConfig(user_agent="CustomAgent/2.0"))
        ma = MagicMock()
        mc = MagicMock()
        mt = MagicMock()
        ma.TCPConnector.return_value = mc
        ma.ClientTimeout.return_value = mt
        ma.ClientSession.return_value = MagicMock()
        with patch.dict("sys.modules", {"aiohttp": ma}):
            await p._ensure_session()
            ma.ClientSession.assert_called_once_with(
                connector=mc, timeout=mt, headers={"User-Agent": "CustomAgent/2.0"}
            )

    @pytest.mark.asyncio
    async def test_session_default_timeouts(self):
        p = EnterpriseProxy(config=ProxyConfig(default_timeout=45.0, default_connect_timeout=12.0))
        ma = MagicMock()
        ma.TCPConnector.return_value = MagicMock()
        ma.ClientTimeout.return_value = MagicMock()
        ma.ClientSession.return_value = MagicMock()
        with patch.dict("sys.modules", {"aiohttp": ma}):
            await p._ensure_session()
            ma.ClientTimeout.assert_called_once_with(total=45.0, connect=12.0)


# ============================================================================
# 2 (cont). Request handling pipeline
# ============================================================================


class TestEnterpriseProxyRequest:
    @pytest.mark.asyncio
    async def test_framework_not_configured(self):
        with pytest.raises(FrameworkNotConfiguredError) as ei:
            await EnterpriseProxy().request(framework="x", method="GET", path="/t")
        assert ei.value.framework == "x"

    @pytest.mark.asyncio
    async def test_framework_disabled(self, disabled_framework_config):
        p = EnterpriseProxy(frameworks={"d": disabled_framework_config})
        with pytest.raises(ProxyError) as ei:
            await p.request(framework="d", method="GET", path="/t")
        assert ei.value.code == "FRAMEWORK_DISABLED"

    @pytest.mark.asyncio
    async def test_builds_url(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        s = _make_mock_session()
        p._session = s
        await p.request(framework="openai", method="GET", path="/v1/models")
        assert s.request.call_args[1]["url"] == "https://api.openai.com/v1/models"

    @pytest.mark.asyncio
    async def test_merges_default_headers(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        s = _make_mock_session()
        p._session = s
        await p.request(
            framework="openai", method="GET", path="/v1/models", headers={"X-Custom": "value"}
        )
        assert "X-Custom" in s.request.call_args[1]["headers"]

    @pytest.mark.asyncio
    async def test_json_body_serialized(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        s = _make_mock_session()
        p._session = s
        await p.request(
            framework="openai",
            method="POST",
            path="/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        kw = s.request.call_args[1]
        assert json.loads(kw["data"])["model"] == "gpt-4"
        assert kw["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_raw_data_body(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        s = _make_mock_session()
        p._session = s
        await p.request(framework="openai", method="POST", path="/v1/audio", data=b"raw audio data")
        assert s.request.call_args[1]["data"] == b"raw audio data"

    @pytest.mark.asyncio
    async def test_returns_proxy_response(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_mock_session(status=200, body=b'{"result": "ok"}')
        r = await p.request(framework="openai", method="GET", path="/v1/models")
        assert isinstance(r, ProxyResponse)
        assert r.status_code == 200
        assert r.body == b'{"result": "ok"}'
        assert r.framework == "openai"
        assert r.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_sanitization_validation(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        with pytest.raises(SanitizationError, match="Header injection"):
            await p.request(
                framework="openai", method="GET", path="/v1/models", headers={"Evil\nHeader": "v"}
            )

    @pytest.mark.asyncio
    async def test_suspicious_url_blocked(self):
        c = ExternalFrameworkConfig(base_url="https://api.example.com")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        with pytest.raises(SanitizationError, match="Suspicious URL"):
            await p.request(framework="t", method="GET", path="/<script>alert(1)</script>")

    @pytest.mark.asyncio
    async def test_method_uppercased(self):
        c = ExternalFrameworkConfig(base_url="https://api.test.com")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        s = _make_mock_session()
        p._session = s
        await p.request(framework="t", method="post", path="/test")
        assert s.request.call_args[1]["method"] == "POST"

    @pytest.mark.asyncio
    async def test_json_sets_content_type(self):
        c = ExternalFrameworkConfig(base_url="https://api.test.com")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        s = _make_mock_session()
        p._session = s
        await p.request(framework="t", method="POST", path="/test", json={"k": "v"})
        assert s.request.call_args[1]["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_json_does_not_override_content_type(self):
        c = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            default_headers={"Content-Type": "application/json; charset=utf-8"},
        )
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        s = _make_mock_session()
        p._session = s
        await p.request(framework="t", method="POST", path="/test", json={"k": "v"})
        expected_ct = "application/json; charset=utf-8"
        assert s.request.call_args[1]["headers"]["Content-Type"] == expected_ct

    @pytest.mark.asyncio
    async def test_response_includes_correlation_id(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_mock_session()
        r = await p.request(
            framework="openai", method="GET", path="/v1/models", correlation_id="corr-123"
        )
        assert r.correlation_id == "corr-123"


# ============================================================================
# 3 (cont). Circuit breaker integration
# ============================================================================


class TestEnterpriseProxyCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_open_blocks(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        for _ in range(5):
            await p._circuit_breakers["openai"].record_failure()
        with pytest.raises(CircuitOpenError) as ei:
            await p.request(framework="openai", method="GET", path="/v1/models")
        assert ei.value.cooldown_remaining > 0

    @pytest.mark.asyncio
    async def test_skip_bypasses(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_mock_session()
        for _ in range(5):
            await p._circuit_breakers["openai"].record_failure()
        r = await p.request(
            framework="openai",
            method="GET",
            path="/v1/models",
            skip_circuit_breaker=True,
            skip_retry=True,
        )
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_success_records(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_mock_session(status=200)
        await p.request(framework="openai", method="GET", path="/v1/models")
        assert p._circuit_breakers["openai"]._failures == 0

    @pytest.mark.asyncio
    async def test_reset(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        cb = p._circuit_breakers["openai"]
        for _ in range(5):
            await cb.record_failure()
        assert await p.reset_circuit_breaker("openai") is True
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_reset_nonexistent(self):
        assert await EnterpriseProxy().reset_circuit_breaker("x") is False


# ============================================================================
# 9. Timeout handling
# ============================================================================


class TestEnterpriseProxyTimeout:
    @pytest.mark.asyncio
    async def test_timeout_raises(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_failing_mock_session(asyncio.TimeoutError)
        with pytest.raises(RequestTimeoutError) as ei:
            await p.request(
                framework="openai", method="GET", path="/v1/models", timeout=5.0, skip_retry=True
            )
        assert ei.value.framework == "openai"
        assert ei.value.timeout == 5.0

    @pytest.mark.asyncio
    async def test_timeout_uses_framework_default(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_failing_mock_session(asyncio.TimeoutError)
        with pytest.raises(RequestTimeoutError) as ei:
            await p.request(framework="openai", method="GET", path="/v1/models", skip_retry=True)
        assert ei.value.timeout == openai_framework_config.timeout


# ============================================================================
# 5. Retry with backoff strategies
# ============================================================================


class TestEnterpriseProxyRetry:
    def test_exponential(self):
        p = EnterpriseProxy()
        s = RetrySettings(
            base_delay=1.0, max_delay=30.0, strategy=RetryStrategy.EXPONENTIAL, jitter=False
        )
        assert p._calculate_retry_delay(0, s) == 1.0
        assert p._calculate_retry_delay(1, s) == 2.0
        assert p._calculate_retry_delay(2, s) == 4.0
        assert p._calculate_retry_delay(3, s) == 8.0

    def test_linear(self):
        p = EnterpriseProxy()
        s = RetrySettings(
            base_delay=1.0, max_delay=30.0, strategy=RetryStrategy.LINEAR, jitter=False
        )
        assert p._calculate_retry_delay(0, s) == 1.0
        assert p._calculate_retry_delay(1, s) == 2.0
        assert p._calculate_retry_delay(2, s) == 3.0

    def test_constant(self):
        p = EnterpriseProxy()
        s = RetrySettings(
            base_delay=2.0, max_delay=30.0, strategy=RetryStrategy.CONSTANT, jitter=False
        )
        assert p._calculate_retry_delay(0, s) == 2.0
        assert p._calculate_retry_delay(5, s) == 2.0

    def test_capped_at_max(self):
        p = EnterpriseProxy()
        s = RetrySettings(
            base_delay=1.0, max_delay=5.0, strategy=RetryStrategy.EXPONENTIAL, jitter=False
        )
        assert p._calculate_retry_delay(10, s) == 5.0

    def test_jitter(self):
        p = EnterpriseProxy()
        s = RetrySettings(
            base_delay=1.0, max_delay=30.0, strategy=RetryStrategy.CONSTANT, jitter=True
        )
        delays = [p._calculate_retry_delay(0, s) for _ in range(50)]
        assert len(set(delays)) > 1
        for d in delays:
            assert 0.5 <= d <= 1.5

    def test_always_non_negative(self):
        p = EnterpriseProxy()
        s = RetrySettings(
            base_delay=0.001, max_delay=30.0, strategy=RetryStrategy.EXPONENTIAL, jitter=True
        )
        for i in range(20):
            assert p._calculate_retry_delay(i, s) >= 0

    @pytest.mark.asyncio
    async def test_retries_on_503(self):
        c = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=2, base_delay=0.01, jitter=False),
        )
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        n = 0

        @asynccontextmanager
        async def mr(**kw):
            nonlocal n
            n += 1
            r = AsyncMock()
            r.status = 503 if n < 3 else 200
            r.headers = {}
            r.read = AsyncMock(return_value=b'{"ok":true}')
            yield r

        s = AsyncMock()
        s.request = MagicMock(side_effect=mr)
        p._session = s
        resp = await p.request(framework="t", method="GET", path="/test")
        assert resp.status_code == 200
        assert n == 3

    @pytest.mark.asyncio
    async def test_exhausted_returns_last(self):
        c = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=1, base_delay=0.01, jitter=False),
        )
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})

        @asynccontextmanager
        async def mr(**kw):
            r = AsyncMock()
            r.status = 503
            r.headers = {}
            r.read = AsyncMock(return_value=b"err")
            yield r

        s = AsyncMock()
        s.request = MagicMock(side_effect=mr)
        p._session = s
        assert (await p.request(framework="t", method="GET", path="/t")).status_code == 503

    @pytest.mark.asyncio
    async def test_skip_retry(self):
        c = ExternalFrameworkConfig(
            base_url="https://api.test.com", retry=RetrySettings(max_retries=3)
        )
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        n = 0

        @asynccontextmanager
        async def mr(**kw):
            nonlocal n
            n += 1
            r = AsyncMock()
            r.status = 503
            r.headers = {}
            r.read = AsyncMock(return_value=b"err")
            yield r

        s = AsyncMock()
        s.request = MagicMock(side_effect=mr)
        p._session = s
        resp = await p.request(framework="t", method="GET", path="/t", skip_retry=True)
        assert n == 1 and resp.status_code == 503

    @pytest.mark.asyncio
    async def test_retries_connection_error(self):
        c = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=2, base_delay=0.01, jitter=False),
        )
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        n = 0

        @asynccontextmanager
        async def mr(**kw):
            nonlocal n
            n += 1
            if n < 3:
                raise ConnectionError("refused")
            r = AsyncMock()
            r.status = 200
            r.headers = {}
            r.read = AsyncMock(return_value=b'{"ok":true}')
            yield r

        s = AsyncMock()
        s.request = MagicMock(side_effect=mr)
        p._session = s
        assert (await p.request(framework="t", method="GET", path="/t")).status_code == 200
        assert n == 3

    @pytest.mark.asyncio
    async def test_retries_os_error(self):
        c = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=1, base_delay=0.01, jitter=False),
        )
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        n = 0

        @asynccontextmanager
        async def mr(**kw):
            nonlocal n
            n += 1
            if n == 1:
                raise OSError("unreachable")
            r = AsyncMock()
            r.status = 200
            r.headers = {}
            r.read = AsyncMock(return_value=b"ok")
            yield r

        s = AsyncMock()
        s.request = MagicMock(side_effect=mr)
        p._session = s
        assert (await p.request(framework="t", method="GET", path="/t")).status_code == 200

    @pytest.mark.asyncio
    async def test_retry_records_cb_failures(self):
        c = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=1, base_delay=0.01, jitter=False),
        )
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})

        @asynccontextmanager
        async def mr(**kw):
            raise ConnectionError("refused")
            yield  # noqa

        s = AsyncMock()
        s.request = MagicMock(side_effect=mr)
        p._session = s
        with pytest.raises(ConnectionError):
            await p.request(framework="t", method="GET", path="/t")
        assert p._circuit_breakers["t"]._failures >= 1

    @pytest.mark.asyncio
    async def test_retry_on_429(self):
        c = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=1, base_delay=0.01, jitter=False),
        )
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        n = 0

        @asynccontextmanager
        async def mr(**kw):
            nonlocal n
            n += 1
            r = AsyncMock()
            r.status = 429 if n == 1 else 200
            r.headers = {}
            r.read = AsyncMock(return_value=b"ok")
            yield r

        s = AsyncMock()
        s.request = MagicMock(side_effect=mr)
        p._session = s
        assert (await p.request(framework="t", method="GET", path="/t")).status_code == 200
        assert n == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        c = ExternalFrameworkConfig(
            base_url="https://api.test.com",
            retry=RetrySettings(max_retries=3, base_delay=0.01, jitter=False),
        )
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        n = 0

        @asynccontextmanager
        async def mr(**kw):
            nonlocal n
            n += 1
            r = AsyncMock()
            r.status = 400
            r.headers = {}
            r.read = AsyncMock(return_value=b"bad")
            yield r

        s = AsyncMock()
        s.request = MagicMock(side_effect=mr)
        p._session = s
        assert (await p.request(framework="t", method="GET", path="/t")).status_code == 400
        assert n == 1


# ============================================================================
# 7 (cont). Health check integration
# ============================================================================


class TestEnterpriseProxyHealthCheck:
    @pytest.mark.asyncio
    async def test_unconfigured(self):
        r = await EnterpriseProxy().check_health("x")
        assert r.status == HealthStatus.UNKNOWN
        assert r.error == "Framework not configured"

    @pytest.mark.asyncio
    async def test_no_health_path(self):
        c = ExternalFrameworkConfig(base_url="https://x.com", health_check_path=None)
        r = await EnterpriseProxy(frameworks={"t": c}).check_health("t")
        assert r.status == HealthStatus.UNKNOWN
        assert "No health check path" in r.error

    @pytest.mark.asyncio
    async def test_healthy(self):
        c = ExternalFrameworkConfig(base_url="https://x.com", health_check_path="/health")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        p._session = _make_mock_session(status=200)
        r = await p.check_health("t")
        assert r.status == HealthStatus.HEALTHY
        assert r.latency_ms is not None
        assert r.error is None
        assert r.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_degraded(self):
        c = ExternalFrameworkConfig(base_url="https://x.com", health_check_path="/health")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        p._session = _make_mock_session(status=503)
        r = await p.check_health("t")
        assert r.status == HealthStatus.DEGRADED
        assert "503" in r.error

    @pytest.mark.asyncio
    async def test_unhealthy(self):
        c = ExternalFrameworkConfig(base_url="https://x.com", health_check_path="/health")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        p._session = _make_failing_mock_session(OSError)
        r = await p.check_health("t")
        assert r.status == HealthStatus.UNHEALTHY
        assert r.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_consecutive_failures(self):
        c = ExternalFrameworkConfig(base_url="https://x.com", health_check_path="/health")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        p._session = _make_failing_mock_session(OSError)
        assert (await p.check_health("t")).consecutive_failures == 1
        assert (await p.check_health("t")).consecutive_failures == 2

    @pytest.mark.asyncio
    async def test_stores_results(self):
        c = ExternalFrameworkConfig(base_url="https://x.com", health_check_path="/health")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        p._session = _make_mock_session(status=200)
        await p.check_health("t")
        assert p._health_results["t"].status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_all(self):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={
                "fw1": ExternalFrameworkConfig(base_url="https://fw1.com", health_check_path="/h"),
                "fw2": ExternalFrameworkConfig(base_url="https://fw2.com", health_check_path=None),
            },
        )
        p._session = _make_mock_session(status=200)
        results = await p.check_all_health()
        assert results["fw1"].status == HealthStatus.HEALTHY
        assert results["fw2"].status == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_start_creates_health_task(self):
        c = ExternalFrameworkConfig(base_url="https://x.com", health_check_path="/health")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        with (
            patch.object(p, "_ensure_session", new_callable=AsyncMock),
            patch.object(p, "_health_check_loop", new_callable=AsyncMock),
        ):
            await p.start()
            assert p._health_check_task is not None
        await p.shutdown()

    @pytest.mark.asyncio
    async def test_start_no_health_task_without_paths(self):
        c = ExternalFrameworkConfig(base_url="https://x.com", health_check_path=None)
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        with patch.object(p, "_ensure_session", new_callable=AsyncMock):
            await p.start()
            assert p._health_check_task is None


# ============================================================================
# Monitoring and statistics
# ============================================================================


class TestEnterpriseProxyMonitoring:
    def test_cb_status_all(self, proxy_with_frameworks):
        s = proxy_with_frameworks.get_circuit_breaker_status()
        assert "openai" in s and "anthropic" in s
        assert s["openai"]["state"] == "closed"

    def test_cb_status_single(self, proxy_with_frameworks):
        s = proxy_with_frameworks.get_circuit_breaker_status("openai")
        assert s["framework"] == "openai"

    def test_cb_status_missing(self, proxy_with_frameworks):
        assert proxy_with_frameworks.get_circuit_breaker_status("x") == {}

    def test_bh_status_all(self, proxy_with_frameworks):
        s = proxy_with_frameworks.get_bulkhead_status()
        assert "openai" in s and "anthropic" in s

    def test_bh_status_single(self, proxy_with_frameworks):
        s = proxy_with_frameworks.get_bulkhead_status("openai")
        assert "active" in s and "max_concurrent" in s

    def test_bh_status_missing(self, proxy_with_frameworks):
        assert proxy_with_frameworks.get_bulkhead_status("x") == {}

    def test_health_all(self, proxy_with_frameworks):
        s = proxy_with_frameworks.get_health_status()
        assert "openai" in s and "anthropic" in s

    def test_health_single(self, proxy_with_frameworks):
        s = proxy_with_frameworks.get_health_status("openai")
        assert s["framework"] == "openai" and s["status"] == "unknown"

    def test_health_missing(self, proxy_with_frameworks):
        assert proxy_with_frameworks.get_health_status("x") == {}

    def test_stats(self, proxy_with_frameworks):
        s = proxy_with_frameworks.get_stats()
        assert all(
            k in s
            for k in ("config", "frameworks", "circuit_breakers", "bulkheads", "health", "hooks")
        )
        assert s["config"]["max_connections"] == 100
        assert s["frameworks"]["openai"]["enabled"] is True
        assert s["hooks"]["pre_request"] == 0

    def test_stats_with_hooks(self, proxy_with_frameworks):
        async def h1(r):
            return r

        async def h2(r, resp):
            pass

        async def h3(r, exc):
            pass

        proxy_with_frameworks.add_pre_request_hook(h1)
        proxy_with_frameworks.add_post_request_hook(h2)
        proxy_with_frameworks.add_error_hook(h3)
        s = proxy_with_frameworks.get_stats()
        assert s["hooks"] == {"pre_request": 1, "post_request": 1, "error": 1}

    def test_stats_framework_details(self, proxy_with_frameworks):
        s = proxy_with_frameworks.get_stats()
        assert s["frameworks"]["openai"]["base_url"] == "https://api.openai.com"
        assert s["frameworks"]["openai"]["timeout"] == 60.0


# ============================================================================
# Audit logging
# ============================================================================


class TestEnterpriseProxyAuditLogging:
    @pytest.mark.asyncio
    async def test_enabled(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=True),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_mock_session()
        with patch.object(p, "_log_request") as ml:
            await p.request(framework="openai", method="GET", path="/v1/models", skip_retry=True)
            ml.assert_called_once()

    @pytest.mark.asyncio
    async def test_disabled(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        p._session = _make_mock_session()
        with patch.object(p, "_log_request") as ml:
            await p.request(framework="openai", method="GET", path="/v1/models", skip_retry=True)
            ml.assert_not_called()


# ============================================================================
# 12. Tenant context injection
# ============================================================================


class TestEnterpriseProxyTenantContext:
    @pytest.mark.asyncio
    async def test_tenant_id_injected(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        s = _make_mock_session()
        p._session = s
        await p.request(framework="openai", method="GET", path="/v1/models", tenant_id="t-abc")
        assert s.request.call_args[1]["headers"]["X-Tenant-ID"] == "t-abc"

    @pytest.mark.asyncio
    async def test_custom_tenant_header(self):
        c = ExternalFrameworkConfig(base_url="https://api.test.com")
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False, tenant_header_name="X-Custom-Tenant"),
            frameworks={"t": c},
        )
        s = _make_mock_session()
        p._session = s
        await p.request(framework="t", method="GET", path="/t", tenant_id="tenant-456")
        assert s.request.call_args[1]["headers"]["X-Custom-Tenant"] == "tenant-456"

    @pytest.mark.asyncio
    async def test_correlation_id_injected(self, openai_framework_config):
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False),
            frameworks={"openai": openai_framework_config},
        )
        s = _make_mock_session()
        p._session = s
        await p.request(
            framework="openai", method="GET", path="/v1/models", correlation_id="corr-xyz"
        )
        assert s.request.call_args[1]["headers"]["X-Correlation-ID"] == "corr-xyz"

    @pytest.mark.asyncio
    async def test_custom_correlation_header(self):
        c = ExternalFrameworkConfig(base_url="https://api.test.com")
        p = EnterpriseProxy(
            config=ProxyConfig(enable_audit_logging=False, correlation_header_name="X-Request-ID"),
            frameworks={"t": c},
        )
        s = _make_mock_session()
        p._session = s
        await p.request(framework="t", method="GET", path="/t", correlation_id="req-789")
        assert s.request.call_args[1]["headers"]["X-Request-ID"] == "req-789"

    @pytest.mark.asyncio
    async def test_no_tenant_when_not_provided(self):
        c = ExternalFrameworkConfig(base_url="https://api.test.com")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        s = _make_mock_session()
        p._session = s
        await p.request(framework="t", method="GET", path="/t")
        assert "X-Tenant-ID" not in s.request.call_args[1]["headers"]

    @pytest.mark.asyncio
    async def test_no_correlation_when_not_provided(self):
        c = ExternalFrameworkConfig(base_url="https://api.test.com")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        s = _make_mock_session()
        p._session = s
        await p.request(framework="t", method="GET", path="/t")
        assert "X-Correlation-ID" not in s.request.call_args[1]["headers"]

    @pytest.mark.asyncio
    async def test_both_injected(self):
        c = ExternalFrameworkConfig(base_url="https://api.test.com")
        p = EnterpriseProxy(config=ProxyConfig(enable_audit_logging=False), frameworks={"t": c})
        s = _make_mock_session()
        p._session = s
        await p.request(
            framework="t", method="GET", path="/t", tenant_id="t-1", correlation_id="c-1"
        )
        h = s.request.call_args[1]["headers"]
        assert h["X-Tenant-ID"] == "t-1"
        assert h["X-Correlation-ID"] == "c-1"
