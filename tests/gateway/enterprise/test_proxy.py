"""Tests for aragora.gateway.enterprise.proxy module.

Covers:
- Configuration dataclass validation
- Circuit breaker state transitions
- Bulkhead slot management
- Request sanitizer (header redaction, body sanitization, validation)
- EnterpriseProxy framework management, hooks, monitoring
- ProxyRequest/ProxyResponse data structures
"""

from __future__ import annotations

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Test exception classes."""

    def test_proxy_error_basic(self):
        err = ProxyError("Something failed")
        assert str(err) == "Something failed"
        assert err.code == "PROXY_ERROR"
        assert err.framework is None
        assert err.details == {}

    def test_proxy_error_with_details(self):
        err = ProxyError("Fail", code="CUSTOM", framework="openai", details={"key": "val"})
        assert err.code == "CUSTOM"
        assert err.framework == "openai"
        assert err.details == {"key": "val"}

    def test_circuit_open_error(self):
        err = CircuitOpenError("openai", 15.3)
        assert "openai" in str(err)
        assert err.code == "CIRCUIT_OPEN"
        assert err.framework == "openai"
        assert err.cooldown_remaining == 15.3

    def test_bulkhead_full_error(self):
        err = BulkheadFullError("anthropic", 50)
        assert "anthropic" in str(err)
        assert err.code == "BULKHEAD_FULL"
        assert err.max_concurrent == 50

    def test_request_timeout_error(self):
        err = RequestTimeoutError("gemini", 30.0)
        assert "gemini" in str(err)
        assert err.code == "REQUEST_TIMEOUT"
        assert err.timeout == 30.0

    def test_framework_not_configured_error(self):
        err = FrameworkNotConfiguredError("unknown")
        assert "unknown" in str(err)
        assert err.code == "FRAMEWORK_NOT_CONFIGURED"

    def test_sanitization_error(self):
        err = SanitizationError("Bad header", framework="openai", details={"header": "x"})
        assert str(err) == "Bad header"
        assert err.code == "SANITIZATION_ERROR"


# =============================================================================
# Configuration Tests
# =============================================================================


class TestCircuitBreakerSettings:
    """Test CircuitBreakerSettings validation."""

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
            cooldown_seconds=30.0,
            half_open_max_calls=1,
        )
        assert settings.failure_threshold == 10

    def test_invalid_failure_threshold(self):
        with pytest.raises(ValueError, match="failure_threshold"):
            CircuitBreakerSettings(failure_threshold=0)

    def test_invalid_success_threshold(self):
        with pytest.raises(ValueError, match="success_threshold"):
            CircuitBreakerSettings(success_threshold=0)

    def test_invalid_cooldown(self):
        with pytest.raises(ValueError, match="cooldown_seconds"):
            CircuitBreakerSettings(cooldown_seconds=-1)

    def test_invalid_half_open_max_calls(self):
        with pytest.raises(ValueError, match="half_open_max_calls"):
            CircuitBreakerSettings(half_open_max_calls=0)


class TestRetrySettings:
    """Test RetrySettings validation."""

    def test_defaults(self):
        settings = RetrySettings()
        assert settings.max_retries == 3
        assert settings.strategy == RetryStrategy.EXPONENTIAL
        assert settings.jitter is True
        assert 429 in settings.retryable_status_codes
        assert 503 in settings.retryable_status_codes

    def test_negative_retries(self):
        with pytest.raises(ValueError, match="max_retries"):
            RetrySettings(max_retries=-1)

    def test_invalid_base_delay(self):
        with pytest.raises(ValueError, match="base_delay"):
            RetrySettings(base_delay=0)

    def test_max_delay_less_than_base(self):
        with pytest.raises(ValueError, match="max_delay"):
            RetrySettings(base_delay=10.0, max_delay=5.0)


class TestBulkheadSettings:
    """Test BulkheadSettings validation."""

    def test_defaults(self):
        settings = BulkheadSettings()
        assert settings.max_concurrent == 50
        assert settings.wait_timeout == 10.0

    def test_invalid_max_concurrent(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            BulkheadSettings(max_concurrent=0)

    def test_invalid_wait_timeout(self):
        with pytest.raises(ValueError, match="wait_timeout"):
            BulkheadSettings(wait_timeout=0)


class TestExternalFrameworkConfig:
    """Test ExternalFrameworkConfig validation."""

    def test_basic_config(self):
        config = ExternalFrameworkConfig(base_url="https://api.example.com")
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 30.0
        assert config.enabled is True

    def test_trailing_slash_removed(self):
        config = ExternalFrameworkConfig(base_url="https://api.example.com/")
        assert config.base_url == "https://api.example.com"

    def test_empty_base_url(self):
        with pytest.raises(ValueError, match="base_url"):
            ExternalFrameworkConfig(base_url="")

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout"):
            ExternalFrameworkConfig(base_url="https://api.example.com", timeout=-1)

    def test_invalid_connect_timeout(self):
        with pytest.raises(ValueError, match="connect_timeout"):
            ExternalFrameworkConfig(base_url="https://api.example.com", connect_timeout=0)


class TestProxyConfig:
    """Test ProxyConfig validation."""

    def test_defaults(self):
        config = ProxyConfig()
        assert config.max_connections == 100
        assert config.tenant_header_name == "X-Tenant-ID"
        assert config.user_agent == "Aragora-EnterpriseProxy/1.0"

    def test_invalid_max_connections(self):
        with pytest.raises(ValueError, match="max_connections"):
            ProxyConfig(max_connections=0)

    def test_invalid_max_per_host(self):
        with pytest.raises(ValueError, match="max_connections_per_host"):
            ProxyConfig(max_connections_per_host=0)


# =============================================================================
# ProxyRequest / ProxyResponse Tests
# =============================================================================


class TestProxyRequest:
    """Test ProxyRequest dataclass."""

    def test_basic_request(self):
        req = ProxyRequest(
            framework="openai",
            method="POST",
            url="https://api.openai.com/v1/chat",
            headers={"Authorization": "Bearer sk-xxx"},
        )
        assert req.framework == "openai"
        assert req.method == "POST"
        assert req.body is None
        assert req.tenant_id is None

    def test_body_hash_none(self):
        req = ProxyRequest(framework="x", method="GET", url="http://x", headers={})
        assert req.body_hash() is None

    def test_body_hash_computed(self):
        req = ProxyRequest(
            framework="x",
            method="POST",
            url="http://x",
            headers={},
            body=b'{"test": true}',
        )
        h = req.body_hash()
        assert h is not None
        assert len(h) == 64  # SHA-256 hex


class TestProxyResponse:
    """Test ProxyResponse dataclass."""

    def test_success_response(self):
        resp = ProxyResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=b'{"ok":true}',
            elapsed_ms=150.0,
            framework="openai",
        )
        assert resp.is_success is True
        assert resp.is_client_error is False
        assert resp.is_server_error is False

    def test_client_error(self):
        resp = ProxyResponse(
            status_code=404,
            headers={},
            body=b"Not found",
            elapsed_ms=50.0,
            framework="openai",
        )
        assert resp.is_success is False
        assert resp.is_client_error is True
        assert resp.is_server_error is False

    def test_server_error(self):
        resp = ProxyResponse(
            status_code=503,
            headers={},
            body=b"Service unavailable",
            elapsed_ms=100.0,
            framework="openai",
        )
        assert resp.is_success is False
        assert resp.is_client_error is False
        assert resp.is_server_error is True

    def test_body_hash(self):
        resp = ProxyResponse(
            status_code=200,
            headers={},
            body=b"test",
            elapsed_ms=10.0,
            framework="x",
        )
        assert len(resp.body_hash()) == 64


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestFrameworkCircuitBreaker:
    """Test FrameworkCircuitBreaker state machine."""

    @pytest.fixture
    def cb(self):
        return FrameworkCircuitBreaker(
            "test-framework",
            CircuitBreakerSettings(
                failure_threshold=3,
                success_threshold=2,
                cooldown_seconds=10.0,
                half_open_max_calls=2,
            ),
        )

    def test_initial_state_closed(self, cb):
        assert cb.state == "closed"
        assert cb.is_open is False
        assert cb.cooldown_remaining == 0.0

    @pytest.mark.asyncio
    async def test_can_proceed_when_closed(self, cb):
        assert await cb.can_proceed() is True

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, cb):
        for i in range(2):
            opened = await cb.record_failure()
            assert opened is False

        opened = await cb.record_failure()
        assert opened is True
        assert cb.state == "open"
        assert cb.is_open is True

    @pytest.mark.asyncio
    async def test_blocks_when_open(self, cb):
        for _ in range(3):
            await cb.record_failure()

        assert await cb.can_proceed() is False

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_cooldown(self, cb):
        for _ in range(3):
            await cb.record_failure()

        # Simulate cooldown elapsed
        cb._open_at = time.time() - 15.0
        assert cb.state == "half-open"
        assert cb.is_open is False

    @pytest.mark.asyncio
    async def test_half_open_allows_limited_calls(self, cb):
        for _ in range(3):
            await cb.record_failure()

        cb._open_at = time.time() - 15.0  # past cooldown

        assert await cb.can_proceed() is True
        assert await cb.can_proceed() is True
        assert await cb.can_proceed() is False  # exceeds half_open_max_calls

    @pytest.mark.asyncio
    async def test_closes_after_success_in_half_open(self, cb):
        for _ in range(3):
            await cb.record_failure()

        cb._open_at = time.time() - 15.0

        await cb.record_success()
        await cb.record_success()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, cb):
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_success()
        assert cb._failures == 0

    @pytest.mark.asyncio
    async def test_reset(self, cb):
        for _ in range(3):
            await cb.record_failure()

        assert cb.state == "open"
        await cb.reset()
        assert cb.state == "closed"
        assert cb._failures == 0

    def test_to_dict(self, cb):
        d = cb.to_dict()
        assert d["framework"] == "test-framework"
        assert d["state"] == "closed"
        assert "failures" in d
        assert "cooldown_remaining" in d


# =============================================================================
# Bulkhead Tests
# =============================================================================


class TestFrameworkBulkhead:
    """Test FrameworkBulkhead slot management."""

    @pytest.fixture
    def bulkhead(self):
        return FrameworkBulkhead(
            "test-framework",
            BulkheadSettings(max_concurrent=2, wait_timeout=0.5),
        )

    def test_initial_state(self, bulkhead):
        assert bulkhead.active_count == 0
        assert bulkhead.available_slots == 2

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, bulkhead):
        async with bulkhead.acquire():
            assert bulkhead.active_count == 1
            assert bulkhead.available_slots == 1
        assert bulkhead.active_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self, bulkhead):
        async with bulkhead.acquire():
            async with bulkhead.acquire():
                assert bulkhead.active_count == 2
                assert bulkhead.available_slots == 0

    @pytest.mark.asyncio
    async def test_bulkhead_full_raises(self, bulkhead):
        async with bulkhead.acquire():
            async with bulkhead.acquire():
                with pytest.raises(BulkheadFullError):
                    async with bulkhead.acquire():
                        pass

    @pytest.mark.asyncio
    async def test_release_on_exception(self, bulkhead):
        try:
            async with bulkhead.acquire():
                raise RuntimeError("Test error")
        except RuntimeError:
            pass
        assert bulkhead.active_count == 0

    def test_to_dict(self, bulkhead):
        d = bulkhead.to_dict()
        assert d["framework"] == "test-framework"
        assert d["max_concurrent"] == 2
        assert d["active"] == 0


# =============================================================================
# Request Sanitizer Tests
# =============================================================================


class TestRequestSanitizer:
    """Test RequestSanitizer security features."""

    @pytest.fixture
    def sanitizer(self):
        return RequestSanitizer(SanitizationSettings())

    def test_sanitize_headers_strips_sensitive(self, sanitizer):
        headers = {
            "Content-Type": "application/json",
            "X-Forwarded-For": "1.2.3.4",
            "Accept": "application/json",
        }
        result = sanitizer.sanitize_headers(headers)
        assert "Content-Type" in result
        assert "Accept" in result
        assert "X-Forwarded-For" not in result

    def test_sanitize_headers_for_logging_redacts(self, sanitizer):
        headers = {
            "Authorization": "Bearer sk-xxx",
            "Content-Type": "application/json",
        }
        result = sanitizer.sanitize_headers(headers, for_logging=True)
        assert result["Authorization"] == "[REDACTED]"
        assert result["Content-Type"] == "application/json"

    def test_sanitize_body_none(self, sanitizer):
        assert sanitizer.sanitize_body_for_logging(None) == ""

    def test_sanitize_body_truncates_large(self, sanitizer):
        body = b"x" * 10000
        result = sanitizer.sanitize_body_for_logging(body)
        assert "truncated" in result

    def test_sanitize_body_redacts_patterns(self, sanitizer):
        body = b'{"api_key": "sk-secret-123", "message": "hello"}'
        result = sanitizer.sanitize_body_for_logging(body)
        assert "sk-secret-123" not in result
        assert "[REDACTED]" in result

    def test_validate_request_header_injection(self, sanitizer):
        req = ProxyRequest(
            framework="x",
            method="GET",
            url="http://x",
            headers={"Evil\r\nHeader": "value"},
        )
        with pytest.raises(SanitizationError, match="injection"):
            sanitizer.validate_request(req)

    def test_validate_request_url_injection(self, sanitizer):
        req = ProxyRequest(
            framework="x",
            method="GET",
            url="javascript:alert(1)",
            headers={},
        )
        with pytest.raises(SanitizationError, match="Suspicious URL"):
            sanitizer.validate_request(req)

    def test_validate_request_clean(self, sanitizer):
        req = ProxyRequest(
            framework="x",
            method="GET",
            url="https://api.example.com/v1/test",
            headers={"Accept": "application/json"},
        )
        sanitizer.validate_request(req)  # Should not raise


# =============================================================================
# HealthCheckResult Tests
# =============================================================================


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_to_dict(self):
        result = HealthCheckResult(
            framework="openai",
            status=HealthStatus.HEALTHY,
            latency_ms=42.5,
        )
        d = result.to_dict()
        assert d["framework"] == "openai"
        assert d["status"] == "healthy"
        assert d["latency_ms"] == 42.5


# =============================================================================
# EnterpriseProxy Tests
# =============================================================================


class TestEnterpriseProxy:
    """Test EnterpriseProxy core functionality."""

    @pytest.fixture
    def proxy_config(self):
        return ProxyConfig(
            default_timeout=30.0,
            max_connections=10,
            enable_audit_logging=False,
        )

    @pytest.fixture
    def framework_config(self):
        return ExternalFrameworkConfig(
            base_url="https://api.example.com",
            timeout=10.0,
            circuit_breaker=CircuitBreakerSettings(failure_threshold=3),
            retry=RetrySettings(max_retries=2),
            bulkhead=BulkheadSettings(max_concurrent=5),
        )

    @pytest.fixture
    def proxy(self, proxy_config, framework_config):
        return EnterpriseProxy(
            config=proxy_config,
            frameworks={"test-fw": framework_config},
        )

    class TestFrameworkManagement:
        """Test framework registration and management."""

        @pytest.fixture
        def proxy(self):
            return EnterpriseProxy()

        def test_register_framework(self, proxy):
            config = ExternalFrameworkConfig(base_url="https://api.new.com")
            proxy.register_framework("new-fw", config)
            assert "new-fw" in proxy.list_frameworks()
            assert proxy.get_framework_config("new-fw") is config

        def test_unregister_framework(self, proxy):
            config = ExternalFrameworkConfig(base_url="https://api.test.com")
            proxy.register_framework("temp", config)
            assert proxy.unregister_framework("temp") is True
            assert "temp" not in proxy.list_frameworks()

        def test_unregister_missing_framework(self, proxy):
            assert proxy.unregister_framework("nonexistent") is False

        def test_get_missing_framework(self, proxy):
            assert proxy.get_framework_config("nonexistent") is None

        def test_list_frameworks(self, proxy):
            proxy.register_framework("a", ExternalFrameworkConfig(base_url="https://a.com"))
            proxy.register_framework("b", ExternalFrameworkConfig(base_url="https://b.com"))
            names = proxy.list_frameworks()
            assert "a" in names
            assert "b" in names

    class TestHookManagement:
        """Test hook add/remove functionality."""

        @pytest.fixture
        def proxy(self):
            return EnterpriseProxy()

        def test_add_and_remove_pre_request_hook(self, proxy):
            async def hook(req):
                return req

            proxy.add_pre_request_hook(hook)
            assert len(proxy._pre_request_hooks) == 1
            assert proxy.remove_pre_request_hook(hook) is True
            assert len(proxy._pre_request_hooks) == 0

        def test_add_and_remove_post_request_hook(self, proxy):
            async def hook(req, resp):
                pass

            proxy.add_post_request_hook(hook)
            assert len(proxy._post_request_hooks) == 1
            assert proxy.remove_post_request_hook(hook) is True

        def test_add_and_remove_error_hook(self, proxy):
            async def hook(req, exc):
                pass

            proxy.add_error_hook(hook)
            assert len(proxy._error_hooks) == 1
            assert proxy.remove_error_hook(hook) is True

        def test_remove_nonexistent_hook(self, proxy):
            async def hook(req):
                return req

            assert proxy.remove_pre_request_hook(hook) is False
            assert proxy.remove_post_request_hook(hook) is False
            assert proxy.remove_error_hook(hook) is False

    class TestMonitoring:
        """Test monitoring and statistics methods."""

        @pytest.fixture
        def proxy(self):
            config = ExternalFrameworkConfig(base_url="https://api.test.com")
            return EnterpriseProxy(frameworks={"fw1": config})

        def test_get_circuit_breaker_status_all(self, proxy):
            status = proxy.get_circuit_breaker_status()
            assert "fw1" in status
            assert status["fw1"]["state"] == "closed"

        def test_get_circuit_breaker_status_single(self, proxy):
            status = proxy.get_circuit_breaker_status("fw1")
            assert status["framework"] == "fw1"

        def test_get_circuit_breaker_status_missing(self, proxy):
            assert proxy.get_circuit_breaker_status("unknown") == {}

        def test_get_bulkhead_status_all(self, proxy):
            status = proxy.get_bulkhead_status()
            assert "fw1" in status

        def test_get_bulkhead_status_single(self, proxy):
            status = proxy.get_bulkhead_status("fw1")
            assert "active" in status

        def test_get_health_status_all(self, proxy):
            status = proxy.get_health_status()
            assert "fw1" in status

        def test_get_health_status_missing(self, proxy):
            assert proxy.get_health_status("unknown") == {}

        def test_get_stats(self, proxy):
            stats = proxy.get_stats()
            assert "config" in stats
            assert "frameworks" in stats
            assert "circuit_breakers" in stats
            assert "bulkheads" in stats
            assert "health" in stats
            assert "hooks" in stats
            assert stats["frameworks"]["fw1"]["enabled"] is True

        @pytest.mark.asyncio
        async def test_reset_circuit_breaker(self, proxy):
            cb = proxy._circuit_breakers["fw1"]
            for _ in range(5):
                await cb.record_failure()
            assert cb.state == "open"

            result = await proxy.reset_circuit_breaker("fw1")
            assert result is True
            assert cb.state == "closed"

        @pytest.mark.asyncio
        async def test_reset_circuit_breaker_missing(self, proxy):
            result = await proxy.reset_circuit_breaker("nonexistent")
            assert result is False

    class TestRequestValidation:
        """Test request validation before execution."""

        @pytest.fixture
        def proxy(self):
            return EnterpriseProxy(
                frameworks={
                    "test": ExternalFrameworkConfig(base_url="https://api.test.com"),
                }
            )

        @pytest.mark.asyncio
        async def test_request_framework_not_configured(self, proxy):
            with pytest.raises(FrameworkNotConfiguredError):
                await proxy.request("unknown", "GET", "/test")

        @pytest.mark.asyncio
        async def test_request_framework_disabled(self, proxy):
            proxy._frameworks["test"].enabled = False
            with pytest.raises(ProxyError, match="disabled"):
                await proxy.request("test", "GET", "/test")

        @pytest.mark.asyncio
        async def test_request_circuit_open(self, proxy):
            cb = proxy._circuit_breakers["test"]
            for _ in range(5):
                await cb.record_failure()

            with pytest.raises(CircuitOpenError):
                await proxy.request("test", "GET", "/test")

    class TestRetryDelay:
        """Test retry delay calculation."""

        @pytest.fixture
        def proxy(self):
            return EnterpriseProxy()

        def test_exponential_backoff(self, proxy):
            settings = RetrySettings(
                strategy=RetryStrategy.EXPONENTIAL,
                base_delay=1.0,
                max_delay=30.0,
                jitter=False,
            )
            assert proxy._calculate_retry_delay(0, settings) == 1.0
            assert proxy._calculate_retry_delay(1, settings) == 2.0
            assert proxy._calculate_retry_delay(2, settings) == 4.0

        def test_linear_backoff(self, proxy):
            settings = RetrySettings(
                strategy=RetryStrategy.LINEAR,
                base_delay=1.0,
                max_delay=30.0,
                jitter=False,
            )
            assert proxy._calculate_retry_delay(0, settings) == 1.0
            assert proxy._calculate_retry_delay(1, settings) == 2.0
            assert proxy._calculate_retry_delay(2, settings) == 3.0

        def test_constant_backoff(self, proxy):
            settings = RetrySettings(
                strategy=RetryStrategy.CONSTANT,
                base_delay=2.0,
                max_delay=30.0,
                jitter=False,
            )
            assert proxy._calculate_retry_delay(0, settings) == 2.0
            assert proxy._calculate_retry_delay(5, settings) == 2.0

        def test_max_delay_cap(self, proxy):
            settings = RetrySettings(
                strategy=RetryStrategy.EXPONENTIAL,
                base_delay=1.0,
                max_delay=5.0,
                jitter=False,
            )
            assert proxy._calculate_retry_delay(10, settings) == 5.0

        def test_jitter_applied(self, proxy):
            settings = RetrySettings(
                strategy=RetryStrategy.CONSTANT,
                base_delay=1.0,
                max_delay=30.0,
                jitter=True,
            )
            delays = {proxy._calculate_retry_delay(0, settings) for _ in range(20)}
            # With jitter, we should get variation
            assert len(delays) > 1

    class TestLifecycle:
        """Test proxy lifecycle management."""

        @pytest.mark.asyncio
        async def test_shutdown_without_start(self):
            proxy = EnterpriseProxy()
            await proxy.shutdown()  # Should not raise

        @pytest.mark.asyncio
        async def test_shutdown_cancels_health_task(self):
            proxy = EnterpriseProxy()

            # Create a real asyncio task that we can cancel
            async def noop():
                await asyncio.sleep(100)

            task = asyncio.create_task(noop())
            proxy._health_check_task = task
            proxy._session = None

            await proxy.shutdown()
            assert task.cancelled()


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enum values."""

    def test_health_status_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_retry_strategy_values(self):
        assert RetryStrategy.EXPONENTIAL.value == "exponential"
        assert RetryStrategy.LINEAR.value == "linear"
        assert RetryStrategy.CONSTANT.value == "constant"
