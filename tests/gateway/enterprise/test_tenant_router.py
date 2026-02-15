"""
Comprehensive tests for TenantRouter and related classes.

Tests cover:
- Exception classes
- Enums
- Data classes (EndpointConfig, EndpointHealth, TenantQuotas, etc.)
- QuotaTracker
- EndpointHealthTracker
- TenantRouter
- TenantRoutingContext
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.enterprise.tenant_router import (
    # Exceptions
    TenantRoutingError,
    TenantNotFoundError,
    NoAvailableEndpointError,
    QuotaExceededError,
    CrossTenantAccessError,
    # Enums
    LoadBalancingStrategy,
    EndpointStatus,
    RoutingEventType,
    # Data classes
    EndpointConfig,
    EndpointHealth,
    TenantQuotas,
    QuotaStatus,
    TenantRoutingConfig,
    RoutingDecision,
    RoutingAuditEntry,
    # Core classes
    QuotaTracker,
    EndpointHealthTracker,
    TenantRouter,
    TenantRoutingContext,
)
from aragora.tenancy.context import TenantContext, TenantNotSetError
from aragora.tenancy.isolation import IsolationLevel


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def basic_endpoint_config():
    """Create a basic endpoint configuration."""
    return EndpointConfig(
        url="https://api.example.com",
        weight=100,
        priority=1,
        timeout=30.0,
        max_retries=3,
        headers={"X-API-Key": "test-key"},
    )


@pytest.fixture
def basic_tenant_quotas():
    """Create basic tenant quotas."""
    return TenantQuotas(
        requests_per_minute=60,
        requests_per_hour=1000,
        requests_per_day=10000,
        concurrent_requests=10,
        bandwidth_bytes_per_minute=10 * 1024 * 1024,
        warn_threshold=0.8,
    )


@pytest.fixture
def basic_tenant_config(basic_endpoint_config, basic_tenant_quotas):
    """Create a basic tenant routing configuration."""
    return TenantRoutingConfig(
        tenant_id="test-tenant",
        endpoints=[basic_endpoint_config],
        quotas=basic_tenant_quotas,
        load_balancing=LoadBalancingStrategy.WEIGHTED_RANDOM,
        enable_fallback=True,
        fallback_endpoints=[
            EndpointConfig(url="https://fallback.example.com", weight=50, priority=2)
        ],
    )


@pytest.fixture
def tenant_router(basic_tenant_config):
    """Create a TenantRouter with basic configuration."""
    return TenantRouter(
        configs=[basic_tenant_config],
        enable_audit=True,
    )


# =============================================================================
# Test Exceptions
# =============================================================================


class TestTenantRoutingError:
    """Tests for TenantRoutingError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = TenantRoutingError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.tenant_id is None
        assert error.code == "ROUTING_ERROR"
        assert error.details == {}

    def test_error_with_tenant_and_details(self):
        """Test error with all attributes."""
        error = TenantRoutingError(
            message="Failed routing",
            tenant_id="tenant-123",
            code="CUSTOM_ERROR",
            details={"key": "value"},
        )
        assert error.tenant_id == "tenant-123"
        assert error.code == "CUSTOM_ERROR"
        assert error.details == {"key": "value"}


class TestTenantNotFoundError:
    """Tests for TenantNotFoundError exception."""

    def test_tenant_not_found(self):
        """Test TenantNotFoundError creation."""
        error = TenantNotFoundError("missing-tenant")
        assert "missing-tenant" in str(error)
        assert error.tenant_id == "missing-tenant"
        assert error.code == "TENANT_NOT_FOUND"

    def test_with_details(self):
        """Test with additional details."""
        error = TenantNotFoundError("missing-tenant", details={"source": "database"})
        assert error.details["source"] == "database"


class TestNoAvailableEndpointError:
    """Tests for NoAvailableEndpointError exception."""

    def test_no_available_endpoint(self):
        """Test NoAvailableEndpointError creation."""
        error = NoAvailableEndpointError("tenant-123")
        assert "tenant-123" in str(error)
        assert error.code == "NO_AVAILABLE_ENDPOINT"


class TestQuotaExceededError:
    """Tests for QuotaExceededError exception."""

    def test_quota_exceeded(self):
        """Test QuotaExceededError creation."""
        error = QuotaExceededError(
            tenant_id="tenant-123",
            quota_type="requests_per_minute",
            limit=100,
            current=105,
            retry_after=30,
        )
        assert error.tenant_id == "tenant-123"
        assert error.quota_type == "requests_per_minute"
        assert error.limit == 100
        assert error.current == 105
        assert error.retry_after == 30
        assert error.code == "QUOTA_EXCEEDED"
        assert error.details["quota_type"] == "requests_per_minute"


class TestCrossTenantAccessError:
    """Tests for CrossTenantAccessError exception."""

    def test_cross_tenant_access(self):
        """Test CrossTenantAccessError creation."""
        error = CrossTenantAccessError(
            requesting_tenant="tenant-a",
            target_tenant="tenant-b",
        )
        assert error.requesting_tenant == "tenant-a"
        assert error.target_tenant == "tenant-b"
        assert error.code == "CROSS_TENANT_ACCESS"
        assert "tenant-a" in str(error)
        assert "tenant-b" in str(error)


# =============================================================================
# Test Enums
# =============================================================================


class TestLoadBalancingStrategy:
    """Tests for LoadBalancingStrategy enum."""

    def test_strategy_values(self):
        """Test all strategy values."""
        assert LoadBalancingStrategy.ROUND_ROBIN.value == "round_robin"
        assert LoadBalancingStrategy.WEIGHTED_RANDOM.value == "weighted_random"
        assert LoadBalancingStrategy.LEAST_CONNECTIONS.value == "least_connections"
        assert LoadBalancingStrategy.PRIORITY.value == "priority"
        assert LoadBalancingStrategy.LATENCY.value == "latency"


class TestEndpointStatus:
    """Tests for EndpointStatus enum."""

    def test_status_values(self):
        """Test all status values."""
        assert EndpointStatus.HEALTHY.value == "healthy"
        assert EndpointStatus.DEGRADED.value == "degraded"
        assert EndpointStatus.UNHEALTHY.value == "unhealthy"
        assert EndpointStatus.UNKNOWN.value == "unknown"


class TestRoutingEventType:
    """Tests for RoutingEventType enum."""

    def test_event_type_values(self):
        """Test all event type values."""
        assert RoutingEventType.ROUTE_SUCCESS.value == "route_success"
        assert RoutingEventType.ROUTE_FALLBACK.value == "route_fallback"
        assert RoutingEventType.ROUTE_FAILED.value == "route_failed"
        assert RoutingEventType.QUOTA_WARNING.value == "quota_warning"
        assert RoutingEventType.QUOTA_EXCEEDED.value == "quota_exceeded"
        assert RoutingEventType.ENDPOINT_HEALTH_CHANGE.value == "endpoint_health_change"
        assert RoutingEventType.CROSS_TENANT_BLOCKED.value == "cross_tenant_blocked"


# =============================================================================
# Test Data Classes
# =============================================================================


class TestEndpointConfig:
    """Tests for EndpointConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EndpointConfig(url="https://api.example.com")
        assert config.weight == 100
        assert config.priority == 1
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.health_check_path == "/health"
        assert config.headers == {}
        assert config.metadata == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EndpointConfig(
            url="https://custom.api.com",
            weight=50,
            priority=2,
            timeout=60.0,
            max_retries=5,
            health_check_path="/ping",
            headers={"Authorization": "Bearer token"},
            metadata={"region": "us-west"},
        )
        assert config.url == "https://custom.api.com"
        assert config.weight == 50
        assert config.priority == 2
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.health_check_path == "/ping"
        assert config.headers["Authorization"] == "Bearer token"
        assert config.metadata["region"] == "us-west"


class TestEndpointHealth:
    """Tests for EndpointHealth dataclass."""

    def test_default_values(self):
        """Test default health values."""
        health = EndpointHealth(endpoint_url="https://api.example.com")
        assert health.status == EndpointStatus.UNKNOWN
        assert health.consecutive_failures == 0
        assert health.latency_ms == 0.0
        assert health.active_connections == 0
        assert health.error_rate == 0.0

    def test_custom_values(self):
        """Test custom health values."""
        health = EndpointHealth(
            endpoint_url="https://api.example.com",
            status=EndpointStatus.HEALTHY,
            consecutive_failures=0,
            latency_ms=150.0,
            active_connections=5,
            error_rate=0.01,
        )
        assert health.status == EndpointStatus.HEALTHY
        assert health.latency_ms == 150.0
        assert health.active_connections == 5


class TestTenantQuotas:
    """Tests for TenantQuotas dataclass."""

    def test_default_values(self):
        """Test default quota values."""
        quotas = TenantQuotas()
        assert quotas.requests_per_minute == 60
        assert quotas.requests_per_hour == 1000
        assert quotas.requests_per_day == 10000
        assert quotas.concurrent_requests == 10
        assert quotas.bandwidth_bytes_per_minute == 10 * 1024 * 1024
        assert quotas.warn_threshold == 0.8

    def test_custom_values(self):
        """Test custom quota values."""
        quotas = TenantQuotas(
            requests_per_minute=100,
            requests_per_hour=2000,
            requests_per_day=20000,
            concurrent_requests=20,
            bandwidth_bytes_per_minute=20 * 1024 * 1024,
            warn_threshold=0.9,
        )
        assert quotas.requests_per_minute == 100
        assert quotas.concurrent_requests == 20


class TestQuotaStatus:
    """Tests for QuotaStatus dataclass."""

    def test_quota_status_creation(self):
        """Test quota status creation."""
        status = QuotaStatus(
            tenant_id="tenant-123",
            used=50,
            remaining=50,
            limit=100,
            reset_time=datetime.now(timezone.utc),
            quota_type="requests_per_minute",
            is_exceeded=False,
            is_warning=False,
            percentage_used=50.0,
        )
        assert status.tenant_id == "tenant-123"
        assert status.used == 50
        assert status.remaining == 50

    def test_to_dict(self):
        """Test conversion to dictionary."""
        reset_time = datetime.now(timezone.utc)
        status = QuotaStatus(
            tenant_id="tenant-123",
            used=80,
            remaining=20,
            limit=100,
            reset_time=reset_time,
            quota_type="requests_per_minute",
            is_exceeded=False,
            is_warning=True,
            percentage_used=80.0,
        )
        result = status.to_dict()
        assert result["tenant_id"] == "tenant-123"
        assert result["used"] == 80
        assert result["remaining"] == 20
        assert result["limit"] == 100
        assert result["is_warning"] is True
        assert result["percentage_used"] == 80.0
        assert result["reset_time"] == reset_time.isoformat()


class TestTenantRoutingConfig:
    """Tests for TenantRoutingConfig dataclass."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = TenantRoutingConfig(
            tenant_id="test-tenant",
            endpoints=[EndpointConfig(url="https://api.example.com")],
        )
        assert config.tenant_id == "test-tenant"
        assert len(config.endpoints) == 1
        assert config.load_balancing == LoadBalancingStrategy.WEIGHTED_RANDOM
        assert config.enable_fallback is True

    def test_empty_tenant_id_raises(self):
        """Test that empty tenant_id raises ValueError."""
        with pytest.raises(ValueError, match="tenant_id is required"):
            TenantRoutingConfig(
                tenant_id="",
                endpoints=[EndpointConfig(url="https://api.example.com")],
            )

    def test_no_endpoints_raises(self):
        """Test that no endpoints raises ValueError."""
        with pytest.raises(ValueError, match="At least one endpoint"):
            TenantRoutingConfig(
                tenant_id="test-tenant",
                endpoints=[],
                fallback_endpoints=[],
            )

    def test_fallback_only_config(self):
        """Test config with only fallback endpoints is valid."""
        config = TenantRoutingConfig(
            tenant_id="test-tenant",
            endpoints=[],
            fallback_endpoints=[EndpointConfig(url="https://fallback.example.com")],
        )
        assert len(config.fallback_endpoints) == 1


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """Test routing decision creation."""
        decision = RoutingDecision(
            target_endpoint="https://api.example.com",
            tenant_context={"X-Tenant-ID": "tenant-123"},
            headers={"Authorization": "Bearer token"},
            used_fallback=False,
            decision_time_ms=5.5,
        )
        assert decision.target_endpoint == "https://api.example.com"
        assert decision.tenant_context["X-Tenant-ID"] == "tenant-123"
        assert decision.used_fallback is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        decision = RoutingDecision(
            target_endpoint="https://api.example.com",
            tenant_context={"X-Tenant-ID": "tenant-123"},
            headers={"Authorization": "Bearer token"},
            used_fallback=True,
            routing_metadata={"strategy": "weighted_random"},
            decision_time_ms=5.5,
        )
        result = decision.to_dict()
        assert result["target_endpoint"] == "https://api.example.com"
        assert result["used_fallback"] is True
        assert result["decision_time_ms"] == 5.5
        assert result["routing_metadata"]["strategy"] == "weighted_random"


# =============================================================================
# Test QuotaTracker
# =============================================================================


class TestQuotaTracker:
    """Tests for QuotaTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh quota tracker."""
        return QuotaTracker()

    @pytest.fixture
    def quotas(self):
        """Create test quotas."""
        return TenantQuotas(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000,
            concurrent_requests=5,
            bandwidth_bytes_per_minute=1024 * 1024,
            warn_threshold=0.8,
        )

    @pytest.mark.asyncio
    async def test_check_and_consume_allowed(self, tracker, quotas):
        """Test successful quota consumption."""
        allowed, status = await tracker.check_and_consume("tenant-1", quotas)
        assert allowed is True
        assert status is None

    @pytest.mark.asyncio
    async def test_concurrent_limit_exceeded(self, tracker, quotas):
        """Test concurrent request limit."""
        # Consume all concurrent slots
        for _ in range(5):
            allowed, _ = await tracker.check_and_consume("tenant-1", quotas)
            assert allowed is True

        # Next request should fail
        allowed, status = await tracker.check_and_consume("tenant-1", quotas)
        assert allowed is False
        assert status is not None
        assert status.quota_type == "concurrent_requests"
        assert status.is_exceeded is True

    @pytest.mark.asyncio
    async def test_per_minute_limit_exceeded(self, tracker, quotas):
        """Test per-minute rate limit."""
        # Consume all per-minute slots
        for i in range(10):
            allowed, _ = await tracker.check_and_consume("tenant-1", quotas)
            assert allowed is True
            # Release concurrent slot immediately
            await tracker.release_concurrent("tenant-1")

        # Next request should fail
        allowed, status = await tracker.check_and_consume("tenant-1", quotas)
        assert allowed is False
        assert status is not None
        assert status.quota_type == "requests_per_minute"

    @pytest.mark.asyncio
    async def test_bandwidth_limit_exceeded(self, tracker, quotas):
        """Test bandwidth limit."""
        # Use most of bandwidth
        allowed, _ = await tracker.check_and_consume("tenant-1", quotas, bytes_size=1000 * 1024)
        assert allowed is True
        await tracker.release_concurrent("tenant-1")

        # Next large request should fail
        allowed, status = await tracker.check_and_consume("tenant-1", quotas, bytes_size=500 * 1024)
        assert allowed is False
        assert status is not None
        assert status.quota_type == "bandwidth_bytes_per_minute"

    @pytest.mark.asyncio
    async def test_release_concurrent(self, tracker, quotas):
        """Test releasing concurrent slots."""
        # Consume all slots
        for _ in range(5):
            await tracker.check_and_consume("tenant-1", quotas)

        # Release one slot
        await tracker.release_concurrent("tenant-1")

        # Should be able to consume again
        allowed, _ = await tracker.check_and_consume("tenant-1", quotas)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_get_status(self, tracker, quotas):
        """Test getting quota status."""
        # Consume some quota
        for _ in range(5):
            await tracker.check_and_consume("tenant-1", quotas)

        status = await tracker.get_status("tenant-1", quotas)
        assert "requests_per_minute" in status
        assert "requests_per_hour" in status
        assert "requests_per_day" in status
        assert "concurrent_requests" in status

        minute_status = status["requests_per_minute"]
        assert minute_status.used == 5
        assert minute_status.remaining == 5
        assert minute_status.percentage_used == 50.0

    @pytest.mark.asyncio
    async def test_reset(self, tracker, quotas):
        """Test resetting quota for a tenant."""
        # Consume some quota
        for _ in range(5):
            await tracker.check_and_consume("tenant-1", quotas)

        # Reset
        await tracker.reset("tenant-1")

        # Should have full quota again
        status = await tracker.get_status("tenant-1", quotas)
        assert status["requests_per_minute"].used == 0
        assert status["concurrent_requests"].used == 0

    @pytest.mark.asyncio
    async def test_per_hour_limit(self, tracker):
        """Test per-hour rate limit."""
        quotas = TenantQuotas(
            requests_per_minute=100,  # High minute limit
            requests_per_hour=5,  # Low hour limit
            requests_per_day=1000,
            concurrent_requests=100,
        )

        # Consume all per-hour slots
        for i in range(5):
            allowed, _ = await tracker.check_and_consume("tenant-1", quotas)
            assert allowed is True
            await tracker.release_concurrent("tenant-1")

        # Next request should fail on hour limit
        allowed, status = await tracker.check_and_consume("tenant-1", quotas)
        assert allowed is False
        assert status.quota_type == "requests_per_hour"

    @pytest.mark.asyncio
    async def test_per_day_limit(self, tracker):
        """Test per-day rate limit."""
        quotas = TenantQuotas(
            requests_per_minute=100,
            requests_per_hour=100,
            requests_per_day=5,  # Low day limit
            concurrent_requests=100,
        )

        # Consume all per-day slots
        for i in range(5):
            allowed, _ = await tracker.check_and_consume("tenant-1", quotas)
            assert allowed is True
            await tracker.release_concurrent("tenant-1")

        # Next request should fail on day limit
        allowed, status = await tracker.check_and_consume("tenant-1", quotas)
        assert allowed is False
        assert status.quota_type == "requests_per_day"


# =============================================================================
# Test EndpointHealthTracker
# =============================================================================


class TestEndpointHealthTracker:
    """Tests for EndpointHealthTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh health tracker."""
        return EndpointHealthTracker(
            unhealthy_threshold=3,
            recovery_timeout=30.0,
        )

    @pytest.mark.asyncio
    async def test_record_success(self, tracker):
        """Test recording successful request."""
        await tracker.record_success("https://api.example.com", latency_ms=100.0)

        health = await tracker.get_health("https://api.example.com")
        assert health.status == EndpointStatus.HEALTHY
        assert health.consecutive_failures == 0
        assert health.latency_ms == 100.0

    @pytest.mark.asyncio
    async def test_record_failure(self, tracker):
        """Test recording failed request.

        With unhealthy_threshold=3, threshold//2=1, so first failure
        triggers DEGRADED (1 >= 1).
        """
        status = await tracker.record_failure("https://api.example.com", "Connection error")

        # With threshold=3, threshold//2=1, so first failure triggers DEGRADED
        assert status == EndpointStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_degraded_after_partial_failures(self, tracker):
        """Test endpoint stays degraded after partial failures."""
        # With threshold=3, threshold//2=1, so even 1 failure triggers degraded
        status = await tracker.record_failure("https://api.example.com", "Error 1")
        assert status == EndpointStatus.DEGRADED

        status = await tracker.record_failure("https://api.example.com", "Error 2")
        assert status == EndpointStatus.DEGRADED  # Still degraded, not yet unhealthy

    @pytest.mark.asyncio
    async def test_unhealthy_after_threshold_failures(self, tracker):
        """Test endpoint becomes unhealthy after threshold failures."""
        for i in range(3):
            status = await tracker.record_failure("https://api.example.com", f"Error {i}")

        assert status == EndpointStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_is_available_healthy(self, tracker):
        """Test availability check for healthy endpoint."""
        await tracker.record_success("https://api.example.com", 100.0)
        available = await tracker.is_available("https://api.example.com")
        assert available is True

    @pytest.mark.asyncio
    async def test_is_available_unhealthy(self, tracker):
        """Test availability check for unhealthy endpoint."""
        # Make endpoint unhealthy
        for _ in range(3):
            await tracker.record_failure("https://api.example.com", "Error")

        available = await tracker.is_available("https://api.example.com")
        assert available is False

    @pytest.mark.asyncio
    async def test_is_available_unknown_endpoint(self, tracker):
        """Test availability check for unknown endpoint (defaults to available)."""
        available = await tracker.is_available("https://unknown.example.com")
        assert available is True

    @pytest.mark.asyncio
    async def test_recovery_after_timeout(self, tracker):
        """Test endpoint recovery after timeout."""
        # Create tracker with short recovery timeout
        short_tracker = EndpointHealthTracker(
            unhealthy_threshold=3,
            recovery_timeout=0.1,
        )

        # Make endpoint unhealthy
        for _ in range(3):
            await short_tracker.record_failure("https://api.example.com", "Error")

        # Should be unavailable
        available = await short_tracker.is_available("https://api.example.com")
        assert available is False

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should be available again (in degraded state)
        available = await short_tracker.is_available("https://api.example.com")
        assert available is True

    @pytest.mark.asyncio
    async def test_get_health_unknown(self, tracker):
        """Test getting health for unknown endpoint."""
        health = await tracker.get_health("https://unknown.example.com")
        assert health.endpoint_url == "https://unknown.example.com"
        assert health.status == EndpointStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_get_all_health(self, tracker):
        """Test getting all endpoint health."""
        await tracker.record_success("https://api1.example.com", 100.0)
        await tracker.record_success("https://api2.example.com", 150.0)

        all_health = await tracker.get_all_health()
        assert len(all_health) == 2
        assert "https://api1.example.com" in all_health
        assert "https://api2.example.com" in all_health

    @pytest.mark.asyncio
    async def test_latency_exponential_moving_average(self, tracker):
        """Test latency calculation uses EMA."""
        await tracker.record_success("https://api.example.com", latency_ms=100.0)
        await tracker.record_success("https://api.example.com", latency_ms=200.0)

        health = await tracker.get_health("https://api.example.com")
        # EMA: 0.7 * 100 + 0.3 * 200 = 130
        assert health.latency_ms == pytest.approx(130.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_success_resets_failures(self, tracker):
        """Test successful request resets failure count."""
        # Record some failures
        await tracker.record_failure("https://api.example.com", "Error")
        await tracker.record_failure("https://api.example.com", "Error")

        # Record success
        await tracker.record_success("https://api.example.com", 100.0)

        health = await tracker.get_health("https://api.example.com")
        assert health.consecutive_failures == 0
        assert health.status == EndpointStatus.HEALTHY


# =============================================================================
# Test TenantRouter
# =============================================================================


class TestTenantRouter:
    """Tests for TenantRouter class."""

    @pytest.mark.asyncio
    async def test_init_with_configs(self, basic_tenant_config):
        """Test router initialization with configs."""
        router = TenantRouter(configs=[basic_tenant_config])
        assert len(router._configs) == 1
        assert "test-tenant" in router._configs

    @pytest.mark.asyncio
    async def test_add_tenant_config(self, tenant_router):
        """Test adding a new tenant configuration."""
        new_config = TenantRoutingConfig(
            tenant_id="new-tenant",
            endpoints=[EndpointConfig(url="https://new.example.com")],
        )
        await tenant_router.add_tenant_config(new_config)
        assert "new-tenant" in tenant_router._configs

    @pytest.mark.asyncio
    async def test_remove_tenant_config(self, tenant_router):
        """Test removing a tenant configuration."""
        result = await tenant_router.remove_tenant_config("test-tenant")
        assert result is True
        assert "test-tenant" not in tenant_router._configs

    @pytest.mark.asyncio
    async def test_remove_nonexistent_config(self, tenant_router):
        """Test removing a nonexistent tenant configuration."""
        result = await tenant_router.remove_tenant_config("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_tenant_config(self, tenant_router):
        """Test getting a tenant configuration."""
        config = await tenant_router.get_tenant_config("test-tenant")
        assert config is not None
        assert config.tenant_id == "test-tenant"

    @pytest.mark.asyncio
    async def test_get_nonexistent_config(self, tenant_router):
        """Test getting a nonexistent tenant configuration."""
        config = await tenant_router.get_tenant_config("nonexistent")
        assert config is None

    @pytest.mark.asyncio
    async def test_list_tenant_configs(self, tenant_router):
        """Test listing all tenant configurations."""
        configs = await tenant_router.list_tenant_configs()
        assert len(configs) == 1
        assert configs[0].tenant_id == "test-tenant"

    @pytest.mark.asyncio
    async def test_route_success(self, tenant_router):
        """Test successful routing with tenant context."""
        async with TenantContext(tenant_id="test-tenant"):
            decision = await tenant_router.route(tenant_id="test-tenant")
            assert decision.target_endpoint == "https://api.example.com"
            assert decision.tenant_context["X-Tenant-ID"] == "test-tenant"
            assert decision.used_fallback is False

    @pytest.mark.asyncio
    async def test_route_no_tenant_context(self, tenant_router):
        """Test routing without tenant context raises error."""
        with pytest.raises(TenantNotSetError):
            await tenant_router.route()

    @pytest.mark.asyncio
    async def test_route_tenant_not_found(self, tenant_router):
        """Test routing for nonexistent tenant raises error."""
        async with TenantContext(tenant_id="nonexistent"):
            with pytest.raises(TenantNotFoundError):
                await tenant_router.route(tenant_id="nonexistent")

    @pytest.mark.asyncio
    async def test_route_cross_tenant_access(self, tenant_router):
        """Test cross-tenant access is blocked."""
        # Add another tenant
        await tenant_router.add_tenant_config(
            TenantRoutingConfig(
                tenant_id="other-tenant",
                endpoints=[EndpointConfig(url="https://other.example.com")],
            )
        )

        async with TenantContext(tenant_id="test-tenant"):
            with pytest.raises(CrossTenantAccessError):
                await tenant_router.route(tenant_id="other-tenant")

    @pytest.mark.asyncio
    async def test_route_quota_exceeded(self, tenant_router):
        """Test routing when quota is exceeded."""
        # Update config with very low quota
        low_quota_config = TenantRoutingConfig(
            tenant_id="limited-tenant",
            endpoints=[EndpointConfig(url="https://limited.example.com")],
            quotas=TenantQuotas(
                requests_per_minute=1,
                concurrent_requests=1,
            ),
        )
        await tenant_router.add_tenant_config(low_quota_config)

        async with TenantContext(tenant_id="limited-tenant"):
            # First request succeeds
            decision = await tenant_router.route(tenant_id="limited-tenant")
            assert decision is not None

            # Second request should fail (concurrent limit)
            with pytest.raises(QuotaExceededError) as exc_info:
                await tenant_router.route(tenant_id="limited-tenant")
            assert exc_info.value.quota_type == "concurrent_requests"

    @pytest.mark.asyncio
    async def test_route_operation_not_allowed(self, tenant_router):
        """Test routing with disallowed operation."""
        restricted_config = TenantRoutingConfig(
            tenant_id="restricted-tenant",
            endpoints=[EndpointConfig(url="https://restricted.example.com")],
            allowed_operations={"read", "list"},
        )
        await tenant_router.add_tenant_config(restricted_config)

        async with TenantContext(tenant_id="restricted-tenant"):
            with pytest.raises(TenantRoutingError) as exc_info:
                await tenant_router.route(
                    tenant_id="restricted-tenant",
                    operation="delete",
                )
            assert exc_info.value.code == "OPERATION_NOT_ALLOWED"

    @pytest.mark.asyncio
    async def test_route_uses_fallback(self, tenant_router):
        """Test routing falls back when primary endpoints unavailable."""
        # Mark primary endpoint as unhealthy
        for _ in range(3):
            await tenant_router._health_tracker.record_failure("https://api.example.com", "Error")

        async with TenantContext(tenant_id="test-tenant"):
            decision = await tenant_router.route(tenant_id="test-tenant")
            assert decision.target_endpoint == "https://fallback.example.com"
            assert decision.used_fallback is True

    @pytest.mark.asyncio
    async def test_route_no_available_endpoint(self, tenant_router):
        """Test routing when no endpoints are available."""
        # Mark all endpoints as unhealthy
        for _ in range(3):
            await tenant_router._health_tracker.record_failure("https://api.example.com", "Error")
            await tenant_router._health_tracker.record_failure(
                "https://fallback.example.com", "Error"
            )

        async with TenantContext(tenant_id="test-tenant"):
            with pytest.raises(NoAvailableEndpointError):
                await tenant_router.route(tenant_id="test-tenant")

    @pytest.mark.asyncio
    async def test_complete_request_success(self, tenant_router):
        """Test completing a successful request."""
        async with TenantContext(tenant_id="test-tenant"):
            decision = await tenant_router.route(tenant_id="test-tenant")

        await tenant_router.complete_request(
            tenant_id="test-tenant",
            endpoint_url=decision.target_endpoint,
            success=True,
            latency_ms=100.0,
        )

        health = await tenant_router.get_endpoint_health(decision.target_endpoint)
        assert health.status == EndpointStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_complete_request_failure(self, tenant_router):
        """Test completing a failed request."""
        async with TenantContext(tenant_id="test-tenant"):
            decision = await tenant_router.route(tenant_id="test-tenant")

        await tenant_router.complete_request(
            tenant_id="test-tenant",
            endpoint_url=decision.target_endpoint,
            success=False,
            latency_ms=100.0,
            error="Connection timeout",
        )

        health = await tenant_router.get_endpoint_health(decision.target_endpoint)
        assert health.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_get_quota_status(self, tenant_router):
        """Test getting quota status for a tenant."""
        async with TenantContext(tenant_id="test-tenant"):
            status = await tenant_router.get_quota_status(tenant_id="test-tenant")
            assert "requests_per_minute" in status
            assert "concurrent_requests" in status

    @pytest.mark.asyncio
    async def test_get_quota_status_no_tenant(self, tenant_router):
        """Test getting quota status without tenant context."""
        with pytest.raises(TenantNotSetError):
            await tenant_router.get_quota_status()

    @pytest.mark.asyncio
    async def test_reset_quota(self, tenant_router):
        """Test resetting quota for a tenant."""
        async with TenantContext(tenant_id="test-tenant"):
            # Consume some quota
            await tenant_router.route(tenant_id="test-tenant")

        # Reset quota
        await tenant_router.reset_quota("test-tenant")

        async with TenantContext(tenant_id="test-tenant"):
            status = await tenant_router.get_quota_status(tenant_id="test-tenant")
            assert status["concurrent_requests"].used == 0

    @pytest.mark.asyncio
    async def test_get_endpoint_health(self, tenant_router):
        """Test getting endpoint health."""
        health = await tenant_router.get_endpoint_health("https://api.example.com")
        assert health.endpoint_url == "https://api.example.com"

    @pytest.mark.asyncio
    async def test_get_all_endpoint_health(self, tenant_router):
        """Test getting all endpoint health."""
        # Generate some health data
        async with TenantContext(tenant_id="test-tenant"):
            await tenant_router.route(tenant_id="test-tenant")

        health = await tenant_router.get_all_endpoint_health()
        # May have entries based on routing
        assert isinstance(health, dict)

    @pytest.mark.asyncio
    async def test_get_tenant_health(self, tenant_router):
        """Test getting tenant health summary."""
        health = await tenant_router.get_tenant_health("test-tenant")
        assert health["tenant_id"] == "test-tenant"
        assert "overall_status" in health
        assert "primary_endpoints" in health
        assert "fallback_endpoints" in health

    @pytest.mark.asyncio
    async def test_get_tenant_health_not_found(self, tenant_router):
        """Test getting health for nonexistent tenant."""
        with pytest.raises(TenantNotFoundError):
            await tenant_router.get_tenant_health("nonexistent")

    @pytest.mark.asyncio
    async def test_get_audit_log(self, tenant_router):
        """Test getting audit log entries."""
        async with TenantContext(tenant_id="test-tenant"):
            await tenant_router.route(tenant_id="test-tenant")

        entries = await tenant_router.get_audit_log(tenant_id="test-tenant")
        assert len(entries) > 0
        assert entries[0]["tenant_id"] == "test-tenant"

    @pytest.mark.asyncio
    async def test_get_audit_log_with_filters(self, tenant_router):
        """Test getting audit log with filters."""
        async with TenantContext(tenant_id="test-tenant"):
            await tenant_router.route(tenant_id="test-tenant")

        entries = await tenant_router.get_audit_log(
            tenant_id="test-tenant",
            event_type=RoutingEventType.ROUTE_SUCCESS,
            limit=50,
        )
        assert all(e["event_type"] == "route_success" for e in entries)

    def test_add_event_handler(self, tenant_router):
        """Test adding an event handler."""
        handler = MagicMock()
        tenant_router.add_event_handler(handler)
        assert handler in tenant_router._event_handlers

    def test_remove_event_handler(self, tenant_router):
        """Test removing an event handler."""
        handler = MagicMock()
        tenant_router.add_event_handler(handler)
        tenant_router.remove_event_handler(handler)
        assert handler not in tenant_router._event_handlers

    @pytest.mark.asyncio
    async def test_event_handler_called(self, tenant_router):
        """Test event handlers are called on routing."""
        handler = MagicMock()
        tenant_router.add_event_handler(handler)

        async with TenantContext(tenant_id="test-tenant"):
            await tenant_router.route(tenant_id="test-tenant")

        assert handler.called
        call_args = handler.call_args[0][0]
        assert call_args.tenant_id == "test-tenant"

    @pytest.mark.asyncio
    async def test_get_stats(self, tenant_router):
        """Test getting router statistics."""
        stats = await tenant_router.get_stats()
        assert stats["tenant_count"] == 1
        assert stats["total_endpoints"] == 2  # 1 primary + 1 fallback
        assert stats["audit_enabled"] is True

    @pytest.mark.asyncio
    async def test_audit_disabled(self):
        """Test router with audit disabled."""
        router = TenantRouter(
            configs=[
                TenantRoutingConfig(
                    tenant_id="test",
                    endpoints=[EndpointConfig(url="https://api.example.com")],
                )
            ],
            enable_audit=False,
        )

        async with TenantContext(tenant_id="test"):
            await router.route(tenant_id="test")

        entries = await router.get_audit_log()
        assert len(entries) == 0


class TestLoadBalancingStrategies:
    """Tests for different load balancing strategies."""

    @pytest.fixture
    def multi_endpoint_config(self):
        """Create config with multiple endpoints."""
        return TenantRoutingConfig(
            tenant_id="multi-endpoint",
            endpoints=[
                EndpointConfig(url="https://api1.example.com", weight=100, priority=1),
                EndpointConfig(url="https://api2.example.com", weight=50, priority=2),
                EndpointConfig(url="https://api3.example.com", weight=25, priority=3),
            ],
            load_balancing=LoadBalancingStrategy.ROUND_ROBIN,
        )

    @pytest.mark.asyncio
    async def test_round_robin_strategy(self, multi_endpoint_config):
        """Test round-robin load balancing."""
        router = TenantRouter(configs=[multi_endpoint_config])

        endpoints_used = []
        async with TenantContext(tenant_id="multi-endpoint"):
            for _ in range(6):
                decision = await router.route(tenant_id="multi-endpoint")
                endpoints_used.append(decision.target_endpoint)
                await router.complete_request(
                    tenant_id="multi-endpoint",
                    endpoint_url=decision.target_endpoint,
                    success=True,
                    latency_ms=100.0,
                )

        # Should cycle through endpoints
        assert "https://api1.example.com" in endpoints_used
        assert "https://api2.example.com" in endpoints_used
        assert "https://api3.example.com" in endpoints_used

    @pytest.mark.asyncio
    async def test_priority_strategy(self, multi_endpoint_config):
        """Test priority-based load balancing."""
        multi_endpoint_config.load_balancing = LoadBalancingStrategy.PRIORITY
        router = TenantRouter(configs=[multi_endpoint_config])

        async with TenantContext(tenant_id="multi-endpoint"):
            decision = await router.route(tenant_id="multi-endpoint")
            # Should always pick highest priority (priority=1)
            assert decision.target_endpoint == "https://api1.example.com"

    @pytest.mark.asyncio
    async def test_latency_strategy(self, multi_endpoint_config):
        """Test latency-based load balancing."""
        multi_endpoint_config.load_balancing = LoadBalancingStrategy.LATENCY
        router = TenantRouter(configs=[multi_endpoint_config])

        # Record different latencies
        await router._health_tracker.record_success("https://api1.example.com", latency_ms=500.0)
        await router._health_tracker.record_success("https://api2.example.com", latency_ms=100.0)
        await router._health_tracker.record_success("https://api3.example.com", latency_ms=300.0)

        async with TenantContext(tenant_id="multi-endpoint"):
            decision = await router.route(tenant_id="multi-endpoint")
            # Should pick lowest latency
            assert decision.target_endpoint == "https://api2.example.com"

    @pytest.mark.asyncio
    async def test_weighted_random_strategy(self, multi_endpoint_config):
        """Test weighted random load balancing distributes traffic."""
        multi_endpoint_config.load_balancing = LoadBalancingStrategy.WEIGHTED_RANDOM
        # Use high quotas to avoid rate limiting during test
        multi_endpoint_config.quotas = TenantQuotas(
            requests_per_minute=1000,
            requests_per_hour=10000,
            requests_per_day=100000,
            concurrent_requests=100,
        )
        router = TenantRouter(configs=[multi_endpoint_config])

        endpoint_counts = {
            "https://api1.example.com": 0,
            "https://api2.example.com": 0,
            "https://api3.example.com": 0,
        }

        async with TenantContext(tenant_id="multi-endpoint"):
            for _ in range(100):
                decision = await router.route(tenant_id="multi-endpoint")
                endpoint_counts[decision.target_endpoint] += 1
                await router.complete_request(
                    tenant_id="multi-endpoint",
                    endpoint_url=decision.target_endpoint,
                    success=True,
                    latency_ms=100.0,
                )

        # Higher weight should get more traffic (with some variance)
        assert (
            endpoint_counts["https://api1.example.com"]
            > endpoint_counts["https://api3.example.com"]
        )

    @pytest.mark.asyncio
    async def test_single_endpoint_optimization(self, basic_tenant_config):
        """Test that single endpoint skips load balancing logic."""
        router = TenantRouter(configs=[basic_tenant_config])

        async with TenantContext(tenant_id="test-tenant"):
            decision = await router.route(tenant_id="test-tenant")
            assert decision.target_endpoint == "https://api.example.com"


# =============================================================================
# Test TenantRoutingContext
# =============================================================================


class TestTenantRoutingContext:
    """Tests for TenantRoutingContext context manager."""

    @pytest.fixture
    def router_for_context(self):
        """Create a router for context testing."""
        return TenantRouter(
            configs=[
                TenantRoutingConfig(
                    tenant_id="context-tenant",
                    endpoints=[EndpointConfig(url="https://context.example.com")],
                )
            ]
        )

    @pytest.mark.asyncio
    async def test_context_manager_basic(self, router_for_context):
        """Test basic context manager usage."""
        async with TenantRoutingContext(router_for_context, "context-tenant") as ctx:
            decision = await ctx.route()
            assert decision.target_endpoint == "https://context.example.com"
            await ctx.complete(success=True, latency_ms=100.0)

    @pytest.mark.asyncio
    async def test_context_auto_complete_on_error(self, router_for_context):
        """Test automatic completion on exception."""
        try:
            async with TenantRoutingContext(router_for_context, "context-tenant") as ctx:
                await ctx.route()
                raise ValueError("Test error")
        except ValueError:
            pass

        # Check that request was marked complete (concurrent slot released)
        status = await router_for_context._quota_tracker.get_status(
            "context-tenant",
            TenantQuotas(),
        )
        assert status["concurrent_requests"].used == 0

    @pytest.mark.asyncio
    async def test_context_no_decision_no_auto_complete(self, router_for_context):
        """Test no auto-complete when no routing decision made."""
        async with TenantRoutingContext(router_for_context, "context-tenant") as ctx:
            # Don't call route()
            pass

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_context_multiple_complete_calls(self, router_for_context):
        """Test multiple complete calls are idempotent."""
        async with TenantRoutingContext(router_for_context, "context-tenant") as ctx:
            decision = await ctx.route()
            await ctx.complete(success=True, latency_ms=100.0)
            # Second call should be no-op
            await ctx.complete(success=True, latency_ms=200.0)


# =============================================================================
# Test Headers and Context Building
# =============================================================================


class TestHeadersAndContextBuilding:
    """Tests for header and context building functionality."""

    @pytest.mark.asyncio
    async def test_tenant_context_headers(self, tenant_router):
        """Test tenant context is properly built."""
        async with TenantContext(tenant_id="test-tenant"):
            decision = await tenant_router.route(tenant_id="test-tenant")

            assert "X-Tenant-ID" in decision.tenant_context
            assert "X-Aragora-Tenant" in decision.tenant_context
            assert "X-Isolation-Level" in decision.tenant_context
            assert "X-Tenant-Hash" in decision.tenant_context

    @pytest.mark.asyncio
    async def test_correlation_id_propagation(self, tenant_router):
        """Test correlation ID is propagated."""
        async with TenantContext(tenant_id="test-tenant"):
            decision = await tenant_router.route(
                tenant_id="test-tenant",
                request={"correlation_id": "req-12345"},
            )

            assert decision.tenant_context.get("X-Correlation-ID") == "req-12345"

    @pytest.mark.asyncio
    async def test_custom_context_headers(self):
        """Test custom context headers are included."""
        config = TenantRoutingConfig(
            tenant_id="custom-headers",
            endpoints=[EndpointConfig(url="https://api.example.com")],
            context_headers={"X-Custom-Header": "custom-value"},
        )
        router = TenantRouter(configs=[config])

        async with TenantContext(tenant_id="custom-headers"):
            decision = await router.route(tenant_id="custom-headers")

            assert decision.tenant_context.get("X-Custom-Header") == "custom-value"

    @pytest.mark.asyncio
    async def test_request_headers_include_endpoint_headers(self, tenant_router):
        """Test request headers include endpoint-specific headers."""
        async with TenantContext(tenant_id="test-tenant"):
            decision = await tenant_router.route(tenant_id="test-tenant")

            # Endpoint config has X-API-Key header
            assert "X-API-Key" in decision.headers

    @pytest.mark.asyncio
    async def test_aragora_router_header(self, tenant_router):
        """Test Aragora router metadata headers."""
        async with TenantContext(tenant_id="test-tenant"):
            decision = await tenant_router.route(tenant_id="test-tenant")

            assert decision.headers.get("X-Aragora-Router") == "TenantRouter"
            assert "X-Aragora-Timestamp" in decision.headers


# =============================================================================
# Test Audit Log Behavior
# =============================================================================


class TestAuditLogBehavior:
    """Tests for audit logging behavior."""

    @pytest.mark.asyncio
    async def test_audit_log_bounded(self):
        """Test audit log doesn't grow unbounded."""
        router = TenantRouter(
            configs=[
                TenantRoutingConfig(
                    tenant_id="audit-test",
                    endpoints=[EndpointConfig(url="https://api.example.com")],
                )
            ],
            enable_audit=True,
        )
        router._max_audit_entries = 10  # Low limit for testing

        async with TenantContext(tenant_id="audit-test"):
            for _ in range(20):
                await router.route(tenant_id="audit-test")
                await router._quota_tracker.release_concurrent("audit-test")

        assert len(router._audit_log) <= 10

    @pytest.mark.asyncio
    async def test_audit_log_filter_by_since(self, tenant_router):
        """Test filtering audit log by timestamp."""
        before = datetime.now(timezone.utc)

        async with TenantContext(tenant_id="test-tenant"):
            await tenant_router.route(tenant_id="test-tenant")

        after = datetime.now(timezone.utc)

        # Filter by before time should return entries
        entries = await tenant_router.get_audit_log(since=before)
        assert len(entries) > 0

        # Filter by after time should return no entries
        future = after + timedelta(hours=1)
        entries = await tenant_router.get_audit_log(since=future)
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_event_handler_exception_handling(self, tenant_router):
        """Test event handler exceptions don't break routing."""

        def bad_handler(entry):
            raise RuntimeError("Handler error")

        tenant_router.add_event_handler(bad_handler)

        # Routing should still work
        async with TenantContext(tenant_id="test-tenant"):
            decision = await tenant_router.route(tenant_id="test-tenant")
            assert decision is not None
