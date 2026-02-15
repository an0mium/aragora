"""Tests for gateway enterprise routing router."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.enterprise.routing.router import (
    EndpointConfig,
    EndpointHealth,
    EndpointHealthTracker,
    EndpointStatus,
    LoadBalancingStrategy,
    NoAvailableEndpointError,
    QuotaExceededError,
    RoutingAuditEntry,
    RoutingDecision,
    RoutingEventType,
    TenantNotFoundError,
    TenantRouter,
    TenantRoutingConfig,
    TenantRoutingError,
)
from aragora.gateway.enterprise.routing.quotas import TenantQuotas


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------


class TestTenantRoutingError:
    """Tests for routing exception hierarchy."""

    def test_base_error(self) -> None:
        err = TenantRoutingError("msg", tenant_id="t1", code="TEST")
        assert str(err) == "msg"
        assert err.tenant_id == "t1"
        assert err.code == "TEST"
        assert err.details == {}

    def test_tenant_not_found(self) -> None:
        err = TenantNotFoundError("t1")
        assert "t1" in str(err)
        assert err.tenant_id == "t1"
        assert err.code == "TENANT_NOT_FOUND"

    def test_no_available_endpoint(self) -> None:
        err = NoAvailableEndpointError("t1")
        assert "t1" in str(err)
        assert err.code == "NO_AVAILABLE_ENDPOINT"

    def test_quota_exceeded(self) -> None:
        err = QuotaExceededError("t1", "requests_per_minute", 100, 105)
        assert "t1" in str(err)
        assert err.quota_type == "requests_per_minute"
        assert err.limit == 100
        assert err.current == 105
        assert err.retry_after is None

    def test_quota_exceeded_with_retry(self) -> None:
        err = QuotaExceededError("t1", "rpm", 60, 61, retry_after=30)
        assert err.retry_after == 30


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for routing enums."""

    def test_load_balancing_strategies(self) -> None:
        assert LoadBalancingStrategy.ROUND_ROBIN.value == "round_robin"
        assert LoadBalancingStrategy.WEIGHTED_RANDOM.value == "weighted_random"
        assert LoadBalancingStrategy.PRIORITY.value == "priority"
        assert LoadBalancingStrategy.LATENCY.value == "latency"
        assert LoadBalancingStrategy.LEAST_CONNECTIONS.value == "least_connections"

    def test_endpoint_status(self) -> None:
        assert EndpointStatus.HEALTHY.value == "healthy"
        assert EndpointStatus.DEGRADED.value == "degraded"
        assert EndpointStatus.UNHEALTHY.value == "unhealthy"
        assert EndpointStatus.UNKNOWN.value == "unknown"

    def test_routing_event_types(self) -> None:
        assert RoutingEventType.ROUTE_SUCCESS.value == "route_success"
        assert RoutingEventType.QUOTA_EXCEEDED.value == "quota_exceeded"
        assert RoutingEventType.CROSS_TENANT_BLOCKED.value == "cross_tenant_blocked"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestEndpointConfig:
    """Tests for EndpointConfig dataclass."""

    def test_defaults(self) -> None:
        ep = EndpointConfig(url="https://api.example.com")
        assert ep.url == "https://api.example.com"
        assert ep.weight == 100
        assert ep.priority == 1
        assert ep.timeout == 30.0
        assert ep.max_retries == 3
        assert ep.health_check_path == "/health"

    def test_custom_values(self) -> None:
        ep = EndpointConfig(url="https://x.com", weight=50, priority=2)
        assert ep.weight == 50
        assert ep.priority == 2


class TestEndpointHealth:
    """Tests for EndpointHealth dataclass."""

    def test_defaults(self) -> None:
        h = EndpointHealth(endpoint_url="https://api.example.com")
        assert h.status == EndpointStatus.UNKNOWN
        assert h.consecutive_failures == 0
        assert h.latency_ms == 0.0


class TestTenantRoutingConfig:
    """Tests for TenantRoutingConfig dataclass."""

    def test_valid_config(self) -> None:
        cfg = TenantRoutingConfig(
            tenant_id="acme",
            endpoints=[EndpointConfig(url="https://acme.example.com")],
        )
        assert cfg.tenant_id == "acme"
        assert len(cfg.endpoints) == 1
        assert cfg.load_balancing == LoadBalancingStrategy.WEIGHTED_RANDOM

    def test_empty_tenant_id_raises(self) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            TenantRoutingConfig(
                tenant_id="",
                endpoints=[EndpointConfig(url="https://x.com")],
            )

    def test_no_endpoints_raises(self) -> None:
        with pytest.raises(ValueError, match="endpoint"):
            TenantRoutingConfig(tenant_id="t1")

    def test_fallback_only_is_valid(self) -> None:
        cfg = TenantRoutingConfig(
            tenant_id="t1",
            fallback_endpoints=[EndpointConfig(url="https://fallback.com")],
        )
        assert len(cfg.endpoints) == 0
        assert len(cfg.fallback_endpoints) == 1


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_to_dict(self) -> None:
        d = RoutingDecision(
            target_endpoint="https://api.example.com",
            tenant_context={"X-Tenant-ID": "t1"},
            decision_time_ms=1.5,
        )
        result = d.to_dict()
        assert result["target_endpoint"] == "https://api.example.com"
        assert result["decision_time_ms"] == 1.5
        assert result["used_fallback"] is False


# ---------------------------------------------------------------------------
# EndpointHealthTracker tests
# ---------------------------------------------------------------------------


class TestEndpointHealthTracker:
    """Tests for EndpointHealthTracker."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint_is_available(self) -> None:
        tracker = EndpointHealthTracker()
        assert await tracker.is_available("https://new.example.com") is True

    @pytest.mark.asyncio
    async def test_record_success_marks_healthy(self) -> None:
        tracker = EndpointHealthTracker()
        await tracker.record_success("https://api.example.com", 50.0)
        health = await tracker.get_health("https://api.example.com")
        assert health.status == EndpointStatus.HEALTHY
        assert health.consecutive_failures == 0
        assert health.latency_ms == 50.0

    @pytest.mark.asyncio
    async def test_record_success_ema_latency(self) -> None:
        tracker = EndpointHealthTracker()
        await tracker.record_success("https://api.example.com", 100.0)
        await tracker.record_success("https://api.example.com", 50.0)
        health = await tracker.get_health("https://api.example.com")
        # EMA: 0.7 * 100 + 0.3 * 50 = 85
        assert abs(health.latency_ms - 85.0) < 0.01

    @pytest.mark.asyncio
    async def test_record_failure_increments(self) -> None:
        tracker = EndpointHealthTracker(unhealthy_threshold=5)
        status = await tracker.record_failure("https://api.example.com", "err")
        assert status == EndpointStatus.HEALTHY  # 1 failure < threshold//2 (2)

    @pytest.mark.asyncio
    async def test_record_failure_degraded(self) -> None:
        tracker = EndpointHealthTracker(unhealthy_threshold=4)
        # threshold // 2 = 2 failures for degraded
        await tracker.record_failure("https://api.example.com", "err")
        status = await tracker.record_failure("https://api.example.com", "err")
        assert status == EndpointStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_record_failure_unhealthy(self) -> None:
        tracker = EndpointHealthTracker(unhealthy_threshold=3)
        await tracker.record_failure("https://api.example.com", "err")
        await tracker.record_failure("https://api.example.com", "err")
        status = await tracker.record_failure("https://api.example.com", "err")
        assert status == EndpointStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_endpoint_not_available(self) -> None:
        tracker = EndpointHealthTracker(unhealthy_threshold=1, recovery_timeout=60.0)
        await tracker.record_failure("https://api.example.com", "err")
        assert await tracker.is_available("https://api.example.com") is False

    @pytest.mark.asyncio
    async def test_unhealthy_endpoint_recovers_after_timeout(self) -> None:
        tracker = EndpointHealthTracker(unhealthy_threshold=1, recovery_timeout=0.0)
        await tracker.record_failure("https://api.example.com", "err")
        # recovery_timeout=0.0 means it should immediately be available again
        assert await tracker.is_available("https://api.example.com") is True

    @pytest.mark.asyncio
    async def test_success_after_failure_resets(self) -> None:
        tracker = EndpointHealthTracker(unhealthy_threshold=2)
        await tracker.record_failure("https://api.example.com", "err")
        await tracker.record_success("https://api.example.com", 10.0)
        health = await tracker.get_health("https://api.example.com")
        assert health.status == EndpointStatus.HEALTHY
        assert health.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_get_all_health(self) -> None:
        tracker = EndpointHealthTracker()
        await tracker.record_success("https://a.com", 10.0)
        await tracker.record_success("https://b.com", 20.0)
        all_h = await tracker.get_all_health()
        assert "https://a.com" in all_h
        assert "https://b.com" in all_h

    @pytest.mark.asyncio
    async def test_get_health_unknown(self) -> None:
        tracker = EndpointHealthTracker()
        health = await tracker.get_health("https://unknown.com")
        assert health.status == EndpointStatus.UNKNOWN


# ---------------------------------------------------------------------------
# TenantRouter tests
# ---------------------------------------------------------------------------


def _make_config(
    tenant_id: str = "t1",
    endpoints: list[str] | None = None,
    fallback_endpoints: list[str] | None = None,
    **kwargs: Any,
) -> TenantRoutingConfig:
    """Helper to build a TenantRoutingConfig."""
    eps = [EndpointConfig(url=u) for u in (endpoints or ["https://primary.example.com"])]
    fb = [EndpointConfig(url=u) for u in (fallback_endpoints or [])]
    return TenantRoutingConfig(
        tenant_id=tenant_id,
        endpoints=eps,
        fallback_endpoints=fb,
        **kwargs,
    )


class TestTenantRouterInit:
    """Tests for TenantRouter initialization."""

    def test_init_no_configs(self) -> None:
        router = TenantRouter()
        assert len(router._configs) == 0

    def test_init_with_configs(self) -> None:
        cfg = _make_config("acme")
        router = TenantRouter(configs=[cfg])
        assert "acme" in router._configs

    def test_init_multiple_configs(self) -> None:
        configs = [_make_config("t1"), _make_config("t2")]
        router = TenantRouter(configs=configs)
        assert len(router._configs) == 2


class TestTenantRouterConfigManagement:
    """Tests for config add/remove/get/list."""

    @pytest.mark.asyncio
    async def test_add_tenant_config(self) -> None:
        router = TenantRouter()
        await router.add_tenant_config(_make_config("new"))
        assert "new" in router._configs

    @pytest.mark.asyncio
    async def test_remove_tenant_config(self) -> None:
        router = TenantRouter(configs=[_make_config("t1")])
        result = await router.remove_tenant_config("t1")
        assert result is True
        assert "t1" not in router._configs

    @pytest.mark.asyncio
    async def test_remove_nonexistent_config(self) -> None:
        router = TenantRouter()
        result = await router.remove_tenant_config("missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_tenant_config(self) -> None:
        cfg = _make_config("t1")
        router = TenantRouter(configs=[cfg])
        result = await router.get_tenant_config("t1")
        assert result is cfg

    @pytest.mark.asyncio
    async def test_get_tenant_config_missing(self) -> None:
        router = TenantRouter()
        result = await router.get_tenant_config("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_tenant_configs(self) -> None:
        configs = [_make_config("a"), _make_config("b")]
        router = TenantRouter(configs=configs)
        result = await router.list_tenant_configs()
        assert len(result) == 2


class TestTenantRouterRouting:
    """Tests for route() method."""

    @pytest.mark.asyncio
    async def test_route_success(self) -> None:
        cfg = _make_config("t1")
        router = TenantRouter(configs=[cfg])

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            decision = await router.route(tenant_id="t1")

        assert decision.target_endpoint == "https://primary.example.com"
        assert decision.used_fallback is False
        assert decision.tenant_context["X-Tenant-ID"] == "t1"

    @pytest.mark.asyncio
    async def test_route_tenant_not_found(self) -> None:
        router = TenantRouter()
        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="missing",
        ):
            with pytest.raises(TenantNotFoundError):
                await router.route(tenant_id="missing")

    @pytest.mark.asyncio
    async def test_route_no_tenant_context(self) -> None:
        from aragora.tenancy.context import TenantNotSetError

        router = TenantRouter()
        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value=None,
        ):
            with pytest.raises(TenantNotSetError):
                await router.route()

    @pytest.mark.asyncio
    async def test_route_cross_tenant_blocked(self) -> None:
        from aragora.gateway.enterprise.routing.isolation import CrossTenantAccessError

        cfg = _make_config("t1")
        router = TenantRouter(configs=[cfg])

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="other-tenant",
        ):
            with pytest.raises(CrossTenantAccessError):
                await router.route(tenant_id="t1")

    @pytest.mark.asyncio
    async def test_route_operation_not_allowed(self) -> None:
        cfg = _make_config("t1", allowed_operations={"read", "write"})
        router = TenantRouter(configs=[cfg])

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            with pytest.raises(TenantRoutingError, match="not allowed"):
                await router.route(tenant_id="t1", operation="delete")

    @pytest.mark.asyncio
    async def test_route_allowed_operation(self) -> None:
        cfg = _make_config("t1", allowed_operations={"read"})
        router = TenantRouter(configs=[cfg])

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            decision = await router.route(tenant_id="t1", operation="read")
        assert decision.target_endpoint == "https://primary.example.com"

    @pytest.mark.asyncio
    async def test_route_uses_fallback_when_primary_unhealthy(self) -> None:
        cfg = _make_config(
            "t1",
            endpoints=["https://primary.example.com"],
            fallback_endpoints=["https://fallback.example.com"],
        )
        router = TenantRouter(configs=[cfg])

        # Mark primary as unhealthy
        for _ in range(3):
            await router._health_tracker.record_failure("https://primary.example.com", "err")

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            decision = await router.route(tenant_id="t1")

        assert decision.target_endpoint == "https://fallback.example.com"
        assert decision.used_fallback is True

    @pytest.mark.asyncio
    async def test_route_no_available_endpoints(self) -> None:
        cfg = _make_config("t1", endpoints=["https://primary.example.com"])
        router = TenantRouter(configs=[cfg])

        # Mark primary as unhealthy
        for _ in range(3):
            await router._health_tracker.record_failure("https://primary.example.com", "err")

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            with pytest.raises(NoAvailableEndpointError):
                await router.route(tenant_id="t1")


class TestTenantRouterCompleteRequest:
    """Tests for complete_request."""

    @pytest.mark.asyncio
    async def test_complete_success(self) -> None:
        router = TenantRouter()
        await router.complete_request("t1", "https://ep.com", success=True, latency_ms=50.0)
        health = await router._health_tracker.get_health("https://ep.com")
        assert health.status == EndpointStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_complete_failure(self) -> None:
        router = TenantRouter()
        await router.complete_request(
            "t1", "https://ep.com", success=False, latency_ms=0, error="timeout"
        )
        health = await router._health_tracker.get_health("https://ep.com")
        assert health.consecutive_failures == 1


class TestTenantRouterLoadBalancing:
    """Tests for load balancing strategies."""

    @pytest.mark.asyncio
    async def test_round_robin(self) -> None:
        cfg = _make_config(
            "t1",
            endpoints=["https://a.com", "https://b.com"],
            load_balancing=LoadBalancingStrategy.ROUND_ROBIN,
        )
        router = TenantRouter(configs=[cfg])

        decisions = []
        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            for _ in range(4):
                d = await router.route(tenant_id="t1")
                decisions.append(d.target_endpoint)
                await router._quota_tracker.release_concurrent("t1")

        assert decisions[0] == "https://a.com"
        assert decisions[1] == "https://b.com"
        assert decisions[2] == "https://a.com"
        assert decisions[3] == "https://b.com"

    @pytest.mark.asyncio
    async def test_priority_selects_lowest(self) -> None:
        cfg = TenantRoutingConfig(
            tenant_id="t1",
            endpoints=[
                EndpointConfig(url="https://low-priority.com", priority=5),
                EndpointConfig(url="https://high-priority.com", priority=1),
            ],
            load_balancing=LoadBalancingStrategy.PRIORITY,
        )
        router = TenantRouter(configs=[cfg])

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            decision = await router.route(tenant_id="t1")

        assert decision.target_endpoint == "https://high-priority.com"

    @pytest.mark.asyncio
    async def test_latency_selects_lowest_latency(self) -> None:
        cfg = _make_config(
            "t1",
            endpoints=["https://slow.com", "https://fast.com"],
            load_balancing=LoadBalancingStrategy.LATENCY,
        )
        router = TenantRouter(configs=[cfg])

        # Record latencies
        await router._health_tracker.record_success("https://slow.com", 200.0)
        await router._health_tracker.record_success("https://fast.com", 10.0)

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            decision = await router.route(tenant_id="t1")

        assert decision.target_endpoint == "https://fast.com"

    @pytest.mark.asyncio
    async def test_single_endpoint_always_selected(self) -> None:
        cfg = _make_config("t1", endpoints=["https://only.com"])
        router = TenantRouter(configs=[cfg])

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            decision = await router.route(tenant_id="t1")

        assert decision.target_endpoint == "https://only.com"


class TestTenantRouterQuotas:
    """Tests for quota management in router."""

    @pytest.mark.asyncio
    async def test_get_quota_status(self) -> None:
        cfg = _make_config("t1")
        router = TenantRouter(configs=[cfg])

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            statuses = await router.get_quota_status("t1")
        assert "requests_per_minute" in statuses

    @pytest.mark.asyncio
    async def test_get_quota_status_not_found(self) -> None:
        router = TenantRouter()
        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="missing",
        ):
            with pytest.raises(TenantNotFoundError):
                await router.get_quota_status("missing")

    @pytest.mark.asyncio
    async def test_reset_quota(self) -> None:
        cfg = _make_config("t1", quotas=TenantQuotas(requests_per_minute=2))
        router = TenantRouter(configs=[cfg])

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            await router.route(tenant_id="t1")
            await router._quota_tracker.release_concurrent("t1")
            await router.route(tenant_id="t1")
            await router._quota_tracker.release_concurrent("t1")

        await router.reset_quota("t1")

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            decision = await router.route(tenant_id="t1")
        assert decision is not None


class TestTenantRouterHealth:
    """Tests for health monitoring."""

    @pytest.mark.asyncio
    async def test_get_endpoint_health(self) -> None:
        router = TenantRouter()
        health = await router.get_endpoint_health("https://unknown.com")
        assert health.status == EndpointStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_get_all_endpoint_health(self) -> None:
        router = TenantRouter()
        await router._health_tracker.record_success("https://a.com", 10.0)
        result = await router.get_all_endpoint_health()
        assert "https://a.com" in result

    @pytest.mark.asyncio
    async def test_get_tenant_health(self) -> None:
        cfg = _make_config(
            "t1",
            endpoints=["https://a.com"],
            fallback_endpoints=["https://b.com"],
        )
        router = TenantRouter(configs=[cfg])
        result = await router.get_tenant_health("t1")
        assert result["tenant_id"] == "t1"
        assert result["total_endpoints"] == 2
        assert len(result["primary_endpoints"]) == 1
        assert len(result["fallback_endpoints"]) == 1

    @pytest.mark.asyncio
    async def test_get_tenant_health_not_found(self) -> None:
        router = TenantRouter()
        with pytest.raises(TenantNotFoundError):
            await router.get_tenant_health("missing")


class TestTenantRouterAudit:
    """Tests for audit logging."""

    @pytest.mark.asyncio
    async def test_audit_enabled(self) -> None:
        cfg = _make_config("t1")
        router = TenantRouter(configs=[cfg], enable_audit=True)

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            await router.route(tenant_id="t1")

        log = await router.get_audit_log()
        assert len(log) >= 1
        assert log[-1]["event_type"] == RoutingEventType.ROUTE_SUCCESS.value

    @pytest.mark.asyncio
    async def test_audit_disabled(self) -> None:
        cfg = _make_config("t1")
        router = TenantRouter(configs=[cfg], enable_audit=False)

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            await router.route(tenant_id="t1")

        log = await router.get_audit_log()
        assert len(log) == 0

    @pytest.mark.asyncio
    async def test_audit_filter_by_tenant(self) -> None:
        configs = [_make_config("t1"), _make_config("t2")]
        router = TenantRouter(configs=configs, enable_audit=True)

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            side_effect=lambda: "t1",
        ):
            await router.route(tenant_id="t1")
            await router._quota_tracker.release_concurrent("t1")

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            side_effect=lambda: "t2",
        ):
            await router.route(tenant_id="t2")

        log = await router.get_audit_log(tenant_id="t1")
        assert all(e["tenant_id"] == "t1" for e in log)

    @pytest.mark.asyncio
    async def test_audit_filter_by_event_type(self) -> None:
        cfg = _make_config("t1")
        router = TenantRouter(configs=[cfg], enable_audit=True)

        with patch(
            "aragora.gateway.enterprise.routing.router.get_current_tenant_id",
            return_value="t1",
        ):
            await router.route(tenant_id="t1")

        log = await router.get_audit_log(event_type=RoutingEventType.ROUTE_SUCCESS)
        assert all(e["event_type"] == "route_success" for e in log)

    @pytest.mark.asyncio
    async def test_audit_bounded(self) -> None:
        router = TenantRouter(enable_audit=True)
        router._max_audit_entries = 5
        for i in range(10):
            await router._log_audit(
                RoutingAuditEntry(
                    timestamp=datetime.now(timezone.utc),
                    tenant_id="t1",
                    event_type=RoutingEventType.ROUTE_SUCCESS,
                )
            )
        assert len(router._audit_log) <= 5

    def test_add_event_handler(self) -> None:
        router = TenantRouter()
        handler = MagicMock()
        router.add_event_handler(handler)
        assert handler in router._event_handlers

    def test_remove_event_handler(self) -> None:
        router = TenantRouter()
        handler = MagicMock()
        router.add_event_handler(handler)
        router.remove_event_handler(handler)
        assert handler not in router._event_handlers

    @pytest.mark.asyncio
    async def test_event_handler_called(self) -> None:
        handler = MagicMock()
        router = TenantRouter(event_handlers=[handler], enable_audit=True)
        await router._log_audit(
            RoutingAuditEntry(
                timestamp=datetime.now(timezone.utc),
                tenant_id="t1",
                event_type=RoutingEventType.ROUTE_SUCCESS,
            )
        )
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_handler_error_does_not_propagate(self) -> None:
        handler = MagicMock(side_effect=RuntimeError("handler broke"))
        router = TenantRouter(event_handlers=[handler], enable_audit=True)
        # Should not raise
        await router._log_audit(
            RoutingAuditEntry(
                timestamp=datetime.now(timezone.utc),
                tenant_id="t1",
                event_type=RoutingEventType.ROUTE_SUCCESS,
            )
        )


class TestTenantRouterStats:
    """Tests for get_stats."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self) -> None:
        router = TenantRouter()
        stats = await router.get_stats()
        assert stats["tenant_count"] == 0
        assert stats["total_endpoints"] == 0
        assert stats["audit_entries"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_tenants(self) -> None:
        configs = [
            _make_config("t1", endpoints=["https://a.com", "https://b.com"]),
            _make_config("t2", endpoints=["https://c.com"]),
        ]
        router = TenantRouter(configs=configs)
        stats = await router.get_stats()
        assert stats["tenant_count"] == 2
        assert stats["total_endpoints"] == 3
