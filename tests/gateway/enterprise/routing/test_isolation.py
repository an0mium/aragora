"""Tests for gateway enterprise routing isolation."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.enterprise.routing.isolation import (
    CrossTenantAccessError,
    IsolationConfig,
    TenantAccessContext,
    TenantContextBuilder,
    TenantRoutingContextManager,
    get_tenant_from_context,
    validate_tenant_access,
)
from aragora.tenancy.isolation import IsolationLevel


# ---------------------------------------------------------------------------
# CrossTenantAccessError
# ---------------------------------------------------------------------------


class TestCrossTenantAccessError:
    """Tests for CrossTenantAccessError exception."""

    def test_basic_error(self) -> None:
        err = CrossTenantAccessError("tenant-a", "tenant-b")
        assert "tenant-a" in str(err)
        assert "tenant-b" in str(err)
        assert err.requesting_tenant == "tenant-a"
        assert err.target_tenant == "tenant-b"
        assert err.code == "CROSS_TENANT_ACCESS"

    def test_details_propagation(self) -> None:
        err = CrossTenantAccessError("a", "b", details={"reason": "blocked"})
        assert err.details["requesting_tenant"] == "a"
        assert err.details["target_tenant"] == "b"
        assert err.details["reason"] == "blocked"

    def test_is_exception(self) -> None:
        assert issubclass(CrossTenantAccessError, Exception)

    def test_no_details(self) -> None:
        err = CrossTenantAccessError("x", "y")
        assert "requesting_tenant" in err.details
        assert "target_tenant" in err.details


# ---------------------------------------------------------------------------
# IsolationConfig
# ---------------------------------------------------------------------------


class TestIsolationConfig:
    """Tests for IsolationConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = IsolationConfig()
        assert cfg.level == IsolationLevel.STRICT
        assert cfg.allow_cross_tenant_read is False
        assert cfg.allow_cross_tenant_write is False
        assert cfg.allowed_peer_tenants == set()
        assert cfg.audit_all_access is True

    def test_custom_values(self) -> None:
        cfg = IsolationConfig(
            level=IsolationLevel.SOFT,
            allow_cross_tenant_read=True,
            allowed_peer_tenants={"peer-1", "peer-2"},
        )
        assert cfg.allow_cross_tenant_read is True
        assert len(cfg.allowed_peer_tenants) == 2


# ---------------------------------------------------------------------------
# TenantAccessContext
# ---------------------------------------------------------------------------


class TestTenantAccessContext:
    """Tests for TenantAccessContext dataclass."""

    def test_basic_creation(self) -> None:
        ctx = TenantAccessContext(
            tenant_id="t1",
            isolation_level="strict",
            tenant_hash="abc123",
        )
        assert ctx.tenant_id == "t1"
        assert ctx.isolation_level == "strict"
        assert ctx.tenant_hash == "abc123"
        assert ctx.correlation_id is None
        assert ctx.headers == {}
        assert ctx.timestamp is not None


# ---------------------------------------------------------------------------
# TenantContextBuilder
# ---------------------------------------------------------------------------


class TestTenantContextBuilder:
    """Tests for TenantContextBuilder."""

    def test_build_context_basic(self) -> None:
        builder = TenantContextBuilder()
        ctx = builder.build_context("tenant-1", IsolationLevel.STRICT)
        assert ctx["X-Tenant-ID"] == "tenant-1"
        assert ctx["X-Aragora-Tenant"] == "tenant-1"
        assert ctx["X-Isolation-Level"] == IsolationLevel.STRICT.value
        assert "X-Tenant-Hash" in ctx

    def test_build_context_with_extra_headers(self) -> None:
        builder = TenantContextBuilder()
        ctx = builder.build_context(
            "tenant-1",
            IsolationLevel.STRICT,
            context_headers={"X-Custom": "value"},
        )
        assert ctx["X-Custom"] == "value"

    def test_build_context_with_correlation_id(self) -> None:
        builder = TenantContextBuilder()
        ctx = builder.build_context(
            "tenant-1",
            IsolationLevel.STRICT,
            request={"correlation_id": "corr-123"},
        )
        assert ctx["X-Correlation-ID"] == "corr-123"

    def test_build_context_no_correlation_id(self) -> None:
        builder = TenantContextBuilder()
        ctx = builder.build_context(
            "tenant-1",
            IsolationLevel.STRICT,
            request={"other": "data"},
        )
        assert "X-Correlation-ID" not in ctx

    def test_build_headers_basic(self) -> None:
        builder = TenantContextBuilder()
        headers = builder.build_headers("tenant-1")
        assert headers["X-Tenant-ID"] == "tenant-1"
        assert headers["X-Aragora-Tenant"] == "tenant-1"
        assert headers["X-Aragora-Router"] == "TenantRouter"
        assert "X-Aragora-Timestamp" in headers

    def test_build_headers_with_endpoint_headers(self) -> None:
        builder = TenantContextBuilder()
        headers = builder.build_headers(
            "tenant-1",
            endpoint_headers={"Authorization": "Bearer xyz"},
        )
        assert headers["Authorization"] == "Bearer xyz"

    def test_build_headers_with_context_headers(self) -> None:
        builder = TenantContextBuilder()
        headers = builder.build_headers(
            "tenant-1",
            context_headers={"X-Env": "prod"},
        )
        assert headers["X-Env"] == "prod"

    def test_compute_tenant_hash(self) -> None:
        expected = hashlib.sha256(b"tenant-1").hexdigest()[:16]
        assert TenantContextBuilder.compute_tenant_hash("tenant-1") == expected

    def test_compute_tenant_hash_deterministic(self) -> None:
        h1 = TenantContextBuilder.compute_tenant_hash("abc")
        h2 = TenantContextBuilder.compute_tenant_hash("abc")
        assert h1 == h2

    def test_compute_tenant_hash_different_tenants(self) -> None:
        h1 = TenantContextBuilder.compute_tenant_hash("tenant-a")
        h2 = TenantContextBuilder.compute_tenant_hash("tenant-b")
        assert h1 != h2


# ---------------------------------------------------------------------------
# validate_tenant_access
# ---------------------------------------------------------------------------


class TestValidateTenantAccess:
    """Tests for validate_tenant_access function."""

    def test_none_requesting_tenant_allowed(self) -> None:
        # No context tenant -> allow
        validate_tenant_access(None, "target")

    def test_same_tenant_allowed(self) -> None:
        validate_tenant_access("tenant-1", "tenant-1")

    def test_cross_tenant_denied_no_config(self) -> None:
        with pytest.raises(CrossTenantAccessError):
            validate_tenant_access("tenant-a", "tenant-b")

    def test_cross_tenant_denied_strict_config(self) -> None:
        cfg = IsolationConfig(allow_cross_tenant_read=False)
        with pytest.raises(CrossTenantAccessError):
            validate_tenant_access("tenant-a", "tenant-b", config=cfg)

    def test_cross_tenant_allowed_by_read_flag(self) -> None:
        cfg = IsolationConfig(allow_cross_tenant_read=True)
        validate_tenant_access("tenant-a", "tenant-b", config=cfg)

    def test_cross_tenant_allowed_by_peer_list(self) -> None:
        cfg = IsolationConfig(allowed_peer_tenants={"tenant-b"})
        validate_tenant_access("tenant-a", "tenant-b", config=cfg)

    def test_cross_tenant_denied_peer_not_in_list(self) -> None:
        cfg = IsolationConfig(allowed_peer_tenants={"tenant-c"})
        with pytest.raises(CrossTenantAccessError):
            validate_tenant_access("tenant-a", "tenant-b", config=cfg)


# ---------------------------------------------------------------------------
# get_tenant_from_context
# ---------------------------------------------------------------------------


class TestGetTenantFromContext:
    """Tests for get_tenant_from_context."""

    def test_returns_current_tenant(self) -> None:
        with patch(
            "aragora.gateway.enterprise.routing.isolation.get_current_tenant_id",
            return_value="tenant-xyz",
        ):
            assert get_tenant_from_context() == "tenant-xyz"

    def test_returns_none_when_not_set(self) -> None:
        with patch(
            "aragora.gateway.enterprise.routing.isolation.get_current_tenant_id",
            return_value=None,
        ):
            assert get_tenant_from_context() is None


# ---------------------------------------------------------------------------
# TenantRoutingContextManager
# ---------------------------------------------------------------------------


class TestTenantRoutingContextManager:
    """Tests for TenantRoutingContextManager."""

    def test_init(self) -> None:
        router = MagicMock()
        mgr = TenantRoutingContextManager(router, "tenant-1")
        assert mgr._tenant_id == "tenant-1"
        assert mgr._router is router
        assert mgr._completed is False
        assert mgr._decision is None

    @pytest.mark.asyncio
    async def test_enter_exit_sets_context(self) -> None:
        router = MagicMock()
        mock_tc = MagicMock()
        mock_tc.__aenter__ = AsyncMock(return_value=mock_tc)
        mock_tc.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.gateway.enterprise.routing.isolation.TenantContext",
            return_value=mock_tc,
        ):
            mgr = TenantRoutingContextManager(router, "tenant-1")
            async with mgr:
                pass

        mock_tc.__aenter__.assert_called_once()
        mock_tc.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_delegates_to_router(self) -> None:
        decision = MagicMock(target_endpoint="https://example.com")
        router = MagicMock()
        router.route = AsyncMock(return_value=decision)

        mock_tc = MagicMock()
        mock_tc.__aenter__ = AsyncMock(return_value=mock_tc)
        mock_tc.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.gateway.enterprise.routing.isolation.TenantContext",
            return_value=mock_tc,
        ):
            mgr = TenantRoutingContextManager(router, "tenant-1")
            async with mgr:
                result = await mgr.route(request={"data": 1})

        assert result is decision
        router.route.assert_called_once_with(
            tenant_id="tenant-1",
            request={"data": 1},
            operation=None,
            bytes_size=0,
        )

    @pytest.mark.asyncio
    async def test_complete_delegates_to_router(self) -> None:
        decision = MagicMock(target_endpoint="https://example.com")
        router = MagicMock()
        router.route = AsyncMock(return_value=decision)
        router.complete_request = AsyncMock()

        mock_tc = MagicMock()
        mock_tc.__aenter__ = AsyncMock(return_value=mock_tc)
        mock_tc.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.gateway.enterprise.routing.isolation.TenantContext",
            return_value=mock_tc,
        ):
            mgr = TenantRoutingContextManager(router, "tenant-1")
            async with mgr:
                await mgr.route()
                await mgr.complete(success=True, latency_ms=42.0)

        router.complete_request.assert_called_once_with(
            tenant_id="tenant-1",
            endpoint_url="https://example.com",
            success=True,
            latency_ms=42.0,
            error=None,
        )

    @pytest.mark.asyncio
    async def test_complete_only_once(self) -> None:
        decision = MagicMock(target_endpoint="https://example.com")
        router = MagicMock()
        router.route = AsyncMock(return_value=decision)
        router.complete_request = AsyncMock()

        mock_tc = MagicMock()
        mock_tc.__aenter__ = AsyncMock(return_value=mock_tc)
        mock_tc.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.gateway.enterprise.routing.isolation.TenantContext",
            return_value=mock_tc,
        ):
            mgr = TenantRoutingContextManager(router, "tenant-1")
            async with mgr:
                await mgr.route()
                await mgr.complete(success=True, latency_ms=10.0)
                await mgr.complete(success=True, latency_ms=20.0)

        assert router.complete_request.call_count == 1

    @pytest.mark.asyncio
    async def test_auto_complete_on_error(self) -> None:
        decision = MagicMock(target_endpoint="https://example.com")
        router = MagicMock()
        router.route = AsyncMock(return_value=decision)
        router.complete_request = AsyncMock()

        mock_tc = MagicMock()
        mock_tc.__aenter__ = AsyncMock(return_value=mock_tc)
        mock_tc.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.gateway.enterprise.routing.isolation.TenantContext",
            return_value=mock_tc,
        ):
            mgr = TenantRoutingContextManager(router, "tenant-1")
            with pytest.raises(ValueError, match="boom"):
                async with mgr:
                    await mgr.route()
                    raise ValueError("boom")

        router.complete_request.assert_called_once()
        call_kwargs = router.complete_request.call_args[1]
        assert call_kwargs["success"] is False
