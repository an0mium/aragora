"""Tests for TenantContext and context management."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from aragora.tenancy.context import (
    TenantContext,
    TenantContextInfo,
    TenantMismatchError,
    TenantNotSetError,
    get_current_tenant,
    get_current_tenant_id,
    require_tenant,
    require_tenant_id,
)
from aragora.tenancy.tenant import Tenant, TenantStatus, TenantTier


class TestTenantContext:
    """Tests for TenantContext class."""

    def test_context_with_tenant_id(self):
        """Test context manager with tenant ID."""
        with TenantContext(tenant_id="test-tenant"):
            assert get_current_tenant_id() == "test-tenant"
        assert get_current_tenant_id() is None

    def test_context_with_tenant_object(self):
        """Test context manager with full tenant object."""
        tenant = Tenant(
            id="test-123",
            name="Test Org",
            slug="test-org",
        )
        with TenantContext(tenant=tenant):
            assert get_current_tenant_id() == "test-123"
            assert get_current_tenant() is tenant
        assert get_current_tenant() is None

    def test_tenant_object_takes_precedence(self):
        """Test that tenant object takes precedence over tenant_id."""
        tenant = Tenant(id="tenant-from-obj", name="Obj Tenant", slug="obj")
        with TenantContext(tenant_id="tenant-from-id", tenant=tenant):
            assert get_current_tenant_id() == "tenant-from-obj"

    def test_nested_contexts(self):
        """Test nested tenant contexts."""
        with TenantContext(tenant_id="outer"):
            assert get_current_tenant_id() == "outer"
            with TenantContext(tenant_id="inner"):
                assert get_current_tenant_id() == "inner"
            assert get_current_tenant_id() == "outer"
        assert get_current_tenant_id() is None

    def test_context_depth_tracking(self):
        """Test that context depth is tracked correctly."""
        initial_depth = TenantContext._depth
        with TenantContext(tenant_id="level1"):
            assert TenantContext._depth == initial_depth + 1
            with TenantContext(tenant_id="level2"):
                assert TenantContext._depth == initial_depth + 2
        assert TenantContext._depth == initial_depth

    def test_context_properties(self):
        """Test TenantContext properties."""
        tenant = Tenant(id="prop-test", name="Prop Test", slug="prop")
        ctx = TenantContext(tenant=tenant)
        assert ctx.tenant_id == "prop-test"
        assert ctx.tenant is tenant

    def test_context_with_none_tenant_id(self):
        """Test context with None tenant_id."""
        with TenantContext(tenant_id=None):
            assert get_current_tenant_id() is None


class TestAsyncTenantContext:
    """Tests for async tenant context operations."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager."""
        async with TenantContext(tenant_id="async-tenant"):
            assert get_current_tenant_id() == "async-tenant"
        assert get_current_tenant_id() is None

    @pytest.mark.asyncio
    async def test_async_nested_contexts(self):
        """Test nested async contexts."""
        async with TenantContext(tenant_id="async-outer"):
            assert get_current_tenant_id() == "async-outer"
            async with TenantContext(tenant_id="async-inner"):
                assert get_current_tenant_id() == "async-inner"
            assert get_current_tenant_id() == "async-outer"

    @pytest.mark.asyncio
    async def test_context_preserved_across_await(self):
        """Test that tenant context is preserved across await points."""
        async with TenantContext(tenant_id="await-test"):
            assert get_current_tenant_id() == "await-test"
            await asyncio.sleep(0.01)
            assert get_current_tenant_id() == "await-test"


class TestGetCurrentTenant:
    """Tests for get_current_tenant and get_current_tenant_id."""

    def test_get_tenant_id_without_context(self):
        """Test getting tenant ID when no context is set."""
        assert get_current_tenant_id() is None

    def test_get_tenant_without_context(self):
        """Test getting tenant when no context is set."""
        assert get_current_tenant() is None

    def test_get_tenant_id_with_context(self):
        """Test getting tenant ID within context."""
        with TenantContext(tenant_id="ctx-test"):
            assert get_current_tenant_id() == "ctx-test"

    def test_get_tenant_with_context(self):
        """Test getting full tenant within context."""
        tenant = Tenant(id="full-tenant", name="Full", slug="full")
        with TenantContext(tenant=tenant):
            result = get_current_tenant()
            assert result is tenant
            assert result.name == "Full"


class TestRequireTenant:
    """Tests for require_tenant and require_tenant_id."""

    def test_require_tenant_id_raises_without_context(self):
        """Test that require_tenant_id raises when no context."""
        with pytest.raises(TenantNotSetError) as exc_info:
            require_tenant_id()
        assert "No tenant ID set" in str(exc_info.value)

    def test_require_tenant_raises_without_context(self):
        """Test that require_tenant raises when no context."""
        with pytest.raises(TenantNotSetError) as exc_info:
            require_tenant()
        assert "No tenant set" in str(exc_info.value)

    def test_require_tenant_id_with_context(self):
        """Test require_tenant_id within context."""
        with TenantContext(tenant_id="required-id"):
            result = require_tenant_id()
            assert result == "required-id"

    def test_require_tenant_with_context(self):
        """Test require_tenant within context."""
        tenant = Tenant(id="required-tenant", name="Required", slug="req")
        with TenantContext(tenant=tenant):
            result = require_tenant()
            assert result is tenant

    def test_require_tenant_with_only_id_raises(self):
        """Test that require_tenant raises when only ID is set."""
        with TenantContext(tenant_id="only-id"):
            with pytest.raises(TenantNotSetError):
                require_tenant()


class TestTenantContextInfo:
    """Tests for TenantContextInfo dataclass."""

    def test_context_info_creation(self):
        """Test TenantContextInfo creation."""
        tenant = Tenant(id="info-test", name="Info", slug="info")
        info = TenantContextInfo(
            tenant_id="info-test",
            tenant=tenant,
            is_set=True,
            depth=2,
        )
        assert info.tenant_id == "info-test"
        assert info.tenant is tenant
        assert info.is_set is True
        assert info.depth == 2

    def test_context_info_without_tenant(self):
        """Test TenantContextInfo without tenant object."""
        info = TenantContextInfo(
            tenant_id="only-id",
            tenant=None,
            is_set=True,
            depth=1,
        )
        assert info.tenant_id == "only-id"
        assert info.tenant is None
        assert info.is_set is True


class TestTenantNotSetError:
    """Tests for TenantNotSetError exception."""

    def test_error_message(self):
        """Test error has correct message."""
        error = TenantNotSetError("Custom message")
        assert str(error) == "Custom message"

    def test_error_inheritance(self):
        """Test error inherits from Exception."""
        assert issubclass(TenantNotSetError, Exception)


class TestTenantMismatchError:
    """Tests for TenantMismatchError exception."""

    def test_error_message(self):
        """Test error has correct message."""
        error = TenantMismatchError("Mismatch occurred")
        assert str(error) == "Mismatch occurred"

    def test_error_inheritance(self):
        """Test error inherits from Exception."""
        assert issubclass(TenantMismatchError, Exception)


class TestThreadSafety:
    """Tests for thread safety of tenant context."""

    def test_contexts_isolated_between_threads(self):
        """Test that tenant contexts are isolated between threads."""
        results = {}

        def set_and_get_tenant(tenant_id: str, key: str):
            with TenantContext(tenant_id=tenant_id):
                # Small sleep to increase chance of interleaving
                import time

                time.sleep(0.01)
                results[key] = get_current_tenant_id()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(set_and_get_tenant, f"tenant-{i}", f"result-{i}") for i in range(4)
            ]
            for f in futures:
                f.result()

        # Each thread should see its own tenant
        for i in range(4):
            assert results[f"result-{i}"] == f"tenant-{i}"

    def test_main_thread_unaffected_by_worker_threads(self):
        """Test that main thread context is not affected by workers."""
        with TenantContext(tenant_id="main-thread"):

            def worker():
                with TenantContext(tenant_id="worker-thread"):
                    pass

            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(worker).result()

            # Main thread should still have its context
            assert get_current_tenant_id() == "main-thread"


class TestContextCleanup:
    """Tests for proper context cleanup."""

    def test_context_cleaned_on_exception(self):
        """Test that context is cleaned up on exception."""
        try:
            with TenantContext(tenant_id="exception-test"):
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert get_current_tenant_id() is None

    @pytest.mark.asyncio
    async def test_async_context_cleaned_on_exception(self):
        """Test that async context is cleaned up on exception."""
        try:
            async with TenantContext(tenant_id="async-exception"):
                raise ValueError("Async exception")
        except ValueError:
            pass

        assert get_current_tenant_id() is None
