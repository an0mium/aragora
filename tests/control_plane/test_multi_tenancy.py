"""
Tests for Control Plane Multi-Tenancy.

Tests cover:
- TenantContext context manager
- with_tenant decorator
- TenantQuota configuration
- TenantState tracking
- TenantEnforcer enforcement
- get_current_tenant / set_current_tenant functions
"""

import pytest
import asyncio

from aragora.control_plane.multi_tenancy import (
    TenantContext,
    TenantQuota,
    TenantState,
    TenantEnforcer,
    get_current_tenant,
    set_current_tenant,
    with_tenant,
)


class TestTenantContextFunctions:
    """Tests for tenant context getter/setter functions."""

    def test_get_current_tenant_default_none(self):
        """Test that get_current_tenant returns None by default."""
        # Reset to ensure clean state
        set_current_tenant(None)
        assert get_current_tenant() is None

    def test_set_and_get_current_tenant(self):
        """Test setting and getting current tenant."""
        set_current_tenant("workspace_123")
        assert get_current_tenant() == "workspace_123"
        # Cleanup
        set_current_tenant(None)

    def test_set_current_tenant_to_none(self):
        """Test clearing current tenant."""
        set_current_tenant("workspace_456")
        assert get_current_tenant() == "workspace_456"
        set_current_tenant(None)
        assert get_current_tenant() is None


class TestTenantContext:
    """Tests for TenantContext context manager."""

    def test_sync_context_manager(self):
        """Test TenantContext as sync context manager."""
        set_current_tenant(None)
        assert get_current_tenant() is None

        with TenantContext("sync_workspace"):
            assert get_current_tenant() == "sync_workspace"

        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test TenantContext as async context manager."""
        set_current_tenant(None)
        assert get_current_tenant() is None

        async with TenantContext("async_workspace"):
            assert get_current_tenant() == "async_workspace"

        assert get_current_tenant() is None

    def test_nested_sync_contexts(self):
        """Test nested sync tenant contexts."""
        set_current_tenant(None)

        with TenantContext("outer"):
            assert get_current_tenant() == "outer"
            with TenantContext("inner"):
                assert get_current_tenant() == "inner"
            assert get_current_tenant() == "outer"

        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_nested_async_contexts(self):
        """Test nested async tenant contexts."""
        set_current_tenant(None)

        async with TenantContext("outer_async"):
            assert get_current_tenant() == "outer_async"
            async with TenantContext("inner_async"):
                assert get_current_tenant() == "inner_async"
            assert get_current_tenant() == "outer_async"

        assert get_current_tenant() is None

    def test_context_restored_on_exception(self):
        """Test that context is restored even on exception."""
        set_current_tenant("original")

        try:
            with TenantContext("temporary"):
                assert get_current_tenant() == "temporary"
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert get_current_tenant() == "original"
        set_current_tenant(None)


class TestWithTenantDecorator:
    """Tests for with_tenant decorator."""

    def test_sync_function_decorator(self):
        """Test with_tenant decorator on sync function."""
        set_current_tenant(None)

        @with_tenant("decorated_workspace")
        def sync_func():
            return get_current_tenant()

        result = sync_func()
        assert result == "decorated_workspace"
        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_async_function_decorator(self):
        """Test with_tenant decorator on async function."""
        set_current_tenant(None)

        @with_tenant("async_decorated")
        async def async_func():
            return get_current_tenant()

        result = await async_func()
        assert result == "async_decorated"
        assert get_current_tenant() is None

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves function metadata."""

        @with_tenant("test")
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    @pytest.mark.asyncio
    async def test_decorator_with_args_kwargs(self):
        """Test decorator passes args and kwargs correctly."""

        @with_tenant("args_test")
        async def func_with_args(a, b, c=None):
            return (a, b, c, get_current_tenant())

        result = await func_with_args(1, 2, c=3)
        assert result == (1, 2, 3, "args_test")


class TestTenantQuota:
    """Tests for TenantQuota dataclass."""

    def test_quota_defaults(self):
        """Test default quota values."""
        quota = TenantQuota()

        assert quota.max_agents == 100
        assert quota.max_concurrent_tasks == 50
        assert quota.max_queued_tasks == 1000
        assert quota.max_task_timeout_seconds == 3600.0
        assert quota.rate_limit_per_minute == 1000

    def test_custom_quota(self):
        """Test custom quota values."""
        quota = TenantQuota(
            max_agents=10,
            max_concurrent_tasks=5,
            max_queued_tasks=100,
            max_task_timeout_seconds=300.0,
            rate_limit_per_minute=60,
        )

        assert quota.max_agents == 10
        assert quota.max_concurrent_tasks == 5
        assert quota.max_queued_tasks == 100
        assert quota.max_task_timeout_seconds == 300.0
        assert quota.rate_limit_per_minute == 60


class TestTenantState:
    """Tests for TenantState dataclass."""

    def test_state_creation(self):
        """Test creating tenant state."""
        state = TenantState(
            tenant_id="tenant_123",
            quota=TenantQuota(max_agents=50),
        )

        assert state.tenant_id == "tenant_123"
        assert state.quota.max_agents == 50

    def test_state_defaults(self):
        """Test tenant state default values."""
        state = TenantState(tenant_id="test")

        assert state.tenant_id == "test"
        assert state.quota is not None
        assert state.registered_agents == 0
        assert state.running_tasks == 0
        assert state.queued_tasks == 0

    def test_can_register_agent(self):
        """Test agent registration capacity check."""
        state = TenantState(
            tenant_id="test",
            quota=TenantQuota(max_agents=2),
            registered_agents=1,
        )

        assert state.can_register_agent() is True

        # At limit
        state.registered_agents = 2
        assert state.can_register_agent() is False

    def test_can_submit_task(self):
        """Test task submission capacity check."""
        state = TenantState(
            tenant_id="test",
            quota=TenantQuota(max_concurrent_tasks=10, max_queued_tasks=100),
            running_tasks=5,
            queued_tasks=50,
        )

        assert state.can_submit_task() is True

        # At concurrent limit
        state.running_tasks = 10
        assert state.can_submit_task() is False

        # At queue limit
        state.running_tasks = 5
        state.queued_tasks = 100
        assert state.can_submit_task() is False

    def test_rate_limit_check(self):
        """Test rate limit checking."""
        state = TenantState(
            tenant_id="test",
            quota=TenantQuota(rate_limit_per_minute=5),
        )

        # Should allow first 5 requests
        for _ in range(5):
            assert state.check_rate_limit() is True

        # 6th request should be blocked
        assert state.check_rate_limit() is False


class TestTenantEnforcer:
    """Tests for TenantEnforcer class."""

    def test_enforcer_creation(self):
        """Test creating tenant enforcer."""
        enforcer = TenantEnforcer()
        assert enforcer is not None

    @pytest.mark.asyncio
    async def test_enforcer_set_tenant_quota(self):
        """Test setting a tenant quota."""
        enforcer = TenantEnforcer()
        quota = TenantQuota(max_agents=25)

        await enforcer.set_tenant_quota("new_tenant", quota)

        state = await enforcer.get_tenant_state("new_tenant")
        assert state is not None
        assert state.quota.max_agents == 25

    @pytest.mark.asyncio
    async def test_enforcer_get_creates_state(self):
        """Test that get_tenant_state creates state if not exists."""
        enforcer = TenantEnforcer()

        state = await enforcer.get_tenant_state("new_tenant")
        # Should create a new state with default quota
        assert state is not None
        assert state.tenant_id == "new_tenant"
        assert state.quota.max_agents == 100  # Default

    @pytest.mark.asyncio
    async def test_enforcer_check_agent_limit(self):
        """Test checking agent limit enforcement."""
        enforcer = TenantEnforcer()
        quota = TenantQuota(max_agents=2)
        await enforcer.set_tenant_quota("limited_tenant", quota)

        # Should allow first two agents
        assert await enforcer.can_register_agent("limited_tenant") is True

    @pytest.mark.asyncio
    async def test_enforcer_multiple_tenants_isolated(self):
        """Test that multiple tenants are isolated."""
        enforcer = TenantEnforcer()

        await enforcer.set_tenant_quota("tenant_a", TenantQuota(max_agents=10))
        await enforcer.set_tenant_quota("tenant_b", TenantQuota(max_agents=20))

        state_a = await enforcer.get_tenant_state("tenant_a")
        state_b = await enforcer.get_tenant_state("tenant_b")

        assert state_a.quota.max_agents == 10
        assert state_b.quota.max_agents == 20

    @pytest.mark.asyncio
    async def test_enforcer_record_activity(self):
        """Test recording agent and task activity."""
        enforcer = TenantEnforcer()

        await enforcer.record_agent_registered("test_tenant")
        state = await enforcer.get_tenant_state("test_tenant")
        assert state.registered_agents == 1

        await enforcer.record_task_submitted("test_tenant")
        state = await enforcer.get_tenant_state("test_tenant")
        assert state.queued_tasks == 1

        await enforcer.record_task_started("test_tenant")
        state = await enforcer.get_tenant_state("test_tenant")
        assert state.queued_tasks == 0
        assert state.running_tasks == 1

    def test_filter_by_tenant(self):
        """Test filtering items by tenant."""
        enforcer = TenantEnforcer()

        class MockItem:
            def __init__(self, item_id, tenant_id):
                self.id = item_id
                self.tenant_id = tenant_id

        items = [
            MockItem("a1", "tenant_1"),
            MockItem("a2", "tenant_1"),
            MockItem("a3", "tenant_2"),
        ]

        filtered = enforcer.filter_by_tenant(items, "tenant_1")
        assert len(filtered) == 2
        assert all(i.tenant_id == "tenant_1" for i in filtered)


class TestTenantIsolation:
    """Integration tests for tenant isolation."""

    @pytest.mark.asyncio
    async def test_concurrent_tenant_contexts(self):
        """Test that concurrent tasks maintain separate tenant contexts."""
        results = []

        async def task_with_tenant(tenant_id: str, delay: float):
            async with TenantContext(tenant_id):
                await asyncio.sleep(delay)
                results.append((tenant_id, get_current_tenant()))

        # Run tasks concurrently with different tenants
        await asyncio.gather(
            task_with_tenant("tenant_1", 0.01),
            task_with_tenant("tenant_2", 0.005),
            task_with_tenant("tenant_3", 0.015),
        )

        # Each task should have seen its own tenant
        for expected, actual in results:
            assert expected == actual
