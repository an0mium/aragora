"""
Multi-Tenancy Enforcement for the Aragora Control Plane.

Provides workspace/tenant isolation for:
- Agent registration and discovery
- Task submission and retrieval
- Health monitoring
- Resource quotas

Each workspace operates as an isolated tenant with its own:
- Agent pool (agents can be shared or dedicated)
- Task queue
- Resource limits
- Configuration

Usage:
    from aragora.control_plane.multi_tenancy import (
        TenantContext,
        TenantEnforcer,
        with_tenant,
    )

    # Create enforcer
    enforcer = TenantEnforcer()

    # Set current tenant context
    async with TenantContext("workspace_123"):
        # All operations within this context are scoped to workspace_123
        agents = await registry.get_available_agents()
        task_id = await scheduler.submit_task(...)

    # Or use decorator
    @with_tenant("workspace_123")
    async def my_function():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, cast

logger = logging.getLogger(__name__)

# Context variable for current tenant
_current_tenant: ContextVar[Optional[str]] = ContextVar("current_tenant", default=None)

# Type variable for generic functions
F = TypeVar("F", bound=Callable[..., Any])


def get_current_tenant() -> Optional[str]:
    """Get the current tenant/workspace ID from context."""
    return _current_tenant.get()


def set_current_tenant(tenant_id: Optional[str]) -> None:
    """Set the current tenant/workspace ID in context."""
    _current_tenant.set(tenant_id)


class TenantContext:
    """
    Context manager for scoped tenant operations.

    All control plane operations within this context will be
    automatically scoped to the specified tenant.

    Usage:
        async with TenantContext("my_workspace"):
            # Operations here are scoped to my_workspace
            agents = await registry.get_available_agents()
    """

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self._token = None

    def __enter__(self) -> "TenantContext":
        self._token = _current_tenant.set(self.tenant_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _current_tenant.reset(self._token)

    async def __aenter__(self) -> "TenantContext":
        self._token = _current_tenant.set(self.tenant_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        _current_tenant.reset(self._token)


def with_tenant(tenant_id: str) -> Callable[[F], F]:
    """
    Decorator to run a function within a tenant context.

    Usage:
        @with_tenant("workspace_123")
        async def my_function():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            async with TenantContext(tenant_id):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with TenantContext(tenant_id):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)

    return decorator


@dataclass
class TenantQuota:
    """
    Resource quotas for a tenant.

    Limits:
        max_agents: Maximum number of agents that can be registered
        max_concurrent_tasks: Maximum concurrent running tasks
        max_queued_tasks: Maximum tasks in queue
        max_task_timeout_seconds: Maximum allowed task timeout
        rate_limit_per_minute: Maximum task submissions per minute
    """

    max_agents: int = 100
    max_concurrent_tasks: int = 50
    max_queued_tasks: int = 1000
    max_task_timeout_seconds: float = 3600.0  # 1 hour
    rate_limit_per_minute: int = 1000


@dataclass
class TenantState:
    """
    Runtime state for a tenant.

    Tracks current usage against quotas.
    """

    tenant_id: str
    quota: TenantQuota = field(default_factory=TenantQuota)
    registered_agents: int = 0
    running_tasks: int = 0
    queued_tasks: int = 0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    # Rate limiting state
    _request_timestamps: List[float] = field(default_factory=list)

    def can_register_agent(self) -> bool:
        """Check if tenant can register another agent."""
        return self.registered_agents < self.quota.max_agents

    def can_submit_task(self) -> bool:
        """Check if tenant can submit another task."""
        if self.running_tasks >= self.quota.max_concurrent_tasks:
            return False
        if self.queued_tasks >= self.quota.max_queued_tasks:
            return False
        return True

    def check_rate_limit(self) -> bool:
        """Check if tenant is within rate limit."""
        now = time.time()
        minute_ago = now - 60

        # Clean old timestamps
        self._request_timestamps = [t for t in self._request_timestamps if t > minute_ago]

        # Check limit
        if len(self._request_timestamps) >= self.quota.rate_limit_per_minute:
            return False

        # Record this request
        self._request_timestamps.append(now)
        return True


class TenantEnforcementError(Exception):
    """Raised when a tenant operation violates constraints."""

    def __init__(self, message: str, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id
        super().__init__(message)


class TenantEnforcer:
    """
    Enforces multi-tenancy constraints in the control plane.

    Provides:
    - Tenant state tracking
    - Quota enforcement
    - Rate limiting
    - Cross-tenant isolation

    Usage:
        enforcer = TenantEnforcer()

        # Check before agent registration
        if not enforcer.can_register_agent("workspace_123"):
            raise TenantEnforcementError("Agent quota exceeded")

        # Check before task submission
        enforcer.enforce_task_submission("workspace_123")

        # Filter resources by tenant
        agents = enforcer.filter_by_tenant(all_agents, "workspace_123")
    """

    def __init__(self, default_quota: Optional[TenantQuota] = None):
        self._default_quota = default_quota or TenantQuota()
        self._tenant_states: Dict[str, TenantState] = {}
        self._tenant_quotas: Dict[str, TenantQuota] = {}
        self._shared_agents: Set[str] = set()  # Agents available to all tenants
        self._lock = asyncio.Lock()

    def _get_or_create_state(self, tenant_id: str) -> TenantState:
        """Get or create tenant state."""
        if tenant_id not in self._tenant_states:
            quota = self._tenant_quotas.get(tenant_id, self._default_quota)
            self._tenant_states[tenant_id] = TenantState(
                tenant_id=tenant_id,
                quota=quota,
            )
        return self._tenant_states[tenant_id]

    async def set_tenant_quota(self, tenant_id: str, quota: TenantQuota) -> None:
        """Set custom quota for a tenant."""
        async with self._lock:
            self._tenant_quotas[tenant_id] = quota
            if tenant_id in self._tenant_states:
                self._tenant_states[tenant_id].quota = quota

    async def get_tenant_state(self, tenant_id: str) -> TenantState:
        """Get current state for a tenant."""
        async with self._lock:
            return self._get_or_create_state(tenant_id)

    async def mark_agent_shared(self, agent_id: str) -> None:
        """Mark an agent as shared (available to all tenants)."""
        async with self._lock:
            self._shared_agents.add(agent_id)

    async def unmark_agent_shared(self, agent_id: str) -> None:
        """Remove shared status from an agent."""
        async with self._lock:
            self._shared_agents.discard(agent_id)

    def is_agent_shared(self, agent_id: str) -> bool:
        """Check if an agent is shared."""
        return agent_id in self._shared_agents

    async def can_register_agent(self, tenant_id: str) -> bool:
        """Check if tenant can register another agent."""
        async with self._lock:
            state = self._get_or_create_state(tenant_id)
            return state.can_register_agent()

    async def can_submit_task(self, tenant_id: str) -> bool:
        """Check if tenant can submit another task."""
        async with self._lock:
            state = self._get_or_create_state(tenant_id)
            if not state.can_submit_task():
                return False
            if not state.check_rate_limit():
                return False
            return True

    async def enforce_agent_registration(self, tenant_id: str) -> None:
        """Enforce constraints for agent registration. Raises if not allowed."""
        if not await self.can_register_agent(tenant_id):
            raise TenantEnforcementError(
                f"Agent quota exceeded for tenant '{tenant_id}'",
                tenant_id=tenant_id,
            )

    async def enforce_task_submission(self, tenant_id: str) -> None:
        """Enforce constraints for task submission. Raises if not allowed."""
        async with self._lock:
            state = self._get_or_create_state(tenant_id)

            if state.running_tasks >= state.quota.max_concurrent_tasks:
                raise TenantEnforcementError(
                    f"Concurrent task limit reached for tenant '{tenant_id}'",
                    tenant_id=tenant_id,
                )

            if state.queued_tasks >= state.quota.max_queued_tasks:
                raise TenantEnforcementError(
                    f"Task queue limit reached for tenant '{tenant_id}'",
                    tenant_id=tenant_id,
                )

            if not state.check_rate_limit():
                raise TenantEnforcementError(
                    f"Rate limit exceeded for tenant '{tenant_id}'",
                    tenant_id=tenant_id,
                )

    async def record_agent_registered(self, tenant_id: str) -> None:
        """Record that an agent was registered for tenant."""
        async with self._lock:
            state = self._get_or_create_state(tenant_id)
            state.registered_agents += 1
            state.last_activity = time.time()

    async def record_agent_unregistered(self, tenant_id: str) -> None:
        """Record that an agent was unregistered for tenant."""
        async with self._lock:
            state = self._get_or_create_state(tenant_id)
            state.registered_agents = max(0, state.registered_agents - 1)
            state.last_activity = time.time()

    async def record_task_submitted(self, tenant_id: str) -> None:
        """Record that a task was submitted for tenant."""
        async with self._lock:
            state = self._get_or_create_state(tenant_id)
            state.queued_tasks += 1
            state.last_activity = time.time()

    async def record_task_started(self, tenant_id: str) -> None:
        """Record that a task started execution for tenant."""
        async with self._lock:
            state = self._get_or_create_state(tenant_id)
            state.queued_tasks = max(0, state.queued_tasks - 1)
            state.running_tasks += 1
            state.last_activity = time.time()

    async def record_task_completed(self, tenant_id: str) -> None:
        """Record that a task completed for tenant."""
        async with self._lock:
            state = self._get_or_create_state(tenant_id)
            state.running_tasks = max(0, state.running_tasks - 1)
            state.last_activity = time.time()

    def filter_by_tenant(
        self,
        items: List[Any],
        tenant_id: str,
        tenant_attr: str = "tenant_id",
        include_shared: bool = True,
    ) -> List[Any]:
        """
        Filter a list of items to only those belonging to the tenant.

        Args:
            items: List of items with tenant attribute
            tenant_id: Tenant to filter for
            tenant_attr: Name of the tenant attribute on items
            include_shared: Include shared items (for agents)

        Returns:
            Filtered list of items
        """
        result = []
        for item in items:
            item_tenant = getattr(item, tenant_attr, None)

            # Item belongs to tenant
            if item_tenant == tenant_id:
                result.append(item)
                continue

            # Check if it's a shared agent
            if include_shared:
                item_id = getattr(item, "agent_id", None) or getattr(item, "id", None)
                if item_id and self.is_agent_shared(item_id):
                    result.append(item)

        return result

    def validate_tenant_access(
        self,
        item: Any,
        tenant_id: str,
        tenant_attr: str = "tenant_id",
    ) -> bool:
        """
        Validate that a tenant has access to an item.

        Returns True if:
        - Item belongs to the tenant
        - Item is a shared agent

        Returns False otherwise.
        """
        item_tenant = getattr(item, tenant_attr, None)
        if item_tenant == tenant_id:
            return True

        # Check if it's a shared agent
        item_id = getattr(item, "agent_id", None) or getattr(item, "id", None)
        if item_id and self.is_agent_shared(item_id):
            return True

        return False

    async def get_all_states(self) -> Dict[str, TenantState]:
        """Get states for all tenants (admin operation)."""
        async with self._lock:
            return dict(self._tenant_states)

    async def cleanup_inactive_tenants(self, inactive_threshold_seconds: float = 86400.0) -> int:
        """
        Remove state for tenants that have been inactive.

        Args:
            inactive_threshold_seconds: Time after which to consider a tenant inactive

        Returns:
            Number of tenants cleaned up
        """
        async with self._lock:
            now = time.time()
            cutoff = now - inactive_threshold_seconds
            to_remove = [
                tid
                for tid, state in self._tenant_states.items()
                if state.last_activity < cutoff
                and state.registered_agents == 0
                and state.running_tasks == 0
                and state.queued_tasks == 0
            ]

            for tid in to_remove:
                del self._tenant_states[tid]

            return len(to_remove)


# Global enforcer instance
_global_enforcer: Optional[TenantEnforcer] = None


def get_global_enforcer() -> TenantEnforcer:
    """Get the global tenant enforcer instance."""
    global _global_enforcer
    if _global_enforcer is None:
        _global_enforcer = TenantEnforcer()
    return _global_enforcer


def set_global_enforcer(enforcer: TenantEnforcer) -> None:
    """Set the global tenant enforcer instance."""
    global _global_enforcer
    _global_enforcer = enforcer
