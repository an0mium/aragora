"""
Enterprise Control Plane for Aragora Multi-Agent System.

This module provides centralized orchestration of heterogeneous AI agents
with distributed coordination, health monitoring, and task scheduling.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                  Control Plane                          │
    │   AgentRegistry │ TaskScheduler │ HealthMonitor         │
    ├─────────────────────────────────────────────────────────┤
    │                  Redis Backend                          │
    │   Service Discovery │ Job Queue │ Health Probes         │
    └─────────────────────────────────────────────────────────┘

Components:
    - AgentRegistry: Service discovery with heartbeats and capability tracking
    - TaskScheduler: Priority-based task distribution with load balancing
    - HealthMonitor: Liveness probes and circuit breaker integration
    - ControlPlaneCoordinator: High-level API unifying all components

Usage:
    from aragora.control_plane import ControlPlaneCoordinator

    # Initialize control plane
    coordinator = await ControlPlaneCoordinator.create()

    # Register an agent
    await coordinator.register_agent(
        agent_id="claude-3",
        capabilities=["debate", "code", "analysis"],
        metadata={"model": "claude-3-opus", "provider": "anthropic"}
    )

    # Submit a task
    task_id = await coordinator.submit_task(
        task_type="debate",
        payload={"question": "Should we use microservices?"},
        required_capabilities=["debate"],
    )

    # Wait for result
    result = await coordinator.wait_for_result(task_id)

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    CONTROL_PLANE_PREFIX: Key prefix (default: aragora:cp:)
    HEARTBEAT_INTERVAL: Agent heartbeat interval in seconds (default: 10)
    TASK_TIMEOUT: Default task timeout in seconds (default: 300)

See docs/CONTROL_PLANE.md for full documentation.
"""

from aragora.control_plane.registry import (
    AgentCapability,
    AgentInfo,
    AgentRegistry,
    AgentStatus,
)
from aragora.control_plane.scheduler import (
    Task,
    TaskPriority,
    TaskScheduler,
    TaskStatus,
)
from aragora.control_plane.health import (
    HealthCheck,
    HealthMonitor,
    HealthStatus,
)
from aragora.control_plane.coordinator import (
    ControlPlaneCoordinator,
    create_control_plane,
)
from aragora.control_plane.leader import (
    LeaderConfig,
    LeaderElection,
    LeaderInfo,
    LeaderState,
)
from aragora.control_plane.multi_tenancy import (
    TenantContext,
    TenantEnforcer,
    TenantEnforcementError,
    TenantQuota,
    TenantState,
    get_current_tenant,
    get_global_enforcer,
    set_current_tenant,
    set_global_enforcer,
    with_tenant,
)
from aragora.control_plane.shared_state import (
    AgentState,
    SharedControlPlaneState,
    TaskState,
    close_shared_state,
    get_shared_state,
    get_shared_state_sync,
    set_shared_state,
)

__all__ = [
    # Registry
    "AgentRegistry",
    "AgentInfo",
    "AgentStatus",
    "AgentCapability",
    # Scheduler
    "TaskScheduler",
    "Task",
    "TaskStatus",
    "TaskPriority",
    # Health
    "HealthMonitor",
    "HealthCheck",
    "HealthStatus",
    # Coordinator
    "ControlPlaneCoordinator",
    "create_control_plane",
    # Leader Election
    "LeaderElection",
    "LeaderConfig",
    "LeaderInfo",
    "LeaderState",
    # Multi-Tenancy
    "TenantContext",
    "TenantEnforcer",
    "TenantEnforcementError",
    "TenantQuota",
    "TenantState",
    "get_current_tenant",
    "get_global_enforcer",
    "set_current_tenant",
    "set_global_enforcer",
    "with_tenant",
    # Shared State
    "SharedControlPlaneState",
    "AgentState",
    "TaskState",
    "get_shared_state",
    "get_shared_state_sync",
    "set_shared_state",
    "close_shared_state",
]
