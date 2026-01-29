"""
Agent Fabric - High-scale agent orchestration substrate.

The Agent Fabric provides:
- Scheduling: Fair task distribution across agents
- Isolation: Per-agent resource boundaries
- Policy: Tool access control and approvals
- Budget: Cost tracking and enforcement
- Lifecycle: Agent spawn, heartbeat, termination
- Telemetry: Metrics, traces, and logs
- Hooks: GUPP-compliant work persistence (Gastown parity)

Usage:
    from aragora.fabric import AgentFabric

    fabric = AgentFabric()
    agent = await fabric.spawn(AgentConfig(model="claude-3-opus"))
    result = await fabric.schedule(task, agent.id)

For GUPP hook persistence:
    from aragora.fabric import HookManager

    hooks = HookManager()
    hook = await hooks.create_hook("agent-1", {"task": "refactor"})
    pending = await hooks.check_pending_hooks()  # GUPP patrol
"""

from .models import (
    AgentConfig,
    AgentHandle,
    AgentInfo,
    BudgetConfig,
    BudgetStatus,
    IsolationConfig,
    Policy,
    PolicyContext,
    PolicyDecision,
    PolicyRule,
    Priority,
    Task,
    TaskHandle,
    TaskStatus,
    Usage,
    UsageReport,
)
from .scheduler import AgentScheduler
from .lifecycle import LifecycleManager
from .policy import PolicyEngine
from .budget import BudgetManager
from .fabric import AgentFabric, AgentPool, FabricConfig, FabricStats
from .hooks import Hook, HookManager, HookManagerConfig, HookStatus

__all__ = [
    # Core facade
    "AgentFabric",
    "AgentPool",
    "FabricConfig",
    "FabricStats",
    # Components
    "AgentScheduler",
    "LifecycleManager",
    "PolicyEngine",
    "BudgetManager",
    # Models
    "AgentConfig",
    "AgentHandle",
    "AgentInfo",
    "BudgetConfig",
    "BudgetStatus",
    "IsolationConfig",
    "Policy",
    "PolicyContext",
    "PolicyDecision",
    "PolicyRule",
    "Priority",
    "Task",
    "TaskHandle",
    "TaskStatus",
    "Usage",
    "UsageReport",
    # Hooks (GUPP - Gastown parity)
    "Hook",
    "HookManager",
    "HookManagerConfig",
    "HookStatus",
]
