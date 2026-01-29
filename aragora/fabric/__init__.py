"""
Agent Fabric - High-scale agent orchestration substrate.

The Agent Fabric provides:
- Scheduling: Fair task distribution across agents
- Isolation: Per-agent resource boundaries
- Policy: Tool access control and approvals
- Budget: Cost tracking and enforcement
- Lifecycle: Agent spawn, heartbeat, termination
- Telemetry: Metrics, traces, and logs

Usage:
    from aragora.fabric import AgentFabric

    fabric = AgentFabric()
    agent = await fabric.spawn(AgentConfig(model="claude-3-opus"))
    result = await fabric.schedule(task, agent.id)
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
]
