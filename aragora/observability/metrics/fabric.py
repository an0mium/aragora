"""
Agent Fabric metrics.

Provides Prometheus metrics for tracking fabric agent pools,
task scheduling, policy decisions, and budget usage.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any
from collections.abc import Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Global metric variables - Agent stats
FABRIC_AGENTS_ACTIVE: Any = None
FABRIC_AGENTS_HEALTH: Any = None
FABRIC_AGENTS_SPAWNED: Any = None
FABRIC_AGENTS_TERMINATED: Any = None

# Task stats
FABRIC_TASKS_QUEUED: Any = None
FABRIC_TASKS_COMPLETED: Any = None
FABRIC_TASK_LATENCY: Any = None
FABRIC_TASK_QUEUE_DEPTH: Any = None

# Policy stats
FABRIC_POLICY_DECISIONS: Any = None
FABRIC_POLICY_APPROVALS_PENDING: Any = None

# Budget stats
FABRIC_BUDGET_USAGE: Any = None
FABRIC_BUDGET_ALERTS: Any = None

_initialized = False


def init_fabric_metrics() -> None:
    """Initialize fabric metrics."""
    global _initialized
    global FABRIC_AGENTS_ACTIVE, FABRIC_AGENTS_HEALTH, FABRIC_AGENTS_SPAWNED
    global FABRIC_AGENTS_TERMINATED, FABRIC_TASKS_QUEUED, FABRIC_TASKS_COMPLETED
    global FABRIC_TASK_LATENCY, FABRIC_TASK_QUEUE_DEPTH, FABRIC_POLICY_DECISIONS
    global FABRIC_POLICY_APPROVALS_PENDING, FABRIC_BUDGET_USAGE, FABRIC_BUDGET_ALERTS

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Agent metrics
        FABRIC_AGENTS_ACTIVE = Gauge(
            "aragora_fabric_agents_active",
            "Number of active agents in fabric",
            ["pool_id"],
        )

        FABRIC_AGENTS_HEALTH = Gauge(
            "aragora_fabric_agents_health",
            "Agent health status counts",
            ["status"],  # healthy, degraded, unhealthy
        )

        FABRIC_AGENTS_SPAWNED = Counter(
            "aragora_fabric_agents_spawned_total",
            "Total agents spawned",
            ["pool_id", "model"],
        )

        FABRIC_AGENTS_TERMINATED = Counter(
            "aragora_fabric_agents_terminated_total",
            "Total agents terminated",
            ["pool_id", "reason"],  # reason: graceful, timeout, error
        )

        # Task metrics
        FABRIC_TASKS_QUEUED = Counter(
            "aragora_fabric_tasks_queued_total",
            "Total tasks queued",
            ["task_type", "priority"],
        )

        FABRIC_TASKS_COMPLETED = Counter(
            "aragora_fabric_tasks_completed_total",
            "Total tasks completed",
            ["task_type", "status"],  # status: success, failed, cancelled
        )

        FABRIC_TASK_LATENCY = Histogram(
            "aragora_fabric_task_latency_seconds",
            "Task execution latency",
            ["task_type"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        FABRIC_TASK_QUEUE_DEPTH = Gauge(
            "aragora_fabric_task_queue_depth",
            "Current task queue depth per agent",
            ["agent_id"],
        )

        # Policy metrics
        FABRIC_POLICY_DECISIONS = Counter(
            "aragora_fabric_policy_decisions_total",
            "Policy decisions made",
            ["decision"],  # allowed, denied, approval_required
        )

        FABRIC_POLICY_APPROVALS_PENDING = Gauge(
            "aragora_fabric_policy_approvals_pending",
            "Number of pending approval requests",
        )

        # Budget metrics
        FABRIC_BUDGET_USAGE = Gauge(
            "aragora_fabric_budget_usage_percent",
            "Budget usage percentage",
            ["entity_id", "entity_type"],  # entity_type: agent, pool, tenant
        )

        FABRIC_BUDGET_ALERTS = Counter(
            "aragora_fabric_budget_alerts_total",
            "Budget alerts triggered",
            ["entity_id", "alert_type"],  # alert_type: soft_limit, hard_limit
        )

        _initialized = True
        logger.debug("Fabric metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global FABRIC_AGENTS_ACTIVE, FABRIC_AGENTS_HEALTH, FABRIC_AGENTS_SPAWNED
    global FABRIC_AGENTS_TERMINATED, FABRIC_TASKS_QUEUED, FABRIC_TASKS_COMPLETED
    global FABRIC_TASK_LATENCY, FABRIC_TASK_QUEUE_DEPTH, FABRIC_POLICY_DECISIONS
    global FABRIC_POLICY_APPROVALS_PENDING, FABRIC_BUDGET_USAGE, FABRIC_BUDGET_ALERTS

    FABRIC_AGENTS_ACTIVE = NoOpMetric()
    FABRIC_AGENTS_HEALTH = NoOpMetric()
    FABRIC_AGENTS_SPAWNED = NoOpMetric()
    FABRIC_AGENTS_TERMINATED = NoOpMetric()
    FABRIC_TASKS_QUEUED = NoOpMetric()
    FABRIC_TASKS_COMPLETED = NoOpMetric()
    FABRIC_TASK_LATENCY = NoOpMetric()
    FABRIC_TASK_QUEUE_DEPTH = NoOpMetric()
    FABRIC_POLICY_DECISIONS = NoOpMetric()
    FABRIC_POLICY_APPROVALS_PENDING = NoOpMetric()
    FABRIC_BUDGET_USAGE = NoOpMetric()
    FABRIC_BUDGET_ALERTS = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_fabric_metrics()


# =============================================================================
# Recording Functions - Agents
# =============================================================================


def set_agents_active(pool_id: str, count: int) -> None:
    """Set the number of active agents in a pool.

    Args:
        pool_id: Pool identifier
        count: Number of active agents
    """
    _ensure_init()
    FABRIC_AGENTS_ACTIVE.labels(pool_id=pool_id).set(count)


def set_agents_health(healthy: int, degraded: int, unhealthy: int) -> None:
    """Set agent health status counts.

    Args:
        healthy: Number of healthy agents
        degraded: Number of degraded agents
        unhealthy: Number of unhealthy agents
    """
    _ensure_init()
    FABRIC_AGENTS_HEALTH.labels(status="healthy").set(healthy)
    FABRIC_AGENTS_HEALTH.labels(status="degraded").set(degraded)
    FABRIC_AGENTS_HEALTH.labels(status="unhealthy").set(unhealthy)


def record_agent_spawned(pool_id: str, model: str) -> None:
    """Record an agent spawn event.

    Args:
        pool_id: Pool identifier
        model: Model name
    """
    _ensure_init()
    FABRIC_AGENTS_SPAWNED.labels(pool_id=pool_id, model=model).inc()


def record_agent_terminated(pool_id: str, reason: str = "graceful") -> None:
    """Record an agent termination event.

    Args:
        pool_id: Pool identifier
        reason: Termination reason (graceful, timeout, error)
    """
    _ensure_init()
    FABRIC_AGENTS_TERMINATED.labels(pool_id=pool_id, reason=reason).inc()


# =============================================================================
# Recording Functions - Tasks
# =============================================================================


def record_task_queued(task_type: str, priority: str = "normal") -> None:
    """Record a task being queued.

    Args:
        task_type: Type of task (debate, generate, etc.)
        priority: Task priority (critical, high, normal, low)
    """
    _ensure_init()
    FABRIC_TASKS_QUEUED.labels(task_type=task_type, priority=priority).inc()


def record_task_completed(
    task_type: str,
    success: bool,
    latency_seconds: float | None = None,
) -> None:
    """Record task completion.

    Args:
        task_type: Type of task
        success: Whether task succeeded
        latency_seconds: Optional task latency
    """
    _ensure_init()
    status = "success" if success else "failed"
    FABRIC_TASKS_COMPLETED.labels(task_type=task_type, status=status).inc()
    if latency_seconds is not None:
        FABRIC_TASK_LATENCY.labels(task_type=task_type).observe(latency_seconds)


def record_task_cancelled(task_type: str) -> None:
    """Record task cancellation.

    Args:
        task_type: Type of task
    """
    _ensure_init()
    FABRIC_TASKS_COMPLETED.labels(task_type=task_type, status="cancelled").inc()


def set_task_queue_depth(agent_id: str, depth: int) -> None:
    """Set current task queue depth for an agent.

    Args:
        agent_id: Agent identifier
        depth: Queue depth
    """
    _ensure_init()
    FABRIC_TASK_QUEUE_DEPTH.labels(agent_id=agent_id).set(depth)


# =============================================================================
# Recording Functions - Policy
# =============================================================================


def record_policy_decision(decision: str) -> None:
    """Record a policy decision.

    Args:
        decision: Decision type (allowed, denied, approval_required)
    """
    _ensure_init()
    FABRIC_POLICY_DECISIONS.labels(decision=decision).inc()


def set_pending_approvals(count: int) -> None:
    """Set number of pending approval requests.

    Args:
        count: Number of pending approvals
    """
    _ensure_init()
    FABRIC_POLICY_APPROVALS_PENDING.set(count)


# =============================================================================
# Recording Functions - Budget
# =============================================================================


def set_budget_usage(
    entity_id: str,
    entity_type: str,
    usage_percent: float,
) -> None:
    """Set budget usage percentage.

    Args:
        entity_id: Entity identifier
        entity_type: Type of entity (agent, pool, tenant)
        usage_percent: Usage percentage (0-100)
    """
    _ensure_init()
    FABRIC_BUDGET_USAGE.labels(
        entity_id=entity_id,
        entity_type=entity_type,
    ).set(usage_percent)


def record_budget_alert(entity_id: str, alert_type: str = "soft_limit") -> None:
    """Record a budget alert.

    Args:
        entity_id: Entity identifier
        alert_type: Type of alert (soft_limit, hard_limit)
    """
    _ensure_init()
    FABRIC_BUDGET_ALERTS.labels(entity_id=entity_id, alert_type=alert_type).inc()


# =============================================================================
# Convenience Functions
# =============================================================================


def record_fabric_stats(stats: dict) -> None:
    """Record fabric stats from FabricStats dict.

    Args:
        stats: Dictionary from AgentFabric.get_stats()
    """
    _ensure_init()

    # Lifecycle stats
    if "lifecycle" in stats:
        lc = stats["lifecycle"]
        set_agents_health(
            healthy=lc.get("agents_healthy", 0),
            degraded=lc.get("agents_degraded", 0),
            unhealthy=lc.get("agents_unhealthy", 0),
        )
        set_agents_active("default", lc.get("agents_active", 0))

    # Policy stats
    if "policy" in stats:
        pol = stats["policy"]
        set_pending_approvals(pol.get("pending_approvals", 0))


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def track_fabric_task(
    task_type: str,
    priority: str = "normal",
) -> Generator[None, None, None]:
    """Context manager to track fabric task execution.

    Automatically records task queued, completion, and latency.

    Args:
        task_type: Type of task
        priority: Task priority

    Example:
        with track_fabric_task("debate", "high"):
            result = await fabric.execute(task)
    """
    _ensure_init()
    record_task_queued(task_type, priority)
    start = time.perf_counter()
    success = True
    try:
        yield
    except BaseException:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_task_completed(task_type, success, latency)


__all__ = [
    # Metrics
    "FABRIC_AGENTS_ACTIVE",
    "FABRIC_AGENTS_HEALTH",
    "FABRIC_AGENTS_SPAWNED",
    "FABRIC_AGENTS_TERMINATED",
    "FABRIC_TASKS_QUEUED",
    "FABRIC_TASKS_COMPLETED",
    "FABRIC_TASK_LATENCY",
    "FABRIC_TASK_QUEUE_DEPTH",
    "FABRIC_POLICY_DECISIONS",
    "FABRIC_POLICY_APPROVALS_PENDING",
    "FABRIC_BUDGET_USAGE",
    "FABRIC_BUDGET_ALERTS",
    # Functions
    "init_fabric_metrics",
    "set_agents_active",
    "set_agents_health",
    "record_agent_spawned",
    "record_agent_terminated",
    "record_task_queued",
    "record_task_completed",
    "record_task_cancelled",
    "set_task_queue_depth",
    "record_policy_decision",
    "set_pending_approvals",
    "set_budget_usage",
    "record_budget_alert",
    "record_fabric_stats",
    "track_fabric_task",
]
