"""
Control Plane Metrics for Aragora.

Provides Prometheus metrics specifically for the enterprise control plane:
- Agent management (registration, heartbeats, health)
- Task scheduling (submission, completion, queue depth)
- Deliberation tracking (duration, SLA compliance, consensus)
- Policy enforcement (decisions, violations)

Usage:
    from aragora.observability.metrics.control_plane import (
        record_agent_registered,
        record_task_submitted,
        record_deliberation_complete,
    )

    # Record agent registration
    record_agent_registered("agent-001", success=True)

    # Record task completion
    record_task_complete("task-001", "deliberation", success=True, duration=45.2)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Control Plane metrics - initialized lazily
_initialized = False

# Metric instances
CP_AGENTS_REGISTERED: Any = None
CP_AGENTS_ACTIVE: Any = None
CP_AGENT_HEARTBEATS: Any = None
CP_AGENT_HEALTH_CHECKS: Any = None

CP_TASKS_SUBMITTED: Any = None
CP_TASKS_COMPLETED: Any = None
CP_TASKS_FAILED: Any = None
CP_TASK_DURATION: Any = None
CP_TASK_QUEUE_DEPTH: Any = None
CP_TASK_WAIT_TIME: Any = None

CP_DELIBERATIONS_STARTED: Any = None
CP_DELIBERATIONS_COMPLETED: Any = None
CP_DELIBERATION_DURATION: Any = None
CP_DELIBERATION_CONSENSUS: Any = None
CP_DELIBERATION_SLA_STATUS: Any = None
CP_DELIBERATION_AGENT_COUNT: Any = None

CP_POLICY_DECISIONS: Any = None
CP_POLICY_VIOLATIONS: Any = None
CP_POLICY_CHECK_LATENCY: Any = None


def _init_control_plane_metrics() -> bool:
    """Initialize control plane metrics lazily."""
    global _initialized
    global CP_AGENTS_REGISTERED, CP_AGENTS_ACTIVE, CP_AGENT_HEARTBEATS, CP_AGENT_HEALTH_CHECKS
    global CP_TASKS_SUBMITTED, CP_TASKS_COMPLETED, CP_TASKS_FAILED
    global CP_TASK_DURATION, CP_TASK_QUEUE_DEPTH, CP_TASK_WAIT_TIME
    global CP_DELIBERATIONS_STARTED, CP_DELIBERATIONS_COMPLETED
    global CP_DELIBERATION_DURATION, CP_DELIBERATION_CONSENSUS
    global CP_DELIBERATION_SLA_STATUS, CP_DELIBERATION_AGENT_COUNT
    global CP_POLICY_DECISIONS, CP_POLICY_VIOLATIONS, CP_POLICY_CHECK_LATENCY

    if _initialized:
        return True

    try:
        from aragora.observability.config import get_metrics_config

        config = get_metrics_config()

        if not config.enabled:
            _init_noop_metrics()
            _initialized = True
            return False

        from prometheus_client import Counter, Gauge, Histogram

        # Agent metrics
        CP_AGENTS_REGISTERED = Counter(
            "aragora_cp_agents_registered_total",
            "Total agent registrations",
            ["status"],  # success, failure
        )

        CP_AGENTS_ACTIVE = Gauge(
            "aragora_cp_agents_active",
            "Number of currently active agents",
            ["capability"],  # debate, summarize, classify, etc.
        )

        CP_AGENT_HEARTBEATS = Counter(
            "aragora_cp_agent_heartbeats_total",
            "Total agent heartbeats received",
            ["agent_id", "status"],  # healthy, degraded, unhealthy
        )

        CP_AGENT_HEALTH_CHECKS = Counter(
            "aragora_cp_agent_health_checks_total",
            "Total agent health check results",
            ["status"],  # healthy, unhealthy, timeout
        )

        # Task metrics
        CP_TASKS_SUBMITTED = Counter(
            "aragora_cp_tasks_submitted_total",
            "Total tasks submitted to control plane",
            ["task_type", "priority"],
        )

        CP_TASKS_COMPLETED = Counter(
            "aragora_cp_tasks_completed_total",
            "Total tasks completed successfully",
            ["task_type"],
        )

        CP_TASKS_FAILED = Counter(
            "aragora_cp_tasks_failed_total",
            "Total tasks that failed",
            ["task_type", "failure_reason"],
        )

        CP_TASK_DURATION = Histogram(
            "aragora_cp_task_duration_seconds",
            "Task execution duration in seconds",
            ["task_type"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600],
        )

        CP_TASK_QUEUE_DEPTH = Gauge(
            "aragora_cp_task_queue_depth",
            "Current task queue depth",
            ["priority"],
        )

        CP_TASK_WAIT_TIME = Histogram(
            "aragora_cp_task_wait_time_seconds",
            "Time tasks spend waiting in queue",
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300],
        )

        # Deliberation metrics
        CP_DELIBERATIONS_STARTED = Counter(
            "aragora_cp_deliberations_started_total",
            "Total deliberations started",
            ["mode"],  # sync, async, scheduled
        )

        CP_DELIBERATIONS_COMPLETED = Counter(
            "aragora_cp_deliberations_completed_total",
            "Total deliberations completed",
            ["status", "consensus_reached"],  # success/failure, true/false
        )

        CP_DELIBERATION_DURATION = Histogram(
            "aragora_cp_deliberation_duration_seconds",
            "Deliberation duration in seconds",
            ["consensus_reached"],
            buckets=[5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600],
        )

        CP_DELIBERATION_CONSENSUS = Histogram(
            "aragora_cp_deliberation_consensus_confidence",
            "Distribution of consensus confidence scores",
            buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
        )

        CP_DELIBERATION_SLA_STATUS = Counter(
            "aragora_cp_deliberation_sla_total",
            "Deliberation SLA compliance counts",
            ["level"],  # compliant, warning, critical, violated
        )

        CP_DELIBERATION_AGENT_COUNT = Histogram(
            "aragora_cp_deliberation_agent_count",
            "Number of agents participating in deliberations",
            buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

        # Policy metrics
        CP_POLICY_DECISIONS = Counter(
            "aragora_cp_policy_decisions_total",
            "Total policy decisions made",
            ["policy_type", "decision"],  # allow, deny, conditional
        )

        CP_POLICY_VIOLATIONS = Counter(
            "aragora_cp_policy_violations_total",
            "Policy violations detected",
            ["policy_type", "severity"],  # info, warning, critical
        )

        CP_POLICY_CHECK_LATENCY = Histogram(
            "aragora_cp_policy_check_latency_seconds",
            "Policy check latency",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )

        _initialized = True
        logger.debug("Control plane metrics initialized")
        return True

    except ImportError as e:
        logger.warning(f"prometheus_client not installed, control plane metrics disabled: {e}")
        _init_noop_metrics()
        _initialized = True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize control plane metrics: {e}")
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is unavailable."""
    global CP_AGENTS_REGISTERED, CP_AGENTS_ACTIVE, CP_AGENT_HEARTBEATS, CP_AGENT_HEALTH_CHECKS
    global CP_TASKS_SUBMITTED, CP_TASKS_COMPLETED, CP_TASKS_FAILED
    global CP_TASK_DURATION, CP_TASK_QUEUE_DEPTH, CP_TASK_WAIT_TIME
    global CP_DELIBERATIONS_STARTED, CP_DELIBERATIONS_COMPLETED
    global CP_DELIBERATION_DURATION, CP_DELIBERATION_CONSENSUS
    global CP_DELIBERATION_SLA_STATUS, CP_DELIBERATION_AGENT_COUNT
    global CP_POLICY_DECISIONS, CP_POLICY_VIOLATIONS, CP_POLICY_CHECK_LATENCY

    noop = _NoOpMetric()
    CP_AGENTS_REGISTERED = noop
    CP_AGENTS_ACTIVE = noop
    CP_AGENT_HEARTBEATS = noop
    CP_AGENT_HEALTH_CHECKS = noop
    CP_TASKS_SUBMITTED = noop
    CP_TASKS_COMPLETED = noop
    CP_TASKS_FAILED = noop
    CP_TASK_DURATION = noop
    CP_TASK_QUEUE_DEPTH = noop
    CP_TASK_WAIT_TIME = noop
    CP_DELIBERATIONS_STARTED = noop
    CP_DELIBERATIONS_COMPLETED = noop
    CP_DELIBERATION_DURATION = noop
    CP_DELIBERATION_CONSENSUS = noop
    CP_DELIBERATION_SLA_STATUS = noop
    CP_DELIBERATION_AGENT_COUNT = noop
    CP_POLICY_DECISIONS = noop
    CP_POLICY_VIOLATIONS = noop
    CP_POLICY_CHECK_LATENCY = noop


class _NoOpMetric:
    """No-op metric for when Prometheus is unavailable."""

    def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
        return self

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, value: float) -> None:
        pass


# ============================================================================
# Recording Functions
# ============================================================================


def record_agent_registered(agent_id: str, success: bool) -> None:
    """Record agent registration event."""
    _init_control_plane_metrics()
    if CP_AGENTS_REGISTERED:
        CP_AGENTS_REGISTERED.labels(status="success" if success else "failure").inc()


def set_active_agents(capability: str, count: int) -> None:
    """Set the number of active agents with a capability."""
    _init_control_plane_metrics()
    if CP_AGENTS_ACTIVE:
        CP_AGENTS_ACTIVE.labels(capability=capability).set(count)


def record_agent_heartbeat(agent_id: str, status: str) -> None:
    """Record agent heartbeat."""
    _init_control_plane_metrics()
    if CP_AGENT_HEARTBEATS:
        CP_AGENT_HEARTBEATS.labels(agent_id=agent_id, status=status).inc()


def record_agent_health_check(status: str) -> None:
    """Record agent health check result."""
    _init_control_plane_metrics()
    if CP_AGENT_HEALTH_CHECKS:
        CP_AGENT_HEALTH_CHECKS.labels(status=status).inc()


def record_task_submitted(task_type: str, priority: str = "normal") -> None:
    """Record task submission."""
    _init_control_plane_metrics()
    if CP_TASKS_SUBMITTED:
        CP_TASKS_SUBMITTED.labels(task_type=task_type, priority=priority).inc()


def record_task_completed(task_type: str, duration_seconds: float) -> None:
    """Record successful task completion."""
    _init_control_plane_metrics()
    if CP_TASKS_COMPLETED:
        CP_TASKS_COMPLETED.labels(task_type=task_type).inc()
    if CP_TASK_DURATION:
        CP_TASK_DURATION.labels(task_type=task_type).observe(duration_seconds)


def record_task_failed(task_type: str, failure_reason: str) -> None:
    """Record task failure."""
    _init_control_plane_metrics()
    if CP_TASKS_FAILED:
        CP_TASKS_FAILED.labels(task_type=task_type, failure_reason=failure_reason).inc()


def set_task_queue_depth(priority: str, depth: int) -> None:
    """Set current task queue depth."""
    _init_control_plane_metrics()
    if CP_TASK_QUEUE_DEPTH:
        CP_TASK_QUEUE_DEPTH.labels(priority=priority).set(depth)


def record_task_wait_time(wait_seconds: float) -> None:
    """Record task wait time in queue."""
    _init_control_plane_metrics()
    if CP_TASK_WAIT_TIME:
        CP_TASK_WAIT_TIME.observe(wait_seconds)


def record_deliberation_started(mode: str = "sync") -> None:
    """Record deliberation start."""
    _init_control_plane_metrics()
    if CP_DELIBERATIONS_STARTED:
        CP_DELIBERATIONS_STARTED.labels(mode=mode).inc()


def record_deliberation_completed(
    success: bool,
    consensus_reached: bool,
    duration_seconds: float,
    consensus_confidence: Optional[float] = None,
    agent_count: Optional[int] = None,
) -> None:
    """Record deliberation completion."""
    _init_control_plane_metrics()

    if CP_DELIBERATIONS_COMPLETED:
        CP_DELIBERATIONS_COMPLETED.labels(
            status="success" if success else "failure",
            consensus_reached=str(consensus_reached).lower(),
        ).inc()

    if CP_DELIBERATION_DURATION:
        CP_DELIBERATION_DURATION.labels(
            consensus_reached=str(consensus_reached).lower(),
        ).observe(duration_seconds)

    if consensus_confidence is not None and CP_DELIBERATION_CONSENSUS:
        CP_DELIBERATION_CONSENSUS.observe(consensus_confidence)

    if agent_count is not None and CP_DELIBERATION_AGENT_COUNT:
        CP_DELIBERATION_AGENT_COUNT.observe(agent_count)


def record_deliberation_sla(level: str) -> None:
    """Record deliberation SLA compliance level.

    Args:
        level: One of "compliant", "warning", "critical", "violated"
    """
    _init_control_plane_metrics()
    if CP_DELIBERATION_SLA_STATUS:
        CP_DELIBERATION_SLA_STATUS.labels(level=level).inc()


def record_policy_decision(
    policy_type: str,
    decision: str,
    latency_seconds: Optional[float] = None,
) -> None:
    """Record policy decision.

    Args:
        policy_type: Type of policy (e.g., "task_dispatch", "agent_access")
        decision: Decision made (e.g., "allow", "deny", "conditional")
        latency_seconds: Optional time taken for policy check
    """
    _init_control_plane_metrics()
    if CP_POLICY_DECISIONS:
        CP_POLICY_DECISIONS.labels(policy_type=policy_type, decision=decision).inc()
    if latency_seconds is not None and CP_POLICY_CHECK_LATENCY:
        CP_POLICY_CHECK_LATENCY.observe(latency_seconds)


def record_policy_violation(policy_type: str, severity: str) -> None:
    """Record policy violation.

    Args:
        policy_type: Type of policy violated
        severity: Severity level (info, warning, critical)
    """
    _init_control_plane_metrics()
    if CP_POLICY_VIOLATIONS:
        CP_POLICY_VIOLATIONS.labels(policy_type=policy_type, severity=severity).inc()


__all__ = [
    # Metrics
    "CP_AGENTS_REGISTERED",
    "CP_AGENTS_ACTIVE",
    "CP_AGENT_HEARTBEATS",
    "CP_AGENT_HEALTH_CHECKS",
    "CP_TASKS_SUBMITTED",
    "CP_TASKS_COMPLETED",
    "CP_TASKS_FAILED",
    "CP_TASK_DURATION",
    "CP_TASK_QUEUE_DEPTH",
    "CP_TASK_WAIT_TIME",
    "CP_DELIBERATIONS_STARTED",
    "CP_DELIBERATIONS_COMPLETED",
    "CP_DELIBERATION_DURATION",
    "CP_DELIBERATION_CONSENSUS",
    "CP_DELIBERATION_SLA_STATUS",
    "CP_DELIBERATION_AGENT_COUNT",
    "CP_POLICY_DECISIONS",
    "CP_POLICY_VIOLATIONS",
    "CP_POLICY_CHECK_LATENCY",
    # Recording functions
    "record_agent_registered",
    "set_active_agents",
    "record_agent_heartbeat",
    "record_agent_health_check",
    "record_task_submitted",
    "record_task_completed",
    "record_task_failed",
    "set_task_queue_depth",
    "record_task_wait_time",
    "record_deliberation_started",
    "record_deliberation_completed",
    "record_deliberation_sla",
    "record_policy_decision",
    "record_policy_violation",
]
