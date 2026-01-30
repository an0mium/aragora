"""
Workflow template metrics.

Provides Prometheus metrics for tracking workflow template operations including:
- Post-debate workflow triggers
- Workflow template creation
- Workflow template execution with latency tracking
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Post-debate workflow trigger metrics
WORKFLOW_TRIGGERS: Any = None

# Workflow template metrics
WORKFLOW_TEMPLATES_CREATED: Any = None
WORKFLOW_TEMPLATE_EXECUTIONS: Any = None
WORKFLOW_TEMPLATE_EXECUTION_LATENCY: Any = None

_initialized = False


def init_workflow_metrics() -> None:
    """Initialize workflow template metrics."""
    global _initialized
    global WORKFLOW_TRIGGERS
    global WORKFLOW_TEMPLATES_CREATED, WORKFLOW_TEMPLATE_EXECUTIONS
    global WORKFLOW_TEMPLATE_EXECUTION_LATENCY

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Histogram

        WORKFLOW_TRIGGERS = Counter(
            "aragora_workflow_triggers_total",
            "Post-debate workflow triggers",
            ["status"],
        )

        WORKFLOW_TEMPLATES_CREATED = Counter(
            "aragora_workflow_templates_created_total",
            "Workflow templates created",
            ["pattern", "template_id"],
        )

        WORKFLOW_TEMPLATE_EXECUTIONS = Counter(
            "aragora_workflow_template_executions_total",
            "Workflow template executions",
            ["pattern", "status"],
        )

        WORKFLOW_TEMPLATE_EXECUTION_LATENCY = Histogram(
            "aragora_workflow_template_execution_latency_seconds",
            "Workflow template execution latency",
            ["pattern"],
            buckets=[0.5, 1, 2.5, 5, 10, 30, 60, 120, 300],
        )

        _initialized = True
        logger.debug("Workflow metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True
    except Exception as e:
        logger.warning(f"Failed to initialize workflow metrics: {e}")
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global WORKFLOW_TRIGGERS
    global WORKFLOW_TEMPLATES_CREATED, WORKFLOW_TEMPLATE_EXECUTIONS
    global WORKFLOW_TEMPLATE_EXECUTION_LATENCY

    noop = NoOpMetric()
    WORKFLOW_TRIGGERS = noop
    WORKFLOW_TEMPLATES_CREATED = noop
    WORKFLOW_TEMPLATE_EXECUTIONS = noop
    WORKFLOW_TEMPLATE_EXECUTION_LATENCY = noop


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_workflow_metrics()


# =============================================================================
# Post-Debate Workflow Recording Functions
# =============================================================================


def record_workflow_trigger(success: bool) -> None:
    """Record a post-debate workflow trigger.

    Args:
        success: Whether the workflow trigger succeeded
    """
    _ensure_init()
    status = "success" if success else "error"
    WORKFLOW_TRIGGERS.labels(status=status).inc()


# =============================================================================
# Workflow Template Recording Functions
# =============================================================================


def record_workflow_template_created(pattern: str, template_id: str) -> None:
    """Record a workflow template creation.

    Args:
        pattern: The workflow pattern (e.g., debate_followup, periodic_sync)
        template_id: Unique identifier for the template
    """
    _ensure_init()
    WORKFLOW_TEMPLATES_CREATED.labels(pattern=pattern, template_id=template_id).inc()


def record_workflow_template_execution(
    pattern: str,
    success: bool,
    latency_seconds: float,
) -> None:
    """Record a workflow template execution.

    Args:
        pattern: The workflow pattern being executed
        success: Whether the execution succeeded
        latency_seconds: Time taken for execution in seconds
    """
    _ensure_init()
    status = "success" if success else "error"
    WORKFLOW_TEMPLATE_EXECUTIONS.labels(pattern=pattern, status=status).inc()
    WORKFLOW_TEMPLATE_EXECUTION_LATENCY.labels(pattern=pattern).observe(latency_seconds)


@contextmanager
def track_workflow_template_execution(pattern: str) -> Generator[None, None, None]:
    """Context manager to track workflow template execution.

    Records both the execution count and latency automatically.

    Args:
        pattern: The workflow pattern being executed

    Example:
        with track_workflow_template_execution("debate_followup"):
            await execute_workflow(template)
    """
    _ensure_init()
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_workflow_template_execution(pattern, success, latency)


__all__ = [
    # Metrics
    "WORKFLOW_TRIGGERS",
    "WORKFLOW_TEMPLATES_CREATED",
    "WORKFLOW_TEMPLATE_EXECUTIONS",
    "WORKFLOW_TEMPLATE_EXECUTION_LATENCY",
    # Functions
    "init_workflow_metrics",
    "record_workflow_trigger",
    "record_workflow_template_created",
    "record_workflow_template_execution",
    "track_workflow_template_execution",
]
