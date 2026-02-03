"""
Core utilities and shared dependencies for workflow handlers.

This module contains shared imports, utilities, and state management used across
all workflow handler submodules.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal, cast

from aragora.server.http_utils import run_async as _run_async

from aragora.workflow.types import (
    WorkflowDefinition,
    WorkflowCategory,
    StepDefinition,
    StepResult,
    TransitionRule,
)
from aragora.workflow.engine import WorkflowEngine
from aragora.workflow.persistent_store import get_workflow_store, PersistentWorkflowStore
from aragora.audit.unified import audit_data

# Sentinel type for unauthenticated requests
_UnauthenticatedSentinel = Literal["unauthenticated"]

# RBAC availability check
try:
    import aragora.rbac  # noqa: F401

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False

# Metrics imports
try:
    from aragora.observability.metrics import record_rbac_check, track_handler

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

    def record_rbac_check(*args: Any, **kwargs: Any) -> None:
        pass

    def track_handler(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        def decorator(fn: Any) -> Any:
            return fn

        return decorator


logger = logging.getLogger(__name__)


def _step_result_to_dict(step: StepResult) -> dict[str, Any]:
    """Convert a StepResult dataclass to a JSON-serializable dictionary.

    StepResult contains datetime fields that need to be converted to ISO format
    strings for JSON serialization.
    """
    return {
        "step_id": step.step_id,
        "step_name": step.step_name,
        "status": step.status.value if hasattr(step.status, "value") else str(step.status),
        "started_at": step.started_at.isoformat() if step.started_at else None,
        "completed_at": step.completed_at.isoformat() if step.completed_at else None,
        "duration_ms": step.duration_ms,
        "output": step.output,
        "error": step.error,
        "metrics": step.metrics,
        "retry_count": step.retry_count,
    }


# =============================================================================
# Persistent Storage (SQLite-backed)
# =============================================================================


def _get_store() -> PersistentWorkflowStore:
    """Get the persistent workflow store.

    Note: get_workflow_store() returns WorkflowStoreType which is a Union of
    PersistentWorkflowStore and PostgresWorkflowStore. Both have the same
    interface, so we cast to the base type for consistency.
    """
    try:
        pkg = sys.modules.get("aragora.server.handlers.workflows")
        override = getattr(pkg, "_get_store", None) if pkg is not None else None
        if override is not None and override is not _get_store:
            return cast(PersistentWorkflowStore, override())
    except Exception as e:
        logger.debug("Failed to resolve _get_store override: %s", e)
    return cast(PersistentWorkflowStore, get_workflow_store())


def _call_store_method(result: Any) -> Any:
    """Handle both sync and async store method results.

    PostgresWorkflowStore methods are async while PersistentWorkflowStore methods
    are sync. This helper ensures we properly await async results when needed.
    """
    import asyncio

    if asyncio.iscoroutine(result):
        return _run_async(result)
    return result


_engine: WorkflowEngine | None = None


def _get_engine() -> WorkflowEngine:
    """Lazily initialize the workflow engine to avoid import-time side effects."""
    global _engine
    try:
        pkg = sys.modules.get("aragora.server.handlers.workflows")
        override = getattr(pkg, "_engine", None) if pkg is not None else None
        if override is not None and override is not _engine:
            return cast(WorkflowEngine, override)
    except Exception as e:
        logger.debug("Failed to resolve _engine override: %s", e)
    if _engine is None:
        _engine = WorkflowEngine()
    return _engine


# In-memory template store for built-in and YAML templates
class _TemplateStore:
    """In-memory storage for workflow templates."""

    def __init__(self) -> None:
        self.templates: dict[str, WorkflowDefinition] = {}


_store = _TemplateStore()


# Re-export commonly used items for convenience
__all__ = [
    # Logging
    "logger",
    # Utilities
    "_step_result_to_dict",
    "_get_store",
    "_call_store_method",
    "_get_engine",
    "_store",
    "_run_async",
    # Types
    "WorkflowDefinition",
    "WorkflowCategory",
    "StepDefinition",
    "StepResult",
    "TransitionRule",
    "PersistentWorkflowStore",
    "_UnauthenticatedSentinel",
    # RBAC
    "RBAC_AVAILABLE",
    "METRICS_AVAILABLE",
    "record_rbac_check",
    "track_handler",
    "audit_data",
]
