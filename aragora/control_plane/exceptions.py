"""
Control Plane Domain Exceptions.

Provides specific exception types for the control plane to enable
proper error handling and recovery:

- Connection errors: Redis/backend unavailable
- Task lifecycle errors: Not found, invalid state, timeout
- Agent errors: Unavailable, overloaded
- Resource errors: Quota exceeded, rate limited
- Policy errors: Violations, conflicts
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ControlPlaneError(Exception):
    """Base exception for all control plane errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable


# =============================================================================
# Connection Errors (Transient - usually recoverable with retry)
# =============================================================================


class SchedulerConnectionError(ControlPlaneError):
    """Redis or backend connection failure.

    Typically transient and recoverable with retry/backoff.
    """

    def __init__(
        self,
        message: str = "Failed to connect to scheduler backend",
        backend: str = "redis",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details, recoverable=True)
        self.backend = backend


class PolicyCacheConnectionError(ControlPlaneError):
    """Policy cache (Redis) connection failure."""

    def __init__(
        self,
        message: str = "Failed to connect to policy cache",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details, recoverable=True)


# =============================================================================
# Task Lifecycle Errors
# =============================================================================


class TaskNotFoundError(ControlPlaneError):
    """Task doesn't exist in the scheduler."""

    def __init__(
        self,
        task_id: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Task not found: {task_id}",
            details={"task_id": task_id},
            recoverable=False,
        )
        self.task_id = task_id


class TaskTimeoutError(ControlPlaneError):
    """Task exceeded its timeout limit."""

    def __init__(
        self,
        task_id: str,
        timeout_seconds: float,
        elapsed_seconds: float,
        message: Optional[str] = None,
    ):
        super().__init__(
            message
            or f"Task {task_id} timed out after {elapsed_seconds:.1f}s (limit: {timeout_seconds}s)",
            details={
                "task_id": task_id,
                "timeout_seconds": timeout_seconds,
                "elapsed_seconds": elapsed_seconds,
            },
            recoverable=True,  # Can be retried
        )
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class InvalidTaskStateError(ControlPlaneError):
    """Task is in the wrong state for the requested operation.

    For example, trying to complete a task that's already completed,
    or trying to cancel a task that's not running.
    """

    def __init__(
        self,
        task_id: str,
        current_state: str,
        expected_states: list[str],
        operation: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message
            or f"Cannot {operation} task {task_id}: state is '{current_state}', expected one of {expected_states}",
            details={
                "task_id": task_id,
                "current_state": current_state,
                "expected_states": expected_states,
                "operation": operation,
            },
            recoverable=False,
        )
        self.task_id = task_id
        self.current_state = current_state
        self.expected_states = expected_states
        self.operation = operation


class TaskClaimError(ControlPlaneError):
    """Failed to claim a task for execution."""

    def __init__(
        self,
        task_id: str,
        worker_id: str,
        reason: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Worker {worker_id} failed to claim task {task_id}: {reason}",
            details={
                "task_id": task_id,
                "worker_id": worker_id,
                "reason": reason,
            },
            recoverable=True,
        )
        self.task_id = task_id
        self.worker_id = worker_id
        self.reason = reason


# =============================================================================
# Agent Errors
# =============================================================================


class AgentUnavailableError(ControlPlaneError):
    """Agent is not responding or not registered."""

    def __init__(
        self,
        agent_id: str,
        reason: str = "not responding",
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Agent unavailable: {agent_id} ({reason})",
            details={"agent_id": agent_id, "reason": reason},
            recoverable=True,
        )
        self.agent_id = agent_id
        self.reason = reason


class AgentOverloadedError(ControlPlaneError):
    """Agent has too many concurrent tasks."""

    def __init__(
        self,
        agent_id: str,
        current_tasks: int,
        max_tasks: int,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Agent {agent_id} overloaded: {current_tasks}/{max_tasks} tasks",
            details={
                "agent_id": agent_id,
                "current_tasks": current_tasks,
                "max_tasks": max_tasks,
            },
            recoverable=True,
        )
        self.agent_id = agent_id
        self.current_tasks = current_tasks
        self.max_tasks = max_tasks


# =============================================================================
# Resource Errors
# =============================================================================


class ResourceQuotaExceededError(ControlPlaneError):
    """Organization or workspace quota exceeded."""

    def __init__(
        self,
        resource_type: str,
        current_usage: float,
        quota_limit: float,
        org_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Quota exceeded for {resource_type}: {current_usage}/{quota_limit}",
            details={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "quota_limit": quota_limit,
                "org_id": org_id,
                "workspace_id": workspace_id,
            },
            recoverable=False,
        )
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.quota_limit = quota_limit
        self.org_id = org_id
        self.workspace_id = workspace_id


class RateLimitExceededError(ControlPlaneError):
    """Rate limit exceeded for operations."""

    def __init__(
        self,
        operation: str,
        limit: int,
        window_seconds: int,
        retry_after_seconds: Optional[float] = None,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Rate limit exceeded for {operation}: {limit} per {window_seconds}s",
            details={
                "operation": operation,
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after_seconds": retry_after_seconds,
            },
            recoverable=True,
        )
        self.operation = operation
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after_seconds = retry_after_seconds


# =============================================================================
# Policy Errors
# =============================================================================


class PolicyConflictError(ControlPlaneError):
    """Two or more policies conflict with each other."""

    def __init__(
        self,
        policy_ids: list[str],
        conflict_type: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Policy conflict ({conflict_type}): {policy_ids}",
            details={
                "policy_ids": policy_ids,
                "conflict_type": conflict_type,
            },
            recoverable=False,
        )
        self.policy_ids = policy_ids
        self.conflict_type = conflict_type


class PolicyNotFoundError(ControlPlaneError):
    """Policy doesn't exist."""

    def __init__(
        self,
        policy_id: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Policy not found: {policy_id}",
            details={"policy_id": policy_id},
            recoverable=False,
        )
        self.policy_id = policy_id


class PolicyEvaluationError(ControlPlaneError):
    """Error during policy evaluation."""

    def __init__(
        self,
        policy_id: str,
        reason: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Policy evaluation failed for {policy_id}: {reason}",
            details={"policy_id": policy_id, "reason": reason},
            recoverable=True,
        )
        self.policy_id = policy_id
        self.reason = reason


# =============================================================================
# Policy Store Errors
# =============================================================================


class PolicyStoreAccessError(ControlPlaneError):
    """Store read/write failure during policy operations.

    Raised when the policy store (database) is unavailable or
    operations fail due to connection issues.
    """

    def __init__(
        self,
        operation: str,  # "read", "write", "list", "sync"
        reason: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Policy store {operation} failed: {reason}",
            details={"operation": operation, "reason": reason},
            recoverable=True,
        )
        self.operation = operation
        self.reason = reason


class PolicyConversionError(ControlPlaneError):
    """Policy format conversion failed.

    Raised when converting between compliance and control plane
    policy formats fails.
    """

    def __init__(
        self,
        policy_id: str,
        source_format: str,
        target_format: str,
        reason: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message
            or f"Failed to convert policy {policy_id} from {source_format} to {target_format}: {reason}",
            details={
                "policy_id": policy_id,
                "source_format": source_format,
                "target_format": target_format,
                "reason": reason,
            },
            recoverable=False,
        )
        self.policy_id = policy_id
        self.source_format = source_format
        self.target_format = target_format
        self.reason = reason


class CallbackExecutionError(ControlPlaneError):
    """User-provided callback raised exception.

    Raised when a violation callback or conflict callback fails.
    """

    def __init__(
        self,
        callback_type: str,  # "violation", "conflict"
        original_error: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"{callback_type.capitalize()} callback failed: {original_error}",
            details={"callback_type": callback_type, "original_error": original_error},
            recoverable=True,
        )
        self.callback_type = callback_type
        self.original_error = original_error


class MetricsRecordingError(ControlPlaneError):
    """Prometheus metrics recording failed.

    Raised when recording policy metrics fails. This is typically
    non-critical and should not interrupt policy evaluation.
    """

    def __init__(
        self,
        metric_name: str,
        reason: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Failed to record metric {metric_name}: {reason}",
            details={"metric_name": metric_name, "reason": reason},
            recoverable=True,
        )
        self.metric_name = metric_name
        self.reason = reason


# =============================================================================
# Serialization Errors
# =============================================================================


class TaskSerializationError(ControlPlaneError):
    """Failed to serialize or deserialize task data."""

    def __init__(
        self,
        task_id: Optional[str],
        operation: str,  # "serialize" or "deserialize"
        reason: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Failed to {operation} task {task_id or 'unknown'}: {reason}",
            details={
                "task_id": task_id,
                "operation": operation,
                "reason": reason,
            },
            recoverable=False,
        )
        self.task_id = task_id
        self.operation = operation
        self.reason = reason


# =============================================================================
# Region Errors
# =============================================================================


class RegionUnavailableError(ControlPlaneError):
    """Target region is not available."""

    def __init__(
        self,
        region_id: str,
        reason: str = "region offline",
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Region unavailable: {region_id} ({reason})",
            details={"region_id": region_id, "reason": reason},
            recoverable=True,
        )
        self.region_id = region_id
        self.reason = reason


class RegionRoutingError(ControlPlaneError):
    """Failed to route task to appropriate region."""

    def __init__(
        self,
        task_id: str,
        target_region: Optional[str],
        available_regions: list[str],
        reason: str,
        message: Optional[str] = None,
    ):
        super().__init__(
            message or f"Cannot route task {task_id} to region {target_region}: {reason}",
            details={
                "task_id": task_id,
                "target_region": target_region,
                "available_regions": available_regions,
                "reason": reason,
            },
            recoverable=True,
        )
        self.task_id = task_id
        self.target_region = target_region
        self.available_regions = available_regions
        self.reason = reason


__all__ = [
    # Base
    "ControlPlaneError",
    # Connection
    "SchedulerConnectionError",
    "PolicyCacheConnectionError",
    # Task lifecycle
    "TaskNotFoundError",
    "TaskTimeoutError",
    "InvalidTaskStateError",
    "TaskClaimError",
    # Agent
    "AgentUnavailableError",
    "AgentOverloadedError",
    # Resource
    "ResourceQuotaExceededError",
    "RateLimitExceededError",
    # Policy
    "PolicyConflictError",
    "PolicyNotFoundError",
    "PolicyEvaluationError",
    # Policy store
    "PolicyStoreAccessError",
    "PolicyConversionError",
    "CallbackExecutionError",
    "MetricsRecordingError",
    # Serialization
    "TaskSerializationError",
    # Region
    "RegionUnavailableError",
    "RegionRoutingError",
]
