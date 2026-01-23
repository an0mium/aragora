"""Task lifecycle event emissions for NotificationDispatcher.

This module provides helper functions to emit task lifecycle notifications
to the NotificationDispatcher. These events are emitted when tasks are
submitted, claimed, completed, or failed.

Usage:
    from aragora.control_plane.task_events import (
        emit_task_submitted,
        emit_task_completed,
        set_task_event_dispatcher,
    )

    # Set a custom dispatcher (optional - will use default if not set)
    set_task_event_dispatcher(dispatcher)

    # Emit task events
    await emit_task_submitted(task_id, task_type, priority, workspace_id)
    await emit_task_completed(task_id, task_type, agent_id, duration_seconds)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.control_plane.notifications import NotificationDispatcher

from aragora.control_plane.channels import NotificationEventType, NotificationPriority

logger = logging.getLogger(__name__)

_dispatcher: Optional["NotificationDispatcher"] = None


def set_task_event_dispatcher(dispatcher: "NotificationDispatcher") -> None:
    """Set the dispatcher for task events.

    Args:
        dispatcher: NotificationDispatcher instance to use for task events
    """
    global _dispatcher
    _dispatcher = dispatcher
    logger.info("task_event_dispatcher_set")


def get_task_event_dispatcher() -> Optional["NotificationDispatcher"]:
    """Get the current task event dispatcher.

    Returns the configured dispatcher, or falls back to the default
    notification dispatcher if none was explicitly set.

    Returns:
        NotificationDispatcher instance or None if not available
    """
    global _dispatcher
    if _dispatcher is None:
        from aragora.control_plane.notifications import get_default_notification_dispatcher

        return get_default_notification_dispatcher()
    return _dispatcher


def _map_priority(task_priority: str) -> NotificationPriority:
    """Map task priority string to notification priority.

    Args:
        task_priority: Task priority as string (urgent, high, normal, low)

    Returns:
        Corresponding NotificationPriority
    """
    mapping = {
        "urgent": NotificationPriority.URGENT,
        "high": NotificationPriority.HIGH,
        "normal": NotificationPriority.NORMAL,
        "low": NotificationPriority.LOW,
    }
    return mapping.get(task_priority.lower(), NotificationPriority.NORMAL)


async def emit_task_submitted(
    task_id: str,
    task_type: str,
    priority: str,
    workspace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit TASK_SUBMITTED notification.

    Called when a new task is submitted to the scheduler.

    Args:
        task_id: Unique task identifier
        task_type: Type of the task
        priority: Task priority level
        workspace_id: Optional workspace for filtering notifications
        metadata: Additional task metadata
    """
    dispatcher = get_task_event_dispatcher()
    if not dispatcher:
        return

    try:
        await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_SUBMITTED,
            title=f"Task Submitted: {task_type}",
            body=f"New {priority} priority task queued for execution",
            priority=_map_priority(priority),
            workspace_id=workspace_id,
            metadata={
                "task_id": task_id,
                "task_type": task_type,
                "priority": priority,
                **(metadata or {}),
            },
        )
    except Exception as e:
        logger.warning(
            "task_event_emission_failed", extra={"event": "task_submitted", "error": str(e)}
        )


async def emit_task_claimed(
    task_id: str,
    task_type: str,
    agent_id: str,
    workspace_id: Optional[str] = None,
) -> None:
    """Emit TASK_CLAIMED notification.

    Called when an agent claims a task for execution.

    Args:
        task_id: Unique task identifier
        task_type: Type of the task
        agent_id: ID of the agent that claimed the task
        workspace_id: Optional workspace for filtering notifications
    """
    dispatcher = get_task_event_dispatcher()
    if not dispatcher:
        return

    try:
        await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_CLAIMED,
            title=f"Task Claimed: {task_type}",
            body=f"Agent `{agent_id}` claimed task `{task_id[:8]}...` for execution",
            priority=NotificationPriority.NORMAL,
            workspace_id=workspace_id,
            metadata={
                "task_id": task_id,
                "task_type": task_type,
                "agent_id": agent_id,
            },
        )
    except Exception as e:
        logger.warning(
            "task_event_emission_failed", extra={"event": "task_claimed", "error": str(e)}
        )


async def emit_task_completed(
    task_id: str,
    task_type: str,
    agent_id: str,
    duration_seconds: float,
    workspace_id: Optional[str] = None,
) -> None:
    """Emit TASK_COMPLETED notification.

    Called when a task is successfully completed.

    Args:
        task_id: Unique task identifier
        task_type: Type of the task
        agent_id: ID of the agent that completed the task
        duration_seconds: Time taken to complete the task
        workspace_id: Optional workspace for filtering notifications
    """
    dispatcher = get_task_event_dispatcher()
    if not dispatcher:
        return

    try:
        await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_COMPLETED,
            title=f"Task Completed: {task_type}",
            body=f"Task `{task_id[:8]}...` completed by agent `{agent_id}` in {duration_seconds:.1f}s",
            priority=NotificationPriority.NORMAL,
            workspace_id=workspace_id,
            metadata={
                "task_id": task_id,
                "task_type": task_type,
                "agent_id": agent_id,
                "duration_seconds": duration_seconds,
            },
        )
    except Exception as e:
        logger.warning(
            "task_event_emission_failed", extra={"event": "task_completed", "error": str(e)}
        )


async def emit_task_failed(
    task_id: str,
    task_type: str,
    agent_id: Optional[str],
    error: str,
    will_retry: bool,
    workspace_id: Optional[str] = None,
) -> None:
    """Emit TASK_FAILED notification.

    Called when a task fails execution. Uses URGENT priority for failures
    that won't be retried.

    Args:
        task_id: Unique task identifier
        task_type: Type of the task
        agent_id: ID of the agent that was executing (None if unassigned)
        error: Error message describing the failure
        will_retry: Whether the task will be retried
        workspace_id: Optional workspace for filtering notifications
    """
    dispatcher = get_task_event_dispatcher()
    if not dispatcher:
        return

    try:
        agent_info = f" by agent `{agent_id}`" if agent_id else ""
        retry_info = " (will retry)" if will_retry else " (max retries exceeded)"

        await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_FAILED,
            title=f"Task Failed: {task_type}",
            body=f"Task `{task_id[:8]}...` failed{agent_info}{retry_info}: {error[:200]}",
            priority=NotificationPriority.URGENT if not will_retry else NotificationPriority.HIGH,
            workspace_id=workspace_id,
            metadata={
                "task_id": task_id,
                "task_type": task_type,
                "agent_id": agent_id,
                "error": error,
                "will_retry": will_retry,
            },
        )
    except Exception as e:
        logger.warning(
            "task_event_emission_failed", extra={"event": "task_failed", "error": str(e)}
        )


async def emit_task_timeout(
    task_id: str,
    task_type: str,
    agent_id: Optional[str],
    elapsed_seconds: float,
    timeout_seconds: float,
    workspace_id: Optional[str] = None,
) -> None:
    """Emit TASK_TIMEOUT notification.

    Called when a task exceeds its execution timeout.

    Args:
        task_id: Unique task identifier
        task_type: Type of the task
        agent_id: ID of the agent that was executing (None if unassigned)
        elapsed_seconds: Time elapsed before timeout
        timeout_seconds: Configured timeout threshold
        workspace_id: Optional workspace for filtering notifications
    """
    dispatcher = get_task_event_dispatcher()
    if not dispatcher:
        return

    try:
        agent_info = f" (agent: {agent_id})" if agent_id else ""

        await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_TIMEOUT,
            title=f"Task Timeout: {task_type}",
            body=f"Task `{task_id[:8]}...` timed out after {elapsed_seconds:.0f}s{agent_info} (limit: {timeout_seconds:.0f}s)",
            priority=NotificationPriority.HIGH,
            workspace_id=workspace_id,
            metadata={
                "task_id": task_id,
                "task_type": task_type,
                "agent_id": agent_id,
                "elapsed_seconds": elapsed_seconds,
                "timeout_seconds": timeout_seconds,
            },
        )
    except Exception as e:
        logger.warning(
            "task_event_emission_failed", extra={"event": "task_timeout", "error": str(e)}
        )


async def emit_task_retried(
    task_id: str,
    task_type: str,
    attempt: int,
    max_retries: int,
    workspace_id: Optional[str] = None,
) -> None:
    """Emit TASK_RETRIED notification.

    Called when a task is requeued for retry after a failure.

    Args:
        task_id: Unique task identifier
        task_type: Type of the task
        attempt: Current retry attempt number
        max_retries: Maximum allowed retries
        workspace_id: Optional workspace for filtering notifications
    """
    dispatcher = get_task_event_dispatcher()
    if not dispatcher:
        return

    try:
        await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_RETRIED,
            title=f"Task Retried: {task_type}",
            body=f"Task `{task_id[:8]}...` requeued for retry (attempt {attempt}/{max_retries})",
            priority=NotificationPriority.NORMAL,
            workspace_id=workspace_id,
            metadata={
                "task_id": task_id,
                "task_type": task_type,
                "attempt": attempt,
                "max_retries": max_retries,
            },
        )
    except Exception as e:
        logger.warning(
            "task_event_emission_failed", extra={"event": "task_retried", "error": str(e)}
        )


async def emit_task_cancelled(
    task_id: str,
    task_type: str,
    reason: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> None:
    """Emit TASK_CANCELLED notification.

    Called when a task is cancelled before completion.

    Args:
        task_id: Unique task identifier
        task_type: Type of the task
        reason: Optional reason for cancellation
        workspace_id: Optional workspace for filtering notifications
    """
    dispatcher = get_task_event_dispatcher()
    if not dispatcher:
        return

    try:
        reason_info = f": {reason}" if reason else ""

        await dispatcher.dispatch(
            event_type=NotificationEventType.TASK_CANCELLED,
            title=f"Task Cancelled: {task_type}",
            body=f"Task `{task_id[:8]}...` was cancelled{reason_info}",
            priority=NotificationPriority.NORMAL,
            workspace_id=workspace_id,
            metadata={
                "task_id": task_id,
                "task_type": task_type,
                "reason": reason,
            },
        )
    except Exception as e:
        logger.warning(
            "task_event_emission_failed", extra={"event": "task_cancelled", "error": str(e)}
        )


__all__ = [
    # Dispatcher management
    "set_task_event_dispatcher",
    "get_task_event_dispatcher",
    # Event emissions
    "emit_task_submitted",
    "emit_task_claimed",
    "emit_task_completed",
    "emit_task_failed",
    "emit_task_timeout",
    "emit_task_retried",
    "emit_task_cancelled",
]
