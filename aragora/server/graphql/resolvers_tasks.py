"""
GraphQL Resolvers for Task and System operations.

Contains query, mutation, and subscription resolvers for tasks
and system health, plus transform functions for task data.

Separated from resolvers.py for maintainability.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, cast

from .resolvers import (
    ResolverContext,
    ResolverResult,
    _normalize_health_status,
    _normalize_priority,
    _normalize_task_status,
    _to_iso_datetime,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Task Transform Functions
# =============================================================================


def _transform_task(task: Any) -> dict[str, Any]:
    """Transform internal task format to GraphQL format."""
    if isinstance(task, dict):
        return {
            "id": task.get("id", ""),
            "type": task.get("task_type") or task.get("type", ""),
            "status": _normalize_task_status(task.get("status")),
            "priority": _normalize_priority(task.get("priority")),
            "assignedAgent": task.get("assigned_agent"),
            "result": task.get("result"),
            "createdAt": _to_iso_datetime(task.get("created_at")),
            "completedAt": _to_iso_datetime(task.get("completed_at")),
            "payload": task.get("payload"),
            "metadata": task.get("metadata"),
        }

    # Handle object-based task
    status = getattr(task, "status", None)
    if hasattr(status, "value"):
        status = status.value

    priority = getattr(task, "priority", None)
    if hasattr(priority, "name"):
        priority = priority.name

    return {
        "id": getattr(task, "id", ""),
        "type": getattr(task, "task_type", ""),
        "status": _normalize_task_status(status),
        "priority": _normalize_priority(priority),
        "assignedAgent": getattr(task, "assigned_agent", None),
        "result": getattr(task, "result", None),
        "createdAt": _to_iso_datetime(getattr(task, "created_at", None)),
        "completedAt": _to_iso_datetime(getattr(task, "completed_at", None)),
        "payload": getattr(task, "payload", None),
        "metadata": getattr(task, "metadata", None),
    }


# =============================================================================
# Task Query Resolvers
# =============================================================================


class TaskQueryResolvers:
    """Query resolvers for task and system operations."""

    @staticmethod
    async def resolve_task(
        ctx: ResolverContext,
        id: str,
    ) -> ResolverResult:
        """Resolve a single task by ID.

        Args:
            ctx: Resolver context
            id: Task ID

        Returns:
            ResolverResult with task data
        """
        try:
            coordinator: Any = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            task = await coordinator.get_task(id)
            if not task:
                return ResolverResult(errors=[f"Task not found: {id}"])

            data = _transform_task(task)
            return ResolverResult(data=data)

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Data access or transformation errors
            logger.exception(f"Error resolving task {id}: {e}")
            return ResolverResult(errors=[f"Failed to resolve task: {e}"])

    @staticmethod
    async def resolve_tasks(
        ctx: ResolverContext,
        status: str | None = None,
        type: str | None = None,
        limit: int = 20,
    ) -> ResolverResult:
        """Resolve a list of tasks with optional filtering.

        Args:
            ctx: Resolver context
            status: Optional status filter
            type: Optional task type filter
            limit: Maximum results

        Returns:
            ResolverResult with TaskConnection data
        """
        try:
            coordinator: Any = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            # Get tasks based on status filter
            from aragora.control_plane.scheduler import TaskStatus as CPTaskStatus

            tasks: list[Any] = []
            if status:
                try:
                    cp_status = CPTaskStatus(status.lower())
                    tasks = await coordinator._scheduler.list_by_status(cp_status, limit=limit)
                except ValueError:
                    pass
            else:
                # Get pending and running tasks
                pending = await coordinator._scheduler.list_by_status(
                    CPTaskStatus.PENDING, limit=limit
                )
                running = await coordinator._scheduler.list_by_status(
                    CPTaskStatus.RUNNING, limit=limit
                )
                tasks = list(running) + list(pending)

            # Filter by type if provided
            if type:
                tasks = [t for t in tasks if t.task_type == type]

            transformed = [_transform_task(t) for t in tasks[:limit]]

            return ResolverResult(
                data={
                    "tasks": transformed,
                    "total": len(transformed),
                    "hasMore": len(tasks) > limit,
                }
            )

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Data access or transformation errors
            logger.exception(f"Error resolving tasks: {e}")
            return ResolverResult(errors=[f"Failed to resolve tasks: {e}"])

    @staticmethod
    async def resolve_system_health(ctx: ResolverContext) -> ResolverResult:
        """Resolve system health status.

        Args:
            ctx: Resolver context

        Returns:
            ResolverResult with SystemHealth data
        """
        try:
            import time

            coordinator = ctx.server_context.get("control_plane_coordinator")

            components = []
            overall_status = "HEALTHY"

            # Check coordinator
            if coordinator:
                components.append(
                    {
                        "name": "Coordinator",
                        "status": "HEALTHY",
                        "latencyMs": 0,
                        "error": None,
                    }
                )
            else:
                components.append(
                    {
                        "name": "Coordinator",
                        "status": "UNHEALTHY",
                        "latencyMs": None,
                        "error": "Not initialized",
                    }
                )
                overall_status = "DEGRADED"

            # Check storage
            storage = ctx.server_context.get("storage")
            if storage:
                components.append(
                    {
                        "name": "Storage",
                        "status": "HEALTHY",
                        "latencyMs": 5,
                        "error": None,
                    }
                )
            else:
                components.append(
                    {
                        "name": "Storage",
                        "status": "UNHEALTHY",
                        "latencyMs": None,
                        "error": "Not available",
                    }
                )
                overall_status = "DEGRADED"

            # Check ELO system
            elo_system = ctx.server_context.get("elo_system")
            if elo_system:
                components.append(
                    {
                        "name": "ELO System",
                        "status": "HEALTHY",
                        "latencyMs": 2,
                        "error": None,
                    }
                )

            start_time_val = ctx.server_context.get("_start_time", time.time())
            start_time = cast(float, start_time_val) if start_time_val is not None else time.time()
            uptime = int(time.time() - start_time)

            return ResolverResult(
                data={
                    "status": overall_status,
                    "uptimeSeconds": uptime,
                    "version": "2.1.0",
                    "components": components,
                }
            )

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Data access or transformation errors
            logger.exception(f"Error resolving system health: {e}")
            return ResolverResult(errors=[f"Failed to resolve system health: {e}"])

    @staticmethod
    async def resolve_stats(ctx: ResolverContext) -> ResolverResult:
        """Resolve system statistics.

        Args:
            ctx: Resolver context

        Returns:
            ResolverResult with SystemStats data
        """
        try:
            coordinator: Any = ctx.server_context.get("control_plane_coordinator")

            stats: dict[str, Any] = {
                "activeJobs": 0,
                "queuedJobs": 0,
                "completedJobsToday": 0,
                "availableAgents": 0,
                "busyAgents": 0,
                "totalAgents": 0,
                "documentsProcessedToday": 0,
            }

            if coordinator:
                cp_stats = await coordinator.get_stats()
                scheduler_stats = cp_stats.get("scheduler", {})
                registry_stats = cp_stats.get("registry", {})
                by_status = scheduler_stats.get("by_status", {})

                stats["activeJobs"] = by_status.get("running", 0)
                stats["queuedJobs"] = by_status.get("pending", 0)
                stats["completedJobsToday"] = by_status.get("completed", 0)
                stats["availableAgents"] = registry_stats.get("available_agents", 0)
                stats["busyAgents"] = registry_stats.get("by_status", {}).get("busy", 0)
                stats["totalAgents"] = registry_stats.get("total_agents", 0)
                stats["documentsProcessedToday"] = scheduler_stats.get("by_type", {}).get(
                    "document_processing", 0
                )

            return ResolverResult(data=stats)

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Data access or transformation errors
            logger.exception(f"Error resolving stats: {e}")
            return ResolverResult(errors=[f"Failed to resolve stats: {e}"])


# =============================================================================
# Task Mutation Resolvers
# =============================================================================


class TaskMutationResolvers:
    """Mutation resolvers for task operations."""

    @staticmethod
    async def resolve_submit_task(
        ctx: ResolverContext,
        input: dict[str, Any],
    ) -> ResolverResult:
        """Submit a new task to the control plane.

        Args:
            ctx: Resolver context
            input: SubmitTaskInput fields

        Returns:
            ResolverResult with created task data
        """
        try:
            task_type = input.get("taskType")
            if not task_type:
                return ResolverResult(errors=["Task type is required"])

            coordinator: Any = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            from aragora.control_plane.scheduler import TaskPriority

            priority_str = input.get("priority", "NORMAL")
            try:
                priority = TaskPriority[priority_str.upper()]
            except KeyError:
                priority = TaskPriority.NORMAL

            task_id = await coordinator.submit_task(
                task_type=task_type,
                payload=input.get("payload", {}),
                required_capabilities=input.get("requiredCapabilities", []),
                priority=priority,
                timeout_seconds=input.get("timeoutSeconds"),
                metadata=input.get("metadata", {}),
            )

            # Get created task
            task = await coordinator.get_task(task_id)
            if task:
                return ResolverResult(data=_transform_task(task))

            return ResolverResult(
                data={
                    "id": task_id,
                    "type": task_type,
                    "status": "PENDING",
                    "priority": priority_str,
                    "assignedAgent": None,
                    "result": None,
                    "createdAt": datetime.now().isoformat(),
                    "completedAt": None,
                    "payload": input.get("payload"),
                    "metadata": input.get("metadata"),
                }
            )

        except (KeyError, AttributeError, TypeError, ValueError, RuntimeError) as e:
            # Task creation, scheduling, or transformation errors
            logger.exception(f"Error submitting task: {e}")
            return ResolverResult(errors=[f"Failed to submit task: {e}"])

    @staticmethod
    async def resolve_cancel_task(
        ctx: ResolverContext,
        id: str,
    ) -> ResolverResult:
        """Cancel a pending or running task.

        Args:
            ctx: Resolver context
            id: Task ID

        Returns:
            ResolverResult with cancelled task data
        """
        try:
            coordinator: Any = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            success = await coordinator.cancel_task(id)
            if not success:
                return ResolverResult(errors=[f"Task not found or already completed: {id}"])

            # Get updated task
            task = await coordinator.get_task(id)
            if task:
                return ResolverResult(data=_transform_task(task))

            return ResolverResult(
                data={
                    "id": id,
                    "status": "CANCELLED",
                }
            )

        except (KeyError, AttributeError, TypeError, ValueError, RuntimeError) as e:
            # Task access, cancellation, or transformation errors
            logger.exception(f"Error cancelling task: {e}")
            return ResolverResult(errors=[f"Failed to cancel task: {e}"])


# =============================================================================
# Task Subscription Resolvers
# =============================================================================


class TaskSubscriptionResolvers:
    """Subscription resolvers for task operations."""

    @staticmethod
    async def subscribe_task_updates(
        ctx: ResolverContext,
        task_id: str | None = None,
    ):
        """Subscribe to task updates.

        Args:
            ctx: Resolver context
            task_id: Optional specific task ID to subscribe to

        Yields:
            TaskEvent data
        """
        ws_manager = ctx.server_context.get("ws_manager")
        if not ws_manager:
            raise RuntimeError("WebSocket manager not available")

        queue: asyncio.Queue = asyncio.Queue()

        try:
            while True:
                event = await queue.get()
                yield {
                    "type": event.get("type", "update"),
                    "taskId": event.get("task_id", task_id),
                    "data": event.get("data", {}),
                    "timestamp": datetime.now().isoformat(),
                }
        except asyncio.CancelledError:
            pass
