"""
MCP Control Plane Tools.

Provides programmatic access to control plane operations:
- Agent management (register, unregister, list, health)
- Task operations (submit, status, cancel)
- System health and resource utilization
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global coordinator instance (lazily initialized)
_coordinator: Optional[Any] = None


async def _get_coordinator() -> Any:
    """Get or create the control plane coordinator."""
    global _coordinator

    if _coordinator is not None:
        return _coordinator

    try:
        from aragora.control_plane.coordinator import ControlPlaneCoordinator

        _coordinator = await ControlPlaneCoordinator.create()
        return _coordinator
    except Exception as e:
        logger.warning(f"Could not create coordinator: {e}")
        return None


# =============================================================================
# Agent Operations
# =============================================================================


async def register_agent_tool(
    agent_id: str,
    capabilities: str = "debate",
    model: str = "unknown",
    provider: str = "unknown",
) -> Dict[str, Any]:
    """
    Register a new agent with the control plane.

    Args:
        agent_id: Unique identifier for the agent
        capabilities: Comma-separated capabilities (e.g., "debate,code,analysis")
        model: Model name (e.g., "claude-3-opus", "gpt-4")
        provider: Provider name (e.g., "anthropic", "openai")

    Returns:
        Dict with registration result and agent info
    """
    if not agent_id:
        return {"error": "agent_id is required"}

    cap_list = [c.strip() for c in capabilities.split(",") if c.strip()]
    if not cap_list:
        cap_list = ["debate"]

    coordinator = await _get_coordinator()
    if not coordinator:
        return {
            "error": "Control plane not available",
            "note": "Redis may not be running or control plane not initialized",
        }

    try:
        agent = await coordinator.register_agent(
            agent_id=agent_id,
            capabilities=cap_list,
            model=model,
            provider=provider,
        )
        return {
            "success": True,
            "agent": {
                "agent_id": agent.agent_id,
                "capabilities": list(agent.capabilities),
                "status": agent.status.value,
                "model": agent.model,
                "provider": agent.provider,
                "registered_at": agent.registered_at,
            },
        }
    except Exception as e:
        return {"error": f"Failed to register agent: {e}"}


async def unregister_agent_tool(agent_id: str) -> Dict[str, Any]:
    """
    Unregister an agent from the control plane.

    Args:
        agent_id: Agent to unregister

    Returns:
        Dict with unregistration result
    """
    if not agent_id:
        return {"error": "agent_id is required"}

    coordinator = await _get_coordinator()
    if not coordinator:
        return {"error": "Control plane not available"}

    try:
        success = await coordinator.unregister_agent(agent_id)
        return {
            "success": success,
            "agent_id": agent_id,
            "message": "Agent unregistered" if success else "Agent not found",
        }
    except Exception as e:
        return {"error": f"Failed to unregister agent: {e}"}


async def list_registered_agents_tool(
    capability: str = "",
    only_available: bool = True,
) -> Dict[str, Any]:
    """
    List all registered agents with health status.

    Args:
        capability: Optional capability filter (e.g., "debate", "code")
        only_available: Only return available agents (default: True)

    Returns:
        Dict with list of agents and their status
    """
    coordinator = await _get_coordinator()
    if not coordinator:
        # Fallback to static list
        return {
            "agents": [
                {
                    "agent_id": "anthropic-api",
                    "status": "unknown",
                    "capabilities": ["debate", "code"],
                },
                {"agent_id": "openai-api", "status": "unknown", "capabilities": ["debate", "code"]},
                {"agent_id": "grok", "status": "unknown", "capabilities": ["debate"]},
            ],
            "count": 3,
            "note": "Control plane not available - showing static fallback",
        }

    try:
        if capability:
            agents = await coordinator.list_agents(
                capability=capability,
                only_available=only_available,
            )
        else:
            agents = await coordinator.list_agents(only_available=only_available)

        return {
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "status": a.status.value,
                    "capabilities": list(a.capabilities),
                    "model": a.model,
                    "provider": a.provider,
                    "is_available": a.is_available(),
                    "tasks_completed": a.tasks_completed,
                    "tasks_failed": a.tasks_failed,
                    "avg_latency_ms": round(a.avg_latency_ms, 2),
                    "region_id": a.region_id,
                }
                for a in agents
            ],
            "count": len(agents),
            "filter": {"capability": capability, "only_available": only_available},
        }
    except Exception as e:
        return {"error": f"Failed to list agents: {e}"}


async def get_agent_health_tool(agent_id: str) -> Dict[str, Any]:
    """
    Get detailed health status for a specific agent.

    Args:
        agent_id: Agent to query

    Returns:
        Dict with agent health information
    """
    if not agent_id:
        return {"error": "agent_id is required"}

    coordinator = await _get_coordinator()
    if not coordinator:
        return {"error": "Control plane not available"}

    try:
        agent = await coordinator.get_agent(agent_id)
        if not agent:
            return {"error": f"Agent '{agent_id}' not found"}

        health = coordinator.get_agent_health(agent_id)
        is_available = coordinator.is_agent_available(agent_id)

        return {
            "agent_id": agent_id,
            "status": agent.status.value,
            "is_available": is_available,
            "last_heartbeat": agent.last_heartbeat,
            "heartbeat_age_seconds": round(time.time() - agent.last_heartbeat, 2),
            "tasks_completed": agent.tasks_completed,
            "tasks_failed": agent.tasks_failed,
            "avg_latency_ms": round(agent.avg_latency_ms, 2),
            "current_task_id": agent.current_task_id,
            "health_check": (
                {
                    "status": health.status.value if health else "unknown",
                    "last_check": health.last_check if health else None,
                    "consecutive_failures": health.consecutive_failures if health else 0,
                }
                if health
                else None
            ),
        }
    except Exception as e:
        return {"error": f"Failed to get agent health: {e}"}


# =============================================================================
# Task Operations
# =============================================================================


async def submit_task_tool(
    task_type: str,
    payload: str = "{}",
    required_capabilities: str = "",
    priority: str = "normal",
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """
    Submit a task to the control plane for execution.

    Args:
        task_type: Type of task (e.g., "debate", "code_review", "analysis")
        payload: JSON string with task data
        required_capabilities: Comma-separated capabilities required
        priority: Task priority (low, normal, high, urgent)
        timeout_seconds: Task timeout in seconds

    Returns:
        Dict with task ID and submission status
    """
    if not task_type:
        return {"error": "task_type is required"}

    coordinator = await _get_coordinator()
    if not coordinator:
        return {"error": "Control plane not available"}

    # Parse payload
    import json

    try:
        payload_dict = json.loads(payload) if payload else {}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON payload: {e}"}

    # Parse capabilities
    cap_list = (
        [c.strip() for c in required_capabilities.split(",") if c.strip()]
        if required_capabilities
        else None
    )

    # Parse priority
    from aragora.control_plane.scheduler import TaskPriority

    priority_map = {
        "low": TaskPriority.LOW,
        "normal": TaskPriority.NORMAL,
        "high": TaskPriority.HIGH,
        "urgent": TaskPriority.URGENT,
    }
    task_priority = priority_map.get(priority.lower(), TaskPriority.NORMAL)

    try:
        task_id = await coordinator.submit_task(
            task_type=task_type,
            payload=payload_dict,
            required_capabilities=cap_list,
            priority=task_priority,
            timeout_seconds=float(timeout_seconds),
        )
        return {
            "success": True,
            "task_id": task_id,
            "task_type": task_type,
            "priority": priority.lower(),
            "timeout_seconds": timeout_seconds,
            "message": "Task submitted successfully",
        }
    except Exception as e:
        return {"error": f"Failed to submit task: {e}"}


async def get_task_status_tool(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a task.

    Args:
        task_id: Task ID to query

    Returns:
        Dict with task status and details
    """
    if not task_id:
        return {"error": "task_id is required"}

    coordinator = await _get_coordinator()
    if not coordinator:
        return {"error": "Control plane not available"}

    try:
        task = await coordinator.get_task(task_id)
        if not task:
            return {"error": f"Task '{task_id}' not found"}

        return {
            "task_id": task.id,
            "task_type": task.task_type,
            "status": task.status.value,
            "priority": task.priority.name.lower(),
            "created_at": task.created_at,
            "assigned_at": task.assigned_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "assigned_agent": task.assigned_agent,
            "retries": task.retries,
            "max_retries": task.max_retries,
            "timeout_seconds": task.timeout_seconds,
            "result": task.result,
            "error": task.error,
            "is_timed_out": task.is_timed_out(),
        }
    except Exception as e:
        return {"error": f"Failed to get task status: {e}"}


async def cancel_task_tool(task_id: str) -> Dict[str, Any]:
    """
    Cancel a pending or running task.

    Args:
        task_id: Task ID to cancel

    Returns:
        Dict with cancellation result
    """
    if not task_id:
        return {"error": "task_id is required"}

    coordinator = await _get_coordinator()
    if not coordinator:
        return {"error": "Control plane not available"}

    try:
        success = await coordinator.cancel_task(task_id)
        return {
            "success": success,
            "task_id": task_id,
            "message": "Task cancelled" if success else "Task not found or already completed",
        }
    except Exception as e:
        return {"error": f"Failed to cancel task: {e}"}


async def list_pending_tasks_tool(
    task_type: str = "",
    limit: int = 20,
) -> Dict[str, Any]:
    """
    List tasks in the pending queue.

    Args:
        task_type: Optional filter by task type
        limit: Maximum tasks to return (default: 20)

    Returns:
        Dict with list of pending tasks
    """
    coordinator = await _get_coordinator()
    if not coordinator:
        return {"error": "Control plane not available"}

    try:
        from aragora.control_plane.scheduler import TaskStatus

        tasks = await coordinator._scheduler.list_by_status(TaskStatus.PENDING, limit=limit)

        # Filter by task_type if specified
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]

        return {
            "tasks": [
                {
                    "task_id": t.id,
                    "task_type": t.task_type,
                    "priority": t.priority.name.lower(),
                    "created_at": t.created_at,
                    "required_capabilities": list(t.required_capabilities),
                    "timeout_seconds": t.timeout_seconds,
                }
                for t in tasks
            ],
            "count": len(tasks),
            "filter": {"task_type": task_type} if task_type else None,
        }
    except Exception as e:
        return {"error": f"Failed to list pending tasks: {e}"}


# =============================================================================
# Health & Status
# =============================================================================


async def get_control_plane_status_tool() -> Dict[str, Any]:
    """
    Get overall control plane health and status.

    Returns:
        Dict with system health, agent counts, and task statistics
    """
    coordinator = await _get_coordinator()
    if not coordinator:
        return {
            "status": "unavailable",
            "error": "Control plane not available",
            "note": "Redis may not be running or control plane not initialized",
        }

    try:
        # Get system health
        health = coordinator.get_system_health()

        # Get comprehensive stats
        stats = await coordinator.get_stats()

        return {
            "status": health.value if health else "unknown",
            "registry": stats.get("registry", {}),
            "scheduler": stats.get("scheduler", {}),
            "health_monitor": stats.get("health", {}),
            "config": stats.get("config", {}),
            "knowledge_mound": stats.get("knowledge_mound"),
        }
    except Exception as e:
        return {"error": f"Failed to get control plane status: {e}"}


async def trigger_health_check_tool(agent_id: str = "") -> Dict[str, Any]:
    """
    Trigger a health check for an agent or all agents.

    Args:
        agent_id: Specific agent to check, or empty for all agents

    Returns:
        Dict with health check results
    """
    coordinator = await _get_coordinator()
    if not coordinator:
        return {"error": "Control plane not available"}

    try:
        if agent_id:
            # Check specific agent
            is_available = coordinator.is_agent_available(agent_id)
            health = coordinator.get_agent_health(agent_id)
            return {
                "agent_id": agent_id,
                "is_available": is_available,
                "health": (
                    {
                        "status": health.status.value if health else "unknown",
                        "last_check": health.last_check if health else None,
                    }
                    if health
                    else None
                ),
            }
        else:
            # Get system health
            system_health = coordinator.get_system_health()
            agents = await coordinator.list_agents(only_available=False)

            return {
                "system_health": system_health.value if system_health else "unknown",
                "agents_checked": len(agents),
                "agents_available": len([a for a in agents if a.is_available()]),
                "agents_offline": len([a for a in agents if not a.is_available()]),
            }
    except Exception as e:
        return {"error": f"Failed to trigger health check: {e}"}


async def get_resource_utilization_tool() -> Dict[str, Any]:
    """
    Get resource utilization metrics for the control plane.

    Returns:
        Dict with CPU, memory, queue depths, and quota usage
    """
    coordinator = await _get_coordinator()
    if not coordinator:
        return {"error": "Control plane not available"}

    try:
        stats = await coordinator.get_stats()

        # Calculate queue depths from scheduler stats
        scheduler_stats = stats.get("scheduler", {})
        by_status = scheduler_stats.get("by_status", {})

        return {
            "queue_depths": {
                "pending": by_status.get("pending", 0),
                "running": by_status.get("running", 0),
                "completed": by_status.get("completed", 0),
                "failed": by_status.get("failed", 0),
            },
            "agents": {
                "total": stats.get("registry", {}).get("total_agents", 0),
                "available": stats.get("registry", {}).get("available_agents", 0),
                "by_status": stats.get("registry", {}).get("by_status", {}),
            },
            "tasks_by_type": scheduler_stats.get("by_type", {}),
            "tasks_by_priority": scheduler_stats.get("by_priority", {}),
        }
    except Exception as e:
        return {"error": f"Failed to get resource utilization: {e}"}


__all__ = [
    # Agent operations
    "register_agent_tool",
    "unregister_agent_tool",
    "list_registered_agents_tool",
    "get_agent_health_tool",
    # Task operations
    "submit_task_tool",
    "get_task_status_tool",
    "cancel_task_tool",
    "list_pending_tasks_tool",
    # Health & status
    "get_control_plane_status_tool",
    "trigger_health_check_tool",
    "get_resource_utilization_tool",
]
