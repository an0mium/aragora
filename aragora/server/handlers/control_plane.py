"""
Control Plane HTTP Handlers for Aragora.

Provides REST API endpoints for the enterprise control plane:
- Agent registration and discovery
- Task submission and status
- Health monitoring
- Control plane statistics and metrics

Endpoints:
    GET  /api/control-plane/agents           - List registered agents
    POST /api/control-plane/agents           - Register an agent
    GET  /api/control-plane/agents/:id       - Get agent info
    DELETE /api/control-plane/agents/:id     - Unregister agent
    POST /api/control-plane/agents/:id/heartbeat - Send heartbeat

    POST /api/control-plane/tasks            - Submit a task
    GET  /api/control-plane/tasks/:id        - Get task status
    POST /api/control-plane/tasks/:id/complete - Complete task
    POST /api/control-plane/tasks/:id/fail   - Fail task
    POST /api/control-plane/tasks/:id/cancel - Cancel task

    GET  /api/control-plane/health           - System health
    GET  /api/control-plane/health/:agent_id - Agent health
    GET  /api/control-plane/stats            - Control plane statistics
    GET  /api/control-plane/queue            - Job queue (pending/running tasks)
    GET  /api/control-plane/metrics          - Dashboard metrics
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine, Dict, Optional, TypeVar

T = TypeVar("T")

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit, user_rate_limit

logger = logging.getLogger(__name__)


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from sync context safely.

    Uses asyncio.run() which creates a new event loop, runs the coroutine,
    and closes the loop. This is the recommended pattern for calling async
    code from sync handlers.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    return asyncio.run(coro)


class ControlPlaneHandler(BaseHandler):
    """
    HTTP handler for control plane operations.

    Provides REST API access to the ControlPlaneCoordinator for
    agent management, task scheduling, and health monitoring.
    """

    # Class-level coordinator (set during server initialization)
    coordinator: Optional[Any] = None

    def __init__(self, server_context: Dict[str, Any]):
        """Initialize with server context."""
        super().__init__(server_context)

    def _get_coordinator(self) -> Optional[Any]:
        """Get the control plane coordinator."""
        # Try class-level first, then context
        if self.__class__.coordinator is not None:
            return self.__class__.coordinator
        return self.ctx.get("control_plane_coordinator")

    def _get_stream(self) -> Optional[Any]:
        """Get the control plane stream server for event emissions."""
        return self.ctx.get("control_plane_stream")

    def _emit_event(self, emit_method: str, *args, **kwargs) -> None:
        """Emit an event to the control plane stream (fire-and-forget).

        Args:
            emit_method: Name of the emit method on the stream server
            *args, **kwargs: Arguments to pass to the emit method
        """
        stream = self._get_stream()
        if not stream:
            return
        try:
            method = getattr(stream, emit_method, None)
            if method:
                _run_async(method(*args, **kwargs))
        except Exception as e:
            # Don't let stream errors affect the main request
            logger.debug(f"Stream emission error: {e}")

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/control-plane/")

    # =========================================================================
    # GET Handlers
    # =========================================================================

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle GET requests."""
        # /api/control-plane/agents
        if path == "/api/control-plane/agents":
            return self._handle_list_agents(query_params)

        # /api/control-plane/agents/:id
        if path.startswith("/api/control-plane/agents/") and path.count("/") == 4:
            agent_id = path.split("/")[-1]
            return self._handle_get_agent(agent_id)

        # /api/control-plane/tasks/:id
        if path.startswith("/api/control-plane/tasks/") and path.count("/") == 4:
            task_id = path.split("/")[-1]
            return self._handle_get_task(task_id)

        # /api/control-plane/health
        if path == "/api/control-plane/health":
            return self._handle_system_health()

        # /api/control-plane/health/:agent_id
        if path.startswith("/api/control-plane/health/") and path.count("/") == 4:
            agent_id = path.split("/")[-1]
            return self._handle_agent_health(agent_id)

        # /api/control-plane/stats
        if path == "/api/control-plane/stats":
            return self._handle_stats()

        # /api/control-plane/queue
        if path == "/api/control-plane/queue":
            return self._handle_get_queue(query_params)

        # /api/control-plane/metrics
        if path == "/api/control-plane/metrics":
            return self._handle_get_metrics()

        return None

    def _handle_list_agents(self, query_params: Dict[str, Any]) -> HandlerResult:
        """List registered agents."""
        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        capability = query_params.get("capability")
        only_available = query_params.get("available", "true").lower() == "true"

        try:
            agents = _run_async(
                coordinator.list_agents(
                    capability=capability,
                    only_available=only_available,
                )
            )

            return json_response({
                "agents": [a.to_dict() for a in agents],
                "total": len(agents),
            })
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return error_response(str(e), 500)

    def _handle_get_agent(self, agent_id: str) -> HandlerResult:
        """Get agent by ID."""
        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        try:
            agent = _run_async(
                coordinator.get_agent(agent_id)
            )

            if not agent:
                return error_response(f"Agent not found: {agent_id}", 404)

            return json_response(agent.to_dict())
        except Exception as e:
            logger.error(f"Error getting agent {agent_id}: {e}")
            return error_response(str(e), 500)

    def _handle_get_task(self, task_id: str) -> HandlerResult:
        """Get task by ID."""
        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        try:
            task = _run_async(
                coordinator.get_task(task_id)
            )

            if not task:
                return error_response(f"Task not found: {task_id}", 404)

            return json_response(task.to_dict())
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {e}")
            return error_response(str(e), 500)

    def _handle_system_health(self) -> HandlerResult:
        """Get system health status."""
        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        try:
            health_status = coordinator.get_system_health()
            all_health = coordinator._health_monitor.get_all_health()

            return json_response({
                "status": health_status.value,
                "agents": {
                    agent_id: hc.to_dict()
                    for agent_id, hc in all_health.items()
                },
            })
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return error_response(str(e), 500)

    def _handle_agent_health(self, agent_id: str) -> HandlerResult:
        """Get health status for specific agent."""
        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        try:
            health = coordinator.get_agent_health(agent_id)

            if not health:
                return error_response(f"No health data for agent: {agent_id}", 404)

            return json_response(health.to_dict())
        except Exception as e:
            logger.error(f"Error getting agent health {agent_id}: {e}")
            return error_response(str(e), 500)

    def _handle_stats(self) -> HandlerResult:
        """Get control plane statistics."""
        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        try:
            stats = _run_async(
                coordinator.get_stats()
            )

            return json_response(stats)
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return error_response(str(e), 500)

    def _handle_get_queue(self, query_params: Dict[str, Any]) -> HandlerResult:
        """Get current job queue (pending and running tasks)."""
        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        try:
            from aragora.control_plane.scheduler import TaskStatus
            from datetime import datetime

            limit = int(query_params.get("limit", ["50"])[0]) if isinstance(
                query_params.get("limit"), list
            ) else int(query_params.get("limit", 50))

            # Get pending and running tasks
            pending = _run_async(
                coordinator._scheduler.list_by_status(TaskStatus.PENDING, limit=limit)
            )
            running = _run_async(
                coordinator._scheduler.list_by_status(TaskStatus.RUNNING, limit=limit)
            )

            # Format tasks as jobs for the frontend
            def task_to_job(task) -> Dict[str, Any]:
                # Calculate progress based on status
                progress = 0.0
                if task.status.value == "running":
                    # If running, estimate progress or use metadata
                    progress = task.metadata.get("progress", 0.5)
                elif task.status.value == "completed":
                    progress = 1.0

                return {
                    "id": task.id,
                    "type": task.task_type,
                    "name": task.metadata.get("name", f"{task.task_type} task"),
                    "status": task.status.value,
                    "progress": progress,
                    "started_at": datetime.fromtimestamp(task.started_at).isoformat() if task.started_at else None,
                    "created_at": datetime.fromtimestamp(task.created_at).isoformat() if task.created_at else None,
                    "document_count": task.payload.get("document_count", 0),
                    "agents_assigned": [task.assigned_agent] if task.assigned_agent else [],
                    "priority": task.priority.name.lower(),
                }

            jobs = [task_to_job(t) for t in running] + [task_to_job(t) for t in pending]

            return json_response({
                "jobs": jobs,
                "total": len(jobs),
            })
        except Exception as e:
            logger.error(f"Error getting queue: {e}")
            return error_response(str(e), 500)

    def _handle_get_metrics(self) -> HandlerResult:
        """Get control plane metrics for dashboard."""
        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        try:
            # Get comprehensive stats
            stats = _run_async(coordinator.get_stats())

            scheduler_stats = stats.get("scheduler", {})
            registry_stats = stats.get("registry", {})

            by_status = scheduler_stats.get("by_status", {})
            agent_by_status = registry_stats.get("by_status", {})

            # Calculate metrics for dashboard
            active_jobs = by_status.get("running", 0)
            queued_jobs = by_status.get("pending", 0)
            completed_jobs = by_status.get("completed", 0)

            agents_available = registry_stats.get("available_agents", 0)
            agents_busy = agent_by_status.get("busy", 0)
            total_agents = registry_stats.get("total_agents", 0)

            return json_response({
                "active_jobs": active_jobs,
                "queued_jobs": queued_jobs,
                "completed_jobs": completed_jobs,
                "agents_available": agents_available,
                "agents_busy": agents_busy,
                "total_agents": total_agents,
                # These could come from a metrics store if available
                "documents_processed_today": scheduler_stats.get("by_type", {}).get("document_processing", 0),
                "audits_completed_today": scheduler_stats.get("by_type", {}).get("audit", 0),
                "tokens_used_today": 0,  # Would need token tracking integration
            })
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return error_response(str(e), 500)

    # =========================================================================
    # POST Handlers
    # =========================================================================

    @user_rate_limit(action="agent_call")
    @rate_limit(rpm=60, limiter_name="control_plane_post")
    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        # /api/control-plane/agents
        if path == "/api/control-plane/agents":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_register_agent(body, handler)

        # /api/control-plane/agents/:id/heartbeat
        if path.endswith("/heartbeat") and "/agents/" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                agent_id = parts[-2]
                body, err = self.read_json_body_validated(handler)
                if err:
                    return err
                return self._handle_heartbeat(agent_id, body, handler)

        # /api/control-plane/tasks
        if path == "/api/control-plane/tasks":
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_submit_task(body, handler)

        # /api/control-plane/tasks/:id/complete
        if path.endswith("/complete") and "/tasks/" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                task_id = parts[-2]
                body, err = self.read_json_body_validated(handler)
                if err:
                    return err
                return self._handle_complete_task(task_id, body, handler)

        # /api/control-plane/tasks/:id/fail
        if path.endswith("/fail") and "/tasks/" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                task_id = parts[-2]
                body, err = self.read_json_body_validated(handler)
                if err:
                    return err
                return self._handle_fail_task(task_id, body, handler)

        # /api/control-plane/tasks/:id/cancel
        if path.endswith("/cancel") and "/tasks/" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                task_id = parts[-2]
                return self._handle_cancel_task(task_id, handler)

        # /api/control-plane/tasks/:id/claim
        if path.endswith("/claim") and "/tasks/" in path:
            body, err = self.read_json_body_validated(handler)
            if err:
                return err
            return self._handle_claim_task(body, handler)

        return None

    def _handle_register_agent(self, body: Dict[str, Any], handler: Any) -> HandlerResult:
        """Register a new agent."""
        # Require authentication for agent registration
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        agent_id = body.get("agent_id")
        if not agent_id:
            return error_response("agent_id is required", 400)

        capabilities = body.get("capabilities", [])
        model = body.get("model", "unknown")
        provider = body.get("provider", "unknown")
        metadata = body.get("metadata", {})

        try:
            agent = _run_async(
                coordinator.register_agent(
                    agent_id=agent_id,
                    capabilities=capabilities,
                    model=model,
                    provider=provider,
                    metadata=metadata,
                )
            )

            # Emit event for real-time streaming
            self._emit_event(
                "emit_agent_registered",
                agent_id=agent_id,
                capabilities=capabilities,
                model=model,
                provider=provider,
            )

            return json_response(agent.to_dict(), status=201)
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            return error_response(str(e), 500)

    def _handle_heartbeat(
        self, agent_id: str, body: Dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Handle agent heartbeat."""
        # Require authentication for heartbeats
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        status = body.get("status")

        try:
            from aragora.control_plane.registry import AgentStatus

            agent_status = AgentStatus(status) if status else None

            success = _run_async(
                coordinator.heartbeat(agent_id, agent_status)
            )

            if not success:
                return error_response(f"Agent not found: {agent_id}", 404)

            return json_response({"acknowledged": True})
        except Exception as e:
            logger.error(f"Error processing heartbeat: {e}")
            return error_response(str(e), 500)

    def _handle_submit_task(self, body: Dict[str, Any], handler: Any) -> HandlerResult:
        """Submit a new task."""
        # Require authentication for task submission
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        task_type = body.get("task_type")
        if not task_type:
            return error_response("task_type is required", 400)

        payload = body.get("payload", {})
        required_capabilities = body.get("required_capabilities", [])
        priority = body.get("priority", "normal")
        timeout_seconds = body.get("timeout_seconds")
        metadata = body.get("metadata", {})

        try:
            from aragora.control_plane.scheduler import TaskPriority

            priority_enum = TaskPriority[priority.upper()]

            task_id = _run_async(
                coordinator.submit_task(
                    task_type=task_type,
                    payload=payload,
                    required_capabilities=required_capabilities,
                    priority=priority_enum,
                    timeout_seconds=timeout_seconds,
                    metadata=metadata,
                )
            )

            # Emit event for real-time streaming
            self._emit_event(
                "emit_task_submitted",
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                required_capabilities=required_capabilities,
            )

            return json_response({"task_id": task_id}, status=201)
        except KeyError:
            return error_response(f"Invalid priority: {priority}", 400)
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return error_response(str(e), 500)

    def _handle_claim_task(self, body: Dict[str, Any], handler: Any) -> HandlerResult:
        """Claim a task for an agent."""
        # Require authentication for claiming tasks
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        agent_id = body.get("agent_id")
        if not agent_id:
            return error_response("agent_id is required", 400)

        capabilities = body.get("capabilities", [])
        block_ms = body.get("block_ms", 5000)

        try:
            task = _run_async(
                coordinator.claim_task(
                    agent_id=agent_id,
                    capabilities=capabilities,
                    block_ms=block_ms,
                )
            )

            if not task:
                return json_response({"task": None})

            # Emit event for real-time streaming
            self._emit_event(
                "emit_task_claimed",
                task_id=task.id,
                agent_id=agent_id,
            )

            return json_response({"task": task.to_dict()})
        except Exception as e:
            logger.error(f"Error claiming task: {e}")
            return error_response(str(e), 500)

    def _handle_complete_task(
        self, task_id: str, body: Dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Mark task as completed."""
        # Require authentication for completing tasks
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        result = body.get("result")
        agent_id = body.get("agent_id")
        latency_ms = body.get("latency_ms")

        try:
            success = _run_async(
                coordinator.complete_task(
                    task_id=task_id,
                    result=result,
                    agent_id=agent_id,
                    latency_ms=latency_ms,
                )
            )

            if not success:
                return error_response(f"Task not found: {task_id}", 404)

            # Emit event for real-time streaming
            self._emit_event(
                "emit_task_completed",
                task_id=task_id,
                agent_id=agent_id or "unknown",
                result=result,
            )

            return json_response({"completed": True})
        except Exception as e:
            logger.error(f"Error completing task: {e}")
            return error_response(str(e), 500)

    def _handle_fail_task(self, task_id: str, body: Dict[str, Any], handler: Any) -> HandlerResult:
        """Mark task as failed."""
        # Require authentication for failing tasks
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        error = body.get("error", "Unknown error")
        agent_id = body.get("agent_id")
        latency_ms = body.get("latency_ms")
        requeue = body.get("requeue", True)

        try:
            success = _run_async(
                coordinator.fail_task(
                    task_id=task_id,
                    error=error,
                    agent_id=agent_id,
                    latency_ms=latency_ms,
                    requeue=requeue,
                )
            )

            if not success:
                return error_response(f"Task not found: {task_id}", 404)

            # Emit event for real-time streaming
            self._emit_event(
                "emit_task_failed",
                task_id=task_id,
                agent_id=agent_id or "unknown",
                error=error,
                retries_left=0,  # Coordinator tracks retries
            )

            return json_response({"failed": True})
        except Exception as e:
            logger.error(f"Error failing task: {e}")
            return error_response(str(e), 500)

    def _handle_cancel_task(self, task_id: str, handler: Any) -> HandlerResult:
        """Cancel a task."""
        # Require authentication for canceling tasks
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        try:
            success = _run_async(
                coordinator.cancel_task(task_id)
            )

            if not success:
                return error_response(f"Task not found or already completed: {task_id}", 404)

            return json_response({"cancelled": True})
        except Exception as e:
            logger.error(f"Error cancelling task: {e}")
            return error_response(str(e), 500)

    # =========================================================================
    # DELETE Handlers
    # =========================================================================

    def handle_delete(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle DELETE requests."""
        # /api/control-plane/agents/:id
        if path.startswith("/api/control-plane/agents/") and path.count("/") == 4:
            agent_id = path.split("/")[-1]
            return self._handle_unregister_agent(agent_id, handler)

        return None

    def _handle_unregister_agent(self, agent_id: str, handler: Any) -> HandlerResult:
        """Unregister an agent."""
        # Require authentication for agent unregistration
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        try:
            success = _run_async(
                coordinator.unregister_agent(agent_id)
            )

            if not success:
                return error_response(f"Agent not found: {agent_id}", 404)

            # Emit event for real-time streaming
            self._emit_event(
                "emit_agent_unregistered",
                agent_id=agent_id,
                reason="manual_unregistration",
            )

            return json_response({"unregistered": True})
        except Exception as e:
            logger.error(f"Error unregistering agent: {e}")
            return error_response(str(e), 500)
