"""
Control Plane API Handler.

Provides agent orchestration and monitoring endpoints:
- List/monitor running agents
- Pause/resume individual agents
- View and manage processing queue
- Real-time dashboard updates via WebSocket

Usage:
    GET    /api/control-plane/agents         - List running agents
    GET    /api/control-plane/agents/{id}    - Get agent details
    POST   /api/control-plane/agents/{id}/pause   - Pause agent
    POST   /api/control-plane/agents/{id}/resume  - Resume agent
    GET    /api/control-plane/queue          - View processing queue
    POST   /api/control-plane/queue/prioritize    - Reorder queue
    GET    /api/control-plane/metrics        - System metrics
    WS     /api/control-plane/stream         - Real-time updates
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from aragora.server.handlers.base import BaseHandler

logger = logging.getLogger(__name__)


# In-memory agent registry (would sync with actual agent system in production)
_agents: dict[str, dict[str, Any]] = {}
_task_queue: list[dict[str, Any]] = []
_stream_clients: list[asyncio.Queue] = []
_metrics: dict[str, Any] = {
    "total_tasks_processed": 0,
    "total_findings_generated": 0,
    "active_sessions": 0,
    "agent_uptime": {},
}


class ControlPlaneHandler(BaseHandler):
    """
    Handler for control plane endpoints.

    Provides visibility and control over the agent orchestration system.
    """

    ROUTES = [
        "/api/control-plane/agents",
        "/api/control-plane/agents/{agent_id}",
        "/api/control-plane/agents/{agent_id}/pause",
        "/api/control-plane/agents/{agent_id}/resume",
        "/api/control-plane/agents/{agent_id}/metrics",
        "/api/control-plane/queue",
        "/api/control-plane/queue/prioritize",
        "/api/control-plane/metrics",
        "/api/control-plane/stream",
        "/api/control-plane/health",
    ]

    async def handle_request(self, request: Any) -> Any:
        """Route request to appropriate handler."""
        method = request.method
        path = str(request.path)

        # Parse agent_id from path if present
        agent_id = None
        if "/agents/" in path:
            parts = path.split("/agents/")
            if len(parts) > 1:
                remaining = parts[1].split("/")
                agent_id = remaining[0]

        # Route to appropriate handler
        if path.endswith("/agents") and method == "GET":
            return await self._list_agents(request)
        elif agent_id and path.endswith("/pause"):
            return await self._pause_agent(request, agent_id)
        elif agent_id and path.endswith("/resume"):
            return await self._resume_agent(request, agent_id)
        elif agent_id and path.endswith("/metrics"):
            return await self._get_agent_metrics(request, agent_id)
        elif agent_id and method == "GET":
            return await self._get_agent(request, agent_id)
        elif path.endswith("/queue/prioritize"):
            return await self._prioritize_queue(request)
        elif path.endswith("/queue"):
            return await self._get_queue(request)
        elif path.endswith("/metrics"):
            return await self._get_metrics(request)
        elif path.endswith("/stream"):
            return await self._stream_updates(request)
        elif path.endswith("/health"):
            return await self._health_check(request)

        return self._error_response(404, "Endpoint not found")

    async def _list_agents(self, request: Any) -> dict[str, Any]:
        """
        List all registered agents with their current status.

        Query params:
        - status: filter by status (active, paused, idle)
        - type: filter by agent type
        """
        status_filter = request.query.get("status")
        type_filter = request.query.get("type")

        # Get agents from registry (or create mock data if empty)
        agents = list(_agents.values())

        if not agents:
            # Populate with default agents for demo purposes
            agents = self._get_default_agents()
            for agent in agents:
                _agents[agent["id"]] = agent

        # Apply filters
        if status_filter:
            agents = [a for a in agents if a["status"] == status_filter]
        if type_filter:
            agents = [a for a in agents if a["type"] == type_filter]

        return self._json_response(
            200,
            {
                "agents": agents,
                "total": len(agents),
                "active": sum(1 for a in agents if a["status"] == "active"),
                "paused": sum(1 for a in agents if a["status"] == "paused"),
                "idle": sum(1 for a in agents if a["status"] == "idle"),
            },
        )

    async def _get_agent(self, request: Any, agent_id: str) -> dict[str, Any]:
        """Get details for a specific agent."""
        agent = _agents.get(agent_id)
        if not agent:
            return self._error_response(404, f"Agent {agent_id} not found")

        return self._json_response(200, agent)

    async def _pause_agent(self, request: Any, agent_id: str) -> dict[str, Any]:
        """Pause a running agent."""
        agent = _agents.get(agent_id)
        if not agent:
            return self._error_response(404, f"Agent {agent_id} not found")

        if agent["status"] != "active":
            return self._error_response(400, f"Agent is not active (current: {agent['status']})")

        agent["status"] = "paused"
        agent["paused_at"] = datetime.utcnow().isoformat()

        await self._broadcast_update(
            {
                "type": "agent_paused",
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        logger.info(f"Paused agent {agent_id}")

        return self._json_response(200, agent)

    async def _resume_agent(self, request: Any, agent_id: str) -> dict[str, Any]:
        """Resume a paused agent."""
        agent = _agents.get(agent_id)
        if not agent:
            return self._error_response(404, f"Agent {agent_id} not found")

        if agent["status"] != "paused":
            return self._error_response(400, f"Agent is not paused (current: {agent['status']})")

        agent["status"] = "active"
        agent["paused_at"] = None
        agent["resumed_at"] = datetime.utcnow().isoformat()

        await self._broadcast_update(
            {
                "type": "agent_resumed",
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        logger.info(f"Resumed agent {agent_id}")

        return self._json_response(200, agent)

    async def _get_agent_metrics(self, request: Any, agent_id: str) -> dict[str, Any]:
        """Get metrics for a specific agent."""
        agent = _agents.get(agent_id)
        if not agent:
            return self._error_response(404, f"Agent {agent_id} not found")

        metrics = {
            "agent_id": agent_id,
            "tasks_completed": agent.get("tasks_completed", 0),
            "findings_generated": agent.get("findings_generated", 0),
            "average_response_time_ms": agent.get("avg_response_time", 0),
            "error_rate": agent.get("error_rate", 0.0),
            "last_active": agent.get("last_active"),
            "uptime_seconds": agent.get("uptime_seconds", 0),
        }

        return self._json_response(200, metrics)

    async def _get_queue(self, request: Any) -> dict[str, Any]:
        """
        Get the current processing queue.

        Returns tasks ordered by priority and submission time.
        """
        # Get queue (or create mock data if empty)
        queue = _task_queue

        if not queue:
            queue = self._get_sample_queue()
            _task_queue.extend(queue)

        return self._json_response(
            200,
            {
                "tasks": queue,
                "total": len(queue),
                "by_priority": {
                    "high": sum(1 for t in queue if t.get("priority") == "high"),
                    "normal": sum(1 for t in queue if t.get("priority") == "normal"),
                    "low": sum(1 for t in queue if t.get("priority") == "low"),
                },
                "by_status": {
                    "pending": sum(1 for t in queue if t.get("status") == "pending"),
                    "processing": sum(1 for t in queue if t.get("status") == "processing"),
                },
            },
        )

    async def _prioritize_queue(self, request: Any) -> dict[str, Any]:
        """
        Reorder tasks in the queue.

        Expected body:
        {
            "task_id": "task-123",
            "priority": "high" | "normal" | "low",
            "position": 0  // optional: move to specific position
        }
        """
        try:
            body = await self._parse_json_body(request)
        except (json.JSONDecodeError, ValueError, TypeError):
            return self._error_response(400, "Invalid JSON body")

        task_id = body.get("task_id")
        if not task_id:
            return self._error_response(400, "task_id is required")

        priority = body.get("priority")
        position = body.get("position")

        # Find and update task
        task = None
        task_index = None
        for i, t in enumerate(_task_queue):
            if t.get("id") == task_id:
                task = t
                task_index = i
                break

        if not task:
            return self._error_response(404, f"Task {task_id} not found")

        # Update priority
        if priority:
            task["priority"] = priority

        # Move to position
        if position is not None and task_index != position:
            _task_queue.pop(task_index)
            _task_queue.insert(min(position, len(_task_queue)), task)

        await self._broadcast_update(
            {
                "type": "queue_updated",
                "task_id": task_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        logger.info(f"Updated priority for task {task_id}")

        return self._json_response(200, {"success": True, "task": task})

    async def _get_metrics(self, request: Any) -> dict[str, Any]:
        """
        Get system-wide metrics.

        Returns aggregated metrics across all agents and sessions.
        """
        # Calculate real-time metrics
        agents = list(_agents.values())

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "agents": {
                "total": len(agents),
                "active": sum(1 for a in agents if a["status"] == "active"),
                "paused": sum(1 for a in agents if a["status"] == "paused"),
                "idle": sum(1 for a in agents if a["status"] == "idle"),
            },
            "queue": {
                "total_tasks": len(_task_queue),
                "pending": sum(1 for t in _task_queue if t.get("status") == "pending"),
                "processing": sum(1 for t in _task_queue if t.get("status") == "processing"),
            },
            "processing": {
                "total_tasks_processed": _metrics["total_tasks_processed"],
                "total_findings_generated": _metrics["total_findings_generated"],
                "active_sessions": _metrics["active_sessions"],
            },
            "performance": {
                "avg_task_duration_ms": self._calculate_avg_task_duration(agents),
                "tasks_per_minute": self._calculate_throughput(agents),
                "error_rate": self._calculate_error_rate(agents),
            },
        }

        return self._json_response(200, metrics)

    async def _stream_updates(self, request: Any) -> Any:
        """
        Stream real-time control plane updates via WebSocket or SSE.

        Events include:
        - agent_status_changed
        - queue_updated
        - task_completed
        - metrics_update (every 5 seconds)
        """
        queue: asyncio.Queue = asyncio.Queue()
        _stream_clients.append(queue)

        async def event_generator():
            try:
                # Send initial state
                yield f"data: {json.dumps({'type': 'connected', 'timestamp': datetime.utcnow().isoformat()})}\n\n"

                # Send current metrics
                agents = list(_agents.values())
                yield f"data: {json.dumps({'type': 'initial_state', 'agents': len(agents), 'queue': len(_task_queue)})}\n\n"

                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=5.0)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        # Send periodic metrics update
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"

            finally:
                if queue in _stream_clients:
                    _stream_clients.remove(queue)

        return self._sse_response(event_generator())

    async def _health_check(self, request: Any) -> dict[str, Any]:
        """
        Health check endpoint for the control plane.

        Returns overall system health status.
        """
        agents = list(_agents.values())
        active_agents = sum(1 for a in agents if a["status"] == "active")

        health = {
            "status": "healthy" if active_agents > 0 or not agents else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "agents": {
                    "status": "healthy" if active_agents > 0 else "no_active_agents",
                    "active": active_agents,
                    "total": len(agents),
                },
                "queue": {
                    "status": "healthy",
                    "tasks": len(_task_queue),
                },
                "api": {
                    "status": "healthy",
                },
            },
        }

        return self._json_response(200, health)

    async def _broadcast_update(self, event: dict[str, Any]):
        """Broadcast event to all connected stream clients."""
        for queue in _stream_clients:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def _get_default_agents(self) -> list[dict[str, Any]]:
        """Get default agent configurations."""
        return [
            {
                "id": "agent-gemini-scanner",
                "name": "Gemini 3 Pro Scanner",
                "type": "scanner",
                "model": "gemini-3-pro",
                "status": "active",
                "role": "Full document analysis",
                "capabilities": ["security", "compliance", "quality"],
                "tasks_completed": 0,
                "findings_generated": 0,
                "avg_response_time": 0,
                "error_rate": 0.0,
                "created_at": datetime.utcnow().isoformat(),
                "last_active": datetime.utcnow().isoformat(),
            },
            {
                "id": "agent-claude-reasoner",
                "name": "Claude Reasoner",
                "type": "reasoner",
                "model": "claude-3.5-sonnet",
                "status": "active",
                "role": "Deep analysis and verification",
                "capabilities": ["reasoning", "verification", "consistency"],
                "tasks_completed": 0,
                "findings_generated": 0,
                "avg_response_time": 0,
                "error_rate": 0.0,
                "created_at": datetime.utcnow().isoformat(),
                "last_active": datetime.utcnow().isoformat(),
            },
            {
                "id": "agent-gpt-verifier",
                "name": "GPT-4 Verifier",
                "type": "verifier",
                "model": "gpt-4-turbo",
                "status": "idle",
                "role": "Adversarial verification",
                "capabilities": ["verification", "adversarial"],
                "tasks_completed": 0,
                "findings_generated": 0,
                "avg_response_time": 0,
                "error_rate": 0.0,
                "created_at": datetime.utcnow().isoformat(),
                "last_active": None,
            },
        ]

    def _get_sample_queue(self) -> list[dict[str, Any]]:
        """Get sample queue tasks."""
        return [
            {
                "id": f"task-{uuid4().hex[:8]}",
                "type": "document_audit",
                "priority": "high",
                "status": "pending",
                "document_id": "doc-001",
                "audit_type": "security",
                "created_at": datetime.utcnow().isoformat(),
            },
            {
                "id": f"task-{uuid4().hex[:8]}",
                "type": "document_audit",
                "priority": "normal",
                "status": "pending",
                "document_id": "doc-002",
                "audit_type": "compliance",
                "created_at": datetime.utcnow().isoformat(),
            },
        ]

    def _calculate_avg_task_duration(self, agents: list) -> float:
        """Calculate average task duration across agents."""
        if not agents:
            return 0.0
        times = [a.get("avg_response_time", 0) for a in agents if a.get("avg_response_time")]
        return sum(times) / len(times) if times else 0.0

    def _calculate_throughput(self, agents: list) -> float:
        """Calculate tasks per minute."""
        return sum(a.get("tasks_completed", 0) for a in agents) / 60.0

    def _calculate_error_rate(self, agents: list) -> float:
        """Calculate overall error rate."""
        if not agents:
            return 0.0
        rates = [a.get("error_rate", 0) for a in agents]
        return sum(rates) / len(rates) if rates else 0.0

    async def _parse_json_body(self, request: Any) -> dict[str, Any]:
        """Parse JSON body from request."""
        if hasattr(request, "json"):
            return await request.json()
        elif hasattr(request, "body"):
            body = await request.body()
            return json.loads(body)
        return {}

    def _json_response(self, status: int, data: Any) -> dict[str, Any]:
        """Create a JSON response."""
        return {
            "status": status,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(data, default=str),
        }

    def _error_response(self, status: int, message: str) -> dict[str, Any]:
        """Create an error response."""
        return self._json_response(status, {"error": message})

    def _sse_response(self, generator) -> dict[str, Any]:
        """Create an SSE response."""
        return {
            "status": 200,
            "headers": {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
            "body": generator,
        }


__all__ = ["ControlPlaneHandler"]
