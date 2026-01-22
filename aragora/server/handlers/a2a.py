"""
A2A Protocol HTTP Handler.

Exposes the A2A (Agent-to-Agent) protocol through the unified server.

Endpoints:
- GET /api/a2a/agents - List all available agents
- GET /api/a2a/agents/:name - Get agent card by name
- POST /api/a2a/tasks - Submit a task
- GET /api/a2a/tasks/:id - Get task status
- DELETE /api/a2a/tasks/:id - Cancel task
- POST /api/a2a/tasks/:id/stream - Stream task (WebSocket upgrade)
- GET /api/a2a/.well-known/agent.json - Discovery endpoint
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Singleton A2A server
_a2a_server: Optional[Any] = None


def get_a2a_server():
    """Get or create the A2A server singleton."""
    global _a2a_server
    if _a2a_server is None:
        from aragora.protocols.a2a import A2AServer

        _a2a_server = A2AServer()
    return _a2a_server


class A2AHandler(BaseHandler):
    """Handler for A2A protocol endpoints."""

    ROUTES = [
        # Discovery
        "/api/v1/a2a/.well-known/agent.json",
        "/.well-known/agent.json",
        # Agent listing
        "/api/v1/a2a/agents",
        "/api/v1/a2a/agents/*",
        # Tasks
        "/api/v1/a2a/tasks",
        "/api/v1/a2a/tasks/*",
        # OpenAPI spec
        "/api/v1/a2a/openapi.json",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        if path.startswith("/api/v1/a2a/"):
            return True
        if path == "/.well-known/agent.json":
            return True
        return False

    @rate_limit(rpm=120)
    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route A2A requests."""
        method = handler.command if hasattr(handler, "command") else "GET"

        # Discovery endpoint
        if path in ("/.well-known/agent.json", "/api/v1/a2a/.well-known/agent.json"):
            return self._handle_discovery()

        # OpenAPI spec
        if path == "/api/v1/a2a/openapi.json":
            return self._handle_openapi()

        # Remove prefix
        subpath = path[8:] if path.startswith("/api/v1/a2a") else path

        # Agents
        if subpath == "/agents":
            return self._handle_list_agents()
        if subpath.startswith("/agents/"):
            agent_name = subpath[8:]
            return self._handle_get_agent(agent_name)

        # Tasks
        if subpath == "/tasks" and method == "POST":
            return self._handle_submit_task(handler)
        if subpath.startswith("/tasks/"):
            task_id = subpath[7:]
            # Handle stream suffix
            if task_id.endswith("/stream"):
                task_id = task_id[:-7]
                return self._handle_stream_task(task_id, handler)
            if method == "GET":
                return self._handle_get_task(task_id)
            if method == "DELETE":
                return self._handle_cancel_task(task_id)

        return error_response("Unknown A2A endpoint", 404)

    def _handle_discovery(self) -> HandlerResult:
        """Handle agent discovery request."""
        server = get_a2a_server()
        agents = server.list_agents()

        # Return primary agent card for discovery
        if agents:
            primary = agents[0]
            return json_response(primary.to_dict())

        return json_response(
            {
                "name": "aragora",
                "version": "1.0.0",
                "description": "Aragora multi-agent decision engine",
                "capabilities": ["debate", "audit", "critique", "research"],
                "endpoints": {
                    "agents": "/api/v1/a2a/agents",
                    "tasks": "/api/v1/a2a/tasks",
                },
            }
        )

    def _handle_openapi(self) -> HandlerResult:
        """Return OpenAPI specification."""
        server = get_a2a_server()
        spec = server.get_openapi_spec()
        return json_response(spec)

    def _handle_list_agents(self) -> HandlerResult:
        """List all available agents."""
        server = get_a2a_server()
        agents = server.list_agents()

        return json_response(
            {
                "agents": [a.to_dict() for a in agents],
                "total": len(agents),
            }
        )

    def _handle_get_agent(self, name: str) -> HandlerResult:
        """Get agent by name."""
        server = get_a2a_server()
        agent = server.get_agent(name)

        if not agent:
            return error_response(f"Agent not found: {name}", 404)

        return json_response(agent.to_dict())

    def _handle_submit_task(self, handler: Any) -> HandlerResult:
        """Submit a task for execution."""
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        # Validate required fields
        if "instruction" not in data:
            return error_response("Missing required field: instruction", 400)

        # Create task request
        from aragora.protocols.a2a import TaskRequest, AgentCapability, ContextItem
        import uuid

        task_id = data.get("task_id", str(uuid.uuid4()))
        capability = None
        if data.get("capability"):
            try:
                capability = AgentCapability(data["capability"])
            except ValueError:
                pass

        context = []
        for ctx in data.get("context", []):
            context.append(
                ContextItem(
                    type=ctx.get("type", "text"),
                    content=ctx.get("content", ""),
                    metadata=ctx.get("metadata", {}),
                )
            )

        request = TaskRequest(  # type: ignore[call-arg]
            task_id=task_id,
            instruction=data["instruction"],
            capability=capability,
            context=context,
            priority=data.get("priority"),
            deadline=data.get("deadline"),
            metadata=data.get("metadata", {}),
        )

        # Execute task
        server = get_a2a_server()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(server.handle_task(request))
            return json_response(result.to_dict())
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return error_response(f"Task execution failed: {str(e)[:100]}", 500)

    def _handle_get_task(self, task_id: str) -> HandlerResult:
        """Get task status."""
        server = get_a2a_server()
        result = server.get_task_status(task_id)

        if not result:
            return error_response(f"Task not found: {task_id}", 404)

        return json_response(result.to_dict())

    def _handle_cancel_task(self, task_id: str) -> HandlerResult:
        """Cancel a running task."""
        server = get_a2a_server()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            success = loop.run_until_complete(server.cancel_task(task_id))
            if success:
                return HandlerResult(
                    status_code=204,
                    content_type="application/json",
                    body=b"",
                    headers={},
                )
            return error_response(f"Task not found or not cancellable: {task_id}", 404)
        except Exception as e:
            logger.error(f"Task cancellation failed: {e}")
            return error_response(f"Cancellation failed: {str(e)[:100]}", 500)

    def _handle_stream_task(self, task_id: str, handler: Any) -> HandlerResult:
        """Handle streaming task request (returns upgrade required)."""
        # Note: Actual streaming requires WebSocket which is handled separately
        return json_response(
            {
                "message": "Use WebSocket connection for streaming",
                "ws_path": f"/ws/a2a/tasks/{task_id}/stream",
            },
            status=426,
        )


# Handler factory
_a2a_handler: Optional["A2AHandler"] = None


def get_a2a_handler(server_context: Optional[Dict] = None) -> "A2AHandler":
    """Get or create the A2A handler instance."""
    global _a2a_handler
    if _a2a_handler is None:
        if server_context is None:
            server_context = {}
        _a2a_handler = A2AHandler(server_context)  # type: ignore[arg-type]
    return _a2a_handler


__all__ = ["A2AHandler", "get_a2a_handler", "get_a2a_server"]
