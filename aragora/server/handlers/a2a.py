"""
A2A Protocol HTTP Handler.

Stability: STABLE

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

import json
import logging
import re
from typing import Any

from aragora.resilience import CircuitBreaker
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.rbac.decorators import require_permission

logger = logging.getLogger(__name__)

# =============================================================================
# Resilience Configuration
# =============================================================================

# Circuit breaker for A2A protocol service
_a2a_circuit_breaker = CircuitBreaker(
    name="a2a_handler",
    failure_threshold=5,
    cooldown_seconds=30.0,
)


def get_a2a_circuit_breaker() -> CircuitBreaker:
    """Get the circuit breaker for A2A protocol service."""
    return _a2a_circuit_breaker


def get_a2a_circuit_breaker_status() -> dict[str, Any]:
    """Get current status of the A2A protocol circuit breaker."""
    return _a2a_circuit_breaker.to_dict()


# Validation patterns for A2A protocol
# Agent names: alphanumeric, hyphens, underscores, dots (for versioning), 1-64 chars
AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")

# Task IDs: UUID format or alphanumeric with hyphens, 1-128 chars
TASK_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")

# Maximum lengths for input validation
MAX_INSTRUCTION_LENGTH = 10000
MAX_CONTEXT_ITEMS = 50
MAX_CONTEXT_CONTENT_LENGTH = 100000
MAX_METADATA_KEYS = 20
MAX_METADATA_VALUE_LENGTH = 1000
MAX_BODY_SIZE = 1024 * 1024  # 1 MB


def validate_agent_name(name: str) -> tuple[bool, str | None]:
    """Validate an A2A agent name.

    Args:
        name: Agent name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Agent name is required"
    if len(name) > 64:
        return False, "Agent name must be 64 characters or less"
    if not AGENT_NAME_PATTERN.match(name):
        return (
            False,
            "Agent name must start with alphanumeric and contain only letters, numbers, dots, hyphens, or underscores",
        )
    return True, None


def validate_task_id(task_id: str) -> tuple[bool, str | None]:
    """Validate an A2A task ID.

    Args:
        task_id: Task ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not task_id:
        return False, "Task ID is required"
    if len(task_id) > 128:
        return False, "Task ID must be 128 characters or less"
    if not TASK_ID_PATTERN.match(task_id):
        return (
            False,
            "Task ID must start with alphanumeric and contain only letters, numbers, hyphens, or underscores",
        )
    return True, None


def validate_task_request_body(data: dict) -> tuple[bool, str | None]:
    """Validate the task submission request body.

    Args:
        data: Parsed JSON body

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object"

    instruction = data.get("instruction")
    if not instruction:
        return False, "Missing required field: instruction"
    if not isinstance(instruction, str):
        return False, "instruction must be a string"
    if len(instruction) > MAX_INSTRUCTION_LENGTH:
        return False, f"instruction must be {MAX_INSTRUCTION_LENGTH} characters or less"

    # Validate optional task_id if provided
    if "task_id" in data:
        task_id = data["task_id"]
        if not isinstance(task_id, str):
            return False, "task_id must be a string"
        is_valid, err = validate_task_id(task_id)
        if not is_valid:
            return False, err

    # Validate capability if provided
    if "capability" in data:
        capability = data["capability"]
        if not isinstance(capability, str):
            return False, "capability must be a string"
        valid_capabilities = {
            "debate",
            "consensus",
            "critique",
            "synthesis",
            "audit",
            "verification",
            "code_review",
            "document_analysis",
            "research",
            "reasoning",
        }
        if capability.lower() not in valid_capabilities:
            return (
                False,
                f"Invalid capability: {capability}. Must be one of: {', '.join(sorted(valid_capabilities))}",
            )

    # Validate priority if provided
    if "priority" in data:
        priority = data["priority"]
        if not isinstance(priority, str):
            return False, "priority must be a string"
        valid_priorities = {"low", "normal", "high", "urgent"}
        if priority.lower() not in valid_priorities:
            return (
                False,
                f"Invalid priority: {priority}. Must be one of: {', '.join(sorted(valid_priorities))}",
            )

    # Validate context if provided
    if "context" in data:
        context = data["context"]
        if not isinstance(context, list):
            return False, "context must be an array"
        if len(context) > MAX_CONTEXT_ITEMS:
            return False, f"context must have {MAX_CONTEXT_ITEMS} items or fewer"

        for i, ctx in enumerate(context):
            if not isinstance(ctx, dict):
                return False, f"context[{i}] must be an object"
            if "type" in ctx and not isinstance(ctx.get("type"), str):
                return False, f"context[{i}].type must be a string"
            if "content" in ctx:
                content = ctx["content"]
                if not isinstance(content, str):
                    return False, f"context[{i}].content must be a string"
                if len(content) > MAX_CONTEXT_CONTENT_LENGTH:
                    return (
                        False,
                        f"context[{i}].content must be {MAX_CONTEXT_CONTENT_LENGTH} characters or less",
                    )
            if "metadata" in ctx and not isinstance(ctx.get("metadata"), dict):
                return False, f"context[{i}].metadata must be an object"

    # Validate metadata if provided
    if "metadata" in data:
        metadata = data["metadata"]
        if not isinstance(metadata, dict):
            return False, "metadata must be an object"
        if len(metadata) > MAX_METADATA_KEYS:
            return False, f"metadata must have {MAX_METADATA_KEYS} keys or fewer"
        for key, value in metadata.items():
            if not isinstance(key, str):
                return False, "metadata keys must be strings"
            if isinstance(value, str) and len(value) > MAX_METADATA_VALUE_LENGTH:
                return (
                    False,
                    f"metadata value for '{key}' exceeds maximum length of {MAX_METADATA_VALUE_LENGTH}",
                )

    # Validate timeout_ms if provided
    if "timeout_ms" in data:
        timeout = data["timeout_ms"]
        if not isinstance(timeout, int):
            return False, "timeout_ms must be an integer"
        if timeout < 1000 or timeout > 3600000:  # 1 second to 1 hour
            return False, "timeout_ms must be between 1000 and 3600000"

    return True, None


# Singleton A2A server
_a2a_server: Any | None = None


def get_a2a_server():
    """Get or create the A2A server singleton."""
    global _a2a_server
    if _a2a_server is None:
        from aragora.protocols.a2a import A2AServer

        _a2a_server = A2AServer()
    return _a2a_server


class A2AHandler(BaseHandler):
    """Handler for A2A protocol endpoints.

    Stability: STABLE

    Features:
    - Circuit breaker pattern for service resilience
    - Rate limiting (30-120 requests/minute depending on endpoint)
    - RBAC permission checks (a2a:read, a2a:create)
    - Comprehensive input validation for task submissions
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

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

    @rate_limit(requests_per_minute=120)
    @require_permission("a2a:read")
    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
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
            return await self._handle_submit_task(handler)
        if subpath.startswith("/tasks/"):
            task_id = subpath[7:]
            # Handle stream suffix
            if task_id.endswith("/stream"):
                task_id = task_id[:-7]
                return self._handle_stream_task(task_id, handler)
            if method == "GET":
                return self._handle_get_task(task_id)
            if method == "DELETE":
                return await self._handle_cancel_task(task_id)

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
        # Validate agent name
        is_valid, err = validate_agent_name(name)
        if not is_valid:
            return error_response(err, 400)

        server = get_a2a_server()
        agent = server.get_agent(name)

        if not agent:
            return error_response(f"Agent not found: {name}", 404)

        return json_response(agent.to_dict())

    @rate_limit(requests_per_minute=30)
    @require_permission("a2a:create")
    async def _handle_submit_task(self, handler: Any) -> HandlerResult:
        """Submit a task for execution."""
        # Validate Content-Type
        content_type = handler.headers.get("Content-Type", "")
        if content_type and not content_type.startswith("application/json"):
            return error_response("Content-Type must be application/json", 415)

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > MAX_BODY_SIZE:
                return error_response(
                    f"Request body too large. Maximum size is {MAX_BODY_SIZE} bytes", 413
                )
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            logger.exception("Invalid JSON in task submission request")
            return error_response(f"Invalid JSON: {e}", 400)
        except UnicodeDecodeError:
            logger.exception("Invalid UTF-8 encoding in task submission request")
            return error_response("Request body must be valid UTF-8", 400)

        # Validate request body schema
        is_valid, err = validate_task_request_body(data)
        if not is_valid:
            return error_response(err, 400)

        # Create task request
        from aragora.protocols.a2a import TaskRequest, AgentCapability, ContextItem, TaskPriority
        import uuid

        task_id = data.get("task_id", str(uuid.uuid4()))
        capability: AgentCapability | None = None
        if data.get("capability"):
            try:
                capability = AgentCapability(data["capability"])
            except ValueError:
                pass

        # Parse priority if provided
        priority = TaskPriority.NORMAL
        if data.get("priority"):
            try:
                priority = TaskPriority(data["priority"])
            except ValueError:
                pass

        context: list[ContextItem] = []
        for ctx in data.get("context", []):
            context.append(
                ContextItem(
                    type=ctx.get("type", "text"),
                    content=ctx.get("content", ""),
                    metadata=ctx.get("metadata", {}),
                )
            )

        # Store deadline in metadata if provided (not a TaskRequest field)
        metadata: dict[str, Any] = data.get("metadata", {})
        if data.get("deadline"):
            metadata["deadline"] = data["deadline"]

        request = TaskRequest(
            task_id=task_id,
            instruction=data["instruction"],
            capability=capability,
            context=context,
            priority=priority,
            metadata=metadata,
        )

        # Execute task
        server = get_a2a_server()

        try:
            result = await server.handle_task(request)
            return json_response(result.to_dict())
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.error(f"Task execution failed: {e}")
            return error_response(f"Task execution failed: {str(e)[:100]}", 500)

    def _handle_get_task(self, task_id: str) -> HandlerResult:
        """Get task status."""
        # Validate task ID
        is_valid, err = validate_task_id(task_id)
        if not is_valid:
            return error_response(err, 400)

        server = get_a2a_server()
        result = server.get_task_status(task_id)

        if not result:
            return error_response(f"Task not found: {task_id}", 404)

        return json_response(result.to_dict())

    async def _handle_cancel_task(self, task_id: str) -> HandlerResult:
        """Cancel a running task."""
        # Validate task ID
        is_valid, err = validate_task_id(task_id)
        if not is_valid:
            return error_response(err, 400)

        server = get_a2a_server()

        try:
            success = await server.cancel_task(task_id)
            if success:
                return HandlerResult(
                    status_code=204,
                    content_type="application/json",
                    body=b"",
                    headers={},
                )
            return error_response(f"Task not found or not cancellable: {task_id}", 404)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
            logger.error(f"Task cancellation failed: {e}")
            return error_response(f"Cancellation failed: {str(e)[:100]}", 500)

    def _handle_stream_task(self, task_id: str, handler: Any) -> HandlerResult:
        """Handle streaming task request (returns upgrade required)."""
        # Validate task ID
        is_valid, err = validate_task_id(task_id)
        if not is_valid:
            return error_response(err, 400)

        # Note: Actual streaming requires WebSocket which is handled separately
        return json_response(
            {
                "message": "Use WebSocket connection for streaming",
                "ws_path": f"/ws/a2a/tasks/{task_id}/stream",
            },
            status=426,
        )


# Handler factory
_a2a_handler: A2AHandler | None = None


def get_a2a_handler(server_context: dict[str, Any] | None = None) -> A2AHandler:
    """Get or create the A2A handler instance."""
    global _a2a_handler
    if _a2a_handler is None:
        if server_context is None:
            server_context = {}
        _a2a_handler = A2AHandler(server_context)
    return _a2a_handler


__all__ = [
    "A2AHandler",
    "get_a2a_handler",
    "get_a2a_server",
    "get_a2a_circuit_breaker",
    "get_a2a_circuit_breaker_status",
]
