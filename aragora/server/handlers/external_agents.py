"""External Agent Gateway endpoint handlers.

Endpoints:
- POST /api/external-agents/tasks          - Submit task to external agent
- GET  /api/external-agents/tasks/{id}     - Get task status/result
- DELETE /api/external-agents/tasks/{id}   - Cancel task
- GET  /api/external-agents/adapters       - List registered adapters
- GET  /api/external-agents/health         - Health check all adapters
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_bounded_string_param,
    json_response,
)
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiters
_submit_limiter = RateLimiter(requests_per_minute=10)
_read_limiter = RateLimiter(requests_per_minute=60)


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result(timeout=120)
    return asyncio.run(coro)


class ExternalAgentsHandler(BaseHandler):
    """Handler for external agent gateway endpoints."""

    ROUTES = [
        "/api/external-agents/tasks",
        "/api/external-agents/tasks/*",
        "/api/external-agents/adapters",
        "/api/external-agents/health",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        path = strip_version_prefix(path)
        if path in self.ROUTES:
            return True
        if path.startswith("/api/external-agents/tasks/"):
            return True
        return False

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Handle GET requests."""
        path = strip_version_prefix(path)

        # Rate limit
        client_ip = get_client_ip(handler)
        if not _read_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        # Auth
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Route
        if path == "/api/external-agents/adapters":
            return self._list_adapters()

        if path == "/api/external-agents/health":
            adapter_name = get_bounded_string_param(query_params, "adapter", "", max_length=100)
            return self._health_check(adapter_name or None)

        if path.startswith("/api/external-agents/tasks/"):
            task_id = path.split("/api/external-agents/tasks/", 1)[1]
            if not task_id or "/" in task_id:
                return error_response("Invalid task ID", 400)
            return self._get_task(task_id, user)

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests."""
        path = strip_version_prefix(path)

        if path != "/api/external-agents/tasks":
            return None

        # Rate limit (stricter for submissions)
        client_ip = get_client_ip(handler)
        if not _submit_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        # Auth
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Parse body
        self.set_request_context(handler, query_params)
        body, body_err = self.read_json_body_validated(handler)
        if body_err:
            return body_err

        return self._submit_task(body, user)

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle DELETE requests."""
        path = strip_version_prefix(path)

        if not path.startswith("/api/external-agents/tasks/"):
            return None

        # Rate limit
        client_ip = get_client_ip(handler)
        if not _read_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        # Auth
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        task_id = path.split("/api/external-agents/tasks/", 1)[1]
        if not task_id or "/" in task_id:
            return error_response("Invalid task ID", 400)

        return self._cancel_task(task_id, user)

    # =========================================================================
    # Internal handlers
    # =========================================================================

    def _list_adapters(self) -> HandlerResult:
        """List registered external agent adapters."""
        try:
            from aragora.agents.external.registry import ExternalAgentRegistry

            adapters = []
            for spec in ExternalAgentRegistry.list_specs():
                adapters.append(
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "config_class": spec.config_class.__name__,
                    }
                )

            return json_response(
                {
                    "adapters": adapters,
                    "total": len(adapters),
                }
            )
        except Exception as e:
            logger.error(f"Failed to list adapters: {e}")
            return error_response("Failed to list adapters", 500)

    def _health_check(self, adapter_name: str | None = None) -> HandlerResult:
        """Health check adapters."""
        try:
            from aragora.agents.external.config import ExternalAgentConfig
            from aragora.agents.external.registry import ExternalAgentRegistry

            results = []
            specs = ExternalAgentRegistry.list_specs()

            for spec in specs:
                if adapter_name and spec.name != adapter_name:
                    continue

                try:
                    config = (
                        spec.config_class()
                        if spec.config_class != ExternalAgentConfig
                        else ExternalAgentConfig(adapter_name=spec.name)
                    )
                    adapter = spec.adapter_class(config)
                    health = _run_coro(adapter.health_check())
                    results.append(health.to_dict())
                except Exception as e:
                    results.append(
                        {
                            "adapter_name": spec.name,
                            "healthy": False,
                            "error": str(e),
                        }
                    )

            return json_response(
                {
                    "health": results,
                    "total": len(results),
                }
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return error_response("Health check failed", 500)

    def _submit_task(self, body: dict[str, Any], user: Any) -> HandlerResult:
        """Submit a task to an external agent."""
        try:
            from aragora.agents.external.models import TaskRequest, ToolPermission
            from aragora.agents.external.proxy import (
                ExternalAgentProxy,
                PolicyDeniedError,
                ProxyConfig,
            )
            from aragora.agents.external.registry import ExternalAgentRegistry

            # Validate required fields
            task_type = body.get("task_type")
            prompt = body.get("prompt")
            if not task_type or not prompt:
                return error_response("task_type and prompt are required", 400)

            if len(prompt) > 10000:
                return error_response("Prompt too long (max 10000 chars)", 400)

            # Get adapter
            adapter_name = body.get("adapter", "openhands")
            if not ExternalAgentRegistry.is_registered(adapter_name):
                return error_response(
                    f"Unknown adapter: {adapter_name}. "
                    f"Available: {list(ExternalAgentRegistry.get_registered_names())}",
                    400,
                )

            adapter = ExternalAgentRegistry.create(adapter_name)

            # Parse tool permissions
            tool_permissions: set[ToolPermission] = set()
            for perm_str in body.get("tool_permissions", []):
                try:
                    tool_permissions.add(ToolPermission(perm_str))
                except ValueError:
                    return error_response(f"Invalid tool permission: {perm_str}", 400)

            # Build request
            user_id = str(user.id) if hasattr(user, "id") else str(user.get("id", "unknown"))
            tenant_id = user.org_id if hasattr(user, "org_id") else user.get("org_id")

            request = TaskRequest(
                task_type=task_type,
                prompt=prompt,
                context=body.get("context", {}),
                tool_permissions=tool_permissions,
                timeout_seconds=min(body.get("timeout_seconds", 3600.0), 7200.0),
                max_steps=min(body.get("max_steps", 100), 500),
                workspace_id=body.get("workspace_id"),
                user_id=user_id,
                tenant_id=tenant_id,
                metadata=body.get("metadata", {}),
            )

            # Build auth context for proxy
            from aragora.rbac.models import AuthorizationContext

            roles = (
                set(user.roles) if hasattr(user, "roles") else set(user.get("roles", ["member"]))
            )
            auth_context = AuthorizationContext(
                user_id=user_id,
                org_id=tenant_id,
                roles=roles,
                permissions=set(),
            )

            # Submit via proxy
            proxy = ExternalAgentProxy(
                adapter,
                auth_context,
                ProxyConfig(enable_policy_checks=True),
            )

            start_time = time.perf_counter()
            try:
                task_id = _run_coro(proxy.submit_task(request))
            except PolicyDeniedError as e:
                return error_response(f"Policy denied: {e.reason}", 403)

            duration = time.perf_counter() - start_time

            # Record metrics
            _record_metrics("submit", adapter_name, task_type, duration)

            return json_response(
                {
                    "task_id": task_id,
                    "status": "pending",
                    "adapter": adapter_name,
                },
                status=201,
            )

        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            return error_response(f"Task submission failed: {str(e)}", 500)

    def _get_task(self, task_id: str, user: Any) -> HandlerResult:
        """Get task status and result."""
        try:
            from aragora.agents.external.models import TaskStatus
            from aragora.agents.external.proxy import ExternalAgentProxy, ProxyConfig
            from aragora.agents.external.registry import ExternalAgentRegistry

            # Determine adapter from task_id prefix or default
            adapter_name = task_id.split("-")[0] if "-" in task_id else "openhands"
            if not ExternalAgentRegistry.is_registered(adapter_name):
                adapter_name = "openhands"

            if not ExternalAgentRegistry.is_registered(adapter_name):
                return error_response(f"No adapter available for task {task_id}", 404)

            adapter = ExternalAgentRegistry.create(adapter_name)

            user_id = str(user.id) if hasattr(user, "id") else str(user.get("id", "unknown"))
            from aragora.rbac.models import AuthorizationContext

            auth_context = AuthorizationContext(
                user_id=user_id,
                roles=set(),
                permissions=set(),
            )

            proxy = ExternalAgentProxy(
                adapter,
                auth_context,
                ProxyConfig(enable_policy_checks=False),
            )

            status = _run_coro(proxy.get_task_status(task_id))

            response: dict[str, Any] = {
                "task_id": task_id,
                "status": status.value,
            }

            # Include result if terminal
            if status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT,
            ):
                result = _run_coro(proxy.get_task_result(task_id))
                response["result"] = result.to_dict()

            return json_response(response)

        except KeyError:
            return error_response(f"Task not found: {task_id}", 404)
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return error_response(f"Failed to get task: {str(e)}", 500)

    def _cancel_task(self, task_id: str, user: Any) -> HandlerResult:
        """Cancel a running task."""
        try:
            from aragora.agents.external.proxy import ExternalAgentProxy, ProxyConfig
            from aragora.agents.external.registry import ExternalAgentRegistry

            adapter_name = task_id.split("-")[0] if "-" in task_id else "openhands"
            if not ExternalAgentRegistry.is_registered(adapter_name):
                adapter_name = "openhands"

            if not ExternalAgentRegistry.is_registered(adapter_name):
                return error_response(f"No adapter available for task {task_id}", 404)

            adapter = ExternalAgentRegistry.create(adapter_name)

            user_id = str(user.id) if hasattr(user, "id") else str(user.get("id", "unknown"))
            from aragora.rbac.models import AuthorizationContext

            auth_context = AuthorizationContext(
                user_id=user_id,
                roles=set(),
                permissions=set(),
            )

            proxy = ExternalAgentProxy(
                adapter,
                auth_context,
                ProxyConfig(enable_policy_checks=False),
            )

            cancelled = _run_coro(proxy.cancel_task(task_id))

            return json_response(
                {
                    "task_id": task_id,
                    "cancelled": cancelled,
                }
            )

        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return error_response(f"Failed to cancel task: {str(e)}", 500)


def _record_metrics(
    operation: str,
    adapter: str,
    task_type: str = "",
    duration: float = 0.0,
) -> None:
    """Record Prometheus metrics for external agent operations."""
    try:
        from aragora.server.prometheus import (
            record_external_agent_task,
            record_external_agent_duration,
        )

        if operation == "submit":
            record_external_agent_task(adapter, "submitted")
            if duration > 0:
                record_external_agent_duration(adapter, task_type, duration)
    except ImportError:
        pass  # Prometheus not available
    except Exception as e:
        logger.debug(f"Metrics recording failed: {e}")
