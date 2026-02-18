"""
Agent management handlers for Control Plane.

Provides REST API endpoints for:
- Agent registration and discovery
- Agent heartbeats and lifecycle management
- Agent unregistration
"""

from __future__ import annotations

import inspect
import logging
import sys
from typing import Any, cast

from aragora.server.http_utils import run_async as _run_async
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from aragora.server.handlers.openapi_decorator import api_endpoint
from aragora.server.handlers.utils.decorators import has_permission as _has_permission
from aragora.server.handlers.utils.decorators import require_permission

logger = logging.getLogger(__name__)


def _get_has_permission():
    control_plane = sys.modules.get("aragora.server.handlers.control_plane")
    if control_plane is not None:
        candidate = getattr(control_plane, "has_permission", None)
        if callable(candidate):
            return candidate
    return _has_permission


async def _await_if_needed(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


class AgentHandlerMixin:
    """
    Mixin class providing agent management handlers.

    This mixin provides methods for:
    - Listing registered agents
    - Getting agent details
    - Registering new agents
    - Handling agent heartbeats
    - Unregistering agents
    """

    # These methods are expected from the base class (ControlPlaneHandler).
    # Default implementations provide graceful degradation when the mixin
    # is used without the concrete class overriding them.
    def _get_coordinator(self) -> Any | None:
        """Get the control plane coordinator.

        Returns the coordinator instance, or None if not initialized.
        The concrete ControlPlaneHandler overrides this to pull from
        the class attribute or server context.
        """
        return getattr(self, "ctx", {}).get("control_plane_coordinator")

    def _require_coordinator(self) -> tuple[Any | None, HandlerResult | None]:
        """Return coordinator and None, or None and error response if not initialized."""
        coord = self._get_coordinator()
        if not coord:
            return None, error_response("Control plane not initialized", 503)
        return coord, None

    def _handle_coordinator_error(self, error: Exception, operation: str) -> HandlerResult:
        """Unified error handler for coordinator operations."""
        if isinstance(error, (ValueError, KeyError, AttributeError)):
            logger.warning(f"Data error in {operation}: {type(error).__name__}: {error}")
            return error_response(safe_error_message(error, "control plane"), 400)
        logger.error(f"Error in {operation}: {error}")
        return error_response(safe_error_message(error, "control plane"), 500)

    def _get_stream(self) -> Any | None:
        """Get the control plane stream server."""
        return getattr(self, "ctx", {}).get("control_plane_stream")

    def _emit_event(
        self,
        emit_method: str,
        *args: Any,
        max_retries: int = 3,
        base_delay: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Emit an event to the control plane stream.

        Default implementation is a no-op. The concrete ControlPlaneHandler
        overrides this with retry logic and async scheduling.
        """
        stream = self._get_stream()
        if not stream:
            return
        method = getattr(stream, emit_method, None)
        if method:
            try:
                _run_async(method(*args, **kwargs))
            except (RuntimeError, OSError, TypeError, ValueError, AttributeError) as e:
                logger.warning(f"Stream emission failed for {emit_method}: {e}")

    def require_auth_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]:
        """Require authentication and return user or error."""
        # Cast super() to Any - mixin expects base class to provide this method
        return cast(Any, super()).require_auth_or_error(handler)

    # Attribute declaration - provided by BaseHandler
    ctx: dict[str, Any]

    # =========================================================================
    # Agent Handlers
    # =========================================================================

    @api_endpoint(
        method="GET",
        path="/api/control-plane/agents",
        summary="List registered agents",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:agents.read")
    def _handle_list_agents(self, query_params: dict[str, Any]) -> HandlerResult:
        """List registered agents."""
        coordinator, err = self._require_coordinator()
        if err:
            return err

        capability = query_params.get("capability")
        only_available = query_params.get("available", "true").lower() == "true"

        try:
            agents = _run_async(
                coordinator.list_agents(
                    capability=capability,
                    only_available=only_available,
                )
            )

            return json_response(
                {
                    "agents": [a.to_dict() for a in agents],
                    "total": len(agents),
                }
            )
        except (RuntimeError, TimeoutError) as e:
            logger.error(f"Runtime error listing agents: {type(e).__name__}: {e}")
            return error_response(safe_error_message(e, "control plane"), 503)
        except (ValueError, KeyError, AttributeError, OSError, TypeError) as e:
            return self._handle_coordinator_error(e, "list_agents")

    @api_endpoint(
        method="GET",
        path="/api/control-plane/agents/{agent_id}",
        summary="Get agent by ID",
        tags=["Control Plane"],
    )
    @require_permission("controlplane:agents.read")
    def _handle_get_agent(self, agent_id: str) -> HandlerResult:
        """Get agent by ID."""
        coordinator, err = self._require_coordinator()
        if err:
            return err

        try:
            agent = _run_async(coordinator.get_agent(agent_id))

            if not agent:
                return error_response(f"Agent not found: {agent_id}", 404)

            return json_response(agent.to_dict())
        except (ValueError, KeyError, AttributeError, OSError, TypeError, RuntimeError) as e:
            return self._handle_coordinator_error(e, f"get_agent:{agent_id}")

    @api_endpoint(
        method="POST",
        path="/api/control-plane/agents",
        summary="Register a new agent",
        tags=["Control Plane"],
    )
    def _handle_register_agent(self, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Register a new agent."""
        # Require authentication for agent registration
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Check permission for agent management
        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:agents"
        ):
            return error_response("Permission denied", 403)

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
        except (ValueError, KeyError, AttributeError, OSError, TypeError, RuntimeError) as e:
            logger.error(f"Error registering agent: {e}")
            return error_response(safe_error_message(e, "control plane"), 500)

    async def _handle_register_agent_async(
        self, body: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Register a new agent (async context)."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:agents"
        ):
            return error_response("Permission denied", 403)

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
            agent = await _await_if_needed(
                coordinator.register_agent(
                    agent_id=agent_id,
                    capabilities=capabilities,
                    model=model,
                    provider=provider,
                    metadata=metadata,
                )
            )

            self._emit_event(
                "emit_agent_registered",
                agent_id=agent_id,
                capabilities=capabilities,
                model=model,
                provider=provider,
            )

            return json_response(agent.to_dict(), status=201)
        except (ValueError, KeyError, AttributeError, OSError, TypeError, RuntimeError) as e:
            logger.error(f"Error registering agent: {e}")
            return error_response(safe_error_message(e, "control plane"), 500)

    @api_endpoint(
        method="POST",
        path="/api/control-plane/agents/{agent_id}/heartbeat",
        summary="Send agent heartbeat",
        tags=["Control Plane"],
    )
    def _handle_heartbeat(self, agent_id: str, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Handle agent heartbeat."""
        # Require authentication for heartbeats
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Check permission for agent management
        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:agents"
        ):
            return error_response("Permission denied", 403)

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        status = body.get("status")

        try:
            from aragora.control_plane.registry import AgentStatus

            agent_status = AgentStatus(status) if status else None

            success = _run_async(coordinator.heartbeat(agent_id, agent_status))

            if not success:
                return error_response(f"Agent not found: {agent_id}", 404)

            return json_response({"acknowledged": True})
        except (ValueError, KeyError, AttributeError, OSError, TypeError, RuntimeError) as e:
            logger.error(f"Error processing heartbeat: {e}")
            return error_response(safe_error_message(e, "control plane"), 500)

    async def _handle_heartbeat_async(
        self, agent_id: str, body: dict[str, Any], handler: Any
    ) -> HandlerResult:
        """Handle agent heartbeat (async context)."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:agents"
        ):
            return error_response("Permission denied", 403)

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        status = body.get("status")

        try:
            from aragora.control_plane.registry import AgentStatus

            agent_status = AgentStatus(status) if status else None

            success = await _await_if_needed(coordinator.heartbeat(agent_id, agent_status))

            if not success:
                return error_response(f"Agent not found: {agent_id}", 404)

            return json_response({"acknowledged": True})
        except (ValueError, KeyError, AttributeError, OSError, TypeError, RuntimeError) as e:
            logger.error(f"Error processing heartbeat: {e}")
            return error_response(safe_error_message(e, "control plane"), 500)

    @api_endpoint(
        method="DELETE",
        path="/api/control-plane/agents/{agent_id}",
        summary="Unregister an agent",
        tags=["Control Plane"],
    )
    def _handle_unregister_agent(self, agent_id: str, handler: Any) -> HandlerResult:
        """Unregister an agent."""
        # Require authentication for agent unregistration
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        # Check permission for agent management
        if not _get_has_permission()(
            user.role if hasattr(user, "role") else None, "controlplane:agents"
        ):
            return error_response("Permission denied", 403)

        coordinator = self._get_coordinator()
        if not coordinator:
            return error_response("Control plane not initialized", 503)

        try:
            success = _run_async(coordinator.unregister_agent(agent_id))

            if not success:
                return error_response(f"Agent not found: {agent_id}", 404)

            # Emit event for real-time streaming
            self._emit_event(
                "emit_agent_unregistered",
                agent_id=agent_id,
                reason="manual_unregistration",
            )

            return json_response({"unregistered": True})
        except (ValueError, KeyError, AttributeError, OSError, TypeError, RuntimeError) as e:
            logger.error(f"Error unregistering agent: {e}")
            return error_response(safe_error_message(e, "control plane"), 500)
