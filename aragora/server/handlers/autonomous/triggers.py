"""Scheduled trigger HTTP handlers.

Stability: STABLE
- Circuit breaker protection for trigger operations
- All endpoints require authentication
"""

from __future__ import annotations

import logging

from aiohttp import web

from aragora.autonomous import ScheduledTrigger
from aragora.server.handlers.utils.auth import (
    get_auth_context,
    UnauthorizedError,
    ForbiddenError,
)
from aragora.server.handlers.utils import parse_json_body
from aragora.rbac.checker import get_permission_checker
from aragora.rbac.decorators import require_permission
from aragora.resilience import get_circuit_breaker

logger = logging.getLogger(__name__)

# Circuit breaker for trigger operations
_trigger_circuit_breaker = None


def _get_circuit_breaker():
    """Get or create circuit breaker for trigger operations."""
    global _trigger_circuit_breaker
    if _trigger_circuit_breaker is None:
        _trigger_circuit_breaker = get_circuit_breaker(
            "scheduled_triggers",
            failure_threshold=5,
            cooldown_seconds=30,
        )
    return _trigger_circuit_breaker


# RBAC permission keys for autonomous operations
AUTONOMOUS_READ_PERMISSION = "autonomous:read"
AUTONOMOUS_WRITE_PERMISSION = "autonomous:write"

# Global scheduled trigger instance
_scheduled_trigger: ScheduledTrigger | None = None


def get_scheduled_trigger() -> ScheduledTrigger:
    """Get or create the global scheduled trigger instance."""
    global _scheduled_trigger
    if _scheduled_trigger is None:
        _scheduled_trigger = ScheduledTrigger()
    return _scheduled_trigger


def set_scheduled_trigger(trigger: ScheduledTrigger) -> None:
    """Set the global scheduled trigger instance."""
    global _scheduled_trigger
    _scheduled_trigger = trigger


class TriggerHandler:
    """HTTP handlers for scheduled trigger operations."""

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    @staticmethod
    @require_permission("autonomous:triggers:read")
    async def list_triggers(request: web.Request) -> web.Response:
        """
        List all scheduled triggers.

        GET /api/autonomous/triggers

        Requires authentication and 'autonomous:read' permission.

        Returns:
            List of scheduled triggers
        """
        try:
            # Check circuit breaker
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Trigger service temporarily unavailable"},
                    status=503,
                )

            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_READ_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            trigger = get_scheduled_trigger()
            triggers = trigger.list_triggers()

            return web.json_response(
                {
                    "success": True,
                    "triggers": [
                        {
                            "id": t.id,
                            "name": t.name,
                            "interval_seconds": t.interval_seconds,
                            "cron_expression": t.cron_expression,
                            "enabled": t.enabled,
                            "last_run": t.last_run.isoformat() if t.last_run else None,
                            "next_run": t.next_run.isoformat() if t.next_run else None,
                            "run_count": t.run_count,
                            "max_runs": t.max_runs,
                            "metadata": t.metadata,
                        }
                        for t in triggers
                    ],
                    "count": len(triggers),
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized listing triggers: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden listing triggers: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error listing triggers: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to list triggers"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:triggers:write")
    async def add_trigger(request: web.Request) -> web.Response:
        """
        Add a new scheduled trigger.

        POST /api/autonomous/triggers

        Requires authentication and 'autonomous:write' permission.

        Body:
            trigger_id: str - Unique identifier
            name: str - Human-readable name
            interval_seconds: int (optional) - Run every N seconds
            cron_expression: str (optional) - Cron expression
            enabled: bool - Whether trigger is active
            max_runs: int (optional) - Maximum number of runs
            metadata: dict (optional) - Additional metadata (topic, agents, rounds)

        Returns:
            Created trigger
        """
        try:
            # Check circuit breaker
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Trigger service temporarily unavailable"},
                    status=503,
                )

            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_WRITE_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            data, err = await parse_json_body(request, context="add_trigger")
            if err:
                return err
            trigger_id = data.get("trigger_id")
            name = data.get("name")

            if not trigger_id or not name:
                return web.json_response(
                    {"success": False, "error": "trigger_id and name are required"},
                    status=400,
                )

            trigger = get_scheduled_trigger()
            config = trigger.add_trigger(  # type: ignore[call-arg]
                id=trigger_id,
                name=name,
                interval_seconds=data.get("interval_seconds"),
                cron_expression=data.get("cron_expression"),
                enabled=data.get("enabled", True),
                max_runs=data.get("max_runs"),
                metadata=data.get("metadata"),
            )

            return web.json_response(
                {
                    "success": True,
                    "trigger": {
                        "id": config.id,
                        "name": config.name,
                        "interval_seconds": config.interval_seconds,
                        "enabled": config.enabled,
                        "next_run": config.next_run.isoformat() if config.next_run else None,
                    },
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized adding trigger: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden adding trigger: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error adding trigger: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to create trigger"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:triggers:write")
    async def remove_trigger(request: web.Request) -> web.Response:
        """
        Remove a scheduled trigger.

        DELETE /api/autonomous/triggers/{trigger_id}

        Requires authentication and 'autonomous:write' permission.

        Returns:
            Success status
        """
        trigger_id = request.match_info.get("trigger_id")

        try:
            # Check circuit breaker
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Trigger service temporarily unavailable"},
                    status=503,
                )

            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_WRITE_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            trigger = get_scheduled_trigger()
            if hasattr(trigger, "remove_trigger"):
                success = trigger.remove_trigger(trigger_id)
            else:
                # Backward-compatible path for mocks without remove_trigger
                if hasattr(trigger, "_triggers"):
                    success = bool(trigger._triggers.pop(trigger_id, None))
                else:
                    success = False

            if not success:
                return web.json_response(
                    {"success": False, "error": "Trigger not found"},
                    status=404,
                )

            return web.json_response(
                {
                    "success": True,
                    "trigger_id": trigger_id,
                    "removed": True,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized removing trigger: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden removing trigger: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error removing trigger: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to remove trigger"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:triggers:write")
    async def enable_trigger(request: web.Request) -> web.Response:
        """
        Enable a scheduled trigger.

        POST /api/autonomous/triggers/{trigger_id}/enable

        Requires authentication and 'autonomous:write' permission.

        Returns:
            Success status
        """
        trigger_id = request.match_info.get("trigger_id")

        try:
            # Check circuit breaker
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Trigger service temporarily unavailable"},
                    status=503,
                )

            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_WRITE_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            trigger = get_scheduled_trigger()
            success = trigger.enable_trigger(trigger_id)

            if not success:
                return web.json_response(
                    {"success": False, "error": "Trigger not found"},
                    status=404,
                )

            return web.json_response(
                {
                    "success": True,
                    "trigger_id": trigger_id,
                    "enabled": True,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized enabling trigger: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden enabling trigger: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error enabling trigger: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to enable trigger"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:triggers:write")
    async def disable_trigger(request: web.Request) -> web.Response:
        """
        Disable a scheduled trigger.

        POST /api/autonomous/triggers/{trigger_id}/disable

        Requires authentication and 'autonomous:write' permission.

        Returns:
            Success status
        """
        trigger_id = request.match_info.get("trigger_id")

        try:
            # Check circuit breaker
            cb = _get_circuit_breaker()
            if not cb.can_execute():
                return web.json_response(
                    {"success": False, "error": "Trigger service temporarily unavailable"},
                    status=503,
                )

            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_WRITE_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            trigger = get_scheduled_trigger()
            success = trigger.disable_trigger(trigger_id)

            if not success:
                return web.json_response(
                    {"success": False, "error": "Trigger not found"},
                    status=404,
                )

            return web.json_response(
                {
                    "success": True,
                    "trigger_id": trigger_id,
                    "enabled": False,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized disabling trigger: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden disabling trigger: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error disabling trigger: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to disable trigger"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:triggers:write")
    async def start_scheduler(request: web.Request) -> web.Response:
        """
        Start the trigger scheduler.

        POST /api/autonomous/triggers/start

        Requires authentication and 'autonomous:write' permission.

        Returns:
            Success status
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_WRITE_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            trigger = get_scheduled_trigger()
            await trigger.start()

            return web.json_response(
                {
                    "success": True,
                    "scheduler_running": True,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized starting scheduler: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden starting scheduler: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.error("Error starting scheduler: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to start scheduler"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:triggers:write")
    async def stop_scheduler(request: web.Request) -> web.Response:
        """
        Stop the trigger scheduler.

        POST /api/autonomous/triggers/stop

        Requires authentication and 'autonomous:write' permission.

        Returns:
            Success status
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_WRITE_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            trigger = get_scheduled_trigger()
            await trigger.stop()

            return web.json_response(
                {
                    "success": True,
                    "scheduler_running": False,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized stopping scheduler: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden stopping scheduler: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
            logger.error("Error stopping scheduler: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to stop scheduler"},
                status=500,
            )

    @staticmethod
    def register_routes(app: web.Application, prefix: str = "/api/v1/autonomous") -> None:
        """Register trigger routes with the application."""
        app.router.add_get(
            f"{prefix}/triggers",
            TriggerHandler.list_triggers,
        )
        app.router.add_post(
            f"{prefix}/triggers",
            TriggerHandler.add_trigger,
        )
        app.router.add_delete(
            f"{prefix}/triggers/{{trigger_id}}",
            TriggerHandler.remove_trigger,
        )
        app.router.add_post(
            f"{prefix}/triggers/{{trigger_id}}/enable",
            TriggerHandler.enable_trigger,
        )
        app.router.add_post(
            f"{prefix}/triggers/{{trigger_id}}/disable",
            TriggerHandler.disable_trigger,
        )
        app.router.add_post(
            f"{prefix}/triggers/start",
            TriggerHandler.start_scheduler,
        )
        app.router.add_post(
            f"{prefix}/triggers/stop",
            TriggerHandler.stop_scheduler,
        )
