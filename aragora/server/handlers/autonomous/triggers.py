"""Scheduled trigger HTTP handlers."""

import logging
from typing import Optional

from aiohttp import web

from aragora.autonomous import ScheduledTrigger

logger = logging.getLogger(__name__)

# Global scheduled trigger instance
_scheduled_trigger: Optional[ScheduledTrigger] = None


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

    @staticmethod
    async def list_triggers(request: web.Request) -> web.Response:
        """
        List all scheduled triggers.

        GET /api/autonomous/triggers

        Returns:
            List of scheduled triggers
        """
        try:
            trigger = get_scheduled_trigger()
            triggers = trigger.list_triggers()

            return web.json_response({
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
            })

        except Exception as e:
            logger.error(f"Error listing triggers: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def add_trigger(request: web.Request) -> web.Response:
        """
        Add a new scheduled trigger.

        POST /api/autonomous/triggers

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
            data = await request.json()
            trigger_id = data.get("trigger_id")
            name = data.get("name")

            if not trigger_id or not name:
                return web.json_response(
                    {"success": False, "error": "trigger_id and name are required"},
                    status=400,
                )

            trigger = get_scheduled_trigger()
            config = trigger.add_trigger(
                trigger_id=trigger_id,
                name=name,
                interval_seconds=data.get("interval_seconds"),
                cron_expression=data.get("cron_expression"),
                enabled=data.get("enabled", True),
                max_runs=data.get("max_runs"),
                metadata=data.get("metadata"),
            )

            return web.json_response({
                "success": True,
                "trigger": {
                    "id": config.id,
                    "name": config.name,
                    "interval_seconds": config.interval_seconds,
                    "enabled": config.enabled,
                    "next_run": config.next_run.isoformat() if config.next_run else None,
                },
            })

        except Exception as e:
            logger.error(f"Error adding trigger: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def remove_trigger(request: web.Request) -> web.Response:
        """
        Remove a scheduled trigger.

        DELETE /api/autonomous/triggers/{trigger_id}

        Returns:
            Success status
        """
        trigger_id = request.match_info.get("trigger_id")

        try:
            trigger = get_scheduled_trigger()
            success = trigger.remove_trigger(trigger_id)

            if not success:
                return web.json_response(
                    {"success": False, "error": "Trigger not found"},
                    status=404,
                )

            return web.json_response({
                "success": True,
                "trigger_id": trigger_id,
                "removed": True,
            })

        except Exception as e:
            logger.error(f"Error removing trigger: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def enable_trigger(request: web.Request) -> web.Response:
        """
        Enable a scheduled trigger.

        POST /api/autonomous/triggers/{trigger_id}/enable

        Returns:
            Success status
        """
        trigger_id = request.match_info.get("trigger_id")

        try:
            trigger = get_scheduled_trigger()
            success = trigger.enable_trigger(trigger_id)

            if not success:
                return web.json_response(
                    {"success": False, "error": "Trigger not found"},
                    status=404,
                )

            return web.json_response({
                "success": True,
                "trigger_id": trigger_id,
                "enabled": True,
            })

        except Exception as e:
            logger.error(f"Error enabling trigger: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def disable_trigger(request: web.Request) -> web.Response:
        """
        Disable a scheduled trigger.

        POST /api/autonomous/triggers/{trigger_id}/disable

        Returns:
            Success status
        """
        trigger_id = request.match_info.get("trigger_id")

        try:
            trigger = get_scheduled_trigger()
            success = trigger.disable_trigger(trigger_id)

            if not success:
                return web.json_response(
                    {"success": False, "error": "Trigger not found"},
                    status=404,
                )

            return web.json_response({
                "success": True,
                "trigger_id": trigger_id,
                "enabled": False,
            })

        except Exception as e:
            logger.error(f"Error disabling trigger: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def start_scheduler(request: web.Request) -> web.Response:
        """
        Start the trigger scheduler.

        POST /api/autonomous/triggers/start

        Returns:
            Success status
        """
        try:
            trigger = get_scheduled_trigger()
            await trigger.start()

            return web.json_response({
                "success": True,
                "scheduler_running": True,
            })

        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def stop_scheduler(request: web.Request) -> web.Response:
        """
        Stop the trigger scheduler.

        POST /api/autonomous/triggers/stop

        Returns:
            Success status
        """
        try:
            trigger = get_scheduled_trigger()
            await trigger.stop()

            return web.json_response({
                "success": True,
                "scheduler_running": False,
            })

        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    def register_routes(app: web.Application, prefix: str = "/api/autonomous") -> None:
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
