"""
Notification event handlers for plan lifecycle events.

Subscribes to PLAN_COMPLETED/PLAN_FAILED events and routes
human-readable notifications through the NotificationService.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from aragora.events.types import StreamEvent

logger = logging.getLogger(__name__)


class NotificationHandlersMixin:
    """Mixin providing notification handlers for plan lifecycle events.

    This mixin requires the implementing class to provide:
    - stats: dict - Handler statistics tracking
    - _is_km_handler_enabled(handler_name: str) -> bool - Feature flag check
    """

    stats: dict
    _is_km_handler_enabled: Callable[[str], bool]

    def _handle_plan_completed_notification(self, event: StreamEvent) -> None:
        """Send notification when a plan completes successfully."""
        try:
            data = event.data
            plan_id = data.get("plan_id", "unknown")
            duration = data.get("duration_seconds", 0)
            tasks_completed = data.get("tasks_completed", 0)
            tasks_total = data.get("tasks_total", 0)
            execution_mode = data.get("execution_mode", "workflow")

            title = "Plan Execution Completed"
            message = (
                f"Plan {plan_id[:12]}... completed successfully in {duration:.1f}s. "
                f"Tasks: {tasks_completed}/{tasks_total} ({execution_mode} mode)."
            )

            self._send_plan_notification(
                title=title,
                message=message,
                severity="info",
                plan_id=plan_id,
                workspace_id=data.get("workspace_id", ""),
            )

            self.stats.setdefault("plan_notification", {"events": 0, "errors": 0})
            self.stats["plan_notification"]["events"] += 1

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error("Plan completion notification failed: %s", e)
            self.stats.setdefault("plan_notification", {"events": 0, "errors": 0})
            self.stats["plan_notification"]["errors"] += 1

    def _handle_plan_failed_notification(self, event: StreamEvent) -> None:
        """Send notification when a plan fails."""
        try:
            data = event.data
            plan_id = data.get("plan_id", "unknown")
            error = data.get("error", "Unknown error")
            duration = data.get("duration_seconds", 0)

            title = "Plan Execution Failed"
            message = f"Plan {plan_id[:12]}... failed after {duration:.1f}s: {error[:200]}"

            self._send_plan_notification(
                title=title,
                message=message,
                severity="warning",
                plan_id=plan_id,
                workspace_id=data.get("workspace_id", ""),
            )

            self.stats.setdefault("plan_notification", {"events": 0, "errors": 0})
            self.stats["plan_notification"]["events"] += 1

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error("Plan failure notification failed: %s", e)
            self.stats.setdefault("plan_notification", {"events": 0, "errors": 0})
            self.stats["plan_notification"]["errors"] += 1

    @staticmethod
    def _send_plan_notification(
        title: str,
        message: str,
        severity: str,
        plan_id: str,
        workspace_id: str = "",
    ) -> None:
        """Route notification through NotificationService."""
        try:
            from aragora.notifications.service import get_notification_service
            from aragora.notifications.models import Notification

            service = get_notification_service()
            notification = Notification(
                title=title,
                message=message,
                severity=severity,
                resource_type="plan",
                resource_id=plan_id,
            )
            # notify() is async - schedule it
            import asyncio

            coro = service.notify(notification)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                asyncio.run(coro)
            logger.debug("Sent plan notification: %s", title)

        except ImportError:
            logger.debug("NotificationService not available")
        except (RuntimeError, TypeError, AttributeError, ValueError, OSError) as e:
            logger.debug("Notification delivery failed (non-critical): %s", e)
