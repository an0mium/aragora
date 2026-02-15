"""
Notification history and delivery stats endpoints.

Provides:
- GET /api/v1/notifications/history (paginated notification history)
- GET /api/v1/notifications/delivery-stats (success rate, DLQ count)
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.versioning.compat import strip_version_prefix

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_int_param,
    get_string_param,
    json_response,
)
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

_notification_history_limiter = RateLimiter(requests_per_minute=30)


class NotificationHistoryHandler(BaseHandler):
    """Handler for notification history and delivery stats."""

    ROUTES = ["/api/v1/notifications/history", "/api/v1/notifications/delivery-stats"]

    def __init__(self, ctx: dict[str, Any] | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        cleaned = strip_version_prefix(path)
        return cleaned in (
            "/api/notifications/history",
            "/api/notifications/delivery-stats",
        )

    @require_permission("notifications:read")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        cleaned = strip_version_prefix(path)

        client_ip = get_client_ip(handler)
        if not _notification_history_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if cleaned == "/api/notifications/history":
            return self._get_history(query_params)
        elif cleaned == "/api/notifications/delivery-stats":
            return self._get_delivery_stats()

        return None

    def _get_notification_service(self) -> Any:
        """Get notification service from context or global."""
        service = self.ctx.get("notification_service")
        if service:
            return service
        try:
            from aragora.notifications.service import get_notification_service

            return get_notification_service()
        except (ImportError, Exception):
            return None

    def _get_history(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get paginated notification history."""
        limit = get_int_param(query_params, "limit", 50)
        limit = max(1, min(limit, 200))
        offset = get_int_param(query_params, "offset", 0)
        offset = max(0, offset)
        channel_filter = get_string_param(query_params, "channel")

        service = self._get_notification_service()
        if not service:
            return error_response("Notification service not available", 503)

        try:
            # Map channel string to enum if provided
            channel_enum = None
            if channel_filter:
                try:
                    from aragora.notifications.models import NotificationChannel

                    channel_enum = NotificationChannel(channel_filter.lower())
                except (ValueError, ImportError):
                    return error_response(
                        f"Invalid channel: {channel_filter}. Valid: slack, email, webhook",
                        400,
                    )

            history = service.get_history(
                limit=limit + offset,  # Fetch enough for offset
                channel=channel_enum,
            )

            # Apply offset
            paginated = history[offset : offset + limit]

            return json_response(
                {
                    "notifications": paginated,
                    "count": len(paginated),
                    "total": len(history),
                    "limit": limit,
                    "offset": offset,
                    "channel": channel_filter,
                }
            )
        except Exception as e:
            logger.error("Notification history failed: %s: %s", type(e).__name__, e)
            return error_response("Failed to get notification history", 500)

    def _get_delivery_stats(self) -> HandlerResult:
        """Get notification delivery statistics."""
        service = self._get_notification_service()
        if not service:
            return error_response("Notification service not available", 503)

        try:
            history = service.get_history(limit=1000)

            total = 0
            success = 0
            failed = 0
            by_channel: dict[str, dict[str, int]] = {}

            for entry in history:
                results = entry.get("results", [])
                for r in results:
                    total += 1
                    ch = r.get("channel", "unknown")
                    if ch not in by_channel:
                        by_channel[ch] = {"total": 0, "success": 0, "failed": 0}
                    by_channel[ch]["total"] += 1

                    if r.get("success"):
                        success += 1
                        by_channel[ch]["success"] += 1
                    else:
                        failed += 1
                        by_channel[ch]["failed"] += 1

            success_rate = (success / total * 100) if total > 0 else 0.0

            # Get DLQ count if dispatcher available
            dlq_count = 0
            try:
                from aragora.control_plane.notifications import get_notification_dispatcher

                dispatcher = get_notification_dispatcher()
                if dispatcher and hasattr(dispatcher, "dead_letter_count"):
                    dlq_count = dispatcher.dead_letter_count
            except (ImportError, Exception):
                pass

            return json_response(
                {
                    "total_notifications": total,
                    "successful": success,
                    "failed": failed,
                    "success_rate": round(success_rate, 1),
                    "dlq_count": dlq_count,
                    "by_channel": by_channel,
                }
            )
        except Exception as e:
            logger.error("Delivery stats failed: %s: %s", type(e).__name__, e)
            return error_response("Failed to get delivery stats", 500)
