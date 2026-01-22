"""
HTTP Handler for Knowledge Sharing Notifications.

Provides endpoints for managing in-app notifications:
- GET /api/knowledge/notifications - Get notifications for current user
- GET /api/knowledge/notifications/count - Get unread count
- POST /api/knowledge/notifications/{id}/read - Mark notification as read
- POST /api/knowledge/notifications/read-all - Mark all as read
- POST /api/knowledge/notifications/{id}/dismiss - Dismiss notification
- GET /api/knowledge/notifications/preferences - Get notification preferences
- PUT /api/knowledge/notifications/preferences - Update preferences
"""

import logging
from typing import Any, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for notification endpoints
_notifications_limiter = RateLimiter(requests_per_minute=60)


class SharingNotificationsHandler(BaseHandler):
    """Handler for knowledge sharing notification endpoints.

    Endpoints:
        GET  /api/knowledge/notifications - Get notifications
        GET  /api/knowledge/notifications/count - Get unread count
        POST /api/knowledge/notifications/{id}/read - Mark as read
        POST /api/knowledge/notifications/read-all - Mark all as read
        POST /api/knowledge/notifications/{id}/dismiss - Dismiss
        GET  /api/knowledge/notifications/preferences - Get preferences
        PUT  /api/knowledge/notifications/preferences - Update preferences
    """

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/knowledge/notifications")

    def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Handle GET requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _notifications_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        # Require authentication
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        user_id = user.get("sub") or user.get("user_id") or user.get("id", "anonymous")  # type: ignore[union-attr]

        if path == "/api/v1/knowledge/notifications":
            return self._get_notifications(user_id, query_params)

        if path == "/api/v1/knowledge/notifications/count":
            return self._get_unread_count(user_id)

        if path == "/api/v1/knowledge/notifications/preferences":
            return self._get_preferences(user_id)

        return None

    def handle_post(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _notifications_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        # Require authentication
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        user_id = user.get("sub") or user.get("user_id") or user.get("id", "anonymous")  # type: ignore[union-attr]

        if path == "/api/v1/knowledge/notifications/read-all":
            return self._mark_all_read(user_id)

        # Check for notification ID in path
        if path.startswith("/api/v1/knowledge/notifications/") and "/read" in path:
            # Extract notification ID: /api/knowledge/notifications/{id}/read
            parts = path.split("/")
            if len(parts) >= 5:
                notification_id = parts[4]
                return self._mark_read(user_id, notification_id)

        if path.startswith("/api/v1/knowledge/notifications/") and "/dismiss" in path:
            parts = path.split("/")
            if len(parts) >= 5:
                notification_id = parts[4]
                return self._dismiss(user_id, notification_id)

        return None

    def handle_put(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Handle PUT requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _notifications_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        # Require authentication
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        user_id = user.get("sub") or user.get("user_id") or user.get("id", "anonymous")  # type: ignore[union-attr]

        if path == "/api/v1/knowledge/notifications/preferences":
            return self._update_preferences(user_id, handler)

        return None

    def _get_notifications(
        self,
        user_id: str,
        query_params: dict[str, Any],
    ) -> HandlerResult:
        """Get notifications for the current user."""
        try:
            from aragora.knowledge.mound.notifications import (
                NotificationStatus,
                get_notifications_for_user,
            )

            # Parse query params
            limit = int(query_params.get("limit", 20))
            offset = int(query_params.get("offset", 0))
            status_str = query_params.get("status")

            status = None
            if status_str:
                try:
                    status = NotificationStatus(status_str)
                except ValueError:
                    return error_response(f"Invalid status: {status_str}", 400)

            notifications = get_notifications_for_user(
                user_id=user_id,
                status=status,
                limit=min(limit, 100),  # Cap at 100
                offset=offset,
            )

            return json_response(
                {
                    "notifications": [n.to_dict() for n in notifications],
                    "count": len(notifications),
                    "limit": limit,
                    "offset": offset,
                }
            )

        except Exception as e:
            logger.error(f"Failed to get notifications: {e}")
            return error_response("Failed to get notifications", 500)

    def _get_unread_count(self, user_id: str) -> HandlerResult:
        """Get count of unread notifications."""
        try:
            from aragora.knowledge.mound.notifications import get_unread_count

            count = get_unread_count(user_id)
            return json_response({"unread_count": count})

        except Exception as e:
            logger.error(f"Failed to get unread count: {e}")
            return error_response("Failed to get unread count", 500)

    def _mark_read(self, user_id: str, notification_id: str) -> HandlerResult:
        """Mark a notification as read."""
        try:
            from aragora.knowledge.mound.notifications import mark_notification_read

            success = mark_notification_read(notification_id, user_id)
            if success:
                return json_response({"success": True, "notification_id": notification_id})
            else:
                return error_response("Notification not found", 404)

        except Exception as e:
            logger.error(f"Failed to mark notification read: {e}")
            return error_response("Failed to mark notification read", 500)

    def _mark_all_read(self, user_id: str) -> HandlerResult:
        """Mark all notifications as read."""
        try:
            from aragora.knowledge.mound.notifications import mark_all_notifications_read

            count = mark_all_notifications_read(user_id)
            return json_response({"success": True, "count": count})

        except Exception as e:
            logger.error(f"Failed to mark all notifications read: {e}")
            return error_response("Failed to mark all notifications read", 500)

    def _dismiss(self, user_id: str, notification_id: str) -> HandlerResult:
        """Dismiss a notification."""
        try:
            from aragora.knowledge.mound.notifications import get_notification_store

            store = get_notification_store()
            success = store.dismiss_notification(notification_id, user_id)
            if success:
                return json_response({"success": True, "notification_id": notification_id})
            else:
                return error_response("Notification not found", 404)

        except Exception as e:
            logger.error(f"Failed to dismiss notification: {e}")
            return error_response("Failed to dismiss notification", 500)

    def _get_preferences(self, user_id: str) -> HandlerResult:
        """Get notification preferences."""
        try:
            from aragora.knowledge.mound.notifications import get_notification_preferences

            prefs = get_notification_preferences(user_id)
            return json_response(
                {
                    "user_id": prefs.user_id,
                    "email_on_share": prefs.email_on_share,
                    "email_on_unshare": prefs.email_on_unshare,
                    "email_on_permission_change": prefs.email_on_permission_change,
                    "in_app_enabled": prefs.in_app_enabled,
                    "telegram_enabled": prefs.telegram_enabled,
                    "webhook_url": prefs.webhook_url,
                    "quiet_hours_start": prefs.quiet_hours_start,
                    "quiet_hours_end": prefs.quiet_hours_end,
                }
            )

        except Exception as e:
            logger.error(f"Failed to get preferences: {e}")
            return error_response("Failed to get preferences", 500)

    def _update_preferences(self, user_id: str, handler: Any) -> HandlerResult:
        """Update notification preferences."""
        try:
            from aragora.knowledge.mound.notifications import (
                NotificationPreferences,
                set_notification_preferences,
            )

            body, err = self.read_json_body_validated(handler)
            if err:
                return err

            prefs = NotificationPreferences(
                user_id=user_id,
                email_on_share=body.get("email_on_share", True),
                email_on_unshare=body.get("email_on_unshare", False),
                email_on_permission_change=body.get("email_on_permission_change", True),
                in_app_enabled=body.get("in_app_enabled", True),
                telegram_enabled=body.get("telegram_enabled", False),
                webhook_url=body.get("webhook_url"),
                quiet_hours_start=body.get("quiet_hours_start"),
                quiet_hours_end=body.get("quiet_hours_end"),
            )

            set_notification_preferences(prefs)

            return json_response(
                {
                    "success": True,
                    "message": "Preferences updated",
                }
            )

        except Exception as e:
            logger.error(f"Failed to update preferences: {e}")
            return error_response("Failed to update preferences", 500)


__all__ = ["SharingNotificationsHandler"]
