"""
Notification preferences endpoint.

Provides:
- GET /api/v1/notifications/preferences (get user notification preferences)
- PUT /api/v1/notifications/preferences (update user preferences)
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from ..utils.decorators import require_permission
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

_preferences_limiter = RateLimiter(requests_per_minute=30)

# Default notification preferences
_DEFAULT_PREFERENCES: dict[str, Any] = {
    "channels": {
        "slack": True,
        "email": True,
        "webhook": True,
    },
    "event_types": {
        "finding_created": True,
        "audit_completed": True,
        "checkpoint_approval": True,
        "budget_alert": True,
        "compliance_finding": True,
        "workflow_progress": True,
        "cost_anomaly": True,
        "debate_completed": True,
    },
    "quiet_hours": {
        "enabled": False,
        "start": "22:00",
        "end": "08:00",
        "timezone": "UTC",
    },
    "digest_mode": False,
}

# In-memory preferences store (production would use a database)
_user_preferences: dict[str, dict[str, Any]] = {}


class NotificationPreferencesHandler(BaseHandler):
    """Handler for notification preferences."""

    ROUTES = ["/api/v1/notifications/preferences"]

    def __init__(self, ctx: dict[str, Any] | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        cleaned = strip_version_prefix(path)
        return cleaned == "/api/notifications/preferences"

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle GET for preferences."""
        cleaned = strip_version_prefix(path)
        if cleaned != "/api/notifications/preferences":
            return None

        client_ip = get_client_ip(handler)
        if not _preferences_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded. Please try again later.", 429)

        return self._get_preferences(handler)

    @require_permission("notifications:manage_preferences")
    def handle_put(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle PUT for updating preferences."""
        cleaned = strip_version_prefix(path)
        if cleaned != "/api/notifications/preferences":
            return None

        client_ip = get_client_ip(handler)
        if not _preferences_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded. Please try again later.", 429)

        return self._update_preferences(handler)

    def _get_user_id(self, handler: Any) -> str:
        """Extract user ID from request context."""
        user = self.get_current_user(handler)
        if user and hasattr(user, "user_id"):
            return user.user_id
        # Fallback for testing
        return "anonymous"

    def _get_preferences(self, handler: Any) -> HandlerResult:
        """Get notification preferences for current user."""
        user_id = self._get_user_id(handler)

        prefs = _user_preferences.get(user_id)
        if prefs is None:
            prefs = copy.deepcopy(_DEFAULT_PREFERENCES)

        return json_response({
            "user_id": user_id,
            "preferences": prefs,
        })

    def _update_preferences(self, handler: Any) -> HandlerResult:
        """Update notification preferences for current user."""
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        user_id = self._get_user_id(handler)

        # Get existing prefs or defaults
        current = _user_preferences.get(user_id, copy.deepcopy(_DEFAULT_PREFERENCES))

        # Validate and merge updates
        updates = body.get("preferences", body)

        # Validate channels
        if "channels" in updates:
            channels = updates["channels"]
            if not isinstance(channels, dict):
                return error_response("'channels' must be an object", 400)
            valid_channels = {"slack", "email", "webhook"}
            for ch in channels:
                if ch not in valid_channels:
                    return error_response(
                        f"Invalid channel: {ch}. Valid: {', '.join(sorted(valid_channels))}",
                        400,
                    )
                if not isinstance(channels[ch], bool):
                    return error_response(f"Channel '{ch}' value must be boolean", 400)
            if isinstance(current.get("channels"), dict):
                current["channels"].update(channels)
            else:
                current["channels"] = channels

        # Validate event_types
        if "event_types" in updates:
            event_types = updates["event_types"]
            if not isinstance(event_types, dict):
                return error_response("'event_types' must be an object", 400)
            for et, enabled in event_types.items():
                if not isinstance(enabled, bool):
                    return error_response(f"Event type '{et}' value must be boolean", 400)
            if isinstance(current.get("event_types"), dict):
                current["event_types"].update(event_types)
            else:
                current["event_types"] = event_types

        # Validate quiet hours
        if "quiet_hours" in updates:
            qh = updates["quiet_hours"]
            if not isinstance(qh, dict):
                return error_response("'quiet_hours' must be an object", 400)
            if isinstance(current.get("quiet_hours"), dict):
                current["quiet_hours"].update(qh)
            else:
                current["quiet_hours"] = qh

        # Validate digest mode
        if "digest_mode" in updates:
            if not isinstance(updates["digest_mode"], bool):
                return error_response("'digest_mode' must be boolean", 400)
            current["digest_mode"] = updates["digest_mode"]

        _user_preferences[user_id] = current

        return json_response({
            "user_id": user_id,
            "preferences": current,
            "updated": True,
        })
