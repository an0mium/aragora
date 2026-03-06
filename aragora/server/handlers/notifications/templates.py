"""
Notification template override endpoints.

Provides:
- GET  /api/v1/notifications/templates          (list built-in templates)
- GET  /api/v1/notifications/templates/overrides (get user overrides)
- PUT  /api/v1/notifications/templates/overrides (save override)
- DELETE /api/v1/notifications/templates/overrides/<template_id> (delete override)

Template overrides are persisted via ``NotificationTemplateStore`` so they
survive server restarts.
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
    handle_errors,
)
from ..utils.decorators import require_permission
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

_templates_limiter = RateLimiter(requests_per_minute=30)


# ---------------------------------------------------------------------------
# Built-in notification templates
# ---------------------------------------------------------------------------

BUILT_IN_TEMPLATES: dict[str, dict[str, Any]] = {
    "debate_completed": {
        "id": "debate_completed",
        "name": "Debate Completed",
        "subject": "Debate completed: {task}",
        "body": "The debate on '{task}' has concluded. Consensus: {consensus}.",
        "channels": ["slack", "email"],
    },
    "finding_created": {
        "id": "finding_created",
        "name": "Finding Created",
        "subject": "New finding: {title}",
        "body": "A new {severity} finding has been created: {title}.",
        "channels": ["slack", "email", "webhook"],
    },
    "budget_alert": {
        "id": "budget_alert",
        "name": "Budget Alert",
        "subject": "Budget alert: {threshold}% spent",
        "body": "Your budget has reached {threshold}% of the limit.",
        "channels": ["slack", "email"],
    },
    "consensus_reached": {
        "id": "consensus_reached",
        "name": "Consensus Reached",
        "subject": "Consensus reached on: {task}",
        "body": "Multi-agent consensus reached with {confidence}% confidence.",
        "channels": ["slack"],
    },
    "workflow_progress": {
        "id": "workflow_progress",
        "name": "Workflow Progress",
        "subject": "Workflow update: {workflow_name}",
        "body": "Workflow '{workflow_name}' is now {status}.",
        "channels": ["email", "webhook"],
    },
}


def _get_template_store():
    """Lazy-load the template store to avoid circular imports."""
    from aragora.storage.notification_template_store import (
        get_notification_template_store,
    )

    return get_notification_template_store()


class NotificationTemplateHandler(BaseHandler):
    """Handler for notification template overrides."""

    ROUTES = [
        "/api/v1/notifications/templates",
        "/api/v1/notifications/templates/overrides",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        cleaned = strip_version_prefix(path)
        return cleaned in (
            "/api/notifications/templates",
            "/api/notifications/templates/overrides",
        ) or cleaned.startswith("/api/notifications/templates/overrides/")

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Handle GET requests for templates and overrides."""
        cleaned = strip_version_prefix(path)

        client_ip = get_client_ip(handler)
        if not _templates_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if cleaned == "/api/notifications/templates":
            return self._list_templates()

        if cleaned == "/api/notifications/templates/overrides":
            return self._get_overrides(handler)

        return None

    @handle_errors("notification template override update")
    @require_permission("notifications:manage_templates")
    def handle_put(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle PUT for saving a template override."""
        cleaned = strip_version_prefix(path)
        if cleaned != "/api/notifications/templates/overrides":
            return None

        client_ip = get_client_ip(handler)
        if not _templates_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded. Please try again later.", 429)

        return self._save_override(handler)

    @handle_errors("notification template override delete")
    @require_permission("notifications:manage_templates")
    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle DELETE for removing a template override."""
        cleaned = strip_version_prefix(path)
        prefix = "/api/notifications/templates/overrides/"
        if not cleaned.startswith(prefix):
            return None

        client_ip = get_client_ip(handler)
        if not _templates_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded. Please try again later.", 429)

        template_id = cleaned[len(prefix) :]
        if not template_id:
            return error_response("Missing template_id in path", 400)

        return self._delete_override(handler, template_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_user_id(self, handler: Any) -> str:
        """Extract user ID from request context."""
        user = self.get_current_user(handler)
        if user and hasattr(user, "user_id"):
            return user.user_id
        return "anonymous"

    def _list_templates(self) -> HandlerResult:
        """List all built-in notification templates."""
        templates = [copy.deepcopy(t) for t in BUILT_IN_TEMPLATES.values()]
        return json_response({"templates": templates, "count": len(templates)})

    def _get_overrides(self, handler: Any) -> HandlerResult:
        """Get template overrides for the current user."""
        user_id = self._get_user_id(handler)

        try:
            store = _get_template_store()
            # Handler is sync; use the thread-safe file read directly
            overrides = store._read_user_file(user_id)

            return json_response(
                {
                    "user_id": user_id,
                    "overrides": overrides,
                    "count": len(overrides),
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Failed to load template overrides: %s", exc)
            return error_response("Failed to load template overrides", 500)

    def _save_override(self, handler: Any) -> HandlerResult:
        """Save a template override for the current user."""
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        template_id = body.get("template_id")
        if not template_id:
            return error_response("Missing 'template_id' in request body", 400)

        overrides = body.get("overrides")
        if not isinstance(overrides, dict):
            return error_response("'overrides' must be an object", 400)

        user_id = self._get_user_id(handler)

        try:
            store = _get_template_store()
            # Sync write (handler is sync)
            with store._lock:
                data = store._read_user_file(user_id)
                data[template_id] = overrides
                store._write_user_file(user_id, data)

            return json_response(
                {
                    "user_id": user_id,
                    "template_id": template_id,
                    "overrides": overrides,
                    "saved": True,
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Failed to save template override: %s", exc)
            return error_response("Failed to save template override", 500)

    def _delete_override(self, handler: Any, template_id: str) -> HandlerResult:
        """Delete a template override for the current user."""
        user_id = self._get_user_id(handler)

        try:
            store = _get_template_store()
            with store._lock:
                data = store._read_user_file(user_id)
                if template_id not in data:
                    return error_response(f"No override found for template '{template_id}'", 404)
                del data[template_id]
                if data:
                    store._write_user_file(user_id, data)
                else:
                    path = store._user_path(user_id)
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass

            return json_response(
                {
                    "user_id": user_id,
                    "template_id": template_id,
                    "deleted": True,
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Failed to delete template override: %s", exc)
            return error_response("Failed to delete template override", 500)
