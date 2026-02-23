# mypy: disable-error-code="no-redef"
"""
HTTP handlers for Shared Inbox - main handler class.

The SharedInboxHandler class routes requests to the appropriate handler functions.
"""
# mypy: disable-error-code="assignment,attr-defined,index"

from __future__ import annotations

import logging
import sys
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
    handle_errors,
)
from aragora.rbac.decorators import require_permission
from aragora.server.validation.query_params import safe_query_int
from aragora.server.validation.security import sanitize_user_input

from .models import MessageStatus
from .validators import (
    MAX_INBOX_NAME_LENGTH,
    MAX_INBOX_DESCRIPTION_LENGTH,
    validate_inbox_input,
    validate_tag,
)
from .storage import (
    _get_activity_store as _get_activity_store_impl,
    _get_rules_store as _get_rules_store_impl,
    _get_store as _get_store_impl,
    _log_activity as _log_activity_impl,
)
from .inbox_handlers import (
    handle_create_shared_inbox,
    handle_list_shared_inboxes,
    handle_get_shared_inbox,
    handle_get_inbox_messages,
    handle_assign_message,
    handle_update_message_status,
    handle_add_message_tag,
)
from .rule_handlers import (
    handle_create_routing_rule,
    handle_list_routing_rules,
    handle_update_routing_rule,
    handle_delete_routing_rule,
    handle_test_routing_rule,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Store helpers (shims for test overrides)
# =============================================================================


def _get_store() -> Any:  # type: ignore[override]
    module = sys.modules.get("aragora.server.handlers._shared_inbox_handler")
    if module is not None:
        patched = getattr(module, "_get_store", None)
        if patched is not None and patched is not _get_store:
            return patched()
    return _get_store_impl()


def _get_rules_store() -> Any:  # type: ignore[override]
    module = sys.modules.get("aragora.server.handlers._shared_inbox_handler")
    if module is not None:
        patched = getattr(module, "_get_rules_store", None)
        if patched is not None and patched is not _get_rules_store:
            return patched()
    return _get_rules_store_impl()


def _get_activity_store() -> Any:  # type: ignore[override]
    module = sys.modules.get("aragora.server.handlers._shared_inbox_handler")
    if module is not None:
        patched = getattr(module, "_get_activity_store", None)
        if patched is not None and patched is not _get_activity_store:
            return patched()
    return _get_activity_store_impl()


def _log_activity(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
    module = sys.modules.get("aragora.server.handlers._shared_inbox_handler")
    if module is not None:
        patched = getattr(module, "_log_activity", None)
        if patched is not None and hasattr(patched, "assert_called"):
            return patched(*args, **kwargs)
        patched_store = getattr(module, "_get_activity_store", None)
        if patched_store is not None and patched_store is not _get_activity_store:
            store = patched_store()
            if store:
                try:
                    from aragora.storage.inbox_activity_store import InboxActivity

                    activity = InboxActivity(*args, **kwargs)
                    store.log_activity(activity)
                except (
                    ImportError,
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    OSError,
                    RuntimeError,
                ) as e:
                    logging.getLogger(__name__).warning("Failed to log inbox activity: %s", e)
            return None
    return _log_activity_impl(*args, **kwargs)


logger = logging.getLogger(__name__)


# =============================================================================
# Shared Inbox Handlers
# =============================================================================


class SharedInboxHandler(BaseHandler):
    """
    HTTP handler for shared inbox endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/inbox/shared",
        "/api/v1/inbox/routing/rules",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/inbox/shared/",
        "/api/v1/inbox/routing/rules/",
    ]

    def __init__(self, ctx: dict[str, Any]) -> None:
        """Initialize with server context."""
        super().__init__(ctx)

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route shared inbox endpoint requests."""
        return None

    @handle_errors("shared inbox operation")
    @require_permission("inbox:create")
    async def handle_post_shared_inbox(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/v1/inbox/shared"""
        workspace_id = data.get("workspace_id")
        name = data.get("name")

        if not workspace_id or not name:
            return error_response("workspace_id and name required", 400)

        # Validate inbox inputs
        is_valid, error = validate_inbox_input(
            name=name,
            description=data.get("description"),
            email_address=data.get("email_address"),
        )
        if not is_valid:
            return error_response(error, 400)

        # Sanitize name and description
        sanitized_name = sanitize_user_input(name, max_length=MAX_INBOX_NAME_LENGTH)
        sanitized_description = None
        if data.get("description"):
            sanitized_description = sanitize_user_input(
                data["description"], max_length=MAX_INBOX_DESCRIPTION_LENGTH
            )

        result = await handle_create_shared_inbox(
            workspace_id=workspace_id,
            name=sanitized_name,
            description=sanitized_description,
            email_address=data.get("email_address"),
            connector_type=data.get("connector_type"),
            team_members=data.get("team_members"),
            admins=data.get("admins"),
            settings=data.get("settings"),
            created_by=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:read")
    async def handle_get_shared_inboxes(self, params: dict[str, Any]) -> HandlerResult:
        """GET /api/v1/inbox/shared"""
        workspace_id = params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id required", 400)

        result = await handle_list_shared_inboxes(
            workspace_id=workspace_id,
            user_id=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:read")
    async def handle_get_shared_inbox(self, params: dict[str, Any], inbox_id: str) -> HandlerResult:
        """GET /api/v1/inbox/shared/:id"""
        result = await handle_get_shared_inbox(inbox_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    @require_permission("inbox:read")
    async def handle_get_inbox_messages(
        self, params: dict[str, Any], inbox_id: str
    ) -> HandlerResult:
        """GET /api/v1/inbox/shared/:id/messages"""
        result = await handle_get_inbox_messages(
            inbox_id=inbox_id,
            status=params.get("status"),
            assigned_to=params.get("assigned_to"),
            tag=params.get("tag"),
            limit=safe_query_int(params, "limit", default=50, min_val=1, max_val=1000),
            offset=safe_query_int(params, "offset", default=0, min_val=0, max_val=100000),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @handle_errors("shared inbox operation")
    @require_permission("inbox:manage")
    async def handle_post_assign_message(
        self, data: dict[str, Any], inbox_id: str, message_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/shared/:id/messages/:msg_id/assign"""
        assigned_to = data.get("assigned_to")
        if not assigned_to:
            return error_response("assigned_to required", 400)

        # Validate and sanitize assigned_to (user ID)
        if not isinstance(assigned_to, str):
            return error_response("assigned_to must be a string", 400)

        if len(assigned_to) > 200:
            return error_response("assigned_to exceeds maximum length of 200", 400)

        # Sanitize user ID
        sanitized_assigned_to = sanitize_user_input(assigned_to, max_length=200)
        if not sanitized_assigned_to:
            return error_response("assigned_to cannot be empty", 400)

        result = await handle_assign_message(
            inbox_id=inbox_id,
            message_id=message_id,
            assigned_to=sanitized_assigned_to,
            assigned_by=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @handle_errors("shared inbox operation")
    @require_permission("inbox:manage")
    async def handle_post_update_status(
        self, data: dict[str, Any], inbox_id: str, message_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/shared/:id/messages/:msg_id/status"""
        status = data.get("status")
        if not status:
            return error_response("status required", 400)

        # Validate status is a valid MessageStatus
        try:
            MessageStatus(status)
        except ValueError:
            valid_statuses = ", ".join(s.value for s in MessageStatus)
            return error_response(
                f"Invalid status '{status}'. Valid statuses: {valid_statuses}", 400
            )

        result = await handle_update_message_status(
            inbox_id=inbox_id,
            message_id=message_id,
            status=status,
            updated_by=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @handle_errors("shared inbox operation")
    @require_permission("inbox:manage")
    async def handle_post_add_tag(
        self, data: dict[str, Any], inbox_id: str, message_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/shared/:id/messages/:msg_id/tag"""
        tag = data.get("tag")
        if not tag:
            return error_response("tag required", 400)

        # Validate tag format and length
        is_valid, error = validate_tag(tag)
        if not is_valid:
            return error_response(error, 400)

        result = await handle_add_message_tag(
            inbox_id=inbox_id,
            message_id=message_id,
            tag=tag,
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @handle_errors("shared inbox operation")
    @require_permission("inbox:admin")
    async def handle_post_routing_rule(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/v1/inbox/routing/rules"""
        workspace_id = data.get("workspace_id")
        name = data.get("name")
        conditions = data.get("conditions", [])
        actions = data.get("actions", [])

        if not workspace_id or not name or not conditions or not actions:
            return error_response("workspace_id, name, conditions, and actions required", 400)

        result = await handle_create_routing_rule(
            workspace_id=workspace_id,
            name=name,
            conditions=conditions,
            actions=actions,
            condition_logic=data.get("condition_logic", "AND"),
            priority=data.get("priority", 5),
            enabled=data.get("enabled", True),
            description=data.get("description"),
            created_by=self._get_user_id(),
            inbox_id=data.get("inbox_id"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @require_permission("inbox:read")
    async def handle_get_routing_rules(self, params: dict[str, Any]) -> HandlerResult:
        """GET /api/v1/inbox/routing/rules"""
        workspace_id = params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id required", 400)

        result = await handle_list_routing_rules(
            workspace_id=workspace_id,
            enabled_only=params.get("enabled_only", "false").lower() == "true",
            limit=safe_query_int(params, "limit", default=100, min_val=1, max_val=1000),
            offset=safe_query_int(params, "offset", default=0, min_val=0, max_val=100000),
            inbox_id=params.get("inbox_id"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @handle_errors("shared inbox operation")
    @require_permission("inbox:admin")
    async def handle_patch_routing_rule(self, data: dict[str, Any], rule_id: str) -> HandlerResult:
        """PATCH /api/v1/inbox/routing/rules/:id"""
        result = await handle_update_routing_rule(rule_id, data)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @handle_errors("shared inbox operation")
    @require_permission("inbox:admin")
    async def handle_delete_routing_rule(self, rule_id: str) -> HandlerResult:
        """DELETE /api/v1/inbox/routing/rules/:id"""
        result = await handle_delete_routing_rule(rule_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    @handle_errors("shared inbox operation")
    @require_permission("inbox:admin")
    async def handle_post_test_routing_rule(
        self, data: dict[str, Any], rule_id: str
    ) -> HandlerResult:
        """POST /api/v1/inbox/routing/rules/:id/test"""
        workspace_id = data.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id required", 400)

        result = await handle_test_routing_rule(rule_id, workspace_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"
