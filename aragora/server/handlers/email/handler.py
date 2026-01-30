"""
Email HTTP handler class for server routing integration.

This module provides the EmailHandler class that routes HTTP requests
to the appropriate email handler functions.
"""

from __future__ import annotations

from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    ServerContext,
    error_response,
    success_response,
)

# Import handler functions from submodules
from .prioritization import (
    handle_prioritize_email,
    handle_rank_inbox,
    handle_email_feedback,
)
from .categorization import (
    handle_categorize_email,
    handle_categorize_batch,
    handle_feedback_batch,
    handle_apply_category_label,
)
from .oauth import (
    handle_gmail_oauth_url,
    handle_gmail_oauth_callback,
    handle_gmail_status,
)
from .context import (
    handle_get_context,
    handle_get_email_context_boost,
)
from .inbox import handle_fetch_and_rank_inbox
from .config import handle_get_config, handle_update_config
from .vip import handle_add_vip, handle_remove_vip


class EmailHandler(BaseHandler):
    """
    HTTP handler for email prioritization endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/email/prioritize",
        "/api/v1/email/rank-inbox",
        "/api/v1/email/feedback",
        "/api/v1/email/feedback/batch",
        "/api/v1/email/inbox",
        "/api/v1/email/config",
        "/api/v1/email/vip",
        "/api/v1/email/categorize",
        "/api/v1/email/categorize/batch",
        "/api/v1/email/categorize/apply-label",
        "/api/v1/email/gmail/oauth/url",
        "/api/v1/email/gmail/oauth/callback",
        "/api/v1/email/gmail/status",
        "/api/v1/email/context/boost",
    ]

    # Prefix for dynamic routes like /api/email/context/:email_address
    ROUTE_PREFIXES = ["/api/v1/email/context/"]

    def __init__(self, ctx: ServerContext):
        """Initialize with server context."""
        super().__init__(ctx)

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        # Check prefix routes (e.g., /api/email/context/:email_address)
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix) and path != prefix.rstrip("/"):
                return True
        return False

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route email endpoint requests."""
        # This handler uses async methods, so we return None here
        # and let the server's async handling mechanism process it
        # The actual handling is done via HTTP method-specific handlers
        return None

    async def handle_post_prioritize(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/prioritize"""
        email_data = data.get("email", {})
        force_tier = data.get("force_tier")
        user_id = self._get_user_id()

        result = await handle_prioritize_email(
            email_data,
            user_id,
            force_tier=force_tier,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_rank_inbox(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/rank-inbox"""
        emails = data.get("emails", [])
        limit = data.get("limit")
        user_id = self._get_user_id()

        result = await handle_rank_inbox(
            emails,
            user_id,
            limit=limit,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_feedback(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/feedback"""
        email_id = data.get("email_id")
        action = data.get("action")
        email_data = data.get("email")
        user_id = self._get_user_id()

        if not email_id or not action:
            return error_response("email_id and action required", 400)

        result = await handle_email_feedback(
            email_id,
            action,
            user_id,
            email_data=email_data,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_feedback_batch(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/feedback/batch"""
        items = data.get("items", [])
        user_id = self._get_user_id()

        if not items:
            return error_response("items array required", 400)

        result = await handle_feedback_batch(
            items,
            user_id,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_categorize(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/categorize"""
        email_data = data.get("email", {})
        user_id = self._get_user_id()

        result = await handle_categorize_email(
            email_data,
            user_id,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_categorize_batch(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/categorize/batch"""
        emails = data.get("emails", [])
        concurrency = data.get("concurrency", 10)
        user_id = self._get_user_id()

        if not emails:
            return error_response("emails array required", 400)

        result = await handle_categorize_batch(
            emails,
            user_id,
            concurrency,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_categorize_apply_label(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/categorize/apply-label"""
        email_id = data.get("email_id")
        category = data.get("category")
        user_id = self._get_user_id()

        if not email_id or not category:
            return error_response("email_id and category required", 400)

        result = await handle_apply_category_label(
            email_id,
            category,
            user_id,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_inbox(self, params: dict[str, Any]) -> HandlerResult:
        """GET /api/email/inbox"""
        user_id = self._get_user_id()
        labels = params.get("labels", "").split(",") if params.get("labels") else None
        limit = int(params.get("limit", 50))
        include_read = params.get("include_read", "").lower() == "true"

        result = await handle_fetch_and_rank_inbox(
            user_id=user_id,
            labels=labels,
            limit=limit,
            include_read=include_read,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        elif result.get("needs_auth"):
            return error_response(result.get("error"), 401)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_config(self, params: dict[str, Any]) -> HandlerResult:
        """GET /api/email/config"""
        user_id = self._get_user_id()
        result = await handle_get_config(
            user_id,
            auth_context=self._get_auth_context(),
        )
        return success_response(result)

    async def handle_put_config(self, data: dict[str, Any]) -> HandlerResult:
        """PUT /api/email/config"""
        user_id = self._get_user_id()
        result = await handle_update_config(
            user_id,
            data,
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_vip(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/vip"""
        user_id = self._get_user_id()
        email = data.get("email")
        domain = data.get("domain")

        result = await handle_add_vip(
            user_id,
            email,
            domain,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_delete_vip(self, data: dict[str, Any]) -> HandlerResult:
        """DELETE /api/email/vip"""
        user_id = self._get_user_id()
        email = data.get("email")
        domain = data.get("domain")

        result = await handle_remove_vip(
            user_id,
            email,
            domain,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_gmail_oauth_url(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/gmail/oauth/url"""
        redirect_uri = data.get("redirect_uri")
        state = data.get("state", "")
        scopes = data.get("scopes", "readonly")

        if not redirect_uri:
            return error_response("redirect_uri required", 400)

        result = await handle_gmail_oauth_url(
            redirect_uri,
            state,
            scopes,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_gmail_oauth_callback(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/gmail/oauth/callback"""
        code = data.get("code")
        redirect_uri = data.get("redirect_uri")
        user_id = self._get_user_id()

        if not code or not redirect_uri:
            return error_response("code and redirect_uri required", 400)

        result = await handle_gmail_oauth_callback(
            code,
            redirect_uri,
            user_id,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_gmail_status(self, params: dict[str, Any]) -> HandlerResult:
        """GET /api/email/gmail/status"""
        user_id = self._get_user_id()
        result = await handle_gmail_status(
            user_id,
            auth_context=self._get_auth_context(),
        )
        return success_response(result)

    async def handle_get_context(self, params: dict[str, Any], email_address: str) -> HandlerResult:
        """GET /api/email/context/:email_address"""
        user_id = self._get_user_id()
        result = await handle_get_context(
            email_address,
            user_id,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_context_boost(self, data: dict[str, Any]) -> HandlerResult:
        """POST /api/email/context/boost"""
        email_data = data.get("email", {})
        user_id = self._get_user_id()

        result = await handle_get_email_context_boost(
            email_data,
            user_id,
            auth_context=self._get_auth_context(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        # In production, extract from JWT token
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"

    def _get_auth_context(self) -> Any | None:
        """Get auth context from handler ctx."""
        return self.ctx.get("auth_context")
