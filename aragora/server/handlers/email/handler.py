"""
Email HTTP handler class for server routing integration.

This module provides the EmailHandler class that routes HTTP requests
to the appropriate email handler functions.
"""

from __future__ import annotations

from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
    handle_errors,
)
from aragora.server.handlers.openapi_decorator import api_endpoint
from aragora.server.handlers.utils.rate_limit import rate_limit

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

    def __init__(self, ctx: dict[str, Any]):
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/prioritize",
        summary="Score an email for priority",
        description="Score a single email with priority signals and return a rationale.",
        tags=["Email", "Prioritization"],
        responses={
            "200": {"description": "Priority result returned"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=60)  # READ operation
    @require_permission("email:read")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/rank-inbox",
        summary="Rank inbox emails by priority",
        description="Rank multiple emails by priority and return ordered results.",
        tags=["Email", "Prioritization"],
        responses={
            "200": {"description": "Ranked inbox results returned"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=60)  # READ operation
    @require_permission("email:read")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/feedback",
        summary="Record prioritization feedback",
        description="Record user feedback for a single email action.",
        tags=["Email", "Feedback"],
        responses={
            "200": {"description": "Feedback recorded"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=20)  # WRITE operation
    @require_permission("email:update")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/feedback/batch",
        summary="Record batch feedback",
        description="Record feedback actions for a batch of emails.",
        tags=["Email", "Feedback"],
        responses={
            "200": {"description": "Batch feedback recorded"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=20)  # WRITE operation
    @require_permission("email:update")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/categorize",
        summary="Categorize a single email",
        description="Categorize a single email and return the assigned category.",
        tags=["Email", "Categorization"],
        responses={
            "200": {"description": "Categorization result returned"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=60)  # READ operation
    @require_permission("email:read")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/categorize/batch",
        summary="Categorize emails in batch",
        description="Categorize multiple emails and return results and stats.",
        tags=["Email", "Categorization"],
        responses={
            "200": {"description": "Batch categorization results returned"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=60)  # READ operation
    @require_permission("email:read")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/categorize/apply-label",
        summary="Apply category label",
        description="Apply the suggested category label to an email.",
        tags=["Email", "Categorization"],
        responses={
            "200": {"description": "Label applied"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=20)  # WRITE operation
    @require_permission("email:update")
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

    @api_endpoint(
        method="GET",
        path="/api/v1/email/inbox",
        summary="Fetch and rank inbox",
        description="Fetch recent inbox emails and return ranked results.",
        tags=["Email", "Prioritization"],
        parameters=[
            {"name": "labels", "in": "query", "schema": {"type": "string"}},
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
            {"name": "include_read", "in": "query", "schema": {"type": "boolean"}},
        ],
        responses={
            "200": {"description": "Inbox results returned"},
            "401": {"description": "Unauthorized"},
            "400": {"description": "Invalid request"},
        },
    )
    @rate_limit(requests_per_minute=10)  # SYNC operation
    @require_permission("email:read")
    async def handle_get_inbox(self, params: dict[str, Any]) -> HandlerResult:
        """GET /api/email/inbox"""
        user_id = self._get_user_id()
        labels = params.get("labels", "").split(",") if params.get("labels") else None
        try:
            limit = int(params.get("limit", 50))
        except (ValueError, TypeError):
            limit = 50
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

    @api_endpoint(
        method="GET",
        path="/api/v1/email/config",
        summary="Get email config",
        description="Get the email prioritization configuration.",
        tags=["Email", "Config"],
        responses={
            "200": {"description": "Config returned"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=60)  # READ operation
    @require_permission("email:read")
    async def handle_get_config(self, params: dict[str, Any]) -> HandlerResult:
        """GET /api/email/config"""
        user_id = self._get_user_id()
        result = await handle_get_config(
            user_id,
            auth_context=self._get_auth_context(),
        )
        return success_response(result)

    @handle_errors("email operation")
    @api_endpoint(
        method="PUT",
        path="/api/v1/email/config",
        summary="Update email config",
        description="Update the email prioritization configuration.",
        tags=["Email", "Config"],
        responses={
            "200": {"description": "Config updated"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=20)  # WRITE operation
    @require_permission("email:update")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/vip",
        summary="Add VIP email/domain",
        description="Add a VIP email address or domain for priority boosting.",
        tags=["Email", "VIP"],
        responses={
            "200": {"description": "VIP added"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=20)  # WRITE operation
    @require_permission("email:update")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="DELETE",
        path="/api/v1/email/vip",
        summary="Remove VIP email/domain",
        description="Remove a VIP email address or domain.",
        tags=["Email", "VIP"],
        responses={
            "200": {"description": "VIP removed"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=20)  # WRITE operation
    @require_permission("email:delete")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/gmail/oauth/url",
        summary="Get Gmail OAuth URL",
        description="Generate Gmail OAuth URL for user authorization.",
        tags=["Email", "Gmail", "OAuth"],
        responses={
            "200": {"description": "OAuth URL returned"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=20)  # WRITE operation
    @require_permission("email:create")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/gmail/oauth/callback",
        summary="Handle Gmail OAuth callback",
        description="Exchange OAuth code for Gmail tokens.",
        tags=["Email", "Gmail", "OAuth"],
        responses={
            "200": {"description": "Gmail authenticated"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=20)  # WRITE operation
    @require_permission("email:create")
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

    @api_endpoint(
        method="GET",
        path="/api/v1/email/gmail/status",
        summary="Get Gmail connection status",
        description="Check Gmail OAuth connection status.",
        tags=["Email", "Gmail", "OAuth"],
        responses={
            "200": {"description": "Status returned"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=60)  # READ operation
    @require_permission("email:read")
    async def handle_get_gmail_status(self, params: dict[str, Any]) -> HandlerResult:
        """GET /api/email/gmail/status"""
        user_id = self._get_user_id()
        result = await handle_gmail_status(
            user_id,
            auth_context=self._get_auth_context(),
        )
        return success_response(result)

    @api_endpoint(
        method="GET",
        path="/api/v1/email/context/{email_address}",
        summary="Get email context",
        description="Get cross-channel context for a specific email address.",
        tags=["Email", "Context"],
        parameters=[
            {
                "name": "email_address",
                "in": "path",
                "required": True,
                "schema": {"type": "string"},
            }
        ],
        responses={
            "200": {"description": "Context returned"},
            "400": {"description": "Invalid request"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=60)  # READ operation
    @require_permission("email:read")
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

    @handle_errors("email operation")
    @api_endpoint(
        method="POST",
        path="/api/v1/email/context/boost",
        summary="Get email context boost",
        description="Get context-based priority boost signals for an email.",
        tags=["Email", "Context"],
        responses={
            "200": {"description": "Boost info returned"},
            "400": {"description": "Invalid request body"},
            "401": {"description": "Unauthorized"},
        },
    )
    @rate_limit(requests_per_minute=60)  # READ operation
    @require_permission("email:read")
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
