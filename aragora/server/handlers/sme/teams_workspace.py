"""
Teams Workspace API Handlers.

Provides management APIs for Microsoft Teams workspace integrations:
- GET /api/v1/sme/teams/tenants - List connected tenants
- POST /api/v1/sme/teams/tenants - Create tenant configuration
- GET /api/v1/sme/teams/tenants/:tenant_id - Get tenant details
- PATCH /api/v1/sme/teams/tenants/:tenant_id - Update tenant details
- POST /api/v1/sme/teams/tenants/:tenant_id/test - Test connection
- DELETE /api/v1/sme/teams/tenants/:tenant_id - Disconnect tenant
- GET /api/v1/sme/teams/tenants/:tenant_id/channels - List available channels
- GET /api/v1/sme/teams/channels/:tenant_id - List available channels (legacy)
- GET /api/v1/sme/teams/oauth/start - Start OAuth flow
- GET /api/v1/sme/teams/oauth/callback - OAuth callback
- POST /api/v1/sme/teams/subscribe - Subscribe channel to notifications
- GET /api/v1/sme/teams/subscriptions - List subscriptions
- DELETE /api/v1/sme/teams/subscriptions/:id - Remove subscription
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from ..base import (
    error_response,
    get_string_param,
    handle_errors,
    json_response,
)
from ..utils.responses import HandlerResult
from ..secure import SecureHandler
from aragora.rbac.decorators import require_permission
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for Teams workspace APIs (30 requests per minute)
_teams_limiter = RateLimiter(requests_per_minute=30)


class TeamsWorkspaceHandler(SecureHandler):
    """Handler for Teams workspace management endpoints.

    Provides APIs for managing Microsoft Teams workspace connections,
    listing channels, and subscribing to notifications.
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    RESOURCE_TYPE = "teams_workspace"

    ROUTES = [
        "/api/v1/sme/teams/workspaces",
        "/api/v1/sme/teams/tenants",
        "/api/v1/sme/teams/channels",
        "/api/v1/sme/teams/subscribe",
        "/api/v1/sme/teams/subscriptions",
        "/api/v1/sme/teams/oauth/start",
        "/api/v1/sme/teams/oauth/callback",
    ]

    # Regex patterns for parameterized routes
    ROUTE_PATTERNS = [
        (re.compile(r"^/api/v1/sme/teams/workspaces/([^/]+)/test$"), "workspace_test"),
        (re.compile(r"^/api/v1/sme/teams/tenants/([^/]+)/test$"), "workspace_test"),
        (re.compile(r"^/api/v1/sme/teams/tenants/([^/]+)/channels$"), "workspace_channels"),
        (re.compile(r"^/api/v1/sme/teams/tenants/([^/]+)$"), "workspace_detail"),
        (re.compile(r"^/api/v1/sme/teams/workspaces/([^/]+)$"), "workspace_detail"),
        (re.compile(r"^/api/v1/sme/teams/channels/([^/]+)$"), "channels"),
        (re.compile(r"^/api/v1/sme/teams/subscriptions/([^/]+)$"), "subscription_detail"),
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        for pattern, _ in self.ROUTE_PATTERNS:
            if pattern.match(path):
                return True
        return False

    def _match_route(self, path: str) -> tuple[str | None, str | None]:
        """Match a path against parameterized routes.

        Returns:
            Tuple of (route_name, extracted_id) or (None, None) if no match.
        """
        for pattern, route_name in self.ROUTE_PATTERNS:
            match = pattern.match(path)
            if match:
                return route_name, match.group(1)
        return None, None

    def handle(
        self,
        path: str,
        query_params: dict,
        handler: Any,
        method: str = "GET",
    ) -> HandlerResult | None:
        """Route Teams workspace requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _teams_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for Teams workspace: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        # Handle static routes
        if path in ("/api/v1/sme/teams/workspaces", "/api/v1/sme/teams/tenants"):
            if method == "GET":
                return self._list_workspaces(handler, query_params)
            if method == "POST":
                return self._create_tenant(handler, query_params)
            return error_response("Method not allowed", 405)

        if path == "/api/v1/sme/teams/oauth/start":
            if method == "GET":
                return self._handle_oauth_start(handler, query_params)
            return error_response("Method not allowed", 405)

        if path == "/api/v1/sme/teams/oauth/callback":
            if method == "GET":
                code = query_params.get("code")
                state = query_params.get("state")
                if not code:
                    return error_response("Missing OAuth code", 400)
                return self._handle_oauth_callback(code, state, handler)
            return error_response("Method not allowed", 405)

        if path == "/api/v1/sme/teams/subscribe":
            if method == "POST":
                return self._subscribe_channel(handler, query_params)
            return error_response("Method not allowed", 405)

        if path == "/api/v1/sme/teams/subscriptions":
            if method == "GET":
                return self._list_subscriptions(handler, query_params)
            return error_response("Method not allowed", 405)

        # Handle parameterized routes
        route_name, param_id = self._match_route(path)
        if route_name:
            if route_name == "workspace_detail":
                if method == "GET":
                    return self._get_workspace(handler, query_params, param_id)
                elif method == "PATCH":
                    return self._update_tenant(handler, query_params, param_id)
                elif method == "DELETE":
                    return self._disconnect_workspace(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

            if route_name == "workspace_test":
                if method == "POST":
                    return self._test_connection(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

            if route_name in ("workspace_channels", "channels"):
                if method == "GET":
                    return self._list_channels(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

            if route_name == "subscription_detail":
                if method == "DELETE":
                    return self._delete_subscription(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

        return error_response("Not found", 404)

    def _get_workspace_store(self) -> Any:
        """Get Teams workspace store instance."""
        from aragora.storage.teams_workspace_store import get_teams_workspace_store

        return get_teams_workspace_store()

    def _get_subscription_store(self) -> Any:
        """Get channel subscription store instance."""
        from aragora.storage.channel_subscription_store import (
            get_channel_subscription_store,
        )

        return get_channel_subscription_store()

    def _get_user_and_org(self, handler: Any, user: Any) -> tuple[Any, Any, HandlerResult | None]:
        """Get user and organization from context."""
        user_store = self.ctx.get("user_store")
        if not user_store:
            return None, None, error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return None, None, error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        if not org:
            return None, None, error_response("No organization found", 404)

        return db_user, org, None

    def _parse_json_body(self, handler: Any) -> tuple[dict[str, Any] | None, HandlerResult | None]:
        """Parse JSON body from handler."""
        import json as json_lib

        try:
            body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
            data = json_lib.loads(body.decode("utf-8")) if body else {}
            return data, None
        except (json_lib.JSONDecodeError, ValueError):
            return None, error_response("Invalid JSON body", 400)

    @handle_errors("list Teams workspaces")
    @require_permission("sme:workspaces:read")
    def _list_workspaces(
        self,
        handler: Any,
        query_params: dict,
        user: Any = None,
    ) -> HandlerResult:
        """
        List connected Teams workspaces for the organization.

        Query Parameters:
            limit: Maximum results (default: 50)
            offset: Pagination offset (default: 0)

        Returns:
            JSON response with workspace list:
            {
                "workspaces": [...],
                "total": 3,
                "limit": 50,
                "offset": 0
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        limit = int(get_string_param(handler, "limit", "50"))
        offset = int(get_string_param(handler, "offset", "0"))

        store = self._get_workspace_store()
        if hasattr(store, "get_by_org"):
            workspaces = store.get_by_org(org.id)
        else:
            workspaces = store.get_by_aragora_tenant(org.id)

        # Apply pagination
        paginated = workspaces[offset : offset + limit]

        return json_response(
            {
                "workspaces": [w.to_dict() for w in paginated],
                "total": len(workspaces),
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("create Teams tenant")
    @require_permission("sme:workspaces:write")
    def _create_tenant(
        self,
        handler: Any,
        query_params: dict,
        user: Any = None,
    ) -> HandlerResult:
        """Create a Teams tenant entry."""
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        data, err = self._parse_json_body(handler)
        if err:
            return err
        data = data or {}

        tenant_id = data.get("tenant_id")
        client_id = data.get("client_id") or data.get("bot_id")
        client_secret = data.get("client_secret") or data.get("access_token")

        if not tenant_id:
            return error_response("tenant_id is required", 400)
        if not client_id or not client_secret:
            return error_response("client_id and client_secret are required", 400)

        tenant_name = data.get("tenant_name") or tenant_id

        from aragora.storage.teams_workspace_store import TeamsWorkspace

        workspace = TeamsWorkspace(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            access_token=client_secret,
            bot_id=client_id,
            installed_at=datetime.now(timezone.utc).timestamp(),
            installed_by=db_user.id,
            aragora_tenant_id=org.id,
            is_active=True,
        )

        store = self._get_workspace_store()
        if not store.save(workspace):
            return error_response("Failed to save tenant", 500)

        return json_response({"tenant": workspace.to_dict()}, status=201)

    @handle_errors("update Teams tenant")
    @require_permission("sme:workspaces:write")
    def _update_tenant(
        self,
        handler: Any,
        query_params: dict,
        tenant_id: str,
        user: Any = None,
    ) -> HandlerResult:
        """Update Teams tenant details."""
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        data, err = self._parse_json_body(handler)
        if err:
            return err
        data = data or {}

        store = self._get_workspace_store()
        workspace = store.get(tenant_id)
        if not workspace:
            return error_response("Tenant not found", 404)
        if workspace.aragora_tenant_id != org.id:
            return error_response("Tenant not found", 404)

        if "tenant_name" in data:
            workspace.tenant_name = data["tenant_name"]
        if "client_id" in data or "bot_id" in data:
            workspace.bot_id = data.get("client_id") or data.get("bot_id")
        if "client_secret" in data or "access_token" in data:
            workspace.access_token = data.get("client_secret") or data.get("access_token")
        if "is_active" in data:
            workspace.is_active = bool(data["is_active"])

        if not store.save(workspace):
            return error_response("Failed to update tenant", 500)

        return json_response({"tenant": workspace.to_dict()})

    @handle_errors("get Teams workspace")
    @require_permission("sme:workspaces:read")
    def _get_workspace(
        self,
        handler: Any,
        query_params: dict,
        tenant_id: str,
        user: Any = None,
    ) -> HandlerResult:
        """
        Get details for a specific Teams workspace.

        Path Parameters:
            tenant_id: Azure AD tenant ID

        Returns:
            JSON response with workspace details
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        store = self._get_workspace_store()
        workspace = store.get(tenant_id)

        if not workspace:
            return error_response("Workspace not found", 404)

        # Verify workspace belongs to this org
        if workspace.aragora_tenant_id != org.id:
            return error_response("Workspace not found", 404)

        return json_response({"workspace": workspace.to_dict()})

    @handle_errors("test Teams connection")
    @require_permission("sme:workspaces:write")
    def _test_connection(
        self,
        handler: Any,
        query_params: dict,
        tenant_id: str,
        user: Any = None,
    ) -> HandlerResult:
        """
        Test connection to a Teams workspace.

        Path Parameters:
            tenant_id: Azure AD tenant ID

        Returns:
            JSON response with connection status:
            {
                "status": "connected",
                "tenant_id": "...",
                "tenant_name": "...",
                "bot_id": "...",
                "token_valid": true,
                "tested_at": "..."
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        store = self._get_workspace_store()
        workspace = store.get(tenant_id)

        if not workspace:
            return error_response("Workspace not found", 404)

        if workspace.aragora_tenant_id != org.id:
            return error_response("Workspace not found", 404)

        # Check token expiration
        token_valid = True
        if workspace.token_expires_at:
            token_valid = workspace.token_expires_at > datetime.now(timezone.utc).timestamp()

        # Try to make a test API call
        connection_status = "connected"
        error_message = None

        try:
            # Simple validation - check if token format looks valid
            if not workspace.access_token or len(workspace.access_token) < 10:
                connection_status = "invalid_token"
                token_valid = False
        except (TypeError, AttributeError) as e:
            logger.warning(f"Teams connection test failed for {tenant_id}: {e}")
            connection_status = "error"
            error_message = "Connection test failed"

        result = {
            "status": connection_status,
            "tenant_id": workspace.tenant_id,
            "tenant_name": workspace.tenant_name,
            "bot_id": workspace.bot_id,
            "token_valid": token_valid,
            "tested_at": datetime.now(timezone.utc).isoformat(),
        }

        if error_message:
            result["error"] = error_message

        return json_response(result)

    @handle_errors("disconnect Teams workspace")
    @require_permission("sme:workspaces:write")
    def _disconnect_workspace(
        self,
        handler: Any,
        query_params: dict,
        tenant_id: str,
        user: Any = None,
    ) -> HandlerResult:
        """
        Disconnect (deactivate) a Teams workspace.

        Path Parameters:
            tenant_id: Azure AD tenant ID

        Returns:
            JSON response confirming disconnection
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        store = self._get_workspace_store()
        workspace = store.get(tenant_id)

        if not workspace:
            return error_response("Workspace not found", 404)

        if workspace.aragora_tenant_id != org.id:
            return error_response("Workspace not found", 404)

        # Deactivate (soft delete)
        success = store.deactivate(tenant_id)

        if not success:
            return error_response("Failed to disconnect workspace", 500)

        # Also deactivate any subscriptions for this workspace
        sub_store = self._get_subscription_store()
        from aragora.storage.channel_subscription_store import ChannelType

        subscriptions = sub_store.get_by_org(org.id, channel_type=ChannelType.TEAMS)
        for sub in subscriptions:
            if sub.workspace_id == tenant_id:
                sub_store.deactivate(sub.id)

        logger.info(f"Disconnected Teams workspace {tenant_id} for org {org.id}")

        return json_response(
            {
                "disconnected": True,
                "tenant_id": tenant_id,
                "message": "Workspace disconnected successfully",
            }
        )

    @handle_errors("list Teams channels")
    @require_permission("sme:workspaces:read")
    def _list_channels(
        self,
        handler: Any,
        query_params: dict,
        tenant_id: str,
        user: Any = None,
    ) -> HandlerResult:
        """
        List available channels for a Teams workspace.

        Path Parameters:
            tenant_id: Azure AD tenant ID

        Query Parameters:
            team_id: Optional team ID to filter channels

        Returns:
            JSON response with channel list:
            {
                "channels": [
                    {
                        "id": "...",
                        "team_id": "...",
                        "display_name": "General",
                        "description": "...",
                        "web_url": "..."
                    }
                ],
                "tenant_id": "..."
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        store = self._get_workspace_store()
        workspace = store.get(tenant_id)

        if not workspace:
            return error_response("Workspace not found", 404)

        if workspace.aragora_tenant_id != org.id:
            return error_response("Workspace not found", 404)

        team_id = get_string_param(handler, "team_id", None)

        # Try to fetch channels from Teams API
        channels: list[dict[str, Any]] = []

        try:
            from aragora.connectors.enterprise.collaboration.teams import (
                TeamsEnterpriseConnector,
            )

            _connector = TeamsEnterpriseConnector(  # noqa: F841
                tenant_id=workspace.tenant_id,
            )
            # Note: In production, this would make actual API calls
            # For now, return placeholder indicating API integration needed
            channels = [
                {
                    "id": "placeholder",
                    "team_id": team_id or "default",
                    "display_name": "Channel listing requires Graph API integration",
                    "description": "Configure Microsoft Graph API credentials to list channels",
                    "web_url": "",
                }
            ]
        except ImportError:
            logger.warning("Teams connector not available")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to list Teams channels: {e}")

        return json_response(
            {
                "channels": channels,
                "tenant_id": tenant_id,
            }
        )

    @handle_errors("subscribe Teams channel")
    @require_permission("sme:channels:subscribe")
    def _subscribe_channel(
        self,
        handler: Any,
        query_params: dict,
        user: Any = None,
    ) -> HandlerResult:
        """
        Subscribe a Teams channel to receive notifications.

        Request Body:
            {
                "tenant_id": "...",
                "channel_id": "...",
                "channel_name": "General",
                "event_types": ["receipt", "budget_alert"]
            }

        Returns:
            JSON response with subscription details
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        # Parse request body
        import json as json_lib

        try:
            body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
            data = json_lib.loads(body.decode("utf-8")) if body else {}
        except (json_lib.JSONDecodeError, ValueError):
            return error_response("Invalid JSON body", 400)

        tenant_id = data.get("tenant_id")
        channel_id = data.get("channel_id")
        channel_name = data.get("channel_name")
        event_types = data.get("event_types", ["receipt", "budget_alert"])

        if not tenant_id or not channel_id:
            return error_response("tenant_id and channel_id are required", 400)

        # Verify workspace exists and belongs to org
        ws_store = self._get_workspace_store()
        workspace = ws_store.get(tenant_id)

        if not workspace:
            return error_response("Workspace not found", 404)

        if workspace.aragora_tenant_id != org.id:
            return error_response("Workspace not found", 404)

        # Create subscription
        from aragora.storage.channel_subscription_store import (
            ChannelSubscription,
            ChannelType,
            EventType,
        )

        # Parse event types
        parsed_events = []
        for et in event_types:
            try:
                parsed_events.append(EventType(et))
            except ValueError:
                return error_response(f"Invalid event type: {et}", 400)

        subscription = ChannelSubscription(
            id="",  # Will be generated
            org_id=org.id,
            channel_type=ChannelType.TEAMS,
            channel_id=channel_id,
            workspace_id=tenant_id,
            channel_name=channel_name,
            event_types=parsed_events,
            created_at=0,  # Will be set
            created_by=db_user.id,
            is_active=True,
            config={},
        )

        sub_store = self._get_subscription_store()
        try:
            created = sub_store.create(subscription)
            logger.info(
                f"Created Teams subscription {created.id} for channel {channel_id} in org {org.id}"
            )
            return json_response({"subscription": created.to_dict()}, status=201)
        except ValueError as e:
            logger.warning("Handler error: %s", e)
            return error_response("Conflict", 409)

    @handle_errors("list Teams subscriptions")
    @require_permission("sme:channels:subscribe")
    def _list_subscriptions(
        self,
        handler: Any,
        query_params: dict,
        user: Any = None,
    ) -> HandlerResult:
        """
        List Teams channel subscriptions for the organization.

        Query Parameters:
            event_type: Filter by event type (receipt, budget_alert, etc.)

        Returns:
            JSON response with subscription list
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        from aragora.storage.channel_subscription_store import ChannelType, EventType

        event_type_str = get_string_param(handler, "event_type", None)
        event_type = None
        if event_type_str:
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                pass

        sub_store = self._get_subscription_store()
        subscriptions = sub_store.get_by_org(
            org.id,
            channel_type=ChannelType.TEAMS,
            event_type=event_type,
        )

        return json_response(
            {
                "subscriptions": [s.to_dict() for s in subscriptions],
                "total": len(subscriptions),
            }
        )

    @handle_errors("delete Teams subscription")
    @require_permission("sme:channels:subscribe")
    def _delete_subscription(
        self,
        handler: Any,
        query_params: dict,
        subscription_id: str,
        user: Any = None,
    ) -> HandlerResult:
        """
        Delete a Teams channel subscription.

        Path Parameters:
            subscription_id: Subscription ID to delete

        Returns:
            JSON response confirming deletion
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        sub_store = self._get_subscription_store()
        subscription = sub_store.get(subscription_id)

        if not subscription:
            return error_response("Subscription not found", 404)

        # Verify subscription belongs to this org
        if subscription.org_id != org.id:
            return error_response("Subscription not found", 404)

        success = sub_store.delete(subscription_id)

        if not success:
            return error_response("Failed to delete subscription", 500)

        logger.info(f"Deleted Teams subscription {subscription_id} for org {org.id}")

        return json_response(
            {
                "deleted": True,
                "subscription_id": subscription_id,
            }
        )

    @handle_errors("start Teams OAuth")
    @require_permission("sme:workspaces:write")
    def _handle_oauth_start(
        self,
        handler: Any,
        query_params: dict,
        user: Any = None,
    ) -> HandlerResult:
        """Start Teams OAuth flow (placeholder for SME flow)."""
        return json_response(
            {
                "status": "oauth_start",
                "message": "Teams OAuth flow not configured for SME endpoint",
            }
        )

    @handle_errors("Teams OAuth callback")
    @require_permission("sme:workspaces:write")
    def _handle_oauth_callback(
        self,
        code: str,
        state: str | None,
        handler: Any,
        user: Any = None,
    ) -> HandlerResult:
        """Handle Teams OAuth callback (placeholder)."""
        if not code:
            return error_response("Missing OAuth code", 400)
        return json_response({"status": "oauth_callback", "code": code, "state": state})


__all__ = ["TeamsWorkspaceHandler"]
