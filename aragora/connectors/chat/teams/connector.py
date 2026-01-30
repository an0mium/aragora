# mypy: ignore-errors
"""
Microsoft Teams Chat Connector.

Implements ChatPlatformConnector for Microsoft Teams using
Bot Framework and Adaptive Cards.

Includes circuit breaker protection for fault tolerance.

Environment Variables:
- TEAMS_APP_ID: Bot application ID
- TEAMS_APP_PASSWORD: Bot application password
- TEAMS_TENANT_ID: Optional tenant ID for single-tenant apps
- TEAMS_REQUEST_TIMEOUT: HTTP request timeout in seconds (default: 30)
- TEAMS_UPLOAD_TIMEOUT: File upload/download timeout in seconds (default: 120)
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.connectors.chat.base import ChatPlatformConnector

import aragora.connectors.chat.teams._constants as _tc
from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin
from aragora.connectors.chat.teams._files import TeamsFilesMixin
from aragora.connectors.chat.teams._events import TeamsEventsMixin
from aragora.connectors.chat.teams._channels import TeamsChannelsMixin

try:
    import httpx
except ImportError:
    pass

logger = logging.getLogger(__name__)


class TeamsConnector(
    TeamsMessagingMixin,
    TeamsFilesMixin,
    TeamsEventsMixin,
    TeamsChannelsMixin,
    ChatPlatformConnector,
):
    """
    Microsoft Teams connector using Bot Framework.

    Supports:
    - Sending messages with Adaptive Cards
    - Responding to commands and interactions
    - File uploads via OneDrive integration
    - Threaded conversations

    Includes circuit breaker protection for fault tolerance against
    Bot Framework API failures and rate limiting.
    """

    def __init__(
        self,
        app_id: str | None = None,
        app_password: str | None = None,
        tenant_id: str | None = None,
        request_timeout: float | None = None,
        upload_timeout: float | None = None,
        **config: Any,
    ):
        """
        Initialize Teams connector.

        Args:
            app_id: Bot application ID (defaults to TEAMS_APP_ID env var)
            app_password: Bot application password (defaults to TEAMS_APP_PASSWORD)
            tenant_id: Optional tenant ID for single-tenant apps
            request_timeout: HTTP request timeout in seconds (default from TEAMS_REQUEST_TIMEOUT env var or 30s)
            upload_timeout: File upload/download timeout in seconds (default from TEAMS_UPLOAD_TIMEOUT env var or 120s)
            **config: Additional configuration
        """
        super().__init__(
            bot_token=app_password or _tc.TEAMS_APP_PASSWORD,
            signing_secret=None,  # Teams uses JWT validation
            request_timeout=request_timeout or _tc.TEAMS_REQUEST_TIMEOUT,
            **config,
        )
        self.app_id = app_id or _tc.TEAMS_APP_ID
        self.app_password = app_password or _tc.TEAMS_APP_PASSWORD
        self.tenant_id = tenant_id or _tc.TEAMS_TENANT_ID
        self._upload_timeout = upload_timeout or _tc.TEAMS_UPLOAD_TIMEOUT
        self._access_token: str | None = None
        self._token_expires: float = 0
        # Separate token cache for Microsoft Graph API
        self._graph_token: str | None = None
        self._graph_token_expires: float = 0

    @property
    def platform_name(self) -> str:
        return "teams"

    @property
    def platform_display_name(self) -> str:
        return "Microsoft Teams"

    async def _get_access_token(self) -> str:
        """
        Get or refresh Bot Framework access token.

        Uses _http_request for retry logic and circuit breaker protection.
        """
        import time

        if self._access_token and time.time() < self._token_expires - 60:
            return self._access_token

        if not _tc.HTTPX_AVAILABLE:
            raise RuntimeError("httpx required for Teams API calls")

        # Use _http_request which handles circuit breaker, retries, and backoff
        success, data, error = await self._http_request(
            method="POST",
            url=_tc.BOT_FRAMEWORK_AUTH_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": self.app_id,
                "client_secret": self.app_password,
                "scope": "https://api.botframework.com/.default",
            },
            operation="get_access_token",
        )

        if not success or not data:
            raise RuntimeError(f"Failed to get Bot Framework token: {error}")

        self._access_token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 3600)

        return self._access_token

    async def _get_graph_token(self) -> str:
        """
        Get or refresh Microsoft Graph API access token.

        Graph API uses a separate OAuth flow from Bot Framework.
        Requires ChannelMessage.Read.All and Files.ReadWrite.All permissions.
        Uses _http_request for retry logic and circuit breaker protection.
        """
        import time

        if self._graph_token and time.time() < self._graph_token_expires - 60:
            return self._graph_token

        if not _tc.HTTPX_AVAILABLE:
            raise RuntimeError("httpx required for Graph API calls")

        if not self.tenant_id:
            raise RuntimeError("Tenant ID required for Graph API. Set TEAMS_TENANT_ID env var.")

        auth_url = _tc.GRAPH_AUTH_URL.format(tenant=self.tenant_id)

        # Use _http_request which handles circuit breaker, retries, and backoff
        success, data, error = await self._http_request(
            method="POST",
            url=auth_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "client_id": self.app_id,
                "client_secret": self.app_password,
                "scope": _tc.GRAPH_SCOPE_FILES,
            },
            operation="get_graph_token",
        )

        if not success or not data:
            raise RuntimeError(f"Failed to get Graph API token: {error}")

        self._graph_token = data["access_token"]
        self._graph_token_expires = time.time() + data.get("expires_in", 3600)

        return self._graph_token

    async def _graph_api_request(
        self,
        endpoint: str,
        method: str = "GET",
        json_data: dict | None = None,
        data: bytes | None = None,
        content_type: str | None = None,
        operation: str = "graph_api",
        **kwargs: Any,
    ) -> tuple[bool, dict | None, str | None]:
        """
        Make a Microsoft Graph API request with auth and circuit breaker.

        Args:
            endpoint: API endpoint (will be appended to GRAPH_API_BASE)
            method: HTTP method
            json_data: Optional JSON body
            data: Optional raw bytes body (for file uploads)
            content_type: Content-Type header for raw data
            operation: Operation name for logging
            **kwargs: Additional arguments passed to _http_request

        Returns:
            Tuple of (success, response_json, error_message)
        """
        try:
            token = await self._get_graph_token()
        except (RuntimeError, httpx.HTTPError, httpx.TimeoutException, OSError, KeyError) as e:
            return False, None, f"Failed to get Graph token: {e}"

        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            **_tc.build_trace_headers(),  # Distributed tracing
        }

        if content_type:
            headers["Content-Type"] = content_type

        # Build the full URL
        url = f"{_tc.GRAPH_API_BASE}{endpoint}"

        return await self._http_request(
            method=method,
            url=url,
            headers=headers,
            json=json_data,
            data=data,
            operation=operation,
        )
