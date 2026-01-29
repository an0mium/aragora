# mypy: ignore-errors
"""
Slack Chat Connector.

Implements ChatPlatformConnector for Slack using
Slack's Web API and Block Kit.

Environment Variables:
- SLACK_BOT_TOKEN: Bot OAuth token (xoxb-...)
- SLACK_SIGNING_SECRET: For webhook verification
- SLACK_WEBHOOK_URL: For incoming webhooks

Resilience Features:
- Circuit breaker protection against Slack API failures
- Exponential backoff retry logic for transient errors
- Configurable timeouts on all API calls
- Rate limit handling (429 responses)

This package was refactored from a single slack.py module.
All public names are re-exported for backward compatibility.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Re-export client-level utilities and constants
from .client import (
    CIRCUIT_BREAKER_COOLDOWN,
    CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    HTTPX_AVAILABLE,
    SLACK_API_BASE,
    SLACK_BOT_TOKEN,
    SLACK_SIGNING_SECRET,
    SLACK_WEBHOOK_URL,
    WorkspaceRateLimit,
    WorkspaceRateLimitRegistry,
    _classify_slack_error,
    _exponential_backoff,
    _is_retryable_error,
    _wait_for_rate_limit,
    get_rate_limit_registry,
)

# Re-export thread manager
from .threads import SlackThreadManager

# Import mixins (not re-exported, used only for class composition)
from .events import SlackEventsMixin
from .messages import SlackMessagesMixin

# Import base class
from aragora.connectors.chat.base import ChatPlatformConnector
from aragora.resilience import get_circuit_breaker

try:
    from aragora.observability.tracing import build_trace_headers
except ImportError:

    def build_trace_headers() -> dict[str, str]:
        return {}

class SlackConnector(SlackMessagesMixin, SlackEventsMixin, ChatPlatformConnector):
    """
    Slack connector using Slack Web API.

    Supports:
    - Sending messages with Block Kit
    - Slash commands
    - Interactive components (buttons, menus)
    - File uploads
    - Threaded conversations
    - Ephemeral messages

    Resilience Features:
    - Circuit breaker protection against API failures
    - Exponential backoff retry for transient errors
    - Configurable timeouts on all API calls
    """

    def __init__(
        self,
        bot_token: str | None = None,
        signing_secret: str | None = None,
        webhook_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
        use_circuit_breaker: bool = True,
        workspace_id: str | None = None,
        enable_token_refresh: bool = True,
        **config: Any,
    ):
        """
        Initialize Slack connector.

        Args:
            bot_token: Bot OAuth token (defaults to SLACK_BOT_TOKEN)
            signing_secret: Webhook signing secret
            webhook_url: Incoming webhook URL
            timeout: Request timeout in seconds (default 30)
            max_retries: Maximum retry attempts for transient errors (default 3)
            use_circuit_breaker: Whether to use circuit breaker (default True)
            workspace_id: Slack workspace ID for multi-workspace token management
            enable_token_refresh: Whether to auto-refresh expired tokens (default True)
            **config: Additional configuration
        """
        super().__init__(
            bot_token=bot_token or SLACK_BOT_TOKEN,
            signing_secret=signing_secret or SLACK_SIGNING_SECRET,
            webhook_url=webhook_url or SLACK_WEBHOOK_URL,
            **config,
        )
        self._timeout = timeout
        self._max_retries = max_retries
        self._use_circuit_breaker = use_circuit_breaker
        self._workspace_id = workspace_id
        self._enable_token_refresh = enable_token_refresh
        self._workspace_store: Any = None  # Lazy-loaded

        # Initialize circuit breaker
        if use_circuit_breaker:
            self._circuit_breaker = get_circuit_breaker(
                "slack_api",
                failure_threshold=CIRCUIT_BREAKER_THRESHOLD,
                cooldown_seconds=CIRCUIT_BREAKER_COOLDOWN,
            )
        else:
            self._circuit_breaker = None

    @property
    def platform_name(self) -> str:
        return "slack"

    @property
    def platform_display_name(self) -> str:
        return "Slack"

    @property
    def is_available(self) -> bool:
        """Check if httpx is installed."""
        return HTTPX_AVAILABLE

    @property
    def is_configured(self) -> bool:
        """Check if bot token is configured."""
        return bool(self.bot_token)

    async def _perform_health_check(self, timeout: float) -> bool:
        """Verify Slack API connectivity with auth.test.

        Uses _slack_api_request with a single attempt for quick feedback.
        """
        if not self.bot_token:
            return False

        success, _, _ = await self._slack_api_request(
            "auth.test",
            operation="health_check",
            timeout=timeout,
            max_retries=1,
        )
        return success

    def _get_headers(self) -> dict[str, str]:
        """Get authorization headers with trace context for distributed tracing."""
        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        # Add trace context headers for distributed tracing
        headers.update(build_trace_headers())
        return headers

    def _get_workspace_store(self) -> Any:
        """Get or create workspace store for token management."""
        if self._workspace_store is None:
            try:
                from aragora.storage.slack_workspace_store import get_slack_workspace_store

                self._workspace_store = get_slack_workspace_store()
            except ImportError:
                logger.debug("Slack workspace store not available")
        return self._workspace_store

    async def _validate_token(self) -> bool:
        """Validate the current token using Slack auth.test API.

        Returns:
            True if token is valid, False otherwise
        """
        return await self._perform_health_check(self._timeout)

    async def _attempt_token_refresh(self) -> bool:
        """Attempt to refresh the token for the current workspace.

        Returns:
            True if token was refreshed successfully, False otherwise
        """
        if not self._enable_token_refresh or not self._workspace_id:
            return False

        store = self._get_workspace_store()
        if not store:
            logger.debug("No workspace store available for token refresh")
            return False

        # Check if token needs refresh
        if not store.is_token_expired(self._workspace_id):
            logger.debug(f"Token for workspace {self._workspace_id} not expired, skipping refresh")
            return False

        # Get OAuth credentials from environment
        client_id = os.environ.get("SLACK_CLIENT_ID", "")
        client_secret = os.environ.get("SLACK_CLIENT_SECRET", "")

        if not client_id or not client_secret:
            logger.warning("SLACK_CLIENT_ID or SLACK_CLIENT_SECRET not set, cannot refresh token")
            return False

        # Attempt refresh
        logger.info(f"Attempting token refresh for workspace: {self._workspace_id}")
        workspace = store.refresh_workspace_token(
            self._workspace_id,
            client_id=client_id,
            client_secret=client_secret,
        )

        if workspace:
            # Update connector's token with refreshed token
            self.bot_token = workspace.access_token
            logger.info(f"Token refreshed successfully for workspace: {self._workspace_id}")
            return True

        logger.error(f"Token refresh failed for workspace: {self._workspace_id}")
        return False

    def _is_auth_error(self, error: str | None) -> bool:
        """Check if error indicates an authentication/token issue."""
        if not error:
            return False
        auth_errors = {
            "invalid_auth",
            "token_revoked",
            "token_expired",
            "account_inactive",
            "not_authed",
        }
        return error.lower() in auth_errors

    async def _slack_api_request(
        self,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        operation: str = "api_call",
        *,
        method: str = "POST",
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        form_data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        """
        Make a Slack API request with circuit breaker, retry, and timeout.

        Centralizes the resilience pattern for all Slack API calls.

        Args:
            endpoint: API endpoint (e.g., "chat.postMessage", "conversations.info")
            payload: JSON payload to send (deprecated, use json_data instead)
            operation: Operation name for logging
            method: HTTP method - "GET" or "POST" (default: "POST")
            params: Query parameters for GET requests
            json_data: JSON body for POST requests (takes precedence over payload)
            form_data: Form data for multipart POST requests (e.g., file uploads)
            files: File data for multipart uploads (e.g., {"file": (name, content, type)})
            timeout: Optional timeout override
            max_retries: Optional retry override for this call

        Returns:
            Tuple of (success, response_data, error_message)
        """
        import httpx as _httpx

        if not HTTPX_AVAILABLE:
            return False, None, "httpx not available"

        # Check circuit breaker
        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            remaining = self._circuit_breaker.cooldown_remaining()
            return False, None, f"Circuit breaker open (retry in {remaining:.0f}s)"

        last_error: str | None = None
        url = f"{SLACK_API_BASE}/{endpoint}"
        request_timeout = timeout if timeout is not None else self._timeout

        # Support both old-style payload and new-style json_data
        body = json_data if json_data is not None else payload

        retries = max_retries if max_retries is not None else self._max_retries
        for attempt in range(retries):
            try:
                async with _httpx.AsyncClient(timeout=request_timeout) as client:
                    if method.upper() == "GET":
                        response = await client.get(
                            url,
                            headers=self._get_headers(),
                            params=params,
                        )
                    elif files is not None:
                        # File upload - use form data and files, not JSON body
                        response = await client.post(
                            url,
                            headers={"Authorization": f"Bearer {self.bot_token}"},
                            data=form_data,
                            files=files,
                        )
                    else:
                        response = await client.post(
                            url,
                            headers=self._get_headers(),
                            json=body,
                        )
                    data = response.json()

                    if data.get("ok"):
                        if self._circuit_breaker:
                            self._circuit_breaker.record_success()
                        return True, data, None
                    else:
                        error = data.get("error", "Unknown error")
                        last_error = error

                        # Check if retryable
                        if _is_retryable_error(response.status_code, error):
                            if attempt < retries - 1:
                                logger.warning(
                                    f"Slack {operation} retryable error: {error} "
                                    f"(attempt {attempt + 1}/{retries})"
                                )
                                # Use Retry-After header for rate limits (429), exponential backoff otherwise
                                if response.status_code == 429:
                                    await _wait_for_rate_limit(response, attempt)
                                else:
                                    await _exponential_backoff(attempt)
                                continue

                        # Check for auth errors - attempt token refresh
                        if self._is_auth_error(error) and attempt < retries - 1:
                            logger.warning(
                                f"Slack {operation} auth error: {error}, attempting token refresh"
                            )
                            if await self._attempt_token_refresh():
                                # Token refreshed, retry immediately
                                continue
                            # Token refresh failed, fall through to non-retryable

                        # Non-retryable error
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        return False, data, error

            except _httpx.TimeoutException:
                last_error = f"Request timeout after {request_timeout}s"
                classified = _classify_slack_error(last_error)
                if attempt < retries - 1:
                    logger.warning(
                        f"[slack] {operation} timeout (attempt {attempt + 1}/{retries}) "
                        f"[{type(classified).__name__}]"
                    )
                    await _exponential_backoff(attempt)
                    continue
                # Final attempt failed
                logger.error(f"[slack] {operation} timeout after {retries} attempts")

            except _httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
                classified = _classify_slack_error(last_error)
                if attempt < retries - 1:
                    logger.warning(
                        f"[slack] {operation} network error (attempt {attempt + 1}/{retries}) "
                        f"[{type(classified).__name__}]"
                    )
                    await _exponential_backoff(attempt)
                    continue
                # Final attempt failed
                logger.error(f"[slack] {operation} network error after {retries} attempts: {e}")

            except Exception as e:
                # Unexpected error - don't retry, classify for metrics
                last_error = f"Unexpected error: {e}"
                classified = _classify_slack_error(last_error)
                logger.exception(
                    f"[slack] {operation} unexpected error [{type(classified).__name__}]: {e}"
                )
                break

        # All retries exhausted - classify final error for metrics
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()
        if last_error:
            classified = _classify_slack_error(last_error)
            logger.debug(f"[slack] {operation} final error type: {type(classified).__name__}")
        return False, None, last_error or "Unknown error"

__all__ = [
    # Main classes
    "SlackConnector",
    "SlackThreadManager",
    # Client utilities
    "HTTPX_AVAILABLE",
    "SLACK_API_BASE",
    "SLACK_BOT_TOKEN",
    "SLACK_SIGNING_SECRET",
    "SLACK_WEBHOOK_URL",
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRIES",
    "CIRCUIT_BREAKER_THRESHOLD",
    "CIRCUIT_BREAKER_COOLDOWN",
    "_classify_slack_error",
    "_is_retryable_error",
    "_exponential_backoff",
    "_wait_for_rate_limit",
    "WorkspaceRateLimit",
    "WorkspaceRateLimitRegistry",
    "get_rate_limit_registry",
]
