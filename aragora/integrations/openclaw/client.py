"""
OpenClaw API Client.

Provides a Python client for interacting with OpenClaw instances,
supporting both the REST API and WebSocket connections.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class OpenClawConfig:
    """Configuration for OpenClaw connection."""

    # Connection settings
    base_url: str = "http://localhost:8080"
    api_key: str | None = None
    timeout_seconds: int = 30

    # WebSocket settings
    ws_url: str | None = None
    ws_reconnect: bool = True
    ws_reconnect_delay: float = 1.0
    ws_max_reconnect_delay: float = 60.0

    # Authentication
    auth_token: str | None = None
    client_id: str | None = None
    client_secret: str | None = None

    # TLS settings
    verify_ssl: bool = True
    ca_bundle: str | None = None

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.5


@dataclass
class OpenClawResponse:
    """Response from OpenClaw API."""

    success: bool
    data: Any = None
    error: str | None = None
    status_code: int = 200
    latency_ms: float = 0.0


class OpenClawClient:
    """
    Client for OpenClaw API.

    Supports:
    - Shell command execution
    - File operations (read, write, delete)
    - Browser control (navigate, screenshot)
    - Input control (keyboard, mouse)
    - WebSocket streaming

    Example:
    ```python
    from aragora.integrations.openclaw import OpenClawClient, OpenClawConfig

    config = OpenClawConfig(
        base_url="http://localhost:8080",
        api_key="your-api-key",
    )
    client = OpenClawClient(config)

    # Execute a shell command
    result = await client.execute_shell("ls -la")

    # Navigate browser
    result = await client.navigate("https://example.com")

    # Take screenshot
    screenshot = await client.screenshot()
    ```
    """

    def __init__(
        self,
        config: OpenClawConfig | None = None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        """
        Initialize the client.

        Args:
            config: OpenClaw configuration
            event_callback: Optional callback for client events
        """
        self._config = config or OpenClawConfig()
        self._event_callback = event_callback
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_connected = False
        self._stats = {
            "requests_made": 0,
            "requests_failed": 0,
            "total_latency_ms": 0.0,
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["X-API-Key"] = self._config.api_key
        if self._config.auth_token:
            headers["Authorization"] = f"Bearer {self._config.auth_token}"
        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
    ) -> OpenClawResponse:
        """Make an HTTP request to OpenClaw."""
        start_time = time.time()
        url = f"{self._config.base_url}{endpoint}"

        for attempt in range(self._config.max_retries):
            try:
                session = await self._get_session()
                headers = self._get_headers()

                async with session.request(
                    method,
                    url,
                    json=data,
                    headers=headers,
                    ssl=self._config.verify_ssl,
                ) as response:
                    latency = (time.time() - start_time) * 1000
                    self._stats["requests_made"] += 1
                    self._stats["total_latency_ms"] += latency

                    if response.status >= 400:
                        error_text = await response.text()
                        return OpenClawResponse(
                            success=False,
                            error=error_text,
                            status_code=response.status,
                            latency_ms=latency,
                        )

                    try:
                        result = await response.json()
                    except (ValueError, aiohttp.ContentTypeError) as e:
                        logger.debug(
                            f"JSON parse failed, falling back to text: {type(e).__name__}: {e}"
                        )
                        result = await response.text()

                    return OpenClawResponse(
                        success=True,
                        data=result,
                        status_code=response.status,
                        latency_ms=latency,
                    )

            except asyncio.TimeoutError:
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
                    continue
                self._stats["requests_failed"] += 1
                return OpenClawResponse(
                    success=False,
                    error="Request timed out",
                    latency_ms=(time.time() - start_time) * 1000,
                )

            except (aiohttp.ClientError, OSError) as e:
                if attempt < self._config.max_retries - 1:
                    logger.debug(
                        f"OpenClaw connection error (attempt {attempt + 1}): {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
                    continue
                self._stats["requests_failed"] += 1
                return OpenClawResponse(
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                    latency_ms=(time.time() - start_time) * 1000,
                )

        return OpenClawResponse(
            success=False,
            error="Max retries exceeded",
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def execute_shell(self, command: str) -> OpenClawResponse:
        """
        Execute a shell command.

        Args:
            command: Shell command to execute

        Returns:
            OpenClawResponse with stdout/stderr
        """
        return await self._request("POST", "/api/shell", {"command": command})

    async def read_file(self, path: str) -> OpenClawResponse:
        """
        Read a file.

        Args:
            path: File path to read

        Returns:
            OpenClawResponse with file content
        """
        return await self._request("GET", f"/api/files?path={path}")

    async def write_file(self, path: str, content: str) -> OpenClawResponse:
        """
        Write a file.

        Args:
            path: File path to write
            content: Content to write

        Returns:
            OpenClawResponse indicating success
        """
        return await self._request(
            "POST",
            "/api/files",
            {
                "path": path,
                "content": content,
            },
        )

    async def delete_file(self, path: str) -> OpenClawResponse:
        """
        Delete a file.

        Args:
            path: File path to delete

        Returns:
            OpenClawResponse indicating success
        """
        return await self._request("DELETE", f"/api/files?path={path}")

    async def navigate(self, url: str) -> OpenClawResponse:
        """
        Navigate browser to URL.

        Args:
            url: URL to navigate to

        Returns:
            OpenClawResponse with page info
        """
        return await self._request("POST", "/api/browser/navigate", {"url": url})

    async def screenshot(self) -> OpenClawResponse:
        """
        Take a screenshot.

        Returns:
            OpenClawResponse with base64 image data
        """
        return await self._request("GET", "/api/browser/screenshot")

    async def type_text(self, text: str) -> OpenClawResponse:
        """
        Type text via keyboard.

        Args:
            text: Text to type

        Returns:
            OpenClawResponse indicating success
        """
        return await self._request("POST", "/api/input/keyboard", {"text": text})

    async def click(self, x: int, y: int) -> OpenClawResponse:
        """
        Click at coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            OpenClawResponse indicating success
        """
        return await self._request(
            "POST",
            "/api/input/mouse",
            {
                "action": "click",
                "x": x,
                "y": y,
            },
        )

    async def api_call(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> OpenClawResponse:
        """
        Make an API call through OpenClaw.

        Args:
            url: API URL to call
            params: Optional request parameters

        Returns:
            OpenClawResponse with API response
        """
        return await self._request(
            "POST",
            "/api/proxy",
            {
                "url": url,
                "params": params or {},
            },
        )

    async def health_check(self) -> OpenClawResponse:
        """
        Check OpenClaw health status.

        Returns:
            OpenClawResponse with health info
        """
        return await self._request("GET", "/health")

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        avg_latency = 0.0
        if self._stats["requests_made"] > 0:
            avg_latency = self._stats["total_latency_ms"] / self._stats["requests_made"]

        return {
            **self._stats,
            "avg_latency_ms": avg_latency,
            "connected": self._session is not None and not self._session.closed,
            "ws_connected": self._ws_connected,
        }

    async def close(self) -> None:
        """Close the client and release resources."""
        if self._ws:
            await self._ws.close()
            self._ws = None
            self._ws_connected = False

        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> OpenClawClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
