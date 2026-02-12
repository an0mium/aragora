"""
Server fixture for E2E smoke tests.

Provides a pytest fixture that starts a real UnifiedServer instance
with dynamic port allocation and automatic cleanup.

Usage:
    @pytest.mark.asyncio
    async def test_health(live_server):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{live_server.base_url}/healthz") as resp:
                assert resp.status == 200
"""

from __future__ import annotations

import asyncio
import socket
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from collections.abc import AsyncGenerator

import aiohttp
import pytest
import pytest_asyncio


def find_free_port() -> int:
    """Find an available port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@dataclass
class LiveServerInfo:
    """Information about a running test server."""

    http_port: int
    ws_port: int
    control_plane_port: int
    nomic_loop_port: int
    host: str = "127.0.0.1"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.http_port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.ws_port}"

    @property
    def control_plane_ws_url(self) -> str:
        return f"ws://{self.host}:{self.control_plane_port}"


async def wait_for_server(
    url: str,
    timeout: float = 30.0,
    interval: float = 0.5,
) -> bool:
    """Wait for server to respond to health check.

    Args:
        url: Health check URL to poll
        timeout: Maximum time to wait in seconds
        interval: Time between poll attempts

    Returns:
        True if server is ready, False if timeout exceeded
    """
    import logging

    logger = logging.getLogger(__name__)
    start = time.monotonic()
    last_error = None
    attempts = 0

    while time.monotonic() - start < timeout:
        attempts += 1
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        logger.info(
                            f"Server ready after {attempts} attempts ({time.monotonic() - start:.1f}s)"
                        )
                        return True
                    else:
                        last_error = f"HTTP {resp.status}"
        except aiohttp.ClientConnectorError as e:
            last_error = f"Connection refused: {e}"
        except asyncio.TimeoutError:
            last_error = "Request timeout"
        except (aiohttp.ClientError, OSError) as e:
            last_error = str(e)
        await asyncio.sleep(interval)

    logger.error(f"Server health check failed after {timeout}s ({attempts} attempts): {last_error}")
    return False


@pytest_asyncio.fixture
async def live_server(tmp_path: Path) -> AsyncGenerator[LiveServerInfo, None]:
    """Start a real UnifiedServer for E2E testing.

    This fixture:
    1. Allocates dynamic ports to avoid conflicts
    2. Creates an isolated nomic directory in tmp_path
    3. Starts the server in a background task
    4. Waits for health check to pass
    5. Yields server info for tests to use
    6. Cancels the server task on cleanup

    Args:
        tmp_path: Pytest's temporary directory fixture

    Yields:
        LiveServerInfo with ports and URLs for the running server
    """
    # Allocate dynamic ports
    http_port = find_free_port()
    ws_port = find_free_port()
    control_plane_port = find_free_port()
    nomic_loop_port = find_free_port()

    # Create isolated nomic directory
    nomic_dir = tmp_path / ".nomic"
    nomic_dir.mkdir(parents=True, exist_ok=True)

    server_info = LiveServerInfo(
        http_port=http_port,
        ws_port=ws_port,
        control_plane_port=control_plane_port,
        nomic_loop_port=nomic_loop_port,
    )

    # Import server module
    from aragora.server.unified_server import UnifiedServer

    # Create server instance
    server = UnifiedServer(
        http_port=http_port,
        ws_port=ws_port,
        control_plane_port=control_plane_port,
        nomic_loop_port=nomic_loop_port,
        http_host="127.0.0.1",
        ws_host="127.0.0.1",
        nomic_dir=nomic_dir,
        enable_persistence=False,  # Disable Supabase for tests
    )

    # Start server in background task
    server_task = asyncio.create_task(server.start(use_parallel_init=False))

    try:
        # Give the HTTP server thread time to start
        await asyncio.sleep(2.0)

        # Wait for server to be ready - try multiple health endpoints
        health_urls = [
            f"{server_info.base_url}/healthz",
            f"{server_info.base_url}/api/health",
            f"{server_info.base_url}/health",
        ]

        is_ready = False
        for health_url in health_urls:
            is_ready = await wait_for_server(health_url, timeout=25.0)
            if is_ready:
                break

        if not is_ready:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            pytest.fail(f"Server failed to start within 30 seconds (tried: {health_urls})")

        yield server_info

    finally:
        # Cleanup: cancel server task with graceful handling
        server_task.cancel()
        try:
            # Give a brief window for graceful shutdown
            await asyncio.wait_for(server_task, timeout=2.0)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            # Force kill if still running
            pass
        except (OSError, ConnectionError, RuntimeError):
            # Ignore socket/connection errors during teardown
            pass


@pytest_asyncio.fixture
async def http_client(live_server: LiveServerInfo) -> AsyncGenerator[aiohttp.ClientSession, None]:
    """Provide an aiohttp client session configured for the live server."""
    async with aiohttp.ClientSession(
        base_url=live_server.base_url,
        timeout=aiohttp.ClientTimeout(total=10),
    ) as session:
        yield session
