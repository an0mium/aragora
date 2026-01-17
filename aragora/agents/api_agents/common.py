"""
Shared imports and constants for API-based agents.

This module provides common imports used across all agent implementations
to avoid code duplication.
"""

import asyncio
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Optional

import aiohttp

from aragora.agents.base import CritiqueMixin
from aragora.agents.errors import (
    AgentAPIError,
    AgentCircuitOpenError,
    AgentConnectionError,
    AgentRateLimitError,
    AgentStreamError,
    AgentTimeoutError,
    handle_agent_errors,
)
from aragora.agents.registry import AgentRegistry
from aragora.config import DB_TIMEOUT_SECONDS, get_api_key, get_settings
from aragora.core import Agent, Critique, Message
from aragora.utils.error_sanitizer import sanitize_error_text as _sanitize_error_message

logger = logging.getLogger(__name__)

# =============================================================================
# Connection Pool Configuration
# =============================================================================

# Per-host connection limit (prevents overwhelming single provider)
DEFAULT_CONNECTIONS_PER_HOST = 10

# Total connection limit across all hosts
DEFAULT_TOTAL_CONNECTIONS = 100

# Connection timeout for establishing new connections
DEFAULT_CONNECT_TIMEOUT = 30.0

# Total request timeout (for full request/response cycle)
DEFAULT_REQUEST_TIMEOUT = 120.0


def _get_connection_limits() -> tuple[int, int]:
    """Get connection limits from settings or defaults."""
    settings = get_settings()
    per_host = getattr(settings.agent, "connections_per_host", DEFAULT_CONNECTIONS_PER_HOST)
    total = getattr(settings.agent, "total_connections", DEFAULT_TOTAL_CONNECTIONS)
    return per_host, total


# Global session manager for connection pooling
_session_lock = threading.Lock()
_shared_connector: Optional[aiohttp.TCPConnector] = None
_connector_loop_id: Optional[int] = None  # Track which event loop owns the connector


def get_shared_connector() -> aiohttp.TCPConnector:
    """Get or create a shared TCP connector with connection limits.

    Uses a singleton pattern to reuse connections across requests,
    reducing connection establishment overhead and preventing resource
    exhaustion from too many simultaneous connections.

    The connector is recreated if called from a different event loop,
    since aiohttp connectors are bound to the event loop they were created in.

    Returns:
        Configured TCPConnector instance

    Thread-safe: Uses lock for lazy initialization
    """
    global _shared_connector, _connector_loop_id
    with _session_lock:
        # Get current event loop id (if any)
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            # No running loop - connector will be created for whatever loop uses it first
            current_loop_id = None

        # Recreate connector if it's closed, None, or bound to a different loop
        need_new_connector = (
            _shared_connector is None
            or _shared_connector.closed
            or (current_loop_id is not None and _connector_loop_id != current_loop_id)
        )

        if need_new_connector:
            # Close old connector if it exists and is still open
            if _shared_connector is not None and not _shared_connector.closed:
                try:
                    # Note: close() returns a coroutine, but we schedule it
                    # for execution rather than awaiting in this sync context
                    _shared_connector.close()
                except Exception:  # noqa: BLE001 - Cleanup errors should not propagate
                    pass

            per_host, total = _get_connection_limits()
            _shared_connector = aiohttp.TCPConnector(
                limit=total,
                limit_per_host=per_host,
                ttl_dns_cache=300,  # Cache DNS for 5 minutes
                enable_cleanup_closed=True,  # Clean up closed connections
            )
            _connector_loop_id = current_loop_id
            logger.debug(
                f"Created shared TCP connector: limit={total}, "
                f"per_host={per_host}, loop_id={current_loop_id}"
            )
        return _shared_connector


def create_client_session(
    timeout: Optional[float] = None,
    connector: Optional[aiohttp.TCPConnector] = None,
) -> aiohttp.ClientSession:
    """Create an aiohttp ClientSession with proper connection limits.

    This factory function ensures all API agents use consistent connection
    pooling settings, preventing resource exhaustion.

    Args:
        timeout: Request timeout in seconds (default: DEFAULT_REQUEST_TIMEOUT)
        connector: Custom connector (default: shared connector with limits)

    Returns:
        Configured ClientSession

    Example:
        async with create_client_session() as session:
            async with session.post(url, json=data) as response:
                ...

    Note:
        The session should be used with async context manager to ensure
        proper cleanup. The shared connector is NOT closed when the
        session closes - this is intentional for connection reuse.
    """
    if connector is None:
        connector = get_shared_connector()

    if timeout is None:
        timeout = DEFAULT_REQUEST_TIMEOUT

    client_timeout = aiohttp.ClientTimeout(
        total=timeout,
        connect=DEFAULT_CONNECT_TIMEOUT,
    )

    return aiohttp.ClientSession(
        connector=connector,
        connector_owner=False,  # Don't close connector when session closes
        timeout=client_timeout,
    )


async def close_shared_connector() -> None:
    """Close the shared connector, releasing all connections.

    Call this during application shutdown to properly clean up
    connection resources. Safe to call multiple times.
    """
    global _shared_connector, _connector_loop_id
    with _session_lock:
        if _shared_connector is not None and not _shared_connector.closed:
            await _shared_connector.close()
            _shared_connector = None
            _connector_loop_id = None
            logger.debug("Closed shared TCP connector")


# Maximum buffer size for streaming responses (prevents DoS via memory exhaustion)
# Now configurable via ARAGORA_STREAM_BUFFER_SIZE env var
def _get_stream_buffer_size() -> int:
    """Get max stream buffer size from settings."""
    return get_settings().agent.stream_buffer_size


MAX_STREAM_BUFFER_SIZE = 10 * 1024 * 1024  # Default fallback, use _get_stream_buffer_size()


def calculate_retry_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.3,
) -> float:
    """
    Calculate retry delay with exponential backoff and random jitter.

    Jitter prevents thundering herd when multiple clients recover simultaneously
    after a provider outage. The delay is randomized within a range around the
    exponential backoff value.

    Args:
        attempt: Current retry attempt (0-indexed)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)
        jitter_factor: Fraction of delay to randomize (default: 0.3 = ±30%)

    Returns:
        Delay in seconds with jitter applied

    Example:
        attempt=0: ~1s (0.7-1.3s with 30% jitter)
        attempt=1: ~2s (1.4-2.6s)
        attempt=2: ~4s (2.8-5.2s)
        attempt=3: ~8s (5.6-10.4s)
    """
    # Calculate base exponential delay
    delay = min(base_delay * (2**attempt), max_delay)

    # Apply random jitter: delay ± (jitter_factor * delay)
    jitter = delay * jitter_factor * random.uniform(-1, 1)

    # Ensure minimum delay of 0.1s
    return max(0.1, delay + jitter)


# Default timeout between stream chunks (30 seconds)
# Now configurable via ARAGORA_STREAM_CHUNK_TIMEOUT env var
def _get_stream_chunk_timeout() -> float:
    """Get stream chunk timeout from settings."""
    return get_settings().agent.stream_chunk_timeout


STREAM_CHUNK_TIMEOUT = 90.0  # Default fallback (increased for long-form models)


async def iter_chunks_with_timeout(
    response_content: aiohttp.StreamReader,
    chunk_timeout: float | None = None,
) -> AsyncGenerator[bytes, None]:
    """
    Async generator that wraps response content iteration with per-chunk timeout.

    Prevents indefinite blocking if a stream stalls (server stops sending
    chunks but keeps connection alive). Each chunk must arrive within the
    timeout period or asyncio.TimeoutError is raised.

    Args:
        response_content: aiohttp response.content object with iter_any() method
        chunk_timeout: Maximum seconds to wait for each chunk (default: 30s)

    Yields:
        bytes: Raw chunk data from the stream

    Raises:
        asyncio.TimeoutError: If no chunk received within timeout period

    Example:
        async for chunk in iter_chunks_with_timeout(response.content):
            buffer += chunk.decode('utf-8', errors='ignore')
    """
    # Use config default if not specified
    if chunk_timeout is None:
        chunk_timeout = _get_stream_chunk_timeout()

    async_iter = response_content.iter_any().__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(async_iter.__anext__(), timeout=chunk_timeout)
            yield chunk
        except StopAsyncIteration:
            break


class SSEStreamParser:
    """
    Server-Sent Events (SSE) stream parser for API streaming responses.

    Consolidates the common SSE parsing pattern used across OpenAI, Anthropic,
    and other API agents. Handles buffer management, line parsing, and JSON
    extraction with DoS protection.

    Usage:
        parser = SSEStreamParser(
            content_extractor=lambda e: e.get('choices', [{}])[0].get('delta', {}).get('content', '')
        )
        async for content in parser.parse_stream(response.content):
            yield content

    For Anthropic (different JSON structure):
        parser = SSEStreamParser(
            content_extractor=lambda event: (
                event.get('delta', {}).get('text', '')
                if event.get('type') == 'content_block_delta'
                else ''
            )
        )
    """

    def __init__(
        self,
        content_extractor: Callable[[dict], str],
        done_marker: str = "[DONE]",
        max_buffer_size: int | None = None,
        chunk_timeout: float | None = None,
    ):
        """
        Initialize the SSE parser.

        Args:
            content_extractor: Function to extract text content from parsed JSON event.
                              Takes a dict (parsed JSON) and returns str (content to yield).
            done_marker: String that indicates end of stream (default: "[DONE]")
            max_buffer_size: Maximum buffer size in bytes (DoS protection).
                            Defaults to ARAGORA_STREAM_BUFFER_SIZE config.
            chunk_timeout: Timeout for each chunk in seconds.
                          Defaults to ARAGORA_STREAM_CHUNK_TIMEOUT config.
        """
        self.content_extractor = content_extractor
        self.done_marker = done_marker
        self.max_buffer_size = (
            max_buffer_size if max_buffer_size is not None else _get_stream_buffer_size()
        )
        self.chunk_timeout = (
            chunk_timeout if chunk_timeout is not None else _get_stream_chunk_timeout()
        )

    async def parse_stream(
        self,
        response_content: aiohttp.StreamReader,
        agent_name: str = "agent",
    ) -> AsyncGenerator[str, None]:
        """
        Parse an SSE stream and yield content chunks.

        Args:
            response_content: aiohttp response.content StreamReader
            agent_name: Name for logging (optional)

        Yields:
            Content strings extracted from the stream

        Raises:
            RuntimeError: If buffer exceeds maximum size or connection error
            asyncio.TimeoutError: If chunk timeout exceeded
        """
        buffer = ""
        try:
            async for chunk in iter_chunks_with_timeout(response_content, self.chunk_timeout):
                buffer += chunk.decode("utf-8", errors="ignore")

                # DoS protection: prevent unbounded buffer growth
                if len(buffer) > self.max_buffer_size:
                    raise AgentStreamError(
                        agent_name=agent_name, message="Streaming buffer exceeded maximum size"
                    )

                # Process complete SSE lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    # Skip empty lines and non-data lines
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Remove 'data: ' prefix

                    # Check for end marker
                    if data_str == self.done_marker:
                        return

                    # Parse JSON and extract content
                    try:
                        event = json.loads(data_str)
                        if not isinstance(event, dict):
                            logger.debug(
                                f"[{agent_name}] Unexpected JSON type: {type(event).__name__}"
                            )
                            continue
                        content = self.content_extractor(event)
                        if content:
                            yield content
                    except json.JSONDecodeError as e:
                        # Log malformed JSON for debugging, skip gracefully
                        logger.debug(f"[{agent_name}] Malformed JSON in stream: {e}")
                        continue

        except asyncio.TimeoutError:
            logger.warning(f"[{agent_name}] Streaming timeout")
            raise
        except aiohttp.ClientError as e:
            logger.warning(f"[{agent_name}] Streaming connection error: {e}")
            raise AgentConnectionError(
                f"Streaming connection error: {e}", agent_name=agent_name
            ) from e


# Pre-configured parsers for common providers
def create_openai_sse_parser() -> SSEStreamParser:
    """Create an SSE parser configured for OpenAI API responses."""

    def extract_openai_content(event: dict) -> str:
        choices = event.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            return ""
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""
        delta = first_choice.get("delta")
        if not isinstance(delta, dict):
            return ""
        content = delta.get("content", "")
        return content if isinstance(content, str) else ""

    return SSEStreamParser(content_extractor=extract_openai_content)


def create_anthropic_sse_parser() -> SSEStreamParser:
    """Create an SSE parser configured for Anthropic API responses."""

    def extract_anthropic_content(event: dict) -> str:
        if event.get("type") != "content_block_delta":
            return ""
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return ""
        if delta.get("type") != "text_delta":
            return ""
        text = delta.get("text", "")
        return text if isinstance(text, str) else ""

    return SSEStreamParser(content_extractor=extract_anthropic_content)


__all__ = [
    # Standard library
    "asyncio",
    "aiohttp",
    "json",
    "logging",
    "os",
    "random",
    "re",
    "threading",
    "time",
    "dataclass",
    "Optional",
    "AsyncGenerator",
    # Aragora imports
    "CritiqueMixin",
    "AgentAPIError",
    "AgentCircuitOpenError",
    "AgentConnectionError",
    "AgentRateLimitError",
    "AgentStreamError",
    "AgentTimeoutError",
    "handle_agent_errors",
    "AgentRegistry",
    "DB_TIMEOUT_SECONDS",
    "get_api_key",
    "Agent",
    "Critique",
    "Message",
    "_sanitize_error_message",
    # Module-level
    "logger",
    "MAX_STREAM_BUFFER_SIZE",
    "calculate_retry_delay",
    "STREAM_CHUNK_TIMEOUT",
    "iter_chunks_with_timeout",
    # Connection pooling
    "DEFAULT_CONNECTIONS_PER_HOST",
    "DEFAULT_TOTAL_CONNECTIONS",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_REQUEST_TIMEOUT",
    "get_shared_connector",
    "create_client_session",
    "close_shared_connector",
    # SSE parsing
    "SSEStreamParser",
    "create_openai_sse_parser",
    "create_anthropic_sse_parser",
]
