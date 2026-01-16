"""
Streaming utilities for API agents.

Provides shared SSE (Server-Sent Events) parsing logic to avoid code duplication
across multiple API agent implementations.
"""

__all__ = [
    "StreamingMixin",
    "MAX_STREAM_BUFFER_SIZE",
]

import json
import logging
from typing import Any, AsyncIterator

from aragora.exceptions import StreamingError

logger = logging.getLogger(__name__)

# Maximum buffer size for streaming responses (1MB) - DoS protection
MAX_STREAM_BUFFER_SIZE = 1024 * 1024


class StreamingMixin:
    """Mixin providing shared SSE streaming logic for API agents.

    This mixin extracts the common SSE parsing pattern used by:
    - OpenAI API
    - Anthropic API
    - Grok API
    - OpenRouter API

    Each uses the standard SSE format:
        data: {"json": "object"}
        data: [DONE]

    Usage:
        class MyAgent(APIAgent, StreamingMixin):
            async def generate_stream(self, prompt, context):
                async with session.post(url, ...) as response:
                    async for content in self.parse_sse_stream(
                        response, format_type="openai"
                    ):
                        yield content
    """

    async def parse_sse_stream(
        self,
        response: Any,
        format_type: str = "openai",
    ) -> AsyncIterator[str]:
        """Parse SSE stream and yield content tokens.

        Args:
            response: aiohttp response object with .content.iter_any() method
            format_type: One of "openai", "anthropic", "grok" - determines
                         how to extract content from parsed JSON events

        Yields:
            String content tokens from the stream

        Raises:
            RuntimeError: If buffer exceeds MAX_STREAM_BUFFER_SIZE
        """
        buffer = ""
        async for chunk in response.content.iter_any():
            buffer += chunk.decode("utf-8", errors="ignore")

            # Prevent unbounded buffer growth (DoS protection)
            if len(buffer) > MAX_STREAM_BUFFER_SIZE:
                raise StreamingError("Streaming buffer exceeded maximum size (1MB limit)")

            # Process complete SSE lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                # Skip empty lines and non-data lines
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove 'data: ' prefix

                # Check for stream termination
                if data_str == "[DONE]":
                    return

                try:
                    event = json.loads(data_str)
                    content = self._extract_content_from_event(event, format_type)
                    if content:
                        yield content
                except json.JSONDecodeError:
                    # Skip malformed JSON events
                    pass

    def _extract_content_from_event(self, event: dict[str, Any], format_type: str) -> str:
        """Extract content string from parsed SSE event.

        Args:
            event: Parsed JSON event from SSE stream
            format_type: Provider type determining event structure

        Returns:
            Extracted content string, or empty string if not found
        """
        if format_type == "anthropic":
            # Anthropic format: {"type": "content_block_delta", "delta": {"text": "..."}}
            event_type = event.get("type", "")
            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                text = delta.get("text", "") if isinstance(delta, dict) else ""
                return str(text) if text else ""
            return ""

        elif format_type in ("openai", "grok", "openrouter"):
            # OpenAI-compatible format: {"choices": [{"delta": {"content": "..."}}]}
            choices = event.get("choices", [])
            if choices and isinstance(choices, list) and len(choices) > 0:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    delta = first_choice.get("delta", {})
                    if isinstance(delta, dict):
                        content = delta.get("content", "")
                        return str(content) if content else ""
            return ""

        else:
            logger.warning(f"Unknown streaming format type: {format_type}")
            return ""
