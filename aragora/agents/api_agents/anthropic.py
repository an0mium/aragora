"""
Anthropic API agent with OpenRouter fallback support.
"""

import aiohttp
import asyncio
import json
import logging
from typing import AsyncGenerator

from aragora.agents.api_agents.base import APIAgent
from aragora.agents.api_agents.common import (
    Message,
    Critique,
    handle_agent_errors,
    AgentAPIError,
    AgentRateLimitError,
    AgentConnectionError,
    AgentStreamError,
    AgentTimeoutError,
    get_api_key,
    _sanitize_error_message,
    MAX_STREAM_BUFFER_SIZE,
    iter_chunks_with_timeout,
)
from aragora.agents.fallback import QuotaFallbackMixin
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register(
    "anthropic-api",
    default_model="claude-sonnet-4-20250514",
    default_name="claude-api",
    agent_type="API",
    env_vars="ANTHROPIC_API_KEY",
    accepts_api_key=True,
)
class AnthropicAPIAgent(QuotaFallbackMixin, APIAgent):
    """Agent that uses Anthropic API directly (without CLI).

    Supports automatic fallback to OpenRouter when Anthropic API returns
    billing/quota errors (e.g., "credit balance is too low").

    Uses QuotaFallbackMixin for shared quota detection and fallback logic.
    """

    # Model mapping from Anthropic to OpenRouter format (used by QuotaFallbackMixin)
    OPENROUTER_MODEL_MAP = {
        "claude-opus-4-5-20251101": "anthropic/claude-sonnet-4",
        "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
        "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
        "claude-3-opus-20240229": "anthropic/claude-3-opus",
        "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
        "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
    }
    DEFAULT_FALLBACK_MODEL = "anthropic/claude-sonnet-4"

    def __init__(
        self,
        name: str = "claude-api",
        model: str = "claude-opus-4-5-20251101",
        role: str = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        enable_fallback: bool = True,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or get_api_key("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1",
        )
        self.agent_type = "anthropic"
        self.enable_fallback = enable_fallback
        self._fallback_agent = None  # Cached by QuotaFallbackMixin

    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        retryable_exceptions=(AgentRateLimitError, AgentConnectionError, AgentTimeoutError),
    )
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using Anthropic API.

        Falls back to OpenRouter if billing/quota errors are encountered
        and OPENROUTER_API_KEY is set.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/messages"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": full_prompt}],
        }

        if self.system_prompt:
            payload["system"] = self.system_prompt

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)

                    # Check if this is a quota/billing error and fallback is enabled
                    if self.is_quota_error(response.status, error_text):
                        result = await self.fallback_generate(prompt, context, response.status)
                        if result is not None:
                            return result

                    raise AgentAPIError(
                        f"Anthropic API error {response.status}: {sanitized}",
                        agent_name=self.name,
                        status_code=response.status,
                    )

                data = await response.json()

                try:
                    return data["content"][0]["text"]
                except (KeyError, IndexError):
                    raise AgentAPIError(
                        f"Unexpected Anthropic response format: {data}",
                        agent_name=self.name,
                    )

    async def generate_stream(self, prompt: str, context: list[Message] | None = None) -> AsyncGenerator[str, None]:
        """Stream tokens from Anthropic API.

        Yields chunks of text as they arrive from the API using SSE.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/messages"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": True,
        }

        if self.system_prompt:
            payload["system"] = self.system_prompt

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)

                    # Check for quota/billing errors and fallback to OpenRouter
                    if self.is_quota_error(response.status, error_text):
                        async for chunk in self.fallback_generate_stream(prompt, context, response.status):
                            yield chunk
                        return

                    raise AgentStreamError(
                        f"Anthropic streaming API error {response.status}: {sanitized}",
                        agent_name=self.name,
                    )

                # Anthropic uses SSE format: data: {...}\n\n
                buffer = ""
                try:
                    # Use timeout wrapper to prevent hanging on stalled streams
                    async for chunk in iter_chunks_with_timeout(response.content):
                        buffer += chunk.decode('utf-8', errors='ignore')
                        # Prevent unbounded buffer growth (DoS protection)
                        if len(buffer) > MAX_STREAM_BUFFER_SIZE:
                            raise AgentStreamError(
                                "Streaming buffer exceeded maximum size",
                                agent_name=self.name,
                            )

                        # Process complete SSE lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()

                            if not line or not line.startswith('data: '):
                                continue

                            data_str = line[6:]  # Remove 'data: ' prefix

                            if data_str == '[DONE]':
                                return

                            try:
                                event = json.loads(data_str)
                                event_type = event.get('type', '')

                                # Handle content_block_delta events
                                if event_type == 'content_block_delta':
                                    delta = event.get('delta', {})
                                    if delta.get('type') == 'text_delta':
                                        text = delta.get('text', '')
                                        if text:
                                            yield text

                            except json.JSONDecodeError:
                                continue
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.name}] Streaming timeout")
                    raise
                except aiohttp.ClientError as e:
                    logger.warning(f"[{self.name}] Streaming connection error: {e}")
                    raise AgentStreamError(
                        f"Streaming connection error: {e}",
                        agent_name=self.name,
                        cause=e,
                    )

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using Anthropic API."""
        critique_prompt = f"""Analyze this proposal critically:

Task: {task}

Proposal:
{proposal}

Provide structured feedback:
- ISSUES: Specific problems (bullet points)
- SUGGESTIONS: Improvements (bullet points)
- SEVERITY: 0.0-1.0 rating
- REASONING: Brief explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, "proposal", proposal)


__all__ = ["AnthropicAPIAgent"]
