"""
Anthropic API agent with OpenRouter fallback support.
"""

import aiohttp
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
    create_anthropic_sse_parser,
    create_client_session,
)
from aragora.agents.fallback import QuotaFallbackMixin
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register(
    "anthropic-api",
    default_model="claude-opus-4-5-20251101",
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

        # Apply generation parameters from persona if set
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        if self.system_prompt:
            payload["system"] = self.system_prompt

        async with create_client_session(timeout=self.timeout) as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
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

                # Record token usage for billing
                usage = data.get("usage", {})
                self._record_token_usage(
                    tokens_in=usage.get("input_tokens", 0),
                    tokens_out=usage.get("output_tokens", 0),
                )

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

        # Apply generation parameters from persona if set
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        if self.system_prompt:
            payload["system"] = self.system_prompt

        async with create_client_session(timeout=self.timeout) as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
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

                # Use SSEStreamParser for consistent SSE parsing
                try:
                    parser = create_anthropic_sse_parser()
                    async for content in parser.parse_stream(response.content, self.name):
                        yield content
                except RuntimeError as e:
                    raise AgentStreamError(str(e), agent_name=self.name)

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
