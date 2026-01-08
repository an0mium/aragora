"""
OpenAI-compatible API agent mixin.

Provides common implementation for agents using OpenAI-compatible APIs:
- OpenAI
- Grok (xAI)
- OpenRouter (and its model variants)
- Any other OpenAI-compatible endpoint

This eliminates ~150 lines of duplicate code per agent.
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
    _sanitize_error_message,
    create_openai_sse_parser,
)
from aragora.agents.fallback import QuotaFallbackMixin

logger = logging.getLogger(__name__)


class OpenAICompatibleMixin(QuotaFallbackMixin):
    """
    Mixin providing OpenAI-compatible API implementation.

    Subclasses must define:
    - OPENROUTER_MODEL_MAP: dict mapping models to OpenRouter equivalents
    - DEFAULT_FALLBACK_MODEL: default OpenRouter model for fallback
    - agent_type: str identifying the agent type
    - base_url: API base URL
    - api_key: API key

    Optional overrides:
    - _build_extra_headers(): Add provider-specific headers
    - _build_extra_payload(): Add provider-specific payload fields
    - max_tokens: Default 4096
    """

    # Subclasses should define these
    OPENROUTER_MODEL_MAP: dict[str, str] = {}
    DEFAULT_FALLBACK_MODEL: str = "openai/gpt-4o"

    # Default max tokens (can be overridden)
    max_tokens: int = 4096

    # Expected from base class (APIAgent) - declared for type checking
    api_key: str | None
    base_url: str | None
    model: str
    name: str
    agent_type: str
    timeout: int

    # Methods inherited from CritiqueMixin (via APIAgent):
    # - _build_context_prompt(context: list[Message]) -> str
    # - _parse_critique(response: str, target_agent: str, target_content: str) -> Critique

    def _build_headers(self) -> dict:
        """Build request headers. Override to add provider-specific headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        extra = self._build_extra_headers()
        if extra:
            headers.update(extra)
        return headers

    def _build_extra_headers(self) -> dict | None:
        """Override to add provider-specific headers."""
        return None

    def _build_messages(self, full_prompt: str) -> list[dict]:
        """Build messages array with optional system prompt."""
        messages = [{"role": "user", "content": full_prompt}]
        if hasattr(self, 'system_prompt') and self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def _build_payload(self, messages: list[dict], stream: bool = False) -> dict:
        """Build request payload. Override to add provider-specific fields."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if stream:
            payload["stream"] = True
        extra = self._build_extra_payload()
        if extra:
            payload.update(extra)
        return payload

    def _build_extra_payload(self) -> dict | None:
        """Override to add provider-specific payload fields."""
        return None

    def _parse_response(self, data: dict) -> str:
        """Parse response content from API response."""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise AgentAPIError(
                f"Unexpected {self.agent_type} response format: {data}",
                agent_name=self.name,
            )

    def _get_endpoint_url(self) -> str:
        """Get the chat completions endpoint URL."""
        return f"{self.base_url}/chat/completions"

    def _get_error_prefix(self) -> str:
        """Get error message prefix for this agent type."""
        return self.agent_type.title() if hasattr(self, 'agent_type') else "API"

    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        retryable_exceptions=(AgentRateLimitError, AgentConnectionError, AgentTimeoutError),
    )
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using the OpenAI-compatible API."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = self._get_endpoint_url()
        headers = self._build_headers()
        messages = self._build_messages(full_prompt)
        payload = self._build_payload(messages, stream=False)

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

                    # Check for quota/billing errors and fallback
                    if self.is_quota_error(response.status, error_text):
                        result = await self.fallback_generate(prompt, context, response.status)
                        if result is not None:
                            return result

                    raise AgentAPIError(
                        f"{self._get_error_prefix()} API error {response.status}: {sanitized}",
                        agent_name=self.name,
                        status_code=response.status,
                    )

                data = await response.json()
                return self._parse_response(data)

    async def generate_stream(
        self, prompt: str, context: list[Message] | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the OpenAI-compatible API."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = self._get_endpoint_url()
        headers = self._build_headers()
        messages = self._build_messages(full_prompt)
        payload = self._build_payload(messages, stream=True)

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

                    # Check for quota errors and fallback
                    if self.is_quota_error(response.status, error_text):
                        async for chunk in self.fallback_generate_stream(
                            prompt, context, response.status
                        ):
                            yield chunk
                        return

                    raise AgentStreamError(
                        f"{self._get_error_prefix()} streaming API error {response.status}: {sanitized}",
                        agent_name=self.name,
                    )

                # Use SSE parser for consistent streaming
                try:
                    parser = create_openai_sse_parser()
                    async for content in parser.parse_stream(response.content, self.name):
                        yield content
                except RuntimeError as e:
                    raise AgentStreamError(str(e), agent_name=self.name)

    async def critique(
        self, proposal: str, task: str, context: list[Message] | None = None
    ) -> Critique:
        """Critique a proposal using the API."""
        critique_prompt = f"""Critically analyze this proposal:

Task: {task}
Proposal: {proposal}

Format your response as:
ISSUES:
- issue 1
- issue 2

SUGGESTIONS:
- suggestion 1
- suggestion 2

SEVERITY: X.X
REASONING: explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, "proposal", proposal)


__all__ = ["OpenAICompatibleMixin"]
