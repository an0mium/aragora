"""
OpenAI API agent with OpenRouter fallback support.
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
    create_openai_sse_parser,
)
from aragora.agents.fallback import QuotaFallbackMixin
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register(
    "openai-api",
    default_model="gpt-4o",
    default_name="openai-api",
    agent_type="API",
    env_vars="OPENAI_API_KEY",
    accepts_api_key=True,
)
class OpenAIAPIAgent(QuotaFallbackMixin, APIAgent):
    """Agent that uses OpenAI API directly (without CLI).

    Includes automatic fallback to OpenRouter when OpenAI quota is exceeded (429 error).
    The fallback uses the same GPT model via OpenRouter's API.

    Uses QuotaFallbackMixin for shared quota detection and fallback logic.
    """

    # Model mapping from OpenAI to OpenRouter format (used by QuotaFallbackMixin)
    OPENROUTER_MODEL_MAP = {
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-4": "openai/gpt-4",
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "gpt-5.2": "openai/gpt-4o",  # Fallback to gpt-4o if gpt-5.2 not available
    }
    DEFAULT_FALLBACK_MODEL = "openai/gpt-4o"

    def __init__(
        self,
        name: str = "openai-api",
        model: str = "gpt-5.2",
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
            api_key=api_key or get_api_key("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )
        self.agent_type = "openai"
        self.enable_fallback = enable_fallback
        self._fallback_agent = None  # Cached by QuotaFallbackMixin

    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        retryable_exceptions=(AgentRateLimitError, AgentConnectionError, AgentTimeoutError),
    )
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using OpenAI API."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{"role": "user", "content": full_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }

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

                    # Check if this is a quota error and fallback is enabled
                    if self.is_quota_error(response.status, error_text):
                        result = await self.fallback_generate(prompt, context, response.status)
                        if result is not None:
                            return result

                    raise AgentAPIError(
                        f"OpenAI API error {response.status}: {sanitized}",
                        agent_name=self.name,
                        status_code=response.status,
                    )

                data = await response.json()

                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise AgentAPIError(
                        f"Unexpected OpenAI response format: {data}",
                        agent_name=self.name,
                    )

    async def generate_stream(self, prompt: str, context: list[Message] | None = None) -> AsyncGenerator[str, None]:
        """Stream tokens from OpenAI API.

        Yields chunks of text as they arrive from the API using SSE.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{"role": "user", "content": full_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True,
        }

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

                    # Check if this is a quota error and fallback is enabled
                    if self.is_quota_error(response.status, error_text):
                        async for token in self.fallback_generate_stream(prompt, context, response.status):
                            yield token
                        return

                    raise AgentStreamError(
                        f"OpenAI streaming API error {response.status}: {sanitized}",
                        agent_name=self.name,
                    )

                # Use SSEStreamParser for consistent SSE parsing
                try:
                    parser = create_openai_sse_parser()
                    async for content in parser.parse_stream(response.content, self.name):
                        yield content
                except RuntimeError as e:
                    raise AgentStreamError(str(e), agent_name=self.name)

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using OpenAI API."""
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


__all__ = ["OpenAIAPIAgent"]
