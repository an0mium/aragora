"""
OpenAI API agent with OpenRouter fallback support.
"""

import aiohttp
import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, AsyncGenerator, Optional

from aragora.agents.api_agents.base import APIAgent

if TYPE_CHECKING:
    from aragora.agents.api_agents.openrouter import OpenRouterAgent
from aragora.agents.api_agents.common import (
    Message,
    Critique,
    handle_agent_errors,
    AgentRateLimitError,
    AgentConnectionError,
    AgentTimeoutError,
    get_api_key,
    _sanitize_error_message,
    MAX_STREAM_BUFFER_SIZE,
)
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
class OpenAIAPIAgent(APIAgent):
    """Agent that uses OpenAI API directly (without CLI).

    Includes automatic fallback to OpenRouter when OpenAI quota is exceeded (429 error).
    The fallback uses the same GPT model via OpenRouter's API.
    """

    # Model mapping from OpenAI to OpenRouter format
    OPENROUTER_MODEL_MAP = {
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-4": "openai/gpt-4",
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "gpt-5.2": "openai/gpt-4o",  # Fallback to gpt-4o if gpt-5.2 not available
    }

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
        self._fallback_agent = None  # Lazy-loaded OpenRouter fallback

    def _get_fallback_agent(self) -> "OpenRouterAgent":
        """Get or create the OpenRouter fallback agent."""
        if self._fallback_agent is None:
            # Lazy import to avoid circular dependency
            from aragora.agents.api_agents.openrouter import OpenRouterAgent

            # Map the model to OpenRouter format
            openrouter_model = self.OPENROUTER_MODEL_MAP.get(self.model, "openai/gpt-4o")

            self._fallback_agent = OpenRouterAgent(
                name=f"{self.name}_fallback",
                model=openrouter_model,
                role=self.role,
                system_prompt=self.system_prompt,
                timeout=self.timeout,
            )
            logger.info(f"Created OpenRouter fallback agent with model {openrouter_model}")
        return self._fallback_agent

    def _is_quota_error(self, status_code: int, error_text: str) -> bool:
        """Check if the error is a quota/rate limit error."""
        if status_code == 429:
            return True
        # Also check for quota-related messages in other error codes
        quota_keywords = ["quota", "rate_limit", "insufficient_quota", "exceeded"]
        return any(kw in error_text.lower() for kw in quota_keywords)

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
                    if self.enable_fallback and self._is_quota_error(response.status, error_text):
                        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
                        if openrouter_key:
                            logger.warning(
                                f"OpenAI quota exceeded (status {response.status}), "
                                f"falling back to OpenRouter for {self.name}"
                            )
                            fallback = self._get_fallback_agent()
                            return await fallback.generate(prompt, context)
                        else:
                            logger.warning(
                                "OpenAI quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
                            )

                    raise RuntimeError(f"OpenAI API error {response.status}: {sanitized}")

                data = await response.json()

                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected OpenAI response format: {data}")

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
                    if self.enable_fallback and self._is_quota_error(response.status, error_text):
                        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
                        if openrouter_key:
                            logger.warning(
                                f"OpenAI quota exceeded (status {response.status}), "
                                f"falling back to OpenRouter streaming for {self.name}"
                            )
                            fallback = self._get_fallback_agent()
                            # Yield from fallback's stream
                            async for token in fallback.generate_stream(prompt, context):
                                yield token
                            return
                        else:
                            logger.warning(
                                "OpenAI quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
                            )

                    raise RuntimeError(f"OpenAI streaming API error {response.status}: {sanitized}")

                # OpenAI uses SSE format: data: {...}\n\n
                buffer = ""
                try:
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode('utf-8', errors='ignore')
                        # Prevent unbounded buffer growth (DoS protection)
                        if len(buffer) > MAX_STREAM_BUFFER_SIZE:
                            raise RuntimeError("Streaming buffer exceeded maximum size")

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
                                choices = event.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content

                            except json.JSONDecodeError:
                                continue
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.name}] Streaming timeout")
                    raise
                except aiohttp.ClientError as e:
                    logger.warning(f"[{self.name}] Streaming connection error: {e}")
                    raise RuntimeError(f"Streaming connection error: {e}")

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
