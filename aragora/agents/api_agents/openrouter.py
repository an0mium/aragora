"""
OpenRouter agent and provider-specific subclasses.
"""

import asyncio
import logging
import os
from typing import AsyncGenerator

import aiohttp

from aragora.agents.api_agents.base import APIAgent
from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentConnectionError,
    AgentRateLimitError,
    AgentStreamError,
    Critique,
    Message,
    _sanitize_error_message,
    create_client_session,
    create_openai_sse_parser,
    get_api_key,
)
from aragora.agents.api_agents.rate_limiter import get_openrouter_limiter
from aragora.agents.registry import AgentRegistry
from aragora.config import DB_TIMEOUT_SECONDS
from aragora.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)


@AgentRegistry.register(
    "openrouter",
    default_model="deepseek/deepseek-chat-v3-0324",
    agent_type="API (OpenRouter)",
    env_vars="OPENROUTER_API_KEY",
    description="Generic OpenRouter - specify model via 'model' parameter",
)
class OpenRouterAgent(APIAgent):
    """Agent that uses OpenRouter API for access to many models.

    OpenRouter provides unified access to models like DeepSeek, Llama, Mistral,
    and others through an OpenAI-compatible API.

    Supported models (via model parameter):
    - deepseek/deepseek-chat (DeepSeek V3)
    - deepseek/deepseek-reasoner (DeepSeek R1)
    - meta-llama/llama-3.3-70b-instruct
    - mistralai/mistral-large-2411
    - google/gemini-2.0-flash-exp:free
    - anthropic/claude-3.5-sonnet
    - openai/gpt-4o
    """

    def __init__(
        self,
        name: str = "openrouter",
        role: str = "analyst",
        model: str = "deepseek/deepseek-chat",
        system_prompt: str | None = None,
        timeout: int = 300,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=get_api_key("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.agent_type = "openrouter"
        if system_prompt:
            self.system_prompt = system_prompt

    def _build_context_prompt(
        self,
        context: list[Message] | None = None,
        truncate: bool = False,
        sanitize_fn: object = None,
    ) -> str:
        """Build context prompt from message history.

        OpenRouter-specific: limits to 5 messages, truncates each to 500 chars.

        Args:
            context: List of previous messages
            truncate: Ignored (OpenRouter always truncates for rate limiting)
            sanitize_fn: Ignored (OpenRouter uses simple truncation)
        """
        if not context:
            return ""
        prompt = "Previous discussion:\n"
        for msg in context[-5:]:
            prompt += f"- {msg.agent} ({msg.role}): {msg.content[:500]}...\n"
        return prompt + "\n"

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using OpenRouter API with rate limiting and retry."""
        max_retries = 3
        base_delay = 30  # Start with 30s backoff for rate limits

        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://aragora.ai",
            "X-Title": "Aragora Multi-Agent Debate",
        }

        messages = [{"role": "user", "content": full_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }

        last_error = None
        for attempt in range(max_retries):
            # Acquire rate limit token for each attempt
            limiter = get_openrouter_limiter()
            if not await limiter.acquire(timeout=DB_TIMEOUT_SECONDS):
                raise AgentRateLimitError(
                    "OpenRouter rate limit exceeded, request timed out",
                    agent_name=self.name,
                )

            try:
                async with create_client_session(timeout=self.timeout) as session:
                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        # Update rate limit state from headers
                        limiter.update_from_headers(dict(response.headers))

                        if response.status == 429:
                            # Rate limited - use centralized backoff
                            # This records failure and calculates delay with jitter
                            backoff_delay = limiter.record_rate_limit_error(429)

                            # Check for Retry-After header override
                            retry_after_header = response.headers.get("Retry-After")
                            if retry_after_header:
                                try:
                                    wait_time = min(float(retry_after_header), 300)
                                except ValueError:
                                    wait_time = min(backoff_delay, 300)
                            else:
                                wait_time = min(backoff_delay, 300)

                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"OpenRouter rate limited (429), waiting {wait_time:.0f}s before retry {attempt + 2}/{max_retries}"
                                )
                                await asyncio.sleep(wait_time)
                                last_error = "Rate limited (429)"
                                continue
                            else:
                                raise AgentRateLimitError(
                                    f"OpenRouter rate limited (429) after {max_retries} retries",
                                    agent_name=self.name,
                                )

                        if response.status != 200:
                            error_text = await response.text()
                            sanitized = _sanitize_error_message(error_text)
                            raise AgentAPIError(
                                f"OpenRouter API error {response.status}: {sanitized}",
                                agent_name=self.name,
                                status_code=response.status,
                            )

                        data = await response.json()

                        # Record token usage for billing (OpenAI format)
                        usage = data.get("usage", {})
                        self._record_token_usage(
                            tokens_in=usage.get("prompt_tokens", 0),
                            tokens_out=usage.get("completion_tokens", 0),
                        )

                        try:
                            content = data["choices"][0]["message"]["content"]
                            # Success - reset backoff state
                            limiter.record_success()
                            return content
                        except (KeyError, IndexError):
                            raise AgentAPIError(
                                f"Unexpected OpenRouter response format: {data}",
                                agent_name=self.name,
                            )

            except aiohttp.ClientError as e:
                limiter.release_on_error()
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2**attempt)
                    logger.warning(
                        f"OpenRouter connection error, waiting {wait_time:.0f}s before retry: {e}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise AgentConnectionError(
                    f"OpenRouter connection failed after {max_retries} retries: {last_error}",
                    agent_name=self.name,
                    cause=e,
                )

        # Should not reach here, but satisfy mypy
        raise AgentAPIError(
            f"OpenRouter request failed after {max_retries} retries: {last_error}",
            agent_name=self.name,
        )

    async def generate_stream(
        self, prompt: str, context: list[Message] | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from OpenRouter API with rate limiting and retry.

        Yields chunks of text as they arrive from the API using SSE.
        Implements retry logic with exponential backoff for 429 rate limit errors.
        """
        max_retries = 3
        base_delay = 2.0

        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://aragora.ai",
            "X-Title": "Aragora Multi-Agent Debate",
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

        last_error = None
        for attempt in range(max_retries):
            # Acquire rate limit token for each attempt
            limiter = get_openrouter_limiter()
            if not await limiter.acquire(timeout=DB_TIMEOUT_SECONDS):
                raise AgentRateLimitError(
                    "OpenRouter rate limit exceeded, request timed out",
                    agent_name=self.name,
                )

            try:
                async with create_client_session(timeout=self.timeout) as session:
                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        # Update rate limit state from headers
                        limiter.update_from_headers(dict(response.headers))

                        if response.status == 429:
                            # Rate limited - use centralized backoff
                            backoff_delay = limiter.record_rate_limit_error(429)

                            # Check for Retry-After header override
                            retry_after_header = response.headers.get("Retry-After")
                            if retry_after_header:
                                try:
                                    wait_time = min(float(retry_after_header), 300)
                                except ValueError:
                                    wait_time = min(backoff_delay, 300)
                            else:
                                wait_time = min(backoff_delay, 300)

                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"OpenRouter streaming rate limited (429), waiting {wait_time:.0f}s before retry {attempt + 2}/{max_retries}"
                                )
                                await asyncio.sleep(wait_time)
                                last_error = "Rate limited (429)"
                                continue
                            else:
                                raise AgentRateLimitError(
                                    f"OpenRouter streaming rate limited (429) after {max_retries} retries",
                                    agent_name=self.name,
                                )

                        if response.status != 200:
                            error_text = await response.text()
                            sanitized = _sanitize_error_message(error_text)
                            raise AgentStreamError(
                                f"OpenRouter streaming API error {response.status}: {sanitized}",
                                agent_name=self.name,
                            )

                        # Use SSEStreamParser for consistent SSE parsing (OpenAI-compatible)
                        try:
                            parser = create_openai_sse_parser()
                            async for content in parser.parse_stream(response.content, self.name):
                                yield content
                            # Success - reset backoff state
                            limiter.record_success()
                        except RuntimeError as e:
                            raise AgentStreamError(str(e), agent_name=self.name)
                        # Successfully streamed - exit retry loop
                        return

            except aiohttp.ClientError as e:
                limiter.release_on_error()
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2**attempt)
                    logger.warning(
                        f"OpenRouter streaming connection error, waiting {wait_time:.0f}s before retry: {e}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise AgentConnectionError(
                    f"OpenRouter streaming failed after {max_retries} retries: {last_error}",
                    agent_name=self.name,
                    cause=e,
                )

    async def critique(
        self, proposal: str, task: str, context: list[Message] | None = None
    ) -> Critique:
        """Critique a proposal using OpenRouter API."""
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


# Convenience aliases for specific OpenRouter models
@AgentRegistry.register(
    "deepseek",
    default_model="deepseek/deepseek-chat-v3-0324",
    agent_type="API (OpenRouter)",
    env_vars="OPENROUTER_API_KEY",
    description="DeepSeek V3 - excellent for coding/math, very cost-effective",
)
class DeepSeekAgent(OpenRouterAgent):
    """DeepSeek V3.2 via OpenRouter - latest model with integrated thinking + tool-use."""

    def __init__(
        self,
        name: str = "deepseek",
        role: str = "analyst",
        model: str = "deepseek/deepseek-v3.2",
        system_prompt: str | None = None,
    ):
        super().__init__(
            name=name,
            role=role,
            model=model,
            system_prompt=system_prompt,
        )
        self.agent_type = "deepseek"


@AgentRegistry.register(
    "deepseek-r1",
    default_model="deepseek/deepseek-r1",
    agent_type="API (OpenRouter)",
    env_vars="OPENROUTER_API_KEY",
    description="DeepSeek R1 - chain-of-thought reasoning model",
)
class DeepSeekReasonerAgent(OpenRouterAgent):
    """DeepSeek R1 via OpenRouter - reasoning model with chain-of-thought."""

    def __init__(
        self,
        name: str = "deepseek-r1",
        role: str = "analyst",
        model: str = "deepseek/deepseek-reasoner",
        system_prompt: str | None = None,
    ):
        super().__init__(
            name=name,
            role=role,
            model=model,
            system_prompt=system_prompt,
        )
        self.agent_type = "deepseek-r1"


class DeepSeekV3Agent(OpenRouterAgent):
    """DeepSeek V3.2 via OpenRouter - integrated thinking + tool-use, GPT-5 class reasoning."""

    def __init__(
        self, name: str = "deepseek-v3", role: str = "analyst", system_prompt: str | None = None
    ):
        super().__init__(
            name=name,
            role=role,
            model="deepseek/deepseek-v3.2",  # V3.2 with integrated thinking + tool-use
            system_prompt=system_prompt,
        )
        self.agent_type = "deepseek-v3"


@AgentRegistry.register(
    "llama",
    default_model="meta-llama/llama-3.3-70b-instruct",
    agent_type="API (OpenRouter)",
    env_vars="OPENROUTER_API_KEY",
    description="Llama 3.3 70B Instruct",
)
class LlamaAgent(OpenRouterAgent):
    """Llama 3.3 70B via OpenRouter."""

    def __init__(
        self,
        name: str = "llama",
        role: str = "analyst",
        model: str = "meta-llama/llama-3.3-70b-instruct",
        system_prompt: str | None = None,
    ):
        super().__init__(
            name=name,
            role=role,
            model=model,
            system_prompt=system_prompt,
        )
        self.agent_type = "llama"


@AgentRegistry.register(
    "mistral",
    default_model="mistralai/mistral-large-2411",
    agent_type="API (OpenRouter)",
    env_vars="OPENROUTER_API_KEY",
    description="Mistral Large",
)
class MistralAgent(OpenRouterAgent):
    """Mistral Large via OpenRouter."""

    def __init__(
        self,
        name: str = "mistral",
        role: str = "analyst",
        model: str = "mistralai/mistral-large-2411",
        system_prompt: str | None = None,
    ):
        super().__init__(
            name=name,
            role=role,
            model=model,
            system_prompt=system_prompt,
        )
        self.agent_type = "mistral"


@AgentRegistry.register(
    "qwen",
    default_model="qwen/qwen-2.5-coder-32b-instruct",
    agent_type="API (OpenRouter)",
    env_vars="OPENROUTER_API_KEY",
    description="Qwen 2.5 Coder - Alibaba's code-focused model, strong at technical tasks",
)
class QwenAgent(OpenRouterAgent):
    """Alibaba Qwen 2.5 Coder via OpenRouter - excellent for code generation and analysis."""

    def __init__(
        self,
        name: str = "qwen",
        role: str = "analyst",
        model: str = "qwen/qwen-2.5-coder-32b-instruct",
        system_prompt: str | None = None,
    ):
        super().__init__(
            name=name,
            role=role,
            model=model,
            system_prompt=system_prompt,
        )
        self.agent_type = "qwen"


@AgentRegistry.register(
    "qwen-max",
    default_model="qwen/qwen-max",
    agent_type="API (OpenRouter)",
    env_vars="OPENROUTER_API_KEY",
    description="Qwen Max - Alibaba's flagship model for complex reasoning",
)
class QwenMaxAgent(OpenRouterAgent):
    """Alibaba Qwen Max via OpenRouter - flagship for complex reasoning tasks."""

    def __init__(
        self,
        name: str = "qwen-max",
        role: str = "analyst",
        model: str = "qwen/qwen-max",
        system_prompt: str | None = None,
    ):
        super().__init__(
            name=name,
            role=role,
            model=model,
            system_prompt=system_prompt,
        )
        self.agent_type = "qwen-max"


@AgentRegistry.register(
    "yi",
    default_model="01-ai/yi-large",
    agent_type="API (OpenRouter)",
    env_vars="OPENROUTER_API_KEY",
    description="Yi Large - 01.AI's flagship model with balanced capabilities",
)
class YiAgent(OpenRouterAgent):
    """01.AI Yi Large via OpenRouter - balanced reasoning with cross-cultural perspective."""

    def __init__(
        self,
        name: str = "yi",
        role: str = "analyst",
        model: str = "01-ai/yi-large",
        system_prompt: str | None = None,
    ):
        super().__init__(
            name=name,
            role=role,
            model=model,
            system_prompt=system_prompt,
        )
        self.agent_type = "yi"


@AgentRegistry.register(
    "kimi",
    default_model="moonshot-v1-8k",
    agent_type="API (Kimi/Moonshot)",
    env_vars="KIMI_API_KEY",
    description="Kimi - Moonshot AI's flagship model with strong reasoning",
)
class KimiAgent(APIAgent):
    """Moonshot AI Kimi - strong reasoning and Chinese language capabilities.

    Uses Moonshot's OpenAI-compatible API directly.
    """

    def __init__(
        self,
        name: str = "kimi",
        role: str = "analyst",
        model: str = "moonshot-v1-8k",
        system_prompt: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(name=name, role=role)
        self.model = model
        self.system_prompt = system_prompt
        self.api_key = api_key or os.environ.get("KIMI_API_KEY")
        self.base_url = "https://api.moonshot.cn/v1"
        self.agent_type = "kimi"

        if not self.api_key:
            raise ValueError("KIMI_API_KEY environment variable not set")

    async def generate(self, prompt: str, context: list | None = None) -> str:
        """Generate response using Moonshot API."""

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add context
        context_str = self._build_context_prompt(context)
        if context_str:
            messages.append({"role": "user", "content": context_str})

        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }

        async with create_client_session(timeout=self.timeout) as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ExternalServiceError(
                        service="Kimi API", reason=error_text, status_code=response.status
                    )

                data = await response.json()
                return data["choices"][0]["message"]["content"]


__all__ = [
    "OpenRouterAgent",
    "DeepSeekAgent",
    "DeepSeekReasonerAgent",
    "DeepSeekV3Agent",
    "LlamaAgent",
    "MistralAgent",
    "QwenAgent",
    "QwenMaxAgent",
    "YiAgent",
    "KimiAgent",
]
