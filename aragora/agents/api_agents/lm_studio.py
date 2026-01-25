"""
LM Studio agent for local LLM inference.

LM Studio uses an OpenAI-compatible API, making integration straightforward.
Default endpoint: http://localhost:1234/v1
"""

import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Optional

import aiohttp

from aragora.agents.api_agents.base import APIAgent
from aragora.core_types import AgentRole
from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentConnectionError,
    AgentRateLimitError,
    AgentTimeoutError,
    Critique,
    Message,
    _sanitize_error_message,
    create_client_session,
    handle_agent_errors,
)
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register(
    "lm-studio",
    default_model="local-model",
    agent_type="API",
    requires="LM Studio running locally (https://lmstudio.ai/)",
    env_vars="LM_STUDIO_HOST (optional, defaults to localhost:1234)",
    description="Local LLM inference via LM Studio (OpenAI-compatible)",
)
class LMStudioAgent(APIAgent):
    """Agent that uses LM Studio's OpenAI-compatible API."""

    def __init__(
        self,
        name: str = "lm-studio",
        model: str = "local-model",
        role: AgentRole = "proposer",
        timeout: int = 180,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        resolved_base = base_url or os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
        # Ensure /v1 suffix for OpenAI compatibility
        if not resolved_base.endswith("/v1"):
            resolved_base = resolved_base.rstrip("/") + "/v1"

        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            base_url=resolved_base,
        )
        self.agent_type = "lm-studio"
        self.max_tokens = max_tokens

    async def is_available(self) -> bool:
        """Check if LM Studio server is running and accessible."""
        try:
            async with create_client_session(timeout=5.0) as session:
                async with session.get(f"{self.base_url}/models") as response:
                    return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False

    async def list_models(self) -> list[dict]:
        """List available models loaded in LM Studio.

        Returns:
            List of model info dicts with 'id', 'object', 'owned_by' keys.
        """
        async with create_client_session(timeout=10.0) as session:
            try:
                async with session.get(f"{self.base_url}/models") as response:
                    if response.status != 200:
                        return []
                    data = await response.json()
                    return data.get("data", [])
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError) as e:
                logger.warning(f"Failed to list LM Studio models: {e}")
                return []

    async def get_loaded_model(self) -> Optional[str]:
        """Get the currently loaded model in LM Studio.

        Returns:
            Model ID if one is loaded, None otherwise.
        """
        models = await self.list_models()
        if models:
            return models[0].get("id")
        return None

    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        retryable_exceptions=(AgentRateLimitError, AgentConnectionError, AgentTimeoutError),
    )
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using LM Studio's OpenAI-compatible API."""
        messages = []

        # Add system prompt if set
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add context messages
        if context:
            for msg in context:
                role = "assistant" if msg.agent == self.name else "user"
                messages.append({"role": role, "content": msg.content})

        # Add the prompt
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        async with create_client_session(timeout=float(self.timeout)) as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        raise AgentAPIError(
                            f"LM Studio API error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    try:
                        data = await response.json()
                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                        raise AgentAPIError(
                            f"LM Studio returned invalid JSON: {e}",
                            agent_name=self.name,
                        )

                    # Extract content from OpenAI-style response
                    choices = data.get("choices", [])
                    if not choices:
                        return ""
                    return choices[0].get("message", {}).get("content", "")

            except aiohttp.ClientConnectorError as e:
                raise AgentConnectionError(
                    f"Cannot connect to LM Studio at {self.base_url}. "
                    "Is LM Studio running with a model loaded?",
                    agent_name=self.name,
                    cause=e,
                )

    async def generate_stream(
        self, prompt: str, context: list[Message] | None = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using LM Studio API.

        Args:
            prompt: The prompt to generate from
            context: Optional conversation context

        Yields:
            Response tokens as they are generated
        """
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if context:
            for msg in context:
                role = "assistant" if msg.agent == self.name else "user"
                messages.append({"role": role, "content": msg.content})

        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        async with create_client_session(timeout=float(self.timeout)) as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        raise AgentAPIError(
                            f"LM Studio API error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    async for line in response.content:
                        line_str = line.decode().strip()
                        if not line_str or line_str == "data: [DONE]":
                            continue
                        if line_str.startswith("data: "):
                            try:
                                data = json.loads(line_str[6:])
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                if content := delta.get("content"):
                                    yield content
                            except json.JSONDecodeError:
                                continue

            except aiohttp.ClientConnectorError as e:
                raise AgentConnectionError(
                    f"Cannot connect to LM Studio at {self.base_url}. "
                    "Is LM Studio running with a model loaded?",
                    agent_name=self.name,
                    cause=e,
                )

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        """Critique a proposal using LM Studio."""
        target_desc = f" from {target_agent}" if target_agent else ""
        critique_prompt = f"""You are a critical reviewer. Analyze this proposal{target_desc}:

Task: {task}

Proposal:
{proposal}

Provide structured feedback:
ISSUES:
- issue 1
- issue 2

SUGGESTIONS:
- suggestion 1
- suggestion 2

SEVERITY: X (0=trivial, 10=critical)
REASONING: explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, target_agent or "proposal", proposal)


__all__ = ["LMStudioAgent"]
