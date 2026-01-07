"""
Grok agent for xAI's Grok API.
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
    "grok",
    default_model="grok-3",
    agent_type="API",
    env_vars="XAI_API_KEY or GROK_API_KEY",
)
class GrokAgent(APIAgent):
    """Agent that uses xAI's Grok API (OpenAI-compatible).

    Uses the xAI API at https://api.x.ai/v1 with models like grok-3.
    """

    def __init__(
        self,
        name: str = "grok",
        model: str = "grok-4",
        role: str = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or get_api_key("XAI_API_KEY", "GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        self.agent_type = "grok"

    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        retryable_exceptions=(AgentRateLimitError, AgentConnectionError, AgentTimeoutError),
    )
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using Grok API."""

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
                    raise RuntimeError(f"Grok API error {response.status}: {sanitized}")

                data = await response.json()

                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected Grok response format: {data}")

    async def generate_stream(self, prompt: str, context: list[Message] | None = None) -> AsyncGenerator[str, None]:
        """Stream tokens from Grok API."""
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
                    raise RuntimeError(f"Grok streaming API error {response.status}: {sanitized}")

                try:
                    buffer = ""
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode('utf-8', errors='ignore')

                        # Prevent unbounded buffer growth (DoS protection)
                        if len(buffer) > MAX_STREAM_BUFFER_SIZE:
                            raise RuntimeError("Streaming buffer exceeded maximum size")

                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()

                            if not line or not line.startswith('data: '):
                                continue

                            data_str = line[6:]

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
        """Critique a proposal using Grok API."""
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


__all__ = ["GrokAgent"]
