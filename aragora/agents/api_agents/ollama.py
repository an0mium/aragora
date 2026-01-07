"""
Ollama agent for local LLM inference.
"""

import aiohttp
import json
import logging
import os

from aragora.agents.api_agents.base import APIAgent
from aragora.agents.api_agents.common import (
    Message,
    Critique,
    handle_agent_errors,
    AgentRateLimitError,
    AgentConnectionError,
    AgentTimeoutError,
    _sanitize_error_message,
)
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register(
    "ollama",
    default_model="llama3.2",
    agent_type="API",
    requires="Ollama running locally (brew install ollama && ollama serve)",
    env_vars="OLLAMA_HOST (optional, defaults to localhost:11434)",
)
class OllamaAgent(APIAgent):
    """Agent that uses local Ollama API."""

    def __init__(
        self,
        name: str = "ollama",
        model: str = "llama3.2",
        role: str = "proposer",
        timeout: int = 180,
        base_url: str | None = None,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            base_url=base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
        self.agent_type = "ollama"

    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        retryable_exceptions=(AgentRateLimitError, AgentConnectionError, AgentTimeoutError),
    )
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using Ollama API."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        raise RuntimeError(f"Ollama API error {response.status}: {sanitized}")

                    try:
                        data = await response.json()
                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                        raise RuntimeError(f"Ollama returned invalid JSON: {e}")
                    return data.get("response", "")

            except aiohttp.ClientConnectorError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Is Ollama running? Start with: ollama serve"
                )

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using Ollama."""
        critique_prompt = f"""You are a critical reviewer. Analyze this proposal:

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

SEVERITY: X.X (0.0 minor to 1.0 critical)
REASONING: explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, "proposal", proposal)


__all__ = ["OllamaAgent"]
