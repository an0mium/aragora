"""
Ollama agent for local LLM inference.

Supports:
- Streaming responses
- Model listing and pulling
- Health checks
- Multiple model variants
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
    "ollama",
    default_model="llama3.2",
    agent_type="API",
    requires="Ollama running locally (brew install ollama && ollama serve)",
    env_vars="OLLAMA_HOST (optional, defaults to localhost:11434)",
    description="Local LLM inference via Ollama",
)
class OllamaAgent(APIAgent):
    """Agent that uses local Ollama API with streaming support."""

    def __init__(
        self,
        name: str = "ollama",
        model: str = "llama3.2",
        role: AgentRole = "proposer",
        timeout: int = 180,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            base_url=base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
        self.agent_type = "ollama"

    async def is_available(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            async with create_client_session(timeout=5.0) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False

    async def list_models(self) -> list[dict]:
        """List available models on the Ollama server.

        Returns:
            List of model info dicts with 'name', 'size', 'modified_at' keys.
        """
        async with create_client_session(timeout=10.0) as session:
            try:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status != 200:
                        return []
                    data = await response.json()
                    return data.get("models", [])
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError) as e:
                logger.warning(f"Failed to list Ollama models: {e}")
                return []

    async def pull_model(self, model_name: str) -> AsyncGenerator[dict, None]:
        """Pull a model from Ollama registry.

        Yields progress updates during download.

        Args:
            model_name: Name of model to pull (e.g., 'llama3.2', 'codellama')

        Yields:
            Dict with 'status', 'completed', 'total' keys
        """
        async with create_client_session(timeout=3600.0) as session:  # 1 hour for large models
            async with session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": True},
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode())
                            yield data
                        except json.JSONDecodeError:
                            continue

    async def model_info(self, model_name: Optional[str] = None) -> dict:
        """Get detailed info about a model.

        Args:
            model_name: Model to query (defaults to self.model)

        Returns:
            Dict with model details including parameters, template, etc.
        """
        model = model_name or self.model
        async with create_client_session(timeout=10.0) as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/show",
                    json={"name": model},
                ) as response:
                    if response.status != 200:
                        return {}
                    return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError) as e:
                logger.warning(f"Failed to get model info: {e}")
                return {}

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

        async with create_client_session(timeout=float(self.timeout)) as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        raise AgentAPIError(
                            f"Ollama API error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    try:
                        data = await response.json()
                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                        raise AgentAPIError(
                            f"Ollama returned invalid JSON: {e}",
                            agent_name=self.name,
                        )
                    return data.get("response", "")

            except aiohttp.ClientConnectorError as e:
                raise AgentConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Is Ollama running? Start with: ollama serve",
                    agent_name=self.name,
                    cause=e,
                )

    async def generate_stream(
        self, prompt: str, context: list[Message] | None = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Ollama API.

        Args:
            prompt: The prompt to generate from
            context: Optional conversation context

        Yields:
            Response tokens as they are generated
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
        }

        async with create_client_session(timeout=float(self.timeout)) as session:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        raise AgentAPIError(
                            f"Ollama API error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode())
                                if token := data.get("response"):
                                    yield token
                                if data.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue

            except aiohttp.ClientConnectorError as e:
                raise AgentConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Is Ollama running? Start with: ollama serve",
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
        """Critique a proposal using Ollama."""
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


__all__ = ["OllamaAgent"]
