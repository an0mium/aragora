# Custom Agent Development Guide

Build your own agents for Aragora stress-tests. This guide covers creating API agents, CLI agents, and integrating with the registry.

## Agent Architecture

All agents implement a common interface:

```python
class Agent:
    name: str           # Unique identifier
    role: str           # "proposer", "critic", "synthesizer"

    async def generate(self, prompt: str, context: dict) -> str:
        """Generate a response to the prompt."""

    async def critique(self, content: str, context: dict) -> str:
        """Critique another agent's content."""
```

## Creating an API Agent

### Step 1: Create the Agent File

Create a new module under `aragora/agents/` (for example `my_provider.py`):

```python
"""
MyProvider API Agent.

Integrates with MyProvider's API for adversarial stress-tests.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import httpx

from aragora.agents.registry import AgentRegistry
from .base import APIAgent, APIAgentConfig

logger = logging.getLogger(__name__)


@AgentRegistry.register(
    "my-provider",
    default_model="my-model-v1",
    agent_type="API",
    env_vars="MY_PROVIDER_API_KEY",
    description="MyProvider AI for adversarial stress-tests",
    accepts_api_key=True,
)
class MyProviderAgent(APIAgent):
    """Agent using MyProvider's API."""

    BASE_URL = "https://api.myprovider.com/v1"

    def __init__(
        self,
        name: str = "my-provider",
        role: str = "proposer",
        model: str = "my-model-v1",
        api_key: Optional[str] = None,
    ):
        super().__init__(name=name, role=role)
        self.model = model
        self.api_key = api_key or os.getenv("MY_PROVIDER_API_KEY")

        if not self.api_key:
            raise ValueError("MY_PROVIDER_API_KEY environment variable required")

        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    async def generate(self, prompt: str, context: dict = None) -> str:
        """Generate a response using MyProvider API."""
        messages = self._build_messages(prompt, context)

        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
            },
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def critique(self, content: str, context: dict = None) -> str:
        """Critique another agent's content."""
        critique_prompt = f"""You are an adversarial reviewer in a decision stress-test.

Analyze the following content and provide constructive criticism:

{content}

Focus on:
1. Logical flaws or unsupported claims
2. Missing considerations
3. Potential risks or edge cases
4. Alternative perspectives

Be specific and actionable in your critique."""

        return await self.generate(critique_prompt, context)

    def _build_messages(self, prompt: str, context: dict = None) -> list:
        """Build message list for API call."""
        messages = []

        # System message based on role
        system_prompts = {
            "proposer": "You are a thoughtful proposer in a decision stress-test. Make clear, risk-aware arguments.",
            "critic": "You are a critical analyst. Red-team the proposal and surface flaws, risks, and missing considerations.",
            "synthesizer": "You synthesize multiple perspectives into a balanced, defensible conclusion.",
        }
        messages.append({
            "role": "system",
            "content": system_prompts.get(self.role, system_prompts["proposer"]),
        })

        # Add context if provided
        if context and context.get("history"):
            for msg in context["history"][-5:]:  # Last 5 messages
                messages.append({
                    "role": "user" if msg["agent"] != self.name else "assistant",
                    "content": f"[{msg['agent']}]: {msg['content']}",
                })

        # Add the prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
```

### Step 2: Register in the Package

Add to `aragora/agents/api_agents/__init__.py`:

```python
from .my_provider import MyProviderAgent

__all__ = [
    # ... existing exports
    "MyProviderAgent",
]
```

### Step 3: Test Your Agent

```python
# tests/agents/test_my_provider.py
import pytest
from unittest.mock import AsyncMock, patch

from aragora.agents.api_agents.my_provider import MyProviderAgent


class TestMyProviderAgent:
    """Tests for MyProviderAgent."""

    def test_init_requires_api_key(self):
        """Should require API key."""
        with pytest.raises(ValueError, match="API_KEY"):
            MyProviderAgent(api_key=None)

    def test_init_with_api_key(self):
        """Should initialize with API key."""
        agent = MyProviderAgent(api_key="test-key")
        assert agent.name == "my-provider"
        assert agent.role == "proposer"

    @pytest.mark.asyncio
    async def test_generate(self):
        """Should generate response."""
        agent = MyProviderAgent(api_key="test-key")

        with patch.object(agent.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Test response"}}]
            }
            mock_post.return_value.raise_for_status = lambda: None

            result = await agent.generate("Test prompt")
            assert result == "Test response"

    @pytest.mark.asyncio
    async def test_critique(self):
        """Should critique content."""
        agent = MyProviderAgent(api_key="test-key", role="critic")

        with patch.object(agent, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Critique: Missing edge cases"

            result = await agent.critique("Some content to review")
            assert "Critique" in result
```

## Creating a CLI Agent

For agents that wrap CLI tools (like `claude` or `codex`):

```python
# Example file: my_cli.py (place under `aragora/agents/`)

import subprocess
import asyncio
from aragora.agents.registry import AgentRegistry
from .base import CLIAgent


@AgentRegistry.register(
    "my-cli",
    agent_type="CLI",
    requires="my-cli tool (npm install -g my-cli)",
    description="CLI agent wrapping my-cli tool",
)
class MyCLIAgent(CLIAgent):
    """Agent wrapping a CLI tool."""

    COMMAND = "my-cli"

    def __init__(self, name: str = "my-cli", role: str = "proposer"):
        super().__init__(name=name, role=role)
        self._verify_cli_available()

    def _verify_cli_available(self):
        """Check if CLI tool is installed."""
        try:
            subprocess.run(
                [self.COMMAND, "--version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(f"{self.COMMAND} not found. Install with: npm install -g {self.COMMAND}")

    async def generate(self, prompt: str, context: dict = None) -> str:
        """Generate response via CLI."""
        # Write prompt to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                self.COMMAND,
                "generate",
                "--input", prompt_file,
                "--format", "text",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(f"CLI error: {stderr.decode()}")

            return stdout.decode().strip()
        finally:
            import os
            os.unlink(prompt_file)
```

## Using OpenRouter for Quick Integration

For models available on OpenRouter, use the base class:

```python
# Quick integration via OpenRouter
from aragora.agents.api_agents.openrouter import OpenRouterAgent
from aragora.agents.registry import AgentRegistry


@AgentRegistry.register(
    "my-openrouter-model",
    default_model="my-provider/my-model",
    agent_type="API (OpenRouter)",
    env_vars="OPENROUTER_API_KEY",
    description="My model via OpenRouter",
)
class MyOpenRouterModel(OpenRouterAgent):
    """Access my-model via OpenRouter."""

    def __init__(self, name: str = "my-model", role: str = "proposer"):
        super().__init__(
            name=name,
            role=role,
            model="my-provider/my-model",
        )
```

## Advanced Features

### Streaming Responses

```python
async def generate_stream(self, prompt: str, context: dict = None):
    """Stream response tokens."""
    async with self.client.stream(
        "POST",
        "/chat/completions",
        json={"model": self.model, "messages": [...], "stream": True},
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if content := data["choices"][0]["delta"].get("content"):
                    yield content
```

### Error Handling and Fallback

```python
from aragora.agents.fallback import with_fallback

@with_fallback("openrouter")  # Fall back to OpenRouter on failure
class MyProviderAgent(APIAgent):
    ...
```

### Rate Limiting

```python
from aragora.resilience import RateLimiter

class MyProviderAgent(APIAgent):
    def __init__(self, ...):
        super().__init__(...)
        self.rate_limiter = RateLimiter(
            max_requests=60,
            time_window=60,  # seconds
        )

    async def generate(self, prompt: str, context: dict = None) -> str:
        await self.rate_limiter.acquire()
        return await self._generate_impl(prompt, context)
```

## Testing Your Agent in Debates

```python
# Interactive test
import asyncio
from aragora.agents.registry import AgentRegistry, register_all_agents

register_all_agents()

async def test_debate():
    agent = AgentRegistry.create("my-provider", role="proposer")

    response = await agent.generate(
        "Should we use microservices or monolith?",
        context={"topic": "architecture"}
    )
    print(f"Response: {response}")

    critique = await agent.critique(response)
    print(f"Self-critique: {critique}")

asyncio.run(test_debate())
```

## Checklist

Before submitting your agent:

- [ ] Implements `generate()` and `critique()` methods
- [ ] Registered with `@AgentRegistry.register`
- [ ] Has unit tests
- [ ] Handles API errors gracefully
- [ ] Documents required environment variables
- [ ] Added to `__init__.py` exports
- [ ] Tested in a real debate

## Examples

See existing agents for reference:
- `aragora/agents/api_agents/anthropic.py` - Full-featured API agent
- `aragora/agents/api_agents/openrouter.py` - OpenRouter integration
- `aragora/agents/cli_agents.py` - CLI agent patterns

---

*Build agents. Shape debates. Improve outcomes.*
