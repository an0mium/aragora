# Agent Development Guide

Build custom AI agents for the Aragora debate framework.

## Overview

Aragora agents are autonomous participants in multi-agent debates. Each agent:
- Responds to prompts with positions and arguments
- Critiques other agents' responses
- Votes on proposals and consensus
- Learns from debate outcomes via ELO ratings

## Quick Start

### Minimal Agent

```python
from aragora.agents.base import BaseAgent
from aragora.core import Response

class MyAgent(BaseAgent):
    """A simple custom agent."""

    name = "my-agent"
    provider = "custom"

    async def generate(self, prompt: str, **kwargs) -> Response:
        # Your logic here
        return Response(
            content="My response to the prompt",
            agent=self.name,
            metadata={"confidence": 0.85}
        )
```

### Register and Use

```python
from aragora import Arena, Environment, DebateProtocol
from aragora.agents import register_agent

# Register your agent
register_agent("my-agent", MyAgent)

# Use in a debate
env = Environment(task="Discuss the best sorting algorithm")
protocol = DebateProtocol(rounds=3)
arena = Arena(env, agents=["claude", "gpt-4", "my-agent"], protocol=protocol)

result = await arena.run()
```

## Agent Architecture

### Base Agent Interface

```python
class BaseAgent(ABC):
    """Abstract base class for all agents."""

    name: str                    # Unique identifier
    provider: str                # Provider name (anthropic, openai, custom)
    model: str | None = None     # Model identifier

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Response:
        """Generate a response to the prompt."""
        pass

    async def critique(
        self,
        response: Response,
        context: DebateContext
    ) -> Critique:
        """Critique another agent's response."""
        # Default implementation uses generate()
        prompt = self._build_critique_prompt(response, context)
        result = await self.generate(prompt)
        return Critique.from_response(result)

    async def vote(
        self,
        proposals: list[Response],
        context: DebateContext
    ) -> Vote:
        """Vote on proposals."""
        # Default implementation uses generate()
        prompt = self._build_vote_prompt(proposals, context)
        result = await self.generate(prompt)
        return Vote.from_response(result)
```

### Response Model

```python
@dataclass
class Response:
    content: str                           # Main response text
    agent: str                             # Agent that generated it
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    # Optional structured data
    position: str | None = None            # Agent's stance
    arguments: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)
```

## Building API Agents

### OpenAI-Compatible Agent

```python
import httpx
from aragora.agents.base import BaseAgent
from aragora.core import Response

class OpenAICompatibleAgent(BaseAgent):
    """Agent for any OpenAI-compatible API."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4"
    ):
        self.name = name
        self.provider = "openai-compatible"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Response:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        response.raise_for_status()
        data = response.json()

        return Response(
            content=data["choices"][0]["message"]["content"],
            agent=self.name,
            metadata={
                "model": self.model,
                "usage": data.get("usage", {})
            }
        )
```

### Anthropic Agent

```python
import anthropic
from aragora.agents.base import BaseAgent
from aragora.core import Response

class AnthropicAgent(BaseAgent):
    """Agent using Anthropic's Claude models."""

    def __init__(
        self,
        name: str = "claude",
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None
    ):
        self.name = name
        self.provider = "anthropic"
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Response:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant participating in a debate.",
            messages=[{"role": "user", "content": prompt}]
        )

        return Response(
            content=response.content[0].text,
            agent=self.name,
            metadata={
                "model": self.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        )
```

## Building Local Agents

### Ollama Agent

```python
import httpx
from aragora.agents.base import BaseAgent
from aragora.core import Response

class OllamaAgent(BaseAgent):
    """Agent for local Ollama models."""

    def __init__(
        self,
        name: str = "ollama",
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434"
    ):
        self.name = name
        self.provider = "ollama"
        self.model = model
        self.base_url = base_url

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Response:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": False
                },
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

        return Response(
            content=data["response"],
            agent=self.name,
            metadata={
                "model": self.model,
                "eval_count": data.get("eval_count"),
                "eval_duration": data.get("eval_duration")
            }
        )
```

### vLLM Agent

```python
from aragora.agents.base import BaseAgent
from aragora.core import Response
from openai import AsyncOpenAI

class VLLMAgent(BaseAgent):
    """Agent for vLLM server."""

    def __init__(
        self,
        name: str = "vllm",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000/v1"
    ):
        self.name = name
        self.provider = "vllm"
        self.model = model
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed"  # vLLM doesn't require auth by default
        )

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> Response:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return Response(
            content=response.choices[0].message.content,
            agent=self.name,
            metadata={"model": self.model}
        )
```

## Specialized Agents

### Expert Agent

An agent with domain expertise:

```python
class ExpertAgent(BaseAgent):
    """Agent with specialized domain knowledge."""

    def __init__(
        self,
        name: str,
        domain: str,
        expertise_prompt: str,
        base_agent: BaseAgent
    ):
        self.name = name
        self.provider = "expert"
        self.domain = domain
        self.expertise_prompt = expertise_prompt
        self.base_agent = base_agent

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs
    ) -> Response:
        enhanced_system = f"""You are an expert in {self.domain}.

{self.expertise_prompt}

{system or ''}"""

        response = await self.base_agent.generate(
            prompt=prompt,
            system=enhanced_system,
            **kwargs
        )
        response.agent = self.name
        response.metadata["domain"] = self.domain
        return response

# Usage
security_expert = ExpertAgent(
    name="security-expert",
    domain="cybersecurity",
    expertise_prompt="""You have deep expertise in:
- Application security (OWASP Top 10)
- Network security and penetration testing
- Cryptography and secure protocols
- Security architecture and threat modeling

Always consider security implications in your responses.""",
    base_agent=AnthropicAgent()
)
```

### Devil's Advocate Agent

An agent that challenges consensus:

```python
class DevilsAdvocateAgent(BaseAgent):
    """Agent that deliberately challenges the majority position."""

    def __init__(self, base_agent: BaseAgent):
        self.name = "devils-advocate"
        self.provider = "meta"
        self.base_agent = base_agent

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        **kwargs
    ) -> Response:
        contrarian_system = """You are a devil's advocate. Your role is to:
1. Identify the most commonly held position
2. Argue against it with compelling counterarguments
3. Find weaknesses and edge cases
4. Challenge assumptions

Be intellectually rigorous, not contrarian for its own sake."""

        return await self.base_agent.generate(
            prompt=prompt,
            system=contrarian_system + (f"\n\n{system}" if system else ""),
            **kwargs
        )
```

### Ensemble Agent

Combines multiple agents:

```python
class EnsembleAgent(BaseAgent):
    """Agent that synthesizes responses from multiple sub-agents."""

    def __init__(
        self,
        name: str,
        agents: list[BaseAgent],
        synthesizer: BaseAgent
    ):
        self.name = name
        self.provider = "ensemble"
        self.agents = agents
        self.synthesizer = synthesizer

    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> Response:
        # Gather responses from all sub-agents
        import asyncio
        responses = await asyncio.gather(*[
            agent.generate(prompt, **kwargs)
            for agent in self.agents
        ])

        # Synthesize into a single response
        synthesis_prompt = f"""Given these responses to the prompt "{prompt}":

{chr(10).join(f'Agent {r.agent}: {r.content}' for r in responses)}

Synthesize these into a single, coherent response that captures the best insights from each."""

        final = await self.synthesizer.generate(synthesis_prompt)
        final.agent = self.name
        final.metadata["sub_responses"] = [r.to_dict() for r in responses]
        return final
```

## Agent Configuration

### From YAML

```yaml
# agents.yaml
agents:
  claude-expert:
    type: anthropic
    model: claude-sonnet-4-20250514
    temperature: 0.7
    system: "You are a helpful expert assistant."

  gpt-analyst:
    type: openai
    model: gpt-4-turbo
    temperature: 0.3
    system: "You are a careful analyst who examines evidence."

  local-llama:
    type: ollama
    model: llama3.2:70b
    base_url: http://localhost:11434
```

Load configuration:

```python
from aragora.agents import load_agents_from_config

agents = load_agents_from_config("agents.yaml")
```

### From Environment

```python
from aragora.agents import auto_configure_agents

# Auto-detects available API keys and configures agents
agents = auto_configure_agents()
# Returns: ["claude", "gpt-4"] if both API keys are set
```

## Error Handling

### Retry Logic

```python
from aragora.resilience import CircuitBreaker, with_retry

class ResilientAgent(BaseAgent):
    """Agent with built-in resilience."""

    def __init__(self, base_agent: BaseAgent):
        self.name = base_agent.name
        self.provider = base_agent.provider
        self.base_agent = base_agent
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60
        )

    @with_retry(max_attempts=3, backoff_factor=2)
    async def generate(self, prompt: str, **kwargs) -> Response:
        if not self.circuit_breaker.allow_request():
            raise CircuitOpenError(f"Circuit open for {self.name}")

        try:
            response = await self.base_agent.generate(prompt, **kwargs)
            self.circuit_breaker.record_success()
            return response
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
```

### Fallback Agents

```python
from aragora.agents import FallbackAgent

# Primary: Claude, Fallback: GPT-4, Last resort: OpenRouter
agent = FallbackAgent(
    primary=AnthropicAgent(),
    fallbacks=[
        OpenAIAgent(),
        OpenRouterAgent(model="anthropic/claude-3-opus")
    ]
)
```

## Testing Agents

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_my_agent_generate():
    agent = MyAgent()

    response = await agent.generate("What is 2+2?")

    assert response.content
    assert response.agent == "my-agent"
    assert "confidence" in response.metadata

@pytest.mark.asyncio
async def test_my_agent_critique():
    agent = MyAgent()
    other_response = Response(content="The answer is 5", agent="other")
    context = DebateContext(topic="Math problem")

    critique = await agent.critique(other_response, context)

    assert critique.score is not None
    assert critique.feedback
```

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_in_debate():
    from aragora import Arena, Environment, DebateProtocol

    env = Environment(task="Simple test debate")
    protocol = DebateProtocol(rounds=1)
    arena = Arena(env, agents=["my-agent", "claude"], protocol=protocol)

    result = await arena.run()

    assert result.status == "completed"
    assert any(r.agent == "my-agent" for r in result.responses)
```

### Mock Agents for Testing

```python
class MockAgent(BaseAgent):
    """Deterministic agent for testing."""

    def __init__(self, responses: list[str]):
        self.name = "mock"
        self.provider = "mock"
        self.responses = responses
        self.call_count = 0

    async def generate(self, prompt: str, **kwargs) -> Response:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return Response(content=response, agent=self.name)
```

## Best Practices

### 1. Handle Rate Limits

```python
import asyncio
from aragora.utils.rate_limit import RateLimiter

class RateLimitedAgent(BaseAgent):
    def __init__(self, base_agent: BaseAgent, requests_per_minute: int = 60):
        self.base_agent = base_agent
        self.limiter = RateLimiter(requests_per_minute)

    async def generate(self, prompt: str, **kwargs) -> Response:
        await self.limiter.acquire()
        return await self.base_agent.generate(prompt, **kwargs)
```

### 2. Log All Interactions

```python
import structlog

logger = structlog.get_logger()

class LoggingAgent(BaseAgent):
    async def generate(self, prompt: str, **kwargs) -> Response:
        logger.info("agent.generate.start", agent=self.name, prompt_length=len(prompt))

        try:
            response = await self._generate_impl(prompt, **kwargs)
            logger.info("agent.generate.success",
                       agent=self.name,
                       response_length=len(response.content))
            return response
        except Exception as e:
            logger.error("agent.generate.error", agent=self.name, error=str(e))
            raise
```

### 3. Validate Responses

```python
from pydantic import BaseModel, validator

class ValidatedResponse(BaseModel):
    content: str
    confidence: float

    @validator("content")
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Response content cannot be empty")
        return v

    @validator("confidence")
    def confidence_in_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v
```

### 4. Support Streaming

```python
from typing import AsyncIterator

class StreamingAgent(BaseAgent):
    async def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        async for chunk in self._stream_impl(prompt, **kwargs):
            yield chunk

    async def generate(self, prompt: str, **kwargs) -> Response:
        """Non-streaming wrapper."""
        chunks = []
        async for chunk in self.generate_stream(prompt, **kwargs):
            chunks.append(chunk)
        return Response(content="".join(chunks), agent=self.name)
```

## Agent Registry

### Register Custom Agents

```python
from aragora.agents import register_agent, get_agent

# Register
register_agent("my-agent", MyAgent)
register_agent("expert-security", lambda: ExpertAgent(...))

# Retrieve
agent = get_agent("my-agent")
```

### List Available Agents

```python
from aragora.agents import list_agents

agents = list_agents()
# ['claude', 'gpt-4', 'gemini', 'my-agent', 'expert-security', ...]
```

## Related Documentation

- [API Reference](API_REFERENCE.md) - Full API documentation
- [Custom Agents](CUSTOM_AGENTS.md) - More agent examples
- [Architecture](ARCHITECTURE.md) - System architecture
- [Memory Architecture](MEMORY_ARCHITECTURE.md) - How agents learn
