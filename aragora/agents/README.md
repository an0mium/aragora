# Agents Module

Multi-provider agent management for AI model orchestration.

## Quick Start

```python
from aragora.agents import create_agent, wrap_agent, AirlockConfig

# Create agent with automatic fallback
agent = create_agent("anthropic-api", name="claude-debate")

# Wrap with resilience layer
config = AirlockConfig(generate_timeout=240.0)
resilient = wrap_agent(agent, config)

# Generate with automatic fallback on quota
response = await resilient.generate("Analyze this problem...")
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `Agent` | Abstract base with `generate()` and `critique()` methods |
| `APIAgent` | Base for API agents with circuit breaker and rate limiting |
| `CLIAgent` | Base for CLI agents with subprocess management |
| `AirlockProxy` | Resilience wrapper with timeouts and fallback |
| `QuotaFallbackMixin` | Automatic OpenRouter fallback on rate limits |
| `AgentPerformanceMonitor` | Tracks success rates, latency, tokens |

## Supported Providers

| Provider | Type | Key Models |
|----------|------|------------|
| Anthropic | CLI/API | Claude Opus, Sonnet, Haiku |
| OpenAI | CLI/API | GPT-4o, GPT-4 Turbo |
| Google | CLI/API | Gemini 2.0 Flash, Pro |
| xAI | CLI/API | Grok 2, Grok 4 |
| Mistral | API | Large, Codestral |
| Deepseek | OpenRouter | v3, Reasoner, Coder |
| Meta | OpenRouter | Llama 3.1, 4 variants |
| Alibaba | OpenRouter | Qwen 2.5-Coder, Max |
| OpenRouter | API | 50+ models (unified fallback) |
| Ollama | API | Local models |
| LM Studio | API | Local inference |

## Architecture

```
agents/
├── base.py           # Agent, CritiqueMixin, BaseDebateAgent
├── cli_agents.py     # CLI wrappers (Claude, Codex, Gemini, Grok)
├── fallback.py       # QuotaFallbackMixin, AgentFallbackChain
├── airlock.py        # AirlockProxy resilience wrapper
├── api_agents/       # Direct API implementations
│   ├── base.py       # APIAgent base class
│   ├── anthropic.py  # Anthropic Claude API
│   ├── openai.py     # OpenAI GPT API
│   ├── gemini.py     # Google Gemini API
│   ├── grok.py       # xAI Grok API
│   ├── mistral.py    # Mistral API
│   ├── openrouter.py # 40+ OpenRouter models
│   ├── ollama.py     # Local Ollama models
│   └── lm_studio.py  # LM Studio local
├── configs/          # YAML team configurations
└── errors/           # Error classification
```

## Key Patterns

- **Fallback Chain**: Automatic provider switching on quotas
- **Circuit Breaker**: Prevents cascading failures
- **Rate Limiting**: Per-provider adaptive backoff
- **Performance Monitoring**: Latency and success tracking

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [docs/AGENT_DEVELOPMENT.md](../../docs/AGENT_DEVELOPMENT.md) - Agent development guide
