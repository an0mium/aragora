# Agent Catalog (Overview)

Aragora supports multiple agent backends for vetted decisionmaking. This page summarizes
the agent families and how to select them. For the full up-to-date catalog,
see `AGENTS.md` at the repository root or call `list_available_agents()` at runtime.

**Allowlist note:** Server-side validation uses `ALLOWED_AGENT_TYPES` in
`aragora/config/settings.py`. Some registered agents are opt-in and not
allowlisted by default.

## Agent Families

### CLI Agents (allowlisted)

Use external CLI tools with local credentials:

- `claude`, `codex`, `openai`, `gemini-cli`, `grok-cli`, `qwen-cli`,
  `deepseek-cli`, `kilocode`

### Direct API Agents

Call provider APIs directly:

- Allowlisted: `anthropic-api`, `openai-api`, `gemini`, `grok`, `ollama`
- Opt-in: `mistral-api`, `codestral`, `lm-studio`, `kimi-legacy`
- Fine-tuned (opt-in): `tinker`, `tinker-llama`, `tinker-qwen`, `tinker-deepseek`

### OpenRouter Agents (allowlisted)

OpenRouter-backed unified access:

- `deepseek`, `deepseek-r1`, `llama`, `mistral`, `qwen`, `qwen-max`, `yi`,
  `kimi`, `kimi-thinking`, `llama4-maverick`, `llama4-scout`, `sonar`,
  `command-r`, `jamba`, `openrouter`

### External Framework Proxies (allowlisted)

- `external-framework`, `openclaw`, `crewai`, `autogen`, `langgraph`

### Built-In

- `demo`

## Choosing Agents

Use heterogeneous mixes to reduce correlated blind spots:

```python
from aragora.agents import create_agent

agents = [
    create_agent("anthropic-api", name="proposer", role="proposer"),
    create_agent("openai-api", name="critic", role="critic"),
    create_agent("gemini", name="judge", role="synthesizer"),
]
```

## Roles

- **proposer**: Generates initial responses
- **critic**: Finds flaws and counterarguments
- **synthesizer**: Produces final consensus outputs

## Related Docs

- [Custom agents](../guides/CUSTOM_AGENTS.md)
- [Agent development](AGENT_DEVELOPMENT.md)
- [Agent selection](AGENT_SELECTION.md)
