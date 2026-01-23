# Agent Catalog (Overview)

Aragora supports multiple agent backends for robust decisionmaking. This page summarizes
the agent families and how to select them. For the full up-to-date catalog,
see `AGENTS.md` at the repository root.

## Agent Families

### CLI Agents

Use external CLI tools with local credentials:

- `claude`, `codex`, `openai`, `gemini-cli`, `grok-cli`, `qwen-cli`,
  `deepseek-cli`, `kilocode`

### Direct API Agents

Call provider APIs directly:

- `anthropic-api`, `openai-api`, `gemini`, `grok`, `mistral-api`, `codestral`,
  `ollama`, `kimi`

### OpenRouter Agents

OpenRouter-backed unified access:

- `deepseek`, `deepseek-r1`, `llama`, `mistral`, `qwen`, `qwen-max`, `yi`

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

- [Custom agents](CUSTOM_AGENTS.md)
- [Agent development](AGENT_DEVELOPMENT.md)
- [Agent selection](AGENT_SELECTION.md)
