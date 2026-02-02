# Integrations Module

External integrations for Aragora, providing connectors to workflow automation platforms (Zapier, Make, n8n), LangChain integration, and a base framework for chat platform adapters.

## Overview

The integrations module enables Aragora to:

- **LangChain Integration**: Use Aragora as a tool, retriever, or chain component in LangChain applications
- **Workflow Automation**: Connect to Zapier, Make (Integromat), and n8n for automated workflows
- **Chat Platform Base**: Provide a foundation for chat platform connectors (Telegram, WhatsApp, Slack)
- **Data Formatting**: Standardize debate results, consensus data, and errors for external consumption

## Architecture

```
aragora/integrations/
├── __init__.py          # Module exports
├── base.py              # BaseIntegration for chat platforms
├── langchain.py         # LangChain tools, retrievers, and chains
├── zapier.py            # Zapier webhook integration
├── make.py              # Make (Integromat) integration
└── n8n.py               # n8n workflow integration
```

## Key Classes

### LangChain Integration

- **`AragoraTool`**: LangChain tool for running debates
- **`AragoraRetriever`**: LangChain retriever for querying Aragora knowledge
- **`AragoraCallbackHandler`**: LangChain callback handler for monitoring debate execution
- **`create_aragora_chain()`**: Helper to create a LangChain chain with Aragora integration

### Base Integration

- **`BaseIntegration`**: Abstract base class for chat platform integrations
  - Provides text formatting helpers (markdown, plain text)
  - Session management for multi-turn conversations
  - Rate limiting and throttling
  - Error handling and recovery

### Data Models

- **`FormattedDebateData`**: Standardized debate result format
- **`FormattedConsensusData`**: Standardized consensus data format
- **`FormattedErrorData`**: Standardized error format
- **`FormattedLeaderboardData`**: Standardized leaderboard format

## Usage Example

### LangChain Integration

```python
from langchain.agents import initialize_agent, AgentType
from langchain_anthropic import ChatAnthropic
from aragora.integrations.langchain import (
    AragoraTool,
    AragoraRetriever,
    AragoraCallbackHandler,
    create_aragora_chain,
)

# Create an Aragora tool for debates
aragora_tool = AragoraTool(
    name="debate",
    description="Run a multi-agent debate to analyze a question",
    api_url="http://localhost:8080",
    api_token="your-token",
)

# Create a retriever for knowledge queries
retriever = AragoraRetriever(
    api_url="http://localhost:8080",
    api_token="your-token",
    search_type="semantic",
    k=5,
)

# Use in a LangChain agent
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
agent = initialize_agent(
    tools=[aragora_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[AragoraCallbackHandler()],
)

# Run the agent
result = agent.run("What are the pros and cons of microservices?")

# Or use the chain helper
chain = create_aragora_chain(
    llm=llm,
    api_url="http://localhost:8080",
    api_token="your-token",
)
result = await chain.ainvoke({"question": "Should we use GraphQL or REST?"})
```

### Custom Chat Platform Integration

```python
from aragora.integrations.base import BaseIntegration

class DiscordIntegration(BaseIntegration):
    """Discord bot integration for Aragora."""

    def __init__(self, bot_token: str, api_url: str, api_token: str):
        super().__init__(api_url=api_url, api_token=api_token)
        self.bot_token = bot_token

    async def handle_message(self, message):
        # Use inherited session management
        session = self.get_or_create_session(message.channel.id)

        # Format response using inherited helpers
        if message.content.startswith("!debate"):
            topic = message.content[7:].strip()
            result = await self.run_debate(topic, session)

            # Use inherited formatting
            formatted = self.format_debate_result(result)
            await message.channel.send(formatted)

    def format_debate_result(self, result):
        # Use inherited markdown formatter
        return self.to_markdown(
            title="Debate Result",
            consensus=result.consensus,
            confidence=result.confidence,
            summary=result.final_answer,
        )
```

### Workflow Automation (Zapier/Make/n8n)

```python
from aragora.integrations.zapier import ZapierWebhook
from aragora.integrations.make import MakeScenario
from aragora.integrations.n8n import N8nWorkflow

# Zapier integration
zapier = ZapierWebhook(webhook_url="https://hooks.zapier.com/...")
await zapier.send_debate_result(debate_result)

# Make (Integromat) integration
make = MakeScenario(webhook_url="https://hook.eu1.make.com/...")
await make.trigger_scenario(
    event="debate_completed",
    data=debate_result.to_dict(),
)

# n8n integration
n8n = N8nWorkflow(
    base_url="https://your-n8n.com",
    workflow_id="abc123",
    api_key="your-api-key",
)
await n8n.execute_workflow(input_data={"topic": "AI safety"})
```

## Integration Points

### With Debate Engine
- `AragoraTool` wraps debate execution for LangChain agents
- Webhook integrations send debate results to automation platforms
- Session management tracks multi-turn debate conversations

### With Knowledge Mound
- `AragoraRetriever` queries knowledge for RAG applications
- Supports semantic and keyword search modes
- Configurable result count and filtering

### With Chat Connectors
- `BaseIntegration` provides foundation for Telegram, WhatsApp, Slack connectors
- Standardized formatting ensures consistent user experience
- Rate limiting protects against abuse

### With External Workflows
- Zapier triggers for "new debate" and "consensus reached" events
- Make scenarios for complex multi-step automations
- n8n workflows for self-hosted automation

## Data Formats

### FormattedDebateData

```python
{
    "id": "debate-123",
    "topic": "Should we adopt microservices?",
    "consensus_reached": True,
    "confidence": 0.85,
    "final_answer": "Microservices are recommended for...",
    "rounds": 3,
    "participants": ["claude", "gpt4", "gemini"],
    "duration_seconds": 45.2,
    "created_at": "2024-01-15T10:30:00Z",
}
```

### FormattedConsensusData

```python
{
    "reached": True,
    "confidence": 0.85,
    "method": "supermajority",
    "votes": {"accept": 4, "reject": 1},
    "reasoning": "Strong agreement on core points...",
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_API_URL` | Aragora API base URL | `http://localhost:8080` |
| `ARAGORA_API_TOKEN` | API authentication token | - |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `false` |
| `ZAPIER_WEBHOOK_URL` | Zapier webhook URL | - |
| `MAKE_WEBHOOK_URL` | Make webhook URL | - |
| `N8N_BASE_URL` | n8n instance URL | - |

## See Also

- `aragora/connectors/chat/` - Chat platform connectors (Telegram, WhatsApp)
- `aragora/connectors/slack.py` - Slack connector
- `aragora/connectors/github.py` - GitHub connector
- `docs/INTEGRATIONS.md` - Full integration guide
