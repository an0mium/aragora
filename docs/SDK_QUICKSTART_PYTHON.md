# Python SDK Quickstart

Get started with Aragora in under 5 minutes.

## Installation

```bash
pip install aragora-client
```

## Basic Usage

```python
import asyncio
from aragora_client import AragoraClient

async def main():
    async with AragoraClient("http://localhost:8080", api_key="your-key") as client:
        # Run a debate
        result = await client.debates.run(
            task="Should we use microservices or monolith?",
            agents=["anthropic-api", "openai-api"],
        )
        print(f"Conclusion: {result.consensus.conclusion}")

asyncio.run(main())
```

## Full Example

```python
import asyncio
from aragora_client import AragoraClient

async def main():
    client = AragoraClient(
        base_url="http://localhost:8080",
        api_key="your-api-key",  # optional for local dev
    )

    # 1. Create a debate
    debate = await client.debates.create(
        task="Design a rate limiter for our API",
        agents=["anthropic-api", "openai-api", "gemini-api"],
        max_rounds=3,
    )
    print(f"Created debate: {debate.debate_id}")

    # 2. Wait for completion
    result = await client.debates.run(
        task="What caching strategy should we use?",
        timeout=120.0,
    )

    # 3. Access results
    print(f"Status: {result.status}")
    print(f"Consensus: {result.consensus.conclusion}")
    print(f"Confidence: {result.consensus.confidence}")

    # 4. Get decision receipt
    receipt = await client.decisions.get(result.debate_id)
    print(f"Receipt ID: {receipt.receipt_id}")

asyncio.run(main())
```

## Key APIs

| API | Description |
|-----|-------------|
| `client.debates.run()` | Run debate and wait for result |
| `client.debates.create()` | Create debate (non-blocking) |
| `client.debates.get(id)` | Get debate details |
| `client.decisions.get(id)` | Get decision receipt |
| `client.agents.list()` | List available agents |
| `client.health.check()` | Check API health |

## Environment Variables

```bash
export ARAGORA_API_URL="http://localhost:8080"
export ARAGORA_API_KEY="your-api-key"
```

```python
import os
from aragora_client import AragoraClient

client = AragoraClient(
    base_url=os.getenv("ARAGORA_API_URL"),
    api_key=os.getenv("ARAGORA_API_KEY"),
)
```

## Next Steps

- [Full API Reference](https://docs.aragora.ai/sdk/python)
- [Graph Debates](https://docs.aragora.ai/features/graph-debates)
- [Workflows](https://docs.aragora.ai/features/workflows)
- [WebSocket Streaming](https://docs.aragora.ai/features/streaming)
