# Python SDK Quickstart

> **Note:** For the comprehensive Python SDK guide with advanced features and streaming, see **[Python Quickstart Guide](guides/python-quickstart.md)**.
>
> This page provides a minimal 5-minute quickstart using the `aragora-sdk` package.

Get started with Aragora in under 5 minutes.

## Installation

```bash
pip install aragora-sdk
```

## Basic Usage

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient("http://localhost:8080", api_key="your-key") as client:
        # Run a debate
        result = await client.debates.run(
            task="Should we use microservices or monolith?",
            agents=["anthropic-api", "openai-api"],
        )
        if result.consensus:
            print(f"Conclusion: {result.consensus.conclusion}")

asyncio.run(main())
```

## Full Example

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient(
        base_url="http://localhost:8080",
        api_key="your-api-key",  # optional for local dev
    ) as client:
        # 1. Create a debate
        created = await client.debates.create(
            task="Design a rate limiter for our API",
            agents=["anthropic-api", "openai-api", "gemini-api"],
            max_rounds=3,
        )
        debate_id = created["id"]
        print(f"Created debate: {debate_id}")

        # 2. Wait for completion
        result = await client.debates.run(
            task="What caching strategy should we use?",
            timeout=120.0,
        )

        # 3. Access results
        if result.consensus:
            print(f"Status: {result.status}")
            print(f"Consensus: {result.consensus.conclusion}")
            print(f"Confidence: {result.consensus.confidence}")

        # 4. Fetch debate details
        debate = await client.debates.get(debate_id)
        print(f"Debate task: {debate.task}")

asyncio.run(main())
```

## Key APIs

| API | Description |
|-----|-------------|
| `client.debates.run()` | Run debate and wait for result |
| `client.debates.create()` | Create debate (non-blocking) |
| `client.debates.get(id)` | Get debate details |
| `client.agents.list()` | List available agents |
| `await client.health()` | Check API health |
