# Python SDK Quickstart

Get started with the Aragora Python SDK in 5 minutes.

## Installation

```bash
pip install aragora-sdk
```

If you want to run a local server, install the full control plane package:

```bash
pip install aragora
```

Or install from source:

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora
pip install -e .
```

## Prerequisites

Start the Aragora server (self-hosted or use a hosted API). For local dev:

```bash
# Terminal 1: Start the server
python -m aragora.server.unified_server --port 8080
```

Set API keys for at least one provider:

```bash
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

## Basic Usage

All SDK calls are async. The examples below use `asyncio.run()`.

### 1. Create a Client

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        # Check server health
        health = await client.health()
        print(f"Server status: {health.status}")

asyncio.run(main())
```

### 2. Run a Debate

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        # Run a debate and wait for completion
        result = await client.debates.run(
            task="Should we use microservices or a monolith for our new project?",
            agents=["anthropic-api", "openai-api"],
            max_rounds=3,
        )

        if result.consensus:
            conclusion = result.consensus.conclusion or ""
            print(f"Consensus reached: {result.consensus.reached}")
            print(f"Confidence: {result.consensus.confidence:.1%}")
            print(f"Final answer: {conclusion[:500]}...")

asyncio.run(main())
```

### 3. Create and Poll a Debate

For more control, create a debate and poll for status:

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        # Create debate (returns immediately)
        created = await client.debates.create(
            task="What's the best database for a real-time analytics platform?",
            agents=["anthropic-api", "openai-api", "gemini"],
            max_rounds=2,
            consensus_threshold=0.6,
        )

        debate_id = created["id"]
        print(f"Debate ID: {debate_id}")

        # Poll for completion
        while True:
            status = await client.debates.get(debate_id)
            if status.status == "completed":
                print("Completed!")
                break
            await asyncio.sleep(2)

asyncio.run(main())
```

## Real-time Streaming

Stream debate events in real-time:

```python
import asyncio
from aragora_sdk import AragoraClient
from aragora_sdk.websocket import stream_debate

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        created = await client.debates.create(
            task="Design a caching strategy",
            agents=["anthropic-api", "openai-api"],
        )
        debate_id = created["id"]

    async for event in stream_debate("http://localhost:8080", debate_id):
        if event.type == "agent_message":
            print(event.data.get("content", ""))
        elif event.type == "consensus":
            print(f"Consensus: {event.data}")
        elif event.type == "debate_end":
            break

asyncio.run(main())
```

## Gauntlet: Adversarial Validation

Stress-test decisions with adversarial AI personas:

```python
import asyncio
from pathlib import Path
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        receipt = await client.gauntlet.run_and_wait(
            input_content=Path("policy.md").read_text(),
            input_type="policy",
            persona="gdpr",
        )

        print(f"Score: {receipt.score}")
        for finding in receipt.findings:
            print(f"  [{finding.severity}] {finding.description}")

asyncio.run(main())
```

## Agent Rankings

Query agent performance:

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        leaderboard = await client.agents.get_leaderboard()

        for agent in leaderboard[:10]:
            print(f"{agent.name}: {agent.elo_rating:.0f} ELO")

        # Get specific agent profile
        agent = await client.agents.get("anthropic-api")
        print(f"Agent: {agent.name}")
        print(f"Rating: {agent.elo_rating:.0f}")
        print(f"Win rate: {agent.win_rate:.1%}")

asyncio.run(main())
```

## Advanced Features

### Graph Debates (Branching)

Explore multiple solution paths:

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        created = await client.graph_debates.create(
            task="Design a notification system",
            max_rounds=3,
            agents=["anthropic-api", "openai-api"],
        )
        graph_id = created["id"]

        graph = await client.graph_debates.get(graph_id)
        for branch in graph.branches:
            print(f"Branch: {branch.approach} (divergence {branch.divergence_score:.2f})")

asyncio.run(main())
```

### Matrix Debates (Parallel Scenarios)

Test multiple scenarios in parallel:

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        created = await client.matrix_debates.create(
            task="Evaluate authentication approaches",
            scenarios=[
                {"name": "high_traffic", "parameters": {"context": "10M daily users"}},
                {"name": "regulated", "parameters": {"context": "HIPAA compliance required"}},
                {"name": "startup", "parameters": {"context": "Minimum viable product"}},
            ],
            agents=["anthropic-api", "openai-api"],
        )
        matrix_id = created["id"]

        conclusions = await client.matrix_debates.get_conclusions(matrix_id)
        print("Universal conclusions:", conclusions.universal)
        for scenario, points in conclusions.conditional.items():
            print(f"{scenario}: {points}")

asyncio.run(main())
```

### Formal Verification

Verify claims with formal methods:

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        result = await client.verification.verify(
            claim="The system handles all edge cases correctly",
            backend="z3",
        )

        print(f"Status: {result.status.value}")

asyncio.run(main())
```

## Error Handling

```python
import asyncio
from aragora_sdk import AragoraClient
from aragora_sdk.exceptions import (
    AragoraAuthenticationError,
    AragoraNotFoundError,
    AragoraTimeoutError,
    AragoraError,
)

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        try:
            await client.debates.get("invalid-id")
        except AragoraNotFoundError as e:
            print(f"Not found: {e}")
        except AragoraAuthenticationError:
            print("Invalid API token")
        except AragoraTimeoutError:
            print("Request timed out")
        except AragoraError as e:
            print(f"API error: {e.message} (status {e.status})")

asyncio.run(main())
```

## Configuration

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    client = AragoraClient(
        base_url="http://localhost:8080",
        api_key="your-api-token",  # Optional auth
        timeout=60.0,
        headers={"X-Custom-Header": "value"},
    )

    # Remember to close the client when not using a context manager
    await client.close()

asyncio.run(main())
```

## Next Steps

- [TypeScript SDK Quickstart](./typescript-quickstart.md)
- [API Reference](../API_REFERENCE.md)
- [Examples](../../examples/README.md)
- [Gauntlet Guide](../GAUNTLET.md)
