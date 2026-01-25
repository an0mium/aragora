# Python SDK Quickstart

Get started with the Aragora Python SDK in 5 minutes.

## Installation

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

Start the Aragora server:

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

### 1. Create a Client

```python
from aragora.client import AragoraClient

# Connect to local server
client = AragoraClient(base_url="http://localhost:8080")

# Check server health
health = client.system.health()
print(f"Server status: {health.status}")
```

### 2. Run a Debate

```python
# Run a debate and wait for completion
result = client.debates.run(
    task="Should we use microservices or a monolith for our new project?",
    agents=["anthropic-api", "openai-api"],
    rounds=3,
)

print(f"Consensus reached: {result.consensus.reached}")
print(f"Confidence: {result.consensus.confidence:.1%}")
print(f"Final answer: {result.consensus.final_answer[:500]}...")
```

### 3. Create and Poll a Debate

For more control, create a debate and poll for status:

```python
# Create debate (returns immediately)
debate = client.debates.create(
    task="What's the best database for a real-time analytics platform?",
    agents=["anthropic-api", "openai-api", "gemini"],
    rounds=2,
    consensus="majority",
)

print(f"Debate ID: {debate.debate_id}")
print(f"Status: {debate.status}")

# Poll for completion
import time
while True:
    status = client.debates.get(debate.debate_id)
    if status.status == "completed":
        print(f"Completed! Consensus: {status.consensus.reached}")
        break
    time.sleep(2)
```

## Async Usage

For async applications:

```python
import asyncio
from aragora.client import AragoraClient

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        # Health check
        health = await client.system.health_async()
        print(f"Status: {health.status}")

        # Run debate asynchronously
        result = await client.debates.run_async(
            task="React vs Vue vs Svelte?",
            agents=["anthropic-api", "gemini"],
        )
        print(result.consensus.final_answer)

asyncio.run(main())
```

## Real-time Streaming

Stream debate events in real-time:

```python
from aragora.client import stream_debate

# Stream events as they happen
async for event in stream_debate(
    base_url="http://localhost:8080",
    task="Design a caching strategy",
    agents=["anthropic-api", "openai-api"],
):
    if event.type == "agent_message":
        print(f"[{event.agent}]: {event.content[:100]}...")
    elif event.type == "consensus":
        print(f"Consensus reached: {event.data}")
```

## Gauntlet: Adversarial Validation

Stress-test decisions with adversarial AI personas:

```python
from pathlib import Path

# Validate a policy document
receipt = client.gauntlet.run_and_wait(
    input_content=Path("policy.md").read_text(),
    input_type="policy",
    persona="gdpr",
    profile="thorough",
)

print(f"Verdict: {receipt.verdict}")
print(f"Risk Score: {receipt.risk_score}")

for finding in receipt.findings:
    print(f"  [{finding.severity}] {finding.title}")
```

## Agent Rankings

Query agent performance:

```python
# Get agent leaderboard
rankings = client.leaderboard.list(limit=10)

for i, agent in enumerate(rankings, 1):
    print(f"{i}. {agent.name}: {agent.elo:.0f} ELO")

# Get specific agent profile
agent = client.agents.get("anthropic-api")
print(f"Agent: {agent.name}")
print(f"Rating: {agent.elo}")
print(f"Win rate: {agent.win_rate:.1%}")
```

## Advanced Features

### Graph Debates (Branching)

Explore multiple solution paths:

```python
# Create a graph debate
graph = client.graph_debates.create(
    root_topic="Design a notification system",
    branch_depth=3,
    agents=["anthropic-api", "openai-api"],
)

# Explore branches
for branch in graph.branches:
    print(f"Branch: {branch.topic}")
    print(f"  Path: {' -> '.join(branch.path)}")
```

### Matrix Debates (Parallel Scenarios)

Test multiple scenarios in parallel:

```python
# Create a matrix debate
matrix = client.matrix_debates.create(
    base_topic="Evaluate authentication approaches",
    scenarios=[
        {"name": "high_traffic", "context": "10M daily users"},
        {"name": "regulated", "context": "HIPAA compliance required"},
        {"name": "startup", "context": "Minimum viable product"},
    ],
    agents=["anthropic-api", "openai-api"],
)

# Get results for each scenario
for scenario in matrix.scenarios:
    print(f"{scenario.name}: {scenario.recommendation}")
```

### Formal Verification

Verify claims with formal methods:

```python
# Verify a claim
result = client.verification.verify(
    claim="The system handles all edge cases correctly",
    context="Based on the test coverage report...",
    backend="z3",  # or "lean"
)

print(f"Status: {result.status}")
print(f"Verified: {result.verified}")
```

## Error Handling

```python
from aragora.client import (
    AragoraAPIError,
    RateLimitError,
    AuthenticationError,
    NotFoundError,
)

try:
    result = client.debates.get("invalid-id")
except NotFoundError:
    print("Debate not found")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except AuthenticationError:
    print("Invalid API token")
except AragoraAPIError as e:
    print(f"API error: {e.message}")
```

## Configuration

```python
from aragora.client import AragoraClient, RetryConfig

# Configure retries and rate limiting
client = AragoraClient(
    base_url="http://localhost:8080",
    api_token="your-token",  # Optional auth
    retry_config=RetryConfig(
        max_retries=3,
        backoff_factor=0.5,
    ),
    rate_limit_rps=10,
    timeout=60,
)
```

## Next Steps

- [TypeScript SDK Quickstart](./typescript-quickstart.md)
- [API Reference](../API_REFERENCE.md)
- [Examples](../../examples/README.md)
- [Gauntlet Guide](../GAUNTLET.md)
