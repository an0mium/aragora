# aragora-client

Python SDK for the Aragora control plane for multi-agent deliberation across organizational knowledge and channels.

## Installation

```bash
pip install aragora-client
```

## Quick Start

```python
import asyncio
from aragora_client import AragoraClient

async def main():
    client = AragoraClient("http://localhost:8080")

    # Run a debate
    debate = await client.debates.run(
        task="Should we use microservices?",
        agents=["anthropic-api", "openai-api"],
    )

    print(f"Consensus: {debate.consensus.conclusion}")

asyncio.run(main())
```

## API Reference

### Client Initialization

```python
from aragora_client import AragoraClient

client = AragoraClient(
    base_url="http://localhost:8080",
    api_key="your-api-key",      # optional
    timeout=30.0,                 # optional, default 30s
    headers={"X-Custom": "value"} # optional
)

# Use as context manager for automatic cleanup
async with AragoraClient("http://localhost:8080") as client:
    debate = await client.debates.run(task="...")
```

### Debates API

```python
# Create a debate
response = await client.debates.create(
    task="Design a rate limiter",
    agents=["anthropic-api", "openai-api"],
    max_rounds=5,
    consensus_threshold=0.8,
)

# Get debate details
debate = await client.debates.get("debate-123")

# List debates
debates = await client.debates.list(limit=10, status="completed")

# Run debate and wait for completion
result = await client.debates.run(
    task="Should we use TypeScript?",
    timeout=300.0,  # optional timeout
)
```

### Graph Debates API

Graph debates support automatic branching when agents identify different approaches.

```python
# Create graph debate
response = await client.graph_debates.create(
    task="Design a distributed system",
    agents=["anthropic-api", "openai-api"],
    max_rounds=5,
    branch_threshold=0.5,
    max_branches=10,
)

# Get debate with branches
debate = await client.graph_debates.get("debate-123")

# Get branches
branches = await client.graph_debates.get_branches("debate-123")
```

### Matrix Debates API

Matrix debates run the same question across different scenarios.

```python
# Create matrix debate
response = await client.matrix_debates.create(
    task="Should we adopt microservices?",
    scenarios=[
        {"name": "small_team", "parameters": {"team_size": 5}},
        {"name": "large_team", "parameters": {"team_size": 50}},
        {"name": "high_traffic", "parameters": {"rps": 100000}, "is_baseline": True},
    ],
    max_rounds=3,
)

# Get conclusions
conclusions = await client.matrix_debates.get_conclusions("matrix-123")
print(f"Universal: {conclusions.universal}")
print(f"Conditional: {conclusions.conditional}")
```

### Verification API

Formal verification of claims using Z3 or Lean 4.

```python
# Verify a claim
result = await client.verification.verify(
    claim="All primes > 2 are odd",
    backend="z3",  # "z3" | "lean"
    timeout=30,
)

if result.status == "valid":
    print("Claim is valid!")
    print(f"Formal translation: {result.formal_translation}")

# Check backend status
status = await client.verification.status()
```

### Agents API

```python
# List available agents
agents = await client.agents.list()

# Get agent profile
agent = await client.agents.get("anthropic-api")
print(f"ELO rating: {agent.elo_rating}")

# Get match history
history = await client.agents.history("anthropic-api", limit=20)

# Get rivals and allies
rivals = await client.agents.rivals("anthropic-api")
allies = await client.agents.allies("anthropic-api")
```

### Gauntlet API

Adversarial validation of specifications.

```python
# Run gauntlet
response = await client.gauntlet.run(
    input_content="Your spec content here...",
    input_type="spec",
    persona="security",
)

# Get receipt
receipt = await client.gauntlet.get_receipt(response["gauntlet_id"])
print(f"Score: {receipt.score}")
print(f"Findings: {receipt.findings}")

# Run and wait for completion
result = await client.gauntlet.run_and_wait(
    input_content=spec_content,
    persona="devil_advocate",
)
```

### Selection API

Agent selection plugins for team building.

```python
# List available plugins
plugins = await client.selection.list_plugins()
print(f"Scorers: {[s.name for s in plugins.scorers]}")
print(f"Team Selectors: {[t.name for t in plugins.team_selectors]}")

# Score agents for a task
scores = await client.selection.score_agents(
    task_description="Design a distributed cache system",
    primary_domain="systems",
    scorer="elo_weighted",
)

for agent in scores:
    print(f"{agent.name}: {agent.score}")

# Select an optimal team
team = await client.selection.select_team(
    task_description="Build a secure authentication system",
    min_agents=3,
    max_agents=5,
    diversity_preference=0.7,
    quality_priority=0.8,
)

print(f"Team: {[f'{a.name} ({a.role})' for a in team.agents]}")
print(f"Expected quality: {team.expected_quality}")
```

### Memory API

```python
# Get analytics
analytics = await client.memory.analytics(days=30)
print(f"Total entries: {analytics.total_entries}")
print(f"Learning velocity: {analytics.learning_velocity}")

# Get tier-specific stats
fast_tier = await client.memory.tier_stats("fast")

# Take manual snapshot
snapshot = await client.memory.snapshot()
```

### Health Check

```python
health = await client.health()
print(f"Status: {health.status}")
print(f"Version: {health.version}")
```

## WebSocket Streaming

Stream debate events in real-time.

### Class-based API

```python
from aragora_client import DebateStream

debate_id = "debate-123"
stream = DebateStream("ws://localhost:8765", debate_id)

stream.on("agent_message", lambda e: print(f"Agent: {e.data}"))
stream.on("consensus", lambda e: print("Consensus reached!"))
stream.on("debate_end", lambda e: stream.disconnect())
stream.on_error(lambda e: print(f"Error: {e}"))

await stream.connect()
```

### Async Iterator API

```python
from aragora_client import stream_debate

async for event in stream_debate("ws://localhost:8765", "debate-123"):
    print(event.type, event.data)

    if event.type == "debate_end":
        break
```

### WebSocket Options

```python
stream = DebateStream(
    "ws://localhost:8765",
    "debate-123",
    reconnect=True,             # Auto-reconnect on disconnect
    reconnect_interval=1.0,     # Base reconnect delay (seconds)
    max_reconnect_attempts=5,   # Max reconnect attempts
    heartbeat_interval=30.0,    # Heartbeat ping interval (seconds)
)
```

## Error Handling

```python
from aragora_client import (
    AragoraError,
    AragoraConnectionError,
    AragoraAuthenticationError,
    AragoraNotFoundError,
    AragoraValidationError,
    AragoraTimeoutError,
)

try:
    await client.debates.get("nonexistent-123")
except AragoraNotFoundError as e:
    print(f"Resource: {e.resource}")
    print(f"ID: {e.resource_id}")
except AragoraError as e:
    print(f"Code: {e.code}")
    print(f"Status: {e.status}")
    print(f"Message: {e.message}")
    print(f"Details: {e.details}")
```

## Type Hints

All types are exported for use in your application:

```python
from aragora_client import (
    # Debate types
    Debate,
    DebateStatus,
    ConsensusResult,
    GraphDebate,
    GraphBranch,
    MatrixDebate,
    MatrixConclusion,
    # Verification types
    VerificationResult,
    VerificationStatus,
    # Agent types
    AgentProfile,
    GauntletReceipt,
    # Event types
    DebateEvent,
    # Selection types
    SelectionPlugins,
    TeamSelection,
    AgentScore,
)
```

## Advanced Patterns

### Retry with Exponential Backoff

```python
from aragora_client import AragoraClient, AragoraError

async def with_retry(fn, max_retries=3, base_delay=1.0):
    last_error = None

    for attempt in range(max_retries):
        try:
            return await fn()
        except AragoraError as e:
            last_error = e
            # Don't retry client errors (4xx)
            if e.status and e.status < 500:
                raise
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

    raise last_error

# Usage
debate = await with_retry(
    lambda: client.debates.run(task="Design a system")
)
```

### Concurrent Debates with Semaphore

```python
import asyncio
from aragora_client import AragoraClient

async def run_debates_concurrent(tasks: list[str], max_concurrent: int = 3):
    client = AragoraClient("http://localhost:8080")
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(task: str):
        async with semaphore:
            return await client.debates.run(task=task)

    return await asyncio.gather(*[run_with_semaphore(t) for t in tasks])

# Run multiple debates with controlled concurrency
tasks = ["Design auth system", "Choose database", "API architecture"]
debates = await run_debates_concurrent(tasks, max_concurrent=2)
```

## Requirements

- Python 3.10+
- httpx >= 0.25.0
- websockets >= 12.0
- pydantic >= 2.0.0

## License

MIT
