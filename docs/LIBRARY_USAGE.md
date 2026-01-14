# Programmatic API Usage Guide

This guide covers using Aragora as a Python library for direct integration into your applications, without the HTTP server or CLI.

## Quick Reference

| Use Case | Module | Primary Class |
|----------|--------|---------------|
| Run debates | `aragora.debate` | `Arena` |
| Create agents | `aragora.agents` | `Agent`, API-specific classes |
| Memory systems | `aragora.memory` | `ContinuumMemory`, `ConsensusMemory` |
| Event handling | `aragora.spectate` | `SpectatorStream` |
| Gauntlet testing | `aragora.gauntlet` | `GauntletRunner` |

---

## Basic Usage

### Running a Simple Debate

```python
import asyncio
from aragora.core import Environment
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol
from aragora.agents.api_agents import AnthropicAPIAgent, OpenAIAPIAgent

async def run_debate():
    # Create environment with the task
    env = Environment(task="Should we use microservices or a monolith?")

    # Create agents
    agents = [
        AnthropicAPIAgent(name="anthropic-api"),
        OpenAIAPIAgent(name="openai-api"),
    ]

    # Configure debate protocol
    protocol = DebateProtocol(
        rounds=3,
        consensus="majority",  # "unanimous", "majority", "supermajority", "hybrid"
    )

    # Create and run arena
    arena = Arena(env, agents, protocol)
    result = await arena.run()

    # Access results
    print(f"Consensus reached: {result.consensus_reached}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Final answer: {result.final_answer}")

    return result

# Run
result = asyncio.run(run_debate())
```

### Using Pre-configured Agent Sets

```python
from aragora.agents.base import create_agent, list_available_agents

# List available agents (returns metadata by type)
available = list_available_agents()
# list(available.keys()) -> ['anthropic-api', 'openai-api', 'gemini', 'grok', ...]

# Create agents by type
anthropic_agent = create_agent("anthropic-api")
openai_agent = create_agent("openai-api")

# With custom model selection
anthropic_custom = create_agent(
    "anthropic-api",
    model="claude-opus-4-5-20251101",
)
anthropic_custom.set_generation_params(temperature=0.7)
```

---

## Creating Custom Agents

### Basic Custom Agent

```python
from aragora.agents.registry import AgentRegistry
from aragora.core import Agent, Critique, Message

@AgentRegistry.register("my-custom", default_model="custom", agent_type="Custom")
class MyCustomAgent(Agent):
    """A custom agent implementation."""

    def __init__(self, name: str = "my-custom", model: str = "custom", role: str = "proposer"):
        super().__init__(name=name, model=model, role=role)

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response to the prompt."""
        return self._call_your_api(prompt)

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
    ) -> Critique:
        """Critique another agent's proposal."""
        analysis = self._analyze_proposal(proposal, task)
        return Critique(
            agent=self.name,
            target_agent="proposal",
            target_content=proposal[:200],
            issues=[analysis],
            suggestions=[],
            severity=0.5,
            reasoning="Heuristic critique",
        )
```

### Registering Custom Agents

```python
from aragora.agents.base import create_agent

# Create via the registry after registration
agent = create_agent("my-custom")
```

---

## Memory Systems

### Continuum Memory (Multi-tier)

```python
from aragora.memory.continuum import ContinuumMemory, MemoryTier

# Initialize memory
memory = ContinuumMemory(db_path=".nomic/continuum.db")

# Store with tier
memory.store(
    content="Important insight from debate",
    tier=MemoryTier.MEDIUM,  # FAST, MEDIUM, SLOW, GLACIAL
    metadata={"debate_id": "abc123", "topic": "architecture"},
)

# Retrieve by relevance
relevant = memory.retrieve(
    query="microservices vs monolith",
    limit=5,
    min_tier=MemoryTier.MEDIUM,
)

# Memory tiers and their retention
# FAST:    1 minute  - immediate context
# MEDIUM:  1 hour    - session memory
# SLOW:    1 day     - cross-session learning
# GLACIAL: 1 week    - long-term patterns
```

### Consensus Memory

```python
from aragora.memory.consensus import ConsensusMemory, ConsensusRecord

# Initialize
consensus_mem = ConsensusMemory(db_path=".nomic/consensus.db")

# Store debate outcome
record = ConsensusRecord(
    topic="Rate limiter design",
    conclusion="Token bucket algorithm is preferred",
    confidence=0.85,
    participating_agents=["anthropic-api", "openai-api"],
    strength="strong",
)
consensus_mem.store(record)

# Find similar past debates
similar = consensus_mem.find_similar_debates(
    topic="API throttling strategy",
    limit=5,
)

# Check if topic is settled
is_settled = consensus_mem.is_topic_settled(
    topic="Rate limiting approach",
    min_confidence=0.8,
)
```

---

## Event Handling

### Subscribing to Debate Events

```python
from aragora.spectate.stream import SpectatorStream

async def run_with_events():
    spectator = SpectatorStream()

    # Subscribe to events
    @spectator.on("round_start")
    async def on_round(event):
        print(f"Round {event.round_number} starting...")

    @spectator.on("agent_message")
    async def on_message(event):
        print(f"{event.agent}: {event.content[:100]}...")

    @spectator.on("critique")
    async def on_critique(event):
        print(f"{event.agent} critiques {event.target}: {event.vote}")

    @spectator.on("consensus")
    async def on_consensus(event):
        print(f"Consensus: {event.reached} ({event.confidence:.2%})")

    # Run debate with spectator
    arena = Arena(env, agents, protocol, spectator=spectator)
    result = await arena.run()

    return result
```

### Available Event Types

| Event | Data Fields |
|-------|-------------|
| `debate_start` | `task`, `agents`, `protocol` |
| `round_start` | `round_number`, `total_rounds` |
| `agent_message` | `agent`, `content`, `role` |
| `critique` | `agent`, `target`, `content`, `vote` |
| `vote` | `agent`, `vote`, `confidence` |
| `consensus` | `reached`, `confidence`, `final_answer` |
| `debate_end` | `result`, `duration_seconds` |

---

## Gauntlet (Adversarial Testing)

### Running Gauntlet Validation

```python
from aragora.gauntlet import GauntletRunner, GauntletConfig, AttackCategory

async def stress_test_document():
    config = GauntletConfig(
        attack_categories=[
            AttackCategory.SECURITY,
            AttackCategory.COMPLIANCE,
            AttackCategory.SCALABILITY,
        ],
        agents=["anthropic-api", "openai-api", "gemini"],
        rounds_per_attack=2,
    )

    runner = GauntletRunner(config)

    # Test a document/spec
    with open("my_spec.md") as f:
        content = f.read()

    result = await runner.run(content)

    # Get decision receipt
    receipt = result.to_receipt()
    print(f"Overall verdict: {receipt.verdict}")
    print(f"Risk score: {receipt.risk_score}/100")

    # Check vulnerabilities
    for vuln in result.vulnerabilities:
        print(f"[{vuln.severity}] {vuln.category}: {vuln.description}")

    return result
```

### Using Preset Gauntlets

```python
from aragora.gauntlet import (
    GDPR_GAUNTLET,
    HIPAA_GAUNTLET,
    SECURITY_GAUNTLET,
    CODE_REVIEW_GAUNTLET,
)

# Use a compliance preset
runner = GauntletRunner(GDPR_GAUNTLET)
result = await runner.run(privacy_policy_text)
```

---

## Advanced Configuration

### Arena Configuration Options

```python
from aragora.debate.orchestrator import Arena, ArenaConfig
from aragora.ranking.elo import EloSystem
from aragora.memory.continuum import ContinuumMemory

# Full configuration
config = ArenaConfig(
    # Ranking
    elo_system=EloSystem(db_path=".nomic/elo.db"),

    # Memory
    continuum=ContinuumMemory(db_path=".nomic/continuum.db"),

    # Consensus memory for historical context
    consensus_memory=ConsensusMemory(db_path=".nomic/consensus.db"),

    # Role rotation (agents take turns as proposer/critic)
    role_rotation=RoleRotationConfig(
        enabled=True,
        rotate_every_round=True,
    ),

    # Convergence detection (early stop if agents agree)
    convergence_threshold=0.9,

    # Circuit breaker (handle failing agents)
    circuit_breaker_enabled=True,
    failure_threshold=3,
)

arena = Arena(env, agents, protocol, config=config)
```

### Protocol Customization

```python
from aragora.debate.protocol import DebateProtocol

protocol = DebateProtocol(
    rounds=5,
    consensus="hybrid",  # Combines voting with semantic similarity

    # Early stopping
    early_stop_threshold=0.95,  # Stop if confidence exceeds this
    min_rounds=2,  # But run at least this many rounds

    # Critique settings
    require_justification=True,
    min_critique_length=50,

    # Timeout
    round_timeout_seconds=120,
    total_timeout_seconds=600,
)
```

---

## Error Handling

### Handling Agent Failures

```python
from aragora.agents.errors import AgentError, RateLimitError, APIKeyError

async def robust_debate():
    try:
        result = await arena.run()
    except RateLimitError as e:
        print(f"Rate limited by {e.provider}, retrying in {e.retry_after}s")
        await asyncio.sleep(e.retry_after)
        result = await arena.run()
    except APIKeyError as e:
        print(f"Invalid API key for {e.provider}")
        raise
    except AgentError as e:
        print(f"Agent {e.agent_name} failed: {e.message}")
        # Arena's circuit breaker may have already handled this
        if arena.circuit_breaker.is_open(e.agent_name):
            print(f"Agent {e.agent_name} disabled due to repeated failures")
```

### Circuit Breaker Pattern

```python
from aragora.debate.protocol import CircuitBreaker

# Arena uses circuit breaker internally
# Access it to check agent health
breaker = arena.circuit_breaker

# Check if agent is healthy
if not breaker.is_open("openai-api"):
    # Agent is available
    pass

# Manually reset a tripped breaker
breaker.reset("openai-api")
```

---

## Storage Backend Selection

### Using PostgreSQL for Production

```python
import os

# Set environment variables
os.environ["ARAGORA_DB_BACKEND"] = "postgresql"
os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/aragora"

# Or configure in code
from aragora.storage.backends import get_database_backend, PostgreSQLBackend

# Get configured backend
db = get_database_backend()

# Or explicitly create PostgreSQL backend
db = PostgreSQLBackend(
    database_url="postgresql://user:pass@localhost:5432/aragora",
    pool_size=5,
    pool_max_overflow=10,
)
```

---

## Integration Patterns

### FastAPI Integration

```python
from fastapi import FastAPI, BackgroundTasks
from aragora.debate.orchestrator import Arena
from aragora.core import Environment

app = FastAPI()

@app.post("/debates")
async def create_debate(task: str, background_tasks: BackgroundTasks):
    """Start a debate and return immediately."""
    debate_id = generate_id()

    async def run_debate():
        env = Environment(task=task)
        arena = Arena(env, agents, protocol)
        result = await arena.run()
        # Store result
        await store_result(debate_id, result)

    background_tasks.add_task(run_debate)
    return {"debate_id": debate_id, "status": "started"}

@app.get("/debates/{debate_id}")
async def get_debate(debate_id: str):
    """Get debate result."""
    result = await get_stored_result(debate_id)
    return result
```

### WebSocket Streaming

```python
import asyncio
from fastapi import WebSocket
from aragora.server.stream import SyncEventEmitter
from aragora.server.stream.arena_hooks import create_arena_hooks

@app.websocket("/ws/debates/{debate_id}")
async def debate_stream(websocket: WebSocket, debate_id: str):
    await websocket.accept()

    emitter = SyncEventEmitter(loop_id=debate_id)
    hooks = create_arena_hooks(emitter)

    def forward_event(event):
        asyncio.create_task(websocket.send_json(event.to_dict()))

    emitter.subscribe(forward_event)
    arena = Arena(env, agents, protocol, event_hooks=hooks, event_emitter=emitter, loop_id=debate_id)
    result = await arena.run()

    await websocket.send_json({"type": "debate_end", "data": result.to_dict()})
    await websocket.close()
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [CUSTOM_AGENTS.md](CUSTOM_AGENTS.md) - Detailed agent customization
- [API_REFERENCE.md](API_REFERENCE.md) - Full API documentation
- [GETTING_STARTED.md](GETTING_STARTED.md) - CLI and server usage
