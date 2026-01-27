# Aragora Python SDK

The Aragora Python SDK provides a type-safe interface for interacting with the Aragora API.
Prefer `/api/v1` endpoints for SDK usage; unversioned `/api` endpoints remain supported but are deprecated for SDK clients.

## Package Options

- `aragora` - Full control plane package with sync + async SDK (server + CLI + client).
- `aragora-client` - Lightweight, async-only SDK for remote API use (`/api/v1`).
- `aragora-sdk` - Deprecated; use `aragora-client` instead.

## Installation

```bash
pip install aragora-client
```

Or install the full control plane package (includes the SDK and server):

```bash
pip install aragora
```

Or from source:

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora
pip install -e .
```

## Quick Start

### Standalone SDK (aragora-client)

```python
import asyncio
from aragora_client import AragoraClient

async def main():
    client = AragoraClient("http://localhost:8080")
    debate = await client.debates.run(
        task="Should we use microservices?",
        agents=["anthropic-api", "openai-api"],
    )
    print(f"Consensus: {debate.consensus.conclusion}")

asyncio.run(main())
```

### Full SDK (aragora) - Synchronous Usage

```python
from aragora.client import AragoraClient

# Create client
client = AragoraClient(base_url="http://localhost:8080")

# Create a debate
response = client.debates.create(task="Should we use microservices?")
print(f"Debate started: {response.debate_id}")

# Get debate result
debate = client.debates.get(response.debate_id)
print(f"Status: {debate.status}")
```

### Full SDK (aragora) - Asynchronous Usage

```python
import asyncio
from aragora.client import AragoraClient

async def main():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        # Create and wait for debate completion
        debate = await client.debates.run_async(
            task="Design a distributed cache",
            timeout=600,  # 10 minutes
        )
        print(f"Consensus: {debate.consensus}")

asyncio.run(main())
```

## Configuration

```python
client = AragoraClient(
    base_url="http://localhost:8080",  # API server URL
    api_key="your-api-key",            # Optional authentication
    timeout=60,                         # Request timeout in seconds
)
```

## Core APIs

### Debates

Standard debates with propose-critique-revise workflow.

```python
# Create a debate
response = client.debates.create(
    task="Should we adopt Kubernetes?",
    agents=["anthropic-api", "openai-api", "gemini"],
    rounds=3,
    consensus="majority",  # unanimous, majority, supermajority, hybrid
    context="For a startup with 5 engineers",
)

# Get debate details
debate = client.debates.get(response.debate_id)

# List recent debates
debates = client.debates.list(limit=20, status="completed")

# Create and wait for completion (blocking)
debate = client.debates.run(
    task="Design a rate limiter",
    timeout=600,
)
```

### Gauntlet (Adversarial Validation)

Stress-test specifications, policies, and architectures.

```python
# Start a gauntlet analysis
response = client.gauntlet.run(
    input_content="Your specification or policy text...",
    input_type="text",     # text, policy, code
    persona="security",    # security, gdpr, hipaa, ai_act, soc2
    profile="default",     # quick, default, thorough
)

# Get decision receipt
receipt = client.gauntlet.get_receipt(response.gauntlet_id)
print(f"Verdict: {receipt.verdict}")
print(f"Risk Score: {receipt.risk_score:.0%}")

for finding in receipt.findings:
    print(f"- [{finding.severity}] {finding.title}")

# Run and wait for completion
receipt = client.gauntlet.run_and_wait(
    input_content="...",
    timeout=900,
)
```

### Graph Debates (Branching Discussions)

Graph debates allow automatic branching when agents identify fundamentally different approaches.

```python
# Create graph debate
response = client.graph_debates.create(
    task="Design a distributed system architecture",
    agents=["anthropic-api", "openai-api"],
    max_rounds=5,
    branch_threshold=0.5,  # Divergence threshold for branching
    max_branches=5,
)

# Get debate with all branches
debate = client.graph_debates.get(response.debate_id)

# Get branches separately
branches = client.graph_debates.get_branches(response.debate_id)
for branch in branches:
    print(f"Branch: {branch.name} ({len(branch.nodes)} nodes)")
```

### Matrix Debates (Parallel Scenarios)

Matrix debates run the same question across different scenarios to identify universal vs conditional conclusions.

```python
# Create matrix debate
response = client.matrix_debates.create(
    task="Should we adopt microservices?",
    agents=["anthropic-api", "openai-api"],
    scenarios=[
        {"name": "small_team", "parameters": {"team_size": 5}, "is_baseline": True},
        {"name": "large_team", "parameters": {"team_size": 50}},
        {"name": "high_scale", "parameters": {"requests_per_sec": 1_000_000}},
    ],
    max_rounds=3,
)

# Get matrix debate results
matrix = client.matrix_debates.get(response.matrix_id)

# Get conclusions (universal vs conditional)
conclusions = client.matrix_debates.get_conclusions(response.matrix_id)
print("Universal conclusions (true in all scenarios):")
for c in conclusions.universal:
    print(f"  - {c}")

print("\nConditional conclusions:")
for scenario, findings in conclusions.conditional.items():
    print(f"  {scenario}:")
    for f in findings:
        print(f"    - {f}")
```

### Verification (Formal Methods)

Verify claims using formal methods (Z3, Lean, Coq).

```python
# Verify a claim
result = client.verification.verify(
    claim="All prime numbers greater than 2 are odd",
    context="Number theory",
    backend="z3",      # z3, lean, coq
    timeout=30,
)

print(f"Status: {result.status}")  # valid, invalid, unknown, error
if result.proof:
    print(f"Proof: {result.proof}")
if result.counterexample:
    print(f"Counterexample: {result.counterexample}")

# Check backend availability
status = client.verification.status()
for backend in status.backends:
    print(f"{backend.name}: {'available' if backend.available else 'unavailable'}")
```

### Memory Analytics

Monitor memory tier performance and get optimization recommendations.

```python
# Get comprehensive analytics
analytics = client.memory.analytics(days=30)
print(f"Total entries: {analytics.total_entries}")
print(f"Learning velocity: {analytics.learning_velocity:.2f}")

for tier in analytics.tiers:
    print(f"{tier.tier_name}: {tier.entry_count} entries, {tier.hit_rate:.0%} hit rate")

for rec in analytics.recommendations:
    print(f"[{rec.impact}] {rec.type}: {rec.description}")

# Get stats for specific tier
tier_stats = client.memory.tier_stats("fast", days=7)

# Take manual snapshot
snapshot = client.memory.snapshot()
```

### Agents

Discover available agents and their capabilities.

```python
# List all agents
agents = client.agents.list()
for agent in agents:
    print(f"{agent.agent_id}: ELO {agent.elo_rating}, {agent.win_rate:.0%} win rate")

# Get specific agent profile
profile = client.agents.get("anthropic-api")
print(f"Capabilities: {profile.capabilities}")
```

### Leaderboard

ELO rankings across all agents.

```python
# Get top agents
rankings = client.leaderboard.get(limit=10)
for entry in rankings:
    trend = {"up": "+", "down": "-", "stable": "="}[entry.recent_trend]
    print(f"#{entry.rank} {entry.agent_id}: {entry.elo_rating} ({trend})")
```

### Replays

View and export debate replays.

```python
# List replays
replays = client.replays.list(limit=10)

# Get full replay with events
replay = client.replays.get(replays[0].replay_id)
for event in replay.events:
    print(f"[{event.timestamp}] {event.event_type}: {event.content[:50]}...")

# Export to JSON/CSV
data = client.replays.export(replay.replay_id, format="json")

# Delete replay
client.replays.delete(replay.replay_id)
```

### Explainability

Inspect decision explanations, evidence chains, and counterfactuals.

```python
explanation = client.explainability.get_explanation(debate_id)
evidence = client.explainability.get_evidence(debate_id)
summary = client.explainability.get_summary(debate_id, format="markdown")

batch = client.explainability.create_batch([debate_id], include_evidence=True)
status = client.explainability.get_batch_status(batch.batch_id)
```

### Organizations

Manage organizations and membership.

```python
org = client.organizations.get(org_id)
members = client.organizations.list_members(org_id)
client.organizations.invite_member(org_id, email="user@acme.com", role="member")

memberships = client.organizations.list_user_organizations()
if memberships:
    client.organizations.switch_organization(memberships[0].org_id)
```

### Compliance Policies

Define policies and review violations.

```python
policies, total = client.policies.list(limit=50)
policy = client.policies.get("policy-123")

violations, _ = client.policies.list_violations(status="open")
result = client.policies.check(
    content="We store EU customer data in us-east-1",
    frameworks=["gdpr"],
)
```

### Tenants

Administer tenants for multi-tenant deployments.

```python
tenants, total = client.tenants.list()
tenant = client.tenants.create(name="Acme Corp", slug="acme", tier="enterprise")
usage = client.tenants.get_usage(tenant.id)
client.tenants.update_quotas(tenant.id, {"debates_per_month": 5000})
```

### Health Check

```python
health = client.health()
print(f"Status: {health.status}")
print(f"Version: {health.version}")
print(f"Uptime: {health.uptime_seconds:.0f}s")
```

## Type-Safe Models

The SDK uses Pydantic models for all request/response types:

```python
from aragora.client.models import (
    # Debates
    Debate, DebateStatus, DebateCreateRequest, DebateCreateResponse,
    DebateRound, AgentMessage, Vote, ConsensusResult, ConsensusType,

    # Gauntlet
    GauntletReceipt, GauntletVerdict, Finding,
    GauntletRunRequest, GauntletRunResponse,

    # Graph debates
    GraphDebate, GraphDebateBranch, GraphDebateNode,
    GraphDebateCreateRequest, GraphDebateCreateResponse,

    # Matrix debates
    MatrixDebate, MatrixScenario, MatrixScenarioResult, MatrixConclusion,
    MatrixDebateCreateRequest, MatrixDebateCreateResponse,

    # Verification
    VerifyClaimRequest, VerifyClaimResponse, VerifyStatusResponse,
    VerificationStatus, VerificationBackend,

    # Memory
    MemoryAnalyticsResponse, MemoryTierStats, MemoryRecommendation,
    MemorySnapshotResponse,

    # Agents
    AgentProfile, LeaderboardEntry,

    # Replays
    Replay, ReplaySummary, ReplayEvent,

    # General
    HealthCheck, APIError,
)
```

## Error Handling

```python
from aragora.client import AragoraClient, AragoraAPIError

client = AragoraClient(base_url="http://localhost:8080")

try:
    debate = client.debates.get("nonexistent-id")
except AragoraAPIError as e:
    print(f"Error: {e}")
    print(f"Code: {e.code}")
    print(f"Status: {e.status_code}")
```

Common error codes:
- `NOT_FOUND` (404): Resource doesn't exist
- `VALIDATION_ERROR` (400): Invalid request parameters
- `RATE_LIMITED` (429): Too many requests
- `UNAUTHORIZED` (401): Missing or invalid API key
- `INTERNAL_ERROR` (500): Server error

## Examples

### Basic Debate Workflow

```python
from aragora.client import AragoraClient

client = AragoraClient(base_url="http://localhost:8080")

# Run a complete debate
debate = client.debates.run(
    task="Design a secure authentication system",
    agents=["anthropic-api", "openai-api", "mistral-api"],
    rounds=3,
    consensus="majority",
)

if debate.consensus and debate.consensus.reached:
    print(f"Agreement: {debate.consensus.agreement:.0%}")
    print(f"Answer: {debate.consensus.final_answer}")
else:
    print("No consensus reached")
    for round in debate.rounds:
        for msg in round.messages:
            print(f"{msg.agent_id}: {msg.content[:100]}...")
```

### Streaming Debate Events (Python)

```python
import asyncio
from aragora.streaming import AragoraWebSocket

async def stream_debate(debate_id: str):
    ws = AragoraWebSocket(base_url="https://api.aragora.ai", api_key="YOUR_API_KEY")

    def on_message(event):
        data = event.get("data", {}) if isinstance(event, dict) else {}
        print(data.get("content", ""))

    ws.on("agent_message", on_message)
    await ws.connect(debate_id=debate_id)

    # Wait for consensus (or handle any other events)
    await ws.once("consensus", timeout=60)
    await ws.disconnect()

asyncio.run(stream_debate("debate-123"))
```

**Dependency:** `pip install websockets`
**Auth:** `api_key` is sent as a `token` query parameter; use a proxy to inject
`Authorization` headers if your WebSocket server requires header auth.

### Gauntlet for Policy Review

```python
from aragora.client import AragoraClient

client = AragoraClient(base_url="http://localhost:8080")

policy = """
Privacy Policy:
We collect user email and browsing history.
Data is stored indefinitely.
Third parties may access data for advertising.
"""

receipt = client.gauntlet.run_and_wait(
    input_content=policy,
    input_type="policy",
    persona="gdpr",
    profile="thorough",
)

print(f"Verdict: {receipt.verdict.value}")
print(f"Risk Score: {receipt.risk_score:.0%}")

for finding in receipt.findings:
    if finding.severity in ("critical", "high"):
        print(f"\n[{finding.severity.upper()}] {finding.title}")
        print(f"  {finding.description}")
        if finding.mitigation:
            print(f"  Fix: {finding.mitigation}")
```

### Matrix Debate for Decision Analysis

```python
import asyncio
from aragora.client import AragoraClient

async def analyze_decision():
    async with AragoraClient(base_url="http://localhost:8080") as client:
        # Compare microservices decision across team sizes
        response = await client.matrix_debates.create_async(
            task="Should we refactor our monolith to microservices?",
            scenarios=[
                {"name": "startup", "parameters": {"team_size": 5, "budget": "low"}},
                {"name": "scaleup", "parameters": {"team_size": 25, "budget": "medium"}},
                {"name": "enterprise", "parameters": {"team_size": 100, "budget": "high"}},
            ],
        )

        # Wait for completion (poll)
        import asyncio
        while True:
            matrix = await client.matrix_debates.get_async(response.matrix_id)
            if matrix.status.value in ("completed", "failed"):
                break
            await asyncio.sleep(5)

        conclusions = await client.matrix_debates.get_conclusions_async(response.matrix_id)

        print("Universal conclusions:")
        for c in conclusions.universal:
            print(f"  - {c}")

        print("\nConditional conclusions:")
        for scenario, findings in conclusions.conditional.items():
            print(f"\n  {scenario}:")
            for f in findings:
                print(f"    - {f}")

asyncio.run(analyze_decision())
```

## Related Documentation

- [API Reference](API_REFERENCE.md) - Full REST API documentation
- [WebSocket Events](WEBSOCKET_EVENTS.md) - Real-time streaming events
- [Gauntlet Guide](GAUNTLET.md) - Adversarial validation details
- [Graph Debates](GRAPH_DEBATES.md) - Branching debate documentation
- [Matrix Debates](MATRIX_DEBATES.md) - Parallel scenario debates
