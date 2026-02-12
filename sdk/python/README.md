# Aragora Python SDK

The canonical Python client for the Aragora multi-agent debate platform.

## Installation

```bash
pip install aragora-sdk
```

## Quick Start

```python
from aragora_sdk import AragoraClient

# Create a client
client = AragoraClient(
    base_url="https://api.aragora.ai",
    api_key="your-api-key"
)

# Create a debate
debate = client.debates.create(
    task="Should we adopt microservices architecture?"
)

print(f"Debate created: {debate['debate_id']}")

# Get debate messages
messages = client.debates.get_messages(debate['debate_id'])
for msg in messages['messages']:
    print(f"{msg['agent']}: {msg['content'][:100]}...")
```

## Async Usage

```python
from aragora_sdk import AragoraAsyncClient

async def main():
    async with AragoraAsyncClient(
        base_url="https://api.aragora.ai",
        api_key="your-api-key"
    ) as client:
        # Create a debate
        debate = await client.debates.create(
            task="What's the best approach for state management?"
        )

        # List agents
        agents = await client.agents.list()
        print(f"Available agents: {len(agents['agents'])}")

import asyncio
asyncio.run(main())
```

## Configuration

```python
from aragora_sdk import AragoraClient

client = AragoraClient(
    # Required: API base URL
    base_url="https://api.aragora.ai",

    # Optional: API key for authentication
    api_key="your-api-key",

    # Optional: Request timeout in seconds (default: 30)
    timeout=60.0,

    # Optional: Maximum retry attempts (default: 3)
    max_retries=5,

    # Optional: Base delay between retries in seconds (default: 1.0)
    retry_delay=1.0,
)
```

## Real-time Streaming

Stream debate events as they happen using WebSockets:

```python
from aragora_sdk import AragoraAsyncClient

async def stream_debate():
    async with AragoraAsyncClient(
        base_url="https://api.aragora.ai",
        api_key="your-api-key"
    ) as client:
        # Create WebSocket connection
        ws = client.stream.connect()
        await ws.open()

        # Subscribe to a debate
        await ws.subscribe("debate-id")

        # Handle events
        async for event in ws.events():
            if event.type == "debate_start":
                print(f"Debate started: {event.data['task']}")
            elif event.type == "agent_message":
                print(f"[Round {event.data['round']}] "
                      f"{event.data['agent']}: {event.data['content']}")
            elif event.type == "critique":
                print(f"{event.data['critic']} critiques "
                      f"{event.data['target']}")
            elif event.type == "vote":
                print(f"{event.data['agent']} votes: "
                      f"{event.data['vote']}")
            elif event.type == "consensus":
                print(f"Consensus: {event.data['final_answer']}")
            elif event.type == "debate_end":
                print(f"Ended: {event.data['status']}")
                break

asyncio.run(stream_debate())
```

## API Reference

### Debates

```python
# List debates
debates = client.debates.list(limit=10, status="completed")

# Get a specific debate
debate = client.debates.get("debate-id")

# Create a debate
result = client.debates.create(
    task="Your question here",
    agents=["claude", "gpt-4"],
    rounds=3,
    consensus="majority"  # "majority" | "unanimous" | "weighted" | "semantic"
)

# Search debates
results = client.debates.search("AI safety")

# Export debate
markdown = client.debates.export("debate-id", format="markdown")
```

### Agents

```python
# List all agents
agents = client.agents.list()

# Get agent details
agent = client.agents.get("claude")

# Get agent profile with stats
profile = client.agents.get_profile("claude")

# Get leaderboard
leaderboard = client.agents.leaderboard()

# Compare agents
comparison = client.agents.compare(["claude", "gpt-4"])
```

### Explainability

```python
# Get full explanation
explanation = client.explainability.get("debate-id",
    include_factors=True,
    include_counterfactuals=True,
    include_provenance=True
)

# Get contributing factors
factors = client.explainability.factors("debate-id",
    min_contribution=0.1
)

# Generate counterfactual
counterfactual = client.explainability.counterfactual("debate-id",
    hypothesis="What if agent X had a different opinion?",
    affected_agents=["claude"]
)

# Get narrative summary
narrative = client.explainability.narrative("debate-id",
    format="executive_summary"
)
```

### Workflows

```python
# List workflow templates
templates = client.workflows.list_templates(
    category="analysis",
    tags="security,compliance"
)

# Create workflow from pattern
workflow = client.workflows.instantiate_pattern("pattern-id",
    name="My Security Workflow",
    description="Custom security analysis",
    category="security",
    agents=["claude", "gpt-4"]
)

# Execute workflow
execution = client.workflows.execute("workflow-id",
    input_data="document content..."
)
```

### Gauntlet (Decision Receipts)

```python
# List receipts
receipts = client.gauntlet.list(verdict="approved")

# Get receipt details
receipt = client.gauntlet.get("receipt-id")

# Verify receipt integrity
result = client.receipts.verify("receipt-id")
print(f"Valid: {result['valid']}, Hash: {result['hash']}")

# Export receipt
sarif = client.receipts.export("receipt-id", format="sarif")
```

### Template Marketplace

```python
# Get marketplace categories
categories = client.marketplace.categories()

# Browse templates
templates = client.marketplace.browse(
    category="security",
    sort_by="downloads"
)

# Import a template
imported = client.marketplace.import_template("template-id")

# Rate a template
client.marketplace.rate("template-id", rating=5)

# Publish a template
client.marketplace.publish(
    template_id="my-template",
    name="Security Audit",
    description="Comprehensive security analysis workflow",
    category="security",
    tags=["audit", "compliance"]
)
```

### Control Plane

```python
# Agent registry
client.control_plane.register_agent(
    agent_id="custom-agent",
    capabilities=["reasoning", "code_review"],
    metadata={"version": "1.0.0"}
)

agents = client.control_plane.list_agents()
health = client.control_plane.agent_health("anthropic-api")

# Task scheduling
task = client.control_plane.submit_task(
    task_type="debate",
    payload={"question": "Should we use GraphQL?"},
    priority=8,
    required_capabilities=["reasoning"]
)

# Resource monitoring
status = client.control_plane.status()
print(f"Active agents: {status['active_agents']}")
print(f"Pending tasks: {status['pending_tasks']}")
```

### Batch Operations

```python
# Submit batch of debates
batch = client.batch.submit_debates(
    debates=[
        {"task": "Review microservices architecture"},
        {"task": "Evaluate caching strategy"},
    ],
    priority="high",
    callback_url="https://example.com/webhook"
)

# Check batch status
status = client.batch.get_status(batch["batch_id"])

# List batches
batches = client.batch.list(status="processing")
```

### Verification

```python
# Generate a formal proof
proof = client.verification.generate_proof("debate-id")

# Verify a proof
result = client.verification.verify(proof["proof_id"])
```

## Namespace APIs (Advanced)

The Python SDK provides 90+ namespace APIs for granular control over all platform features. Access them as attributes on the client:

| Namespace | Description |
|-----------|-------------|
| `client.debates` | Debate CRUD, search, export |
| `client.agents` | Agent registry, profiles, stats |
| `client.workflows` | Templates, patterns, execution |
| `client.explainability` | Factors, counterfactuals, narratives |
| `client.marketplace` | Template marketplace operations |
| `client.control_plane` | Agent registry, task scheduling |
| `client.batch` | Bulk debate operations |
| `client.knowledge` | Knowledge mound access |
| `client.memory` | Memory tier management |
| `client.consensus` | Consensus detection |
| `client.ranking` | ELO rankings, tournaments |
| `client.gauntlet` | Gauntlet runner, findings |
| `client.receipts` | Decision receipt management |
| `client.verification` | Formal proof generation |
| `client.sme` | SME quick decisions, risk assessment |
| `client.rbac` | Role-based access control |
| `client.auth` | Authentication, OAuth |
| `client.tenants` | Multi-tenancy management |
| `client.connectors` | External integrations |
| `client.pulse` | Trending topics |
| `client.analytics` | Analytics dashboards |
| `client.billing` | Billing and usage |
| `client.backups` | Backup management |
| `client.policies` | Policy governance |
| `client.webhooks` | Webhook configuration |
| `client.notifications` | Notification delivery |

## Type Exports

The SDK exports 196 types for type-safe development:

```python
from aragora_sdk import (
    # Core types
    DebateResult, DebateConfig, DebateMessage,
    AgentProfile, AgentStats,
    ConsensusResult, ConsensusMethod,
    # Workflow types
    WorkflowTemplate, WorkflowExecution,
    # Explainability
    ExplanationResult, ExplanationFactor,
    # Enterprise
    TenantConfig, RBACRole, Permission,
    # ... 196 types total
)

# Or import from generated types
from aragora.generated_types import (
    DebateResult,
    AgentProfile,
    WorkflowTemplate,
    # etc.
)
```

## Error Handling

```python
from aragora_sdk import (
    AragoraClient,
    AragoraError,
    RateLimitError,
    NotFoundError,
    AuthenticationError,
    ValidationError,
)

client = AragoraClient(base_url="https://api.aragora.ai")

try:
    debate = client.debates.get("invalid-id")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except NotFoundError as e:
    print(f"Not found: {e.message}")
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
except ValidationError as e:
    print(f"Validation error: {e.message}")
except AragoraError as e:
    print(f"API error [{e.status_code}]: {e.message}")
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy aragora

# Linting
ruff check aragora
```

## License

MIT License - see LICENSE file for details.
