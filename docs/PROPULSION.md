# Propulsion Engine Guide

The Propulsion Engine implements the Gastown pattern for push-based work assignment in multi-agent debates. Instead of agents polling for work, work is actively pushed to handlers when it becomes available.

## Overview

The Propulsion system provides:
- **Push-based execution**: Work flows to handlers automatically when ready
- **Priority scheduling**: Critical work processed before normal priority
- **Deadline awareness**: Payloads expire if not processed in time
- **Handler filtering**: Route work to specific handlers based on payload attributes
- **Chained execution**: Sequential multi-stage pipelines
- **Retry with backoff**: Automatic retry on transient failures
- **Hook integration**: Lifecycle events via HookManager

## Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Debate Phase   │────>│  PropulsionEngine │────>│  Handler Pool   │
│  (Proposals)    │     │                   │     │                 │
└─────────────────┘     └─────────┬─────────┘     └────────┬────────┘
                                  │                        │
                        ┌─────────┴─────────┐              │
                        │  PropulsionPayload │              │
                        │  - data            │              │
                        │  - priority        │              │
                        │  - deadline        │              │
                        │  - routing_key     │              │
                        └─────────┬─────────┘              │
                                  │                        │
                        ┌─────────┴─────────┐              │
                        │  Event Dispatch   │<─────────────┘
                        │  proposals_ready  │
                        │  critiques_ready  │
                        │  revisions_done   │
                        └───────────────────┘
```

## Quick Start

```python
from aragora.debate.propulsion import (
    PropulsionEngine,
    PropulsionPayload,
    PropulsionPriority,
    propulsion_handler,
    get_propulsion_engine,
)

# Create engine
engine = PropulsionEngine()

# Register handlers for different stages
engine.register_handler("proposals_ready", handle_critiques)
engine.register_handler("critiques_ready", handle_revisions)
engine.register_handler("revisions_done", handle_consensus)

# Push work to next stage
await engine.propel("proposals_ready", PropulsionPayload(
    data={"proposals": proposals, "debate_id": "debate-123"},
    source_molecule_id="debate-123",
    priority=PropulsionPriority.HIGH,
))

# Chain multiple stages
await engine.chain([
    ("proposals_ready", payload1),
    ("critiques_ready", payload2),
    ("revisions_done", payload3),
])
```

## Core Classes

### PropulsionPriority

Priority levels determine execution order:

```python
from aragora.debate.propulsion import PropulsionPriority

class PropulsionPriority(Enum):
    CRITICAL = 0    # Must be processed immediately
    HIGH = 1        # Process before normal priority
    NORMAL = 2      # Default priority
    LOW = 3         # Process when no higher priority work
    BACKGROUND = 4  # Process during idle time
```

### PropulsionPayload

Work item containing data and metadata for routing:

```python
from aragora.debate.propulsion import PropulsionPayload
from datetime import datetime, timezone, timedelta

payload = PropulsionPayload(
    # Required: The actual work data
    data={"proposals": [...], "debate_id": "debate-123"},

    # Priority (default: NORMAL)
    priority=PropulsionPriority.HIGH,

    # Optional deadline - payload expires if not processed
    deadline=datetime.now(timezone.utc) + timedelta(minutes=5),

    # Source tracking
    source_molecule_id="debate-123",
    source_stage="proposal_generation",

    # Routing hints
    routing_key="claude-team",
    agent_affinity="claude-3-opus",  # Prefer specific agent

    # Retry configuration
    max_attempts=3,
)

# Check payload status
if payload.is_expired():
    print("Payload past deadline")

if payload.can_retry():
    print(f"Attempt {payload.attempt_count}/{payload.max_attempts}")

# Serialize for logging/storage
payload_dict = payload.to_dict()
```

### PropulsionResult

Result returned from handler execution:

```python
from aragora.debate.propulsion import PropulsionResult

# Results contain execution details
result = PropulsionResult(
    payload_id="abc123",
    success=True,
    handler_name="critique_handler",
    result={"critiques": [...]},
    duration_ms=1234.5,
)

# Check for errors
if not result.success:
    print(f"Handler failed: {result.error_message}")
```

## PropulsionEngine

The engine orchestrates work distribution across handlers.

### Initialization

```python
from aragora.debate.propulsion import PropulsionEngine
from aragora.debate.hooks import HookManager

# Basic initialization
engine = PropulsionEngine()

# With custom settings
engine = PropulsionEngine(
    hook_manager=HookManager(),  # Optional hook integration
    max_concurrent=10,           # Max parallel handler executions
)
```

### Handler Registration

```python
# Register with function reference
def handle_proposals(payload: PropulsionPayload):
    proposals = payload.data["proposals"]
    return {"processed": len(proposals)}

unregister = engine.register_handler(
    event_type="proposals_ready",
    handler=handle_proposals,
    name="proposal_processor",
    priority=PropulsionPriority.HIGH,
)

# Async handlers supported
async def handle_critiques(payload: PropulsionPayload):
    await asyncio.sleep(0.1)  # Simulate work
    return {"critiques": [...]}

engine.register_handler("critiques_ready", handle_critiques)

# With filter function - only handle specific payloads
def filter_claude_only(payload: PropulsionPayload) -> bool:
    return payload.agent_affinity == "claude"

engine.register_handler(
    "proposals_ready",
    handle_claude_proposals,
    filter_fn=filter_claude_only,
)

# Unregister when done
unregister()

# Or unregister by name
engine.unregister_handler("proposals_ready", "proposal_processor")
```

### Propelling Work

```python
# Basic propel - fires all matching handlers
results = await engine.propel("proposals_ready", payload)

for result in results:
    if result.success:
        print(f"{result.handler_name} completed in {result.duration_ms}ms")
    else:
        print(f"{result.handler_name} failed: {result.error_message}")
```

### Chained Execution

Execute multiple stages in sequence:

```python
# Chain stops on first failure by default
all_results = await engine.chain([
    ("proposals_ready", proposal_payload),
    ("critiques_ready", critique_payload),
    ("revisions_done", revision_payload),
])

# Continue even if stages fail
all_results = await engine.chain(
    events=[...],
    stop_on_failure=False,
)

# Results is list of lists (one per stage)
for stage_idx, stage_results in enumerate(all_results):
    print(f"Stage {stage_idx}: {len(stage_results)} handlers executed")
```

### Retry with Backoff

Automatic retry on transient failures:

```python
# Retry up to 3 times with exponential backoff
results = await engine.propel_with_retry(
    "proposals_ready",
    payload,
    max_retries=3,
    backoff_base=1.0,  # 1s, 2s, 4s delays
)
```

### Broadcast

Send to multiple event types simultaneously:

```python
# Broadcast payload to multiple stages
results_by_event = await engine.broadcast(
    event_types=["logging", "metrics", "audit"],
    payload=payload,
)

for event_type, results in results_by_event.items():
    print(f"{event_type}: {len(results)} handlers")
```

### Statistics

```python
stats = engine.get_stats()
print(f"Total propelled: {stats['total_propelled']}")
print(f"Successful: {stats['successful']}")
print(f"Failed: {stats['failed']}")
print(f"Retried: {stats['retried']}")
print(f"Handlers: {stats['registered_handlers']}")

# Get specific result
result = engine.get_result("payload-id", "handler-name")

# Clear stored results
engine.clear_results()
```

## Decorator Registration

Use the decorator for automatic registration:

```python
from aragora.debate.propulsion import propulsion_handler, PropulsionPriority

@propulsion_handler("proposals_ready")
async def handle_proposals(payload: PropulsionPayload):
    """Automatically registered on import."""
    return {"status": "processed"}

@propulsion_handler("critiques_ready", priority=PropulsionPriority.HIGH)
async def handle_critiques(payload: PropulsionPayload):
    """High-priority handler."""
    return {"critiques": [...]}
```

## Global Engine

Access the singleton engine instance:

```python
from aragora.debate.propulsion import (
    get_propulsion_engine,
    reset_propulsion_engine,
)

# Get singleton instance
engine = get_propulsion_engine()

# Reset for testing
reset_propulsion_engine()
```

## Debate Integration

The propulsion engine integrates with debate phases:

### Standard Event Types

| Event Type | Fired When | Typical Handler |
|------------|------------|-----------------|
| `proposals_ready` | Initial proposals generated | Start critique phase |
| `critiques_ready` | Critiques complete | Start revision phase |
| `revisions_done` | Revisions complete | Check consensus |
| `consensus_reached` | Consensus achieved | Finalize debate |
| `round_complete` | Round ends | Start next round |

### Example Debate Flow

```python
from aragora.debate.propulsion import (
    PropulsionEngine,
    PropulsionPayload,
    PropulsionPriority,
)

engine = PropulsionEngine()

# Register phase handlers
async def start_critiques(payload: PropulsionPayload):
    """Handle proposals_ready by generating critiques."""
    proposals = payload.data["proposals"]
    debate_id = payload.data["debate_id"]

    # Generate critiques for each proposal
    critiques = await generate_critiques(proposals)

    # Propel to next stage
    await engine.propel("critiques_ready", PropulsionPayload(
        data={"critiques": critiques, "debate_id": debate_id},
        source_molecule_id=debate_id,
        source_stage="proposals_ready",
    ))

    return {"critique_count": len(critiques)}

async def start_revisions(payload: PropulsionPayload):
    """Handle critiques_ready by generating revisions."""
    critiques = payload.data["critiques"]
    debate_id = payload.data["debate_id"]

    # Generate revisions based on critiques
    revisions = await generate_revisions(critiques)

    # Propel to consensus check
    await engine.propel("revisions_done", PropulsionPayload(
        data={"revisions": revisions, "debate_id": debate_id},
        source_molecule_id=debate_id,
        source_stage="critiques_ready",
    ))

    return {"revision_count": len(revisions)}

async def check_consensus(payload: PropulsionPayload):
    """Handle revisions_done by checking for consensus."""
    revisions = payload.data["revisions"]
    debate_id = payload.data["debate_id"]

    consensus = detect_consensus(revisions)

    if consensus.reached:
        await engine.propel("consensus_reached", PropulsionPayload(
            data={"consensus": consensus, "debate_id": debate_id},
            source_molecule_id=debate_id,
        ))

    return {"consensus_reached": consensus.reached}

# Register handlers
engine.register_handler("proposals_ready", start_critiques)
engine.register_handler("critiques_ready", start_revisions)
engine.register_handler("revisions_done", check_consensus)

# Start the debate flow
await engine.propel("proposals_ready", PropulsionPayload(
    data={"proposals": initial_proposals, "debate_id": "debate-123"},
    source_molecule_id="debate-123",
    priority=PropulsionPriority.NORMAL,
))
```

## Hook Integration

The engine fires hooks for lifecycle events:

```python
from aragora.debate.hooks import HookManager, HookType

hook_manager = HookManager()

# Register hook for propulsion events
async def on_propel(source_stage, target_stage, payload):
    print(f"Propelling from {source_stage} to {target_stage}")

hook_manager.register(HookType.ON_PROPEL, on_propel)

# Engine uses hook manager
engine = PropulsionEngine(hook_manager=hook_manager)

# Hook fires on every propel
await engine.propel("proposals_ready", payload)
# Output: "Propelling from None to proposals_ready"
```

## Best Practices

1. **Use meaningful event types** - Names like `proposals_ready` are clearer than `stage1`

2. **Set appropriate priorities** - Reserve CRITICAL for time-sensitive work

3. **Configure deadlines** - Prevent stale work from clogging the system:
   ```python
   payload = PropulsionPayload(
       data={...},
       deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
   )
   ```

4. **Use filters for routing** - Instead of checking inside handlers:
   ```python
   engine.register_handler(
       "proposals_ready",
       claude_handler,
       filter_fn=lambda p: p.agent_affinity == "claude",
   )
   ```

5. **Track source stages** - Helps with debugging and auditing:
   ```python
   payload.source_stage = "previous_stage"
   ```

6. **Handle failures gracefully** - Use retry for transient errors:
   ```python
   results = await engine.propel_with_retry(event, payload, max_retries=3)
   ```

7. **Monitor statistics** - Track success rates and performance:
   ```python
   stats = engine.get_stats()
   if stats["failed"] > stats["successful"]:
       logger.warning("High failure rate detected")
   ```

## Error Handling

```python
# Results include error details
results = await engine.propel("proposals_ready", payload)

for result in results:
    if not result.success:
        logger.error(
            f"Handler {result.handler_name} failed: {result.error_message}"
        )

        # Payload tracks last error
        if payload.can_retry():
            results = await engine.propel_with_retry(
                "proposals_ready", payload
            )

# Expired payloads return immediately
if payload.is_expired():
    logger.warning(f"Payload {payload.id} expired")
```

## Testing

```python
from aragora.debate.propulsion import (
    PropulsionEngine,
    PropulsionPayload,
    reset_propulsion_engine,
)

def test_propulsion_flow():
    # Reset global engine between tests
    reset_propulsion_engine()

    engine = PropulsionEngine()
    received = []

    async def test_handler(payload):
        received.append(payload.data)
        return {"status": "ok"}

    engine.register_handler("test_event", test_handler)

    # Propel and verify
    results = asyncio.run(engine.propel(
        "test_event",
        PropulsionPayload(data={"key": "value"})
    ))

    assert len(results) == 1
    assert results[0].success
    assert received[0]["key"] == "value"

def test_chain_stops_on_failure():
    engine = PropulsionEngine()

    async def failing_handler(payload):
        raise ValueError("Simulated failure")

    engine.register_handler("stage1", failing_handler)
    engine.register_handler("stage2", lambda p: "never reached")

    results = asyncio.run(engine.chain([
        ("stage1", PropulsionPayload(data={})),
        ("stage2", PropulsionPayload(data={})),
    ]))

    # Only stage1 executed
    assert len(results) == 1
    assert not results[0][0].success
```

## See Also

- [Debate Phases](DEBATE_PHASES.md) - Phase implementations that use propulsion
- [Hooks System](HOOKS.md) - Lifecycle hooks integration
- [Workflow Engine](WORKFLOWS.md) - Higher-level workflow orchestration
