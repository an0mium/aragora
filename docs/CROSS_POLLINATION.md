# Cross-Pollination Architecture

Aragora's cross-pollination system enables different subsystems to communicate and share data through a unified event-driven architecture. This creates powerful synergies where improvements in one area automatically benefit others.

## Overview

```
Event Flow:
                                    ┌──────────────────────────────────┐
                                    │    CrossSubscriberManager        │
                                    │                                  │
┌─────────────────┐                 │  ┌────────────────────────────┐ │
│ Arena (Debates) │─────Events────▶ │  │ memory_to_rlm              │ │
└─────────────────┘                 │  │ elo_to_debate              │ │
                                    │  │ knowledge_to_memory        │ │
┌─────────────────┐                 │  │ calibration_to_agent       │ │
│ ContinuumMemory │─────Events────▶ │  │ evidence_to_insight        │ │
└─────────────────┘                 │  │ mound_to_memory            │ │
                                    │  └────────────────────────────┘ │
┌─────────────────┐                 └──────────────────────────────────┘
│ KnowledgeMound  │─────Events────▶
└─────────────────┘

┌─────────────────┐
│ ELO System      │─────Events────▶
└─────────────────┘
```

## Key Components

### 1. CrossSubscriberManager

Central hub for cross-subsystem event routing.

```python
from aragora.events.cross_subscribers import get_cross_subscriber_manager

manager = get_cross_subscriber_manager()

# Register a custom handler
@manager.subscribe(StreamEventType.MEMORY_STORED)
def on_memory_stored(event):
    print(f"Memory stored: {event.data}")

# Or register manually
manager.register("my_handler", StreamEventType.AGENT_ELO_UPDATED, my_handler_fn)
```

### 2. Arena Event Bridge

Connects Arena's internal EventBus to the CrossSubscriberManager.

```python
from aragora.events.arena_bridge import ArenaEventBridge, create_arena_bridge

# Automatic - enabled by default via ArenaBuilder
arena = ArenaBuilder(env, agents).build()

# Manual setup
bridge = create_arena_bridge(arena.event_bus)
bridge.connect_to_cross_subscribers()
```

### 3. Evidence-Provenance Bridge

Links evidence snippets to belief claims for provenance tracking.

```python
from aragora.reasoning.evidence_bridge import get_evidence_bridge

bridge = get_evidence_bridge()

# Register evidence
bridge.register_evidence(evidence_snippet, claim_id="claim_001")

# Link to claims with relevance scoring
bridge.link_to_claim(evidence_id, claim_id, relevance=0.9)

# Create evidence chains
chain = bridge.create_evidence_chain("chain_001", [evidence1, evidence2])
```

### 4. RLM Training Integration

Automatically collects debate trajectories for reinforcement learning.

```python
from aragora.rlm.debate_integration import (
    get_debate_trajectory_collector,
    create_training_hook,
)

# Get collected trajectories
collector = get_debate_trajectory_collector()
trajectories = collector.get_trajectories(limit=100)

# Hook is auto-installed via ArenaBuilder (enabled by default)
arena = ArenaBuilder(env, agents).with_rlm_training(True).build()
```

## Event Types

The system supports these cross-pollination event types:

| Event Type | Source | Description |
|------------|--------|-------------|
| `MEMORY_STORED` | ContinuumMemory | New memory item stored |
| `MEMORY_RETRIEVED` | ContinuumMemory | Memory item accessed |
| `AGENT_ELO_UPDATED` | EloSystem | Agent rating changed |
| `AGENT_CALIBRATION_CHANGED` | CalibrationTracker | Agent calibration updated |
| `AGENT_FALLBACK_TRIGGERED` | Agent | Fallback provider activated |
| `KNOWLEDGE_INDEXED` | KnowledgeMound | New knowledge indexed |
| `KNOWLEDGE_QUERIED` | KnowledgeMound | Knowledge queried |
| `MOUND_UPDATED` | KnowledgeMound | Mound structure changed |

## Built-in Handlers

### memory_to_rlm
Records memory access patterns to inform RLM compression strategies.

### elo_to_debate
Updates AgentPool weights based on ELO changes.

### knowledge_to_memory
Syncs high-confidence knowledge nodes to ContinuumMemory.

### calibration_to_agent
Updates agent confidence weights based on calibration data.

### evidence_to_insight
Stores high-confidence evidence as insights.

### mound_to_memory
Handles knowledge mound structure changes.

## Configuration

Settings can be controlled via environment variables or code:

```python
from aragora.config.settings import get_settings

settings = get_settings()

# Check integration settings
print(settings.integration.rlm_training_enabled)      # True
print(settings.integration.knowledge_mound_enabled)   # True
print(settings.integration.cross_subscribers_enabled) # True
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_INTEGRATION_RLM_TRAINING` | `true` | Enable RLM training collection |
| `ARAGORA_INTEGRATION_KNOWLEDGE_MOUND` | `true` | Enable Knowledge Mound integration |
| `ARAGORA_INTEGRATION_KNOWLEDGE_THRESHOLD` | `0.85` | Min confidence for ingestion |
| `ARAGORA_INTEGRATION_CROSS_SUBSCRIBERS` | `true` | Enable cross-subscribers |
| `ARAGORA_INTEGRATION_ARENA_BRIDGE` | `true` | Enable Arena event bridge |
| `ARAGORA_INTEGRATION_AUTO_REVALIDATION` | `false` | Enable auto revalidation |
| `ARAGORA_INTEGRATION_EVIDENCE_BRIDGE` | `true` | Enable evidence bridge |

## Observability API

Monitor cross-pollination status via HTTP endpoints:

```bash
# Get subscriber statistics
curl http://localhost:8080/api/cross-pollination/stats

# List all registered subscribers
curl http://localhost:8080/api/cross-pollination/subscribers

# Get Arena event bridge status
curl http://localhost:8080/api/cross-pollination/bridge

# Reset statistics (for testing)
curl -X POST http://localhost:8080/api/cross-pollination/reset
```

## Knowledge Revalidation

Automatic staleness detection and revalidation via debates:

```python
from aragora.knowledge.mound import RevalidationScheduler

scheduler = RevalidationScheduler(
    knowledge_mound=mound,
    staleness_threshold=0.7,
    check_interval_seconds=3600,
    revalidation_method="debate",  # or "evidence", "expert"
)

# Start background monitoring
await scheduler.start()

# Manual check
task_ids = await scheduler.check_and_schedule_revalidations()
```

## Best Practices

1. **Enable default integrations** - The defaults are optimized for most use cases
2. **Monitor via observability endpoints** - Check `/api/cross-pollination/stats` regularly
3. **Use ArenaBuilder** - It automatically wires up all integrations
4. **Set appropriate thresholds** - Higher thresholds = less noise, lower thresholds = more data
5. **Consider memory impact** - Disable unused integrations in resource-constrained environments

## Troubleshooting

### Events not being delivered

1. Check if cross-subscribers are enabled: `settings.integration.cross_subscribers_enabled`
2. Verify the Arena event bridge is connected: `GET /api/cross-pollination/bridge`
3. Check for handler errors in logs

### High memory usage

1. Reduce trajectory collection: `.with_rlm_training(False)` in ArenaBuilder
2. Increase knowledge ingestion threshold to reduce stored items
3. Clear collected data periodically: `collector.clear()`

### Slow debate performance

1. Disable auto-revalidation: `ARAGORA_INTEGRATION_AUTO_REVALIDATION=false`
2. Use async handlers for expensive operations
3. Consider sampling events instead of processing all

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ARAGORA SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   Arena     │    │  Memory     │    │  Knowledge  │    │    ELO      │ │
│  │ (Debates)   │    │ (Continuum) │    │   (Mound)   │    │  (Rankings) │ │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘ │
│         │                  │                  │                  │        │
│         │    Events        │    Events        │    Events        │        │
│         └──────────────────┴──────────────────┴──────────────────┘        │
│                                    │                                       │
│                                    ▼                                       │
│         ┌──────────────────────────────────────────────────────────┐      │
│         │              CrossSubscriberManager                       │      │
│         │                                                           │      │
│         │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │      │
│         │   │memory_to │ │ elo_to   │ │knowledge │ │evidence_ │   │      │
│         │   │   _rlm   │ │ _debate  │ │_to_memory│ │to_insight│   │      │
│         │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │      │
│         └────────┼────────────┼────────────┼────────────┼──────────┘      │
│                  │            │            │            │                  │
│         ┌────────▼────┐ ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐          │
│         │ RLM Trainer │ │ AgentPool │ │ Memory  │ │  Insight  │          │
│         │(Trajectories│ │ (Weights) │ │  Sync   │ │  Store    │          │
│         └─────────────┘ └───────────┘ └─────────┘ └───────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
