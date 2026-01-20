# Cross-Functional Feature Integration Guide

This document describes how to enable and configure the cross-functional features that connect different subsystems in Aragora.

## Overview

Aragora's cross-functional features allow different subsystems to share data and enhance each other:

| Feature | Purpose | Default |
|---------|---------|---------|
| KnowledgeBridgeHub | Unified access to MetaLearner, Evidence, Pattern bridges | Auto-enabled with KnowledgeMound |
| MemoryCoordinator | Atomic writes across memory systems | `enable_coordinated_writes=True` |
| SelectionFeedbackLoop | Performance-based agent selection | `enable_performance_feedback=True` |
| CrossDebateMemory | Institutional knowledge injection | `enable_cross_debate_memory=True` |
| Post-debate Workflows | Automated processing after debates | `enable_post_debate_workflow=False` |
| Evidence Storage | Persist collected evidence | Auto-enabled with KnowledgeBridgeHub |
| Culture Observation | Extract organizational patterns | Auto-enabled with KnowledgeMound |

## Configuration

### Basic Setup

```python
from aragora import Arena, Environment, DebateProtocol
from aragora.debate.arena_config import ArenaConfig

config = ArenaConfig(
    # Knowledge integration
    enable_knowledge_retrieval=True,
    enable_knowledge_ingestion=True,
    enable_cross_debate_memory=True,

    # Memory coordination
    enable_coordinated_writes=True,

    # Performance feedback
    enable_performance_feedback=True,

    # Post-debate automation (disabled by default)
    enable_post_debate_workflow=False,
)

arena = Arena.from_config(
    env=Environment(task="Design a caching strategy"),
    agents=agents,
    protocol=DebateProtocol(rounds=3),
    config=config,
)

result = await arena.run()
```

### Advanced Configuration

#### Memory Coordinator

The MemoryCoordinator provides atomic writes across multiple memory systems:

```python
from aragora.memory.coordinator import MemoryCoordinator, CoordinatorOptions

coordinator = MemoryCoordinator(
    continuum_memory=continuum,
    consensus_memory=consensus,
    critique_store=critique_store,
    knowledge_mound=mound,
)

config = ArenaConfig(
    enable_coordinated_writes=True,
    memory_coordinator=coordinator,
    coordinator_parallel_writes=False,  # Sequential for safety
    coordinator_rollback_on_failure=True,
)
```

#### Selection Feedback Loop

Adjusts agent selection weights based on debate performance:

```python
from aragora.debate.selection_feedback import SelectionFeedbackLoop

feedback_loop = SelectionFeedbackLoop(
    learning_rate=0.15,
    decay_factor=0.9,
    min_debates_for_adjustment=5,
)

config = ArenaConfig(
    enable_performance_feedback=True,
    selection_feedback_loop=feedback_loop,
    feedback_loop_weight=0.15,
)
```

#### Cross-Debate Memory

Injects institutional knowledge from past debates:

```python
from aragora.memory.cross_debate_rlm import CrossDebateMemory

cross_memory = CrossDebateMemory(
    critique_store=critique_store,
    max_context_items=10,
)

config = ArenaConfig(
    enable_cross_debate_memory=True,
    cross_debate_memory=cross_memory,
)
```

#### Post-Debate Workflows

Trigger automated processing after high-confidence debates:

```python
from aragora.workflow.engine import WorkflowEngine
from aragora.workflow.patterns import refinement_workflow

config = ArenaConfig(
    enable_post_debate_workflow=True,
    post_debate_workflow=refinement_workflow,
    post_debate_workflow_threshold=0.7,  # Min confidence
)
```

## Feature Interactions

### Knowledge Flow

```
Debate Start
    │
    ├─► CrossDebateMemory queries past debates
    │       └─► Injects institutional knowledge
    │
    ├─► KnowledgeMound provides relevant context
    │       └─► Cached via TTL-based caching
    │
    └─► Belief cruxes injected from similar debates

Debate Rounds
    │
    └─► Evidence collected and linked

Debate End
    │
    ├─► EvidenceBridge stores evidence in mound
    │
    ├─► Culture patterns extracted and stored
    │
    ├─► MemoryCoordinator writes atomically to:
    │       ├─► ConsensusMemory
    │       ├─► ContinuumMemory
    │       ├─► CritiqueStore
    │       └─► KnowledgeMound
    │
    ├─► SelectionFeedbackLoop adjusts agent weights
    │
    └─► Post-debate workflow triggered (if enabled)
```

### Performance Impact

| Feature | Latency Impact | Memory Impact |
|---------|---------------|---------------|
| CrossDebateMemory | +50-200ms (cached) | ~10KB per debate |
| MemoryCoordinator | +20-50ms | Minimal |
| SelectionFeedbackLoop | +5-10ms | ~1KB per agent |
| EvidenceBridge | +30-100ms | Depends on evidence |
| Culture observation | +10-50ms | ~5KB per debate |

## Disabling Features

All cross-functional features can be disabled via ArenaConfig:

```python
config = ArenaConfig(
    # Disable all cross-functional features
    enable_knowledge_retrieval=False,
    enable_knowledge_ingestion=False,
    enable_cross_debate_memory=False,
    enable_coordinated_writes=False,
    enable_performance_feedback=False,
    enable_post_debate_workflow=False,
    enable_belief_guidance=False,
)
```

## Monitoring

### Event Types

New event types emitted by cross-functional features:

- `KNOWLEDGE_INJECTED` - Institutional knowledge added to context
- `EVIDENCE_STORED` - Evidence persisted in mound
- `CULTURE_OBSERVED` - Patterns extracted from debate
- `MEMORY_COORDINATED` - Atomic write completed
- `SELECTION_FEEDBACK` - Agent weights adjusted
- `WORKFLOW_TRIGGERED` - Post-debate workflow started

### Metrics

Enable telemetry to track cross-functional feature performance:

```python
config = ArenaConfig(
    enable_telemetry=True,
)
```

Metrics available:
- `aragora_knowledge_cache_hits_total`
- `aragora_knowledge_cache_misses_total`
- `aragora_memory_coordinator_writes_total`
- `aragora_selection_feedback_adjustments_total`
- `aragora_workflow_triggers_total`

## Troubleshooting

### Common Issues

1. **CrossDebateMemory not injecting context**
   - Check `enable_cross_debate_memory=True`
   - Verify `cross_debate_memory` is provided or CritiqueStore has historical data

2. **MemoryCoordinator rollbacks**
   - Check logs for `[memory_coordinator]` entries
   - Verify all memory systems are properly initialized

3. **SelectionFeedbackLoop not adjusting weights**
   - Requires `min_debates_for_adjustment` debates (default: 5)
   - Check `enable_performance_feedback=True`

4. **Workflows not triggering**
   - Check `enable_post_debate_workflow=True`
   - Verify debate confidence >= `post_debate_workflow_threshold`
   - Ensure `post_debate_workflow` is provided

### Debug Logging

Enable debug logging for cross-functional features:

```python
import logging
logging.getLogger("aragora.debate.phases.feedback_phase").setLevel(logging.DEBUG)
logging.getLogger("aragora.debate.phases.context_init").setLevel(logging.DEBUG)
logging.getLogger("aragora.memory.coordinator").setLevel(logging.DEBUG)
```
