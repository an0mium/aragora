---
title: Cross-Functional Feature Integration Guide
description: Cross-Functional Feature Integration Guide
---

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

The MemoryCoordinator provides atomic writes across multiple memory systems with full transaction support including rollback on partial failure.

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

**Transaction Rollback**: When `coordinator_rollback_on_failure=True`, partial failures trigger automatic rollback of successful writes. Each memory system has a registered delete method:

| System | Rollback Method | Behavior |
|--------|-----------------|----------|
| Continuum | `delete(memory_id, archive=True)` | Archives pattern before deletion |
| Consensus | `delete_consensus(id, cascade_dissents=True)` | Removes consensus + associated dissents |
| Critique | `delete_debate(id, cascade_critiques=True)` | Removes debate record + critiques |
| Mound | `delete_item(item_id)` | Removes knowledge item |

**Custom Rollback Handlers**: Override default rollback behavior:

```python
async def custom_continuum_rollback(op: WriteOperation) -> bool:
    # Custom cleanup logic
    logger.warning(f"Rolling back continuum write: {op.result}")
    return True  # Return True if rollback succeeded

coordinator.register_rollback_handler("continuum", custom_continuum_rollback)
```

**Transaction Inspection**: Check transaction status after commit:

```python
tx = await coordinator.commit_debate_outcome(ctx)

if tx.partial_failure:
    failed_ops = tx.get_failed_operations()
    for op in failed_ops:
        logger.error(f"{op.target} failed: {op.error}")

if tx.rolled_back:
    logger.warning("Transaction was rolled back due to failures")
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

---

## Phase 8: Cross-Pollination Integrations

Phase 8 introduces feedback loops that make the system self-improving by connecting subsystems that previously operated independently.

### Overview

| Feature | Purpose | Default |
|---------|---------|---------|
| HookHandlerRegistry | Automatic event wiring across subsystems | `enable_hook_handlers=True` |
| PerformanceEloIntegrator | Performance metrics influence ELO adjustments | `enable_performance_elo=True` |
| OutcomeMemoryBridge | Successful outcomes promote memories | `enable_outcome_memory=True` |
| TricksterCalibrator | Auto-calibrate hollow consensus detection | `enable_trickster_calibration=True` |
| Checkpoint Memory State | Checkpoints include memory for restoration | `checkpoint_include_memory=True` |

### Configuration

```python
from aragora.debate.arena_config import ArenaConfig

config = ArenaConfig(
    # Phase 8 Cross-Pollination (all enabled by default)
    enable_hook_handlers=True,           # Wire subsystems to hook lifecycle
    enable_performance_elo=True,         # Performance affects ELO K-factor
    enable_outcome_memory=True,          # Outcomes promote/demote memories
    enable_trickster_calibration=True,   # Auto-tune detection sensitivity
    checkpoint_include_memory=True,      # Include memory in checkpoints
)
```

### Hook Handler Registry

The `HookHandlerRegistry` automatically wires subsystems to the debate lifecycle hooks:

```python
from aragora.debate.hook_handlers import HookHandlerRegistry, create_hook_handler_registry

# Automatic wiring via convenience function
registry = create_hook_handler_registry(
    hook_manager=hook_manager,
    analytics=analytics_coordinator,
    continuum_memory=continuum,
    calibration_tracker=calibration,
    outcome_tracker=outcome_tracker,
    performance_monitor=performance_monitor,
    trickster=trickster,
    auto_register=True,  # Automatically register all handlers
)

# Or manual control
registry = HookHandlerRegistry(
    hook_manager=hook_manager,
    subsystems={
        "analytics": analytics_coordinator,
        "continuum_memory": continuum,
    },
)
count = registry.register_all()
print(f"Registered \{count\} handlers")
```

**Hooks registered:**
- `POST_ROUND` → Analytics round tracking
- `POST_GENERATE` → Response timing and quality metrics
- `POST_DEBATE` → Memory writes, outcome recording
- `PRE_CONSENSUS` → Trickster hollow consensus check
- `ON_CONVERGENCE` → Convergence tracking

Declarative hook configs can also be layered on top of the HookManager. See
[HOOKS.md](HOOKS.md) for YAML configuration and built-in actions.

### Performance-ELO Integration

The `PerformanceEloIntegrator` modulates ELO K-factors based on agent performance:

```python
from aragora.ranking.performance_integrator import PerformanceEloIntegrator

integrator = PerformanceEloIntegrator(
    performance_monitor=performance_monitor,
    elo_system=elo_system,
    response_quality_weight=0.4,      # Quality score impact
    latency_weight=0.1,               # Faster = slight bonus
    consistency_weight=0.2,           # Low variance = bonus
    participation_weight=0.3,         # Active engagement
    min_calls_for_adjustment=10,      # Min data before adjusting
    k_factor_range=(0.7, 1.5),        # K-factor multiplier bounds
)

# Compute K-factor multipliers for ELO update
multipliers = integrator.compute_k_multipliers(["agent-1", "agent-2"])
# Returns: {"agent-1": 1.2, "agent-2": 0.9}
```

**Performance components:**
- Response quality (calibration, coherence)
- Latency (faster responses = bonus)
- Consistency (low variance = bonus)
- Participation (active engagement)

### Outcome-Memory Bridge

The `OutcomeMemoryBridge` promotes memories that contribute to successful outcomes:

```python
from aragora.memory.outcome_bridge import OutcomeMemoryBridge

bridge = OutcomeMemoryBridge(
    continuum_memory=continuum,
    success_boost_weight=0.1,         # Importance boost per success
    failure_penalty_weight=0.05,      # Importance penalty per failure
    promotion_threshold=3,            # Successes needed for promotion
    demotion_threshold=5,             # Failures needed for demotion
)

# Record memory usage during debate
bridge.record_memory_usage("memory-id", "debate-id")

# Process outcome after debate
result = bridge.process_outcome(outcome)
print(f"Updated: {result.updated}, Promoted: {result.promoted}")
```

**Flow:**
1. Track which memories were used in debate
2. After outcome, update success/failure counts
3. Promote memories with high success rates to faster tiers
4. Demote memories with high failure rates to slower tiers

### Trickster Auto-Calibration

The `TricksterCalibrator` auto-tunes hollow consensus detection sensitivity:

```python
from aragora.debate.trickster_calibrator import TricksterCalibrator

calibrator = TricksterCalibrator(
    trickster=trickster,
    min_samples=20,                   # Min debates before calibrating
    recalibrate_interval=50,          # Debates between recalibrations
    false_positive_tolerance=0.15,    # Max false positive rate
    sensitivity_bounds=(0.3, 0.9),    # Min/max sensitivity
    adjustment_step=0.05,             # Sensitivity adjustment size
)

# Record intervention during debate
calibrator.record_intervention("debate-id", intervention_count=1)

# Record outcome after debate
calibrator.record_debate_outcome(outcome)

# Check if recalibration needed
result = calibrator.maybe_recalibrate()
if result and result.calibrated:
    print(f"Sensitivity: {result.old_sensitivity:.2f} → {result.new_sensitivity:.2f}")
```

**Calibration logic:**
- Too many false positives → lower sensitivity
- Too many misses → raise sensitivity
- Bounded adjustments prevent oscillation

### Checkpoint Memory State

Checkpoints now include memory state for complete debate restoration:

```python
from aragora.debate.checkpoint import CheckpointManager
from aragora.memory.continuum import ContinuumMemory

# Export memory snapshot
memory = ContinuumMemory(db_path="debate.db")
snapshot = memory.export_snapshot(tiers=[MemoryTier.FAST, MemoryTier.MEDIUM])

# Create checkpoint with memory
checkpoint = await manager.create_checkpoint(
    debate_id="debate-123",
    task="Complex analysis",
    current_round=5,
    total_rounds=10,
    messages=messages,
    continuum_memory_state=snapshot,  # Include memory snapshot
)

# Later, restore checkpoint
resumed = await manager.resume_from_checkpoint(checkpoint.checkpoint_id)

# Restore memory state
new_memory = ContinuumMemory(db_path="restored.db")
result = new_memory.restore_snapshot(resumed.checkpoint.continuum_memory_state)
print(f"Restored {result['restored']} memory entries")
```

### Data Flow Diagram

```
                    ┌─────────────────────────────────────────┐
                    │            Debate Lifecycle              │
                    └─────────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
        ┌─────────────────────┐              ┌─────────────────────┐
        │   HookManager       │◄────────────►│  HookHandlerRegistry │
        │   (Event Bus)       │              │  (Auto-wiring)       │
        └─────────────────────┘              └─────────────────────┘
                    │
    ┌───────────────┼───────────────┬───────────────┐
    ▼               ▼               ▼               ▼
┌────────┐   ┌────────────┐   ┌──────────┐   ┌───────────┐
│Analytics│   │Performance │   │ Trickster │   │ Memory    │
│Coord.  │   │ Monitor    │   │ Detector  │   │ Systems   │
└────────┘   └────────────┘   └──────────┘   └───────────┘
                    │               │               │
                    ▼               ▼               ▼
        ┌─────────────────┐  ┌──────────────┐ ┌──────────────┐
        │ Performance-ELO │  │  Trickster   │ │ Outcome-Mem  │
        │  Integrator     │  │  Calibrator  │ │   Bridge     │
        └─────────────────┘  └──────────────┘ └──────────────┘
                    │               │               │
                    ▼               ▼               ▼
              ┌───────────┐   ┌──────────┐   ┌──────────────┐
              │ ELO       │   │ Adjusted │   │ Promoted/    │
              │ K-Factors │   │Sensitivity│   │ Demoted Mem  │
              └───────────┘   └──────────┘   └──────────────┘
```

### Metrics

New metrics for Phase 8 features:

- `aragora_hook_handlers_registered_total` - Registered hook handlers
- `aragora_performance_elo_adjustments_total` - ELO K-factor modulations
- `aragora_outcome_memory_promotions_total` - Memory tier promotions
- `aragora_outcome_memory_demotions_total` - Memory tier demotions
- `aragora_trickster_calibrations_total` - Sensitivity recalibrations
- `aragora_checkpoint_memory_snapshots_total` - Memory snapshots created

### Debug Logging

```python
import logging
logging.getLogger("aragora.debate.hook_handlers").setLevel(logging.DEBUG)
logging.getLogger("aragora.ranking.performance_integrator").setLevel(logging.DEBUG)
logging.getLogger("aragora.memory.outcome_bridge").setLevel(logging.DEBUG)
logging.getLogger("aragora.debate.trickster_calibrator").setLevel(logging.DEBUG)
```

---

## Phase 9: Cross-Pollination Feedback Loops

Phase 9 completes the cross-pollination by adding bidirectional feedback between subsystems.

### Overview

| Feature | Purpose | Default |
|---------|---------|---------|
| Calibration → Proposals | Temperature scaling for proposal confidence | Auto-enabled with CalibrationTracker |
| Learning Efficiency → ELO | Agents who learn quickly get ELO bonuses | Auto-enabled with EloSystem |
| Verification → Vote Confidence | Verification results adjust vote weights | Auto-enabled |
| Voting Accuracy → ELO | Correct votes boost ELO | Auto-enabled |
| Adaptive Debate Rounds | Memory tiers inform debate length | `enable_adaptive_rounds=True` |
| RLM Compression Caching | Reuse compression hierarchies | `enable_caching=True` |

### Configuration

```python
from aragora.debate.arena_config import ArenaConfig

config = ArenaConfig(
    # Calibration scaling (auto-enabled with calibration_tracker)
    enable_calibration=True,

    # Adaptive rounds from memory strategy
    enable_adaptive_rounds=True,

    # ELO system with learning efficiency
    enable_elo_voting_accuracy=True,  # Track voting patterns
    enable_elo_learning_bonus=True,   # Bonus for consistent learners
)
```

### Calibration → Proposals

Proposal confidence values are adjusted based on agent calibration history:

```python
from aragora.debate.phases.proposal_phase import ProposalPhase
from aragora.agents.calibration import CalibrationTracker

# Calibration tracker with history
tracker = CalibrationTracker(db_path="calibration.db")

# ProposalPhase uses calibration automatically
phase = ProposalPhase(calibration_tracker=tracker)

# Raw confidence 0.7 might become 0.55 for overconfident agents
# or 0.82 for underconfident agents
calibrated = phase._get_calibrated_confidence("agent_name", 0.7, ctx)
```

**How it works:**
1. Agent submits proposal with implicit 0.7 confidence
2. CalibrationTracker looks up agent's temperature scaling
3. If agent has 10+ predictions, temperature scaling is applied
4. Adjusted confidence is stored with the position

### Learning Efficiency → ELO

Agents who improve consistently get ELO bonuses:

```python
from aragora.ranking.elo import EloSystem

elo = EloSystem(db_path="elo.db")

# Get learning efficiency metrics
efficiency = elo.get_learning_efficiency("agent_name", domain="coding")
# Returns:
# {
#   "elo_gain_rate": 3.5,       # Average ELO gain per debate
#   "consistency_score": 0.75,  # How steady the improvement is
#   "learning_category": "rapid",  # rapid/steady/slow/declining
#   "has_meaningful_data": True
# }

# Apply learning bonus (called automatically in FeedbackPhase)
bonus = elo.apply_learning_bonus(
    agent_name="agent_name",
    domain="coding",
    debate_id="debate-123",
)
```

**Learning categories:**
- `rapid`: gain_rate > 5, consistency > 0.6 → 1.5x bonus
- `steady`: gain_rate > 2, consistency > 0.5 → 0.75x bonus
- `slow`: gain_rate > 0 → 0.25x bonus
- `declining`: No bonus

### Verification → Vote Confidence

Verified/disproven proposals adjust vote confidence:

```python
from aragora.debate.phases.consensus_verification import ConsensusVerifier

verifier = ConsensusVerifier()

# After verification phase
votes = [Vote(choice="agent_a", confidence=0.7)]
verification_results = {"agent_a": {"verified": 1, "disproven": 0}}

verifier.adjust_vote_confidence_from_verification(
    votes, verification_results, proposals
)

# Verified: confidence * 1.3 = 0.91 (capped at 0.99)
# Disproven: confidence * 0.3 = 0.21 (floored at 0.01)
```

### Voting Accuracy → ELO

Agents who vote for the consensus winner get small ELO bonuses:

```python
from aragora.ranking.elo import EloSystem

elo = EloSystem(db_path="elo.db")

# Record voting accuracy
elo.update_voting_accuracy(
    agent_name="agent_a",
    voted_for_consensus=True,
    domain="general",
    debate_id="debate-123",
    apply_elo_bonus=True,
)

# Get voting accuracy stats
stats = elo.get_voting_accuracy("agent_a")
# Returns:
# {
#   "total_votes": 25,
#   "correct_votes": 20,
#   "accuracy": 0.8,
#   "has_meaningful_data": True
# }
```

### Adaptive Debate Rounds

Memory tiers inform optimal debate length:

```python
from aragora.debate.strategy import DebateStrategy
from aragora.memory.continuum import ContinuumMemory

memory = ContinuumMemory(db_path="memory.db")
strategy = DebateStrategy(continuum_memory=memory)

# Estimate optimal rounds based on memory
rec = strategy.estimate_rounds("Design a caching strategy")
# Returns StrategyRecommendation:
# {
#   "estimated_rounds": 2,  # Quick validation (glacial memory found)
#   "confidence": 0.85,
#   "reasoning": "High-confidence memory found for similar task",
#   "relevant_memories": ["mem_123", "mem_456"]
# }
```

**Round estimation logic:**
- Glacial memory with > 0.9 confidence → 2 rounds (quick validation)
- Any glacial memory → 3 rounds (standard debate)
- No relevant memory → 5 rounds (exploration debate)

### RLM Compression Caching

Reuse compression hierarchies across debates:

```python
from aragora.rlm.bridge import AragoraRLM, RLMHierarchyCache

# Create cache (uses knowledge mound if available)
cache = RLMHierarchyCache(knowledge_mound=mound)

rlm = AragoraRLM(
    hierarchy_cache=cache,
    enable_caching=True,
)

# Compress - cache hit if similar content was compressed before
result = await rlm.compress_and_query(
    messages=messages,
    query="What's the consensus?",
)

# Check cache stats
stats = cache.stats
# Returns:
# {
#   "hits": 15,
#   "misses": 5,
#   "hit_rate": 0.75,
#   "local_cache_size": 20
# }
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Proposal Phase                            │
│  CalibrationTracker ──► Temperature Scaling ──► Proposal Conf.  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Debate Rounds                             │
│  DebateStrategy ──► Memory Query ──► Adaptive Round Count       │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Consensus Phase                            │
│  ConsensusVerifier ──► Verification ──► Vote Confidence Adjust  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Feedback Phase                             │
│  EloSystem ──► Voting Accuracy ──► ELO Bonus                    │
│  EloSystem ──► Learning Efficiency ──► ELO Bonus                │
└─────────────────────────────────────────────────────────────────┘
```

### Metrics

New metrics for Phase 9 features:

- `aragora_calibration_adjustments_total` - Confidence calibrations applied
- `aragora_learning_bonuses_total` - Learning efficiency bonuses applied
- `aragora_verification_confidence_adjustments_total` - Vote confidence changes
- `aragora_voting_accuracy_updates_total` - Voting accuracy records
- `aragora_adaptive_round_changes_total` - Round count adjustments
- `aragora_rlm_cache_hits_total` - Compression cache hits

### Debug Logging

```python
import logging
logging.getLogger("aragora.debate.phases.proposal_phase").setLevel(logging.DEBUG)
logging.getLogger("aragora.debate.phases.feedback_phase").setLevel(logging.DEBUG)
logging.getLogger("aragora.debate.strategy").setLevel(logging.DEBUG)
logging.getLogger("aragora.ranking.elo").setLevel(logging.DEBUG)
logging.getLogger("aragora.rlm.bridge").setLevel(logging.DEBUG)
```
