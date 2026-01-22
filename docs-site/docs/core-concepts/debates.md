---
title: Debate Phases Architecture
description: Debate Phases Architecture
---

# Debate Phases Architecture

This document describes the debate execution phases, their state transitions, and consensus requirements.

## Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEBATE EXECUTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Phase 0          Phase 1           Phase 2           Phase 3              │
│  ┌─────────┐     ┌───────────┐     ┌────────────┐    ┌────────────┐        │
│  │ Context │────▶│ Proposals │────▶│  Debate    │───▶│ Consensus  │        │
│  │  Init   │     │           │     │  Rounds    │    │            │        │
│  └─────────┘     └───────────┘     └────────────┘    └────────────┘        │
│       │               │                  │                 │                │
│       ▼               ▼                  ▼                 ▼                │
│   - History       - Parallel        - Critique       - Voting              │
│   - Patterns      - Position        - Revision       - Judge               │
│   - Research      - Streaming       - Convergence    - Termination         │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 4-6         Phase 7                                                  │
│  ┌────────────┐   ┌────────────┐                                           │
│  │ Analytics  │──▶│ Feedback   │                                           │
│  │            │   │            │                                           │
│  └────────────┘   └────────────┘                                           │
│       │                │                                                    │
│       ▼                ▼                                                    │
│   - Metrics        - ELO Update                                            │
│   - Insights       - Persona                                               │
│   - Verification   - Memory                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 0: Context Initialization

**Module:** `aragora/debate/phases/context_init.py`

### Purpose
Prepares the debate context with historical data, patterns, and research.

### State Transitions
```
START → INITIALIZING → CONTEXT_READY
```

### Key Operations
1. **History Injection**: Fetches relevant past debates from `ConsensusMemory`
2. **Pattern Loading**: Retrieves successful critique patterns from `CritiqueStore`
3. **Research Fetch**: Optional web/document research via connectors
4. **Trending Topics**: Injects current pulse data if enabled

### Configuration
```python
ContextInitializer(
    consensus_memory=...,      # Historical debate outcomes
    critique_store=...,        # Successful patterns
    research_enabled=True,     # Enable evidence collection
    history_limit=5,           # Max historical debates
)
```

## Phase 1: Proposals

**Module:** `aragora/debate/phases/proposal_phase.py`

### Purpose
Generates initial proposals from all proposer agents in parallel.

### State Transitions
```
CONTEXT_READY → GENERATING_PROPOSALS → PROPOSALS_COMPLETE
```

### Execution Pattern
- Uses `asyncio.as_completed()` for parallel generation
- Streams output as each agent finishes
- Circuit breaker filters unavailable agents

### Key Operations
1. Filter proposers via circuit breaker
2. Create parallel generation tasks
3. Process results as they complete
4. Track positions for grounded personas
5. Extract citation needs

### Parallelization
```python
tasks = [asyncio.create_task(generate_proposal(agent)) for agent in proposers]
for completed in asyncio.as_completed(tasks):
    agent, result = await completed
    process_proposal(agent, result)
```

## Phase 2: Debate Rounds

**Module:** `aragora/debate/phases/debate_rounds.py`

### Purpose
Executes the critique → revision loop until consensus or max rounds.

### State Transitions
```
PROPOSALS_COMPLETE → ROUND_N_CRITIQUE → ROUND_N_REVISION →
    [CONVERGENCE_CHECK] → ROUND_N+1_CRITIQUE | CONSENSUS_PHASE
```

### Round Structure
```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Critique   │─────▶│  Revision   │─────▶│ Convergence │
│   Phase     │      │   Phase     │      │   Check     │
└─────────────┘      └─────────────┘      └─────────────┘
      │                     │                    │
      ▼                     ▼                    ▼
 - Parallel            - Parallel           - Semantic
   critique              revision             similarity
 - Topology            - Position           - Early
   routing               tracking             stopping
 - Circuit             - Streaming          - Judge
   breaker                                    termination
```

### Critique Topology Options
- **all-to-all**: Every critic reviews every proposal
- **round-robin**: Deterministic rotation based on agent name hash
- **ring**: Neighbors critique each other
- **star**: Hub agent critiques all, or all critique hub
- **sparse/random-graph**: Probabilistic subset based on sparsity

### Convergence Detection
```python
class ConvergenceDetector:
    def check_convergence(self, proposals: dict[str, str]) -> bool:
        # Compute semantic similarity between all pairs
        # Return True if average similarity > threshold
```

### Early Termination
- **Judge Termination**: Expert agent determines sufficient discussion
- **Early Stopping**: Confidence threshold exceeded
- **Max Rounds**: Protocol limit reached

## Phase 3: Consensus

**Module:** `aragora/debate/phases/consensus_phase.py`

### Purpose
Resolves the debate through voting and determines final answer.

### State Transitions
```
ROUNDS_COMPLETE → VOTING → VOTE_AGGREGATION → CONSENSUS_REACHED | NO_CONSENSUS
```

### Voting Mechanisms
1. **Majority**: Simple majority wins
2. **Supermajority**: 2/3 required
3. **Unanimous**: All must agree
4. **Weighted**: ELO/calibration-weighted votes

### Vote Weighting
```python
class VoteWeightCalculator:
    def calculate_weight(self, agent_name: str) -> float:
        # Base weight from ELO ranking
        elo_weight = (agent.elo - 1000) / 500
        # Calibration bonus (0.5-1.5 multiplier)
        cal_weight = 0.5 + agent.calibration_score
        return elo_weight * cal_weight
```

### Consensus Strength Levels
```python
class ConsensusStrength(Enum):
    UNANIMOUS = "unanimous"    # All agents agreed
    STRONG = "strong"          # >80% agreement
    MODERATE = "moderate"      # 60-80% agreement
    WEAK = "weak"              # 50-60% agreement
    SPLIT = "split"            # No majority
    CONTESTED = "contested"    # Active disagreement
```

## Phases 4-6: Analytics

**Module:** `aragora/debate/phases/analytics_phase.py`

### Purpose
Post-consensus analysis and reporting.

### Key Operations
1. **Pattern Tracking**: Record failed patterns for learning
2. **Metrics Recording**: Duration, rounds, outcome
3. **Insight Extraction**: Key learnings from debate
4. **Relationship Updates**: Agent interaction metrics
5. **Disagreement Report**: Document dissenting views
6. **Grounded Verdict**: Evidence-backed conclusion
7. **Formal Verification**: Optional Z3/Lean proofs
8. **Belief Analysis**: Propagation network updates

## Phase 7: Feedback

**Module:** `aragora/debate/phases/feedback_phase.py`

### Purpose
Updates persistent systems with debate outcomes.

### Feedback Loops
1. **ELO Update**: Record match results
2. **Persona Performance**: Update agent traits
3. **Position Resolution**: Resolve ledger positions
4. **Relationship Metrics**: Update interaction weights
5. **Moment Detection**: Identify narrative moments
6. **Debate Indexing**: Add to embeddings database
7. **Flip Detection**: Track position reversals
8. **Memory Storage**: Store in ContinuumMemory

## Circuit Breaker Integration

**Module:** `aragora/resilience.py`

### States
```
CLOSED → (failures >= threshold) → OPEN → (cooldown elapsed) → HALF_OPEN
                                                                    │
                    ┌──────────────────────────────────────────────┘
                    │
                    ▼
            (trial success) → CLOSED
            (trial failure) → OPEN
```

### Configuration
```python
CircuitBreaker(
    failure_threshold=3,           # Consecutive failures to open
    cooldown_seconds=60.0,         # Time before half-open
    half_open_success_threshold=2, # Successes to fully close
)
```

### Usage in Phases
```python
# Filter available agents
available = circuit_breaker.filter_available_agents(agents)

# Record outcomes
try:
    result = await agent.generate(prompt)
    circuit_breaker.record_success(agent.name)
except Exception:
    circuit_breaker.record_failure(agent.name)
```

## Event Emission

Throughout all phases, events are emitted for real-time monitoring:

| Event Type | Phase | Description |
|------------|-------|-------------|
| `debate_start` | 1 | Debate initiated |
| `round_start` | 2 | Round N beginning |
| `agent_message` | 1,2 | Agent response |
| `critique` | 2 | Critique issued |
| `vote` | 3 | Vote cast |
| `consensus` | 3 | Consensus reached |
| `debate_end` | 4-6 | Debate complete |

## Error Handling

### Timeout Protection
All agent operations wrapped with configurable timeout:
```python
async def _with_timeout(self, coro, agent_name: str, timeout: float = 90.0):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        self.circuit_breaker.record_failure(agent_name)
        raise
```

### Graceful Degradation
- Circuit breaker skips failing agents
- Partial results returned on timeout
- Fallback providers (OpenRouter) for quota errors

## See Also

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System overview
- [MEMORY_STRATEGY.md](../MEMORY_STRATEGY.md) - Memory tier details
- [API_REFERENCE.md](../API_REFERENCE.md) - HTTP/WebSocket API
