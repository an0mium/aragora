# Debate Orchestration Architecture

This document describes the orchestration architecture for Aragora's multi-agent debate system.

## Overview

The debate orchestration system coordinates AI agents through structured debate phases, managing proposal generation, critique exchange, revision cycles, and consensus formation. The architecture follows a **phase-based execution model** with clean separation of concerns through context passing and subsystem bridges.

**Key Stats:**
- 40+ integrated subsystems
- 7 execution phases
- 9 cross-pollination bridges
- 3-tier convergence detection

## Core Components

### Arena (`aragora/debate/orchestrator.py`)

The `Arena` class is the central orchestrator managing the complete debate lifecycle.

```python
from aragora import Arena, Environment, DebateProtocol

env = Environment(task="Design a rate limiter for API endpoints")
protocol = DebateProtocol(rounds=3, consensus="majority")
arena = Arena(env, agents, protocol)
result = await arena.run()
```

**Responsibilities:**
- Initialize and coordinate 40+ subsystems
- Execute debate phases sequentially
- Handle timeouts and error recovery
- Integrate telemetry and observability
- Manage agent lifecycle and channels

**Key Methods:**
- `run(correlation_id)` - Public entry point with timeout handling
- `_run_inner(correlation_id)` - Internal orchestrator coordinating all phases

### DebateContext (`aragora/debate/context.py`)

Immutable container passed between all phases for state management.

```python
@dataclass
class DebateContext:
    # Immutable inputs
    env: Environment
    agents: List[Agent]
    debate_id: str
    correlation_id: str
    domain: str
    org_id: str

    # Result storage
    result: DebateResult

    # Phase-specific state
    proposals: List[Message]
    critiques: List[Critique]
    votes: List[Vote]
    messages: List[Message]

    # Agent workspaces
    workspaces: Dict[str, AgentWorkspace]
```

**Pattern:** Phases read from context and write state back without tight coupling to Arena.

## Phase Architecture

### PhaseExecutor (`aragora/debate/phase_executor.py`)

Orchestrates sequential phase execution with error handling and timeouts.

```
Total timeout: 600s (configurable)
Per-phase timeout: 120s (configurable)
Critical phases: consensus (must run even if earlier phases fail)
Optional phases: analytics, feedback (can be skipped on timeout)
```

### Phase Sequence

| Phase | Name | Description | Class |
|-------|------|-------------|-------|
| 0 | Context Initialization | Gather context, patterns, evidence | `ContextInitializer` |
| 1 | Proposal | Initial proposer responses | `ProposalPhase` |
| 2 | Debate Rounds | Critique/revision cycles | `DebateRoundsPhase` |
| 3 | Consensus | Voting and agreement | `ConsensusPhase` |
| 4-6 | Analytics | Metrics and insights | `AnalyticsPhase` |
| 7 | Feedback | Memory and updates | `FeedbackPhase` |

### Execution Flow

```
Arena._run_inner()
  │
  ├─ Initialize convergence detector
  ├─ Extract domain from task
  ├─ Initialize Knowledge Mound context
  ├─ Apply culture-informed protocol adjustments
  ├─ Create DebateContext
  ├─ Classify task complexity
  ├─ Apply performance-based agent selection
  ├─ Assign hierarchy roles (Gastown pattern)
  ├─ Setup agent-to-agent channels
  ├─ Check budget constraints
  ├─ Initialize GUPP hook tracking
  │
  ▼
PhaseExecutor.execute(ctx)
  ├─► Phase 0: ContextInitializer.execute(ctx)
  ├─► Phase 1: ProposalPhase.execute(ctx)
  ├─► Phase 2: DebateRoundsPhase.execute(ctx)
  ├─► Phase 3: ConsensusPhase.execute(ctx)
  ├─► Phase 4-6: AnalyticsPhase.execute(ctx)
  └─► Phase 7: FeedbackPhase.execute(ctx)
  │
  ▼
ctx.finalize_result()
```

## Phase Details

### Phase 0: Context Initialization

**File:** `aragora/debate/phases/context_init.py`

Prepares debate context by gathering relevant information:
- Load historical context from memory
- Retrieve trending topics from Pulse
- Collect evidence from sources
- Inject Knowledge Mound context
- Apply culture hints

### Phase 1: Proposal (`aragora/debate/phases/proposal_phase.py`)

Generates initial proposals from agents:
- Position tracking and recording
- Role assignment updates
- Propulsion engine integration
- Concurrent proposal generation

### Phase 2: Debate Rounds (`aragora/debate/phases/debate_rounds.py`)

Main debate loop with critique and revision cycles:
- Critique generation (concurrent per agent)
- Revision generation based on critiques
- Convergence detection for early termination
- Trickster (hollow consensus detection)
- Rhetorical observer (pattern detection)
- RLM compression (context optimization after round 3)
- Evidence refresh during rounds

**Concurrency Settings:**
```python
MAX_CONCURRENT_PROPOSALS = 5
MAX_CONCURRENT_CRITIQUES = 10
MAX_CONCURRENT_REVISIONS = 5
```

### Phase 3: Consensus (`aragora/debate/phases/consensus_phase.py`)

Voting and agreement resolution:
- Voting orchestration
- Judge selection and evaluation
- Consensus proof generation
- Claim verification (Z3-backed formal verification)

### Phases 4-6: Analytics (`aragora/debate/phases/analytics_phase.py`)

Metrics computation and insights:
- Grounded verdict creation (with citations)
- Disagreement reporting
- Relationship updates
- Performance metrics

### Phase 7: Feedback (`aragora/debate/phases/feedback_phase.py`)

Memory storage and updates:
- ELO system updates
- Persona evolution
- Position ledger updates
- Knowledge Mound ingestion
- Post-debate workflow triggers
- Broadcast pipeline integration

## Consensus & Proofs

### ConsensusProof (`aragora/debate/consensus.py`)

Full audit trail for decision verification:

```python
@dataclass
class ConsensusProof:
    # Core
    final_claim: str
    confidence: float
    consensus_reached: bool

    # Voting
    votes: List[ConsensusVote]
    supporting_agents: List[str]
    dissenting_agents: List[str]

    # Records
    claims: List[Claim]
    dissents: List[DissentRecord]
    unresolved_tensions: List[UnresolvedTension]

    # Provenance
    evidence_chain: List[Evidence]
    reasoning_summary: str
```

**Key Properties:**
- `checksum` - SHA-256 hash for proof integrity
- `agreement_ratio` - Proportion of supporting agents
- `has_strong_consensus` - >80% agreement AND >0.7 confidence
- `get_dissent_summary()` - Markdown dissent summary
- `get_blind_spots()` - Identify missing perspectives

### Evidence Types

```python
class Evidence:
    evidence_id: str
    source: str  # agent or tool
    content: str
    evidence_type: str  # argument/data/citation/tool_output
    supports_claim: bool
    strength: float  # 0-1
```

## Convergence Detection

### ConvergenceDetector (`aragora/debate/convergence.py`)

Semantic convergence detection for early termination when agents' positions align.

**3-Tier Fallback Strategy:**
1. **SentenceTransformer** - Best accuracy (requires sentence-transformers)
2. **TF-IDF** - Good accuracy (requires scikit-learn)
3. **Jaccard** - Always available (zero dependencies)

**Advanced Metrics (G3 Framework):**

| Metric | Purpose | Weight |
|--------|---------|--------|
| Semantic Similarity | Text similarity | 40% |
| Argument Diversity | Unique vs total arguments | 20% |
| Evidence Convergence | Shared citations | 20% |
| Stance Volatility | Position changes | 20% |

```python
@dataclass
class ConvergenceResult:
    converged: bool
    status: str  # converged/diverging/refining
    min_similarity: float
    avg_similarity: float
    per_agent_similarity: Dict[str, float]
    consecutive_stable_rounds: int
```

## Recovery & Resilience

### RecoveryCoordinator (`aragora/debate/recovery_coordinator.py`)

Coordinates recovery actions for debate issues (stalls, deadlocks, failures).

**Recovery Actions:**

| Action | Description |
|--------|-------------|
| `NUDGE` | Send reminder/prompt to agent |
| `REPLACE` | Replace agent with another |
| `SKIP` | Skip agent's turn |
| `RESET_ROUND` | Reset current round |
| `FORCE_VOTE` | Force early voting |
| `INJECT_MEDIATOR` | Add mediating agent |
| `ESCALATE` | Escalate to human |
| `ABORT` | Abort debate |

**Configuration:**
```python
@dataclass
class RecoveryConfig:
    # Agent replacement
    max_agent_failures: int = 3
    max_agent_stalls: int = 2
    replacement_pool: List[str] = []

    # Deadlock strategies
    cycle_resolution_strategy: str = "inject_mediator"
    mutual_block_strategy: str = "nudge"
    semantic_loop_strategy: str = "nudge"
    convergence_failure_strategy: str = "force_vote"
```

### DebateWitness (`aragora/debate/witness.py`)

Observer that monitors debate progress and detects issues:
- Per-agent activity tracking
- Stall detection (timeout, repeated content, circular arguments)
- Progress status monitoring

**Stall Reasons:**
- `TIMEOUT` - No response within threshold
- `REPEATED_CONTENT` - Content too similar to previous
- `CIRCULAR_ARGUMENTS` - Arguments forming a cycle
- `NO_PROGRESS` - Round not advancing
- `AGENT_FAILURE` - Agent threw an error

### DeadlockDetector (`aragora/debate/deadlock_detector.py`)

Detects deadlock patterns in debates:

| Deadlock Type | Description |
|---------------|-------------|
| `CYCLE` | Circular reference in arguments |
| `MUTUAL_BLOCK` | Two agents blocking each other |
| `SEMANTIC_LOOP` | Repeated argument patterns |
| `CONVERGENCE_FAILURE` | No progress toward consensus |

## Subsystem Coordination

### SubsystemCoordinator (`aragora/debate/subsystem_coordinator.py`)

Centralized management of optional tracking and detection subsystems.

**Subsystem Groups:**

1. **Position Systems**
   - `position_tracker` - Real-time stance tracking
   - `position_ledger` - Persistent position history

2. **Agent Ranking**
   - `elo_system` - Agent skill ratings
   - `calibration_tracker` - Prediction accuracy

3. **Memory Systems**
   - `consensus_memory` - Historical outcomes
   - `dissent_retriever` - Minority viewpoints
   - `continuum_memory` - Cross-debate learning

4. **Detection Systems**
   - `flip_detector` - Position reversals
   - `moment_detector` - Significant moments

5. **Relationship Systems**
   - `relationship_tracker` - Agent interactions
   - `tier_analytics_tracker` - Memory tier ROI

## Cross-Pollination Bridges

Bridges connect subsystems for feedback loops:

| Bridge | Source | Target | Purpose |
|--------|--------|--------|---------|
| Performance Router | PerformanceMonitor | AgentRouter | Performance → selection |
| Outcome Complexity | OutcomeTracker | ComplexityGovernor | Complexity → timeout |
| Analytics Selection | AnalyticsPhase | TeamSelector | Metrics → composition |
| Novelty Selection | NoveltyTracker | AgentSelector | Staleness → diversity |
| Relationship Bias | RelationshipTracker | AgentRouter | Relationships → teams |
| Checkpoint | Debate state | Storage | Save/restore checkpoints |
| Event | Event bus | Spectators | Distributed notifications |
| KM Outcome | DebateResult | KnowledgeMound | Consensus → knowledge |

## Knowledge Management

### ArenaKnowledgeManager (`aragora/debate/knowledge_manager.py`)

Manages Knowledge Mound operations:
- Knowledge retrieval before debates (seed context)
- Outcome ingestion after debates (store consensus)
- Auto-revalidation for stale knowledge
- Culture-informed protocol adjustments

**Culture Hints:**
- Preferred consensus approach
- Additional critique rounds
- Early termination threshold
- Domain-specific strategies

## Telemetry & Observability

### OpenTelemetry Integration

```python
# Span creation
with tracer.start_as_current_span("debate") as span:
    span.set_attribute("debate_id", debate_id)
    span.set_attribute("domain", domain)
    span.set_attribute("complexity", complexity)
    span.set_attribute("agent_count", len(agents))
```

**Metrics Tracked:**
- `ACTIVE_DEBATES` - Counter of concurrent debates
- Debate outcome tracking
- Circuit breaker metrics
- Per-phase performance

**Structured Logging:**
- `debate_start` event - debate_id, complexity, agent_count, domain
- `debate_end` event - status, duration, consensus, confidence, rounds
- Correlation context for distributed tracing

## Error Handling

### Timeout Recovery

```python
try:
    result = await asyncio.wait_for(
        phase_executor.execute(ctx),
        timeout=protocol.timeout_seconds
    )
except asyncio.TimeoutError:
    # Return partial results
    return ctx.partial_result()
```

### Crash Recovery (GUPP)

The system uses hook tracking for crash recovery:
1. Create pending debate bead
2. Track incomplete operations
3. Complete hook tracking on success
4. Recover from checkpoints on restart

## Configuration

### Protocol Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `rounds` | Number of debate rounds | 3 |
| `timeout_seconds` | Debate timeout | 600 |
| `judge_selection` | Judge selection strategy | "auto" |
| `consensus_mechanism` | Consensus method | "majority" |
| `verify_claims_during_consensus` | Enable Z3 verification | false |
| `enable_hook_tracking` | GUPP crash recovery | true |
| `enable_km_belief_sync` | KM → Belief Network | false |
| `enable_rhetorical_observer` | Pattern detection | false |
| `enable_trickster` | Echo chamber detection | false |

### ArenaConfig

```python
@dataclass
class ArenaConfig:
    # Agent pool and selection
    agents: List[Agent]
    agent_selector: Optional[AgentSelector] = None

    # Memory systems
    critique_store: Optional[CritiqueStore] = None
    continuum_memory: Optional[ContinuumMemory] = None

    # Knowledge systems
    knowledge_mound: Optional[KnowledgeMound] = None

    # Tracking systems
    elo_system: Optional[ELOSystem] = None
    calibration_tracker: Optional[CalibrationTracker] = None

    # Advanced features
    enable_checkpointing: bool = False
    enable_performance_monitoring: bool = True
```

## Design Patterns

| Pattern | Usage |
|---------|-------|
| **Context** | DebateContext passed between phases |
| **Phase** | Sequential phase execution via PhaseExecutor |
| **Bridge** | Subsystems connected via bridges |
| **Observer** | Witness observes progress, Recovery reacts |
| **Factory** | ArenaInitializer creates subsystems |
| **Strategy** | Adaptive rounds, team selection |
| **Coordinator** | SubsystemCoordinator manages optional systems |
| **Circuit Breaker** | Handles agent failures gracefully |

## File Reference

| Component | File | Key Classes |
|-----------|------|-------------|
| Main Orchestrator | `orchestrator.py` | Arena |
| Phase Execution | `phase_executor.py` | PhaseExecutor |
| Phase Initialization | `arena_phases.py` | init_phases() |
| Shared Context | `context.py` | DebateContext |
| Consensus Proofs | `consensus.py` | ConsensusProof |
| Convergence | `convergence.py` | ConvergenceDetector |
| Recovery | `recovery_coordinator.py` | RecoveryCoordinator |
| Subsystems | `subsystem_coordinator.py` | SubsystemCoordinator |
| Knowledge | `knowledge_manager.py` | ArenaKnowledgeManager |
| Witness | `witness.py` | DebateWitness |
| Deadlock | `deadlock_detector.py` | DeadlockDetector |

## See Also

- [Architecture Overview](./ARCHITECTURE.md)
- [Agent Development](./core-concepts/agent-development.md)
- [Memory Systems](./core-concepts/memory.md)
- [Knowledge Mound](./core-concepts/knowledge-mound.md)
- [Control Plane](./enterprise/control-plane.md)
