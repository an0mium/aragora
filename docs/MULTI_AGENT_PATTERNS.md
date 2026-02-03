# Multi-Agent Patterns Guide

This guide covers the multi-agent orchestration patterns available in Aragora, including debate modes, agent composition strategies, workflow patterns, and common anti-patterns.

For the lower-level orchestration architecture, see [ORCHESTRATION.md](./ORCHESTRATION.md).

---

## Table of Contents

1. [Debate Modes](#debate-modes)
2. [Agent Team Composition](#agent-team-composition)
3. [Workflow Patterns](#workflow-patterns)
4. [Consensus Mechanisms](#consensus-mechanisms)
5. [Memory & Learning Patterns](#memory--learning-patterns)
6. [Resilience Patterns](#resilience-patterns)
7. [Common Anti-Patterns](#common-anti-patterns)
8. [Pattern Selection Guide](#pattern-selection-guide)

---

## Debate Modes

Aragora supports three debate modes, each suited to different problem types.

### Standard Linear Debate

The default mode. Agents take turns proposing, critiquing, and revising through a fixed number of rounds.

**Best for:** Well-defined questions with clear evaluation criteria.

```python
from aragora import Arena, Environment, DebateProtocol

env = Environment(task="Which caching strategy should we use?")
protocol = DebateProtocol(rounds=3, consensus="majority")
arena = Arena(env, agents, protocol)
result = await arena.run()
```

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rounds` | 3 | Number of debate rounds |
| `timeout_seconds` | 600 | Total debate timeout |
| `consensus_mechanism` | `"majority"` | How consensus is determined |
| `judge_selection` | `"auto"` | Judge selection strategy |

**When to use:**
- Binary or well-scoped decisions
- When you need fast turnaround (1-3 rounds)
- Questions where agent perspectives are complementary

### Graph Debates (Branching)

When agents disagree beyond a threshold, the debate branches into parallel tracks exploring different perspectives, then merges results.

**Best for:** Complex decisions where disagreement reveals genuinely different valid approaches.

```python
from aragora.debate.graph_orchestrator import GraphDebateOrchestrator
from aragora.debate.graph import BranchPolicy, MergeStrategy

policy = BranchPolicy(
    min_disagreement=0.6,      # Branch at 60% disagreement
    max_branches=4,
    auto_merge=True,
    merge_strategy=MergeStrategy.SYNTHESIS,
    convergence_threshold=0.8,
)

orchestrator = GraphDebateOrchestrator(env, agents, policy)
result = await orchestrator.run()
```

**Merge strategies:**

| Strategy | Description | Use when |
|----------|-------------|----------|
| `SYNTHESIS` | Combines insights from all branches into unified position | Branches explore complementary aspects |
| `VOTE` | Selects the branch with most agent support | Branches are mutually exclusive options |
| `BEST_EVIDENCE` | Selects the branch with highest evidence quality | Evidence strength varies significantly |

**When to use:**
- Architecture decisions with multiple viable approaches
- Policy questions where trade-offs matter
- Any question where "it depends" is the honest answer

**API:**
```
POST /api/v2/debates/graph
GET  /api/v2/debates/graph/{id}
GET  /api/v2/debates/graph/{id}/branches
```

### Matrix Debates (Scenario-Based)

Runs the same question across multiple scenarios in parallel, then compares results to identify universal vs conditional conclusions.

**Best for:** Decisions that depend heavily on context or assumptions.

```python
# Define scenarios
scenarios = [
    Scenario(
        name="High-write workload",
        parameters={"writes_per_second": 100000, "data_size_tb": 50},
        constraints=["Must support ACID transactions"],
        is_baseline=True,
    ),
    Scenario(
        name="Read-heavy analytics",
        parameters={"reads_per_second": 500000, "data_size_tb": 200},
        constraints=["Sub-second query latency"],
    ),
]
```

**Output structure:**
- **Universal conclusions:** True across all scenarios (high confidence)
- **Conditional conclusions:** Context-specific recommendations with scenario tags

**When to use:**
- "Which database should we use?" (depends on workload)
- Capacity planning across growth scenarios
- Risk assessment under different market conditions

**API:**
```
POST /api/v2/debates/matrix
GET  /api/v2/debates/matrix/{id}/scenarios
GET  /api/v2/debates/matrix/{id}/conclusions
```

---

## Agent Team Composition

### Team Selection System

The `TeamSelector` (`aragora/debate/team_selector.py`) scores agents using weighted factors:

| Factor | Weight | Source |
|--------|--------|--------|
| Agent CV (unified profile) | 0.35 | Composite capability scores |
| Domain capability | 0.25 | Task-domain matching |
| KM expertise | 0.25 | Knowledge Mound stored expertise |
| Pattern affinity | 0.20 | Historical success patterns |
| ELO rating | 0.30 | Win/loss performance rating |
| Calibration accuracy | 0.20 | Prediction reliability |
| Delegation routing | 0.20 | Task routing score |
| Cultural fit | 0.15 | Team composition balance |

### Domain-Capability Mapping

Agents have known strengths by domain:

| Domain | Recommended Agents |
|--------|--------------------|
| Code | Claude, Codex, Codestral, DeepSeek, GPT |
| Research | Claude, Gemini, GPT, DeepSeek-R1 |
| Reasoning | Claude, DeepSeek-R1, GPT, Gemini |
| Creative | Claude, GPT, Gemini, Llama |

### Agent Hierarchy (Gastown Model)

For complex debates, agents can be assigned hierarchical roles:

```python
from aragora.debate.hierarchy import AgentHierarchy, HierarchyConfig

config = HierarchyConfig(
    max_orchestrators=1,
    max_monitors=2,
    min_workers=2,
    capability_weight=0.4,
    elo_weight=0.3,
    affinity_weight=0.3,
    auto_promote=True,
)
```

| Role | Count | Responsibility |
|------|-------|----------------|
| **Orchestrator** | 1 | Coordinates debate flow and synthesis |
| **Monitor** | 1-2 | Observes for quality issues, stuck debates, violations |
| **Worker** | 2+ | Executes debate tasks (propose, critique, revise) |

**Auto-promotion:** If a worker consistently outperforms monitors, `auto_promote=True` allows dynamic role reassignment.

### Composition Strategies

**Diverse team (recommended for most cases):**
```python
agents = ["claude", "gpt4", "gemini"]  # Different providers, different strengths
```

**Specialist team (domain-specific):**
```python
agents = ["claude", "codestral", "deepseek"]  # All strong at code
```

**Adversarial team (stress-testing ideas):**
```python
# Use graph debates with low disagreement threshold
policy = BranchPolicy(min_disagreement=0.3)  # Branch early and often
```

---

## Workflow Patterns

Workflow patterns (`aragora/workflow/patterns/`) compose agents into structured multi-step pipelines.

### Dialectic (Thesis-Antithesis-Synthesis)

Classical dialectical reasoning with dedicated roles.

```python
from aragora.workflow.patterns.dialectic import DialecticPattern

pattern = DialecticPattern(
    thesis_agent="claude",
    antithesis_agent="gpt4",
    synthesis_agent="claude",
    thesis_stance="supportive",
    include_meta_analysis=True,
    timeout_per_step=120.0,
)
```

**Flow:** Thesis -> Antithesis -> Synthesis -> (optional) Meta-Analysis

**When to use:** Evaluating a specific proposal where you want structured opposition.

### Hierarchical (Manager-Worker Delegation)

A manager agent decomposes the task, delegates to workers, and integrates results.

```python
from aragora.workflow.patterns.hierarchical import HierarchicalPattern

pattern = HierarchicalPattern(
    manager_agent="claude",
    worker_agents=["gpt4", "gemini", "claude"],
    max_subtasks=4,
    timeout_per_worker=120.0,
)
```

**Flow:** Decompose -> Parse Subtasks -> Dispatch (parallel) -> Manager Review

**When to use:** Large tasks that can be split into independent parts.

### Map-Reduce

Split input into chunks, process in parallel, aggregate results.

```python
from aragora.workflow.patterns.map_reduce import MapReducePattern

pattern = MapReducePattern(
    split_strategy="chunks",  # chunks, lines, sections, files
    chunk_size=4000,
    map_agent="claude",
    reduce_agent="gpt4",
    parallel_limit=5,
    timeout_per_chunk=60.0,
)
```

**Flow:** Split -> Map (parallel) -> Reduce

**When to use:** Document analysis, code review across many files, data processing.

### Review Cycle (Iterative Refinement)

One agent drafts, another reviews, looping until convergence.

```python
from aragora.workflow.patterns.review_cycle import ReviewCyclePattern

pattern = ReviewCyclePattern(
    draft_agent="claude",
    review_agent="gpt4",
    max_iterations=3,
    convergence_threshold=0.85,
    review_criteria=["correctness", "efficiency", "readability"],
)
```

**Flow:** Draft -> Review -> Check Convergence -> (loop or output)

**When to use:** Code generation, document writing, any task requiring iterative polish.

### Hive Mind (Parallel Consensus)

All agents process the input independently, then merge via consensus.

```python
from aragora.workflow.patterns.hive_mind import HiveMindPattern

pattern = HiveMindPattern(
    agents=["claude", "gpt4", "gemini"],
    consensus_mode="synthesis",  # weighted, majority, synthesis
    consensus_threshold=0.7,
    include_dissent=True,
    timeout_per_agent=120.0,
)
```

**Flow:** All agents in parallel -> Consensus merge

**When to use:** High-stakes decisions where you want independent assessments before any cross-pollination.

### Post-Debate (Automated Follow-Up)

Runs after a debate completes to extract knowledge, notify stakeholders, and store results.

```python
from aragora.workflow.patterns.post_debate import PostDebatePattern, PostDebateConfig

pattern = PostDebatePattern(
    config=PostDebateConfig(
        store_consensus=True,
        extract_facts=True,
        notify_webhook="https://hooks.example.com/decisions",
        generate_summary=True,
        workspace_id="default",
    )
)
```

**Flow:** Extract Knowledge -> Store in KnowledgeMound -> Notify -> Generate Summary

Enable via Arena config: `enable_post_debate_workflow=True`

---

## Consensus Mechanisms

### Consensus Proof

Every debate produces a `ConsensusProof` (`aragora/debate/consensus.py`) with auditable provenance:

```python
@dataclass
class ConsensusProof:
    final_claim: str
    confidence: float           # 0-1
    consensus_reached: bool
    votes: List[ConsensusVote]  # AGREE, DISAGREE, ABSTAIN, CONDITIONAL
    supporting_agents: List[str]
    dissenting_agents: List[str]
    dissents: List[DissentRecord]
    unresolved_tensions: List[UnresolvedTension]
    evidence_chain: List[Evidence]
    checksum: str               # SHA-256 integrity hash
```

**Key properties:**
- `has_strong_consensus`: >80% agreement AND >0.7 confidence
- `agreement_ratio`: Proportion of supporting agents
- `get_blind_spots()`: Identifies missing perspectives

### Convergence Detection

The `ConvergenceDetector` (`aragora/debate/convergence.py`) monitors agent positions to detect early convergence or persistent divergence.

**3-tier fallback strategy:**
1. **SentenceTransformer** - Best accuracy (requires `sentence-transformers`)
2. **TF-IDF** - Good accuracy (requires `scikit-learn`)
3. **Jaccard** - Always available, zero dependencies

**Convergence weights:**
| Signal | Weight |
|--------|--------|
| Semantic similarity | 40% |
| Argument diversity | 20% |
| Evidence convergence | 20% |
| Stance volatility | 20% |

---

## Memory & Learning Patterns

### Selection Feedback Loop

Adjusts agent selection weights based on actual debate performance (`aragora/debate/selection_feedback.py`).

```python
FeedbackLoopConfig(
    performance_to_selection_weight=0.15,
    calibration_to_elo_weight=0.1,
    min_debates_for_feedback=3,
    feedback_decay_factor=0.9,
    max_adjustment=0.5,
    recency_window_days=30,
)
```

Enable via Arena config: `enable_performance_feedback=True`

The loop tracks per-agent: win rate, timeout rate, calibration score, domain-specific performance, and response time. Adjustments are bounded by `max_adjustment` to prevent runaway selection bias.

### Cross-Debate Memory

Maintains institutional knowledge from past debates (`aragora/memory/cross_debate_rlm.py`).

**Access tiers:**

| Tier | Duration | Detail Level |
|------|----------|-------------|
| Hot | 24h | Full detail |
| Warm | 7d | Summary level |
| Cold | 30d | Abstract level |
| Archive | >30d | Minimal context |

Enable via Arena config: `enable_cross_debate_memory=True`

Uses RLM (Recursive Language Models) when available for programmatic context examination, falling back to hierarchical compression.

### Continuum Memory (4-Tier Learning)

Based on Google Research's Nested Learning paradigm (`aragora/memory/continuum/core.py`):

| Tier | Half-Life | Update Frequency | Purpose |
|------|-----------|------------------|---------|
| Fast | 1h | Immediate | Current debate context |
| Medium | 24h | Per-round | Session-level learning |
| Slow | 7d | Per-cycle | Cross-session patterns |
| Glacial | 30d | Monthly | Long-term institutional knowledge |

Entries promote/demote between tiers based on stability score, access frequency, and importance.

### Memory Coordination

`MemoryCoordinator` (`aragora/memory/coordinator.py`) provides atomic writes across multiple memory systems:

```python
MemoryCoordinationConfig(
    enable_coordinated_writes=True,
    coordinator_parallel_writes=False,
    coordinator_rollback_on_failure=True,
    coordinator_min_confidence_for_mound=0.7,
)
```

Coordinates: ContinuumMemory, ConsensusMemory, CritiqueStore, KnowledgeMound.

---

## Resilience Patterns

### Agent Fallback Chain

When a primary agent hits rate limits or quota errors, Aragora falls back to alternatives:

```
Primary Agent (Claude) -> OpenRouter Fallback -> Circuit Breaker Open
```

**Detected errors:** 429 (rate limit), 403 (quota), 408/504/524 (timeouts)

See `aragora/agents/fallback.py` for the `QuotaFallbackMixin`.

### Airlock Proxy

Wraps agent calls with timeout handling, response sanitization, and fallback responses:

```python
AirlockConfig(
    generate_timeout=240.0,    # Proposal generation
    critique_timeout=180.0,    # Critique generation
    vote_timeout=120.0,        # Voting
    max_retries=1,
    extract_json=True,         # Sanitize malformed output
    fallback_on_timeout=True,
)
```

Enable via Arena config: `use_airlock=True`

### Rhetorical Observer

Passive observer that detects argument patterns without interfering with the debate:

- Concession, rebuttal, synthesis detection
- Appeal to authority/evidence tracking
- Technical depth measurement
- Rhetorical question detection

Enable via Arena config: `enable_rhetorical_observer=True`

---

## Common Anti-Patterns

### 1. Too Many Rounds

**Problem:** Setting `rounds=10` hoping for better results.
**Reality:** Most debates converge by round 3. Extra rounds add latency and cost without improving quality. The convergence detector will signal early termination when appropriate.

**Fix:** Start with `rounds=3`, use convergence detection, increase only if convergence isn't reached.

### 2. Homogeneous Agent Teams

**Problem:** Using 5 instances of the same model (e.g., all Claude).
**Reality:** Same model, same biases. You get agreement, not consensus.

**Fix:** Mix providers. A team of [Claude, GPT-4, Gemini] produces more robust decisions than [Claude, Claude, Claude].

### 3. Skipping the Critique Phase

**Problem:** Using agents only for proposals, ignoring critique/revision cycles.
**Reality:** The value of multi-agent debate is in the adversarial feedback loop, not parallel generation.

**Fix:** Use the standard debate protocol with full critique rounds. If you only need parallel generation, use the Hive Mind pattern instead.

### 4. Ignoring Dissent

**Problem:** Treating consensus as binary (reached/not reached) and discarding minority opinions.
**Reality:** Dissent often contains valuable information about edge cases, risks, or assumptions.

**Fix:** Check `consensus_proof.dissents` and `consensus_proof.unresolved_tensions`. The `get_blind_spots()` method helps identify what was missed.

### 5. Over-Branching in Graph Debates

**Problem:** Setting `min_disagreement=0.2` causes every minor difference to spawn branches.
**Reality:** Excessive branching increases cost and complexity without proportional value.

**Fix:** Set `min_disagreement=0.6` or higher. Branch only on genuinely different approaches, not stylistic differences.

### 6. Missing Post-Debate Knowledge Capture

**Problem:** Running debates without persisting results to the Knowledge Mound.
**Reality:** Each debate produces institutional knowledge that can inform future decisions.

**Fix:** Enable `enable_post_debate_workflow=True` and `store_consensus=True` in the post-debate config.

### 7. Using Hierarchy When Flat Is Enough

**Problem:** Assigning Orchestrator/Monitor/Worker roles for a 2-agent debate.
**Reality:** Hierarchy adds overhead. It's valuable for 4+ agents on complex tasks, not for simple questions.

**Fix:** Use flat team composition for <4 agents. Reserve hierarchy for complex, multi-phase debates.

---

## Pattern Selection Guide

| Situation | Recommended Pattern | Debate Mode |
|-----------|-------------------|-------------|
| Quick yes/no decision | Standard, 2 rounds | Linear |
| Architecture decision | Standard, 3 rounds | Linear or Graph |
| "It depends" question | Matrix debate | Matrix |
| Complex multi-faceted problem | Graph debate | Graph |
| Document review | Map-Reduce workflow | N/A |
| Code generation + review | Review Cycle workflow | N/A |
| High-stakes decision | Hive Mind + Standard debate | Linear |
| Evaluating a specific proposal | Dialectic workflow | N/A |
| Large task decomposition | Hierarchical workflow | N/A |
| Risk assessment | Matrix debate, multiple scenarios | Matrix |

### Decision Tree

```
Is the question well-scoped with clear evaluation criteria?
├── Yes → Standard Linear Debate (2-3 rounds)
│         Are there genuinely different valid approaches?
│         ├── Yes → Graph Debate (branch on disagreement)
│         └── No → Stay with Linear
└── No → Does the answer depend on context/assumptions?
          ├── Yes → Matrix Debate (scenario-based)
          └── No → Can the task be decomposed?
                    ├── Yes → Hierarchical or Map-Reduce workflow
                    └── No → Hive Mind (parallel independent assessment)
```

---

## Related Documentation

- [ORCHESTRATION.md](./ORCHESTRATION.md) - Low-level orchestration architecture
- [GRAPH_DEBATES.md](./GRAPH_DEBATES.md) - Graph debate implementation details
- [MATRIX_DEBATES.md](./MATRIX_DEBATES.md) - Matrix debate implementation details
- [RESILIENCE_PATTERNS.md](./RESILIENCE_PATTERNS.md) - Circuit breakers, retry, timeout patterns

---

*Last updated: 2026-02-01*
