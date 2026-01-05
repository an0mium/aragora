# Aragora Feature Documentation

This document provides detailed documentation for all 30+ features implemented in aragora through 8 phases of self-improvement.

## Table of Contents

- [Phase 1: Foundation](#phase-1-foundation)
- [Phase 2: Learning](#phase-2-learning)
- [Phase 3: Evidence & Resilience](#phase-3-evidence--resilience)
- [Phase 4: Agent Evolution](#phase-4-agent-evolution)
- [Phase 5: Intelligence](#phase-5-intelligence)
- [Phase 6: Formal Reasoning](#phase-6-formal-reasoning)
- [Phase 7: Reliability & Audit](#phase-7-reliability--audit)
- [Phase 8: Advanced Debates](#phase-8-advanced-debates)

---

## Phase 1: Foundation

### ContinuumMemory
**File:** `aragora/memory/continuum.py`

Multi-timescale learning system that organizes memories into fast, medium, and slow tiers based on recency and importance.

```python
from aragora.memory.continuum import ContinuumMemory

memory = ContinuumMemory(db_path="continuum.db")
memory.store("Important insight", tier="FAST", agent="claude")
recent = memory.retrieve(tier="FAST", limit=10)
```

**Key Methods:**
- `store(content, tier, agent)` - Store content in a specific tier
- `retrieve(tier, limit)` - Retrieve recent memories from a tier
- `promote(memory_id)` - Promote memory to a higher tier
- `decay()` - Age out old memories

### ReplayRecorder
**File:** `aragora/replay/replay.py`

Records all cycle events for later analysis and replay.

```python
from aragora.replay.replay import DebateRecorder, DebateReplayer

recorder = DebateRecorder(storage_dir="replays")
filepath = recorder.save_debate(result, metadata={"agents": ["claude", "codex"]})

replayer = DebateReplayer(storage_dir="replays")
debates = replayer.list_debates()
replayer.replay_debate("debate_20260103_120000_abc12345.json", speed=2.0)
```

### MetaLearner
**File:** `aragora/learning/meta.py`

Self-tuning hyperparameter optimization based on debate outcomes.

```python
from aragora.learning.meta import MetaLearner

learner = MetaLearner(db_path="meta.db")
suggestions = learner.suggest_adjustments({
    "consensus_rate": 0.75,
    "avg_rounds": 2.5,
    "avg_confidence": 0.85
})
```

### IntrospectionAPI
**File:** `aragora/introspection/api.py`

Agent self-awareness and reflection capabilities.

```python
from aragora.introspection.api import IntrospectionAPI

api = IntrospectionAPI()
state = api.get_agent_state(agent_name)
api.record_reflection(agent_name, "I noticed my critiques were too harsh")
```

### ArgumentCartographer
**File:** `aragora/visualization/mapper.py`

Builds directed graphs of debate logic in real-time for visualization.

```python
from aragora.visualization.mapper import ArgumentCartographer

cartographer = ArgumentCartographer()
cartographer.start_debate("debate-123", task="Design a cache")
cartographer.add_proposal("claude", "Use Redis with LRU")
cartographer.add_critique("codex", "claude", ["No failover strategy"], severity=0.7)
graph = cartographer.get_graph()
```

### WebhookDispatcher
**File:** `aragora/webhooks/dispatcher.py`

Sends notifications to external systems on debate events.

```python
from aragora.webhooks.dispatcher import WebhookDispatcher

dispatcher = WebhookDispatcher()
dispatcher.register("https://example.com/webhook", events=["debate_start", "consensus_reached"])
await dispatcher.dispatch("debate_start", {"task": "Design API"})
```

---

## Phase 2: Learning

### ConsensusMemory
**File:** `aragora/memory/consensus.py`

Tracks which topics have reached consensus and which remain contested.

```python
from aragora.memory.consensus import ConsensusMemory

memory = ConsensusMemory(db_path="consensus.db")
memory.record_consensus("API versioning", reached=True, confidence=0.9)
status = memory.get_topic_status("API versioning")
contested = memory.get_contested_topics(min_debates=3)
```

### InsightExtractor
**File:** `aragora/insights/extractor.py`

Extracts patterns and insights from completed debates.

```python
from aragora.insights.extractor import InsightExtractor

extractor = InsightExtractor(db_path="insights.db")
insights = extractor.extract_from_debate(result)
patterns = extractor.get_winning_patterns(min_occurrences=3)
```

---

## Phase 3: Evidence & Resilience

### MemoryStream
**File:** `aragora/memory/stream.py`

Per-agent persistent memory that survives across debates.

```python
from aragora.memory.stream import MemoryStream

stream = MemoryStream(db_path="streams.db")
stream.add_memory("claude", "User prefers functional style")
memories = stream.get_memories("claude", limit=10)
```

### LocalDocsConnector
**File:** `aragora/connectors/local_docs.py`

Searches local codebase for evidence to ground claims.

```python
from aragora.connectors.local_docs import LocalDocsConnector

connector = LocalDocsConnector(root_path=".")
evidence = await connector.search("rate limiting implementation")
content = await connector.fetch("aragora/debate/orchestrator.py")
```

### CounterfactualOrchestrator
**File:** `aragora/debate/counterfactual.py`

Resolves deadlocks by exploring alternative debate branches.

```python
from aragora.debate.counterfactual import CounterfactualOrchestrator

orchestrator = CounterfactualOrchestrator()
if orchestrator.detect_deadlock(messages):
    branches = await orchestrator.fork_debate(context)
    best = orchestrator.merge_branches(branches)
```

### CapabilityProber
**File:** `aragora/agents/prober.py`

Tests agent capabilities to ensure quality.

```python
from aragora.agents.prober import CapabilityProber

prober = CapabilityProber()
results = await prober.probe_agent(agent, tests=["code_generation", "critique"])
if results.passed:
    print(f"Agent scored {results.score}")
```

### DebateTemplates
**File:** `aragora/debate/templates.py`

Structured formats for different debate types.

```python
from aragora.debate.templates import DebateTemplates

templates = DebateTemplates()
template = templates.get("code_review")
structured_task = template.format(code=code_snippet)
```

### AgentCircuitBreaker
**File:** `scripts/nomic_loop.py`

Prevents cascade failures when agents repeatedly fail. Trips after configurable failure threshold and auto-resets after cooldown cycles.

```python
from scripts.nomic_loop import AgentCircuitBreaker

breaker = AgentCircuitBreaker(
    failure_threshold=3,  # Trip after 3 consecutive failures
    cooldown_cycles=2     # Wait 2 cycles before retrying
)

# Check before calling agent
if breaker.is_open("claude"):
    print("Agent tripped - skipping")
else:
    try:
        result = await agent.call()
        breaker.record_success("claude")
    except Exception:
        breaker.record_failure("claude")

# Get circuit status
status = breaker.get_status()  # {"claude": {"open": True, "failures": 3}}
```

**States:**
- **CLOSED**: Normal operation, agent is healthy
- **OPEN**: Tripped, skipping agent calls
- **HALF-OPEN**: Testing if agent recovered (first call after cooldown)

### Cycle Timeout
**File:** `scripts/nomic_loop.py`

Prevents runaway cycles from consuming unlimited time. Configurable per-cycle timeout with phase-level deadline checks.

```python
loop = NomicLoop(
    max_cycle_seconds=7200  # 2 hour max per cycle (default)
)

# Or override per-run:
await loop.run_cycle(max_cycle_seconds=3600)  # 1 hour limit
```

**Behavior:**
- Logs warning at 50% time consumed
- Graceful shutdown at deadline with partial results
- Phase-level checks prevent hanging in any single phase
- Timeout triggers circuit breaker for slow agents

### Agent Health Check
**File:** `aragora/debate/orchestrator.py`

15-second connectivity probe before debates to verify agents are reachable.

```python
# Arena performs health check on startup
arena = Arena(environment=env, agents=agents)
# Logs: "[health] claude: OK (0.8s)" or "[health] codex: FAILED (timeout)"
```

Unhealthy agents are excluded from the debate automatically.

---

## Phase 4: Agent Evolution

### PersonaManager
**File:** `aragora/agents/personas.py`

Manages agent personalities, traits, and expertise evolution.

```python
from aragora.agents.personas import PersonaManager

manager = PersonaManager(db_path="personas.db")
persona = manager.get_persona("claude-visionary")
manager.update_trait("claude-visionary", "expertise", "distributed systems")
manager.record_success("claude-visionary", domain="caching")
```

### PromptEvolver
**File:** `aragora/evolution/prompts.py`

Evolves agent system prompts based on successful patterns.

```python
from aragora.evolution.prompts import PromptEvolver

evolver = PromptEvolver(db_path="prompts.db")
evolver.record_success(agent_name, prompt_version, score=0.95)
new_prompt = evolver.evolve(agent_name)
```

### Tournament
**File:** `aragora/competition/tournament.py`

Periodic competitive benchmarking between agents.

```python
from aragora.competition.tournament import Tournament

tournament = Tournament(agents=["claude", "codex", "gemini"])
results = await tournament.run(tasks=benchmark_tasks)
rankings = tournament.get_rankings()
```

---

## Phase 5: Intelligence

### ConvergenceDetector
**File:** `aragora/debate/convergence.py`

Detects when debate has converged for early stopping.

```python
from aragora.debate.convergence import ConvergenceDetector

detector = ConvergenceDetector(threshold=0.85)
status = detector.check(messages)
if status.converged:
    print(f"Converged at similarity {status.similarity:.2f}")
```

### MetaCritiqueAnalyzer
**File:** `aragora/meta/critique.py`

Analyzes the debate process itself and provides recommendations.

```python
from aragora.meta.critique import MetaCritiqueAnalyzer

analyzer = MetaCritiqueAnalyzer()
analysis = analyzer.analyze_debate(result)
recommendations = analysis.recommendations
```

### EloSystem
**File:** `aragora/ranking/elo.py`

Persistent skill tracking using Elo ratings.

```python
from aragora.ranking.elo import EloSystem

elo = EloSystem(db_path="elo.db")
elo.record_match(winner="claude", loser="codex", domain="security")
rating = elo.get_rating("claude")
```

### AgentSelector
**File:** `aragora/selection/selector.py`

Smart agent team selection based on task requirements.

```python
from aragora.selection.selector import AgentSelector

selector = AgentSelector(elo_system=elo, persona_manager=personas)
team = selector.select(task="Design authentication", team_size=3)
```

### RiskRegister
**File:** `aragora/risk/register.py`

Tracks items with low consensus for future attention.

```python
from aragora.risk.register import RiskRegister

register = RiskRegister(db_path="risks.db")
register.add_risk("SQL injection vectors", confidence=0.4, severity="high")
high_risks = register.get_risks(min_severity="high")
```

---

## Phase 6: Formal Reasoning

### ClaimsKernel
**File:** `aragora/reasoning/claims.py`

Structured typed claims with evidence tracking.

```python
from aragora.reasoning.claims import ClaimsKernel, Claim, ClaimType

kernel = ClaimsKernel()
claim = Claim(
    content="This algorithm is O(n log n)",
    claim_type=ClaimType.MATHEMATICAL,
    evidence=["complexity_proof.md"]
)
kernel.add_claim(claim)
```

### ProvenanceManager
**File:** `aragora/provenance/manager.py`

Cryptographic evidence chain integrity.

```python
from aragora.provenance.manager import ProvenanceManager

provenance = ProvenanceManager(db_path="provenance.db")
chain = provenance.create_chain(evidence_id, source="github.com/...")
verified = provenance.verify_chain(chain_id)
```

### BeliefNetwork
**File:** `aragora/reasoning/belief.py`

Probabilistic reasoning over uncertain claims.

```python
from aragora.reasoning.belief import BeliefNetwork

network = BeliefNetwork()
network.add_belief("system_secure", prior=0.7)
network.add_evidence("passed_audit", supports="system_secure", strength=0.9)
posterior = network.query("system_secure")
```

### ProofExecutor
**File:** `aragora/verification/executor.py`

Executes code to verify claims programmatically.

```python
from aragora.verification.executor import ProofExecutor

executor = ProofExecutor()
result = await executor.verify_claim(
    claim="Function returns sorted list",
    test_code="assert is_sorted(my_func([3,1,2]))"
)
```

### ScenarioMatrix
**File:** `aragora/testing/scenarios.py`

Robustness testing across multiple scenarios.

```python
from aragora.testing.scenarios import ScenarioMatrix

matrix = ScenarioMatrix()
matrix.add_scenario("high_load", {"requests_per_sec": 10000})
matrix.add_scenario("network_partition", {"partition_prob": 0.1})
results = await matrix.run_all(system_under_test)
```

---

## Phase 7: Reliability & Audit

### EnhancedProvenanceManager
**File:** `aragora/provenance/enhanced.py`

Extends ProvenanceManager with staleness detection for living documents.

```python
from aragora.provenance.enhanced import EnhancedProvenanceManager

provenance = EnhancedProvenanceManager(db_path="provenance.db")
provenance.set_staleness_threshold(hours=24)
stale = provenance.get_stale_evidence()
```

### CheckpointManager
**File:** `aragora/checkpoint/manager.py`

Pause/resume and crash recovery for long-running debates.

```python
from aragora.checkpoint.manager import CheckpointManager

checkpoint = CheckpointManager(db_path="checkpoints.db")
checkpoint.save(debate_id, state)
restored = checkpoint.restore(debate_id)
```

### BreakpointManager
**File:** `aragora/checkpoint/breakpoints.py`

Human intervention points in automated processes.

```python
from aragora.checkpoint.breakpoints import BreakpointManager, BreakpointConfig

breakpoints = BreakpointManager(
    config=BreakpointConfig(min_confidence=0.5, max_deadlock_rounds=3)
)
if breakpoints.should_pause(state):
    await breakpoints.wait_for_human()
```

### ReliabilityScorer
**File:** `aragora/reasoning/reliability.py`

Scores claim confidence based on evidence quality.

```python
from aragora.reasoning.reliability import ReliabilityScorer

scorer = ReliabilityScorer(provenance=provenance_manager)
score = scorer.score_claim(claim_id)
report = scorer.get_reliability_report(claim_id)
```

### DebateTracer
**File:** `aragora/debate/traces.py`

Audit logs with deterministic replay capability.

```python
from aragora.debate.traces import DebateTracer

tracer = DebateTracer(debate_id="d123", task="Design API", agents=["claude", "codex"])
tracer.record_proposal("claude", "Use REST with versioning")
tracer.record_critique("codex", "claude", issues=["No GraphQL support"])
trace = tracer.finalize({"consensus": True})
```

---

## Phase 8: Advanced Debates

### PersonaLaboratory
**File:** `aragora/agents/laboratory.py`

A/B testing, emergent trait detection, and cross-pollination.

```python
from aragora.agents.laboratory import PersonaLaboratory

lab = PersonaLaboratory(persona_manager=personas, db_path="lab.db")
experiment = lab.create_experiment(
    agent="claude",
    variant_traits=["more_concise", "code_focused"]
)
lab.record_trial(experiment.id, is_control=False, success=True)
emergent = lab.detect_emergent_traits()
lab.cross_pollinate_traits("claude", "codex", "security_focus")
```

### SemanticRetriever
**File:** `aragora/memory/embeddings.py`

Find similar past critiques using embeddings.

```python
from aragora.memory.embeddings import SemanticRetriever

retriever = SemanticRetriever(db_path="embeddings.db")
await retriever.embed_and_store("critique-123", "The error handling is insufficient")
similar = await retriever.find_similar("needs better error handling", limit=5)
```

### FormalVerificationManager
**File:** `aragora/verification/formal.py`

Z3 SMT solver for verifying logical and mathematical claims.

```python
from aragora.verification.formal import FormalVerificationManager

verifier = FormalVerificationManager()
result = await verifier.verify_claim(
    "For all x > 0, x + 1 > x",
    claim_type="mathematical"
)
if result.status == "PROVED":
    print(f"Proof: {result.proof}")
```

### DebateGraph
**File:** `aragora/debate/graph.py`

DAG-based debates for complex multi-path disagreements.

```python
from aragora.debate.graph import DebateGraph, GraphDebateOrchestrator

orchestrator = GraphDebateOrchestrator(agents=agents)
result = await orchestrator.run_debate(task)
graph = orchestrator.graph
paths = graph.get_all_paths()
```

### DebateForker
**File:** `aragora/debate/forking.py`

Parallel branch exploration when agents fundamentally disagree.

```python
from aragora.debate.forking import DebateForker, ForkDetector

detector = ForkDetector()
decision = detector.should_fork(messages, round_num, agents)

if decision.should_fork:
    forker = DebateForker()
    merge_result = await forker.run_branches(decision, base_context)
    print(f"Winning hypothesis: {merge_result.winning_hypothesis}")
```

---

## Agent Cognition: Structured Thinking Protocols

Each agent in the nomic loop uses a specialized thinking protocol that guides their analysis and proposal generation.

### Overview

Structured Thinking Protocols ensure agents:
- Explore before proposing (avoid assumptions)
- Show their reasoning chain (transparent decision-making)
- Consider alternatives (avoid premature convergence)
- Ground proposals in evidence (reference specific code)

### Protocol Details

| Agent | Protocol Steps | Focus |
|-------|----------------|-------|
| **Claude** | EXPLORE → PLAN → REASON → PROPOSE | Architecture, system cohesion |
| **Codex** | TRACE → ANALYZE → DESIGN → VALIDATE | Implementation, code quality |
| **Gemini** | EXPLORE → ENVISION → REASON → PROPOSE | Product vision, user impact |
| **Grok** | DIVERGE → CONNECT → SYNTHESIZE → GROUND | Creative solutions, novel patterns |

### Implementation

The protocols are injected via system prompts in `scripts/nomic_loop.py`:

```python
# Example: Claude's structured thinking protocol
self.claude.system_prompt = """You are a visionary architect for aragora.

=== STRUCTURED THINKING PROTOCOL ===
When analyzing a task:
1. EXPLORE: First understand the current state - read relevant files, trace code paths
2. PLAN: Design your approach before implementing - consider alternatives
3. REASON: Show your thinking step-by-step - explain tradeoffs
4. PROPOSE: Make concrete, actionable proposals with clear impact

When using Claude Code:
- Use 'Explore' mode to deeply understand the codebase before proposing
- Use 'Plan' mode to design implementation approaches with user approval
"""
```

### Benefits

1. **Higher Quality Proposals**: Agents analyze before proposing
2. **Transparent Reasoning**: Other agents can critique the reasoning, not just conclusions
3. **Evidence-Grounded**: Proposals reference specific files and code patterns
4. **Complementary Perspectives**: Each agent's protocol highlights different aspects

---

## Integration in Nomic Loop

All features are integrated into the nomic loop (`scripts/nomic_loop.py`) using an optional import pattern:

```python
# Optional feature import
try:
    from aragora.feature.module import FeatureClass
    FEATURE_AVAILABLE = True
except ImportError:
    FEATURE_AVAILABLE = False
    FeatureClass = None

# Initialize if available
if FEATURE_AVAILABLE:
    self.feature = FeatureClass(db_path=str(self.nomic_dir / "feature.db"))
    print(f"[feature] Feature enabled")
```

Features degrade gracefully—if a feature's dependencies are missing, the loop continues without it.

---

## Feature Dependencies

| Feature | Dependencies |
|---------|--------------|
| SemanticRetriever | sentence-transformers OR openai |
| FormalVerificationManager | z3-solver |
| BeliefNetwork | numpy |
| ConvergenceDetector | sentence-transformers OR sklearn |

Install optional dependencies:
```bash
pip install sentence-transformers z3-solver numpy scikit-learn
```

---

## Phase 9: Truth Grounding (Recent)

### FlipDetector
**File:** `aragora/insights/flip_detector.py`

Semantic position reversal detection that tracks when agents change their positions on claims.

```python
from aragora.insights.flip_detector import FlipDetector

detector = FlipDetector(db_path="positions.db")
flips = detector.detect_flips_for_agent("claude", lookback_positions=50)
consistency = detector.get_agent_consistency("claude")
print(f"Consistency score: {consistency.consistency_score:.2%}")
```

**Flip Types:**
- `contradiction` - Direct opposite position
- `refinement` - Minor adjustment
- `retraction` - Complete withdrawal
- `qualification` - Adding nuance

### GroundedPersonaManager
**File:** `aragora/agents/grounded.py`

Truth-grounded persona tracking that links agent identities to verifiable performance history.

```python
from aragora.agents.grounded import GroundedPersonaManager

manager = GroundedPersonaManager(db_path="personas.db")
identity = manager.get_grounded_identity("claude")
print(f"Win rate: {identity.win_rate:.2%}")
print(f"Calibration: {identity.calibration_score:.2f}")
```

**Key Metrics:**
- Position history with debate outcomes
- Calibration scoring (prediction accuracy)
- Domain expertise tracking
- Inter-agent relationship metrics

### TruthGroundingSystem
**File:** `aragora/agents/truth_grounding.py`

Central system for maintaining epistemic accountability across debates.

```python
from aragora.agents.truth_grounding import TruthGroundingSystem

system = TruthGroundingSystem(db_path="grounding.db")
system.record_position("claude", "claim-123", 0.85, "debate-456")
accuracy = system.compute_calibration("claude")
```

### CalibrationTracker
**File:** `aragora/agents/calibration.py`

Tracks prediction accuracy using Brier scoring and ECE (Expected Calibration Error).

```python
from aragora.agents.calibration import CalibrationTracker

tracker = CalibrationTracker(db_path="calibration.db")
tracker.record_prediction("claude", confidence=0.85, correct=True, domain="security")
summary = tracker.get_calibration_summary("claude")
print(f"Brier score: {summary['brier_score']:.4f}")
print(f"ECE: {summary['ece']:.4f}")
curve = tracker.get_calibration_curve("claude")  # For visualization
```

**Key Metrics:**
- Brier score (lower is better calibration)
- ECE (Expected Calibration Error)
- Per-domain calibration breakdown
- Calibration curve data for plotting

### DissentRetriever (Enhanced)
**File:** `aragora/memory/consensus.py`

Enhanced dissent retrieval with contrarian views and risk warnings.

```python
from aragora.memory.consensus import DissentRetriever

retriever = DissentRetriever(memory)
contrarian = retriever.find_contrarian_views("The approach should use caching")
risks = retriever.find_risk_warnings("Database migration")
context = retriever.get_debate_preparation_context("New feature design")
```

### DebateForker
**File:** `aragora/debate/forking.py`

Parallel branch exploration for deadlock resolution.

```python
from aragora.debate.forking import ForkDetector, DebateForker

detector = ForkDetector(disagreement_threshold=0.7)
decision = detector.should_fork(messages, round_num, agents)

if decision:
    forker = DebateForker(agents, protocol)
    merge_result = await forker.run_branches(decision.branches, base_context)
    print(f"Winner: {merge_result.winning_branch}")
```

### EnhancedProvenanceManager
**File:** `aragora/reasoning/provenance_enhanced.py`

Staleness detection and evidence validation for claims.

```python
from aragora.reasoning.provenance_enhanced import EnhancedProvenanceManager

manager = EnhancedProvenanceManager()
staleness = await manager.check_staleness(claims, changed_files)
if staleness.needs_redebate:
    print(f"Stale claims: {staleness.stale_claims}")
```

### BeliefPropagationAnalyzer (Integrated)
**File:** `aragora/reasoning/belief.py`

Now integrated into Arena for automatic crux identification and evidence suggestions.

```python
from aragora.reasoning.belief import BeliefNetwork, BeliefPropagationAnalyzer

# Automatically used in Arena after consensus:
# - result.debate_cruxes: Key claims that drive disagreement
# - result.evidence_suggestions: Claims needing more evidence
```

**Crux Analysis:**
- Identifies claims with high centrality and high uncertainty
- Suggests evidence targets to reduce debate uncertainty
- Computes consensus probability

### ContinuumMemory (Integrated)
**File:** `aragora/memory/continuum.py`

Now integrated into Arena for cross-debate learning context.

```python
from aragora.memory.continuum import ContinuumMemory, MemoryTier

# Arena uses continuum_memory parameter:
arena = Arena(
    environment=env,
    agents=agents,
    continuum_memory=ContinuumMemory("continuum.db"),  # NEW
)
# Relevant past learnings are automatically injected into agent context
```

**Memory Tiers:**
- FAST (1 day half-life) - Recent patterns
- MEDIUM (1 week) - Recurring patterns
- SLOW (1 month) - Established patterns
- GLACIAL (1 year) - Foundational knowledge

---

## Phase 10: Thread-Safe Audience Participation

### Thread-Safe Arena Mailbox
**File:** `aragora/debate/orchestrator.py`

Decouples event ingestion from consumption using a thread-safe queue pattern for live audience interaction.

```python
from aragora.debate.orchestrator import Arena, DebateProtocol

arena = Arena(
    environment=env,
    agents=agents,
    protocol=DebateProtocol(rounds=3),
    event_emitter=emitter,
    loop_id="my-debate-123",
    strict_loop_scoping=True,  # Only accept events for this loop
)

# WebSocket thread calls _handle_user_event - enqueues to thread-safe queue
# Debate thread calls _drain_user_events before each prompt build
```

**Key Features:**
- `_user_event_queue: queue.Queue` - Thread-safe event buffer
- `_handle_user_event()` - Enqueues without blocking debate loop
- `_drain_user_events()` - Moves events to lists at safe sync points
- `strict_loop_scoping` - Rejects events not matching current loop_id

### Loop Scoping

Multi-tenant event isolation ensures votes/suggestions go to correct debate:

```python
# Events with matching loop_id are processed
event.loop_id = "my-debate-123"  # Accepted

# Events from other loops are ignored
event.loop_id = "other-debate"  # Dropped

# Strict mode also rejects events without loop_id
arena = Arena(..., strict_loop_scoping=True)
```

### Test Coverage

Comprehensive tests in `tests/test_audience_participation.py`:
- `TestMailboxThreadSafety` - Concurrent enqueue/drain safety
- `TestLoopScoping` - Multi-tenant event isolation
- `TestEdgeCases` - Empty queue, malformed events
- `TestSuggestionIntegration` - Vote/suggestion separation

---

## Phase 11: Operational Modes

### Modes System
**File:** `aragora/modes/`

The modes system allows switching between different operational configurations for agents and debates.

### OperationalModes
**File:** `aragora/modes/operational.py`

Switches between different agent tool configurations:

```python
from aragora.modes.operational import OperationalMode, set_mode

# Available modes:
# - default: Standard debate tools
# - research: Web search + document retrieval enabled
# - implementation: Code editing + git operations enabled

set_mode(OperationalMode.IMPLEMENTATION)
```

### CapabilityProber
**File:** `aragora/modes/prober.py`

Tests agent capabilities for quality assurance and vulnerability detection.

```python
from aragora.modes.prober import CapabilityProber, ProbeType

prober = CapabilityProber()
report = await prober.probe_agent(
    target_agent=agent,
    probe_types=[
        ProbeType.CONTRADICTION,
        ProbeType.HALLUCINATION,
        ProbeType.CONFIDENCE_CALIBRATION,
    ],
)
print(f"Vulnerability score: {report.vulnerability_score}")
```

**Probe Types:**
- `contradiction` - Tests for logical inconsistencies
- `hallucination` - Tests for fabricated information
- `sycophancy` - Tests for agreement bias
- `confidence_calibration` - Tests confidence accuracy
- `reasoning_depth` - Tests multi-step reasoning
- `edge_case` - Tests boundary conditions

### RedTeamMode
**File:** `aragora/modes/redteam.py`

Adversarial analysis mode for stress-testing debate conclusions.

```python
from aragora.modes.redteam import RedTeamMode, AttackType

mode = RedTeamMode()
result = await mode.run_redteam(
    target_proposal="Use microservices architecture",
    red_team_agents=red_agents,
    attack_types=[AttackType.DEVILS_ADVOCATE, AttackType.EDGE_CASE],
)
```

**Attack Types:**
- `devils_advocate` - Argue opposite position
- `edge_case` - Find failure scenarios
- `assumption_challenge` - Question premises
- `scale_test` - Test at different scales
- `adversarial` - Active exploitation attempts

---

## Phase 12: Grounded Personas v2

### Overview

Grounded Personas v2 is a comprehensive system for building agent identities from verifiable performance history. Unlike traditional persona systems that rely on predefined traits, Grounded Personas emerge from actual debate outcomes, creating trustworthy agent identities backed by evidence.

**Core Philosophy:**
- Personas are *earned* through performance, not assigned
- All claims about an agent are backed by verifiable debate history
- Relationships between agents emerge from actual interactions
- Significant moments define agent identity naturally

### PositionLedger
**File:** `aragora/agents/grounded.py`

Tracks every position an agent takes across all debates, enabling historical analysis of consistency and expertise.

```python
from aragora.agents.grounded import PositionLedger, Position

ledger = PositionLedger(db_path="personas.db")

# Record a position
position_id = ledger.record_position(
    agent_name="claude",
    claim="Microservices are better for this use case",
    confidence=0.85,
    debate_id="debate-123",
    round_num=2,
    domain="architecture",
)

# Get agent's position history
positions = ledger.get_agent_positions("claude", limit=50)

# Resolve position outcome after debate
ledger.resolve_position(position_id, outcome="correct")  # or "incorrect", "pending"

# Check for position reversals
ledger.mark_reversal(position_id, reversal_debate_id="debate-456")
```

**Key Fields:**
- `claim` - The position statement
- `confidence` - Agent's stated confidence (0.0-1.0)
- `outcome` - Was the position correct? (pending/correct/incorrect)
- `reversed` - Did agent later reverse this position?
- `domain` - Topic domain for expertise tracking

### RelationshipTracker
**File:** `aragora/agents/grounded.py`

Tracks inter-agent relationships based on actual debate interactions.

```python
from aragora.agents.grounded import RelationshipTracker

tracker = RelationshipTracker(db_path="personas.db")

# Update from debate results
tracker.update_from_debate(
    debate_id="debate-123",
    participants=["claude", "gemini", "codex"],
    winner="claude",
    votes={"claude": "option_a", "gemini": "option_a", "codex": "option_b"},
    critiques=[
        {"agent": "codex", "target": "claude"},
        {"agent": "gemini", "target": "codex"},
    ],
)

# Get relationship between two agents
rel = tracker.get_relationship("claude", "gemini")
print(f"Alliance score: {rel.alliance_score:.2f}")
print(f"Rivalry score: {rel.rivalry_score:.2f}")
print(f"Influence score: {rel.influence_score:.2f}")

# Get agent's full network
network = tracker.get_agent_network("claude")
# Returns: {allies: [...], rivals: [...], influences: [...], influenced_by: [...]}
```

**Relationship Types:**
- `alliance_score` - How often agents vote together
- `rivalry_score` - How often agents oppose each other
- `influence_score` - How much this agent influences the other's positions
- `debate_count` - Number of shared debates

### MomentDetector
**File:** `aragora/agents/grounded.py`

Detects genuinely significant narrative moments from debate history.

```python
from aragora.agents.grounded import MomentDetector, SignificantMoment

detector = MomentDetector(
    elo_system=elo,
    position_ledger=ledger,
    relationship_tracker=tracker,
)

# Detect upset victory
moment = detector.detect_upset_victory(
    winner="underdog_agent",
    loser="top_ranked_agent",
    debate_id="debate-789",
)
if moment:
    detector.record_moment(moment)
    print(f"Upset! Significance: {moment.significance_score:.2%}")

# Detect calibration vindication
moment = detector.detect_calibration_vindication(
    agent_name="claude",
    prediction_confidence=0.92,
    was_correct=True,
    domain="security",
    debate_id="debate-101",
)

# Detect streak achievements
moment = detector.detect_streak_achievement(
    agent_name="gemini",
    streak_type="win",
    streak_length=7,
    debate_id="debate-102",
)

# Get narrative summary
summary = detector.get_narrative_summary("claude", limit=5)
# Returns formatted markdown of agent's defining moments
```

**Moment Types:**
- `upset_victory` - Lower-rated agent defeats higher-rated (100+ ELO diff)
- `position_reversal` - High-confidence position changed with evidence
- `calibration_vindication` - 85%+ confidence prediction proven correct
- `streak_achievement` - 5+ consecutive wins or losses
- `domain_mastery` - Becomes #1 ranked in a domain
- `consensus_breakthrough` - Rivals reach agreement

### PersonaSynthesizer
**File:** `aragora/agents/grounded.py`

Synthesizes complete agent personas from all grounded data sources.

```python
from aragora.agents.grounded import PersonaSynthesizer

synthesizer = PersonaSynthesizer(
    position_ledger=ledger,
    calibration_tracker=calibration,
    relationship_tracker=tracker,
    moment_detector=detector,
    elo_system=elo,
)

# Generate complete grounded persona
persona = synthesizer.synthesize("claude")

print(persona.identity_summary)
# "claude is a well-calibrated architect (Brier: 0.12) with expertise in
#  security and distributed systems. They've maintained 78% consistency
#  across 234 positions. Key rival: codex. Key ally: gemini."

print(persona.expertise_profile)
# {"security": 0.89, "architecture": 0.82, "testing": 0.65}

print(persona.calibration_summary)
# {"brier_score": 0.12, "ece": 0.08, "total_predictions": 156}

print(persona.relationship_brief)
# {"allies": ["gemini"], "rivals": ["codex"], "influences": ["grok"]}

# Get opponent briefing for debate prep
briefing = synthesizer.get_opponent_briefing("claude", opponent="codex")
# Returns strategic information about how claude typically debates codex
```

### Arena Integration

The Grounded Personas system is integrated into Arena for automatic tracking:

```python
from aragora.debate.orchestrator import Arena

arena = Arena(
    environment=env,
    agents=agents,
    # Grounded Personas v2 integrations
    calibration_tracker=CalibrationTracker("calibration.db"),
    relationship_tracker=RelationshipTracker("relationships.db"),
    moment_detector=MomentDetector(elo_system=elo),
)

# After each debate, Arena automatically:
# 1. Records positions to PositionLedger
# 2. Updates CalibrationTracker with prediction accuracy
# 3. Updates RelationshipTracker with debate outcomes
# 4. Detects significant moments (upsets, vindications)
# 5. Stores all data for persona synthesis
```

### API Endpoints

The following endpoints expose Grounded Personas data:

| Endpoint | Description |
|----------|-------------|
| `GET /api/agent/{name}/network` | Agent's relationship network |
| `GET /api/agent/{name}/moments?limit=N` | Significant moments |
| `GET /api/agent/{name}/consistency` | Consistency score from FlipDetector |
| `GET /api/consensus/dissents?limit=N` | Dissenting views on topics |

### Benefits

1. **Trust Through Evidence**: Agent claims are backed by verifiable history
2. **Natural Narratives**: Stories emerge from actual performance
3. **Fair Reputation**: Agents earn their standings through debate outcomes
4. **Rich Context**: Debate participants understand each other's tendencies
5. **Accountability**: Position reversals and calibration failures are tracked

---

## API Reference

See [API_REFERENCE.md](./API_REFERENCE.md) for complete HTTP and WebSocket API documentation.
