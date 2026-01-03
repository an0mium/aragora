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
