# Aragora Architecture

This document describes the high-level architecture of aragora, the AI red team / decision stress-test platform. The multi-agent debate system is the engine that powers adversarial validation.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ARAGORA SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           AGENT LAYER                                  │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │  │
│  │  │ Claude  │ │  Codex  │ │ Gemini  │ │  Grok   │ │ OpenAI  │          │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │  │
│  │       │           │           │           │           │                │  │
│  │       └───────────┴─────┬─────┴───────────┴───────────┘                │  │
│  │                         ▼                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    PersonaManager + EloSystem                    │  │  │
│  │  │            Traits, Expertise, Skill Ratings, Selection           │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          DEBATE LAYER                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    Arena (Orchestrator)                          │  │  │
│  │  │    Role Assignment • Round Management • Context Accumulation     │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                 │                                      │  │
│  │  ┌──────────────┬───────────────┼───────────────┬──────────────────┐  │  │
│  │  ▼              ▼               ▼               ▼                  ▼  │  │
│  │  DebateGraph   DebateForker   Protocol    ConvergenceDetector  Tracer│  │
│  │  (DAG-based)  (Parallel)    (Sequential)    (Early Stop)      (Audit)│  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         REASONING LAYER                                │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │  │
│  │  │ClaimsKernel │ │BeliefNetwork│ │ProofExecutor│ │FormalVerifier   │  │  │
│  │  │(Structured) │ │(Probabilist)│ │(Executable) │ │(Z3/Lean)        │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘  │  │
│  │                                 │                                      │  │
│  │  ┌─────────────┐ ┌─────────────┐                                      │  │
│  │  │ Provenance  │ │ Reliability │                                      │  │
│  │  │ (Evidence)  │ │ (Confidence)│                                      │  │
│  │  └─────────────┘ └─────────────┘                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          MEMORY LAYER                                  │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │  │
│  │  │ContinuumMem │ │MemoryStream │ │ConsensusMem │ │SemanticRetriever│  │  │
│  │  │(Timescales) │ │(Per-Agent)  │ │(Topic Track)│ │(Embeddings)     │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘  │  │
│  │                                 │                                      │  │
│  │  ┌─────────────┐ ┌─────────────┐                                      │  │
│  │  │CritiqueStore│ │InsightStore │                                      │  │
│  │  │(Patterns)   │ │(Learning)   │                                      │  │
│  │  └─────────────┘ └─────────────┘                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        PERSISTENCE LAYER                               │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐  │  │
│  │  │   SQLite    │ │  Supabase   │ │ Checkpoint  │ │    Replays      │  │  │
│  │  │ (Local DB)  │ │ (Cloud)     │ │ (Recovery)  │ │ (History)       │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
aragora/
├── agents/                 # Agent implementations
│   ├── base.py            # Abstract Agent class
│   ├── cli_agents.py      # CLI tool wrappers (Claude, Codex, etc.)
│   ├── api_agents.py      # API-based agents (Anthropic, OpenAI, etc.)
│   ├── fallback.py        # QuotaFallbackMixin for provider failover
│   ├── streaming.py       # StreamingMixin for SSE parsing
│   ├── personas.py        # PersonaManager for agent traits
│   ├── laboratory.py      # PersonaLaboratory for A/B testing
│   ├── prober.py          # CapabilityProber for quality assurance
│   ├── grounded.py        # GroundedPersona (truth-based personas)
│   ├── relationships.py   # RelationshipTracker (agent relationships)
│   └── positions.py       # PositionTracker (position history)
│
├── debate/                 # Core debate infrastructure
│   ├── orchestrator.py    # Arena class (~1,650 LOC after extraction refactors)
│   ├── memory_manager.py  # MemoryManager (extracted from orchestrator)
│   ├── prompt_builder.py  # PromptBuilder (extracted from orchestrator)
│   ├── security_barrier.py# SecurityBarrier, TelemetryVerifier
│   ├── telemetry_config.py# TelemetryConfig (observation levels)
│   ├── convergence.py     # ConvergenceDetector
│   ├── graph.py           # DebateGraph (DAG-based debates)
│   ├── forking.py         # DebateForker (parallel branches)
│   ├── traces.py          # DebateTracer (audit logs)
│   ├── checkpoint.py      # CheckpointManager
│   ├── templates.py       # DebateTemplates
│   ├── breakpoints.py     # DebateBreakpointManager
│   ├── scenarios.py       # ScenarioMatrix
│   └── phases/            # Phase executors (extracted)
│       ├── base.py        # PhaseExecutor base class
│       ├── proposal.py    # ProposalPhase
│       ├── critique.py    # CritiquePhase
│       ├── voting.py      # VotingPhase
│       └── synthesis.py   # SynthesisPhase
│
├── reasoning/              # Logical reasoning components
│   ├── claims.py          # ClaimsKernel (structured claims)
│   ├── belief.py          # BeliefNetwork (probabilistic)
│   ├── provenance.py      # ProvenanceManager
│   ├── reliability.py     # ReliabilityScorer
│   └── risk.py            # RiskRegister
│
├── verification/           # Verification subsystems
│   ├── executor.py        # ProofExecutor (code execution)
│   ├── formal.py          # FormalVerificationManager (Z3)
│   └── scenarios.py       # ScenarioMatrix
│
├── memory/                 # Memory systems
│   ├── store.py           # CritiqueStore (SQLite patterns)
│   ├── continuum.py       # ContinuumMemory (timescales)
│   ├── stream.py          # MemoryStream (per-agent)
│   ├── consensus.py       # ConsensusMemory (topic tracking)
│   └── embeddings.py      # SemanticRetriever
│
├── evolution/              # Self-improvement
│   ├── prompts.py         # PromptEvolver
│   └── meta.py            # MetaLearner
│
├── connectors/             # External data sources
│   ├── base.py            # BaseConnector protocol
│   ├── local_docs.py      # LocalDocsConnector
│   ├── github.py          # GitHubConnector
│   ├── web.py             # WebConnector
│   ├── youtube_uploader.py# YouTubeUploaderConnector (OAuth 2.0)
│   └── twitter_poster.py  # TwitterPosterConnector (OAuth 1.0a)
│
├── maintenance/            # System maintenance utilities
│   └── db_maintenance.py  # DatabaseMaintenance (WAL, VACUUM)
│
├── monitoring/             # System monitoring
│   └── simple_observer.py # SimpleObserver (agent failure tracking)
│
├── resilience.py          # CircuitBreaker for agent failure handling
│
├── visualization/          # Debate visualization
│   ├── mapper.py          # ArgumentCartographer
│   └── exporter.py        # Graph export utilities
│
├── live/                   # Live dashboard (Next.js)
│   ├── src/               # React components
│   └── public/            # Static assets
│
├── server/                 # WebSocket/HTTP server
│   ├── unified_server.py  # Unified server (106+ endpoints)
│   ├── handlers/          # Request handlers by domain
│   │   ├── base.py        # BaseHandler, ttl_cache decorator
│   │   ├── debates.py     # Debate CRUD and exports
│   │   ├── agents.py      # Agent profiles and rankings
│   │   ├── analytics.py   # System analytics
│   │   └── ...            # 41 handler modules
│   └── stream/            # Streaming infrastructure (refactored)
│       ├── servers.py     # WebSocket server classes
│       ├── broadcaster.py # Event broadcasting
│       ├── state_manager.py # Connection state
│       ├── loop_manager.py  # Loop lifecycle
│       ├── message_handlers.py # Message routing
│       └── serializers.py # Event serialization
│
├── core.py                # Core types (Message, Critique, Vote, etc.)
└── __init__.py            # Package exports

scripts/
├── nomic_loop.py          # Main nomic loop implementation
└── run_nomic_with_stream.py  # Streaming wrapper

docs/
├── FEATURES.md            # Detailed feature documentation
├── NOMIC_LOOP.md          # Nomic loop documentation
└── ARCHITECTURE.md        # This file
```

## Data Flow

### Standard Debate Flow

```
1. Environment Setup
   ├── Task definition
   ├── Agent selection (AgentSelector)
   └── Protocol configuration

2. Proposal Phase
   ├── Each agent generates initial proposal
   └── Proposals stored as Messages

3. Critique Phase (per round)
   ├── Agents critique each other's proposals
   ├── Critiques have severity scores (0-1)
   └── ArgumentCartographer builds graph

4. Revision Phase
   ├── Proposers incorporate critiques
   ├── ConvergenceDetector checks similarity
   └── Early stop if converged

5. Voting Phase
   ├── Agents vote on best proposal
   ├── Votes weighted by Elo rating
   └── Confidence scores recorded

6. Consensus Phase
   ├── Judge synthesizes final answer
   ├── Dissenting views preserved
   └── DebateResult created

7. Learning Phase
   ├── CritiqueStore records patterns
   ├── InsightExtractor extracts lessons
   ├── PromptEvolver updates prompts
   └── EloSystem updates ratings
```

### Nomic Loop Flow

```
1. Context Gathering (Phase 0)
   ├── All agents read codebase
   ├── LocalDocsConnector provides evidence
   └── Genesis analysis (optional)

2. Proposal (Phase 1)
   ├── Agents propose improvements
   ├── Proposals structured with impact/complexity
   └── ProvenanceManager tracks sources

3. Debate (Phase 2)
   ├── Arena orchestrates critique rounds
   ├── DebateGraph for complex disagreements
   └── DebateForker for parallel exploration

4. Voting (Phase 3)
   ├── Elo-weighted votes
   ├── MetaCritiqueAnalyzer evaluates process
   └── RiskRegister tracks low-consensus items

5. Implementation (Phase 4)
   ├── Winning agent implements via CLI
   ├── Changes sandboxed to approved files
   └── CheckpointManager saves state

6. Verification (Phase 5)
   ├── Syntax check (py_compile)
   ├── Import check (import aragora)
   └── Test suite (pytest)

7. Commit (Phase 6)
   ├── Auto-commit if verification passes
   ├── ReplayRecorder saves cycle
   └── Cycle repeats
```

## Component Interactions

### Memory Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                      MEMORY INTEGRATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Before Debate:                                                  │
│  ├── SemanticRetriever: Find similar past critiques             │
│  ├── ConsensusMemory: Check topic status (settled/contested)    │
│  └── CritiqueStore: Retrieve relevant patterns                  │
│                                                                  │
│  During Debate:                                                  │
│  ├── MemoryStream: Agent-specific context                       │
│  ├── ContinuumMemory: Store novel insights                      │
│  └── ProvenanceManager: Track evidence chains                   │
│                                                                  │
│  After Debate:                                                   │
│  ├── InsightExtractor: Extract winning patterns                 │
│  ├── SemanticRetriever: Store critique embeddings               │
│  └── ConsensusMemory: Update topic status                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Evolution

```
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT EVOLUTION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Structured Thinking Protocols (per-agent cognition)            │
│  ├── Claude:  EXPLORE → PLAN → REASON → PROPOSE                 │
│  ├── Codex:   TRACE → ANALYZE → DESIGN → VALIDATE               │
│  ├── Gemini:  EXPLORE → ENVISION → REASON → PROPOSE             │
│  └── Grok:    DIVERGE → CONNECT → SYNTHESIZE → GROUND           │
│                                                                  │
│  PersonaManager                                                  │
│  ├── Trait storage (expertise, personality)                     │
│  ├── Success tracking by domain                                 │
│  └── Balanced/aggressive/defensive stance                       │
│                                                                  │
│  PersonaLaboratory                                               │
│  ├── A/B testing: Control vs variant personas                   │
│  ├── Emergent trait detection from success patterns             │
│  └── Cross-pollination between agents                           │
│                                                                  │
│  PromptEvolver                                                   │
│  ├── Track prompt → outcome correlations                        │
│  ├── Evolve prompts based on success rate                       │
│  └── Preserve winning prompt elements                           │
│                                                                  │
│  EloSystem                                                       │
│  ├── Skill ratings updated after each debate                    │
│  ├── Domain-specific ratings                                    │
│  └── Used for vote weighting and team selection                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Extension Points

### Adding New Agents

1. Create wrapper in `aragora/agents/cli_agents.py`
2. Implement the `Agent` protocol from `aragora/core.py`
3. Register in the agent factory

```python
class MyAgent(Agent):
    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        # Call your LLM
        pass

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
        # Generate critique
        pass
```

### Adding New Connectors

1. Inherit from `BaseConnector` in `aragora/connectors/base.py`
2. Implement `search()` and `fetch()` methods
3. Register with ProvenanceManager for evidence tracking

```python
class MyConnector(BaseConnector):
    async def search(self, query: str, limit: int = 10) -> list[Evidence]:
        # Search your data source
        pass

    async def fetch(self, evidence_id: str) -> Evidence:
        # Fetch specific evidence
        pass
```

### Adding New Verification Backends

1. Implement the `FormalBackend` protocol in `aragora/verification/formal.py`
2. Register with FormalVerificationManager

```python
class MyBackend(FormalBackend):
    def verify(self, claim: str, proof: str) -> FormalProofResult:
        # Verify using your prover
        pass
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_DB_PATH` | SQLite database path | `aragora.db` |
| `ARAGORA_NOMIC_DIR` | Nomic loop state directory | `.nomic` |
| `SUPABASE_URL` | Supabase project URL | None |
| `SUPABASE_KEY` | Supabase anon key | None |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `GOOGLE_API_KEY` | Google AI key | None |

### Feature Flags

Features degrade gracefully if dependencies are missing:

```python
try:
    from aragora.feature import FeatureClass
    FEATURE_AVAILABLE = True
except ImportError:
    FEATURE_AVAILABLE = False
```

### Protocol Flags (DebateProtocol)

Opt-in features controlled via protocol configuration:

| Flag | Description | Default |
|------|-------------|---------|
| `enable_calibration` | Record prediction accuracy for calibration curves | `False` |
| `enable_rhetorical_observer` | Passive commentary on debate dynamics | `False` |
| `enable_trickster` | Hollow consensus detection | `False` |
| `trickster_sensitivity` | Threshold for trickster challenges | `0.7` |
| `enable_breakpoints` | Human-in-the-loop intervention points | `False` |
| `role_rotation` | Cognitive role rotation (Heavy3-inspired) | `True` |
| `convergence_detection` | Semantic convergence auto-detection | `True` |
| `vote_grouping` | Merge semantically similar vote choices | `True` |

### ArenaConfig Options

Dependency injection via ArenaConfig:

| Option | Description | Default |
|--------|-------------|---------|
| `performance_monitor` | AgentPerformanceMonitor instance | `None` |
| `enable_performance_monitor` | Auto-create PerformanceMonitor | `False` |
| `enable_telemetry` | Prometheus/Blackbox emission | `False` |
| `use_airlock` | Wrap agents with AirlockProxy | `False` |
| `airlock_config` | Custom AirlockConfig | `None` |
| `population_manager` | Genesis PopulationManager | `None` |
| `auto_evolve` | Trigger evolution after debates (gated) | `False` |
| `breeding_threshold` | Min confidence for evolution | `0.8` |

Safety gates:
- Set `ARAGORA_ALLOW_AUTO_EVOLVE=1` to enable `auto_evolve`.
- Set `ARAGORA_ALLOW_PROMPT_EVOLVE=1` to enable prompt evolution.

## Performance Considerations

1. **Embedding Caching**: SemanticRetriever caches embeddings to avoid recomputation
2. **Checkpoint Intervals**: Configure based on debate length and failure risk
3. **Parallel Debate Branches**: DebateForker runs branches concurrently
4. **Connection Pooling**: Database connections are pooled for efficiency
5. **Lazy Loading**: Heavy components (Z3, embeddings) loaded on demand
