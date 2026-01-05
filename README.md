# aragora (Agent Agora): Multi-Agent Debate Framework

> *"The truth is the whole."* — Hegel

A society of heterogeneous AI agents that discuss, critique, improve each other's responses, and learn from successful patterns. Aragora implements dialectical reasoning: truth emerges not from any single perspective, but through the productive tension of contradiction and synthesis.

**Domain**: [aragora.ai](https://aragora.ai) (available)

## Inspiration & Citations

aragora synthesizes ideas from these excellent open-source projects:

### Foundational Inspiration
- **[Stanford Generative Agents](https://github.com/joonspk-research/generative_agents)** - Memory + reflection architecture
- **[ChatArena](https://github.com/chatarena/chatarena)** - Game environments for multi-agent interaction
- **[LLM Multi-Agent Debate](https://github.com/composable-models/llm_multiagent_debate)** - ICML 2024 consensus mechanisms
- **[UniversalBackrooms](https://github.com/scottviteri/UniversalBackrooms)** - Multi-model infinite conversations
- **[Project Sid](https://github.com/altera-al/project-sid)** - Emergent civilization with 1000+ agents

### Borrowed Patterns (MIT/Apache Licensed)

We gratefully acknowledge these projects whose patterns we adapted:

| Project | What We Borrowed | License |
|---------|------------------|---------|
| **[ai-counsel](https://github.com/AI-Counsel/ai-counsel)** | Semantic convergence detection (3-tier fallback: SentenceTransformer → TF-IDF → Jaccard), vote option grouping, per-agent similarity tracking | MIT |
| **[DebateLLM](https://github.com/Tsinghua-MARS-Lab/DebateLLM)** | Agreement intensity modulation (0-10 scale), asymmetric debate roles (affirmative/negative/neutral stances), judge-based termination | Apache 2.0 |
| **[CAMEL-AI](https://github.com/camel-ai/camel)** | Multi-agent orchestration patterns, critic agent design | Apache 2.0 |
| **[CrewAI](https://github.com/joaomdmoura/crewAI)** | Agent role and task patterns | MIT |
| **[AIDO](https://github.com/aido-research/aido)** | Consensus variance tracking (strong/medium/weak classification), reputation-weighted voting concepts | MIT |

See `aragora/debate/convergence.py` and `aragora/debate/orchestrator.py` for implementations.

## Philosophical Foundation

Aragora's architecture embodies principles from Hegelian dialectics:

| Dialectical Concept | Aragora Implementation |
|---------------------|------------------------|
| **Thesis → Antithesis → Synthesis** | Propose → Critique → Revise loop |
| **Aufhebung** (sublation) | Judge synthesizes best elements, preserving value while transcending limitations |
| **Contradiction as motor** | Critiques (disagreement) drive improvement, not consensus-seeking |
| **Negation of negation** | Proposal → Critique (negation) → Revision (higher unity) |
| **Truth as totality** | No single agent has complete truth; it emerges from multi-perspectival synthesis |

The **nomic loop** (self-modifying rules) mirrors Hegel's concept of Spirit developing through its own internal contradictions—the system's structure becomes an object of its own dialectical process.

## Key Features

### Core Capabilities
- **Heterogeneous Agents**: Mix Claude, GPT/Codex, Gemini, Grok, and local models in the same debate
- **Structured Debate Protocol**: Propose → Critique → Revise loop with configurable rounds
- **Multiple Consensus Mechanisms**: Majority voting, unanimous, judge-based, or none
- **Self-Improvement**: SQLite-based pattern store learns from successful critiques
- **CLI Interface**: One command, multiple agents working behind the scenes

### Nomic Loop (Self-Improving System)
The **Nomic Loop** is aragora's autonomous self-improvement system—a society of AI agents that debates and implements improvements to its own codebase:

- **Multi-Phase Debate**: Context gathering → Proposal → Critique → Voting → Implementation
- **4 Specialized Agents**: Claude (visionary), Codex (engineer), Gemini (architect), Grok (lateral thinker)
- **Auto-Implementation**: Consensus proposals are automatically implemented and tested
- **Crash Recovery**: Checkpoint/resume system survives interruptions
- **Live Streaming**: Watch debates in real-time at [live.aragora.ai](https://live.aragora.ai)

```bash
# Run the nomic loop
python scripts/run_nomic_with_stream.py run --cycles 24 --auto
```

### Live Dashboard
Real-time debate visualization at **[live.aragora.ai](https://live.aragora.ai)**:
- WebSocket streaming of agent proposals and critiques
- Argument graph visualization
- Late-joiner state sync
- Multi-loop support

## Quick Start

```bash
# Clone and install
git clone https://github.com/an0mium/aragora.git
cd aragora
pip install -e .

# Set at least one API key
export ANTHROPIC_API_KEY=your-key  # or OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY

# Run a debate with API agents (recommended - no CLI tools needed)
python -m aragora.debate "Design a rate limiter for 1M requests/sec" \
  --agents anthropic-api openai-api

# With more agents and custom consensus
python -m aragora.debate "Implement a secure auth system" \
  --agents anthropic-api openai-api gemini-api grok \
  --rounds 4 \
  --consensus majority
```

> **See [docs/QUICKSTART.md](docs/QUICKSTART.md) for the complete 5-minute setup guide.**

## Prerequisites

**API Agents (Recommended):** Just set your API keys - no additional tools needed:

```bash
# Set one or more API keys in .env or environment
ANTHROPIC_API_KEY=sk-ant-xxx    # For Claude
OPENAI_API_KEY=sk-xxx           # For GPT models
GEMINI_API_KEY=AIzaSy...        # For Gemini
XAI_API_KEY=xai-xxx             # For Grok
```

**CLI Agents (Optional):** For local CLI-based agents, install the corresponding tools:

```bash
# OpenAI Codex CLI
npm install -g @openai/codex

# Claude CLI (claude-code)
npm install -g @anthropic-ai/claude-code

# Google Gemini CLI
npm install -g @google/gemini-cli

# xAI Grok CLI
npm install -g grok-cli
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AAGORA FRAMEWORK                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│    │ Claude │ │ Codex  │ │ Gemini │ │  Grok  │ │ OpenAI │       │
│    └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘       │
│        │    ┌─────┴─────┐    │    ┌─────┴─────┐    │            │
│        │    │   Qwen    │    │    │ Deepseek  │    │            │
│        │    └─────┬─────┘    │    └─────┬─────┘    │            │
│        └──────────┴──────────┴──────────┴──────────┘            │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    ARENA (Orchestrator)                  │    │
│  │  • Role assignment (proposer, critic, synthesizer)       │    │
│  │  • Round management                                      │    │
│  │  • Context accumulation                                  │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   DEBATE PROTOCOL                        │    │
│  │  • Propose → Critique → Revise loop                      │    │
│  │  • Sparse/all-to-all/round-robin topology                │    │
│  │  • Majority/unanimous/judge consensus                    │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   CRITIQUE STORE                         │    │
│  │  • SQLite-based pattern storage                          │    │
│  │  • Issue categorization (security, performance, etc.)    │    │
│  │  • Success rate tracking                                 │    │
│  │  • Export for fine-tuning                                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Basic Debate

```python
import asyncio
from aragora.agents import create_agent
from aragora.debate import Arena, DebateProtocol
from aragora.core import Environment
from aragora.memory import CritiqueStore

# Create heterogeneous agents
agents = [
    create_agent("codex", name="codex_proposer", role="proposer"),
    create_agent("claude", name="claude_critic", role="critic"),
    create_agent("codex", name="codex_synth", role="synthesizer"),
]

# Define task
env = Environment(
    task="Design a distributed cache with LRU eviction",
    max_rounds=3,
)

# Configure debate
protocol = DebateProtocol(
    rounds=3,
    consensus="majority",
)

# Run with memory
memory = CritiqueStore("debates.db")
arena = Arena(env, agents, protocol, memory)
result = asyncio.run(arena.run())

print(result.final_answer)
print(f"Consensus: {result.consensus_reached} ({result.confidence:.0%})")
```

### CLI Commands

```bash
# Run a debate
aragora ask "Your task here" --agents codex,claude --rounds 3

# View statistics
aragora stats

# View learned patterns
aragora patterns --type security --limit 20

# Export for training
aragora export --format jsonl > training_data.jsonl
```

## Debate Protocol

Each debate follows the dialectical structure of thesis → antithesis → synthesis:

1. **Round 0: Thesis (Initial Proposals)**
   - Proposer agents generate initial responses to the task
   - Multiple perspectives on the same problem

2. **Rounds 1-N: Antithesis (Critique & Revise)**
   - Agents critique each other's proposals (productive negation)
   - Identify issues with severity scores (0-1)
   - Provide concrete suggestions
   - Proposers revise, incorporating valid critiques (negation of negation)

3. **Synthesis (Consensus Phase)**
   - All agents vote on best proposal
   - Judge synthesizes best elements from competing proposals (*Aufhebung*)
   - Judge selection is randomized or voted to prevent systematic bias
   - Final answer transcends individual limitations

## Self-Improvement (Reflexive Development)

Aragora learns from successful debates through a process analogous to Hegel's self-developing Spirit—the system examines its own patterns, finds inadequacies, and transcends to higher forms:

1. **Pattern Storage**: Successful critique→fix patterns are indexed by issue type
2. **Retrieval**: Future debates can retrieve relevant patterns (learning from history)
3. **Prompt Evolution**: Agent system prompts evolve based on what works
4. **Nomic Loop**: The system debates *changes to itself*, making its own rules an object of dialectical inquiry
5. **Export**: Patterns can be exported for fine-tuning

```python
# Retrieve successful patterns
from aragora.memory import CritiqueStore

store = CritiqueStore("debates.db")
security_patterns = store.retrieve_patterns(issue_type="security", min_success=3)

for pattern in security_patterns:
    print(f"Issue: {pattern.issue_text}")
    print(f"Fix: {pattern.suggestion_text}")
    print(f"Success rate: {pattern.success_rate:.0%}")
```

## Implemented Features (54 Components)

Aragora has evolved through 11 phases of self-improvement, with the nomic loop debating and implementing each feature:

### Phase 1: Foundation
| Feature | Description |
|---------|-------------|
| **ContinuumMemory** | Multi-timescale learning (fast/medium/slow tiers) |
| **ReplayRecorder** | Cycle event recording for analysis |
| **MetaLearner** | Self-tuning hyperparameters |
| **IntrospectionAPI** | Agent self-awareness and reflection |
| **ArgumentCartographer** | Real-time debate graph visualization |
| **WebhookDispatcher** | External event notifications |

### Phase 2: Learning
| Feature | Description |
|---------|-------------|
| **ConsensusMemory** | Track settled vs contested topics across debates |
| **InsightExtractor** | Post-debate pattern learning and extraction |

### Phase 3: Evidence & Resilience
| Feature | Description |
|---------|-------------|
| **MemoryStream** | Per-agent persistent memory |
| **LocalDocsConnector** | Evidence grounding from codebase |
| **CounterfactualOrchestrator** | Deadlock resolution via forking |
| **CapabilityProber** | Agent quality assurance testing |
| **DebateTemplates** | Structured debate formats |

### Phase 4: Agent Evolution
| Feature | Description |
|---------|-------------|
| **PersonaManager** | Agent traits and expertise evolution |
| **PromptEvolver** | Prompt evolution from winning patterns |
| **Tournament** | Periodic competitive benchmarking |

### Phase 5: Intelligence
| Feature | Description |
|---------|-------------|
| **ConvergenceDetector** | Early stopping via semantic convergence |
| **MetaCritiqueAnalyzer** | Debate process feedback and recommendations |
| **EloSystem** | Persistent agent skill tracking |
| **AgentSelector** | Smart agent team selection |
| **RiskRegister** | Low-consensus risk tracking |

### Phase 6: Formal Reasoning
| Feature | Description |
|---------|-------------|
| **ClaimsKernel** | Structured typed claims with evidence tracking |
| **ProvenanceManager** | Cryptographic evidence chain integrity |
| **BeliefNetwork** | Probabilistic reasoning over uncertain claims |
| **ProofExecutor** | Executable verification of claims |
| **ScenarioMatrix** | Robustness testing across scenarios |

### Phase 7: Reliability & Audit
| Feature | Description |
|---------|-------------|
| **EnhancedProvenanceManager** | Staleness detection for living documents |
| **CheckpointManager** | Pause/resume and crash recovery |
| **BreakpointManager** | Human intervention breakpoints |
| **ReliabilityScorer** | Claim confidence scoring |
| **DebateTracer** | Audit logs and deterministic replay |

### Phase 8: Advanced Debates
| Feature | Description |
|---------|-------------|
| **PersonaLaboratory** | A/B testing, emergent traits, cross-pollination |
| **SemanticRetriever** | Pattern matching for similar critiques |
| **FormalVerificationManager** | Z3 theorem proving for logical claims |
| **DebateGraph** | DAG-based debates for complex disagreements |
| **DebateForker** | Parallel branch exploration |

### Phase 9: Truth Grounding
| Feature | Description |
|---------|-------------|
| **FlipDetector** | Semantic position reversal detection |
| **CalibrationTracker** | Prediction accuracy tracking (Brier score) |
| **GroundedPersonaManager** | Evidence-linked persona synthesis |
| **PositionTracker** | Agent position history with verification |

### Phase 10: Thread-Safe Audience Participation
| Feature | Description |
|---------|-------------|
| **ArenaMailbox** | Thread-safe event queue for live interaction |
| **LoopScoping** | Session-isolated streaming events |

### Phase 11: Operational Modes
| Feature | Description |
|---------|-------------|
| **OperationalModes** | Agent tool configuration switching |
| **CapabilityProber** | Agent vulnerability testing |
| **RedTeamMode** | Adversarial analysis of proposals |

## Roadmap

- [x] **Phase 1-11**: Core framework with 54 integrated features ✓
- [x] **Position Flip Detection**: Track agent position reversals and consistency scores ✓
- [x] **Hybrid Model Architecture**: Gemini=Designer, Claude=Implementer, Codex=Verifier ✓
- [x] **Security Hardening**: API key header auth, rate limiting, input validation ✓
- [ ] **Phase 12**: LeanBackend for Lean 4 theorem proving
- [ ] **Phase 13**: Emergent society simulation (ala Project Sid)
- [ ] **Phase 14**: Multi-codebase coordination

## Deployment

### AWS Lightsail (Production)

The aragora API server runs on AWS Lightsail with Cloudflare Tunnel:

```bash
# Deploy to Lightsail
./deploy/lightsail-setup.sh

# The server runs at api.aragora.ai via Cloudflare Tunnel
# Frontend at live.aragora.ai via Cloudflare Pages
```

Configuration:
- **Instance**: Ubuntu 22.04, nano_3_0 ($5/month)
- **API Server**: Port 8080 (HTTP) + 8765 (WebSocket)
- **Tunnel**: Cloudflare Tunnel proxies api.aragora.ai

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/leaderboard` | Agent ELO rankings |
| `GET /api/matches/recent` | Recent debate results |
| `GET /api/agent/{name}/history` | Agent ELO history |
| `GET /api/insights/recent` | Extracted debate insights |
| `GET /api/flips/recent` | Position reversal events |
| `GET /api/flips/summary` | Flip statistics by type/agent |
| `GET /api/agent/{name}/consistency` | Agent consistency score |
| `GET /api/calibration/leaderboard` | Agents ranked by calibration (accuracy vs confidence) |
| `GET /api/agent/{name}/calibration` | Detailed calibration metrics for agent |
| `GET /api/personas` | All agent personas with traits/expertise |
| `GET /api/agent/{name}/persona` | Persona for specific agent |
| `GET /api/agent/{name}/performance` | Agent performance summary |
| `GET /api/learning/evolution` | Learning pattern evolution over time |
| `WS /ws` | Real-time debate streaming |

### WebSocket Events

```typescript
// Event types streamed via WebSocket
type EventType =
  | "debate_start" | "debate_end"
  | "round_start" | "agent_message"
  | "critique" | "vote" | "consensus"
  | "flip_detected"   // Position reversal
  | "memory_recall"   // Historical context
  | "match_recorded"  // ELO update
  | "audience_drain"  // User events processed
  | "calibration"     // Accuracy metrics update
```

## Security

Aragora implements several security measures:

- **API Key Protection**: Gemini API keys are transmitted via HTTP headers, not URL parameters
- **Rate Limiting**: Thread-safe rate limiting with configurable limits per minute
- **Input Validation**: API parameters are validated and capped to prevent resource exhaustion
- **Content-Length Validation**: POST requests validated against max size (100MB general, 10MB JSON)
- **Multipart Limits**: Max 100 multipart form parts prevents DoS via form flooding
- **Path Traversal Protection**: Static file serving validates paths against base directory
- **CORS Validation**: Origin allowlist prevents unauthorized cross-origin requests (no wildcards)
- **Security Headers**: X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, Referrer-Policy
- **Generic Error Messages**: Internal errors don't leak stack traces to clients
- **Error Message Sanitization**: API error responses redact patterns resembling API keys and tokens
- **Process Cleanup**: CLI agents properly kill and await zombie processes on exceptions
- **Backpressure Control**: Stream event queues are capped to prevent memory exhaustion
- **WebSocket Message Limits**: Max message size of 64KB prevents memory exhaustion
- **Debate Timeouts**: Configurable per-debate and per-round timeouts prevent runaway processes
- **Connection Health**: WebSocket ping/pong (30s/10s) detects stale connections
- **Thread Safety**: Double-checked locking for shared executor initialization
- **Secure Client IDs**: Cryptographically random WebSocket client identifiers
- **JSON Parse Timeout**: 5-second timeout prevents CPU-bound DoS attacks
- **Payload Validation**: WebSocket payloads validated for structure and size (max 10KB)
- **Upload Rate Limiting**: IP-based limits (5/min, 30/hour) prevent storage DoS

Configure security via environment variables:
```bash
export ARAGORA_API_TOKEN="your-secret-token"    # Enable token auth
export ARAGORA_ALLOWED_ORIGINS="https://aragora.ai,https://live.aragora.ai"
export ARAGORA_TOKEN_TTL=3600                   # Token lifetime in seconds
export ARAGORA_WS_MAX_SIZE=65536                # Max WebSocket message size (bytes)
```

## Contributing

Contributions welcome! Areas of interest:

- Additional agent backends (Llama, Mistral, Cohere)
- Debate visualization
- Benchmark datasets
- Prompt engineering for better critiques
- Self-improvement mechanisms

## License

MIT

## Acknowledgments

This project was inspired by a conversation exploring the intersection of:
- Multi-agent AI systems
- Competitive-collaborative dynamics
- Self-improvement through critique
- Emergent behavior in AI societies
- Hegelian dialectics and the structure of reason

The name "aragora" evokes the Greek *agora* (ἀγορά)—the public assembly where citizens debated and reached collective decisions through reasoned discourse.

Special thanks to the researchers behind Generative Agents, ChatArena, and Project Sid for pioneering this space, and to Hegel for the insight that contradiction is not a flaw to avoid but the engine of development.
