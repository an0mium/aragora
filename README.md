# aragora (Agent Agora): AI Red Team for Decision Stress-Testing

[![Tests](https://github.com/an0mium/aragora/actions/workflows/test.yml/badge.svg)](https://github.com/an0mium/aragora/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/an0mium/aragora/branch/main/graph/badge.svg)](https://codecov.io/gh/an0mium/aragora)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aragora is an **adversarial validation engine**. It stress-tests high-stakes specs, architectures, policies, and code by running multi-agent red-team debates and producing audit-ready Decision Receipts, risk heatmaps, and dissent trails.

Debate is the engine. The product is a defensible decision record.

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

## Debate Engine (Dialectic Roots)

The dialectic framing is the **internal engine**; users get adversarial validation outputs (decision receipts, risk heatmaps, dissent trails). If you're here for outcomes, this section is optional background.

Aragora's debate engine draws from Hegelian dialectics:

| Dialectical Concept | Aragora Implementation |
|---------------------|------------------------|
| **Thesis → Antithesis → Synthesis** | Propose → Critique → Revise loop |
| **Aufhebung** (sublation) | Judge synthesizes best elements, preserving value while transcending limitations |
| **Contradiction as motor** | Critiques (disagreement) drive improvement, not consensus-seeking |
| **Negation of negation** | Proposal → Critique (negation) → Revision (higher unity) |
| **Truth as totality** | No single agent has complete truth; it emerges from multi-perspectival synthesis |

The **nomic loop** (self-modifying rules) mirrors this internal engine by debating and refining its own processes. It is experimental—run in a sandbox and require human review before any auto-commit.

## Key Features

### Core Capabilities
- **Gauntlet Mode (Decision Stress-Test)**: Red-team attacks, probes, and scenario tests for specs, policies, and architectures
- **Decision Receipts**: Audit-ready artifacts with evidence chains and dissent tracking
- **Heterogeneous Agents**: Mix Claude, GPT, Gemini, Grok, Mistral (EU perspective), and Chinese models like DeepSeek, Qwen, and Kimi, plus local models in the same debate
- **Structured Debate Protocol**: Propose → Critique → Revise loop with configurable rounds
- **Multiple Consensus Mechanisms**: Majority voting, unanimous, judge-based, or none
- **Self-Improvement**: SQLite-based pattern store learns from successful critiques
- **CLI Interface**: One command, multiple agents working behind the scenes
- **12+ Model Providers**: Anthropic, OpenAI, Google, xAI, Mistral, DeepSeek, Qwen, Kimi, and more via OpenRouter

### Nomic Loop (Self-Improving System)
The **Nomic Loop** is aragora's autonomous self-improvement system—a society of AI agents that debates and implements improvements to its own codebase:

- **Multi-Phase Debate**: Context gathering → Proposal → Critique → Voting → Implementation
- **4 Specialized Agents**: Claude (visionary), Codex (engineer), Gemini (architect), Grok (lateral thinker)
- **Auto-Implementation**: Consensus proposals are automatically implemented and tested
- **Crash Recovery**: Checkpoint/resume system survives interruptions
- **Live Streaming**: Watch debates in real-time at [aragora.ai](https://aragora.ai)

```bash
# Run the nomic loop (experimental; review changes before auto-commit)
python scripts/run_nomic_with_stream.py run --cycles 24 --auto
```

### Live Dashboard
Real-time debate visualization at **[aragora.ai](https://aragora.ai)**:
- WebSocket streaming of agent proposals and critiques
- Argument graph visualization
- Late-joiner state sync
- Multi-loop support

## AI Red Team Code Review

Get **unanimous AI consensus** on your pull requests. When 3 independent AI models agree on an issue, you know it's worth fixing. This is Gauntlet configured for code-level stress-testing.

```bash
# Review a PR
git diff main | aragora review

# Review a GitHub PR URL
aragora review https://github.com/owner/repo/pull/123

# Try without API keys
aragora review --demo
```

**What you get:**

| Section | What It Means |
|---------|--------------|
| **Unanimous Issues** | All AI models agree → High confidence, fix first |
| **Split Opinions** | Models disagree → See the tradeoff, you decide |
| **Risk Areas** | Low confidence → Manual review recommended |

Example output:
```
### Unanimous Issues
> All AI models agree - address these first
- SQL injection in search_users() - user input concatenated into query
- Missing input validation on file upload endpoint

### Split Opinions
| Topic | For | Against |
|-------|-----|---------|
| Add request rate limiting | Claude, GPT-4 | Gemini |
```

**GitHub Actions**: Automatically review every PR with the included workflow.

## Gauntlet Mode - Adversarial Stress Testing

Stress-test your specifications, architectures, and policies before they ship:

```bash
# Test a specification for security vulnerabilities
aragora gauntlet spec.md --input-type spec --profile quick

# GDPR compliance audit
aragora gauntlet policy.yaml --input-type policy --persona gdpr

# Full adversarial stress test with HTML report
aragora gauntlet architecture.md --profile thorough --output report.html
```

**What you get:**

| Attack Type | What It Tests |
|-------------|--------------|
| **Red Team** | Security holes, injection points, auth bypasses |
| **Devil's Advocate** | Logic flaws, hidden assumptions, edge cases |
| **Scaling Critic** | Performance bottlenecks, SPOF, thundering herd |
| **Compliance** | GDPR, HIPAA, SOC 2, AI Act violations |

**Decision receipts** provide cryptographic audit trails for every finding, ready for regulatory review.

CI decision gate:

```bash
aragora gauntlet architecture.md --profile thorough --output receipt.html
```

GitHub Action: `.github/workflows/aragora-gauntlet.yml`

See [docs/GAUNTLET.md](docs/GAUNTLET.md) for full documentation and [docs/AGENT_SELECTION.md](docs/AGENT_SELECTION.md) for agent recommendations.

Case studies:
- `docs/case-studies/README.md`

## Quick Start

Start here: [docs/START_HERE.md](docs/START_HERE.md) for the canonical 5-minute setup.

```bash
# Clone and install
git clone https://github.com/an0mium/aragora.git
cd aragora
pip install -e .

# Set at least one API key
export ANTHROPIC_API_KEY=your-key  # or OPENAI_API_KEY, GEMINI_API_KEY, XAI_API_KEY

# Run a debate with API agents (recommended - no CLI tools needed)
aragora ask "Design a rate limiter for 1M requests/sec" \
  --agents anthropic-api,openai-api

# With more agents and custom consensus
aragora ask "Implement a secure auth system" \
  --agents anthropic-api,openai-api,gemini,grok \
  --rounds 4 \
  --consensus majority
```

> **See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for the complete setup and usage guide.**

### Python SDK

Use the type-safe Python SDK for programmatic access:

```python
from aragora.client import AragoraClient

# Synchronous
client = AragoraClient(base_url="http://localhost:8080")
debate = client.debates.run(task="Should we adopt microservices?")
print(f"Consensus: {debate.consensus.reached}")

# Asynchronous
async with AragoraClient(base_url="http://localhost:8080") as client:
    debate = await client.debates.run_async(task="Design a rate limiter")
    receipt = await client.gauntlet.run_and_wait(input_content="spec.md")
```

See [docs/SDK_GUIDE.md](docs/SDK_GUIDE.md) for full API reference.

### Chat Integrations

Post debate notifications to Discord and Slack:

```python
from aragora.integrations.discord import DiscordConfig, DiscordIntegration

discord = DiscordIntegration(DiscordConfig(
    webhook_url="https://discord.com/api/webhooks/..."
))
await discord.send_consensus_reached(debate_id, topic, "majority", result)
```

See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) for setup instructions.

## Supported Entry Points

Stable interfaces (recommended):
- `aragora gauntlet` for decision stress-tests (CLI)
- `aragora ask` for exploratory debates (CLI)
- `aragora serve` for the unified API + WebSocket server
- `python -m aragora` as a CLI alias
- `python -m aragora.server` for the server in scripts/automation

Experimental/research (may change; use in a sandbox):
- `scripts/nomic_loop.py` and `scripts/run_nomic_with_stream.py`
- `aragora improve` (self-improvement mode)

## Prerequisites

**API Agents (Recommended):** Just set your API keys - no additional tools needed:

```bash
# Set one or more API keys in .env or environment
ANTHROPIC_API_KEY=sk-ant-xxx    # For Claude (Opus 4.5, Sonnet 4)
OPENAI_API_KEY=sk-xxx           # For GPT models
GEMINI_API_KEY=AIzaSy...        # For Gemini 2.5
XAI_API_KEY=xai-xxx             # For Grok 4
MISTRAL_API_KEY=xxx             # For Mistral Large, Codestral
OPENROUTER_API_KEY=sk-or-xxx    # For DeepSeek, Qwen, Yi (multi-model access)
KIMI_API_KEY=xxx                # For Kimi (Moonshot, China perspective)
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
│                         ARAGORA FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│    │ Claude │ │ Gemini │ │  Grok  │ │Mistral │ │ OpenAI │       │
│    └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘       │
│        │    ┌─────┴─────┐    │    ┌─────┴─────┐    │            │
│        │    │ DeepSeek  │    │    │   Qwen    │    │            │
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
# Run a decision stress-test
aragora ask "Your task here" --agents codex,claude --rounds 3

# View statistics
aragora stats

# View learned patterns
aragora patterns --type security --limit 20

# Run the API + WebSocket server
aragora serve

# System health check
aragora doctor

# Export for training
aragora export --format jsonl > training_data.jsonl
```

## Debate Protocol (Stress-Test Engine)

Each stress-test session follows the debate engine structure of thesis → antithesis → synthesis:

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

Aragora learns from successful stress-tests through a structured feedback loop; the engine critiques itself to harden future outputs. This is powerful but risky—keep the nomic loop sandboxed and human-reviewed.

1. **Pattern Storage**: Successful critique→fix patterns are indexed by issue type
2. **Retrieval**: Future debates can retrieve relevant patterns (learning from history)
3. **Prompt Evolution**: Agent system prompts evolve based on what works
4. **Nomic Loop**: The system debates *changes to itself*, making its own rules an object of dialectical inquiry (guarded by safety gates)
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

## Implemented Features (65+ Components)

Aragora has evolved through 21 phases of self-improvement, with the nomic loop debating and implementing each feature:

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

- [x] **Phase 1-21**: Core framework with 65+ integrated features ✓
- [x] **Position Flip Detection**: Track agent position reversals and consistency scores ✓
- [x] **Hybrid Model Architecture**: Gemini=Designer, Claude=Implementer, Codex=Verifier ✓
- [x] **Security Hardening**: API key header auth, rate limiting, input validation ✓
- [x] **Feature Integration**: PerformanceMonitor, CalibrationTracker, Airlock, Telemetry ✓
- [x] **Multi-Provider Agents**: Mistral, DeepSeek, Qwen, Yi, Kimi via direct API and OpenRouter ✓
- [ ] **LeanBackend**: Lean 4 theorem proving integration
- [ ] **Emergent Society**: Society simulation (ala Project Sid)
- [ ] **Multi-Codebase**: Cross-repository coordination

## Deployment

### AWS Lightsail (Production)

The aragora API server runs on AWS Lightsail with Cloudflare Tunnel:

```bash
# Deploy to Lightsail
./deploy/lightsail-setup.sh

# The server runs at api.aragora.ai via Cloudflare Tunnel
# Frontend at aragora.ai via Cloudflare Pages
```

Configuration:
- **Instance**: Ubuntu 22.04, nano_3_0 ($5/month)
- **HTTP API**: Port 8080
- **WebSocket**: Port 8765 (ws://host:8765 or ws://host:8765/ws)
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
| `WS /ws` | Real-time debate streaming (WebSocket server port) |

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
export ARAGORA_ALLOWED_ORIGINS="https://aragora.ai,https://www.aragora.ai"
export ARAGORA_TOKEN_TTL=3600                   # Token lifetime in seconds
export ARAGORA_WS_MAX_MESSAGE_SIZE=65536        # Max WebSocket message size (bytes)
```

## Contributing

Contributions welcome! Areas of interest:

- Additional agent backends (Cohere, Inflection, Reka)
- Debate visualization enhancements
- Benchmark datasets for agent evaluation
- Prompt engineering for better critiques
- Self-improvement mechanism research
- Lean 4 theorem proving integration

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
