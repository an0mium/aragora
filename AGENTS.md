# Aragora Agents

Aragora is the control plane for multi-agent robust decisionmaking across organizational knowledge and channels. It implements structured robust decisionmaking through a society of heterogeneous AI agents. This document describes the agent system architecture.

## Agent Types

Aragora supports 20+ agent types across three backends:

### CLI-Based Agents

These agents invoke external CLI tools (use these agent type IDs with `create_agent()`):

| Agent Type | CLI Tool | Default Model |
|------------|----------|---------------|
| `claude` | `claude` (claude-code) | claude-sonnet-4 |
| `codex` | `codex` | gpt-5.2-codex |
| `openai` | `openai` | gpt-4o |
| `gemini-cli` | `gemini` | gemini-3-pro-preview |
| `grok-cli` | `grok` | grok-4 |
| `qwen-cli` | `qwen` | qwen3-coder |
| `deepseek-cli` | `deepseek` | deepseek-v3 |
| `kilocode` | `kilocode` | gemini-explorer (provider id) |

### API-Based Agents (Direct)

These agents make direct HTTP API calls to provider endpoints:

| Agent Type | API | Default Model | Env Var |
|------------|-----|---------------|---------|
| `anthropic-api` | Anthropic | claude-opus-4-5-20251101 | `ANTHROPIC_API_KEY` |
| `openai-api` | OpenAI | gpt-5.2 | `OPENAI_API_KEY` |
| `gemini` | Google | gemini-3-pro-preview | `GEMINI_API_KEY` |
| `grok` | xAI | grok-3 | `XAI_API_KEY` |
| `mistral-api` | Mistral | mistral-large-2512 | `MISTRAL_API_KEY` |
| `codestral` | Mistral | codestral-latest | `MISTRAL_API_KEY` |
| `ollama` | Local Ollama | llama3.2 | `OLLAMA_HOST` |
| `kimi` | Moonshot | moonshot-v1-8k | `KIMI_API_KEY` |

### API-Based Agents (via OpenRouter)

These agents use OpenRouter for unified multi-model access:

| Agent Type | Model | Description |
|------------|-------|-------------|
| `deepseek` | deepseek/deepseek-chat-v3-0324 | DeepSeek V3 (chat) - excellent for coding |
| `deepseek-r1` | deepseek/deepseek-r1 | DeepSeek R1 - chain-of-thought reasoning |
| `llama` | meta-llama/llama-3.3-70b-instruct | Llama 3.3 70B |
| `mistral` | mistralai/mistral-large-2411 | Mistral Large via OpenRouter |
| `qwen` | qwen/qwen-2.5-coder-32b-instruct | Qwen 2.5 Coder |
| `qwen-max` | qwen/qwen-max | Qwen Max - flagship reasoning |
| `yi` | 01-ai/yi-large | Yi Large - balanced capabilities |

All OpenRouter agents require `OPENROUTER_API_KEY`.

## Agent Creation

Use the factory function to create agents:

```python
from aragora.agents import create_agent

# CLI agents
agent = create_agent("claude", name="claude_proposer", role="proposer")
agent = create_agent("codex", name="codex_critic", role="critic")

# API agents
agent = create_agent("anthropic-api", name="claude_api", role="synthesizer", api_key="...")
agent = create_agent("gemini", name="gemini_judge", role="synthesizer", api_key="...")
agent = create_agent("ollama", name="local_agent", model="llama3.2")
```

## Agent Roles

Each agent has a role that determines its behavior in debates:

- **proposer** - Generates initial responses to tasks
- **critic** - Analyzes and critiques other proposals
- **synthesizer** - Creates final consensus answers from multiple proposals

## Core Agent Interface

All agents implement the abstract `Agent` class from `aragora/core.py`:

```python
class Agent(ABC):
    name: str
    model: str
    role: str  # "proposer", "critic", "synthesizer"
    system_prompt: str
    stance: str  # "affirmative", "negative", "neutral"

    async def generate(self, prompt: str, context: list[Message] = None) -> str
    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique
    async def vote(self, proposals: dict[str, str], task: str) -> Vote
```

## Agent Personas

Agents can have personas with expertise domains and personality traits:

**Expertise Domains:**
- security, performance, architecture, testing, error_handling
- concurrency, api_design, database, frontend, devops, documentation, code_style

**Personality Traits:**
- thorough, pragmatic, innovative, conservative
- diplomatic, direct, collaborative, contrarian

```python
from aragora.agents import PersonaManager

personas = PersonaManager("aragora_personas.db")
personas.create_persona(
    "claude_proposer",
    description="Visionary architect of solutions",
    traits=["innovative", "collaborative"],
    expertise={"architecture": 0.8, "api_design": 0.7},
)
```

## Truth-Grounded Identities

Agents maintain evidence-based identities through position tracking:

- **Position Ledger** - Records every claim, confidence level, and outcome
- **Calibration Scores** - Tracks prediction accuracy per domain
- **Relationship Metrics** - Rivalry, alliance, and influence scores between agents

## Debate Orchestration

The `Arena` class orchestrates multi-agent debates:

```python
from aragora.agents import create_agent
from aragora.debate import Arena, DebateProtocol
from aragora.core import Environment
from aragora.memory import CritiqueStore

agents = [
    create_agent("anthropic-api", name="proposer", role="proposer"),
    create_agent("openai-api", name="critic", role="critic"),
    create_agent("gemini", name="judge", role="synthesizer"),
]

env = Environment(task="Design a rate limiter", max_rounds=3)
protocol = DebateProtocol(rounds=3, consensus="majority")
memory = CritiqueStore("debates.db")

arena = Arena(env, agents, protocol, memory)
result = await arena.run()
```

### Debate Topologies

- **all-to-all** - Everyone critiques everyone
- **round-robin** - Deterministic cycle
- **ring** - Circular neighborhood critiques
- **star** - Hub agent central to all critiques
- **sparse** - Random subset based on sparsity parameter
- **random-graph** - Randomized connections

### Consensus Mechanisms

- **majority** - Plurality vote wins
- **unanimous** - All agents must agree
- **judge** - Single judge synthesizes best elements
- **none** - No voting, collect all proposals

## ELO Ranking System

Agents are ranked using an ELO-based skill system:

```python
from aragora.ranking import EloSystem

elo = EloSystem("aragora_elo.db")
elo.record_match(
    debate_id="debate_123",
    winner="claude_proposer",
    participants=["claude_proposer", "codex_critic"],
    domain="architecture",
    scores={"claude_proposer": 0.8, "codex_critic": 0.6},
)
```

## Nomic Loop Integration

The nomic loop (`scripts/nomic_loop.py`) is a self-improving cycle that leverages all agent features:

### Integrated Features

| Feature | Integration Point |
|---------|------------------|
| **Belief Analysis** | Runs on every debate, identifies contested/crux claims |
| **Capability Probing** | Probes agents before debates, weights votes by reliability |
| **Counterfactual Branches** | Resolves deadlocks by exploring alternative assumptions |
| **ELO Team Selection** | Selects agents by domain-specific expertise scores |
| **Evidence Staleness** | Checks claims against changed files, flags for re-debate |
| **Persona Evolution** | Applies winning experiment variants to agent traits |
| **Position Tracking** | Records all claims with outcomes for calibration |

### NomicIntegration Hub

The `NomicIntegration` class coordinates advanced features:

```python
from aragora.nomic.integration import create_nomic_integration

integration = create_nomic_integration(
    elo_system=elo,
    enable_probing=True,
    enable_belief_analysis=True,
    enable_staleness_check=True,
    enable_counterfactual=True,
)

# After debate
analysis = await integration.full_post_debate_analysis(
    result,
    arena=arena,
    claims_kernel=claims_kernel,
    changed_files=changed_files,
)
```

## Key Files

| File | Purpose |
|------|---------|
| `aragora/core.py` | Core abstractions (Agent, Message, Critique, Vote, DebateResult) |
| `aragora/agents/base.py` | Agent factory and type definitions |
| `aragora/agents/cli_agents.py` | CLI-based agent implementations |
| `aragora/agents/api_agents.py` | API-based agent implementations |
| `aragora/agents/personas.py` | Persona management and traits |
| `aragora/agents/laboratory.py` | Emergent persona evolution experiments |
| `aragora/agents/grounded.py` | Truth-grounded identity tracking |
| `aragora/agents/truth_grounding.py` | Position ledger for evidence tracking |
| `aragora/debate/orchestrator.py` | Arena class and DebateProtocol |
| `aragora/nomic/integration.py` | NomicIntegration feature coordination hub |
| `aragora/ranking/elo.py` | ELO skill ranking system |
| `aragora/memory/store.py` | CritiqueStore pattern database |
| `scripts/nomic_loop.py` | Self-improving nomic loop orchestrator |

## Theoretical Foundation

Aragora implements Hegelian dialectics:

| Concept | Implementation |
|---------|----------------|
| Thesis → Antithesis → Synthesis | Propose → Critique → Revise loop |
| Aufhebung (sublation) | Judge synthesizes best elements |
| Contradiction as motor | Critiques drive improvement |
| Truth as totality | Emerges from multi-perspectival synthesis |
