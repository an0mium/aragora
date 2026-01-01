# aagora (Agent Agora): Multi-Agent Debate Framework

> A society of heterogeneous AI agents that discuss, critique, improve each other's responses, and learn from successful patterns.

**Domain**: [aagora.ai](https://aagora.ai) (available)

## Inspiration

aagora synthesizes ideas from:
- **[Stanford Generative Agents](https://github.com/joonspk-research/generative_agents)** - Memory + reflection architecture
- **[ChatArena](https://github.com/chatarena/chatarena)** - Game environments for multi-agent interaction
- **[LLM Multi-Agent Debate](https://github.com/composable-models/llm_multiagent_debate)** - ICML 2024 consensus mechanisms
- **[UniversalBackrooms](https://github.com/scottviteri/UniversalBackrooms)** - Multi-model infinite conversations
- **[Project Sid](https://github.com/altera-al/project-sid)** - Emergent civilization with 1000+ agents

## Key Features

- **Heterogeneous Agents**: Mix Claude, GPT/Codex, Gemini, and local models in the same debate
- **Structured Debate Protocol**: Propose → Critique → Revise loop with configurable rounds
- **Multiple Consensus Mechanisms**: Majority voting, unanimous, judge-based, or none
- **Self-Improvement**: SQLite-based pattern store learns from successful critiques
- **CLI Interface**: One command, multiple agents working behind the scenes

## Quick Start

```bash
# Clone and install
git clone https://github.com/an0mium/aagora.git
cd aagora
pip install -e .

# Run a debate
aagora ask "Design a rate limiter for 1M requests/sec" --agents codex,claude

# With more agents and rounds
aagora ask "Implement a secure auth system" \
  --agents codex:proposer,claude:critic,openai:synthesizer \
  --rounds 4 \
  --consensus judge
```

## Prerequisites

You need at least one of these CLI tools installed:

```bash
# OpenAI Codex CLI
npm install -g @openai/codex

# Claude CLI (claude-code)
npm install -g @anthropic-ai/claude-code

# Google Gemini CLI
npm install -g @google/gemini-cli

# xAI Grok CLI
npm install -g grok-cli

# Alibaba Qwen Code CLI
npm install -g @qwen-code/qwen-code

# Deepseek CLI
pip install deepseek-cli

# OpenAI CLI
pip install openai
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
from aagora.agents import create_agent
from aagora.debate import Arena, DebateProtocol
from aagora.core import Environment
from aagora.memory import CritiqueStore

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
aagora ask "Your task here" --agents codex,claude --rounds 3

# View statistics
aagora stats

# View learned patterns
aagora patterns --type security --limit 20

# Export for training
aagora export --format jsonl > training_data.jsonl
```

## Debate Protocol

Each debate follows this structure:

1. **Round 0: Initial Proposals**
   - Proposer agents generate initial responses to the task

2. **Rounds 1-N: Critique & Revise**
   - Critic agents analyze each proposal
   - Identify issues (severity 0-1)
   - Provide concrete suggestions
   - Proposers revise based on critiques

3. **Consensus Phase**
   - All agents vote on best proposal
   - Synthesizer may combine best elements
   - Final answer selected by consensus mechanism

## Self-Improvement

aagora learns from successful debates:

1. **Pattern Storage**: Successful critique→fix patterns are indexed by issue type
2. **Retrieval**: Future debates can retrieve relevant patterns
3. **Prompt Evolution**: Agent system prompts can be updated based on what works
4. **Export**: Patterns can be exported for fine-tuning

```python
# Retrieve successful patterns
from aagora.memory import CritiqueStore

store = CritiqueStore("debates.db")
security_patterns = store.retrieve_patterns(issue_type="security", min_success=3)

for pattern in security_patterns:
    print(f"Issue: {pattern.issue_text}")
    print(f"Fix: {pattern.suggestion_text}")
    print(f"Success rate: {pattern.success_rate:.0%}")
```

## Roadmap

- [x] **Phase 1**: Core debate loop (current)
- [ ] **Phase 2**: Semantic retrieval for pattern matching
- [ ] **Phase 3**: Prompt evolution based on success patterns
- [ ] **Phase 4**: Meta-critique (agents critique the debate process)
- [ ] **Phase 5**: Emergent society simulation (ala Project Sid)

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

Special thanks to the researchers behind Generative Agents, ChatArena, and Project Sid for pioneering this space.
