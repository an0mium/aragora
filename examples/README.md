# Aragora Examples

Quick examples to demonstrate Aragora's core capabilities.

## Prerequisites

Set at least one API key:
```bash
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="..."  # for Grok
```

For best results, set 2-3 keys to enable multi-agent debates.

## Examples

### 1. Simple Multi-Agent Debate (2-5 min)

```bash
python examples/01_simple_debate.py
```

Multiple AI agents debate a topic, critique each other, and reach consensus. This demonstrates the core value proposition of Aragora.

**What you'll see:**
- Agents with different roles (proposer, critic, synthesizer)
- Multi-round debate with critique and revision
- Consensus detection with confidence score

### 2. Tournament & Leaderboard (10-30 min)

```bash
python examples/02_tournament.py
```

Agents compete across multiple topics with ELO ranking. Shows how Aragora tracks agent performance over time.

**What you'll see:**
- Round-robin tournament format
- Multiple debate tasks
- Final standings with wins, points, and ELO ratings

### 3. Nomic Loop - Self-Improvement (5-10 min)

```bash
python examples/03_nomic_loop.py
```

Demonstrates Aragora's unique self-improvement capability (runs in dry-run mode).

**What you'll see:**
- Debate phase: agents propose improvements
- Design phase: agents architect the solution
- Implement phase: code generation (simulated)
- Verify phase: testing (simulated)

## Troubleshooting

**"Need at least 2 agents"**
- Set more API keys (at least 2 providers needed for debate)

**Timeout errors**
- Some API providers may be slow; wait for completion
- Try setting `ARAGORA_AGENT_TIMEOUT` to increase timeout

**Rate limit errors**
- Set `OPENROUTER_API_KEY` for automatic fallback
- Wait and retry

## Full Documentation

- [CLAUDE.md](../CLAUDE.md) - Architecture overview
- [docs/STATUS.md](../docs/STATUS.md) - Feature status
- [scripts/nomic_loop.py](../scripts/nomic_loop.py) - Full nomic loop implementation
