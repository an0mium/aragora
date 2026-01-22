---
title: Aragora Developer Quick Start
description: Aragora Developer Quick Start
---

# Aragora Developer Quick Start

> Quick reference for developers working with Aragora

## Project Structure

```
aragora/
├── agents/           # Agent implementations (CLI, API)
├── debate/           # Core debate engine
│   ├── orchestrator.py    # Arena class (main coordinator)
│   ├── phases/            # Modular phase executors
│   └── context.py         # Shared debate state
├── memory/           # Memory systems
│   ├── continuum.py       # Multi-tier memory (fast/medium/slow/glacial)
│   ├── consensus.py       # Topic consensus tracking
│   └── postgres_*.py      # PostgreSQL implementations
├── ranking/          # ELO rating system
│   ├── elo.py             # SQLite-based EloSystem
│   └── postgres_database.py # PostgreSQL EloDatabase
├── server/           # HTTP/WebSocket API
│   ├── unified_server.py  # Main server
│   └── handlers/          # Request handlers
└── observability/    # Metrics and tracing
    └── metrics/           # Modular metrics (by domain)
```

## Running a Debate

```python
from aragora import Arena, Environment, DebateProtocol

# Configure
env = Environment(task="Design a rate limiter")
protocol = DebateProtocol(rounds=3, consensus="majority")

# Select agents
agents = ["claude", "gpt4", "gemini"]

# Run
arena = Arena(env, agents, protocol)
result = await arena.run()

print(f"Consensus: {result.consensus_reached}")
print(f"Answer: {result.final_answer}")
```

## Debate Phases

| Phase | Name | Description |
|-------|------|-------------|
| 0 | Context Init | Inject history, patterns, research context |
| 1 | Proposals | Generate initial proposer responses |
| 2 | Debate Rounds | Critique/revision loop with convergence detection |
| 3 | Consensus | Voting, weight calculation, winner determination |
| 4-6 | Analytics | Metrics, insights, verdict generation |
| 7 | Feedback | ELO updates, persona refinement, memory persistence |

## PostgreSQL Stores

All core stores have PostgreSQL implementations:

| Store | Class | Tables |
|-------|-------|--------|
| Consensus Memory | `PostgresConsensusMemory` | `consensus`, `dissent`, `verified_proofs` |
| Critique Store | `PostgresCritiqueStore` | `debates`, `critiques`, `patterns`, `agent_reputation` |
| Continuum Memory | `PostgresContinuumMemory` | `continuum_memory`, `tier_transitions` |
| ELO Rankings | `PostgresEloDatabase` | `elo_ratings`, `tournaments`, `matches` |

### Usage

```python
from aragora.memory.postgres_consensus import get_postgres_consensus_memory
from aragora.ranking.postgres_database import get_postgres_elo_database

# Set DATABASE_URL environment variable
memory = await get_postgres_consensus_memory()
elo_db = await get_postgres_elo_database()
```

## Memory Tiers

| Tier | TTL | Purpose |
|------|-----|---------|
| Fast | 1 min | Immediate context |
| Medium | 1 hour | Session memory |
| Slow | 1 day | Cross-session learning |
| Glacial | 1 week | Long-term patterns |

## Key Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `ANTHROPIC_API_KEY` | Claude API | One API key required |
| `OPENAI_API_KEY` | GPT API | One API key required |
| `DATABASE_URL` | PostgreSQL connection | For production |
| `OPENROUTER_API_KEY` | Fallback provider | Recommended |

## Adding a New Agent

```python
from aragora.core import Agent, Message, Critique

class MyAgent(Agent):
    name = "my_agent"

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        # Call your LLM
        return response

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
        # Generate critique
        return Critique(
            agent=self.name,
            target_agent=target,
            issues=["Issue 1"],
            suggestions=["Suggestion 1"],
            severity=0.5,
        )
```

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Specific module
pytest tests/debate/ -v

# Integration tests (requires DATABASE_URL)
DATABASE_URL=postgresql://... pytest tests/integration/ -v

# Quick syntax check
python -c "import aragora"
```

## Common Commands

```bash
# Start server
python -m aragora.server.unified_server --port 8080

# Run nomic loop
python scripts/run_nomic_with_stream.py run --cycles 3

# Check codebase health
python scripts/verify_system_health.py
```

## Key Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](../core-concepts/architecture) | System architecture overview |
| [API_REFERENCE.md](../api/reference) | REST API documentation |
| [POSTGRESQL_MIGRATION.md](../deployment/postgresql-migration) | PostgreSQL setup guide |
| [AGENT_DEVELOPMENT.md](../core-concepts/agent-development) | Creating custom agents |
| [RLM_GUIDE.md](./RLM_GUIDE.md) | Recursive Language Model usage |
| [MEMORY_TIERS.md](../core-concepts/memory) | Memory system design |

## Metrics

Metrics are organized by domain in `aragora/observability/metrics/`:

- `core.py` - Debate and request metrics
- `stores.py` - Storage operation metrics
- `bridge.py` - Knowledge bridge metrics
- `security.py` - Auth and security metrics
- `gauntlet.py` - Gauntlet run metrics

## Getting Help

- Check [CLAUDE.md](../CLAUDE.md) for AI assistant context
- See [docs/STATUS.md](../contributing/status) for feature implementation status
- Review [docs/ADR/](./ADR/) for architectural decisions
