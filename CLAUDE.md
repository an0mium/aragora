# Claude Code Integration Guide

This document provides context for Claude Code when working with the Aragora codebase.

## Project Overview

Aragora is a multi-agent debate framework where heterogeneous AI agents discuss, critique, and improve each other's responses. It implements self-improvement through the **Nomic Loop** - an autonomous cycle where agents debate improvements, design solutions, implement code, and verify changes.

## Key Architecture

```
aragora/
├── debate/           # Core debate orchestration
│   ├── orchestrator.py     # Arena class - main debate engine (1,742 LOC)
│   ├── memory_manager.py   # Memory coordination (extracted)
│   ├── prompt_builder.py   # Prompt construction (extracted)
│   ├── security_barrier.py # Telemetry redaction (extracted)
│   ├── consensus.py        # Consensus detection and proofs
│   └── convergence.py      # Semantic similarity detection
├── agents/           # Agent implementations
│   ├── cli_agents.py    # CLI-based agents (claude, codex, gemini)
│   ├── api_agents/      # API-based agents (directory)
│   │   ├── anthropic.py    # Anthropic API agent
│   │   ├── openai.py       # OpenAI API agent
│   │   ├── gemini.py       # Google Gemini agent
│   │   ├── openrouter.py   # OpenRouter multi-model agent
│   │   └── ollama.py       # Local Ollama agent
│   └── fallback.py      # Quota detection and OpenRouter fallback
├── memory/           # Learning and persistence
│   ├── continuum.py     # Multi-tier memory (1,232 LOC)
│   └── consensus.py     # Historical debate outcomes
├── server/           # HTTP/WebSocket API
│   └── unified_server.py  # Main server (1,248 LOC, 72+ endpoints)
├── ranking/          # Agent skill tracking
│   └── elo.py           # ELO ratings and calibration (1,709 LOC)
├── resilience.py     # CircuitBreaker for agent failure handling
└── verification/     # Proof generation
    └── formal.py        # Z3/Lean verification backends
```

## Protected Files

The following files should NOT be modified by autonomous agents:
- `CLAUDE.md` - This file
- `core.py` - Core dataclasses and types
- `aragora/__init__.py` - Package exports
- `.env` - Environment configuration (never commit)
- `scripts/nomic_loop.py` - Critical for self-improvement safety

## Nomic Loop Context

The nomic loop (`scripts/nomic_loop.py`) is the autonomous self-improvement cycle:

1. **Phase 0: Context** - Gather codebase understanding
2. **Phase 1: Debate** - Agents propose improvements
3. **Phase 2: Design** - Architecture planning
4. **Phase 3: Implement** - Code generation (Codex/Claude)
5. **Phase 4: Verify** - Tests and checks

**Safety Features:**
- All changes backed up before implementation
- Protected files checksummed
- Automatic rollback on verification failure
- Human approval required for dangerous changes

## Important Patterns

### Debate Protocol
```python
from aragora import Arena, Environment, DebateProtocol

env = Environment(task="Design a rate limiter")
protocol = DebateProtocol(rounds=3, consensus="majority")
arena = Arena(env, agents, protocol)
result = await arena.run()
```

### Memory Tiers
- **Fast (1 min)**: Immediate context, high importance
- **Medium (1 hour)**: Session memory
- **Slow (1 day)**: Cross-session learning
- **Glacial (1 week)**: Long-term patterns

### Event Streaming
All debates emit events to WebSocket clients:
- `debate_start`, `round_start`, `agent_message`
- `critique`, `vote`, `consensus`, `debate_end`

## Common Tasks

### Running the Server
```bash
python -m aragora.server.unified_server --port 8080
```

### Running with Streaming
```bash
python scripts/run_nomic_with_stream.py run --cycles 3
```

### Testing
```bash
pytest tests/ -v
```

## Safety Guidelines

1. **Never modify protected files** without explicit approval
2. **Always run tests** after code changes
3. **Preserve existing functionality** - avoid breaking changes
4. **Use rate limiting** for API calls (respect provider limits)
5. **Log all changes** for audit trails
6. **Backup before modify** - always create backups

## Environment Variables

Required:
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` - At least one AI provider

Optional:
- `GEMINI_API_KEY`, `XAI_API_KEY`, `GROK_API_KEY` - Additional providers
- `OPENROUTER_API_KEY` - Fallback provider (auto-used when OpenAI returns 429)
- `SUPABASE_URL`, `SUPABASE_KEY` - Persistence
- `ARAGORA_API_TOKEN` - Auth token
- `ARAGORA_ALLOWED_ORIGINS` - CORS origins

See `docs/ENVIRONMENT.md` for full reference.

## Helpful Commands

```bash
# Check syntax
python -c "import ast; ast.parse(open('file.py').read())"

# Run specific test
pytest tests/test_orchestrator.py -v

# Check git status
git status && git diff --stat
```

## Feature Status

Well-integrated:
- Memory systems (CritiqueStore, ContinuumMemory)
- ELO rankings and tournaments
- Debate templates and TrendingTopicsPanel
- Verification proofs
- Belief networks (add_claim, propagation)
- CircuitBreaker for agent failure handling
- OpenRouter fallback for OpenAI quota errors
- User participation (votes/suggestions via WebSocket)
- TTL caching for expensive queries

Partially integrated:
- Pulse (trending topics) - works but may need API keys
- Evidence collection
- Calibration scoring

Recent additions (2026-01):
- `CircuitBreaker` class in aragora/resilience.py - handles failing agents gracefully
- `SecurityBarrier` and `TelemetryVerifier` in debate/security_barrier.py - telemetry redaction
- `MemoryManager` in debate/memory_manager.py - memory coordination (extracted from orchestrator)
- `PromptBuilder` in debate/prompt_builder.py - prompt construction (extracted from orchestrator)
- OpenRouter fallback in api_agents/ - auto-fallback when OpenAI returns 429
- `ttl_cache` decorator in handlers/base.py - simple TTL caching
- `from __future__ import annotations` added to 28 core modules for modern type hints
- Connector exception hierarchy in connectors/exceptions.py - structured error handling

See `docs/STATUS.md` for detailed feature status.
