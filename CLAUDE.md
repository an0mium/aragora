# Claude Code Integration Guide

Context for Claude Code when working with the Aragora codebase.

## Quick Reference

| What | Where | Key Files |
|------|-------|-----------|
| Debate engine | `aragora/debate/` | `orchestrator.py`, `consensus.py` |
| Agents | `aragora/agents/` | `cli_agents.py`, `api_agents/` |
| Server | `aragora/server/` | `unified_server.py`, `handlers/` |
| Memory | `aragora/memory/` | `continuum.py`, `consensus.py` |
| Nomic loop | `scripts/` | `nomic_loop.py`, `run_nomic_with_stream.py` |
| Reasoning | `aragora/reasoning/` | `belief.py`, `provenance.py`, `claims.py` |
| Workflow | `aragora/workflow/` | `engine.py`, `patterns/`, `nodes/` |
| RLM | `aragora/rlm/` | `factory.py`, `bridge.py`, `handler.py` |
| Enterprise | `aragora/auth/`, `aragora/tenancy/` | `oidc.py`, `isolation.py` |
| Connectors | `aragora/connectors/` | `slack.py`, `github.py`, `teams.py` |
| Resilience | `aragora/` | `resilience.py` (circuit breaker, 34KB) |

## Project Overview

Aragora is a multi-agent debate framework where heterogeneous AI agents discuss, critique, and improve each other's responses. It implements self-improvement through the **Nomic Loop** - an autonomous cycle where agents debate improvements, design solutions, implement code, and verify changes.

**Codebase Scale:** 1000+ Python modules | 34,400+ tests | 117 debate modules | 65 HTTP handlers + 15 WebSocket streams | 24+ enterprise connectors

## Architecture

```
aragora/
├── debate/           # Core debate orchestration
│   ├── orchestrator.py     # Arena class - main debate engine
│   ├── phases/             # Extracted phase implementations
│   ├── team_selector.py    # Agent team selection (ELO + calibration)
│   ├── memory_manager.py   # Memory coordination
│   ├── prompt_builder.py   # Prompt construction
│   ├── consensus.py        # Consensus detection and proofs
│   └── convergence.py      # Semantic similarity detection
├── agents/           # Agent implementations
│   ├── cli_agents.py       # CLI agents (claude, codex, gemini, grok)
│   ├── api_agents/         # API agents directory
│   │   ├── anthropic.py    # Anthropic API agent
│   │   ├── openai.py       # OpenAI API agent
│   │   ├── mistral.py      # Mistral API agent (Large, Codestral)
│   │   ├── grok.py         # xAI Grok agent
│   │   └── openrouter.py   # OpenRouter (DeepSeek, Llama, Qwen, Yi, Kimi)
│   ├── fallback.py         # OpenRouter fallback on quota errors
│   └── airlock.py          # AirlockProxy for agent resilience
├── memory/           # Learning and persistence
│   ├── continuum.py        # Multi-tier memory (fast/medium/slow/glacial)
│   └── consensus.py        # Historical debate outcomes
├── server/           # HTTP/WebSocket API
│   ├── unified_server.py   # Main server (~275 endpoints)
│   ├── handlers/           # HTTP endpoint handlers (119 modules)
│   └── stream/             # WebSocket streaming (14 modules)
├── ranking/          # Agent skill tracking
│   └── elo.py              # ELO ratings and calibration
├── resilience.py     # CircuitBreaker for agent failure handling
└── verification/     # Proof generation
    └── formal.py           # Z3/Lean verification backends
```

## Protected Files

**Do NOT modify without explicit approval:**
- `CLAUDE.md` - This file
- `core.py` - Core dataclasses and types
- `aragora/__init__.py` - Package exports
- `.env` - Environment configuration (never commit)
- `scripts/nomic_loop.py` - Critical for self-improvement safety

## Nomic Loop

The autonomous self-improvement cycle (`scripts/nomic_loop.py`):

| Phase | Name | Purpose |
|-------|------|---------|
| 0 | Context | Gather codebase understanding |
| 1 | Debate | Agents propose improvements |
| 2 | Design | Architecture planning |
| 3 | Implement | Code generation (Codex/Claude) |
| 4 | Verify | Tests and checks |

**Safety features:** Automatic backups, protected file checksums, rollback on failure, human approval for dangerous changes.

## Common Patterns

### Running a Debate
```python
from aragora import Arena, Environment, DebateProtocol

env = Environment(task="Design a rate limiter")
protocol = DebateProtocol(rounds=3, consensus="majority")
arena = Arena(env, agents, protocol)
result = await arena.run()
```

### Memory Tiers
| Tier | TTL | Purpose |
|------|-----|---------|
| Fast | 1 min | Immediate context |
| Medium | 1 hour | Session memory |
| Slow | 1 day | Cross-session learning |
| Glacial | 1 week | Long-term patterns |

### WebSocket Events
`debate_start`, `round_start`, `agent_message`, `critique`, `vote`, `consensus`, `debate_end`

## Commands

```bash
# Start server
python -m aragora.server.unified_server --port 8080

# Run nomic loop with streaming
python scripts/run_nomic_with_stream.py run --cycles 3

# Run tests
pytest tests/ -v

# Check syntax
python -c "import ast; ast.parse(open('file.py').read())"

# Quick git check
git status && git diff --stat
```

## Environment Variables

**Required** (at least one):
- `ANTHROPIC_API_KEY` - Anthropic API (Claude)
- `OPENAI_API_KEY` - OpenAI API (GPT)

**Recommended:**
- `OPENROUTER_API_KEY` - Fallback when primary APIs fail (auto-used on 429)
- `MISTRAL_API_KEY` - Mistral API (Large, Codestral)

**Optional:**
- `GEMINI_API_KEY`, `XAI_API_KEY`, `GROK_API_KEY` - Additional providers
- `SUPABASE_URL`, `SUPABASE_KEY` - Persistence
- `ARAGORA_API_TOKEN` - Auth token
- `ARAGORA_ALLOWED_ORIGINS` - CORS origins

See `docs/ENVIRONMENT.md` for full reference.

## Safety Guidelines

1. **Never modify protected files** without explicit approval
2. **Always run tests** after code changes
3. **Preserve existing functionality** - avoid breaking changes
4. **Use rate limiting** for API calls (respect provider limits)
5. **Backup before modify** - always create backups
6. **Log all changes** for audit trails

## Feature Status

**Test Suite:** 34,400+ tests across 928 test files

**Core (stable):**
- Debate orchestration (Arena, consensus, convergence)
- Memory systems (CritiqueStore, ContinuumMemory)
- ELO rankings and tournaments
- Agent fallback (OpenRouter on quota errors)
- CircuitBreaker for agent failure handling
- WebSocket event streaming
- User participation (votes/suggestions)

**Integrated:**
- PerformanceMonitor - via Arena and AutonomicExecutor
- CalibrationTracker - via `enable_calibration` protocol flag
- AirlockProxy - via `use_airlock` ArenaConfig option
- RhetoricalObserver - via `enable_rhetorical_observer`
- Trickster - hollow consensus detection via `enable_trickster`
- SecurityBarrier - telemetry redaction
- Graph/Matrix debate APIs
- RLM (Recursive Language Models) - REPL-based programmatic context access (NOT compression)
- Belief Network - claim provenance tracking
- Workflow Engine - DAG-based automation

**Enterprise (production-ready):**
- Authentication - OIDC/SAML SSO, MFA (TOTP/HOTP), API key management
- Multi-Tenancy - Tenant isolation, resource quotas, usage metering
- Security - AES-256-GCM encryption, rate limiting, circuit breakers
- Compliance - SOC 2 controls, GDPR support, audit trails
- Observability - Prometheus metrics, Grafana dashboards, OpenTelemetry tracing

**Partial:**
- Pulse (trending topics) - works but may need API keys
- Evidence collection - functional but limited connectors
- Knowledge Mound - experimental (Phase A1)

See `docs/STATUS.md` for 74+ detailed feature statuses.

## Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/COMMERCIAL_OVERVIEW.md` | Commercial positioning, 85% readiness assessment |
| `docs/ENTERPRISE_FEATURES.md` | Enterprise capabilities reference |
| `docs/FEATURE_DISCOVERY.md` | Complete feature catalog (100+ features) |
| `docs/STATUS.md` | Detailed feature implementation status |
| `docs/API_REFERENCE.md` | REST API documentation |
