# Claude Code Integration Guide

Context for Claude Code when working with the Aragora codebase.

## Quick Reference

| What | Where | Key Files |
|------|-------|-----------|
| Debate engine | `aragora/debate/` | `orchestrator.py`, `consensus.py` |
| Agents | `aragora/agents/` | `cli_agents.py`, `api_agents/` |
| Server | `aragora/server/` | `unified_server.py`, `handlers/`, `startup.py` |
| Memory | `aragora/memory/` | `continuum.py`, `consensus.py`, `coordinator.py` |
| Nomic loop | `scripts/` | `nomic_loop.py`, `run_nomic_with_stream.py` |
| Reasoning | `aragora/reasoning/` | `belief.py`, `provenance.py`, `claims.py` |
| Workflow | `aragora/workflow/` | `engine.py`, `patterns/`, `nodes/`, `templates/` |
| RLM | `aragora/rlm/` | `factory.py`, `bridge.py`, `handler.py` |
| Knowledge | `aragora/knowledge/` | `bridges.py`, `mound/`, `mound/resilience.py` |
| Explainability | `aragora/explainability/` | `builder.py`, `factors.py` |
| Gauntlet | `aragora/gauntlet/` | `receipts.py`, `runner.py`, `findings.py` |
| Enterprise | `aragora/auth/`, `aragora/tenancy/` | `oidc.py`, `isolation.py` |
| Connectors | `aragora/connectors/` | `slack.py`, `github.py`, `chat/`, `enterprise/streaming/` |
| Streaming | `aragora/connectors/enterprise/streaming/` | `kafka.py`, `rabbitmq.py` |
| Chat routing | `aragora/server/` | `debate_origin.py`, `result_router.py` |
| TTS/Voice | `aragora/server/stream/` | `tts_integration.py`, `voice_stream.py` |
| Control Plane | `aragora/control_plane/` | `policy.py`, `scheduler.py`, `notifications.py` |
| Resilience | `aragora/` | `resilience.py` (circuit breaker, 34KB) |
| KM Resilience | `aragora/knowledge/mound/` | `resilience.py` (retry, health, invalidation) |
| RBAC v2 | `aragora/rbac/` | `models.py`, `checker.py`, `decorators.py` |
| Backup | `aragora/backup/` | `manager.py` (disaster recovery) |
| Ops | `aragora/ops/` | `deployment_validator.py` (runtime validation) |

## Project Overview

Aragora is the **control plane for multi-agent robust decisionmaking across organizational knowledge and channels**. It orchestrates 15+ AI models—Claude, GPT, Gemini, Grok, Mistral, DeepSeek, Qwen, and more—to debate your organization's knowledge and deliver defensible decisions to any channel. It implements self-improvement through the **Nomic Loop** - an autonomous cycle where agents debate improvements, design solutions, implement code, and verify changes.

**Codebase Scale:** 1000+ Python modules | 45,100+ tests | 1220 test files | 117 debate modules | 70 HTTP handlers + 15 WebSocket streams | 26+ enterprise connectors

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
│   # Configurable concurrency: MAX_CONCURRENT_PROPOSALS, MAX_CONCURRENT_CRITIQUES, MAX_CONCURRENT_REVISIONS
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
│   ├── consensus.py        # Historical debate outcomes
│   └── coordinator.py      # Atomic cross-system memory writes
├── knowledge/        # Unified knowledge management
│   ├── bridges.py          # KnowledgeBridgeHub, MetaLearner, Evidence bridges
│   └── mound/              # KnowledgeMound with sync, revalidation
│       └── adapters/       # KM adapters (9 total)
│           └── factory.py  # Auto-create adapters from Arena subsystems
├── connectors/       # External integrations
│   ├── chat/               # Telegram, WhatsApp connectors
│   └── enterprise/
│       └── streaming/      # Event stream ingestion
│           ├── kafka.py    # Apache Kafka consumer
│           └── rabbitmq.py # RabbitMQ consumer/publisher
├── server/           # HTTP/WebSocket API
│   ├── unified_server.py   # Main server (~275 endpoints)
│   ├── startup.py          # Server startup sequence
│   ├── debate_origin.py    # Bidirectional chat result routing
│   ├── handlers/           # HTTP endpoint handlers (119 modules)
│   │   └── social/         # Chat platform handlers (Telegram, WhatsApp)
│   └── stream/             # WebSocket streaming (14 modules)
│       ├── tts_integration.py  # TTS for voice/chat
│       └── voice_stream.py     # Voice session management
├── ranking/          # Agent skill tracking
│   └── elo.py              # ELO ratings and calibration
├── resilience.py     # CircuitBreaker for agent failure handling
├── control_plane/    # Enterprise orchestration (142 tests)
│   ├── registry.py        # Agent discovery with heartbeats
│   ├── scheduler.py       # Priority-based task distribution
│   ├── health.py          # Liveness probes and monitoring
│   └── coordinator.py     # Unified control plane API
├── rbac/             # Role-based access control v2
│   ├── models.py           # Permission, Role, RoleAssignment dataclasses
│   ├── defaults.py         # 6 default roles, 50+ permissions
│   ├── checker.py          # PermissionChecker with caching
│   ├── decorators.py       # @require_permission, @require_role
│   ├── middleware.py       # HTTP route protection
│   └── audit.py            # Authorization audit logging
├── backup/           # Disaster recovery
│   └── manager.py          # BackupManager with incremental support
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

**Test Suite:** 45,100+ tests across 1,220 test files

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
- KnowledgeBridgeHub - unified access to MetaLearner, Evidence, Pattern bridges
- MemoryCoordinator - atomic cross-system writes via `enable_coordinated_writes`
- SelectionFeedbackLoop - performance-based agent selection via `enable_performance_feedback`
- CrossDebateMemory - institutional knowledge injection via `enable_cross_debate_memory`
- Post-debate workflows - automated processing via `enable_post_debate_workflow`
- Chat connectors - Telegram, WhatsApp integration for debate interfaces
- Leader election - distributed coordination via `aragora.control_plane.leader`
- Streaming connectors - Kafka and RabbitMQ for enterprise event ingestion
- Bidirectional chat routing - `debate_origin.py` routes results to originating platform
- Adapter factory - auto-create KM adapters from Arena subsystems
- TTS integration - voice synthesis for debates and chat channels
- Decision Explainability - natural language explanations, factor decomposition, counterfactuals
- Workflow Templates - 15+ pre-built templates across 6 categories, pattern factories
- Gauntlet Receipts - cryptographic audit trails with SHA-256 hashing
- Gauntlet Defense - proposer_agent param enables attack/defend cycles
- KM Resilience - ResilientPostgresStore with retry, health monitoring, cache invalidation

**Enterprise (production-ready):**
- Authentication - OIDC/SAML SSO, MFA (TOTP/HOTP), API key management
- Multi-Tenancy - Tenant isolation, resource quotas, usage metering
- Security - AES-256-GCM encryption, rate limiting, circuit breakers
- Compliance - SOC 2 controls, GDPR support, audit trails
- Observability - Prometheus metrics, Grafana dashboards, OpenTelemetry tracing
- RBAC v2 - Fine-grained permissions (50+), role hierarchy, decorators, middleware
- Backup/DR - Incremental backups, retention policies, disaster recovery drills
- Control Plane - Agent registry, task scheduler, health monitoring, policy governance (170+ tests)
  - PolicyConflictDetector - Detects contradictory policies before they cause issues
  - RedisPolicyCache - Distributed cache for fast policy evaluation
  - PolicySyncScheduler - Continuous background policy synchronization
  - Omnichannel notifications - Debate → Slack/Teams/Email/Webhook delivery
  - ReceiptAdapter - Decision receipts auto-persist to Knowledge Mound

**Integrated:**
- Knowledge Mound - STABLE Phase A2 (100% integrated, 950+ tests passing)
  - 14 adapters (Continuum, Consensus, Critique, Evidence, Pulse, Insights, ELO, Belief, Cost, Receipt, ControlPlane, RLM, Culture, Ranking)
  - Visibility, sharing, federation, global knowledge
  - Semantic search, validation feedback, cross-debate learning
  - SLO alerting with Prometheus metrics
  - Phase A2: Contradiction detection, confidence decay, RBAC governance, analytics, knowledge extraction
- Pulse (trending topics) - STABLE (358+ tests passing)
  - HackerNews, Reddit, Twitter ingestors
  - Quality filtering, freshness scoring, source weighting
  - Integration with debate context and prompt building
- Evidence collection - STABLE with KM integration

See `docs/STATUS.md` for 74+ detailed feature statuses.

## Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/COMMERCIAL_OVERVIEW.md` | Commercial positioning, 85% readiness assessment |
| `docs/ENTERPRISE_FEATURES.md` | Enterprise capabilities reference |
| `docs/FEATURE_DISCOVERY.md` | Complete feature catalog (100+ features) |
| `docs/STATUS.md` | Detailed feature implementation status |
| `docs/API_REFERENCE.md` | REST API documentation |
