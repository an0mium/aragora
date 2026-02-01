# Feature Discovery Guide

Complete catalog of Aragora features organized by domain. Use this guide to discover capabilities and understand the codebase structure.

*Last updated: February 2026*

---

## Quick Navigation

| Domain | Features | Primary Location |
|--------|----------|------------------|
| [Debate Engine](#debate-engine) | 15+ | `aragora/debate/` |
| [Agent System](#agent-system) | 20+ | `aragora/agents/` |
| [Knowledge Management](#knowledge-management) | 12+ | `aragora/knowledge/` |
| [Memory System](#memory-system) | 8+ | `aragora/memory/` |
| [Workflow Engine](#workflow-engine) | 10+ | `aragora/workflow/` |
| [Enterprise Features](#enterprise-features) | 25+ | Various |
| [Integrations](#integrations) | 30+ | `aragora/connectors/` |
| [API & SDK](#api--sdk) | 200+ endpoints | `aragora/server/` |

---

## Debate Engine

Core multi-agent debate orchestration for robust decision-making.

### Core Features

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| Arena | Stable | `debate/orchestrator.py` | Main debate engine with configurable protocols |
| Consensus Detection | Stable | `debate/consensus.py` | Multi-strategy consensus (majority, unanimous, weighted) |
| Convergence | Stable | `debate/convergence.py` | Semantic similarity detection for argument convergence |
| Team Selection | Stable | `debate/team_selector.py` | ELO-based agent selection with calibration |
| Prompt Builder | Stable | `debate/prompt_builder.py` | Dynamic prompt construction |

### Advanced Features

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| Graph Debates | Stable | `debate/graph/` | Graph-structured argument mapping |
| Matrix Debates | Stable | `debate/matrix/` | Multi-dimensional debate analysis |
| Hybrid Debates | Stable | `handlers/hybrid_debate_handler.py` | External + internal agent debates |
| Debate Breakpoints | Stable | `debate/breakpoints.py` | Pause/resume debate execution |
| Rhetorical Observer | Stable | `debate/rhetorical_observer.py` | Argument quality analysis |
| Trickster | Stable | `debate/trickster.py` | Hollow consensus detection |

### Configuration

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| ArenaConfig | Stable | `debate/arena_config.py` | Centralized debate configuration |
| DebateProtocol | Stable | `core.py` | Protocol parameters (rounds, consensus) |
| Orchestrator Hooks | Stable | `debate/orchestrator_hooks.py` | Extension points for custom logic |

---

## Agent System

Multi-model agent orchestration supporting 15+ AI providers.

### Supported Providers

| Provider | Type | Location | Models |
|----------|------|----------|--------|
| Anthropic | API | `agents/api_agents/anthropic.py` | Claude 3.5, Claude Opus 4.5 |
| OpenAI | API | `agents/api_agents/openai.py` | GPT-4o, GPT-4 Turbo, o1 |
| Google | API | `agents/api_agents/gemini.py` | Gemini Pro, Ultra |
| Mistral | API | `agents/api_agents/mistral.py` | Mistral Large, Codestral |
| xAI | API | `agents/api_agents/grok.py` | Grok |
| OpenRouter | API | `agents/api_agents/openrouter.py` | DeepSeek, Llama, Qwen, Yi |
| Ollama | Local | `agents/api_agents/ollama.py` | Any GGUF model |
| LM Studio | Local | `agents/api_agents/lm_studio.py` | Any local model |

### Agent Features

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| AirlockProxy | Stable | `agents/airlock.py` | Agent resilience with circuit breaker |
| Fallback Chain | Stable | `agents/fallback.py` | Automatic OpenRouter fallback |
| Rate Limiter | Stable | `agents/api_agents/rate_limiter.py` | Per-provider rate limiting |
| Calibration | Stable | `agents/calibration.py` | Agent performance calibration |
| Personas | Stable | `agents/personas.py` | Configurable agent personalities |
| ELO Rankings | Stable | `ranking/elo.py` | Agent skill tracking |

### External Agents

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| External Framework | Stable | `agents/external/` | CrewAI, LangGraph, AutoGen integration |
| A2A Protocol | Stable | `handlers/a2a.py` | Agent-to-Agent communication |
| External Agents API | Stable | `handlers/external_agents.py` | External agent task management |

---

## Knowledge Management

Unified knowledge layer with Knowledge Mound at the core.

### Knowledge Mound

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| Knowledge Mound Core | Stable | `knowledge/mound/` | Central knowledge repository |
| Semantic Search | Stable | `knowledge/mound/adapters/performance/search.py` | Vector-based knowledge retrieval |
| Contradiction Detection | Stable | `knowledge/mound/contradictions.py` | Identify conflicting knowledge |
| Confidence Decay | Stable | `knowledge/mound/confidence.py` | Time-based confidence scoring |
| Federation | Stable | `knowledge/mound/federation.py` | Cross-instance knowledge sharing |

### Adapters (14 Total)

| Adapter | Location | Purpose |
|---------|----------|---------|
| Continuum | `adapters/continuum_adapter.py` | Memory tier integration |
| Consensus | `adapters/consensus_adapter.py` | Debate outcome storage |
| Critique | `adapters/critique_adapter.py` | Critique pattern storage |
| Evidence | `adapters/evidence_adapter.py` | Evidence management |
| Pulse | `adapters/pulse_adapter.py` | Trending topics |
| Insights | `adapters/insights_adapter.py` | Analytical insights |
| ELO | `adapters/elo_adapter.py` | Agent rankings |
| Belief | `adapters/belief_adapter.py` | Belief network integration |
| Cost | `adapters/cost_adapter.py` | Cost tracking |
| Receipt | `adapters/receipt_adapter.py` | Decision receipts |
| ControlPlane | `adapters/control_plane_adapter.py` | Control plane data |
| RLM | `adapters/rlm_adapter.py` | RLM context |
| Culture | `adapters/culture_adapter.py` | Organizational culture |
| Ranking | `adapters/ranking_adapter.py` | Performance rankings |

### Knowledge Bridges

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| KnowledgeBridgeHub | Stable | `knowledge/bridges.py` | Unified bridge access |
| MetaLearner | Stable | `knowledge/bridges.py` | Cross-debate learning |
| Evidence Bridge | Stable | `knowledge/bridges.py` | Evidence integration |
| Pattern Bridge | Stable | `knowledge/bridges.py` | Pattern recognition |

---

## Memory System

Multi-tier memory for context preservation and learning.

### Memory Tiers

| Tier | TTL | Location | Purpose |
|------|-----|----------|---------|
| Fast | 1 min | `memory/continuum.py` | Immediate context |
| Medium | 1 hour | `memory/continuum.py` | Session memory |
| Slow | 1 day | `memory/continuum.py` | Cross-session learning |
| Glacial | 1 week | `memory/continuum.py` | Long-term patterns |

### Memory Features

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| ContinuumMemory | Stable | `memory/continuum.py` | Multi-tier memory manager |
| ConsensusMemory | Stable | `memory/consensus.py` | Historical debate outcomes |
| CritiqueStore | Stable | `memory/store.py` | Critique pattern storage |
| MemoryCoordinator | Stable | `memory/coordinator.py` | Atomic cross-system writes |
| CrossDebateMemory | Stable | `memory/cross_debate.py` | Institutional knowledge injection |

---

## Workflow Engine

DAG-based automation with 15+ pre-built templates.

### Core Components

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| Workflow Engine | Stable | `workflow/engine.py` | DAG execution engine |
| Pattern Factory | Stable | `workflow/patterns/` | Pre-built workflow patterns |
| Template Registry | Stable | `workflow/templates/` | Template management |
| Node Library | Stable | `workflow/nodes/` | Reusable workflow nodes |

### Template Categories

| Category | Templates | Description |
|----------|-----------|-------------|
| Knowledge | 3 | Knowledge extraction, validation, pruning |
| Decision | 2 | Decision-making workflows |
| Compliance | 2 | Audit and compliance checks |
| Onboarding | 2 | User/team onboarding |
| Integration | 3 | Data sync and integration |
| Notification | 3 | Alert and notification routing |

---

## Enterprise Features

Production-ready enterprise capabilities.

### Authentication & Authorization

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| OIDC/SAML SSO | Stable | `auth/oidc.py` | Single sign-on |
| MFA (TOTP/HOTP) | Stable | `auth/mfa.py` | Multi-factor authentication |
| API Key Management | Stable | `auth/api_keys.py` | API key lifecycle |
| RBAC v2 | Stable | `rbac/` | 50+ permissions, role hierarchy |
| SCIM Provisioning | Stable | `handlers/scim_handler.py` | User provisioning |

### Multi-Tenancy

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| Tenant Isolation | Stable | `tenancy/isolation.py` | Data isolation |
| Resource Quotas | Stable | `tenancy/quotas.py` | Usage limits |
| Usage Metering | Stable | `tenancy/metering.py` | Consumption tracking |

### Security

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| AES-256-GCM Encryption | Stable | `security/encryption.py` | Data encryption |
| Rate Limiting | Stable | `server/middleware/rate_limit.py` | Request throttling |
| Circuit Breakers | Stable | `resilience.py` | Failure isolation |
| Security Headers | Stable | `server/middleware/security_headers.py` | CSP, HSTS, X-Frame-Options |

### Compliance

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| SOC 2 Controls | Stable | `compliance/soc2.py` | Compliance controls |
| GDPR Support | Stable | `privacy/gdpr.py` | Data privacy |
| Audit Trails | Stable | `audit/` | Comprehensive logging |
| Gauntlet Receipts | Stable | `gauntlet/receipts.py` | Cryptographic audit trails |

### Observability

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| Prometheus Metrics | Stable | `server/metrics.py` | Metrics export |
| OpenTelemetry Tracing | Stable | `observability/tracing.py` | Distributed tracing |
| SLO Monitoring | Stable | `handlers/slo.py` | Service level objectives |

### Backup & DR

| Feature | Status | Location | Description |
|---------|--------|----------|-------------|
| BackupManager | Stable | `backup/manager.py` | Incremental backups |
| Disaster Recovery | Stable | `handlers/dr_handler.py` | DR drills and recovery |

---

## Integrations

### Communication Platforms

| Platform | Status | Location | Features |
|----------|--------|----------|----------|
| Slack | Stable | `handlers/social/slack.py` | Bot, OAuth, webhooks |
| Microsoft Teams | Stable | `handlers/bots/teams.py` | Bot, OAuth, cards |
| Discord | Stable | `handlers/bots/discord.py` | Bot, OAuth, slash commands |
| Telegram | Stable | `handlers/bots/telegram.py` | Bot, webhooks |
| WhatsApp | Stable | `handlers/bots/whatsapp.py` | Business API |
| Zoom | Stable | `handlers/bots/zoom.py` | Meetings, webhooks |
| Google Chat | Stable | `handlers/google_chat.py` | Bot, cards |

### Email Providers

| Provider | Status | Location | Features |
|----------|--------|----------|----------|
| Gmail | Stable | `handlers/features/gmail*.py` | OAuth, labels, threads |
| Outlook | Stable | `handlers/features/outlook.py` | OAuth, folders, search |
| Generic SMTP | Stable | `connectors/email/` | Send/receive |

### Enterprise Systems

| System | Status | Location | Features |
|--------|--------|----------|----------|
| Salesforce | Stable | `connectors/enterprise/salesforce.py` | CRM sync |
| HubSpot | Stable | `connectors/enterprise/hubspot.py` | CRM sync |
| Zendesk | Stable | `connectors/enterprise/zendesk.py` | Support tickets |
| Jira | Stable | `connectors/enterprise/jira.py` | Issue tracking |
| GitHub | Stable | `connectors/github.py` | PR reviews, webhooks |

### Data Streaming

| System | Status | Location | Features |
|--------|--------|----------|----------|
| Kafka | Stable | `connectors/enterprise/streaming/kafka.py` | Event ingestion |
| RabbitMQ | Stable | `connectors/enterprise/streaming/rabbitmq.py` | Message queue |
| CDC | Stable | `connectors/enterprise/database/cdc.py` | Change data capture |

---

## API & SDK

### REST API

- **Endpoints**: 461 API endpoints
- **OpenAPI Spec**: `docs/api/openapi.yaml`
- **Reference**: `docs/API_REFERENCE.md`
- **Stability Levels**: `docs/API_STABILITY.md`

### WebSocket Streaming

- **Streams**: 22 WebSocket streams
- **Events**: `debate_start`, `round_start`, `agent_message`, `critique`, `vote`, `consensus`, `debate_end`
- **Location**: `server/stream/`

### SDKs

| SDK | Location | Status |
|-----|----------|--------|
| Python | `sdk/python/` | Stable |
| TypeScript | `sdk/typescript/` | Stable |

---

## Self-Improvement (Nomic Loop)

Autonomous improvement cycle for codebase evolution.

| Phase | Purpose | Location |
|-------|---------|----------|
| Context | Gather codebase understanding | `scripts/nomic_loop.py` |
| Debate | Agents propose improvements | `nomic/meta_planner.py` |
| Design | Architecture planning | `nomic/task_decomposer.py` |
| Implement | Code generation | `nomic/branch_coordinator.py` |
| Verify | Tests and checks | `nomic/autonomous_orchestrator.py` |

---

## Feature Discovery Tips

### Finding Features by Use Case

1. **"I need to run a multi-agent debate"** → Start with `aragora/debate/orchestrator.py`
2. **"I want to integrate with Slack"** → Check `aragora/server/handlers/social/slack.py`
3. **"I need to add authentication"** → Look at `aragora/auth/` and `aragora/rbac/`
4. **"I want to store knowledge"** → Use `aragora/knowledge/mound/`
5. **"I need workflow automation"** → See `aragora/workflow/engine.py`

### Searching the Codebase

```bash
# Find all handlers
grep -r "class.*Handler" aragora/server/handlers/ --include="*.py"

# Find all adapters
grep -r "class.*Adapter" aragora/ --include="*.py"

# Find all API endpoints
grep -r "@api_endpoint" aragora/ --include="*.py"

# Find all permissions
grep -r "require_permission" aragora/ --include="*.py"
```

### Key Entry Points

| Purpose | Entry Point |
|---------|-------------|
| Start server | `python -m aragora.server.unified_server` |
| Run debate | `from aragora import Arena, Environment` |
| Access knowledge | `from aragora.knowledge.mound import KnowledgeMound` |
| Run workflow | `from aragora.workflow.engine import WorkflowEngine` |

---

## See Also

- [CLAUDE.md](../CLAUDE.md) - Integration guide
- [STATUS.md](STATUS.md) - Release status
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [ENTERPRISE_FEATURES.md](ENTERPRISE_FEATURES.md) - Enterprise capabilities
- [HANDLER_PATTERNS.md](HANDLER_PATTERNS.md) - Handler implementation patterns
