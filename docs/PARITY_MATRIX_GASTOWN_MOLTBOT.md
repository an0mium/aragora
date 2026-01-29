# Gastown + Moltbot Parity Matrix (Extension Targets)

Status legend:
- **present**: capability exists in Aragora core as a first-class feature
- **partial**: related capability exists but lacks specific workflow/UX/semantics
- **missing**: no clear equivalent in Aragora core today

This matrix treats Gastown and Moltbot parity as extension layers on top of the
Aragora enterprise decision control plane.

## Core invariants (Aragora must remain)
- Enterprise decision control plane for multi-agent vetted decisionmaking
- Evidence-based outputs with defensible receipts and audit trails
- Multi-channel delivery (Slack, Teams, Discord, Telegram, WhatsApp, voice)
- RBAC, compliance, multi-tenancy, and governance

---

## Gastown Parity (Developer Orchestration Extension)

Gastown is a multi-agent workspace manager for Claude Code. Its core insight is
that **all work state lives in git** -- sessions are ephemeral, work persists.
It uses a Kubernetes-inspired model: Mayor=scheduler, Rig=node, Polecat=pod.

Reference: https://github.com/steveyegge/gastown

### Construct-Level Mapping

| Gastown Construct | Purpose | Aragora Equivalent | Status | Gap Summary |
|---|---|---|---|---|
| **Town** | Root workspace directory managing all rigs/agents | `control_plane/coordinator.py` | partial | CP coordinates agents but has no workspace directory concept |
| **Mayor** | Chief-of-staff AI coordinator; decomposes goals, distributes work | `debate/orchestrator.py` (Arena), `nomic/autonomous_orchestrator.py` | partial | Arena orchestrates debates; needs goal decomposition for SE tasks |
| **Deacon** | Background daemon with exponential-backoff patrol cycles | `control_plane/health.py` + `resilience.py` circuit breakers | partial | Health monitoring exists; lacks continuous patrol with backoff |
| **Dogs / Boot** | Maintenance agents + watchdog-of-the-watchdog | `control_plane/watchdog.py` (ThreeTierWatchdog) | present | Three-tier watchdog covers mechanical + boot + deacon tiers |
| **Rigs** | Per-repo project containers with isolated agent contexts | Debate `Environment` + `TenantContext` | partial | Tenancy isolation exists; no per-repo container abstraction |
| **Crew** | Long-lived named agents with persistent context | `agents/cli_agents.py` (claude, codex, gemini, grok) with ELO history | partial | Named agents exist with ratings; lack cross-session context persistence |
| **Polecats** | Ephemeral worker agents (12-30 concurrent) in git worktrees | `debate/agent_pool.py` + Arena orchestration | partial | Concurrent agents exist; lack isolated filesystem per agent |
| **Witness** | Per-rig agent monitoring Polecats and Refinery | `debate/witness.py` (DebateWitness) | present | DebateWitness monitors per-debate progress and stalls |
| **Refinery** | Per-rig merge queue manager with backpressure | `debate/consensus.py` (ConsensusProof) | partial | Consensus integrates outputs; no git merge queue concept |
| **Hooks** | Pinned Bead per agent = persistent work queue in git worktree | No equivalent | **missing** | Core gap: Aragora has no git-worktree-backed per-agent task persistence |
| **Beads** | Git-backed atomic work units (JSONL, prefix+5-char IDs) | `nomic/task_decomposer.py` tasks, Gauntlet receipts | partial | Work items exist but not git-backed JSONL with atomic tracking |
| **Convoys** | Bundled work orders tracking multiple beads as peers | Nomic Loop cycles, debate session groups | partial | Grouping exists conceptually; no first-class convoy tracking |
| **Wisps** | Ephemeral beads (not persisted to git) | `memory/continuum.py` fast tier | present | Fast-tier memory is transient by design |
| **Molecules / Formulas** | Multi-step workflow templates (TOML-based) | `workflow/engine.py` + `workflow/patterns/` | present | Workflow engine + pattern library exists |
| **Nudge** | Real-time inter-agent messaging | `debate/protocol_messages/` | partial | Protocol messages exist within debates; no cross-debate agent mail |
| **Seance** | Query previous sessions for decisions/context | `memory/continuum.py` (cross-session), `knowledge/mound/` | present | Multi-tier memory + KnowledgeMound provide historical context |
| **Dashboard** | Web UI for convoy/agent/hook status | Debate streaming WebSocket events | partial | Real-time debate streaming; no workspace management dashboard |
| **`gt` CLI** | Unified CLI for workspace operations | `scripts/self_develop.py`, `scripts/nomic_loop.py` | partial | CLI tools exist but not Gastown-style workspace commands |

### Key Design Principles to Implement

| Gastown Principle | Description | Aragora Status |
|---|---|---|
| **GUPP** | "If there is work on your Hook, YOU MUST RUN IT" -- auto-resume | **missing** |
| **MEOW** | Molecular Expression of Work -- break goals into trackable atoms | partial (task decomposer) |
| **NDI** | Nondeterministic Idempotence -- useful outcomes from unreliable processes | partial (retry/circuit breaker) |
| **Sessions are cattle** | Agent sessions disposable; work state persists in git | **missing** (sessions carry state) |
| **Git is the database** | All state in git = free versioning/rollback/distribution | **missing** |

### Critical Gaps (Ordered by Priority)

1. **Hook persistence** -- Git worktree-backed per-agent work queues that survive crashes
2. **GUPP auto-resume** -- Agents automatically resuming pending work without prompting
3. **Rig abstraction** -- Per-repo project containers with isolated agent pools
4. **Refinery merge queue** -- Intelligent merge gating with backpressure awareness
5. **Inter-agent messaging** -- Direct nudge/mail between agents outside debates
6. **Convoy tracking** -- First-class work batch lifecycle management

---

## Moltbot Parity (Consumer/Device Extension)

Moltbot is a local-first personal AI assistant that runs on user devices with
multi-channel inbox, voice, and live canvas capabilities.

Reference: https://github.com/moltbot/moltbot

### Feature-Level Mapping

| Moltbot Feature | Purpose | Aragora Equivalent | Status | Gap Summary |
|---|---|---|---|---|
| **Local-first gateway** | Device-local routing + auth control plane | `server/unified_server.py` (server-centric) | **missing** | Aragora is server/cloud-oriented; needs local daemon |
| **Multi-channel inbox** | Unified inbox across WhatsApp/Telegram/Slack/Discord/Signal/iMessage/Teams/etc | `connectors/chat/` | partial | Channels exist; no unified consumer "inbox" aggregation |
| **Multi-agent routing** | Per-channel/account agent assignment | `control_plane/scheduler.py` task routing | partial | Agent selection exists; not channel-specific routing UX |
| **Voice wake + talk** | On-device always-on speech I/O | `server/stream/tts_integration.py`, `server/stream/voice_stream.py` | partial | TTS + voice streaming exist; no on-device wake word |
| **Live Canvas (A2UI)** | Real-time interactive visual canvas | `canvas/manager.py` + `server/stream/canvas_stream.py` | partial | Canvas infra exists; consumer-grade A2UI not yet exposed |
| **Onboarding wizard** | Guided first-run setup experience | No equivalent | **missing** | No guided onboarding flow |
| **Device nodes** | Companion apps / device capabilities registry | No equivalent | **missing** | No device-node model |
| **Skill marketplace** | User-installable skills/plugins | `aragora/plugins/` | partial | Plugin system exists; no consumer marketplace UX |
| **Security pairing** | Allowlist / DM pairing defaults | `aragora/rbac/`, `auth/` | partial | Enterprise RBAC exists; consumer pairing UX missing |
| **Computer use** | Browser/shell/screen interaction | `sandbox/` | partial | Sandbox exists but not a full browser/screen automation surface |

### Critical Gaps (Ordered by Priority)

1. **Local gateway daemon** -- Device-local service for routing and auth
2. **Live Canvas / A2UI** -- Real-time visual collaboration surface
3. **Onboarding wizard** -- Guided setup for first-time users
4. **Device node model** -- Registry of device capabilities and permissions
5. **Voice wake word** -- On-device always-listening trigger
6. **Computer use sandbox** -- Policy-gated browser/shell/screen actions

---

## Cross-Cutting Extension Requirements

| Capability | Purpose | Aragora Module | Status | Notes |
|---|---|---|---|---|
| **Agent Fabric** | High-scale scheduling + isolation for 50+ concurrent agents | `control_plane/scheduler.py`, `control_plane/registry.py` | partial | Scheduler + registry exist; needs workspace-scoped pools and git worktree isolation |
| **Policy engine** | Tool access, approvals, sandboxing | `rbac/`, `control_plane/policy.py` | partial | RBAC + policy exist; need device-level policy and approval gates |
| **Audit + replay** | Every device/agent action logged | `gauntlet/receipt.py`, audit logging | present | Decision receipts exist; extend to device actions |
| **Cost + budget controls** | Prevent runaway usage at workspace level | `control_plane/cost_enforcement.py` | partial | Per-task cost checks exist; need cumulative workspace budgets |
| **Safe computer use** | UI/browser/shell with approvals | `sandbox/` | partial | Sandbox exists but needs browser/screen actions and policy gating |
| **Workspace quotas** | Resource limits per project/tenant | `control_plane/multi_tenancy.py` | partial | Tenant isolation exists; no per-workspace quota management |

---

## Existing Aragora Infrastructure (Reusable for Extensions)

These modules provide the foundation for both parity extensions:

| Module | Location | Relevance |
|---|---|---|
| TaskScheduler | `control_plane/scheduler.py` | Task dispatch, throttling, and routing |
| AgentRegistry | `control_plane/registry.py` | Agent liveness and capability tracking |
| HealthMonitor | `control_plane/health.py` | Periodic probes, circuit breakers, cascading failure detection |
| CircuitBreaker | `resilience.py` | Global registry, provider-based config, auto-pruning |
| TenantContext | `control_plane/multi_tenancy.py` | ContextVar-based workspace scoping |
| CostEnforcer | `control_plane/cost_enforcement.py` | Pre-submission checks, throttle levels, priority adjustment |
| AirlockProxy | `agents/airlock.py` | Per-operation timeouts, response sanitization, fallback |
| AgentTelemetry | `agents/telemetry.py` | Token usage, duration, success/failure tracking |
| ThreeTierWatchdog | `control_plane/watchdog.py` | Mechanical/Boot/Deacon monitoring tiers |
| Coordinator | `control_plane/coordinator.py` | Unified agent/task operations and orchestration glue |
| Workflow Engine | `workflow/engine.py` | DAG-based automation with reusable patterns |
| Memory Continuum | `memory/continuum.py` | Multi-tier memory and retention policies |
| Knowledge Mound | `knowledge/mound/` | Knowledge storage, retrieval, and evidence linking |

---

## Build vs Integrate Recommendation

| Extension | Recommendation | Rationale |
|---|---|---|
| **Hook persistence** | Build | Core to Aragora's value; git worktree management is infrastructure-level |
| **Rig/Workspace abstraction** | Build | Extends existing TenantContext and Coordinator |
| **Convoy tracking** | Build | Extends existing workflow engine and task decomposer |
| **Mayor-style coordinator** | Build | Extends existing Arena and AutonomousOrchestrator |
| **Refinery merge queue** | Build | Specialized; extends existing consensus patterns |
| **Local gateway** | Build or Fork | May warrant separate service; evaluate Moltbot gateway as starting point |
| **Live Canvas** | Integrate | Complex UI; evaluate existing open-source canvas solutions |
| **Voice wake** | Integrate | Specialized hardware/ML; use existing speech frameworks |
| **Onboarding wizard** | Build | Product-specific; must match Aragora's identity |
| **Computer use sandbox** | Build | Security-critical; must be policy-integrated from inception |

---

## Implementation Priority Matrix

### P0: Foundation (Weeks 1-4) - Must build first

| Component | Module | Builds On | Deliverable |
|---|---|---|---|
| Agent Pool Manager | `aragora/fabric/pool.py` | AgentRegistry | Per-pool quotas, affinity rules, lifecycle |
| Per-Agent Queues | `aragora/fabric/queue.py` | TaskScheduler | Semaphore-based concurrency, backpressure |
| Resource Limits Enforcer | `aragora/fabric/limits.py` | CostEnforcer | Per-agent tokens, requests, execution time |
| Execution Wrapper | `aragora/fabric/executor.py` | AirlockProxy | Process isolation, timeout enforcement |

### P1: Extension Requirements (Weeks 5-8) - Enables parity features

| Component | Module | Builds On | Deliverable |
|---|---|---|---|
| Hook Persistence | `aragora/fabric/hooks.py` | Git worktree | Per-agent git-backed task queues (GUPP) |
| Workspace Manager | `aragora/workspace/manager.py` | TenantContext | Rig abstraction with isolated agent pools |
| Convoy Tracker | `aragora/workspace/convoy.py` | Workflow engine | Work batch lifecycle, artifact tracking |
| Local Gateway | `aragora/gateway/server.py` | unified_server | Device-local routing and auth daemon |

### P2: Advanced Capabilities (Weeks 9-12) - Full parity

| Component | Module | Builds On | Deliverable |
|---|---|---|---|
| Computer Use Sandbox | `aragora/sandbox/executor.py` | Policy engine | Browser/shell with approval gates |
| Live Canvas | `aragora/canvas/` | WebSocket streaming | Real-time visual collaboration |
| Onboarding Wizard | `aragora/onboarding/` | New | Guided setup CLI/UI flow |
| Voice Wake | `aragora/voice/wake.py` | voice_stream | On-device always-listening trigger |

---

## Critical Gaps for 50+ Agent Scale (from codebase analysis)

1. **Per-agent resource limits**: No concurrent task limits, token rate limiting, or execution budgets per agent
2. **Agent pool management**: No grouping, affinity rules, or dedicated vs shared pools
3. **Execution isolation**: Logical only (TenantContext), no process/container isolation
4. **Concurrency control**: No per-agent semaphores or backpressure mechanisms
5. **Agent lifecycle**: No provisioning API, versioning, warm-up, or graceful shutdown
6. **Fine-grained cost tracking**: CostTracker is per-workspace, not per-agent
