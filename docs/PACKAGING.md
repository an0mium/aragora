# Aragora Packaging Strategy

> **Thesis:** Aragora is a verifiable AI operations platform with three pillars:
> adversarial decisioning, policy-gated execution, and provenance attestation.
> Ship each pillar as a certified module, not a monolith.

---

## Tiered Module Architecture

### Tier 1: Core Primitives (always installed)

The minimal trust stack. Every Aragora deployment includes these.

| Module | Purpose | LOC | Tests |
|--------|---------|-----|-------|
| `aragora/debate/` | Adversarial multi-model debate orchestration | ~104K | 13,500+ |
| `aragora/gauntlet/` | Red-team stress testing and decision receipts | ~8K | 1,000+ |
| `aragora/ranking/` | ELO ratings, calibration, agent trust scoring | ~3K | 500+ |
| `aragora/knowledge/mound/` | Institutional memory with 45 adapters | ~15K | 4,300+ |
| `aragora/core_types.py` | Shared type definitions | ~750 | — |

**Install:** `pip install aragora`

### Tier 2: Gateway Pack (OpenClaw)

Policy-gated autonomous execution with human approvals and audit trails.

| Module | Purpose | LOC | Tests |
|--------|---------|-----|-------|
| `aragora/compat/openclaw/` | OpenClaw compatibility layer | ~2K | 100+ |
| `aragora/server/handlers/openclaw/` | Gateway HTTP handlers | ~4K | 200+ |
| `aragora/compat/openclaw/standalone.py` | Standalone deployable gateway | ~500 | 14 |

**Install:** `pip install aragora[gateway]`

**Value prop:** "Policy-first autonomous execution gateway with human approvals
and auditable controls — the missing middleware between your AI agents and
production systems."

### Tier 3: Blockchain Pack (ERC-8004)

Cross-organization trust portability via on-chain agent identity and reputation.

| Module | Purpose | LOC | Tests |
|--------|---------|-----|-------|
| `aragora/blockchain/` | ERC-8004 contract wrappers, Web3 provider | ~2,400 | 182 |
| `aragora/connectors/blockchain/` | Evidence source from on-chain data | ~2,000 | 127 |
| `aragora/knowledge/mound/adapters/erc8004_adapter.py` | Bidirectional KM sync | ~1,000 | 48 |

**Install:** `pip install aragora[blockchain]` (requires `web3`)

**Value prop:** "Portable agent reputation across organizations — when your
agent has a verified track record on-chain, new partners trust it without
re-evaluation."

### Tier 4: Enterprise Pack

Production security, compliance, and multi-tenancy.

| Module | Purpose |
|--------|---------|
| `aragora/auth/` | OIDC/SAML SSO, MFA, API key management |
| `aragora/rbac/` | Fine-grained permissions (360+), role hierarchy |
| `aragora/tenancy/` | Tenant isolation, resource quotas |
| `aragora/compliance/` | SOC 2 controls, GDPR, audit trails |
| `aragora/security/` | AES-256-GCM encryption, key rotation |
| `aragora/backup/` | Disaster recovery, incremental backups |

**Install:** `pip install aragora[enterprise]`

### Tier 5: Connector Pack

Integrate with external platforms and services.

| Module | Purpose |
|--------|---------|
| `aragora/connectors/chat/` | Telegram, WhatsApp, Slack, Discord, Teams, Signal |
| `aragora/connectors/enterprise/` | OneDrive, Google Drive, SharePoint, Outlook |
| `aragora/connectors/enterprise/streaming/` | Kafka, RabbitMQ event ingestion |
| `aragora/integrations/` | Slack, Email, Discord, Teams, Zapier, LangChain |

**Install:** `pip install aragora[connectors]`

### Tier 6: Experimental

Alpha-quality capabilities. APIs may change.

| Module | Purpose |
|--------|---------|
| `aragora/genesis/` | Fractal resolution, agent evolution |
| `aragora/visualization/` | Argument cartography, logic mapping |
| `aragora/sandbox/` | Docker-based safe code execution |
| `aragora/introspection/` | Agent self-awareness, meta-cognition |
| `aragora/computer_use/` | Browser and desktop automation |

**Install:** `pip install aragora[experimental]`

---

## Standalone Packages

These can be installed independently, without the main Aragora package.

### `aragora-debate`

Minimal adversarial debate engine. Zero dependencies.

```bash
pip install aragora-debate
```

| File | Purpose |
|------|---------|
| `types.py` | Core types (Agent, Message, Critique, Vote, DebateConfig, etc.) |
| `arena.py` | Debate orchestrator (propose → critique → vote) |
| `debate.py` | High-level 5-line Debate API + `create_agent` factory |
| `receipt.py` | Decision receipts with HMAC-SHA256 signing |
| `agents.py` | Provider agents (Claude, OpenAI, Mistral, Gemini) |
| `evidence.py` | Evidence quality scoring + hollow consensus detection |
| `convergence.py` | Convergence tracking across rounds (stdlib only) |
| `trickster.py` | Evidence-powered challenge injection |
| `cross_analysis.py` | Cross-proposal evidence validation |
| `events.py` | Event/callback system for real-time monitoring |
| `_mock.py` / `styled_mock.py` | Mock agents for testing and demos |

**When to use:** You want adversarial debate without the full platform.
Embed in existing AI pipelines, CrewAI workflows, or LangGraph chains.

---

## Future Extractions

Modules currently developed inside Aragora that are explicitly designed for future extraction as standalone OSS substrates with permissive licensing.

### `agent-bridge` *(in-flight, extraction target)*

State-preserving heterogeneous CLI harness interop: a small operator-local broker that lets independent agent harnesses (Claude Code, Codex CLI, Factory Droid, Aider, Gemini CLI, Cursor Agent, Goose, Cline, …) hold a structured ongoing dialog and cross-check each other while each retains its own native session state, tools, and context. Lives at `aragora/swarm/agent_bridge/`.

The intersection that existing frameworks don't cover — **heterogeneous + state-preserving + dialog/cross-check + permissive license** — is the unmet market need this targets. MCP shares tools but not turns; A2A assumes HTTP-service agents with cards, not CLI harnesses; AutoGen / LangGraph / CrewAI / OpenHands all absorb the agent into their own runtime. The agent-ops plane lets *real* harnesses converse while preserving each one's authentic behavior.

**Designed-in extraction-readiness disciplines** (so the eventual `pip install agent-bridge` is zero-rewrite):

1. Package boundary discipline: `aragora/swarm/agent_bridge/` is import-clean of Aragora-specific code (no `debate/`, no `review/`, no `thesis/`)
2. Footer Protocol as a versioned wire spec (semver, LSP-style)
3. Adopt-where-useful, depend-where-stable: speak A2A envelope shape on the wire; expose MCP-style tool sharing as an optional transport
4. Pluggable transports: subprocess (current), HTTP, WebSocket, future gRPC
5. Reference adapters for ≥4 harnesses
6. License: Apache-2.0 (patent-grant clarity in a multi-vendor space)

**Status.** Live-smoke transport layer passing; write API in PR (#6629); positioning + market-fit + sequencing in [`docs/plans/2026-04-25-agent-ops-plane-market-fit.md`](plans/2026-04-25-agent-ops-plane-market-fit.md). Extraction itself is post-Aragora-validation; until then it ships as part of the core install.

---

## SDK Coverage

| Namespace | Python SDK | TypeScript SDK | Status |
|-----------|-----------|---------------|--------|
| openclaw | 22/22 endpoints | 22/22 endpoints | Parity |
| blockchain | 6/7 endpoints | 6/7 endpoints | Parity (2 server stubs) |
| debates | Full | Full | Parity |
| auth | Full | Full | Parity |
| gateway | Full | Full | Parity |
| audit | Full | Full | Parity |

Parity is enforced by `tests/sdk/test_contract_parity.py`.

---

## Go-to-Market Positioning

### Primary: "Verifiable AI Operations Platform"

> End-to-end Decision → Action → Audit → Attestation chain in one self-hosted stack.

**Buyer:** Engineering leaders at regulated companies (finance, healthcare, legal)
who need AI-assisted decisions with audit trails before the EU AI Act deadline
(August 2, 2026).

### Differentiators vs. Existing OSS

| Capability | CrewAI | LangGraph | AutoGen | **Aragora** |
|-----------|--------|-----------|---------|-------------|
| Agent relationship | Collaborative | Collaborative | Collaborative | **Adversarial** |
| Decision receipts | No | No | No | **Yes (HMAC-SHA256)** |
| Dissent tracking | No | No | No | **Yes** |
| Calibrated confidence | No | No | No | **Yes (Brier scores)** |
| Policy-gated execution | No | No | No | **Yes (OpenClaw)** |
| On-chain reputation | No | No | No | **Yes (ERC-8004)** |
| EU AI Act compliance | No | No | No | **Articles 9-15** |

### Packaging for Different Buyers

| Buyer | Package | Entry Point |
|-------|---------|-------------|
| **Developer exploring** | `pip install aragora-debate` | Standalone debate engine |
| **Team adopting** | `pip install aragora` | Core platform |
| **Enterprise deploying** | `pip install aragora[enterprise,gateway]` | Full stack |
| **Web3 organization** | `pip install aragora[blockchain]` | Trust portability |

---

## Install Validation

Tier isolation is enforced by `tests/packaging/test_tiers.py` (51 tests):

| Test Class | What It Validates |
|-----------|-------------------|
| `TestTier1Core` | Core debate/gauntlet/ranking/knowledge imports (always succeed) |
| `TestTier2Gateway` | OpenClaw gateway imports with zero extra deps |
| `TestTier3Blockchain` | Blockchain module imports; web3 is lazy |
| `TestTier4Enterprise` | Auth/RBAC/security/compliance imports |
| `TestTier5Connectors` | Streaming connectors import even without aiokafka/aio-pika |
| `TestTier6Experimental` | Genesis/introspection/visualization imports |
| `TestTierIsolation` | Core tier does NOT require web3, playwright, or python3-saml |
| `TestDependencyGroups` | pyproject.toml declares all 5 tier groups |

Run the tier tests:

```bash
pytest tests/packaging/test_tiers.py -v
```

### Dependency Graph

```
                    ┌──────────────┐
                    │   aragora    │ (core: debate, gauntlet, ranking, knowledge)
                    └──────┬───────┘
          ┌────────────────┼────────────────┬──────────────────┐
          ▼                ▼                ▼                  ▼
   ┌──────────┐    ┌──────────────┐  ┌────────────┐    ┌──────────────┐
   │ gateway  │    │  enterprise  │  │ blockchain │    │  connectors  │
   │ (no deps)│    │  saml,redis  │  │  web3      │    │  kafka,amqp  │
   └──────────┘    │ prometheus   │  └────────────┘    │  twilio      │
                   └──────────────┘                    └──────────────┘
                                                              │
                                                       ┌──────────────┐
                                                       │ experimental │
                                                       │  playwright  │
                                                       │  networkx    │
                                                       └──────────────┘
```

Each tier is independently installable. No tier forces another tier's
dependencies into your environment.
