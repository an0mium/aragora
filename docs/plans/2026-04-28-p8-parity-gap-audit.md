# Pillar 8 (P8) Parity Gap Audit — Agents-as-Co-Equal-Consumers

> **Status:** read-only audit. No code, no PRs, no migrations. Strategic input for the next bounded P8 deliverable.
> **Scope:** the parity gap between the human-consumer surface (`/api/auth/*`, `/api/v1/billing/*`, `/api/v1/marketplace/*`, `/api/v2/receipts/*`, `/api/v1/budgets`, `/api/v1/reputation/*` SDK) and the agent-consumer surface (`/api/v1/a2a/*`, `/api/v1/blockchain/*`, MCP tools, `aragora/protocols/a2a/receipts.py`, `aragora/reputation/*`, `aragora/blockchain/compute_budget.py`).
> **Author:** Droid (Exec mode, 2026-04-28 overnight planning).
> **Anchor doc:** [`docs/CANONICAL_GOALS.md` line 197](../CANONICAL_GOALS.md) — *"Agents and humans participate in the same substrate as co-equal consumers, with portable reputation tied to objectively verifiable outcomes through external truth oracles."*
> **Companion docs:** [`AGENT_CIVILIZATION_SUBSTRATE.md`](AGENT_CIVILIZATION_SUBSTRATE.md), [`AGENT_CONSUMER_SURFACE.md`](AGENT_CONSUMER_SURFACE.md), [`SKIN_IN_THE_GAME_REPUTATION.md`](SKIN_IN_THE_GAME_REPUTATION.md).

---

## 0. Why this audit exists

Pillar 8 is the only canonical pillar of the seven north-star outcomes whose explicit success criterion is **symmetry**: humans and agents must be co-equal consumers of the same runtime truth. Every other pillar (debate quality, memory provenance, receipts, evidence, freshness, evolution) is *about a property of the runtime*; P8 is *about who is allowed to consume that runtime as a first-class principal*.

The supporting plan (`AGENT_CONSUMER_SURFACE.md`) commits to a stronger formulation:

> *"Every consumer surface ships in two forms, agent-readable and human-readable, backed by the same runtime truth."*

Today the platform ships almost every consumer surface in **only one** form (human), with a smaller set of agent-readable mirrors (A2A protocol, ERC-8004 blockchain endpoints, MCP tools). The mirrors do not cover the same surfaces the human consumer hits, and where they do cover the same surfaces, they are not backed by the same runtime truth in the same direction.

This audit answers three questions:

1. For each consumer surface, **does an agent form exist** and **does it touch the same backing data** as the human form?
2. Where there is a gap, **what is the severity** in terms of P8's co-equal-consumer claim?
3. What is the **single bounded first deliverable** (~200–400 LOC) that closes the highest-severity gap with the lowest blast radius and the cleanest demo?

The deliverable is intentionally **one** read-only file. There is no implementation in this audit.

---

## 1. How to read this audit

For each of six surfaces I record:

| Field | Meaning |
|---|---|
| **Human form** | The endpoint, CLI, or SDK call a human (or human-controlled UI) uses to consume this surface today. Cited as `path:line`. |
| **Agent form** | The endpoint, MCP tool, or signed contract that a software agent uses to consume the same surface programmatically without screen-scraping. Cited as `path:line`. If absent, marked **MISSING**. |
| **Backing parity** | Whether both forms are backed by the same runtime truth (same store, same write path, same authority). Three states: ✅ same backing, ⚠️ partial / drift risk, ❌ different backing or one side absent. |
| **Severity for P8** | `low` / `medium` / `high` / `critical`. High and critical means this gap, taken alone, falsifies P8 as a co-equal-consumer claim. |
| **First deliverable proposal** | A bounded ~200–400 LOC change that closes the gap without expanding scope. Always one concrete change, never a roadmap. |

`path:line` citations are pinned to the worktree at `/Users/armand/Development/aragora` on branch `docs/2026-04-28-overnight-planning` at the time of audit. When a path is referenced by directory only, that means the existence of the directory itself is the citation (e.g., `aragora/protocols/a2a/`).

The audit only **classifies and proposes**. It does not edit code, queue tickets, or trigger automation.

---

## 2. Summary table

| # | Surface | Human form | Agent form | Backing | Severity |
|---|---|---|---|---|---|
| 1 | Registration / identity | `POST /api/auth/register` (`aragora/server/handlers/auth/login.py:42`), `POST /api/v1/auth/signup` (`aragora/server/handlers/auth/signup_handlers.py:158`) | `POST /api/v1/blockchain/agents` (`aragora/server/handlers/erc8004.py:559`) — gated, queued, on-chain only | ❌ different backing, no shared identity primitive | **critical** |
| 2 | Capability discovery | `GET /api/v1/agents` (`aragora/server/handlers/agents/agents.py:390`), `GET /api/v1/marketplace/listings` (`aragora/server/handlers/marketplace_pilot.py:118`) | `GET /api/v1/a2a/agents`, `GET /.well-known/agent.json` (`aragora/server/handlers/a2a.py:350`–`351`), `list_agents_tool` MCP (`aragora/mcp/tools_module/agent.py:16`) | ⚠️ ELO-side parity but no per-agent capability-with-reputation slice yet | **medium** |
| 3 | Billing / metering | `GET /api/v1/billing/usage` (`aragora/server/handlers/usage_metering.py`), `GET /api/v1/budgets` (`aragora/server/handlers/budgets.py`) | `aragora/blockchain/compute_budget.py:65` (in-memory primitive only) — **no HTTP endpoint** | ❌ no agent-readable surface; humans see usage, agents do not | **high** |
| 4 | Decision receipts | `GET /api/v2/receipts/{id}` (`aragora/server/handlers/receipts.py:368`) | `aragora/protocols/a2a/receipts.py` `AgentReceipt` (contract-only, **no live emitter**, no endpoint) | ❌ schema exists, wire format does not | **high** (highest leverage) |
| 5 | Reputation reads / writes | SDK `/api/v1/reputation/all`, `/api/v1/reputation/{agent}`, `/api/v1/reputation/history`, `/api/v1/reputation/domain` (`sdk/python/aragora_sdk/namespaces/reputation.py:45`) | `GET /api/v1/blockchain/agents/{token_id}/reputation`, `POST /api/v1/blockchain/agents/{token_id}/reputation` (ERC-8004, gated); `aragora/reputation/store.py` (in-memory, flag-gated) | ⚠️ split backing — SDK reads ELO/calibration, blockchain reads ERC-8004, AGT-05 store is invisible to either | **medium** |
| 6 | Marketplace participation | `GET /api/v1/marketplace/listings`, `POST /api/v1/marketplace/listings/{id}/install` (`aragora/server/handlers/marketplace_pilot.py:148`) | **MISSING** — no agent-bid, agent-list, agent-settle surface; no MCP tool for marketplace install/rate either | ❌ marketplace is human-consumer only | **high** |

**Single highest-severity gap with the cleanest first deliverable:** Surface 4 (decision receipts), because the agent-readable schema (`AgentReceipt`) is already implemented, signed, content-addressed, and verifiable; only the wire endpoint is missing. Closing it is a ~250 LOC change that immediately raises three other surfaces (5, 3, 6) into solvable territory.

---

## 3. Per-surface detail

### Surface 1 — Registration / identity

#### Human form

A human creates an Aragora account through one of two flows:

1. The legacy auth flow `POST /api/auth/register`, decorated at `aragora/server/handlers/auth/login.py:42` with `@api_endpoint(method="POST", path="/api/auth/register", ...)`. The handler implementation is `handle_register` at `login.py:90`. It accepts `{email, password, name, organization}`, validates email and password, creates the user and (optionally) an org via `user_store.create_user`, hashes the password, mints an access/refresh JWT pair via `aragora.billing.jwt_auth.create_token_pair`, and returns the user envelope plus the token pair.
2. The newer signup flow `POST /api/v1/auth/signup` (`handle_signup` at `aragora/server/handlers/auth/signup_handlers.py:158`; route documented at line 11 of the module docstring). This is the surface used by the live frontend onboarding wizard.

After registration, the human can retrieve their own principal via `GET /api/auth/me` (`aragora/server/handlers/auth/handler.py:763`, `_handle_get_me`). They can mint or revoke API keys via `POST /api/auth/api-key`, `DELETE /api/auth/api-key`, `GET /api/auth/api-keys`, `DELETE /api/auth/api-keys/{prefix}` (all in `aragora/server/handlers/auth/api_keys.py:45`–`218`).

The CLI surface for the same flow is `aragora/cli/api_keys.py:1`, which stores LLM API keys in a macOS Keychain or encrypted file backend, and shells out to the same HTTP endpoints for tenant principal operations.

The backing store is a `UserStore` resolved via `handler_instance._get_user_store()` (sql + Redis-cached). Org and user records live in the canonical Postgres / SQLite store (`aragora/storage/postgres_store.py`, `aragora/db/`).

#### Agent form

The closest agent-side analog is `POST /api/v1/blockchain/agents` defined in `aragora/server/handlers/erc8004.py:559` (`handle_register_agent`). Its semantics are:

- Decorated with `@require_permission("blockchain:write")`, `@rate_limit(requests_per_minute=10)`, `@with_timeout(120.0)`.
- Accepts `{agent_uri, metadata, requested_by, approval_id, receipt_id}`.
- Calls `enqueue_register_agent_action(...)` which **queues** the registration as a pending action requiring approval; the response is `status=202` with `requires_approval=True`. The agent is not registered synchronously; a human has to approve the action.
- Backed by `IdentityRegistryContract` in `aragora/blockchain/contracts/identity.py`, which writes to the on-chain ERC-8004 registry (when configured).

Adjacent agent-side discovery surfaces exist:

- The A2A discovery endpoint `GET /.well-known/agent.json` and `GET /api/v1/a2a/agents` in `aragora/server/handlers/a2a.py:350`–`353`. These advertise the agents that *Aragora itself* is willing to expose to A2A clients. They do not register external agents.
- The MCP `register_agent_tool` in `aragora/mcp/tools_module/control_plane.py:1`. This registers an agent into the **control plane** (the in-process registry used by the scheduler), not into a tenant identity primitive.

#### Backing parity

❌ Different backing.

- The human form writes to `UserStore` (relational tenant DB) and mints JWTs / API keys signed by `aragora.billing.jwt_auth`.
- The agent form writes (or queues a write) to the ERC-8004 on-chain registry through `IdentityRegistryContract`. It does not appear in `UserStore`. Agents registered this way cannot mint API keys, do not have a JWT principal, and do not have any of the tenant primitives (org, RBAC, billing customer record).
- The MCP control-plane register surface writes to a third in-process registry that is reset every server restart, has no identity attestation, and is not visible to either of the above.

There is no shared identity primitive that says *"this principal — whether human or agent — exists, has these capabilities, and can authenticate against the same authority"*. P8 calls for exactly that.

#### Severity for P8

**Critical.**

Without a shared identity primitive, every other parity claim is inferential: an agent can call `POST /api/v1/blockchain/agents` to assert an on-chain identity, but it cannot then turn around and call `GET /api/v1/billing/usage` for that identity, because billing has no concept of an on-chain agent ID. The platform principle from `AGENT_CONSUMER_SURFACE.md` ("two forms, same runtime truth") is structurally unreachable while identity itself is split.

#### First deliverable proposal

**Out of scope for the *first* deliverable.** Identity unification is the largest of the six gaps; it pulls in `UserStore`, `IdentityRegistryContract`, JWT minting, API-key minting, RBAC role assignment, and tenant org provisioning. A correct fix is multi-PR and is not bounded to ~200–400 LOC.

What can be bounded and is the right *second* deliverable (after surface 4): a read-only **principal resolver** at `GET /api/v1/principals/{id}` that maps any of {`user_id`, `api_key_prefix`, `agent_uri`, `wallet_address`, `erc8004_token_id`} to a single `Principal` envelope `{id, kind: "human" | "agent", display_name, capabilities[], reputation_summary | null}`. ~250 LOC. Pure read; no schema migration. Establishes the noun before the verbs.

This audit recommends **deferring** identity-unification reads until after surface 4 has been wired, because surface 4 forces the canonical content-addressed receipt to exist, and the receipt's `issuer` field is the exact place where principal identity becomes legible.

---

### Surface 2 — Capability discovery

#### Human form

A human discovers what Aragora can do through three orthogonal surfaces:

1. **Agent registry list:** `GET /api/v1/agents` at `aragora/server/handlers/agents/agents.py:390` (`_list_agents`). Returns the leaderboard agents (43 registered types per `AGENTS.md`), with optional `include_stats=true` adding ELO, match count, and calibration. Backed by `EloSystem` (`aragora/ranking/elo.py`) and `CalibrationTracker`.
2. **Local LLM detection:** `GET /api/v1/agents/local` (`agents.py:450`), `GET /api/v1/agents/local/status` (`agents.py:487`), `GET /api/v1/agents/health` (`agents.py:531`), `GET /api/v1/agents/availability` (`agents.py:700`). Returns Ollama / LM Studio detection results plus circuit-breaker health.
3. **Marketplace catalog:** `GET /api/v1/marketplace/listings` (the `MarketplacePilotHandler.handle` GET path at `aragora/server/handlers/marketplace_pilot.py:118`, with the read decorator stack at line 116) — browse with filters; `GET /api/v1/marketplace/listings/featured`; `GET /api/v1/marketplace/listings/stats`; `GET /api/v1/marketplace/listings/{id}` for detail. Older `/api/v1/marketplace/templates` is at `aragora/server/handlers/marketplace.py:259`. Backed by `MarketplaceService` (`aragora/marketplace/service.py`).

The CLI mirror lives at `aragora/cli/marketplace.py:1` (`aragora marketplace list/search/get/export/import/categories/rate/use`).

#### Agent form

This is the surface where agent parity is **closest** to existing.

1. **A2A discovery:**
   - `GET /.well-known/agent.json` (`aragora/server/handlers/a2a.py:351`, `366`) — the standard A2A discovery endpoint advertising what this server offers as an agent.
   - `GET /api/v1/a2a/agents` (`a2a.py:353`) — list available agents in A2A shape.
   - `GET /api/v1/a2a/agents/*` (`a2a.py:354`) — per-agent card.
   - `GET /api/v1/a2a/openapi.json` (`a2a.py:359`, `383`) — the A2A protocol's own OpenAPI surface.

2. **MCP discovery tools:**
   - `list_agents_tool` (`aragora/mcp/tools_module/agent.py:16`) returns the agent registry.
   - `list_registered_agents_tool` (`aragora/mcp/tools_module/control_plane.py`) returns the control-plane registry.
   - `list_workflow_templates_tool` (`aragora/mcp/tools_module/workflow.py`).
   - The full MCP tool inventory (~70+ tools across `aragora/mcp/tools_module/`) is itself a machine-readable capability surface, served by `AragoraMCPServer` at `aragora/mcp/server.py:1`.

3. **Blockchain discovery:** `GET /api/v1/blockchain/agents` (`erc8004.py:618`) lists on-chain agents. `GET /api/v1/blockchain/config` advertises chain configuration.

#### Backing parity

⚠️ Partial. Two specific drifts:

- **Capability-with-reputation slice missing.** `AGENT_CONSUMER_SURFACE.md §S2` requires `GET /api/v1/a2a/agents/<id>/capabilities` returning *capabilities + reputation summary in the same envelope*. The current `/api/v1/a2a/agents/*` returns capabilities but does not join reputation; reputation is a separate fetch through `/api/v1/blockchain/agents/{id}/reputation` or the SDK `/api/v1/reputation/{agent}`. A capability discovery without reputation context forces every consumer to do the join client-side, which agents will do incorrectly more often than humans.
- **Marketplace catalog is not in the agent form.** The marketplace `/api/v1/marketplace/listings` is a human surface only; there is no MCP tool that wraps it and no A2A path that exposes it. So an external agent that wants to *find a workflow template, agent pack, or skill* on the marketplace must screen-scrape the human listing endpoint or have the marketplace listing path baked into its prompt.

The `/api/v1/agents` and `/api/v1/a2a/agents` paths share the `EloSystem` backing, so the basic identity-and-stats slice has parity.

#### Severity for P8

**Medium.**

Discovery is the easiest surface to give parity to because the human form is already mostly read-only and JSON. The gap is the missing capability+reputation join and the absent marketplace mirror, not absent infrastructure.

#### First deliverable proposal

A single new endpoint `GET /api/v1/a2a/agents/{id}/capabilities` that joins:

- the agent card from `/api/v1/a2a/agents/{id}`,
- the leaderboard slice from `EloSystem` (ELO, win-rate, calibration),
- the AGT-05 in-memory reputation summary from `aragora.reputation.ReputationStore.agent_score(id)` when `ARAGORA_REPUTATION_FLOW_ENABLED` is set, otherwise `null`,
- the on-chain reputation summary from `IdentityRegistryContract.get_reputation` when configured, otherwise `null`.

Returns one canonical envelope; both human and agent forms can call it. ~200 LOC: one handler, four reads, schema-versioned response. Fully read-only; no migration; safely additive.

This audit recommends this as the **third** deliverable after surface 4 and the principal resolver, because both reputation backings (ERC-8004 and AGT-05 in-memory) are still maturing, so a JSON envelope that nullably wraps either is the right level of commitment today.

---

### Surface 3 — Billing / metering

#### Human form

Humans consume billing/metering through a deep set of endpoints handled by:

- `aragora/server/handlers/usage_metering.py` (`UsageMeteringHandler` at line 36; routes registered at lines 49–58):
  - `GET /api/v1/billing/usage` (dispatch at line 92)
  - `GET /api/v1/billing/usage/summary` (line 95)
  - `GET /api/v1/billing/usage/breakdown` (line 98)
  - `GET /api/v1/billing/usage/export` (line 104)
  - `GET /api/v1/billing/limits` (line 101)
  - `GET /api/v1/quotas`, `GET /api/v1/quotas/usage`, `GET /api/v1/quotas/{resource}`
- `aragora/server/handlers/billing/core.py`:
  - `GET /api/v1/billing/plans`
  - `GET /api/v1/billing/subscription`
  - `POST /api/v1/billing/checkout`
  - `POST /api/v1/billing/portal`
  - `POST /api/v1/billing/cancel`
  - `POST /api/v1/billing/resume`
- `aragora/server/handlers/budgets.py` (`BudgetsHandler`):
  - `GET /api/v1/budgets`, `POST /api/v1/budgets`
  - `GET /api/v1/budgets/{id}`, `PATCH /api/v1/budgets/{id}`, `DELETE /api/v1/budgets/{id}`
  - `GET /api/v1/budgets/{id}/alerts`, `POST /api/v1/budgets/{id}/alerts/{alert_id}/acknowledge`
  - `POST /api/v1/budgets/{id}/override`, `DELETE /api/v1/budgets/{id}/override/{user_id}`
  - `POST /api/v1/budgets/{id}/reset`
  - `GET /api/v1/budgets/{id}/transactions`
  - `GET /api/v1/budgets/{id}/trends`
  - `GET /api/v1/budgets/summary`, `GET /api/v1/budgets/trends`
  - `POST /api/v1/budgets/check`
- `aragora/server/handlers/billing/cost_dashboard.py`:
  - `GET /api/v1/billing/dashboard`
- `aragora/cli/billing.py` and `aragora/cli/commands/billing_ops.py` mirror these for terminal use.

The billing principal is the org / user resolved via JWT or API key. The metering store sits behind `aragora/billing/{cost_tracker,budget_manager,metering,forecaster}.py`.

#### Agent form

The platform has the *primitive* but not the *surface*.

`aragora/blockchain/compute_budget.py:65` defines `ComputeBudgetManager` with:

- `allocate(agent_id, task_complexity)` → `int`
- `charge(agent_id, tokens_used)` → `None`
- `reward_accuracy(agent_id, epistemic_score)` → `int`
- `penalize_inaccuracy(agent_id, epistemic_score)` → `int`
- `get_budget(agent_id)` → `ComputeBudget`

This is the on-chain-aware compute budget primitive, with optional `StakingRegistry` and `ReputationRegistryContract` backing. It is **not exposed via HTTP**. There is no `GET /api/v1/a2a/agents/{id}/budget`, no MCP `get_compute_budget_tool`, and no SDK method.

The closest live agent-side billing surface is the chargeback path *implied* by `AGENT_CONSUMER_SURFACE.md §S3` ("every billable event emits a signed receipt with the consumed budget delta") — but that path is not wired. `aragora/billing/calibration_cost_bridge.py` exists but is a calibration bridge, not an agent-readable budget endpoint.

#### Backing parity

❌ The human and agent forms do not share a backing because the agent form does not exist as an external surface.

Even if `ComputeBudgetManager` were exposed via HTTP, it would not be backed by the same store as `aragora.billing.metering`, so an agent reading its budget and a human reading the same org's usage would see two different numbers. The reconciliation between compute-credits (agent-economy) and usage-tier limits (human-economy) is unbuilt.

#### Severity for P8

**High.**

P8's substrate claim hinges on agents being able to *transact*: pay other agents, be paid by humans, watch their own budget, get billed for the same units humans get billed for. Today an agent cannot ask "how much budget do I have left?" through any documented public surface. Without that, every other agent capability is gated on a human watching the meter.

#### First deliverable proposal

A read-only HTTP exposure of `ComputeBudgetManager.get_budget(agent_id)` at:

```
GET /api/v1/a2a/agents/{agent_id}/budget
```

Returns:

```json
{
  "agent_id": "agt_...",
  "compute_credits": {
    "total": 1200,
    "used": 480,
    "earned": 60,
    "penalty": 20,
    "available": 760
  },
  "billing_principal": {"kind": "agent", "id": "agt_..."},
  "schema_version": "1.0"
}
```

~150 LOC: one handler, one in-memory read, RBAC `blockchain:read`, schema-versioned response. Does **not** wire chargeback receipts (that requires surface 4 first); does **not** unify with `aragora/billing/metering.py` (that requires the principal resolver). Pure read; safely additive.

This is recommended as the **fourth** deliverable, after surface 4 (receipts), the principal resolver (surface 1's bounded slice), and the capability+reputation join (surface 2). Doing the budget read before receipts would publish an unsigned number with no audit trail.

---

### Surface 4 — Decision receipts

#### Human form

Humans consume decision receipts through `aragora/server/handlers/receipts.py`:

- `GET /api/v2/receipts` — list with filters (line 12 in module docstring; routes registered at line 368)
- `GET /api/v2/receipts/search` — full-text search (line 13)
- `GET /api/v2/receipts/{receipt_id}` — fetch one (line 14)
- `GET /api/v2/receipts/{receipt_id}/export` — export json/html/md/pdf (line 15)
- `GET /api/v2/receipts/{receipt_id}/verify` — verify integrity + signature (line 16)
- `POST /api/v2/receipts/{receipt_id}/verify` — body-driven verify (line 17)
- `POST /api/v2/receipts/{receipt_id}/verify-signature` — verify cryptographic signature only (line 18)
- `POST /api/v2/receipts/verify-batch` — batch sig verify (line 19)
- `POST /api/v2/receipts/sign-batch` — batch sign (line 20)
- `POST /api/v2/receipts/batch-export` — batch export to ZIP (line 21)
- `GET /api/v2/receipts/stats` (line 22)
- `POST /api/v2/receipts/{receipt_id}/share` (line 23)
- `GET /api/v2/receipts/share/{token}` (line 24)
- `GET /api/v1/receipts/deliveries` — legacy delivery history bridge (line 25)

Receipts are stored in the `ReceiptStore` (`aragora/server/handlers/receipts.py`, plus `aragora/export/decision_receipt.py` for schema). The receipt format is a JSON envelope with HTML rendering, designed for human inspection.

#### Agent form

This is the most strategically interesting gap. The agent-readable receipt format **already exists** as a contract:

`aragora/protocols/a2a/receipts.py` defines `AgentReceipt` at line 230 (with the build classmethod at line 263). Key properties:

- **Content-addressed.** `receipt_id` is a SHA-256-derived hash of the canonical payload (`receipt.py:_canonical_payload`, `_sha256_hex`).
- **Ed25519-signed.** Build path `AgentReceipt.build(...)` accepts a private signing key, signs the canonical payload + key metadata, embeds `signature_key_id`, `signature_algorithm`, `verification_key_sha256`, and a base64 signature. Verify path `AgentReceipt.verify_signature(...)` works with a key resolver `(issuer, key_id) → public_key`.
- **Schema-versioned.** `schema_version = "1.0"` (`AGENT_RECEIPT_SCHEMA_VERSION`); forward-compat by major version.
- **Subject-typed.** `subject_kind` + `subject` lets the same envelope wrap a decision text, a CruxSet, a claim, a prediction outcome, or a debate result.
- **Carries reputation deltas.** `reputation_deltas_applied` is a tuple of `ReputationDelta(agent_id, domain, delta, reason)` so a receipt is also the audit trail for AGT-05 settlement.
- **Carries dissent.** `dissent` tuple records minority positions, addressing the closed-loop-reputation risk from `AGENT_CONSUMER_SURFACE.md §Risks`.
- **Freshness + settlement.** `freshness_sla_seconds` (default 24h) and `settlement_window_seconds` (default 7d) are first-class, supporting the "stale evidence" outcome from AGT-04.

The module docstring is explicit:

> *"Activation: this module is contract-only. Nothing publishes or signs receipts automatically. Server endpoints that emit AgentReceipts will land in a follow-up PR..."*

There is **no live emitter**. There is **no endpoint that returns an `AgentReceipt`**. There is no MCP tool that wraps it. The schema is invisible to consumers.

#### Backing parity

❌ Schema exists, wire format does not.

Concretely:
- A consumer calling `GET /api/v2/receipts/{id}` today gets the human-format envelope.
- There is no `GET /api/v2/receipts/{id}/agent-form` or `GET /api/v1/a2a/receipts/{id}`.
- The MCP tool `get_decision_receipt_tool` (`aragora/mcp/tools_module/knowledge.py`) returns the human envelope, not the `AgentReceipt`. Its result is a JSON dict but is not Ed25519-signed and is not content-addressed in the AGT-02 sense.
- The CLI surface for receipts (e.g., the `aragora receipts ...` family) returns the human envelope.

#### Severity for P8

**High.** This is the single highest-leverage gap, even though it is not the largest gap.

It is high-severity because:

1. **It falsifies P8 most directly.** A co-equal consumer must be able to *prove* it consumed a decision and act on it without screen-scraping. AgentReceipt is exactly the proof primitive; it is sitting on the shelf.
2. **It unblocks three other gaps.** Surface 5 (reputation) needs receipts to be the audit trail for AGT-05 settlement. Surface 3 (billing) needs receipts as the chargeback envelope. Surface 6 (marketplace) needs receipts as the agent-bid acceptance proof.
3. **The blast radius is small.** AgentReceipt has no transitive dependency on UserStore, JWT, ERC-8004 chain config, or marketplace state. It only depends on the Ed25519 signing key and the canonical receipt data already in `ReceiptStore`.
4. **The demo is clean.** "Fetch a receipt as JSON, verify the signature with the public key, parse the CruxSet, settle a reputation delta" is a reproducible end-to-end demo in <100 lines of client code.

#### First deliverable proposal — **recommended as the actual first P8 deliverable**

**Endpoint:** `GET /api/v2/receipts/{receipt_id}/agent-form`

**Behavior:**

1. Look up the human receipt by `receipt_id` from `ReceiptStore`.
2. Map fields onto `AgentReceipt.build(...)` arguments:
   - `issuer` ← Aragora server identity (`AGT_RECEIPT_ISSUER` env var; default `aragora-server`).
   - `subject_kind` ← receipt type (`"decision"`, `"debate_result"`, `"crux_set"`).
   - `subject` ← the human receipt's machine-readable core (`debate_id`, `task`, `consensus`, `agents`, `votes`, …).
   - `cruxset` ← the AGT-01 CruxSet when present on the human receipt; `None` otherwise.
   - `dissent` ← extracted from the human receipt's vote/critique tuple.
   - `reputation_deltas_applied` ← from the AGT-05 settlement record when `ARAGORA_REPUTATION_FLOW_ENABLED` is on; empty tuple otherwise.
   - `freshness_sla_seconds` / `settlement_window_seconds` ← receipt-type defaults from a small lookup table.
   - `provenance` ← `{"source": "aragora", "human_receipt_id": ..., "schema_versions": {"human": ..., "agent": "1.0"}}`.
   - `signing_key` ← read from `AGT_RECEIPT_SIGNING_KEY` (Ed25519 raw or PEM).
   - `signature_key_id` ← from `AGT_RECEIPT_SIGNING_KEY_ID`; default `default`.
3. Return `AgentReceipt.to_json()` with `Content-Type: application/json` and a header `X-Receipt-Schema-Version: 1.0`.

**Plus a verify endpoint:** `POST /api/v2/receipts/{receipt_id}/agent-form/verify`

- Accepts `{verification_key?, key_id?}` in the body; if absent, resolves via the same env var the issuer used.
- Returns `{verified: bool, receipt_id_match: bool, signature_match: bool, fresh: bool, within_settlement: bool}`.

**Optional MCP tool:** `get_agent_receipt_tool(receipt_id)` wrapping the GET endpoint, exposing the same JSON to MCP clients without HTTP.

**Bound:** ~250 LOC across one handler module (`aragora/server/handlers/agent_receipts.py`), one minor edit to `aragora/mcp/tools_module/knowledge.py` for the MCP wrapper, one entry in the OpenAPI manifest, one test file.

**Why this is the right first deliverable:**

| Criterion | Evidence |
|---|---|
| Bounded LOC | The schema, signing, verification, content addressing, and freshness logic all already exist. The new code is glue + handler + tests. |
| No new dependencies | Uses `cryptography` (already in `aragora/protocols/a2a/receipts.py`), `ReceiptStore` (already wired), `aragora.reputation.store` (already in repo). |
| Closes the highest-severity gap | Surface 4 (high) directly. |
| Unblocks at least three other gaps | Surface 5 (reputation deltas now have a wire format), Surface 3 (chargeback receipts now have a shape), Surface 6 (agent-bid acceptance now has a content-addressed proof). |
| Reversible | Pure read endpoint; no schema migration; failure mode is HTTP 503 plus the existing human endpoint stays untouched. |
| Demoable in one paragraph | "Fetch JSON, verify with public key, parse CruxSet, settle reputation delta." |
| Aligned with `AGENT_CONSUMER_SURFACE.md §S4` and the AGT-02 sequencing table | Step 1 ("Agent-readable receipt schema spec") is already in the repo; this is step 1.5 ("publish + verify endpoint"). |

**Out of scope for this deliverable:**

- Wiring receipt emission into the live debate finalize path. (Receipts are still produced by the existing path; this just adds an alternate read shape.)
- Anchoring AgentReceipts on-chain via `aragora/blockchain/receipt_anchor.py`. (That is a follow-up.)
- Computing AGT-05 reputation deltas inside the receipt path. (The AGT-05 store provides them; this just embeds them when present.)

---

### Surface 5 — Reputation reads / writes

#### Human form

The Python SDK at `sdk/python/aragora_sdk/namespaces/reputation.py:36–85` exposes:

- `client.reputation.list_all()` → `GET /api/v1/reputation/all`
- `client.reputation.get(agent)` → `GET /api/v1/reputation/{agent}`
- `client.reputation.get_history(agent?, period?)` → `GET /api/v1/reputation/history`
- `client.reputation.get_by_domain(domain)` → `GET /api/v1/reputation/domain`

The async mirror at lines 99–135 has the same shape.

The CLI surface (`aragora reputation ...`) and the live frontend dashboards (`aragora/live/src/app/.../reputation/`) consume the same SDK.

The backing on the server side is the ELO + calibration stack (`aragora/ranking/elo.py`, `aragora/agents/calibration_tracker.py`) plus the reputation calibration bridge `aragora/reputation/selection_bridge.py:1` (`ReputationCalibrationBridge`).

#### Agent form

There are **three** agent-side reputation surfaces, and they do not agree with each other or with the SDK:

1. **ERC-8004 on-chain reputation.** `GET /api/v1/blockchain/agents/{token_id}/reputation` (`erc8004.py`, `handle_get_reputation`) and `POST /api/v1/blockchain/agents/{token_id}/reputation` (feedback submission). Backed by `ReputationRegistryContract` in `aragora/blockchain/contracts/reputation.py`. This is the "official" on-chain reputation.

2. **AGT-05 in-memory reputation flow.** `aragora/reputation/store.py:96` (`ReputationStore`) defines an append-only JSONL ledger of `ReputationDelta`s, plus `agent_score(agent_id)` returning `AgentScore(agent_id, running_score, delta_count, domains)`. Activated only when `ARAGORA_REPUTATION_FLOW_ENABLED=1`. Has no HTTP endpoint of its own; the only public surface that touches it is `aragora/reputation/selection_bridge.py` (used internally by team selection).

3. **MCP knowledge tool.** `get_decision_receipt_tool` (`aragora/mcp/tools_module/knowledge.py`) returns receipts that include reputation deltas as part of the receipt envelope. This is read-only, indirect, and closely coupled with surface 4.

#### Backing parity

⚠️ Split backing.

The SDK (`/api/v1/reputation/*`) reads from the ELO stack. The blockchain endpoint reads from the on-chain registry. The AGT-05 store is invisible to either. None of the three is the canonical source.

The write story is even more split:
- The SDK has no public write path (intentional — humans don't write reputation directly).
- The blockchain endpoint has `POST /api/v1/blockchain/agents/{token_id}/reputation` for submitting feedback, but it requires `blockchain:write` and is gated on chain configuration.
- The AGT-05 settlement function `aragora.reputation.settlement.settle_claim` is the *only* legitimate write path per `SKIN_IN_THE_GAME_REPUTATION.md`; it is not exposed via HTTP at all.

`AGENT_CONSUMER_SURFACE.md §S5` is explicit:

> *"reputation write path is mediated only by AGT-05 (claims→predictions→resolution→reputation flow); agents cannot self-promote"*

The current state respects "agents cannot self-promote" but only because no public write surface exists. The downside: agents and humans cannot agree on what an agent's reputation is, because they read three different stores.

#### Severity for P8

**Medium.**

The split backing is reconcilable because all three sources have a stable identity key (the agent_id / token_id / agent_uri). What is missing is the join. The risk to P8 is not catastrophic — humans and agents can both read *some* reputation surface — but the answers will diverge under load, and the divergence is silent.

#### First deliverable proposal

A read-only join endpoint at:

```
GET /api/v1/a2a/agents/{agent_id}/reputation
```

Returns:

```json
{
  "agent_id": "agt_...",
  "summary": {
    "running_score": 0.42,
    "delta_count": 17,
    "domains": ["debate_position", "code_pr", "prediction_market"]
  },
  "sources": {
    "elo": {"rating": 1547, "matches": 42, "wins": 28},
    "calibration": {"score": 0.78, "samples": 32},
    "agt05_store": {"running_score": 0.42, "delta_count": 17, "fresh_through": "2026-04-28T00:00:00Z"},
    "erc8004": {"on_chain_score": null, "feedback_count": null, "chain_id": 11155111}
  },
  "schema_version": "1.0"
}
```

Each source is reported separately; the `summary` is the AGT-05 view (the canonical one per `SKIN_IN_THE_GAME_REPUTATION.md`) when enabled, falling back to ELO when not. ~200 LOC: one handler, four reads (with each guarded by its own feature flag), one schema-versioned response. Pure read; no write.

This is recommended as the **fifth** deliverable, after surfaces 4, 1 (principal resolver), 2 (capabilities), and 3 (budget). The reason for that order: the reputation join is most useful *after* receipts can be agent-readable, because then the join can also surface the receipts that produced each delta.

---

### Surface 6 — Marketplace participation

#### Human form

Humans participate in the marketplace through:

- `aragora/server/handlers/marketplace_pilot.py:117` (read) and `:148` (write):
  - `GET /api/v1/marketplace/listings` (browse)
  - `GET /api/v1/marketplace/listings/featured`
  - `GET /api/v1/marketplace/listings/stats`
  - `GET /api/v1/marketplace/listings/{id}` (detail)
  - `POST /api/v1/marketplace/listings/{id}/install`
  - `POST /api/v1/marketplace/listings/{id}/rate`
  - `POST /api/v1/marketplace/listings/{id}/launch-debate`
- `aragora/server/handlers/marketplace.py:259` (older templates path):
  - `GET /api/v1/marketplace/templates`, `GET /api/v1/marketplace/templates/{id}`
  - `POST /api/v1/marketplace/templates`, `DELETE /api/v1/marketplace/templates/{id}`
  - `POST /api/v1/marketplace/templates/{id}/ratings`, `GET .../ratings`
  - `POST /api/v1/marketplace/templates/{id}/star`
  - `GET /api/v1/marketplace/categories`
  - `GET /api/v1/marketplace/templates/{id}/export`, `POST /api/v1/marketplace/templates/import`

The CLI mirror lives at `aragora/cli/marketplace.py:1` (`aragora marketplace list/search/get/export/import/categories/rate/use`). Backed by `aragora/marketplace/service.py` (`MarketplaceService`).

#### Agent form

**MISSING.**

Concretely, none of the following exist:

- An A2A path for marketplace browsing or installation.
- An MCP tool for marketplace install / rate / launch-debate. The MCP tool inventory at `aragora/mcp/tools_module/` does not include a `marketplace` module.
- An agent-bid surface (an agent offering a capability for compute credits).
- An agent-list surface (an agent posting itself as a marketplace listing).
- An agent-settle surface (an agent claiming payment after delivering a marketplace-promised capability).

The closest agent-side surface is the A2A task submission endpoint `POST /api/v1/a2a/tasks` (`aragora/server/handlers/a2a.py:356`), which is a transient task-submission path, not a marketplace listing-and-settlement path. Tasks are submitted, run, and discarded; they do not become marketplace listings, do not have ratings, do not appear in the catalog, and cannot be browsed by other agents looking for capabilities.

#### Backing parity

❌ Marketplace is human-consumer only.

#### Severity for P8

**High.**

P8's strongest formulation in `AGENT_CONSUMER_SURFACE.md §S3` is "agent A can pay agent B for a debate, with both sides emitting verifiable chargeback receipts." This is a marketplace participation claim. With no agent-side marketplace, the claim is currently false.

That said, this is the **largest** gap in scope: it touches surface 1 (identity, because agent listers need a stable principal), surface 3 (billing, because settlement requires compute-credit transfer), surface 4 (receipts, because the chargeback envelope is `AgentReceipt`), and surface 5 (reputation, because every settled marketplace transaction should produce an AGT-05 delta). It is correctly **last** in the sequencing table from `AGENT_CONSUMER_SURFACE.md` (step 6, "Operator-parity surfaces").

#### First deliverable proposal

The bounded first slice is **read-only marketplace mirroring for agents**, not agent-side participation. Specifically:

```
GET /api/v1/a2a/marketplace/listings
GET /api/v1/a2a/marketplace/listings/{id}
```

These wrap `MarketplaceService.list_listings()` and the per-listing detail, returning the same JSON envelope but with:
- A schema-versioned response.
- An `agent_readable: true` flag in each listing.
- Only listings flagged as agent-installable (a new boolean on `MarketplaceListing` defaulting to `false`, set to `true` for the workflow templates that don't require interactive UI).

Plus an MCP tool `list_marketplace_listings_tool` wrapping the GET path.

~250 LOC: two handlers, one boolean flag added to the listing dataclass, one MCP tool, one test. **No** agent-bid path, **no** agent-list path, **no** agent-settle path. Those follow once surfaces 1, 3, 4 are wired.

This is recommended as the **sixth** (final) deliverable in the bounded P8 sequence proposed by this audit.

---

## 4. Cross-cutting findings

### 4.1 The "two forms, same runtime truth" principle is partially observed

Of the six surfaces:

- **One** ships in both forms backed by the same truth (surface 2 — discovery, partially: ELO-side parity yes, capability+reputation join no).
- **Three** ship a human form with an agent-readable schema but no live emitter (surface 4 — receipts) or no HTTP exposure of the agent-side primitive (surface 3 — budget; surface 5 — AGT-05 store).
- **Two** ship in human form only (surface 1 — registration is split between two backings; surface 6 — marketplace has no agent surface at all).

The shape of the gap is consistent: **the agent primitives almost always exist as Python contracts; the wire surface does not.** This is good news. It means the first two or three P8 deliverables can be pure handler-glue PRs of <300 LOC each, with no schema migrations, no new dependencies, and no live behavior change for the human surface.

### 4.2 The principal resolver is the second-most-strategic missing piece

Identity unification (surface 1) is the only "critical" severity in this audit. But it cannot be solved in a single ~200–400 LOC PR because it touches `UserStore`, `IdentityRegistryContract`, the JWT signer, the API-key store, RBAC, and tenant org provisioning.

The bounded surrogate is a **principal resolver** at `GET /api/v1/principals/{id}`:

```json
{
  "id": "...",
  "kind": "human" | "agent",
  "display_name": "...",
  "capabilities": ["debate", "code_review", ...],
  "reputation_summary": null | {...},
  "sources": {
    "user_store": {...} | null,
    "identity_registry": {...} | null,
    "control_plane": {...} | null
  }
}
```

This is read-only, joins the existing three identity backings, and lets every other surface (receipts, budgets, marketplace, reputation) reference principals through one shape. ~250 LOC. It is **not** the first deliverable because it has no signed audit shape; that is what AgentReceipt provides. But it is the second deliverable.

### 4.3 The AGT-05 reputation flow is fully implemented but invisible

`aragora/reputation/__init__.py:1` exports a complete settlement pipeline:

- `StakeableClaim` and `ResolvedClaim` (`aragora/reputation/types.py`) — the unified shape across AGT-04 prediction-market resolution and AGT-01 CruxSet resolution.
- `settle_claim(claim, resolved)` (`aragora/reputation/settlement.py`) — the proper-scoring-rule implementation.
- `ReputationStore` (`aragora/reputation/store.py`) — the in-memory ledger with optional JSONL persistence and exponential decay.
- `ReputationCalibrationBridge` (`aragora/reputation/selection_bridge.py`) — the dispatch-eligibility integration with team selection.
- `anchor_delta` (`aragora/reputation/anchor.py`) — the on-chain anchoring path.
- `bridge_from_market_position` and `bridge_from_crux_position` — the two-source ingest path.

All of this is gated behind `ARAGORA_REPUTATION_FLOW_ENABLED`. There is no HTTP surface that exposes any of it. Every external consumer (human or agent) is structurally unable to see the canonical reputation answer.

This is a different problem from the surface-5 gap, which was specifically about *reads*. This is about the *flag itself*: the canonical answer exists, is gated by a single environment variable, and has zero documented activation path. The audit recommends keeping the flag (it correctly reflects the AGT-05 / AGT-04 dependency) but documenting that the flag is the gate, and that surface 5 reads should report `agt05_store: null` until the flag is set.

### 4.4 ERC-8004 chain dependency is a load-bearing brittleness

Surfaces 1 (registration), 5 (reputation), and 6 (marketplace settlement) all currently route through ERC-8004 contracts when configured. The contracts are wrappers around web3 ABIs (`aragora/blockchain/contracts/{identity,reputation,staking,validation}.py`). When the chain is not configured, every one of these surfaces returns 503.

For local development, demo, and staging use cases — which are where P8 will first be exercised — the chain is rarely configured. This means the agent surface is *also* effectively absent in those environments, even where the human surface works.

The audit recommends that every P8 deliverable in this proposal include a **flag-gated, in-memory fallback** mirroring the chain shape, so the agent surface degrades gracefully when the chain is absent. This is consistent with how `aragora/reputation/store.py` already handles the AGT-05 case.

### 4.5 The MCP tool surface is the one surface that ships at parity

The MCP tool inventory at `aragora/mcp/tools_module/` (~70+ tools across `agent.py`, `audit.py`, `browser.py`, `business_memory.py`, `canvas.py`, `chat_actions.py`, `checkpoint.py`, `codebase.py`, `context_tools.py`, `control_plane.py`, `debate.py`, `evidence.py`, `gauntlet.py`, `integrations.py`, `knowledge.py`, `memory.py`, `pipeline.py`, `self_improve.py`, `trending.py`, `verification.py`, `workflow.py`) is the closest the platform has to a fully agent-readable consumer surface. It does not require HTTP, JWT, or RBAC; it goes through the MCP transport layer.

The gap is that MCP tools are organized around *capabilities* (run a debate, query knowledge, list agents) rather than around the *consumer surfaces* this audit cares about (register an account, check your budget, fetch a signed receipt, install a marketplace listing). The right move is not to expand MCP to cover all six P8 surfaces; it is to add a thin MCP tool **per HTTP endpoint** added by P8 deliverables, so agents that consume Aragora through MCP and agents that consume it through HTTP see the same data.

### 4.6 The CLI is parity-blind by accident

Every CLI command in `aragora/cli/` is structurally an agent-readable surface — it returns JSON when `--json` is set, it speaks to the same HTTP endpoints as the SDK and the live frontend. The `aragora marketplace ...`, `aragora billing ...`, `aragora reputation ...`, `aragora receipts ...`, `aragora agents ...` families are already at parity *for capability*; they just don't sign their output, don't expose Ed25519 verification, and don't return AgentReceipts.

The implication for the P8 sequencing: the CLI does not need a separate parity pass. Once the HTTP endpoints proposed in this audit exist, the CLI gets parity by adding `--agent-form` flags that swap the response shape from human-format to AgentReceipt or principal-resolver shape. ~30 LOC per command, deferred to after the HTTP work.

### 4.7 The AGT-02 sequencing in `AGENT_CONSUMER_SURFACE.md` is the right ordering, with one swap

The plan's sequencing is:

| Step | Plan deliverable |
|---|---|
| 1 | Agent-readable receipt schema spec |
| 2 | A2A registration endpoint + identity receipt emission |
| 3 | A2A capability discovery endpoint |
| 4 | Compute-budget billing with chargeback receipts |
| 5 | Reputation read endpoint (read-only) |
| 6 | Operator-parity surfaces |

Step 1 (schema spec) is **already in the repo** at `aragora/protocols/a2a/receipts.py`. It is contract-only; the schema landed but the wire endpoint did not.

This audit recommends a slightly different bounded ordering for the next P8 deliverables, leading with what's left of step 1:

1. **AgentReceipt wire endpoint** (`GET /api/v2/receipts/{id}/agent-form` + verify) — the actual first-deliverable proposal in §3 surface 4. ~250 LOC.
2. **Principal resolver** (`GET /api/v1/principals/{id}`) — the bounded surrogate for surface 1. ~250 LOC.
3. **Capability+reputation join** (`GET /api/v1/a2a/agents/{id}/capabilities`) — surface 2 first deliverable. ~200 LOC.
4. **Compute-budget read** (`GET /api/v1/a2a/agents/{id}/budget`) — surface 3 first deliverable. ~150 LOC.
5. **Reputation-source join** (`GET /api/v1/a2a/agents/{id}/reputation`) — surface 5 first deliverable. ~200 LOC.
6. **Marketplace mirror** (`GET /api/v1/a2a/marketplace/listings`) — surface 6 first deliverable. ~250 LOC.

Total: ~1,300 LOC across six PRs, each independently mergeable, each strictly read-only, each on a different handler module so merge conflicts are minimal. Identity write unification (the full surface 1) lands after all six in a separate plan.

---

## 5. Recommended P8 first deliverable

**The first P8 deliverable should be the AgentReceipt wire endpoint (§3 Surface 4).**

### Recap of the deliverable

- New endpoint: `GET /api/v2/receipts/{receipt_id}/agent-form` returning `AgentReceipt.to_json()`.
- New endpoint: `POST /api/v2/receipts/{receipt_id}/agent-form/verify` returning `{verified, receipt_id_match, signature_match, fresh, within_settlement}`.
- New MCP tool: `get_agent_receipt_tool(receipt_id)` wrapping the GET path.
- New file: `aragora/server/handlers/agent_receipts.py` (~150 LOC handler + glue).
- Edit: `aragora/mcp/tools_module/knowledge.py` to register the new tool (~30 LOC).
- Edit: `aragora/server/handler_registry/core.py` to register the new handler (~10 LOC).
- New tests: `tests/handlers/test_agent_receipts.py` (~80 LOC).

### Why this is correct

| Criterion | This deliverable |
|---|---|
| Closes highest-severity gap with smallest blast radius | Yes — surface 4, ~250 LOC, one new module. |
| Uses existing primitives, no new dependencies | `AgentReceipt`, `ReceiptStore`, `cryptography`, MCP server — all in repo. |
| Pure read; safely additive; reversible | Yes — no schema migration, no live human-surface change. |
| Demoable end-to-end | "Fetch agent-form receipt, verify with public key, parse CruxSet" — <100 lines of client. |
| Unblocks at least three other gaps | Receipts as chargeback shape (3), reputation-delta wire format (5), agent-bid acceptance proof (6). |
| Aligned with `AGENT_CONSUMER_SURFACE.md §S4` and AGT-02 step 1 | Yes — completes the half of step 1 that was contract-only. |
| Does not require chain configuration | Correct — Ed25519 keys live in env vars, no web3 connection needed. |
| Does not require AGT-05 to be enabled | Correct — `reputation_deltas_applied` is `()` when AGT-05 is off; receipt is still valid. |
| Activates dormant code | Yes — `aragora/protocols/a2a/receipts.py` becomes a live wire format instead of contract-only. |

### Acceptance criteria the audit recommends for the deliverable

1. An external agent can `GET` an agent-form receipt for any receipt that exists in the human surface, with no authentication change.
2. The returned JSON has `schema_version: "1.0"`, content-addressed `receipt_id` matching the canonical hash, Ed25519 signature, and dissent + reputation_deltas_applied fields populated when those data are present.
3. `POST .../verify` returns `verified: true` for an unmodified receipt and `verified: false` for any of: tampered subject, wrong public key, expired freshness SLA.
4. The MCP tool returns the same JSON envelope the HTTP endpoint returns.
5. The human endpoint `GET /api/v2/receipts/{id}` is unchanged in shape and behavior.
6. The new handler returns `503` (not `500`) when `AGT_RECEIPT_SIGNING_KEY` is absent, with a clear error body.
7. Test coverage: build → fetch → verify → modify → verify-fails round trip.

### Out of scope for the first deliverable

- Anchoring the AgentReceipt on-chain (deferred; uses `aragora/blockchain/receipt_anchor.py` later).
- Computing AGT-05 reputation deltas inside the receipt path (the AGT-05 store provides them; this deliverable just embeds them when present).
- Wiring receipt emission into the live debate finalize path (already happens; this deliverable adds an alternate read shape).
- Identity unification (surface 1; covered by the principal resolver as the next deliverable).
- Per-domain reputation slices (surface 5; deferred).
- Agent-side marketplace participation (surface 6; deferred).

### Risk register for the deliverable

| Risk | Mitigation |
|---|---|
| Ed25519 signing key leaks via env var | Store key in `AGT_RECEIPT_SIGNING_KEY` already in `.env` (gitignored); document; add a key rotation path in the follow-up PR. The risk is no worse than for any other server-side secret. |
| Receipt-data mismatch between human and agent forms | The agent form is *derived* from the human form, not stored separately; mismatch is impossible by construction. The signature hash binds the agent form to its inputs. |
| AgentReceipt schema becomes incompatible with the on-chain anchoring path | The schema is already specified as forward-compatible by major version (`AGENT_RECEIPT_SCHEMA_VERSION = "1.0"`). On-chain anchoring will only need a cid → tx_hash mapping, not a schema change. |
| Adoption: nobody calls the endpoint | The MCP tool wrapper means every MCP-speaking agent (Claude Desktop, Cursor, etc.) gets it for free. The CLI will get an `--agent-form` flag in a follow-up. |

---

## 6. Out of scope for this audit

This audit does **not** propose:

- Any code changes, refactors, or migrations.
- Identity unification beyond the principal resolver.
- A fiat billing path. `AGENT_CONSUMER_SURFACE.md` explicitly defers fiat to a later regulated lane.
- Wallet management UX. `aragora/blockchain/wallet.py` exists; the audit does not touch its surface.
- Sybil resistance policy. `AGENT_CONSUMER_SURFACE.md §Risks` keeps "keyed identity with manual approval for production reputation effects" as the policy. The audit endorses keeping that policy; it is orthogonal to the wire-format work.
- Onchain anchoring of receipts. Already specified at `aragora/blockchain/receipt_anchor.py`; out of scope for the first deliverable.
- Per-vertical reputation slices (legal, healthcare, financial, software). The verticals already have their own scoring (`aragora/verticals/`); P8 parity is a separate concern.
- Any expansion of the MCP tool inventory beyond a single wrapper for the recommended deliverable.
- Changes to the Nomic Loop or self-improvement orchestrator. P8 lives at the consumer-surface layer; the Nomic Loop lives below it.
- Deprecation of any human surface. Every existing human endpoint stays exactly as it is.
- Changes to the OpenAPI generation pipeline beyond adding the new endpoints to the manifest.

---

## 7. Citations

All `path:line` references below are pinned to the worktree at `/Users/armand/Development/aragora` on branch `docs/2026-04-28-overnight-planning` at the time of audit.

### Surface 1 — Registration / identity

- `aragora/server/handlers/auth/login.py:42` — `POST /api/auth/register` decorator.
- `aragora/server/handlers/auth/login.py:90` — `handle_register` implementation.
- `aragora/server/handlers/auth/login.py:203` — `POST /api/auth/login`.
- `aragora/server/handlers/auth/signup_handlers.py:11` — module docstring listing `POST /api/v1/auth/signup`.
- `aragora/server/handlers/auth/signup_handlers.py:158` — `handle_signup` implementation.
- `aragora/server/handlers/auth/handler.py:144` — `/api/auth/me` route registration.
- `aragora/server/handlers/auth/handler.py:763` — `_handle_get_me` implementation.
- `aragora/server/handlers/auth/api_keys.py:45` — `POST /api/auth/api-key`.
- `aragora/server/handlers/auth/api_keys.py:152` — `DELETE /api/auth/api-key`.
- `aragora/server/handlers/auth/api_keys.py:218` — `GET /api/auth/api-keys`.
- `aragora/server/handlers/erc8004.py:559` — `handle_register_agent` (`POST /api/v1/blockchain/agents`).
- `aragora/server/handlers/erc8004.py:626` — `BLOCKCHAIN_HANDLERS` dict.
- `aragora/blockchain/contracts/identity.py` — `IdentityRegistryContract` (ERC-8004 identity).
- `aragora/mcp/tools_module/control_plane.py` — `register_agent_tool` (control-plane registry, not tenant identity).

### Surface 2 — Capability discovery

- `aragora/server/handlers/agents/agents.py:390` — `GET /api/v1/agents` (`_list_agents`).
- `aragora/server/handlers/agents/agents.py:450` — `GET /api/v1/agents/local`.
- `aragora/server/handlers/agents/agents.py:487` — `GET /api/v1/agents/local/status`.
- `aragora/server/handlers/agents/agents.py:531` — `GET /api/v1/agents/health`.
- `aragora/server/handlers/agents/agents.py:700` — `GET /api/v1/agents/availability`.
- `aragora/server/handlers/marketplace_pilot.py:116` — read decorator stack (`@require_permission("marketplace:read")` + `@rate_limit`).
- `aragora/server/handlers/marketplace_pilot.py:118` — `MarketplacePilotHandler.handle` (read).
- `aragora/server/handlers/marketplace_pilot.py:147` — write decorator stack (`@handle_errors` + `@require_permission("marketplace:write")` + `@rate_limit`).
- `aragora/server/handlers/marketplace_pilot.py:150` — `MarketplacePilotHandler.handle_post` (write).
- `aragora/server/handlers/a2a.py:350-353` — A2A discovery routes.
- `aragora/server/handlers/a2a.py:359` — `/api/v1/a2a/openapi.json`.
- `aragora/mcp/tools_module/agent.py:16` — `list_agents_tool`.
- `aragora/mcp/tools_module/control_plane.py` — `list_registered_agents_tool`.
- `aragora/mcp/tools_module/workflow.py` — `list_workflow_templates_tool`.
- `aragora/mcp/server.py:1` — `AragoraMCPServer` (the MCP discovery surface itself).
- `aragora/mcp/tools_module/`  (directory containing ~70+ MCP tools).

### Surface 3 — Billing / metering

- `aragora/server/handlers/usage_metering.py:5-7` — module docstring listing usage routes.
- `aragora/server/handlers/usage_metering.py:36` — `UsageMeteringHandler` class.
- `aragora/server/handlers/usage_metering.py:49-58` — `ROUTES` registration.
- `aragora/server/handlers/usage_metering.py:92` — `GET /api/v1/billing/usage` dispatch.
- `aragora/server/handlers/billing/core.py` — billing subscription / checkout / portal.
- `aragora/server/handlers/billing/cost_dashboard.py` — `GET /api/v1/billing/dashboard`.
- `aragora/server/handlers/budgets.py:1` — `BudgetsHandler` module docstring with full route list.
- `aragora/blockchain/compute_budget.py:65` — `ComputeBudgetManager`.
- `aragora/billing/calibration_cost_bridge.py` — calibration→cost bridge (not an external surface).
- `aragora/cli/billing.py` — CLI mirror.
- `aragora/cli/commands/billing_ops.py` — billing ops CLI.

### Surface 4 — Decision receipts

- `aragora/server/handlers/receipts.py:12-25` — module docstring listing all receipt routes.
- `aragora/server/handlers/receipts.py:368` — `ROUTES` registration.
- `aragora/protocols/a2a/receipts.py:1` — module docstring ("contract-only; no live emitter").
- `aragora/protocols/a2a/receipts.py:32` — `AGENT_RECEIPT_SCHEMA_VERSION`.
- `aragora/protocols/a2a/receipts.py:166` — `DissentEntry` dataclass.
- `aragora/protocols/a2a/receipts.py:198` — `ReputationDelta` dataclass.
- `aragora/protocols/a2a/receipts.py:230` — `AgentReceipt` dataclass.
- `aragora/protocols/a2a/receipts.py:263` — `AgentReceipt.build()` classmethod.
- `aragora/mcp/tools_module/knowledge.py` — `get_decision_receipt_tool`, `verify_decision_receipt_tool`, `build_decision_integrity_tool`.
- `aragora/export/decision_receipt.py` — human receipt schema.
- `aragora/blockchain/receipt_anchor.py` — on-chain anchoring path (out of scope for the first deliverable).

### Surface 5 — Reputation reads / writes

- `sdk/python/aragora_sdk/namespaces/reputation.py:36-85` — `ReputationAPI` (sync).
- `sdk/python/aragora_sdk/namespaces/reputation.py:99-135` — `AsyncReputationAPI`.
- `aragora/server/handlers/erc8004.py` — `handle_get_reputation` (`GET /api/v1/blockchain/agents/{id}/reputation`).
- `aragora/blockchain/contracts/reputation.py` — `ReputationRegistryContract` (ERC-8004).
- `aragora/reputation/__init__.py:1` — AGT-05 module docstring.
- `aragora/reputation/store.py:96` — `ReputationStore`.
- `aragora/reputation/types.py` — `StakeableClaim`, `ResolvedClaim`, `ReputationDelta`.
- `aragora/reputation/settlement.py` — `settle_claim` (the canonical write path).
- `aragora/reputation/selection_bridge.py:1` — `ReputationCalibrationBridge`.
- `aragora/reputation/anchor.py` — on-chain anchoring.
- `aragora/reputation/bridge.py` — `bridge_from_market_position`.
- `aragora/reputation/crux_bridge.py` — `bridge_from_crux_position`.

### Surface 6 — Marketplace participation

- `aragora/server/handlers/marketplace_pilot.py:14-25` — module docstring listing all listing routes.
- `aragora/server/handlers/marketplace_pilot.py:117` — read GET handler.
- `aragora/server/handlers/marketplace_pilot.py:148` — write POST handler.
- `aragora/server/handlers/marketplace.py:14-23` — older `/api/v1/marketplace/templates` routes.
- `aragora/marketplace/service.py` — `MarketplaceService` (`list_listings`, `install_listing`, `rate_listing`).
- `aragora/cli/marketplace.py:1` — CLI mirror.

### Cross-cutting

- `docs/CANONICAL_GOALS.md:197` — Pillar 8 north-star outcome.
- `docs/plans/AGENT_CIVILIZATION_SUBSTRATE.md` — strategic context for AGT-* deliverables.
- `docs/plans/AGENT_CONSUMER_SURFACE.md` — the canonical AGT-02 plan (the contract this audit checks parity against).
- `docs/plans/SKIN_IN_THE_GAME_REPUTATION.md` — the canonical AGT-05 plan (the truth-oracle backbone for reputation).
- `aragora/mcp/server.py:1` — MCP server contract.
- `aragora/cli/api_keys.py:1` — API key store (Keychain / file).

---

## 8. Decision summary

> **Most strategic gap:** decision receipts (surface 4). The AgentReceipt schema is implemented, signed, content-addressed, and verifiable; only the wire endpoint is missing. Closing it is a ~250 LOC change that immediately unblocks chargeback receipts (surface 3), reputation-delta wire format (surface 5), and agent-bid acceptance proof (surface 6), and is the only candidate among the six surfaces that satisfies "highest severity, smallest blast radius, cleanest demo, no chain dependency, no AGT-05 dependency, no schema migration." Recommend it as the next bounded P8 deliverable.
