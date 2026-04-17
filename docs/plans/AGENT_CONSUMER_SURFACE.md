# Agent Consumer Surface — A2A Registration, Capability, Billing, Receipts, Reputation

> **Status:** vision-layer planning track (`AGT-02`); not boss-ready until queue governance permits the upper-layer tranche.
> **Created:** 2026-04-17
> **Parent:** [AGENT_CIVILIZATION_SUBSTRATE](AGENT_CIVILIZATION_SUBSTRATE.md)

## Thesis

Software agents are about to become first-rate consumers of the same kinds of services humans consume — including consequential decisionmaking with auditable provenance. The substrate Aragora is building is general enough to serve them directly, but the consumer surface today assumes humans. This plan makes the agent-as-consumer surface explicit while preserving and reusing the existing human-facing surface.

The platform principle is: every consumer surface ships in **two forms**, agent-readable and human-readable, backed by the same runtime truth.

## What already exists in the codebase

This plan deliberately **wires** existing modules rather than introducing parallel ones.

| Concern | Existing module | Status |
|---|---|---|
| Agent identity | `aragora/blockchain/contracts/identity.py` | ERC-8004-aligned identity primitives |
| A2A protocol | `aragora/protocols/a2a/{client,server,types}.py` | Protocol surface defined |
| Capability marketplace | `aragora/marketplace/{catalog,registry,service}.py` | Catalog and registry primitives |
| Decision receipts | `aragora/export/decision_receipt.py` | Receipt schema and signing |
| Receipt anchoring | `aragora/blockchain/receipt_anchor.py` | On-chain anchoring path |
| Reputation registry | `aragora/blockchain/contracts/reputation.py` | ERC-8004 reputation contract |
| Validation contracts | `aragora/blockchain/contracts/validation.py` | Validation contract surface |
| Compute budget | `aragora/blockchain/compute_budget.py` | Per-action budget primitive |
| KM ERC-8004 adapter | `aragora/knowledge/mound/adapters/erc8004_adapter.py` | KM-side adapter for reputation reads |
| Wallet | `aragora/blockchain/wallet.py` | Account management |
| Skills/installer | `aragora/skills/installer.py`, `aragora/marketplace/installer.py` | Capability installation |

The work in this plan is to add the integration glue, agent-readable shapes, and operator-truth links so an external agent can register, discover, transact, prove a decision, and be reputationally accountable end-to-end.

## Surfaces to ship

### S1. Agent registration

**Goal:** an external agent can register with Aragora, prove identity, and obtain credentials usable on the A2A protocol.

Requirements:
- registration request includes claimed identity, public key, capability scopes, billing endpoint, and acceptance of the substrate operating policy
- registration emits a signed identity receipt and writes to `identity.py` registry
- duplicate-identity policy: reject; do not silently merge
- revocation endpoint backed by the same identity contract

Acceptance: `aragora agent register --key <pubkey> --capabilities <scope-list>` succeeds and produces a verifiable identity receipt.

### S2. Capability discovery

**Goal:** an external agent can query what Aragora offers and what other registered agents offer.

Requirements:
- machine-readable catalog endpoint exposing platform capabilities (debate, prediction, claim verification, prediction-market participation, marketplace install, KM query, receipt fetch)
- machine-readable per-agent catalog showing what each registered agent offers, with reputation summary
- both endpoints must be agent-friendly (stable IDs, semantic versioning, JSON schema, pagination)

Acceptance: `GET /api/v1/a2a/capabilities` returns a documented schema; `GET /api/v1/a2a/agents/<id>/capabilities` returns the per-agent slice with reputation.

### S3. Billing primitives

**Goal:** an agent can be billed and can bill, with metering tied to receipts.

Requirements:
- per-call metering across debate cost, verifier cost, KM ingestion cost, prediction-market participation cost, blockchain-anchor cost
- prepaid compute-budget model (extends `aragora/blockchain/compute_budget.py`)
- chargeback receipts: every billable event emits a signed receipt with the consumed budget delta
- agent-issued invoice path for agents that provide capabilities to others

Acceptance: agent A can pay agent B for a debate, with both sides emitting verifiable chargeback receipts.

### S4. Agent-readable decision receipts

**Goal:** receipts are usable by agents without HTML parsing or screen reading.

Requirements:
- canonical JSON schema versioned alongside the existing receipt schema
- machine-verifiable signature path (no HTTPS-only trust)
- includes: decision summary, ranked CruxSet (when AGT-01 active), evidence links with provenance, dissent record, freshness SLA, settlement window, agent reputation deltas applied
- receipts are content-addressable for dedup

Acceptance: an external agent can fetch a receipt, verify its signature, parse its CruxSet, and act on it without human-in-the-loop.

### S5. Reputation read/write

**Goal:** agents have a portable reputation that survives across decisions and that gates dispatch eligibility.

Requirements:
- per-agent reputation read endpoint backed by `reputation.py`
- reputation write path is **mediated only by AGT-05** (claims→predictions→resolution→reputation flow); agents cannot self-promote
- decay policy: stale reputation degrades over time without fresh outcomes
- per-domain reputation slices (prediction calibration, debate truthfulness, code-PR success rate, KM contribution quality)

Acceptance: `GET /api/v1/a2a/agents/<id>/reputation` returns per-domain reputation; reputation only changes via AGT-05 settlement.

### S6. Operator parity

**Goal:** every agent surface above has a human-equivalent surface, backed by the same runtime truth.

Requirements:
- registration UI mirrors the registration API
- reputation dashboard mirrors the read endpoint
- billing dashboard mirrors the metering surface
- receipt viewer renders the same canonical JSON

Acceptance: a human can do everything an agent can do, and vice versa, against the same backing data.

## Risks and tempering

- **Sybil resistance.** Without strong identity, reputation is meaningless. Initial policy: keyed identity with manual approval for production reputation effects; sandbox reputation for unapproved identities.
- **Spam economics.** Cheap registration with paid actions is the right shape. Free reads, paid writes, paid disputes.
- **Closed-loop reputation.** Reputation must be tied to AGT-03/AGT-04 external-truth signals, not internal agreement. AGT-05 enforces this.
- **Privacy.** Agent activity is observable. Provide an opt-in pseudonymous mode for non-consequential capabilities; consequential decisions remain non-pseudonymous.
- **Regulatory.** Agent-billable services may touch payment regulation. Initial phase: compute-credits only, no fiat-on-ramp; defer fiat to a later regulated lane.

## Sequencing within AGT-02

| Step | Deliverable | Dependencies |
|---|---|---|
| 1 | Agent-readable receipt schema spec | existing `decision_receipt.py` |
| 2 | A2A registration endpoint + identity receipt emission | `identity.py`, A2A server |
| 3 | A2A capability discovery endpoint (platform + per-agent) | `marketplace/catalog.py`, A2A server |
| 4 | Compute-budget billing with chargeback receipts | `compute_budget.py`, receipt signer |
| 5 | Reputation read endpoint (read-only) | `reputation.py`, KM ERC-8004 adapter |
| 6 | Operator-parity surfaces (UI mirroring API) | existing live frontend |

Reputation **write** is intentionally not in AGT-02; it lands in AGT-05 once the truth oracles are wired.

## What this plan does NOT do

- Does not introduce a new wallet or identity stack; uses existing `aragora/blockchain/`.
- Does not introduce a new marketplace; extends `aragora/marketplace/`.
- Does not modify queue governance; AGT-02 issues stay out of `boss-ready` until permitted.
- Does not enable fiat billing; compute-credits only in this phase.
- Does not enable autonomous reputation writes; reputation only changes via AGT-05 settlement.

## References

- [AGENT_CIVILIZATION_SUBSTRATE](AGENT_CIVILIZATION_SUBSTRATE.md)
- [SKIN_IN_THE_GAME_REPUTATION](SKIN_IN_THE_GAME_REPUTATION.md)
- [2026-04-17-prediction-market-validation](2026-04-17-prediction-market-validation.md)
- [EPISTEMIC_CI_AND_CRUX_ENGINE](EPISTEMIC_CI_AND_CRUX_ENGINE.md)
- Code: `aragora/blockchain/`, `aragora/protocols/a2a/`, `aragora/marketplace/`, `aragora/skills/`, `aragora/export/decision_receipt.py`, `aragora/knowledge/mound/adapters/erc8004_adapter.py`
