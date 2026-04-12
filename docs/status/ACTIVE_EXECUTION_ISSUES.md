# Active Execution Issues

Last updated: 2026-04-12

This document is the canonical epic, milestone, and execution-issue tree for the current roadmap tranche.

Use it to create or reconcile GitHub issues. Keep issue titles stable when possible so docs, dashboards, and GitHub can stay aligned.

## Priority Order

1. **Reliability Substrate**
2. **Bounded Autonomy Control Plane**
3. **Trust-Wedge Product Loops**
4. **Unified DAG Workbench** (parallel second track)
5. **Memory & Context Fabric**
6. **Decision Integrity Core**
7. **Commercialization & Scale-Out**

## GitHub Issue Links

The strict status reconciler requires this canonical map to link the live execution backlog directly.

- Enterprise assurance carryover: [#273](https://github.com/synaptent/aragora/issues/273), [#274](https://github.com/synaptent/aragora/issues/274), [#509](https://github.com/synaptent/aragora/issues/509)
- Current execution epics: [#804](https://github.com/synaptent/aragora/issues/804), [#805](https://github.com/synaptent/aragora/issues/805), [#806](https://github.com/synaptent/aragora/issues/806)
- Current execution lanes: [#807](https://github.com/synaptent/aragora/issues/807), [#808](https://github.com/synaptent/aragora/issues/808), [#809](https://github.com/synaptent/aragora/issues/809), [#810](https://github.com/synaptent/aragora/issues/810), [#811](https://github.com/synaptent/aragora/issues/811), [#812](https://github.com/synaptent/aragora/issues/812), [#813](https://github.com/synaptent/aragora/issues/813), [#814](https://github.com/synaptent/aragora/issues/814), [#815](https://github.com/synaptent/aragora/issues/815), [#816](https://github.com/synaptent/aragora/issues/816), [#817](https://github.com/synaptent/aragora/issues/817), [#818](https://github.com/synaptent/aragora/issues/818), [#819](https://github.com/synaptent/aragora/issues/819), [#820](https://github.com/synaptent/aragora/issues/820)

## Reverse-Staged Rocket Bootstrap

This is the near-term sequencing layer across the full task tree. Do not treat all `[30d]` work as equally active at once.

| Booster | Immediate aim | Main issue groups |
|---|---|---|
| **B0 — Corpus** | Prove `>=50%` no-rescue success on a fixed benchmark corpus | `RS-01..03`, `TW-02..03` |
| **B1 — Assist** | Auto-draft safe work orders and validation plans | `BC-04..06`, `TW-07..09` |
| **B2 — Guard** | Require contracts and production-like preflight before auto-run | `RS-04..09` |
| **B3 — Repair** | Productize retry, salvage, quarantine, and session reuse | `BC-01..03`, `RS-10..11` |
| **B4 — Multi** | Extend proven loops across hosts with truthful state | `BC-07..12`, `UDW-01..03` |

## Epic 1 — Reliability Substrate

**Outcome:** make bounded unattended execution trustworthy on real multi-host backlogs.

### Milestone 1.1 — Failure Taxonomy & Benchmark Corpus `[30d]`

- [x] **RS-01** Define canonical terminal-truth classes for auth, publication, validation, runtime, and task-shape failures
- [x] **RS-02** Harvest benchmark fixtures from real `needs_human` and publication-failure receipts
- [x] **RS-03** Add a benchmark scoring lane and regression guardrails in CI

### Milestone 1.2 — Worker Contracts & Credential Envelopes `[30-90d]`

- [x] **RS-04** Introduce persisted `WorkerContract` objects with checksum and admission rules
- [x] **RS-05** Introduce `CredentialEnvelope` slices for runner, git, GitHub API, provider, and verification auth
- [ ] **RS-06** Require launcher, supervisor, and tranche queue to dispatch only from complete contracts

### Milestone 1.3 — Contract-Aware Preflight `[30-90d]`

- [ ] **RS-07** Build `aragora swarm preflight run --contract ...`
- [ ] **RS-08** Validate scratch read/write/commit/push/draft-PR flow through the production code path
- [ ] **RS-09** Replace shell-only host checks with receipt-backed preflight wrappers

### Milestone 1.4 — Ledger & Self-Heal `[90d]`

- [ ] **RS-10** Mirror probes, queue state, contracts, and receipts into `AutonomyLedger`
- [ ] **RS-11** Add quarantine and fallback rules for auth drift, rate limits, permission mismatch, and publication failures
- [ ] **RS-12** Cut `studio-health`, reporter, and status surfaces to ledger-backed truth

## Epic 2 — Bounded Autonomy Control Plane

**Outcome:** make live autonomy sessions inspectable, resumable, and operable.

### Milestone 2.1 — Interactive Sessions & Repair Journal `[90d]`

- [ ] **BC-01** Persist session state across `explore -> plan -> edit -> verify -> repair -> publish`
- [ ] **BC-02** Resume retries from prior session state instead of fresh prompts
- [ ] **BC-03** Emit precise blocker evidence and repair transcripts for failed runs

### Milestone 2.2 — Task Sanitizer & Admission Gate `[30-90d]`

- [x] **BC-04** Add sanitizer outcomes: accepted, rewritten, dropped, quarantined
- [x] **BC-05** Detect truncated, contradictory, or impossible tasks before dispatch
- [ ] **BC-06** Preserve original versus sanitized task text for audit

### Milestone 2.3 — Truthful Lane / Integrator State `[90d]`

- [ ] **BC-07** Unify lane, host, runner, and publication state into one operator model
- [ ] **BC-08** Add pause, resume, retry, salvage, and quarantine controls against live state
- [ ] **BC-09** Produce one integrator view for contracts, receipts, blockers, and merge readiness

### Milestone 2.4 — Multi-Host Soak & Nomic on Substrate `[90d]`

- [ ] **BC-10** Run repeated multi-host soak tests on the bounded backlog
- [ ] **BC-11** Route Nomic-generated work through the same substrate as operator work
- [ ] **BC-12** Define explicit stop/go criteria for unattended 12-hour runs

## Epic 3 — Trust-Wedge Product Loops

**Outcome:** prove that real customers get value from the substrate now.

### Milestone 3.1 — Autonomous Software Execution Benchmark `[30d]`

- [ ] **TW-01** Prove `prompt -> spec -> code -> verify -> PR` loops on a fixed benchmark corpus of bounded repos/issues
- [ ] **TW-02** Measure rescue rate, verification pass rate, wall-clock throughput, and no-rescue completion rate
- [ ] **TW-03** Convert human rescues into benchmark fixtures and product requirements

### Milestone 3.2 — Inbox / Operator Action Loops `[30-90d]`

- [ ] **TW-04** Preserve receipt-before-action guarantees for inbox workflows
- [ ] **TW-05** Reuse contracts, memory, and approval policies across non-code actions
- [ ] **TW-06** Capture operator feedback tied to receipts

### Milestone 3.3 — Prompt-to-Spec Handoff `[30-90d]`

- [ ] **TW-07** Turn vague prompts into reviewable specs with explicit constraints and evals
- [ ] **TW-08** Hand approved specs into debate and execution without manual rewrite
- [ ] **TW-09** Surface missing context and weak acceptance criteria before work begins

### Milestone 3.4 — Design-Partner Recurrence `[90-365d]`

- [ ] **TW-10** Establish weekly design-partner operating cadence on real workloads
- [ ] **TW-11** Publish truthful proof packs and before/after benchmarks
- [ ] **TW-12** Decide which wedges graduate to packaged offerings

## Epic 4 — Unified DAG Workbench

**Outcome:** ship the GUI as a truthful control plane, not a mock visualization.

### Milestone 4.1 — Canonical Graph Model `[30-90d]`

- [ ] **UDW-01** Unify ideas, goals, actions, orchestration, and receipt node schemas
- [ ] **UDW-02** Map contracts, approvals, and provenance links onto the graph model
- [ ] **UDW-03** Define graph APIs backed by live runtime and ledger state

### Milestone 4.2 — Reviewable Stage Transitions `[90d]`

- [ ] **UDW-04** Show prompt -> spec -> work-order -> receipt transitions explicitly
- [ ] **UDW-05** Allow human approval, rejection, and replan at stage boundaries
- [ ] **UDW-06** Surface dissent, evidence, and risk at each handoff

### Milestone 4.3 — Operator Workbench `[90-365d]`

- [ ] **UDW-07** Add lane board, run replay, and intervention controls
- [ ] **UDW-08** Add DAG comparison and branching views
- [ ] **UDW-09** Add receipt-linked run history and audit navigation

### Milestone 4.4 — Full Idea-to-Execution Canvas `[365d]`

- [ ] **UDW-10** Ship editable ideas -> goals -> actions -> orchestration canvas
- [ ] **UDW-11** Add live collaboration and role-specific views
- [ ] **UDW-12** Add cross-project portfolio view with DAG-level dependencies

## Epic 5 — Memory & Context Fabric

**Outcome:** make shared knowledge a trustworthy execution advantage.

### Milestone 5.1 — Permissioned Memory Model `[30-90d]`

- [ ] **MCF-01** Add trust tiers, provenance, and access boundaries to memory items
- [ ] **MCF-02** Carry taint and provenance annotations from retrieved context into specs and receipts
- [ ] **MCF-03** Define per-workflow memory policies for software, inbox, policy, and research lanes

### Milestone 5.2 — Large-Context Packing `[90d]`

- [ ] **MCF-04** Build relevance-ranked context packing for big repos and mixed sources
- [ ] **MCF-05** Track context budgets, truncation, and evidence coverage
- [ ] **MCF-06** Benchmark quality lift from large-context packing versus baseline

### Milestone 5.3 — Broad Ingestion & Normalization `[90-365d]`

- [ ] **MCF-07** Normalize ingestion across repos, docs, APIs, chat, and receipts
- [ ] **MCF-08** Build durable source adapters with provenance and permission metadata
- [ ] **MCF-09** Add gap detection for stale context and missing evidence

### Milestone 5.4 — Shared Knowledge Base `[365d]`

- [ ] **MCF-10** Tie cross-run learning loops to outcomes and receipts
- [ ] **MCF-11** Add org-level knowledge graph and retrieval analytics
- [ ] **MCF-12** Support memory export and portability for customer trust

## Epic 6 — Decision Integrity Core

**Outcome:** preserve Aragora's core trust differentiation.

### Milestone 6.1 — Debate Quality & Calibration `[30-90d]`

- [ ] **DIC-01** Improve truth weighting, dissent capture, and hollow-consensus detection
- [ ] **DIC-02** Expand benchmark coverage across decision and execution workflows
- [ ] **DIC-03** Make debate quality signals visible in receipts and operator UI

### Milestone 6.2 — External Verification & Policy Gates `[90d]`

- [ ] **DIC-04** Require external verifiers for high-impact decisions where policy demands it
- [ ] **DIC-05** Tie policy gates to execution permissions and approval flows
- [ ] **DIC-06** Fail closed when verification requirements are unmet

### Milestone 6.3 — Receipt Chain & Compliance Bundles `[90-365d]`

- [ ] **DIC-07** Strengthen cryptographic receipt envelopes and provenance links
- [ ] **DIC-08** Extend compliance artifact bundles for regulated workflows
- [ ] **DIC-09** Add settlement and review loops for long-horizon outcomes

### Milestone 6.4 — Explainability & Comparison `[365d]`

- [ ] **DIC-10** Add side-by-side debate and execution comparison surfaces
- [ ] **DIC-11** Show idea -> receipt -> later settlement lineage end-to-end
- [ ] **DIC-12** Produce board- and regulator-ready exports with dissent summaries

## Epic 7 — Commercialization & Scale-Out

**Outcome:** translate proof into a durable market position without overclaiming.

### Milestone 7.1 — Packaging & Narrative `[30-90d]`

- [ ] **CS-01** Align roadmap, goals, and commercial docs to one wedge-first story
- [ ] **CS-02** Define packaging for tool, teammate, foreman, and substrate stages
- [ ] **CS-03** Keep all external claims gated by measured proof

### Milestone 7.2 — SMB Operating System `[90-365d]`

- [ ] **CS-04** Productize plan, review, and execute loops for small teams with limited time
- [ ] **CS-05** Add role-based surfaces for founders, operators, and engineering leads
- [ ] **CS-06** Define pricing and onboarding around time-to-trust rather than token volume

### Milestone 7.3 — Vertical & Enterprise Proof `[365d]`

- [ ] **CS-07** Package regulated-workflow receipts and deployment modes
- [ ] **CS-08** Build healthcare, finance, legal, and government proof packs
- [ ] **CS-09** Sequence enterprise assurance after wedge proof rather than before it

### Milestone 7.4 — Platform Expansion `[365d]`

- [ ] **CS-10** Expand from software execution into broader cross-functional workflows
- [ ] **CS-11** Support heterogeneous external agents on shared knowledge and receipts
- [ ] **CS-12** Define federation and portability boundaries for the platform era

## Issue-Creation Rules

- Create issues narrowly enough that one bounded run can prove success or failure.
- Every issue should map to a stage, horizon, and proof artifact.
- Close issues with receipts, benchmark movement, or explicit proof — not optimism.
- Reopen or split issues when a “fix” only papers over the true blocker.
