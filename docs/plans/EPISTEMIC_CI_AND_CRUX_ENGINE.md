# Epistemic CI, Crux Engine, and Epistemic Runtime Plan

> **Status:** additive future Decision Integrity tranche
> **Created:** 2026-04-16
> **Queue policy:** planning issues only; do not add `boss-ready` until the proof-first Foreman gate permits this tranche

## Thesis

Aragora should not only help organizations make decisions or execute tasks. It should help them know what they believe, why they believe it, what evidence is fresh, which disagreements are load-bearing, and what work should happen when reality changes.

This plan combines three compatible product directions:

- **Crux Engine:** for any contested question, surface the 3-5 load-bearing disagreements where a changed fact or framing would change the conclusion.
- **Epistemic CI:** convert important organizational claims into executable, evidence-linked objects that can pass, fail, go stale, or create bounded repair work.
- **Epistemic Runtime:** treat code as a living argument by attaching assumptions, debate receipts, evidence, and verifier contracts to code units, then detecting epistemic decay when reality changes.

The crux engine locates where reasonable agents diverge. Epistemic CI keeps claims and evidence from silently decaying. The Epistemic Runtime extends that discipline to code paths: if the assumptions behind a function, route, script, or policy surface decay, Aragora should know, fail safely, and produce bounded repair work with new receipts. Together they form a decision-integrity layer on top of Aragora's debate, receipt, provenance, and proof-first runtime substrate.

## Why This Belongs In Aragora

Aragora already has most of the necessary primitives:

- `DebateProtocol` and Arena consensus paths for multi-agent disagreement and synthesis.
- `aragora/debate/convergence.py` for agreement detection that can be inverted into divergence scoring.
- `aragora/interrogation/` for prioritized question generation from ambiguous or underspecified prompts.
- `aragora/explainability/builder.py` for counterfactual analysis that can test whether a proposed crux is load-bearing.
- `aragora/export/decision_receipt.py` for verified and unverified claim fields.
- `aragora/gauntlet/` receipt models, signing, storage, and export paths.
- `aragora/knowledge/mound/adapters/receipt_adapter.py` for ingesting receipt-derived claims into memory.
- Provenance tests and claim-evidence citation machinery.
- Proof-first queue governance and `ShiftLedger` patterns for turning failures into bounded work without broad queue farming.

The missing abstraction is not another generic agent loop. It is a first-class **claim/crux/proof-carrying-code object model** that binds debate, evidence, receipts, memory, runtime safety, and bounded follow-up work.

## Non-Goals

- Do not replace the current proof-first autonomy wedge.
- Do not make broad external claims before the bounded execution substrate is stable.
- Do not create live `boss-ready` queue work from failed claims or unresolved cruxes until queue governance explicitly allows it.
- Do not build a new parallel reliability stack.
- Do not duplicate receipt, provenance, or knowledge-mound plumbing.
- Do not introduce unsupervised production hot-swapping. The first runtime shape is report-only, then guarded fallback/quarantine, then shadow replacement or pull-request generation. Any live hot-swap behavior must remain opt-in, receipt-backed, and limited to explicitly allowlisted demo or pure-policy surfaces until a later safety gate permits more.

## Core Contracts

### Executable Claim

An executable claim is a versioned statement with evidence and a verification contract.

```yaml
claim_id: b0.truth.success_rate
statement: "The benchmark truth surface is complete and fresh on current main."
owner: proof-first-runtime
scope: repo
confidence: high
evidence:
  - path: docs/status/B0_BENCHMARK_TRUTH_STATUS.md
  - workflow: Benchmark Truth Publication
freshness_sla_hours: 24
verification:
  kind: command
  command: python3 scripts/build_benchmark_truth_artifact.py --check
failure:
  severity: blocking
  allowed_action: report_only
receipts:
  - type: benchmark_truth_publication
```

Initial statuses should be deliberately boring: `pass`, `fail`, `stale`, `unsupported`, and `error`.

### Crux

A crux is a load-bearing disagreement. It is not merely a dissenting opinion; it is a candidate fact, framing, value, or constraint where flipping the answer would plausibly flip the decision.

```yaml
crux_id: crux.guard.expansion.requires.green_soaks
question: "Should Aragora widen B2 guard expansion now?"
statement: "Three consecutive green 12-hour soaks are required before widening B2."
positions:
  - side: require_soaks
    agents: [codex]
  - side: accept_shorter_productive_soak
    agents: [claude]
load_bearing_score: 0.86
counterfactual: "If one productive 12-hour soak is accepted as sufficient, the next action changes from soak hardening to B2 issue creation."
evidence_gaps:
  - "No settled policy on productive-soak versus empty-queue-soak equivalence."
candidate_verifier: docs/status/BC12_SOAK_POLICY.md
```

### CruxSet

A CruxSet is the signed debate output:

- question
- decision or candidate decision
- ranked cruxes
- opposing positions
- evidence gaps
- counterfactual notes
- verifier candidates
- receipt and provenance links

### Proof-Carrying Code Unit

A proof-carrying code unit is a code path with an attached argument for why it is safe, useful, and still true.

```yaml
code_unit_id: proof_first.shift.green_criteria
symbol: scripts.run_proof_first_shift.evaluate_green_shift
source_path: scripts/run_proof_first_shift.py
owner: proof-first-runtime
decision_receipts:
  - receipt_id: decision.bc12.green_shift_criteria
claims:
  - claim_id: bc12.benchmark_surface_fresh
  - claim_id: bc12.queue_canonical_or_empty
assumptions:
  - "Benchmark freshness can be determined from the published proof surface."
  - "Queue can be considered intentionally idle only when no canonical work is eligible."
verifiers:
  - kind: command
    command: python3 -m pytest tests/scripts/test_run_proof_first_shift.py -q
freshness_sla_hours: 24
decay_policy:
  failed_claim: repair_required
  stale_evidence: report_only
fallback_policy:
  default: fail_closed
  operator_message: "Green-shift criteria are stale or unsupported; do not widen guard scope."
```

The initial proof-carrying-code layer should be manifest-based and read-only. It attaches receipts, claims, and verifiers to code; it does not rewrite code at runtime.

### Epistemic Decay Signal

An epistemic decay signal is the runtime or CI observation that a code unit's proof is weakening.

```yaml
decay_id: decay.proof_first.shift.green_criteria.2026-04-16
code_unit_id: proof_first.shift.green_criteria
integrity_score: 0.42
reasons:
  - failed_claim: bc12.benchmark_surface_fresh
  - unresolved_crux: crux.soak_equivalence
recommended_action: repair_required
quarantine_policy: report_only
evidence:
  - docs/status/B0_BENCHMARK_TRUTH_STATUS.md
  - docs/status/NEXT_STEPS_CANONICAL.md
receipt_required: true
```

Decay signals are deliberately narrower than generic alerts: they explain which assumption, claim, receipt, verifier, or unresolved crux weakened a code path's justification.

## Implementation Sequence

### DIC-13: Executable Claim Manifest

Issue: [#6023](https://github.com/synaptent/aragora/issues/6023)

Define the first manifest contract for explicit repo claims. Start with manually curated claims from recurring proof surfaces, not automatic extraction.

Acceptance shape:

- documented schema
- examples under a stable docs/status path
- fields for owner, evidence, freshness, verifier, failure action, and receipt links
- no queue mutation

### DIC-14: Claim Verification Runner

Issue: [#6024](https://github.com/synaptent/aragora/issues/6024)

Add a safe local runner that verifies explicit claim manifests and emits JSON. This is the Epistemic CI equivalent of a minimal test runner.

Acceptance shape:

- `pass`, `fail`, `stale`, `unsupported`, `error`
- no network requirement in focused tests
- JSON output suitable for status docs and receipts
- no issue creation yet

### DIC-15: CruxSet Contract And Consensus Mode

Issue: [#6025](https://github.com/synaptent/aragora/issues/6025)

Add a CruxSet contract and a non-default Arena path that produces ranked crux candidates from debate output.

Acceptance shape:

- deterministic mocked-agent tests
- no replacement of existing consensus modes
- load-bearing score and evidence-gap fields
- counterfactual hook points for later validation

### DIC-16: Receipt And Knowledge Mound Provenance

Issue: [#6026](https://github.com/synaptent/aragora/issues/6026)

Persist executable claims and CruxSet outputs through receipt and Knowledge Mound paths.

Acceptance shape:

- backwards-compatible receipt metadata
- claim/crux IDs linked to evidence and source receipt
- Knowledge Mound ingestion preserves verification status
- focused tests around receipt ingestion

### DIC-17: Failed-Claim / Open-Crux Follow-Up Bridge

Issue: [#6027](https://github.com/synaptent/aragora/issues/6027)

Turn failed claims or unresolved high-load-bearing cruxes into exactly one bounded follow-up issue, initially dry-run only.

Acceptance shape:

- one failure creates at most one bounded issue proposal
- generated issues do not receive `boss-ready` unless the current tranche permits them
- no broad restock behavior
- test coverage for queue-governance constraints

### DIC-18: Organizational Truth Map Report

Issue: [#6028](https://github.com/synaptent/aragora/issues/6028)

Create a read-only operator report showing current executable claims, evidence freshness, failed/stale claims, open cruxes, and linked receipts/issues.

Acceptance shape:

- claim ID, statement, owner, status, evidence age, verifier, follow-up link
- CruxSet summary fields when available
- read-only report first
- no queue mutation

### DIC-19: Proof-Carrying Code Unit Constraint Graph

Issue: [#6030](https://github.com/synaptent/aragora/issues/6030)

Define the manifest and constraint graph for code paths that carry their assumptions, evidence, receipts, freshness requirements, verifier commands, decay policy, and fallback policy.

Acceptance shape:

- schema for code unit ID, source path, symbol, owner, linked claims, assumptions, receipts, verifiers, freshness SLA, decay policy, and fallback policy
- at least one low-risk example tied to an existing proof-first code path
- compatibility with executable claim and CruxSet identifiers
- read-only behavior first; no runtime mutation

### DIC-20: Epistemic Decay Monitor

Issue: [#6031](https://github.com/synaptent/aragora/issues/6031)

Evaluate proof-carrying code units against claim verification results, freshness, receipt availability, unresolved cruxes, and synthetic world-state events.

Acceptance shape:

- machine-readable integrity score and decay reasons
- reason classes such as `stale_evidence`, `failed_claim`, `unresolved_crux`, `missing_receipt`, and `verifier_error`
- no queue mutation
- focused tests with synthetic dependency/API/CVE-like invalidation events

### DIC-21: Fail-Closed Quarantine Policy

Issue: [#6032](https://github.com/synaptent/aragora/issues/6032)

Define how proof decay maps to report-only, degrade, fallback, quarantine, or repair-required actions before any live behavior changes.

Acceptance shape:

- policy model keyed by decay severity and code-unit class
- receipt/provenance output for every non-report-only recommendation
- live routing or production hot-swap prohibited unless an explicit allowlist enables it
- ambiguous or over-budget decay fails closed

### DIC-22: Verified Replacement Pipeline

Issue: [#6033](https://github.com/synaptent/aragora/issues/6033)

Turn decayed proof into a bounded repair candidate through Arena debate, verifier hooks, receipt generation, and shadow or pull-request output.

Acceptance shape:

- decay signal produces a bounded repair spec with linked claims, cruxes, validation commands, and receipt context
- replacement debate captures a new decision or CruxSet receipt
- formal-verifier hook is available where constraints can be expressed
- default output is a patch, PR, or shadow candidate, not in-memory production hot-swap
- tests prove the no-hot-swap guardrail

## DIC-23..28: Dialectical Runtime Synthesis Layer (Planning Only)

DIC-13..22 ships the primitives. The additive synthesis layer DIC-23..28 ties those primitives into a single observable loop so code paths evolve as inspectable, receipt-carrying arguments rather than static text. Full design doc: [2026-04-18-dialectical-runtime-synthesis.md](2026-04-18-dialectical-runtime-synthesis.md). Epic: [#6223](https://github.com/synaptent/aragora/issues/6223).

This section is **strictly additive** to DIC-13..22. It does not reorder, re-scope, or replace any existing DIC item. It is planning truth only; no live queue scope; activation gate is the same as DIC-13..22 plus DIC-20/21/22 being production-green.

### DIC-23: Dialectical Runtime Loop Orchestrator

Issue: [#6217](https://github.com/synaptent/aragora/issues/6217)

Thin event-driven orchestrator joining decay signals, crux-finder debates, quarantine policy, and verified-replacement proposals into one chained receipt stream. Default is report-only telemetry; every escalation is flag-gated. Module: `aragora/epistemic/runtime_loop.py`.

### DIC-24: Epistemic Genealogy Ledger

Issue: [#6218](https://github.com/synaptent/aragora/issues/6218)

Read-only ancestry view over existing receipt, claim, and CruxSet stores — "why does this code unit look the way it does today?" Answers with the full decision → decay → crux → repair → new decision chain. Module: `aragora/epistemic/genealogy.py`.

### DIC-25: Adversarial World-State Stress-Test

Issue: [#6219](https://github.com/synaptent/aragora/issues/6219)

Operator-curated catalog of plausible-future perturbations (CVE drops, API rate-limit shifts, corpus revisions) that probe proof-carrying code units offline and report fragility deltas before reality invalidates them. Module: `aragora/epistemic/stress_test.py`.

### DIC-26: Belief Coherence Monitor

Issue: [#6220](https://github.com/synaptent/aragora/issues/6220)

Uses the existing `BeliefNetwork` to detect hard contradictions, evidence conflicts, and confidence rot across the organisation's claim ledger. Report-only; feeds DIC-17 bridge only when explicitly enabled. Module: `aragora/epistemic/coherence.py`.

### DIC-27: Operator Crux Arbitration

Issue: [#6221](https://github.com/synaptent/aragora/issues/6221)

First-class receipt type for human resolution of persistent cruxes — cruxes that stay load-bearing across N consecutive debates on the same question family. Arbitrations pin priors, carry expiry, and are reversible with their own receipts. CLI: `aragora crux arbitrate`. Module: `aragora/epistemic/arbitration.py`.

### DIC-28: Proactive Crux Gardening

Issue: [#6222](https://github.com/synaptent/aragora/issues/6222)

Scheduled re-examination of resolved and outstanding cruxes: is the evidence still fresh, have contradicting claims emerged, have fragility scores shifted? Report-only by default; optional DIC-17 feed. Module: `aragora/epistemic/gardening.py`.

## Relationship To The Current Roadmap

This tranche belongs under **Decision Integrity Core**, not the immediate Reliability Substrate lane.

Near-term proof-first autonomy remains blocking. Epistemic CI, Crux Engine, and Epistemic Runtime work should not become a reason to interrupt BC-12 soaks, B2 guard proof, or queue discipline. The first safe use is as an explicit, manually curated claim layer over already-published proof surfaces. Automatic extraction, failed-claim issue generation, crux-driven work creation, proof-carrying code enforcement, quarantine, and verified replacement all come later.

## First Dogfood Targets

Good initial claims:

- benchmark truth publication is fresh and complete on current `main`
- `TW03_RESCUE_PRODUCTIZATION_STATUS.md` links repeated rescue classes to fixtures or issues
- live queue contains no `boss-ready` work outside the allowed tranche
- BC-12 green-shift criteria are emitted by proof-first shifts
- external docs remain narrower than measured proof

Good initial crux questions:

- whether one productive 12-hour soak is equivalent to one empty-queue green soak
- whether B2 guard expansion may begin before three consecutive green shifts
- whether failed claim reports should create issues automatically or remain operator-only
- whether CruxSet belongs as a consensus mode or a standalone subsystem
- whether dialectical runtime events should ever mutate live queue state or remain report-only indefinitely (DIC-23)
- whether operator arbitrations should carry a hard expiry or a periodic revalidation (DIC-27)

Good initial proof-carrying code units:

- proof-first green-shift criteria in `scripts/run_proof_first_shift.py`
- benchmark truth publication checks that gate proof-first expansion
- queue-governance rules that prevent non-canonical `boss-ready` restock
- status-doc reconciliation paths that prevent external claims from exceeding proof

## Quality Bar

The tranche is ready to enter active implementation only when:

- the proof-first loop has stable long-run evidence
- recurring proof surfaces stay fresh without babysitting
- queue governance can prevent claim/crux follow-up work from becoming generic churn
- receipt and Knowledge Mound paths can preserve provenance without schema drift
- runtime safety policy can distinguish report-only decay from fallback, quarantine, repair-required, and any future opt-in hot-swap class
