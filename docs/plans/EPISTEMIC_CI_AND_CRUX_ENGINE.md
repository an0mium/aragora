# Epistemic CI and Crux Engine Plan

> **Status:** additive future Decision Integrity tranche
> **Created:** 2026-04-16
> **Queue policy:** planning issues only; do not add `boss-ready` until the proof-first Foreman gate permits this tranche

## Thesis

Aragora should not only help organizations make decisions or execute tasks. It should help them know what they believe, why they believe it, what evidence is fresh, which disagreements are load-bearing, and what work should happen when reality changes.

This plan combines two compatible product directions:

- **Crux Engine:** for any contested question, surface the 3-5 load-bearing disagreements where a changed fact or framing would change the conclusion.
- **Epistemic CI:** convert important organizational claims into executable, evidence-linked objects that can pass, fail, go stale, or create bounded repair work.

The crux engine locates where reasonable agents diverge. Epistemic CI keeps claims and evidence from silently decaying. Together they form a decision-integrity layer on top of Aragora's debate, receipt, provenance, and proof-first runtime substrate.

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

The missing abstraction is not another generic agent loop. It is a first-class **claim/crux object model** that binds debate, evidence, receipts, memory, and bounded follow-up work.

## Non-Goals

- Do not replace the current proof-first autonomy wedge.
- Do not make broad external claims before the bounded execution substrate is stable.
- Do not create live `boss-ready` queue work from failed claims or unresolved cruxes until queue governance explicitly allows it.
- Do not build a new parallel reliability stack.
- Do not duplicate receipt, provenance, or knowledge-mound plumbing.

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

## Relationship To The Current Roadmap

This tranche belongs under **Decision Integrity Core**, not the immediate Reliability Substrate lane.

Near-term proof-first autonomy remains blocking. Epistemic CI should not become a reason to interrupt BC-12 soaks, B2 guard proof, or queue discipline. The first safe use is as an explicit, manually curated claim layer over already-published proof surfaces. Automatic extraction, failed-claim issue generation, and crux-driven work creation all come later.

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

## Quality Bar

The tranche is ready to enter active implementation only when:

- the proof-first loop has stable long-run evidence
- recurring proof surfaces stay fresh without babysitting
- queue governance can prevent claim/crux follow-up work from becoming generic churn
- receipt and Knowledge Mound paths can preserve provenance without schema drift
