# Aragora Roadmap Revision: Local Advocates as an Augmentation Layer (Draft v0.1)

**Status:** draft, pending operator settlement
**Owner:** Armand
**Date:** 2026-05-22
**Related:** `docs/specs/LOCAL_ADVOCATE_TRAINING_PIPELINE.md`,
`scripts/aft_extract_training_data.py`, `scripts/aft_harness.py`,
`memory/feedback_substrate_freeze_external_proof.md`,
`memory/feedback_use_real_intelligence.md`,
`memory/feedback_pr_triage_policy.md`,
`docs/REVIEW_AUTHORITY_PRINCIPLES.md`

## Why this exists

The May 21–22 codex/claude debate produced a proposal that we re-interpreted
several times before settling. The proposal is:

> One core primitive within Aragora could be an ensemble of small, fast,
> locally runnable, locally finetunable open-weight models that serve as
> proxies and advocates for the interests of a human, in lieu of somewhat
> dumber specialized classifiers, when interacting with API-based frontier
> models.

This document is the **non-pivoting** version of that proposal. It positions
local advocates as a new layer in the existing architecture, not a
replacement for any of the five pillars or the existing debate/receipt
substrate. The intent is to clarify how the layer fits, what it is allowed
to do, what it must not do, and which existing roadmap items it changes,
adds to, or leaves alone.

Two prior memories anchor the framing:

- `memory/feedback_use_real_intelligence.md` — use frontier LLMs for all
  classification/routing/disambiguation work that currently relies on
  regex/heuristics. Advocates do **not** weaken that rule; they extend it
  by adding *operator-specific* intelligence at a lower price point for a
  narrow class of routine decisions.
- `memory/feedback_substrate_freeze_external_proof.md` — when an agent
  loop is "producing more loop", redirect to existing benchmarks and ship
  one external-proof artifact. The advocate layer must be evaluated under
  the same discipline: one Advocate Feasibility Test, one falsifiable
  result, one decision about whether to expand.

## Where advocates fit

```text
+--------------------------------------------------------------+
|  Operator surface (CLI, Inbox, Live, Channels, Bots, Spec)   |
+--------------------------------------------------------------+
                          |
                          v
+--------------------------------------------------------------+
|  Advocate layer (NEW, optional, per-operator, per-task)      |
|  - PR triage advocate                                        |
|  - Inbox triage advocate (v0.2)                              |
|  - Calendar / approvals advocate (v0.3, only if v0.1 lands)  |
|                                                              |
|  Roles per advocate: propose, abstain, escalate, audit       |
+--------------------------------------------------------------+
                          |
        (propose+confidence; escalate when below threshold)
                          v
+--------------------------------------------------------------+
|  Existing Aragora debate substrate                           |
|  Arena → consensus → cross-verification → receipt → memory    |
|  (this is where Tier 3+ decisions still land, unchanged)     |
+--------------------------------------------------------------+
                          |
                          v
+--------------------------------------------------------------+
|  Receipt + governance + execution                            |
|  DecisionReceipt, RBAC, audit, EU AI Act bundle, settlement  |
+--------------------------------------------------------------+
```

The advocate layer sits *between* the operator surface and the debate
substrate. Its only job, in v0.1, is to *propose* a decision and a
confidence. The advocate never performs a side effect, never writes to a
receipt store, never approves anything on its own. When confidence is below
a per-operator threshold, the request continues into the existing debate
substrate exactly as it does today.

This is the central design rule. Advocates **augment**; they never bypass.

## What this is not

1. **Not a pivot.** No pillar moves. No subsystem is deprecated. The
   debate substrate, receipts, KM, RBAC, observability, EU AI Act bundle
   generator — all of these are unchanged by v0.1. The advocate layer is a
   new optional preamble, not a replacement.
2. **Not a frontier-replacement story.** The advocates handle a narrow
   slice of operator decisions that currently consume frontier tokens and
   produce a tiny amount of information per dollar. The frontier is still
   the technical reviewer for everything else.
3. **Not a "small model superiority" claim.** We expect local advocates to
   be *worse* on novel inputs and *cheaper* on routine ones. The roadmap
   change is only worthwhile if the cost-quality frontier shifts for
   bounded routine tasks.
4. **Not multi-tenant.** Each advocate is one operator's revealed-policy
   artifact. Cross-operator advocates require their own governance review
   and are explicitly deferred.

## What this is

1. A **per-operator, per-task augmentation layer** trained on the
   operator's own decision history.
2. A **cost-quality frontier shift** for routine bounded decisions that
   would otherwise burn frontier calls.
3. A **privacy primitive** for cases where the operator does not want the
   full content of a decision to leave their machine. The corpus extractor
   intentionally takes only low-information features so the trained policy
   does not memorize private content.
4. A **falsifiable experiment** with one pre-registered hypothesis and
   one refutation rule (see `scripts/aft_harness.py`).

## Roadmap changes

The following table lists every change to the existing roadmap. Items not
listed are unchanged.

| Roadmap item | Status before | Status after | Change |
|---|---|---|---|
| Frontier-debate substrate | active | active | None. |
| Decision receipt + EU AI Act bundle | active | active | None. |
| Inbox trust wedge | shipped, UI rejected | unchanged | Inbox advocate is a v0.2 candidate but does not change wedge mechanics. |
| Smart provider routing | active | active | Advocate is a *prefix* to routing, not a replacement. |
| Spec / interrogation pipeline | active | active | None. |
| Crux-finder mode (Apr 16) | P1 | P1 | None. |
| Agent-civilization substrate (Apr 17) | planning only | planning only | None. |
| Trust-compound plan (Apr 17) | active | active | None. |
| **Advocate Feasibility Test (v0.1)** | — | NEW, this PR | Land harness + extractor + spec as draft. No live wiring. |
| **PR triage advocate** | — | NEW, gated on AFT v0.1 result | If AFT v0.1 passes, train one adapter, wire as *proposal-only* preamble to the existing review-queue flow. |
| **Inbox triage advocate** | — | candidate v0.2 | Gated on PR triage advocate landing and on operator agreeing the inbox wedge is worth re-attacking. |
| **Cross-operator advocate registry** | — | deferred | Out of scope until we have at least two operators running v0.1 advocates for ≥30 days. |

## Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | Advocate learns *frontier* policy instead of *operator* policy. | Train on operator-curated labels only. The corpus extractor pulls from the operator's own merge/close decisions, not from frontier suggestions. Periodic re-extraction catches drift. |
| R2 | Advocate is over-confident and gets routed past the frontier on bad decisions. | Brier-score the advocate offline; require a calibration threshold before the runtime layer trusts the confidence; abstain by default below the threshold. |
| R3 | Advocate is wired into a path it should not be wired into. | Tier 3 (proposal only) and Tier 4 (any path that *initiates* writes) gates per `docs/REVIEW_AUTHORITY_PRINCIPLES.md`. |
| R4 | Privacy regression: training data leaks operator-private content via the adapter. | Extractor pulls only low-information features (no diffs, no comment bodies). LoRA capacity is bounded. Operator policy artifact stays local. |
| R5 | Roadmap distraction: advocate work eats engineering time that should go to the frontier substrate. | AFT v0.1 is bounded scope (one extractor, one harness, two spec docs). v0.2 only opens on a positive v0.1 result with operator settlement. |
| R6 | Single-operator advocates are not generalizable, so the cost-quality argument does not compound. | True, and that is the point. We are not claiming generalization. We are claiming per-operator amortization across many routine decisions, which is the only regime where a small finetuned model beats a frontier call. |
| R7 | Tinker availability shifts; hosted training disappears. | The pipeline targets local MLX as the reference path. Tinker is a *control* used to bound the data-vs-method question. Loss of Tinker delays the calibration check but not the path. |
| R8 | The advocate is right *on average* but spectacularly wrong on edge cases that matter (e.g. it auto-routes a Tier 4 PR as `merged_fast`). | Tier hint in the rationale seeds biases the advocate toward `open_aged` for governance-adjacent paths; harness reports per-class precision so the operator sees tier-specific failure modes. Tier 4 surface paths are explicitly bound to human preapproval regardless of advocate confidence. |

## Governance posture

Per `docs/REVIEW_AUTHORITY_PRINCIPLES.md`:

- The harness, extractor, and spec docs in this PR are **Tier 1** (additive
  internal code with no live caller). They land via the model-quorum gate
  with focused dogfood (run the harness on the holdout, confirm baseline
  numbers are sane, confirm stubbed conditions are clearly labeled).
- The first wiring of a trained advocate into the review-queue flow as a
  *proposal-only* preamble is **Tier 3** (semantic correctness with
  potential reputation effect). Model quorum prepares the packet; operator
  explicitly accepts or rejects the risk.
- Any wiring of the advocate into a path that initiates GitHub or external
  writes is **Tier 4** (merge-authority self-modification adjacent).
  Requires human preapproval before implementation and before merge.

The advocate layer never escapes the model-quorum + human-settlement
discipline; it just changes who proposes the first draft of a decision.

## Success criteria

For the **AFT v0.1** experiment (this PR's reason for existing):

- The harness runs end-to-end on the holdout produced by the extractor.
- All three conditions complete without errors.
- The summary report shows accuracy, Brier, latency, cost, and pairwise
  significance for all three conditions.
- The pre-registered hypothesis file is written verbatim alongside the
  results.

For the **advocate layer** to advance past v0.1:

- The local advocate must clear at least the H2 (cost-quality frontier)
  threshold of `scripts/aft_harness.py::PRE_REGISTERED_HYPOTHESES`.
- The result write-up at `docs/status/AFT_RESULT_v0.1.md` must be
  reviewed and signed by the operator.
- No falsification rule fires.

If those criteria are met, v0.2 begins with one additional task (most
likely inbox triage). If not, the advocate-ensemble hypothesis is
**falsified for this codebase** and we do not expand it.

## Settlement

The operator settles this roadmap revision at one of two granularities:

1. **Accept as draft, gate further work on AFT v0.1 outcome.** This PR
   ships the harness, extractor, and specs. No live wiring. Next merge
   only happens after the AFT v0.1 report.
2. **Reject.** This PR is closed. The harness and extractor remain as a
   reusable skeleton for any future bounded-task experiment; the spec
   docs are deleted or archived.

The default recommendation is option 1, because the artifacts in this PR
have value as a generic falsification harness even if the advocate
hypothesis itself is later rejected.

## Open questions for the operator

1. Is "PR triage" the right first task, or should v0.1 start on inbox
   triage despite the UI rejection memory? PR triage has more historical
   data (`gh pr list` is rich) and a clearer label space, which is why
   v0.1 starts there.
2. Should the advocate's confidence threshold for escalation be a global
   knob or per-task? v0.1 assumes per-task to avoid coupling tasks.
3. Is there an objection to Tinker as the hosted-control path? If so,
   v0.1 runs Tier A only and skips the data-vs-method calibration check.
