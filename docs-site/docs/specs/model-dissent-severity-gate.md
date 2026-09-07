---
title: Model Dissent Severity Gate
description: Model Dissent Severity Gate
---

# Model Dissent Severity Gate

## Problem

The merge-quorum gate currently treats any exact-head model comment with a
negative verdict, such as `Verdict: CHANGES-REQUESTED`, as unresolved dissent.
That is safe but too coarse: a reviewer that lists only `[P2]` or `[P3]`
follow-up work blocks the same way as a `[P0]` gate bypass or `[P1]` correctness
defect. The result is repair churn where advisory findings move the PR head and
invalidate otherwise good evidence.

## Phase 1 Rule

Behind `ARAGORA_ENABLE_SEVERITY_GATED_DISSENT=1`, exact-head model-review
dissent blocks merge quorum only when the comment carries:

- a real `[P0]` or `[P1]` finding, or
- an explicit non-empty `Blockers:` / `Blocking findings:` field.

`[P2]` and `[P3]` findings remain visible review findings, but they are advisory
for merge-quorum blocking. They do not count as supportive evidence and must not
satisfy model quorum.

With the flag disabled, the legacy rule remains in force: a negative verdict
blocks.

This mirrors the existing structured-review severity gate in
`aragora.review.builder._apply_severity_gate()`: `REPAIR_FIRST` with no high
findings is downgraded to `APPROVE_WITH_FOLLOWUPS`, while missing severity data
preserves legacy blocking behavior.

## Trust Boundary

Evidence counting stays fail-closed. The evidence linter still rejects negative
verdict comments as countable support. The new rule only changes whether a
negative model comment is promoted to `unresolved_dissent` in merge-packet and
collector disposition logic.

## Follow-Up Phases

Phase 2 should independently triage claimed `[P0]` / `[P1]` findings before
blocking by using existing Aragora verification machinery:

- `aragora.debate.cross_verification.CrossVerificationEngine`
- `aragora.evaluation.llm_judge.LLMJudge`
- reviewer calibration from `aragora.ranking.calibration_engine.DomainCalibrationEngine`

Phase 3 should record final finding disposition into calibration / feedback
stores so repeat nitpick patterns lower future reviewer weight.
