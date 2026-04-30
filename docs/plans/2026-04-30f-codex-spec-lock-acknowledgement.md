# Round 30f — Codex/GPT Spec-Lock Acknowledgement

*Audience:* Codex/GPT and follow-on δ implementers.
*Author:* Factory/Claude, reconciled by Codex after #6898 landed.
*Status:* ACKNOWLEDGED FOR CONSERVATIVE δ SUBSET — #6898 is the implementation acknowledgement.

---

## Current State

The original planning contract required Codex/GPT acknowledgement before δ implementation began. That state is now superseded by #6898, which merged the first conservative #6375 implementation slice:

- `InsufficiencyReceipt.v1` was added beside the existing `ThresholdUpdateReceipt.v1`.
- `ThresholdRecalibrationScheduler.run_receipt_from_sample()` and `run_receipt_from_source()` now emit insufficiency receipts for below-floor, placeholder, or human-numerator schema-gap data.
- `ReviewQueueInvalidationEventSource` adapts the existing stores into the scheduler contract.
- `scripts/measure_invalidation_baseline.py` is dry-run by default; `--write-receipt` is the only mutation path and writes local JSON under `.aragora/review-queue/thresholds/` or an explicit receipt directory.
- The live local run emitted `insufficiency_receipt.v1` with `sample_count=0`, `additional_dispatches_needed=50`, and reasons including `schema_gap_human_numerator`.

This means #6375 is **not closed**. The honest Round 30f δ result is an auditable insufficiency surface, not a measured thesis-threshold replacement.

---

## Acknowledged Rules

Codex/GPT acknowledged and implemented the conservative subset of the Round 30f rules as follows:

- No new invalidation signals were added.
- No `docs/THESIS.md` update was made.
- No H2 pilot, DIC/AGT breadth, marketplace work, public Receipt-as-API work, or production dispatch mutation was introduced.
- The historic `ThresholdUpdateReceipt.v1` path remains backward-compatible.
- The stricter Round 30f path refuses to treat placeholder or schema-gap data as a threshold update.

---

## Source Boundary Actually Landed In #6898

#6898 intentionally used a narrower event-source policy than the broader planning target:

1. Auto-handle calibration rows provide auto-handled numerator and denominator data.
2. Review-queue settlement receipts provide the human-settled denominator.
3. Review-queue settlement receipts provide human invalidation numerator data only when explicit future-schema fields are present, such as `reverted_at`, `post_merge_incident`, or `redo_pr`.
4. `.aragora/overnight/boss_metrics.jsonl` is **not** treated as a human-invalidation numerator in #6898.

Future PRs may widen this source boundary, but they must explicitly state whether they are superseding this conservative #6898 policy and must provide tests for any new mapping.

---

## Follow-On δ Work Requires A New Acknowledgement

Any future δ expansion must explicitly acknowledge or supersede the seven rules in `docs/plans/2026-04-30f-round-spec.md` before implementation if it does any of the following:

- Scans `.aragora/overnight/boss_metrics.jsonl` as numerator evidence.
- Adds GitHub timeline reads.
- Adds new invalidation signals or candidate signal types.
- Edits `docs/THESIS.md`.
- Claims #6375 is closed.
- Changes the receipt schema version.
- Changes the threshold deviation policy.

The required acknowledgement format is:

```text
spec-acknowledged: docs/plans/2026-04-30f-round-spec.md, post-#6898 reconciliation.
exceptions:
  - <none or named exceptions>
will not close #6375 without a measured threshold_update_receipt.v1 or explicit operator approval of insufficiency handling.
```

---

## Why This Reconciliation Exists

The planning PR and #6898 raced. Without this reconciliation, the planning docs would tell future agents to wait for work that already landed and would point them at a future `aragora/triage/invalidation_event_source.py` path that is not the actual merged implementation. This file is now the handoff boundary: #6898 is the conservative δ implementation; broader #6375 closure still requires measured data and a fresh acknowledgement.

— Round 30f planning lane, reconciled after #6898 merge.
