# Semantic-risk settlement packets — AGT-cascade

**Date:** 2026-04-28
**Author:** review droid (this run, on `docs/2026-04-28-evolution-round`)
**PRs covered:** #6772 [AGT-03 calibration curve], #6791 [AGT-05 delta reversal schema],
#6793 [AGT-05 team selector wiring]
**Author of all three PRs:** `an0mium`
**Scope:** read-only review of each PR diff at its current head SHA.  Recommendation
on admin-squash order vs. requesting further changes.

This document is **not** a settlement-gate filing — the workflow's own "Epistemic
Hygiene + Settlement Gate" check is already SUCCESS on all three PRs.  This is a
human-readable companion explaining what we'd be agreeing to if we admin-squash.

---

## Pre-flight (shared)

| PR | Branch | Head SHA | Files | +/- | Mergeable | Review |
|----|--------|----------|-------|-----|-----------|--------|
| #6772 | `vision-incubator/agt-03-calibration-curve` | `68b70b9b687a` | 2 | +282/-0 | MERGEABLE (BLOCKED) | REVIEW_REQUIRED |
| #6791 | `vision-incubator/agt-05-delta-reversal` | `34fb5f95935c` | 4 | +312/-4 | MERGEABLE (BLOCKED) | REVIEW_REQUIRED |
| #6793 | `vision-incubator/agt-05-team-selector-wiring` | `4f5101f78e4d` | 2 | +165/-0 | MERGEABLE (BLOCKED) | REVIEW_REQUIRED |

`BLOCKED` reflects only `REVIEW_REQUIRED`; every required CI check is SUCCESS,
including `Epistemic Hygiene + Settlement Gate`, `V1 Scope Lock Gate`,
`PR Scope`, `Quality Gates`, and the full test-fast matrix.

---

## Packet 1 — PR #6772 (AGT-03 calibration curve)

### Scope confirmation
Adds `CalibrationBin(low, high, count, fraction_yes, mean_predicted)` frozen
dataclass and `ManifoldBrierScorer.calibration_curve(*, n_bins=10) -> list[CalibrationBin]`
to `aragora/metrics/manifold_brier.py`.  Public re-export added to `__all__`.
Test file gains `TestCalibrationBin` and `TestCalibrationCurve` — including
parametrized boundary tests at `0.3`, `0.6`, `0.7`, `1.0` and at all
`i/10` decimal boundaries.

### Code-path semantic risks
- **Boundary placement**: uses `Decimal(str(p))` then floor-divides by `n_bins`
  to dodge the IEEE 754 trap where `0.3 / 0.1 == 2.9999…`.  This is the
  principled fix; the parametrized tests prove the boundaries are inclusive on
  the low side.
- **Last-bin clamp**: `idx = min(idx, n_bins - 1)` ensures `p == 1.0` lands in
  the last bin; tested.
- **Empty bins**: `fraction_yes` and `mean_predicted` are `None` (not 0.0) when
  `count == 0`; reflected in `to_dict()`.
- **Validation**: rejects `n_bins` outside `[2, 100]` with `ValueError`.

### Blast radius
None.  No call site in the codebase invokes `calibration_curve()` yet; this is
the producer slice.  The companion consumer slice would be a separate PR that
renders the bins.

### Reversibility
Pure revert is safe.  No on-disk format change; no schema migration; no
existing caller behaviour altered.

### Cross-PR dependency
None.  Independent of #6791 and #6793.

### Halt criteria
None observed.  Feature is gated by the pre-existing
`ARAGORA_MANIFOLD_BRIER_ENABLED` flag (the same gate the `add()` and
`rolling_score()` methods already use); off by default.

### Signoff readiness
**Ready to admin-squash.**  Smallest-blast PR of the three.

---

## Packet 2 — PR #6791 (AGT-05 delta reversal schema)

### Scope confirmation
Adds `ReputationDeltaReversed` frozen dataclass to `aragora/reputation/types.py`
and `ReputationStore.reverse_delta(delta_id, *, reason, now) -> ReputationDeltaReversed`
plus `ReputationStore.reversals_for(agent_id)` to `aragora/reputation/store.py`.
Re-exports added in `aragora/reputation/__init__.py`.

### Code-path semantic risks
This PR is the only one of the three with deletions (`+312/-4`); the deletions
are localized and intentional:

```diff
-        if not deltas:
+        live = [d for d in deltas if d.delta_id not in self._reversals]
+        if not live:
             return 0.0
         if not apply_decay:
-            return sum(d.delta for d in deltas)
+            return sum(d.delta for d in live)
         now = datetime.now(tz=UTC)
-        return sum(d.delta * _decay_weight(d, now) for d in deltas)
+        return sum(d.delta * _decay_weight(d, now) for d in live)
```

`get_score()` now filters reversed deltas.  The behaviour change is **a
no-op for callers that never call `reverse_delta()`** (and `_reversals` is
empty by default).  This is the deliberate semantic of the slice.

Other code-path observations:
- `_delta_by_id` index is populated in two places: `record_delta` and
  `load_from_file`.  Both paths are covered.  The index is never pruned — a
  reversed delta still occupies an entry, which is correct because the
  `_reversals` map is the single source of truth for "is this delta live".
- `_reversal_path` is auto-derived from the main path:
  `path.parent / (path.stem + ".reversals.jsonl")`.  Two stores in the same
  directory with the same `stem` would collide; reasonable in practice but
  worth noting.  Not configurable.
- `reverse_delta()` is idempotent (returns the existing reversal if the same
  `delta_id` is reversed twice); the persisted JSONL contains a single line
  per reversal, verified by `test_persisted_reversal_is_idempotent_after_reload`.
- `reversal_id = "rev_" + sha256(json.dumps({original_delta_id, reversed_at}))[:16]`.
  Two reversals at literally the same UTC microsecond would collide on `id`,
  but the dictionary is keyed on `original_delta_id` so the second call is
  short-circuited before that point.  Not a real risk.

### Blast radius
- `aragora/reputation/store.ReputationStore`: every existing caller reads
  `get_score()` semantics that now filter `_reversals`.  Empty by default
  → no behavioural change.
- One additional file (`*.reversals.jsonl`) appears next to the ledger only
  when `reverse_delta()` is invoked.

### Reversibility
Pure revert is safe.  Companion `*.reversals.jsonl` files left on disk after
revert are silently ignored (no loader path remains).  In-memory state
trivially returns to the pre-PR `if not deltas: return 0.0` semantics.

### Cross-PR dependency
None.  `ReputationDeltaReversed` is **not** imported by #6793.  AGT-05 is
sliced into independent sub-deliverables.

### Halt criteria
None observed.  The PR body's gating section names
`ARAGORA_REPUTATION_FLOW_ENABLED`, but in this slice the gate is implicit:
`reverse_delta()` is opt-in (must be called explicitly), and the
`get_score()` filter is a no-op until the first reversal is persisted.

### Signoff readiness
**Ready to admin-squash.**  The 4 deletions are localized and the test for
the durability case (`test_reversal_after_reload_survives_second_reload`)
proves that disk → memory → disk → memory round-trips.

---

## Packet 3 — PR #6793 (AGT-05 team selector wiring)

### Scope confirmation
Adds two fields to `TeamSelectionConfig`:
- `enable_agt05_reputation_selection: bool = False`
- `reputation_bridge_config: ReputationBridgeConfig | None = None`

Adds `reputation_store: ReputationStore | None = None` parameter to
`TeamSelector.__init__`.  When `flag and store is not None`, replaces
`self.calibration_tracker` with `ReputationCalibrationBridge`.

### Code-path semantic risks
- **Imports**: `ReputationCalibrationBridge` and `ReputationBridgeConfig`
  resolve to `aragora/reputation/selection_bridge.py`, which exists on
  `origin/main` (blob `bb9d8b9b`); the bridge is dormant by default
  per the env-flag check inside it.  Confirmed by inspecting the file
  on `origin/main`.
- **Replacement of caller-supplied tracker**: when the flag is on AND a
  store is supplied, an explicitly passed `calibration_tracker` is
  superseded by the bridge.  This is asserted by
  `test_explicit_tracker_superseded_by_bridge_when_flag_on`.  Operator
  surprise risk: if a future caller passes a custom tracker with the AGT-05
  flag on, their tracker will be ignored.  Acceptable in v1; would be a
  reasonable follow-up to log a `logger.warning` when the override happens.
- **Double-gate semantics**: the test `test_neutral_when_env_flag_unset`
  proves that even with the config flag on AND a store with deltas,
  `get_brier_score` returns `0.5` (neutral) unless the env flag
  `ARAGORA_REPUTATION_FLOW_ENABLED` is also set.  This means the wiring
  is genuinely dormant after merge.
- **TYPE_CHECKING imports** for `ReputationBridgeConfig` and
  `ReputationStore` — the runtime import for `ReputationCalibrationBridge`
  is local inside the conditional.  Clean.

### Blast radius
None for the default config.  When opted-in by both flag layers, the bridge's
score replaces the calibration scorer for all team-selection events on
that `TeamSelector` instance.  Per-instance, not global.

### Reversibility
Pure revert is safe.  No persistent state.  No effect on stores/files.

### Cross-PR dependency
None.  The bridge module the wiring imports has been on `main` since before
this PR was opened.  Independent of #6791.

### Halt criteria
None observed.

### Signoff readiness
**Ready to admin-squash.**  This is the wedge slice — without it, the
existing dormant `selection_bridge.py` has no in-tree caller.

---

## Admin-squash recommendation

### Recommendation: authorize admin-squash for all three, in this order

1. **#6772** (AGT-03 calibration curve) — fully isolated, additive only,
   smallest blast radius.
2. **#6791** (AGT-05 delta reversal schema) — establishes durable
   reversal capability in the store before any consumer that emits
   reversals.  Has the only deletions of the three; landing it under
   review-light conditions is the highest-stakes of the three but still
   well-tested and double-gated.
3. **#6793** (AGT-05 team selector wiring) — wedge that lights up the
   already-dormant `ReputationCalibrationBridge` for callers that opt in
   via both layers (`enable_agt05_reputation_selection=True` *and*
   `ARAGORA_REPUTATION_FLOW_ENABLED=1`).

The order is logical, not strictly required.  All three are mutually
independent at the diff level (verified above).

### Caveats worth recording before squashing

1. **Single-author concentration.**  All three PRs are from `an0mium`
   under `vision-incubator/agt-*` branches.  The Settlement Gate already
   accepts these as scope-locked; a human reviewer signoff after merge —
   even an async one — would be appropriate to keep the heterogeneous-review
   record clean.  This is consistent with thesis Premise 3.

2. **Codex's halt directive on the evolution-round.**  Codex paused the
   evolution-round dogfood at Phase 3 and asked us not to open *new* PRs
   until the current cascade drains.  These three are *existing* PRs in
   that cascade; admin-squashing them helps drain the cascade.  The halt
   does not apply to them.

3. **No semantic-risk packet template existed in-tree before this run.**
   This document was written from first principles (`Slice / Gating`
   convention from the PR bodies + diff inspection).  Recommend keeping
   the format and applying it to future AGT-cascade landings.

4. **#6791 deletes 4 lines.**  Of the three, this is the only PR that
   changes the meaning of an existing public method (`get_score()`).
   The change is a no-op until the first reversal is recorded.  Operators
   running the store with the existing JSONL ledger see identical
   behaviour until they explicitly call `reverse_delta()`.

5. **Operator surprise in #6793.**  When the AGT-05 flag is on with a
   store provided, an explicitly passed `calibration_tracker` is silently
   dropped in favour of the bridge.  Recommend a follow-up issue to add a
   `logger.warning` for the override case.  Not a blocker.

### What would change my recommendation

- A required check flips from SUCCESS to FAILURE during the squash window.
- A reviewer requests changes that re-target the feature flags or the
  default-off semantics.
- Discovery that `ReputationStore` is being instantiated by a code path
  that *cannot* tolerate the new `_reversals` filter — none found in this
  review (the filter is a no-op pre-reversal).

### What I would still want logged after squash

- A short note in `docs/status/` or a follow-up issue confirming that the
  `enable_agt05_reputation_selection` flag remains off in the default
  shipped config and the env flag remains unset in the production env file.
- The companion follow-up to #6772: a renderer / API surface that
  consumes `calibration_curve()` so the producer doesn't sit unused.

---

## Bottom line

All three PRs are **safe to admin-squash today** in the order
**#6772 → #6791 → #6793**.  The double-gating (config defaults plus env
flags) means production behaviour is unchanged at merge.  The only
semantic-behaviour delta is in #6791's `get_score()`, which is a no-op
until the first explicit `reverse_delta()` call.
