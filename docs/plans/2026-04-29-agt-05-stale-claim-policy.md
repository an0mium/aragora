# 2026-04-29 — AGT-05 stale-claim policy proposal (docs)

> **Status:** vision-layer planning track (`AGT-05`); proposal-only, no code change.
> Builds on `docs/plans/SKIN_IN_THE_GAME_REPUTATION.md` and the Phase E summary in
> `docs/plans/2026-04-29-refined-round-briefing.md`.

## Context

The 2026-04-29 round shadow-mode dogfood (Phase E in that round, *not* this round's Phase E)
exercised the AGT-05 reputation-flow path with `reputation_flow_enabled=False`. It demonstrated
that the bridge correctly emits the expected calibration deltas across five resolution states:

| Verdict        | Calibration delta | Outcome label |
| -------------- | ----------------: | ------------- |
| `PASS`         | +2                | `yes`         |
| `FAIL`         | -2                | `no`          |
| `STALE`        | -2                | `no`          |
| `UNSUPPORTED`  | 0                 | `inconclusive`|
| `ERROR`        | 0                 | `inconclusive`|

The shadow-mode receipt confirmed that all five deltas are emitted symmetrically in observation
mode. What it did *not* answer is the **operating policy** governing the `STALE` axis: **when
should a claim be classified as stale, and what should happen to its reputation delta if the
underlying evidence is judged stale rather than wrong?**

This document proposes an explicit stale-claim policy and the operational guardrails needed
before AGT-05 can leave shadow mode and start writing on-chain or to the canonical position
ledger.

## Why this matters

Treating `STALE` and `FAIL` identically on the calibration axis creates two failure modes:

1. **False decay penalty.** A claim that was *right at the time* but whose evidence has since
   moved (a refactored API surface, a deprecated flag, a settled lawsuit) will be penalized as
   if it were *wrong*. Over time this drives the reputation curve toward conservative,
   never-claim agents — a well-known calibration-pathology in scoring rules with asymmetric
   penalties.

2. **Suppressed evidence-decay signal.** Conversely, if every stale claim is silently elided
   (delta=0), agents are not incentivized to renew evidence, and the staleness rate of the
   knowledge base climbs invisibly.

The current shadow-mode default (`STALE → no, delta=-2`) encodes choice (1). The other policy
points encode it implicitly. We need an explicit, reversible position before the bridge writes
anything observable downstream.

## Proposal: three-axis stale-claim policy

For every resolution event whose verdict is `STALE`, AGT-05 shall record:

1. **`evidence_age_at_resolution_days`** — float, the age (in days) of the underlying
   evidence at resolution time. This is a new payload requirement for the future
   implementation PR, derived from the claim's evidence record.
2. **`half_life_used_days`** — float, the half-life that produced the staleness verdict.
   AGT-05 already carries `decay_half_life_days` on the reputation delta path; the
   future implementation must expose the value used for this specific stale decision
   in the shadow/live receipt.
3. **`policy_decision`** — enum, one of `decay_penalty | renewal_required | abstain`, set by
   the policy below.

### Policy table

| Condition                                                                  | Decision           | Calibration delta |
| -------------------------------------------------------------------------- | ------------------ | ----------------: |
| `evidence_age_at_resolution_days < 0.5 × half_life_used_days`              | `decay_penalty`    | -2 (current)      |
| `0.5 × half_life ≤ evidence_age < 1.5 × half_life`                         | `renewal_required` | 0 (abstain)       |
| `evidence_age_at_resolution_days ≥ 1.5 × half_life_used_days`              | `abstain`          | 0 (abstain)       |

**Reading.**

- If a claim is judged stale very recently relative to its half-life, the staleness was
  premature; treating it as a calibration miss is reasonable (`decay_penalty`).
- If a claim is judged stale in the half-life's middle band, the right interpretation is
  *evidence renewal needed*; the agent is given a no-op delta and a `claim_renewal_id` to
  re-evidence the claim.
- If a claim's evidence is so old that staleness is structural rather than incremental, the
  policy abstains; this is the regime where decay-penalty becomes a false-decay penalty.

Pseudocode for the bridge:

```python
def _stale_calibration_delta(
    age_days: float, half_life_days: float
) -> tuple[int, str]:
    if half_life_days <= 0:
        return -2, "decay_penalty"  # legacy path, stays
    ratio = age_days / half_life_days
    if ratio < 0.5:
        return -2, "decay_penalty"
    if ratio < 1.5:
        return 0, "renewal_required"
    return 0, "abstain"
```

### Default behavior under `reputation_flow_enabled=False`

Shadow mode (the current default) **must** emit all three policy decisions even though no
on-chain or position-ledger writes happen. This lets us pre-seed an empty calibration ledger
before flipping the flag.

The future shadow-mode receipt schema should therefore be extended to include
`policy_decision` per resolution event. This docs PR does not change the current receipt
schema.

## Reversibility plan

This proposal can be reverted in three steps:

1. Set bridge to a mode that always emits `decay_penalty` (current behavior).
2. Drop the `policy_decision` field from receipts.
3. Remove the policy table from `aragora/reputation/policies.py` once that module exists.

No on-chain commitments are made by this proposal.

## What this proposal explicitly does NOT do

- **No code change.** This is a docs-only PR. The policy table is a target the future
  AGT-05 implementation PR will adopt; the staleness scorer in
  `aragora/gauntlet/runner.py` is unchanged.
- **No flag flip.** `reputation_flow_enabled` stays `False` by default and through this PR.
- **No new dependencies.** The decay half-life is already represented on the AGT-05
  reputation delta path as `decay_half_life_days`. The future implementation PR still
  needs to add `evidence_age_at_resolution_days`, `half_life_used_days`, and
  `policy_decision` to the shadow/live receipt payload before this policy can be enforced.

## Open questions for follow-up rounds

1. Should `renewal_required` increment a `claims_pending_renewal` counter on the agent's
   position ledger, surfacing a queue for human-tier agents to revisit?
2. Should `abstain` carry a small *positive* delta to acknowledge that the agent had the
   discipline to abstain rather than chase decayed evidence?
3. Should `decay_half_life_days` be claim-class-specific (e.g., longer for legal-domain
   claims, shorter for software-API claims), and if so, where does that registry live?

These are intentionally left open. The next AGT-05 implementation round will pick them up.

## References

- `docs/plans/SKIN_IN_THE_GAME_REPUTATION.md` — AGT-05 master plan.
- `docs/plans/2026-04-29-refined-round-briefing.md` — Phase E shadow-mode summary,
  including the local-only receipt/report paths produced during the round.
- `aragora/reputation/__init__.py` — current `reputation_flow_enabled()` gate.
- `aragora/reputation/claim_verifier_bridge.py` — current `STALE -> no` mapping.
- `aragora/reputation/types.py` — current `ReputationDelta.decay_half_life_days` field.
