# AGT-05 Reputation Signal Flow — End-to-End Smoke Result (2026-05-14)

## Why this exists

The agent-civilization substrate roadmap (`docs/plans/AGENT_CIVILIZATION_SUBSTRATE.md`) defines AGT-05 as **ERC-8004 reputation flow wiring (claims → predictions → resolution → reputation → dispatch)**. The substrate is now in place:

| Code | Status | Landing PR(s) |
|------|--------|---------------|
| AGT-03 (Manifold rolling Brier) | landed | #6707, #6772, #6932, #7146 |
| AGT-04 (Synthetic GitHub markets) | landed | #6944, #7133, #7139 |
| AGT-05 (ReputationCalibrationBridge) | landed | #6731 |
| AGT-05 (Metaculus → reputation bridge) | landed | #6944 |

What was **not yet demonstrated** was the complete signal flow running end-to-end on the same code path a live debate would use. This document captures that demonstration.

## What runs

```
scripts/agt_05_reputation_flow_smoke.py
```

Exercises:

```
MetaculusQuestion (resolved) + agent prediction
    → bridge_from_metaculus_question  (aragora/reputation/metaculus_bridge.py)
    → (StakeableClaim, ResolvedClaim)
    → settle_claim                    (aragora/reputation/settlement.py)
    → ReputationDelta
```

Flag-gated on `ARAGORA_REPUTATION_FLOW_ENABLED=1`. Hermetic — no live Manifold/Metaculus API calls. Three synthetic resolved questions × three agents (well-calibrated, indifferent, anti-calibrated).

## Run

```
ARAGORA_REPUTATION_FLOW_ENABLED=1 python3 scripts/agt_05_reputation_flow_smoke.py
```

## Captured output (2026-05-14T19:44:33Z)

```
AGT-05 Reputation Flow Smoke Test
==============================================================================
  ARAGORA_REPUTATION_FLOW_ENABLED=1
  scoring_rule=brier_proper
  agents=['claude-opus-4-7', 'gpt-4.1', 'demo-anti']
  questions=['manifold-1001', 'manifold-1002', 'manifold-1003']

question       agent                     pred_p outcome   brier   payout   delta
------------------------------------------------------------------------------
manifold-1001  claude-opus-4-7             0.90 yes      0.010    0.980    9.80
manifold-1001  gpt-4.1                     0.60 yes      0.160    0.680    6.80
manifold-1001  demo-anti                   0.10 yes      0.810   -0.620   -6.20
manifold-1002  claude-opus-4-7             0.85 yes      0.023    0.955    9.55
manifold-1002  gpt-4.1                     0.55 yes      0.202    0.595    5.95
manifold-1002  demo-anti                   0.05 yes      0.902   -0.805   -8.05
manifold-1003  claude-opus-4-7             0.10 no       0.010    0.980    9.80
manifold-1003  gpt-4.1                     0.45 no       0.203    0.595    5.95
manifold-1003  demo-anti                   0.90 no       0.810   -0.620   -6.20
------------------------------------------------------------------------------

Per-agent total reputation delta (Brier-proper, stake_units=10 per Q):
  claude-opus-4-7           +29.150
  demo-anti                 -20.450
  gpt-4.1                   +18.700
```

## Interpretation

The Brier-proper scoring rule produces exactly the structure we want:

- **claude-opus-4-7** (well-calibrated, p ≈ outcome): **+29.15** — clear positive reputation
- **gpt-4.1** (predictions near 0.5): **+18.70** — small positive, indifferent predictors aren't punished but aren't strongly rewarded either
- **demo-anti** (systematically wrong, p ≈ 1 − outcome): **-20.45** — clear negative reputation

The payout fraction `1 − 2·Brier` is symmetric around break-even at Brier=0.5 (payout 0.0): a perfect predictor at Brier=0.0 earns +1.0× stake; a perfectly wrong predictor at Brier=1.0 loses −1.0× stake.

## What this proves

1. **The full reputation flow runs cleanly with the flag enabled.** No flag-OFF paths to dodge, no missing dependencies.
2. **Brier-proper scoring produces the expected signal shape.** Calibrated agents accumulate positive reputation, anti-calibrated agents accumulate negative reputation, indifferent agents drift small-positive.
3. **The bridge is the right shape for production wiring.** Any module that can produce a `MetaculusQuestion` (or Manifold-equivalent) + a per-agent `predicted_probability` can plug straight into this path.

## What this does NOT do (deliberately out of scope)

- **No `apply_delta` call.** The script computes deltas; it does not write them to any on-chain registry or ledger. That's the next AGT-05 sub-deliverable (registry-write integration).
- **No `Arena` wiring.** This is calibration on synthetic data, not a live debate. Wiring an Arena post-debate hook to call this same code path is the production integration step.
- **No `team_selector` reading.** The end goal — `team_selector` querying reputation to bias agent selection — is the *visibility* milestone for AGT-05; this is upstream of that.
- **No on-chain ERC-8004 calls.** `aragora/blockchain/contracts/reputation.py` is the contract wrapper; this script does not touch it.

## Next AGT-05 sub-deliverables

1. **Registry persistence**: wire `settle_claim` → `ReputationDelta` → an in-memory or local-store ledger (`aragora/reputation/ledger.py` if not present, otherwise extend existing store). Flag-gated.
2. **Arena post-debate hook**: when a debate produces an extractable prediction (e.g., via the existing belief network or the prover-estimator surface), submit it as a `StakeableClaim` and settle on resolution. Flag-gated on `ARAGORA_REPUTATION_FLOW_ENABLED` AND on a per-Arena opt-in.
3. **`team_selector` visibility**: read the ledger from `aragora.debate.team_selector` and surface it as one signal among the existing ELO/calibration/persona signals. Default-OFF.

Each of these is a separate small PR. This smoke result is the foundation that proves the substrate is calibrated and ready for them.

## Provenance

- Repo: synaptent/aragora
- Script: `scripts/agt_05_reputation_flow_smoke.py`
- Worktree: `.worktrees/codex-auto/claude-20260514-194402-e0d8d9d4`
- Branch: `demo/agt-05-reputation-flow-smoke`
- Ran at: 2026-05-14T19:44:33Z
- Flag: `ARAGORA_REPUTATION_FLOW_ENABLED=1`
- Scoring rule: `brier_proper`
- Decay half-life: 30 days (default)
