# Skin-in-the-Game Reputation Flow — Claims → Predictions → Resolution → Reputation → Dispatch

> **Status:** vision-layer planning track (`AGT-05`); not boss-ready until queue governance permits the upper-layer tranche.
> **Created:** 2026-04-17
> **Parent:** [AGENT_CIVILIZATION_SUBSTRATE](AGENT_CIVILIZATION_SUBSTRATE.md)
> **Depends on:** AGT-02 (consumer surface), AGT-03 (Manifold), AGT-04 (synthetic GitHub markets), DIC-15 (CruxSet contract), DIC-16 (receipt/KM provenance)

## Thesis

A reputation system without external resolution is agents agreeing with agents. The existing `aragora/blockchain/contracts/{identity,reputation,staking,validation}.py` primitives are correctly designed but currently not wired to a live settlement loop. This plan defines the unified flow that turns each consequential agent action into a verifiable, time-bounded, cryptographically attested reputational consequence.

The discipline is: **stake before you speak, settle when reality lands, lose dispatch if you are wrong too often.**

## The unified flow

```
   ┌─────────────┐    ┌────────┐    ┌─────────────────┐    ┌────────────┐    ┌──────────────┐    ┌──────────────────┐
   │ claim/stance│ -> │ stake  │ -> │ resolution event│ -> │ settlement │ -> │ reputation Δ │ -> │ dispatch update  │
   └─────────────┘    └────────┘    └─────────────────┘    └────────────┘    └──────────────┘    └──────────────────┘
```

Each stage is a real module today. This plan wires them.

| Stage | Existing module | What lands in AGT-05 |
|---|---|---|
| claim/stance | `aragora/reasoning/claims.py`, `aragora/reasoning/crux_detector.py`, Arena debate output, Manifold/Metaculus prediction | normalize across sources into `StakeableClaim` |
| stake | `aragora/blockchain/contracts/staking.py`, `aragora/blockchain/compute_budget.py` | per-claim stake commitment with refund/forfeit policy |
| resolution event | Manifold market resolution (AGT-03), Metaculus closure, synthetic GitHub event (AGT-04), DIC-14 claim verifier output, oracle attestation | unified `ResolutionEvent` shape |
| settlement | `aragora/blockchain/receipt_settlement.py`, existing `Time Is Part of Settlement` doctrine | settlement-window enforcement, dispute path |
| reputation Δ | `aragora/blockchain/contracts/reputation.py`, `aragora/knowledge/mound/adapters/erc8004_adapter.py` | per-domain reputation update with decay |
| dispatch update | debate `team_selector`, calibration tracker, ELO, swarm dispatch policy | dispatch eligibility gated by per-domain reputation |

## Core types

### StakeableClaim

```yaml
claim_id: claim.agent.<agent>.<domain>.<hash>
agent_id: <agent-id>
domain: prediction_market | debate_position | code_pr | km_contribution | crux_resolution
statement: "<machine-parseable statement>"
position: long | short | for | against | <typed-position>
stake:
  budget_units: <int>
  policy: forfeit_on_loss | scaled | refund_on_inconclusive
resolution:
  source: manifold | metaculus | synthetic_github | claim_verifier | oracle
  resolution_id: <external id>
  expected_window: { open: <iso>, close: <iso> }
provenance:
  receipt_id: <decision-receipt-id>
  arena_run_id: <optional>
  crux_id: <optional>
```

### ResolutionEvent

```yaml
resolution_id: <source-specific id>
source: manifold | metaculus | synthetic_github | claim_verifier | oracle
resolved_at: <iso>
outcome: yes | no | inconclusive | invalid
score:
  brier: <float 0-1>           # for probabilistic predictions
  binary_correct: <bool>        # for binary stances
  partial_credit: <float 0-1>   # for graded outcomes
attestation:
  signer: <oracle-pubkey>
  signature: <base64>
```

### ReputationDelta

```yaml
delta_id: rep.<agent>.<domain>.<resolution-id>
agent_id: <agent>
domain: <domain>
delta: <signed float>
reason:
  resolution_id: <id>
  claim_id: <id>
  scoring_rule: brier_proper | log_loss | binary | flat
applied_at: <iso>
decay_policy:
  half_life_days: <int>
  floor: <float>
  ceiling: <float>
```

## Scoring rules

The system supports multiple scoring rules per domain so calibration is not collapsed into a single signal:

| Domain | Default scoring rule | Notes |
|---|---|---|
| prediction_market | Brier (proper) | rolling 90-day window, decayed |
| debate_position | binary correctness from CruxSet resolution | verified against later evidence |
| code_pr | merged within window with no rollback | settlement window 30 days |
| km_contribution | downstream usage with no contradiction event | 60-day window |
| crux_resolution | counterfactual confirmation per DIC-15 | requires crux to be reified |

Each domain decays independently. Aggregate reputation is a weighted sum visible to `team_selector`.

## Dispatch eligibility

Dispatch eligibility is computed from per-domain reputation against per-domain thresholds:

```yaml
dispatch_policy:
  prediction_market:
    floor_brier_90d: 0.20         # below this, suspend prediction-domain dispatch
    promotion_brier_90d: 0.12     # above this, enter senior-prediction class
  debate_position:
    min_truth_rate_30d: 0.55
  code_pr:
    min_merge_rate_30d: 0.40
    rollback_rate_30d_max: 0.10
```

Suspension is **soft by default** (downweight in selector) and **hard only** for explicitly opted-in lanes. Hard suspension always emits a receipt with reasons and reinstatement criteria.

## Existing contract surface to extend

The `aragora/blockchain/contracts/` directory already defines:

- `identity.py` — agent identity registration
- `reputation.py` — reputation registry (ERC-8004-shaped)
- `staking.py` — stake commitment and slashing primitives
- `validation.py` — validation contract surface

This plan does **not** introduce new contract files. It defines the call sequence, schema versioning, and integration glue that make these contracts a live flow.

## Settlement, dispute, and re-opening

Following the canonical doctrine that **time is part of settlement**:

- Each ResolutionEvent has a dispute window. Within the window, an agent may file a counter-attestation with stake.
- After the window, the resolution is final unless re-opened by an explicit policy event (e.g. the oracle is later proven wrong).
- A re-opened resolution rolls back the corresponding ReputationDelta and emits a `ReputationDeltaReversed` event.
- Reversal events are themselves a domain agents can be reputational on (anti-fragility).

## Risks and tempering

- **Adversarial agents farming easy markets.** Mitigation: per-domain reputation, weighting by stake size, decay so old wins don't carry forever.
- **Oracle compromise.** Mitigation: prefer multiple independent oracles per domain; require attestation by signer key registered through `validation.py`.
- **Cold-start:** new agents have no reputation. Mitigation: bootstrap reputation from a sandbox lane with reduced stake limits before promoting to production dispatch.
- **Reputation as ranked list invites Goodhart.** Mitigation: surface per-domain breakdown in operator views, not a single score.
- **Slashing as adversarial weapon.** Mitigation: slashing requires meeting an explicit policy; protocols/coordination prevents accidental slashing on transient errors.

## Sequencing within AGT-05

| Step | Deliverable | Dependencies |
|---|---|---|
| 1 | `StakeableClaim` and `ResolutionEvent` schemas | DIC-15 CruxSet contract for crux-domain claims |
| 2 | Stake commitment path through `staking.py` with refund/forfeit policy | existing staking primitives |
| 3 | Resolution ingestion adapters (Manifold, Metaculus, synthetic GH) | AGT-03, AGT-04 |
| 4 | Settlement and dispute window enforcement via `receipt_settlement.py` | existing receipt anchoring |
| 5 | Reputation update path via `reputation.py` and ERC-8004 KM adapter | existing reputation contract |
| 6 | Dispatch eligibility integration with `team_selector` and ELO | debate selection path |
| 7 | Reversal/re-opening path | settlement window enforcement |

## What this plan does NOT do

- Does not introduce new contract files; uses existing primitives.
- Does not enable hard slashing by default; soft downweighting is the default policy.
- Does not auto-mutate dispatch policy; threshold values are operator-configured initially.
- Does not move AGT-05 issues into `boss-ready` until queue governance permits the upper-layer tranche.

## References

- [AGENT_CIVILIZATION_SUBSTRATE](AGENT_CIVILIZATION_SUBSTRATE.md)
- [AGENT_CONSUMER_SURFACE](AGENT_CONSUMER_SURFACE.md)
- [2026-04-17-prediction-market-validation](2026-04-17-prediction-market-validation.md)
- [EPISTEMIC_CI_AND_CRUX_ENGINE](EPISTEMIC_CI_AND_CRUX_ENGINE.md)
- Code: `aragora/blockchain/contracts/`, `aragora/blockchain/receipt_settlement.py`, `aragora/reasoning/{claims,crux_detector}.py`, `aragora/knowledge/mound/adapters/erc8004_adapter.py`, debate `team_selector`
