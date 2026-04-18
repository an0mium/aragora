# G5 extension — Diversity Floor in Team Selection (Measurement Prototype)

**Status:** Planning truth only. **Not `boss-ready`** until the proof-first Foreman gate opens per [NEXT_STEPS_CANONICAL.md](../../status/NEXT_STEPS_CANONICAL.md).
**Labels:** `track:G` `area:ensemble` `area:selection` `type:measurement-prototype` `boss-ready:no-until-foreman-gate`

## What this is

A narrow **measurement prototype** under existing [Track G5 (CruxDetector Activation in Live Debates)](../../plans/ARAGORA_EVOLUTION_ROADMAP.md) that extends `aragora/debate/team_selector.py` with a measured diversity floor on top of ELO + calibration. Not a new selection track. Not a new registry. Not a new protocol-level gate.

## Question

Does greedy quality-only selection in `team_selector.py` produce measurably homogenized failure modes on the existing bounded corpus that a simple diversity floor (pairwise disagreement entropy + provider-family share) would reduce?

## Scope (bounded)

1. Offline replay against receipts from the existing bounded corpus: reconstruct what ensembles greedy selection would have chosen, with and without a diversity floor.
2. Measure the share of debates whose failures cluster by provider family or by model lineage under each arm.
3. Report the delta as a benchmark artifact.

## Acceptance criteria

- [ ] Replay harness under `benchmarks/g5_diversity_floor/` (new) — uses existing receipt store; no production-path changes
- [ ] Benchmark artifact: `benchmarks/g5_diversity_floor/REPORT_<date>.md` with measured homogenization deltas
- [ ] Report concludes with one of: **measurable reduction in correlated-failure clustering → propose a production wiring under G5**; **no measurable reduction → close the experiment and document**
- [ ] No changes to `aragora/debate/team_selector.py` or to production selection behaviour

## Explicitly out of scope

- Any production change to `team_selector.py`
- Any protocol-level or schema-level diversity gate (the retired I3 surface)
- Any rotating benchmark corpus maintained in production
- Any operator dashboard or admin surface
- Any policy enforcement

## Foreman-gate posture

Per [NEXT_STEPS_CANONICAL.md](../../status/NEXT_STEPS_CANONICAL.md), **the current obligation is operationalizing the proof-first loop, not adding new roadmap scope**. This issue is a **replay-only benchmark**. It **must not** carry `boss-ready` or enter the live dispatch queue until the proof-first Foreman gate opens.

## Related

- [Track G5 in ARAGORA_EVOLUTION_ROADMAP.md](../../plans/ARAGORA_EVOLUTION_ROADMAP.md)
- `aragora/debate/team_selector.py` — existing selection code under measurement
- [Biological-timescale analogies brief (non-canonical framing for the diversity metaphor)](../../research/2026-04-18-biological-timescale-analogies-brief.md)
- [NEXT_STEPS_CANONICAL.md proof-first gate](../../status/NEXT_STEPS_CANONICAL.md)
