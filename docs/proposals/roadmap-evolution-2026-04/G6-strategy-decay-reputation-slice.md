# G3 extension — Strategy-Decay Reputation Slice (Measurement Prototype)

**Status:** Planning truth only. **Not `boss-ready`** until the proof-first Foreman gate opens per [NEXT_STEPS_CANONICAL.md](../../status/NEXT_STEPS_CANONICAL.md).
**Labels:** `track:G` `area:reputation` `type:measurement-prototype` `boss-ready:no-until-foreman-gate`

## What this is

A narrow **measurement prototype** under existing [Track G3 (Skin-in-the-Game Reputation Flow)](../../plans/ARAGORA_EVOLUTION_ROADMAP.md). Not a new reputation subsystem; a sub-item that extends the existing slice model with one additional, benchmarkable behaviour.

## Question

When an agent's reputation slice is tied to participation in a specific exploit strategy that becomes publicly known or measurably crowded, does faster decay of that slice produce better-calibrated downstream dispatch decisions on a receipt-backed synthetic corpus than holding the slice constant?

## Scope (bounded)

1. Build a small synthetic corpus of `(agent, strategy, epoch, observed-outcome)` tuples that models the crowded-strategy scenario. This corpus lives under `benchmarks/g3_strategy_decay/` — not production data.
2. Prototype a decay function on top of the existing G3 reputation-slice math. Prototype may live in a scratch script, not production code, until acceptance.
3. Compare dispatch decisions with and without the decay on the synthetic corpus.

## Acceptance criteria

- [ ] Synthetic corpus + generator script lands under `benchmarks/g3_strategy_decay/`
- [ ] Benchmark artifact: `benchmarks/g3_strategy_decay/REPORT_<date>.md` with measured calibration deltas between decay-on and decay-off arms
- [ ] Report concludes with one of: **measurable improvement → propose production wiring under G3**; **no measurable improvement → close the experiment and document**
- [ ] No changes to `aragora/ranking/`, `aragora/blockchain/contracts/`, or dispatch code paths

## Explicitly out of scope

- Any production reputation-writer change
- Any ERC-8004 schema extension
- Any dispatch-policy change
- Any public-signal monitor for real-world strategy lifecycle
- Any registry of exploit strategies (that was the retired E8 surface)

## Foreman-gate posture

Per [NEXT_STEPS_CANONICAL.md](../../status/NEXT_STEPS_CANONICAL.md), **the current obligation is operationalizing the proof-first loop, not adding new roadmap scope**. This issue is a **benchmark-only prototype**. It **must not** carry `boss-ready` or enter the live dispatch queue until the proof-first Foreman gate opens.

## Related

- [Track G3 in ARAGORA_EVOLUTION_ROADMAP.md](../../plans/ARAGORA_EVOLUTION_ROADMAP.md)
- [SKIN_IN_THE_GAME_REPUTATION design doc](../../plans/SKIN_IN_THE_GAME_REPUTATION.md)
- [Biological-timescale analogies brief (non-canonical framing for the decay metaphor)](../../research/2026-04-18-biological-timescale-analogies-brief.md)
- [NEXT_STEPS_CANONICAL.md proof-first gate](../../status/NEXT_STEPS_CANONICAL.md)
