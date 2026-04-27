# Roadmap Evolution — 2026-04-18 Proposal Bundle (Trimmed)

This bundle was initially drafted as 12 epic-grade issues spanning three proposed new tracks (H Surrogate Distillation, I Population-Under-Selection, J Multi-Timescale Memory). After a skeptical-read pass on 2026-04-18, the bundle was **trimmed to 4 bounded, benchmark-gated, Foreman-gated issues** under existing tracks B/E/G.

## Critique that drove the trim

1. [P1] New top-level tracks duplicated canonical tracks D/E/G and weakened canonical clarity.
2. [P1] [NEXT_STEPS_CANONICAL.md](../../status/NEXT_STEPS_CANONICAL.md) explicitly says "the current obligation is operationalizing the proof-first loop, **not adding new roadmap scope**".
3. [P2] A third roadmap doc (the now-retired `docs/ROADMAP_EVOLUTION.md`) created source-of-truth drift.
4. [P2] Biology framing belongs in a research brief, not in doctrine. The surviving heuristics need benchmark-artifact acceptance, not metaphor.

## Surviving issues (4)

Each is a **measurement-only / benchmark-only** experiment under an existing track. None carries `boss-ready`. None enters the live dispatch queue until the proof-first Foreman gate opens.

| File | Track | Type | What it measures |
|------|-------|------|------------------|
| [E6-disagreement-as-signal.md](E6-disagreement-as-signal.md) | E1 | measurement experiment | Does pre-consensus disagreement predict settlement divergence? |
| [G6-strategy-decay-reputation-slice.md](G6-strategy-decay-reputation-slice.md) | G3 | measurement prototype | Does faster decay on crowded-strategy slices improve dispatch calibration on a synthetic corpus? |
| [G7-diversity-maintenance-gates.md](G7-diversity-maintenance-gates.md) | G5 | measurement prototype | Does greedy quality-only team selection produce measurably homogenized failure modes on the existing corpus? |
| [B4-cheap-signal-routing-benchmark.md](B4-cheap-signal-routing-benchmark.md) | B4 | benchmarked decision policy | On the existing bounded corpus, does cheap-signal routing preserve rescue/verification rate while reducing cost, with auditable routing receipts? |

## Retired issues (9)

These drafts were retired. Each file now carries a tombstone pointing at where the surviving narrow idea (if any) lives and why the epic was rejected:

- [D5-germline-somatic-memetic-separation.md](D5-germline-somatic-memetic-separation.md)
- [E8-exploit-strategy-registry.md](E8-exploit-strategy-registry.md)
- [H1-surrogate-training-corpus.md](H1-surrogate-training-corpus.md)
- [H3-routing-gate-with-receipt.md](H3-routing-gate-with-receipt.md)
- [I1-meta-ecosystem-ledger.md](I1-meta-ecosystem-ledger.md)
- [I2-bounded-protocol-ab-tests.md](I2-bounded-protocol-ab-tests.md)
- [I4-goodhart-sentinel.md](I4-goodhart-sentinel.md)
- [J1-germline-permanence-contract.md](J1-germline-permanence-contract.md)
- [J2-somatic-isolation.md](J2-somatic-isolation.md)

(They remain in the directory only because the sandbox does not currently permit hard deletion. Remove with `git rm` on the next commit pass.)

## Rules of the road

1. Any issue opened from this bundle must carry the label `boss-ready:no-until-foreman-gate`.
2. Acceptance criteria must be a **benchmark artifact or proof surface**, not an engineering spec.
3. No issue here enters the live dispatch queue until the proof-first Foreman gate opens per [NEXT_STEPS_CANONICAL.md](../../status/NEXT_STEPS_CANONICAL.md).
4. Significance results that graduate to production wiring must do so as a narrow, feature-flagged extension of the existing track, not as a new track.

## Related

- [CANONICAL_GOALS.md](../../CANONICAL_GOALS.md) — 8 pillars; unchanged by this bundle
- [ARAGORA_EVOLUTION_ROADMAP.md](../../plans/ARAGORA_EVOLUTION_ROADMAP.md) — Tracks A–G; the narrow surviving sub-bullets live here
- [Biological-timescale analogies research brief (non-canonical)](../../research/2026-04-18-biological-timescale-analogies-brief.md)
- [NEXT_STEPS_CANONICAL.md — proof-first Foreman gate and `boss-ready` queue governance](../../status/NEXT_STEPS_CANONICAL.md)
