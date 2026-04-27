# E1 extension — Pre-Consensus Disagreement-as-Signal Measurement

**Status:** Planning truth only. **Not `boss-ready`** until the proof-first Foreman gate opens per [NEXT_STEPS_CANONICAL.md](../../status/NEXT_STEPS_CANONICAL.md).
**Labels:** `track:E` `area:debate` `area:measurement` `type:measurement-experiment` `boss-ready:no-until-foreman-gate`

## What this is

A narrow **measurement experiment** under existing [Track E1 (Debate Quality and Calibration)](../../plans/ARAGORA_EVOLUTION_ROADMAP.md). Not a new subsystem. Not a new track. Not a new debate primitive.

## Question

Do pre-consensus disagreement patterns among heterogeneous evaluators predict later settlement divergence on the existing bounded corpus?

Today Aragora treats disagreement primarily as a problem surface (hollow-consensus detection, Trickster, dissent tracking). The hypothesis is that the *distribution* of disagreement early in a debate carries predictive information about whether the eventual consensus will be revised at settlement. If so, it is a usable feature for E5 crux detection, difficulty scoring, and calibrated confidence. If not, the hypothesis stops being load-bearing and this experiment is closed.

## Scope (bounded)

1. Extract pre-consensus disagreement features from receipts in the existing bounded corpus — no new feature extractor modules outside of a short measurement script.
2. Compute correlation with settled-outcome divergence (for receipts where settlement is observed).
3. Report the correlation, confidence intervals, and sensitivity analysis as a benchmark artifact.

## Acceptance criteria

- [ ] Measurement script lands under `benchmarks/e1_disagreement_signal/` (new directory) — not a production module
- [ ] Benchmark artifact published: `benchmarks/e1_disagreement_signal/REPORT_<date>.md` with (a) correlation, (b) confidence interval, (c) sensitivity analysis, (d) failure-mode analysis
- [ ] No changes to `aragora/debate/` or `aragora/reasoning/` — this issue does not wire any consumer
- [ ] Report concludes with one of: **significant → propose a follow-up under E5 crux detection**; **non-significant → close and document**

## Explicitly out of scope

- Any production feature extractor under `aragora/debate/`
- Any routing consumer
- Any dashboard or admin surface
- Any cross-cutting modification to the debate orchestrator
- Wiring into the admission gate, the crux detector, or team selection

## Foreman-gate posture

Per [NEXT_STEPS_CANONICAL.md](../../status/NEXT_STEPS_CANONICAL.md), **the current obligation is operationalizing the proof-first loop, not adding new roadmap scope**. This issue is a **measurement-only experiment** whose entire artifact is a benchmark report. It **must not** carry `boss-ready`, enter the live dispatch queue, or be auto-decomposed into implementation work until the proof-first Foreman gate opens.

If the measurement does land and reports significance, the follow-up — wiring a consumer under E5 — still sits behind the same gate.

## Related

- [Track E1 in ARAGORA_EVOLUTION_ROADMAP.md](../../plans/ARAGORA_EVOLUTION_ROADMAP.md)
- [Biological-timescale analogies brief (non-canonical framing)](../../research/2026-04-18-biological-timescale-analogies-brief.md)
- [NEXT_STEPS_CANONICAL.md proof-first gate](../../status/NEXT_STEPS_CANONICAL.md)
