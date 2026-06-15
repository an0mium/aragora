# Biological-Timescale Analogies for Aragora Memory and Selection — Research Brief

> **Status:** NON-CANONICAL RESEARCH BRIEF.
> **Authority:** None. This document is framings, not roadmap. It does not bind architecture, create tracks, or add scope to the [canonical roadmap](../plans/ARAGORA_EVOLUTION_ROADMAP.md) or [CANONICAL_GOALS.md](../CANONICAL_GOALS.md).
> **Gate:** Per [NEXT_STEPS_CANONICAL.md](../status/NEXT_STEPS_CANONICAL.md), the current obligation is operationalizing the proof-first loop, not adding new roadmap scope. Any bullet in this brief that graduates into roadmap work must do so by surviving a benchmark artifact and passing through an existing track (D, E, G, or B), not as a new track.
> **Date:** 2026-04-18

## Purpose

This brief captures four biology-flavored framings that came up in product strategy discussion and evaluates whether any of them should survive as **measurable design heuristics** under existing roadmap tracks. It does so by:

1. stating the analogy in its shortest useful form
2. identifying the Aragora primitive it maps onto
3. proposing a measurement that would make the analogy load-bearing rather than decorative
4. naming the existing track that should own any follow-up

If a framing cannot produce a benchmark artifact or a proof surface, it stays in this brief and does not graduate.

## Framings

### 1. Germline / somatic / memetic timescale separation

Biology separates very-slow (DNA), fast-mutable (somatic cells and immune memory), and horizontally-transferable (memetic, cultural) memory. The failure mode the analogy highlights is **silent merging** — a short-lived session artifact informing long-lived retrieval without a receipt.

Aragora primitive: ContinuumMemory, Unified Memory Gateway, 42 KM adapters.

Measurable design heuristic (under Track D): any module that writes across memory tiers must declare a cross-tier consolidation path, and silent merges are treated as a finding in architectural review. This does not require a new tier, a new adapter, or a new subsystem — it is a design invariant that applies to the existing Memory and Context Fabric.

Owner: Track D. Already reflected as a sub-bullet under [D1 Permissioned Memory Model](../plans/ARAGORA_EVOLUTION_ROADMAP.md).

### 2. Red Queen dynamics and strategy decay

Evolutionary systems where a specific edge depends on an opponent's failure mode lose that edge as the opponent learns. Fund-of-funds operators model this as alpha decay: tactical edges are depreciating assets, not evergreen moats.

Aragora primitive: Skin-in-the-Game reputation (G3), selection over exploit strategies.

Measurable design heuristic (under Track G): a reputation slice tied to participation in a specific exploit strategy decays faster once the strategy is measurably crowded or publicly known. The benchmark artifact is a receipt-backed synthetic corpus that shows measured decay behaviour on known-decaying strategies.

Owner: Track G, sub-bullet under [G3 Skin-in-the-Game Reputation Flow](../plans/ARAGORA_EVOLUTION_ROADMAP.md). Not a new track.

### 3. Disagreement among cheap evaluators as a first-class signal

Ensembles exist primarily to error-correct through independent observers. The usual framing of disagreement inside Aragora (hollow-consensus detection, Trickster, dissent tracking) is disagreement-as-problem. The symmetric positive framing is: pre-consensus disagreement is itself predictive data about which questions are hard, which will be overturned at settlement, and which contain a real crux.

Aragora primitive: Debate orchestration, truth-scorer, calibration tracker, crux detector.

Measurable design heuristic (under Track E): measure whether pre-consensus disagreement patterns correlate with later settlement divergence on the existing bounded corpus. If the correlation is real and significant, it becomes a usable feature for crux detection and difficulty scoring. If the correlation is weak or absent, the framing stops being load-bearing.

Owner: Track E, sub-bullet under [E1 Debate Quality and Calibration](../plans/ARAGORA_EVOLUTION_ROADMAP.md). This is a measurement question first; only after a measurable correlation lands does any downstream routing consumer become appropriate.

### 4. Population-under-selection diversity maintenance

Biological and economic populations that optimize greedily on a single fitness metric homogenize, which loads bearing weight on correlated failure modes. The canonical antidote is explicit diversity maintenance, not faith that aggregation will preserve heterogeneity.

Aragora primitive: `team_selector.py` (ELO + calibration).

Measurable design heuristic (under Track G): extend team selection with a measured diversity floor (pairwise disagreement entropy plus provider-family share). The benchmark artifact is a comparison on the same fixed corpus showing that greedy-quality selection produces measurably homogenized failure modes which the diversity floor reduces.

Owner: Track G, sub-bullet under [G5 CruxDetector Activation in Live Debates](../plans/ARAGORA_EVOLUTION_ROADMAP.md). Not a new track.

## Framings that do not survive

The following adjacent framings were explored and do not survive into roadmap sub-bullets. They are documented here so the decision is auditable.

- **"Surrogate distillation as a new track" (NNUE / chess analog).** The useful core survives as a *benchmarked decision policy* under [Track B4 Multi-Host Soak](../plans/ARAGORA_EVOLUTION_ROADMAP.md) — cheap-signal-to-verification routing, evaluated by labelled comparison against the fixed benchmark. A dedicated track would duplicate the existing bounded-autonomy benchmark substrate and violate the proof-first Foreman gate.
- **"Population-under-selection as a new track."** The useful core survives as two narrow sub-bullets (G3 strategy-decay slice, G5 diversity floor). A dedicated track would duplicate G3 and G5.
- **"Multi-timescale memory as a new track."** The useful core survives as the D1 design heuristic above. A dedicated track would duplicate D.
- **Stacked generalization, Complementary Learning Systems, Neural Darwinism, and related ML-history framings.** These are generative prior art for thinking about the above heuristics; they do not, on their own, propose work that is not already covered by the sub-bullets above.

## Rules of the road for this brief

1. Nothing here carries `boss-ready`. Nothing here is canonical. Nothing here adds a track.
2. Any framing that graduates to roadmap work must (a) map onto an existing track, (b) define a benchmark artifact or proof surface as acceptance, and (c) respect the existing delayed-track rule governing `DIC-13..22` and `DIC-23..28` — planning-truth only until the proof-first Foreman gate opens.
3. If a framing proves load-bearing via benchmark artifact, move the surviving bullet into the appropriate track doc and update this brief to point at the landed change. Do not let the brief grow into a parallel roadmap.

## Related

- [CANONICAL_GOALS.md](../CANONICAL_GOALS.md) — 8 pillars; unchanged by this brief
- [ARAGORA_EVOLUTION_ROADMAP.md](../plans/ARAGORA_EVOLUTION_ROADMAP.md) — Tracks A–G; the narrow surviving bullets live there as sub-items
- [NEXT_STEPS_CANONICAL.md](../status/NEXT_STEPS_CANONICAL.md) — the proof-first Foreman gate and `boss-ready` queue governance that bounds when any of these can graduate
