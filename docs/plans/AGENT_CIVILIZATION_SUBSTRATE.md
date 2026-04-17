# Aragora as Substrate for Agent Civilization

> **Status:** additive long-horizon vision synthesis — extends but does not replace [CANONICAL_GOALS](../CANONICAL_GOALS.md), [ARAGORA_EVOLUTION_ROADMAP](ARAGORA_EVOLUTION_ROADMAP.md), [EPISTEMIC_CI_AND_CRUX_ENGINE](EPISTEMIC_CI_AND_CRUX_ENGINE.md), and [COMMERCIAL_OVERVIEW](../COMMERCIAL_OVERVIEW.md).
> **Created:** 2026-04-17
> **Queue policy:** planning truth only. None of the AGT-* gates below may carry `boss-ready` until the proof-first Foreman gate permits the upper-layer tranche, in line with the existing rule that delays `DIC-13..22`.

## Why this document exists

CANONICAL_GOALS already names crux-finding, cryptographic receipts, the terrarium-not-organism doctrine, and the Tool→Teammate→Foreman→Chief-of-Staff→Organization-Substrate stage model. The Evolution Roadmap already names Track E (Decision Integrity Core) including E5 (Crux Engine, Epistemic CI, Epistemic Runtime). EPISTEMIC_CI_AND_CRUX_ENGINE.md plans DIC-13..22 in detail.

What is **not yet explicit** in those documents:

1. The long-horizon thesis that Aragora is the **substrate for agent civilization**, not only for human-operated organizations.
2. The treatment of **agents as first-rate consumers** alongside humans, with their own registration, capability discovery, billing, receipts, and reputation surfaces.
3. The mechanism by which **skin-in-the-game accountability** turns the existing `aragora/blockchain/contracts/{identity,reputation,staking,validation}.py` primitives into a live reputation flow tied to resolved cruxes and time-bounded predictions.
4. How **external truth oracles** (prediction markets, public verifiable streams, synthetic GitHub markets) supply the ground-truth signal the reputation flow needs without requiring a sales motion.
5. A **productivity metric** to replace empty-queue stability soaks once substrate stability is proven.
6. **Capability checkpoints** for the booster-rocket thesis so the meta-system is required to graduate from substrate to surface.

This document captures those six extensions and sequences them as a vision-layer planning track (`AGT-01..AGT-06`) that runs in parallel to the substrate-first execution gate.

## Thesis

The decision-grade autonomy substrate Aragora is building — heterogeneous-model debate, cryptographically attested receipts, verifiable provenance, fail-closed admission, ledger-backed truth, crux detection — is a useful platform for human-operated organizations. It is also the missing platform for an emerging population of software agents that need to make consequential, auditable, time-bounded decisions on each other's behalf and on humans' behalf.

The model labs will build coding agents and personal assistants. They are unlikely to build:

- multi-agent crux resolution that surfaces the load-bearing disagreement
- cryptographically attested decision provenance with later-settlement preservation
- skin-in-the-game agent reputation tied to objectively verifiable outcomes
- a consumer surface that lets agents register, stake, transact, and earn reputation
- an environment whose physics rewards evidence, verification, and explicit dissent over fluent assertion

That is the layer Aragora is uniquely positioned to occupy. It is also the layer that compounds: as the substrate stabilizes and the verifiable-outcome corpus grows, the calibration data on every participating agent becomes itself a moat.

## Why this is compatible with the substrate-first gate

The current execution obligation in NEXT_STEPS_CANONICAL is to make bounded unattended execution boring before widening claims. That obligation is **not** in tension with this vision. Three compatibility points:

1. **Substrate is foundation, not competitor.** Boss loop, merge arbiter, queue governance, ShiftLedger, proof-first publication, BC-12 soaks — these are the load-bearing primitives that everything in this document depends on. None of the AGT-* work would function without them.
2. **Planning truth is allowed; live queue scope is not widened.** The same rule that governs DIC-13..22 (open issues, design work, schema definition allowed; no `boss-ready` until permitted) applies here. The proof-first reconciler will continue to strip `boss-ready` from any AGT-* issue until the gate explicitly opens.
3. **The vision sharpens the substrate metric.** A move from "did the controller crash for 12 hours?" to "did the system produce N verifiable improvements per agent-hour without rescue?" is itself a tightening of the substrate gate, not a relaxation. Empty-queue stability proves the meta-system did not regress; verifiable productivity proves the meta-system is paying for its own existence.

## What the vision adds (beyond what is already canonical)

### 1. Agents as first-rate consumers

The product surface today assumes humans are the consumer. The substrate is general enough to serve agents directly, but the registration, capability discovery, billing, receipt, and reputation surfaces still need explicit agent-readable shapes.

Concretely:

- agent registration through the existing identity primitives (`aragora/blockchain/contracts/identity.py`)
- capability discovery through the A2A protocol (`aragora/protocols/a2a/`) plus marketplace catalog (`aragora/marketplace/`)
- billing primitives that meter compute, debate cost, and verifier cost per agent
- agent-readable decision receipts (machine-parseable, signature-verifiable) emitted by the existing `aragora/export/decision_receipt.py` path
- reputation read/write surfaces that read from and write to the ERC-8004 reputation registry (`aragora/blockchain/contracts/reputation.py`)

This work is detailed in [AGENT_CONSUMER_SURFACE](AGENT_CONSUMER_SURFACE.md).

### 2. Skin-in-the-game accountability

The reputation primitives exist. The flow that turns resolved outcomes into reputation deltas does not.

The unified flow is:

```
claim or stance → stake (compute, capital, or reputation) → resolution event → settlement → reputation update → dispatch eligibility update
```

Each stage maps to existing modules:

- **claim/stance:** `aragora/reasoning/claims.py`, `aragora/reasoning/crux_detector.py`, Arena debate output
- **stake:** `aragora/blockchain/contracts/staking.py`, `aragora/blockchain/compute_budget.py`
- **resolution event:** Manifold/Metaculus/synthetic-market resolution, `aragora/blockchain/receipt_settlement.py`
- **settlement:** existing `Time Is Part of Settlement` doctrine in CANONICAL_GOALS
- **reputation update:** `aragora/blockchain/contracts/reputation.py`, `aragora/knowledge/mound/adapters/erc8004_adapter.py`
- **dispatch eligibility:** debate `team_selector`, calibration tracker, ELO

This work is detailed in [SKIN_IN_THE_GAME_REPUTATION](SKIN_IN_THE_GAME_REPUTATION.md).

### 3. External truth oracles via prediction markets

Without design partners, the reputation flow has no external ground-truth source. The system would optimize agents on agreement with each other, which is a closed loop and a doom-loop risk.

Public verifiable streams supply external ground truth without requiring a sales motion:

- **Manifold Markets** — play money, full API, zero regulatory exposure, designed for bots
- **Metaculus** — calibration tracking, public API, no trading mechanics
- **Synthetic GitHub markets** — predict PR merges, issue resolutions, CI outcomes
- **Kalshi** — graduate to real-money markets only after calibration is stable
- **Polymarket / Augur / Limitless** — defer; regulatory weather varies

This work is detailed in [2026-04-17-prediction-market-validation](2026-04-17-prediction-market-validation.md).

### 4. Productivity metric replacing empty-queue idle soaks

Empty-queue BC-12 soaks prove the controller did not crash. They do not prove the system produces value. Once the substrate is stable enough to sustain itself, the gating metric should shift to something the booster-rocket thesis is required to lift.

Proposed metric: **verifiable improvements per agent-hour (VIAH)**.

```
VIAH = (
    (merged_autonomous_prs * 1.0)
    + (cruxes_correctly_detected_pre_resolution * 0.5)
    + (predictions_resolved_above_brier_threshold * 0.5)
    - (rescues_required * 0.5)
    - (failed_claims_promoted_without_repair * 1.0)
) / agent_hours
```

VIAH is computed weekly, persisted to ShiftLedger, and surfaced through the existing operator-truth path. It supplements, not replaces, the no-rescue success rate from TW-02. The substrate is paying for itself when VIAH trends up week-over-week without operator babysitting.

### 5. Capability checkpoints for the booster-rocket thesis

The booster-rocket investment in autonomous infrastructure is defensible if and only if the boosters actually lift. Without explicit checkpoints, scaffolding work becomes open-ended and self-reinforcing.

Proposed checkpoints (target dates relative to gate-open):

| Checkpoint | Window | Pass condition | Action if not met |
|---|---|---|---|
| **CP-1: Sustained substrate** | gate-open + 4 weeks | 3 consecutive green BC-12 soaks (1 idle + 2 productive); no LaunchAgent respawn-failure incidents requiring human kickstart | Pause AGT-* planning work; debug substrate self-healing |
| **CP-2: Live crux activation** | CP-1 + 4 weeks | CruxDetector emits ranked CruxSet on >=20 real debates per week; at least one CruxSet linked to a follow-up issue or claim | Reduce AGT scope to crux-only and re-evaluate |
| **CP-3: External truth signal** | CP-2 + 4 weeks | Manifold + Metaculus integration produces >=100 resolved predictions per agent per week; calibration curve is stable | Defer reputation wiring; debug prediction pipeline |
| **CP-4: Reputation flow live** | CP-3 + 4 weeks | At least one agent has reputation delta drive a real dispatch-eligibility change in production | Reduce to read-only reputation surface; revisit policy |
| **CP-5: Productivity-positive** | CP-4 + 4 weeks | VIAH trends positive over rolling 4-week window without operator rescue spike | Pause new boosters; consolidate existing layers |

Failing any checkpoint does not kill the vision. It downscales the next investment until the underlying booster is proven.

### 6. Wire-vs-shelve discipline (preserves existing breadth)

The user's preference is integrate over cut. The risk is that 230+ features bit-rot. The compromise is explicit classification per subsystem rather than deletion:

- **(A) Substrate path** — actively maintained, on the critical path to the self-improving organism
- **(B) Showcase application** — demonstrates the platform end-to-end (Inbox Trust Wedge, autonomy benchmarks, prediction-market validation, A2A consumer surface)
- **(C) Shelved with revisit pointer** — preserved in repo, maintenance suspended, README link to the doc that will revive it

`docs/STRANDED_FEATURES_AUDIT.md` is the existing inventory; this discipline extends it. Nothing is deleted; bit-rot risk is contained by suspending maintenance with explicit revival criteria.

## AGT-* sequencing (vision-layer planning track)

These codes mirror the existing `CS-*`, `BC-*`, `RS-*`, `DIC-*` taxonomy. They are planning truth, not live queue scope. The proof-first reconciler MUST strip `boss-ready` from AGT-* issues until queue governance permits this tranche, exactly as it does for `DIC-13..22`.

| Code | Title | Depends on | First milestone |
|------|-------|-----------|-----------------|
| `AGT-01` | Activate CruxDetector in live Arena debates | DIC-15 (#6025), Issue [#6035](https://github.com/synaptent/aragora/issues/6035) | CruxDetector emits ranked CruxSet on production debate path under flag |
| `AGT-02` | A2A consumer surface (registration, capability discovery, billing, agent receipts) | existing `aragora/protocols/a2a/`, `aragora/marketplace/` | Agents can register, discover capabilities, transact, and consume receipts via documented A2A endpoints |
| `AGT-03` | Manifold integration with rolling Brier scoring | AGT-02 | Aragora agents predict on Manifold, store predictions/resolutions, emit per-agent Brier scores |
| `AGT-04` | Synthetic GitHub prediction markets | none (internal) | Internal market predicts PR merges and issue closures with verifiable resolution within 30 days |
| `AGT-05` | ERC-8004 reputation flow wiring (claims→predictions→resolution→reputation→dispatch) | AGT-03, AGT-04, DIC-16, existing `aragora/blockchain/contracts/reputation.py` | Resolved prediction or crux outcome produces a reputation delta visible to team_selector |
| `AGT-06` | Verifiable improvements per agent-hour (VIAH) metric | RS-10 ShiftLedger | Weekly VIAH computed, persisted, and surfaced through operator-truth path |

Sequencing rule: AGT-01 and AGT-04 may proceed in parallel (both internal, no external dependency). AGT-02 and AGT-03 may proceed in parallel after AGT-01 lands. AGT-05 requires both AGT-03 and AGT-04. AGT-06 requires AGT-05 to be meaningful.

## What this document does NOT change

- The current gate stays the proof-first substrate gate in NEXT_STEPS_CANONICAL.
- `boss-ready` queue rules are unchanged.
- DIC-13..22 sequencing is unchanged; this document layers on top of them.
- BC-12 soak policy is unchanged in the short term; AGT-06 (VIAH) is the eventual replacement, not an immediate one.
- External claims still must lag measured proof.
- The terrarium-not-organism doctrine stands. Agents earning reputation in this substrate are operating inside a designed environment, not granted autonomy beyond it.

## Tempering principles to remember

These are decision aids when the substrate-vs-surface tension flares up:

1. **Substrate must graduate to surface.** Permanent scaffolding is a tar pit. Each booster needs a checkpoint where it is required to produce downstream value.
2. **External truth oracles are non-negotiable.** A reputation system without external resolution is agents agreeing with agents. Manifold/Metaculus/synthetic markets supply the bite.
3. **Wire-vs-shelve over delete.** The 230+ features are an asset only if classified. Premature deletion is also a failure mode.
4. **Platform with showcases, not platform alone.** Inbox Trust Wedge, autonomy benchmarks, A2A consumer surface, prediction-market validation are the showcases that prove the platform works without narrowing it to one vertical.
5. **Agents and humans as co-equal consumers.** Every consumer surface (registration, billing, receipts, reputation) ships in agent-readable and human-readable form.
6. **Time is part of settlement.** Receipts preserve dissent and assumptions for delayed resolution. Reputation updates can be re-opened when later evidence arrives.

## References

- [CANONICAL_GOALS](../CANONICAL_GOALS.md)
- [ARAGORA_EVOLUTION_ROADMAP](ARAGORA_EVOLUTION_ROADMAP.md)
- [EPISTEMIC_CI_AND_CRUX_ENGINE](EPISTEMIC_CI_AND_CRUX_ENGINE.md)
- [NEXT_STEPS_CANONICAL](../status/NEXT_STEPS_CANONICAL.md)
- [COMMERCIAL_OVERVIEW](../COMMERCIAL_OVERVIEW.md)
- [WHY_ARAGORA](../WHY_ARAGORA.md)
- [SELF_IMPROVING_ARAGORA](SELF_IMPROVING_ARAGORA.md)
- [AGENT_CONSUMER_SURFACE](AGENT_CONSUMER_SURFACE.md)
- [SKIN_IN_THE_GAME_REPUTATION](SKIN_IN_THE_GAME_REPUTATION.md)
- [2026-04-17-prediction-market-validation](2026-04-17-prediction-market-validation.md)
- [2026-04-16-crux-mode-design](2026-04-16-crux-mode-design.md)
- Existing primitives: `aragora/blockchain/contracts/`, `aragora/protocols/a2a/`, `aragora/reasoning/crux_detector.py`, `aragora/marketplace/`
