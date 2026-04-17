# Prediction-Market Validation Plan — Manifold, Metaculus, and Synthetic GitHub Markets

> **Status:** vision-layer planning track (`AGT-03` Manifold, `AGT-04` synthetic GH, downstream Metaculus); not boss-ready until queue governance permits the upper-layer tranche.
> **Created:** 2026-04-17
> **Parent:** [AGENT_CIVILIZATION_SUBSTRATE](AGENT_CIVILIZATION_SUBSTRATE.md)
> **Sibling:** [SKIN_IN_THE_GAME_REPUTATION](SKIN_IN_THE_GAME_REPUTATION.md)

## Why this plan exists

The skin-in-the-game reputation flow needs an **external truth oracle**. Without one, the reputation system collapses into agents agreeing with agents — a closed loop with no signal. Public verifiable streams supply the bite without requiring a sales motion.

This plan specifies the venue stack, integration architecture, risk controls, and graduation path for using prediction markets as the primary external resolution source for `AGT-05`.

## Why prediction markets are the right fit

The user's intuition is correct: prediction markets are open-ended, verifiable, time-bounded, and provide real skin-in-the-game without requiring real money in the early phases. The properties that make them appropriate:

| Property | Why it matters |
|---|---|
| Open-ended | Works for any domain reputation we want to build, not just one vertical |
| Verifiable | Resolutions are public, signed, and auditable |
| Time-bounded | Each market has a resolution date, so reputation updates close out |
| Real skin-in-the-game (where elected) | Stakes create real cost for being wrong |
| Adversarial | Other traders apply real pressure, which makes the test meaningful |
| Public APIs | Manifold and Metaculus both expose stable bot-friendly APIs |
| Calibration-native | The whole field is built around scoring rules (Brier, log loss) |

## Why other models warned against this (and why those warnings don't apply here)

Most model-warned objections pattern-match "automated" + "trading" + "real money" → CFTC/SEC/gambling regulatory risk + alignment concerns about autonomous financial agents. Those concerns are legitimate **for real-money trading**. They are not legitimate for the validation-environment use case proposed here, because:

1. The early phases use **only play-money** venues (Manifold mana, Metaculus points).
2. The goal is **calibration measurement**, not profit extraction.
3. Stake sizing and position limits are bounded explicitly so even a bug cannot cause meaningful adverse market impact.
4. Real-money venues are **deferred** until calibration is stable and proper compliance setup is in place.

The signal worth extracting from the warnings: don't move to real-money venues casually; treat the regulatory step as a deliberate graduation event with legal review.

## Venue stack (in order of adoption)

### Phase 1: Manifold Markets (AGT-03 primary)

- **Money model:** play money (mana)
- **Regulatory exposure:** none
- **Why first:** designed for bots from day one, hundreds of bots already trade there, full public API, welcoming community
- **What we measure:** rolling Brier score per agent, calibration curve, market-impact footprint
- **Position limits:** start at 50 mana per market, scale only after observing impact patterns

### Phase 2: Metaculus

- **Money model:** none (forecasting platform with calibration scoring)
- **Regulatory exposure:** none
- **Why second:** pure calibration-focused community, established Brier/log-loss leaderboards, longer-horizon questions complement Manifold's short cycle
- **What we measure:** community-relative calibration, novel-question performance
- **Constraints:** Metaculus does not allow undisclosed bots in some categories; ensure compliance with their bot policy

### Phase 3: Synthetic GitHub markets (AGT-04, runs in parallel from start)

- **Money model:** internal (compute credits via `aragora/blockchain/compute_budget.py`)
- **Regulatory exposure:** none (internal)
- **Why parallel:** fully owned, fastest cycle, no platform dependency
- **Question shapes:**
  - Will PR #X merge in 7 days?
  - Will issue #Y close within 30 days?
  - Will CI pipeline #Z pass on first run for a given branch?
  - Will a given OSS dependency release within target window?
- **Resolution:** automatic via GitHub API + receipt-anchored settlement
- **What we measure:** high-volume calibration on a controlled corpus, complementing the lower-volume external venues

### Phase 4 (deferred): Kalshi

- **Money model:** real (USD)
- **Regulatory exposure:** CFTC-regulated, requires entity setup, KYC, legal review
- **When:** only after Phase 1-3 calibration is stable, the reputation flow is proven, and there is a deliberate decision to move to real-money signal
- **Compliance notes:** Kalshi requires US-resident users; entity-level participation requires onboarding process

### Phase 5 (deferred indefinitely): Polymarket / Augur / Limitless

- **Money model:** crypto (USDC/etc)
- **Regulatory exposure:** geo-restricted in US, regulatory weather changes frequently
- **When:** not in the visible horizon; revisit if regulatory landscape stabilizes

## Architecture

```
┌──────────────────────┐    ┌─────────────────┐    ┌──────────────────────┐
│ Aragora prediction   │--->│ Venue API       │--->│ Position taken       │
│ engine (per agent)   │    │ adapter         │    │ Receipt anchored     │
└──────────────────────┘    └─────────────────┘    └──────────────────────┘
         │                                                    │
         │  reads CruxSet, Arena debate, KM evidence          ▼
         ▼                                          ┌──────────────────────┐
┌──────────────────────┐                            │ Resolution ingestion │
│ StakeableClaim       │<────────────────────-------│ ResolutionEvent      │
│ schema (AGT-05)      │                            │ ↓                    │
└──────────────────────┘                            │ Reputation Δ (AGT-05)│
                                                    └──────────────────────┘
```

Each venue gets one adapter under `aragora/connectors/prediction_markets/`:

- `manifold.py` — Manifold API client + market discovery + position taking
- `metaculus.py` — Metaculus API client + question discovery + prediction submission
- `synthetic_github.py` — internal market creation + GitHub API resolution

All adapters emit a unified `MarketPosition` event consumed by AGT-05 settlement.

## Stake sizing and risk controls

### Manifold

- per-market position cap: 50 mana initially, 200 mana after 30 days of stable behavior
- per-day total cap: 1000 mana
- avoid >5% of market liquidity in any single position
- never trade markets with <30 day resolution windows in initial phase (avoid noisy short-cycle calibration)

### Metaculus

- follow Metaculus bot policy strictly; disclosed-bot status only
- one prediction per question per day
- target categories with clear resolution criteria; avoid subjective questions in initial phase

### Synthetic GitHub markets

- per-market position cap: 100 internal credits
- limit to publicly observable repos to keep resolution unambiguous
- expire stale unresolved markets after 90 days

## Calibration metric and reporting

Per agent, per venue, weekly:

| Metric | Definition |
|---|---|
| Brier score (rolling 90d) | Lower is better; floor is 0 |
| Calibration curve | Predicted probability vs. realized frequency, bucketed |
| Resolution count | Number of markets resolved in window |
| Stake-weighted Brier | Brier weighted by position size |
| Market-impact footprint | Average % of market liquidity moved per position |
| Dispute rate | Fraction of resolutions challenged by other traders |

These metrics feed AGT-05 ReputationDelta computation. The reputation delta scoring rule is **proper Brier**, so honest probability reporting is the maximum-reward strategy.

## Resolution lag and short-cycle markets

Many interesting markets resolve months out. To keep the calibration-update loop fast enough to be useful for dispatch decisions, mix in short-cycle markets:

- **Daily:** sports outcomes, weather targets, daily price thresholds
- **Weekly:** scheduled events, weekly OSS releases, weekly economic data
- **Monthly:** PR merges, issue closures, monthly milestones
- **Quarterly+:** elections, scientific replications, long-horizon predictions

The reputation calculation weights short-cycle and long-cycle markets differently to avoid Goodharting on quick-resolving but low-value calibration.

## Selection bias

Easy markets reward narrow agents. To exercise the full debate-to-decision pipeline:

- enforce a quota of "open-ended judgement" markets per agent per week (e.g. Metaculus essay-style questions)
- prevent farming of pure consensus markets where every reasonable trader agrees
- include cross-domain markets so an agent's reputation is not concentrated in a single category

## Adversarial pressure

This is a **feature** for skin-in-the-game testing. Other traders will reverse-engineer agent strategies, copy good predictions, and exploit predictable behavior. The reputation system gets to learn under that pressure. Operationally:

- expect strategy decay over time as patterns are recognized
- treat exploited strategies as a signal that the agent's reasoning was not novel enough
- include red-team agents internally that intentionally counter-trade detected patterns

## Public exposure of agent reasoning

If receipts and reasoning are posted publicly, expect them to be analyzed and copied. This is acceptable for the validation purpose. For competitive lanes (later, real-money), reasoning would be partial-veiled.

## Sequencing

| Code | Step | Deliverable |
|---|---|---|
| AGT-03.1 | Manifold adapter (read-only) | discover markets, fetch resolutions |
| AGT-03.2 | Manifold adapter (write) | submit predictions, anchor receipts |
| AGT-03.3 | Brier score computation per agent | weekly rolling 90d Brier reported |
| AGT-04.1 | Synthetic market schema | internal market objects with resolution criteria |
| AGT-04.2 | GitHub event resolution | automatic resolution against GitHub API |
| AGT-04.3 | Internal credit bookkeeping | stake forfeit/refund through compute_budget |
| Metaculus | Bot-policy compliance + read adapter | discovered after AGT-03.1, before AGT-03.3 |
| Metaculus | Write adapter | optional follow-up to AGT-03.3 |

## What this plan does NOT do

- Does not enable real-money trading.
- Does not move AGT-03/AGT-04 issues into `boss-ready` until queue governance permits the upper-layer tranche.
- Does not introduce a new wallet or settlement layer; uses existing `aragora/blockchain/`.
- Does not bypass venue terms of service; all bot activity must comply with venue policies.
- Does not commit to a Polymarket/Kalshi integration timeline.

## Risks and tempering

- **Venue policy change.** Venues may change bot policies or rate limits. Mitigation: keep adapters thin; design AGT-05 to function across venue switches.
- **Calibration ≠ usefulness.** A perfectly calibrated agent may still be uninteresting. Calibration is necessary but not sufficient; CruxSet quality is the parallel signal.
- **Time sink.** Building rich integrations for low-value venues. Mitigation: each venue must contribute >100 resolved predictions per agent per quarter to remain wired.
- **Reputation laundering.** An agent could farm easy markets to inflate aggregate reputation. Mitigation: per-domain reputation; aggregate is a weighted sum that surfaces the breakdown to operators.

## References

- [AGENT_CIVILIZATION_SUBSTRATE](AGENT_CIVILIZATION_SUBSTRATE.md)
- [SKIN_IN_THE_GAME_REPUTATION](SKIN_IN_THE_GAME_REPUTATION.md)
- [AGENT_CONSUMER_SURFACE](AGENT_CONSUMER_SURFACE.md)
- [EPISTEMIC_CI_AND_CRUX_ENGINE](EPISTEMIC_CI_AND_CRUX_ENGINE.md)
- Manifold API: <https://docs.manifold.markets/api>
- Metaculus API: <https://www.metaculus.com/api/>
- Code: `aragora/blockchain/`, `aragora/connectors/` (new `prediction_markets/` subdirectory), `aragora/reasoning/`
