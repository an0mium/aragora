# Canonical Model Catalog

**Status:** Phase 1 (enforcement without rewiring) — adjudicated on the
#9073/#9075 review record, 2026-07-16.

## Why

The #9073 and #9075 reviews empirically demonstrated that model identity and
pricing were duplicated across **eleven runtime tables** (model pins, model
selector, routing provider config, billing usage, services metering, debate
costs, pdb invoker, two agent fallback maps, server cost estimation, billing
optimizer tiers). Adversarial review discovered the drift one table per round
— including **three live provider reprices caught mid-review** (gpt-5.5
$2.50/$10 → $5/$30; qwen3.7-max $1.25/$3.75 → $1.475/$4.425; kimi-k2.7-code
$0.72 → $0.75). Review rounds are the most expensive drift detector the
project owns; this catalog makes the detection mechanical.

## The design (three rules)

1. **One typed source.** `aragora/models/catalog.py` defines `ModelSpec`
   (canonical/direct/OpenRouter ids, aliases, USD-per-MTok pricing, context
   and output limits, release + soak dates). `by_any_id()` resolves every
   accepted spelling.
2. **Offline validation, advisory liveness.** Required CI never calls the
   network: it validates against the committed
   `aragora/models/catalog_snapshot.json`. `scripts/model_catalog_drift.py`
   is the advisory live-vs-snapshot differ (`--refresh` rewrites the
   snapshot for a reviewed commit). A scheduled advisory workflow may invoke
   it, but it must never gate a PR on live-catalog reachability.
3. **Governance stays out.** Quorum-family *eligibility* — which model may
   produce merge-authority evidence — lives in
   `aragora/swarm/quorum_evidence.py` under Tier-4 control, never in the
   catalog. The catalog knows prices and soak dates; it does not decide
   authority. (`soak_until` records the 14-day availability rule so
   reviewers/tools can check it; enforcement of the rule remains policy.)

## Phase 1: enforcement without rewiring

No runtime table changed behavior. `tests/models/test_catalog.py` asserts
that every existing table row for an **enforced** model matches the catalog
(`ENFORCED_MODELS`, currently the eight models verified live this week).
A drifting mirror now fails tests in seconds instead of consuming an
adversarial review round.

Known-stale legacy rows (e.g. deepseek-v4-pro at $1.74/$3.48 vs live
$0.435/$0.87; qwen3-max at $0.60/$1.80 vs live $0.78/$3.90) are deliberately
NOT enforced yet: each needs adjudication (some rows may intentionally model
non-OpenRouter direct pricing) before entering `ENFORCED_MODELS`. Add a
model to enforcement by cataloging it and fixing every mirror in the same
commit — the tests tell you exactly which rows disagree.

## Phase 2 (planned, separate PRs)

Consumers migrate to projections generated from the catalog: billing tables
become derived structures, agent fallback maps derive their targets from
`openrouter_id`, and the cost-estimation/optimizer maps import instead of
duplicating. Each migration is a bounded PR gated by the phase-1 tests.

## How to add or reprice a model

1. Edit `aragora/models/catalog.py` (one `ModelSpec`).
2. Run `python3 scripts/model_catalog_drift.py --refresh` and commit the
   snapshot diff (live verification receipt).
3. Run `python3 -m pytest tests/models/` — it lists every mirror row that
   must change; change them in the same commit.
4. New model on a merge-authority surface? Check `soak_until` and the
   governance pins first.
