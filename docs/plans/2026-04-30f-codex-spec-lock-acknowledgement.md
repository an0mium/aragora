# Round 30f — Codex/GPT Spec-Lock Acknowledgement Contract

*Audience:* Codex/GPT (the δ #6375 implementation agent).
*Author:* Factory/Claude, Round 30f planning lane.
*Status:* CONTRACT — to be acknowledged before δ implementation begins.

---

## Purpose

Round 30f's lane δ closes the only outstanding H1 thesis-gap issue (#6375) with empirical baseline measurement. The work involves seven judgment calls that, if made ad-hoc inside the implementation PR, would not be reviewable. This contract names the seven calls explicitly so they are settled *before* code starts, not embedded in commits.

This is the same discipline the project applies to feature flags: pre-register the gate, do not invent it after seeing the data.

---

## What Codex/GPT must acknowledge

Comment on PR `<planning-PR-URL>` (or on `.aragora/evolve-round/2026-04-30f/round-receipt.json` if planning PR is not yet open) with one of:

### Option (a) — Acknowledge as written

```
spec-acknowledged: rules 2.1–2.7 of docs/plans/2026-04-30f-round-spec.md as written.
no exceptions.
will implement against the spec verbatim.
```

### Option (b) — Acknowledge with named exceptions

```
spec-acknowledged: rules 2.1–2.7 of docs/plans/2026-04-30f-round-spec.md, except:
  - rule X.Y: I propose <revision>, because <reason>.
  - rule W.Z: I propose <revision>, because <reason>.
will pause implementation until operator resolves the named exceptions.
```

### Option (c) — Reject

```
spec-rejected: <reason>.
will not implement δ on the current spec.
```

Operator + planning lane resolve any (b) or (c) before implementation begins. **No silent drift.**

---

## What "implement against the spec verbatim" means

The seven judgment calls in §2 of `docs/plans/2026-04-30f-round-spec.md` are *binding* on lane δ's implementation. Concretely:

### §2.1 — Five canonical signals

Lane δ does NOT add new signals. The candidate-signal collection point (`InvalidationCandidate.unclassified_signals`) is the only place new candidates may be recorded; they remain inert until a future round formally adds them to `INVALIDATION_SIGNALS`.

### §2.2 — Authoritative event sources

Lane δ scans only the four named sources, in the named order:
1. `.aragora/overnight/boss_metrics.jsonl` (verified: 406 rows present, 2026-04-30).
2. `.aragora/review-queue/briefs/*.json` (verified: directory present).
3. `.aragora/evolve-round/*/dogfood/unstick-receipts/applied.jsonl` (verified: 1 file present).
4. GitHub PR/issue timeline via `gh api` (read-only; rate-limited; do not fan out at >10 req/s).

If lane δ encounters another candidate source not on this list, it records the source name in the receipt's `unscanned_candidate_sources` field but does NOT scan it.

### §2.3 — Human-settled definition

A decision is human-settled iff:
- a human reviewer left a GitHub `APPROVED` review, OR
- the merge author is not a bot login AND the merge was not via `--admin` bypass.

`admin_merge_allowed` is auto-handled. `merge --admin` is auto-handled. `merge --auto` is auto-handled. The bot-login deny-list is exactly: `factory-droid[bot]`, `github-actions[bot]`, `dependabot[bot]`, `claude-code[bot]`, `codex-cli[bot]`, plus any other login matching `*[bot]`.

### §2.4 — Under-floor behavior

Lane δ writes `InsufficiencyReceipt.v1` (schema in §3 of the round spec) when `n_human_settled_samples < 50`. Lane δ does NOT update `docs/THESIS.md`. Lane δ does NOT close #6375. The receipt's `recommended_data_collection_delta` field must be populated with concrete actionable text (e.g., "dispatch issues #5126/#5128/#5130 plus 18 more H1-01 staged issues; expected to add ~21 additional human-settled samples within 14 days at current cadence; ETA to floor: 21 days from today").

### §2.5 — Threshold-deviation cases

Pre-registered cases (no post-hoc threshold tweaking):

| Measured rate | Lower-CI bound | Action | Round verdict |
| --- | --- | --- | --- |
| 4.5–5.5% | < 5% < upper | Footnote on Commitment 3 | `δ_pass_5pct_confirmed` |
| 5.5–15% | both above 5% | Replace 5% in Commitment 3 | `δ_pass_threshold_revised` |
| > 15% | lower > 10% | Receipt only; investigate | `δ_pass_high_invalidation_investigate` |
| < 2% | upper < 4% | Receipt only; investigate | `δ_pass_low_invalidation_investigate` |
| 2–4.5% or 5.5% with low CI | (border cases) | Footnote on Commitment 3 | `δ_pass_5pct_confirmed` |

The 4.5%/5.5%/15%/2% thresholds are pre-registered. Lane δ does NOT pick "the closest case" — it picks the case whose bounds the measurement actually satisfies.

### §2.6 — Confidence interval

Wilson score 95% CI on `n_invalidated_human_settled / n_total_human_settled`. Use `statsmodels.stats.proportion.proportion_confint(..., method='wilson')` if available; else hand-roll the formula:

```
n = total_human_settled
p_hat = invalidated_human_settled / n
z = 1.96
denom = 1 + z**2 / n
center = (p_hat + z**2 / (2*n)) / denom
spread = z * sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
ci = (center - spread, center + spread)
```

### §2.7 — Receipt determinism

Same inputs → same `receipt_id`. The `receipt_id` is `sha256(canonical_json(body))` where `canonical_json` sorts keys, separates with `(",", ":")` (no whitespace), and excludes the `produced_at` field from the hash. Datetime fields elsewhere are ISO 8601 with UTC `Z` suffix and second precision (no microseconds, no offsets).

---

## What lane δ does NOT do

These are out of scope per §7 of the round spec. Lane δ:

- Does NOT add new invalidation signals.
- Does NOT scan event sources beyond the four named.
- Does NOT include heterogeneous-dialog transcripts as evidence.
- Does NOT change the threshold to anything outside the §2.5 case table.
- Does NOT close #6375 unless the threshold change is well-defined and the data supports it.
- Does NOT update `docs/THESIS.md` unless cases 2.5.1 or 2.5.2 apply.
- Does NOT fan out to GitHub API beyond rate-limited reads of issue/PR timeline for SHAs already discovered locally.
- Does NOT perform any auto-merge.
- Does NOT mutate `aragora/review/invalidation.py` or `aragora/review/threshold_recalibration.py` (these are the existing scaffolding; the new module is `aragora/triage/invalidation_event_source.py`).
- Does NOT ship without unit tests for each of the five signal predicates.

---

## What lane δ MUST do

- Open exactly one PR with the new module + script + receipt + tests + (conditionally) `docs/THESIS.md` amendment.
- Sign the PR body with the round receipt URI (`.aragora/evolve-round/2026-04-30f/round-receipt.json`).
- Tag the PR Tier 2 (≤300 LOC, additive only).
- Reference this spec-lock contract in the PR body via permalink.
- Cross-reference issue #6375 with one of the `δ_*` verdict tags.

---

## Acknowledgement window

Lane δ does not start mutating code until either:
- this contract is acknowledged by Codex/GPT (per the option (a)/(b)/(c) protocol above), OR
- the operator explicitly approves the planning-lane PR (which IS the acknowledgement, with Codex implementing verbatim).

If neither happens within 24h of the planning-lane PR opening, lane δ is descoped from Round 30f. Round 30g picks up #6375 with a fresh spec.

---

## Why this contract exists

Round 30e produced 4 PRs in 12h. Each PR had implementation latitude — there was no spec-lock. That worked because the work was internal-substrate building where ad-hoc decisions were reversible. #6375 is different: it touches `docs/THESIS.md`, the canonical statement of authority. A wrong threshold, or a wrong honest-failure decision, becomes the thesis. The cost of getting it right is trivial (this 4-page contract); the cost of getting it wrong is a thesis-rewrite round.

The contract is also an explicit hand-off ritual between agents. Round 30e proved that heterogeneous agents can work in parallel; Round 30f proves they can also adhere to a written contract before execution. Both are necessary properties for the Aragora substrate to scale.

— Round 30f planning lane (Factory/Claude), 2026-04-30.
