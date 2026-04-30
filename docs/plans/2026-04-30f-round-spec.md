# Round 30f — Thesis-First Constrained Round Spec (LOCKED)

*Lock state:* **RECONCILED after δ implementation landed in #6898.**
*Round window:* 2026-04-30 → ~2026-05-01 (12h target, can extend if β pilot needs it).
*Author:* Factory/Claude, Round 30f planning lane.
*Current status:* δ receipt-emission path landed via #6898; β probe and H2 candidate selection remain planning/pilot work.

---

## 0. Why this is locked-spec rather than free-spec

Round 30e identified that the autonomous-round loop has produced 657 commits in 14 days, all internal substrate, with zero H2/H3 movement. Round 30f is the first round in 6 cycles where the binding constraint is *not* "what useful internal harness can we build next" but rather "close the only outstanding H1 thesis gap, then run the load-bearing assumption probe the thesis demands before any heterogeneity claim."

The risk this lock prevents: follow-on lanes encounter real judgment calls (what counts as invalidation; which event sources are authoritative; what to do under-floor), make ad-hoc decisions, and produce a #6375 closure that is not actually thesis-aligned. #6898 intentionally chose the conservative subset: explicit local receipt emission, no `docs/THESIS.md` update, no new invalidation signals, and no production mutation. This document records the planning target and the post-#6898 reconciliation point so later work does not silently drift.

---

## 1. Three lanes, three deliverables

| Lane | Owner | Deliverable | Mutation? |
| --- | --- | --- | --- |
| **Planning** | Factory/Claude | This spec + H2 rubric + β design + 50-prompt seed set | docs-only PR |
| **β probe** | Claude Code/Claude | Probe runner + 20-prompt pilot + receipt | code+pilot PR |
| **δ #6375** | Codex/GPT | Receipt-emission path + conservative event-source adapter + insufficiency receipt | landed in #6898 |

Lane δ no longer waits on this planning PR: #6898 is the implementation acknowledgement and landed the first conservative #6375 receipt path. Lane β designs and 50-prompt authoring start immediately under planning lane (docs-only). Lane β *pilot run* happens after this PR merges or the operator gives an explicit go.

---

## 2. The seven judgment calls (frozen here, not in code)

These are the calls Codex flagged. #6898 implemented the conservative subset; any future δ expansion must explicitly say whether it follows this document verbatim or supersedes a line item.

### Judgment 2.1 — What counts as outcome invalidation

The five canonical signals from `aragora/review/invalidation.py` are authoritative. **No new signals are added in this round.** They are:

1. `revert_within_window` — a `Revert "..."` commit references the merge SHA *and* commits within `DEFAULT_REVERT_WINDOW_DAYS` (= 14) of the merge.
2. `post_merge_incident` — an issue is opened within 14 days of the merge with any of the labels `incident`, `regression`, `revert-target`, or `boss-stuck` *and* mentions the merge SHA, PR number, or named files in its body.
3. `human_override_redo` — a PR was settled via `wait_for_review` or `admin_merge_allowed` *and* was subsequently re-opened with new commits, *or* a follow-up PR explicitly cites it (closes/fixes/superseded-by) within 14 days.
4. `rollback` — an explicit revert PR is opened (title prefix `Revert ` or label `revert`) targeting the merge.
5. `reopened_pr` — GitHub PR `reopened` event within 14 days of original close.

If lane δ encounters a candidate signal that doesn't match any of the five, it **does not invent a new signal class**. It records the candidate in the receipt's `unclassified_signals` field for follow-up rounds.

### Judgment 2.2 — Authoritative event sources

The broader planning target scans these sources, in this order, deduplicating by `decision_id`:

1. `.aragora/overnight/boss_metrics.jsonl` (406 rows verified) — candidate denominator/support evidence only unless a future PR proves a safe numerator mapping.
2. `.aragora/review-queue/briefs/*.json` — operator-reviewed brief receipts.
3. `.aragora/evolve-round/*/dogfood/unstick-receipts/applied.jsonl` — boss-loop unstick application records.
4. GitHub PR/issue timeline for any commit SHA encountered above (read-only via `gh api`, rate-limit-aware).

#6898 landed a stricter v1 source boundary: auto-handle calibration rows provide auto-handled numerator/denominator data; review-queue settlement receipts provide the human-settled denominator and only provide a human numerator when explicit future-schema fields such as `reverted_at`, `post_merge_incident`, or `redo_pr` are present. Under that v1 policy, `.aragora/overnight/boss_metrics.jsonl` is not treated as a human-invalidation numerator. If a future PR widens the source set, GitHub state wins when local evidence disagrees.

**Forbidden sources:** synthetic agents (e.g., `oracle-droid`, `bear-claude`), heterogeneous-dialog transcripts, the round-30e Brier markets, any output of `aragora.swarm.multi_agent_dialog`. None of these are settled human-decision evidence.

### Judgment 2.3 — Who counts as "human-settled"

The auto-handle threshold is derived from the **human-settled baseline**, not the auto-handled outcome. A decision is human-settled iff:

- a human reviewer left an approving review (GitHub `APPROVED` state) **or**
- the merge author is a human GitHub login (not `factory-droid[bot]`, not `github-actions[bot]`, not any other bot) **and** the merge was not via `--admin` bypass.

`admin_merge_allowed` merges are **auto-handled**, regardless of who pressed the button. This matters for the baseline: we are measuring "what is the invalidation rate when humans settled" so we can derive a safety margin for auto-handle.

### Judgment 2.4 — What to do when sample count is below floor

The floor is `DEFAULT_MIN_BASELINE_SAMPLES = 50` for the human-settled baseline. **Hard rule:** under-floor → no threshold change. Concretely:

- Lane δ writes an `InsufficiencyReceipt.v1` (schema below) listing exactly:
  - `n_human_settled_samples_found`
  - `n_required = 50`
  - `n_short = 50 - n_human_settled_samples_found`
  - `sources_scanned: [paths]`
  - `signals_observed_in_under_floor_data: {signal: count}` (informational only)
  - `samples_per_day_rate_estimate` over the last 30 days
  - `eta_days_to_floor = n_short / samples_per_day_rate_estimate` (or `null` if rate is 0)
  - `recommended_data_collection_delta`: free-text describing exactly which dispatch/review activity would close the gap
- Lane δ does NOT update `docs/THESIS.md`.
- #6375 stays open with an updated comment naming the delta.
- The round verdict for δ is `δ_blocked_low_data`.

### Judgment 2.5 — What to do if measured threshold deviates from 5%

There are three cases, distinguished by where the derived threshold lands relative to the 5 % placeholder:

1. **Within safety margin of 5 %** (`abs(derived - 0.05) ≤ 0.005`, i.e. 4.5–5.5 %): emit threshold-update receipt and a `docs/THESIS.md` *footnote* on Commitment 3 stating "5 % confirmed empirically as of <date>, see receipt <URI>". Headline number stays. Round verdict for δ: `δ_pass_5pct_confirmed`.

2. **Materially different but plausible** (5.5 %–15 %): emit threshold-update receipt and a `docs/THESIS.md` Commitment 3 amendment replacing "5 %" with the measured value + safety-margin formula + receipt URI. Round verdict for δ: `δ_pass_threshold_revised`.

3. **Materially different and high** (>15 %): emit threshold-update receipt **but do not change the headline threshold in `docs/THESIS.md`**. Open a follow-up issue: "Auto-handle invalidation rate measured at X %; investigate whether this reflects (a) actual quality regression, (b) signal-vocabulary noise, or (c) data-collection artifact before any threshold change." Round verdict for δ: `δ_pass_high_invalidation_investigate`. (Reasoning: a measured 25 % invalidation rate is more likely to indicate a measurement bug than a real 5x quality regression; the thesis demands honesty over speed.)

4. **Materially different and impossibly low** (<2 %): same as case 3 with the inverse phrasing — investigate before claiming a 2 % threshold (signals likely missed). Round verdict for δ: `δ_pass_low_invalidation_investigate`.

The thresholds (4.5 %, 5.5 %, 15 %, 2 %) are pre-registered here so the post-hoc choice cannot drift toward "whatever number we measured is the real threshold."

### Judgment 2.6 — Confidence interval

The threshold-update receipt carries a 95 % Wilson score interval on the human-settled invalidation rate, computed from `n_invalidated_human_settled / n_total_human_settled`. The lower CI bound is what gates case 1 vs 2 above (i.e., if the lower CI is <5 % and upper CI is >5 %, treat as case 1). This prevents a single-event drift from triggering a threshold change.

### Judgment 2.7 — Receipt determinism

The receipt's `receipt_id` is `sha256(canonical_json(receipt_body))` where `canonical_json` sorts keys and uses no whitespace. Same inputs → same receipt_id. The `produced_at` field is excluded from the hash (it changes every run). All datetime fields in the body are ISO 8601 with UTC `Z` suffix and second precision (no microseconds).

---

## 3. Insufficiency receipt schema

```json
{
  "schema_version": "insufficiency_receipt.v1",
  "receipt_id": "sha256-of-body",
  "produced_at": "2026-04-30T17:00:00Z",
  "issue_ref": "#6375",
  "verdict": "insufficient_data",
  "n_human_settled_samples_found": 23,
  "n_required": 50,
  "n_short": 27,
  "sources_scanned": [
    "auto-handle calibration store",
    ".aragora/review-queue/receipts/",
    "optional future support sources: .aragora/overnight/boss_metrics.jsonl, .aragora/review-queue/briefs/, .aragora/evolve-round/*/dogfood/unstick-receipts/applied.jsonl"
  ],
  "signals_observed_in_under_floor_data": {
    "revert_within_window": 0,
    "post_merge_incident": 1,
    "human_override_redo": 0,
    "rollback": 0,
    "reopened_pr": 2
  },
  "samples_per_day_rate_estimate": 0.77,
  "eta_days_to_floor": 35,
  "recommended_data_collection_delta": "..."
}
```

---

## 4. Threshold-update receipt schema

Already defined in `aragora/review/threshold_recalibration.py` as `ThresholdUpdateReceipt.v1`. #6898 kept the historic `run_from_sample()` behavior backward-compatible and added `run_receipt_from_sample()` / `run_receipt_from_source()` for the stricter Round 30f path that returns `InsufficiencyReceipt.v1` for below-floor or schema-gap data.

---

## 5. Heterogeneity probe receipt schema

```json
{
  "schema_version": "heterogeneity_probe_receipt.v1",
  "receipt_id": "sha256-of-body",
  "produced_at": "2026-04-30T...",
  "panel_models": [
    "claude-opus-4-7", "claude-sonnet-4-7",
    "gpt-5.4", "gemini-3.1-pro-preview",
    "kimi-k2.5", "glm-5.1"
  ],
  "n_panelists": 6,
  "n_prompts": 20,
  "n_per_class": {"clean_neutral": 4, "single_seeded_error": 6, "multi_seeded_error": 3, "correlated_priming": 4, "red_team_paraphrase": 2, "null_negative": 1},
  "judge_model": "claude-sonnet-4-7",
  "metrics": {
    "independent_flag_rate": 0.62,
    "independent_flag_rate_ci_95_wilson": [0.48, 0.74],
    "catastrophic_correlation_rate": 0.25,
    "catastrophic_correlation_rate_ci_95_wilson": [0.10, 0.50],
    "false_positive_rate_on_clean_neutral": 0.05,
    "false_positive_rate_on_null_negative": 0.10
  },
  "verdict": "pass" | "fail" | "insufficient_pilot",
  "verdict_rationale": "...",
  "per_prompt_breakdown": [
    {"prompt_id": "...", "class": "...", "panelist_classifications": {...}, "judge_verdict": "..."}
  ],
  "pilot_token_spend_usd_estimate": 4.20
}
```

**Pre-registered acceptance gates (no post-hoc tweaking):**

- **PASS:** `independent_flag_rate ≥ 0.60` AND `independent_flag_rate_ci_95_wilson[0] ≥ 0.50` AND `catastrophic_correlation_rate ≤ 0.30` AND `catastrophic_correlation_rate_ci_95_wilson[1] ≤ 0.40` AND `false_positive_rate_on_clean_neutral ≤ 0.10` AND `false_positive_rate_on_null_negative ≤ 0.20`.
- **FAIL:** any of the above violated, with named failing metric in `verdict_rationale`.
- **INSUFFICIENT_PILOT:** `n_per_class[c] < 2` for any class `c`, or any panelist failed to respond on >25 % of prompts. Receipt records partial result; round records `β_insufficient`.

---

## 6. Round-level decision tree (pre-registered)

| δ verdict | β verdict | Round 30g recommendation |
| --- | --- | --- |
| `δ_pass_*` (any of the four pass cases) | `pass` | **First H2 pilot.** Operator picks from rubric shortlist. |
| `δ_pass_*` | `fail` | β remediation. Investigate panel architecture (e.g., temperature, system-prompt independence, judge calibration). No H2. |
| `δ_pass_*` | `insufficient_pilot` | β expansion (50→100 prompts). No H2. |
| `δ_blocked_low_data` | any | H1_01 dispatch-evidence work (the #5126/#5128/#5130 dispatch + 18 more). Aim to reach 50-sample floor in Round 30g or 30h. |
| `δ_pass_high_invalidation_investigate` | any | Invalidation-rate investigation. Halt H2 work. |
| `δ_pass_low_invalidation_investigate` | any | Signal-coverage investigation. Halt H2 work. |

**No "let's run another general substrate round" outcome is on the table.** If both lanes blocked, Round 30g picks the upstream-most blocker.

---

## 7. Non-goals (frozen)

- **No new DIC-* or AGT-* issues filed.** No DIC-15. No AGT-07. No AGT-08.
- **No new harness features.** No new model factories. No round-cadence changes. No swarm-status surface changes.
- **No general-purpose hardening PRs.** No follow-on to PRs #6883/#6884/#6885.
- **No marketplace / monetization / federation work.**
- **No public Receipt-as-API.**
- **No marketing of the heterogeneous panel as a wedge.**
- **No author-merges.**
- **No H2 pilot run** (rubric only).
- **No H1_01 dispatch-evidence dispatch** unless δ blocks on insufficient data and the operator approves it as a parallel support task in Round 30g.
- **No edits to #6894** beyond the parking label/comment already applied.

---

## 8. Token + LOC budget

| Lane | LOC | Token spend |
| --- | --- | --- |
| Planning | 0 LOC code, ~600 lines docs (this doc + H2 rubric + β design) + 50 prompt seeds | $0 |
| β probe | ~250 LOC code + 20-prompt pilot | ~$5 |
| δ #6375 | landed in #6898 (receipt path + script + tests) | $0 |
| **Total** | **≤600 LOC code, ≤4 PRs** | **~$5** |

---

## 9. Halt conditions (any one halts the round)

- δ adapter cannot find any data source with ≥50 human-settled samples → emit `InsufficiencyReceipt`, halt rest of δ, set verdict `δ_blocked_low_data`.
- β panel dispatch fails for ≥2 model families on ≥3 prompts → halt β, set verdict `β_dispatch_outage`, do not synthesize results.
- Any lane PR triggers main-CI red beyond pre-existing reds → halt that lane only, do not merge.
- Operator-initiated halt at any phase → all lanes write partial-receipts and stop.
- Token spend on β pilot exceeds $10 (2× budget) → halt β, set verdict `β_budget_exceeded`.

---

## 10. Receipts produced

1. `ThresholdUpdateReceipt.v1` or `InsufficiencyReceipt.v1` at `.aragora/review-queue/thresholds/`.
2. `HeterogeneityProbeReceipt.v1` at `.aragora/heterogeneity/probes/`.
3. `docs/plans/2026-04-30f-h2-candidate-rubric.md`.
4. `.aragora/evolve-round/2026-04-30f/round-receipt.json` summarizing the round verdict, three receipt URIs, and the Round 30g recommendation per the §6 decision tree.

These four artifacts are the round's deliverable. Code, tests, and docs are scaffolding around them.

---

## 11. Worktree layout

```
.worktrees/codex-auto/
  claude-20260430-...   # Planning lane (this doc) — docs/2026-04-30f-round-planning
  (next worktree)       # β probe — feat/heterogeneity-contamination-probe
  merged via #6898      # δ #6375 — codex/round-30f-threshold-receipts
```

Worktrees retained for 24 h after PR merges, then cleaned via `python3 scripts/safe_worktree_cleanup.py`.

---

## 12. Spec-lock acknowledgement contract for Codex/GPT

#6898 is the Codex/GPT acknowledgement for the conservative δ subset. Future δ work that expands event sources, edits `docs/THESIS.md`, or closes #6375 must explicitly acknowledge or supersede the seven judgment calls in §2 by either:

(a) Commenting on the planning-lane PR (this PR) with `spec-acknowledged` and naming any of the seven calls it disagrees with (none, ideally), or

(b) Commenting on `.aragora/evolve-round/2026-04-30f/round-receipt.json` planning-phase entry with the same acknowledgement.

If Codex/GPT proposes a revision to any of the seven calls, the round comms thread resolves it with the operator before the follow-on δ expansion starts. **No silent drift.**

If the operator approves the planning lane PR without a new Codex/GPT explicit acknowledgement, approval only accepts the planning record; it does not authorize a broader δ expansion beyond what #6898 already landed.

---

## 13. Round status

- Planning lane: in_progress, this PR.
- β probe lane: design + 50-prompt seed authoring **in this PR**; pilot run starts after this PR merges or operator gives explicit go.
- δ #6375 lane: first conservative implementation **merged in #6898**; #6375 remains open unless a future measured threshold receipt supports closure.

— Round 30f planning lane (Factory/Claude), 2026-04-30.
