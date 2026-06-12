# Leverage & Waste Status

<!-- leverage-managed:begin -->
Last updated: 2026-06-10T21:33:55Z

Repo-tracked recurring publication surface for the steering-leverage program
(Operating Plan v2, Phase 0.2): leverage ratio (LR) and waste ratio together.

## Leverage Ratio (LR)

<!-- leverage-lr:begin -->
| Metric | Value |
| --- | --- |
| Window | 2026-06-10T00:00:00Z -> 2026-06-10T21:33:26Z (7d) |
| Merged PRs in window (total) | 32 |
| Merged PRs receipt-backed (verified) | 29 |
| Unique verified receipts backing them | 50 |
| Split factor (receipt-backed PRs / unique receipts; >1 = splitting) | 0.58 |
| Receipts failed verify | 0 |
| Receipt refs unresolved locally | 0 |
| Operator minutes (self-reported) | 25.0 |
| Leverage ratio (verified merged outcomes / operator-minute) | 1.16 |
| Steering integrity (SI) | null — pending: crux_shown, within_attention_budget, not_reversed_on_audit |
| Methodology version | 1 |

Operator-minutes note: baseline day estimate ~25 min: approval messages + design-review read + queue replies
<!-- leverage-lr:end -->

## Waste Ratio (Work-Loss Accounting)

<!-- leverage-waste:begin -->
| Metric | Value |
| --- | --- |
| Window (produced/closed units) | 2026-06-10T00:00:00Z -> 2026-06-10T21:24:37Z |
| Branches pushed, never PR'd | 238 |
| Outbox items expired unpublished | 21 |
| Outbox items lost, never pushed | 355 |
| PRs closed unmerged (window) | 9 |
| Lost units (deduplicated) | 623 |
| Produced units (merged PRs in window) | 32 |
| Waste ratio (lost_units / max(1, produced_units)) | 19.47 |
| Outbox items scanned / unreadable | 869 / 0 |
| Methodology version | 1 |

Unit definitions:
- `branches_pushed_never_prd`: Non-protected branch present on origin that has never been the head ref of any PR and is not already claimed by an outbox loss category.
- `outbox_expired_unpublished`: Outbox item (live or archive) whose expires_at has passed and which never reached a published state (no explicit publication marker, no PR for its branch, not marked already-satisfied).
- `outbox_lost_never_pushed`: Unpublished outbox item whose branch never reached origin — the work exists (or existed) only locally.
- `prs_closed_unmerged`: PR closed without merge, with closed_at inside the window.
- `produced_units`: PRs merged inside the window (merged_at >= window start).
- `lost_units`: Sum of the four loss categories after de-duplication by unit key (branch name when present, else outbox idempotency key); each lost unit counts in exactly one category.
- `waste_ratio`: lost_units / max(1, produced_units).
<!-- leverage-waste:end -->

## Caveats (honest limits of these numbers)

- **Operator-minutes are self-reported.** There is no attention capture yet;
  the denominator is an operator estimate passed explicitly on the CLI, and
  the script refuses to run without it.
- **Steering Integrity (SI) is not yet instrumented.** It is published as
  `null` — never a number — until crux_shown, within_attention_budget, and
  not_reversed_on_audit are actually captured.
- **Receipt linkage is text-based.** A PR counts as receipt-backed when its
  body/comments reference a receipt path that exists locally and verifies;
  this can undercount (receipts not referenced) or be gamed by splitting work
  into more PRs — merged_total, unique_receipts_backed, and split_factor are
  published alongside so splitting is visible, not hidden.
- **Waste units are defined in the waste table above**; categories are
  de-duplicated by unit key (branch name, else outbox idempotency key) so a
  unit of lost work is counted at most once.
<!-- leverage-managed:end -->

## Blind-Period Log

Manual entries below are preserved across re-renders; the publisher never
touches text outside the managed region above.

- 2026-05-27 -> 2026-06-05: automation publisher dead / loop blind
  (source: `.aragora/run-20260610/OPERATOR_STEERING_AUDIT.md`).
- Baseline reference (2026-06-10 outbox harvest, the manual prototype of the
  waste instrument): 134 lost-never-pushed, 45 expired, 37 PRs recovered from
  254 triaged items. The instrumented numbers above scan a wider corpus
  (live + nested archive + archive dirs, 869 items) and so differ from the
  harvest snapshot.

## Queue Composition (substrate cap)

Sprint 4 goal 4 acceptance: the before/after queue-composition measurement of
the substrate cap (cap shipped in #8095; default `ARAGORA_SUBSTRATE_CAP=0.3`
live in `scripts/run_boss_cycle.sh`). The cap is the substrate-cap pattern
applied to the issue generator: it bounds substrate/tooling candidates to at
most `int(max_issues * cap)` per refill so product-surface work is never
crowded out by self-referential loop tooling.

### Measurement basis — SYNTHETIC (live generator yields 0 under proof-first filter)

**Honesty caveat (required reading).** A live before/after composition from
real candidates was **not obtainable** in this window. Running the generator in
dry-run over the real candidate set on 2026-06-11 yields **0 valid candidates**
at *both* cap settings, because the proof-first canonical-priority filter blocks
the queue upstream of the cap:

```
$ python scripts/generate_boss_issues.py --repo synaptent/aragora \
        --dry-run --max-issues 20 --substrate-cap {1.0,0.3}
  Found 75 candidates across 4 categories
Filtered to 0 valid candidates
  Skipped: 0 duplicates, 3 PR conflicts, 72 canonical priority blocks, 0 validation failures
DRY RUN — would create 0 issues:
```

The cap operates *after* this filter, so with 0 candidates reaching it the cap
has nothing to compose either way — identical (empty) output at cap 1.0 and cap
0.3. This is the same condition observed 2026-06-10. The numbers below are
therefore **synthetic**: they are the cap's *mathematical effect* on a
controlled candidate set, taken directly from the proven unit behavior
(`tests/scripts/test_generate_boss_issues.py::TestSubstrateCap`,
`select_with_substrate_cap`), **not** a live composition of real candidates.

### Before / after composition (synthetic; candidate set = 10 substrate + 10 product, max_issues=10)

| Setting | Substrate selected | Product selected | Substrate skipped | Queue composition |
| --- | --- | --- | --- | --- |
| Cap **disabled** (`--substrate-cap 1.0`) | 10 | 0 | 0 | 100% substrate / 0% product |
| Cap **active** (`--substrate-cap 0.3`) | 3 | 7 | 7 | 30% substrate / 70% product |

Reading: under the same candidate pressure, the default cap converts a queue
that *would* be 100% substrate (substrate candidates listed first, filling all
10 slots) into a 30/70 substrate:product split — the `int(10 * 0.3) = 3`
substrate budget, with the 7 freed slots filled by product work and the 7
excess substrate candidates reported as skipped (never silently dropped).

Verified by unit (7/7 passing, 2026-06-11):
`pytest tests/scripts/test_generate_boss_issues.py -k "cap or substrate"`
- `test_cap_limits_substrate_and_product_fills_rest`: 10s+10p @ cap 0.3 → 3 substrate / 7 product, 7 skipped.
- `test_only_substrate_candidates_respects_budget_and_reports_skips`: 10s @ cap 0.3 → 3 selected, 7 skipped.
- `test_cap_of_one_disables`: cap 1.0 → no skips (cap off).
- `test_product_never_skipped_by_cap`: product candidates are never capped.

**What this measurement claims, and what it does not.** It claims the cap's
selection logic produces the stated 30/70 composition under controlled
candidate pressure, proven by unit. It does **not** claim a live refill was
observed composing real candidates this way — none occurred, because the
proof-first filter currently empties the queue before the cap runs. When a real
refill next reaches the cap with mixed candidates, this section should be
updated with the observed live composition replacing the synthetic basis.

<!-- cost-per-settled-pr:begin -->
## Cost per settled PR (#8233 phase 1)

Last updated: 2026-06-12T02:15:35Z

| Metric | Value |
| --- | --- |
| Window | 2026-06-05T02:15:35Z -> 2026-06-12T02:15:35Z (7d) |
| Settled (merged) PRs in window | 154 |
| Settled PRs with attributed cost record | 0 (0% coverage) |
| Attributed recorded cost (USD) | 0.0000 |
| Unattributed recorded cost (USD, receipts) | 0.0000 |
| Total recorded model cost (USD) | 0.0000 |
| Recorded cost per settled PR | $0.0000 (lower bound; see coverage) |
| Routing records in window (with / without cost) | 0 (0 / 0) |
| Receipts scanned (with cost / without / no timestamp) | 120 (0 / 120 / 0) |
| Methodology version | 1 |

**Coverage caveat (required reading).** Only *recorded* model cost is summed —
routing-rationale records with `cost.recorded: true` and receipts carrying a
`cost_summary`. Settled PRs without any cost record are NOT estimated; they are
counted in the denominator and disclosed as uncovered, so the ratio is a lower
bound on true cost. Unattributed receipt costs (receipts do not carry PR
numbers) are included in the total and disclosed separately. Recording starts
with #8233 phase 1; coverage is expected to grow from near zero.
<!-- cost-per-settled-pr:end -->
