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
