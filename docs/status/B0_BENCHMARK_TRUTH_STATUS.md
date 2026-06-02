# B0 Benchmark Truth Status

Last updated: 2026-06-02T23:56:25Z

This is the repo-tracked recurring `TW-02` publication surface for the fixed benchmark corpus.

## Corpus

- Corpus manifest: `docs/benchmarks/corpus.json`
- Corpus id: `tw-01-bounded-execution-v1`
- Revision: `5`
- Recorded on: `2026-05-28`
- Success contract: `mergeable_pr_or_merged_pr`
- Verified expected issues: `5`
- In-progress expected issues: `8`
- Coverage status: `complete`
- Coverage: `13`/`13` issues attempted

## Published Paths

- Latest truth artifact: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/latest.json`
- Latest scorecard: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/latest.json`
- Revision-scoped truth pointer: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/rev-5/latest.json`
- Revision-scoped scorecard pointer: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-5/latest.json`

## Truth Metrics

| Metric | Value |
| --- | --- |
| Verified truth success rate (primary) | 100.0% |
| Full-corpus truth success rate (legacy/context) | 53.8% |
| No-rescue truth success rate | 53.8% |
| Merged-only rate | 53.8% |

## In-Flight Graduation Metrics

| Metric | Value |
| --- | --- |
| In-progress expected issues | 8 |
| In-progress attempted issues | 8 |
| In-progress successful issues | 2 |
| In-progress graduation rate | 25.0% |
| In-progress issue numbers | `#5426`, `#5427`, `#5428`, `#5764`, `#5789`, `#5790`, `#5839`, `#5844` |

## Proxy Metrics

| Metric | Value |
| --- | --- |
| Proxy no-rescue success rate | 76.9% |
| Unique issues attempted | 13 |
| Unique issues succeeded | 10 |
| Unique issues failed | 3 |
| Unique issues neutral | 0 |
| Total ticks | 28 |

## Failure Class Distribution

- `blocked_auth_failure`: 7
- `blocked_not_dispatch_bounded`: 8
- `blocked_sanitation_failed`: 2
- `rescue_no_deliverable`: 1

## Rescue Counts By Type

- `rescue_no_deliverable`: 1

## Previous Published Artifact

- Previous artifact path: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-5/scorecard-20260528T171236Z.json`
- Previous generated_at: `2026-05-28T17:12:36Z`

## Deltas

- Merged-only rate (`merged_only_rate`): 0.1539
- No-rescue truth success rate (`no_rescue_truth_success_rate`): 0.1539
- Proxy no-rescue success rate (`proxy_no_rescue_success_rate`): 0.0000
- Full-corpus truth success rate (legacy/context) (`truth_success_rate`): 0.1539
- Unique issues attempted (`unique_issues_attempted`): 0.0000
