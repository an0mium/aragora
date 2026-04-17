# B0 Benchmark Truth Status

Last updated: 2026-04-17T12:57:27Z

This is the repo-tracked recurring `TW-02` publication surface for the fixed benchmark corpus.

## Corpus

- Corpus manifest: `docs/benchmarks/corpus.json`
- Corpus id: `tw-01-bounded-execution-v1`
- Revision: `3`
- Recorded on: `2026-04-17`
- Success contract: `mergeable_pr_or_merged_pr`
- Verified expected issues: `5`
- In-progress expected issues: `3`
- Coverage status: `complete`
- Coverage: `8`/`8` issues attempted

## Published Paths

- Latest truth artifact: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/latest.json`
- Latest scorecard: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/latest.json`
- Revision-scoped truth pointer: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/rev-3/latest.json`
- Revision-scoped scorecard pointer: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-3/latest.json`

## Truth Metrics

| Metric | Value |
| --- | --- |
| Verified truth success rate (primary) | 100.0% |
| Full-corpus truth success rate (legacy/context) | 62.5% |
| No-rescue truth success rate | 62.5% |
| Merged-only rate | 62.5% |

## In-Flight Graduation Metrics

| Metric | Value |
| --- | --- |
| In-progress expected issues | 3 |
| In-progress attempted issues | 3 |
| In-progress successful issues | 0 |
| In-progress graduation rate | 0.0% |
| In-progress issue numbers | `#5814`, `#5818`, `#5820` |

## Proxy Metrics

| Metric | Value |
| --- | --- |
| Proxy no-rescue success rate | 0.0% |
| Unique issues attempted | 8 |
| Unique issues succeeded | 0 |
| Unique issues failed | 3 |
| Unique issues neutral | 5 |
| Total ticks | 8 |

Proxy note: neutral issue outcomes are current-corpus rows that were neither fresh success nor failure, such as `issue_already_resolved`.

## Proxy Neutral Class Distribution

- `issue_already_resolved`: 5

## Failure Class Distribution

- `blocked_not_dispatch_bounded`: 3

## Rescue Counts By Type

- none

## Previous Published Artifact

- Previous artifact path: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-3/scorecard-20260417T124141Z.json`
- Previous generated_at: `2026-04-17T12:41:41Z`

## Deltas

- `merged_only_rate`: -0.3750
- `no_rescue_truth_success_rate`: -0.3750
- `proxy_no_rescue_success_rate`: 0.0000
- `truth_success_rate`: -0.3750
- `unique_issues_attempted`: 0.0000
