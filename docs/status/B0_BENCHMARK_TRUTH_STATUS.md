# B0 Benchmark Truth Status

Last updated: 2026-04-17T06:04:51Z

This is the repo-tracked recurring `TW-02` publication surface for the fixed benchmark corpus.

## Corpus

- Corpus manifest: `docs/benchmarks/corpus.json`
- Corpus id: `tw-01-bounded-execution-v1`
- Revision: `3`
- Recorded on: `2026-04-17`
- Success contract: `mergeable_pr_or_merged_pr`
- Coverage status: `complete`
- Coverage: `6`/`8` issues attempted

## Published Paths

- Latest truth artifact: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/latest.json`
- Latest scorecard: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/latest.json`
- Revision-scoped truth pointer: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/rev-3/latest.json`
- Revision-scoped scorecard pointer: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-3/latest.json`

## Truth Metrics

| Metric | Value |
| --- | --- |
| Truth success rate | 62.5% |
| No-rescue truth success rate | 62.5% |
| Merged-only rate | 62.5% |

## Proxy Metrics

| Metric | Value |
| --- | --- |
| Proxy no-rescue success rate | 0.0% |
| Unique issues attempted | 6 |
| Unique issues succeeded | 0 |
| Unique issues failed | 1 |
| Unique issues neutral | 5 |
| Total ticks | 8 |

Proxy note: neutral issue outcomes are current-corpus rows that were neither fresh success nor failure, such as `issue_already_resolved`.

## Proxy Neutral Class Distribution

- `issue_already_resolved`: 5

## Failure Class Distribution

- `blocked_not_dispatch_bounded`: 3

## Rescue Counts By Type

- none
