# B0 Benchmark Truth Status

Last updated: 2026-04-15T22:12:28Z

This is the repo-tracked recurring `TW-02` publication surface for the fixed benchmark corpus.

## Corpus

- Corpus manifest: `docs/benchmarks/corpus.json`
- Corpus id: `tw-01-bounded-execution-v1`
- Revision: `2`
- Recorded on: `2026-04-15`
- Success contract: `mergeable_pr_or_merged_pr`
- Coverage status: `incomplete`
- Coverage: `2`/`5` issues attempted
- Missing corpus issues: `1064`, `1641`, `2712`

## Published Paths

- Latest truth artifact: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/latest.json`
- Latest scorecard: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/latest.json`
- Revision-scoped truth pointer: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/rev-2/latest.json`
- Revision-scoped scorecard pointer: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-2/latest.json`

## Truth Metrics

| Metric | Value |
| --- | --- |
| Truth success rate | 0.0% |
| No-rescue truth success rate | 0.0% |
| Merged-only rate | 0.0% |

## Proxy Metrics

| Metric | Value |
| --- | --- |
| Proxy no-rescue success rate | 0.0% |
| Unique issues attempted | 2 |
| Unique issues succeeded | 0 |
| Unique issues failed | 2 |
| Unique issues neutral | 0 |
| Total ticks | 5 |

## Corpus Freshness Verification Warnings

Closed issues with failed GitHub linkage checks were excluded from stale-corpus alerts until verification can be retried cleanly.

- `#873` `Bump @eslint/eslintrc from 3.2.0 to 3.3.0 in /aragora/live`: closed `n/a`, reason `n/a`, truth `no_linked_pr`, linkage `issue_lookup_failed`, error `error connecting to api.github.com
check your internet connection or https://githubstatus.com`
- `#5756` `Fail closed on 8 silent except-Exception-pass in boss_loop.py`: closed `n/a`, reason `n/a`, truth `no_linked_pr`, linkage `issue_lookup_failed`, error `error connecting to api.github.com
check your internet connection or https://githubstatus.com`

## Failure Class Distribution

- `blocked_auth_failure`: 1
- `blocked_sanitation_failed`: 1
- `rescue_no_deliverable`: 3

## Rescue Counts By Type

- `rescue_no_deliverable`: 3
