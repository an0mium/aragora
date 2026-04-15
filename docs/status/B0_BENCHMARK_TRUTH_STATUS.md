# B0 Benchmark Truth Status

Last updated: 2026-04-15T09:08:54Z

This is the repo-tracked recurring `TW-02` publication surface for the fixed benchmark corpus.

## Corpus

- Corpus manifest: `docs/benchmarks/corpus.json`
- Corpus id: `tw-01-bounded-execution-v1`
- Revision: `1`
- Recorded on: `2026-04-13`
- Success contract: `mergeable_pr_or_merged_pr`
- Coverage status: `complete`
- Coverage: `5`/`5` issues attempted

## Published Paths

- Latest truth artifact: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/latest.json`
- Latest scorecard: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/latest.json`
- Revision-scoped truth pointer: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/rev-1/latest.json`
- Revision-scoped scorecard pointer: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-1/latest.json`

## Truth Metrics

| Metric | Value |
| --- | --- |
| Truth success rate | 80.0% |
| No-rescue truth success rate | 80.0% |
| Merged-only rate | 80.0% |

## Proxy Metrics

| Metric | Value |
| --- | --- |
| Proxy no-rescue success rate | 0.0% |
| Unique issues attempted | 5 |
| Unique issues succeeded | 0 |
| Unique issues failed | 1 |
| Unique issues neutral | 4 |
| Total ticks | 6 |

Proxy note: neutral issue outcomes are current-corpus rows that were neither fresh success nor failure, such as `issue_already_resolved`.

## Proxy Neutral Class Distribution

- `issue_already_resolved`: 4

## Corpus Freshness Alerts

Truth metrics still reflect the frozen corpus revision. Closed issues without linked PR truth should be retired or replaced in the next corpus revision.

- `#1733` `fix(swarm): collect finished detached workers before stale lease reap`: closed `2026-03-31T23:45:29Z`, reason `COMPLETED`, truth `no_linked_pr`

## Failure Class Distribution

- `blocked_auth_failure`: 2

## Rescue Counts By Type

- none

## Previous Published Artifact

- Previous artifact path: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-1/scorecard-20260415T044303Z.json`
- Previous generated_at: `2026-04-15T04:43:03Z`

## Deltas

- `merged_only_rate`: 0.2000
- `no_rescue_truth_success_rate`: 0.2000
- `proxy_no_rescue_success_rate`: 0.0000
- `truth_success_rate`: 0.2000
- `unique_issues_attempted`: 0.0000
