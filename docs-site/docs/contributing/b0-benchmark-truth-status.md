---
title: B0 Benchmark Truth Status
description: B0 Benchmark Truth Status
---

# B0 Benchmark Truth Status

Last updated: 2026-06-14T03:52:10Z

This is the repo-tracked recurring `TW-02` publication surface for the fixed benchmark corpus.

## Corpus

- Corpus manifest: `docs/benchmarks/corpus.json`
- Corpus id: `tw-01-bounded-execution-v1`
- Revision: `6`
- Recorded on: `2026-06-05`
- Success contract: `mergeable_pr_or_merged_pr`
- Verified expected issues: `5`
- In-progress expected issues: `8`
- Coverage status: `complete`
- Coverage: `13`/`13` issues attempted

## Published Paths

- Latest truth artifact: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/latest.json`
- Latest scorecard: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/latest.json`
- Revision-scoped truth pointer: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/rev-6/latest.json`
- Revision-scoped scorecard pointer: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-6/latest.json`

## Truth Metrics

| Metric | Value |
| --- | --- |
| Verified truth success rate (primary) | 100.0% |
| Full-corpus truth success rate (legacy/context) | 100.0% |
| No-rescue truth success rate | 100.0% |
| Merged-only rate | 100.0% |

## In-Flight Graduation Metrics

| Metric | Value |
| --- | --- |
| In-progress expected issues | 8 |
| In-progress attempted issues | 8 |
| In-progress successful issues | 8 |
| In-progress graduation rate | 100.0% |
| Expected in-progress issue numbers | `#5182`, `#5183`, `#5184`, `#5186`, `#5426`, `#5427`, `#5839`, `#5844` |
| Live-open expected issue numbers | none |
| Live-closed expected issue numbers | `#5182`, `#5183`, `#5184`, `#5186`, `#5426`, `#5427`, `#5839`, `#5844` |

## Proxy Metrics

| Metric | Value |
| --- | --- |
| Proxy no-rescue success rate | 92.3% |
| Unique issues attempted | 13 |
| Unique issues succeeded | 12 |
| Unique issues failed | 1 |
| Unique issues neutral | 0 |
| Total ticks | 30 |

## Failure Class Distribution

- `blocked_auth_failure`: 5
- `blocked_not_dispatch_bounded`: 12
- `blocked_sanitation_failed`: 1

## Rescue Counts By Type

- none

## Previous Published Artifact

- Previous artifact path: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-6/scorecard-20260606T114257Z.json`
- Previous generated_at: `2026-06-06T11:42:57Z`

## Deltas

- Merged-only rate (`merged_only_rate`): 0.3077
- No-rescue truth success rate (`no_rescue_truth_success_rate`): 0.3077
- Proxy no-rescue success rate (`proxy_no_rescue_success_rate`): 0.0000
- Full-corpus truth success rate (legacy/context) (`truth_success_rate`): 0.3077
- Unique issues attempted (`unique_issues_attempted`): 0.0000
