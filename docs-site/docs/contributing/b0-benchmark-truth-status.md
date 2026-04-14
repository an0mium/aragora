---
title: B0 Benchmark Truth Status
description: B0 Benchmark Truth Status
---

# B0 Benchmark Truth Status

Last updated: 2026-04-14T18:12:28Z

This is the repo-tracked recurring `TW-02` publication surface for the fixed benchmark corpus.

## Corpus

- Corpus manifest: `docs/benchmarks/corpus.json`
- Corpus id: `tw-01-bounded-execution-v1`
- Revision: `1`
- Recorded on: `2026-04-13`
- Success contract: `mergeable_pr_or_merged_pr`
- Coverage status: `incomplete`
- Coverage: `1`/`5` issues attempted
- Missing corpus issues: `1064`, `1641`, `1733`, `2712`

## Published Paths

- Latest truth artifact: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/latest.json`
- Latest scorecard: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/latest.json`
- Revision-scoped truth pointer: `docs/status/generated/benchmark_truth_artifacts/tw-01-bounded-execution-v1/rev-1/latest.json`
- Revision-scoped scorecard pointer: `docs/status/generated/benchmark_scorecards/tw-01-bounded-execution-v1/rev-1/latest.json`

## Truth Metrics

| Metric | Value |
| --- | --- |
| Truth success rate | 60.0% |
| No-rescue truth success rate | 40.0% |
| Merged-only rate | 60.0% |

## Proxy Metrics

| Metric | Value |
| --- | --- |
| No-rescue success rate | 0.0% |
| Unique issues attempted | 1 |
| Unique issues succeeded | 0 |
| Unique issues failed | 1 |
| Total ticks | 4 |

## Failure Class Distribution

- `blocked_sanitation_failed`: 1
- `rescue_no_deliverable`: 3

## Rescue Counts By Type

- `rescue_no_deliverable`: 3
