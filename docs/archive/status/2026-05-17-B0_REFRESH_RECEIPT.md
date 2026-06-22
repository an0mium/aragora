# B0 Refresh Receipt 2026-05-17T14:36:51Z

This receipt records a recurring `TW-01` / `TW-02` publication refresh on
the canonical benchmark corpus, produced from a clean observer reconciled
to `origin/main` at `f14f0c871`.

## Observer

- Branch: `droid/phase1-b0-refresh-20260517`
- Worktree: `.worktrees/codex-auto/claude-20260517-143423-4810eda2`
- Origin main HEAD at publish time: `f14f0c871cf3fe7d0add5d765bbc06c18f959578`
- `aragora swarm shift-status` observer warning at start of session:
  `shift surfaces are benchmark truth stale (45.9h old)`

## Why this refresh

- Previous published truth: `2026-05-15T16:31:15Z` (~45.9h stale)
- Previous before that: `2026-04-30T16:01:56Z` (15-day publication gap)
- Operator rule (`NEXT_STEPS_CANONICAL.md`):
  "proof that recurring benchmark publication stays complete and fresh on
  `main` without operator babysitting" remains an open gap; this refresh
  is a single bounded operator-driven receipt to keep the proof surface
  current while a recurring shim is being designed in a sibling PR (#D).

## What was refreshed

| Path | Before generated_at | After generated_at |
|---|---|---|
| `docs/status/B0_BENCHMARK_TRUTH_STATUS.md` | `2026-05-15T16:31:15Z` | `2026-05-17T14:36:51Z` |
| `docs/status/generated/benchmark_truth_artifacts/.../latest.json` | `2026-05-14T19:40:33Z` | `2026-05-17T14:36:42Z` |
| `docs/status/generated/benchmark_scorecards/.../latest.json` | `2026-05-15T16:31:15Z` | `2026-05-17T14:36:51Z` |

Two new timestamped rev-4 snapshots were also written for historical
diffability:

- `truth-20260517T143631Z.json` (initial truth-artifact run)
- `truth-20260517T143642Z.json` (auto-rebuilt by the scorecard call)
- `scorecard-20260517T143651Z.json`

## Truth metrics after refresh

| Metric | Value |
| --- | --- |
| Verified truth success rate (primary) | 0.0% |
| Full-corpus truth success rate (legacy/context) | 0.0% |
| No-rescue truth success rate | 0.0% |
| Merged-only rate | 0.0% |
| In-progress graduation rate | 0.0% (0/13) |

## Failure class distribution (unchanged)

| Class | Count | Share |
| --- | ---: | ---: |
| `blocked_not_dispatch_bounded` | 8 | 28.6% |
| `blocked_auth_failure` | 7 | 25.0% |
| `blocked_sanitation_failed` | 2 | 7.1% |
| `rescue_no_deliverable` | 1 | 3.6% |
| (other / passes) | 10 | 35.7% |
| **Total ticks** | 28 | 100% |

## Deltas vs previous published artifact

| Metric | Delta |
| --- | ---: |
| `merged_only_rate` | 0.0 |
| `no_rescue_truth_success_rate` | 0.0 |
| `proxy_no_rescue_success_rate` | 0.0 |
| `truth_success_rate` | 0.0 |
| `unique_issues_attempted` | 0.0 |

The corpus has not been actively dispatched since the previous scorecard,
so all truth deltas are zero. This is consistent with `boss-ready` being
empty and the boss-loop log showing `no_suitable_issue` cycles since the
previous publish.

## Reproduction

```bash
# from a managed worktree on current main
python3 scripts/build_benchmark_truth_artifact.py \
  --metrics-file ~/Development/aragora/.aragora/overnight/boss_metrics.jsonl \
  --corpus docs/benchmarks/corpus.json \
  --publish \
  --publish-dir docs/status/generated/benchmark_truth_artifacts

python3 scripts/measure_b0_scorecard.py \
  --metrics ~/Development/aragora/.aragora/overnight/boss_metrics.jsonl \
  --corpus docs/benchmarks/corpus.json \
  --truth-publish-dir docs/status/generated/benchmark_truth_artifacts \
  --publish \
  --publish-dir docs/status/generated/benchmark_scorecards

python3 scripts/render_benchmark_truth_status.py
```

## Next steps after this refresh

- Phase 2 of the current plan productizes the largest failure class
  (`blocked_auth_failure`, 7/28 ticks = 25%) as a benchmark fixture per
  Operating Law: Repeated Rescue Becomes Product.
- Phase 4 publishes a LaunchAgent shim so the freshness probe runs on a
  schedule, removing the operator-babysitting requirement that this
  receipt currently satisfies manually.
