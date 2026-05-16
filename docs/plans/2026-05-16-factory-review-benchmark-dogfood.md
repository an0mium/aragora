# Factory Code Review Benchmark Dogfood Plan

Date: 2026-05-16

## Status

Plan only. No live reviewer routing changes.

## Purpose

Use Factory's open code-review benchmark as an external golden set for Aragora's
`review-pr` path. The goal is to measure Aragora's review protocol against a
portable benchmark, not to import Factory's model ranking as routing truth.

## External Source

Factory's April 29, 2026 benchmark article and open benchmark repository report:

- 50 pull requests across Sentry, Grafana, Keycloak, Discourse, and Cal.com.
- 167 validated golden comments in the final v3 data set.
- At least three runs per model.
- Precision, recall, F1, run variance, outlier handling, and cost-per-PR.
- LLM-judge matching with cross-judge spot checks.

Verified public assets:

- Article: `https://factory.ai/news/code-review-benchmark`
- Benchmark repo: `https://github.com/droid-code-review-evals/review-droid-benchmark`
- Corpus repos:
  - `https://github.com/droid-code-review-evals/droid-sentry`
  - `https://github.com/droid-code-review-evals/droid-grafana`
  - `https://github.com/droid-code-review-evals/droid-keycloak`
  - `https://github.com/droid-code-review-evals/droid-discourse`
  - `https://github.com/droid-code-review-evals/droid-cal_dot_com`

The benchmark repository contains `manifest.json`, per-tool raw comments,
evaluation scripts, and validation outputs under
`results/review_droid_run_gpt_5p2_2026-01-28/validations/`.

## Existing Aragora Surfaces To Reuse

- `aragora review-pr`: remote-head PR review, artifact persistence, optional
  no-publish mode.
- `aragora.review.reviewer_output`: normalized reviewer findings.
- `aragora.review.protocol`: advisory review brief schema and cost/latency
  fields.
- `aragora.review.settlement_outcome`: review outcome truth surface.
- `scripts/score_benchmark.py`: fixture scoring entry point.
- `scripts/run_dogfood_benchmark.py`: repeated-run dogfood harness pattern.
- `docs/status/generated/benchmark_truth_artifacts/`: existing truth-artifact
  publication shape.

## Non-Goals

- Do not change live reviewer routing defaults.
- Do not treat Factory's leaderboard as Aragora model-selection truth.
- Do not add AGT, DIC, Flywheel, or bridge substrate.
- Do not replace B0/proof-loop metrics.
- Do not run the full 50-PR benchmark until the adapter and scoring receipt
  format pass the smoke milestone.

## Proposed Implementation

### PR 1: Benchmark Intake And Smoke Adapter

Files:

- `docs/benchmarks/factory_review_benchmark_manifest.json`
- `scripts/run_factory_review_benchmark_smoke.py`
- `tests/scripts/test_run_factory_review_benchmark_smoke.py`

Behavior:

- Read Factory's `manifest.json`.
- Select a three-PR smoke slice.
- Produce a normalized local run plan for `aragora review-pr --no-publish-review
  --json`.
- Do not execute model calls by default. Provide `--execute` for explicit runs.
- Persist a planned-run receipt with source repo, PR number, head SHA, base ref,
  and Factory validation-file URL.

Smoke slice:

| Source repo | PR | Why |
|---|---:|---|
| `droid-code-review-evals/droid-sentry` | 6 | Python runtime bug plus a known false positive in validation output |
| `droid-code-review-evals/droid-grafana` | 1 | Go race-condition finding |
| `droid-code-review-evals/droid-keycloak` | 7 | Java/properties logic and runtime findings |

### PR 2: Scorer And Receipt Format

Files:

- `scripts/score_factory_review_benchmark.py`
- `tests/scripts/test_score_factory_review_benchmark.py`
- `docs/schemas/factory_review_benchmark_receipt.v1.json`

Behavior:

- Normalize Aragora `ReviewerOutput` findings into benchmark findings.
- Normalize Factory validation rows into golden findings.
- Compute precision, recall, F1, false-positive rate, per-PR cost proxy,
  latency, and run variance.
- Support a judge-swap field:
  `judge_swap_delta_pp`, measured as percentage-point delta between primary
  and secondary semantic matching.
- Fail closed when golden rows or review output cannot be parsed.

Matching policy:

- Exact file match is required.
- Line overlap within a configurable window is preferred but not mandatory when
  the semantic judge marks the same bug class and explanation.
- Bug type must match, or the scorer records the match as partial.
- False positives in Factory's validation output remain first-class negatives.

### PR 3: Three-PR Dogfood Run

Files:

- `docs/status/generated/factory_review_benchmark/smoke/latest.json`
- `docs/status/generated/factory_review_benchmark/smoke/run-<timestamp>.json`
- `docs/status/FACTORY_REVIEW_BENCHMARK_SMOKE.md`

Behavior:

- Run the smoke slice using `aragora review-pr --no-publish-review --json`.
- Score the outputs using PR 2.
- Publish a receipt with exact Aragora revision, provider roster, model IDs,
  token/cost proxy, judge model IDs, and matching policy version.

Gate:

- Smoke must complete without GitHub writes to external benchmark PRs.
- Every reviewed diff must be bound to exact `headRefOid`.
- The report must label all metrics as benchmark-smoke, not live routing truth.

### PR 4: Full 50-PR Run

Only after PRs 1-3 are reviewed:

- Execute all 50 PRs.
- Run each selected Aragora configuration three times.
- Publish mean and standard deviation for precision, recall, F1, cost proxy,
  latency, and judge-swap variance.
- Compare configurations, not model providers alone.

Candidate configurations:

- Current default review panel.
- Single cheap reviewer pass.
- Single frontier reviewer pass.
- Cheap ensemble.
- Default panel with stricter disagreement escalation.

## Acceptance Criteria

PR 1:

- Plan generation works without model credentials.
- Smoke manifest validates against current Factory `manifest.json`.
- Tests cover missing validation files, malformed manifest rows, and
  non-executing default behavior.

PR 2:

- Scorer handles true positive, false positive, missed golden, partial match,
  malformed reviewer output, and missing cost fields.
- Precision, recall, F1, and false-positive rate are deterministic.
- Judge-swap variance is represented even when secondary judge execution is
  deferred.

PR 3:

- Three smoke reviews complete with `--no-publish-review`.
- Receipts are written under `docs/status/generated/factory_review_benchmark/`.
- The report states that metrics are external-benchmark smoke metrics, not
  production routing evidence.

PR 4:

- 50 PRs x 3 runs per selected configuration, or an explicit outlier exclusion
  receipt for each skipped run.
- No live routing default changes in the same PR.

## Risks And Controls

| Risk | Control |
|---|---|
| External benchmark shape changes | Pin source commit SHAs in the manifest receipt |
| Judge bias favors Aragora phrasing | Require judge-swap variance and store raw match explanations |
| Cost blowup on full run | Gate full run behind smoke receipt and explicit operator approval |
| Confusing smoke metrics with product truth | Label every report as external-benchmark smoke until full run is complete |
| Accidental writes to Factory PRs | Always run `review-pr` with `--no-publish-review` |

## Recommended Next Action

Open PR 1 only. Keep it additive and default-read-only. The useful dogfood is
to make Aragora review its own review protocol against an external golden set
while preserving the existing proof-loop focus.
