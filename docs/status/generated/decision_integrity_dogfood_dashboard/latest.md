# Decision-Integrity Dogfood Dashboard

Last updated: 2026-07-05T08:36:54Z

Report-only generated companion for the frozen July dogfood proof artifact. It does not schedule publishing, mutate queues, post comments, or authorize merges.

## Scope

- Repo: `synaptent/aragora`
- Tracking issue: `#8861`
- Source artifact: `docs/artifacts/2026-07-decision-integrity-dogfooding.md`
- GitHub search window: `2026-06-05`..`2026-07-05`
- Query shape: reused from the July artifact, with the window made regenerable.

## Metrics

| Metric | Value | Status | Last updated | Source |
| --- | ---: | --- | --- | --- |
| Merged PRs | `495` | `limited` | `2026-07-05T08:36:54Z` | `repo:synaptent/aragora is:pr is:merged merged:2026-06-05..2026-07-05` |
| Merged PRs with "independent model review" comments | `462` | `limited` | `2026-07-05T08:36:54Z` | `repo:synaptent/aragora is:pr is:merged merged:2026-06-05..2026-07-05 "independent model review" in:comments` |
| Merged PRs with "Verdict: PASS" comments | `464` | `limited` | `2026-07-05T08:36:54Z` | `repo:synaptent/aragora is:pr is:merged merged:2026-06-05..2026-07-05 "Verdict: PASS" in:comments` |
| Merged PRs mentioning "merge-quorum" | `146` | `limited` | `2026-07-05T08:36:54Z` | `repo:synaptent/aragora is:pr is:merged merged:2026-06-05..2026-07-05 "merge-quorum" in:comments` |
| Merged PRs with "CHANGES-REQUESTED" comments | `35` | `limited` | `2026-07-05T08:36:54Z` | `repo:synaptent/aragora is:pr is:merged merged:2026-06-05..2026-07-05 "CHANGES-REQUESTED" in:comments` |
| Merged PRs with [P0] comment markers | `18` | `limited` | `2026-07-05T08:36:54Z` | `repo:synaptent/aragora is:pr is:merged merged:2026-06-05..2026-07-05 "[P0]" in:comments` |
| Merged PRs with [P1] comment markers | `64` | `limited` | `2026-07-05T08:36:54Z` | `repo:synaptent/aragora is:pr is:merged merged:2026-06-05..2026-07-05 "[P1]" in:comments` |
| Merged PRs with [P2] comment markers | `99` | `limited` | `2026-07-05T08:36:54Z` | `repo:synaptent/aragora is:pr is:merged merged:2026-06-05..2026-07-05 "[P2]" in:comments` |
| Merged PRs with [P3] comment markers | `290` | `limited` | `2026-07-05T08:36:54Z` | `repo:synaptent/aragora is:pr is:merged merged:2026-06-05..2026-07-05 "[P3]" in:comments` |
| Merged PRs with "exact-head" comment markers | `427` | `limited` | `2026-07-05T08:36:54Z` | `repo:synaptent/aragora is:pr is:merged merged:2026-06-05..2026-07-05 "exact-head" in:comments` |
| Independent-review marker coverage | `462/495 (93.3%)` | `limited` | `2026-07-05T08:36:54Z` | `derived from independent_model_review_comments/merged_prs` |
| Verdict PASS marker coverage | `464/495 (93.7%)` | `limited` | `2026-07-05T08:36:54Z` | `derived from verdict_pass_comments/merged_prs` |
| Merge-quorum marker coverage | `146/495 (29.5%)` | `limited` | `2026-07-05T08:36:54Z` | `derived from merge_quorum_comments/merged_prs` |
| Exact-head marker coverage | `427/495 (86.3%)` | `limited` | `2026-07-05T08:36:54Z` | `derived from exact_head_marker_comments/merged_prs` |
| Committed settlement receipts verified | `3/3` | `ok` | `2026-07-05T08:36:54Z` | `elves/close-the-loop-20260701:docs/elves/receipts/b3-8767-settlement.json,docs/elves/receipts/b4-8768-settlement.json,docs/elves/receipts/b6-cleanup-batch1.json` |
| Operator-local merge-executor receipts observed | `5` | `local_only` | `2026-07-04T21:45:57Z` | `.aragora/merge_executor/receipts, ~/.aragora/merge-executor-receipts` |

## Stale And Failure Behavior

| Metric | Stale after | Failure behavior | Caveat |
| --- | ---: | --- | --- |
| Merged PRs | 24h | Mark stale after SLA; if the query fails, show failed and rerun manually. | GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. |
| Merged PRs with "independent model review" comments | 24h | Mark stale after SLA; if the query fails, show failed and rerun manually. | GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. |
| Merged PRs with "Verdict: PASS" comments | 24h | Mark stale after SLA; if the query fails, show failed and rerun manually. | GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. |
| Merged PRs mentioning "merge-quorum" | 24h | Mark stale after SLA; if the query fails, show failed and rerun manually. | GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. |
| Merged PRs with "CHANGES-REQUESTED" comments | 24h | Mark stale after SLA; if the query fails, show failed and rerun manually. | GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. |
| Merged PRs with [P0] comment markers | 24h | Mark stale after SLA; if the query fails, show failed and rerun manually. | GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. |
| Merged PRs with [P1] comment markers | 24h | Mark stale after SLA; if the query fails, show failed and rerun manually. | GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. |
| Merged PRs with [P2] comment markers | 24h | Mark stale after SLA; if the query fails, show failed and rerun manually. | GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. |
| Merged PRs with [P3] comment markers | 24h | Mark stale after SLA; if the query fails, show failed and rerun manually. | GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. |
| Merged PRs with "exact-head" comment markers | 24h | Mark stale after SLA; if the query fails, show failed and rerun manually. | GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. This is a phrase marker only; it does not prove the comment SHA matched the PR head. |
| Independent-review marker coverage | 24h | Derived metric follows the staleness/failure behavior of its inputs. | Coverage is based on GitHub Search comment markers and is not a thread-by-thread audit. |
| Verdict PASS marker coverage | 24h | Derived metric follows the staleness/failure behavior of its inputs. | Coverage is based on GitHub Search comment markers and is not a thread-by-thread audit. |
| Merge-quorum marker coverage | 24h | Derived metric follows the staleness/failure behavior of its inputs. | Coverage is based on GitHub Search comment markers and is not a thread-by-thread audit. |
| Exact-head marker coverage | 24h | Derived metric follows the staleness/failure behavior of its inputs. | This is only a live phrase-marker proxy. The remaining stronger grounding gap is a per-PR audit that resolves each evidence comment SHA against the actual merged head. |
| Committed settlement receipts verified | 168h | If any receipt is missing or fails integrity verification, mark the metric failed. | These are committed settlement receipts, not operator-local merge-executor receipts. |
| Operator-local merge-executor receipts observed | 168h | If local paths are absent, mark missing; do not infer public receipt counts from operator-local storage. | Operator merge-executor receipts are local machine artifacts, not repo-visible public proof. |

## Local Receipt Notes

- Operator-local exact-head receipts observed: `5`/`5`
- These local receipts are useful operator proof, but not outsider-verifiable until promoted into repo-visible signed or hash-verifiable artifacts.

## Known Gaps

- `github_search_counts` status `limited`: GitHub Search API counts are live, regenerable marker counts; exact audited truth still requires thread-by-thread PR evidence enumeration.
- `exact_head_marker_comments` status `limited`: GitHub Search API total_count is a live search-index marker count, not a hand-audited exact truth set. This is a phrase marker only; it does not prove the comment SHA matched the PR head.
- `exact_head_marker_coverage` status `limited`: This is only a live phrase-marker proxy. The remaining stronger grounding gap is a per-PR audit that resolves each evidence comment SHA against the actual merged head.
- `operator_local_merge_executor_receipts` status `local_only`: Operator merge-executor receipts are local machine artifacts, not repo-visible public proof.

## Regenerate

```bash
python3 scripts/render_decision_integrity_dogfood_dashboard.py --output docs/status/generated/decision_integrity_dogfood_dashboard/latest.md --json-output docs/status/generated/decision_integrity_dogfood_dashboard/latest.json
```
