# Campaign Dogfood Report — 2026-03-10

## Setup

- **Manifest**: prebuilt, 3 docs-only projects with sequential dependencies
- **Configuration**: max_parallel_ready_projects=1, max_retries_per_project=1, budget=$5
- **Worker model**: claude (via `claude -p <prompt> --yes`)
- **Review model**: claude (heterogeneous review not exercised — see finding)

## Invocations

| # | Command | proj-001 | proj-002 | proj-003 | Stop reason | Cost |
|---|---------|----------|----------|----------|-------------|------|
| 1 | `campaign run --json` | dispatched → needs_revision (clean_exit_no_deliverable) | pending | pending | still_running | $1.50 |
| 2 | `campaign run --json` | retried → skipped (clean_exit_no_deliverable) | pending | pending | still_running | $3.00 |
| 3 | `campaign run --json` | skipped | pending | pending | campaign_blocked | $3.00 |

## Concrete deliverables

None. Both dispatches for proj-001 resulted in `clean_exit_no_deliverable` — the worker ran but did not push a branch, create a PR, or produce any committed artifact.

## Findings

### F1: Worker produces no pushed artifact (root cause of campaign block)

The worker (claude -p ... --yes) executed the prompt, likely made local changes, but the supervised swarm pipeline's concrete-deliverable gate (`_extract_deliverable` in boss_loop.py) requires at least one of: a pr_url, adopted_pr, or branch with commit_shas. The worker process completed cleanly but without pushing, so `_classify_terminal_run_outcome` returned `clean_exit_no_deliverable`.

**Root cause hypothesis**: The worker prompt instructs the edit but the swarm work-order doesn't include an explicit push/commit instruction. The `WorkerLauncher._build_prompt` may not include instructions to commit and push changes. The docs-only scope may also mean the worker considers the change too small to warrant a PR.

**Severity**: High — this will block any docs-only campaign from producing deliverables until the worker prompt or commit flow is fixed.

### F2: `campaign status` and `campaign run` disagree on stop reason

After proj-001 was skipped:
- `campaign run` returned `campaign_blocked` (correct: no ready projects, no active projects)
- `campaign status` returned `still_running` (incorrect: statuses are {skipped, pending, pending} which is not a subset of {blocked, failed, skipped, completed} because pending is present)

The `_compute_stop_reason` function used by both paths does not account for projects that are pending but permanently unreachable (their dependency was skipped, not completed). The status command should also report `campaign_blocked` in this state.

**Severity**: Medium — misleading for operators who check status between runs.

### F3: Heterogeneous review was never exercised

Both worker_model and review_model were set to "claude". The `_canonical_review_model` function should have forced a different model, but since the project never reached `delivered` status, the review gate was never triggered. The dogfood did not validate the review path.

**Severity**: Info — expected consequence of F1. A successful dispatch is needed to exercise review.

### F4: retry_count semantics

After 2 dispatches, retry_count=2 and the project was skipped (max_retries_per_project=1, check is `retry_count > max_retries_per_project`). The count represents "number of dispatch attempts" not "number of retries after the first attempt". With max_retries_per_project=1, the project gets 2 total attempts. This is consistent but the naming is confusing.

**Severity**: Low — cosmetic.

## Manifest coherence

**PASS**: The manifest remained structurally coherent across all 3 invocations:
- project_ids stable
- dependency references intact
- status transitions followed the documented lifecycle (pending → active → needs_revision → active → skipped)
- execution_state.total_cost_usd accumulated correctly ($0 → $1.50 → $3.00)
- last_run_at updated on each invocation
- last_result reflected the most recent dispatch

## Conclusion

The campaign pipeline correctly:
- Resumed from a prebuilt manifest across multiple invocations
- Dispatched one project per invocation (max_parallel=1 honored)
- Tracked retry counts and skipped after exceeding max_retries
- Maintained manifest coherence
- Detected campaign_blocked when no forward progress was possible

The campaign pipeline failed to:
- Produce any concrete deliverable from a docs-only worker dispatch
- Exercise the review gate
- Agree between `status` and `run` on the terminal stop reason

**Pass/fail**: CONDITIONAL PASS for pipeline mechanics, FAIL for end-to-end deliverable production.

The pipeline orchestration layer is sound. The blocker is in the worker dispatch → artifact collection boundary, not in the campaign layer itself.
