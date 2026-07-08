# Merge-State `UNSTABLE` Settlement Triage

Use this runbook when a PR has green branch-protection checks but GitHub reports
`mergeStateStatus=UNSTABLE`, especially during high-churn automation queues.

## Immediate Rule

Do not collect more model-review evidence or retry settlement until the
`UNSTABLE` source is classified. Model quorum cannot repair a GitHub merge-state
summary that is being held open by cancelled or stale check runs.

## Classification

1. Read required checks from branch protection:

   ```bash
   gh api repos/synaptent/aragora/branches/main/protection/required_status_checks \
     --jq '{strict,contexts,checks}'
   gh pr checks <pr> --repo synaptent/aragora --required
   ```

2. Read the full check list and live merge state:

   ```bash
   gh pr checks <pr> --repo synaptent/aragora
   gh pr view <pr> --repo synaptent/aragora \
     --json headRefOid,mergeable,mergeStateStatus,statusCheckRollup
   ```

3. Classify the blocker:

   | Signal | Meaning | Action |
   | --- | --- | --- |
   | Required check is red | hard gate | Repair the PR or main before settlement. |
   | Required checks pass and only advisory checks are pending | queue churn | Recheck once, then proceed under the merge contract if policy allows. |
   | Latest advisory check is `cancelled` before its command ran | merge-state poison | Park target settlement and file or link a CI issue. |
   | Advisory check ran and failed in files touched by the PR | in-scope failure | Repair or explicitly park the PR. |
   | Advisory check ran and failed outside PR scope | possible main regression | Verify on `origin/main`, then open or link a main-health issue. |

## Settlement-Window Head Freeze

Exact-head evidence becomes stale as soon as the PR branch moves. During a
settlement window, treat the head SHA as part of the gate:

1. Record the target `headRefOid` before collecting evidence.
2. After every reviewer dry run, evidence post, quorum rerun, and merge-packet
   read, re-read `headRefOid`.
3. If the head changed, stop the settlement attempt. Do not post stale evidence,
   rerun quorum, or chase the new head in the same cycle.
4. Leave a PR-visible head-churn note naming the old head, the new head, and any
   stale evidence artifact that must not be counted.
5. The session or person that pushed the new head owns the next exact-head
   evidence pass. A different settlement owner may resume only after a fresh
   owner/steering read and a new dry-run artifact for the new head.

This is a coordination guard, not a branch lock. It preserves the exact-head
contract without blocking repair pushes: pushes are allowed, but they invalidate
the current settlement window and transfer responsibility for refreshing
evidence to the pusher or next explicit owner.

## Portability Cancellation Pattern

Issue #9031 tracks the July 8, 2026 instance where `Portability Lint /
portability` cancelled during `actions/checkout` before
`scripts/check_portability.py` ran. The local guard passed on detached
`origin/main`, and main's own portability run was green, but GitHub still
reported `mergeStateStatus=UNSTABLE` for #9027.

This pattern is not evidence that the PR introduced private paths or portability
violations. It is a CI/workflow-concurrency settlement hazard because the
newest check run for the `portability` name can be a cancelled advisory run.

When this pattern appears:

1. Link the affected PR to #9031 or a successor issue.
2. Leave a PR handoff comment with the check name, run URL, and exact
   classification.
3. Stop settlement and evidence-collection loops for that PR until merge state
   becomes `CLEAN` or the CI issue is repaired.
4. Work the CI/runbook/tooling issue as a separate bounded unit.

## Evidence Commands

Use targeted logs to avoid large CI output:

```bash
gh run view <run-id> --repo synaptent/aragora --log \
  | rg -n "operation was canceled|Portability guard|check_portability|fetch --no-tags|##\\[error\\]"
```

Use local reproduction to distinguish a workflow cancellation from a real guard
failure. Check `origin/main` for a possible main regression, then check the
target PR head before clearing the PR source.

```bash
git fetch origin main:refs/remotes/origin/main
main_parent="$(mktemp -d /tmp/aragora-portability-main.XXXXXX)"
main_tmp="$main_parent/worktree"
git worktree add --detach "$main_tmp" origin/main
(cd "$main_tmp" && python3 scripts/check_portability.py)

head_sha="$(gh pr view <pr> --repo synaptent/aragora --json headRefOid --jq .headRefOid)"
git fetch origin "pull/<pr>/head:refs/remotes/pr/<pr>/head"
pr_parent="$(mktemp -d /tmp/aragora-portability-pr.XXXXXX)"
pr_tmp="$pr_parent/worktree"
git worktree add --detach "$pr_tmp" "$head_sha"
test "$(git -C "$pr_tmp" rev-parse HEAD)" = "$head_sha"
(cd "$pr_tmp" && python3 scripts/check_portability.py)
```

If local `origin/main` and the exact PR head both pass, or the exact PR head
already has a prior green portability run and the failed workflow never reached
the guard command, classify the target PR as parked on CI/merge-state noise
rather than source correctness.
