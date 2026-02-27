# CI Lane Discipline Runbook

This runbook keeps agent throughput high while preventing merge queue collapse.

## Goal

Run many agent sessions in parallel for ideation and implementation, but serialize merge integration so CI stays responsive.

## Lane Model

- `R&D lane`:
  - Multiple branches/worktrees are allowed.
  - PRs stay in `draft`.
  - Expensive workflows are skipped for draft PRs.
  - Iterate with local checks (`ruff`, targeted `pytest`).
- `Integrator lane`:
  - Exactly one non-draft PR at a time.
  - Auto-merge enabled on that PR.
  - Required checks must pass on the current head SHA.

## Required Checks (main branch protection)

Keep required contexts minimal and stable:

- `lint`
- `typecheck`
- `sdk-parity`
- `Generate & Validate`
- `TypeScript SDK Type Check`

If this set changes in GitHub branch protection, update:

- `.github/workflows/required-check-priority.yml`
- `scripts/check_required_check_priority_policy.py`

## Fast-Path Rules Implemented

- Draft PR fast-path:
  - Heavy workflows/jobs skip when `pull_request.draft == true`.
- Queue trimming:
  - `Required Check Priority` cancels non-required runs for non-draft PRs.
  - For draft PRs, it cancels all other PR runs on that head.
- Concurrency:
  - Workflows use per-ref concurrency with cancel-in-progress enabled for non-main refs.

## Operator Procedure

1. Choose integrator PR and keep all others in draft.
2. Ensure integrator PR has auto-merge enabled.
3. Let required checks run; ignore stale failures from superseded heads.
4. On first required-check failure, patch only that failure and push.
5. After merge, promote the next draft PR.

## One-Shot Queue Cleanup (optional)

```bash
gh run list --limit 200 --json databaseId,headBranch,status \
| jq -r '.[] | select((.status=="queued" or .status=="in_progress") and .headBranch!="main" and .headBranch!="<integrator-branch>") | .databaseId' \
| xargs -n1 -I{} gh run cancel {}
```

Use one-shot commands only. Avoid long-lived `gh run watch` loops.
