# Main-Head Required Check Backfill Runbook

Use this runbook when a merge to `main` leaves the current `origin/main` head
without one or more protected required check-runs. The goal is to distinguish
normal GitHub scheduling lag from a real missing-context gap, then prepare an
operator-authorized workflow dispatch for only the missing safe contexts.

This runbook is read-first. Do not dispatch workflows, rerun jobs, edit branch
protection, or merge a PR from this procedure without an exact-head operator
authorization prompt.

## Protected Context Map

Read live branch protection first:

```bash
gh api repos/synaptent/aragora/branches/main/protection/required_status_checks \
  --jq '{strict, contexts, checks}'
```

The normal protected contexts map to these workflow files:

| Required context | Workflow file | Safe main dispatch |
| --- | --- | --- |
| `lint` | `.github/workflows/lint.yml` | `gh workflow run lint.yml --repo synaptent/aragora --ref main` |
| `typecheck` | `.github/workflows/lint.yml` | same `lint.yml` dispatch |
| `sdk-parity` | `.github/workflows/sdk-parity.yml` | `gh workflow run sdk-parity.yml --repo synaptent/aragora --ref main` |
| `Generate & Validate` | `.github/workflows/openapi.yml` | `gh workflow run openapi.yml --repo synaptent/aragora --ref main` |
| `TypeScript SDK Type Check` | `.github/workflows/sdk-test.yml` | `gh workflow run sdk-test.yml --repo synaptent/aragora --ref main` |
| `aragora-merge-quorum` | `.github/workflows/aragora-merge-quorum.yml` | Not a safe main-head dispatch; treat as PR-only unless live tooling proves otherwise. |

## Detection

Start from live truth:

```bash
git fetch origin --prune
HEAD_SHA=$(git rev-parse origin/main)
git log -1 --oneline origin/main
```

List exact-head check-runs:

```bash
gh api "repos/synaptent/aragora/commits/${HEAD_SHA}/check-runs?per_page=100" \
  --paginate \
  --jq '.check_runs[] | "\(.name) | \(.status) | \(.conclusion) | \(.html_url)"'
```

List workflows associated with the same commit:

```bash
gh run list --repo synaptent/aragora --commit "${HEAD_SHA}" \
  --json databaseId,workflowName,event,status,conclusion,createdAt,updatedAt,url \
  --limit 100
```

Classify each protected context as one of:

- `green`: an exact-head check-run exists and completed successfully.
- `pending`: an exact-head workflow or check-run exists and is still queued or
  in progress.
- `missing`: no exact-head workflow or check-run exists for the context.
- `failed`: an exact-head check-run exists and completed unsuccessfully.
- `main-unsafe`: the context is not safe to dispatch for `main`, such as
  `aragora-merge-quorum`.

## Timing Rule

Do not declare a required context missing immediately after a merge. If the
`origin/main` commit is less than 15 minutes old, wait and recheck. GitHub may
attach the push-triggered workflows after quorum, auto-revert, or advisory
workflow-run jobs appear.

The 2026-07-07 incident that motivated this runbook showed this timing hazard:

- At `5a4a727e3e46ffde531062b56e5c2fe6d4765e84`, exact-head reads showed only
  smoke, quorum/retrigger, auto-revert, testfixer, and similar advisory jobs.
  The protected push contexts were absent at that snapshot.
- While preparing the runbook, `origin/main` advanced to
  `bf956096fd68946a4d0bd520b1b106ca5cf95535`. Exact-head reads then showed
  `lint`, `typecheck`, `sdk-parity`, `Generate & Validate`, and
  `TypeScript SDK Type Check` attached to push workflows, while
  `aragora-merge-quorum` remained a main-unsafe skipped/PR-only context.

## Backfill Preconditions

Before proposing any dispatch, verify all of the following:

- `origin/main` still equals the exact head in the authorization prompt.
- Live branch protection still requires the context.
- The missing context is one of the safe main-dispatchable contexts in the map.
- No exact-head workflow for that context is already queued or in progress.
- Any failed required context is classified separately with run, job, and log
  clue. Do not mask a real failure by dispatching unrelated workflows.
- `aragora-merge-quorum` is not dispatched for `main`.

## Operator Authorization Prompt

Use this paste-ready prompt when only safe main-dispatchable protected contexts
are missing:

```text
Start from live truth in $HOME/Development/aragora. Goal: dispatch exactly the missing current-main protected workflow contexts for origin/main head <HEAD_SHA>. Do not rerun product-proof, deploy, security, Branch Discipline, aragora-merge-quorum, or unrelated CI.

Check mailbox/owner state read-only/no receipt if possible, git fetch origin --prune, verify origin/main is still <HEAD_SHA>, and recheck exact-head check-runs/statuses. Proceed only if <GREEN_CONTEXTS> are green, aragora-merge-quorum is skipped/not safe for main, and the only missing safely dispatchable protected contexts are <MISSING_CONTEXTS>. Then dispatch exactly:
<ONE gh workflow run COMMAND PER MISSING SAFE CONTEXT>

Report run ids and stop.
```

Example command block when all four safe workflow families are absent:

```bash
gh workflow run lint.yml --repo synaptent/aragora --ref main
gh workflow run sdk-parity.yml --repo synaptent/aragora --ref main
gh workflow run openapi.yml --repo synaptent/aragora --ref main
gh workflow run sdk-test.yml --repo synaptent/aragora --ref main
```

Example command block when only OpenAPI and TypeScript SDK contexts are absent:

```bash
gh workflow run openapi.yml --repo synaptent/aragora --ref main
gh workflow run sdk-test.yml --repo synaptent/aragora --ref main
```

After dispatch, re-run the detection commands and report each run id. Stop
there; do not merge, mark ready, rerun unrelated workflows, or change branch
protection.
