# Node 24 Runtime Contract for `aragora/live` (#9577)

## Mission

Make the supported Node runtime for `aragora/live` explicit and mechanically
consistent across the workflows that install or execute the live frontend.
The contract is Node `24.18.x` at execution time and Node 24 for package
consumers. Prove the contract with a focused repository test and a real
frontend build under the declared runtime.

This is a fresh-main, isolated Tier 4 workflow lane. It does not modify,
re-review, settle, or merge PR #9505. It stops at draft-PR final readiness for
an OWNER decision.

## Authorization and boundaries

The operator explicitly approved this separate follow-up lane and the required
workflow edits on 2026-07-24. That approval does not authorize branch
protection changes, evidence collection, settlement, ready-for-review, or
merge.

On 2026-07-25, the operator explicitly authorized merging current main
`1cd722cc2df2330466af2511f6f3b6b83d496b83` into PR #9591, preserving PR
#9589's dependency and lockfile changes, reconciling the Batch 1 baseline and
contract against that state, and resuming the Node 24 implementation. This is
a branch reconciliation only; it does not authorize merging PR #9591.

In scope:

- `aragora/live/package.json`
- the root package record in `aragora/live/package-lock.json`
- only the Node selector(s) used by `aragora/live` consumers in these 13
  workflows:
  - `.github/workflows/deploy-frontend.yml`
  - `.github/workflows/deploy-secure.yml`
  - `.github/workflows/e2e.yml`
  - `.github/workflows/integration-gate.yml`
  - `.github/workflows/lighthouse.yml`
  - `.github/workflows/lint.yml`
  - `.github/workflows/live-deploy-mode-gate.yml`
  - `.github/workflows/merge-group-frontend-typecheck.yml`
  - `.github/workflows/nightly-full-matrix.yml`
  - `.github/workflows/onramp-integration.yml`
  - `.github/workflows/production-monitor.yml`
  - `.github/workflows/release.yml`
  - `.github/workflows/test.yml`
- `tests/ci/test_live_node_runtime_workflows.py`
- this plan and temporary Elves recovery artifacts

Out of scope:

- any file or branch belonging to PR #9505
- workflow Node selectors used only by the TypeScript SDK, CLI, or another
  package
- dependency, Next.js, Dockerfile, Docker Compose, branch-protection, runner,
  secret, environment, or deployment-setting changes
- quorum evidence, settlement, ready-for-review, or merge
- another Fable or inference consult

## Live-state tripwires

The staging base is `c7c4681eb08d5e7c7966d10dfba3a1520d671319`.
Before Batch 1 product edits:

1. Re-read global and lane steering.
2. Renew lease `00ea44f4-78d` as session
   `elves-node-runtime-contract-9577-20260724`.
3. Fetch `origin/main` and compare it with the staging base.
4. If main advanced, inspect overlap against every in-scope product path.
   Because the branch is published, reconcile only with a merge and record the
   new base; never rebase it. The exact overlap at `1cd722cc...` is separately
   authorized above. Any later unapproved in-scope overlap remains a hard stop.
5. Verify the draft PR exact head before editing.
6. Verify at least three healthy runners with the `aragora` label.
7. Verify no protected required context on current main has been terminal red
   for more than 30 minutes.

Stop and ask the operator if there is current-main overlap, foreign
ownership/lease, steering that narrows this scope, runner capacity below
three, or a sustained required-check incident.

## Runtime contract

- Add `"engines": { "node": ">=24.18.0 <25" }` to
  `aragora/live/package.json`.
- Mirror that field only in the lockfile root package record.
- Use exact `24.18.0` in the 13 GitHub Actions workflow selectors that execute
  `aragora/live`.
- Do not blindly replace every Node 20 selector in a workflow. Preserve
  unrelated package jobs, especially SDK-only jobs.
- Make a Node 24 frontend build an observable affected-path gate. Prefer the
  existing live workflow job that already performs `npm ci`, type checking,
  and `npm run build`; change structure only if the existing trigger or path
  logic cannot prove the affected live paths.

## Batch 1 — implement and prove the contract

1. Inventory every `actions/setup-node` consumer in the 13 workflows and map
   each selector to the package it executes.
2. Add the package and lockfile-root engine contract without regenerating or
   changing dependency records.
3. Change only live-frontend selectors to exact `24.18.0`.
4. Add a focused static contract test that:
   - checks the package and lockfile root engine fields agree;
   - checks the 13 expected workflow files remain in the contract set;
   - proves every job that operates in `aragora/live` resolves to Node
     `24.18.0`;
   - allows unrelated SDK/CLI Node selectors to remain independently managed;
   - proves at least one affected-path workflow performs `npm ci` and a real
     frontend build under that selector.
5. Run focused tests and inspect the diff for accidental workflow or lockfile
   churn.
6. Commit and push Batch 1 only after lease verification and
   `bash scripts/automation_pr_preflight.sh origin/main HEAD`.

Batch 1 acceptance:

- only the approved product files and the focused contract test changed;
- no dependency record or transitive package changed;
- all 13 live consumers resolve to exact Node `24.18.0`;
- package and lockfile root declare `>=24.18.0 <25`;
- focused tests pass;
- `git diff --check` and automation preflight pass.

## Batch 2 — validation, independent review, and final readiness

1. Re-read steering, renew the lease, recheck exact head and main overlap.
2. Run the focused Python workflow-contract test.
3. Under exact Node `24.18.0`, run in `aragora/live`:
   - `npm ci`
   - `npm run lint`
   - `npx tsc --noEmit`
   - `npm run build`
4. Run repository workflow policy checks relevant to every changed workflow,
   including checkout-integrity and required-check priority policy where
   applicable.
5. Run `bash scripts/automation_pr_preflight.sh origin/main HEAD`.
6. Obtain an independent, non-countable review of the exact diff. Do not run
   quorum collection or another inference/Fable consult.
7. Poll draft-PR checks and comments, repair only grounded in-scope blockers,
   and repeat validation after any repair.
8. Remove temporary operational artifacts (`.elves-session.json`,
   survival guide, execution log, and session-only learnings) from the product
   diff. Retain this plan as the durable implementation record.
9. Push the final exact head, update the PR body with commands and results, and
   stop for OWNER settlement/merge handling.

Batch 2 acceptance:

- actual Node `24.18.0` install, lint, typecheck, and frontend build pass;
- focused and repository workflow policy checks pass;
- independent review has no unresolved blocking finding;
- required non-quorum checks are green or a precise ambient blocker is
  recorded;
- the final PR contains no session-only operational artifacts;
- no ready flip, evidence collection, settlement, or merge occurred.

## Stop conditions

Stop immediately and report the exact approval question if:

- a change outside the allowed paths is required;
- any dependency or transitive package record changes;
- a Node selector cannot be attributed safely to `aragora/live`;
- current main overlaps an in-scope product file;
- the exact PR head changes externally;
- another owner or lease claims the branch, PR, or any in-scope file;
- a required check is terminal red on current main for more than 30 minutes;
- the implementation would require branch protection, runner, secret,
  environment, deployment-setting, Dockerfile, Docker Compose, Next.js, or
  dependency changes;
- an independent reviewer reports a blocking issue that cannot be fixed inside
  this scope.

## Delivery

Run mode is finite, with an eight-hour lease budget and no operator hard stop.
The user merges. The Elves lane may implement, validate, review, and prepare a
draft PR, but it must not mark ready, collect countable evidence, settle, or
merge.
