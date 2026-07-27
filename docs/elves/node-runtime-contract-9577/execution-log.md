# Execution Log — Node 24 Runtime Contract (#9577)

## Batch 0/2 — staging

### 2026-07-24T23:47Z — live coordination check

- Re-read the global operator mailbox.
- Confirmed workflow changes remain Tier 4 parked-draft territory.
- Confirmed branch `codex/node24-runtime-contract-9577` did not exist locally
  or remotely.
- Confirmed no matching PR, owner, lane, lease, issue assignee, or issue
  comment.
- Confirmed issue #9577 is open.

### 2026-07-24T23:48Z — main and runner preflight

- Base and remote main:
  `c7c4681eb08d5e7c7966d10dfba3a1520d671319`.
- Protected required contexts queried from branch protection.
- Five non-quorum protected required contexts were green.
- `aragora-merge-quorum` is PR/evidence-specific and skipped on main.
- Uptime Monitor was red but is not a protected required context.
- Four `aragora` runners were online, three Linux and one macOS.

### 2026-07-24T23:49Z — isolation and ownership

- Created:
  `$HOME/.codex/worktrees/node24-runtime-contract-9577/aragora`.
- Created branch `codex/node24-runtime-contract-9577` at the exact staging
  base.
- Initial lease request failed because broad path `tests` overlapped lease
  `4db40cc4-8dd` on an unrelated Boundary 2 test path.
- Retried with exact test path
  `tests/ci/test_live_node_runtime_workflows.py`.
- Claimed lease `00ea44f4-78d` for
  `elves-node-runtime-contract-9577-20260724`.
- Claimed lane `node-runtime-contract-9577-20260724`.

### 2026-07-24T23:50Z — plan

- Added `docs/plans/2026-07-24-node-runtime-contract-9577.md`.
- Plan SHA-256:
  `4a3adbc45ca347b457b30bd0adb923d77cc1b208b5f7aa33792aadf1c712f071`.
- Added the survival guide, learnings, execution log, and staged session
  state.
- No product edit has occurred.

### 2026-07-24T23:54Z — pushed draft staging lane

- Repository hooks and
  `bash scripts/automation_pr_preflight.sh origin/main HEAD` passed.
- Pushed docs-only commit
  `ce5a3524c5f5e75e1ca1148b87b18ce8766a147c`.
- Opened draft PR #9591:
  `https://github.com/synaptent/aragora/pull/9591`.
- Applied `codex`, `codex-automation`, and `operator-review-required`.
- Corrected the lane record from the worktree-local coordination store to the
  shared repository coordination store, then re-read steering: zero messages.
- Exact PR head matched the pushed commit.
- Staging stop condition reached. No product edit has occurred.

## Batch 1/2 — implementation

### 2026-07-24T23:59Z — launch

- Re-read the Elves skill and all recovery artifacts in the required order.
- No `.ai-docs/manifest.md`, constitution, or project TODO exists.
- Verified local and remote branch tips both match the authorized launch head
  `389556c2217f0ea26965703c17f86c65db5bd177`.
- Set the Stop Gate to no and marked Batch 1 in progress.
- Next: live steering, lease, main overlap, runner/main health, baseline,
  rollback tag, and Batch 1 contract.

### 2026-07-25T00:00Z — live tripwires passed

- Steering: zero lane messages.
- Lease `00ea44f4-78d` renewed through
  `2026-07-25T08:00:13Z`.
- PR #9591 remained draft at exact head
  `389556c2217f0ea26965703c17f86c65db5bd177`.
- Current main was
  `8b97a31d1f81ec74d63e30409b867971999bb2aa`; its four-file delta had no
  overlap with the approved product paths.
- All five non-quorum protected main contexts were green; merge quorum was
  skipped on main as expected.
- Four `aragora` runners were online, including three Linux runners.
- Committed launch state and merged the non-overlapping main commit without
  rewriting published branch history.
- Created rollback tag
  `elves/pre-batch-1-node-runtime-contract-9577` at
  `2750e5d09c0c93a8a39c5f668d2eef7edbc6b232`; the generic tag already
  belonged to another historical run and was left unchanged.

### 2026-07-25T00:01Z — pre-edit baseline and survey

- `python3 scripts/check_workflow_checkout_integrity_policy.py --repo-root .`:
  PASS.
- `python3 scripts/check_required_check_priority_policy.py --repo-root .`:
  PASS.
- `python3 -m pytest -q tests/ci`: `197 passed, 1 skipped`.
- Exact `node:24.18-alpine` container:
  - Node `v24.18.0`, npm `11.16.0`;
  - `npm ci`: PASS;
  - `npm run lint`: PASS;
  - `npx tsc --noEmit`: PASS;
  - `npm run build`: PASS, 228 routes.
- Restored the generated `next-env.d.ts` build side effect; worktree product
  state returned clean.
- Independent read-only survey mapped 24 selectors across the approved
  workflows: 21 live-serving selectors should resolve to exact `24.18.0`;
  three SDK/version-only selectors remain Node 20.
- Survey confirmed `live-deploy-mode-gate.yml` skips draft PRs, so the exact
  Node 24 container run is the valid build-proof surface for this parked draft.

### 2026-07-25T00:06Z — hard stop: current-main product overlap

- While baseline validation was running, PR #9589 merged as current main
  `1cd722cc2df2330466af2511f6f3b6b83d496b83`.
- PR #9589 changes `aragora/live/package.json` overrides and regenerates
  `aragora/live/package-lock.json`; it does not add an engine declaration.
- Both paths are core Batch 1 product paths. The plan's current-main overlap
  hard stop fired before the Batch 1 contract or any product edit.
- No #9577 Node engine, workflow, test, package, or lockfile change was made.
- Exact approval required: authorize merging current main `1cd722cc...`,
  preserving PR #9589, reconciling the baseline/contract, and resuming Batch 1.

### 2026-07-25T00:21Z — exact overlap authorization received

- The operator explicitly authorized merging current main
  `1cd722cc2df2330466af2511f6f3b6b83d496b83`, preserving PR #9589's package
  dependency and lockfile changes, reconciling the Batch 1 baseline and
  contract, and resuming the approved Node 24 implementation.
- Re-read the Elves skill and recovery artifacts in the required order.
- Re-verified local, remote-branch, and PR heads at
  `254a1935670d0178d2894bb2d900e43c36e0de24`.
- Re-verified current main at the exact authorized commit.
- Steering remained empty. Lease `00ea44f4-78d` was renewed through
  `2026-07-25T08:20:37Z`.
- All five non-quorum protected checks on current main were green;
  `aragora-merge-quorum` was skipped as expected on main.
- Four `aragora` runners were online, including three Linux runners.
- Set the Stop Gate back to no. Next: merge the authorized main commit,
  verify #9589 preservation, and recapture the pre-edit baseline.
- Reconciled the durable plan with the explicit merge authorization and the
  no-rebase rule for published branches. New plan SHA-256:
  `5f144491f957af73445dca1ed3d08bf8616db8564f091479dd46e8908df01a5d`.

### 2026-07-27T14:19Z — elapsed-time reactivation and reconciliation proof

- Local HEAD is the lane-owned merge commit
  `158701f0a4e90813469a72032e30cd5d0daabf7d`; the remote branch and PR head
  remain unchanged at `254a1935670d0178d2894bb2d900e43c36e0de24`.
- Current main still equals the exact authorized
  `1cd722cc2df2330466af2511f6f3b6b83d496b83`.
- `git diff 1cd722cc..HEAD` is empty for `aragora/live/package.json` and
  `aragora/live/package-lock.json`, proving PR #9589 is preserved before this
  lane's product edits.
- Restored the generated `aragora/live/next-env.d.ts` build side effect via a
  one-line patch; the worktree returned clean.
- Re-read global and lane steering: zero messages.
- The prior lease expired during the elapsed interval. Reclaimed the
  identical narrow path scope without conflict as lease `8e6a0fa7-5ef`.
- Three Linux runners with the `aragora` label are online, satisfying the
  minimum capacity gate.
- Reconciled baseline:
  - workflow checkout-integrity policy: PASS;
  - required-check priority policy: PASS;
  - `python3 -m pytest -q tests/ci`: `197 passed, 1 skipped`.
- An initial container retry mounted only `aragora/live` and therefore hid
  the local `../../sdk/typescript` dependency. That harness result is invalid,
  not a repository failure. The corrected proof mounts the repository root.
- Next: rerun the full exact-Node proof to a terminal result, then write the
  Batch 1 contract and implement.

### 2026-07-27T14:24Z — reconciled baseline complete

- Corrected exact-runtime proof mounted the repository root so the live
  package's `file:../../sdk/typescript` dependency was present.
- Exact `node:24.18-alpine` container:
  - Node `v24.18.0`, npm `11.16.0`;
  - `npm ci`: PASS, 951 packages installed;
  - `npm run lint`: PASS;
  - `npx tsc --noEmit`: PASS;
  - `npm run build`: PASS, all 228 routes generated.
- The inherited install reported 29 high-severity audit findings and existing
  deprecation/install-script warnings. This lane does not change dependencies
  and does not widen scope to remediate them.
- Restored the generated `aragora/live/next-env.d.ts` production-route import
  to its tracked dev-route form. No product diff remained.

### Batch 1 contract

**Behaviors:**

- `aragora/live` declares Node `>=24.18.0 <25` in both package metadata and the
  lockfile root package record.
- Every approved workflow job that installs or executes `aragora/live`
  resolves its `actions/setup-node@v4` selector to exact `24.18.0`.
- The three unrelated selectors remain independently managed at Node 20:
  `release.yml:sdk-parity-gate`, `test.yml:version-check`, and
  `test.yml:sdk-typescript-compile-gate`.
- A focused static test owns the complete 13-workflow/24-selector inventory
  and proves the existing live deploy-mode gate covers package changes and
  performs `npm ci` plus a real runtime build.
- Existing package dependencies, overrides, workflow behavior, and unrelated
  Node jobs are preserved.

**Build on:**

- Extend the existing `engines` convention used by npm package metadata; do
  not add a version manager, Docker, or dependency change.
- Mirror only the package-lock root record (`packages[""]`), following npm's
  package metadata shape; do not regenerate the lockfile.
- Preserve each workflow's existing selector style: update shared
  `NODE_VERSION` values where they exclusively feed live jobs, and update
  literal selectors only in identified live jobs.
- Follow the PyYAML and `Path` patterns in
  `tests/ci/test_release_gates.py` and
  `tests/ci/test_sdk_test_workflow.py`, including handling PyYAML's boolean
  representation of the bare `on:` key.

**Acceptance criteria:**

- [x] Package and lockfile root engines both equal `>=24.18.0 <25`.
- [x] The test enumerates exactly all 24 setup-node selectors across exactly
  the 13 approved workflows.
- [x] All 21 live-serving selectors resolve to `24.18.0`.
- [x] The three named non-live selectors remain Node 20 and no other selector
  is exempted.
- [x] Every live-classified job has an `aragora/live` working directory or
  live lockfile cache path.
- [x] `live-deploy-mode-gate.yml` covers `aragora/live/**`, its internal
  filter covers `aragora/live/package*.json`, and its live job includes
  setup-node, `npm ci`, and `npm run build`.
- [x] Focused test, complete `tests/ci`, workflow policies, diff checks, and
  exact Node 24 install/lint/typecheck/build all pass.
- [x] PR #9589's dependency and override records remain unchanged; the
  lockfile delta contains only the root `engines` field.

**Blast radius:**

- `aragora/live/package.json`: additive runtime metadata read by npm and
  deployment tooling; no script, dependency, or override changes.
- `aragora/live/package-lock.json`: one additive root metadata field; 1,034
  non-root package records must remain byte-equivalent.
- Thirteen protected workflow files: 24 setup-node consumers total; 21 live
  consumers change and three named non-live consumers remain untouched.
- New focused test only; no existing test is modified or weakened.
- Risk: **medium** because protected workflows include mixed-purpose jobs.
  Mitigation is an exhaustive selector map, exact diff review, repository
  policy checks, and a regression-focused independent review.

**Pre-implementation survey:**

- Live inventory re-parsed from current files: 21 live selectors and three
  non-live selectors, matching the earlier independent survey.
- Shared env selectors are safe only in `deploy-frontend.yml`, `e2e.yml`,
  `lighthouse.yml`, and `production-monitor.yml`, where every consumer is a
  live job. `live-deploy-mode-gate.yml` uses a live-job-local env value.
- `release.yml` and `test.yml` require literal, job-specific edits to preserve
  SDK/version jobs.
- The existing live deploy-mode gate already has the required path trigger,
  package filter, install, typecheck, and runtime build shape; no workflow
  structure or trigger change is needed.
- No existing focused live-runtime contract test exists; the new file
  establishes this inventory contract using current `tests/ci` conventions.

### 2026-07-27T14:31Z — Batch 1 implementation and validation

**Implemented:**

- Added Node engine `>=24.18.0 <25` to `aragora/live/package.json` and only
  the lockfile root package record.
- Updated 15 literal/env selector definitions across the 13 approved
  workflows. Those definitions feed exactly 21 live setup-node consumers.
- Preserved Node 20 in the three named SDK/version-only jobs.
- Added `tests/ci/test_live_node_runtime_workflows.py` with three tests
  covering package metadata, the exhaustive selector inventory, live-job
  grounding, and the existing deploy-mode build gate.

**Validation:**

- Focused workflow-contract test: `3 passed`.
- Ruff check and format check for the new test: PASS.
- Workflow checkout-integrity policy: PASS.
- Required-check priority policy: PASS.
- Complete `tests/ci`: `200 passed, 1 skipped`, 12 inherited warnings.
- `git diff --check`: PASS.
- Structured package comparison after deleting only `engines`: byte-equivalent
  to the merged #9589 state.
- Structured lock comparison after deleting only root `engines`:
  byte-equivalent to the merged #9589 state; all 1,034 non-root package
  records are unchanged.
- Exact post-change `node:24.18-alpine` validation:
  - Node `v24.18.0`, npm `11.16.0`;
  - `npm ci`: PASS;
  - `npm run lint`: PASS;
  - `npx tsc --noEmit`: PASS;
  - `npm run build`: PASS, 228 routes.
- Restored the generated `next-env.d.ts` side effect after the build.

**Review:**

- The focused test caught an initially mis-targeted identical selector in
  `release.yml`: the SDK job had changed while the frontend build had not.
  Fixed the workflow selectors without weakening the test, then reran green.
- Fresh independent non-countable reviewer `batch1_review_2` found no
  blocking issue and assessed the scoped regression confidence as HIGH.
- Reviewer independently confirmed 24 selectors, 21 live consumers, three
  Node 20 exceptions, unchanged workflow structure, root-only lock metadata,
  and #9589 preservation.
- No PR review or inline comment exists. The sole issue comment is the
  resolved overlap-stop record; existing checks apply only to remote head
  `254a193...` and will be repolled after push.

**Current-main tripwire:**

- During independent review, main advanced from `1cd722cc...` to
  `b0633c5f76738dfcc5412107a97753be91db6f8d`.
- The one intervening #9593 commit changes only:
  `aragora/cli/commands/review_queue_comment_verdicts.py`,
  `aragora/swarm/settle_plan.py`, `scripts/settle_pr.py`, and
  `tests/swarm/test_settle_plan.py`.
- Overlap with every approved product and operational path is zero.
- Per the published-branch protocol, commit the reviewed Batch 1 work, merge
  this non-overlapping main commit without rebasing, then rerun gates before
  push.

**Regression attestation:**

- Cumulative product diff is limited to the two package metadata files, the
  13 approved workflows, and one new focused test. No unexpected deletion.
- Shared surfaces: 24 workflow setup-node consumers were exhaustively mapped;
  21 live consumers change, three unrelated consumers remain Node 20.
  Package changes are additive, with scripts, dependencies, overrides, and
  1,034 non-root lock records unchanged.
- Test baseline changed from `197 passed, 1 skipped` to
  `200 passed, 1 skipped`: +3 passing tests, no deletion, no new skip.
- Confidence: **HIGH**. The exhaustive inventory failed on the one
  mis-targeted selector and passed only after the mixed-purpose workflow was
  corrected; exact runtime validation and structured metadata comparisons
  cover the remaining regression surface.
- Timing: implementation 12m; validation 16m; independent review 5m.

## Batch 2/2 — validation and final readiness

Pending Batch 1.
