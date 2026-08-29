# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

## Mission

Prepare a fresh-main Tier-4 PR that fixes the three exact-head snapshot-provenance defects found by
the terminal review of PR #9677. Carry one trusted full SHA from decision/settlement through the
merge command in the Codex automation merger, boss drain, and merge arbiter; fail closed on missing,
malformed, changed, or unavailable heads.

## Run Control

- **Run mode:** finite
- **Stop policy:** stage boundary now; after launch, blocker or completed final draft readiness
- **User intent:** "authorize a separate fresh-main PR fixing these three snapshot-provenance paths. After that lands, separately authorize reconciling current main into #9677, regenerate docs, and rerun validation plus terminal review before OWNER settlement."
- **Checkpoint due by:** none
- **Checkpoint semantics:** none
- **May continue after checkpoint:** yes
- **Actual stop conditions:** staging complete in this call; after launch, final draft readiness or a genuine Tier-4/collision blocker
- **Workspace ownership:** dedicated worktree at `$HOME/.codex/worktrees/merge-snapshot-provenance-9677-20260829/aragora`
- **Branch tip at start:** `953c501c2147026c2c996c3f95001580e326ec52`
- **Lease:** `906d51fa-4af`, session `codex-merge-snapshot-provenance-9677-20260829`
- **Merge policy:** user-merges; implementation authorization is not merge authorization
- **Final-response policy:** allowed at the mandatory staging boundary; after launch only when the Stop Gate permits it
- **Batch completion rule:** update execution log and survival guide, commit, push, then re-read this guide before starting later work
- **Re-read rule:** immediately after every commit and push
- **Checkpoint rule:** the staging boundary is mandatory now; after launch, a checkpoint is progress evidence and not a stop condition
- **Continuation rule:** after launch, continue until final draft readiness or a genuine blocker

## Session Budget

- **Started:** 2026-08-29 17:42 America/Chicago
- **User returns:** unknown; assume an eight-hour execution window after launch
- **Checkpoint expectation:** launch-ready draft PR during staging; validated final draft after launch
- **Time budget:** approximately 8 hours after launch, no hard deadline
- **Average batch time so far:** staging only
- **Batches remaining:** 2 of 2

## Stop Gate

- **Planned batches remaining:** 2
- **Stop allowed right now:** yes
- **Why:** Elves requires a separate staging and launch call; product implementation must not start in this call
- **Next required action:** user sends the launch prompt recorded below

## Effort Standard

- Work as hard as you can for the full launched run. Do not be lazy.
- Maintain the same level of care on the terminal review and cleanup as on implementation.
- Do not settle for the minimum acceptable change, the first green check, or a shallow pass while
  a planned verification remains.
- When one task is complete, take the next highest-value action from the plan unless the Stop Gate
  or a genuine authorization/collision blocker requires stopping.

## Forbidden Stop Reasons

These are not valid reasons to stop the launched run while planned work remains:

- A checkpoint was reached after the mandatory staging boundary.
- A commit or push succeeded.
- CI or the focused test suite is green.
- A draft PR exists.
- The user is silent while the explicit launch authorization remains in force.
- One path is fixed while another planned snapshot-provenance path remains.
- A batch is complete but later validation, review, or cleanup work remains.

## Non-Negotiables

- Do not touch PR #9677, branch `fix/merge-halt-shared-guard`, or its current reconciliation worktree.
- No head may be resolved after the eligibility, settlement, or approval snapshot it is meant to bind.
- Missing/malformed heads block before merge subprocess execution; every merge is pinned to the same full SHA.
- Do not edit workflows, runner configuration, protected governance, branch protection, or required checks.
- Do not collect quorum evidence, mark ready, settle, or merge either PR.
- Never rebase, force-push, or use destructive cleanup; stage specific files only.

## Launch Readiness

- [x] Plan saved to disk
- [x] Survival guide initialized
- [x] Learnings initialized
- [x] Execution log initialized
- [x] Dedicated branch and worktree created from fresh main
- [x] Branch lease claimed; no same-object lane collision found
- [x] Draft PR #9874 opened and recorded
- [x] Preflight executed; scoped gates are green and ambient repo-wide warnings are recorded
- [x] Run mode, non-negotiables, and merge policy recorded
- [x] Stop Gate initialized with `Stop allowed right now: no` for post-launch execution; the mandatory staging boundary remains `yes`
- [x] Launch prompt prepared

## Current Phase

**Status:** Launch-ready staging boundary

**Active batch:** Batch 0: session setup complete

**What was just finished:** Batch 0 was committed and pushed, and draft PR #9874 was opened from the isolated fresh-main branch.

**Single next action:** the user sends the launch prompt below; the launched run then renews the lease and begins Batch 1.

## Active Compute

No active paid or long-running compute.

## Next Exact Batch

**Batch:** 1: Carry exact decision heads through all three merge paths

**Scope:**

- Bind Codex automation eligibility snapshot to its merge command.
- Bind boss drain settlement report head to its merge command.
- Make merge arbiter reject missing/malformed heads and always pin the merge.
- Add category tests, focused validation, and mutation break checks.

**Acceptance criteria:**

- [ ] Decision/settlement/approval head equals `--match-head-commit` for all three paths.
- [ ] Missing or malformed trusted heads block before any merge subprocess.
- [ ] Focused tests, Ruff, mypy, drift checks, and mutation checks pass.

**Risk:** Tier-4 merge authority; a missed fallback can merge an unchecked head.

**Rollback tag:** `elves/pre-batch-1`

## Post-Checkpoint Control Loop

Every completed batch must end with a commit and push. Immediately afterward, re-read this survival guide before doing anything else. A pushed checkpoint proves progress; it does not permit stopping while launched work remains.

After each commit and push:

1. Identify the first unfinished task and begin it.
2. Inventory active paid or long-running compute and stop anything stale or ambiguous.
3. Re-read any new operator steering and rewrite run controls if scope or stop behavior changed.
4. Does the Stop Gate still say `Stop allowed right now: no`, or does `.elves-session.json` still
   have `continuation_guard.stop_allowed: false`? If yes, continue immediately.
5. Stop only at final draft readiness, an explicit user stop, or a genuine Tier-4/collision blocker.

## Tool Configuration

```yaml
lint: python3 -m ruff check scripts/merge_codex_automation_prs.py scripts/boss_drain_pass.py aragora/swarm/merge_arbiter.py tests/scripts/test_merge_codex_automation_prs.py tests/swarm/test_boss_drain.py tests/swarm/test_merge_arbiter.py
typecheck: python3 -m mypy scripts/merge_codex_automation_prs.py scripts/boss_drain_pass.py aragora/swarm/merge_arbiter.py
test: python3 -m pytest tests/scripts/test_merge_codex_automation_prs.py tests/swarm/test_boss_drain.py tests/swarm/test_merge_arbiter.py -q --no-header -p no:randomly --timeout=300
e2e: not applicable
review: github-pr-comments plus fresh independent read-only subagent
notification: PR comment only at final readiness if authorized by the run state
```

## Plan and Log Paths

- **Plan:** `docs/elves/merge-snapshot-provenance-9677/plan.md`
- **Learnings:** `docs/elves/merge-snapshot-provenance-9677/learnings.md`
- **Execution log:** `docs/elves/merge-snapshot-provenance-9677/execution-log.md`
- **Branch:** `codex/merge-snapshot-provenance-9677`
- **PR number:** #9874
- **Plan hash at session start:** `be693b42e00277e0e7fdb3d7cfa4501474e7d724e2d0cf27a24855ca0c1f1e55`

## Collision and Live-State Notes

- Shared checkout `$HOME/Development/aragora` is dirty and behind; it is read-only for this run.
- PR #9677 moved during staging to `13fc8850700927e2e4bd4809f77ddf44f282382e` in another worktree. This run does not own that object.
- Main staging base is `953c501c2147026c2c996c3f95001580e326ec52`; re-fetch before implementation and stop if this branch tip moves unexpectedly.
- Main required contexts were green for lint, typecheck, sdk-parity, Generate & Validate, and TypeScript SDK Type Check; main quorum was skipped as expected.

## Launch Prompt

> The run is staged. Start now. Use `$HOME/.codex/worktrees/merge-snapshot-provenance-9677-20260829/aragora` and read `docs/elves/merge-snapshot-provenance-9677/survival-guide.md` first, followed by `.elves-session.json`, learnings, plan, and execution log. Set the Stop Gate to no, re-read operator steering, renew lease `906d51fa-4af`, verify the checked-out and remote branch still derive from the recorded fresh-main base, and create `elves/pre-batch-1`. Execute Batches 1-2 through implementation, focused validation, mutation break tests, independent review, operational-artifact cleanup, commit, push, and final draft readiness. Do not touch PR #9677 or its branch, edit workflows or protected governance, collect quorum evidence, mark ready, settle, or merge. Stop for exact-head OWNER settlement of the new Tier-4 PR.

## After Any Compaction

1. Read this file first.
2. Read the Run Control section and Stop Gate and confirm whether stopping is allowed.
3. Read `.elves-session.json`; verify the current batch, PR number, test baseline, and
   `continuation_guard`.
4. Read learnings, plan, and execution log in that order.
5. Compare the recorded plan hash, inspect active compute, and locate the first incomplete task.
6. Recheck exact branch/head, lease, operator steering, and #9677 collision state before any write.
7. If `continuation_guard.stop_allowed` is false, resume the next exact action without re-deciding
   whether the run should end.

# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART
