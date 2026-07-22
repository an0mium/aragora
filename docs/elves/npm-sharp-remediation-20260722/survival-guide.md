# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

## Mission

Run one finite, isolated dependency-remediation batch from exact current main. Remove
the inherited GHSA-f88m-g3jw-g9cj exposure in `aragora/live` without changing Next.js,
workflows, public behavior, or any non-sharp dependency; validate and independently
review the exact PR head, but never merge it.

## Run Control

- **Run mode:** finite
- **Stop policy:** blocker-only after the fresh launch; this staging call ends only at launch readiness
- **User intent:** "Safest next step is a separate base dependency-remediation lane, followed by separate authorization for #9477 evidence or settlement if desired."
- **Checkpoint due by:** none; the prior hard stop was explicitly removed
- **Checkpoint semantics:** none
- **May continue after checkpoint:** yes; no checkpoint is configured
- **Actual stop conditions:** Batch 1 and final readiness are complete, the operator stops the run, or a true authority/collision/safety blocker is recorded
- **Workspace ownership:** dedicated worktree `$HOME/.codex/worktrees/npm-sharp-remediation-b9Qayh/aragora`, owned only by session `elves-npm-sharp-remediation-20260722`
- **Branch tip at start:** staged current-main integration commit `23b853cb399a811a26a91583fbb0d0854e27d9f4` (base `563331f03e568e5b34c481bde86a5c1f89575c9e`); only later run-state commits by this session are expected, and any other movement is a collision
- **Merge policy:** user-merges; never merge or approve a merge
- **Final-response policy:** allowed for this staging handoff; after launch, disallowed until the Stop Gate again says yes or a true blocker forces it
- **Batch completion rule:** Every completed batch ends with `update execution log -> update survival guide -> update .elves-session.json -> commit -> push`; no completed work remains only in the working tree
- **Re-read rule:** Immediately after every commit and push, re-read this survival guide before doing anything else
- **Checkpoint rule:** no checkpoint is configured; a later delivery checkpoint is not a stop unless the operator explicitly makes it one
- **Continuation rule:** After launch, if work remains and no actual stop condition is met, continue without waiting for acknowledgment

## Session Budget

- **Started:** 2026-07-22 01:20 CDT
- **User returns:** never specified
- **Checkpoint expectation:** launch-ready draft PR and durable run packet from staging
- **Time budget:** unlimited; no hard stop remains
- **Average batch time so far:** not started
- **Batches remaining:** 1 of 1

## Stop Gate

- **Planned batches remaining:** 1
- **Stop allowed right now:** no
- **Why:** the operator explicitly authorized the three Docker base-image substitutions; Batch 1 must continue through implementation, repeat validation, exact-head review, cleanup, and final readiness
- **Next required action:** commit and push cycle-1 review documentation fixes plus the regression attestation, then perform mandatory fresh independent review of the new exact head

## Effort Standard

- Work as hard as you can for the full launched run. Do not be lazy.
- Maintain the same level of effort through final readiness as at the start.
- Do not settle for the minimum acceptable change, the first green check, or a shallow
  pass while deeper compatibility, delta classification, or review work remains.
- When one task is complete, immediately take the next highest-value action in Batch 1.

## Forbidden Stop Reasons

- A checkpoint, commit, or push happened while launched work remains.
- A draft PR exists or one check is green while validation/review remains.
- The user is silent, the remaining work feels large, or a summary was written.
- A batch implementation is locally complete but is not logged, committed, pushed,
  re-read, and reviewed.

## Memory Surfaces

- **Plan:** `docs/plans/npm-sharp-security-remediation-20260722.md`
- **Survival guide:** this file
- **Learnings:** `docs/elves/npm-sharp-remediation-20260722/learnings.md`
- **Execution log:** `docs/elves/npm-sharp-remediation-20260722/execution-log.md`
- **Structured state:** `.elves-session.json`
- **Durable docs:** `.ai-docs/manifest.md` if present; no durable edits are planned

## Non-Negotiables

- PR #9477 is outside this lane: no edits, comments, evidence, settlement, readiness,
  or merge operations.
- No `.github/**`, governance, merge-authority, Next.js, SDK, or non-sharp dependency changes.
- Stop before a product commit/push when the measured delta exceeds five distinct
  transitive packages or includes any out-of-sharp-family package, absent explicit
  operator approval for that exact delta.
- Never merge; security changes are operator-review-required.
- Never use destructive git commands, force-push, or rebase. If the branch tip moves
  unexpectedly, stop and report the collision.
- Never weaken tests. Preserve all existing behavior and inspect the cumulative diff.

## Launch Readiness

- [x] Plan cleaned and saved to disk
- [x] Survival guide generated from the current plan
- [x] Learnings file initialized
- [x] Execution log initialized with the batch contract and preflight evidence
- [x] Dedicated branch and worktree created from exact remote main
- [x] Ownership registry and strict branch lease claimed; no resource conflict found
- [x] Draft PR #9484 opened and recorded
- [x] Preflight reproduced the exact high-severity audit finding
- [x] Run controls and non-negotiables recorded; the hard stop is removed
- [x] Stop Gate initialized with `Stop allowed right now: no` for launched execution; the staging-only handoff gate above is yes
- [x] Launch prompt prepared for the next call

## Current Phase

**Status:** Cycle-1 product review clean; documentation fixes and fresh-head review pending

**Active batch:** Batch 1: Prove and remediate the sharp advisory

**What was just finished:** At published head `40017dc55857fdcede2117e78d3ce30e1f26bcdb`, all six required checks passed and every PR feedback surface was empty. Independent non-countable review found no product blocker. It classified the package-only SDK link and dev-compose Node 18 image as pre-existing warnings, and required corrections to the PR body, retained plan, and regression attestation before cleanup.

**Single next action:** Commit and push the corrected plan/log/session/guide, re-read this guide, then request a fresh exact-head review before cleanup.

## Active Compute

**No active paid or long-running compute.** The bounded Fable consult completed during
staging, no additional consult was run, and no dev server or remote job was left running.

## Next Exact Batch

**Batch:** 1: Prove and remediate the sharp advisory

**Scope:**

- Re-check live steering, branch ownership, exact base, and baseline audit; create the rollback tag.
- Add only the sharp override, regenerate the lockfile, and classify every changed package.
- Align all three `aragora/live/Dockerfile` stages from Node 18.19 Alpine to the
  repository-standard Node 20.11 Alpine under the operator's exact authorization.
- Enforce the dependency-change auto-halt, then validate, review, and hand off only if authorized.

**Acceptance criteria:**

- [ ] Exact audit gate exits 0 with unchanged Next.js and a valid patched sharp tree.
- [ ] Lockfile delta is fully classified and within authority, or product edits are reverted and an exact approval question is emitted.
- [ ] Lint, tests, build, diff checks, PR checks/comments, and final exact-head review are clean.
- [ ] The self-hosted/quickstart Docker build target satisfies the shared Node >=20.9 engine floor.

**Risk:** sharp 0.35.x is a breaking upstream line outside Next 16.2.9's declared range,
and its platform packages may trip the >5-transitive-change auto-halt.

**Measured authority blocker:** 29 transitive entries changed. The in-family portion was
`sharp` itself plus 26 `@img/sharp*` / `@img/sharp-libvips*` entries. Two out-of-family
entries also moved: `@emnapi/runtime` 1.8.1→1.11.2 and
`node_modules/sharp/node_modules/semver` 7.7.4→7.8.5. The operator explicitly
approved this exact delta at 2026-07-22 11:02 CDT; any different delta requires a new halt.

**Rollback tag:** `elves/npm-sharp-remediation-20260722/pre-batch-1` at `c06ab98ac8cd9d40213eaca457cd2a1c74355d8c`. The requested generic `elves/pre-batch-1` name was already occupied locally by unrelated commit `17af7a7a590e3e04543e1f0f1b9df2faf039dc96`, so it was preserved rather than overwritten.

## Post-Checkpoint Control Loop

Every completed batch must end with a commit and push. Immediately after every commit
and push, re-read this survival guide before doing anything else.

1. Re-read operator steering and renew the strict work lease.
2. Confirm the branch tip is one created by this session.
3. Reconcile any active compute; none should remain idle.
4. Poll all PR comments and checks after every push.
5. If the Stop Gate still say `Stop allowed right now: no`, continue immediately.
6. Stop only at a recorded true blocker or after final readiness rewrites the gate to yes.

## Documentation Triggers

- Promote only stable, reusable dependency or validation facts to `learnings.md`.
- No public behavior or architecture docs should change for this dependency-only patch.
- Remove Elves operational artifacts from the final product diff only after the final
  readiness review, then rerun the final review if cleanup changes the head.

## Tool Configuration

```yaml
working-directory: aragora/live
install: npm ci --ignore-scripts
audit: npm audit --omit=dev --audit-level=high
dependency-proof: npm ls next sharp --all
lint: npm run lint
test: npm test -- --runInBand
build: npm run build
repo-check: git diff --check
review: direct-independent-review-without-countable-evidence
notification: pr-comment-only-if-required-for-disposition
```

## Plan and Log Paths

- **Plan:** `docs/plans/npm-sharp-security-remediation-20260722.md`
- **Learnings:** `docs/elves/npm-sharp-remediation-20260722/learnings.md`
- **Execution log:** `docs/elves/npm-sharp-remediation-20260722/execution-log.md`
- **Branch:** `codex/npm-sharp-security-remediation`
- **PR number:** #9484
- **Plan hash at session start:** `7a132f9cde57b7476ed40bb9b2260b65`
- **Current plan hash after review clarification:** `4b8c560f068cba25a65e8ad60228c8fa`

## After Any Compaction

1. Read this file first and re-check the Run Control section and Stop Gate.
2. Read `.elves-session.json`, especially `continuation_guard`, batch state, PR number,
   branch tip, and review dispositions.
3. Read learnings, then the plan, then the execution log, then `.ai-docs/manifest.md` if present.
4. Re-read live operator steering and verify/renew the work lease before mutation.
5. Confirm the first incomplete action from Next Exact Batch and resume immediately.

# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART
