# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

Read in this order before any action: this survival guide -> `.elves-session.json` ->
`docs/elves/public-wedge-contract-closure-20260830/learnings.md` ->
`docs/plans/2026-08-30-public-wedge-contract-closure.md` ->
`docs/elves/public-wedge-contract-closure-20260830/execution-log.md` -> root `AGENTS.md` ->
`docs/AGENT_OPERATING_CONTRACT.md` §Conductor -> `docs/REVIEW_AUTHORITY_PRINCIPLES.md`.

## Mission

Execute the first bounded unit of the Public Wedge Contract-Closure Campaign: compose the existing
zero-key offline demo, native receipt verification, ODR export, and clean-installed standalone
verifier into one deterministic contract with tamper rejection. This branch must not change public
behavior unless that contract exposes a concrete defect during implementation.

## Run Control

- **Run mode:** finite (Batch 1 only; the persistent campaign continues through fresh-main units)
- **Stop policy:** staged-launch boundary now; blocker-only after launch
- **User intent:** "PLEASE IMPLEMENT THIS PLAN: Aragora Public Wedge Contract-Closure Campaign"
- **Checkpoint due by:** none
- **Checkpoint semantics:** none
- **May continue after checkpoint:** yes
- **Actual stop conditions after launch:** Batch 1 governed merge plus verified current-main health,
  or any campaign mechanical termination condition in the plan
- **Workspace ownership:** dedicated worktree
  `$HOME/.codex/worktrees/public-wedge-contract-closure-b1-20260830/aragora`
- **Branch:** `codex/public-wedge-contract-closure-b1-20260830`
- **Branch tip at start (collision tripwire):**
  `eaaac1a07480b64d3ba4a060fbf36773ae36e589`
- **Fresh-main base:** `origin/main` at
  `eaaac1a07480b64d3ba4a060fbf36773ae36e589`
- **Lease:** `1f1ea372-286`, owner session
  `019f807d-64fd-7502-8043-9c42ec27be88`
- **Merge policy:** merge-commit-on-green for Tier 0-2 only, explicitly authorized by the user;
  exact-head evidence/settlement and live helper gates remain mandatory; never squash or admin
- **Final-response policy:** allowed only at the Elves staging boundary or an actual stop condition
- **Batch completion rule:** update log -> update this guide -> commit -> verify lease -> push ->
  re-read this guide
- **Re-read rule:** immediately after every commit and push
- **Checkpoint rule:** no time checkpoint is configured; a pushed checkpoint is progress evidence,
  not permission to stop after launch
- **Continuation rule:** after launch, continue without user acknowledgment while work remains and
  no actual stop condition has fired

## Session Budget

- **Started:** 2026-08-30 17:00 America/Chicago
- **User returns:** unknown
- **Checkpoint expectation:** a staged draft PR now; after launch, exact-head Batch 1 readiness or a
  precise circuit-breaker handoff
- **Time budget:** unlimited within the bounded Batch 1 run
- **Average batch time so far:** not started
- **Batches remaining:** 1 of 1

## Stop Gate

- **Planned batches remaining:** 1
- **Stop allowed right now:** yes
- **Why:** the Elves two-call protocol requires a fresh user launch after staging; product edits are
  forbidden in this setup call
- **Next required action:** user sends the short launch prompt recorded below; then set this gate to
  `no`, re-read steering, renew the lease, verify the collision tripwire, and create the unique
  rollback tag before product edits

## Launch Prompt

> The run is staged. Start now. Use
> `$HOME/.codex/worktrees/public-wedge-contract-closure-b1-20260830/aragora` and read
> `docs/elves/public-wedge-contract-closure-20260830/survival-guide.md` first, followed by
> `.elves-session.json`, learnings, plan, and execution log. Set the Stop Gate to no, re-read
> operator steering, renew lease `1f1ea372-286`, verify the collision tripwire and remote branch,
> and create `elves/public-wedge-contract-closure-b1/pre-batch-1`. Execute Batch 1 through
> implementation, focused and CI-equivalent validation, mutation break testing, independent
> non-countable review, operational-artifact cleanup, exact-head evidence, OWNER settlement, and
> normal governed Tier 0-2 merge. Never use admin or squash merge. Permit at most one bounded P2
> repair; a second P2 parks the PR and ends the campaign. Do not touch any excluded or overlapping
> surface, run inference/Fable, change dependencies/workflows/governance/deployment/secrets, or
> start a second campaign PR before this one is merged and current-main health is verified.

## Effort Standard

- Work as hard as you can for the full bounded run. Do not be lazy.
- Build the smallest proof that closes the full user journey, not a set of helper-level assertions.
- Preserve the exact existing public interfaces and make failures truthful and diagnosable.
- Do not settle for the minimum acceptable change or stop at a green component test while the
  composed contract remains unproved.
- When Batch 1 completes, take the next highest-value action required for exact-head validation,
  review, cleanup, settlement, and governed landing.
- Keep the final product diff within the campaign's eight-file/~500-line bound.

## Forbidden Stop Reasons

After launch, none of the following permits stopping while Batch 1 remains safe and in scope:

- a commit or push succeeded;
- the draft PR or a green CI run exists;
- a local component test passed;
- the user is silent;
- a checkpoint or concise summary was written;
- the remaining validation or review work is substantial.

Only an actual stop condition in the plan, an explicit user stop, or the mandatory staging boundary
currently recorded in the Stop Gate permits a final response.

## Non-Negotiables

- Shared checkout `$HOME/Development/aragora` is observation-only and currently dirty and
  behind; never repair, clean, commit, or push from it.
- Recheck active tasks, lanes, leases, open-PR changed files, reviewer reservations, current-main
  required checks, disk, and outbox before any product edit and before evidence.
- Expected Batch 1 product scope is only `tests/cli/test_receipt_roundtrip.py`. Any need to change
  producer/export/verifier code requires a fresh overlap check and must remain within the existing
  documented contract; otherwise stop.
- Do not touch PR #9015 paths, SDK modes/spectate, keyless-doctor, recurring-status,
  decision-quality, deployment, governance, merge/evidence infrastructure, workflows, protected
  files, dependencies, or secrets.
- Never invoke provider inference. The one Fable cycle permitted by the campaign was unavailable
  before staging and must not be retried in this outer cycle.
- Never weaken an existing test. Adding a composed regression and mutation test is the authorized
  test change.
- Evidence is exact-head and last. Do not collect it before the terminal independent review and
  all non-quorum required checks are green.

## Collision and Ownership Checks

1. `git fetch origin main codex/public-wedge-contract-closure-b1-20260830`
2. Local and remote branch tips must match, and the tip must descend from the recorded
   `.elves-session.json` `staging_plan_head`; inspect every descendant commit and stop on any
   non-self/foreign commit.
3. `python3 scripts/agent_bridge.py read-steering --lane-id
   codex-public-wedge-contract-closure-b1-20260830 --json`
4. `python3 scripts/check_work_lease.py codex/public-wedge-contract-closure-b1-20260830
   --session-id 019f807d-64fd-7502-8043-9c42ec27be88 --renew --json`
5. Re-query all open PR changed files. Reject the unit if any intersects expected Batch 1 files.
6. Re-query live required main checks. The non-required `contract-drift-program-trajectory` failure
   observed at staging is advisory; any required main failure lasting more than 30 minutes triggers
   main-red incident mode.

## Current State

- Clean worktree and branch created from `origin/main` exact head
  `eaaac1a07480b64d3ba4a060fbf36773ae36e589`.
- GitHub auth and branch-protection visibility are healthy.
- Required main contexts `lint`, `typecheck`, `sdk-parity`, `Generate & Validate`, and
  `TypeScript SDK Type Check` passed at the base head; main-only quorum was skipped as expected.
- No open PR intersects the expected Batch 1 files or producer/export paths.
- Manual zero-key chain passed from a temporary directory: demo 0, native verify 0, ODR export 0,
  local `aragora-verify` 0.1.2 wheel build/install 0, standalone verify 0.
- Existing focused baselines: `tests/cli/test_receipt_roundtrip.py` 3 passed; standalone verifier
  CLI/example tests 8 passed.
- The gap is composed proof, not a reproduced product failure.
- Elves 2.33.0 is available; installed 1.12.0 remains pinned for this run.

## Launch Readiness

- [x] Plan cleaned and saved
- [x] Survival guide, learnings, execution log, session JSON, and uncommitted ledger initialized
- [x] Dedicated fresh-main worktree and branch confirmed
- [x] Lane ownership and work lease confirmed
- [x] Draft PR #9903 opened and recorded
- [x] GitHub auth, current-main required checks, open-PR file overlap, and focused baseline checked
- [x] Run mode, merge authorization, exclusions, circuit breakers, and Stop Gate recorded
- [x] Stop Gate initialized with `Stop allowed right now: no` for the launched run; the current
  staged value is temporarily `yes` only for the mandatory fresh-call boundary
- [x] Fresh-call launch prompt prepared

## Current Phase

**Status:** Launch-ready

**Active batch:** Batch 1 - Compose the zero-key offline receipt proof

**What was just finished:** The validated staging packet was committed and pushed, and draft PR
#9903 was opened at staging plan head `ac8d546127bc42de1ced6920fbd5e5194a889c96`.

**Single next action:** Receive the user's fresh launch call, set the Stop Gate to `no`, renew the
lease, recheck steering/head/overlap/main health, create the unique rollback tag, and implement
Batch 1.

## Next Exact Batch

**Batch:** 1 - Compose the zero-key offline receipt proof

**Scope:**

- Add a deterministic clean-install contract in `tests/cli/test_receipt_roundtrip.py` without
  modifying existing assertions.
- Exercise one artifact through offline demo, native verification, ODR export, installed standalone
  verification, and tamper rejection.
- Validate focused receipt/export/walkthrough suites and the current required CI-equivalent gate.

**Acceptance criteria:**

- [ ] Every untampered seam exits `0` and retains the same non-empty receipt ID.
- [ ] The mutated ODR exits `1` through the installed standalone verifier.
- [ ] No provider transport, excluded path, overlapping file, skip, or test weakening is introduced.

**Risk:** Packaging subprocesses could become slow or accidentally resolve PyPI instead of local
source; use absolute source paths and bounded timeouts.

**Rollback tag:** `elves/public-wedge-contract-closure-b1/pre-batch-1`

## Post-Checkpoint Control Loop

Every completed batch must end with a commit and push. Immediately after every commit and push,
re-read this survival guide before doing anything else. A pushed checkpoint is proof of progress,
not permission to stop.

After every launch-time commit and push:

1. Re-read this guide before any other action.
2. Recheck the exact local and remote head against `.elves-session.json`.
3. Renew and verify the branch lease and read operator steering.
4. Reconcile active compute; stop any campaign resource that is idle or ambiguous.
5. Update Current Phase, Stop Gate, Next Exact Batch, execution log, and session JSON if scope or
   stop behavior changed.
6. Does the Stop Gate still say `Stop allowed right now: no`, or does `.elves-session.json` still
   say `continuation_guard.stop_allowed: false`? If yes, continue immediately; do not treat green
   CI or a pushed checkpoint as completion.

## Tool Configuration

```bash
export CI=true
export HOMEBREW_NO_AUTO_UPDATE=1
export NEXT_TELEMETRY_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export NPM_CONFIG_YES=true
```

- Python: `python3` (3.13.0 at staging)
- Focused pytest: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider ...`
- Packaging: build local wheels with absolute source paths; install to temporary targets outside
  the repo; never confuse `aragora-verify` (PyPI spec) with `./aragora-verify` (local source).
- Existing global tag `elves/pre-batch-1` belongs to an older run and must not be moved. Use the
  unique tag `elves/public-wedge-contract-closure-b1/pre-batch-1`.

## Active Compute

- No campaign server, pod, paid job, or inference request is active.
- Temporary packaging directories under `/tmp/aragora-public-wedge*` are disposable local proof
  artifacts and are not tracked.

## Campaign Termination

After a successful governed Batch 1 merge, verify the merge commit and current-main required
checks, update the uncommitted campaign ledger, and continue only from a new fresh-main unit. End
the persistent goal immediately on any successful-termination condition in the plan.

## After Any Compaction

1. Read this file from the top.
2. Read `.elves-session.json`, learnings, plan, and execution log in the stated order.
3. Re-read root `AGENTS.md`, `docs/AGENT_OPERATING_CONTRACT.md` §Conductor, and
   `docs/REVIEW_AUTHORITY_PRINCIPLES.md` before evidence or merge work.
4. Verify worktree, branch, local head, remote head, lane, lease, and open-PR file overlap.
5. Re-read the Run Control section and Stop Gate, then compare them with
   `.elves-session.json`'s `continuation_guard`.
6. Trust those durable controls and Current Phase over recalled chat state.
7. Resume the Single next action; do not restart discovery or retry the unavailable Fable consult.
