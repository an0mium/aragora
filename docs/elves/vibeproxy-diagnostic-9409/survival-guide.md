# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

## Mission

Ship one review-ready PR for issue #9409's first work unit: a sanitized,
no-inference `scripts/check_vibeproxy.py --json` diagnostic with fake-proxy
tests, direct-path regression proof, and accurate documentation. Do not begin
any later #9409 routing, adapter, metrics, trust, burn-in, shadow, or governance
unit on this branch.

## Run Control

- **Run mode:** finite
- **Stop policy:** plan-complete, true blocker, explicit user stop, or deadline
- **User intent:** "yes to all in best order proceed and execute according to your recommendation"
- **Checkpoint due by:** 2026-07-20 20:15 America/Chicago
- **Checkpoint semantics:** hard stop boundary
- **May continue after checkpoint:** no
- **Actual stop conditions:** Batch 1 has passed the Final Readiness Review and is handed off without merge; a genuine blocker has no safe workaround; the user explicitly stops; or the hard-stop deadline arrives.
- **Workspace ownership:** the dedicated checkout returned by `git rev-parse --show-toplevel` on branch `codex/vibeproxy-diagnostic-9409`; the external lane registry records its absolute path
- **Branch tip at start (collision tripwire):** `80fea4be08952286e26ce13481be0d134d2a49d2`
- **Merge policy:** user-merges; this run never merges
- **Final-response policy:** disallowed after launch until the Stop Gate permits it or a true blocker forces a handoff
- **Batch completion rule:** update execution log -> update survival guide -> commit -> push
- **Re-read rule:** after every commit and push, re-read this survival guide before any other action
- **Checkpoint rule:** the recorded deadline is a hard stop; close the current atomic operation, leave a clean handoff, and stop
- **Continuation rule:** before the deadline, continue without waiting while Batch 1 remains and no genuine blocker exists

## Session Budget

- **Started:** 2026-07-20 12:10 America/Chicago
- **User returns:** approximately 2026-07-20 20:15 America/Chicago (eight-hour staging assumption)
- **Checkpoint expectation:** a review-ready diagnostic PR or an exact blocker handoff
- **Time budget:** approximately 8 hours
- **Average batch time so far:** not started
- **Batches remaining:** 1 of 1

## Stop Gate

- **Planned batches remaining:** 0
- **Stop allowed right now:** no
- **Why:** Batch 1 and the fresh Final Readiness Review are complete, but report generation, product-only cleanup, exact-head recheck, ready transition, and notification remain
- **Next required action:** generate the Elves report, remove staging/run artifacts, push, poll the exact cleanup head, mark #9431 ready, and notify without merge

## Effort Standard

- Work as hard as you can for the full run. Do not be lazy.
- Maintain the same effort through implementation, validation, review, and
  documentation.
- Do not settle for the minimum acceptable change, the first green test, or a
  shallow review when deeper verification remains.
- After each legal checkpoint, take the next highest-value action named in the
  plan or Stop Gate.
- Spend substantial effort on fake-proxy failure modes, redaction, direct-path
  regressions, and independent review.

## Non-Negotiables

- VibeProxy is a transport, never a reviewer/provider family.
- Never select or probe port 8317.
- Never emit credentials, authorization headers, raw response bodies, URL
  credentials, query data, prompts, or provider tokens.
- Do not make an inference request from the diagnostic.
- Preserve direct-by-default behavior and every existing URL, redirect, proxy,
  response-size, and deadline safety invariant.
- Do not touch later #9409 work units, workflows, merge authority, protected
  governance docs, public APIs, or SDKs.
- Never modify a test merely to make it pass.
- Never run destructive git commands or share this branch/worktree.
- Never merge this PR.

## Forbidden Stop Reasons

Before the hard-stop deadline, none of these permit stopping while Batch 1
remains:

- a checkpoint was reached
- a clean commit or successful push exists
- a focused test or CI check is green
- the PR is open or CI is pending
- the user is silent
- a useful summary was written
- review feels like a natural stopping point

Continue through the exact next action unless the user explicitly stops, the
hard deadline arrives, or a genuine blocker has no safe workaround.

## Launch Readiness

- [x] Plan cleaned and saved to disk
- [x] Survival guide updated from the current plan
- [x] Learnings file initialized
- [x] Execution log initialized with batch breakdown and preflight notes
- [x] Branch and dedicated checkout created
- [x] Branch lease and lane ownership recorded
- [x] PR opened and recorded
- [x] Preflight run and critical failures cleared; broad baseline warnings are recorded
- [x] Run mode, return time, non-negotiables, and Stop Gate recorded
- [x] Stop Gate initialized with `Stop allowed right now: no`
- [x] Launch prompt prepared for the next call

## Current Phase

**Status:** Final report, artifact cleanup, and ready handoff

**Active batch:** Batch 1: Diagnostic Command, Tests, and Documentation

**What was just finished:** Fresh cumulative review of exact head `828a725923` returned READY with no blocker, warning, or actionable nit. It independently reran 102 focused tests, mypy, and changed-file hooks; GitHub had 83 completed checks with no failures or feedback.

**Single next action:** Generate the temporary Elves report from the current run sources, then remove all staging/run artifacts in the final cleanup commit.

## Active Compute

No paid or long-running compute is owned by this run. A pre-existing local
VibeProxy process is listening on `127.0.0.1:8318`; it may be used for an
optional no-inference smoke test but must not be stopped or reconfigured by
this run. Unrelated Fable/conductor processes are outside this lane.

## Next Exact Batch

**Batch:** 1: Diagnostic Command, Tests, and Documentation

**Scope:**

- Add the sanitized CLI and the smallest safe extensions to the existing client.
- Add fake-proxy category tests and direct-path regression proof.
- Update the VibeProxy guide, validate, review, and hand off without merge.

**Acceptance criteria:**

- [x] Stable credential-free JSON and meaningful exit codes
- [x] No-inference fake proxy proves success and all safety failures
- [x] Existing transport/direct regressions, mypy, hooks, and automation preflight pass
- [x] Exact-tip cumulative review is clean and PR feedback is dispositioned

**Risk:** Medium; diagnostic metadata must not become a credential leak or a second unsafe HTTP implementation.

**Rollback tag:** `elves/vibeproxy-diagnostic-9409/pre-batch-1`

## Post-Checkpoint Control Loop

Every completed batch must end with a commit and push. After every commit and
push, re-read this survival guide before doing anything else. Verify the branch
and remote tip match the expected commits; recheck steering, PR comments, and
checks; name the next single action; and reconcile run-owned compute. If the
Stop Gate still say `Stop allowed right now: no`, continue immediately unless
the hard-stop deadline or a genuine blocker applies.

## Documentation Triggers

- Behavior/schema changes update `docs/guides/VIBEPROXY.md`.
- Stable transport conventions discovered during the batch go to learnings.
- One-off debugging and review evidence stays in the execution log.

## Tool Configuration

```yaml
lint: pre-commit run --files scripts/check_vibeproxy.py aragora/agents/transports/vibeproxy.py tests/scripts/test_check_vibeproxy.py tests/agents/transports/test_vibeproxy.py docs/guides/VIBEPROXY.md
typecheck: mypy --follow-imports=skip scripts/check_vibeproxy.py aragora/agents/transports/vibeproxy.py
build: none
test: python3 -m pytest tests/agents/transports/test_vibeproxy.py tests/scripts/test_check_vibeproxy.py tests/scripts/test_consult_claude.py -q
e2e: none; fake HTTP proxy integration tests are mandatory
smoke: python3 scripts/check_vibeproxy.py --json against 127.0.0.1:8318 only, with no inference request
repository-preflight: bash scripts/automation_pr_preflight.sh origin/main HEAD
review: github-pr-comments plus independent cumulative review
notification: pr-comment
```

## Rollback and Safety Rules

1. The generic `elves/pre-batch-1` name was already occupied by unrelated
   history, so use the published collision-safe tag
   `elves/vibeproxy-diagnostic-9409/pre-batch-1` for this run.
2. Never force-push, rebase a published branch, or discard work destructively.
3. Stage specific files only.
4. If the branch/worktree tip moves unexpectedly, stop as a collision.
5. If recovery is required, branch from the last known-good tag rather than
   rewriting history.

## Plan and Log Paths

- **Plan:** `docs/plans/2026-07-20-vibeproxy-diagnostic-cli.md`
- **Learnings:** `docs/elves/vibeproxy-diagnostic-9409/learnings.md`
- **Execution log:** `docs/elves/vibeproxy-diagnostic-9409/execution-log.md`
- **Survival guide:** `docs/elves/vibeproxy-diagnostic-9409/survival-guide.md`
- **Branch:** `codex/vibeproxy-diagnostic-9409`
- **PR number:** #9431
- **Plan hash at session start:** `102d7e11e48c9ae41cdcb6ab06ef5c05a764fe1b0939aa3ceb22b1f47501dea4`

## After Any Compaction

Read the Run Control section and Stop Gate first, then read, in order: this
guide, `.elves-session.json`, learnings, plan, execution log,
`docs/AGENT_OPERATING_CONTRACT.md` section Conductor,
`docs/REVIEW_AUTHORITY_PRINCIPLES.md`, and `TODO.md`. Inspect the
`continuation_guard`; when `stop_allowed` is false, resume the recorded next
action without re-deciding whether to stop. Verify mailbox, owner, lease,
branch tip, PR head, and active compute before mutating anything. Do not redo
completed work.

# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART
