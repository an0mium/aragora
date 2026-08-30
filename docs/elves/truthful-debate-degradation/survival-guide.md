# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

Read order: this file -> `.elves-session.json` -> `learnings.md` -> `plan.md` ->
`execution-log.md` -> relevant repo governance.

## Mission

Run the Aragora Truthful Debate Degradation campaign. Close one unowned Tier 0-2
contract gap at a time so partial agent failure is explicit from Arena execution
through consensus, MetaPlanner, Decision Receipts, and existing API/stream
consumers, while preserving substantive surviving work.

## Run Control

- **Run mode:** open-ended campaign with bounded PR units
- **Stop policy:** only the explicit staging boundary, a plan termination condition,
  or an explicit user stop
- **User intent:** implement the full campaign autonomously and merge eligible Tier
  0-2 units through the normal exact-head governed path
- **Checkpoint due by:** none
- **Checkpoint semantics:** the current staging boundary is a required handoff, not
  product completion
- **May continue after checkpoint:** yes, in the fresh launch call
- **Actual stop conditions:** the plan's completion or blocked conditions, plus the
  required stage/launch split for this call
- **Workspace ownership:** dedicated worktree
  `$HOME/.codex/worktrees/truthful-debate-degradation-9872-20260830/aragora`
- **Branch:** `codex/truthful-debate-degradation-9872-20260830`
- **Draft PR:** `#9907`
- **Lane:** `codex-b-truthful-debate-degradation-9872`
- **Lease:** `3976a551-db6`, work identity `issue:9872`
- **Branch tip at start:** `0ecbf67178f406351c9741463c6cb8c1f785c802`
- **Plan hash at session start:** `b2b85bac7780d2fddd68f86315d3a92f`
- **Merge policy:** Tier 0-2 only, exact-head evidence and OWNER settlement through
  the repository's normal protected helper path; never `--admin`, force, or bypass
- **Batch completion rule:** update log and guide, commit, push, then immediately
  re-read this guide
- **Re-read rule:** re-read this guide after every commit and push

## Session Budget

- **Started:** 2026-08-30 18:15 America/Chicago
- **User returns:** never specified
- **Checkpoint expectation:** launch-ready Batch 1 lane with clean baseline and one
  draft PR
- **Time budget:** open-ended
- **Batches remaining:** 7 matrix batches, subject to per-cycle ownership refresh

## Stop Gate

- **Planned batches remaining:** 7
- **Stop allowed right now:** yes
- **Why:** Elves requires a fresh launch call after staging; no product batch has
  begun
- **Next required action:** in a fresh call, renew the lease, set this gate to no,
  create `elves/truthful-debate-degradation-9872/pre-batch-1`, and execute Batch 1

After launch, rewrite this gate to `Stop allowed right now: no` while any legal
batch remains.

## Non-Negotiables

- Re-ground exact main, required checks, halt, ownership, overlap, reviewer
  reservation, disk, and outbox at every outer cycle.
- Never touch the seeded exclusions or any newly overlapping active scope.
- Never send live product inference. Deterministic fake agents only.
- One campaign PR open at a time; each product diff stays within eight files and
  approximately 500 changed lines.
- No workflows, protected governance, merge/evidence infrastructure, dependencies,
  deployment, secrets, new endpoints, or breaking schemas.
- Reuse `DebateResult.agent_failures`; never create a competing provenance schema.
- One independent exact-head non-countable review. One bounded P2 repair maximum;
  a further P2 parks the PR and ends the campaign.
- Evidence is last. Never use `--admin`, force-push, amend a published commit, or
  bypass normal settlement.
- The shared checkout is observation-only. Never absorb its tracked or untracked
  changes.

## Current Phase

**Status:** Launch-ready staging

**Active batch:** Batch 1 - MetaPlanner proposer-loss truth (#9872)

**What was just finished:** Live overlap and owner proof, exact-main verification,
lane/lease claim, clean dedicated worktree, and a 114-test deterministic baseline.

**Single next action:** Launch Batch 1 from the exact staged branch after re-reading
steering and renewing lease `3976a551-db6`.

## Active Compute

No active paid or long-running compute. Reviewer/Fable capacity is reserved by
Codex A, so the initial Fable consult is skipped.

## Next Exact Batch

**Batch:** 1 - MetaPlanner proposer-loss truth (#9872)

**Scope:**

- Add proposal-only timeout configuration with an unset compatibility default.
- Classify structured proposer failures and substantive proposals.
- Preserve surviving proposals; use heuristics only when none survive.
- Add goal metadata and truthful legacy receipt reasoning.
- Add deterministic regression and mutation/break tests.

**Acceptance criteria:** use the Batch 1 section in `plan.md` verbatim.

**Risk:** Error placeholders already appear in proposal maps. Classify from the
structured failure record and expected participant order without weakening healthy
debates.

**Rollback tag to create before implementation:**
`elves/truthful-debate-degradation-9872/pre-batch-1`

## Cycle Start Checklist

1. Read this guide and `.elves-session.json`.
2. Read operator steering for the lane and record an outcome if a message exists.
3. Verify local HEAD equals the branch remote; unexpected movement is a collision.
4. Refresh `origin/main` with a targeted fetch.
5. Recheck required main signals and merge halt state.
6. Refresh active Codex tasks, Factory processes, lane/lease ownership, open PR
   file sets, reviewer reservations, disk, and outbox.
7. Renew lease `3976a551-db6`; stop on ownership loss.
8. Skip Fable while another task reserves reviewer capacity.
9. Create the rollback tag, rewrite the Stop Gate to no, and begin Batch 1.

## Validation Commands

Use the repository environment without provider credentials:

```text
python3 -m pytest tests/nomic/test_meta_planner.py -q
python3 -m pytest tests/debate/phases/test_proposal_phase.py tests/debate/test_proposal_phase.py -q
ruff format --check <changed Python files>
ruff check <changed Python files>
mypy <changed source files> --ignore-missing-imports
git diff --check
bash scripts/automation_pr_preflight.sh origin/main HEAD
make ci-required
```

Run the full CI-equivalent gate only after focused tests and mutation break tests
are green. Scrub provider credentials for every product test invocation.

## PR and Review Rules

- Draft PR #9907 is the sole open campaign PR.
- Operational session files are temporary and must be removed from the final
  product diff before readiness.
- Independent review must inspect the exact product head and may not count as
  quorum evidence.
- Any push invalidates prior exact-head review/evidence.
- Countable evidence, settlement, and merge occur only after the branch is
  non-draft, mergeable, exact-head stable, and all non-quorum required checks are
  green.

## Campaign Rotation

After a governed merge:

1. Verify the merge commit and current-main required checks.
2. Release the old lease and complete the old lane.
3. Append the merge receipt to the uncommitted campaign ledger.
4. Refresh all exclusions and run the next discovery pass.
5. Create a new fresh-main worktree, branch, lane, lease, and at most one draft PR
   for the next eligible matrix cell.

## Tool Configuration

- Python: `python3` from the active Aragora development environment
- Ruff: `ruff` from the active Aragora development environment
- GitHub: authenticated `gh`, repository `synaptent/aragora`
- Reviewer: independent subagent or repository reviewer only when unreserved;
  never a live product provider
- Fable: at most one goal cycle per outer cycle, only when explicitly unreserved

## Recovery

If implementation goes wrong, create a recovery branch from the most recent
run-scoped rollback tag. Never reset, rebase, force-push, or delete another
worktree. If branch, remote, or lease ownership moves unexpectedly, stop with an
exact collision handoff.

# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART
