# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

## Mission

Complete the issue #9409 inference-site inventory/static-allowlist prerequisite as one bounded, review-ready PR. Do not change runtime routing or send inference requests.

## Run Control

- **Run mode:** finite
- **Stop policy:** hard deadline or genuine blocker
- **User intent:** "yes to all in best order proceed and execute according to your recommendation"
- **Checkpoint due by:** 2026-07-20 20:15 America/Chicago
- **Checkpoint semantics:** hard stop boundary
- **May continue after checkpoint:** no
- **Actual stop conditions:** final readiness for this bounded PR, a dependency/authority blocker with no safe workaround, or the 20:15 hard stop.
- **Workspace ownership:** dedicated worktree; resolve the exact path with `git rev-parse --show-toplevel`
- **Branch tip at start:** `9247f44918534e9ac29d37b50be53e4b978b41c8`
- **Merge policy:** user-merges; never merge
- **Final-response policy:** disallowed until the Stop Gate says yes
- **Batch completion rule:** update execution log -> update survival guide -> commit -> push.
- **Re-read rule:** after every commit and push, re-read this file before doing anything else.
- **Time allocation:** implement 34%, validate 33%, review 33%.

## Session Budget

- **Started:** 2026-07-20 17:06 America/Chicago
- **Hard stop:** 2026-07-20 20:15 America/Chicago
- **Time budget:** about 3 hours
- **Checkpoint expectation:** a bounded PR with implementation, validation, independent review, and exact readiness state.
- **Batches remaining:** 1 of 1

## Stop Gate

- **Planned batches remaining:** 0
- **Stop allowed right now:** yes, after required report generation and operational-artifact cleanup
- **Why:** Batch 1 is implemented, fully validated, independently reviewed clean at `5b06270451`, and all six required checks pass.
- **Next required action:** generate the Elves Report, remove operational artifacts, push cleanup, and poll the clean PR one final time.

## Non-Negotiables

- VibeProxy is transport only, never a reviewer family.
- Never use port 8317.
- No inference requests, reviewer evidence, burn-in, merge, settlement, or later #9409 runtime units in this PR.
- No runtime routing, CI workflow, governance, public API, SDK, auth, or endpoint-pinning changes.
- Preserve direct-only status for CI, production, credential validation, public gateways, and evidence/merge authority.
- Never use destructive git commands, weaken tests, share the branch/worktree, or merge.

## Launch Readiness

- [x] Plan cleaned and saved to disk
- [x] Survival guide initialized
- [x] Learnings file initialized
- [x] Execution log initialized
- [x] Dedicated branch and worktree created
- [x] Branch/worktree ownership and lease claimed
- [x] Draft PR #9439 opened
- [x] Preflight validation dry run completed
- [x] Run mode, deadline, and non-negotiables recorded
- [x] Stop Gate initialized to no
- [x] Launch instruction approved by the user's latest message

## Current Phase

**Status:** Complete; report and cleanup in progress

**Active batch:** Batch 1: Inventory and enforcement

**What was just finished:** Exact-head independent review found no P0-P2 blocker; all six required GitHub checks pass at `5b06270451`.

**Single next action:** Generate the Elves Report, clean all ephemeral run artifacts, push, and perform the final live poll.

## Active Compute

No active paid or long-running compute. The completed Fable consult is advisory input only.

## Next Exact Batch

**Batch:** 1: Inventory and enforcement

**Scope:**

- Inventory production OpenAI/Anthropic inference construction sites with stable anchors.
- Add an explicit proxy-eligible/direct-only manifest and deterministic static check.
- Add maintenance docs and focused regression tests.

**Acceptance criteria:**

- [x] Exact current inventory passes; unclassified and stale entries fail.
- [x] Port 8317, including zero-padded textual forms, fails closed.
- [x] Protected direct-only categories are asserted.
- [x] Focused and broad gates plus exact repaired-head independent review are clean.

**Risk:** Avoid noisy broad string matching and fragile line-number anchors.

**Rollback tag:** `elves/vibeproxy-inference-allowlist-9409/pre-batch-1`

## Post-Push Checklist

After every push: re-read this guide, poll all PR comments/checks, renew owner liveness if needed, verify no unexpected branch-tip movement, and continue while Stop Gate says no.

## Reactivation Handoff

Read this file, `.elves-session.json`, learnings, the plan, and the execution log in that order. Resume the single next action above without asking for approval.
