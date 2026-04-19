# Batched PR Review Triage

Last updated: 2026-04-19

Status: design doc

## Decision

Aragora should not implement bot auto-approval for engineering PRs.

The correct pattern is:

1. bots prepare bounded PRs continuously
2. machine review produces an advisory packet on the current PR head
3. a human settles the merge decision in one short batched review session
4. approved PRs merge automatically only after the explicit human decision
5. rejected PRs re-enter a bounded repair loop with the rejection reason attached

This preserves the trust wedge Aragora is trying to sell: receipt-backed review
that speeds human judgment without removing it.

## Why This Exists

The rejected auto-approver path crossed the wrong boundary.

Aragora already has:

- producer discipline in [docs/briefs/automation-merge-contract.md](../briefs/automation-merge-contract.md)
- single-PR machine review in [aragora/cli/commands/review_pr.py](../../aragora/cli/commands/review_pr.py)
- auto-merge harvest in [aragora/swarm/merge_arbiter.py](../../aragora/swarm/merge_arbiter.py)
- batch-triage concepts in [docs/strategy/INBOX_TRIAGE_INSTRUMENTATION_PLAN.md](../strategy/INBOX_TRIAGE_INSTRUMENTATION_PLAN.md)
- evidence-first founder review rhythm in [docs/plans/2026-03-25-weekly-founder-review-rhythm.md](2026-03-25-weekly-founder-review-rhythm.md)

The missing piece is not more autonomy. The missing piece is a disciplined
human-in-the-loop settlement layer for open PRs.

## Problem Statement

Today the automation path has three gaps:

1. `review-pr` is framed as a machine reviewer on a single live PR head, but its
   current publish path can emit GitHub `APPROVE` reviews on machine pass.
2. `merge_arbiter` merges ready PRs when checks pass, without a required human
   settlement event.
3. founder review is still per-PR and context-switch heavy, so throughput
   pressure pushes the system toward unsafe "just let the bot finish it"
   instincts.

The goal is to remove review friction without removing human inspection.

## Goals

- Let one human review 20-50 small PRs in 10-15 minutes.
- Keep all machine output advisory until a human makes a settlement decision.
- Reuse existing Aragora review, queue, and merge substrate.
- Bind every human action to the exact PR head SHA and packet SHA.
- Keep repair loops bounded and explicit when a PR is rejected.

## Non-Goals

- Bot auto-approval.
- Bot-only merge on "all green" CI.
- Replacing GitHub review semantics with a parallel hidden control plane.
- Forcing humans to read every diff line when the risk packet is obviously low.
- Broad workflow restocking or queue widening.

## Operating Principle

The automation boundary is:

- non-human-gated: prepare, validate, summarize, prioritize, repair
- human-gated: approve for merge, request changes, waive a flagged risk

That is the right boundary for both product truth and EU-AI-Act-style oversight
claims.

## Proposed Operator Flow

### 1. Producer lane

Unchanged from the automation merge contract:

- one bounded issue or maintenance task
- scoped branch
- targeted validation
- `automation_pr_preflight.sh`
- draft or ready PR depending on truth of the branch state

### 2. Advisory packet generation

Each candidate PR gets a machine-generated review packet keyed to the current
remote head SHA.

Packet inputs:

- PR metadata from GitHub
- current check state
- diff stats and touched paths
- machine review from `review-pr`
- declared validation from the PR body or repair journal
- stale/duplicate/conflict signals

Packet outputs:

- one short reviewer summary
- touched subsystem list
- diff size summary
- validation evidence
- risk flags
- machine recommendation: `approve_candidate`, `needs_human_attention`, or
  `repair_first`

Important: this packet is advisory only. It must never count as a GitHub
approval.

### 3. Morning queue build

The system compiles a founder queue from open automation PRs and sorts it by
review leverage:

1. `ready_now`
   required checks green, no meaningful failures, packet fresh on current head
2. `needs_attention`
   mergeable but packet flags high-risk subsystem, large diff, or weak
   validation
3. `repairable`
   a bounded CI or review blocker exists with a credible next fix lane
4. `parked`
   stale, conflicting, duplicate, oversized, or blocked on non-bounded failures

Default queue should show only `ready_now` and `needs_attention`. Repairable and
parked lanes belong in a second pass.

### 4. Founder review session

The human runs one interactive session, for example:

```bash
aragora review-queue run --limit 30
```

For each PR, the session shows:

- number, title, author branch
- head SHA and packet freshness
- CI summary
- diff stats
- touched subsystems
- machine review summary
- top risk flags
- exact validation command/result

Actions:

- `a` approve and enable merge
- `r` request changes with one-line reason
- `d` defer
- `o` open full diff or changed-file list
- `p` open raw packet JSON
- `q` quit session

The tool should optimize for skim-first review, not blind review. The human can
settle quickly when the packet is low risk and can expand the diff when the
packet says it is warranted.

### 5. Merge and repair routing

- `approve`
  posts a human GitHub `APPROVE` review tied to the current head SHA and enables
  auto-merge, or otherwise marks the PR eligible for merge arbiter harvest
- `request changes`
  posts a human `REQUEST_CHANGES` review, stores the reason in the packet
  receipt, and optionally dispatches a bounded fix worker
- `defer`
  leaves a receipt and optional label such as `review:deferred`

## Proposed CLI Surface

Do not overload inbox `triage`. This is engineering PR review, not email triage.

Add a new queue-oriented surface:

```bash
aragora review-queue build [--limit 100] [--json]
aragora review-queue run [--limit 30] [--ready-only]
aragora review-queue packet <pr>
aragora review-queue act <pr> --approve|--request-changes|--defer
aragora review-queue digest [--hours 24]
```

This keeps the existing `review-pr` command for single-PR deep review and adds
the missing batched settlement loop on top.

## Existing Components To Reuse

### `review-pr`

Reuse:

- remote-head fetch logic
- diff capture
- structured findings
- artifact persistence under `.aragora/review-pr`
- optional bounded fixer loop

Required change:

- split machine advisory review from human settlement review
- machine-generated PR output must default to `COMMENT`, not `APPROVE`

`review-pr` should become a packet generator for queue review, not a bot
approver.

### `merge_arbiter`

Reuse:

- candidate PR listing
- check classification
- ready-lane and full-suite gating

Required change:

- do not merge a ready PR unless there is an explicit human settlement signal

The cleanest rule is:

- merge candidate if and only if
  - checks are green
  - PR is not draft
  - current head SHA matches packet/head seen during approval
  - there is a fresh human approval review or explicit `review:settled` receipt

### founder review rhythm

Reuse the packet pattern from the weekly founder memo:

- proof
- blockers
- stop-doing

The morning PR queue is the daily analogue:

- ready now
- risky now
- repair next
- park

## Packet Schema

Each packet should serialize to JSON and be comment-renderable.

Suggested schema:

```json
{
  "pr_number": 6272,
  "repo": "synaptent/aragora",
  "head_sha": "abc123",
  "base_sha": "def456",
  "packet_sha": "sha256:...",
  "generated_at": "2026-04-19T13:00:00Z",
  "queue_bucket": "ready_now",
  "ci": {
    "required": "green",
    "full_suite": "green",
    "non_passing": []
  },
  "diff": {
    "files_changed": 4,
    "insertions": 72,
    "deletions": 13
  },
  "subsystems": ["swarm", "cli", "docs"],
  "validation": [
    {
      "command": "bash scripts/automation_pr_preflight.sh origin/main HEAD",
      "result": "passed"
    }
  ],
  "risk_flags": [
    "control-plane",
    "docs-sync"
  ],
  "machine_review": {
    "status": "passed",
    "summary": "Scoped observer-truth fix with test coverage.",
    "findings": []
  },
  "recommendation": "needs_human_attention",
  "repair_hint": null
}
```

## Queue Prioritization Rules

Prioritize for human time, not machine neatness.

Sort descending by:

1. merge-ready and fresh packet
2. high leverage, small diff, low risk
3. human-blocked but locally understandable
4. aging ready PRs that are still safe

Demote:

- stale head vs packet mismatch
- conflicting or duplicate PRs
- very large diffs
- PRs with unresolved meaningful failures
- lanes blocked by broad infra or flaky signals without a bounded next action

## Review Semantics

### Machine role

The machine can:

- summarize
- highlight risk
- suggest approval
- suggest repair
- draft the review comment

The machine cannot:

- approve on behalf of a human
- self-waive a risk flag
- flip a PR into mergeable state by itself

### Human role

The human must be the actor that:

- decides "this is safe enough to merge"
- decides "this needs changes"
- decides "this is deferred"

The human does not need to manually reconstruct CI state or diff topology for
every PR. That is the point of the packet.

## Receipt Model

Add a founder-review receipt parallel to inbox triage receipts.

Each session should persist:

- reviewed PRs in order
- packet SHA and head SHA seen by the reviewer
- action taken
- optional reason text
- wall-clock review time
- whether the action later proved correct

This supports future calibration without collapsing the human gate.

## Safety Rails

1. Packet freshness gate
   if head SHA changes after packet generation, the PR cannot be approved
   without packet refresh.
2. Dirty observer warning
   run review sessions only from a clean reconciled worktree.
3. Oversized diff gate
   very large PRs cannot appear in `ready_now`; they require explicit expansion.
4. Advisory-only machine review
   no machine `APPROVE` events.
5. No hidden merge path
   merge arbiter and GitHub auto-merge must both require the same human
   settlement signal.

## Metrics

Track the system on three axes.

### Throughput

- PRs reviewed per session
- median time per PR
- merge latency after human approval

### Quality

- percent of approved PRs reverted or repaired within 48h
- percent of rejected PRs later merged after one repair cycle
- percent of packet recommendations matching human decision

### Trust

- percent of merged automation PRs with explicit human settlement
- percent of merges whose packet/head SHA matched at approval time
- percent of queue items that required diff expansion beyond the packet

## Rollout Plan

### Phase 1: advisory packet only

- make `review-pr` advisory-only by default
- build packet JSON and comment renderer
- no merge-path changes yet

### Phase 2: founder queue CLI

- add `review-queue build/run`
- founder can approve/request changes/defer in one loop
- persist founder-review receipts

### Phase 3: merge settlement enforcement

- require explicit human settlement before merge arbiter merge
- disallow merge on green CI alone for automation PRs

### Phase 4: bounded repair handoff

- `request changes` can dispatch a fix worker with the human reason attached
- repair returns to the same queue with a fresh packet

## Acceptance Criteria

The design is successful when all of the following are true:

1. a human can clear a morning queue of 20+ small PRs in under 15 minutes
2. no automation PR merges without an explicit human settlement event
3. machine packet generation removes most per-PR context reconstruction work
4. rejected PRs re-enter a bounded repair lane instead of generating duplicate
   branches
5. the workflow is something Aragora can honestly present as a design-partner
   example of "AI speeds judgment without replacing oversight"

## Explicit Rejection

Do not build:

- bot auto-approval
- hidden approval labels set by bots
- auto-merge on "all green" without human settlement
- a new subsystem that duplicates `review-pr` and `merge_arbiter`

The right product move is not "close the loop harder." It is "make the human
loop fast enough that keeping it is obviously worth it."
