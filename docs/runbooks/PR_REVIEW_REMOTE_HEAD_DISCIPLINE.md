# PR Review Loop and Remote-Head Discipline

Use this runbook for manual PR review, `aragora review-pr`, and any tranche or
queue operator flow that needs a truthful merge decision.

## Durable Rules

- Review the remote PR head, not the ambient local worktree.
- Treat the latest `review-pr` artifact set as the durable review record.
- Treat queue autonomy as assistive, not as permission to skip review or merge
  gates.
- If truth cannot be established from the remote head, stop with
  `blocked_nonreviewable`.

## Why This Exists

We saw two failure modes that looked like autonomy but were actually drift:

1. Reviewers inspected whatever branch happened to be checked out locally and
   produced stale findings against code that was no longer on the PR head.
2. Queue experimentation taught the same lesson at a higher level: autonomy is
   only durable when it degrades toward truthful inspection and explicit
   confirmation, not when it assumes every lane is safe to merge unattended.

The proven loop is therefore:

1. Read the live PR head.
2. Produce structured review artifacts from that head.
3. If needed, apply a bounded fix on the same branch.
4. Re-review the refreshed remote head.
5. Merge only after the latest remote-head review and required checks agree.

## Proven `review-pr` Loop

From a clone of the repository that owns the PR:

```bash
gh auth status
aragora review-pr <pr-number>
```

`aragora review-pr` does the following:

- Fetches live PR metadata including `headRefName`, `headRefOid`, and
  `baseRefName`.
- Reviews the current remote diff, not local uncommitted state.
- Writes durable artifacts under `.aragora/review-pr/pr-<n>/<timestamp>/`.
- Returns stable statuses:
  - `passed`
  - `changes_requested`
  - `blocked_nonreviewable`

The first-pass artifact directory should contain:

- `run.json` for the run summary
- `review-1.json` for the structured review result
- `review-1.diff` for the exact reviewed diff snapshot

If the first review requests changes and you want Aragora to attempt a bounded
repair on the same branch:

```bash
aragora review-pr <pr-number> --fixer codex --auto-rerun
```

The fix loop prepares a detached worktree from `origin/<headRefName>`, asks the
fixer to address only the blocking findings, pushes back to the same PR branch,
then re-runs review against the refreshed remote head. When that path runs, the
artifact directory also includes `fix.json` and `review-2.json`.

## Queue-v10 Autonomy Lesson

The queue lesson is simple: branch truth beats requested autonomy.

Today, queue processing resolves autonomy conservatively:

- Requested `fire_and_forget` is downgraded to `adaptive` at queue level unless
  queue auto-merge policy is explicitly enabled.
- Writable-lane submissions in `adaptive`, `checkpoint`, and `spectator` default
  to `awaiting_confirmation`, typically with `design-review` as the recommended
  next action.
- Integration only executes a merge automatically when autonomy, merge policy,
  review status, and required checks all line up. Otherwise the correct outcome
  is `awaiting_confirmation`, `needs_human`, or `request_changes`.

Operationally, this means the queue may prepare work, dispatch lanes, collect
reviews, and surface merge recommendations, but the operator should still treat
the latest remote-head review plus check state as the merge gate.

## Intended Operator Workflow

1. Start in a clean clone of the target repository and verify `gh auth status`.
2. Run `aragora review-pr <pr-number>` to establish the current remote-head
   truth.
3. If the result is `blocked_nonreviewable`, fix fetch/auth/repository context
   first. Do not substitute local file inspection for a remote-head review.
4. If the result is `changes_requested`, either fix manually on the PR branch or
   run `aragora review-pr <pr-number> --fixer <agent> --auto-rerun`.
5. If a tranche or queue lane reaches `awaiting_confirmation` or `needs_human`,
   inspect the tranche artifacts, but use `review-pr` on the live PR head before
   approving merge.
6. Merge only when the latest remote-head review is `passed`, required checks are
   green, and the merge policy permits execution.

## Manual Verification Commands

When you need to audit a finding by hand, fetch and read the remote branch
directly:

```bash
gh pr view <pr-number> --repo synaptent/aragora \
  --json headRefName,headRefOid,baseRefName
git fetch origin <headRefName> --quiet
gh pr diff <pr-number> --repo synaptent/aragora
git diff origin/<baseRefName>...origin/<headRefName>
git show origin/<headRefName>:<path>
```

## Minimal Checklist

1. Confirm the current `headRefOid`.
2. Inspect the remote diff that the review is based on.
3. Reproduce any cited finding from `origin/<headRefName>`.
4. If a fix landed, re-review the refreshed remote head before merge.
5. If the queue asks for confirmation, treat that as a real gate rather than a
   documentation artifact.
