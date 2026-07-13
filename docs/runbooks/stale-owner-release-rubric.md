# Stale-Owner Release Rubric

Use this rubric when a conductor, queue-drain lane, or worktree hygiene pass
finds a lane whose recorded owner is no longer live. It is a decision aid, not
permission to mutate state. Release stale ownership only through supported
helpers after the proof below is current.

For worktree cleanup, use
[`RUNBOOK_WORKTREE_FLEET_AUDIT.md`](RUNBOOK_WORKTREE_FLEET_AUDIT.md). For PR
review truth, use
[`PR_REVIEW_REMOTE_HEAD_DISCIPLINE.md`](PR_REVIEW_REMOTE_HEAD_DISCIPLINE.md).
This runbook is scoped to owner release and takeover decisions.

## Release Classes

Classify the lane into exactly one class.

| Class | Criteria | Action |
| --- | --- | --- |
| `ACTIVE` | The owner heartbeat is current, a matching process is running, an active-session marker exists, or the mailbox contains an unresolved owner-specific restriction. | Do not take over. Route work to the owner or wait for release. |
| `PRESERVE` | The worktree is dirty, has untracked files, has unique commits not reachable from `origin/main`, is tied to an open PR head, or helper output cannot prove the state. | Preserve the lane and write one exact handoff/blocker. |
| `RELEASE-CANDIDATE` | The owner is stale, no active-session signal remains, the worktree is clean, unique commits are mapped to a merged PR or explicit retirement record, and no mailbox restriction blocks takeover. | Release or supersede through the supported owner/lane helper, then claim before mutation. |
| `HUMAN-GATED` | Releasing the owner would discard unknown work, bypass an operator restriction, change protected queue state, or settle Tier 3+ risk. | Stop and request an exact operator decision. |

When signals conflict, choose the safer class in this order:

`HUMAN-GATED` over every other class, `ACTIVE` over `PRESERVE`,
and `PRESERVE` over `RELEASE-CANDIDATE`.

## Required Proof

Capture fresh proof in the handoff, blocker comment, or lane receipt before any
release action.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
python3 "$REPO_ROOT/scripts/identify_lane_owner.py" --pr <number> --json
python3 "$REPO_ROOT/scripts/read_operator_steering.py" --pr <number> --json
git -C "$REPO_ROOT" fetch origin --prune
git -C "$REPO_ROOT" worktree list --porcelain
git -C <worktree-path> status --porcelain=v1 --untracked-files=all
git -C <worktree-path> rev-parse HEAD
if git -C <worktree-path> merge-base --is-ancestor HEAD origin/main; then echo "ancestor"; else echo "not-ancestor"; fi
python3 "$REPO_ROOT/scripts/safe_worktree_cleanup.py" --repo "$REPO_ROOT" inspect <worktree-path>
```

Set `REPO_ROOT` from the repository checkout running the helper. Do not derive
`--repo` from `<worktree-path>`; that argument must identify the repository
whose worktree registry is being inspected, while `<worktree-path>` is the
candidate target.

If `read_operator_steering.py` returns messages, write an outcome receipt before
any mutation:

```bash
python3 "$REPO_ROOT/scripts/read_operator_steering.py" --pr <number> --outcome held --outcome-note "<reason>" --json
```

Use the narrowest true outcome (`obeyed`, `held`, `stale`, `superseded`,
`blocked`, or `completed`). A plain `read` receipt proves inspection only; it is
not owner release and does not satisfy an unresolved restriction.

If the target is not a PR lane, use only the selectors the helpers support:
`identify_lane_owner.py --branch <branch>` or `--worktree <worktree-path>`, and
`read_operator_steering.py --branch <branch>` or `--lane-id <lane-id>`. If the
right selector is unclear, classify the lane as `PRESERVE` or `HUMAN-GATED`.

The `merge-base` result is a preservation signal, not the whole decision. Exit
`0` means the worktree head is reachable from `origin/main`. Exit `1` means it is
not reachable; that can still be releasable only when the unique commits are
mapped to a merged squash PR, an explicit fold/adopt receipt, or an explicit
retirement record. Without that mapping, classify as `PRESERVE`.

## Release Steps

1. Fetch the current base and read the mailbox before classifying.
2. Classify the lane with the table above.
3. If the class is `RELEASE-CANDIDATE`, release or supersede through the
   supported helper for that lane type. Do not edit lane files by hand.
4. Claim a fresh lease before any write:

   ```bash
   python3 scripts/check_work_lease.py <branch> --pr <number> --claim --json
   python3 scripts/check_work_lease.py <branch> --claim --json
   ```

   Use the `--pr` form for PR lanes and the plain branch form for branch-only
   or non-PR lanes.

5. Re-run the owner check and proceed only if this session is now the active
   owner.

## Explicit Non-Goals

- Do not delete a worktree as part of owner release.
- Do not force-push, amend a pushed head, or rewrite another owner session.
- Do not post duplicate operator requests when a current blocker already
  exists for the same head.
- Do not collect model evidence merely because the owner is stale; exact-head
  checks and quorum rules still apply.
- Do not treat a stale owner as settlement approval, human risk settlement, or
  permission to bypass branch protection.

## Handoff Template

```text
Stale-owner classification for <lane-or-pr>

- live head:
- owner helper result:
- mailbox result:
- worktree:
- worktree status:
- unique-commit status:
- class:
- action taken:
- next legal action:
```

The next legal action must be one of: wait for active owner, preserve and hand
off, release through helper then claim, or request exact operator decision.
