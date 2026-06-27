# Session brief — droid-9383F0AA (v12 fan-out, Q07)

- Started: 2026-05-18T18:22:02Z
- Ended:   2026-05-18T18:28:00Z
- Lane: `Q07-clean-pr-automerge-gap`
- Branch: none (investigation; no code changes)
- PR: none
- Outcome: shipped (findings)

## Goal

v12 Q07 specced a forensic investigation into why two PRs marked
`mergeable: MERGEABLE / mergeStateStatus: CLEAN` never auto-merged
since their authors stopped iterating on them ~1 day ago:

- PR #7278 — feat(review-queue): keyboard sign-off + per-decision
  timing for packets route
- PR #7268 — docs(settlement): static signoff UI for open-queue packet

## Findings

### PR #7278

| Field | Value |
|---|---|
| state | OPEN |
| isDraft | **True** |
| mergeable | MERGEABLE |
| mergeStateStatus | CLEAN |
| reviewDecision | (empty) |
| author | an0mium |
| created | 2026-05-17T17:00:29Z |
| updated | 2026-05-17T17:00:29Z |
| headRefName | `worktree-packets-keyboard-throughput-20260517` |
| checks | 5 SUCCESS + 3 SKIPPED |

### PR #7268

| Field | Value |
|---|---|
| state | OPEN |
| isDraft | **True** |
| mergeable | MERGEABLE |
| mergeStateStatus | CLEAN |
| reviewDecision | (empty) |
| author | an0mium |
| created | 2026-05-17T14:57:54Z |
| updated | 2026-05-17T17:00:28Z |
| headRefName | `codex/settlement-packet-ui-20260517` |
| checks | 5 SUCCESS + 2 SKIPPED |

## Root cause

Two independent gates block both PRs from any in-tree auto-merger:

### Gate 1 — Draft state

Both PRs are still drafts. Auto-mergers in `scripts/` (notably
`merge_codex_automation_prs.py`) explicitly skip drafts:

```python
elif pr.is_draft:
    reason = "draft"
```

GitHub's native auto-merge (`gh pr merge --auto`) also refuses drafts.

### Gate 2 — Codex namespace filter

`merge_codex_automation_prs.py` is gated to `codex/*` branches:

```python
if not pr.head_ref.startswith("codex/"):
    reason = "not_codex_branch"
```

- PR #7268 (`codex/settlement-packet-ui-...`) would pass this gate.
- PR #7278 (`worktree-packets-keyboard-throughput-...`) is excluded by
  branch prefix and would not be considered even if marked ready.

### Gate 3 — Validation evidence string

The codex merger also requires `_has_validation_evidence(pr.body)`,
which expects specific magic strings in the PR body. Both PR bodies
look substantive (2.9 KB and 1.2 KB respectively) but neither contains
the validation-evidence sentinels the merger expects. Without marking
ready first this gate is moot.

## Why the PRs are stuck

Both PRs were authored by `an0mium` on 2026-05-17 and have not been
touched since (~24 hours of staleness). The author likely intended to
finish polishing before transitioning out of draft, but stalled. CI is
all green; the PRs are fundamentally mergeable but **policy-blocked**
behind the draft-gate.

## Recommendation

No automated action taken — these are another author's PRs and
transitioning them to ready-for-review is an authorship decision.

For the operator:

1. **Quick win on #7268**: it's in the `codex/*` namespace; marking it
   ready (`gh pr ready 7268`) + adding the validation-evidence string to
   the body would let the in-tree codex merger pick it up.
2. **#7278 requires manual review/merge**: not in `codex/*` namespace;
   would need to be reviewed and merged manually (or via `gh pr merge
   --auto` after marking ready).
3. **Optional follow-up**: extend `merge_codex_automation_prs.py` to
   optionally consider `worktree-*` branches too — currently those PRs
   have no autonomous merge path. This is a v13 enhancement scope.

## Files touched

- `docs/status/SESSION_BRIEF_droid-9383F0AA.md` (this)
- `docs/status/Q07-clean-pr-automerge-gap_RECEIPT_droid-9383F0AA.md`
- `docs/status/AGENT_FANOUT_JOURNAL.md` (appended)

## R/D compliance

- R5: lane claimed.
- R11: queried PR state directly via `gh pr view`.
- D1: no destructive operations.
- D2: no PR state mutation (these are another author's drafts).
