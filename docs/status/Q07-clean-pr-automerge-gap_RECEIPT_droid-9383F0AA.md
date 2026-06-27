# Q07 (clean-pr-automerge-gap) receipt

- Session: `droid-9383F0AA`
- Lane: `Q07-clean-pr-automerge-gap`
- PR: none (investigation)
- Started: 2026-05-18T18:22:02Z
- Completed: 2026-05-18T18:28:00Z
- Outcome: shipped (findings; no code changes)

## Result

Forensic analysis of PR #7278 and PR #7268 — both marked
`mergeable: MERGEABLE / mergeStateStatus: CLEAN` but never auto-merged.

**Root cause**: both PRs are still drafts (`isDraft: True`). All
in-tree auto-mergers (`scripts/merge_codex_automation_prs.py`) and
GitHub native auto-merge explicitly skip drafts.

Secondary issues:

- PR #7278 is on a non-`codex/*` branch (`worktree-packets-...`) so
  `merge_codex_automation_prs.py` would not consider it even if marked
  ready.
- Both PR bodies lack the validation-evidence sentinel string the codex
  merger checks for (this matters only after draft-gate clears).

## Recommendation

- Author or operator should `gh pr ready 7268` and add validation-
  evidence string to body — codex merger will then pick it up.
- PR #7278 needs manual merge after draft transition.
- (v13 scope): consider extending `merge_codex_automation_prs.py` to
  optionally process `worktree-*` branches for full coverage.

## R/D compliance

- R5: lane claimed.
- R11: state probed via `gh pr view` (read-only).
- D1: no destructive operations.
- D2: no PR state mutation (drafts belong to another author).

## Lane

`Q07-clean-pr-automerge-gap` released `status=completed`.
