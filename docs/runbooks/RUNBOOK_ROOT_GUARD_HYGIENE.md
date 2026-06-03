# Root Guard Hygiene Runbook

Phase 1 is prevention-only. The guard scripts fail closed and explain why root
is unsafe for git-mutating work. They do not restore, reset, clean, stash, switch
branches, edit local automation configuration, or reconcile worktrees.

## Operator Automation Stanza

Apply this stanza to local automation prompts or local `~/.codex/automations/*`
configuration outside the repository:

> Before any git-mutating action run python3 scripts/assert_not_root_checkout.py and stand down on non-zero. Never git switch/checkout -b/commit/worktree add inside ~/aragora; always create and cd into a disposable worktree first. At start, run python3 scripts/assert_root_clean_on_main.py; if it fails, do read-only work from a clean worktree and do not mutate the shared root.

This repository change intentionally does not edit local `~/.codex/automations/*`
files. Operators apply the stanza locally after the PR lands.

## Guards

Use `python3 scripts/assert_not_root_checkout.py` immediately before any
git-mutating action. It exits:

- `0` when the current git toplevel is a linked or disposable worktree.
- `3` when the current git toplevel is the canonical shared root checkout.
- `2` for usage or system errors.

Use `python3 scripts/assert_root_clean_on_main.py` at automation startup. It
exits:

- `0` only when the canonical shared root is on `main`, has no staged,
  unstaged, or untracked paths, is not in merge/rebase/cherry-pick state, and
  local `HEAD == origin/main`.
- `3` when any root hygiene condition fails.
- `2` for usage or system errors.

Both scripts support `--json` for machine-readable reports. The canonical root
defaults to the primary worktree from `git worktree list --porcelain`; override
with `--canonical-root PATH` or `ARAGORA_CANONICAL_ROOT`.

## Non-Goals

Phase 1 does not implement preserve-first root restoration. A later phase can
design restoration with active-process checks, index-lock checks, unique commit
preservation, and explicit operator authorization.
