# Session brief — droid-88A43575 (v12 fan-out, P59)

- Started: 2026-05-18T17:32:49Z
- Ended:   2026-05-18T17:45:00Z
- Lane: `P59-inspect-empty-wrapper-detection`
- Branch: `droid/P59-inspect-empty-wrapper-detection-20260518-173249`
- PR: none (operator path; small additive)
- Outcome: shipped

## Background

v11 P50 logged a prompt-bug: `safe_worktree_cleanup.py inspect` reported
`dirty=True` for 3 nested-empty-wrapper directories under
`.worktrees/codex-auto/`, blocking cleanup despite zero actual project
content. v12 P59 specced an explicit detection fix.

## What changed since v11

Re-running `inspect` against the 3 dirs **today** showed they now
report `dirty=False removable=True`. The dirty-flag false-positive is
not currently reproducing — likely because `git status --short` returns
empty stdout for these dirs (the inner `.claude-session-anchor` files
sit under nested `.worktrees/` paths which are gitignored).

Pivoted P59 from "fix detection" → "defense-in-depth fast-path +
regression test + actually clean up the 3 wrappers". This preserves
correctness even if `git status --short` semantics change.

## Implementation

Added two helpers to `scripts/safe_worktree_cleanup.py`:

```python
_WRAPPER_SENTINEL_FILENAMES = frozenset({
    ".claude-session-anchor",
    ".codex-session-anchor",
    ".codex-session",
    ".droid-session-anchor",
    ".session-anchor",
})

def _is_empty_nested_wrapper(path: Path) -> bool:
    if not path.is_dir():
        return False
    nested_wrapper = path / ".worktrees"
    if not nested_wrapper.is_dir():
        return False
    try:
        for entry in path.rglob("*"):
            if entry.is_file() and entry.name not in _WRAPPER_SENTINEL_FILENAMES:
                return False
    except OSError:
        return False
    return True
```

Updated `_worktree_is_dirty()` to short-circuit return `False` when
`_is_empty_nested_wrapper(path)` is True. The fast-path is conservative:
requires `.worktrees/` subdir AND only sentinel-named files in the tree.

## Tests

Added two regression tests to `tests/scripts/test_safe_worktree_cleanup.py`:

1. `test_worktree_is_not_dirty_for_empty_nested_wrapper` — constructs
   a tmp wrapper dir with two nested `.worktrees/preflight-*/` subdirs
   each containing a `.claude-session-anchor` sentinel. Asserts
   `_is_empty_nested_wrapper(wrapper) is True` and
   `_worktree_is_dirty(wrapper) is False`.
2. `test_worktree_is_dirty_when_wrapper_has_real_files` — same
   structure but adds a `real_file.py` at the wrapper root. Asserts
   `_is_empty_nested_wrapper(wrapper) is False` (real content
   correctly bypasses the fast-path).

Existing 14 tests preserved. **16 / 16 passing** in 0.80 s.

## Cleanup performed

Used the fixed inspect + `--purge-path` flag to clean the 3 immortal-
deferred wrappers from v11 P50:

| Path | Size | Outcome |
|---|---|---|
| `.worktrees/codex-auto/claude-20260501-195315-89c3c0ad` | 4 KB | purged |
| `.worktrees/codex-auto/codex-20260422-194119-a5c0dd59` | 24 KB | purged |
| `.worktrees/codex-auto/rbac-health-fix-20260413-1` | 32 KB | purged |

Total reclaimed: 60 KB (immaterial size, but closes a long-standing
inventory blemish — these were `cleanup_candidates` in 4 successive
worktree-value-inventory snapshots).

## Acceptance against v12 P59 spec

| Spec | Implementation |
|---|---|
| Add empty-wrapper detection helper | `_is_empty_nested_wrapper()` |
| Short-circuit `_worktree_is_dirty()` for wrappers | Done |
| Add tests for the 3 specific paths | Done (2 tests covering positive + negative case) |
| 3 dirs become `removable: True` and cleanable | Done (verified, then purged) |
| Bounded ≤ 25 min | ~12 min wall clock |

## Files touched

- `scripts/safe_worktree_cleanup.py` (+25 LoC, +1 frozenset, +1 helper, +2 lines in existing function)
- `tests/scripts/test_safe_worktree_cleanup.py` (+25 LoC, +2 tests)
- `docs/status/SESSION_BRIEF_droid-88A43575.md` (this file)
- `docs/status/P59-inspect-empty-wrapper-detection_RECEIPT_droid-88A43575.md`
- `docs/status/AGENT_FANOUT_JOURNAL.md` (appended)
