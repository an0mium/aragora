# Session brief — droid-CCCE0130 (v12 fan-out, P65)

- Started: 2026-05-18T18:09:14Z
- Ended:   2026-05-18T18:20:00Z
- Lane: `P65-swarm-micro-worktree-triage`
- Branch: none (operator path; small additive)
- PR: none
- Outcome: shipped

## Goal

The latest worktree-value inventory flagged **16** wrappers under
`.worktrees/codex-auto/` as `active_or_dirty` despite all of them being
abandoned single-session stubs (April 30 — May 17). The classifier kept
them in `preserve` decisions because `has_active_session()` treated the
presence of a `.claude-session-anchor` file as an active-session signal.

But `.claude-session-anchor` is a **passive wrapper sentinel** laid down
by `scripts/claude-wt` at session start, *not* an active-session lock.
The companion `_has_active_session()` in `codex_worktree_autopilot.py`
correctly uses only the three active locks
(`.claude-session-active`, `.codex_session_active`,
`.nomic-session-active`). The inventory was the outlier.

## Fix

### `scripts/codex_worktree_value_inventory.py`

```python
def has_active_session(candidate_root, repo_path):
    return bool(active_lock_files(candidate_root, repo_path))
```

Removed the secondary `.claude-session-anchor` glob check. Behavior
now matches `_has_active_session()` in `codex_worktree_autopilot.py`.

### `scripts/safe_worktree_cleanup.py`

P59 introduced `_is_empty_nested_wrapper()` requiring a `.worktrees/`
subdir. P65 widened it to also recognize **flat anchor-only** wrappers
(a directory whose only file is a sentinel and which has no other
content). The function now returns `True` for both shapes:

1. Nested wrapper: dir has `.worktrees/<sub>/.session-anchor` only.
2. Flat anchor: dir has just `.claude-session-anchor` only.

Both correctly short-circuit `_worktree_is_dirty()` to `False`. Without
this fix, running `git status --short` from a flat-anchor wrapper walks
up to the parent repo and reports the parent's unstaged changes as
"dirty" — a transient false positive that blocked cleanup.

## Tests

`tests/scripts/test_codex_worktree_value_inventory.py`:

- Added `test_anchor_only_wrapper_is_not_active` — anchor-only dir
  classifies as `no_git_cache_residue` (cleanup_candidate=True).

`tests/scripts/test_safe_worktree_cleanup.py`:

- Added `test_worktree_is_not_dirty_for_flat_anchor_only_wrapper` —
  flat anchor wrapper short-circuits the dirty check.

Total: **38 / 38** inventory tests + **17 / 17** cleanup tests passing.

## Live verification

| Inventory class | Before fix | After fix |
|---|---|---|
| `active_or_dirty` | 16 | **2** (only real active sessions) |
| `no_git_cache_residue` | 0 | **16** |
| `cleanup_candidate_count` | 1 | 17 |

Subsequent bulk purge via `safe_worktree_cleanup remove --purge-path`:
**16 of 17 wrappers removed** (the 17th likely became missing-path
mid-loop). Final inventory: `active_or_dirty: 2` (real sessions),
`cleanup_candidate_count: 5` (remaining edge cases).

## Files touched

- `scripts/codex_worktree_value_inventory.py` (-11 LoC, simplified)
- `scripts/safe_worktree_cleanup.py` (Δ helper: drop `.worktrees/` req)
- `tests/scripts/test_codex_worktree_value_inventory.py` (+19 LoC, 1 test)
- `tests/scripts/test_safe_worktree_cleanup.py` (+11 LoC, 1 test)
- `docs/status/SESSION_BRIEF_droid-CCCE0130.md` (this)
- `docs/status/P65-swarm-micro-worktree-triage_RECEIPT_droid-CCCE0130.md`
- `docs/status/AGENT_FANOUT_JOURNAL.md` (appended)

## R/D compliance

- R5: lane claimed before any file write.
- R11: snapshots compared pre/post fix.
- D1: removals via `safe_worktree_cleanup remove --purge-path` (not raw `rm -rf`).
- D2: removable=True dirty=False verified before each removal.
- D3: behavior aligned with the canonical implementation in
  `codex_worktree_autopilot.py` (not invented).
