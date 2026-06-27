# P65 (swarm-micro-worktree-triage) receipt

- Session: `droid-CCCE0130`
- Lane: `P65-swarm-micro-worktree-triage`
- Branch: none (operator path; small additive)
- PR: none
- Started: 2026-05-18T18:09:14Z
- Completed: 2026-05-18T18:20:00Z
- Outcome: shipped

## Result

Aligned `has_active_session()` in `codex_worktree_value_inventory.py`
with the canonical implementation in `codex_worktree_autopilot.py`:
`.claude-session-anchor` is a passive wrapper sentinel, not an active
lock. Widened `_is_empty_nested_wrapper()` in `safe_worktree_cleanup.py`
to recognize flat anchor-only wrappers in addition to nested ones.

**16 of 17** flagged worktree wrappers purged after the fix.
Inventory `active_or_dirty` count: 16 → 2.

## Tests

- `tests/scripts/test_codex_worktree_value_inventory.py`: 38 / 38
- `tests/scripts/test_safe_worktree_cleanup.py`: 17 / 17

## R/D compliance

- R5: lane claimed before any file write.
- R11: classification diffed pre/post.
- D1: removals via `safe_worktree_cleanup`, not `rm -rf`.
- D2: each wrapper inspected for `removable=True` before removal.

## Code surface

| File | Δ LoC | Change |
|---|---|---|
| `scripts/codex_worktree_value_inventory.py` | -11 | drop anchor from active signal |
| `scripts/safe_worktree_cleanup.py` | ±0 | widen empty-wrapper detection |
| `tests/scripts/test_codex_worktree_value_inventory.py` | +19 | 1 test |
| `tests/scripts/test_safe_worktree_cleanup.py` | +11 | 1 test |

## Lane

`P65-swarm-micro-worktree-triage` released `status=completed`.
