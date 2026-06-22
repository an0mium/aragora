# P58 (inventory-classifier-include-pr-state) receipt

- Session: `droid-56778BD7`
- Lane: `P58-inventory-classifier-include-pr-state`
- Branch: `droid/P58-inventory-classifier-include-pr-state-20260518-174107`
- PR: none (operator path; small additive)
- Started: 2026-05-18T17:41:07Z
- Completed: 2026-05-18T17:50:00Z
- Outcome: shipped

## Result

Added `--include-pr-state` flag to `codex_worktree_value_inventory.py`
that supplements `--skip-gh` with a single bulk gh prefetch of open PR
heads. **7 branches reclassified** from `harvest_candidate` →
`open_pr_or_outbox` (preserved) in live test, exceeds spec ≥7 acceptance.

## Live data (post-P59 cleanup state)

| | Without `--include-pr-state` | With `--include-pr-state` |
|---|---|---|
| `harvest_candidate_count` | 25 (extrapolated) | **18** |
| `open_pr_or_outbox` | 0 | **7** |

## Tests

37 / 37 passing in `tests/scripts/test_codex_worktree_value_inventory.py`
(added 4 new: parser default, cache-hit fast-path, cache-miss empty
return, `classify_candidate` open_pr_or_outbox via cache).

## R/D compliance

- R5: lane claimed before any file write.
- R11: snapshots captured both with and without the flag.
- D1: no destructive operations.
- D2: flag defaults to off; legacy behavior preserved.

## Code surface

| File | Δ LoC | Change |
|---|---|---|
| `scripts/codex_worktree_value_inventory.py` | +47 | prefetch + cache + flag |
| `tests/scripts/test_codex_worktree_value_inventory.py` | +88 | 4 regression tests |
| `docs/status/inventories/codex_worktree_value_20260518T174656Z_droid-56778BD7_include-pr-state.json` | +73 KB | snapshot |

## Lane

`P58-inventory-classifier-include-pr-state` released `status=completed`.
