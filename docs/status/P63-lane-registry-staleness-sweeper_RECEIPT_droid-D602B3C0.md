# P63 (lane-registry-staleness-sweeper) receipt

- Session: `droid-D602B3C0`
- Lane: `P63-lane-registry-staleness-sweeper`
- Branch: `droid/P63-lane-registry-staleness-sweeper-20260518-175108`
- PR: none (operator path; small additive)
- Started: 2026-05-18T17:51:08Z
- Completed: 2026-05-18T18:00:00Z
- Outcome: shipped

## Result

New `scripts/sweep_stale_lane_claims.py` detects stale active lane
claims via 3 signals (branch_missing, worktree_missing, stale_updated_at)
with a grace-period guard to protect freshly-claimed lanes. Default is
`--dry-run`; `--apply` rewrites stale rows in place with
`status=expired`. **9/9 tests** + ruff clean. Live dry-run: 0 stale
rows currently (all 4 active claims are recent).

## R/D compliance

- R5: lane claimed before any file write.
- R11: snapshot captured; default is non-mutating.
- D1: no destructive operations.
- D2: rows rewritten in place; never deleted.

## Code surface

| File | Δ LoC | Change |
|---|---|---|
| `scripts/sweep_stale_lane_claims.py` | +290 | new |
| `tests/scripts/test_sweep_stale_lane_claims.py` | +253 | new (9 tests) |

## Lane

`P63-lane-registry-staleness-sweeper` released `status=completed`.
