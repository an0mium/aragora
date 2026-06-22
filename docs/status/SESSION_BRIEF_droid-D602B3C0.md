# Session brief — droid-D602B3C0 (v12 fan-out, P63)

- Started: 2026-05-18T17:51:08Z
- Ended:   2026-05-18T18:00:00Z
- Lane: `P63-lane-registry-staleness-sweeper`
- Branch: `droid/P63-lane-registry-staleness-sweeper-20260518-175108`
- PR: none (operator path; small additive)
- Outcome: shipped

## Goal

The lane registry at `.aragora/agent-bridge/lanes.json` (51 rows: 26
completed, 20 released, 5 active) is mutated by every claim helper but
never compacted. Sessions that crash before releasing a lane leave
zombie `active` rows that block collision detection for legitimate
new owners reusing the same lane_id / branch / worktree slot.

v12 P63 spec: build a stale-claim detector that scans active rows,
identifies zombies via three independent signals, and (with `--apply`)
rewrites them in place as `status=expired` with a clean reason.

## Implementation

### `scripts/sweep_stale_lane_claims.py` (new, 290 LoC)

Pure-stdlib, no aragora package imports. Reuses the registry path
resolution conventions from `claim_active_agent_lane.py`.

Detection signals:

1. **`branch_missing`** — `lane.branch` is absent from both
   `git branch --list <branch>` (local) and
   `git ls-remote --heads origin <branch>` (remote). Strongest orphan
   signal. **Suppressed** when the row's `updated_at` is newer than
   `--branch-grace-hours` (default 1.0 h) to avoid flagging freshly-
   claimed lanes that haven't yet pushed their branch.
2. **`worktree_missing`** — `lane.worktree` is set but the path does
   not exist on disk.
3. **`stale_updated_at`** — `updated_at` is older than
   `--max-active-age-hours` (default 24 h).

Only rows whose status is in `{active, running, pending, queued, claimed}`
are evaluated. Released/completed/expired/conflict rows are passed
through untouched.

CLI flags:

| Flag | Default | Purpose |
|---|---|---|
| `--registry-path` | repo-resolved | explicit override |
| `--repo` | cwd | repo root for git lookups |
| `--max-active-age-hours` | 24 | stale_updated_at threshold |
| `--branch-grace-hours` | 1 | grace before branch_missing fires |
| `--skip-branch-check` | False | disable branch lookups |
| `--skip-remote-check` | False | local-only branch checks |
| `--apply` | False | rewrite (default dry-run) |
| `--json` | False | machine-readable output |

Atomic write via tempfile + os.replace, matching the
`claim_active_agent_lane.py` write semantics.

### `tests/scripts/test_sweep_stale_lane_claims.py` (new, 215 LoC)

9 regression tests covering:

1. `test_dry_run_reports_stale_without_writing` — full signal stack +
   verifies file untouched.
2. `test_apply_expires_stale_rows` — verifies in-place rewrite,
   conflict_reason captures all reasons, live rows preserved.
3. `test_worktree_missing_signals_stale` — worktree path missing fires
   `worktree_missing` without branch issues.
4. `test_skip_branch_check_ignores_branch_signal` — flag disables
   detection of missing branches.
5. `test_non_active_rows_are_ignored` — released/completed pass through
   even with all signals true.
6. `test_build_parser_defaults` — verify CLI defaults.
7. `test_branch_grace_period_protects_fresh_claims` — fresh claim
   with branch not yet pushed is NOT flagged.
8. `test_resolve_registry_path_prefers_repo` — registry-path resolver.
9. `test_resolve_registry_path_respects_explicit` — explicit override.

**9 / 9 passing** in 0.47 s.

## Live verification

Default dry-run against real registry (`.aragora/agent-bridge/lanes.json`,
51 rows, 4 active after my own claim, P62-codex active, Q03 active,
Q04 active):

| Mode | stale_rows |
|---|---|
| Default (grace 1h, age 24h) | **0** |
| `--branch-grace-hours 0` (force) | 1 (my own P63 lane, branch not yet pushed) |

The grace period correctly protects in-flight lanes. With the safer
default the sweeper finds 0 stale rows — exactly as expected since all
4 active claims are < 1h old.

## Files touched

- `scripts/sweep_stale_lane_claims.py` (+290 LoC, new)
- `tests/scripts/test_sweep_stale_lane_claims.py` (+253 LoC, new)
- `docs/status/SESSION_BRIEF_droid-D602B3C0.md` (this)
- `docs/status/P63-lane-registry-staleness-sweeper_RECEIPT_droid-D602B3C0.md`
- `docs/status/AGENT_FANOUT_JOURNAL.md` (appended)

## R/D compliance

- R5: lane claimed before any file write.
- R11: dry-run is the default; mutation requires `--apply`.
- D1: no destructive operations in dry-run.
- D2: rows preserved (rewritten in place, not deleted); even with
  `--apply` the script never deletes lane history.
