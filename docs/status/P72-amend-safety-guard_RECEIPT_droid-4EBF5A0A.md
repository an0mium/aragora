# P72 (amend-safety-guard) receipt

- Session: `droid-4EBF5A0A`
- Lane: `P72-amend-safety-guard`
- Branch: `codex/droid-20260518-192929-4ebf5a0a`
- PR: none (operator path; small additive)
- Started: 2026-05-18T19:29:29Z
- Completed: 2026-05-18T19:35:00Z
- Outcome: shipped

## Result

Shipped `scripts/guard_amend_pushed.sh`, a self-contained bash helper
that compares the current `HEAD` SHA against the remote tip of the
current (or `--branch`-overridden) branch and refuses to allow an
amend when the two match. Implements rule **R19** from v13 ("never
`--amend` a pushed commit"). Behavior:

| Scenario | Exit | Channel | Message excerpt |
|---|---|---|---|
| HEAD == remote tip | `1` | stderr | `AMEND-BLOCKED: HEAD is already published on <remote>/<branch>. Use a new commit instead.` |
| HEAD ahead of remote | `0` | stdout | `guard_amend_pushed: HEAD <sha> is ahead of <remote>/<branch> ... amend is safe.` |
| Branch absent remotely | `0` | stdout | `guard_amend_pushed: <remote>/<branch> not found remotely — amend is safe.` |
| Detached HEAD / not a repo / bad flag | `2` | stderr | usage hint |

Flags: `--remote NAME` (default `origin`), `--branch NAME` (default
auto-detected via `git rev-parse --abbrev-ref HEAD`), `-h|--help`.

Added a short subsection under "Automation Operating Rules" in
`AGENTS.md` (~11 lines) pointing agents at the helper and at R19.

## Smoke run (this worktree on `main`)

```
$ git rev-parse --abbrev-ref HEAD
codex/droid-20260518-192929-4ebf5a0a
$ git rev-parse HEAD
d8f0469bd88baaf12da50f5eb670ec10b6a50b86

$ bash scripts/guard_amend_pushed.sh
guard_amend_pushed: origin/codex/droid-20260518-192929-4ebf5a0a not found remotely — amend is safe.
# exit 0

$ bash scripts/guard_amend_pushed.sh --branch main
guard_amend_pushed: HEAD d8f0469bd88b is ahead of origin/main (c5309dfa4341) — amend is safe.
# exit 0
```

The blocked path is exercised by the unit tests (see
`tests/scripts/test_guard_amend_pushed.py::test_guard_blocks_when_head_equals_remote`)
which provisions a local bare `origin` and verifies the `AMEND-BLOCKED`
exit-1 contract end-to-end.

## R/D compliance

- R5: lane claimed (`P72-amend-safety-guard`) before any file write.
- R19: deliverable encodes R19 itself; no amend of pushed history
  performed in this session.
- R20: ruff + mypy clean on touched files; no `--no-verify` pushes.
- R21: no operator-queue work touched.
- R25: no worktree deletes.
- D1: no destructive operations; read-only `git ls-remote`.
- D3: AGENTS.md edit is doc-only and additive (11 lines, within the
  user-requested ≤12-line budget; the in-spec out-of-scope note about
  AGENTS.md was explicitly overridden by the lane brief).

## Tests

5/5 passing in `tests/scripts/test_guard_amend_pushed.py`:

- `test_guard_blocks_when_head_equals_remote` — dangerous path.
- `test_guard_allows_when_head_ahead_of_remote` — happy path.
- `test_guard_allows_when_branch_absent_remotely` — branch missing
  on remote.
- `test_guard_respects_explicit_branch_flag` — `--branch` override.
- `test_guard_help_flag_exits_zero` — `--help` ergonomics.

```
$ python3 -m pytest tests/scripts/test_guard_amend_pushed.py -v
... 5 passed in 1.21s
$ python3 -m ruff check tests/scripts/test_guard_amend_pushed.py
All checks passed!
$ python3 -m mypy tests/scripts/test_guard_amend_pushed.py
Success: no issues found in 1 source file
```

## Code surface

| File | Δ LoC | Change |
|---|---|---|
| `scripts/guard_amend_pushed.sh` | +112 | new (executable bash helper) |
| `tests/scripts/test_guard_amend_pushed.py` | +99 | new (5 tests) |
| `AGENTS.md` | +11 | new "Amend-guard helper" subsection |
| `docs/status/AGENT_FANOUT_JOURNAL.md` | +2 | append row |

## Lane

`P72-amend-safety-guard` to be released `status=completed` at Phase 4.

## Out of scope (per lane brief)

- No git hook installation (documentation-only adoption).
- No multi-remote semantics beyond `--remote NAME` (single remote).
- No mutation of `CLAUDE.md`, `scripts/nomic_loop.py`, or `.env`.
