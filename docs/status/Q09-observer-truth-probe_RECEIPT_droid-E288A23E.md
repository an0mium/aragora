# Q09 (observer-truth-probe) receipt

- Session: `droid-E288A23E`
- Lane: `Q09-observer-truth-probe`
- Branch: `codex/droid-20260518-193455-5a1ef07b`
- PR: none (additive, hygiene/proof-loop tooling)
- Started: 2026-05-18T19:35:00Z
- Completed: 2026-05-18T19:42:00Z
- Outcome: shipped

## Result

Adds `scripts/observer_truth_probe.py` -- a read-only probe that
asserts the observer (the git checkout from which proof-loop surfaces
are read) is sitting on a clean `origin/main` checkout, per the
NEXT_STEPS_CANONICAL Observer rule.

Emits JSON with `clean`, `head_sha`, `origin_main_sha`, `ahead`,
`behind`, `untracked_count`, `uncommitted_modified_count`,
`submodule_dirty`, `reasons`, `repo_root`, `checked_at`. CLI accepts
`--repo-root`, `--strict-mode`/`--no-strict-mode` (default strict),
`--no-fetch`, `--quiet` / `--json-only`. Returns exit code 1 in strict
mode when `clean=false`.

## Tests

`tests/scripts/test_observer_truth_probe.py` -- 5 / 5 passing:

- `test_clean_checkout_exits_zero` -- tmp bare-origin + clone clean -> exit 0 + clean=true.
- `test_uncommitted_change_exits_nonzero` -- modify README -> exit 1 + uncommitted_modified_count>=1.
- `test_untracked_file_exits_nonzero` -- add stray file -> exit 1 + untracked_count>=1.
- `test_no_strict_mode_returns_zero_even_when_dirty` -- `--no-strict-mode` returns 0 + payload still flags dirty.
- `test_module_probe_pure_python` -- `probe()` dict has the 11 documented keys; clean fixture -> clean=true.

## Mypy (R20)

`python3 -m mypy scripts/observer_truth_probe.py tests/scripts/test_observer_truth_probe.py`
-> `Success: no issues found in 2 source files`.

## Live smoke

### Clean fresh worktree at current `origin/main` (acceptance case)

`git worktree add --detach /tmp/q09-smoke/clean origin/main` then
`python3 scripts/observer_truth_probe.py --repo-root /tmp/q09-smoke/clean --quiet`:

```json
{
  "ahead": 0,
  "behind": 0,
  "checked_at": "2026-05-18T19:39:58Z",
  "clean": true,
  "head_sha": "469db0e0d7a638e61281d8e8f0794bff6de3a9de",
  "origin_main_sha": "469db0e0d7a638e61281d8e8f0794bff6de3a9de",
  "reasons": [],
  "repo_root": "/private/tmp/q09-smoke/clean",
  "submodule_dirty": false,
  "uncommitted_modified_count": 0,
  "untracked_count": 0
}
```

Exit: `0`. Temp worktree removed via `git worktree remove`.

### Dirty founder root checkout (`~/aragora`)

Read-only probe with `--no-fetch` (no mutation of founder root):

```json
{
  "ahead": 0,
  "behind": 1,
  "checked_at": "2026-05-18T19:40:07Z",
  "clean": false,
  "head_sha": "d8f0469bd88baaf12da50f5eb670ec10b6a50b86",
  "origin_main_sha": "469db0e0d7a638e61281d8e8f0794bff6de3a9de",
  "reasons": [
    "uncommitted_modified=1",
    "behind_origin_main=1",
    "head_mismatch_origin_main"
  ],
  "repo_root": "<repo-root>",
  "submodule_dirty": false,
  "uncommitted_modified_count": 1,
  "untracked_count": 0
}
```

Exit: `1`. Confirms the probe detects the canonical "observer is
behind, has local mods, head diverges from origin/main" dirty case
that the Observer rule warns against.

## R/D compliance

- R5: lane `Q09-observer-truth-probe` claimed at Phase 0 before any file write.
- R19: no `--amend` of pushed history; single new commit.
- R20: mypy preempt on touched files (`scripts/observer_truth_probe.py` + test file) -> 0 issues.
- R21: no `boss-ready` operator-queue issues touched.
- R25: no raw `rm -rf` against worktrees; smoke worktree removed via `git worktree remove`.

## Code surface

| File | Δ LoC | Change |
|---|---|---|
| `scripts/observer_truth_probe.py` | +311 | new additive probe (stdlib-only) |
| `tests/scripts/test_observer_truth_probe.py` | +193 | new tests (5 cases) |
| `docs/status/Q09-observer-truth-probe_RECEIPT_droid-E288A23E.md` | + | this receipt |
| `docs/status/AGENT_FANOUT_JOURNAL.md` | +1 | journal append |

## Out of scope (per spec)

- No CI workflow / runner-label / required-check changes (AGENT_OPERATING_CONTRACT).
- No mutation of B0 / TW03 status surfaces; that is P66 / P67 territory.
- No mutation of the founder root checkout; the dirty smoke is read-only.

## Lane

`Q09-observer-truth-probe` released `status=completed` at Phase 4.
