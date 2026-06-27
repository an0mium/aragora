# P67 — Publication Freshness CI Probe — Receipt

- Lane: `P67-publication-freshness-probe`
- Session: `droid-20260518-192921-1465c7fa`
- Owner session id (registry): `A7B5CEA2-5F76-4B08-AA76-243F726BB14E`
- Worktree: `.worktrees/codex-auto/droid-20260518-192921-1465c7fa`
- Branch: `codex/droid-20260518-192921-1465c7fa`
- UTC timestamp: `2026-05-18T19:33:16Z`

## Scope

Implement the v13 lane P67 deliverable: a stdlib-only probe that reads
the `Last updated:` header on the two recurring proof-surface status
docs (B0 + TW-03) and exits non-zero if either is stale beyond a
configurable window.

This is observability only — no workflow / CI / runner / public-API
surface changes (per `docs/AGENT_OPERATING_CONTRACT.md`).

## Changes

Additive only.

| File | Kind | Notes |
| --- | --- | --- |
| `scripts/probe_proof_surface_freshness.py` | new | 7 KB. Pure stdlib. Reads `docs/status/B0_BENCHMARK_TRUTH_STATUS.md` and `docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md`; parses first `Last updated:` line as ISO 8601 or `YYYY-MM-DD`; emits a JSON document with one record per surface; exits non-zero if any surface is stale. Accepts `--max-age-days` (default 7) and `--surfaces` (default `b0,tw03`). |
| `tests/scripts/test_probe_proof_surface_freshness.py` | new | 11 pytest cases. Covers parser ISO/date/garbage/missing-header paths, fresh/stale single-surface probes, and the three acceptance CLI scenarios: both fresh -> exit 0, one stale -> exit 1 with offender on stderr, malformed `Last updated:` -> exit 2 with clear error. Plus `--surfaces` scoping and an unknown-surface CLI error case. |

No workflow YAML changes. No new requirements pins. No edits to B0 /
TW-03 doc bodies (P66 owns those). No `docs/dev/` README snippet —
documentation lives in the script docstring per the lane brief's "at
most" clause.

## Verification

### pytest

```
$ python3 -m pytest tests/scripts/test_probe_proof_surface_freshness.py -q
...........                                                              [100%]
11 passed in 0.90s
```

### mypy (R20 enforcement)

```
$ python3 -m mypy scripts/probe_proof_surface_freshness.py \
    tests/scripts/test_probe_proof_surface_freshness.py
Success: no issues found in 2 source files
```

### Live probe against `main`

Run inside the worktree, against the real (unmodified) status surfaces
checked into the repo at receipt time.

Command:

```
$ python3 scripts/probe_proof_surface_freshness.py --pretty
```

Stdout:

```json
{
  "fresh": false,
  "max_age_days": 7,
  "surfaces": [
    {
      "age_days": 1.2058,
      "fresh": true,
      "last_updated": "2026-05-17T14:36:51Z",
      "path": "docs/status/B0_BENCHMARK_TRUTH_STATUS.md",
      "surface": "b0"
    },
    {
      "age_days": 31.2748,
      "fresh": false,
      "last_updated": "2026-04-17T12:57:28Z",
      "path": "docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md",
      "surface": "tw03"
    }
  ]
}
```

Stderr:

```
error: stale proof surface(s) detected: tw03(age_days=31.27)
```

Exit code: `1` (one stale surface present — this is the *expected*
healthy probe behavior given that P66 has not yet refreshed TW-03).

The non-zero exit demonstrates the contract end-to-end: the B0 surface
is within the 7-day window (~1.2 days old at receipt time); the TW-03
surface is ~31 days stale and is correctly flagged as the offender on
stderr in the format `tw03(age_days=31.27)`.

## Operator usage notes

- Suggested cadence: every 24 hours from a non-required cron job
  (LaunchAgent or GHA cron). The probe is observability-only and must
  never be wired into the main-branch required check list.
- `--surfaces b0` / `--surfaces tw03` lets a follow-on lane scope the
  probe to just the surface it owns.
- The script never invokes git, gh, or any network call — safe to run
  inside a sandboxed CI runner with no secrets.

## Contract compliance

- R19 (no `--amend` after push): N/A — this is a fresh commit on a
  worktree branch; no remote yet at receipt write time.
- R20 (mypy on every touched file): clean, see above.
- R21 (do not touch `boss-ready` / `codex` / `codex-automation`
  issues): no GitHub mutation performed; no labels touched.
- R25 (no `rm -rf` of worktrees): no cleanup operations performed.
- AGENT_OPERATING_CONTRACT — `additive only`, no workflow / runner /
  public surface changes, no protected file edits.

## Lane registry

Acquired at Phase 0 via `scripts/claim_active_agent_lane.py`:

```
{
  "branch": "codex/droid-20260518-192921-1465c7fa",
  "goal": "P67 publication freshness CI probe",
  "lane_id": "P67-publication-freshness-probe",
  "owner_session": "A7B5CEA2-5F76-4B08-AA76-243F726BB14E",
  "source": "droid",
  "status": "active",
  "updated_at": "2026-05-18T19:29:47Z",
  "worktree": "<repo-root>/.worktrees/codex-auto/droid-20260518-192921-1465c7fa"
}
```

Released at Phase 4 by writing a `--status released` row with the same
`lane_id` + `owner_session`.
