# Lane-Registry Sweep Cadence

Operating runbook for `scripts/sweep_stale_lane_claims.py` — the periodic
hygiene pass for `.aragora/agent-bridge/lanes.json` (the agent-bridge lane
registry written by `scripts/claim_active_agent_lane.py` and read by
`scripts/agent_bridge.py`).

Last updated: 2026-05-18.

## Why this exists

The registry accumulates `active` lane rows over time. Most rows are released
cleanly by the owning session, but some never are:

- A session crashed before releasing the lane.
- A session was killed by the operator.
- An owner script bailed mid-run.

Without periodic cleanup, those zombie rows hold the lane_id, branch, and
worktree slots against new owners. The collision detector then rejects
otherwise-valid work as duplicate, which causes lane renumbering churn (see
the v12 `claude-86616832` journal entry where four `P28-*` lanes stayed
`active` for hours after their journal row already said `shipped`).

The sweep also satisfies the v13 **R24 staleness contract**: any active
identity that has been silent past its expected heartbeat must be
auto-transitioned, not left to mislead the next fan-out wave.

## When to run it

Run **`make sweep-stale-lanes`** (dry-run, read-only):

- Once per day during active fan-out work.
- Immediately **before** kicking off a new wave (`v_N` fan-out, swarm dispatch,
  batch validation run). A dry-run takes <2s and prevents collision-detector
  surprises.
- Whenever `scripts/agent_bridge.py operator-snapshot` reports more active
  lanes than there are real running sessions.
- After recovering from a power loss, container restart, or unclean Codex/
  Claude shutdown.

Run **`make sweep-stale-lanes-apply`** (mutating) **only** after a dry-run has
been reviewed and the candidate rows look correct. The apply rewrites stale
rows in-place with `status=expired` and a `conflict_reason` describing which
signal(s) tripped. Rows are never deleted; downstream auditors can still
inspect the history.

This document does **not** install cron / launchd / systemd entries. The
sweep is an operator-driven cadence and stays that way until a follow-on
v-series lane (`R24` automation) decides to wire it into a scheduled
workflow.

## Invocations

```bash
# Dry-run (default; --dry-run is an explicit alias for self-documenting use).
make sweep-stale-lanes
# Equivalent to:
python3 scripts/sweep_stale_lane_claims.py --dry-run

# Apply (mutating) — only after the dry-run has been audited.
make sweep-stale-lanes-apply
# Equivalent to:
python3 scripts/sweep_stale_lane_claims.py --apply

# Machine-readable JSON (useful in pipes / CI snapshots):
python3 scripts/sweep_stale_lane_claims.py --dry-run --json

# Override the registry path (e.g. when triaging another workspace):
python3 scripts/sweep_stale_lane_claims.py \
    --dry-run \
    --registry-path /path/to/lanes.json \
    --repo /path/to/repo
```

`--dry-run` and `--apply` are **mutually exclusive**; combining them is a
parser error.

## Interpreting the output

The default text output prints a one-line summary followed by one line per
stale record:

```
registry=/.../lanes.json total=51 active=4 stale=1 applied=False
  STALE lane=foo-lane owner=ghost-session reasons=branch_missing,stale_updated_at
```

A stale row trips one or more of these three independent signals:

| Signal              | Meaning                                                                                                                                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `branch_missing`    | `lane.branch` is not present in `git branch --list <branch>` **and** not present in `git ls-remote --heads origin <branch>`. Strongest orphan signal. Skipped for rows younger than `--branch-grace-hours` (default 1h) so fresh claims have time to push. |
| `worktree_missing`  | `lane.worktree` is set but the path does not exist on disk.                                                                                                                                              |
| `stale_updated_at`  | `updated_at` is older than `--max-active-age-hours` (default 24h).                                                                                                                                       |

Multiple signals can fire for the same row — the apply pass concatenates them
into `conflict_reason="stale: <reason1>,<reason2>"`.

## Recovery: re-claiming a misidentified lane

If a sweep accidentally expired a lane that is in fact still alive (e.g.,
because the owning worktree lives outside the repo-local
`.worktrees/` tree, or because `--max-active-age-hours` was set too tight),
re-claim the lane with the canonical helper:

```bash
python3 scripts/claim_active_agent_lane.py \
    --lane-id <original-lane-id> \
    --owner-session <your-session-id> \
    --status active \
    --branch <original-branch> \
    --worktree "$(pwd)" \
    --force
```

The `--force` flag is required because the previous row now has
`status=expired` and `--force` overwrites an existing claim. The new row
gets a fresh `updated_at` and recovers the registry slot.

If you suspect the sweeper itself was over-eager (false-positive on multiple
rows), reduce its aggressiveness on the next run via:

- `--max-active-age-hours 72` — widen the heartbeat threshold to 3 days.
- `--branch-grace-hours 6` — let freshly-claimed lanes ride for 6h before
  branch-missing fires.
- `--skip-branch-check` / `--skip-remote-check` — temporarily disable the
  branch-existence signal entirely if `gh` / network are unreliable.

## Operator cron snippet (reference only — do NOT install)

The runbook deliberately leaves scheduling to operator discretion. Operators
who choose to wire this up may use a snippet like the following in a personal
`crontab -e` (NOT a repo-level cron file):

```cron
# Dry-run lane-registry sweep every 6 hours, log to a rotating file.
# Do NOT add --apply here; the apply step stays a manual escalation.
0 */6 * * * cd /path/to/aragora && \
    /usr/bin/make sweep-stale-lanes \
    >> ~/.aragora/logs/sweep-stale-lanes.log 2>&1
```

The repository does not ship this cron entry by policy: scheduled mutation
of the lane registry is an operator-gated decision (per AGENT_OPERATING_CONTRACT
"never break main / public API / release flow / CI" and the v13 R21 operator-
queue rule).

## See also

- `scripts/sweep_stale_lane_claims.py` — the sweeper itself.
- `scripts/claim_active_agent_lane.py` — the write-side helper that creates
  the rows this sweeper inspects.
- `scripts/agent_bridge.py` — the registry reader; its
  `operator-snapshot` view is the human-readable surface this sweeper keeps
  honest.
- `docs/status/P63-lane-registry-staleness-sweeper_RECEIPT_droid-D602B3C0.md`
  — the v12 lane that shipped the sweeper script.
