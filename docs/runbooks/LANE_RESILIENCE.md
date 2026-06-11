# Lane Resilience Runbook

Two live failure classes from 2026-06-10/11, now systematically detected by
`scripts/fleet_sentinel.py` and remediated (bounded) by `scripts/lane_janitor.py`.

## Failure class A — silent lane death

Coordinator-spawned lanes die at setup, leaving `status=in_progress` ledgers
(`.aragora/run-*/lanes/<lane>.json`) and empty branches on origin that nobody
notices (incident: `elves/run-20260610-c06/c07/c08-*`, found only by a manual
morning sweep).

**Detection** — sentinel check `lane_liveness` breaches when:

- a ledger entry is `in_progress`, older than `--lane-max-age-hours` (default 3),
  and its branch has **zero commits ahead of origin/main** (or is absent); or
- any origin branch matching `elves/*` / `aragora/boss*` is zero-commits-ahead
  with a tip older than 24h (ledger-less orphan sweep).

**What the breach means:** a lane burned its setup window and produced nothing
durable. Its work is NOT lost (there was none); it needs relaunch.

**One-command fix:**

```bash
python3 scripts/lane_janitor.py            # dry-run: show the plan
python3 scripts/lane_janitor.py --apply    # mark dead + queue + sweep branches
```

The janitor (a) marks dead ledgers (`status=dead`, `detected_at`), (b) writes
`.aragora/run-*/lanes/RELAUNCH_QUEUE.md` listing dead lanes with their briefs,
and (c) deletes remote branches only when zero-commits-ahead AND ledger-dead
(or ledger-less orphans in lane namespaces) AND older than `--branch-ttl-hours`
(default 24). **A branch with any unique commit is never deleted.**

**Relaunch-queue convention:** the janitor never relaunches. The coordinator /
operator reads `RELAUNCH_QUEUE.md` and relaunches each lane with its original
brief, checking off entries as they go.

## Failure class B — external API degradation

GitHub GraphQL 502/504 streaks stall the publisher's branch pass; the cached
`github_health` flips `auth_ok:false`, previously breaching the sentinel
without distinguishing a transient blip from a persistent outage.

**Detection** — sentinel check `github_api_health` combines a live probe
(`gh api rate_limit`) with the trailing failed-pass streak in
`.aragora/overnight/codex-automation-publisher.log`:

- **breach** only on persistent degradation: probe fails AND streak >=
  `--persist-threshold` (default 3 consecutive failed passes);
- transient blips (short streak, or probe already recovered) stay
  visible-but-quiet: streak count + last error class (e.g. `HTTP 504`) are
  recorded in the check detail and sentinel ledger, exit stays green;
- **unknown** when the log is unreadable or the probe cannot run.

**What the breach means:** GitHub has been failing the publisher repeatedly AND
is still down right now. Check <https://githubstatus.com>; the publisher's
bounded retry (2 retries, 30s/60s backoff per branch pass, spend caps enforced
inside each attempt) rides out anything shorter.

## Routine invocations

```bash
python3 scripts/fleet_sentinel.py                  # all checks, appends ledger
python3 scripts/fleet_sentinel.py --no-ledger --checks lane_liveness,github_api_health
python3 scripts/lane_janitor.py --json             # machine-readable dry-run
```

Exit codes (sentinel): 0 ok, 1 breach, 2 unknown/blind (outranks breach).
