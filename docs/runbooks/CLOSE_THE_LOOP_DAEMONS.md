# Runbook: Close-the-Loop Daemons (merge executor + harvest engine)

Armed 2026-07-02 under operator authorization (epic #8762, authorization record:
https://github.com/synaptent/aragora/issues/8762#issuecomment-4860561852). These two
launchd user agents close the back half of the delivery loop: the executor merges
quorum-authorized Tier 0-2 PRs; the harvest engine folds closed/merged/orphaned work
back into the backlog.

## What runs where

| Job | Label | Schedule | Command |
|---|---|---|---|
| Merge executor | `com.aragora.ctl-merge-executor` | every 600 s | `scripts/merge_executor.py --repo synaptent/aragora --apply --max-merges 1 --receipt-dir ~/.aragora/merge-executor-receipts --halt-file ~/.aragora/MERGE_EXECUTOR_HALT --disarm-file ~/.aragora/DISARM_MERGE_EXECUTOR` |
| Harvest engine | `com.aragora.ctl-harvest` | daily 07:15 local | `scripts/harvest_outcomes.py --repo synaptent/aragora --apply --max-issues 3 --ledger-path ~/.aragora/harvest_ledger.jsonl` |

- Plists: `~/Library/LaunchAgents/com.aragora.ctl-{merge-executor,harvest}.plist`
- Wrappers: `~/.aragora/bin/ctl_{merge_executor,harvest}_tick.sh` — each tick fetches
  and detaches a dedicated daemon worktree (`.claude/worktrees/daemon-ctl`) to
  `origin/main`, so the daemons always run the latest merged code and never touch a
  dirty checkout.
- Logs: `~/.aragora/logs/ctl-merge-executor.{log,err}`, `~/.aragora/logs/ctl-harvest.{log,err}`
- Merge receipts (one JSON per executed merge): `~/.aragora/merge-executor-receipts/`
- Harvest ledger (append-only JSONL): `~/.aragora/harvest_ledger.jsonl`

> **The `--disarm-file`/`--halt-file` flags are load-bearing.** The script's built-in
> defaults live under the *repo root* (`<repo>/.aragora/merge_executor.disarm`/`.halt`),
> not `~/.aragora/`. The deployed wrapper (`~/.aragora/bin/ctl_merge_executor_tick.sh`)
> passes the flags shown above — and also pre-checks the disarm file itself — so the
> controls below work. If you ever run the executor by hand with `--apply`, pass the
> same flags or your `~/.aragora` kill switch will not be consulted.

## Safety model (why an unattended merger is acceptable)

The executor merges a PR only when ALL hold at the exact head SHA, re-verified
immediately before each merge: quorum packet satisfied (`aragora-merge-quorum`
semantics), required checks green, Tier 0-2 only (Tier 3-4 are listed in the digest,
never acted on), no unresolved dissent, main healthy (check-runs AND commit statuses).
Tier policy: `docs/REVIEW_AUTHORITY_PRINCIPLES.md`. Design/review history: PR #8767
(issue #8759); the harvest engine is PR #8768 (issue #8760).

## Controls

```bash
# EMERGENCY STOP — blocks all future merging (one-way until you remove it)
touch ~/.aragora/DISARM_MERGE_EXECUTOR
# Re-arm
rm ~/.aragora/DISARM_MERGE_EXECUTOR

# Self-halt: on red main the executor writes a persistent halt marker and stops.
# Investigate main first, then re-arm:
rm ~/.aragora/MERGE_EXECUTOR_HALT

# Health check (each tick logs one JSON summary: main_health/scanned/eligible/merged)
tail -20 ~/.aragora/logs/ctl-merge-executor.log
ls ~/.aragora/merge-executor-receipts/
tail -5 ~/.aragora/harvest_ledger.jsonl

# Pause/resume the launchd jobs (repeat per job — both are listed here)
launchctl unload ~/Library/LaunchAgents/com.aragora.ctl-merge-executor.plist
launchctl load   ~/Library/LaunchAgents/com.aragora.ctl-merge-executor.plist
launchctl unload ~/Library/LaunchAgents/com.aragora.ctl-harvest.plist
launchctl load   ~/Library/LaunchAgents/com.aragora.ctl-harvest.plist

# Full uninstall
launchctl unload ~/Library/LaunchAgents/com.aragora.ctl-merge-executor.plist \
                 ~/Library/LaunchAgents/com.aragora.ctl-harvest.plist
rm ~/Library/LaunchAgents/com.aragora.ctl-{merge-executor,harvest}.plist
```

## Weekly operator pass (≤30 min, per the close-the-loop plan)

1. `tail` both logs; confirm ticks are green and `halted=false`.
2. Review new merge receipts — each names the PR, head SHA, tier, and packet evidence.
3. Review harvest ledger deltas; triage any salvage issues it filed.
4. Work the Tier 3-4 settlement queue (packets prepared by lanes; you accept risk).
5. Re-arm anything that self-halted, after fixing the cause.

## Failure modes

| Symptom | Cause | Action |
|---|---|---|
| `MERGE_EXECUTOR_HALT` exists | main went red mid-pass | fix main, delete marker |
| tick logs `disarmed: true` | disarm file present | intentional; remove file to resume |
| no log lines for >30 min | launchd throttling after wrapper error | check `.err` file; `launchctl list \| grep ctl-` |
| executor merges nothing for days | genuinely 0 eligible (quorum-satisfied Tier 0-2 is rare) | verify with an explicit dry run of the underlying script — NOT the wrapper (the wrapper is apply-mode): `cd .claude/worktrees/daemon-ctl && python3 scripts/merge_executor.py --repo synaptent/aragora --json` (no `--apply` → mutates nothing) |
