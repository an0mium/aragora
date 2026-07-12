# First-Error Diagnostic Receipt Protocol

Use this protocol when diagnosing a long-running failure or timeout on an exact
`origin/main` commit. It makes ownership visible before the run starts and
preserves a useful result even when pytest fails, times out, or is terminated.

This is a diagnostic protocol. It does not authorize changing a main-health
halt, rerunning CI, settling or merging a PR, or deleting another diagnostic
worktree.

## 1. Create A Dedicated Worktree

Never run a first-error diagnostic in the shared checkout or take over an
existing diagnostic worktree. Fetch the exact target and create a new detached
worktree:

```bash
REPO="$(git rev-parse --show-toplevel)"
git -C "$REPO" fetch origin --prune
MAIN_SHA="$(git -C "$REPO" rev-parse origin/main)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TMP_ROOT="${TMPDIR:-/tmp}"
TMP_ROOT="${TMP_ROOT%/}"
WT="$TMP_ROOT/aragora-main-first-error-$TS"

git -C "$REPO" worktree add --detach "$WT" "$MAIN_SHA"
mkdir -p "$WT/logs" "$REPO/.aragora/main-red"
```

Do not silently retarget the worktree if `origin/main` moves. Finish or stop
the run against `MAIN_SHA`, then start a new worktree for the new commit.

## 2. Write Ownership Before Starting

Write `logs/OWNER.json` before launching pytest. The marker is a lease-like
claim for this diagnostic object, not a development branch lease.

```json
{
  "schema": "aragora.main_first_error_owner.v1",
  "owner_session": "<lane-or-session-id>",
  "pid": 12345,
  "started_utc": "2026-07-11T17:00:00Z",
  "purpose": "exact-main first-error diagnostic",
  "main_sha": "<40-character SHA>",
  "command_log": "logs/first-error-<timestamp>.log",
  "ttl_hours": 6
}
```

The owner must update the marker before extending the TTL. A missing heartbeat
does not make an unexpired marker safe to reclaim.

## 3. Stream Output To A Durable Log

Use the interpreter, scope, and exclusions from the incident being reproduced.
Save the example as a Bash driver and execute that driver; do not paste its
`exit` paths into an interactive shell. Bound the process group with
`gtimeout` on macOS or `timeout` elsewhere, and preserve the pytest side of
the pipeline:

```bash
#!/usr/bin/env bash

cd "$WT"
LOG="logs/first-error-$TS.log"
TEST_TIMEOUT="${TEST_TIMEOUT:-1800s}"

if command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN=gtimeout
elif command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN=timeout
else
  echo "No supported timeout binary; stop as infra_error" >&2
  exit 2
fi

set +e
PYTHONUNBUFFERED=1 "$TIMEOUT_BIN" --signal=TERM --kill-after=30s "$TEST_TIMEOUT" \
  python -m pytest tests/ -x -v -p no:cacheprovider \
  --ignore=tests/connectors 2>&1 | tee "$LOG"
TEST_RC="${PIPESTATUS[0]}"
set -e
```

`PIPESTATUS` is intentionally Bash-specific, which is why the example is a
driver script rather than an interactive-shell fragment. Set `TEST_TIMEOUT`
for the incident scope: 30 minutes can be appropriate for a focused shard but
is not expected to prove that the full `tests/` suite is green. Record the
selected timeout and test scope in the receipt.

Do not rely on terminal scrollback. A run without a non-empty log is
`infra_error`, even if the operator observed output on screen.

## 4. Write The Report Before Exit

Every run must write a report, including timeout and infrastructure outcomes.
Write it before cleanup, comments, branch publication, or returning from the
driver script.

Required fields:

| Field | Requirement |
| --- | --- |
| `schema` | `aragora.main_first_error_receipt.v1` |
| `main_sha` | Exact 40-character commit tested |
| `started_utc`, `ended_utc`, `wall_seconds` | UTC timestamps and measured duration |
| `command` | Exact command, interpreter, timeout, plugins, and exclusions |
| `exit_code` | Pipeline command exit code, not `tee`'s exit code |
| `outcome` | One of `test_failure`, `timeout_only_no_failure`, `collection_error`, `infra_error`, or `green` |
| `first_nodeid` | First failing or timed-out node ID, or `null` |
| `traceback_head` | Bounded traceback head, or `null` |
| `progress` | Last completed node IDs, percentage, counts, or other bounded progress evidence |
| `log_path`, `log_sha256` | Worktree-relative path and digest |
| `next_action` | One concrete follow-up that does not overstate the result |

A timeout without a failing node ID is `timeout_only_no_failure`, not proof
that main is test-red. A missing dependency, unavailable timeout command, or
empty log is `infra_error`.

Store the report in both locations before the driver exits:

```text
<worktree>/logs/first-error-report-<timestamp>.md
<shared-root>/.aragora/main-red/first-error-report-<timestamp>.md
```

The worktree copy travels with the diagnostic context. The shared `.aragora`
copy survives later helper-mediated worktree cleanup.

## 5. Reclaim Conservatively

Treat an existing diagnostic worktree as owned unless every check below is
clear:

1. No process has the worktree as its current directory or an open file.
2. No matching pytest, timeout wrapper, or diagnostic driver process remains.
3. No unexpired `logs/OWNER.json` exists.
4. Worktree and `logs/` mtimes remain unchanged across a 90-second window.
5. The owner marker TTL has expired, if a marker exists.
6. `safe_worktree_cleanup.py inspect` reports the worktree state without
   unknown or dirty-work blockers.

Use read-only probes:

```bash
ps -eo pid=,etime=,command= |
  awk -v wt="$OLD_WT" \
    'index($0, wt) && /pytest|first-error|timeout|gtimeout/ && $0 !~ /awk -v wt=/'
lsof +D "$OLD_WT" 2>/dev/null

mtime_line() {
  if stat -f '%m %N' "$@" >/dev/null 2>&1; then
    stat -f '%m %N' "$@"
  else
    stat -c '%Y %n' "$@"
  fi
}

mtime_line "$OLD_WT" "$OLD_WT/logs"
sleep 90
mtime_line "$OLD_WT" "$OLD_WT/logs"
python3 scripts/safe_worktree_cleanup.py inspect "$OLD_WT" --json
```

If any signal is ambiguous, preserve the old worktree. Reclaim means create a
new worktree with a new owner marker; it never means reset, clean, edit, or
delete the old one. Cleanup remains a separate helper-mediated action.

## 6. Completion Checklist

- Owner marker written before pytest starts.
- Exact main SHA and exact command captured.
- Timeout wrapper verified before the run.
- Output streamed to a non-empty log.
- Pipeline exit code preserved.
- Outcome classified without turning timeout-only evidence into test-red.
- Report written to worktree and shared `.aragora` locations before exit.
- Old or ambiguous worktrees left untouched.
- Any issue comment links to the durable report and is labeled diagnostic,
  not settlement evidence.

## Incident Note

During conductor cycles 150 and 151 on 2026-07-11, an exact-main first-error
process ended after creating a clean detached worktree but left no owner marker,
log, report, commit, or durable receipt. The next conductor could prove that
the process had stopped, but could not safely distinguish completed work from
abandoned work until a 90-second stability probe. This protocol closes that
coordination gap; it does not adjudicate the underlying main-health incident.
