# Fleet Coordination Runbook

## Purpose

Coordinate many concurrent Codex/Claude worktree sessions without collisions:

- Session visibility (`fleet-status`)
- File ownership claims (`fleet-claim`, `fleet-release`)
- Merge queue ordering (`fleet-queue-add`, `fleet-queue-list`)

This data is also available over control-plane API for dashboard automation.

## CLI Commands

```bash
# 1) View all sessions + tails + inferred orchestration pattern
python -m aragora.cli.main worktree fleet-status --tail 200

# 2) Claim files for one session (exclusive by default)
python -m aragora.cli.main worktree fleet-claim \
  --session-id <session_id> \
  --paths aragora/server/handlers/control_plane/coordination.py tests/handlers/control_plane/test_coordination.py

# 3) Release claims
python -m aragora.cli.main worktree fleet-release --session-id <session_id>

# 4) Queue merge
python -m aragora.cli.main worktree fleet-queue-add \
  --session-id <session_id> \
  --branch codex/my-branch \
  --priority 70 \
  --title "control-plane fleet claim API"

# 5) Inspect merge queue
python -m aragora.cli.main worktree fleet-queue-list
```

## API Endpoints

- `GET /api/v1/coordination/fleet/status`
- `GET /api/v1/coordination/fleet/logs?session_id=<id>`
- `GET /api/v1/coordination/fleet/claims`
- `POST /api/v1/coordination/fleet/claims`
- `POST /api/v1/coordination/fleet/claims/release`
- `GET /api/v1/coordination/fleet/merge-queue`
- `POST /api/v1/coordination/fleet/merge-queue`

## Orchestrator Extension Pattern

`fleet-status` now emits `orchestration_pattern` inferred from session metadata/command. Current labels:

- `gastown`
- `langchain`
- `crewai`
- `langgraph`
- `autogen`
- `openclaw`
- `nomic`
- `generic`

### How to extend to a new orchestrator

1. Update `aragora/worktree/fleet.py` in `infer_orchestration_pattern()` with new markers.
2. If launching via `scripts/codex_session.sh`, pass `--orchestrator <label>` or add marker detection.
3. Add/adjust tests in `tests/worktree/test_fleet.py`.
4. If needed, add routing or policy logic in control-plane based on `orchestration_pattern`.

## Recommended Operating Policy

1. All sessions claim files before edits.
2. Claims are exclusive for code paths, shared only for docs/analysis.
3. Every branch enters merge queue before PR merge.
4. Merge queue item priority:
   - `80-100`: hotfix / release blocker
   - `60-79`: CI or security unblock
   - `40-59`: normal feature work
   - `0-39`: backlog/refactor

