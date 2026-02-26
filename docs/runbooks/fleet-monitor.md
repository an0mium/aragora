# Fleet Monitor Runbook

## Operator Commands

- Fleet status (JSON):
  - `aragora worktree fleet-status --tail 500 --json`
- Fleet status + report file:
  - `aragora worktree fleet-status --tail 500 --report --json`
- Generate report via script:
  - `python scripts/fleet_status_report.py --tail 500 --json`

## API Endpoints

- `GET /api/v1/coordination/fleet/status?tail=500`
- `GET /api/v1/coordination/fleet/logs?tail=500`
- `GET /api/v1/coordination/fleet/claims`
- `POST /api/v1/coordination/fleet/claims`
  - body: `{"owner":"track:qa","paths":["aragora/server/handlers/coordination.py"],"override":false}`
- `GET /api/v1/coordination/fleet/merge-queue`
- `POST /api/v1/coordination/fleet/merge-queue`
  - enqueue body: `{"action":"enqueue","owner":"track:qa","branch":"codex/qa-123"}`
  - advance body: `{"action":"advance"}`

## Expected Guardrails

- Claims API blocks overlapping path claims unless `override=true`.
- Merge queue advance is blocked when required checks are red.
- Drift is flagged when local branch SHA differs from `origin/<branch>` SHA.
