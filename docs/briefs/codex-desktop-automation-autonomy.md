# Codex Desktop Automation Autonomy Contract

This contract keeps the Codex Desktop automation fleet productive for long unattended windows while preserving merge quality.

## Objective

The fleet should continuously turn local evidence into small, reviewable improvements that advance Aragora's roadmap, thesis, and dogfood loop. A successful run ends with one of these outcomes:

- a validated branch and outbox handoff ready for the publisher bridge,
- a repaired or refreshed existing handoff,
- a safely classified cleanup/salvage decision with durable evidence,
- a concise blocker report only when mutation would be unsafe.

## Shared Startup

Every automation should start from live repo evidence, not prior narration:

1. Read its own memory at `/Users/armand/.codex/automations/<automation-id>/memory.md` if present.
2. Inspect `/Users/armand/Development/aragora/.aragora/automation-outbox` and `.aragora/automation-receipts`.
3. Run the relevant local status command before choosing a lane:
   - `python3 scripts/check_codex_desktop_automations.py --json`
   - `python3 scripts/audit_codex_branch_backlog.py --max-branches 200 --json --outbox-dir /Users/armand/Development/aragora/.aragora/automation-outbox --receipt-dir /Users/armand/Development/aragora/.aragora/automation-receipts`
   - `python3 scripts/agent_bridge.py operator-snapshot --json`
4. Treat sandboxed GitHub failure as expected. Use local git, shared outbox files, and receipts as the primary automation substrate.

## Writer Lanes

The four writer lanes run hourly and staggered:

- `engineering-autopilot` at `:05`: primary roadmap-aligned implementation.
- `engineering-autopilot-2` at `:20`: repair existing failing PRs, tests, or local regressions.
- `engineering-autopilot-3` at `:35`: salvage valuable branches/worktrees and retire no-op backlog.
- `engineering-autopilot-3-2` at `:50`: quality improvement, harness friction, dogfood feedback, and tests.

Writer rules:

- Produce at most one branch, one commit stack, or one refreshed handoff per run.
- Do not pause on raw `codex/*` branch count. Use the backlog classifier categories.
- If the publisher/open PR queue is unhealthy, repair or refresh existing work instead of opening a new lane.
- Before handoff, run `bash scripts/automation_pr_preflight.sh origin/main HEAD` when the branch is intended for PR publication.
- If GitHub is unavailable, write or refresh exactly one idempotent JSON handoff in `.aragora/automation-outbox`.

## Dogfood Loop

Automations should use Aragora's own coordination surfaces when they materially improve quality:

- Use `scripts/agent_bridge.py operator-snapshot --json` for persistent session and lane context.
- Use `scripts/agent_bridge.py send ... --lane ...` only for a bounded cross-check with a live target session.
- Use `scripts/agent_bridge_broker.py list-runs` or `show-run` when the backend bridge is already active.
- Use `scripts/swarm_session_mux.py` only for explicitly bounded tmux sessions; do not launch broad worker swarms by default.

Dogfood bugs are first-class work. If a harness command is confusing, falsely blocks productive work, or emits misleading health, fix that before adding new product breadth.

## Backpressure

Backpressure remains useful, but it should redirect work rather than cause no-ops:

- High open PR count means repair, rebase, review, or refresh existing work.
- High outbox count means dedupe, reconcile receipts, or strengthen handoff quality.
- Dirty root checkout means use disposable worktrees and leave root untouched.
- Uncertain cleanup means preserve and record the blocker.

The fleet should be able to run for 12 hours without human intervention by making conservative progress every cycle and accumulating exact evidence for anything it cannot safely mutate.
