# Proof-First tmux Operator Runbook

## Purpose

Use tmux as the coordination layer for parallel proof-first work without turning tmux into the unattended execution substrate.

The unattended substrate remains:

- `python3 scripts/run_proof_first_shift.py`

tmux is for:

- conductor-led parallel implementation and review lanes
- bounded benchmark/docs/monitor tracks
- fast harvesting of lane output without restocking generic queue work

## When To Use This Runbook

Use this runbook when all of the following are true:

- the live `boss-ready` queue is empty or intentionally constrained by proof-first policy
- the next project move is driven by benchmark truth, docs reconciliation, or operator-surface hardening
- you need multiple bounded lanes with disjoint ownership

Do not use this runbook to:

- repopulate generic cleanup issues
- run multiple independent conductors
- replace `run_proof_first_shift.py` for unattended operation

## Core Rules

1. One conductor only.
2. One proof-first objective at a time.
3. One managed worktree per write lane.
4. Disjoint file ownership across lanes.
5. Keep the root checkout for live services and operator verification, not feature editing.
6. Keep the live queue empty unless fresh proof surfaces expose one bounded regression.

## Standard Lane Set

Use at most four lanes for the current proof-first tranche:

- `codex-conductor`
  Orchestrates prompts, harvests output, sequences merges, and keeps queue discipline.
- `benchmark-proof`
  Owns benchmark/publication reconciliation only.
- `docs-proof`
  Owns canonical proof-first docs only.
- `monitor-proof`
  Read-only lane for PR checks, workflow runs, and queue drift.

Recommended ownership:

- `benchmark-proof`
  `scripts/reconcile_b0_pr_truth.py`
  `scripts/build_benchmark_truth_artifact.py`
  `scripts/render_benchmark_truth_status.py`
  focused tests under `tests/scripts/`
- `docs-proof`
  `docs/status/NEXT_STEPS_CANONICAL.md`
  `docs/status/ACTIVE_EXECUTION_ISSUES.md`
  `scripts/reconcile_status_docs.py` and focused tests only if needed
- `monitor-proof`
  read-only; no code changes

## Setup

Start from a managed worktree, not the dirty root checkout.

```bash
python3 scripts/codex_worktree_autopilot.py ensure --agent codex --base main --reconcile --print-path
```

Create prompt files under `~/.aragora/tmux-prompts/` so prompts are reusable and auditable.

## Launch

Launch named sessions with the repo-native wrapper:

```bash
scripts/tmux_session_launcher.sh --name benchmark-proof --agent claude --prompt-file ~/.aragora/tmux-prompts/proof-first/benchmark.md
scripts/tmux_session_launcher.sh --name docs-proof --agent claude --prompt-file ~/.aragora/tmux-prompts/proof-first/docs.md
scripts/tmux_session_launcher.sh --name monitor-proof --agent claude --prompt-file ~/.aragora/tmux-prompts/proof-first/monitor.md
```

Useful commands:

```bash
scripts/tmux_session_launcher.sh --list
scripts/tmux_send_prompt.sh --name benchmark-proof --prompt-file ~/.aragora/tmux-prompts/proof-first/benchmark-followup.md
scripts/tmux_harvest.sh --name benchmark-proof --lines 120
tmux attach -t aragora
```

For richer session discovery, use:

```bash
python3 scripts/swarm_session_mux.py list
python3 scripts/swarm_session_mux.py status --name benchmark-proof
python3 scripts/swarm_session_mux.py tail --name benchmark-proof --lines 120
```

## Conductor Loop

Run this loop until the bounded objective is complete:

1. Establish truth first.
   Check benchmark surface, open PRs, and live queue state before assigning work.
2. Launch only the lanes needed for the current objective.
3. Assign explicit file ownership in each prompt.
4. Harvest every 10-15 minutes or when a lane reports completion.
5. Merge only green, bounded PRs.
6. Re-run benchmark publication only after the benchmark lane lands.
7. Keep the queue empty unless the refreshed proof surfaces expose one new bounded issue.

## Relationship To Proof-First Shift

`run_proof_first_shift.py` remains the only supported unattended operator path.

Use tmux around that path for:

- pre-merge implementation tracks
- read-only monitoring during a soak
- post-run review and evidence harvesting

Do not use tmux to approximate the unattended shift by hand with ad hoc shell babysitting.

## Stop Conditions

Stop and regroup if any of the following occurs:

- two write lanes start touching the same file set
- benchmark truth is stale and no lane is actively fixing or refreshing it
- the queue is repopulated with non-canonical work
- a lane starts broadening scope beyond the bounded proof-first objective

## Success Criteria For This Tranche

For the current roadmap tranche, tmux-assisted work is successful when:

- benchmark truth publication is complete and fresh
- canonical docs match current merged proof
- the queue remains canonical and usually empty
- the next unattended run can be evaluated against explicit `BC-12` green-shift criteria
