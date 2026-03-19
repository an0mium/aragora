# Boss-Loop Live Proof Runbook

Historical note: this runbook captures the bounded execution contract used for the first live Boss-loop proof on 2026-03-19. Current `main` has since absorbed `#1065` and `#1066`, and issue `#1064` is closed. Use this as the template for future bounded live proofs, not as a literal instruction to rerun `#1064`.

## Preconditions

- `#1061` is merged so `--max-ticks 1` returns after dispatch instead of waiting for full worker completion.
- The target issue is labeled `boss-loop-test`, still open, and has an explicit validation contract.
- The runner is registered and fresh.
- The repo is synced to current `origin/main`.

## Launch Contract

Run a single bounded live tick:

```bash
ARAGORA_USER_ID=an0mium python -m aragora.cli.main swarm boss-loop \
  --boss-label-filter boss-loop-test \
  --boss-issue-number <ISSUE_NUMBER> \
  --max-ticks 1 \
  --target-branch main \
  --json
```

Expected result:

- `stop_reason: "max_iterations"`
- exactly one iteration completed
- `worker_status: "running"`
- `worker_outcome: "dispatched"`
- `next_actions[0]` references the active supervisor run id

The command returning promptly is the point. The worker continues detached and must be monitored separately.

## Monitor Contract

After launch, inspect the active run:

```bash
python -m aragora.cli.main swarm status --json
```

Check for:

- the active `run_id`
- work order branch / worktree metadata
- terminal outcome (`completed`, `needs_human`, `failed`)
- any `receipt_id`, `pr_url`, or commit metadata

If the run becomes terminal, preserve the JSON output as part of the evidence bundle.

## Stop Conditions

Stop and mark `needs_human` when any of the following occurs:

- the target issue is closed, stale, or missing explicit validation commands
- the worker edits outside the declared file scope
- the worker reaches `needs_human`, `failed`, or `blocked`
- the launch JSON is non-terminal in a way that contradicts the one-tick contract

Do not widen scope during a live proof. Do not remove the explicit issue number. Do not increase `--max-ticks` beyond `1` for the first proof pattern.

## Evidence Bundle

For acceptance evidence on `#871`, `#990`, and `#1036`, preserve:

- the boss-loop dispatch JSON
- the final supervisor status JSON
- the resulting PR URL or branch and commit metadata
- the bounded validation commands run for the deliverable issue
- the exact `origin/main` SHA used at launch time

## Current Historical Outcome

The first bounded tranche completed through these milestones:

- `#1061` merged the one-tick return-shape fix
- `#1065` merged the stdin / PTY hardening
- `#1066` merged the concrete `#1064` deliverable

That sequence is the reference pattern this runbook is meant to preserve.
