# Overnight Founder Process

## Goal

Keep Aragora making bounded, useful product progress for a single unattended
10-hour window.

This process is optimized for:

- product roadmap refinement
- founder vision/spec clarification
- documentation and planning outputs
- issue-backed bounded implementation slices

It is intentionally **not** optimized for unconstrained autonomous product
philosophy generation.

## Recommended Driver

Use the GitHub issue-backed **Boss loop** as the top-level overnight driver.

Do **not** use raw `scripts/nomic_loop.py` as the primary unattended process
for founder/roadmap work. Nomic is valuable as a bounded implementation and
verification engine, but it is the wrong top-level tool for open-ended product
direction because it can widen scope without a strong external contract.

Why the Boss loop is the right overnight driver:

- it requires explicit issue scope
- it stops truthfully
- it checks runner freshness
- it emits bounded receipts and next actions
- it can keep pulling the next eligible issue without improvising a new mission

## Core Principle

The overnight run should operate on a **prepared founder tranche**:

- 8-12 small GitHub issues
- all labeled `boss-ready`
- all labeled `overnight-founder`
- all with explicit acceptance criteria / validation contract
- all constrained mostly to `docs/**`, `prompts/**`, `ROADMAP.md`, or other
  planning surfaces

The loop should not spend the night inventing its own backlog.

## Before Sleep

### 1. Start a managed session

```bash
cd ~/aragora
./scripts/codex_session.sh --agent codex --orchestrator boss-loop
```

### 2. Make sure the prerequisites are true

```bash
gh auth status
python3 -m aragora.cli.main swarm boss-loop --help >/dev/null
python3 scripts/codex_worktree_autopilot.py ensure --agent codex --base main --reconcile --print-path
```

### 3. Prepare the overnight issue set

Recommended issue themes:

- canonical founder narrative
- ICP + wedge definition
- 30/60/90 plan refinement
- design partner program
- pricing hypotheses
- inbox trust wedge default journey
- PMF scorecard and metrics
- roadmap dependency map
- receipt-first customer proof pack
- operator runbook cleanup

Each issue should be small enough that a single worker can finish it in one
bounded pass.

### 4. Use this issue body pattern

```md
## Goal

Produce or refine <specific founder artifact>.

## Scope

- Allowed files:
  - docs/strategy/**
  - docs/plans/**
- Do not widen into unrelated repo cleanup.

## Acceptance Criteria

- Produces one concrete artifact, not vague notes.
- States assumptions explicitly.
- Names open questions separately from conclusions.
- Includes next actions that can become follow-up issues.

## Validation

- The changed artifact exists in the allowed write scope.
- The artifact contains: summary, assumptions, decisions, open questions, next actions.

## Stop Conditions

- If the issue requires live customer data or a broader architectural decision, stop and record the blocker truthfully.
```

Do not rely on `--allow-missing-validation-contract`.

### 5. Dry-run the selection logic

```bash
ARAGORA_USER_ID=armand python3 -m aragora.cli.main swarm boss-loop \
  --boss-repo synaptent/aragora \
  --label boss-ready \
  --label overnight-founder \
  --no-dispatch \
  --max-ticks 1 \
  --json
```

If this does not select the right issue or returns a blocker, fix the issue
labels/body before going to sleep.

## Overnight Run

### Primary command

```bash
mkdir -p .aragora/overnight

nohup env ARAGORA_USER_ID=armand \
  python3 -m aragora.cli.main swarm boss-loop \
    --boss-repo synaptent/aragora \
    --label boss-ready \
    --label overnight-founder \
    --max-ticks 12 \
    --interval 30 \
    --max-consecutive-failures 2 \
  > .aragora/overnight/founder-boss-loop.log 2>&1 \
  < /dev/null &

echo $! > .aragora/overnight/founder-boss-loop.pid
```

Notes:

- `--max-ticks 12` means "attempt up to 12 issue iterations", not "run for 12
  clock ticks".
- `--interval 30` keeps the loop moving without tight polling.
- `--max-consecutive-failures 2` prevents wasting the whole night on a broken
  lane.

### Optional reconcile sidecar

Use only if you want a lightweight helper to keep managed worktrees tidy while
the boss loop runs.

```bash
nohup bash -lc '
  while true; do
    python3 scripts/codex_worktree_autopilot.py reconcile --all --base main
    sleep 1800
  done
' > .aragora/overnight/worktree-reconcile.log 2>&1 < /dev/null &
echo $! > .aragora/overnight/worktree-reconcile.pid
```

## What Not To Do Overnight

Do not run these as the top-level unattended job:

- `python3 scripts/nomic_loop.py run --auto`
- broad codebase cleanup missions
- multi-surface refactors without issue-backed acceptance criteria
- anything that requires manual merge judgment on main

If you want Nomic involved, use it only inside a bounded issue whose mission is
already defined by the Boss loop.

## Morning Harvest

### 1. Inspect the overnight log

```bash
tail -200 .aragora/overnight/founder-boss-loop.log
```

### 2. Check what shipped or blocked

```bash
gh issue list --repo synaptent/aragora --label overnight-founder --state open
gh pr list --repo synaptent/aragora --search "overnight-founder"
git status --short --branch
```

### 3. Pull out the truthful stop reasons

Typical good outcomes:

- issues completed
- a bounded needs-human blocker with exact next action
- draft PRs or branch-backed deliverables

Bad outcomes:

- repeated failures on the same issue
- widening scope beyond founder artifacts
- silent drift into unrelated implementation work

## If You Want A More Ambitious Version

The next step after this runbook is a founder-specific tranche or queue:

1. write a founder prompt-pack YAML
2. compile it with `aragora swarm tranche plan`
3. inspect and design-review it
4. drive it with tranche watch/queue execution

That is the right evolution if overnight founder runs become a recurring habit.
For tonight, the issue-backed Boss loop is the safer path.
