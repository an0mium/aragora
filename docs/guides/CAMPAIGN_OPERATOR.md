# Campaign Operator Guide

Operate multi-project campaigns using `aragora swarm campaign run`. A campaign is a persistent manifest of projects that the swarm executes one batch at a time. Each invocation dispatches, reconciles, and reviews exactly one iteration, then exits. Repeated invocations resume from where the manifest left off.

## What `campaign run` does

1. If no manifest exists, plans projects from the input source and writes the manifest.
2. If a manifest exists, resumes from it (no replanning).
3. Reconciles any in-flight projects from a previous invocation.
4. Selects ready projects up to `max_parallel_ready_projects`.
5. Dispatches each selected project through the supervised swarm pipeline.
6. Runs a heterogeneous review gate (worker model != review model).
7. Writes results back to the manifest and exits with a stop reason.

Each call does exactly one pass. The caller (you, a cron job, or a Ralph loop) decides whether to invoke again.

## When it plans vs resumes

| Manifest exists? | Source input provided? | Behavior |
|---|---|---|
| No | Yes (exactly one) | Plans projects from input, writes manifest, executes one iteration |
| No | None | Error: `campaign run requires an existing manifest or one of --source-file, --issue-list, --github-query` |
| No | More than one | Error: `exactly one of --source-file, --issue-list, or --github-query` |
| Yes | None | Resumes from manifest, executes one iteration |
| Yes | Any | Error: `cannot supply --source-file, --issue-list, or --github-query when resuming from an existing manifest` |

**The exactly-one-input rule applies only when no manifest exists.** Once a manifest is written, all future invocations ignore input flags and resume from the manifest. Passing input flags alongside an existing manifest is an error.

## CLI arguments

```
aragora swarm campaign run [options]
```

| Flag | Default | Purpose |
|---|---|---|
| `--source-file PATH` | — | Markdown/text file with roadmap items (one per line) |
| `--issue-list NUMS` | — | Comma-separated GitHub issue numbers |
| `--github-query QUERY` | — | GitHub issue search query string |
| `--manifest PATH` | `.aragora/campaign_manifest.yaml` | Manifest file path |
| `--output PATH` | same as `--manifest` | Where to write a new manifest |
| `--planner-model MODEL` | `claude` | Model for planning decomposition |
| `--worker-model MODEL` | `codex` | Model for project execution |
| `--review-model MODEL` | `claude` | Model for heterogeneous review |
| `--max-parallel-ready-projects N` | `1` | Max projects dispatched per iteration |
| `--budget-limit N` | `50.0` | Total campaign budget in USD |
| `--json` | off | Emit structured JSON output |

## Stop reasons

Every `campaign run` invocation exits with one of these stop reasons in its output:

| Stop reason | Meaning | What to do |
|---|---|---|
| `still_running` | Projects were dispatched or are in-flight. More work remains. | Invoke `campaign run` again. |
| `campaign_complete` | All projects are completed or skipped. Nothing left. | Done. Inspect the manifest for deliverables. |
| `campaign_blocked` | All remaining projects are blocked, failed, or skipped. No in-flight work. | Inspect the manifest. Fix the blocking issue manually, then resume. |
| `budget_exhausted` | Cumulative cost reached `budget_limit_usd`. | Increase the budget in the manifest or accept the partial result. |
| `time_limit_exceeded` | Elapsed time since the current invocation's start exceeded `time_limit_hours`. In practice this only fires if a single `execute_once` dispatch takes longer than the limit. | Increase the limit in the manifest or accept the partial result. |

**`still_running` is normal.** Most iterations return this. It means the campaign is making progress and you should call `campaign run` again.

## Conservative rollout guidance

For your first campaign, use these conservative defaults:

1. **Start from a prebuilt manifest.** Write the manifest YAML by hand (or use `campaign plan` once) rather than having `campaign run` plan from a source file. This lets you inspect every project spec before any work starts.

2. **Set `max_parallel_ready_projects: 1`.** The CLI default is already 1. This ensures one project completes and is reviewed before the next starts.

3. **Heterogeneous review stays blocking.** The review model must differ from the worker model (enforced automatically). A project cannot reach `completed` status without passing review. Do not bypass this.

4. **No replanning on resume.** Once a manifest exists, `campaign run` never modifies the project list. It only advances project statuses. If you need to change the plan, edit the manifest YAML directly or delete it and replan.

5. **Set a budget limit.** The default is $50. For a first run, consider lowering it: edit `budget_limit_usd` in the manifest.

6. **Set a time limit.** The default is 8 hours. For a first run, consider lowering it: edit `time_limit_hours` in the manifest.

## Project lifecycle

Each project in the manifest follows this status progression:

```
pending → active → delivered → completed
                 ↘ needs_revision → active (retry)
                 ↘ failed (if max retries exceeded → skipped)
                 ↘ blocked (needs human intervention)
```

- **pending**: Not yet dispatched. Waiting for dependencies to complete.
- **active**: Dispatched to a worker. In-flight.
- **delivered**: Worker produced a deliverable (PR, branch). Awaiting review.
- **needs_revision**: Review requested changes. Will retry up to `max_retries_per_project` (default: 2).
- **completed**: Passed review. Terminal.
- **failed**: Exhausted retries without passing review. Terminal.
- **blocked**: Worker reported a human-required blocker. Terminal until manual intervention.
- **skipped**: Failed project that exceeded retry limit. Terminal.

## Operator examples

### Example 1: Source-file bootstrap

Create a roadmap file with one item per line:

```
# roadmap.md
- Add retry ledger in aragora/swarm/campaign.py
- Improve status output in aragora/cli/commands/swarm.py
- Write campaign operator docs in docs/guides/
```

Plan and execute the first iteration:

```bash
aragora swarm campaign run \
  --source-file roadmap.md \
  --worker-model codex \
  --review-model claude \
  --budget-limit 20 \
  --json
```

Output (JSON):
```json
{
  "mode": "campaign-run",
  "invocation_mode": "planned_then_executed",
  "manifest_path": ".aragora/campaign_manifest.yaml",
  "campaign_id": "campaign-a1b2c3d4",
  "stop_reason": "still_running",
  "dispatched_projects": [
    {"project_id": "proj-001", "status": "active", "outcome": "deliverable_created"}
  ]
}
```

The manifest is now written. All future invocations resume from it.

### Example 2: Resume from manifest

After the first run, invoke again with no source flags:

```bash
aragora swarm campaign run --json
```

This reads the existing manifest at `.aragora/campaign_manifest.yaml`, reconciles any in-flight projects from the previous invocation, and dispatches the next batch.

Repeat until `stop_reason` is `campaign_complete`, `campaign_blocked`, or a budget/time limit:

```bash
# Check status without dispatching
aragora swarm campaign status --json

# Resume execution
aragora swarm campaign run --json
```

### Example 3: Ralph loop usage

The campaign pipeline is designed for Ralph loops. Each `campaign run` call is idempotent: it reads the manifest, does one batch of work, writes results back, and exits. The Ralph loop wrapper calls it repeatedly.

```bash
# Ralph loop: same command repeated until campaign completes
while :; do
  output=$(aragora swarm campaign run --json 2>&1)
  stop_reason=$(echo "$output" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stop_reason',''))" 2>/dev/null)

  case "$stop_reason" in
    campaign_complete)
      echo "Campaign finished."
      break
      ;;
    campaign_blocked|budget_exhausted|time_limit_exceeded)
      echo "Campaign stopped: $stop_reason"
      break
      ;;
    still_running|"")
      # Normal: more work to do. Loop again.
      ;;
  esac
done
```

**Why `campaign run` and not `campaign plan` + `campaign execute` separately:**
`campaign run` handles the plan-or-resume decision automatically. If you call `plan` then `execute` separately, you risk replanning over an existing manifest or executing without a plan. `campaign run` is the single entry point that does the right thing based on whether a manifest exists.

## Manifest format

The manifest is a YAML file at `.aragora/campaign_manifest.yaml` (configurable via `--manifest`). It is locked with `fcntl.flock` during reads and writes to prevent corruption from concurrent access.

Key fields:

```yaml
campaign_id: campaign-a1b2c3d4
created_at: "2026-03-10T12:00:00+00:00"
source_kind: source_file        # source_file | issue_list | github_query
source_ref: roadmap.md
planner_model: claude
worker_model: codex
review_model: claude
max_parallel_ready_projects: 1
max_retries_per_project: 2
budget_limit_usd: 50.0
time_limit_hours: 8.0
manifest_version: 1
projects:
  - project_id: proj-001
    title: "Add retry ledger"
    status: pending
    # ... spec, file_scope_hints, acceptance_criteria, etc.
execution_state:
  total_cost_usd: 0.0
  last_run_at: null
```

You can edit `budget_limit_usd`, `time_limit_hours`, `max_parallel_ready_projects`, and `max_retries_per_project` between invocations. Do not edit `campaign_id`, `projects[].project_id`, or `projects[].status` unless you understand the state machine.

## Prebuilt manifest workflow

For maximum control, write the manifest by hand before the first `campaign run`:

1. Use `campaign plan --source-file roadmap.md --json` to generate a draft manifest.
2. Inspect every project's `spec`, `file_scope_hints`, and `acceptance_criteria`.
3. Edit as needed. Remove projects you don't want. Adjust constraints.
4. Set conservative limits: `max_parallel_ready_projects: 1`, `budget_limit_usd: 20`.
5. Save to `.aragora/campaign_manifest.yaml`.
6. Run `aragora swarm campaign run --json` to start execution from the prebuilt manifest.

This is the recommended workflow for the first campaign on any codebase.

## Troubleshooting

**`campaign_blocked` but I expected `still_running`:**
Check the manifest. All non-terminal projects may be in `blocked` or `failed` state. Look at `last_run_outcome` and `review.findings` on each project to understand why.

**A project is stuck in `active` across multiple invocations:**
The reconciler checks in-flight runs on each invocation. If `_refresh_run_dict` cannot find the run (crashed worker, cleaned worktree), the project stays `active` indefinitely. Manually set its status to `failed` in the manifest to unblock downstream projects.

**Budget or time limit hit too early:**
Edit `budget_limit_usd` or `time_limit_hours` in the manifest and run again. These limits are checked on each invocation, not enforced by a background timer.

**Review keeps requesting changes:**
Each retry appends review findings as constraints to the next worker dispatch. After `max_retries_per_project` failures (default: 2), the project is marked `skipped`. Increase `max_retries_per_project` in the manifest if needed, or fix the issue manually.
