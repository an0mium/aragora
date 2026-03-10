# Dogfood #6 Evidence Report

**Date:** March 10, 2026
**Purpose:** Record the first confirmed end-to-end campaign execution where the
swarm pipeline produced a real deliverable, passed review, and persisted
campaign state after the worker no-deliverable failure chain was fixed.

## Summary

Dogfood `#6` is the first validated narrow-lane campaign run showing:
- worker dispatch succeeded
- the worker produced a real file change
- auto-commit succeeded
- deliverable extraction succeeded
- review executed and passed
- campaign state was updated to mark the project completed

This does **not** prove broad autonomous self-repair readiness.
It **does** prove that the core campaign loop can complete at least one bounded
documentation task end-to-end on mainline code after the swarm fixes in PRs
`#917`, `#918`, and `#919`.

## Environment

- Worktree: `.worktrees/dogfood-6`
- Source branch baseline: `main`
- Campaign manifest:
  [`/.worktrees/dogfood-6/.aragora/campaign_manifest.yaml`](/Users/armand/Development/aragora/.worktrees/dogfood-6/.aragora/campaign_manifest.yaml)
- Review model: `claude`
- Worker model: `codex`

## Preceding Fix Chain

Dogfood #6 depended on three swarm hardening fixes:

| Dogfood Run | Issue | Fix |
|---|---|---|
| `#1-#4` | pre-commit hooks blocked auto-commit; collect timeout issues | `#918` |
| `#5` | porcelain path parsing truncated paths and caused false scope violations | `#919` |
| `#6` | first successful end-to-end deliverable cycle | validated |

## Verified Result

From the persisted manifest state:

- `campaign_id`: `campaign-dogfood-6-2026-03-10`
- `proj-001` status: `completed`
- `proj-001` outcome: `deliverable_created`
- `proj-001` review status: `passed`
- `proj-001` branch: `codex/swarm-a83ea62b-subtask_`
- `proj-001` commit:
  `8caf91930043cf8658c5b3fd47b8f5b80e515f4e`
- changed file:
  [`docs/guides/SWARM_DOGFOOD_OPERATOR.md`](/Users/armand/Development/aragora/docs/guides/SWARM_DOGFOOD_OPERATOR.md)
- diff size: `20` lines
- campaign stop reason after first pass: `still_running`
- downstream project `proj-002`: remained pending, correctly blocked by dependency

## What Was Proven

### 1. Deliverable Detection Worked

The worker result was classified as `deliverable_created` rather than
`clean_exit_no_deliverable`, which had been the blocking failure mode in prior
dogfood runs.

### 2. Auto-Commit Path Worked

The completed project recorded a non-empty `commit_shas` list, proving that the
worker’s change was committed and surfaced as a deliverable artifact.

### 3. Scope Enforcement Did Not False-Fail

The project modified exactly one file in the declared scope:
- [`docs/guides/SWARM_DOGFOOD_OPERATOR.md`](/Users/armand/Development/aragora/docs/guides/SWARM_DOGFOOD_OPERATOR.md)

This is important because the immediate bug fixed in `#919` was false scope
violation caused by truncated porcelain paths.

### 4. Heterogeneous Review Ran

The project review was executed and marked `passed`. The persisted findings show
that the review step inspected metadata and accepted the deliverable.

### 5. Campaign Dependency Handling Worked

`proj-002` remained pending after `proj-001` completed, and the campaign stop
reason was `still_running` rather than a spurious terminal failure. That is the
expected state for a sequential manifest after a single execution iteration.

## Limits Of This Evidence

Dogfood #6 proves a **narrow bounded lane**, not general autonomy.

It does **not** yet prove:
- repeated reliability over many runs
- correctness of broader code-change tasks
- robustness of merge-gated code modifications
- composed verification against runtime paths
- broad self-repair across canonical subsystems

The review for `proj-001` also explicitly noted that approval was based on
metadata consistency rather than full diff-content verification. That is
acceptable for this proof point, but not sufficient for broader autonomy.

## Phase Interpretation

This run is evidence that:
- `Phase 0A` is credible and close to proven for a narrow governance lane
- `Phase 0B` has begun, but is not yet proven

This run should count as:
- `1` successful bounded campaign task

It should **not** be treated as:
- permission for broad autonomous repo repair
- permission for autonomous merges outside a constrained lane

## Next Required Proof

Before advancing the bootstrap gates, require:

1. `10` consecutive bounded campaign tasks without manual rescue
2. no silent task loss
3. no false `clean_exit_no_deliverable`
4. persisted receipts, verification artifacts, and review artifacts for each run
5. at least some tasks that modify canonical code paths and pass real verification

## Recommended Follow-Up

- Link this report from the bootstrap gates document
- turn the dogfood execution path into a required orchestrator truth-suite
- run a `Phase 0A` campaign against governance tasks only
- continue blocking expansion-surface autonomy until repeated proof is established
