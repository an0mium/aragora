## Campaign Pipeline Validated — Conservative Baseline

On 2026-03-10, Aragora's campaign pipeline completed its first full end-to-end
conservative validation using a prebuilt two-project YAML manifest (`dogfood-6`).
The run finished with `campaign_complete` after spending `$3.00` of a `$5.00`
budget.

The validated rollout shape is:
- Prebuilt YAML manifest
- `max_parallel_ready_projects: 1`
- `worker_model: codex`
- `review_model: claude`
- `use_managed_session_script: false`
- Sequential dependency gating through manifest `dependencies`

This validation proved:
- Codex workers can produce concrete deliverables and push them
- Claude heterogeneous review can execute and pass on those deliverables
- Dependency ordering is enforced correctly across sequential projects
- Manifest state, stop reasons, and budget accounting remain coherent
- File-scope enforcement passes on legitimate outputs

Fixes required before dogfood-6:
- `#916`: auto-push, `WorkerOutcome` enum
- `#917`: disable nested worktree for campaign workers
- `#918`: `--no-verify` in auto-commit, timeout fix
- `#919`: porcelain path truncation fix

Observed result profile:
- `proj-001`: `deliverable_created`, review `passed`, branch `codex/swarm-a83ea62b-subtask_`
- `proj-002`: `deliverable_created`, review `passed`, branch `codex/swarm-a56eac88-subtask_`

Deferred validation remains for parallel execution, richer reviewer diff
inspection, retry/recovery scenarios, budget/time-limit exhaustion paths,
live receipt-emission proof, and later campaign phases.
