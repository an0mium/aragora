# Automation Merge Contract

This contract applies to both local Codex app automations and Aragora native boss-loop workers. The goal is not to maximize the number of opened PRs. The goal is to maximize PRs that are small, reviewable, validated, and mergeable without salvage work.

## Producer Contract

Every automation run should start from current `origin/main`, work on one bounded issue or one bounded maintenance task, and keep the branch scoped to the files needed for that task. It should prefer a disposable worktree when multiple agents are active.

The run must not publish a PR for analysis-only output, a no-op branch, a branch containing session artifacts, or a branch whose final state is known to fail the declared validation. If it cannot finish, it should leave a handoff with the branch, failing command, blocker, and next action instead of opening a misleading PR.

## Shared Preflight

Before pushing or opening an automation PR, run:

```bash
bash scripts/automation_pr_preflight.sh origin/main HEAD
```

For local Codex app automation branches, `scripts/publish_codex_automation_branches.py --apply` runs this preflight by default before it pushes and opens a PR. Use `--skip-preflight` only for manual recovery when a human has already inspected the diff.

For local Codex app automation handoffs, `scripts/publish_automation_handoffs.py --apply` reads structured memory blocks, deduplicates them against existing GitHub issues and PRs, and creates missing `boss-ready` issues from a normal shell with working `gh` access. Scout automations should leave structured handoffs in memory/inbox when GitHub writes fail instead of using browser fallback for issue or PR creation.

If a handoff explicitly targets an already-open pull request (for example `PR #6288` in the task title or evidence block), the handoff publisher should treat it as PR follow-up work instead of minting a new `boss-ready` issue. PR-targeted follow-ups must not consume boss-ready issue capacity.

For Aragora boss-loop workers, run the same script against the worker branch before merge arbitration:

```bash
bash scripts/automation_pr_preflight.sh origin/main <worker-branch>
```

The preflight fails on empty diffs, whitespace errors, and committed session artifacts such as worker logs, active-session markers, repair journals, or event directories. It also warns when source or config changes have no corresponding test-path changes so the PR body can explicitly name the validation command that covered the change.

## PR Body Contract

An automation PR should include:

- the issue or task source
- a short summary of changed behavior
- the exact validation command and result
- any skipped validation with the reason
- known risks or assumptions
- repair journal or handoff context when this is a retry

Draft PRs are acceptable when the publisher is only preserving useful branch state. Ready PRs should have a passing preflight, scoped diff, and explicit validation evidence.

## Merge Gate

Automation PRs are merge candidates only when:

- the diff is scoped to the task
- CI or targeted validation passes
- there is no duplicate open PR for the same issue or branch
- any cross-host retry consumed the prior repair journal or explicitly explains why it ignored it
- generated artifacts and local coordination files are absent from the branch

If any gate fails, the next useful action is a repair attempt with the failure output in the handoff envelope, not another fresh worker staring at the same repo state.

## Required vs Advisory Checks

The merge decision should use GitHub branch protection as the authoritative hard gate. As of this contract, that means the required status checks configured on `main`, one approving review, and a scoped diff with validation evidence in the PR body.

Advisory workflows are still useful evidence, but they should not create a hidden second merge policy. Treat advisory checks as follows:

- Passing advisory checks increase confidence but are not required for low-risk automation PRs.
- Matrix shards named `test-fast (...)`, including server shards such as `test-fast (server, tests/server, 30)`, are advisory unless they are explicitly configured as required branch-protection checks.
- Cancelled advisory checks caused by a newer push are queue churn, not a blocker. Re-run them only when the cancelled workflow is directly relevant to the changed files.
- Failed advisory checks are blockers only when the failure is in-scope for the PR diff or reveals a mainline regression that would be worsened by the PR.
- Summary-only jobs such as analytics, admission signals, and AI review comments should prefer warnings and PR comments over failing statuses.

Use a fast lane for docs-only, tests-only, and narrow CI reliability PRs: if the required checks pass, the diff is scoped, and one reviewer approves, the PR is mergeable even if unrelated advisory lanes are pending or skipped. Use the full lane for product, security, deploy, data migration, and cross-cutting architecture changes: relevant advisory lanes should be green or explicitly waived in the PR body before admin merge.

## Queue Hygiene

High-churn automation loses throughput when multiple branches compete for the same issue or when stale PRs keep requesting review. Before opening or merging another PR, check for duplicate open PRs, dirty draft branches, and obsolete salvage branches for the same task. Close, draft, or supersede stale PRs with a short comment that names the replacement branch or next repair action.

## Publish Bridge

`scripts/run_codex_automation_publisher.sh` is the non-coding publish bridge for local automation output. It should run from a user shell or LaunchAgent context with stable `gh` credentials. It writes a local queue-status cache to `.aragora/automation-github-status/latest.json`, publishes eligible clean `codex/*` branches into PRs with `scripts/publish_codex_automation_branches.py`, and only then drains structured handoffs into GitHub issues with `scripts/publish_automation_handoffs.py`. The order matters: validated branch-to-PR conversion has priority over creating another issue about a branch that is already locally publishable.

The bridge intentionally does not use browser fallback. Browser profiles can be locked by other tools, and interactive safety prompts make browser-based GitHub publishing unreliable for unattended automations. If `gh` is unavailable, automations should keep the handoff in memory/inbox and let the next bridge pass retry.

The publisher bridge has an explicit GitHub health probe at `scripts/github_cli_health.py`. Sandboxed coding automations should treat a failed probe as a network/auth boundary for direct GitHub writes, not as a reason to stop local work. They should inspect `.aragora/automation-github-status/latest.json`, shared `.aragora/automation-outbox/`, shared `.aragora/automation-receipts/`, and publisher logs before deciding whether to revalidate an existing branch, refresh one handoff, or record an unchanged-state memory note. Do not retry `gh issue create`, `gh pr create`, `gh workflow run`, or merge-watch commands from inside the sandbox.

If every existing open `codex/*` PR is unhealthy, `scripts/publish_codex_automation_branches.py --apply` pauses by default with `publish_paused_reason=open_pr_queue_unhealthy`. A human or GitHub-capable bridge may pass `--allow-unhealthy-queue-publish` for a constrained queue-drain run. That flag does not ignore preflight and does not ignore `--max-open-prs`; it only permits otherwise eligible, preflighted branches to publish when the current queue is red. Use it sparingly, preferably with `--limit 1`, for automation-infrastructure or CI-repair branches that plausibly unblock the queue.

Do not use a raw local `git branch --list 'codex/*'` count as the unpublished-work backlog gate. Local developer machines can retain thousands of stale historical `codex/*` refs that are already merged, patch-equivalent, diverged from the current base, or local-only archaeology. Use `python3 scripts/audit_codex_branch_backlog.py --json` and gate on `summary.publishable_branch_backlog` instead. That metric counts recent unique local work and stale unique remote branches that are not behind the base, but intentionally excludes stale local-only refs and diverged repair candidates so writer automations keep producing useful local code when GitHub is sandboxed.

For steady local operation, install the publisher bridge as a LaunchAgent from a normal user shell:

```bash
bash scripts/install_codex_automation_publisher_launchd.sh
```

Use `bash scripts/status_codex_automation_publisher_launchd.sh` to inspect the job and `bash scripts/uninstall_codex_automation_publisher_launchd.sh` to remove it.

## Prompt Snippet For Codex App Automations

Use this repo's automation merge contract at `docs/briefs/automation-merge-contract.md`. Work on one bounded issue or maintenance task. Make the smallest credible change, run the targeted validation, and before publishing run `bash scripts/automation_pr_preflight.sh origin/main HEAD`. If GitHub health is degraded or publishing fails, switch to handoff-only mode: leave the exact branch, failing command, blocker, and next action in structured memory/inbox evidence for `scripts/run_codex_automation_publisher.sh` instead of opening a misleading PR. Do not use browser fallback for GitHub publishing.

## Prompt Snippet For Boss-Loop Operators

Run the boss loop with shared coordination state when multiple hosts are active:

```bash
export ARAGORA_DEV_COORDINATION_DB=/Users/armand/Development/aragora/.aragora/dev_coordination.sqlite3
export ARAGORA_BOSS_VERIFIED_RUNNER_TARGET=0
python3.11 -u -m aragora.cli.main swarm boss-loop \
  --boss-repo synaptent/aragora \
  --target-branch main \
  --worker-model claude \
  --label boss-ready \
  --max-ticks 20 \
  --interval 60 \
  --autonomy full-auto \
  --max-hours 8 \
  --boss-max-parallel-dispatches 1 \
  --allow-claude-write
```

Keep host parallelism conservative until repair journals, handoff envelopes, and the publisher preflight show that retries are improving PR quality rather than generating duplicate work.
