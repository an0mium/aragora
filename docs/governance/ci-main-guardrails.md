# CI And Main-Branch Guardrails

**Status:** Governance reference
**Last updated:** April 25, 2026
**Purpose:** Define the autonomy boundary for CI, runner, release, and
main-branch work so foundation-hardening can move quickly without breaking the
tooling that enables autonomous development.

## Core Rule

Break unreleased branch behavior freely when the change is reversible and the
PR makes the risk obvious. Do not casually break the tools, gates, or release
paths that make repair and review possible.

In practice:

- Product behavior may regress on a PR branch while an agent is iterating.
- `main` should not be knowingly merged red.
- Public API, SDK, website, release, and operator-visible behavior need normal
  review, even before product-market fit.
- CI, runner fleet, required checks, pre-commit hooks, worktree autopilot, and
  release automation are tool surfaces. Treat them as higher-risk than ordinary
  product code.

## Tool Surfaces

Changes to these surfaces require explicit PR-body risk notes and independent
review before merge:

- GitHub Actions workflows that affect required checks, path filters, runner
  labels, workflow concurrency, auto-revert, release, deploy, security, or
  secret/OIDC configuration.
- Self-hosted runner labels, fleet membership, AMIs, Docker/toolchain
  provisioning, or scheduling assumptions.
- Pre-commit hooks, local lint/type/test wrappers, and scripts that developers
  use to prepare PRs.
- Worktree/session automation including `scripts/codex_session.sh`,
  `scripts/codex_worktree_autopilot.py`, and safe cleanup tooling.
- Branch protection or merge policy.
- Release tagging, publishing, and version-bump workflows.

Small docs-only changes that describe these surfaces are low risk. Behavioral
changes to these surfaces are not.

## Workflow And Runner PR Checklist

Every workflow or runner PR should answer these before merge:

1. What runner labels can the job land on?
2. Are all required host tools available on every eligible runner?
3. If a job uses Docker, is Docker verified for the exact runner label set?
4. Does the workflow rely on OS-specific package managers such as `apt-get`,
   `dnf`, `brew`, or `choco`?
5. Does the change affect required-check names or branch-protection semantics?
6. Does a skipped, cancelled, or failed helper job fail open?
7. Was `actionlint` run, and does the repo config know any new custom labels?
8. For dispatch-only or scheduled workflows, was a manual dispatch smoke run
   performed when practical?

If any answer is unknown, keep the PR open until the assumption is verified or
documented as an explicit follow-up risk.

## Main-Red Incident Mode

When `main` has a terminal failure on a required check, stop roadmap work and
switch to incident mode.

Incident mode priority:

1. Confirm whether the failure is real, stale, cancelled, or unrelated to the
   latest merge.
2. Re-run or inspect logs only when it reduces uncertainty.
3. If the latest merge caused the failure and the fix is not obvious, revert
   the merge rather than stacking speculative fixes.
4. Resume roadmap work only after required checks are green or a human has
   explicitly waived the failure.

If `main` remains red for more than two hours, do not open new roadmap PRs.
Use the next work cycle for bisect, revert, or a focused fix PR.

## Autonomous Scope Tiers

Allowed without additional approval:

- Additive tests, docs, and low-risk refactors that do not touch tool surfaces.
- Patch-level dependency-floor alignment with clear local verification.
- Route or handler consolidation PRs that preserve behavior and include
  dispatch/snapshot tests.

Requires explicit review before merge:

- Workflow, runner, release, or required-check changes.
- Major dependency upgrades or dependency changes with broad transitive churn.
- Public API, CLI, REST, SDK, schema, or migration removals.
- Any PR whose safe rollback requires more than a normal Git revert.

Requires explicit human approval before execution:

- Force-pushes, branch deletion with unique commits, destructive cleanup, secret
  edits, `.env` edits, protected-file edits, and tag pushes.

## Reversibility Budget

If two PRs from the same roadmap wave need to be reverted for distinct reasons,
stop that wave. The wave is carrying too much uncertainty; write a short
postmortem, split the scope, and restart with narrower PRs.
