# Release Worktree Hygiene

Use this flow for release operations and CI reconciliation to avoid branch drift and mixed-session state.
For fleet-wide side-worktree audit, harvest, and cleanup classification, use
[`RUNBOOK_WORKTREE_FLEET_AUDIT.md`](RUNBOOK_WORKTREE_FLEET_AUDIT.md). This
release runbook is scoped to release worktrees; the fleet runbook covers
`.worktrees/`, `.claude/worktrees/`, `.codex/worktrees/`, and
`/private/tmp/aragora-*`.

## 1. Validate your current worktree

```bash
bash scripts/check_release_worktree_hygiene.sh main
```

This check fails when:
- the worktree has uncommitted changes
- `HEAD` is ahead/behind `origin/main`
- you are on a non-`main` named branch (detached `HEAD` is allowed)

## 2. Use a clean release worktree

```bash
git fetch origin --prune
git worktree add /tmp/aragora-release-$(date +%s) origin/main
cd /tmp/aragora-release-*
```

Run release gates and merge operations from this worktree only.

## 3. Run the release-readiness gate

```bash
bash scripts/ci_release_readiness.sh
```

## 4. Optional self-host static validation

```bash
python scripts/check_self_host_compose.py
```

For runtime compose verification, use a dedicated host with Docker daemon access and a non-placeholder env file.
