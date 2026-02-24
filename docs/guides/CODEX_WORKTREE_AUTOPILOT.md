# Codex Worktree Autopilot

Use this guide when many Codex/Claude sessions are modifying Aragora simultaneously and you want minimal manual routing.

## Why

`scripts/codex_worktree_autopilot.py` keeps sessions in disposable, managed worktrees so state drift is handled automatically.

## Core Commands

```bash
# Start Codex in an auto-managed worktree
./scripts/codex_session.sh

# Start a named lane in an isolated managed-dir
./scripts/codex_session.sh --agent codex-ci --managed-dir .worktrees/codex-auto-ci

# Ensure or create a managed worktree for an agent
python3 scripts/codex_worktree_autopilot.py ensure --agent codex --base main --reconcile

# Reconcile all managed worktrees onto origin/main
python3 scripts/codex_worktree_autopilot.py reconcile --all --base main

# Cleanup stale/expired managed worktrees
python3 scripts/codex_worktree_autopilot.py cleanup --base main --ttl-hours 24

# One-shot maintenance (non-destructive merge integration + cleanup)
python3 scripts/codex_worktree_autopilot.py maintain --base main --strategy merge --ttl-hours 24

# Inspect managed sessions
python3 scripts/codex_worktree_autopilot.py status
```

## Recommended Operating Loop

1. Start each agent with `./scripts/codex_session.sh --agent <name>`.
2. Before major test/fix cycles, run `... maintain --base main --strategy merge --ttl-hours 24`.
3. Use `... reconcile --all --base main` when you need explicit sync reporting.
4. Use short-lived worktrees; do not keep long-running stale session trees.

## Makefile Shortcuts

```bash
make codex-session
make worktree-ensure
make worktree-reconcile
make worktree-cleanup
make worktree-maintain
```

## Auto-Maintainer (macOS launchd)

```bash
# Install background maintainer (every 5 minutes)
make worktree-maintainer-install

# Check status
make worktree-maintainer-status

# Uninstall
make worktree-maintainer-uninstall
```

The maintainer runs non-destructive upkeep by default:
- Integration strategy: `merge`
- Branch retention: keeps local `codex/*` branches (`--no-delete-branches`)
- Safety: skips worktrees with active processes (use `--include-active` only if needed)
- Daemon mode: `--reconcile-only` (no cleanup deletions in background)
