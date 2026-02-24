# Codex Worktree Autopilot

Use this guide when many Codex/Claude sessions are modifying Aragora simultaneously and you want minimal manual routing.

## Why

`scripts/codex_worktree_autopilot.py` keeps sessions in disposable, managed worktrees so state drift is handled automatically.

## Core Commands

```bash
# Start Codex in an auto-managed worktree
./scripts/codex_session.sh

# Ensure or create a managed worktree for an agent
python3 scripts/codex_worktree_autopilot.py ensure --agent codex --base main --reconcile

# Reconcile all managed worktrees onto origin/main
python3 scripts/codex_worktree_autopilot.py reconcile --all --base main

# Cleanup stale/expired managed worktrees
python3 scripts/codex_worktree_autopilot.py cleanup --base main --ttl-hours 24

# Inspect managed sessions
python3 scripts/codex_worktree_autopilot.py status
```

## Recommended Operating Loop

1. Start each agent with `./scripts/codex_session.sh --agent <name>`.
2. Before major test/fix cycles, run `... reconcile --all --base main`.
3. After merges/pushes, run `... cleanup --ttl-hours 24`.
4. Use short-lived worktrees; do not keep long-running stale session trees.

## Makefile Shortcuts

```bash
make codex-session
make worktree-ensure
make worktree-reconcile
make worktree-cleanup
```
