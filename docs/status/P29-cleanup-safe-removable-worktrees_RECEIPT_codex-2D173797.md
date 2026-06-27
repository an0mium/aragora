# P29 Receipt: Cleanup Safe-Removable Worktrees

Generated: 2026-05-18T12:14:19Z

## Summary

Removed 9 helper-approved worktree paths with `scripts/safe_worktree_cleanup.py inspect` followed by `scripts/safe_worktree_cleanup.py remove --purge-path`. No raw deletion, branch deletion, PR mutation, issue mutation, process killing, or protected-path work was performed.

## Removed

| Path | Size before | Inspect proof | Result |
| --- | ---: | --- | --- |
| `.worktrees/codex-auto/claude-20260501-195315-89c3c0ad` | 4.0K | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `.worktrees/codex-auto/claude-20260517-143423-4810eda2` | 358M | `removable=true`, `active_session=false`, `dirty=false`, `patch_equivalent_to_origin_main=true`, `open_prs=[]` | removed |
| `.worktrees/codex-auto/claude-20260517-144003-b211740c` | 474M | `removable=true`, `active_session=false`, `dirty=false`, `patch_equivalent_to_origin_main=true`, `open_prs=[]` | removed |
| `.worktrees/codex-auto/claude-20260517-171434-066430b1` | 479M | `removable=true`, `active_session=false`, `dirty=false`, `patch_equivalent_to_origin_main=true`, `open_prs=[]` | removed |
| `.worktrees/codex-auto/claude-20260517-174858-47953d00` | 329M | `removable=true`, `active_session=false`, `dirty=false`, `patch_equivalent_to_origin_main=true`, `open_prs=[]` | removed |
| `.worktrees/codex-auto/codex-20260422-194119-a5c0dd59` | 24K | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `.worktrees/codex-auto/rbac-health-fix-20260413-1` | 32K | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `.worktrees/codex-pr7170-e92e7b50` | 4.0K | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `.worktrees/codex-salvage-h1-direct-dispatch-invocation` | 4.0K | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |

## Disk Evidence

- Before cleanup: `df -h .` reported 48 GiB free.
- Before cleanup: `.worktrees` was 25G; `.worktrees/codex-auto` was 15G.
- After cleanup: `df -h .` reported 50 GiB free.
- After cleanup: `.worktrees` was 23G; `.worktrees/codex-auto` was 13G.

## Observer Evidence

- Same-work pre-claim check printed no `COLLISION_RISK`.
- Lane claim: `P29-cleanup-safe-removable-worktrees`, owner `codex-2D173797`.
- Raw registry before claim showed active `Q01-repair-7292-metrics-drift`; #7292 was not touched.
- `agent_bridge.py health` showed only two pre-existing missing temp-worktree records.
- `triage_open_prs.py --json` reported 31 results, all Bucket C.

## Deferred / Not Done

- No remote branches were deleted.
- No harvest candidates were opened as PRs.
- No #7292 mutation was attempted because an active lane already owned the repair.
- No update to the worktree inventory artifact was attempted; P40 was fresh.
