# P45 Receipt: Legacy Worktree Dir Audit + Clean

Generated: 2026-05-18T05:45:39Z

## Summary

Cleaned stale Claude worktree entries with `scripts/safe_worktree_cleanup.py inspect` followed by `scripts/safe_worktree_cleanup.py remove --purge-path` only for paths that individually reported `removable: true`. No source files, PRs, branches, issues, labels, automations, raw transcripts, or protected paths were changed.

## Removed

| Path | Size before | Inspect proof | Result |
| --- | ---: | --- | --- |
| `/Users/armand/.claude-worktrees/aragora/agt-review` | 647M | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `/Users/armand/.claude-worktrees/aragora/dogfood-6796-ac2bc747` | 853M | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `/Users/armand/.claude-worktrees/aragora/round-31b-parallel` | 648M | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `/Users/armand/.claude-worktrees/aragora/round-31a-parallel` | 647M | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `/Users/armand/.claude-worktrees/aragora/round-2026-04-30b` | 1.9G | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `/Users/armand/.claude-worktrees/aragora/round-2026-04-30c` | 1.9G | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `/Users/armand/.claude-worktrees/aragora/round-2026-04-30d` | 1.9G | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged |
| `/Users/armand/.claude-worktrees/aragora/youthful-saha-cdd7ab` | 4.0K | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]` | purged after second inspect |

## Skipped

| Path | Reason |
| --- | --- |
| `.worktrees/codex-pr7170-e92e7b50` | fresh inspection reported `dirty_worktree`; 4.0K |
| `.worktrees/codex-salvage-h1-direct-dispatch-invocation` | fresh inspection reported `dirty_worktree`; 4.0K |
| `/Users/armand/.claude-worktrees/aragora/merge-authority-followup-20260428` | fresh inspection reported `branch_ahead_of_origin_main`; 322M |
| `/Users/armand/.claude-worktrees/aragora` | parent preserved because it contains the blocked `merge-authority-followup-20260428` child |

## Disk Evidence

- Before cleanup: `df -h .` reported 57 GiB free.
- Before cleanup: `/Users/armand/.claude-worktrees` was 8.7G.
- After cleanup: `/Users/armand/.claude-worktrees` is 322M.
- After cleanup: `df -h .` reported 63 GiB free.
- `.worktrees` remained about 29G; P45 did not remove repo-local paths because the named candidates were dirty.

## Observer Evidence

- Raw lane registry before claim showed active `P28-refresh-worktree-value-inventory` and `Q01-repair-7292-admin-merge`; no P45 collision.
- Same-work pre-claim check printed no `COLLISION_RISK`.
- `agent_bridge.py health` reported two unrelated prunable temp-worktree records that were already missing on disk.
- #7292 was not touched. It was already owned by `Q01-repair-7292-admin-merge` and remained outside this lane.

## Deferred

P29 can continue repo-local cleanup for candidates that pass fresh safe inspections. P44 can handle remote branch cleanup separately. The preserved Claude worktree child needs salvage/review before removal because it is far ahead of `origin/main`.
