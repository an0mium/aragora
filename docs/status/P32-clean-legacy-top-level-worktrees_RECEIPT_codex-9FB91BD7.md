# P32 Receipt: Clean Legacy Top-Level Worktrees

Generated: 2026-05-18T05:26:38Z

## Summary

Removed exactly one safe legacy top-level worktree through `scripts/safe_worktree_cleanup.py remove`. No source files, PRs, branches, issues, labels, automations, or protected paths were changed.

## Removed

| Path | Branch | Inspect proof | Size before | Result |
| --- | --- | --- | --- | --- |
| `.worktrees/codex-cross-agent-collision-control` | `codex/cross-agent-collision-control-20260517` | `removable=true`, `active_session=false`, `dirty=false`, `open_prs=[]`, `patch_equivalent_to_origin_main=true` | 850M | removed |

Removal command:

```bash
python3 scripts/safe_worktree_cleanup.py remove .worktrees/codex-cross-agent-collision-control --json
```

Helper output reported:

```json
{
  "removed": true,
  "branch_deleted": false,
  "path_purged": false,
  "status": "removed"
}
```

## Skipped

| Path | Size | Reason |
| --- | ---: | --- |
| `.worktrees/codex-inventory-runtime-budget` | 456M | open PR #7259; branch ahead of `origin/main` |
| `.worktrees/codex-lane-collision-hardening-followup` | 759M | open PR #7290; branch ahead of `origin/main` |
| `.worktrees/codex-operator-decisions-postmerge-hardening` | 453M | open PR #7293; branch ahead of `origin/main` |

## Disk Evidence

- Before candidate removal: `.worktrees` was 28G.
- After candidate removal: `.worktrees` was 27G.
- Filesystem after removal: `57Gi` free on `/System/Volumes/Data`.

## Observer Evidence

- `list_active_agent_sessions.py --json --max-pr-fetch 50 --skip-codex-desktop` showed no P32 collision before claim.
- Lane claim: `P32-clean-legacy-top-level-worktrees`, owner `codex-9FB91BD7`.
- `agent_bridge.py --json health` showed no lane collisions after claim, but did report two unrelated prunable temp-worktree records missing on disk.
- `triage_open_prs.py --json` reported 28 open PR results, all Bucket C.

## Deferred

No further P32 removal was safe in this session because the remaining target directories are tied to open PRs. P29/P31 can use the refreshed inventory and branch salvage tools for broader cleanup after their own safe inspections.
