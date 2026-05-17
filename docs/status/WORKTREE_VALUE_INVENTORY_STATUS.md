# Worktree Value Inventory Status

Last updated: 2026-05-17T04:08:00Z

This repo-tracked status surface captures the most recent run of `scripts/codex_worktree_value_inventory.py` against the canonical and legacy Aragora worktree roots, classifying each checkout as preserve / harvest_candidate / cleanup_candidate.

## Summary

- Source inventory: `/private/tmp/wvi_canonical.json`
- Base ref: `origin/main`
- Roots: `/Users/armand/Development/aragora/.worktrees/codex-auto`
- Total candidates: `45`
- Active sessions: `12`
- Registered worktrees: `30`
- Candidates with open PR: `0`
- Dirty checkouts: `0`

## Classifications

- `active_or_dirty`: 12
- `no_git_cache_residue`: 3
- `patch_equivalent_or_merged`: 2
- `unique_unharvested`: 28

## Decisions

- `cleanup_candidate`: 5
- `harvest_candidate`: 28
- `preserve`: 12

## Harvest Candidates

| Path | Branch | Classification | Ahead | Behind | Dirty | Active |
| --- | --- | --- | --- | --- | --- | --- |
| `ragora/.worktrees/codex-auto/claude-20260517-011811-b5b49881` | `droid/a2-pr3-admission-recovery-rescue-map-and-fixtures-20260516` | `unique_unharvested` | 1 | 2 | no | no |
| `ragora/.worktrees/codex-auto/claude-20260517-014513-4b437897` | `droid/phase1-fastapi-observer-truth-20260516` | `unique_unharvested` | 1 | 1 | no | no |
| `aragora/.worktrees/codex-auto/codex-20260516-171432-ae417793` | `codex/corpus-aware-dispatch-upgrade-7209` | `unique_unharvested` | 1 | 16 | no | no |
| `aragora/.worktrees/codex-auto/codex-20260516-184833-3348d471` | `codex/review-pr-advisory-hardening` | `unique_unharvested` | 1 | 12 | no | no |
| `aragora/.worktrees/codex-auto/codex-20260517-013727-4aa9468a` | `-` | `unique_unharvested` | 1 | 2 | no | no |
| `aragora/.worktrees/codex-auto/codex-20260517-014233-61822f8b` | `codex/agent-bridge-process-census` | `unique_unharvested` | 1 | 1 | no | no |
| `elopment/aragora/.worktrees/codex-auto/codex-activity-repair` | `-` | `unique_unharvested` | 7 | 12 | no | no |
| `ees/codex-auto/harvest-github-cli-health-unused-sys-20260516` | `codex/harvest-github-cli-health-unused-sys-20260516` | `unique_unharvested` | 1 | 12 | no | no |
| `agora/.worktrees/codex-auto/harvest-metrics-refresh-20260516` | `codex/harvest-metrics-refresh-20260516` | `unique_unharvested` | 1 | 6 | no | no |
| `ees/codex-auto/harvest-preflight-rescue-guard-tests-20260516` | `codex/harvest-preflight-rescue-guard-tests-20260516` | `unique_unharvested` | 2 | 12 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-3844d54a-micro-1` | `codex/swarm-3844d54a-micro-1` | `unique_unharvested` | 1 | 12 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-3844d54a-micro-2` | `codex/swarm-3844d54a-micro-2` | `unique_unharvested` | 1 | 12 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-3844d54a-micro-3` | `codex/swarm-3844d54a-micro-3` | `unique_unharvested` | 2 | 12 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-3844d54a-micro-4` | `codex/swarm-3844d54a-micro-4` | `unique_unharvested` | 3 | 12 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5a555597-micro-1` | `codex/swarm-5a555597-micro-1` | `unique_unharvested` | 1 | 15 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5a555597-micro-2` | `codex/swarm-5a555597-micro-2` | `unique_unharvested` | 1 | 15 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5a555597-micro-3` | `codex/swarm-5a555597-micro-3` | `unique_unharvested` | 1 | 14 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5a555597-micro-4` | `codex/swarm-5a555597-micro-4` | `unique_unharvested` | 2 | 14 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5a555597-micro-5` | `codex/swarm-5a555597-micro-5` | `unique_unharvested` | 3 | 14 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5f57bb72-micro-1` | `codex/swarm-5f57bb72-micro-1` | `unique_unharvested` | 1 | 20 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5f57bb72-micro-2` | `codex/swarm-5f57bb72-micro-2` | `unique_unharvested` | 1 | 20 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5f57bb72-micro-3` | `codex/swarm-5f57bb72-micro-3` | `unique_unharvested` | 1 | 20 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5f57bb72-micro-4` | `codex/swarm-5f57bb72-micro-4` | `unique_unharvested` | 2 | 20 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5f57bb72-micro-5` | `codex/swarm-5f57bb72-micro-5` | `unique_unharvested` | 4 | 20 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-848a8081-micro-2` | `codex/swarm-848a8081-micro-2` | `unique_unharvested` | 1 | 22 | no | no |

*Truncated: 3 additional rows omitted.*

## Cleanup Candidates

| Path | Branch | Classification | Ahead | Behind | Dirty | Active |
| --- | --- | --- | --- | --- | --- | --- |
| `ragora/.worktrees/codex-auto/claude-20260501-195315-89c3c0ad` | `-` | `no_git_cache_residue` | - | - | no | no |
| `ragora/.worktrees/codex-auto/claude-20260517-015647-9ea8f6b7` | `droid/phase2-worktree-value-inventory-20260516` | `patch_equivalent_or_merged` | 0 | 1 | no | no |
| `aragora/.worktrees/codex-auto/codex-20260422-194119-a5c0dd59` | `-` | `no_git_cache_residue` | - | - | no | no |
| `aragora/.worktrees/codex-auto/codex-20260517-013517-967984c4` | `codex/fix-worktree-inventory-managed-root` | `patch_equivalent_or_merged` | 1 | 1 | no | no |
| `ent/aragora/.worktrees/codex-auto/rbac-health-fix-20260413-1` | `-` | `no_git_cache_residue` | - | - | no | no |

## Preserved (active or dirty)

| Path | Branch | Classification | Ahead | Behind | Dirty | Active |
| --- | --- | --- | --- | --- | --- | --- |
| `ragora/.worktrees/codex-auto/claude-20260430-041011-3b22a0f5` | `-` | `active_or_dirty` | - | - | no | yes |
| `ragora/.worktrees/codex-auto/claude-20260430-042801-d58ae59e` | `-` | `active_or_dirty` | - | - | no | yes |
| `ragora/.worktrees/codex-auto/claude-20260430-044410-204ba6b3` | `-` | `active_or_dirty` | - | - | no | yes |
| `ragora/.worktrees/codex-auto/claude-20260430-125421-513d6f8d` | `-` | `active_or_dirty` | - | - | no | yes |
| `ragora/.worktrees/codex-auto/claude-20260430-132040-05232470` | `-` | `active_or_dirty` | - | - | no | yes |
| `ragora/.worktrees/codex-auto/claude-20260430-132610-4446bddf` | `-` | `active_or_dirty` | - | - | no | yes |
| `ragora/.worktrees/codex-auto/claude-20260501-174558-6eed6d67` | `-` | `active_or_dirty` | - | - | no | yes |
| `ragora/.worktrees/codex-auto/claude-20260501-230411-bc06dedc` | `-` | `active_or_dirty` | - | - | no | yes |
| `ragora/.worktrees/codex-auto/claude-20260503-144747-b1c1b5ad` | `-` | `active_or_dirty` | - | - | no | yes |
| `ragora/.worktrees/codex-auto/claude-20260503-150130-e21247fc` | `-` | `active_or_dirty` | - | - | no | yes |
| `ragora/.worktrees/codex-auto/claude-20260507-171638-9eaae229` | `-` | `active_or_dirty` | - | - | no | yes |
| `aragora/.worktrees/codex-auto/codex-20260515-152237-14d11f01` | `-` | `active_or_dirty` | - | - | no | yes |

## Provenance

- Generator: `scripts/codex_worktree_value_inventory.py` (see PR #7250 for canonical+legacy smart roots; PR #7253/#7254 for managed-session lookup and foreign-worktree preservation).
- Publisher: `scripts/publish_worktree_value_inventory.py` (this file). Read-only; never deletes worktrees or branches.
