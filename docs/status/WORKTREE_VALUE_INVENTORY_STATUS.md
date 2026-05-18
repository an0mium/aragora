# Worktree Value Inventory Status

Last updated: 2026-05-18T17:03:10Z

This repo-tracked status surface captures the most recent run of `scripts/codex_worktree_value_inventory.py` against the canonical and legacy Aragora worktree roots, classifying each checkout as preserve / harvest_candidate / cleanup_candidate.

## Summary

- Source inventory: `/private/tmp/v11_smart_inv.json`
- Base ref: `origin/main`
- Roots: `/Users/armand/Development/aragora/.worktrees/codex-auto`
- Total candidates: `44`
- Active sessions: `12`
- Registered worktrees: `29`
- Candidates with open PR: `0`
- Dirty checkouts: `2`

## Classifications

- `active_or_dirty`: 14
- `no_git_cache_residue`: 3
- `patch_equivalent_or_merged`: 1
- `receipt_protected`: 1
- `unique_unharvested`: 25

## Decisions

- `cleanup_candidate`: 4
- `harvest_candidate`: 25
- `preserve`: 15

## Harvest Candidates

| Path | Branch | Classification | Ahead | Behind | Dirty | Active |
| --- | --- | --- | --- | --- | --- | --- |
| `ragora/.worktrees/codex-auto/claude-20260517-165018-00bb191c` | `worktree-packets-keyboard-throughput-20260517` | `unique_unharvested` | 3 | 38 | no | no |
| `ragora/.worktrees/codex-auto/claude-20260518-041625-69149fbf` | `claude/P24-canonical-test-definitions-count-drift-20260518-041606` | `unique_unharvested` | 1 | 19 | no | no |
| `ragora/.worktrees/codex-auto/claude-20260518-043718-a1054f6f` | `claude/P28-A-identify-lane-owner-20260518-043722` | `unique_unharvested` | 2 | 19 | no | no |
| `ragora/.worktrees/codex-auto/claude-20260518-051359-40073f53` | `claude/P29-steering-mailbox-writer-20260518-051405` | `unique_unharvested` | 1 | 15 | no | no |
| `ragora/.worktrees/codex-auto/claude-20260518-053518-29cc1ddc` | `claude/P30-operator-snapshot-steering-messages-20260518-053523` | `unique_unharvested` | 1 | 12 | no | no |
| `aragora/.worktrees/codex-auto/droid-20260517-204819-018cc17d` | `droid/P02-freshness-probe-rerun-20260517-204809` | `unique_unharvested` | 2 | 27 | no | no |
| `aragora/.worktrees/codex-auto/droid-20260518-002344-c62e03f6` | `droid/P16-stage2-auto-merge-bucket-a-20260518-002325` | `unique_unharvested` | 12 | 2 | no | no |
| `aragora/.worktrees/codex-auto/droid-20260518-015653-1486bdb7` | `droid/P17-stage3-triage-bucket-c-batcher-20260518-015641` | `unique_unharvested` | 1 | 20 | no | no |
| `agora/.worktrees/codex-auto/harvest-metrics-refresh-20260516` | `codex/harvest-metrics-refresh-20260516` | `unique_unharvested` | 1 | 51 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-3844d54a-micro-1` | `codex/swarm-3844d54a-micro-1` | `unique_unharvested` | 1 | 57 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-3844d54a-micro-2` | `codex/swarm-3844d54a-micro-2` | `unique_unharvested` | 1 | 57 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-3844d54a-micro-3` | `codex/swarm-3844d54a-micro-3` | `unique_unharvested` | 2 | 57 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-3844d54a-micro-4` | `codex/swarm-3844d54a-micro-4` | `unique_unharvested` | 3 | 57 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5a555597-micro-1` | `codex/swarm-5a555597-micro-1` | `unique_unharvested` | 1 | 60 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5a555597-micro-2` | `codex/swarm-5a555597-micro-2` | `unique_unharvested` | 1 | 60 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5a555597-micro-4` | `codex/swarm-5a555597-micro-4` | `unique_unharvested` | 2 | 59 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5a555597-micro-5` | `codex/swarm-5a555597-micro-5` | `unique_unharvested` | 3 | 59 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5f57bb72-micro-1` | `codex/swarm-5f57bb72-micro-1` | `unique_unharvested` | 1 | 65 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5f57bb72-micro-2` | `codex/swarm-5f57bb72-micro-2` | `unique_unharvested` | 1 | 65 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5f57bb72-micro-3` | `codex/swarm-5f57bb72-micro-3` | `unique_unharvested` | 1 | 65 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5f57bb72-micro-4` | `codex/swarm-5f57bb72-micro-4` | `unique_unharvested` | 2 | 65 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5f57bb72-micro-5` | `codex/swarm-5f57bb72-micro-5` | `unique_unharvested` | 4 | 65 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-848a8081-micro-2` | `codex/swarm-848a8081-micro-2` | `unique_unharvested` | 1 | 67 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-848a8081-micro-3` | `codex/swarm-848a8081-micro-3` | `unique_unharvested` | 1 | 67 | no | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-848a8081-micro-5` | `codex/swarm-848a8081-micro-5` | `unique_unharvested` | 3 | 67 | no | no |

## Cleanup Candidates

| Path | Branch | Classification | Ahead | Behind | Dirty | Active |
| --- | --- | --- | --- | --- | --- | --- |
| `ragora/.worktrees/codex-auto/claude-20260501-195315-89c3c0ad` | `-` | `no_git_cache_residue` | - | - | no | no |
| `aragora/.worktrees/codex-auto/codex-20260422-194119-a5c0dd59` | `-` | `no_git_cache_residue` | - | - | no | no |
| `aragora/.worktrees/codex-auto/droid-20260518-041442-0e8aade8` | `droid/P20-model-pins-frontier-aligned-20260518-041438` | `patch_equivalent_or_merged` | 1 | 19 | no | no |
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
| `ragora/.worktrees/codex-auto/claude-20260517-180111-e41d43ca` | `worktree-triage-classifier-20260517` | `active_or_dirty` | 4 | 32 | yes | no |
| `aragora/.worktrees/codex-auto/codex-20260515-152237-14d11f01` | `-` | `active_or_dirty` | - | - | no | yes |
| `aragora/.worktrees/codex-auto/droid-20260517-231527-ce5c3b15` | `droid/P13a-canonical-km-adapter-count-drift-20260517-231510` | `active_or_dirty` | 2 | 23 | yes | no |
| `lopment/aragora/.worktrees/codex-auto/swarm-5a555597-micro-3` | `codex/swarm-5a555597-micro-3` | `receipt_protected` | 1 | 59 | no | no |

## Provenance

- Generator: `scripts/codex_worktree_value_inventory.py` (see PR #7250 for canonical+legacy smart roots; PR #7253/#7254 for managed-session lookup and foreign-worktree preservation).
- Publisher: `scripts/publish_worktree_value_inventory.py` (this file). Read-only; never deletes worktrees or branches.
