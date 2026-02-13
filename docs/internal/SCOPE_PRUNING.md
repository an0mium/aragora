# Scope Pruning Analysis

Last updated: 2026-02-12

## Codebase Breakdown

| Tier | Category | LOC | % | Action |
|------|----------|-----|---|--------|
| 1 | Defensible Core (debate, gauntlet, knowledge, memory, ranking, reasoning, verification) | 252K | 17% | INVEST |
| 2 | Essential Infrastructure (server, agents, storage, CLI, config, resilience) | 547K | 37% | MAINTAIN, prune server/ |
| 3 | Enterprise Features (RBAC, audit, billing, security, auth, compliance) | 135K | 9% | MAINTAIN |
| 4 | Connectors (connectors, gateway, integrations, MCP) | 181K | 12% | CONSOLIDATE |
| 5 | Scope Creep (workflow, nomic, control_plane, blockchain, etc.) | 200K | 25% | EXTRACT/DEPRECATE |

## Completed Deletions

### Dead code removed (6,128 LOC)
- `aragora/coding/` — test generator, zero external imports
- `aragora/pipelines/` — essay synthesis, zero external imports
- `aragora/voice/` — wake word detector, exported but zero internal usage

### Handler consolidation
- Deleted `aragora/server/handlers/versioning/` (38 LOC re-export wrapper)
- Deleted `aragora/server/handlers/oauth_providers/` (duplicate of `_oauth/`)
- Updated imports to canonical module paths

### Speech module stub
- `aragora/speech/` deleted, handler returns 501

## Remaining Tier 5 Modules (candidates for extraction)

### Extract to `aragora-experimental` package
- `workflow/` (81 files, 31K LOC) — DAG engine, overlaps with LangGraph
- `nomic/` (87 files, 40K LOC) — self-improvement loop, experimental
- `genesis/` (7 files, 2.6K LOC) — agent evolution
- `evolution/` (6 files, 2.7K LOC) — evolution system

### Gate behind enterprise flag
- `control_plane/` (44 files, 25K LOC) — agent registry/scheduler

### Consolidation opportunities
- `channels/` + `bots/` into `connectors/` (25 files, 7.5K LOC)
- `autonomous/` into `nomic/` (4 files, 2.7K LOC)

## Server Handler Stats (620 modules, 273K LOC)

### Priority consolidation targets
1. OAuth: 3 implementations → 1 canonical (`_oauth/`)
2. Inbox: `inbox/` + `shared_inbox/` potential merge
3. Features: 37 root-level files → organized subdirectories
4. Thin wrappers: 32 directories under 700 LOC
