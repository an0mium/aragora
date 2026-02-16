# Nomic-to-Core Integration Plan

*Created: February 15, 2026*

## Current State

The nomic features are already well-integrated:

| Feature | CLI | Server REST | Status |
|---------|-----|-------------|--------|
| HardenedOrchestrator | `aragora self-improve` | - | CLI only |
| MetaPlanner | `--dry-run` preview | - | CLI only |
| Worktree isolation | `aragora worktree` + auto in self-improve | - | CLI only |
| Nomic loop control | `aragora nomic` | `/api/nomic/control/*` | Both |
| Orchestration | - | `/api/v1/orchestration/*` | Server only |
| Hierarchical coordinator | - | - | Neither |
| CI feedback | auto in verify phase | - | Internal only |
| Cross-cycle learning | auto in HardenedOrchestrator | - | Internal only |

## Gap Analysis

### Gap 1: No REST API for Self-Improvement Runs

**Problem:** `aragora self-improve` is CLI-only. No way to trigger, monitor, or cancel self-improvement runs from the web UI, Slack, or other integrations.

**Proposed endpoints:**
```
POST /api/v1/self-improve/start
  Body: { goal, tracks?, budget_limit?, max_cycles?, dry_run? }
  Returns: { run_id, status: "started" | "preview", plan? }

GET  /api/v1/self-improve/:run_id
  Returns: { status, progress, subtasks, receipts, cost_so_far }

POST /api/v1/self-improve/:run_id/cancel
  Returns: { status: "cancelled", cleanup_summary }

GET  /api/v1/self-improve/history
  Returns: [{ run_id, goal, status, created_at, completed_at, cost }]

WS   /api/v1/self-improve/:run_id/stream
  Events: subtask_started, subtask_completed, gauntlet_result, merge_result
```

**Handler:** `aragora/server/handlers/self_improve.py` (~200 LOC)
- Reuses `HardenedOrchestrator` from CLI command
- Stores runs in `SelfImproveRunStore` (sqlite-backed, same as debate sessions)
- WebSocket stream reuses spectate infrastructure

**Effort:** ~200 LOC handler + ~100 LOC store + ~60 tests

### Gap 2: Hierarchical Coordinator Not Exposed

**Problem:** `HierarchicalCoordinator` (Planner/Worker/Judge) is only usable programmatically. Neither CLI nor server expose it.

**Proposed CLI flag:**
```bash
aragora self-improve "Goal" --hierarchical
# Uses HierarchicalCoordinator instead of flat execute_goal
```

**Proposed server parameter:**
```json
POST /api/v1/self-improve/start
{ "goal": "...", "mode": "hierarchical" }
```

**Effort:** ~20 LOC CLI + ~10 LOC handler

### Gap 3: Chat-Triggered Self-Improvement

**Problem:** Can't trigger self-improvement from Slack/Teams/Telegram. Already possible for debates via `debate_origin.py`.

**Proposed:**
- Add `SelfImproveOrigin` to `debate_origin.py` pattern
- Slack: `/aragora improve "Add retry logic"` slash command
- Teams: `@aragora improve "Fix error handling"` mention
- Result routes back to originating channel

**Effort:** ~100 LOC across 3 connector handlers

### Gap 4: Worktree Status in Server Dashboard

**Problem:** No visibility into active worktrees from web UI.

**Proposed endpoint:**
```
GET /api/v1/worktrees
  Returns: [{ branch, path, track, created_at, assignment_id, status }]

DELETE /api/v1/worktrees/:branch
  Cleans up a specific worktree
```

**Effort:** ~60 LOC handler + ~20 tests

## Priority Matrix

| # | Gap | Effort | Impact | Priority |
|---|-----|--------|--------|----------|
| 1 | Self-improve REST API | ~360 LOC | High (enables web UI, API clients) | **P0** |
| 2 | Hierarchical CLI flag | ~30 LOC | Medium (power users) | **P1** |
| 3 | Chat-triggered improve | ~100 LOC | Medium (Slack/Teams users) | **P2** |
| 4 | Worktree dashboard | ~80 LOC | Low (admin visibility) | **P3** |

## Implementation Order

1. **Sprint 1 (P0):** Self-improve REST API — unlocks web UI and programmatic access
2. **Sprint 2 (P1+P2):** Hierarchical flag + chat triggers — completes user-facing surface
3. **Sprint 3 (P3):** Worktree dashboard — admin tooling

## Dependencies

- Gap 1 depends on: nothing (HardenedOrchestrator already works)
- Gap 2 depends on: nothing (HierarchicalCoordinator already works)
- Gap 3 depends on: Gap 1 (needs REST API to route to)
- Gap 4 depends on: nothing (BranchCoordinator.list_worktrees() already works)

## Non-Goals

- **No new orchestrator classes** — HardenedOrchestrator is the canonical entry point
- **No new decomposition logic** — TaskDecomposer already handles all goal types
- **No new CI integration** — CI feedback already flows through verify phase
