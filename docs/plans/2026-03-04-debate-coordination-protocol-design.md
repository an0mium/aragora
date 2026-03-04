# Debate as Coordination Protocol — Design Doc

> **Approach:** Dogfood Aragora's own Arena to coordinate heterogeneous AI agents
> (Claude, Codex, Gemini) working on separate worktrees/branches.

## Guiding Principle

**Make bad outcomes cheap to fix, not hard to create.** Prefer fast-forward merges,
auto-revert on test failure, and lightweight conflict detection over gates and approval
workflows. Never block an agent from pushing code — detect and fix problems after the fact.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────┐
│              Coordination Arena                   │
│  (aragora.coordination.arena)                     │
│                                                   │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Session    │  │ File     │  │ Conflict     │  │
│  │ Registry   │  │ Claims   │  │ Resolver     │  │
│  │            │  │          │  │ (Arena)      │  │
│  └─────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│        │              │               │           │
│        └──────────────┼───────────────┘           │
│                       │                           │
│              ┌────────┴────────┐                  │
│              │  Event Bus      │                  │
│              │  (file-based)   │                  │
│              └────────┬────────┘                  │
│                       │                           │
└───────────────────────┼───────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────┴────┐    ┌─────┴────┐    ┌─────┴────┐
   │ Claude  │    │ Codex    │    │ Gemini   │
   │ Session │    │ Session  │    │ Session  │
   │ (wt-1)  │    │ (wt-2)   │    │ (wt-3)  │
   └─────────┘    └──────────┘    └──────────┘
```

**Key decisions:**
- File-based event bus (JSON in `.aragora_coordination/`), not Redis — works without infra
- Sessions are peers, no central orchestrator process required
- Arena debates run in-process when conflicts arise (not a long-running server)
- Everything is advisory — agents can ignore claims and push anyway

## 2. Session Registration & Discovery

Each agent session writes a registration file on startup:

```
.aragora_coordination/
├── sessions/
│   ├── claude-abc123.json    # { agent: "claude", worktree: "...", pid: 1234, started: "...", focus: "Track C" }
│   └── codex-def456.json
├── claims/
│   └── ...
└── events/
    └── ...
```

- Registration is a JSON file named `{agent}-{short-id}.json`
- File is removed on clean session exit (best-effort, not guaranteed)
- Stale sessions detected by PID liveness check (same as existing `.claude-session-active`)
- Discovery: `glob(".aragora_coordination/sessions/*.json")`

**No central process.** Any session can read the directory to see who else is active.

## 3. File Claim Protocol

Before editing a file, a session writes a claim:

```json
{
  "session": "claude-abc123",
  "files": ["aragora/server/handlers/auth.py"],
  "intent": "Refactoring OIDC flow",
  "claimed_at": "2026-03-04T10:00:00Z",
  "ttl_minutes": 30
}
```

**Rules:**
- Claims are **advisory, not blocking** — agents can edit unclaimed files freely
- If another session already claims a file, the new session gets a warning but can proceed
- Claims auto-expire after TTL (default 30 min)
- On conflict: log a warning, continue working, resolve at merge time

**Why advisory, not mandatory:** Mandatory locks create deadlocks, slow iteration,
and require a central authority. Advisory claims give agents enough information to
avoid conflicts without blocking anyone.

## 4. Conflict Resolution via Arena Debate

When two sessions modify the same file and their branches can't fast-forward merge:

1. **Detection:** `SemanticConflictDetector` (already exists in `branch_coordinator.py`)
   runs AST-aware diff to classify conflict severity
2. **Trivial conflicts** (import ordering, whitespace, non-overlapping hunks): auto-merge
3. **Semantic conflicts** (same function modified differently): trigger a 2-round Arena debate
   - Agents: the two sessions' agents + a neutral "Architect" agent
   - Protocol: each side presents their change rationale, architect picks winner or proposes synthesis
   - Output: a merge commit or revert recommendation
4. **Fallback:** If debate fails or times out, keep both branches, flag for human review

**Cost control:** Debates only run for semantic conflicts (rare). Most merges are trivial.

## 5. Integration Points

Uses existing Aragora primitives:
- `aragora.debate.orchestrator.Arena` — conflict resolution debates
- `aragora.nomic.branch_coordinator.SemanticConflictDetector` — AST diff
- `aragora.events.dispatcher` — internal event routing
- `scripts/codex_worktree_autopilot.py` — worktree lifecycle (already has session locks)

New code lives in `aragora/coordination/`:
- `registry.py` — session registration and discovery
- `claims.py` — file claim protocol
- `resolver.py` — conflict detection + Arena debate trigger
- `bus.py` — file-based event bus

## 6. What This System Does NOT Do

- **No PR admission controller** — agents push freely, CI catches problems
- **No mandatory review gates** — fast iteration over safety theater
- **No central orchestrator** — fully peer-to-peer, no single point of failure
- **No blocking locks** — advisory claims only, agents always proceed
- **No aggressive cleanup** — worktree lifecycle stays as-is (PID-based session locks)

---

## Success Criteria

1. Two concurrent Claude sessions on overlapping files produce a clean merge >80% of the time
2. Zero increase in per-PR merge latency (no new gates)
3. Conflict resolution debates complete in <30 seconds
4. System works with zero infrastructure beyond the git repo itself
