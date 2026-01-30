# Canonical APIs: Convoys/Beads/Workspaces + Gateway/Inbox/Routing

**Status:** Proposed  
**Owner:** Architecture  
**Date:** 2026-01-29

---

## Purpose

Define the **single source of truth** for core orchestration primitives and the
compatibility adapters that sit on top. This prevents duplication and ensures
consistent behavior across enterprise and parity layers (Gastown/OpenClaw).

---

## Canonical Layer A: Convoys / Beads / Workspaces

**Canonical package:** `aragora/nomic/*`

### Responsibilities
- **Convoy lifecycle** (create → assign → merge → close)
- **Bead definition** (unit of work, ownership, metadata)
- **Hook queue dispatch** (multi-agent execution handoffs)
- **Audit trail** (task provenance + outcomes)

### Primary Modules
- `aragora/nomic/convoys.py`
- `aragora/nomic/beads.py`
- `aragora/nomic/hook_queue.py`
- `aragora/nomic/convoy_coordinator.py`

### Adapters
- `aragora/extensions/gastown/*`  
  Thin compatibility layer; **must not implement core logic**.
- `aragora/workspace/*`  
  Deprecated; must forward to canonical APIs.

### Public API Surface (initial)
- `create_convoy(...)`
- `add_bead(...)`
- `assign_bead(...)`
- `commit_bead(...)`
- `close_convoy(...)`

---

## Canonical Layer B: Gateway / Inbox / Capability Routing

**Canonical package:** `aragora/gateway/*`

### Responsibilities
- **Gateway lifecycle** (server start/stop)
- **Device registry + capabilities**
- **Inbox ingestion + persistence**
- **Capability routing + policy enforcement**

### Primary Modules
- `aragora/gateway/server.py`
- `aragora/gateway/device_registry.py`
- `aragora/gateway/inbox.py`
- `aragora/gateway/capability_router.py`
- `aragora/gateway/persistence.py`

### Adapters
- `aragora/extensions/moltbot/*`  
  Compatibility adapter for OpenClaw protocols; **no core logic**.
- `aragora/server/handlers/features/unified_inbox.py`  
  Delegates to gateway storage.

### Public API Surface (initial)
- `register_device(...)`
- `update_capabilities(...)`
- `enqueue_message(...)`
- `route_capability(...)`
- `persist_message(...)`

---

## Required Conventions

- **No duplication:** New functionality must land in canonical packages only.
- **Adapters only:** Extension modules should wrap, not reimplement.
- **Auditability:** All actions must emit structured events for audit trails.
- **Policy gating:** All routing + execution must consult approval/RBAC gates.

---

## Migration Checklist

1. Define canonical interfaces in Nomic + Gateway modules.
2. Replace extension logic with adapter calls.
3. Add deprecation warnings in legacy modules.
4. Remove legacy paths after stable cutover.
