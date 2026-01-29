# Primitive Consolidation Decision Memo

**Status:** Proposed  
**Owner:** Architecture  
**Date:** 2026-01-29  
**Scope:** Convoys/Beads/Workspaces and Gateway/Inbox/Routing

---

## Problem

Aragora currently contains duplicated implementations of:

- **Convoys / Beads / Workspaces** across:
  - `aragora/nomic/*`
  - `aragora/extensions/gastown/*`
  - `aragora/workspace/*`
- **Gateway / Inbox / Capability Routing** across:
  - `aragora/gateway/*`
  - `aragora/extensions/moltbot/*`
  - `aragora/server/handlers/features/unified_inbox.py` and storage

This duplication blocks consistent behavior, makes auditing difficult, and prevents a clean Gastown/Moltbot parity layer.

---

## Decision (Proposed)

### Canonical Convoy/Bead/Workspace Layer

**Adopt `aragora/nomic` as the canonical implementation** for:
- Convoys
- Beads
- Hook queues / multi-agent execution

**Rationale**
- Nomic is the planned execution engine for “true test” automation.
- It already has convoy coordination and git-backed bead storage.
- It aligns with multi-agent coding integration and audit trails.

**Adapters**
- `aragora/extensions/gastown/*` becomes a thin adapter on top of Nomic.
- `aragora/workspace/*` is deprecated and replaced with Nomic-backed storage.

---

### Canonical Gateway / Inbox / Routing Layer

**Adopt `aragora/gateway` as the canonical implementation** for:
- Gateway server lifecycle
- Device registry
- Capability routing
- Inbox persistence

**Rationale**
- Gateway already includes the core routing/persistence logic.
- Server handlers can delegate to gateway primitives.

**Adapters**
- `aragora/extensions/moltbot/*` becomes a compatibility adapter for Moltbot protocols.
- `aragora/server/handlers/features/unified_inbox.py` delegates to gateway storage.

---

## Alternatives Considered

### Option A: Canonicalize `aragora/extensions/gastown`
**Pros:** Matches Gastown semantics.  
**Cons:** Requires re-plumbing Nomic loop to extension layer; more inversion of control.

### Option B: Canonicalize `aragora/workspace`
**Pros:** Simple JSONL persistence.  
**Cons:** Weak integration with Nomic and multi-agent execution; lacks hook queue.

### Option C: Canonicalize `aragora/extensions/moltbot`
**Pros:** Direct consumer parity.  
**Cons:** Gateway/server-level routing already lives elsewhere; extension is thinner.

---

## Migration Plan

1. **Define canonical APIs** in Nomic and Gateway modules.
2. **Add adapters** for Gastown + Moltbot extensions.
3. **Deprecate old paths** with warnings + doc updates.
4. **Remove legacy modules** after migrations complete.

---

## Impact

- **Positive:** Clear source of truth, simpler audits, easier multi-agent orchestration.
- **Risk:** Migration complexity; requires adapters and transition period.

---

## Next Steps

- Create API boundary docs for canonical layers.
- Add test coverage to guarantee adapter behavior.
- Update routing so server handlers call gateway primitives directly.

