# Nomic “True Test” Execution Plan

**Status:** Proposed  
**Owner:** Nomic/Platform  
**Date:** 2026-01-29

---

## Phase 1: Debate Profile Wiring (1–2 weeks)

**Goal:** Make 8–9 round / 8‑agent debate the default for the Nomic loop.

### Tasks
- Wire `NomicDebateProfile` into `scripts/nomic_loop.py`.
- Remove hard‑coded agent roster from Nomic loop.
- Align `DebatePhase` defaults with profile config.

### Acceptance
- Nomic loop runs full structured debate by default.
- 8 frontier agents selected from `AgentSettings.default_agent_list`.

---

## Phase 2: Deep Context Builder (2–4 weeks)

**Goal:** Replace summary-only context with indexed + RLM-backed context.

### Tasks
- Implement a `ContextBuilder` that:
  - Indexes repo into Knowledge Mound.
  - Builds a hierarchical map of modules/files.
  - Uses TRUE RLM REPL for targeted retrieval.
- Expose env toggles for TRUE RLM path.
- Add size-safe configuration for large context (10M token equivalent).

### Acceptance
- Debate context includes codebase index + RLM retrieval.
- Works for large repos without full prompt stuffing.

---

## Phase 3: Multi‑Agent Implementation (4–6 weeks)

**Goal:** Use Gastown-style convoys/beads to implement decisions with cross-checks.

### Tasks
- Implement `NomicImplementExecutor`:
  - Convert plan → beads.
  - Assign beads to agents via HookQueue.
  - Require peer review + targeted tests per bead.
- Wire executor into Nomic loop.

### Acceptance
- Multi-agent execution produces merged changes.
- Review + verification gates block unsafe changes.

---

## Phase 4: Governance + Safety (Ongoing)

**Goal:** Ensure approval + RBAC gates apply to all execution paths.

### Tasks
- Enforce approval gate in executor path.
- Add audit trails for all bead outcomes.

### Acceptance
- No auto-commit without explicit approval.
- Full provenance stored for every change.

