# Aragora Parity + Agent Fabric Backlog (Gastown + OpenClaw)

**Version:** 2.4.x  
**Created:** January 2026  
**Scope:** Enterprise core + Developer (Gastown) + Consumer (OpenClaw) extensions  
**Goal:** Maintain Aragora as an enterprise decision control plane while adding opt-in parity layers.

---

## Executive Summary

This backlog focuses on three priorities in the correct order:
1. **Align defaults + contracts** so the system behaves consistently across API/UI/docs.
2. **Unify duplicated primitives** (convoys/beads/workspaces, gateway/inbox/routing) into canonical APIs.
3. **Upgrade the Nomic loop** to run a true multi-agent debate and multi-agent implementation flow.

Parity with Gastown and OpenClaw is treated as **opt-in extension layers**, gated by policy, with clear identity and security boundaries.

---

## Milestones (Best Order)

### 0–2 Weeks: Stabilize + Decide

**M1. Defaults & Contracts**
- **Goal:** Make 9-round / 8-agent / judge consensus the canonical defaults everywhere.
- **Acceptance:**
  - Defaults consistent across settings, API, UI, docs, OpenAPI.
  - CI detects default drift.

**M2. Primitive Consolidation Decision**
- **Goal:** Select canonical APIs for:
  - convoys/beads/workspaces
  - gateway/inbox/capability routing
- **Acceptance:** One source of truth per primitive + deprecation plan.

**M3. Parity Matrix + Acceptance Tests**
- **Goal:** Define MVP parity scope for Gastown + OpenClaw with tests.
- **Acceptance:** Each parity item mapped to module + test plan.

---

### 30 Days: Unify + Wire Core

**M4. Convoy/Bead/Workspace Unification v1**
- **Goal:** Canonical API with adapters from legacy systems.
- **Acceptance:**
  - Single CRUD API.
  - Legacy paths forward into the canonical layer.
  - Tests updated.

**M5. Gateway/Inbox/Routing Unification v1**
- **Goal:** One gateway pipeline with compatibility adapters.
- **Acceptance:** Single routing pipeline for inbox + capabilities.

**M6. Nomic Debate Profile**
- **Goal:** 8–9 rounds + 8 frontier agents as the default Nomic profile.
- **Acceptance:** Nomic loop runs full structured debate by default.

---

### 60 Days: Scale + Deep Context

**M7. Agent Fabric MVP**
- **Goal:** Scheduler, isolation, budgets, telemetry.
- **Acceptance:** Safe 50–100 concurrent agent runs with cost limits.

**M8. True RLM Context Builder**
- **Goal:** Codebase index + REPL-backed context for large repos.
- **Acceptance:** Debate uses indexed context (not summary-only).

**M9. Multi-Agent Implement Executor**
- **Goal:** Gastown-style convoys/beads driven from Nomic plan.
- **Acceptance:** Plan → multi-agent changes → peer review → tests → commit.

---

### 90 Days: Parity MVPs + Hardening

**M10. Gastown Parity MVP**
- **Goal:** Developer workflow parity (workspaces, hooks, convoys, beads).
- **Acceptance:** End-to-end multi-agent dev workflow with persistence.

**M11. Moltbot Parity MVP**
- **Goal:** Gateway protocol compatibility + device node skeletons + canvas/voice/browser control v1.
- **Acceptance:** Local assistant flow works with device pairing + routing.

**M12. Governance Hardening**
- **Goal:** Unified RBAC + approval + sandboxing for device/computer-use.
- **Acceptance:** Audit trail for all actions; policy gates enforced.

---

## Epics & Issues

### Epic A — Defaults + Contract Alignment
- **A1:** Centralize defaults in settings and generate UI/API constants.
- **A2:** OpenAPI defaults + examples updated to 9 rounds and judge consensus.
- **A3:** Docs reflect default profile and max rounds.

### Epic B — Canonical Primitives
- **B1:** Convoy/Bead/Workspace canonical API.
- **B2:** Gateway/Inbox/Capability routing canonical pipeline.
- **B3:** Adapters for legacy paths and staged deprecation.

### Epic C — Nomic “True Test” Upgrade
- **C1:** Nomic debate profile (8–9 rounds, 8 agents).
- **C2:** Codebase index + True RLM REPL context builder.
- **C3:** Multi-agent implementation executor (convoys + hook queue + review gate).

### Epic D — Agent Fabric MVP
- **D1:** Scheduler + isolation model.
- **D2:** Budget controls + telemetry.
- **D3:** Cross-agent verification hooks.

### Epic E — Gastown Parity (Developer Profile)
- **E1:** Workspace + hook persistence.
- **E2:** Convoy/Bead lifecycle + handoff.
- **E3:** CLI flows + dashboard endpoints.

### Epic F — Moltbot Parity (Consumer Profile)
- **F1:** Gateway protocol adapter.
- **F2:** Device node pairing + registry.
- **F3:** Canvas + voice + browser control compatibility.
- **F4:** Consumer channel expansion.

### Epic G — Governance + Safety
- **G1:** Unified RBAC across gateway/device/computer-use.
- **G2:** Approval workflow standardization.
- **G3:** Audit trail end-to-end.

---

## Risks & Dependencies

- **SDKs are incomplete:** Do not auto-regenerate from OpenAPI until endpoints stabilize.
- **Context scale:** RLM REPL must handle large repo sizes safely (size caps + chunking).
- **Scope creep:** Keep parity layers opt-in and gated by policy.
