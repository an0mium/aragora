# Nomic “True Test” Wiring Spec

**Status:** Proposed  
**Owner:** Nomic/Platform  
**Date:** 2026-01-29  
**Goal:** Full-power multi-agent debate + deep codebase context + multi-agent implementation.

---

## Objectives

1. **Full‑power debate** by default (8–9 rounds, 8 frontier agents, judge consensus).
2. **Deep codebase understanding** using TRUE RLM (REPL) + indexed context.
3. **Multi‑agent implementation** using Gastown-style convoys/beads with cross‑checks.

---

## Current Gaps

- Nomic loop defaults to short debates (2 rounds, 5 agents).
- Context gathering is summary‑based and not a durable index.
- Implement phase falls back to single-agent execution when no executor is wired.

---

## Architecture (Target)

### 1) Debate Profile (8–9 rounds, 8 agents)

**Canonical component:** `aragora/nomic/debate_profile.py`

**Change required**
- Ensure Nomic loop uses `NomicDebateProfile` by default.
- Align `rounds` with `STRUCTURED_ROUND_PHASES` (0–8).

**Integration points**
- `scripts/nomic_loop.py` → use `NomicDebateProfile` for protocol + agent roster.
- `aragora/nomic/phases/debate.py` → default to profile config.

---

### 2) Deep Context Builder (TRUE RLM)

**Goal:** Replace summary-only context with indexed + queryable codebase context.

**Components**
- **Indexer:** Ingest repo into Knowledge Mound.
  - Source: `aragora/knowledge/mound/repository_orchestrator.py`
- **RLM Query:** Use TRUE RLM REPL for large context traversal.
  - Sources: `aragora/rlm/bridge.py`, `aragora/knowledge/mound/api/rlm.py`

**Proposal**
- Add a `ContextBuilder` that:
  1. Indexes the repo into Knowledge Mound.
  2. Builds a hierarchical map of modules/files.
  3. Uses TRUE RLM REPL for targeted retrieval (not full prompt stuffing).
- Ensure support for **10M-token equivalent** contexts by:
  - Chunked ingestion
  - REPL-based retrieval by path/module
  - Configurable `max_content_bytes` and disk-backed storage

**Integration points**
- `aragora/nomic/phases/context.py` → call `ContextBuilder` and inject output.
- `scripts/nomic_loop.py` → expose env toggles for TRUE RLM path.

---

### 3) Multi-Agent Implement Executor (Gastown-style)

**Goal:** Plan → Convoys → Beads → Multi-agent edits → Review → Verify → Commit

**Components**
- Convoys/Beads: `aragora/nomic/convoys.py`, `aragora/nomic/beads.py`
- Hook queue: `aragora/nomic/hook_queue.py`
- Coordinator: `aragora/nomic/convoy_coordinator.py`

**Proposal**
- Implement a `NomicImplementExecutor` that:
  1. Converts the design spec into beads.
  2. Creates a convoy per module/area.
  3. Assigns beads to multiple agents via HookQueue.
  4. Requires peer review + targeted tests before merge.

**Integration points**
- `aragora/nomic/phases/implement.py` → accept `implement_executor` hook.
- `scripts/nomic_loop.py` → wire executor into loop.

---

## Acceptance Criteria

1. **Debate:** Nomic runs 8–9 round debate with 8 agents by default.
2. **Context:** Debate context includes indexed codebase map + RLM-backed retrieval.
3. **Implementation:** Multi-agent executor produces merged changes with review + tests.
4. **Safety:** Approval gates remain enforced for commit.

---

## Rollout Plan

1. **Phase 1:** Wire `NomicDebateProfile` into Nomic loop defaults.
2. **Phase 2:** Add ContextBuilder + RLM toggles (behind env flags).
3. **Phase 3:** Implement multi-agent executor and enable with policy gates.

