# Test-Only File Triage

**Date:** 2026-02-12
**Scope:** Top 20 largest Python files that have tests but zero production imports.
**Method:** For each file, checked: (1) absolute imports across `aragora/`, (2) relative imports in sibling modules, (3) `__init__.py` re-exports, (4) CLI/script usage, (5) handler registry wiring, (6) dynamic imports.

## Summary

| Category  | Count | Lines   | Action |
|-----------|-------|---------|--------|
| WIRE_IN   | 10    | 10,002  | Import from production code paths |
| KEEP      | 5     | 5,022   | Migration/optional/standalone tooling |
| DELETE    | 5     | 4,938   | Dead code, remove |
| **Total** | **20**| **19,962** | |

---

## WIRE_IN (Should be imported by production code)

These files provide real functionality that is simply missing its wiring point.

### 1. `aragora/storage/ar_invoice_store.py` (1,413 lines, 1 test file)

**Purpose:** AR invoice storage backends (InMemory, SQLite, PostgreSQL) for accounts receivable.
**Gap:** No server handler, CLI command, or service imports this store. The companion `invoice_processor.py` (which IS wired to the invoices handler) does not use it either.
**Recommendation:** Wire into `aragora/server/handlers/invoices.py` or a dedicated AR handler. Without a consumer, the store pattern (factory, backends, migrations) is orphaned.

### 2. `aragora/storage/invoice_store.py` (1,256 lines, 4 test files)

**Purpose:** AP invoice storage backends. Similar pattern to `ar_invoice_store.py`.
**Gap:** Despite `invoice_processor.py` being wired into `server/handlers/invoices.py`, the processor never imports this store. Data created by the processor has no persistence path.
**Recommendation:** Wire `get_invoice_store()` into `InvoiceProcessor` or the invoices handler for persistence.

### 3. `aragora/debate/forking.py` (929 lines, 16 test files)

**Purpose:** Debate forking -- spawn parallel debate branches when agents fundamentally disagree, compare outcomes, merge results.
**Gap:** Core debate feature with extensive tests but not imported by `orchestrator.py`, `arena_phases.py`, or any handler. This is a significant capability that should be accessible.
**Recommendation:** Wire into the Arena via an `enable_forking` protocol flag or expose via `/api/v2/debates/:id/fork` endpoint.

### 4. `aragora/knowledge/vector_store.py` (815 lines, 8 test files)

**Purpose:** Weaviate-backed vector storage for KnowledgeNodes with semantic search, workspace scoping, graph filtering.
**Gap:** Only self-referenced in docstring. The KnowledgeMound uses its own stores but never delegates to this vector adapter.
**Recommendation:** Wire as optional backend in `KnowledgeMoundCore` for semantic search queries, or expose via knowledge query engine.

### 5. `aragora/rbac/ownership.py` (797 lines, 24 test files)

**Purpose:** Resource ownership tracking with implicit owner permissions, ownership transfer, audit trail. Supplements explicit RBAC grants.
**Gap:** Never imported by `PermissionChecker`, middleware, or any handler. The 24 test files indicate significant investment.
**Recommendation:** Wire `OwnershipManager` into `PermissionChecker.check_permission()` as a fallback before denying access.

### 6. `aragora/rbac/emergency.py` (784 lines, 8 test files)

**Purpose:** Break-glass emergency access -- time-limited elevated permissions with extended audit logging and post-incident review.
**Gap:** Not imported by RBAC middleware, auth handlers, or any endpoint. Being investigated by another agent (Task #2).
**Recommendation:** Wire into RBAC middleware as an override path (check emergency grants before denying). Expose via `/api/v2/admin/emergency-access` endpoint.

### 7. `aragora/privacy/anonymization.py` (791 lines, 5 test files)

**Purpose:** HIPAA Safe Harbor de-identification (18 identifiers), k-anonymity, differential privacy. Critical for healthcare vertical.
**Gap:** Not imported by any handler, compliance module, or export path.
**Recommendation:** Wire into privacy handlers and the GDPR/HIPAA compliance pipeline. Essential for healthcare_hipaa vertical weight profile.

### 8. `aragora/config/feature_flags.py` (760 lines, 11 test files)

**Purpose:** Centralized feature flag registry with validation, usage tracking, env override, hierarchical resolution.
**Gap:** Not imported anywhere. Feature flags are scattered across ArenaConfig, TenantConfig, and env vars.
**Recommendation:** Wire as the canonical feature flag source. Replace ad-hoc `os.environ.get("ARAGORA_FEATURE_*")` calls with `FeatureFlagRegistry.is_enabled()`.

### 9. `aragora/debate/voting_engine.py` (745 lines, 2 test files)

**Purpose:** Unified voting engine consolidating vote collection, semantic grouping, weighted counting, and consensus calculation from 3 separate modules.
**Gap:** Not imported by any phase or orchestrator. The existing `voting.py` and `vote_aggregator.py` are still used directly.
**Recommendation:** Wire as replacement for the fragmented voting modules. This is a consolidation module waiting to be activated.

### 10. `aragora/nomic/global_work_queue.py` (806 lines, 1 test file)

**Purpose:** Unified priority queue merging bead/convoy work items with dynamic reprioritization.
**Gap:** Not imported by `autonomous_orchestrator.py`, `branch_coordinator.py`, or any nomic module.
**Recommendation:** Wire into the nomic loop as the work scheduling layer between `TaskDecomposer` output and `BranchCoordinator` execution.

---

## KEEP (Migration/optional/standalone tooling)

These files serve a specific purpose that doesn't require regular production imports.

### 11. `aragora/migrations/sqlite_to_postgres.py` (1,123 lines, 2 test files)

**Purpose:** Full lifecycle SQLite-to-PostgreSQL migration orchestrator (discovery, schema translation, data migration, verification, rollback). Runnable as CLI: `python -m aragora.migrations.sqlite_to_postgres`.
**Rationale:** Migration script. Deliberately standalone -- invoked via `__main__` or CLI, not imported at runtime.
**Status:** KEEP as-is.

### 12. `aragora/migrations/v20260113000000_consolidate_databases.py` (841 lines, 1 test file)

**Purpose:** Database consolidation migration (20+ legacy SQLite DBs to 4 consolidated DBs). Versioned migration with CLI entry point.
**Rationale:** One-time migration script. Deliberately standalone.
**Status:** KEEP as-is.

### 13. `aragora/server/stream/debate_stream_server.py` (1,341 lines, 4 test files)

**Purpose:** WebSocket-only debate streaming server using the `websockets` library. Alternative to the full HTTP+WS unified server.
**Rationale:** Deliberately separate -- used for lightweight WS-only deployments. Standalone server, not imported by the main server.
**Status:** KEEP. Consider documenting as an alternative deployment option.

### 14. `aragora/audit/codebase_auditor.py` (816 lines, 2 test files)

**Purpose:** Codebase analysis for nomic loop pre-cycle auditing -- security scanning, documentation drift, improvement opportunities.
**Rationale:** Nomic loop utility, used programmatically but not as a library import. Invoked through the nomic loop phases.
**Status:** KEEP. Wire into `scripts/nomic_loop.py` if not already used there.

### 15. `aragora/knowledge/mound/consistency_validator.py` (875 lines, 1 test file)

**Purpose:** Unified consistency checking for Knowledge Mound -- referential integrity, contradiction detection, staleness, confidence decay.
**Rationale:** Maintenance utility. Should be available as on-demand validation, not always-on.
**Status:** KEEP. Consider wiring into a `/api/v2/knowledge/validate` admin endpoint.

---

## DELETE (Dead code)

These files provide no unique value or duplicate functionality that already exists elsewhere.

### 16. `aragora/storage/ar_invoice_store.py` -- *See WIRE_IN above; if no consumer is planned, DELETE.*

*Note: If there is no plan to build AR invoice functionality, both `ar_invoice_store.py` and `invoice_store.py` should be deleted as a pair (2,669 lines).*

### 16. `aragora/memory/postgres_continuum.py` (1,265 lines, 3 test files)

**Purpose:** PostgreSQL backend for ContinuumMemory.
**Gap:** The main `ContinuumMemory` in `aragora/memory/continuum/` uses its own SQLite backend and never references this PostgreSQL variant. The `postgres_store.py` in `knowledge/mound/` already handles PostgreSQL persistence for the knowledge layer.
**Rationale:** Duplicate persistence layer. ContinuumMemory already has its own scaling path via the knowledge mound postgres store.
**Status:** DELETE -- orphaned PostgreSQL backend with no integration path.

### 17. `aragora/memory/postgres_critique.py` (1,150 lines, 2 test files)

**Purpose:** PostgreSQL backend for CritiqueStore.
**Gap:** Like `postgres_continuum`, this is a parallel PostgreSQL backend that was never wired into the main CritiqueStore.
**Rationale:** Same pattern as above. The CritiqueStore works with SQLite and the KM adapter handles persistence for production.
**Status:** DELETE -- orphaned PostgreSQL backend.

### 18. `aragora/memory/postgres_consensus.py` (859 lines, 2 test files)

**Purpose:** PostgreSQL backend for ConsensusMemory.
**Gap:** Third instance of the same orphaned pattern.
**Rationale:** ConsensusMemory uses SQLite with KM adapter for production persistence.
**Status:** DELETE -- orphaned PostgreSQL backend.

### 19. `aragora/knowledge/postgres_fact_store.py` (886 lines, 1 test file)

**Purpose:** PostgreSQL fact store with tsvector full-text search.
**Gap:** Not imported by any knowledge module. The KnowledgeMound has its own PostgreSQL store.
**Rationale:** Superseded by `knowledge/mound/postgres_store.py`.
**Status:** DELETE -- superseded.

### 20. `aragora/agents/doc_generator.py` (1,038 lines, 1 test file)

**Purpose:** AI agent for generating documentation (docstrings, API docs, ADRs).
**Gap:** Not imported by any agent registry, handler, or CLI command. The agent system already has its own documentation capabilities through the standard agent pipeline.
**Rationale:** Standalone utility that was never integrated into the agent registry or any workflow.
**Status:** DELETE -- unintegrated utility agent.

---

## Additional Candidates (Rank 21-25, for reference)

| Rank | File | Lines | Tests | Category |
|------|------|-------|-------|----------|
| 21 | `billing/cost_governance.py` | 987 | 1 | WIRE_IN (cost policy engine, not connected to billing middleware) |
| 22 | `server/persistent_origin_store.py` | 779 | 1 | WIRE_IN (debate origin routing, not connected to debate_origin.py) |
| 23 | `gateway/openclaw_sandbox.py` | 757 | 2 | WIRE_IN (action sandbox, not connected to gateway) |
| 24 | `server/handlers/sme_success_dashboard.py` | 824 | 2 | FALSE POSITIVE (wired via handler_registry/admin.py) |
| 25 | `integrations/email_oauth.py` | 1,120 | 1 | WIRE_IN (credential store, not connected to email handlers) |

---

## Line Count Breakdown

**DELETE candidates:** 5,198 lines (postgres_continuum 1,265 + postgres_critique 1,150 + postgres_consensus 859 + postgres_fact_store 886 + doc_generator 1,038)

**WIRE_IN candidates:** 10,002 lines of functional code that should be connected to production paths.

**KEEP candidates:** 5,022 lines of migration/utility code that is correctly standalone.
