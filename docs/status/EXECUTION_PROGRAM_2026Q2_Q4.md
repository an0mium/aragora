# Execution Program 2026 Q2-Q4

Last updated: 2026-02-23

Related:
- `docs/status/NEXT_STEPS_CANONICAL.md`
- `docs/CAPABILITY_MATRIX.md`
- `docs/connectors/STATUS.md`
- `docs/GA_CHECKLIST.md`
- `ROADMAP.md`

## Program Intent

Build Aragora into a complete control plane for:
1. Decision integrity (debate + evidence + receipts + verification)
2. Multi-agent implementation of decisions (workflows + computer-use + policy gates)
3. Human interactivity across channels (chat/voice/email/web)
4. Institutional memory across time scales with RBAC and auditability
5. Production-scale reliability, deployment, and measurable business outcomes

## Source-of-Truth Snapshot

As of 2026-02-23 (updated 2026-02-24):
- Capability matrix coverage: API 100%, CLI 100%, Python SDK 100%, TypeScript SDK 100%, UI 100%, catalog mapped 100%
- All previously "missing" surfaces (consensus_detection, compliance_framework, rbac_v2) now implemented
- Connector maturity: 103 Production, 48 Beta, 0 Stub
- GA checklist: 58/59 complete, primary blocker listed as external penetration test
- Note: Surface parity gaps are CLOSED. Remaining work is hardening, promotion, and productization.

## Supported Features (Implemented)

### 1) Multi-agent decisioning core

- Arena orchestration and phased debate loop
- Weighted voting and consensus machinery
- Evidence grounding and claim linkage
- Nomic loop integration and checkpoints
- Workflow engine with sequential/parallel/conditional execution

Primary evidence:
- `aragora/debate/orchestrator.py`
- `aragora/debate/voting_engine.py`
- `aragora/reasoning/evidence_grounding.py`
- `aragora/nomic/integration.py`
- `aragora/workflow/engine.py`

### 2) Agent fabric

- Registry-backed agent creation and validation
- 42+ agent types across CLI/API/local/OpenRouter/external frameworks
- Persona and calibration plumbing

Primary evidence:
- `aragora/agents/registry.py`
- `aragora/agents/base.py`
- `aragora/agents/cli_agents.py`
- `AGENTS.md`

### 3) Memory and institutional learning

- 4-tier memory (fast/medium/slow/glacial)
- Continuum memory and retention gate
- Memory gateway for cross-system writes and retrieval
- ELO + calibration signals for agent reliability

Primary evidence:
- `aragora/memory/tier_manager.py`
- `aragora/memory/continuum/core.py`
- `aragora/memory/retention_gate.py`
- `aragora/memory/gateway.py`
- `aragora/ranking/elo.py`

### 4) Knowledge management

- Knowledge mound core, search, ingestion, quality, staleness, and operations modules
- Adapter ecosystem and bridge surfaces

Primary evidence:
- `aragora/knowledge/mound_core.py`
- `aragora/knowledge/query_engine.py`
- `aragora/knowledge/mound/ops/`
- `aragora/knowledge/README.md`

### 5) Decision integrity and auditability

- Decision receipt generation and export
- Verification and compliance artifact pathways

Primary evidence:
- `aragora/export/decision_receipt.py`
- `docs/integration/decision-receipts.md`

### 6) API, streaming, and SDK surfaces

- OpenAPI generation and validation pipeline
- REST + WebSocket server stack
- Python + TypeScript SDKs with parity checking

Primary evidence:
- `scripts/generate_openapi.py`
- `scripts/validate_openapi_routes.py`
- `tests/server/openapi/test_contract_matrix.py`
- `sdk/python/README.md`
- `sdk/typescript/README.md`

### 7) Security, policy, tenancy, and compliance

- RBAC decorators and middleware
- Policy engine (allow/deny/escalate/budget)
- Tenant isolation and workspace controls
- Privacy deletion/retention/audit support
- EU AI Act artifact generation paths
- Secret rotation and encrypted secret handling

Primary evidence:
- `aragora/rbac/decorators.py`
- `aragora/policy/engine.py`
- `aragora/tenancy/isolation.py`
- `aragora/privacy/deletion.py`
- `aragora/compliance/eu_ai_act.py`
- `aragora/security/token_rotation.py`

### 8) Computer-use and MCP

- Computer-use API handler and persistent task/policy storage
- CLI command surface for computer-use operations
- MCP server with tool metadata and runnable tool endpoints

Primary evidence:
- `aragora/server/handlers/computer_use_handler.py`
- `aragora/computer_use/storage.py`
- `aragora/cli/commands/computer_use.py`
- `aragora/mcp/server.py`
- `aragora/mcp/tools.py`

### 9) Channel and connector ecosystem

- Production channel connectors include Slack, Telegram, WhatsApp, Discord, Teams, Google Chat, Signal, iMessage
- Enterprise connectors across collaboration, CRM, documents, databases, streaming, healthcare, and ITSM

Primary evidence:
- `docs/connectors/STATUS.md`
- `aragora/connectors/registry.py`

### 10) Deployment, observability, and resilience

- Production docker compose stack
- Kubernetes deployment and monitoring manifests
- Prometheus/Grafana/Alertmanager profile wiring
- DR and backup runbooks and jobs

Primary evidence:
- `deploy/docker-compose.production.yml`
- `docs/deployment/KUBERNETES.md`
- `docs/observability/OBSERVABILITY.md`
- `docs/deployment/DISASTER_RECOVERY.md`

### 11) Product interfaces

- CLI: `ask`, `quickstart`, `gauntlet`, `review`, `serve`
- Live web surfaces with debate templates and dashboard routes

Primary evidence:
- `aragora/cli/commands/debate.py`
- `aragora/cli/commands/quickstart.py`
- `aragora/cli/gauntlet.py`
- `aragora/cli/review.py`
- `aragora/cli/commands/server.py`
- `aragora/live/src/components/LandingPage.tsx`

## Partially Supported and Missing Surfaces

### 1) Explicit capability matrix gaps

From `docs/CAPABILITY_MATRIX.md`:
- Missing API: `consensus_detection`
- Missing CLI: `compliance_framework`, `consensus_detection`, `rbac_v2`
- Missing SDK: `consensus_detection`
- Missing UI: `compliance_framework`, `consensus_detection`, `continuum_memory`, `debate_orchestration`, `decision_integrity`, `graph_debates`, `knowledge_mound`, `marketplace`, `matrix_debates`, `nomic_loop`, `rbac_v2`, `workflow_engine`
- Missing CHANNELS: `agent_team_selection`, `compliance_framework`, `consensus_detection`, `continuum_memory`, `graph_debates`, `knowledge_mound`, `marketplace`, `matrix_debates`, `nomic_loop`, `rbac_v2`, `vertical_specialists`, `workflow_engine`

### 2) Connector maturity debt

From `docs/connectors/STATUS.md`:
- Production: 103
- Beta: 48
- Stub: 0

All 4 former stub connectors have been promoted to Production:
- `aragora/connectors/communication/sendgrid.py` -- email activity search, templates, query sanitization
- `aragora/connectors/communication/twilio.py` -- SMS/MMS/call history, query sanitization
- `aragora/connectors/productivity/trello.py` -- card/board search via Trello API
- `aragora/connectors/social/instagram.py` -- media/comments via Graph API

Remaining maturity debt is concentrated in 48 Beta connectors that lack circuit breaker patterns and advanced retry logic.

### 3) Readiness and release-gating debt

- PR fast lanes exclude slow/load/e2e/integration classes
- ~~Release workflow does not depend on security/nightly suites~~ **RESOLVED**: release.yml now blocks on npm audit HIGH/CRITICAL
- ~~npm audit path is non-blocking for dependency checks~~ **RESOLVED**: two-pass audit (moderate non-blocking, high/critical blocking)
- ~~Typecheck strictness is concentrated in selected modules only~~ **PARTIALLY RESOLVED**: `frontend-typecheck` CI job added for PR path filtering
- Skip baseline zero-tolerance enforced (no more THRESHOLD=2 fallback)

Primary evidence:
- `.github/workflows/test.yml`
- `.github/workflows/release.yml`
- `.github/workflows/security.yml`
- `.github/workflows/lint.yml`
- `scripts/test_tiers.sh`

### 4) Frontend build and test health

- 170+ pages exist in `aragora/live/` but TypeScript strictness varies
- ~~No unified frontend test suite in CI~~ **PARTIALLY RESOLVED**: `frontend-typecheck` job runs `tsc --noEmit` on PRs touching `aragora/live/`
- Component tests exist for some features but coverage is unmeasured

Primary evidence:
- `aragora/live/tsconfig.json`
- `aragora/live/src/app/`

### 5) Developer documentation and onboarding

- Quickstart command exists (`aragora quickstart`) but end-to-end validation is manual
- SDK guides exist but no measured onboarding time target
- No automated doc freshness or link-rot checking

Primary evidence:
- `aragora/cli/commands/quickstart.py`
- `docs/SDK_GUIDE.md`
- `docs/EXTENDED_README.md`

### 6) Self-improvement loop production readiness

- Nomic Loop, MetaPlanner, BranchCoordinator, and TaskDecomposer all implemented
- Self-develop CLI works in dry-run but production cycle metrics not tracked
- Cross-cycle learning via KnowledgeMound exists but efficiency not measured

Primary evidence:
- `scripts/self_develop.py`
- `aragora/nomic/meta_planner.py`
- `aragora/nomic/calibration_monitor.py`

### 7) FastAPI migration breadth

- FastAPI surface exists but does not yet represent full legacy handler breadth

Primary evidence:
- `aragora/server/ARCHITECTURE.md`
- `aragora/server/fastapi/routes/`

## Dependency-Driven Roadmap

### Phase 0: Release Integrity and Truth Baseline (2026-02-24 to 2026-03-15)

Goal: make status and release confidence unambiguous.

Deliverables:
- Release workflow blocks on security + integration smoke + nightly evidence checks
- Skip marker governance tightened
- Capability/GA/roadmap status reconciliation automation
- External pentest execution started with tracked remediation plan

### Phase 1: Surface Parity Closure (2026-03-16 to 2026-04-15)

Goal: close matrix-gapped product surfaces.

Deliverables:
- `consensus_detection` available in API, SDK, CLI
- `compliance_framework` and `rbac_v2` CLI surfaces
- Decision-integrity UI workbench covering debate/workflow/memory/nomic visibility
- FastAPI migration wave for critical routes

### Phase 2: Channel Productization and FinOps (2026-04-16 to 2026-05-31)

Goal: productionize human-in-the-loop channels and spend controls.

Deliverables:
- Slack and Teams thread lifecycle with receipts and approvals
- Workspace budget policy engine
- Per-debate cost accounting in receipts
- Spend analytics dashboard
- Top beta connectors promoted (all stubs already removed pre-program)

### Phase 3: Scale and Reliability (2026-06-01 to 2026-07-31)

Goal: meet reliability and performance targets under load.

Deliverables:
- Debate latency reduction
- Streaming reliability hardening
- Load-testing SLO enforcement
- Offsite backup automation + restore drill evidence
- Public status page and SLA instrumentation

### Phase 4: Ecosystem and Analytics (2026-08-01 to 2026-10-31)

Goal: improve developer ecosystem and measurable value reporting.

Deliverables:
- Interactive API explorer
- SDK codegen pipeline hardening
- Example application pack
- Outcome analytics and knowledge gap detection
- Marketplace pilot

## Owner Model (Role-Based)

- `@team-platform`: release gates, CI, FastAPI migration, contract governance
- `@team-core`: debate/runtime/workflow/memory/decision integrity
- `@team-integrations`: channels/connectors/computer-use external systems
- `@team-finops`: budgets, cost accounting, spend analytics
- `@team-risk`: security, privacy, compliance, pentest closure, MFA
- `@team-sre`: deployment, reliability, observability, backups, SLOs
- `@team-analytics`: KPI data products, outcomes, insight surfaces
- `@team-sdk`: SDK parity, codegen, API explorer integration
- `@team-growth`: docs/examples/marketplace and adoption workflows

## KPI Set and Targets

### Program KPIs

1. `Release Gate Completeness`
- Target: 100% of tagged releases blocked by security + integration gates
- Data source: `.github/workflows/release.yml` run metadata

2. `Capability Coverage (CLI)`
- Target: 78.6% -> >= 90%
- Data source: `docs/CAPABILITY_MATRIX.md`

3. `Capability Coverage (UI)`
- Target: 14.3% -> >= 35%
- Data source: `docs/CAPABILITY_MATRIX.md`

4. `Connector Production Ratio`
- Baseline: Production 103, Beta 48, Stub 0
- Target: Production 103 -> >= 120, Beta 48 -> <= 25, Stub 0 (achieved)
- Data source: `docs/connectors/STATUS.md`

5. `Debate Runtime`
- Target: median runtime reduction >= 25%
- Data source: benchmark and load-test workflows

6. `Streaming Reliability`
- Target: stream error rate <= 0.5%
- Data source: stream tests and production metrics

7. `Budget Protection`
- Target: budget-overrun debates <= 3%
- Data source: billing and policy telemetry

8. `Receipt Completeness`
- Target: >= 95% debates produce verifiable receipts with cost + evidence sections
- Data source: receipt store/export verification jobs

9. `Re-debate from Staleness`
- Target: >= 70% stale claims trigger automatic re-debate workflow
- Data source: nomic + memory lifecycle metrics

10. `Admin MFA Enforcement`
- Target: 100% admin users MFA-enforced
- Data source: auth policy logs

11. `External Pentest Closure`
- Target: 0 open HIGH/CRITICAL findings
- Data source: pentest tracking artifacts

12. `Availability`
- Target: >= 99.9% monthly
- Data source: uptime/status monitoring

13. `Frontend Build Health`
- Target: 0 TypeScript build errors; frontend test coverage >= 40%
- Data source: `npx tsc --noEmit` in CI; Jest/Vitest coverage reports

14. `Developer Onboarding Time`
- Target: New developer can run first debate in <= 10 minutes
- Data source: Quickstart walkthrough validation runs; user testing feedback

15. `Self-Improvement Cycle Efficiency`
- Target: >= 60% of Nomic Loop cycles produce merged improvements; average cycle time <= 30 minutes
- Data source: Nomic Loop telemetry and branch coordinator metrics

## 30/60/90 Execution Plan

### Day 30 (2026-03-25)

Primary outcomes:
- Release integrity gates hardened
- Status reconciliation automated
- Pentest engagement active
- Frontend TypeScript build gate established in CI
- Doc link-rot checking active

Must-hit KPIs:
- Release Gate Completeness >= 90% for new tags
- Skip marker baseline reduced by at least 20
- Frontend Build Health: 0 TS errors on main

### Day 60 (2026-04-24)

Primary outcomes:
- Surface parity gaps largely closed for API/SDK/CLI
- Decision-integrity UI baseline shipped
- Developer onboarding validated and measured
- Frontend component test baseline established

Must-hit KPIs:
- CLI coverage >= 86%
- SDK missing `consensus_detection` gap closed
- Developer Onboarding Time measured and <= 15 min
- Frontend test coverage >= 20%

### Day 90 (2026-05-24)

Primary outcomes:
- Slack/Teams thread lifecycle productized
- FinOps controls live
- Connector maturity trend established

Must-hit KPIs:
- Budget Protection metric active and <= 5% overrun during rollout
- Stub connectors: 0 (achieved; all 4 promoted to Production pre-program)
- Channel workflow success >= 95% on pilot tenants

## Backlog Artifacts

- Milestones: `docs/status/EXECUTION_MILESTONES_2026Q2.csv`
- Issues (import-ready): `docs/status/EXECUTION_BACKLOG_2026Q2.csv`
- Brain-dump execution map: `docs/status/BRAIN_DUMP_EXECUTION_MAP_2026Q2.md`
- Brain-dump issues (import-ready): `docs/status/BRAIN_DUMP_BACKLOG_2026Q2.csv`
- Import helper: `scripts/import_execution_backlog.py`
