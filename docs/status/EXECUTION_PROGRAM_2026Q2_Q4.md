# Execution Program 2026 Q2-Q4

Last updated: 2026-03-22

Related:
- `docs/status/NEXT_STEPS_CANONICAL.md`
- `docs/status/ACTIVE_EXECUTION_ISSUES.md`
- `docs/status/BRAIN_DUMP_EXECUTION_MAP_2026Q2.md`
- `docs/plans/ARAGORA_IDEA_TO_EXECUTION_STRATEGY.md`

## Program Intent

Build Aragora around the three truthful near-term wedges that now exist on `main`:
1. Receipt-gated decision-to-action on narrow recurring workflows
2. Bounded repo execution under supervisor/integrator control
3. A local-first idea-to-execution workbench that makes those loops legible and editable

The long-range thesis still stands. The program reset is about sequence: finish the real wedge first, then widen.

## Source-of-Truth Snapshot

As of 2026-03-22:
- `main` includes tranche/overnight hardening through `#1117`, then March 21-22 proof-surface closures through `#1138`
- `#1108` is still the first merged queue artifact proving the tranche system can recover and publish real work
- `#1110` merged the API-key / first-user-journey slice onto `main`
- `#1111`, `#1131`, `#1132`, and `#1134` merged KM retrieval, KM writeback, default enablement, and settlement-hook outcome wiring onto the canonical debate path
- `#1118` and `#1119` made receipts and integrations flows more truthful on the user-facing surface
- `#1124`, `#1126`, `#1127`, `#1133`, and `#1138` materially strengthened the execution operator contract: evidence survives detach, terminal state reconciles, lane view is authoritative, completed deliverables publish, and review inspects the remote PR head
- `#1135` made OpenClaw action dispatch real enough to count as part of the wedge
- `#1136` and `#1137` made the public proof surface and pipeline live-state UI materially more truthful

The near-term program should therefore not be described as "finish surface parity" or "close enterprise readiness." The real job is to stitch these merged slices into repeatable product loops.

## Product Wedge On `main`

### 1) Inbox Trust Wedge

- Narrow receipt-gated Gmail actions are shipped
- The system already proves "receipt before action" on a real recurring operational workflow
- This remains the best first design-partner loop because the trigger, approval path, and outcome are all concrete

### 2) Truthful Default Debate / Public Proof Surface

- The repo now has a merged API-key setup slice, KM-backed default debate recall, live receipts, and a truthful public proof surface
- The remaining gap is continuity: these slices still need to feel like one default path instead of several honest islands

### 3) Bounded Repo Execution

- Ralph V14 proved the bounded autonomous repo loop under explicit merge policy
- The tranche/supervisor/integrator stack has since gained truthful terminalization, better publish behavior, preserved review evidence, and authoritative live state
- The remaining gap is universal per-lane provenance/receipt discipline and operator-facing merge-readiness clarity

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
- 43 agent types across CLI/API/local/OpenRouter/external frameworks
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

## Remaining Near-Term Gaps

1. One end-to-end default journey still does not feel continuous.
   Pieces are merged, but the system still behaves like several honest islands instead of one obvious default loop.
2. The frontend remains shell-heavy outside the proven slices.
   Truthful surfaces improved, but page count still overstates usability.
3. No universal per-lane receipt/provenance contract exists across every swarm lane.
   The operator story is much stronger, but not yet universally canonical.
4. Design-partner PMF loops are not yet repeated enough.
   The wedge is real; the repeatability evidence is still thin.
5. The idea-to-execution workbench is still partial.
   Live state is landing, but unified stage-transition editing is not yet the default shell.

## Dependency-Driven Roadmap

### Phase 0: Truthfulness Baseline And Autonomy Proofs (through 2026-03-21) ✅ COMPLETE ENOUGH TO BUILD ON

Goal: establish truthful receipts, bounded autonomy proofs, and honest surface/state reporting.

Delivered:
- Ralph V14 benchmark
- Trust wedge core
- Tranche queue hardening through `#1117`
- First merged queue artifact (`#1108`)
- First merged product-loop slices for API-key setup, KM retrieval, live receipts, truthful integrations/public state, and operator/integrator visibility

### Phase 1: Active Wedge Closure (2026-03-22 to 2026-04-19) 🔄 CURRENT

Goal: make the merged proof slices feel like three repeatable product loops.

Deliverables:
- One truthful default path: credentials/provider routing -> debate -> KM-enriched context -> receipt -> visible result
- Truthful-by-default public and operator surfaces: `/demo`, integrations status/edit, receipts, pipeline live state
- Canonical bounded-lane operator contract: authoritative lane view, preserved evidence, remote-head review, completed-lane publish, explicit blocked next steps
- Inbox trust wedge kept as the first recurring partner workflow

### Phase 2: PMF Harvest And Partner Loop (2026-04-20 to 2026-05-31)

Goal: prove the wedge repeats weekly for real users.

Deliverables:
- 3-5 design partners each running one bounded workflow weekly
- OpenClaw used on one narrow real action-dispatch path
- Five functional frontend paths dogfooded continuously
- PMF scorecards and first case-study evidence attached to the merged wedges

### Phase 3: Unified Workbench And Learning Loop (2026-06-01 to 2026-07-31)

Goal: turn the truthful wedges into one local-first idea-to-execution shell.

Deliverables:
- Stage-transition review slices across ideas -> goals -> actions -> execution
- Execution outcomes revising upstream plans, not just emitting receipts
- Per-lane provenance visible in the workbench
- Multi-agent scale-out only after the operator/readiness contract is routine

### Phase 4: Ecosystem, FinOps, And Broader Enterprise Productization (2026-08-01 to 2026-10-31)

Goal: widen only after the wedge is repeatable.

Deliverables:
- Spend/budget surfaces that matter to active partner workflows
- Ecosystem and API explorer packaging
- Broader enterprise/compliance productization tied to real adoption, not speculative breadth

## Owner Model (Role-Based)

- `@team-platform`: release truth, queue/tranche, integrator/operator surfaces, provenance contracts
- `@team-core`: debate/runtime/KM/openclaw/settlement/default loop
- `@team-integrations`: inbox and other narrow action surfaces
- `@team-growth`: five functional frontend paths, public proof surface, workbench UX
- `@team-analytics`: PMF scorecards, proof metrics, case-study evidence
- `@team-risk`: keep assurance and compliance warm without outranking wedge closure
- `@team-sre`: reliability, deployment, and observability for the active wedges

## KPI Reset (Near-Term)

1. `Time To First Truthful Result`
- Target: <= 15 minutes on the guided path
- Data source: onboarding + receipt timing instrumentation

2. `Trust Wedge Receipt Before Action`
- Target: 100%
- Data source: trust wedge execution/receipt verification logs

3. `Default Debate KM Enrichment`
- Target: >= 80% of default debates retrieve relevant KM context
- Data source: debate context injection telemetry

4. `Core Surface Truthfulness`
- Target: 0 known optimistic/demo-only states on `/demo`, integrations status/edit, receipts, and pipeline live state
- Data source: dogfood checklists + incident/issues

5. `Bounded Lane Truthful Terminalization`
- Target: >= 95% of bounded runs end in deliverable or explicit blocked reason
- Data source: supervisor/tranche outcome telemetry

6. `Integrator Visibility`
- Target: 100% of publishable lanes have authoritative state plus review target
- Data source: integrator receipts and review metadata

7. `OpenClaw Narrow-Path Success`
- Target: >= 90% on the chosen real dispatch path
- Data source: computer-use/openclaw execution telemetry

8. `Design Partner Weekly Recurrence`
- Target: 3+ partners running one bounded workflow weekly by end of Phase 2
- Data source: PMF scorecards

## 30/60/90 Execution Plan

### Day 30 (2026-04-21)

Primary outcomes:
- Default product loop is dogfoodable end to end
- Truthful public/operator state exists on the key proof surfaces
- Queue/tranche operator contract is stable enough for repeated internal runs

Must-hit KPIs:
- Time To First Truthful Result <= 20 minutes
- Trust Wedge Receipt Before Action = 100%
- Core Surface Truthfulness incidents trending to 0

### Day 60 (2026-05-21)

Primary outcomes:
- Design-partner pilots running on the trust wedge, public proof, or swarm/review surfaces
- OpenClaw narrow-path dispatch validated
- Five functional frontend paths in weekly use

Must-hit KPIs:
- 3 active weekly partner workflows
- OpenClaw Narrow-Path Success >= 90%
- Bounded Lane Truthful Terminalization >= 95%

### Day 90 (2026-06-20)

Primary outcomes:
- Workbench begins to unify the active wedges
- Execution results feed back into planning/KM
- PMF decision uses measured repeatability rather than backlog volume

Must-hit KPIs:
- Default Debate KM Enrichment >= 80%
- Integrator Visibility = 100% on publishable lanes
- At least one stage transition is live, reviewable, and used in dogfood

## Backlog Artifacts

- Canonical short-horizon order: `docs/status/NEXT_STEPS_CANONICAL.md`
- Live GitHub backlog map: `docs/status/ACTIVE_EXECUTION_ISSUES.md`
- Near-term execution map: `docs/status/BRAIN_DUMP_EXECUTION_MAP_2026Q2.md`
- Legacy import artifacts remain available if needed:
  - `docs/status/EXECUTION_BACKLOG_2026Q2.csv`
  - `docs/status/EXECUTION_MILESTONES_2026Q2.csv`
  - `docs/status/BRAIN_DUMP_BACKLOG_2026Q2.csv`
