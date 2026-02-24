# Brain Dump to Execution Map (2026 Q2)

Last updated: 2026-02-24

Related:
- `docs/status/EXECUTION_PROGRAM_2026Q2_Q4.md`
- `docs/status/EXECUTION_BACKLOG_2026Q2.csv`
- `docs/CAPABILITY_MATRIX.md`
- `docs/connectors/STATUS.md`

## Best Order (Priority Sequence)

### P0: Stop the bleeding (Weeks 1-3)

1. Web app E2E reliability and auth flow correctness
2. Production deployment reliability and rollback discipline
3. Security automation closure (key rotation + secrets + SSRF verification)
4. Debate live experience baseline (real streaming + intervention)

### P1: Productize core differentiation (Weeks 4-8)

5. Oracle real-time streaming + live TTS playback
6. Multi-agent worktree isolation and hierarchical coordination
7. Frontend parity for backend capabilities
8. Surface and connect stranded features in UI workflows

### P2: Autonomy and control plane leverage (Weeks 9-12)

9. Autonomous self-improvement (self-directed next-step selection)
10. Idea-to-execution DAG with interactive and automated execution loops
11. Model lineup refresh and automated model drift management

### P3: Positioning and GTM articulation (continuous)

12. Differentiate visibly against single-agent sessions with measurable trust, memory, and adversarial evidence

## Translation of the 12 Ideas

## 1) Make aragora.ai actually work end-to-end

Outcome:
- No loading hangs, no login dead ends, no post-auth routing failures, no fatal React hydration/render crashes in critical flows.

Code anchors:
- `aragora/live/src/`
- `aragora/server/handlers/auth/`
- `aragora/auth/`
- `docs/api/API_REFERENCE.md` (`/api/v1/auth/*`, `/api/auth/oauth/*`)

Owner:
- `@team-growth` + `@team-platform`

Primary KPIs:
- Login-to-debate success >= 98%
- First meaningful page render p95 <= 2.5s
- Critical frontend error rate <= 0.2%

## 2) Oracle streaming with real-time TTS

Outcome:
- Oracle transitions from batch response to low-latency token stream with synchronized speech output.

Code anchors:
- `aragora/server/stream/`
- `aragora/server/stream/tts_integration.py`
- `aragora/server/stream/voice_stream.py`
- `aragora/live/src/store/debateStore.ts`

Owner:
- `@team-core` + `@team-growth`

Primary KPIs:
- Time-to-first-token <= 1.5s
- Time-to-first-audio <= 2.0s
- Oracle session completion >= 97%

## 3) Autonomous self-improvement that chooses goals

Outcome:
- System proposes and ranks its own next goals from telemetry and quality signals, then executes under policy constraints.

Code anchors:
- `scripts/nomic_loop.py`
- `aragora/nomic/autonomous_orchestrator.py`
- `aragora/nomic/meta_planner.py`
- `aragora/nomic/hierarchical_coordinator.py`

Owner:
- `@team-core`

Primary KPIs:
- >= 60% of improvement cycles are system-proposed
- >= 70% of accepted auto-goals improve tracked quality metrics within 2 cycles

## 4) Ideas-to-execution visual DAG

Outcome:
- Visual pipeline from idea clusters -> goals/principles -> task graph -> agent assignment -> execution and review.

Code anchors:
- `aragora/pipeline/`
- `aragora/workflow/`
- `aragora/live/src/app/(app)/`
- `docs/research/IDEA_TO_EXECUTION_PIPELINE.md`

Owner:
- `@team-growth` + `@team-core`

Primary KPIs:
- 80% of created idea graphs reach executable task state
- Median idea-to-first-executed-task <= 20 minutes

## 5) Strong differentiation vs single-agent sessions

Outcome:
- Productized evidence of adversarial vetting, calibrated trust, and institutional memory over time.

Code anchors:
- `aragora/debate/`
- `aragora/ranking/elo.py`
- `aragora/memory/`
- `aragora/export/decision_receipt.py`

Owner:
- `@team-analytics` + `@team-growth`

Primary KPIs:
- >= 90% decisions include measurable dissent + synthesis traces
- Trust calibration error decreases >= 20% over rolling 30 days

## 6) Surface stranded features

Outcome:
- Existing capabilities become discoverable and integrated in coherent user flows.

Code anchors:
- `docs/FEATURE_DISCOVERY.md`
- `docs/status/FEATURES.md`
- `aragora/live/src/`
- `docs/CAPABILITY_MATRIX.md`

Owner:
- `@team-growth` + `@team-platform`

Primary KPIs:
- Feature activation breadth doubles for top 20 dormant modules
- UI capability coverage 14.3% -> >= 35%

## 7) Production deployment reliability

Outcome:
- Deterministic deployment pipeline with health-gated rollout, instant rollback, and config correctness.

Code anchors:
- `.github/workflows/deploy-secure.yml`
- `.github/workflows/deploy-frontend.yml`
- `docs/ops/RUNBOOK.md`
- `deploy/`

Owner:
- `@team-sre`

Primary KPIs:
- Deployment success >= 99%
- Mean time to restore (MTTR) <= 15 minutes
- Drift between deployed SHA and expected SHA = 0

## 8) Frontend page coverage for backend feature set

Outcome:
- Backend-first capabilities (intelligence, pipeline canvas, knowledge flow, performance) are represented with usable UI surfaces.

Code anchors:
- `aragora/live/src/app/(app)/`
- `aragora/server/handlers/`
- `docs/CAPABILITY_MATRIX.md`

Owner:
- `@team-growth`

Primary KPIs:
- UI coverage >= 50% of mapped capabilities by end of phase
- >= 70% of top API domains have at least one first-party UI page

## 9) Multi-agent worktree coordination and recovery

Outcome:
- High-throughput parallel execution with isolated worktrees, hierarchy-based task dispatch, auto-restart, and safe merge/reconcile.

Code anchors:
- `scripts/setup_worktrees.sh`
- `scripts/merge_worktrees.sh`
- `aragora/nomic/branch_coordinator.py`
- `docs/plans/SELF_IMPROVING_ARAGORA.md`
- `docs/plans/NOMIC_CORE_INTEGRATION.md`

Owner:
- `@team-core` + `@team-platform`

Primary KPIs:
- Parallel-session efficiency >= 60% (from ~15-20%)
- Stalled session auto-recovery success >= 90%
- Merge conflict abort rate <= 10%

## 10) Debate experience improvement

Outcome:
- Debates feel truly live with token streaming, reasoning visibility, and explicit user interventions.

Code anchors:
- `aragora/server/stream/`
- `aragora/spectate/`
- `aragora/live/src/components/debate/`
- `docs/api/API_REFERENCE.md` (intervene endpoints)

Owner:
- `@team-growth` + `@team-core`

Primary KPIs:
- Time-to-first-token <= 1.0s in standard debate view
- Intervention action success >= 98%
- Session engagement time +25%

## 11) Update model lineup and provider defaults

Outcome:
- Current default models and mappings are refreshed and governed with compatibility tests and rollback policy.

Code anchors:
- `AGENTS.md`
- `aragora/config/settings.py`
- `aragora/agents/model_selector.py`
- `aragora/agents/api_agents/openrouter.py`

Owner:
- `@team-sdk` + `@team-core`

Primary KPIs:
- Provider compatibility pass rate 100% on supported lineup
- Model fallback correctness >= 99%

## 12) Security hardening automation

Outcome:
- Fully automated key rotation posture with AWS Secrets Manager integration, rotation workflows, and enforced SSRF guards.

Code anchors:
- `aragora/security/token_rotation.py`
- `aragora/security/ssrf_protection.py`
- `scripts/rotate_keys.py`
- `docs/enterprise/SECRETS_MIGRATION.md`

Owner:
- `@team-risk` + `@team-sre`

Primary KPIs:
- 100% production secrets rotated within policy windows
- SSRF validation coverage for outbound URL fetch points = 100%
- Security audit HIGH/CRITICAL findings = 0

## Dependencies and Constraints

1. P0 reliability must be completed before major DAG/automation UX rollout.
2. Frontend parity depends on stable auth/session and deployment paths.
3. Autonomous self-improvement must remain policy-gated and audit-logged.
4. Worktree orchestration depends on branch hygiene and merge gate reliability.
5. Model lineup updates require parity checks across SDK, server, and docs.

## Immediate Sprint Slice (next 2 weeks)

1. Auth + routing E2E fixes for onboarding, OAuth callbacks, and post-login debate redirect.
2. Release pipeline hardening for frontend/backend deploy verification against expected SHA.
3. Oracle stream/TTS first-token-first-audio prototype behind feature flag.
4. Worktree coordinator default path audit and stalled-session watchdog.
5. Security rotation automation dry run with Secrets Manager and rotation telemetry.

