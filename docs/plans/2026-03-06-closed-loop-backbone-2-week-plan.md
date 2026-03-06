# Closed-Loop Backbone: 2-Week Execution Plan

Last updated: 2026-03-06
Status: Active implementation plan
Owner: Platform program (Backend, Frontend, QA, Security, Product)

This plan operationalizes the architecture in `docs/architecture/CLOSED_LOOP_BACKBONE.md`.
It complements, and does not supersede, `docs/status/NEXT_STEPS_CANONICAL.md`.
Issue-ready breakdown: `docs/plans/2026-03-06-closed-loop-backbone-implementation-tickets.md`

## Objective

Establish one canonical closed loop for Aragora:

`vague intent or idea intake -> interrogation/spec -> debate + quality gate -> plan -> controlled execution -> verify + self-repair -> signed receipt -> outcome feedback -> nomic reprioritization`

The point of this sprint is not adding more surfaces. The point is to make the existing surfaces compose into one reliable system.

## Constraints

1. Canonical reliability and security gates remain blocking:
   - test isolation
   - connector exception hygiene
   - offline/demo golden path
   - SDK/version alignment
   - self-host readiness
   - pentest closure gate
2. High-impact automation must remain receipt-gated.
3. New work should prefer wiring existing components over creating parallel orchestration paths.

## End-of-Sprint Definition of Done

By day 14, Aragora should have all of the following:

1. One documented canonical backbone with named handoff artifacts.
2. One fail-closed spec path from vague prompt to execution-grade task/spec.
3. One canonical orchestration path from debated decision to execution attempt.
4. One post-execution loop for verification, bug-fix, receipt generation, and outcome feedback.
5. One end-to-end golden path test from intake to receipt.
6. One dogfood benchmark profile that exercises the full loop and emits machine-readable outcomes.
7. One explicit security overlay for context taint, external verification policy, and pentest/red-team hooks.

## Non-Goals

1. New broad product surfaces.
2. Marketplace/federation/comms expansion.
3. Replacing working subsystems that only need wiring.
4. Treating self-improvement as a separate architecture from user-driven execution.

## Canonical Workstreams

### Workstream A: Freeze the Backbone and Entry Contracts

Goal: define the one backbone every serious flow must use.

Primary files:
- `docs/architecture/CLOSED_LOOP_BACKBONE.md`
- `aragora/pipeline/unified_orchestrator.py`
- `aragora/pipeline/decision_plan/core.py`
- `aragora/pipeline/outcome_feedback.py`
- `aragora/pipeline/receipt_generator.py`

Acceptance:
1. Every major entrypoint maps to one of the canonical handoff artifacts.
2. No stage is allowed to skip directly from prompt or idea intake to execution.
3. Nomic, IdeaCloud, Prompt Engine, and Canvas flows are described as producers/consumers of the same backbone, not alternate architectures.

### Workstream B: Fail-Closed Prompt and Spec Upgrading

Goal: underspecified prompts cannot flow into execution as soft prose.

Primary files:
- `aragora/interrogation/engine.py`
- `aragora/server/handlers/interrogation/handler.py`
- `aragora/server/handlers/prompt_engine/handler.py`
- `aragora/pipeline/decision_plan/factory.py`
- `aragora/prompt_engine/spec_validator.py`

Acceptance:
1. Execution-grade specs must include constraints, acceptance criteria, verification plan, rollback, and owner file scopes.
2. Missing required fields block downstream execution rather than degrading silently.
3. User-facing and self-improve flows emit the same spec shape.

### Workstream C: Canonical Deliberation to Plan Handoff

Goal: debate output becomes a durable decision package, not just a final answer string.

Primary files:
- `aragora/debate/orchestrator.py`
- `aragora/debate/post_debate_coordinator.py`
- `aragora/pipeline/decision_integrity.py`
- `aragora/pipeline/decision_plan/factory.py`
- `aragora/pipeline/unified_orchestrator.py`

Acceptance:
1. Debate output carries dissent, quality result, and provenance into planning.
2. Quality gate failure blocks planning in automated lanes.
3. Plan creation consumes structured debate artifacts rather than lossy text extraction where possible.

### Workstream D: Execution, Verification, and Self-Repair

Goal: execution is observable, retryable, and policy-bounded.

Primary files:
- `aragora/pipeline/executor.py`
- `aragora/pipeline/execution_bridge.py`
- `aragora/pipeline/unified_orchestrator.py`
- `aragora/server/handlers/pipeline/execute.py`
- `aragora/server/handlers/tasks/execution.py`

Acceptance:
1. The canonical path emits execution attempt artifacts with status, diff, and verification payloads.
2. Bug-fix logic runs after failed verification, not as an unrelated sidecar.
3. High-impact execution still requires policy and receipt validation.

### Workstream E: Receipt, Outcome Feedback, and Nomic Reuse

Goal: receipts become the control primitive and outcomes become reusable learning data.

Primary files:
- `aragora/pipeline/receipt_generator.py`
- `aragora/pipeline/outcome_feedback.py`
- `aragora/server/handlers/pipeline/receipts.py`
- `aragora/server/handlers/receipts.py`
- `aragora/server/handlers/self_improve.py`
- `aragora/pipeline/meta_loop.py`

Acceptance:
1. Every canonical run produces a receipt or an explicit blocked/no-receipt outcome.
2. Outcome feedback writes enough structure for Nomic reprioritization.
3. Self-improve consumes the same outcome/receipt data model as user-driven flows.

### Workstream F: Security Overlay and Release Gate

Goal: the backbone is explicit about AI-native attacks and release risk.

Primary files:
- `docs/plans/2026-03-05-ai-attack-vector-resistance-design.md`
- `docs/security/THREAT_MODEL.md`
- `scripts/check_pentest_findings.py`
- `.github/workflows/security.yml`
- `aragora/control_plane/policy.py`

Acceptance:
1. Context taint, external verification policy, and pentest findings are visible in the backbone.
2. High-impact automated decisions have an explicit external-verifier insertion point.
3. External pentest remains a release gate, not a documentation note.

## Day-by-Day Plan

### Days 1-2: Backbone Freeze

1. Publish target architecture and handoff artifacts.
2. Identify which current entrypoints normalize to `IntakeBundle` and which already start at `SpecBundle`.
3. Mark any bypasses that go prompt -> execute or idea -> execute without canonical handoff artifacts.

Deliverables:
- `docs/architecture/CLOSED_LOOP_BACKBONE.md`
- implementation issue list for bypasses

### Days 3-4: Fail-Closed Spec Path

1. Align interrogation and prompt-engine outputs on one execution-grade spec schema.
2. Enforce required fields in validator/factory layers.
3. Ensure self-improve goals can enter the same spec path.

Deliverables:
- one canonical spec contract
- tests for pass/fail completeness

### Days 5-6: Deliberation Package to Plan

1. Carry dissent, quality score, and provenance into planning.
2. Ensure automated lanes stop on failed quality gate.
3. Remove any lossy translation that drops high-value debate metadata before planning.

Deliverables:
- structured debate-to-plan contract
- regression tests for dissent/quality preservation

### Days 7-8: Execution and Self-Repair

1. Standardize execution attempt artifact shape.
2. Run verification before receipt issuance.
3. Make bug-fix loop the first-class recovery path after failed verification.

Deliverables:
- execution attempt contract
- verify/fix/retest path in canonical orchestrator

### Days 9-10: Receipt, Feedback, Nomic Reuse

1. Standardize receipt envelope for successful and blocked outcomes.
2. Write outcome feedback in a way Nomic can consume directly.
3. Ensure self-improve and user flows converge on the same feedback/receipt primitives.

Deliverables:
- receipt envelope contract
- outcome feedback contract

### Days 11-12: Security Overlay

1. Add explicit context-taint and trust-tier handling to the architecture and implementation backlog.
2. Define the external-verifier insertion point for high-impact actions.
3. Verify pentest/red-team hooks are reflected in release gating.

Deliverables:
- threat-to-stage mapping
- implementation backlog for G2 -> G1 -> G4 -> G3

### Days 13-14: Proof Run

1. Run one end-to-end golden path test from intake to receipt.
2. Run one dogfood profile that exercises the closed loop.
3. Record blockers, missing artifacts, and bypasses as explicit defects.

Deliverables:
- green golden-path test
- dogfood artifact bundle
- closeout report with next sprint carryovers

## Canonical Entry Points To Normalize

These entry points should converge onto the backbone instead of staying as parallel systems:

1. Prompt Engine
   - `aragora/server/handlers/prompt_engine/handler.py`
2. Interrogation
   - `aragora/server/handlers/interrogation/handler.py`
3. Pipeline Canvas / universal graph
   - `aragora/server/handlers/canvas_pipeline.py`
   - `aragora/server/handlers/pipeline/*.py`
4. IdeaCloud export/promote flow
   - `aragora/ideacloud/adapters/pipeline_bridge.py`
   - `aragora/ideacloud/core.py`
5. Self-improve / Nomic
   - `aragora/server/handlers/self_improve.py`
   - `scripts/nomic_loop.py`

## Required Tests and Gates

The sprint is not complete without the following:

1. One E2E test covering:
   - intake
   - spec
   - debate
   - plan
   - execute
   - verify
   - receipt
2. One dogfood profile covering the same stages.
3. Canonical project gates remain green:
   - `scripts/check_connector_exception_handling.py`
   - `scripts/check_self_host_compose.py`
   - `scripts/check_pentest_findings.py`
   - `scripts/run_offline_golden_path.sh`

## Decision Rules During the Sprint

1. If a feature needs a parallel pipeline, stop and justify it against the closed-loop backbone.
2. If a flow cannot emit the canonical artifact for its stage, it is incomplete.
3. If a high-impact path cannot produce a valid receipt, it cannot auto-execute.
4. If the sprint ends with multiple orchestration paths still bypassing the backbone, the sprint is not complete even if individual features improved.
