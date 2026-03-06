# Closed-Loop Backbone: Implementation Tickets

Last updated: 2026-03-06
Status: Issue-ready breakdown

This file translates `docs/plans/2026-03-06-closed-loop-backbone-2-week-plan.md`
into issue-sized implementation tickets.

## Ticket vs Issue

In practice:

1. a `ticket` is the work item
2. an `issue` is the tracker record for that work item in GitHub, Linear, or another system

So for this sprint, treat them as the same thing unless there is a reason to split one ticket into multiple tracker records.

## Recommended Order

1. Backbone contracts and adapters
2. Fail-closed spec validation
3. Debate-to-plan preservation
4. Execution artifact normalization
5. Verification and self-repair normalization
6. Receipt and outcome feedback normalization
7. Security overlay and taint propagation
8. Golden-path and dogfood proof runs

## Platform Contracts

### CLB-001: Add canonical backbone contract module

Owner:
Platform

Primary files:
- `aragora/pipeline/backbone_contracts.py`
- `tests/pipeline/test_backbone_contracts.py`

Acceptance:
1. canonical bundle types exist for intake, spec, receipt, and outcome feedback
2. adapters normalize Prompt Engine, Interrogation, receipt, and outcome types
3. missing execution-grade spec fields are surfaced explicitly

### CLB-002: Document current entrypoint-to-contract normalization

Owner:
Platform

Primary files:
- `docs/architecture/CLOSED_LOOP_BACKBONE.md`
- `docs/plans/2026-03-06-closed-loop-backbone-2-week-plan.md`

Acceptance:
1. Prompt Engine, Interrogation, IdeaCloud, Canvas, and Nomic entrypoints map to canonical bundles
2. known bypasses are called out explicitly

## Prompt and Spec Path

### CLB-003: Unify Prompt Engine and Interrogation on one execution-grade spec contract

Owner:
Backend

Primary files:
- `aragora/interrogation/engine.py`
- `aragora/server/handlers/interrogation/handler.py`
- `aragora/server/handlers/prompt_engine/handler.py`
- `aragora/pipeline/backbone_contracts.py`

Acceptance:
1. both paths emit a shared spec bundle shape
2. unanswered questions and missing execution fields are preserved

### CLB-004: Enforce fail-closed execution-grade validation

Owner:
Backend

Primary files:
- `aragora/prompt_engine/spec_validator.py`
- `aragora/pipeline/decision_plan/factory.py`
- `tests/prompt_engine/`
- `tests/pipeline/`

Acceptance:
1. missing constraints, acceptance criteria, verification material, rollback, or file scopes block automated lanes
2. manual lanes can surface warnings without silently auto-executing

## Deliberation and Planning

### CLB-005: Preserve dissent and quality verdict into planning

Owner:
Reasoning + Backend

Primary files:
- `aragora/debate/orchestrator.py`
- `aragora/debate/post_debate_coordinator.py`
- `aragora/pipeline/decision_integrity.py`
- `aragora/pipeline/decision_plan/factory.py`

Acceptance:
1. plan creation can consume structured dissent and quality outputs
2. automated planning halts on failed quality verdict

### CLB-006: Add canonical deliberation bundle adapter

Owner:
Platform

Primary files:
- `aragora/pipeline/backbone_contracts.py`
- `tests/pipeline/test_backbone_contracts.py`

Acceptance:
1. debate outputs can be normalized into a stable handoff artifact
2. provenance, unresolved risks, and diversity data are not lost

## Execution and Verification

### CLB-007: Normalize execution attempt artifacts

Owner:
Backend

Primary files:
- `aragora/pipeline/executor.py`
- `aragora/pipeline/execution_bridge.py`
- `aragora/server/handlers/pipeline/execute.py`
- `aragora/server/handlers/tasks/execution.py`

Acceptance:
1. execution attempt output has a stable shape for status, artifacts, diff, and policy decisions
2. plan execution and task execution use compatible shapes

### CLB-008: Make bug-fix loop first-class after verification failure

Owner:
Backend

Primary files:
- `aragora/pipeline/unified_orchestrator.py`
- `aragora/pipeline/verification_plan.py`
- `tests/pipeline/test_unified_orchestrator.py`

Acceptance:
1. failed verification routes into bounded self-repair
2. retest result is carried forward into receipt/outcome layers

## Receipt and Feedback

### CLB-009: Normalize receipt envelope across pipeline outcomes

Owner:
Backend + Compliance

Primary files:
- `aragora/pipeline/receipt_generator.py`
- `aragora/server/handlers/pipeline/receipts.py`
- `aragora/server/handlers/receipts.py`

Acceptance:
1. successful and blocked outcomes share a consistent receipt envelope
2. policy result, taint summary, and provenance are always available

### CLB-010: Normalize outcome feedback for Nomic reuse

Owner:
Platform + Nomic

Primary files:
- `aragora/pipeline/outcome_feedback.py`
- `aragora/pipeline/meta_loop.py`
- `aragora/server/handlers/self_improve.py`

Acceptance:
1. outcome feedback records carry receipt reference and next-action guidance
2. Nomic can reprioritize from the same record used by product flows

## Security Overlay

### CLB-011: Introduce canonical trust-tier and taint propagation fields

Owner:
Security + Platform

Primary files:
- `aragora/pipeline/backbone_contracts.py`
- intake-related handlers and adapters
- receipt generation layers

Acceptance:
1. trust tier is attached at intake
2. taint can propagate into spec, deliberation, and receipt layers

### CLB-012: Add external-verifier insertion point for high-impact actions

Owner:
Security + Control Plane

Primary files:
- `aragora/control_plane/policy.py`
- receipt/policy handlers
- execution handlers

Acceptance:
1. high-impact actions can require independent external verification before final promotion
2. policy result is carried into the receipt envelope

## Proof

### CLB-013: Add canonical golden-path test

Owner:
QA + Platform

Primary files:
- `tests/pipeline/`
- relevant handler/orchestrator tests

Acceptance:
1. one test covers intake -> spec -> debate -> plan -> execute -> verify -> receipt
2. failures identify the broken contract stage

### CLB-014: Add closed-loop dogfood profile

Owner:
QA + Reasoning

Primary files:
- dogfood scripts and docs under `docs/plans/`

Acceptance:
1. one dogfood run emits machine-readable artifacts for each canonical stage
2. the run fails if required stage artifacts are absent
