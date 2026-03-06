# Closed-Loop Backbone Architecture

Last updated: 2026-03-06
Status: Target architecture

This document defines the canonical backbone for Aragora's product and self-improvement flows.
It complements:

1. `docs/CANONICAL_GOALS.md`
2. `docs/plans/ARAGORA_EVOLUTION_ROADMAP.md`
3. `docs/status/NEXT_STEPS_CANONICAL.md`
4. `docs/plans/2026-03-06-closed-loop-backbone-2-week-plan.md`

## Purpose

Aragora already contains most of the needed subsystems:

- IdeaCloud and Pulse for intake
- interrogation and prompt engine for vague-prompt upgrading
- debate and quality gating for adversarial validation
- plan/dependency/workflow layers for execution planning
- execution bridges and handlers for action
- receipts, compliance artifacts, and policy layers for control
- outcome feedback and Nomic for self-improvement

The problem is not missing pieces. The problem is that these pieces can still behave like parallel stacks.

This document fixes that by declaring one canonical backbone.

## The Canonical Backbone

```mermaid
flowchart LR
    A["Intake Sources"] --> B["IntakeBundle"]
    B --> C["Interrogation / Prompt Upgrade"]
    C --> D["SpecBundle"]
    D --> E["Debate + Quality Gate"]
    E --> F["DeliberationBundle"]
    F --> G["DecisionPlanBundle"]
    G --> H["Controlled Execution"]
    H --> I["ExecutionBundle"]
    I --> J["Verify + Bug-Fix"]
    J --> K["VerificationBundle"]
    K --> L["Receipt + Policy Gate"]
    L --> M["ReceiptEnvelope"]
    M --> N["OutcomeFeedbackRecord"]
    N --> O["Nomic Reprioritization"]
    O -. "new goals / retries" .-> B
```

## First Principle

One system, one backbone.

These are not separate architectures:

1. user prompt to pipeline
2. IdeaCloud to pipeline
3. self-improve / Nomic
4. debate-to-execution

They are all entrypoints into the same backbone at different stages.

## Architectural Rules

1. All serious automation must pass through named handoff artifacts.
2. Execution cannot be triggered directly from raw prompts or raw idea graphs.
3. Receipts are the control primitive for high-impact execution.
4. Dissent, provenance, and taint must survive stage transitions.
5. Nomic uses the same backbone as product flows; it is not exempt from the control plane.

## Canonical Stages

### Stage 0: Intake and Context Assembly

Purpose:
normalize all upstream inputs into one intake artifact.

Canonical producers:

1. Prompt Engine and interrogation handlers
2. Pipeline canvas
3. IdeaCloud export and cluster promotion
4. Pulse / KM / Obsidian retrieval
5. self-improve goal creation

Current anchors:

- `aragora/interrogation/engine.py`
- `aragora/server/handlers/interrogation/handler.py`
- `aragora/server/handlers/prompt_engine/handler.py`
- `aragora/ideacloud/adapters/pipeline_bridge.py`
- `aragora/pipeline/input_extension.py`

Canonical artifact:
`IntakeBundle`

Required fields:

1. `source_kind`
2. `raw_intent`
3. `context_refs`
4. `trust_tiers`
5. `origin_metadata`
6. `taint_flags`

Boundary:
intake may collect and enrich context, but it does not authorize execution.

### Stage 1: Interrogation and Specification

Purpose:
turn vague intent into an execution-grade spec.

Current anchors:

- `aragora/interrogation/engine.py`
- `aragora/prompt_engine/*`
- `aragora/pipeline/decision_plan/factory.py`

Canonical artifact:
`SpecBundle`

Required fields:

1. `problem_statement`
2. `objectives`
3. `constraints`
4. `acceptance_criteria`
5. `verification_plan`
6. `rollback_plan`
7. `owner_file_scopes`
8. `open_questions`

Boundary:
specification may block or request clarification; it does not mutate the world.

### Stage 2: Deliberation and Quality Validation

Purpose:
stress-test the spec and intended action using heterogeneous debate.

Current anchors:

- `aragora/debate/orchestrator.py`
- `aragora/debate/post_debate_coordinator.py`
- `aragora/pipeline/decision_integrity.py`
- `aragora/pipeline/unified_orchestrator.py`

Canonical artifact:
`DeliberationBundle`

Required fields:

1. `proposal_summary`
2. `dissent_ledger`
3. `quality_verdict`
4. `quality_scores`
5. `provider_diversity_report`
6. `provenance_refs`
7. `unresolved_risks`

Boundary:
debate produces a decision package, not direct execution.

### Stage 3: Planning

Purpose:
map validated intent into executable tasks, dependencies, gates, and approvals.

Current anchors:

- `aragora/pipeline/decision_plan/core.py`
- `aragora/pipeline/decision_plan/factory.py`
- `aragora/pipeline/verification_plan.py`
- `aragora/pipeline/dag_operations.py`

Canonical artifact:
`DecisionPlanBundle`

Required fields:

1. `steps`
2. `dependencies`
3. `execution_mode`
4. `approval_requirements`
5. `verification_targets`
6. `rollback_targets`
7. `budget_and_policy_constraints`

Boundary:
planning can schedule work, but policy and receipt gates still govern execution.

### Stage 4: Controlled Execution

Purpose:
run the plan through approved execution bridges and track concrete attempts.

Current anchors:

- `aragora/pipeline/executor.py`
- `aragora/pipeline/execution_bridge.py`
- `aragora/server/handlers/pipeline/execute.py`
- `aragora/server/handlers/tasks/execution.py`
- `aragora/server/handlers/workflows/execution.py`

Canonical artifact:
`ExecutionBundle`

Required fields:

1. `attempts`
2. `artifacts`
3. `diffs`
4. `logs`
5. `policy_decisions`
6. `execution_status`

Boundary:
execution performs work, but it does not decide for itself whether the result is acceptable.

### Stage 5: Verification and Self-Repair

Purpose:
evaluate outcomes, retry safe repairs, and determine whether the result is promotable.

Current anchors:

- `aragora/pipeline/unified_orchestrator.py`
- `aragora/pipeline/outcome_feedback.py`
- `aragora/pipeline/verification_plan.py`
- bug-fix integrations reachable through `apply_fix_and_retest`

Canonical artifact:
`VerificationBundle`

Required fields:

1. `checks_run`
2. `check_results`
3. `bug_fix_attempts`
4. `final_verification_status`
5. `remaining_failures`

Boundary:
self-repair may improve a failing execution attempt, but it does not rewrite the original intent or silently bypass policy.

### Stage 6: Receipt and Policy Gate

Purpose:
convert a verified or blocked outcome into a control-plane artifact.

Current anchors:

- `aragora/pipeline/receipt_generator.py`
- `aragora/server/handlers/pipeline/receipts.py`
- `aragora/server/handlers/receipts.py`
- `aragora/control_plane/policy.py`
- gauntlet receipt handlers

Canonical artifact:
`ReceiptEnvelope`

Required fields:

1. `receipt_id`
2. `artifact_hash`
3. `signature`
4. `verdict`
5. `confidence`
6. `dissent`
7. `taint_summary`
8. `policy_gate_result`
9. `provenance_chain`

Boundary:
the receipt is not just an audit artifact. For high-impact automation it is the execution-control artifact.

### Stage 7: Outcome Feedback and Settlement

Purpose:
record what happened, what failed, what settled, and what should influence future trust.

Current anchors:

- `aragora/pipeline/outcome_feedback.py`
- settlement and calibration subsystems
- compliance artifact generators

Canonical artifact:
`OutcomeFeedbackRecord`

Required fields:

1. `receipt_ref`
2. `objective_fidelity`
3. `quality_outcome`
4. `execution_outcome`
5. `settlement_hooks`
6. `calibration_updates`
7. `next_action_recommendation`

Boundary:
feedback is reusable learning data, not just logging.

### Stage 8: Nomic Reprioritization

Purpose:
convert outcomes into future improvement work.

Current anchors:

- `scripts/nomic_loop.py`
- `aragora/pipeline/meta_loop.py`
- `aragora/server/handlers/self_improve.py`
- `aragora/nomic/*`

Canonical behavior:

1. Nomic consumes `OutcomeFeedbackRecord` and `ReceiptEnvelope`.
2. New self-improvement goals re-enter the backbone through `IntakeBundle` or `SpecBundle`.
3. Nomic does not bypass receipt, verification, or policy stages for meaningful actions.

## Trust and Taint Model

The architecture needs explicit trust tiers because context injection is a first-class risk.

Canonical trust tiers:

1. operator-authored
2. signed trusted repo/config
3. internal retrieved knowledge
4. external retrieved content
5. model-generated content

Rules:

1. trust tier must be attached at intake time
2. taint must propagate across artifacts if lower-trust context materially shaped the output
3. receipts must expose taint summaries for human review and policy decisions

This is the architectural home for roadmap items:

1. G2: taint tracking
2. G1: signed context manifests
3. G4: mandatory external verification
4. G3: runtime model attestation

## Canonical Boundaries

### IdeaCloud

Role:
intake, clustering, promotion, and export.

Must not:
directly authorize execution or act as a parallel planning engine.

### Prompt Engine and Interrogation

Role:
upgrade vague intent into a spec.

Must not:
directly trigger execution without passing through deliberation, planning, and receipt logic.

### Debate and Gauntlet

Role:
produce adversarially validated decision packages.

Must not:
be treated as sufficient authority for execution on their own.

### Execution and Bug-Fix

Role:
run plans and repair failed attempts inside bounded policy.

Must not:
change objectives silently or bypass receipt/policy gates.

### Nomic

Role:
learn from outcomes and originate new improvement goals.

Must not:
be a side-channel around the same control plane imposed on user-driven flows.

## Canonical Entry Point Mapping

| Entry point | Normalize to | Notes |
|---|---|---|
| Prompt engine run | `IntakeBundle` -> `SpecBundle` | default vague-prompt path |
| Interrogation API | `IntakeBundle` -> `SpecBundle` | clarification-first path |
| IdeaCloud export/promote | `IntakeBundle` | idea graph enters backbone here |
| Pipeline canvas run | `SpecBundle` or `DecisionPlanBundle` | depending on stage state |
| Self-improve start | `IntakeBundle` or `SpecBundle` | same backbone, different producer |

## Canonical Golden Path

The first path Aragora should make boring and reliable is:

1. user submits vague prompt or promoted idea cluster
2. system emits `SpecBundle`
3. debate emits `DeliberationBundle`
4. plan emits `DecisionPlanBundle`
5. execution emits `ExecutionBundle`
6. verify and bug-fix emit `VerificationBundle`
7. receipt layer emits `ReceiptEnvelope`
8. feedback layer emits `OutcomeFeedbackRecord`
9. Nomic uses that record for reprioritization

If a path cannot do this, it is not yet a canonical production path.

## What This Architecture Deliberately Avoids

1. multiple orchestration stacks with separate contracts
2. direct prompt-to-action shortcuts
3. hidden quality-gate downgrades in automated lanes
4. detached self-improvement loops that ignore product receipts and outcomes
5. treating compliance and security as post-hoc documentation rather than control-plane behavior
