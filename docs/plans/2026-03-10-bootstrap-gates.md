# Bootstrap Gates For Aragora Self-Improvement

**Date:** March 10, 2026
**Purpose:** Define the minimum conditions Aragora must meet before it is
allowed to autonomously plan, implement, verify, and merge changes against
larger portions of the Aragora codebase.

## Why This Exists

Aragora has enough real infrastructure to plausibly build Aragora with Aragora:
- debate kernel
- server runtime
- queue and worker surfaces
- swarm orchestration
- nomic planning and execution machinery

But the repo is still structurally accretive, with broad compatibility,
scaffolding, and unclear subsystem boundaries. The immediate objective is not
"full autonomy." The immediate objective is to prove that Aragora can govern
its own planning and verification loop safely enough to improve itself in a
bounded lane.

This is a bootstrap problem.

## Core Principle

Aragora is only allowed to repair larger parts of itself after it can do all of
the following in a narrow safe lane:
- plan bounded work
- assign bounded work
- cross-check outputs
- run real verification
- reject bad work
- recover from stuck or failed execution

If a human must manually stitch the loop together, Aragora is not yet good
enough to broadly fix Aragora.

## Execution Policy

Autonomous scope expands only when a lower phase has demonstrated:
- reproducible task execution
- explicit verification artifacts
- reliable failure escalation
- bounded cost
- no silent drift against architecture decisions

Autonomy should expand by proof, not by confidence.

## Phase 0A: Prove Governance

### Goal

Prove that Aragora can reason, plan, track, verify, and escalate work in a safe
governance lane without human rescue.

### Allowed Scope

- ADR generation
- subsystem ledger generation
- entrypoint inventory generation
- deploy/runtime truth documentation
- backlog decomposition
- campaign manifest generation
- execution receipts and plan artifacts

### Not Allowed

- autonomous merges to `main`
- production deploy changes
- schema migrations
- broad refactors
- feature generation in expansion surfaces

### Required Capabilities

- one canonical campaign spec
- one canonical task state model
- one artifact store for plan, diff, receipt, and verification outputs
- one stuck-task detector
- one failure escalation path
- one reviewer or synthesizer step that can reject bad outputs
- one canonical receipt format for executed work

### Required Deliverables

- campaign schema document
- task lifecycle document
- escalation policy
- verification artifact format
- review/rejection policy

### Exit Gate

Phase 0A is complete only if all of the following are true:
- 10 consecutive documentation/governance tasks complete without manual rescue
- each task emits a plan artifact, diff artifact, verification artifact, and receipt
- no silent failures occur
- no duplicate or conflicting task execution occurs
- every rejected task has a recorded rejection reason

If any of those fail, remain in Phase 0A.

## Phase 0B: Prove Verified Execution

### Goal

Prove that Aragora can make small code changes in canonical paths and verify
them correctly before any autonomous merge authority is widened.

### Allowed Scope

- CI/config drift fixes
- packaging truth fixes
- deploy command normalization
- startup-path cleanup
- small health check additions
- test additions in canonical subsystems
- small refactors in canonical runtime paths

### Not Allowed

- broad API migrations
- large-file decomposition
- expansion-surface feature work
- connector proliferation
- schema or storage migrations without explicit approval

### Required Capabilities

- merge gate always enabled
- bounded retry policy
- worker failure isolation
- cost and budget caps enforced
- truth-suite lane required for merge eligibility
- explicit human approval required for merge

### Verification Standard

Every task in this phase must produce:
- code diff
- required checks result
- truth-suite result
- reviewer verdict
- receipt with final status

### Exit Gate

Phase 0B is complete only if all of the following are true:
- 10 consecutive code-change tasks succeed in canonical subsystems
- all changes pass required checks and the truth-suite
- at least 3 tasks include real composed verification, not only unit tests
- no merge is reverted because of orchestrator-caused regression
- no budget overrun occurs without escalation

If any of those fail, remain in Phase 0B or drop back to Phase 0A.

## Phase 1: Controlled Self-Repair

### Goal

Allow Aragora to start fixing core architectural debt under explicit boundaries.

### Allowed Scope

- backend consolidation work
- API surface classification
- domain boundary enforcement
- memory/knowledge contract cleanup
- nomic subsystem hardening
- worker/runtime normalization
- test taxonomy and truth-suite expansion

### Constraints

- work is limited to `canonical` and `core-but-messy` subsystems
- no autonomous work in broad expansion surfaces unless it supports canonical paths
- no autonomous schema migrations without explicit approval
- no autonomous broad refactor without a prior split plan
- no autonomous deletion campaign without a residue inventory and rollback plan

### Required Capabilities

- canonical runtime ADR implemented
- canonical worker model implemented
- domain boundary checks active in CI
- truth-suite covers backend startup, debate flow, worker flow, and persistence flow
- observability baseline exists for backend, worker, and orchestrator surfaces

### Exit Gate

Phase 1 is complete only if all of the following are true:
- canonical runtime story is implemented, not just documented
- worker model is normalized across deploy surfaces
- boundary checks block new cross-layer drift
- truth-suite is required in CI for canonical changes
- observability signals can distinguish healthy from degraded core flows

If any of those fail, remain in Phase 1.

## Phase 2: Broader Repo Repair

### Goal

Allow Aragora to repair broader repo structure after proving control of the core.

### Allowed Scope

- handler rationalization
- frontend contract cleanup
- selective connector cleanup
- compatibility reduction
- large-file decomposition
- residue inventory and retirement planning

### Still Not Allowed

- speculative new product surfaces
- bulk connector generation
- large autonomous migrations across undefined boundaries
- uncontrolled parallel refactors touching multiple unclassified subsystems

### Exit Gate

Phase 2 should only continue if:
- rollback rate remains low
- orchestrator-caused regressions remain near zero
- quality gates remain on by default
- verification artifacts remain complete and reproducible

Otherwise, shrink scope back to Phase 1.

## Hard Stop Conditions

If any of the following occur, autonomy drops back one phase immediately:

- silent task loss
- merge without real verification
- repeated stuck workers without escalation
- budget overrun without explicit approval
- production-path regression caused by orchestrator-issued change
- campaign outputs contradict accepted ADRs or subsystem contracts
- duplicate execution of the same task causing conflicting changes

## First Five Bootstrap Tickets

These should be implemented before attempting broader self-repair:

1. Canonical campaign and task schema
2. Failure escalation and stuck-task detector
3. Merge gate with required verification artifacts
4. Minimal truth-suite for orchestrator-controlled changes
5. Canonical receipt/report format for plan -> change -> verify

## What Counts As "Good Enough To Fix Itself"

Aragora is good enough to broadly fix Aragora only when it can:
- generate a bounded plan
- assign bounded work
- cross-check heterogeneous agent output
- run real verification against composed paths
- reject bad output without human intervention
- escalate or recover from failure automatically
- stay inside architecture constraints and budget limits

Until then, the self-improvement engine should be treated as a constrained
operator-assist system, not a general autonomous maintainer.

## Relationship To Other Planning Documents

This document does not replace:
- [`ROADMAP.md`](/Users/armand/Development/aragora/ROADMAP.md)
- product or launch roadmaps
- feature-gap tracking
- subsystem-specific design docs

It is the execution gating policy for when Aragora is allowed to use Aragora to
repair Aragora.

## Recommended Next Step

Use this document to define the first Phase 0A campaign:
- generate ADRs for runtime and worker canon
- generate subsystem ledger
- generate entrypoint inventory
- generate deploy truth table
- require receipts and verification artifacts for every task

Related artifacts:
- evidence report:
  [`docs/plans/2026-03-10-dogfood-6-evidence.md`](/Users/armand/Development/aragora/docs/plans/2026-03-10-dogfood-6-evidence.md)
- Phase 0A manifest template:
  [`docs/plans/phase0a_campaign_manifest.yaml`](/Users/armand/Development/aragora/docs/plans/phase0a_campaign_manifest.yaml)
