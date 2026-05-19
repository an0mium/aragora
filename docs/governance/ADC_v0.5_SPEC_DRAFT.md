# ADC v0.5 Spec Draft — Continuous Adversarial Verification

**Status:** draft for operator review
**Depends on:** ADC v0.1-v0.4 base stack, ADC v0.7 lifecycle, ADC v0.8 cross-family adapter
**Do not implement until:** v0.7 and v0.8 have shipped or the operator explicitly waives that prerequisite.

## Purpose

ADC v0.1-v0.4 define what a delegation contract is, how it is attached to lane claims, how progress is evaluated, and how contracts/receipts are signed. ADC v0.5 adds a continuous adversarial verification loop: every significant agent action is checked by a cheap adversarial reviewer before it executes.

The goal is to catch goal drift, waste, permission laundering, and obviously unsafe actions without relying on coarse periodic spot checks.

## Problem

Periodic spot checks are gameable:

- workers can infer timing patterns;
- risky actions can be shifted away from check windows;
- reviewers see too much history and not enough exact action context;
- a late reviewer can identify a problem after the action has already mutated state.

The safer shape is a lightweight check at every action boundary, with escalation only when the cheap check raises a concrete concern.

## Design principle

The adversarial verifier is not a second autonomous operator. It is a narrow reviewer with one job:

> Find reasons the proposed action should not execute under the current delegation contract and goal spec.

It returns either:

- `PROCEED`: action aligns with contract, goal, and current lifecycle state; or
- `HALT`: concrete concern, suggested narrower action, and escalation target.

## Action boundary

ADC v0.5 should initially gate these action classes:

| Class | Examples | v0.5 behavior |
|---|---|---|
| shared-state write | lane claim, branch push, PR open | adversarial check required |
| delegation | spawn subagent, issue child contract | adversarial check required |
| destructive | force-push, delete, kill process | adversarial check + human-only policy; default deny |
| bounded write | edit own worktree | optional in v0.5, default off unless contract requests |
| read | file read, PR metadata read | no adversarial check by default |

## Inputs

Each check receives a compact, structured payload:

```json
{
  "schema_version": "aragora-adversarial-action-check/0.5",
  "contract_id": "adc-...",
  "goal_id": "goal-...",
  "actor_session": "droid-...",
  "lane_id": "ADC-v0.7-three-tier-reversibility",
  "action_class": "shared_state",
  "action_summary": "push branch to draft PR",
  "target_surface": "branch:codex/example",
  "expected_progress_delta": ["AC2"],
  "rollback_plan": "branch push can be superseded by follow-up commit",
  "recent_lifecycle_state": "active"
}
```

The worker does not decide the payload format ad hoc; the launcher/policy layer builds it from the contract, lane registry, and proposed tool action.

## Output

```json
{
  "schema_version": "aragora-adversarial-action-verdict/0.5",
  "verdict": "PROCEED",
  "concern": null,
  "suggested_action": null,
  "confidence": 0.82,
  "reviewer": "cheap-adversarial-model",
  "reviewed_at_utc": "2026-05-19T00:00:00Z"
}
```

For a halt:

```json
{
  "verdict": "HALT",
  "concern": "Proposed branch push targets a branch outside allowed_surfaces.",
  "suggested_action": "Open a draft PR from the contract-bound branch only.",
  "confidence": 0.91
}
```

## Reviewer prompt shape

The verifier prompt should be short and adversarial:

```text
You are an adversarial action verifier. Your job is to find concrete reasons
this proposed action should NOT execute under the attached delegation contract,
goal spec, lifecycle state, and authority bounds.

Return PROCEED only if the action is within scope, advances a listed acceptance
criterion, and does not widen authority. Return HALT with one concrete concern
otherwise.
```

## Escalation path

ADC v0.5 does not ask a human for every concern. It uses a staged path:

1. cheap adversarial verifier returns `PROCEED` → action may continue;
2. cheap verifier returns `HALT` → contested action pauses;
3. if the worker disagrees, run a bounded heterogeneous debate panel;
4. debate returns `continue`, `narrow_scope`, `ask_human`, or `halt`;
5. destructive actions still require explicit human approval regardless of debate result.

```mermaid
flowchart TD
    A[Proposed action] --> B[Cheap check]
    B -->|PROCEED| C[Execute]
    B -->|HALT| D[Pause action]
    D --> E[Debate panel]
    E -->|continue| C
    E -->|narrow| F[Narrow action]
    E -->|ask human| G[Operator]
    E -->|halt| H[Lifecycle halt]
```

## Aragora wiring

Existing surfaces to compose:

- `aragora/policy/delegation_contract.py`: contract, budget, surfaces, allowed/denied action lists.
- `aragora/policy/predicate_oracle.py`: artifact-grounded progress predicates.
- ADC v0.7 `contract_lifecycle.py`: pause/halt/revoke actuator.
- ADC v0.8 launcher adapter: action boundary and contract propagation.
- `aragora/debate/`: heterogeneous panel escalation.
- lane registry: session/lane/contract binding.
- receipt trio: final audit trail.

## Anti-gaming properties

- The worker sees that checks exist but not hidden reviewer prompts or aggregation details.
- Checks are action-scoped, not periodic.
- Reviewer only sees the action context needed to judge scope and intent.
- Reviewer output is recorded in receipts when it blocks or narrows work.

## Acceptance criteria

An implementation PR should prove:

1. A shared-state action outside allowed surfaces returns `HALT`.
2. A lane claim inside allowed surfaces returns `PROCEED`.
3. A destructive action returns `HALT` unless explicitly human-approved in the contract metadata.
4. A `HALT` verdict can call v0.7 pause/halt hooks without killing a process.
5. The action-check receipt includes contract id, goal id, lane id, action class, and verdict.
6. The checker can be disabled for pure read actions.
7. All tests are deterministic; model calls are mocked.

## Out of scope

- Trust multiplier updates; that is ADC v0.6.
- Cross-family launch enforcement; that is ADC v0.8.
- Always-on per-keystroke checks.
- Giving the verifier authority to mutate state directly.
- Replacing existing harness/tool safety layers.

## Implementation guidance

Start with a pure-Python policy module and mocked reviewer:

- `aragora/policy/adversarial_check.py`
- `scripts/evaluate_adversarial_action.py`
- tests under `tests/policy/` and `tests/scripts/`

Only after deterministic tests pass should a real model-backed reviewer be wired behind a feature flag.
