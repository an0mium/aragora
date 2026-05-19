# PARKED — ADC Implementation Prompt Pack — 2026-05-19

**Status:** parked prompts only
**Do not dispatch from this document without operator approval.**
**Global boundaries:** no merges, no labels, no mark-ready, no protected-file edits, no destructive cleanup, no force-push, no branch mutation outside the assigned implementation branch.

## PARKED prompt — ADC v0.7 lifecycle

```text
You are implementing ADC v0.7 Three-Tier Reversibility for Aragora.

Prerequisite gate:
- Refuse to proceed unless #7361, #7358, and #7360 are merged to main, or the operator explicitly names an integration branch containing v0.1-v0.4.
- Do not dispatch subagents unless the operator explicitly authorizes them.

Goal:
Implement lifecycle pause/halt/revoke for Delegation Contracts without process killing.

Deliverables:
1. aragora/policy/contract_lifecycle.py
   - LifecycleEvent
   - LifecycleDecision
   - ContractState
   - append-only JSONL load/reduce helpers
   - pause/resume/halt/revoke event builders
   - fail-closed lifecycle evaluator
2. scripts/update_contract_lifecycle.py
   - --contract PATH
   - --event pause|resume|halt|revoke
   - --reason TEXT
   - --actor TEXT
   - --ledger PATH
   - --dry-run
   - --json
3. scripts/check_contract_lifecycle.py
   - --contract PATH
   - --action-class read|bounded_write|shared_state|delegation|destructive
   - --ledger PATH
   - documented exit codes
4. Tests:
   - tests/policy/test_contract_lifecycle.py
   - tests/scripts/test_update_contract_lifecycle.py
   - tests/scripts/test_check_contract_lifecycle.py

Required behavior:
- active permits actions only if the contract permits them.
- paused denies writes/spawns but allows audit reads.
- halted and revoked are terminal for writes/spawns.
- revoked outranks halt, pause, and resume.
- unreadable lifecycle ledger fails closed for writes/spawns.
- v0.4 signing is used for lifecycle receipts when present.
- missing v0.4 signing support refuses signed-required mode.

Validation:
- python -m pytest tests/policy/test_contract_lifecycle.py tests/scripts/test_update_contract_lifecycle.py tests/scripts/test_check_contract_lifecycle.py -q
- git diff --check
- project preflight before push.

Open a draft PR only. Do not merge or mark ready.
```

## PARKED prompt — ADC v0.8 envelope + validator

```text
You are implementing ADC v0.8 envelope + validator for Aragora.

Prerequisite gate:
- Refuse to proceed unless ADC v0.7 lifecycle has shipped, or the operator explicitly waives that prerequisite.
- Refuse signed-required mode unless #7361 signing symbols are present on the base branch.
- If ADC_v0.8_CROSS_FAMILY_ADAPTER_PLAN.md is not present, carry it into the integration branch or keep the prompt self-contained.

Goal:
Implement the contract-on-disk envelope and parent-contract validation surface.

Deliverables:
1. aragora/policy/contract_envelope.py
   - ContractEnvelope dataclass
   - canonical JSON load/dump
   - schema validation
   - contract + goal extraction
   - effective-scope intersection
   - v0.4 signature verification when required
   - v0.7 lifecycle validation when required
2. scripts/validate_parent_contract.py
   - --parent-contract PATH
   - --action-class read|bounded_write|shared_state|delegation|destructive
   - --require-signed
   - --allow-unsigned-dry-run
   - --require-lifecycle
   - --json
3. Tests:
   - tests/policy/test_contract_envelope.py
   - tests/scripts/test_validate_parent_contract.py

Fail closed for:
- malformed envelope
- scope widening
- expired contract
- missing parent contract
- unsigned contract in signed-required mode
- paused/halted/revoked lifecycle state for worker launch
- missing lifecycle support in lifecycle-required mode

Validation:
- python -m pytest tests/policy/test_contract_envelope.py tests/scripts/test_validate_parent_contract.py -q
- git diff --check
- project preflight before push.

Open a draft PR only. Do not launch workers.
```

## PARKED prompt — ADC v0.8 Droid adapter

```text
You are implementing the ADC v0.8 contract-bound Factory Droid adapter.

Prerequisite gate:
- Refuse to proceed unless ADC v0.8 envelope + validator exists on the base branch.
- Refuse real launches unless the operator explicitly authorizes them. Initial PR should support dry-run only.

Goal:
Make Droid launches accept --parent-contract and propagate validated contract scope into the child worktree.

Deliverables:
1. scripts/launch_contract_bound_droid.py
   - --lane-id
   - --prompt
   - --parent-contract
   - --dry-run
   - --json
2. Behavior:
   - call validate_parent_contract before launch
   - compute effective permissions by intersection
   - copy envelope into child worktree as ADC_PARENT_CONTRACT.json
   - export ARAGORA_PARENT_CONTRACT
   - export ARAGORA_DELEGATION_CONTRACT_ID
   - export ARAGORA_GOAL_ID
   - export ARAGORA_CONTRACT_ENVELOPE_ID
   - prepend worker instructions requiring lane claim and receipt binding
3. Tests:
   - tests/scripts/test_launch_contract_bound_droid.py
   - relevant Droid harness tests

Validation:
- dry-run tests must prove exact launch plan without spawning a worker.
- bash/python compile checks for scripts.
- project preflight before push.

Open a draft PR only. Do not dispatch a real Droid worker.
```

## PARKED prompt — ADC v0.8 Claude/Codex stubs

```text
You are implementing honest ADC v0.8 validation stubs for Claude Code and Codex surfaces.

Prerequisite gate:
- Refuse to proceed unless ADC v0.8 envelope + validator exists on the base branch.
- Do not claim hard runtime tool-call interception unless the harness actually enforces it.

Goal:
Validate and record parent contracts at Claude/Codex startup surfaces without overclaiming enforcement.

Deliverables:
1. Startup integration examples for:
   - Claude Code
   - Codex CLI worker startup
   - Codex Desktop / rollout metadata binding where available
2. Docs or code comments that explicitly state:
   - validates envelope
   - prints effective scope
   - records contract fields for receipts/lane claims
   - does not intercept every tool call in Claude/Codex harnesses
3. Tests:
   - validation behavior
   - receipt-field extraction
   - limitation text present

Validation:
- python -m pytest relevant harness/stub tests -q
- project preflight before push.

Open a draft PR only.
```

## PARKED prompt — ADC v0.5 adversarial checks

```text
You are implementing ADC v0.5 Continuous Adversarial Verification.

Prerequisite gate:
- Prefer to wait until v0.7 and v0.8 exist.
- If the operator authorizes an earlier slice, implement deterministic mock-first policy only.

Goal:
Create an action-boundary adversarial check that can halt or narrow unsafe proposed actions before they mutate state.

Deliverables:
1. aragora/policy/adversarial_check.py
   - ActionCheckRequest
   - ActionCheckVerdict
   - ActionCheckReceipt
   - StaticActionPolicy
   - MockAdversarialReviewer
2. scripts/evaluate_adversarial_action.py
   - accepts JSON request
   - emits JSON verdict
   - no real model calls by default
3. Tests:
   - out-of-scope shared-state action returns HALT
   - in-scope lane claim returns PROCEED
   - destructive action without human approval returns HALT
   - paused/halted/revoked lifecycle returns HALT
   - pure read bypasses model reviewer
   - receipt is append-only and deterministic

Hard rule:
No real model calls in the first implementation PR. Model-backed reviewers must be feature-flagged and disabled in CI.

Open a draft PR only.
```

## PARKED prompt — ADC v0.6 trust multiplier

```text
You are implementing ADC v0.6 ELO Trust Multiplier.

Prerequisite gate:
- Prefer to wait until v0.5 action-check receipts exist.
- If the operator authorizes an earlier slice, implement static trust-budget calculation only with synthetic receipts.

Goal:
Compute effective budget = min(contract_budget * trust_multiplier, operator_cap) per agent and action class, using receipt-backed trust deltas.

Deliverables:
1. aragora/policy/trust_multiplier.py
   - TrustMultiplierRecord
   - TrustUpdateEvent
   - default policy table
   - clamp/floor/ceiling math
   - append-only JSONL reducer
2. scripts/update_trust_multiplier.py
   - consumes synthetic or v0.5 receipts
   - emits append-only trust update
   - --dry-run and --json
3. Tests:
   - defaults by action class
   - green-check receipt increases within ceiling
   - halt/revoke decreases
   - self-reported progress alone does not update
   - operator cap wins over trust expansion
   - corrupt ledger fails closed for increases

Hard rule:
Trust multipliers never authorize an action denied by the contract, lifecycle state, destructive-action policy, or operator cap.

Open a draft PR only.
```
