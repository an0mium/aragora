# ADC v0.8 Prompt Pack Readiness — 2026-05-19

**Status:** readiness packet only
**Prompt source:** parked `.aragora/v16-dispatch/ADC-v0.8-*.md` files from the prior Factory worktree
**Dispatch gate:** do not dispatch v0.8 until v0.7 ships, or the operator explicitly waives that prerequisite.

## Purpose

ADC v0.8 is the cross-family adapter layer: the contract should travel with a worker launch regardless of whether the worker is Factory Droid, Claude Code, or Codex. The parked prompts are useful but gitignored. This tracked readiness document preserves the dispatch preconditions, verified symbols, expected deliverables, and enforcement limitations without committing executable launch artifacts.

## Readiness verdict

Not ready to dispatch.

Required before v0.8 work starts:

1. ADC v0.1 on `main` — complete.
2. ADC v0.4 `#7361` on `main` — pending operator review/merge.
3. ADC v0.2 `#7358` and v0.3 `#7360` on `main` — pending operator review/merge.
4. ADC v0.7 lifecycle state machine on `main` or explicit operator waiver.
5. If PR #7367 is not merged, carry `docs/governance/ADC_v0.8_CROSS_FAMILY_ADAPTER_PLAN.md` into the v0.8 integration branch before dispatch.

## Verified current-main symbols

The parked prompts correctly reference these current `main` symbols:

| Symbol | Source | Use in v0.8 |
|---|---|---|
| `DelegationContract` | `aragora.policy` | parent authority object |
| `GoalSpec` | `aragora.policy` | goal/acceptance binding |
| `AcceptanceCriterion` | `aragora.policy` | deterministic acceptance criteria |
| `AllowedSurfaces` | `aragora.policy` | branch/file/worktree/PR scope |
| `ContractBudget` | `aragora.policy` | gas/budget scope |
| `ContractValidationError` | `aragora.policy` | fail-closed validation |
| `narrow_for_child` | `aragora.policy` | child contract narrowing |
| `make_root_contract` | `aragora.policy` | tests/fixtures |
| `parse_predicate` | `aragora.policy` | goal predicate parsing |
| `evaluate_predicate` | `aragora.policy` | deterministic predicate checks |

## Verified v0.4 symbols from `#7361`

These symbols are verified on rebased `#7361` head `9093ddbc48ac4dd2ccaa3364b527b660c089a41e`, but are not on `main` until `#7361` merges:

| Symbol | Intended use |
|---|---|
| `SIGNING_SCHEMA_VERSION` | signing schema guard |
| `SigningError` | sign failure handling |
| `VerificationError` | verification failure handling |
| `VerificationResult` | structured verification result |
| `canonical_contract_payload` | stable signing payload |
| `sign_contract` | contract signing |
| `verify_contract` | contract verification |
| `sign_receipt` | receipt signing |
| `verify_receipt` | receipt verification |
| `is_contract_signed` | signed/unsigned mode gate |
| `signing_key_available` | fail-closed dependency check |

## Planned v0.7 symbols

These are planned by the v0.7 blueprint and should be imported only after v0.7 lands:

- `get_contract_state`
- `ContractState`
- lifecycle reducers/checkers that return active/paused/halted/revoked decisions

Executable v0.8 code must either:

1. depend on v0.7 and import these symbols directly; or
2. run in unsigned/lifecycle-disabled dry-run mode and clearly label lifecycle checks as unavailable.

It must not silently treat missing lifecycle support as `active`.

## Prompt lanes

### `ADC-v0.8-envelope-and-validator`

Goal:

> Implement the contract-on-disk envelope and validation surface.

Expected deliverables:

- `aragora/policy/contract_envelope.py`
- `ContractEnvelope` dataclass
- canonical JSON load/dump
- schema validation
- signature verification via v0.4
- lifecycle state check via v0.7
- `scripts/validate_parent_contract.py`
- deterministic tests for signed, unsigned dry-run, malformed, scope-widening, paused, halted, and revoked cases

Fail-closed rules:

- invalid envelope → non-zero exit;
- missing parent contract → non-zero exit;
- missing v0.4 when signed mode is required → non-zero exit;
- missing v0.7 when lifecycle mode is required → non-zero exit;
- paused/halted/revoked → non-zero exit for worker launch.

### `ADC-v0.8-droid-adapter`

Goal:

> Make Factory Droid worker launches contract-bound by accepting `--parent-contract <path>` and propagating validated scope into the child worktree.

Expected deliverables:

- `scripts/launch_contract_bound_droid.py`
- flags `--lane-id`, `--prompt`, `--parent-contract`, `--dry-run`
- parent envelope validation before launch
- child permission intersection
- copied `ADC_PARENT_CONTRACT.json`
- exported:
  - `ARAGORA_PARENT_CONTRACT`
  - `ARAGORA_DELEGATION_CONTRACT_ID`
  - `ARAGORA_GOAL_ID`
  - `ARAGORA_CONTRACT_ENVELOPE_ID`
- worker instruction prelude requiring lane claim and receipt binding

The first implementation PR should support `--dry-run` and prove the launch plan without actually spawning a worker in tests.

### `ADC-v0.8-claude-codex-stubs`

Goal:

> Provide honest validation stubs for Claude Code and Codex without overclaiming runtime enforcement.

Expected deliverables:

- examples for Claude Code startup;
- examples for Codex CLI worker startup;
- examples for Codex Desktop/rollout metadata binding where available;
- docs section explicitly stating limitations;
- tests for validation and receipt-field extraction.

Hard limitation text to preserve:

> Claude/Codex stubs may validate and report lifecycle state, but must not claim to intercept every tool call unless the harness supports that enforcement boundary.

## Contract envelope shape

Suggested v0.8 envelope:

```json
{
  "schema_version": "aragora-contract-envelope/0.8",
  "envelope_id": "env-...",
  "contract": {},
  "goal_spec": {},
  "parent_envelope_id": null,
  "issued_for": {
    "agent_family": "droid",
    "owner_session": "droid-...",
    "lane_id": "ADC-v0.8-droid-adapter"
  },
  "effective_scope": {
    "allowed_actions": ["read:*"],
    "allowed_surfaces": {},
    "budget": {}
  },
  "lifecycle": {
    "required": true,
    "state": "active",
    "checked_at_utc": "2026-05-19T00:00:00Z"
  },
  "signing": {
    "required": true,
    "contract_signed": true,
    "verified_at_utc": "2026-05-19T00:00:00Z"
  }
}
```

The envelope stores the effective scope that was validated at launch time. The source contract remains canonical, and downstream receipts should cite both contract id and envelope id.

## Adapter enforcement matrix

| Family | Hard launch gate? | Tool-call interception? | Honest v0.8 claim |
|---|---:|---:|---|
| Factory Droid | yes, through wrapper | partial/launcher-mediated | can refuse launch and pass scope/env into worker |
| Claude Code | startup validation only unless harness changes | not guaranteed | can validate envelope, print scope, and bind receipts |
| Codex CLI | startup validation only unless harness changes | not guaranteed | can validate envelope, print scope, and bind receipts |
| Codex Desktop | metadata binding where available | not guaranteed | can record parent contract in rollout/session metadata |

## Tests to require before dispatch

Suggested files:

- `tests/policy/test_contract_envelope.py`
- `tests/scripts/test_validate_parent_contract.py`
- `tests/scripts/test_launch_contract_bound_droid.py`
- `tests/swarm/agent_bridge/test_harnesses_claude.py`
- `tests/swarm/agent_bridge/test_harnesses_codex.py`
- `tests/swarm/agent_bridge/test_harnesses_droid.py`

Core cases:

1. valid signed envelope passes;
2. unsigned envelope passes only in explicit dry-run/unsigned mode;
3. malformed JSON fails;
4. schema mismatch fails;
5. signature verification failure fails;
6. child scope widening fails;
7. paused contract fails worker launch;
8. halted contract fails worker launch;
9. revoked contract fails worker launch;
10. missing lifecycle support fails when lifecycle is required;
11. droid dry-run prints exact launch plan;
12. Claude/Codex stubs record limitation text and do not claim full tool interception.

## Dispatch checklist

- [ ] Base stack `#7361`, `#7358`, `#7360` has landed on `main`.
- [ ] v0.7 lifecycle is on `main`, or operator explicitly waives lifecycle prerequisite.
- [ ] `ADC_v0.8_CROSS_FAMILY_ADAPTER_PLAN.md` is present in the target branch.
- [ ] v0.8 prompt pack is refreshed against current symbols.
- [ ] `validate_parent_contract.py` prompt is dispatched before adapter prompts.
- [ ] Droid adapter prompt runs before Claude/Codex stubs if hard enforcement is desired first.
- [ ] All PRs remain draft until operator review.

## Stop conditions

Do not dispatch v0.8 if:

- any base-stack PR is still open;
- v0.7 has not shipped and no explicit waiver exists;
- v0.4 signing symbols drift from the verified names above;
- the prompt would require protected-file edits;
- the implementation would claim enforcement that the harness cannot provide.
