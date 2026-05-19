# ADC v0.7/v0.8 Test Matrix — 2026-05-19

**Status:** fallback queue artifact
**Scope:** planned tests after ADC base stack lands
**No code changes included:** this matrix is an implementation planning aid.

## Current coverage on `main`

Current `main` has ADC v0.1 coverage:

| Area | Existing file | Covered |
|---|---|---|
| contract schema/narrowing | `tests/policy/test_delegation_contract.py` | root validation, depth, child narrowing, budgets, destructive deny, expiry, surface containment, v0.1 unsigned-only rule |
| predicate oracle | `tests/policy/test_predicate_oracle.py` | parser basics, filesystem predicates, PR/issue/branch/commit/test predicates with subprocess mocked |
| lane registry | `tests/scripts/test_claim_active_agent_lane.py` | lane claim/refresh/conflict behavior, resource collision detection, stale release, CLI roundtrip |
| agent bridge/launcher | `tests/scripts/test_agent_bridge.py`, `tests/scripts/test_agent_bridge_broker.py`, `tests/swarm/agent_bridge/*`, `tests/swarm/test_worker_launcher.py` | generic launch/harness behavior, not yet ADC-aware |

Coverage gaps:

- `GoalSpec.validate()` edge cases need direct tests.
- `AllowedSurfaces.pr_numbers` and `worktree_globs` are lightly covered.
- `ContractBudget` negative fields need complete coverage.
- predicate parser needs quoted-comma and malformed-quote edge tests.
- no lifecycle ledger/reducer tests exist.
- no contract envelope/adapter tests exist.
- no launcher fail-closed tests exist for missing/expired/unsigned/revoked parent contracts.

## v0.7 lifecycle matrix

Suggested file: `tests/policy/test_contract_lifecycle.py`

| ID | Scenario | Expected |
|---|---|---|
| L1 | no events for valid contract | state `active` |
| L2 | `pause` event | bounded/shared/delegation actions denied; audit reads allowed |
| L3 | `resume` after `pause` | state `active`; actions evaluated by contract scope |
| L4 | `halt` from `active` | terminal `halted`; writes/spawns denied |
| L5 | `halt` from `paused` | terminal `halted`; resume denied |
| L6 | `revoke` from `active` | terminal `revoked`; writes/spawns denied |
| L7 | `revoke` after `halt` | state `revoked`; revoke outranks halt |
| L8 | stale later `resume` after `revoke` | still `revoked` |
| L9 | corrupt lifecycle ledger | write/spawn checks fail closed |
| L10 | missing lifecycle ledger in required mode | write/spawn checks fail closed |
| L11 | missing lifecycle ledger in optional dry-run mode | explicit warning, no silent `active` |
| L12 | descendant contract found through parent chain | parent revoke blocks descendant |
| L13 | lifecycle event with invalid schema | ignored for state increase, recorded as error |
| L14 | signed receipt verification failure | fail closed in signed-required mode |

Suggested file: `tests/scripts/test_update_contract_lifecycle.py`

| ID | Scenario | Expected |
|---|---|---|
| U1 | dry-run pause | JSON event printed; ledger unchanged |
| U2 | append pause | ledger receives one JSONL event |
| U3 | resume from paused | event appended and state active |
| U4 | resume from halted | non-zero exit |
| U5 | revoke from any non-revoked state | event appended |
| U6 | append is atomic | no partial file after simulated failure |
| U7 | invalid contract file | exit 3 |

Suggested file: `tests/scripts/test_check_contract_lifecycle.py`

| ID | Scenario | Expected exit |
|---|---|---:|
| C1 | active + read | 0 |
| C2 | active + delegation, depth/budget valid | 0 |
| C3 | paused + read | 0 |
| C4 | paused + bounded_write | 2 |
| C5 | halted + read | 0 with audit-only warning |
| C6 | halted + delegation | 2 |
| C7 | revoked + shared_state | 2 |
| C8 | invalid contract | 3 |
| C9 | unreadable ledger | 4 |
| C10 | missing v0.4 in signed-required mode | 5 |

## v0.8 envelope matrix

Suggested file: `tests/policy/test_contract_envelope.py`

| ID | Scenario | Expected |
|---|---|---|
| E1 | valid signed envelope | validates |
| E2 | unsigned envelope in dry-run mode | validates with warning |
| E3 | unsigned envelope in signed-required mode | fails |
| E4 | malformed JSON | fails with schema error |
| E5 | unexpected schema version | fails |
| E6 | child allowed action widens parent | fails |
| E7 | child branch glob widens parent | fails |
| E8 | child budget exceeds parent | fails |
| E9 | expired contract | fails |
| E10 | revoked contract | fails |
| E11 | paused contract for worker launch | fails |
| E12 | paused contract for audit read | validates as audit-only |
| E13 | missing lifecycle support in required mode | fails |
| E14 | missing v0.4 signing support in required mode | fails |
| E15 | envelope id and contract id are stable in receipt refs | validates |

Suggested file: `tests/scripts/test_validate_parent_contract.py`

| ID | CLI | Expected |
|---|---|---|
| V1 | `--parent-contract valid.json --json` | exit 0 with effective scope |
| V2 | missing file | non-zero |
| V3 | invalid schema | non-zero |
| V4 | `--require-signed` with unsigned envelope | non-zero |
| V5 | `--allow-unsigned-dry-run` with unsigned envelope | exit 0 with warning |
| V6 | `--action-class delegation` with paused lifecycle | non-zero |
| V7 | `--action-class read` with halted lifecycle | exit 0 audit-only |
| V8 | corrupt lifecycle ledger | non-zero fail-closed |

## v0.8 adapter matrix

Suggested file: `tests/scripts/test_launch_contract_bound_droid.py`

| ID | Scenario | Expected |
|---|---|---|
| D1 | dry-run valid parent | prints launch plan, no worker spawned |
| D2 | invalid parent | refuses before launch |
| D3 | paused parent | refuses before launch |
| D4 | revoked parent | refuses before launch |
| D5 | effective env vars | plan includes `ARAGORA_PARENT_CONTRACT`, `ARAGORA_DELEGATION_CONTRACT_ID`, `ARAGORA_GOAL_ID`, `ARAGORA_CONTRACT_ENVELOPE_ID` |
| D6 | child worktree copy | plan includes `ADC_PARENT_CONTRACT.json` copy target |
| D7 | missing prompt | refuses |
| D8 | missing launcher | refuses |

Suggested existing harness files to extend:

- `tests/swarm/agent_bridge/test_harnesses_claude.py`
- `tests/swarm/agent_bridge/test_harnesses_codex.py`
- `tests/swarm/agent_bridge/test_harnesses_droid.py`
- `tests/swarm/test_worker_launcher.py`

| ID | Scenario | Expected |
|---|---|---|
| H1 | Claude startup stub receives parent contract | validates and records limitation |
| H2 | Codex startup stub receives parent contract | validates and records limitation |
| H3 | Droid startup wrapper receives parent contract | hard launch gate |
| H4 | Codex Desktop metadata binding | parent contract id recorded where available |
| H5 | stub limitation text | no claim of full tool interception |
| H6 | worker receipt binding | contract id + envelope id carried into receipt |

## v0.5/v0.6 bridge tests

Suggested file: `tests/policy/test_adversarial_check.py`

| ID | Scenario | Expected |
|---|---|---|
| A1 | out-of-scope branch push | `HALT` |
| A2 | in-scope lane claim | `PROCEED` |
| A3 | destructive action without approval | `HALT` |
| A4 | pure read | `PROCEED` without model reviewer |
| A5 | paused lifecycle | `HALT` |
| A6 | mocked reviewer halt | receipt written |

Suggested file: `tests/policy/test_trust_multiplier.py`

| ID | Scenario | Expected |
|---|---|---|
| T1 | new agent defaults by action class | table values |
| T2 | green-check event | positive delta within ceiling |
| T3 | halt event | negative delta |
| T4 | revoke event | strong negative delta |
| T5 | self-reported progress only | no update |
| T6 | operator cap lower than multiplier result | cap wins |
| T7 | corrupt trust ledger | fail closed for increases |

## Validation command bundles

### Current-main smoke

```bash
python -m pytest tests/policy/test_delegation_contract.py tests/policy/test_predicate_oracle.py -q
```

### After v0.7 implementation

```bash
python -m pytest \
  tests/policy/test_contract_lifecycle.py \
  tests/scripts/test_update_contract_lifecycle.py \
  tests/scripts/test_check_contract_lifecycle.py \
  tests/scripts/test_claim_active_agent_lane.py \
  tests/scripts/test_evaluate_goal_progress.py \
  -q
```

### After v0.8 implementation

```bash
python -m pytest \
  tests/policy/test_contract_envelope.py \
  tests/scripts/test_validate_parent_contract.py \
  tests/scripts/test_launch_contract_bound_droid.py \
  tests/swarm/agent_bridge/test_harnesses_claude.py \
  tests/swarm/agent_bridge/test_harnesses_codex.py \
  tests/swarm/agent_bridge/test_harnesses_droid.py \
  tests/swarm/test_worker_launcher.py \
  -q
```

### After v0.5/v0.6 implementation

```bash
python -m pytest \
  tests/policy/test_adversarial_check.py \
  tests/policy/test_trust_multiplier.py \
  tests/scripts/test_evaluate_adversarial_action.py \
  tests/scripts/test_update_trust_multiplier.py \
  -q
```

## Operator notes

- Keep first implementation PRs deterministic and mock-first.
- Do not let v0.8 claim full Claude/Codex tool interception until the harness can enforce it.
- Do not allow v0.6 trust multipliers to override contract denies, lifecycle denies, or destructive-action human gates.
- Treat generated `.aragora/*/*.jsonl` ledgers as operator-owned artifacts; tests should use temporary directories by default.
