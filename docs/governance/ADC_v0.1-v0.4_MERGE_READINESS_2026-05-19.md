# ADC v0.1 → v0.4 Merge Readiness Packet

**Date:** 2026-05-19
**Purpose:** operator-visible gate before any ADC v0.7 dispatch
**Verdict:** **BLOCKED — do not dispatch v0.7 yet**

## Summary

The previous stack audit recommended merge order:

```text
#7357 → #7361 → {#7358, #7360}
```

That order has drifted. `#7357` is now merged, but `#7361` is currently `CONFLICTING` / `DIRTY` against `main`, so the second step in the audit order is no longer directly mergeable. v0.7 must remain parked until the operator either resolves/rebases v0.4 or explicitly authorizes a named integration branch.

## PR readiness table

| PR | Version | State | Draft | Head SHA | Mergeable | Merge state | CI rollup | Blockers | Recommendation |
|---|---|---:|---:|---|---|---|---|---|---|
| [#7357](https://github.com/synaptent/aragora/pull/7357) | v0.1 schema + predicate oracle | MERGED | no | `94f0b688129b0b121b091ddc9f4c93fe257a046d` | UNKNOWN | UNKNOWN | 88 total: 62 success, 25 skipped, 1 cancelled | already merged | `ready-for-base` / no action |
| [#7361](https://github.com/synaptent/aragora/pull/7361) | v0.4 HMAC signing | OPEN | yes | `e39a558c6639ca906a543064ccabeb2587f9dc49` | CONFLICTING | DIRTY | 70 total: 17 success, 53 skipped, 0 failure, 0 pending | draft, review-required, conflicts with current `main` | `needs-rebase` |
| [#7358](https://github.com/synaptent/aragora/pull/7358) | v0.2 lane-registry hookup | OPEN | yes | `d926a9749f23b8ac097a2ec8573df7e63a11f738` | MERGEABLE | BLOCKED | 6 total: 4 success, 2 skipped, 0 failure, 0 pending | draft + review-required | `needs-review` / later merge after v0.4 is resolved |
| [#7360](https://github.com/synaptent/aragora/pull/7360) | v0.3 progress ledger | OPEN | yes | `7b0cf9ea426fe2cc78b3e7298f1db7618931a8db` | MERGEABLE | BLOCKED | 70 total: 17 success, 53 skipped, 0 failure, 0 pending | draft + review-required | `needs-review` / later merge after v0.4 is resolved |

## File-overlap observations

All three open ADC PRs still touch parts of the v0.1 trust-kernel surface:

- `aragora/policy/__init__.py`
- `aragora/policy/delegation_contract.py`
- `aragora/policy/predicate_oracle.py`
- `docs/governance/DELEGATION_CONTRACT_V0_1_SPEC.md`
- `tests/policy/test_delegation_contract.py`
- `tests/policy/test_predicate_oracle.py`

`#7361` additionally adds signing-specific surfaces:

- `aragora/policy/contract_signing.py`
- `scripts/sign_delegation_contract.py`
- `tests/policy/test_contract_signing.py`
- `tests/scripts/test_sign_delegation_contract.py`

`#7358` adds the lane-registry integration in `scripts/claim_active_agent_lane.py`.

`#7360` adds the progress evaluator in `scripts/evaluate_goal_progress.py`.

## Gate conclusion

The audit order remains conceptually correct, but it is no longer mechanically ready because `#7361` must be rebased or otherwise repaired after `#7357` landed.

Do **not** dispatch ADC v0.7 until one of these happens:

1. `#7361` is rebased/repaired and the base stack lands in a refreshed order; or
2. the operator explicitly authorizes an integration branch containing v0.1-v0.4 as the v0.7 base.

## Operator decision needed

Choose one:

- **Repair-and-merge path:** rebase/repair `#7361`, then settle `#7358` and `#7360`.
- **Integration-branch path:** authorize a named branch that composes `#7361`, `#7358`, and `#7360` on top of current `main`, then dispatch v0.7 against that branch.

No merges, labels, ready-state flips, or v0.7 dispatches were performed by this packet.
