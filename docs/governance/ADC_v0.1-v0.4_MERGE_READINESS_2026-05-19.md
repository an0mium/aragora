# ADC v0.1 → v0.4 Merge Readiness Packet

**Date:** 2026-05-19
**Purpose:** operator-visible gate before any ADC v0.7 dispatch
**Verdict:** **READY_FOR_OPERATOR_REVIEW — base stack mergeable; awaiting operator merge**

## Summary

The previous stack audit recommended merge order:

```text
#7357 → #7361 → {#7358, #7360}
```

The audit order is mechanically restored after `#7361`'s rebase (head `9093ddbc48ac4dd2ccaa3364b527b660c089a41e`). `#7357` is merged; `#7361`, `#7358`, and `#7360` are all `MERGEABLE` / `BLOCKED` on review and may be merged in the audit order.

## PR readiness table

| PR | Version | State | Draft | Head SHA | Mergeable | Merge state | CI rollup | Blockers | Recommendation |
|---|---|---:|---:|---|---|---|---|---|---|
| [#7357](https://github.com/synaptent/aragora/pull/7357) | v0.1 schema + predicate oracle | MERGED | no | `94f0b688129b0b121b091ddc9f4c93fe257a046d` | UNKNOWN | UNKNOWN | 88 total: 62 success, 25 skipped, 1 cancelled | already merged | `ready-for-base` / no action |
| [#7361](https://github.com/synaptent/aragora/pull/7361) | v0.4 HMAC signing | OPEN | yes | `9093ddbc48ac4dd2ccaa3364b527b660c089a41e` | MERGEABLE | BLOCKED | 68 total: 16 success, 52 skipped, 0 failure, 0 cancelled | draft, review-required | `needs-review` |
| [#7358](https://github.com/synaptent/aragora/pull/7358) | v0.2 lane-registry hookup | OPEN | yes | `d926a9749f23b8ac097a2ec8573df7e63a11f738` | MERGEABLE | BLOCKED | 6 total: 4 success, 2 skipped, 0 failure, 0 pending | draft + review-required | `needs-review` |
| [#7360](https://github.com/synaptent/aragora/pull/7360) | v0.3 progress ledger | OPEN | yes | `7b0cf9ea426fe2cc78b3e7298f1db7618931a8db` | MERGEABLE | BLOCKED | 70 total: 17 success, 53 skipped, 0 failure, 0 pending | draft + review-required | `needs-review` |

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

The audit order is mechanically ready again: `#7357` is merged, `#7361` has been rebased onto current `main`, and `#7358` / `#7360` are independently mergeable.

ADC v0.7 may be dispatched once the operator merges the base stack (`#7361`, then `#7358` and `#7360`). The merge decision is operator-only; no agent should auto-merge.

## Operator decision needed

Merge the base stack in order `#7361 → {#7358, #7360}`, then authorize v0.7 dispatch by running the prepared invocation at `.aragora/v16-dispatch/dispatch-adc-v0.7.sh`.

No merges, labels, ready-state flips, or v0.7 dispatches were performed by this packet.
