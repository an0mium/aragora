# ADC Operator Post-Merge Runbook — 2026-05-19

**Status:** operator runbook
**Scope:** what to do after `#7361`, `#7358`, and `#7360` land
**No automation executed:** this document is a checklist, not a dispatch.

## Decision gate

Current gate remains:

```text
merge/review #7361 → re-check #7358/#7360 → merge/review #7358 and #7360 → authorize v0.7
```

Do not run v0.7 dispatch helpers before the base stack is present on `main`.

## Pre-merge read-only checks

```bash
for pr in 7361 7358 7360 7367; do
  gh pr view "$pr" --json number,state,isDraft,headRefOid,mergeable,mergeStateStatus,reviewDecision,title,url
  gh pr checks "$pr"
done
```

Expected before merge:

- `#7361` head `9093ddbc48ac4dd2ccaa3364b527b660c089a41e`
- `#7358` head `d926a9749f23b8ac097a2ec8573df7e63a11f738`
- `#7360` head `7b0cf9ea426fe2cc78b3e7298f1db7618931a8db`
- no failing checks
- review/merge blocks only

## Merge step 1 — v0.4 signing

Merge/review `#7361`.

After it lands:

```bash
git fetch origin main
git checkout main
git pull --ff-only origin main

python -m py_compile aragora/policy/contract_signing.py scripts/sign_delegation_contract.py
env -u ARAGORA_CONTEXT_SIGNING_KEY python -m pytest \
  tests/policy/test_contract_signing.py \
  tests/policy/test_delegation_contract.py \
  tests/scripts/test_sign_delegation_contract.py \
  -q
```

If these fail, stop before merging v0.2/v0.3 and ask the v0.4 owner to repair.

## Re-check downstream PRs

```bash
for pr in 7358 7360; do
  gh pr view "$pr" --json number,state,isDraft,headRefOid,mergeable,mergeStateStatus,reviewDecision,title,url
  gh pr checks "$pr"
done
```

If either is no longer `MERGEABLE` or has failing checks:

1. do not merge it;
2. ask its owner to rebase/repair only that branch;
3. re-run the post-repair checks.

## Merge step 2 — v0.2 lane registry

Merge/review `#7358` once it is clean.

Post-merge:

```bash
git fetch origin main
git checkout main
git pull --ff-only origin main

python -m py_compile scripts/claim_active_agent_lane.py
python -m pytest \
  tests/scripts/test_claim_active_agent_lane.py \
  tests/policy/test_delegation_contract.py \
  tests/policy/test_predicate_oracle.py \
  -q
```

If the test suite creates lane registry artifacts, inspect them before cleanup and do not delete untracked files unless they are exact temp/test artifacts and the operator approves.

## Merge step 3 — v0.3 progress ledger

Merge/review `#7360` once it is clean.

Post-merge:

```bash
git fetch origin main
git checkout main
git pull --ff-only origin main

python -m py_compile scripts/evaluate_goal_progress.py
python -m pytest \
  tests/scripts/test_evaluate_goal_progress.py \
  tests/policy/test_predicate_oracle.py \
  tests/policy/test_delegation_contract.py \
  -q
```

Confirm dry-run mode does not write `.aragora/progress-ledger/*.jsonl` unless `--apply` is explicitly passed.

## Final base-stack smoke

```bash
python -m pytest \
  tests/policy/test_delegation_contract.py \
  tests/policy/test_predicate_oracle.py \
  tests/policy/test_contract_signing.py \
  tests/scripts/test_claim_active_agent_lane.py \
  tests/scripts/test_evaluate_goal_progress.py \
  tests/scripts/test_sign_delegation_contract.py \
  -q
```

Optional import smoke:

```bash
python - <<'PY'
from aragora.policy import DelegationContract, GoalSpec, evaluate_predicate
from aragora.policy import sign_contract, verify_contract, sign_receipt, verify_receipt
print("ADC base stack imports ok")
PY
```

## v0.7 dispatch preflight

Only after the final base-stack smoke passes:

1. locate or recreate `.aragora/v16-dispatch/dispatch-adc-v0.7.sh`;
2. run `bash -n .aragora/v16-dispatch/dispatch-adc-v0.7.sh`;
3. ensure local prompt exists:

```bash
test -f "$HOME/.factory/specs/2026-05-19-adc-v0.7-three-tier-reversibility-droid-prompt.md"
```

4. inspect the helper output;
5. run it only if the operator is intentionally authorizing v0.7.

The helper should refuse unless all three base PRs are `MERGED`.

## Rollback routing

If a post-merge issue is isolated:

| Failing surface | First rollback candidate |
|---|---|
| signing/import/signature validation | `#7361` |
| lane claim fields/registry metadata | `#7358` |
| progress ticks/ledger application | `#7360` |
| docs only | `#7367` or follow-on docs PR |

If all three landed and the issue is systemic, revert newest causal merge first, then work backward. Do not delete generated lane/progress JSONL artifacts without explicit operator approval.

## Stop conditions

Stop and request operator direction if:

- a post-merge validation command fails;
- a downstream PR becomes `DIRTY`/`CONFLICTING`;
- signing behavior depends on a secret only present on one machine;
- the v0.7 helper is missing and cannot be reconstructed from the preservation doc;
- any automation proposes a merge, label, mark-ready, force-push, or dispatch without explicit operator approval.
