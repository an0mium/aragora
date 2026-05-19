# ADC Parked Artifact Preservation — 2026-05-19

**Status:** preservation packet only
**Source worktree:** `/Users/armand/Development/aragora/.worktrees/codex-auto/droid-20260519-151711-92e43188`
**Target branch:** `codex/adc-follow-on-deepening-20260519`
**Operator gate:** no v0.7/v0.8 dispatch until `#7361 → {#7358, #7360}` lands on `main`

## Purpose

Claude could not physically inspect Factory's gitignored v0.7/v0.8 artifacts from its checkout. This packet records where those artifacts were found, their hashes, their validation status, and enough safe reconstruction detail that the operator can recover them later without relying on hidden workspace memory.

No ignored executable artifact is committed by this packet. This document is a non-executable preservation/reconstruction note.

## Located artifacts

| Artifact | Located | SHA-256 | Validation |
|---|---:|---|---|
| `.aragora/v16-dispatch/dispatch-adc-v0.7.sh` | yes | `03751114711df883bb7650b2b6533a42aa6a570c10d63d9fc5b140fc7a96a48e` | `bash -n` passed |
| `.aragora/v16-dispatch/ADC-v0.8-envelope-and-validator.md` | yes | `15e2821cf13655b80f13204c156e348e3d30acb7d40b28556c3c0080f4e680e5` | read-only inspected |
| `.aragora/v16-dispatch/ADC-v0.8-droid-adapter.md` | yes | `b2930117f7c7f5f8970e649299b9ed98d0c776536f7f2f5f2c06766ac41cba15` | read-only inspected |
| `.aragora/v16-dispatch/ADC-v0.8-claude-codex-stubs.md` | yes | `e5f34ef1a582b0af724a285f66e50052cd6b199f4f784e8f17b8d32b0e857ea1` | read-only inspected |

## v0.7 dispatch helper behavior

The helper is intentionally interactive and fail-closed:

1. resolves `REPO_ROOT` from `.aragora/v16-dispatch`;
2. points at local hardened prompt `$HOME/.factory/specs/2026-05-19-adc-v0.7-three-tier-reversibility-droid-prompt.md`;
3. refuses unless PRs `#7361`, `#7358`, and `#7360` are `MERGED`;
4. refuses if the prompt is missing;
5. refuses if `.aragora/v13-dispatch/launch_lane.sh` is missing or non-executable;
6. prints the exact launch command;
7. requires the operator to type `DISPATCH`;
8. only then runs `bash .aragora/v13-dispatch/launch_lane.sh ADC-v0.7-three-tier-reversibility <prompt>`.

Reconstruction script:

```bash
mkdir -p .aragora/v16-dispatch
cat > .aragora/v16-dispatch/dispatch-adc-v0.7.sh <<'EOF'
#!/usr/bin/env bash
# Operator instruction: run this only AFTER #7361, #7358, and #7360 are merged to main.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROMPT="$HOME/.factory/specs/2026-05-19-adc-v0.7-three-tier-reversibility-droid-prompt.md"
LANE_ID="ADC-v0.7-three-tier-reversibility"

cd "$REPO_ROOT"

echo "ADC v0.7 dispatch preview"
echo "repo: $REPO_ROOT"
echo "lane: $LANE_ID"
echo "prompt: $PROMPT"
echo
echo "Checking base stack is merged..."

for pr in 7361 7358 7360; do
  state="$(gh pr view "$pr" --json state --jq .state)"
  if [[ "$state" != "MERGED" ]]; then
    echo "REFUSING: PR #$pr is $state, expected MERGED." >&2
    exit 1
  fi
done

if [[ ! -f "$PROMPT" ]]; then
  echo "REFUSING: prompt missing: $PROMPT" >&2
  exit 1
fi

if [[ ! -x ".aragora/v13-dispatch/launch_lane.sh" ]]; then
  echo "REFUSING: launcher missing or not executable: .aragora/v13-dispatch/launch_lane.sh" >&2
  exit 1
fi

echo "Base stack merged. This will launch one droid worker via:"
echo "  bash .aragora/v13-dispatch/launch_lane.sh $LANE_ID $PROMPT"
echo
read -r -p "Type DISPATCH to launch ADC v0.7: " answer
if [[ "$answer" != "DISPATCH" ]]; then
  echo "Aborted."
  exit 0
fi

bash .aragora/v13-dispatch/launch_lane.sh "$LANE_ID" "$PROMPT"
EOF
chmod +x .aragora/v16-dispatch/dispatch-adc-v0.7.sh
bash -n .aragora/v16-dispatch/dispatch-adc-v0.7.sh
```

## v0.8 prompt pack summary

All three parked v0.8 prompts carry the same hard gate:

> Do not dispatch until ADC v0.7 has shipped or the operator explicitly waives that prerequisite.

### Envelope + validator

Lane: `ADC-v0.8-envelope-and-validator`

Safe deliverables captured:

- `aragora/policy/contract_envelope.py` with `ContractEnvelope`, canonical JSON load/dump, schema validation, signing verification via ADC v0.4, and lifecycle refusal via ADC v0.7.
- `scripts/validate_parent_contract.py` with `--parent-contract PATH`, `--json`, and clear non-zero exits for invalid, paused, halted, revoked, or unsigned-disallowed contracts.
- Tests for valid signed envelopes, unsigned dry-run envelopes, malformed schema, permission intersection, and paused/halted/revoked refusal.

Important drift note already fixed in the parked prompt:

> `docs/governance/ADC_v0.8_CROSS_FAMILY_ADAPTER_PLAN.md` is carried by PR #7367 and is not on `main` until #7367 lands. If this prompt is dispatched before #7367 merges, copy the plan document into the integration branch/worktree first or keep this prompt self-contained.

### Droid adapter

Lane: `ADC-v0.8-droid-adapter`

Safe deliverables captured:

- `scripts/launch_contract_bound_droid.py`;
- flags `--lane-id`, `--prompt`, `--parent-contract`, `--dry-run`;
- fail-closed parent envelope validation before launch;
- effective permission intersection;
- copied `ADC_PARENT_CONTRACT.json` in child worktree;
- exported `ARAGORA_PARENT_CONTRACT`, `ARAGORA_DELEGATION_CONTRACT_ID`, `ARAGORA_GOAL_ID`, `ARAGORA_CONTRACT_ENVELOPE_ID`;
- worker-instruction prelude requiring lane claim and receipt binding.

### Claude/Codex stubs

Lane: `ADC-v0.8-claude-codex-stubs`

Safe deliverables captured:

- validation examples for Claude Code startup, Codex CLI worker startup, and Codex Desktop/rollout metadata binding where available;
- explicit limitation text: these stubs validate and record scope but do not intercept every tool call unless the underlying harness provides that boundary;
- tests for validation behavior and receipt-field extraction.

## Verified symbol references

Current `main` provides:

- `DelegationContract`
- `GoalSpec`
- `AcceptanceCriterion`
- `AllowedSurfaces`
- `ContractBudget`
- `ContractValidationError`
- `narrow_for_child`
- `make_root_contract`
- predicate-oracle helpers including `parse_predicate` and `evaluate_predicate`

Rebased `#7361` provides:

- `SIGNING_SCHEMA_VERSION`
- `SigningError`
- `VerificationError`
- `VerificationResult`
- `canonical_contract_payload`
- `sign_contract`
- `verify_contract`
- `sign_receipt`
- `verify_receipt`
- `is_contract_signed`
- `signing_key_available`

Planned v0.7 symbols used by v0.8 prompts:

- `get_contract_state`
- `ContractState`

These v0.7 symbols do not exist on `main` yet and must not be referenced by executable code until the lifecycle PR lands or the implementation guards imports behind explicit dependency checks.

## Recovery checklist

1. Confirm `#7361`, `#7358`, and `#7360` are merged to `main`.
2. Recreate `.aragora/v16-dispatch/dispatch-adc-v0.7.sh` from the reconstruction block above if the prior worktree has been removed.
3. Confirm the hardened local v0.7 prompt exists at `$HOME/.factory/specs/2026-05-19-adc-v0.7-three-tier-reversibility-droid-prompt.md`.
4. Run `bash -n .aragora/v16-dispatch/dispatch-adc-v0.7.sh`.
5. Review the staged command printed by the helper.
6. Type `DISPATCH` only when the operator intentionally authorizes v0.7.

## Boundaries observed

This preservation pass did not execute the dispatch helper, did not create a child worker, did not modify ignored artifacts, did not mutate GitHub, and did not edit protected files.
