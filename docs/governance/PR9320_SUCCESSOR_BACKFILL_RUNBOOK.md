# PR #9320 Successor Historical-Backfill Runbook

Status: preparation only. This document does not authorize workflow dispatch, Tier-4 settlement,
merge, immutable publication, attestation, or a claim that the successor capsule is authenticated.
Those terminal actions belong to `cdg-9320-successor-backfill-capsule-finalization` after a later
exact PR/head/receipt/release-byte delegation.

## Frozen historical identity

| Field | Exact value |
|---|---|
| Repository | `synaptent/aragora` |
| PR | `9320` |
| Recorded PR base | `14d1ef53e23c5466c0491ed93f72752944c78cd4` |
| PR head | `aba6b14c94eca3a9c825b1a303ea67684d5f8daa` |
| Squash merge | `0b28f68b9f4d204ae14814169093723ea84c1364` |
| Merge first parent | `e448b840dad03ee28accd218c14a27fa8b87c7b4` |
| Head tree | `e5c6c3d07a918cf43fffed6d4a9f472bc10a674a` |
| Merge tree | `79c1c374eed261c42468dc526d837e726e73425a` |
| First-parent patch | `5874` bytes, SHA-256 `a5c94ff5c9d32a60c055d5ae67b21935dd7f98aae6f868ab1d68e300bb604455` |
| Old immutable release | API ID `363450207`, tag `backfill-0b28f68b9f4d204ae14814169093723ea84c1364` |
| Successor tag | `backfill-v2-0b28f68b9f4d204ae14814169093723ea84c1364` |

Release `363450207` is immutable superseded historical evidence. Never edit, replace, delete,
reuse, or upload to it.

## Preparation outputs

The repository provides:

- `scripts/build_contract_drift_historical_backfill.py`, the canonical builder/verifier;
- `scripts/schemas/contract-drift-historical-backfill-capsule-v2.schema.json`, the strict payload
  schema;
- an exact-pair `workflow_dispatch` path in `Contract Drift Governance` that fetches only
  `refs/pull/9320/head`, verifies it equals the supplied immutable head SHA, and uploads the
  successful raw analyzer result;
- a completion-triggered `Contract Drift Historical Backfill Finalizer` that runs only after the
  producer workflow is a successful completed `workflow_dispatch`, downloads that exact run's
  analyzer artifact, authenticates the completed producer run/attempt/job/check plus the six
  historical contexts, and uploads the complete canonical receipt envelope;
- `backfill-v2-<merge_sha>` support in the manual `actions/attest@v4` signer workflow.

The builder emits exactly `manifest.json`, `payload.json`, and `checksums.txt`. Canonical JSON is
UTF-8 without BOM, compact sorted-key serialization, and exactly one terminal LF.

## Finalization checkpoint P0: exact authority

Before any remote mutation, record:

1. the parked Tier-4 implementation PR number, exact base SHA, exact head SHA, and complete changed
   paths;
2. a current unconsumed operator delegation naming that exact pair and the authorized actions;
3. `origin/main`, repository future-release immutability, authenticated `Administration:read`, and
   rule-suite access;
4. the OpenAPI release-envelope finalization terminal evidence required by the feature precondition;
5. release `363450207` identity and unchanged immutable status.

Any mismatch or movement stops the procedure.

## Finalization checkpoint P1: settle and merge implementation

Use the repository Tier-4 chronology only:

1. authenticate exact base/head, checks, discussion, and evidence;
2. post only delegated byte-exact model evidence;
3. run the pre-settlement quorum;
4. perform one delegated settle-only human settlement;
5. obtain a distinct post-settlement quorum success;
6. require the final `settle_tier4_pr.py --check` result to pass;
7. re-read `origin/main == BASE_SHA`;
8. merge normally, non-admin, with `--match-head-commit HEAD_SHA`;
9. prove merge first parent equals `BASE_SHA` and merge tree equals the reviewed head tree.

Do not continue from a partial or moved state. Persist the merged implementation SHA as the next
checkpoint input.

## Finalization checkpoint P2: execute the exact historical receipt

Dispatch the merged `Contract Drift Governance` producer workflow at the exact merged
implementation SHA with:

```text
historical_backfill=true
historical_base_sha=14d1ef53e23c5466c0491ed93f72752944c78cd4
historical_head_sha=aba6b14c94eca3a9c825b1a303ea67684d5f8daa
historical_merge_sha=0b28f68b9f4d204ae14814169093723ea84c1364
historical_first_parent_sha=e448b840dad03ee28accd218c14a27fa8b87c7b4
```

This preparation feature must not run that dispatch. The completion-triggered finalizer must then
run only after the producer is complete and successful. It must not attempt to authenticate its own
in-progress job as already successful. The publication finalizer must discover workflow runs without
conclusion filters, reconcile pagination, enumerate attempts, select the intended producer run, and
bind:

- workflow run ID and run attempt;
- attempt-specific job ID and check-run ID;
- `contract-drift-main-receipt-0b28f68b9f4d204ae14814169093723ea84c1364`;
- the receipt `source_sha`, which is the exact merged implementation/checker ref dispatched for
  the successful run and is intentionally distinct from the historical `merge_sha`;
- exact base/head/merge/first-parent tuple, both tree identities, byte-identical
  base-to-head and first-parent-to-merge patch digest/length, and semantic delta paths;
- all six successful historical required-context run/attempt/job/check/app identities.

The completion-triggered finalizer's uploaded `contract-drift-main-receipt.json` is itself the canonical
`contract-drift-historical-backfill-receipt-v1` envelope and can be inserted directly as the
builder input's `receipt` object. Persist those exact downloaded bytes and their digest. A failed,
cancelled, incomplete, missing-finalizer, raw-analyzer-only, or generic push receipt is terminal
failure.

## Finalization checkpoint P3: draft release and deterministic bytes

Only if the final payload requires `release_api_id`, create one unpublished draft release using the
distinct successor tag, then freeze its API ID. Never publish the draft in this checkpoint.

Build the final input document with:

- the successful P2 receipt identity, including `receipt.source_sha == authority_source_sha` and
  `receipt.merge_sha == 0b28f68b9f4d204ae14814169093723ea84c1364`;
- current exact-ref authority, dependency, inventory, public-symbol, route-boundary, category,
  original-ID, projection-schema, SDK-provenance, and SDK/core/extended partition digests;
- all `655` projection memberships and all `666` method-specific edges;
- a new passing rule-suite record for `refs/heads/main`;
- stable attestation identity claims for `actions/attest@v4`;
- non-precedential `historical_nonconforming` disposition;
- explicit supersession of release `363450207`.

Build twice into two nonexistent output paths. The builder creates each directory exclusively and
rejects a pre-existing path, extra file, symlink, or rogue subdirectory:

```bash
python3 -B scripts/build_contract_drift_historical_backfill.py \
  --repo-root "$REPO" --input "$INPUT" --authority-manifest "$AUTHORITY_MANIFEST" \
  --output-dir "$RUN1" --json
python3 -B scripts/build_contract_drift_historical_backfill.py \
  --repo-root "$REPO" --input "$INPUT" --authority-manifest "$AUTHORITY_MANIFEST" \
  --output-dir "$RUN2" --json
cmp "$RUN1/manifest.json" "$RUN2/manifest.json"
cmp "$RUN1/payload.json" "$RUN2/payload.json"
cmp "$RUN1/checksums.txt" "$RUN2/checksums.txt"
```

Record byte lengths and SHA-256 values for all three assets. Do not upload until the later
delegation names these exact bytes.

## Finalization checkpoint P4: immutable publication

Under an exact-byte publication delegation only:

1. re-query future release immutability and require `enabled=true`;
2. require the successor tag to resolve exactly to the historical merge SHA;
3. upload exactly the three frozen assets to the unpublished draft;
4. re-download and byte-compare every asset;
5. publish the successor release once, immutable, with no tag or asset reuse;
6. run `gh release verify` twice and `gh release verify-asset` for each asset twice;
7. capture the release API ID, asset API IDs, tag target, byte lengths, and digests.

No old release operation is permitted.

## Finalization checkpoint P5: attestation and rule suite

Dispatch `.github/workflows/contract-drift-boundary.yml` at the published successor tag only under
the later delegation. Require:

- `actions/attest@v4`;
- signer workflow `synaptent/aragora/.github/workflows/contract-drift-boundary.yml`;
- source digest `0b28f68b9f4d204ae14814169093723ea84c1364`;
- predicate type `https://in-toto.io/attestation/release/v0.2`;
- repository `synaptent/aragora`;
- exact subject SHA-256 values for the three frozen assets.

Run `gh attestation verify` twice per asset. Capture the passing rule-suite ID/result immediately
and require its repository, `refs/heads/main`, and `after_sha` to match the payload.

## Finalization checkpoint P6: negative probes

All probes must fail closed:

- omitted required payload field;
- failed or cancelled receipt;
- absent/unverified attestation;
- payload, manifest, or checksum tamper;
- wrong subject digest;
- wrong signer workflow or signer SAN;
- wrong repository;
- failed, bypassed, or mismatched rule suite;
- incomplete/mutated projection edge set;
- release/tag/asset replacement, deletion, or reuse;
- movement of `main`, selected workflow run, or run attempt.

Never perform a destructive probe against release `363450207`.

## Finalization checkpoint P7: movement requery and terminal record

After all verification, re-query:

- current `main`;
- newest relevant workflow run and attempt;
- successor release/tag/assets;
- attestation subjects;
- rule-suite identity/result.

If any identity moved, restart from the last safe checkpoint rather than normalizing the movement
away. Only after stable equality may the finalizer write a terminal authenticated successor-capsule
record for the successor boundary. Preparation fixtures, synthetic IDs, or this runbook are not that
terminal record.
