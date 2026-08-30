# Public Wedge Contract-Closure Learnings

## Durable facts

- A component-level receipt test is not proof of the public wedge. The same artifact must cross
  every boundary: producer -> native verifier -> ODR exporter -> independently installed verifier.
- `aragora receipt verify` validates the native `DecisionReceipt`; `aragora-verify` validates the
  portable ODR. Treating those names as interchangeable conceals a real contract boundary.
- The offline demo exports a valid unsigned ODR on a machine without signing configuration.
  Standalone verification without `--pubkey` exits `0` for schema/digest/quorum correctness while
  truthfully warning that authenticity is not established.
- For a local verifier wheel, pass the absolute `aragora-verify/` directory (or
  `./aragora-verify`) to pip. The bare token `aragora-verify` resolves the published PyPI package
  and can silently test an older version.
- Open-PR file intersection, not a process sighting by itself, is the useful collision test. At
  staging, no open PR touched `tests/cli/test_receipt_roundtrip.py`, `aragora/cli/demo.py`,
  `aragora/cli/commands/receipt.py`, or the walkthrough contract test.

## Staging evidence

- Base: `eaaac1a07480b64d3ba4a060fbf36773ae36e589`.
- Existing native round-trip test: 3 passed.
- Existing standalone CLI/example tests: 8 passed.
- Manual one-artifact chain from a temporary directory: all application stages exited `0`.
- In-repo standalone verifier wheel: `aragora_verify-0.1.2-py3-none-any.whl`; build, isolated
  target install, and verification all exited `0`.
- Current main required contexts passed at staging. A separate Contract Drift Governance job failed
  in `contract-drift-program-trajectory`, but that context is not branch-protection-required.

## Scope guards

- PR #9015 owns in-repo/standalone ODR verifier parity; this batch must not alter verifier parity
  logic.
- PR #9894 owns keyless-doctor CLI paths; this batch must not touch `aragora/cli/doctor.py` or its
  tests.
- PR #9112 owns SDK modes/spectate work. SDK parity is a later matrix cell after a fresh overlap
  check, not part of Batch 1.
- Decision-quality, recurring-status, deployment, governance, and merge/evidence work are active
  elsewhere and excluded from the campaign.

## Batch 1 implementation facts

- A schema-valid edit to an unsigned ODR cannot be authenticated: changing the free-form verdict
  value still verifies because the canonical digest is recomputed and there is no signature to
  bind it. Batch 1 therefore proves structural fail-closed behavior by removing the required
  `claim.verdict`; signed authenticity remains journey cell 4 and outside this PR.
- `aragora-verify` uses a `src/` package layout. Direct package tests require
  `PYTHONPATH=src`; running them from the monorepo root without that boundary fails collection and
  is a harness error, not a verifier failure.
- The composed test builds both local wheels with `--no-deps`, installs them to a temporary target,
  probes both module paths, and runs every public command from an unrelated temporary directory.
- Root focused regressions passed 126 tests; standalone verifier CLI/example regressions passed 8.
- The CI-equivalent required suite passed lint, shrink-only mypy baseline, version alignment, all
  three SDK parity checks, generated OpenAPI contract verification, and route validation.
