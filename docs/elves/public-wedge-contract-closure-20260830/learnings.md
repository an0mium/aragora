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
