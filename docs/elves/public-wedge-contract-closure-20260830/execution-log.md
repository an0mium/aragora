# Public Wedge Contract-Closure Execution Log

## 2026-08-30 - Staging

- Created persistent Codex goal `019f807d-64fd-7502-8043-9c42ec27be88` for the full campaign.
- Read the Elves skill, root operating instructions, §Conductor, review authority, and coordination
  guidance before taking repository actions.
- Refreshed `origin/main` to `eaaac1a07480b64d3ba4a060fbf36773ae36e589`.
- Observed the shared checkout only: it is dirty and `main` is ahead 1/behind 120; no shared-root
  state was modified.
- Created fresh worktree
  `$HOME/.codex/worktrees/public-wedge-contract-closure-b1-20260830/aragora` and branch
  `codex/public-wedge-contract-closure-b1-20260830` from that exact base.
- Claimed agent-bridge lane `codex-public-wedge-contract-closure-b1-20260830` and work lease
  `1f1ea372-286` for the current Codex session.
- Refreshed active tasks, lanes, processes, open PRs/files, reviewer reservations, required checks,
  disk, and outbox. No expected Batch 1 file overlap was found. Broad-writer activation was not
  attempted because the outbox is non-empty and unrelated to this unit.
- Recorded the previously attempted bounded Fable goal-cycle as unavailable after its permitted
  attempts failed. It is not being retried this cycle and supplied no authority.
- Ran Elves install doctor: installed 1.12.0, available 2.33.0. Chose not to change tooling during
  the staged run.
- Ran clean-main focused baseline:
  - `tests/cli/test_receipt_roundtrip.py`: 3 passed.
  - `aragora-verify/tests/test_cli.py` plus `test_example_live_receipt.py`: 8 passed.
- Ran the public chain in a temporary directory with provider keys removed:
  - offline demo -> native receipt: exit 0;
  - native receipt verify: exit 0;
  - ODR export: exit 0;
  - in-repo `aragora-verify` 0.1.2 wheel build/install: exit 0;
  - standalone ODR verify: exit 0.
- Identified the first unowned gap: these stages are tested separately against different fixtures,
  but no regression follows one demo-produced artifact through the entire public chain or its
  tampered counterpart.
- No product files were edited during staging.

## Decisions made

- Treat Batch 1 as a tests-first contract closure, not a product-code fix. Runtime behavior is
  currently correct, but the user-visible guarantee can regress undetected across seams.
- Keep expected product scope to one existing test file. Any discovered need for product code gets
  a new overlap/risk check before editing.
- Do not move the pre-existing global `elves/pre-batch-1` tag. Use a campaign-specific rollback
  tag after the fresh launch.
- Follow the Elves two-call boundary: stage and open the draft PR now, then wait for the user's
  short launch prompt before product implementation.

## Validation receipts

| Receipt | Result |
|---|---|
| GitHub authentication | active account `scarmani`, repository access healthy |
| Branch protection visibility | six required contexts visible |
| Required current-main checks | five non-quorum required checks green; main-only quorum skipped |
| Open-PR expected-file intersection | none |
| Work lease | held by current session, lease `1f1ea372-286` |
| Manual composed runtime | pass |
| Product edits | none during staging |

## Next

- Committed and pushed the staging packet as
  `ac8d546127bc42de1ced6920fbd5e5194a889c96`; all commit and push hooks passed.
- Opened draft PR #9903, `test(cli): prove clean offline receipt round trip`.
- No product implementation or evidence collection occurred during staging.

Stop at the mandatory fresh-launch boundary. On the next user call, re-read the full durable
packet, set the Stop Gate to `no`, reverify live ownership/head/overlap/main health, create the
unique rollback tag, and implement Batch 1.

## 2026-08-30 - Batch 1 implementation and validation

- Re-read the full launch packet and operator steering, renewed lease `1f1ea372-286`, and verified
  the local and remote tips descended only through the self-created staging metadata commit.
- Refreshed `origin/main`, all required main checks, open-PR changed-file overlap, reviewer
  reservations, disk, processes, and outbox. Main remained healthy and no PR intersected the
  expected Batch 1 paths.
- Created and pushed rollback tag
  `elves/public-wedge-contract-closure-b1/pre-batch-1` at
  `68621d77b24572d6517761170a4e25f3614ab36c` without moving the older global tag.
- Added one composed clean-install contract to `tests/cli/test_receipt_roundtrip.py`. It builds and
  installs the local root and verifier wheels, clears provider/signing/AWS credentials, runs from
  outside the checkout, and follows one receipt through offline demo, native verification, ODR
  export, and standalone JSON verification with receipt/hash continuity checks.
- The first break mutation changed the unsigned ODR verdict to another non-empty string. The
  verifier correctly accepted that schema-valid unsigned artifact, demonstrating that unsigned
  structure validation is not authenticity. Replaced it with removal of required
  `claim.verdict`; the installed verifier exits 1 and reports schema failure. No product/verifier
  logic or PR #9015 path was changed.
- Validation receipts:
  - new composed contract: 1 passed from locally built and isolated wheels;
  - root receipt/export/walkthrough/onboarding/verify suites: 126 passed;
  - standalone verifier CLI/example suites: 8 passed from its `src/` package boundary;
  - Ruff formatting and lint: passed;
  - CI-equivalent required gates: lint, shrink-only mypy baseline, version alignment, SDK parity,
    namespace parity, cross-SDK parity, generated OpenAPI SDK contracts, and route validation all
    passed.
- Product scope remains one existing test file, 237 added lines; the durable plan is the only
  additional final tracked file. Independent review, final cleanup, exact-head CI/evidence,
  settlement, and landing remain.
