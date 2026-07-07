# Settlement Queue Triage Snapshot

Generated: 2026-07-07T21:44:13Z

This snapshot composes with epic #8761, the repeat-blocker park policy landed in
#8990, and the draft-skip filter in #8987. It is documentation only: no PR was
merged, marked ready, commented on, labeled, or rerun while collecting this
state.

Disposition vocabulary comes from
[`2026-07-07-repeat-blocker-park-policy.md`](2026-07-07-repeat-blocker-park-policy.md):
`settle-next`, `repair-exact-blocker`, `park-current-head`,
`defer-active-owner`, `policy-excluded`, and
`superseded-or-close-candidate`.

Next allowed actions use the policy enum: `new head required`,
`repair exact blocker`, `operator override required`,
`safe to retry after transport recovery`, `superseded by replacement PR`, and
`close as obsolete`.

## Required-Check Legend

- `5/5 green + quorum fail`: `lint`, `typecheck`, `sdk-parity`,
  `Generate & Validate`, and `TypeScript SDK Type Check` are green; only
  `aragora-merge-quorum` is failing.
- `no checks reported`: `gh pr checks --required --json` returned no required
  checks for the branch.
- `dirty`: GitHub reports merge conflicts or a dirty merge state even if
  required checks are otherwise green.

## Queue Table

| PR | Title | Branch | Exact head SHA | Required-check rollup | Disposition | Highest known blocker | Next allowed action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| #8995 | fix(governance): ignore nested diff fixture literals | `codex/charter-checker-fixture-precision-20260707` | `09fa08ef8a4edc310e2cf3b61b429ba11253c541` | 5/5 green + quorum fail | `settle-next` | Needs targeted merge-packet, no-dissent check, and exact-head quorum/evidence verification before any settlement prompt. | `operator override required` |
| #8992 | feat(scripts): settlement gate-preflight classifier + runbook | `codex/settlement-preflight-classifier-20260707` | `8ba86322f058290396b4127b0bb56b0c27b1d38f` | 5/5 green + quorum fail | `policy-excluded` | Settlement/steward tooling changes are merge-authority-adjacent; treat as elevated until exact Tier/gate packet proves otherwise. | `operator override required` |
| #8991 | docs(landing): canonicalize public-utility landing + Action setup doc (m6-docs) | `factory/pum-m6-canonical-verbs-and-landing` | `4b8266594166e0bcf6a00bd14c0e4202a488d43e` | 5/5 green + quorum fail | `settle-next` | Needs targeted merge-packet, no-dissent check, and current-head evidence/quorum verification. | `operator override required` |
| #8987 | fix(steward): skip draft PRs during broad settlement selection | `codex/settle-one-skip-drafts-20260707T181215Z` | `0d30c5114c5dbf5a011f53225581cc41d1b6f1bf` | 5/5 green + quorum fail | `policy-excluded` | Settlement selection tooling; do not treat as routine drain without targeted Tier/human-risk confirmation. | `operator override required` |
| #8970 | fix(packaging): raise aragora-verify cryptography floor to >=48.0.1 (m5-install) | `factory/pum-m5-packaging-deps-pr` | `905b3dcfb3bb5e028d712519259c8b58eabc1e90` | 5/5 green + quorum fail | `policy-excluded` | Security/packaging labels plus `operator-review-required`; not a normal-policy queue-drain target. | `operator override required` |
| #8965 | Gate admin squash on live PR state | `codex/review-queue-live-gate-admin-squash-fix` | `babad542405803e3a6b2bc2788a1f52738074d7c` | 5/5 green + quorum fail | `policy-excluded` | Merge-authority gate behavior; Tier-4 treatment unless exact helper packet proves a narrower path. | `operator override required` |
| #8961 | feat(scripts): consolidate founder decision queue across comments and packets | `codex/founder-decision-queue-complete-pending` | `13bf57d45b778e2644cdb55aa09edfdfa9b7dbae` | 5/5 green + quorum fail; dirty | `repair-exact-blocker` | GitHub reports `mergeable=CONFLICTING` and `mergeStateStatus=DIRTY`. | `repair exact blocker` |
| #8948 | feat(scripts): write prompt handoffs to automation outbox | `codex/prompt-handoff-outbox-20260706T1541Z` | `be56200ae6a19c79263c492741c68173eddbf669` | 5/5 green + quorum fail | `park-current-head` | Head still matches the repeat-blocker policy example with Claude timeout/OpenAI `CHANGES-REQUESTED` and a blocking P1. | `repair exact blocker` |
| #8945 | feat(release): verify aragora-verify public install after publish | `codex/verify-post-publish-aragora-verify-20260706T1518Z` | `7b30e8ffbec35a4027d5bc70123b3abf9ca50208` | 5/5 green + quorum fail | `policy-excluded` | Release-flow verification surface; approval-required under the operating contract. | `operator override required` |
| #8931 | fix(settlement): park Dependabot policy exclusions | `codex/settle-policy-park-dependabot-20260706` | `8adc0f8e6a4857a36be8bc0facdb4ab15f28bd12` | 5/5 green + quorum fail | `policy-excluded` | Settlement policy / Dependabot exclusion behavior; not a normal drain target. | `operator override required` |
| #8924 | chore(deps): Update uvicorn requirement from <1.0,>=0.49.0 to >=0.50.0,<1.0 | `dependabot/pip/uvicorn-gte-0.50.0-and-lt-1.0` | `416fb013bd4a0157eb4057a01a63ea5ed42d6408` | no checks reported | `policy-excluded` | Dependabot-origin policy exclusion pending the #8931 path. | `operator override required` |
| #8923 | chore(deps): Update fastapi requirement from <1.0,>=0.138.0 to >=0.139.0,<1.0 | `dependabot/pip/fastapi-gte-0.139.0-and-lt-1.0` | `48ec932139324ba41fd0a110d823d072f64312f6` | no checks reported | `policy-excluded` | Dependabot-origin policy exclusion pending the #8931 path. | `operator override required` |
| #8922 | chore(deps): Update playwright requirement from <2.0,>=1.60.0 to >=1.61.0,<2.0 | `dependabot/pip/playwright-gte-1.61.0-and-lt-2.0` | `3311a665b1b24b556b9bd1529b5cd60f5f9ef72e` | no checks reported | `policy-excluded` | Dependabot-origin policy exclusion pending the #8931 path. | `operator override required` |
| #8921 | chore(deps): Update hatchling requirement from <2.0,>=1.18 to >=1.30.1,<2.0 | `dependabot/pip/hatchling-gte-1.30.1-and-lt-2.0` | `d826520ae9a5e25465cba3013e2fee8894dc9edf` | no checks reported | `policy-excluded` | Dependabot-origin policy exclusion pending the #8931 path. | `operator override required` |
| #8920 | chore(live): bump supabase-js and harden frontend deploy context | `dependabot/npm_and_yarn/aragora/live/supabase/supabase-js-2.110.0` | `6f4ae553e18df22a69fcdf55c430b122e6036e0e` | 5/5 green + quorum fail; dirty | `policy-excluded` | Dependabot-origin and `mergeStateStatus=DIRTY`. | `operator override required` |
| #8917 | chore(deps): Bump the sdk-deps group in /sdk/typescript with 7 updates | `dependabot/npm_and_yarn/sdk/typescript/sdk-deps-f002f4927b` | `0d578344432f9f229c2f5e3b5808c95a5a0153cb` | 5/5 green + quorum fail | `policy-excluded` | Dependabot-origin policy exclusion pending the #8931 path. | `operator override required` |
| #8915 | chore(deps): Bump eslint-config-next from 16.2.6 to 16.2.10 in /aragora/live | `dependabot/npm_and_yarn/aragora/live/eslint-config-next-16.2.10` | `2466c85a9673cfc08eb585ddbca8ddc9146995ba` | 5/5 green + quorum fail | `policy-excluded` | Dependabot-origin policy exclusion pending the #8931 path. | `operator override required` |
| #8908 | feat(lanes): explicit ack/move flow for terminally-receipted steering messages | `codex/steering-message-ack-flow-20260706` | `7848d6ad02551a03bb283b0e60e466a9bb2fd4bb` | 5/5 green + quorum fail | `park-current-head` | Head still matches the repeat-blocker policy example with the ack/apply evidence blocker. | `repair exact blocker` |
| #8879 | fix(evidence): adjudicate severity-gated review stalls | `codex/pr8811-adjudication-stall-salvage-20260705` | `89d17eb9a5500ecde87cf6084a18c5c570bc66cf` | 5/5 green + quorum fail | `policy-excluded` | Evidence/quorum adjudication behavior; treat as elevated settlement tooling until exact Tier packet says otherwise. | `operator override required` |
| #8878 | docs(status): add decision integrity dogfood dashboard | `codex/decision-integrity-dogfood-dashboard-20260705` | `330d164ac26671c5733525b351d5eb14b62ee123` | 5/5 green + quorum fail | `settle-next` | Needs targeted merge-packet, no-dissent check, and current-head evidence/quorum verification. | `operator override required` |

## Settle-Next Top 3

These are not merge authorizations. They are the safest next PRs to inspect
with targeted `merge-packet`, `settle_one_pr.py`, owner checks, and unresolved
dissent checks.

1. #8995 — current head `09fa08ef8a4edc310e2cf3b61b429ba11253c541`; clean
   mergeability, green non-quorum required checks, and no obvious risky surface
   from title/branch in this snapshot.
2. #8991 — current head `4b8266594166e0bcf6a00bd14c0e4202a488d43e`; docs
   landing/setup change with green non-quorum required checks.
3. #8878 — current head `330d164ac26671c5733525b351d5eb14b62ee123`; docs
   status/dashboard change with green non-quorum required checks.

## Policy-Excluded Dependabot-Origin PRs

The following seven Dependabot-origin PRs remain policy-excluded from normal
settlement until the #8931 path or an operator override provides the intended
park/handling behavior:

- #8924 `416fb013bd4a0157eb4057a01a63ea5ed42d6408`
- #8923 `48ec932139324ba41fd0a110d823d072f64312f6`
- #8922 `3311a665b1b24b556b9bd1529b5cd60f5f9ef72e`
- #8921 `d826520ae9a5e25465cba3013e2fee8894dc9edf`
- #8920 `6f4ae553e18df22a69fcdf55c430b122e6036e0e`
- #8917 `0d578344432f9f229c2f5e3b5808c95a5a0153cb`
- #8915 `2466c85a9673cfc08eb585ddbca8ddc9146995ba`

## Live-State Notes

- #8948 is still at the parked head
  `be56200ae6a19c79263c492741c68173eddbf669`.
- #8908 is still at the parked head
  `7848d6ad02551a03bb283b0e60e466a9bb2fd4bb`.
- #8961 and #8920 are dirty/conflicting and should not be sent through evidence
  or settlement until the merge-state blocker is repaired.
- #8924, #8923, #8922, and #8921 returned `no checks reported` from
  `gh pr checks --required --json`; keep them policy-excluded rather than
  treating missing checks as a normal retry target.
- #8990 is already merged, so this snapshot treats the repeat-blocker policy as
  live documentation rather than a pending PR dependency.

## Disposition Counts

| Disposition | Count |
| --- | ---: |
| `settle-next` | 3 |
| `repair-exact-blocker` | 1 |
| `park-current-head` | 2 |
| `defer-active-owner` | 0 |
| `policy-excluded` | 14 |
| `superseded-or-close-candidate` | 0 |
