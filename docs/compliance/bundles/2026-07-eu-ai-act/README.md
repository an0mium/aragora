# EU AI Act Compliance Bundle — July 2026 (W3)

**Status:** DRAFT — content-complete target **2026-07-27**, publish target **2026-07-30**
**Plan:** [`docs/plans/2026-07-09-thirty-day-external-proof-month.md`](../../../plans/2026-07-09-thirty-day-external-proof-month.md) (W3, Jul 23–30)
**Scope:** GPAI / Art. 50 transparency posture ahead of the **Aug 2, 2026** EU AI Act GPAI deadline, plus the Article 14 human-oversight evidence pack (#8230 / ODR-6).
**Assembled:** 2026-07-20
**Status checked:** 2026-07-21 against current main. Issue #9391 remains open;
its latest recorded external probe (2026-07-20) still classifies production as
unreachable.

This bundle packages what Aragora can already prove — signed, dissent-preserving
decision receipts with third-party offline verification — into one auditable
directory. Every artifact below is either present in this directory, or listed
with an honest pending status and the exact step that closes it.

## Bundle contents

| # | Artifact | File | Status |
|---|----------|------|--------|
| 1 | Bundle index (this file) | `README.md` | present |
| 2 | Art. 14 human-oversight evidence pack (JSON, canonical) | `oversight-pack.json` | present |
| 3 | Art. 14 human-oversight evidence pack (Markdown report) | `oversight-pack.md` | present |
| 4 | Source decision receipt (real 3-agent debate, 2026-06-22) | `receipts/2026-06-22-packaging-decision-8263.receipt.json` | present |
| 5 | Signed Open Decision Receipt — **Variant B, locally signed** | `packaging-8263.signed-local.odr.json` | present (local key — see contingency below) |
| 6 | Ed25519 public key for artifact 5 | `local-signing.pubkey.pem` | present (local key — see contingency below) |
| 7 | Receipt-verification walkthrough | `verification-walkthrough.md` | present |
| 8 | Rekor transparency-log note | `rekor-note.md` | present (log entry itself: **pending** — submission is an external publish, held for operator) |
| 9 | ODR-2 (#8225) closure comment draft | `odr2-closure-draft.md` | present (**pending-founder-review** — do not post until reviewed) |
| 10 | Signed **production** receipt — **Variant A** | — | **pending-prod** (blocked on AWS reinstatement, [#9391](https://github.com/synaptent/aragora/issues/9391)) |
| 11 | Crux-cards receipt (`cruxes` block populated) | — | **pending** (see "Crux cards" below) |
| 12 | Founder earned-claim review of the bundle | — | **pending-founder-review** (W3 exit criterion) |

## Remaining exit gates

| Gate | Class | Smallest completion action |
|---|---|---|
| Production-signed receipt | infrastructure/operator | Restore production access under #9391, then run the documented Variant A export and independent verification. |
| Rekor entry | external publish/operator | Review `rekor-note.md`, publish the digest once, and record the returned UUID. |
| Crux-cards receipt | machine-capable, credentialed | Run one real provider-backed debate with `enable_crux_cards=True`, verify the populated `cruxes` block, and add the receipt. |
| Earned-claim and ODR-2 text | founder review | Review this bundle and `odr2-closure-draft.md`; publication and issue closure remain separate operator actions. |

The repository can prepare and validate these artifacts, but none of the four
gates is silently treated as complete by this draft.

## Signed-receipt contingency (Variant A / Variant B)

Production is down: the AWS account is suspended
([#9391](https://github.com/synaptent/aragora/issues/9391)), and the production
Ed25519 ODR signing key lives in AWS Secrets Manager
(`aragora/odr-signing-key`, loaded via `aragora/gauntlet/odr_signing.py` —
by design the key never transits an environment variable and has no local
copy). Until the account is reinstated, no production-signed receipt can be
produced. The bundle is built to ship either way:

### Variant A — production-signed receipt (preferred)

- **What:** the same ODR export signed with the production Ed25519 key from
  AWS Secrets Manager; public key published in-repo and at the
  `.well-known` endpoint.
- **Blocker:** AWS account suspension —
  [#9391](https://github.com/synaptent/aragora/issues/9391). Goodwill-credit
  case pending with AWS Billing/Finance as of 2026-07-17.
- **If reinstated before 2026-07-27:** regenerate artifact 5 with
  `ARAGORA_USE_SECRETS_MANAGER=true aragora receipt export <receipt> --format odr --output <out>`
  (signing is automatic when the secret is reachable), replace artifacts 5–6,
  and delete the gap statement below from the published copy.

### Variant B — locally-signed receipt + dated gap statement (shipping default)

- **What:** artifacts 5–6 in this directory. A real decision receipt (a
  3-agent grok / mistral-api / deepseek debate on the issue #8263 packaging
  decision, 2026-06-22, consensus PASS at 0.80 confidence with consensus
  proof) exported to the JCS-canonical ODR v0.1 profile and signed with a
  **locally generated** Ed25519 key (`key_id=ed25519-7be72c773c6db3a5`).
  The signature is real and verifies offline; what the local key does *not*
  provide is issuer authenticity — the key is not the pinned production
  identity.
- **Gap statement (dated 2026-07-20):** the production signing path is
  blocked by the AWS account suspension
  ([#9391](https://github.com/synaptent/aragora/issues/9391)); the production
  private key is held (inaccessible) in AWS Secrets Manager. This bundle
  therefore demonstrates the *mechanism* (sign → publish key → third-party
  offline verify) with a bundle-local key. **Upgrade path:** on AWS
  reinstatement, re-sign the same ODR bytes with the production key
  (Variant A above — the ODR content digest
  `a00f54fc75207b8b205254290739ebbc287a0b5d189facfe65c442517a922d2a` is
  unchanged by re-signing, since signatures are detached), publish the
  production public key, and anchor the digest in Rekor (`rekor-note.md`).
- **Verify it yourself** (no Aragora account, no API key):

  ```bash
  pip install 'aragora-verify==0.1.1'
  aragora-verify packaging-8263.signed-local.odr.json --pubkey local-signing.pubkey.pem
  ```

  Expected: `=> VERIFIED`, with the honest weakening signals
  `attestation: autonomous` and `confidence: present but uncalibrated`.
  Full walkthrough: `verification-walkthrough.md`.

## Art. 14 oversight pack — generation provenance

Generated 2026-07-20 by:

```bash
aragora compliance oversight-pack --window 30d --fetch-settlements \
  --receipts-dir docs/receipts \
  --receipts-dir docs/compliance/bundles/2026-07-eu-ai-act/receipts \
  --output oversight-pack.json --markdown oversight-pack.md
```

- Clause mapping reference: [`docs/compliance/ART14_OVERSIGHT_PACK.md`](../../ART14_OVERSIGHT_PACK.md) (#9417 / #8230).
- `--fetch-settlements` scanned 424 merged PRs in the window via the
  `aragora/human-settlement` commit status; **1 valid human-settlement
  attestation** was included and **50 were refused** because oversight
  identity equalled execution identity — self-attestation is rejected
  fail-closed and those decisions are recorded as `autonomous`. That refusal
  count is itself Art. 14 evidence that the identity-separation control is
  enforced, not decorative.
- Receipt curation note: the default local receipt store
  (`~/.aragora/receipts`) was found to contain test-run pollution
  (MagicMock artifacts from suite runs). The pack was generated from the
  repo trail (`docs/receipts`) plus this bundle's curated `receipts/`
  directory, which holds the one genuine in-window debate receipt. Curation
  rule: exclude receipts containing mock artifacts or mock-agent demo runs;
  include only receipts of real multi-agent debates.
- Honesty contract: 1 windowed receipt, 0 human-attested — the pack marks
  14(1)/14(3)/14(4)(d) as **partial**, not satisfied. Absence is recorded,
  never implied.

## Crux cards (pending)

Crux cards (#9414, `DebateProtocol.enable_crux_cards`, default OFF) attach
load-bearing disagreements to receipts; the ODR `cruxes` block renders them
to verifiers. **No existing receipt in `docs/receipts/` or the local store
carries a populated `cruxes` block** (checked 2026-07-20), and no CLI flag
exposes `enable_crux_cards` yet (`aragora demo` and `aragora ask` do not
surface it), so generating one requires a real (API-key) debate run:

```python
# Requires provider API keys; run from repo root.
from aragora import Arena, Environment, DebateProtocol
env = Environment(task="<real dogfood question>")
protocol = DebateProtocol(rounds=2, enable_crux_cards=True)
# ... build agents, run arena, save the DecisionReceipt JSON
```

W3 plan line: "crux cards in ≥1 published receipt" — this is the named gap.
When a crux-bearing receipt exists, add it under `receipts/` and update this
table.

## Verification chain (what an auditor can check today)

1. **Receipt integrity** — `aragora receipt verify <receipt.json>` (native
   receipts; artifact-hash check).
2. **Signed ODR** — `aragora-verify <odr.json> --pubkey <pem>` (standalone
   PyPI package, no Aragora dependency): schema conformance, JCS-canonical
   digest, Ed25519 signature, quorum consistency, weakening signals.
3. **Oversight pack integrity** — the pack embeds a `sha256/jcs` integrity
   digest (`oversight-pack.json` → `integrity`).
4. **Transparency log** — pending; see `rekor-note.md` for the exact
   publish-and-verify procedure.

## Bundle PR hygiene

Compliance and evidence bundle PRs carry bundle files only. Generated surfaces
(`METRICS`, `STATUS`, `EXTENDED_README`, `CANONICAL_GOALS`, `ARCHITECTURE`,
`FEATURE_DISCOVERY`, and their docs-site mirrors) stay owned by main's generators;
including them guarantees repeated merge-context conflicts.

## Related documents

- [`docs/compliance/EU_AI_ACT_GUIDE.md`](../../EU_AI_ACT_GUIDE.md) — artifact generation guide (Articles 9/12/13/14/15)
- [`docs/compliance/ART14_OVERSIGHT_PACK.md`](../../ART14_OVERSIGHT_PACK.md) — attestation block + clause mapping
- [`docs/proof/2026-07-10-outsider-receipt-verification-runbook.md`](../../../proof/2026-07-10-outsider-receipt-verification-runbook.md) — outsider verification runbook + gap ledger (#8858)
- [`docs/specs/TAMPER_EVIDENT_TRAIL.md`](../../../specs/TAMPER_EVIDENT_TRAIL.md) — anchor / transparency-log design
- [`docs/security/POST_QUANTUM_CRYPTO_MIGRATION.md`](../../../security/POST_QUANTUM_CRYPTO_MIGRATION.md) — PQC groundwork cited by the ODR-2 closure draft
