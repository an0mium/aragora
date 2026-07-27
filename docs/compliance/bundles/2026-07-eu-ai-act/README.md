# EU AI Act Compliance Bundle — July 2026 (W3)

**Status:** DRAFT — content-complete target **2026-07-27**, publish target **2026-07-30**
**Plan:** [`docs/plans/2026-07-09-thirty-day-external-proof-month.md`](../../../plans/2026-07-09-thirty-day-external-proof-month.md) (W3, Jul 23–30)
**Scope:** GPAI / Art. 50 transparency posture ahead of the **Aug 2, 2026** EU AI Act GPAI deadline, plus the Article 14 human-oversight evidence pack (#8230 / ODR-6).
**Assembled:** 2026-07-20
**Status checked:** 2026-07-27 against current main for the crux-cards section;
the signed-receipt and Art.14 material was last verified 2026-07-24. All
timestamps in this bundle are UTC.

Issue [#9391](https://github.com/synaptent/aragora/issues/9391) remains open and
production was re-probed directly on 2026-07-24
(`https://api.aragora.ai/health` returned no response), so Variant A stays
blocked and Variant B remains the shipping default. The crux-cards gap was
re-attributed on 2026-07-24 after direct testing and its status updated again on
2026-07-27 — see "Crux cards" below,
[#9581](https://github.com/synaptent/aragora/issues/9581) and
[#9644](https://github.com/synaptent/aragora/issues/9644).

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
| 4 | Source decision receipt (real 2026-06-22 debate, two contributing agents) | `receipts/2026-06-22-packaging-decision-8263.receipt.json` | present |
| 5 | Signed Open Decision Receipt — **Variant B, locally signed** | `packaging-8263.signed-local.odr.json` | present (local key — see contingency below) |
| 6 | Ed25519 public key for artifact 5 | `local-signing.pubkey.pem` | present (local key — see contingency below) |
| 7 | Receipt-verification walkthrough | `verification-walkthrough.md` | present |
| 8 | Rekor transparency-log note | `rekor-note.md` | present (log entry itself: **pending** — submission is an external publish, held for operator) |
| 9 | ODR-2 (#8225) closure comment draft | `odr2-closure-draft.md` | present (**pending-founder-review** — do not post until reviewed) |
| 10 | Signed **production** receipt — **Variant A** | — | **pending-prod** (blocked on AWS reinstatement, [#9391](https://github.com/synaptent/aragora/issues/9391)) |
| 11 | Crux-cards receipt (`cruxes` block populated) | — | **pending — edges fixed ([#9643](https://github.com/synaptent/aragora/pull/9643)); dissent attribution still open ([#9644](https://github.com/synaptent/aragora/issues/9644))** (see "Crux cards" below) |
| 12 | Founder earned-claim review of the bundle | — | **pending-founder-review** (W3 exit criterion) |

## Remaining exit gates

| Gate | Class | Smallest completion action |
|---|---|---|
| Production-signed receipt | infrastructure/operator | Restore production access under #9391, then run the documented Variant A export and independent verification. |
| Rekor entry | external publish/operator | Review `rekor-note.md`, publish the digest once, and record the returned UUID. |
| Crux-cards receipt | **product defect** | Edge construction fixed in [#9643](https://github.com/synaptent/aragora/pull/9643). Still needs [#9644](https://github.com/synaptent/aragora/issues/9644) (dissent attribution) **and** a 3+ agent debate — two agents cannot register a disagreement. Verify the `cruxes` block including non-empty `contesting_agents` before adding. Not blocked on credentials or on AWS. |
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

- **What:** artifacts 5–6 in this directory. A real decision receipt (the
  issue #8263 packaging decision, 2026-06-22 — a **two-contributing-agent**
  debate: grok proposed, deepseek critiqued; mistral-api was invited
  (`agents_requested`) but recorded zero contributions, and the signed ODR
  lists only grok and deepseek as participants/supporters; consensus PASS at
  0.80 confidence with consensus proof) exported to the JCS-canonical ODR
  v0.1 profile and signed with a **locally generated** Ed25519 key
  (`key_id=ed25519-7be72c773c6db3a5`).
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
- **14(4)(c) amendment (2026-07-22):** the generated pack originally marked
  14(4)(c) satisfied on "verdict reasoning / explainability". The source
  receipt's `verdict_reasoning` / `final_answer` are in fact truncated
  mid-sentence at 2,000 characters by the receipt persistence path, and the
  local receipt store copy carries the identical truncation, so the full
  synthesis text is not recoverable. The pack was manually amended to mark
  14(4)(c) **partial** with the truncation recorded, and its integrity
  digest recomputed. The signed ODR's `reasoning.summary` reproduces the
  same truncated text; the ODR bytes and signature are unchanged.

## Crux cards (pending)

Crux cards (#9414, `DebateProtocol.enable_crux_cards`, default OFF) attach
load-bearing disagreements to receipts; the ODR `cruxes` block renders them
to verifiers. **No existing receipt in `docs/receipts/` or the local store
carries a populated `cruxes` block** (checked 2026-07-20).

**Gap re-attributed 2026-07-24 — the original attribution was wrong.** This
entry previously stated that no CLI flag exposed `enable_crux_cards` and that
generating a crux-bearing receipt therefore "requires a real (API-key) debate
run". Both halves are now false:

- A CLI flag does exist: `aragora ask --crux-cards`, shipped in #9506 (merged
  2026-07-23T02:43Z, after this bundle was assembled). All timestamps in this
  bundle are UTC, matching what GitHub records.
- Credentials are **not** the blocker. Crux cards cannot currently be produced
  by *any* standard debate — see [#9581](https://github.com/synaptent/aragora/issues/9581).
  Both paths that build the `BeliefNetwork` for crux detection
  (`crux_cards._network_from_messages` and
  `winner_selector.analyze_belief_network`) add claims but never add the
  `SUPPORTS`/`CONTRADICTS` factor edges that
  `CruxDetector.compute_disagreement_scores()` requires, so
  `total_disagreements` is always `0` and no crux clears any threshold.

Evidence for the re-attribution: two real provider-backed debates were run on
2026-07-24 (`aragora ask --agents claude,codex --crux-cards`, 2 and 3 rounds,
deliberately contested prompts). Both produced receipts with
`schema_version: 1.1` and no `cruxes` block.

### Status 2026-07-27 — edge construction fixed; attribution still open

[#9643](https://github.com/synaptent/aragora/pull/9643) (merged
2026-07-27T17:10Z) fixed the missing edges: each `Critique` already records
which agent contested whose proposal and how hard (`severity`), so those now
become the `CONTRADICTS` edges the detector needs. Verified 0 crux cards before
and 4 after on the same contested debate.

**Two conditions still stand between that and a crux-bearing receipt, and both
are stated here rather than discovered at audit:**

1. **A crux receipt needs a 3+ agent debate.** `CruxDetector` registers a
   disagreement only when **≥2 authors other than the claim's own** relate to
   it, so two agents trading reciprocal critiques never reach one. #9643 also
   made `build_crux_cards` return nothing when no disagreement was detected —
   `crux_score` is a composite, so claims can clear the threshold on
   uncertainty and centrality alone, and publishing those as "crux cards" would
   put load-bearing-disagreement claims into a receipt with nothing behind
   them. Absent is honest; mislabelled is not.
2. **Dissent attribution is still empty**, tracked as
   [#9644](https://github.com/synaptent/aragora/issues/9644). `detect_cruxes()`
   calls `network.propagate()` *before* `compute_disagreement_scores()`, so
   belief propagation converges the authors' stances and flattens the variance
   it then measures. **Fixing #9581 alone can therefore yield a populated but
   misattributed `cruxes` block** — the earlier claim in this section that "the
   detector itself is sound" was under-scoped, and is corrected here.

Also unchanged: the fix covers the network the debate builds itself (the
default path). A KM-seeded `ctx.belief_network` — only present under
`enable_km_belief_sync` — uses KM-derived claim ids that message-derived ids
cannot match, so #9581 remains open for those debates.

W3 plan line: "crux cards in ≥1 published receipt" — still the named gap. It is
a **product defect**, not a credentials or infrastructure gap, and remains the
one W3 gap that is neither operator-held nor blocked on AWS. Closing it needs
#9644 fixed *and* a 3+ agent debate run with `--crux-cards`; verify the
populated `cruxes` block **including non-empty `contesting_agents`** before
adding the receipt under `receipts/` and updating the contents table.

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
