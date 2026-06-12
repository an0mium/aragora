# Open Decision Receipt (ODR) — v0.1 Content Profile

**Status:** Tier-2 build spec, 2026-06-11. Part of [#8223] (ODR epic); this
document plus `aragora/gauntlet/odr_schema.json` is the **spec half of
[#8224]** (ODR-1). The code half (canonicalizer, mapper, CLI export, tests)
follows as a separate Tier-2 PR; see "Build phases" below.

## Why this layer exists (positioning, stated honestly)

Action-level agent governance (Microsoft AGT receipts, SCITT/COSE transparency
envelopes, VAP) seals **what happened and whether policy allowed it**. Those
layers deliberately exclude decision quality. The layer regulators now ask for
(EU AI Act Art. 14, NIST AI 600-1 agentic guidance, SR 11-7-successor model
risk guidance) is evidence that a decision was **made well and accountably**:
the rationale, the adversarial-quorum verdict, preserved dissent, calibrated
confidence, the load-bearing cruxes, and proof a human governed the outcome.

ODR is a vendor-neutral **content profile** for exactly that payload. It is
not an envelope, not a signature scheme, and not a transport. It is the JSON
document that envelopes sign and transparency logs anchor.

**Non-claims (honest scope):**

- ODR does **not** prove the deliberation was correct. It makes the
  deliberation inspectable, bindable to a subject, and comparable across
  vendors. Judging adequacy remains the consumer's job; ODR carries the
  evidence.
- ODR does **not** replace the tamper-evident trail
  (`docs/specs/TAMPER_EVIDENT_TRAIL.md`). TET protects the *history of
  actions and intents*; ODR is the *per-decision artifact*. They compose: an
  ODR digest can be a TET intent payload or a Rekor entry (ODR-7).
- v0.1 standardizes the payload **before** public-key signing (ODR-2,
  [#8225]) and the standalone verifier (ODR-3, [#8226]) land. Until then,
  integrity binding comes from the digest definition below plus whatever
  envelope the consumer already trusts.

## Relationship to existing artifacts

| Existing artifact | Where | Relationship |
|---|---|---|
| `DecisionReceipt` (schema_version 1.1) | `aragora/gauntlet/receipt_models.py` | Source of truth. ODR is a lossless-where-present summary projection; mapping table below is normative for the PR-2 mapper. |
| `CruxReceipt` (DIC-16) | `aragora/gauntlet/receipt_models.py` | Referenced (not inlined) by the `cruxes` block until ODR-4 ([#8227]) defines the inline shape. |
| Signing backends (HMAC default; RSA, Ed25519 available) | `aragora/gauntlet/signing.py` | ODR-2 is key distribution + detached envelope + default flip, **not** new cryptography. |
| Settlement metadata / human settlement statuses | `DecisionReceipt.settlement_metadata`, repo settlement flow | Referenced by `human_attestation` until ODR-6 ([#8230]) defines the attestation payload. |
| TET intent chain / anchors | `docs/specs/TAMPER_EVIDENT_TRAIL.md` | ODR digests are anchorable payloads; TET's "Rekor anchor is the seed of the public Open Receipt Standard verifier" line is this spec's lineage. |

## Design rules (normative)

- **R1 — Vendor-neutral verification.** A third party MUST be able to
  schema-validate an ODR and recompute its digest with zero Aragora code.
- **R2 — No fabrication.** A field is either truthfully populated or
  explicitly marked absent. Producers MUST NOT synthesize values for missing
  evidence. This is what the enrichment-block `status` pattern enforces.
- **R3 — Summary artifact.** ODR carries verdict-level evidence, not full
  transcripts. Full agent responses, thinking traces, and KM operations stay
  in the source receipt, referenced by id + hash (size, privacy, and
  stability of canonical bytes all depend on this).
- **R4 — Envelope-free.** The ODR document contains no self-hash and no
  signature. Its digest is computed by consumers and carried in envelopes
  (COSE/SCITT), commit statuses, or transparency logs. Self-referential
  hashes require exclusion rules that are a standing source of verifier bugs.
- **R5 — Strict.** Unknown fields are rejected (`additionalProperties:
  false` everywhere). Extensibility happens through versioning, not through
  silent extra keys.

## The profile

Top-level object. All fourteen members are REQUIRED (absence is expressed
inside enrichment blocks, never by omitting keys; R2).

| Field | Type | Evidentiary meaning |
|---|---|---|
| `odr_version` | const `"0.1"` | Profile version this document conforms to. |
| `receipt_id` | string | Stable identifier of this receipt in the producing system. |
| `issued_at` | string, ISO-8601 UTC (`Z` suffix REQUIRED) | When the receipt was issued. |
| `generator` | object `{name, version}` | Producing software, for provenance and bug forensics. |
| `subject` | object | **What was decided about** (see below). |
| `claim` | object | **What was decided** (see below). |
| `quorum` | object | **Who deliberated and how independent they were** (see below). |
| `confidence` | object `{value: 0..1}` | The system's stated confidence in the verdict. Calibration evidence for this number is the `calibration` block's job. |
| `adversarial_assessment` | enrichment block | How hard the decision was stress-tested (attack/probe counts, robustness). |
| `cruxes` | enrichment block | The load-bearing disagreements (ODR-4). |
| `calibration` | enrichment block | Whether this producer's confidence numbers are historically trustworthy (ODR-5). |
| `human_attestation` | enrichment block | Proof an accountable human accepted the outcome (ODR-6). |
| `routing` | enrichment block | Why these models were trusted with this decision, at what cost ([#8233]). |
| `provenance` | object | Link back to the source system and its event chain (see below). |

### `subject` — what was decided about

| Field | Req | Meaning |
|---|---|---|
| `type` | yes | One of `git_commit`, `pull_request`, `document`, `action`, `decision`, `other`. |
| `identifier` | yes | Natural identifier (commit SHA, PR URL, run id). |
| `digest` | no | `{alg, value}` content hash of the subject when its exact bytes are hashable. SHOULD be present whenever the subject has canonical content. |
| `binding` | no | Prose statement of how the digest/identifier binds the subject (e.g. "digest is SHA-256 of the full input text submitted to the gauntlet run"). |

A receipt whose subject cannot be re-identified is testimony, not evidence;
`subject` is what makes an ODR attach to a real-world artifact.

### `claim` — what was decided

| Field | Req | Meaning |
|---|---|---|
| `statement` | yes | The decision or claim that was vetted. |
| `verdict` | yes | `PASS`, `CONDITIONAL`, or `FAIL`. Producers MUST map internal verdicts onto these three and record the unmapped original in `provenance.source.raw_verdict` when it differs. |
| `reasoning_summary` | yes | Verdict rationale. MAY be empty when the source recorded none; an empty string is the honest representation of "no rationale recorded", never a fabricated one (R2). |

### `quorum` — who deliberated, how independently

| Field | Req | Meaning |
|---|---|---|
| `method` | yes | Consensus mechanism (`majority`, `unanimous`, `judge`, `none`, `prover_estimator`, ...). |
| `reached` | yes | Whether consensus was reached. |
| `participants` | yes | Array of `{agent, model_family, role?, provider?, model?}`. `model_family` is the independent provider lineage (`anthropic`, `openai`, `mistral`, `google`, ...). When the family cannot be determined it MUST be `"unknown"`, and **all `unknown` participants count as a single family** for independence purposes (conservative by construction). |
| `supporting` / `dissenting` | yes | Agent names for and against the verdict. Each listed name MUST appear in `participants`; the two sets MUST be disjoint. |
| `dissent_summaries` | no | The preserved content of dissent. The source model stores dissent texts unattributed (`dissenting_views: list[str]`), so v0.1 does NOT pair summaries to agents; pairing them would be fabrication (R2). |
| `independence` | yes | `{distinct_model_families: int}`. MUST equal the recomputed count from `participants` under the `unknown`-collapse rule. |
| `taint` | no | `{tainted_proposals, trust_score}` when taint analysis ran (G2). |

This block is the artifact's core differentiator: heterogeneous, adversarial,
counted, with dissent preserved rather than averaged away.

### `provenance` — link to the source system

| Field | Req | Meaning |
|---|---|---|
| `source` | yes | `{system, receipt_id, artifact_hash, schema_version, gauntlet_id?, raw_verdict?}` identifying the source receipt. |
| `chain` | no | Source event chain entries `{timestamp, event_type, agent?, description?, evidence_hash?}` (mirrors `ProvenanceRecord`). |
| `costs` | no | Free-form cost summary (informational; not covered by strictness rule R5's spirit, but still canonicalized). |

**Honest integrity note (normative for consumers):**
`DecisionReceipt.artifact_hash` in the source system covers only a subset of
fields (`receipt_id`, `gauntlet_id`, `input_hash`, `risk_summary`, `verdict`,
`confidence`). `provenance.source.artifact_hash` is therefore a **link**, not
an integrity proof of the full source receipt. The ODR digest defined below
covers the **entire** canonical ODR payload; that strengthening is
deliberate and is the reason ODR exists as a separate artifact.

## Enrichment blocks and absent markers (normative)

Each enrichment block is an object whose `status` is one of:

- `{"status": "absent"}` — the evidence does not exist for this decision.
  No other keys allowed. This is a first-class, schema-enforced statement,
  not a missing field: consumers can distinguish "producer has no calibration
  story" from "producer forgot to include it".
- `{"status": "referenced", "ref": {artifact_type, artifact_id, digest?}}` —
  the evidence exists as a separate artifact (e.g. a `CruxReceipt`, a
  settlement status). `digest` binds it when available. `CruxReceipt`
  exposes only a 16-hex truncated checksum today; producers MUST record it
  with `alg: "sha256-trunc16"`, and consumers MUST treat truncated digests
  as link-strength only, never as integrity proof.

Inline `"present"` payloads are **deliberately not defined in v0.1**: their
shapes belong to the children that build them (ODR-4 cruxes [#8227], ODR-5
calibration [#8229], ODR-6 human attestation [#8230], routing rationale
[#8233]) and will arrive as additive v0.2 schema changes. Freezing payloads
for unbuilt evidence would violate R2 in spirit.

`adversarial_assessment` is the exception: its data exists today in every
gauntlet receipt, so v0.1 defines its `present` payload (`attacks_attempted`,
`attacks_successful`, `probes_run`, `vulnerabilities_found`,
`robustness_score`, `risk_summary`).

### Reserved meanings (what each block will attest once its child lands)

| Block | Reserved evidentiary meaning |
|---|---|
| `cruxes` | The minimal set of claims on which the verdict actually pivots, with counterfactual evidence of load-bearing-ness (from the crux finder). |
| `calibration` | Historical reliability of this producer's `confidence.value` (per-domain calibration, e.g. Brier-style), so "0.78 confident" is auditable rather than rhetorical. |
| `human_attestation` | Identity-pinned proof that an accountable human reviewed and accepted (or overrode) the outcome, compatible with the TET settlement-creator pin. |
| `routing` | Which models were trusted with this decision, why (stakes tier, cost/quality rationale), and what it cost. |

## Canonicalization and digest (normative)

- Canonical form is **RFC 8785 (JCS)** applied to the ODR object: UTF-8, no
  insignificant whitespace, lexicographic member ordering, RFC 8785 string
  escaping, ECMAScript number serialization.
- `odr_digest = SHA-256(JCS(odr))`, conveyed wherever the consumer needs it
  (COSE/SCITT envelope, commit status, Rekor entry). It is never embedded in
  the ODR itself (R4).
- Profile constraints that keep JCS implementations boring:
  - All schema-defined member names are ASCII, so RFC 8785's UTF-16
    code-unit ordering coincides with byte ordering **for keys**; producers
    SHOULD also keep free-form object keys (e.g. `risk_summary`,
    `costs`) ASCII.
  - Numbers MUST be finite (RFC 8785 prohibits NaN/Infinity). Producers
    SHOULD round scores to at most 6 decimal places; the residual risk
    (ECMAScript shortest-round-trip formatting) is pinned in PR-2 by the
    RFC 8785 Appendix test vectors.
  - Timestamps are strings, never numbers.

## Envelope and signing compatibility (informative)

ODR is designed to ride standard envelopes rather than invent one:

- **COSE_Sign1 / SCITT signed statement** with the JCS bytes (or their hash)
  as payload is the target shape for ODR-2 ([#8225]). The Ed25519 backend
  already exists in `aragora/gauntlet/signing.py`; ODR-2's actual work is
  public-key distribution, a key-id convention, detached-signature emission,
  and flipping the default away from HMAC (which, being symmetric, proves
  nothing to outsiders).
- **ODR-3** ([#8226], `aragora-verify`) consumes `{ODR JSON, detached
  envelope, public key}` fully offline.
- **ODR-7** ([#8231]) anchors `odr_digest` in a public transparency log
  (Sigstore Rekor), upgrading "our trail says so" to "a public log says so".

## Verifier rules for v0.1 (normative)

A v0.1 verifier MUST:

1. Reject any document whose `odr_version` it does not implement.
2. Validate against the pinned schema (`$id` ending `/odr/0.1/...`);
   unknown fields are failures, not warnings (R5).
3. Recompute JCS bytes and `odr_digest`, and compare against the digest
   provided out-of-band (envelope, status, log entry) when one is supplied.
4. Check `supporting ∪ dissenting ⊆ participants[].agent` and
   `supporting ∩ dissenting = ∅`.
5. Recompute `independence.distinct_model_families` from `participants`
   under the `unknown`-collapse rule and require equality.
6. Treat `sha256-trunc16` reference digests as non-probative links.

What v0.1 verification deliberately does NOT establish: signature validity
(ODR-2/3), deliberation re-execution, or truth of the underlying claim. A
verifier that passes a document is asserting "well-formed, internally
consistent, byte-stable decision evidence", nothing more.

## Mapping: `DecisionReceipt` 1.1 → ODR (normative for the PR-2 mapper)

| DecisionReceipt field | ODR destination | Note |
|---|---|---|
| `receipt_id` | `receipt_id`, `provenance.source.receipt_id` | |
| `timestamp` | `issued_at` | Normalized to UTC `Z`. |
| `gauntlet_id` | `subject.identifier`, `provenance.source.gauntlet_id` | |
| `input_summary` | `claim.statement` | |
| `input_hash` | `subject.digest` (`alg: sha256`) | `subject.type: "document"`; `subject.binding` states the hash covers the full submitted input. |
| `verdict` | `claim.verdict` | Already PASS/CONDITIONAL/FAIL. |
| `verdict_reasoning` | `claim.reasoning_summary` | Empty string preserved as empty (R2). |
| `confidence` | `confidence.value` | |
| `consensus_proof.method/reached` | `quorum.method/reached` | When `consensus_proof` is None: `method: "none"`, `reached: false`. |
| `consensus_proof.supporting_agents/dissenting_agents` | `quorum.supporting/dissenting` | |
| `consensus_proof.tainted_proposals/trust_score` | `quorum.taint` | Omitted when no taint analysis ran. |
| `dissenting_views` | `quorum.dissent_summaries` | Unattributed by source; never paired to agents (R2). |
| `agent_responses[].{agent,role,provider,model}` | `quorum.participants[]` | `model_family` derived from `provider` (normalization table lives in the mapper); response texts NOT copied (R3). |
| `attacks_attempted/attacks_successful/probes_run/vulnerabilities_found/robustness_score/risk_summary` | `adversarial_assessment` (`present`) | |
| `provenance_chain[]` | `provenance.chain[]` | Field-for-field. |
| `cost_summary` | `provenance.costs` | |
| `artifact_hash`, `schema_version` | `provenance.source.artifact_hash/schema_version` | See honest integrity note above. |
| `vulnerability_details`, `explainability`, `thinking_traces`, `km_operations`, `settlement_metadata`, `settlement_status`, `config_used` | **not mapped in v0.1** | Stay in the source receipt (R3). `settlement_metadata`/`settlement_status` become `human_attestation` references when available; `explainability` is an ODR-4/5 candidate. |
| `signature*` fields | **not mapped** | ODR uses detached envelopes (R4, ODR-2). |

## Regulatory crosswalk (informative)

This table states which ODR field supplies **evidentiary support** for which
obligation. It is not a compliance determination, and producers MUST NOT
represent an ODR alone as conformity.

| Obligation | ODR evidence |
|---|---|
| EU AI Act Art. 14(4)(a) — understand capacities and limitations | `confidence.value`, `calibration`, `adversarial_assessment` |
| Art. 14(4)(b) — remain aware of automation bias | `quorum.dissenting`, `quorum.dissent_summaries` (disagreement preserved, not averaged away) |
| Art. 14(4)(c) — correctly interpret output | `claim.reasoning_summary`, `cruxes` |
| Art. 14(4)(d) — decide not to use / disregard | `human_attestation` |
| Art. 14(4)(e) — intervene or interrupt | `human_attestation` + `provenance.chain` |
| Art. 12 — record-keeping | the ODR itself: `issued_at`, `subject` binding, `provenance` |
| NIST AI 600-1 GOVERN (accountability) | `generator`, `human_attestation` |
| NIST AI 600-1 MAP (context) | `subject`, `claim.statement` |
| NIST AI 600-1 MEASURE | `confidence`, `quorum.independence`, `adversarial_assessment` |
| NIST AI 600-1 MANAGE | `claim.verdict`, `routing` |

## Versioning and evolution (normative)

- `odr_version` is `major.minor`. Minor versions are strictly additive (new
  optional members, new enrichment `status` values, new `present` payloads).
  Major versions may break.
- Each version's schema is pinned by `$id`; verifiers MUST validate against
  the schema matching the document's `odr_version` and MUST reject unknown
  versions (verifier rule 1).
- Canonicalization rules never change within a major version (digests must
  stay recomputable forever).

## Build phases (loop-executable)

| Phase | Deliverable | Tier |
|---|---|---|
| P1 (this PR) | This spec + `aragora/gauntlet/odr_schema.json` | 2 (docs + schema) |
| P2 | `aragora/gauntlet/jcs.py` (vendored minimal RFC 8785, validated against the RFC appendix vectors; no new dependency), `aragora/gauntlet/odr.py` (`to_odr`, `odr_canonical_bytes`, `odr_hash`), `aragora receipt export --format odr`, round-trip + golden-vector + schema tests, touched-test run attached as dogfood evidence | 2 |
| then | ODR-2 signing ([#8225]) → ODR-3 verifier ([#8226]) → ODR-4/5/6 flip enrichment blocks to inline `present` (v0.2) → ODR-7 Rekor anchoring ([#8231]) | per child issue |

## Exit metric (falsifiable)

A third party, given only this spec, the schema file, and one ODR document,
can (a) validate conformance and (b) recompute `odr_digest` and match it
against an independently conveyed digest, using **zero Aragora code**.
Falsified if any conforming receipt requires Aragora internals to validate,
or if two independent RFC 8785 implementations produce different canonical
bytes for the same conforming document. PR-2's golden vectors are the
standing regression guard for the second falsifier.

## Example (informative)

```json
{
  "odr_version": "0.1",
  "receipt_id": "gauntlet-a1b2c3d4",
  "issued_at": "2026-06-11T22:14:09Z",
  "generator": {"name": "aragora", "version": "0.9.3"},
  "subject": {
    "type": "document",
    "identifier": "gauntlet-a1b2c3d4",
    "digest": {"alg": "sha256", "value": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"},
    "binding": "digest is SHA-256 of the full input text submitted to the gauntlet run"
  },
  "claim": {
    "statement": "Proposed rate-limiter design for the public API",
    "verdict": "CONDITIONAL",
    "reasoning_summary": "Design survives 11 of 12 attack classes; token-bucket refill race requires a fix before production."
  },
  "quorum": {
    "method": "majority",
    "reached": true,
    "participants": [
      {"agent": "claude_critic", "role": "critic", "provider": "anthropic", "model": "claude-opus-4-8", "model_family": "anthropic"},
      {"agent": "gpt_critic", "role": "critic", "provider": "openai", "model": "gpt-4.1", "model_family": "openai"},
      {"agent": "mistral_judge", "role": "judge", "provider": "mistral", "model": "mistral-large-2512", "model_family": "mistral"}
    ],
    "supporting": ["claude_critic", "mistral_judge"],
    "dissenting": ["gpt_critic"],
    "dissent_summaries": ["Refill race is exploitable under burst load; verdict should be FAIL until fixed."],
    "independence": {"distinct_model_families": 3},
    "taint": {"tainted_proposals": [], "trust_score": 1.0}
  },
  "confidence": {"value": 0.78},
  "adversarial_assessment": {
    "status": "present",
    "attacks_attempted": 12,
    "attacks_successful": 1,
    "probes_run": 9,
    "vulnerabilities_found": 1,
    "robustness_score": 0.83,
    "risk_summary": {"critical": 0, "high": 1, "medium": 0, "low": 0}
  },
  "cruxes": {
    "status": "referenced",
    "ref": {
      "artifact_type": "aragora.crux_receipt",
      "artifact_id": "crux-1f2e3d4c",
      "digest": {"alg": "sha256-trunc16", "value": "8c1f2ab34d5e6f70"}
    }
  },
  "calibration": {"status": "absent"},
  "human_attestation": {"status": "absent"},
  "routing": {"status": "absent"},
  "provenance": {
    "source": {
      "system": "aragora.gauntlet",
      "receipt_id": "gauntlet-a1b2c3d4",
      "artifact_hash": "c2a9d4e6f8013579bdf02468ace13579bdf02468ace13579bdf02468ace13579",
      "schema_version": "1.1"
    },
    "chain": [
      {"timestamp": "2026-06-11T22:13:55Z", "event_type": "verdict", "agent": "mistral_judge", "description": "Judge synthesis recorded", "evidence_hash": ""}
    ],
    "costs": {"total_usd": 0.42}
  }
}
```

## Product note (why this is not just a schema)

Action-level toolkits prove *what happened*. ODR proves *whether it was
decided well and who accountably accepted the risk*. It is the productized
form of what this repo's own loop already emits (quorum evidence, dissent,
settlements, intent anchors), which makes the loop itself the first ODR
producer and the cheapest credible demo: every settled PR can carry a
receipt a stranger can check. "SLSA for decisions" is the position; this
profile is its artifact.

[#8223]: https://github.com/synaptent/aragora/issues/8223
[#8224]: https://github.com/synaptent/aragora/issues/8224
[#8225]: https://github.com/synaptent/aragora/issues/8225
[#8226]: https://github.com/synaptent/aragora/issues/8226
[#8227]: https://github.com/synaptent/aragora/issues/8227
[#8229]: https://github.com/synaptent/aragora/issues/8229
[#8230]: https://github.com/synaptent/aragora/issues/8230
[#8231]: https://github.com/synaptent/aragora/issues/8231
[#8233]: https://github.com/synaptent/aragora/issues/8233
