# Open Decision Receipt (ODR) — Content Profile v0.1

**Status:** Draft v0.1 — Tier 2, issue [#8224](https://github.com/synaptent/aragora/issues/8224),
part of the ODR spine ([#8223](https://github.com/synaptent/aragora/issues/8223)).
**Artifacts:** this spec, `aragora/gauntlet/odr_schema.json` (JSON Schema draft 2020-12),
`aragora/gauntlet/odr_export.py` (reference emitter), `aragora receipt export --format odr`.
**Related:** [`docs/specs/TAMPER_EVIDENT_TRAIL.md`](TAMPER_EVIDENT_TRAIL.md) (trail
*integrity*; this profile supplies the decision *semantics* that ride on it),
issue #8225 (Ed25519 detached signing).

---

## 1. Why a content profile

Trail-integrity standards — IETF SCITT, in-toto/SLSA attestations, Microsoft
agent action receipts — standardize **that** something happened and **that the
record was not rewritten**. They explicitly exclude decision **quality**: what
was claimed, who adversarially examined it, with what model diversity, who
dissented, how confident the system was, and whether a human accepted the
risk.

ODR is a versioned, vendor-neutral **content profile** for exactly that
payload. It deliberately does **not** define an envelope, transport, or
registry: an ODR payload is designed to be carried as the signed statement
inside standard envelopes (SCITT signed statements, COSE detached signatures,
in-toto attestation predicates). One profile, hashed identically everywhere —
"SLSA for decisions."

### Design rules

1. **Lossless where the source has data.** Every field maps from a real field
   of the emitting system (for Aragora: `aragora.gauntlet.receipt_models.DecisionReceipt`).
2. **Honest where it does not.** A field the emitter cannot supply MUST carry
   an explicit absent marker (§3). Emitters MUST NOT fabricate values.
3. **Deterministic bytes.** The hashing basis is RFC 8785 (JCS)
   canonicalization (§5). The same receipt hashes identically on every
   platform and language.
4. **Envelope-agnostic.** `signatures[]` is reserved for detached signatures
   (§6); nothing in the profile depends on a particular envelope.

## 2. Top-level structure

An ODR document is a single JSON object. All thirteen members below are
REQUIRED (the schema enforces this); blocks that the emitter cannot populate
carry absent markers rather than being omitted, so a verifier can distinguish
"not supplied" from "not part of the profile."

| Member | Type | Content |
|---|---|---|
| `odr_version` | string | Profile version, `"0.1"`. |
| `profile` | string | `https://aragora.ai/specs/open-decision-receipt/v0.1`. |
| `receipt_id` | string | Unique id of this receipt. |
| `issued_at` | string \| null | ISO-8601 timestamp from the source receipt; `null` if the source recorded none. |
| `subject` | object | Binding to the decided thing (§4.1). |
| `claim` | object | What is asserted about the subject (§4.2). |
| `reasoning` | object | Reasoning summary or absent marker (§4.3). |
| `quorum` | object | Adversarial-quorum verdict or absent marker (§4.4). |
| `confidence` | object | Calibrated-confidence block or absent marker (§4.5). |
| `cruxes` | object | Crux set or absent marker (§4.6). |
| `attestation` | object | Human-attestation block or explicit `autonomous` disposition (§4.7). |
| `routing` | object | Reserved (§4.8). |
| `signatures` | array | Reserved for detached signatures, empty in v0.1 (§6). |
| `source` | object | Optional provenance link to the native emitting record (§4.9). |

## 3. Absent markers

```json
{ "status": "absent", "reason": "source receipt has no verdict_reasoning" }
```

An absent marker is an object with exactly `status: "absent"` and a non-empty
human-readable `reason`. Its evidentiary meaning: *the emitter looked and the
source record genuinely does not contain this information.* This is a
first-class honesty signal — a receipt full of absent markers is a weak
receipt, visibly, rather than a strong-looking fabricated one. Blocks that are
populated use `status: "present"` (where the schema requires the
discriminator).

## 4. Field semantics and evidentiary meaning

### 4.1 `subject` — what was decided about

| Field | Meaning |
|---|---|
| `identifier` | Stable id of the subject: a git SHA, an action id, a debate/gauntlet id. Binds the decision to one specific thing. |
| `digest` | Content digest of the decision input (`alg` + `value`, e.g. `sha-256`), or absent. With a digest, a verifier can confirm the decision was about *these exact bytes*. |
| `summary` | Optional human-readable description of the subject. |

*Aragora mapping:* `gauntlet_id` → `identifier`, `input_hash` → `digest.value`
(SHA-256), `input_summary` → `summary`.

### 4.2 `claim` — what is asserted

| Field | Meaning |
|---|---|
| `verdict` | The asserted outcome (`PASS`, `CONDITIONAL`, `FAIL`, or emitter-specific). The single load-bearing assertion of the receipt. |
| `statement` | The claim/input under examination, or absent. |

*Aragora mapping:* `verdict` → `verdict`, `input_summary` → `statement`.

### 4.3 `reasoning` — why

`{ "status": "present", "summary": "<text>" }` or absent. The summary is the
emitting system's recorded justification, not a post-hoc rationalization: it
must be the reasoning that was stored with the decision at decision time.

*Aragora mapping:* `verdict_reasoning`.

### 4.4 `quorum` — adversarial-quorum verdict

The core differentiator versus action receipts: **who examined the claim, how
independent they were, and who disagreed.**

| Field | Evidentiary meaning |
|---|---|
| `method` | Consensus mechanism (e.g. `majority`, `adversarial_validation`, `prover_estimator`). |
| `reached` | Whether the quorum converged. |
| `supporting_agents` | Agents endorsing the verdict. |
| `participants[]` | Per-agent `model_family` and `model_id`. The literal `"undisclosed"` means the source recorded no metadata — never a guess. |
| `independence` | `disclosed` (was model diversity recorded at all), `distinct_model_families`, `model_families[]`. Heterogeneous-family review is the substance behind "adversarial"; a quorum of one family is disclosed as such. |
| `dissent` | `present`, `dissenting_agents[]`, `views[]`. Dissent is preserved verbatim — its presence *raises* the evidentiary value of the receipt (the disagreement survived to the record). |

*Aragora mapping:* `consensus_proof` (method/reached/supporting/dissenting),
`agent_responses[].provider`/`.model` (participants and independence),
`dissenting_views` (dissent views). Absent when the source has no
`consensus_proof`.

### 4.5 `confidence` — calibrated confidence

| Field | Meaning |
|---|---|
| `value` | Confidence in `[0, 1]` (`scale: "unit_interval"`). |
| `calibration` | Provenance of calibration: `{ "status": "present", "provenance_ref": {...} }` pointing at the calibration/settlement record, or absent. |

A confidence number without calibration provenance is an *uncalibrated score*
and the profile says so explicitly: emitters MUST mark `calibration` absent
unless a real calibration record exists.

*Aragora mapping:* `confidence` → `value`; `settlement_metadata` (when
populated) → `calibration.provenance_ref` of type
`aragora.settlement_metadata`; otherwise calibration is absent.

### 4.6 `cruxes` — load-bearing disagreement

`{ "status": "present", "items": [...] }` or absent. Crux items identify the
specific claims on which the verdict actually turns (cf. Aragora's
`CruxReceipt`). The native `DecisionReceipt` does not carry a crux set, so the
Aragora emitter marks this absent unless a crux set is supplied explicitly
(`decision_receipt_to_odr(..., crux_set=...)`).

### 4.7 `attestation` — human accountability

| Field | Meaning |
|---|---|
| `disposition` | `"human_attested"` or `"autonomous"`. REQUIRED. |
| `attestor` | Who accepted the risk (REQUIRED when `human_attested`). |
| `attested_at`, `method` | When and how (e.g. `signed_approval`, `settlement_status`). |

`autonomous` is an explicit, first-class disposition — not a missing field.
A consumer can therefore mechanically filter "decisions no human ever looked
at," which is precisely what EU AI Act Article 14 oversight tooling needs.

*Aragora mapping:* the emitter defaults to `autonomous` because
`DecisionReceipt` does not record human sign-off; callers with a real
human-approval record pass it via `attestation=`.

### 4.8 `routing` — reserved

`{ "status": "reserved" }` in v0.1. Reserved for downstream delivery/routing
metadata (channels, jurisdictional residency) in a later minor version.

### 4.9 `source` — native-record provenance

Links the neutral profile back to the emitting system's native record
(`system`, `schema`, `schema_version`, `receipt_id`, `artifact_hash`) so an
auditor can pull the full-fidelity original. Aragora populates it with the
`DecisionReceipt` id and its content-addressable `artifact_hash`.

## 5. Canonicalization and hashing — RFC 8785 (JCS)

The hashing basis of an ODR document is its **RFC 8785 (JSON Canonicalization
Scheme)** serialization:

- UTF-8 output, no insignificant whitespace;
- object members sorted by UTF-16 code units;
- strings minimally escaped per JSON with lowercase `\u00xx` for controls;
- numbers serialized with the ECMAScript `Number::toString` shortest
  round-trip algorithm; `NaN`/`Infinity` are forbidden.

ODR payloads are I-JSON-safe (no numbers needing more than IEEE-754 double
precision), so any conforming JCS implementation produces identical bytes.
The reference implementation is `aragora.gauntlet.odr_export.jcs_canonicalize`
(dependency-free, byte-stability tested against the RFC 8785 number and
sorting examples).

**Content digest:**

```
odr_digest = SHA-256( JCS( odr_document minus the "signatures" member ) )
```

The `signatures` array is excluded so attaching detached signatures never
changes the digest they cover. `aragora.gauntlet.odr_export.odr_content_digest`
implements this.

## 6. Envelopes: ride SCITT/COSE, don't reinvent

ODR intentionally defines **no envelope**. Deployment guidance:

- **SCITT:** the JCS bytes of the ODR document are the signed statement
  payload (`application/json`); registration on a transparency service yields
  the append-only/inclusion properties — exactly the integrity layer that
  [`TAMPER_EVIDENT_TRAIL.md`](TAMPER_EVIDENT_TRAIL.md) builds for this
  repository's own loop. TET answers *"was the record rewritten?"*; ODR
  answers *"what did the decision actually consist of?"*. They compose.
- **COSE / detached signature:** sign `odr_digest` (§5) as a COSE_Sign1
  detached payload, or place Ed25519 signatures in the reserved
  `signatures[]` array (schema shape: `alg`, `key_id`, `signature`,
  `signed_at`). Implementation is issue **#8225** and is out of scope for
  v0.1 — emitters MUST emit `signatures: []`.
- **in-toto:** the ODR document can serve as the predicate of an attestation
  whose subject duplicates `subject.digest`.

## 7. Compliance mapping — EU AI Act Art. 14 / NIST AI 600-1

ODR fields are designed to be the machine-readable evidence behind human
oversight and GenAI risk-management controls:

| ODR field | EU AI Act Art. 14 (Human oversight) | NIST AI 600-1 (GenAI profile) |
|---|---|---|
| `subject` (binding + digest) | 14(4)(a) — enables the overseer to "duly monitor" exactly which input the decision concerns | GV-1.2 / MP-2: documented system context and provenance of inputs |
| `claim.verdict` | 14(4)(c) — output the human must be able to correctly interpret | MS-2.5: traceable system outputs |
| `reasoning.summary` | 14(4)(c)/(d) — interpretation aids; basis for deciding "not to use" the output | MS-2.8: documented rationale supporting explanation |
| `quorum.participants` + `independence` | 14(4)(b) — awareness of automation bias is operationalized by disclosing model-family homogeneity | GV-6.1 / MP-5.1: third-party/model diversity and provenance disclosure |
| `quorum.dissent` | 14(4)(d) — preserved dissent gives the overseer concrete grounds to disregard the output | MS-3.3: capture of disagreement/uncertainty in evaluation |
| `confidence` + `calibration` | 14(4)(b)/(c) — calibrated (or honestly uncalibrated) confidence counters over-reliance | MS-2.3 / MS-4: measured, documented confidence with provenance |
| `cruxes` | 14(4)(d) — identifies the load-bearing points a human should probe before overriding or accepting | MP-2.3: identification of decision-critical assumptions |
| `attestation` | 14(4)(e) — records whether a human exercised the ability to intervene; `autonomous` makes non-intervention auditable | GV-3.2: human oversight roles and responsibilities are recorded per decision |
| `signatures` / JCS digest (§5–6) | 14(1) — effective oversight presupposes the record itself is trustworthy | MS-2.7: integrity/verifiability of AI system records |
| `source` | 14(4)(a) — path back to full-fidelity native record for deeper monitoring | GV-1.5: auditability via linked provenance |

This table maps *evidence availability*, not legal conformity: ODR makes the
facts inspectable; conformity assessment remains the deployer's process (see
`docs/compliance/EU_AI_ACT_GUIDE.md`).

## 8. Conformance

An emitter conforms to ODR v0.1 iff:

1. its output validates against `aragora/gauntlet/odr_schema.json`;
2. every value is sourced from a real record (rule 1) and every unsupplied
   field carries an absent marker (rule 2) — fabricating a value that should
   be absent is non-conformant even if schema-valid;
3. hashing and signing use the JCS basis of §5;
4. `signatures` is `[]` and `routing.status` is `"reserved"`.

A verifier conforms iff it validates the schema, recomputes `odr_digest` from
JCS bytes, and treats `"undisclosed"`/absent markers as *weakening* rather
than failing the receipt (policy thresholds are the verifier's choice).

## 9. Versioning

`odr_version` follows semver-minor semantics: additive optional fields bump
the minor version; any change to canonicalization, required members, or
absent-marker semantics is a new major profile with a new `profile` URI.

## 10. Reference emitter

```bash
aragora receipt export --format odr <receipt-id-or-path> [-o out.odr.json]
```

emits a schema-valid, JCS-canonical ODR document for any stored or on-disk
`DecisionReceipt`. Programmatic use:

```python
from aragora.gauntlet.odr_export import (
    decision_receipt_to_odr, jcs_canonicalize, odr_content_digest,
)

odr = decision_receipt_to_odr(receipt)            # never fabricates
payload = jcs_canonicalize(odr)                   # RFC 8785 bytes
digest = odr_content_digest(odr)                  # SHA-256, signatures-excluded
```
