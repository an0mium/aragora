# ODR Completion Mission — re-point generation from self-maintenance to product

> **For agentic workers / harnesses (Factory `/missions`, nomic loop, elves-aragora):**
> This is a launchable mission directive, not a free-form backlog. It **executes
> epic [#8223](https://github.com/synaptent/aragora/issues/8223)** (Open Decision
> Receipt) to completion; it does **not** define a parallel program. One phase per
> run/sprint; re-read this file, the epic, and the program doc
> (`2026-06-13-conveyor-hardening-program.md`, epic
> [#8344](https://github.com/synaptent/aragora/issues/8344)) at every phase
> boundary. **Verify every issue/PR state live before relying on it** — the state
> table below was captured 2026-06-13 and the loop moves it.

**Mission:** Complete the ODR program — make Aragora's decision receipts
**externally consumable and offline-verifiable by a third party**. That is the
project's own stated value test: a receipt is a moat only if a stranger can
verify it without trusting or running Aragora. Action-level toolkits (Microsoft
AGT) prove *what happened*; ODR proves *whether it was decided well and who
accountably accepted the risk* — the decision-semantics layer above AGT that
SCITT/VAP exclude and the NIST agentic profile / EU AI Act Art. 14 demand with
no existing tooling. This is the differentiated product, and it is the work the
generation loop should be doing instead of maintaining its own pipeline.

**Why this mission exists (the strategic correction):** the autonomous loop has
drifted into self-maintenance. The 2026-06-13 snapshot: ~196 open PRs,
**~57% pipeline-maintenance, ~1.5% product**. The conveyor *quality* gate is
calibrated correctly (106:4 merged:rejected — see the conveyor program §1), so
the problem is not the gate; it is **what the loop chooses to generate**. The
correction is to re-point writer-lane / nomic-loop generation away from
pipeline-maintenance and toward product value, and the product is ODR. Necessary
maintenance does not stop — it moves to a **separate bounded lane** (§3) so it
stops consuming the main generation budget.

---

## 1. Program alignment (this executes #8223 — it does not re-specify it)

The ODR thesis, value chain, and child sequence are defined in epic
[#8223](https://github.com/synaptent/aragora/issues/8223) and need no
restatement. This mission is the *execution discipline* over that epic: tier
honesty, throughput limits, settlement sequencing, and the generation-steering
rule. Adjacent-direction issues (#8233 decision-stakes routing, #8234 jury
optimizer) are explicitly **out of scope** for this mission — they ride the same
thesis but are a separate track; do not let them dilute the spine.

### ODR child sequence and current state (VERIFY LIVE — captured 2026-06-13)

| Child | Issue | Deliverable | State (verify before relying) |
|---|---|---|---|
| **ODR-1** | [#8224](https://github.com/synaptent/aragora/issues/8224) | Vendor-neutral content profile: JSON Schema + JCS canonicalization | **CLOSED/done.** Spec PR [#8246](https://github.com/synaptent/aragora/pull/8246) merged; `docs/specs/OPEN_DECISION_RECEIPT.md` + `aragora/gauntlet/odr_schema.json` on `main`. |
| **ODR-2** | [#8225](https://github.com/synaptent/aragora/issues/8225) | Ed25519 detached public-key signing | **OPEN.** PR [#8275](https://github.com/synaptent/aragora/pull/8275) OPEN, `mergeable_state: blocked`. **Tier 3** (key management) — needs human settlement; sdk-parity on the merge ref needs dep **#8273** (stale-debt paydown) to land first. |
| **ODR-3** | [#8226](https://github.com/synaptent/aragora/issues/8226) | `aragora-verify` standalone offline verifier (PyPI) + `POST /api/receipts/verify` | **OPEN, no PR.** Depends on ODR-1 (done) + ODR-2 (in flight). The keystone deliverable. |
| **ODR-4** | [#8227](https://github.com/synaptent/aragora/issues/8227) | Crux exposure: API + `--crux` CLI + SDK + crux set in receipt | **OPEN.** PR [#8255](https://github.com/synaptent/aragora/pull/8255) OPEN, `blocked`. |
| **ODR-5** | [#8229](https://github.com/synaptent/aragora/issues/8229) | Calibration report API + auditable calibrated confidence in receipt | **OPEN.** PR [#8290](https://github.com/synaptent/aragora/pull/8290) OPEN, `mergeable_state: dirty` — **not merged** (the PR self-reclassifies to Tier 3). |
| **ODR-6** | [#8230](https://github.com/synaptent/aragora/issues/8230) | Human-oversight attestation block + EU AI Act Art. 14 / NIST evidence-pack generator | **OPEN, no PR.** Builds on TET T4 (H2 settlement-creator pin) + the live scarmani precedent. |
| **ODR-7** | [#8231](https://github.com/synaptent/aragora/issues/8231) | Sigstore Rekor public anchoring | **CLOSED/done** 2026-06-12 via the TET T2 anchor publisher (`aragora/trail/rekor.py`); ODR-3's verifier will check inclusion proofs offline. |

### The structure (per #8223 §Sequencing)

- **Spine = ODR-1 → ODR-2 → ODR-3** — *a receipt a stranger can verify offline.*
  ODR-1 is done; ODR-2 is the gating piece; **ODR-3 is the keystone** and the
  thing this mission most exists to land. Until ODR-3 ships, the moat is still
  "leaky" (receipts verify only inside Aragora).
- **Enrichment = ODR-4 / ODR-5 / ODR-6** — crux, calibrated confidence, and
  human-oversight attestation reach the receipt payload + a public API. Each is
  independently valuable but secondary to the spine.
- **Public anchoring = ODR-7** — done; ODR-3 consumes it.

**Right-of-way:** the spine has right-of-way over enrichment, and ODR-3 has
right-of-way over everything (see §4). An enrichment PR must never block or
out-prioritize an open spine PR.

---

## 2. Tier map + checkpoints (honest classification per `docs/REVIEW_AUTHORITY_PRINCIPLES.md`)

Tiers are assigned per the merge-tier table in
[`docs/REVIEW_AUTHORITY_PRINCIPLES.md`](../../REVIEW_AUTHORITY_PRINCIPLES.md) §Merge
Tiers, classified honestly (the gate's own `_classify_model_review_tier` is the
arbiter; where the issue label and the classifier disagree, **the higher tier
wins** — ODR-5/#8290 is the worked example: labelled Tier 2, classified Tier 3
because it adds a public API surface).

| Child | Honest tier | Why | Settlement requirement |
|---|---|---|---|
| ODR-1 (#8224) | 2 → **done** | additive schema + docs + CLI flag | already merged |
| ODR-2 (#8225, PR #8275) | **3** | key management (security-adjacent); new public-key/`.well-known` surface | **human risk settlement** + dep #8273 |
| ODR-3 (#8226) verifier lib | **≤2** | new standalone additive package, no live caller into the gate | model quorum; admin squash allowed |
| ODR-3 (#8226) PyPI publish step | **4** (that step only) | release-workflow touch is workflow policy | **park the publish step for the operator**; ship the library + endpoint at ≤2 first |
| ODR-3 (#8226) `/api/receipts/verify` | **3** | new public API surface | human settlement for the API surface |
| ODR-4 (#8227, PR #8255) | **3** | new public API endpoint + CLI flag + SDK methods | human settlement for the API surface |
| ODR-5 (#8229, PR #8290) | **3** | read-only API but new public surface (`handlers/` + `sdk/`); classifier says Tier 3 | human settlement for the API surface |
| ODR-6 (#8230) attestation + evidence pack | **2–3** for the compliance artifact; **4** for any settlement-verification touch point | the oversight-attestation generator is additive; anything reading/writing `review_queue.py` / settlement status is merge-authority self-mod | settlement-touch parts stay in TET T4 (scarmani-settled) |
| ODR-7 (#8231) | 2 → **done** | additive publisher backend, no secrets (Rekor keyless for hashes) | already merged |

**Tier-4 / scarmani-gated rule:** anything touching `review_queue.py`,
settlement scripts (`scripts/settle_*.py`), or `.github/workflows/` is **Tier 4**
(merge-authority self-modification — the gate cannot be its own arbiter) and is
settled only by `scarmani`, never by an agent-held credential. Key management
(ODR-2, and any future ODR key handling) is **Tier 3** (human settlement). The
verifier library (ODR-3) and most enrichment read-paths are **Tier ≤2**.

**Sequencing gate — Tier-3/4 unblocked:** The scarmani identity pins (#8274 —
TET T4 H1 CODEOWNERS pin + H2 settlement-creator pin) **MERGED 2026-06-13**. The
Tier-3/4 ODR items were correctly held until those pins landed; that gate is now
**open**. Tier-3/4 ODR settlements (ODR-2, the ODR-3 API surface, ODR-4, ODR-5,
ODR-6 settlement-touch) may now proceed under the post-#8274 regime, where the
`aragora/human-settlement` status is pinned to the `scarmani` creator and the
merge-authority paths require code-owner review. Re-verify #8274 is merged and
the H1 branch-protection click-path is active before settling any Tier-4 item.

**ODR-6 is the dogfood proof.** The human-oversight attestation block (ODR-6) is
not only an enrichment payload — it is the **machine-readable form of what this
repo already does manually for Tier-4 settlement**, and therefore the proof of
the project's own audit-trail-honesty claim: a receipt that records *who
accountably accepted the risk, what they saw, and via what mechanism*, with an
explicit `autonomous` disposition where no human settled (absence is recorded,
never implied — per #8230 acceptance). Because the attestation's trust root is
the scarmani settlement-creator pin (TET T4 / H2), **ODR-6 pairs naturally with
the scarmani identity work** and should be sequenced alongside it. Landing ODR-6
turns this repo's own settlement trail into the first public Art. 14 evidence
corpus — the cheapest credible proof the loop can produce.

### Checkpoints (review at each — one phase per run)

- **CP-A — Spine signing settled:** ODR-2 (#8275) Tier-3-settled and merged
  (dep #8273 in first). Spine is signable.
- **CP-B — Spine verifiable (KEYSTONE):** ODR-3 (#8226) library on PyPI-track +
  `/api/receipts/verify` live; a receipt from this repo's loop verifies in a
  fresh venv with only the public key. **This is the mission's pivot point** —
  the moat stops leaking here.
- **CP-C — Enrichment exposed:** ODR-4 (#8255) + ODR-5 (#8290) settled; crux set
  and calibrated-confidence provenance populate receipts and APIs.
- **CP-D — Oversight + dogfood:** ODR-6 (#8230) attestation block in receipts +
  `aragora compliance oversight-pack` produces an Art. 14 bundle citing real
  receipts from this repo's trail.
- **CP-E — Scorecard:** closing scorecard on #8223 (§5 Definition of Done).

---

## 3. Generation-steering rule (the re-point — the load-bearing section)

This is the behavioral correction the mission exists to install. It governs what
the writer lanes / nomic-loop generation choose to produce.

1. **Prefer product/ODR work.** When a generation lane has budget and is
   choosing what to produce, it **prefers** items from this ODR backlog (spine
   first, then enrichment, per §1 right-of-way) over self-maintenance work.
   Product generation is the default; maintenance generation is the exception.

2. **Route maintenance to a separate bounded lane.** Necessary
   pipeline-maintenance (the conveyor-hardening work in #8344: transports,
   janitors, backpressure, review-pipeline defect fixes) does **not** stop — it
   is real work. But it is routed to a **separate, bounded maintenance lane with
   its own budget cap**, NOT the main generation budget. Maintenance and product
   draw from different buckets so maintenance can never crowd out product the way
   it did in the 57%/1.5% snapshot.

3. **Consult the maintenance-ratio before generating.** Generation should
   consult the `pr_value_classifier` maintenance-ratio (the classifier is built
   by a sibling PR — `scripts/pr_value_classifier.py`; it is **not yet on
   `main`**, so treat the ratio as advisory until it lands, then make it
   binding). When the maintenance-ratio is high, **bias hard toward this ODR
   backlog**.

4. **Hard stop — do NOT generate new self-maintenance PRs while the backlog is
   hot.** Make this explicit and enforceable:
   - **open-PR backlog > 60**, OR
   - **maintenance-ratio > 0.5**

   → the loop **MUST NOT generate new self-maintenance PRs**. Drain/triage the
   existing maintenance backlog first (close stale, batch-settle ready, merge or
   abandon), and spend the freed generation budget on ODR. New maintenance
   generation resumes only after the backlog is below 60 **and** the
   maintenance-ratio is at or below 0.5. (The 2026-06-13 snapshot — 196 open
   PRs, ~0.57 ratio — is double-over both thresholds: under this rule the loop
   would be in drain-and-ODR-only mode today.)

5. **Never loosen the review gate to hit a ratio.** Per the conveyor program §6
   lesson 8 and `REVIEW_AUTHORITY_PRINCIPLES.md`: the maintenance-ratio is
   steered by *changing what is generated and draining backlog*, never by
   weakening the merge quorum or relabelling product work as maintenance (or vice
   versa). Classification is honest or the ratio is meaningless.

---

## 4. Merge-gate throughput plan + path-freeze + ODR right-of-way

Mirror the structural discipline of the steering-leverage operating plan and the
conveyor program. The merge gate is calibrated correctly — the constraint is
**throughput through settlement**, not quality — so this section is about not
flooding the gate and sequencing settlements efficiently.

- **Heterogeneous quorum evidence per PR.** Every ODR PR carries a model-quorum
  packet meeting `REVIEW_AUTHORITY_PRINCIPLES.md` §Model Review Quorum: exact
  head SHA, reviewer/provider families, independence from the authoring lane,
  recommendation + dissent, concrete validation/dogfood evidence, tier +
  settlement requirement. For Tier 3-4 the **counted** quorum is **Western-only**
  per the family-eligibility table (§Model family eligibility); Chinese-routed
  families may post advisory comments but do not satisfy the count. No PR is
  presented for settlement without a complete packet.

- **Batch Tier-3 settlements.** ODR-2, ODR-4, ODR-5, and the ODR-3 API surface
  are all Tier 3. Present them to the operator as a **batch** with one
  consolidated risk packet (founder settlement is batch-level risk acceptance per
  `REVIEW_AUTHORITY_PRINCIPLES.md`, not duplicative line-by-line review), so the
  operator settles a cohort in one sitting rather than context-switching per PR.

- **Settle-last.** Get each PR evidence-complete and parked *before* requesting
  settlement; do the human settlement as late as possible against the exact head
  SHA, so no settled head goes stale behind a rebase. Tier-4 items
  (ODR-6 settlement-touch) **park by default** — disclose-and-wait, never
  disclose-and-proceed (conveyor program §6 lesson 4).

- **≤ 8 open mission PRs at a time.** Cap concurrent open ODR PRs at **8**. The
  pipeline is an inventory system (conveyor program §1): admission must not
  outrun settlement. If 8 ODR PRs are open, **drain before generating the next**
  — finish the spine before widening the enrichment front.

- **Path-freeze on the spine surfaces.** While the spine is in flight, treat
  `aragora/gauntlet/odr_schema.json`, `docs/specs/OPEN_DECISION_RECEIPT.md`,
  `aragora/gauntlet/odr_signing.py`, and the `aragora-verify` package as
  **frozen contested surfaces** with one declared owner per surface; other lanes
  steer via comments, not parallel edits (conveyor program §2 class D — surface
  collisions). Schema changes are additive-only (v0.2 enrichment fields) and
  never break a published v0.1 receipt.

- **ODR right-of-way.** An open ODR spine PR has right-of-way over enrichment and
  over maintenance for settlement attention and merge slots. The settlement
  budget is spent spine-first.

---

## 5. Definition of done (closing scorecard on #8223)

The mission is complete when **all** of the following hold and are demonstrated,
not asserted (evidence before claims):

1. **Spine verifiable by a third party offline.** In a fresh venv with no Aragora
   install and no server, `pip install aragora-verify` then verifying a receipt
   produced by **this repo's own loop** returns PASS with only the public key;
   mutating one byte returns FAIL. Rekor inclusion (ODR-7, done) is checkable
   offline-of-Aragora via the verifier. (ODR-1 ✓ + ODR-2 + ODR-3.)

2. **Enrichment payloads exposed via API and in the receipt.** A debate run
   surfaces its **crux set** (ODR-4: API + CLI `--crux` + SDK + receipt field),
   its **calibrated confidence with provenance** (ODR-5: calibration-report API +
   receipt confidence block pointing at it), and — for human-settled decisions —
   a **human-oversight attestation block** (ODR-6), with explicit `autonomous`
   disposition where no human settled. Absence is always recorded, never
   fabricated (the no-fabrication rule from #8224 / the ODR spec).

3. **Art. 14 evidence pack generatable from this repo's trail.** `aragora
   compliance oversight-pack --window 30d` produces an EU AI Act Art. 14 / NIST
   agentic-profile bundle citing **real receipts and scarmani settlements** from
   this repo — the dogfood proof.

4. **Closing scorecard published on #8223:** each child closed or its residual
   honestly stated; the spine demonstrated end-to-end; and a recorded check that
   the generation-steering rule (§3) held during the mission — i.e., the
   maintenance-ratio fell and product PRs were the majority of generation while
   this mission ran. If the ratio did not fall, the steering rule failed and the
   mission is not done regardless of child-issue state.

---

## Cross-references

- Epic: [#8223](https://github.com/synaptent/aragora/issues/8223) (ODR) — the program this executes.
- Conveyor hardening: [#8344](https://github.com/synaptent/aragora/issues/8344) + `docs/superpowers/plans/2026-06-13-conveyor-hardening-program.md` — the maintenance work that moves to the bounded lane (§4 of the program: dogfood loop — the conveyor's receipts *are* the product).
- Identity gate (now merged): [#8274](https://github.com/synaptent/aragora/pull/8274) — TET T4 H1/H2 scarmani pins; the Tier-3/4 unblocker.
- Trust model: [`docs/specs/TAMPER_EVIDENT_TRAIL.md`](../../specs/TAMPER_EVIDENT_TRAIL.md) — witness / anchor / reconcile; ODR-7 is its Rekor anchor productized; ODR-6 attestation roots in its H2 pin.
- Tier + quorum law: [`docs/REVIEW_AUTHORITY_PRINCIPLES.md`](../../REVIEW_AUTHORITY_PRINCIPLES.md) — merge tiers, model-quorum packet, family eligibility.
- ODR profile: [`docs/specs/OPEN_DECISION_RECEIPT.md`](../../specs/OPEN_DECISION_RECEIPT.md) + `aragora/gauntlet/odr_schema.json` — the v0.1 content profile being filled in.
