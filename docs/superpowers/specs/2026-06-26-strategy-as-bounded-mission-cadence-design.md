# Strategy as Bounded Mission Cadence Design

**Goal:** turn the strategic assessment into one durable, resumable execution mission that makes Aragora legible and useful to outsiders: the audit layer for AI decisions, starting with PR governance.

**Canonical tracker:** [#8665](https://github.com/synaptent/aragora/issues/8665)

**Canonical intake:** [`docs/status/ROADMAP_INTAKE_REGISTER.md`](../../status/ROADMAP_INTAKE_REGISTER.md)

**Product sentence:** Aragora gives AI-assisted work a second opinion and a portable receipt: multi-model review in, signed DecisionReceipt out after the ODR offline-verifier spine and Action artifact gates are proven.

---

## Repo-Grounded Findings

This design extends live repo primitives instead of inventing another runtime.

| Surface | Current repo truth | Design implication |
|---|---|---|
| Roadmap intake | `docs/status/ROADMAP_INTAKE_REGISTER.md` exists and is the single-register rule. | Register this mission there; do not create a parallel index. |
| ODR | `docs/specs/OPEN_DECISION_RECEIPT.md`, `aragora/gauntlet/odr_schema.json`, `aragora/gauntlet/odr_export.py`, `aragora/gauntlet/odr_signing.py`, `aragora-verify/`, and the existing ODR completion plan `docs/superpowers/plans/2026-06-13-odr-completion-mission.md` already exist. | M1 extends the ODR-2 -> ODR-3 offline-verifier spine; it does not create a competing ODR executor or freeze v1.0 before the verifier proof lands. |
| GitHub Action | `action.yml` runs advisory `aragora review`, but does not emit ODR/DecisionReceipt artifacts. | M2 is a bridge from quorum evidence to a receipt artifact, then an approval-gated Action change. |
| Mission runtime | `aragora/cli/commands/mission.py`, `aragora/missions/`, and `aragora/swarm/mission.py` exist; native-mission work is being integrated. | Use the mission spine and `GateEvaluation`; do not add a parallel orchestrator. |
| External proof | Existing docs already insist that claims trail measured proof. | The mission exits only when a public/verifiable artifact exists. |

## Non-Goals

- Do not build a new launchd daemon or standing busy loop.
- Do not create a second roadmap register.
- Do not rewrite `action.yml`, `.github/workflows/`, release, or protected surfaces without exact-head operator settlement.
- Do not broaden Aragora into a generic agent platform story while the external proof is still narrow.
- Do not advance from one sub-mission to the next by narrative status alone; the proof gate must verify an artifact.

## Mission Queue

Only one sub-mission is active at a time. The next sub-mission cannot start until the current proof gate returns `PASS`.

| Mission | Scope | Existing tracker(s) | Terminal proof gate |
|---|---|---|---|
| M0: Durable strategy registration | Add this spec, register #8665, and expose the mission queue in the intake register. | #8665, #8650 | Spec and register row are on main; #8665 links the queue and existing epics. |
| M1: ODR spine to v1.0 candidate | Extend the existing ODR completion mission: finish the ODR-2 signing dependency, make the ODR-3 offline verifier prove a live receipt, then promote ODR v0.1 toward a stable v1 candidate with versioning policy, native `DecisionReceipt` mapping, independent verifier fixture, and signing proof. | #8223; existing plan `docs/superpowers/plans/2026-06-13-odr-completion-mission.md` | `aragora-verify` verifies a live Aragora-produced ODR receipt against a checked-in v1 candidate fixture and spec; only then can a GA stability claim be made. |
| M2: GitHub Action wedge | Convert PR quorum/CollectOutcome evidence into a DecisionReceipt/ODR artifact; prepare the approval-required Action update separately. | #8665 plus follow-up issue | A real PR run uploads a receipt artifact that `aragora-verify` accepts. |
| M3: Proof corpus and legibility | Publish a small corpus of real PR governance receipts and trim the front-door story to the Action/receipt wedge. | #8257 plus follow-up issue | Public proof page/release exists with receipt corpus and a README path a newcomer can understand in one sitting. |

## Gate Contract

The gate is a typed mission proof check, not a prose checklist.

Target integration point:

- Extend `aragora/swarm/mission.py::GateEvaluation` with a proof-gate payload when the native mission spine is ready for this mission class.
- Until that lands, the Roadmap Intake Register is the durable queue and each sub-mission uses explicit verification commands in its issue body and PR description.

Gate output shape:

```json
{
  "mission_id": "strategy-decision-receipt-action-proof",
  "sub_mission": "M1",
  "verdict": "pass",
  "artifact": "docs/status/generated/odr_v1/example-live-receipt.odr.json",
  "verification": "aragora-verify docs/status/generated/odr_v1/example-live-receipt.odr.json",
  "evidence": [
    "ODR schema version is v1.0",
    "native DecisionReceipt mapping documented",
    "detached signature verifies"
  ]
}
```

Fail-closed rules:

- `blocked`: artifact missing, verifier fails, or proof was run from a dirty/diverged checkout.
- `needs_human`: Tier 3/4, workflow, release, signing-key, or public-positioning decision needs exact-head operator settlement.
- `pass`: artifact exists, verifier passes, and any required human settlement is already recorded.

## M1 Implementation Plan: ODR Spine to v1.0 Candidate

M1 is the first executable sub-mission after M0. It extends [#8223](https://github.com/synaptent/aragora/issues/8223) through the existing ODR completion plan at [`docs/superpowers/plans/2026-06-13-odr-completion-mission.md`](../plans/2026-06-13-odr-completion-mission.md). That plan's ODR-2 -> ODR-3 spine remains the right-of-way: signing and the offline verifier must settle before this mission claims ODR v1.0 GA.

### Task 1: Versioning and Stability Contract

**Files:**
- Modify: `docs/specs/OPEN_DECISION_RECEIPT.md`
- Modify: `aragora/gauntlet/odr_schema.json`
- Test: `tests/gauntlet/test_odr_export.py`

Steps:

- [ ] Add a `v1.0 candidate stability contract` section to `docs/specs/OPEN_DECISION_RECEIPT.md` stating which fields are intended to become stable, which changes would require a minor version, which changes would require a major version, and how v0.1 consumers migrate after the ODR-3 verifier proof lands.
- [ ] Update `aragora/gauntlet/odr_schema.json` only after the stability contract is clear; do not rename existing fields unless the migration text names the compatibility behavior.
- [ ] Add or update tests in `tests/gauntlet/test_odr_export.py` so `ODR_VERSION`, `ODR_PROFILE_URI`, and the schema `$id` stay synchronized.
- [ ] Run `python3 -m pytest tests/gauntlet/test_odr_export.py -q`.

### Task 2: Native-to-ODR Mapping Matrix

**Files:**
- Modify: `docs/specs/OPEN_DECISION_RECEIPT.md`
- Modify: `aragora/gauntlet/odr_export.py`
- Test: `tests/gauntlet/test_odr_export.py`

Steps:

- [ ] Add a complete mapping matrix from `aragora.gauntlet.receipt_models.DecisionReceipt` fields to ODR fields.
- [ ] For each native field that does not map, record whether it is intentionally omitted, reserved for a future version, or carried through `source`.
- [ ] Add one test that fails if a newly added required ODR top-level key is not covered by the mapping matrix.
- [ ] Run `python3 -m pytest tests/gauntlet/test_odr_export.py -q`.

### Task 3: Independent Verification Fixture

**Files:**
- Create or update: `docs/status/generated/odr_v1/example-live-receipt.odr.json`
- Create or update: `docs/status/generated/odr_v1/README.md`
- Test: `tests/gauntlet/test_odr_signing.py`

Steps:

- [ ] Generate one ODR receipt from a live Aragora `DecisionReceipt` path, not a hand-written JSON object.
- [ ] Store the artifact under `docs/status/generated/odr_v1/` with a short README naming the command that produced it.
- [ ] Verify it with `aragora-verify docs/status/generated/odr_v1/example-live-receipt.odr.json` after installing the `aragora-verify` package, or the in-repo equivalent used by existing tests.
- [ ] Run `python3 -m pytest tests/gauntlet/test_odr_signing.py tests/gauntlet/test_odr_export.py -q`.

### Task 4: Gate Receipt Proof

**Files:**
- Modify: `docs/status/ROADMAP_INTAKE_REGISTER.md`
- Modify: issue [#8223](https://github.com/synaptent/aragora/issues/8223) or a child issue created from it

Steps:

- [ ] Record the M1 proof artifact path and verifier command in the register, preserving the existing ODR completion mission as the authoritative execution spine.
- [ ] Update the GitHub issue with the exact command output and the commit SHA that produced the verified receipt.
- [ ] Keep any signing-key, server endpoint, or protected deployment change parked as `needs_human`; those are not required for the M1 doc/schema proof to land.

## M2 Implementation Plan: GitHub Action Wedge

M2 starts only after M1 passes.

### Task 1: CollectOutcome to Receipt Bridge

**Files:**
- Create: `aragora/swarm/quorum_receipt.py`
- Test: `tests/swarm/test_quorum_receipt.py`

Steps:

- [ ] Convert `CollectOutcome` into a native `DecisionReceipt` with subject = PR head SHA, claim = quorum verdict, quorum = counted reviewer families, dissent = blocking/advisory dissent, and source = merge-quorum metadata.
- [ ] Mark missing fields with explicit absent markers when exporting to ODR.
- [ ] Test PASS, CHANGES-REQUESTED with blocking P1, advisory P2/P3, and no-quorum cases.
- [ ] Run `python3 -m pytest tests/swarm/test_quorum_receipt.py tests/swarm/test_quorum_evidence.py -q`.

### Task 2: Artifact Emission CLI

**Files:**
- Modify: `aragora/cli/commands/review_queue.py` or the existing review-queue command module
- Test: `tests/cli/test_review_queue.py` or the nearest existing review-queue test file

Steps:

- [ ] Add a command that emits a receipt artifact from exact-head collect-evidence data without posting or merging.
- [ ] Require explicit head SHA input or live PR exact-head confirmation.
- [ ] Refuse stale prepared artifacts.
- [ ] Run the focused review-queue tests and `python3 -m pytest tests/swarm/test_quorum_receipt.py -q`.

### Task 3: Action Update Prepared, Not Applied

**Files:**
- Prepare patch for: `action.yml`
- Prepare patch for: `.github/workflows/` only if required

Steps:

- [ ] Draft the Action change so it uploads the ODR receipt as an artifact and comments with a verifier command.
- [ ] Stop before merging any workflow/protected-surface mutation unless exact-head operator settlement is recorded.
- [ ] Run `bash scripts/automation_pr_preflight.sh origin/main HEAD` from the branch worktree before opening the PR.

## M3 Implementation Plan: Proof Corpus and Legibility

M3 starts only after the Action has produced at least one verified receipt artifact.

### Task 1: Receipt Corpus

**Files:**
- Create: `docs/status/generated/decision_receipt_proof/`
- Create: `docs/status/DECISION_RECEIPT_PROOF.md`

Steps:

- [ ] Select a bounded window of PRs with verified receipt artifacts.
- [ ] Record each PR head, receipt path, verifier command, and whether quorum caught anything a single reviewer missed.
- [ ] Publish only facts backed by the receipt corpus.

### Task 2: Front-Door Legibility

**Files:**
- Modify: `README.md`
- Modify: `docs/status/NEXT_STEPS_CANONICAL.md`
- Modify: `ROADMAP.md` only if needed after the corpus exists

Steps:

- [ ] Keep the H1 and first paragraph centered on the audit layer for AI-assisted decisions.
- [ ] Make the GitHub Action / receipt verifier path the first product wedge.
- [ ] Move broader organization-substrate language behind the stage-evolution section so it reads as earned horizon, not the current product promise.

## Cadence

The cadence is a thin gatekeeper, not a second orchestrator.

One tick does exactly this:

1. Read `docs/status/ROADMAP_INTAKE_REGISTER.md`.
2. Find the single active sub-mission in this strategy queue.
3. Run that sub-mission's proof command from a clean current-main observer.
4. If the proof passes, mark the sub-mission done and stop.
5. If the proof fails and the next action is safe, run one bounded worker batch through the existing mission/worker substrate and stop.
6. If the next action is Tier 3/4, workflow, release, protected, signing-key, or product-positioning risk, park with an operator receipt and stop.

Circuit breaker: if no external proof progresses for three consecutive ticks, halt the strategy mission and open a human-readable blocker on #8665.

## Acceptance Criteria

- The register has exactly one row and one mission-queue section for this strategy mission.
- M0 is explicitly merge-gated until this registration PR lands on `main`.
- M1 extends #8223 and the existing ODR completion plan instead of creating a competing receipt spec or executor.
- M2 prepares protected Action/workflow changes but does not land them without exact-head settlement.
- M3 publishes proof before broadening claims.
- A newcomer can state the wedge in one sentence: multi-model review in, signed DecisionReceipt out.
