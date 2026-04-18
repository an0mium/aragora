# Dialectical Runtime — Closing the Loop Between Claims, Cruxes, and Proof-Carrying Code

> **Status:** additive future tranche layered on top of DIC-13..22
> **Created:** 2026-04-18
> **Queue policy:** planning issues only; no `boss-ready`, no live queue mutation
> **Gate:** activation gated on DIC-20 (decay monitor), DIC-21 (quarantine policy), and DIC-22 (verified replacement pipeline) being in place, and on proof-first Foreman reliability per [`NEXT_STEPS_CANONICAL.md`](../status/NEXT_STEPS_CANONICAL.md)
> **Relationship:** strictly **additive** to [`EPISTEMIC_CI_AND_CRUX_ENGINE.md`](EPISTEMIC_CI_AND_CRUX_ENGINE.md) and [`2026-04-16-crux-mode-design.md`](2026-04-16-crux-mode-design.md); does not remove, replace, or reorder any existing DIC item
>
> **GitHub tracking:**
> - Epic: [#6223](https://github.com/synaptent/aragora/issues/6223)
> - DIC-23 Runtime Loop Orchestrator: [#6217](https://github.com/synaptent/aragora/issues/6217)
> - DIC-24 Epistemic Genealogy Ledger: [#6218](https://github.com/synaptent/aragora/issues/6218)
> - DIC-25 Adversarial World-State Stress-Test: [#6219](https://github.com/synaptent/aragora/issues/6219)
> - DIC-26 Belief Coherence Monitor: [#6220](https://github.com/synaptent/aragora/issues/6220)
> - DIC-27 Operator Crux Arbitration: [#6221](https://github.com/synaptent/aragora/issues/6221)
> - DIC-28 Proactive Crux Gardening: [#6222](https://github.com/synaptent/aragora/issues/6222)

## Thesis

DIC-13..22 delivers the building blocks of a decision-integrity layer over code: executable claims, cruxes, proof-carrying code units, decay signals, quarantine policy, and verified-replacement hooks. Each is useful in isolation. None of them, individually, make the system *feel* alive.

The missing abstraction is the **Dialectical Runtime Loop** — a thin orchestration and genealogy layer that binds those primitives into a single, observable, receipt-carrying lifecycle:

```
decision receipt -> claim ledger -> proof-carrying code unit
                                 |
                                 v
                      epistemic decay signal (DIC-20)
                                 |
                                 v
                       crux-finder debate (DIC-15)
                                 |
                                 v
              quarantine/fallback recommendation (DIC-21)
                                 |
                                 v
             verified replacement candidate (DIC-22)
                                 |
                                 v
                  new decision receipt, chained to prior
```

Once that loop exists, code paths stop being static text and start being *living arguments*: every artifact in production knows what justifies it, detects when that justification weakens, names where reasonable agents would disagree about the fix, chooses a safe action under clear policy, and — when allowed — replaces itself through a receipt-carrying debate whose entire ancestry is inspectable.

This is the synthesis point of three product visions that already live in the roadmap:

- **Crux Engine** — locates where reasonable people diverge (DIC-15, `aragora/reasoning/cruxset.py`, `aragora/reasoning/crux_detector.py`).
- **Epistemic CI** — makes important claims executable, evidence-linked, and freshness-gated (DIC-13/14, `aragora/epistemic/claim_verifier.py`).
- **Epistemic Runtime** — attaches claims, cruxes, and verifiers to the code paths they justify (DIC-19/20/21/22, `aragora/epistemic/proof_unit.py`, `aragora/epistemic/decay_monitor.py`).

The Dialectical Runtime Loop is not a new product. It is the connective tissue that lets the three already-approved product directions behave as one system.

## Why This Belongs In Aragora

Aragora already owns every primitive required:

| Capability | Module | Used for |
|---|---|---|
| Executable claims | `aragora/epistemic/claim_verifier.py` | DIC-14 input to decay signals |
| Proof-carrying code units | `aragora/epistemic/proof_unit.py` | DIC-19 anchors for assumptions, receipts, verifiers |
| Decay signals | `aragora/epistemic/decay_monitor.py` | DIC-20 integrity scoring |
| CruxSet contract + emitter | `aragora/reasoning/cruxset.py`, `aragora/reasoning/cruxset_emission.py` | DIC-15 disagreement surfaces |
| Crux-finder consensus mode | `aragora/debate/crux_mode.py` (landed via Crux A1) | Produces CruxSet on demand |
| CruxReceipt + exporter | `aragora/gauntlet/receipt_models.py` (landed via Crux A2) | Signed, exportable disagreement maps |
| Failed-claim / open-crux bridge | `aragora/epistemic/followup.py` | DIC-17 bounded follow-up proposals |
| Organisational truth map | DIC-18 report surface | Current-state belief inventory |
| Belief network + factors | `aragora/reasoning/belief.py` | Coherence checks across claims |
| Formal verification stubs | `aragora/verification/formal.py` (Lean, Z3) | Repair-candidate verifier hooks |
| Nomic Loop orchestration | `aragora/nomic/autonomous_orchestrator.py`, `scripts/nomic_loop.py` | Debate-driven repair path |
| ShiftLedger + proof-first runtime | `scripts/run_proof_first_shift.py`, `aragora/swarm/*` | Bounded work contract for repairs |
| Gauntlet receipt signing | `aragora/gauntlet/receipt_models.py`, `signing.py` | Chainable decision receipts |

Nothing in this plan requires a new subsystem, a new dependency, or a rewrite. It reuses six existing directories and introduces one orchestration module, one genealogy ledger, one stress-test driver, one coherence monitor, one operator arbitration surface, and one scheduled gardener.

## Non-Goals

- **Do not** interrupt, reorder, or re-scope DIC-13..22. Each lands on its own acceptance criteria.
- **Do not** ship live hot-swap behavior. All repair output remains report-only, patch, or pull-request until an explicit safety gate opens per DIC-21's policy model.
- **Do not** add generic auto-dispatch to the live queue. Every follow-up proposal from this tranche flows through the DIC-17 bridge with its existing bounded-work controls.
- **Do not** create a second autonomy stack. The Nomic Loop, Swarm Supervisor, and proof-first queue remain the single execution substrate; this tranche orchestrates *events*, it does not spawn new runners.
- **Do not** replace the existing Knowledge Mound, Belief Network, or provenance models. Genealogy and coherence are *views* over those stores, not new stores.
- **Do not** create `boss-ready` or other live dispatch labels from any proposal until the tranche activation gate opens.

## The Six New Capabilities (DIC-23..28)

Each is planning-only until its prerequisite DIC-* items are green. Each is scoped to be an additive seam onto existing modules, not a rewrite.

### DIC-23 — Dialectical Runtime Loop Orchestrator

**Core idea:** a thin event-driven orchestrator that subscribes to decay signals, triggers targeted crux-finder debates on the decayed unit's assumptions, applies quarantine policy, and emits a single linked repair candidate through the DIC-22 pipeline.

**Module:** `aragora/epistemic/runtime_loop.py` (new), ~200-300 LOC.

**Flow (pseudocode):**

```python
# aragora/epistemic/runtime_loop.py
@dataclass(frozen=True)
class DialecticalEvent:
    event_id: str
    unit: ProofCarryingCodeUnit
    decay: DecaySignal
    cruxset: CruxSet | None
    quarantine_action: str  # report_only | degrade | fail_closed | repair_required
    repair_candidate_id: str | None
    receipt_chain: list[str]  # prior receipt IDs

async def handle_decay(
    unit: ProofCarryingCodeUnit,
    signal: DecaySignal,
    *,
    enable_crux_probe: bool = False,
    enable_repair_proposal: bool = False,
) -> DialecticalEvent:
    """Default: report-only trace. Flags must explicitly opt in."""
    # Step 1: if signal escalates past report_only and crux probing is on,
    #         invoke crux-finder on the unit's assumptions to surface the
    #         load-bearing disagreement about what the repair should be.
    # Step 2: consult DIC-21 policy to select quarantine action.
    # Step 3: if repair proposal is enabled and policy permits, emit a
    #         DIC-22 bounded repair spec linked to this DialecticalEvent.
    # Step 4: write a chained receipt that references the prior decision
    #         receipt, the decay signal, the CruxSet (if present), and the
    #         repair candidate ID (if present).
    ...
```

**Defaults (safety-critical):**

- `enable_crux_probe=False`, `enable_repair_proposal=False`. Pure report-only trace is the first ship.
- When both flags are false, the orchestrator produces a DialecticalEvent and emits events to telemetry, nothing else. This is the "observability before action" posture that the proof-first tranche already requires.
- When `enable_crux_probe=True`, it runs crux-finder with `top_k=3`, `min_score=0.4`, bounded at one debate per unit per 24h.
- When `enable_repair_proposal=True` *and* DIC-21 policy returns `repair_required`, it drafts *one* DIC-22 repair spec; it does not file issues, does not add labels, does not route to the live queue. A follow-up issue comes only through the DIC-17 bridge.

**Why this is valuable even as report-only:** the event stream itself becomes a canonical, per-code-unit "why did this look fragile today?" log — directly usable by the DIC-18 truth map and by operators investigating incidents.

**Acceptance shape:**

- one module producing DialecticalEvents from decay signals
- deterministic tests with mocked `ProofCarryingCodeUnit` and `DecaySignal`
- zero mutation of claim manifests, no issue creation, no queue effect in default mode
- structured telemetry (event IDs, receipt chain, timings) emitted via the existing observability paths

### DIC-24 — Epistemic Genealogy Ledger

**Core idea:** a content-addressed ancestry ledger showing, for any code unit, the full lineage: the original decision debate, every decay event, every crux-finder debate triggered, every repair proposal, and every receipt produced. In other words, "why does this function look the way it does today?"

**Module:** `aragora/epistemic/genealogy.py` (new), ~150-200 LOC. Thin layer over the existing receipt store and Knowledge Mound.

**Record shape:**

```yaml
genealogy_id: gen.proof_first.shift.green_criteria
code_unit_id: proof_first.shift.green_criteria
ancestry:
  - kind: decision_receipt
    receipt_id: decision.bc12.green_shift_criteria
    created_at: 2026-03-30T14:22:00Z
  - kind: decay_signal
    decay_id: decay.proof_first.shift.green_criteria.2026-04-16
    integrity_score: 0.42
    reasons: [failed_claim, stale_evidence]
  - kind: crux_receipt
    receipt_id: crux-7a4f2b9c
    load_bearing_cruxes: [crux.soak_equivalence]
  - kind: repair_proposal
    proposal_id: repair.bc12.green_soak_delta
    status: draft
  - kind: decision_receipt
    receipt_id: decision.bc12.green_shift_criteria.v2
    supersedes: decision.bc12.green_shift_criteria
checksum: sha256:...
```

**Key property:** every receipt type in aragora already has an ID and a checksum. Genealogy is a *join* over the receipt store, not a parallel store. The module is small because the data lives elsewhere; this is the viewer + stable ID assignment + chain-validation code.

**Acceptance shape:**

- read-only lineage retrieval API over existing receipt/KM storage
- deterministic join given a code_unit_id
- chain validation checksum matches an SHA-256 over the canonical-sorted ancestry
- no new persistence model; genealogy records are derived, not stored

**Dogfood target:** render the genealogy of one real code unit (`scripts.run_proof_first_shift.evaluate_green_shift`) as part of the DIC-18 truth map's drill-down view.

### DIC-25 — Adversarial World-State Stress-Test

**Core idea:** *proactively* probe proof-carrying code units against hypothetical world-state perturbations rather than waiting for reality to invalidate them. Output is a ranked list of units whose assumptions are fragile under plausible-future scenarios.

**Module:** `aragora/epistemic/stress_test.py` (new), ~200-250 LOC.

**Mechanism:**

- Synthetic perturbations are expressed as *claim mutations*: "what if claim C (`'Benchmark freshness can be determined from the published proof surface'`) were false tomorrow?"
- For each unit, for each assumption, mutate the claim's verification result to `fail` or `stale` in-memory and re-run the decay monitor (DIC-20) against the mutated state.
- Rank units by the *integrity delta* under each perturbation. Units whose integrity drops below a policy threshold under plausible perturbations are flagged as fragile.
- Optional: run crux-finder (DIC-15) against a unit's assumptions to identify which assumption is load-bearing for its integrity score; this gives a prioritised "if any of these 2 things shift, quarantine this code" list.

**Bounded scope:**

- perturbation catalog is a YAML file under `docs/status/stress_test_scenarios/` curated by operators, not generative
- first catalog targets four plausible-future scenarios: (a) a dependency CVE invalidates a `library_version_ok` claim; (b) an API provider changes rate limits; (c) a benchmark corpus is revised; (d) a policy doc is updated
- runs offline; no network, no live queue mutation, no issue creation
- emits a markdown report + JSON artifact under `docs/status/generated/stress_test_reports/`

**Acceptance shape:**

- deterministic report from a curated scenario set and a set of proof-carrying code units
- integrity deltas are reproducible given the same claim-mutation inputs
- no runtime effect on claim verification results (mutation is in-memory per-run)
- a flag prevents DIC-23 from consuming stress-test output to *create* quarantines in the first ship; stress-test output is operator-visible only

### DIC-26 — Belief Coherence Monitor

**Core idea:** use the existing `BeliefNetwork` to detect contradictions and confidence-rot across the organisation's claim ledger. If claim A asserts X and claim B asserts not-X, or their evidence overlaps in ways that imply contradiction, surface it as a coherence signal before it becomes an incident.

**Module:** `aragora/epistemic/coherence.py` (new), ~150-200 LOC. Reuses `aragora/reasoning/belief.py`.

**Mechanism:**

- Ingest the current DIC-13/14 claim ledger into an ephemeral `BeliefNetwork` (one node per claim, edges for evidence overlap or declared dependencies).
- Run belief propagation to find:
  - **hard contradictions** — two high-confidence claims whose statements are mutually exclusive
  - **evidence conflicts** — two claims sharing an evidence source that they interpret oppositely
  - **confidence rot** — claims whose propagated confidence falls below a threshold due to stale or failed evidence
- Emit a coherence report with a deterministic checksum. Optionally feed coherence issues into the DIC-17 bridge as follow-up proposals.

**Acceptance shape:**

- report-only; no mutation of claim files or receipts
- deterministic output given the same claim set
- a documented threshold matrix (`confidence_rot >= 0.6`, `hard_contradiction_confidence_delta >= 0.4`, etc.)
- focused tests using synthetic claim sets: (a) two contradicting claims surface a hard-contradiction; (b) a claim with decayed evidence surfaces confidence rot; (c) two claims sharing a stale evidence file surface an evidence conflict

**Dogfood target:** run against the initial DIC-13 claim set (benchmark truth, rescue productization, queue governance, etc.) and publish the report under `docs/status/generated/coherence_reports/`.

### DIC-27 — Operator Crux Arbitration

**Core idea:** when a crux persists as load-bearing across N consecutive debates on the same question (the system cannot resolve it by further argument), escalate to a human operator with the full CruxSet as evidence. The operator's resolution becomes a *priors update* for the belief network, with the arbitration itself recorded as a receipt.

**Module:** `aragora/epistemic/arbitration.py` (new), ~150-200 LOC. Plus a small CLI surface: `aragora crux arbitrate <cruxset-id>`.

**Mechanism:**

- A `PersistentCrux` is a crux that appears with `load_bearing_score >= 0.6` in three consecutive debates on questions sharing a `question_family_id` (e.g. repeated disagreement about the same soak-policy question).
- The arbitration record attaches: (a) the CruxSet IDs, (b) the operator's chosen side with a rationale, (c) a bounded expiry window (default 90 days) after which the arbitration is re-examined, (d) the optional evidence citation the operator attached.
- Downstream: the belief network updates its priors on claims tied to the arbitrated crux; DIC-22 repair proposals prefer solutions consistent with the arbitration; any new debate on the same `question_family_id` receives the arbitration as pinned context.
- Arbitrations are **reversible**: an operator can explicitly revoke one with a new receipt; the system never silently removes them.

**Acceptance shape:**

- no auto-escalation to humans in the default ship; operators opt in to the watchlist per question family
- `aragora crux arbitrate` CLI reads a cruxset ID, prompts the operator for the chosen side and rationale, and writes a signed arbitration receipt
- arbitration records are inspectable via the DIC-18 truth map and carry their own checksum
- reversal is a first-class operation with its own receipt, not a deletion

**Why this is distinctive:** most AI systems treat human override as a black-box fallback. This design treats it as *another kind of receipt* — inspectable, reversible, and tied to the specific disagreement it resolved. The goal is to make operator judgment a visible, auditable part of the belief network rather than invisible tribal knowledge.

### DIC-28 — Proactive Crux Gardening

**Core idea:** periodically re-examine resolved and outstanding cruxes against current world state. Are the reasons we decided a crux were settled still valid? Have new cruxes emerged on the same questions? Which previously-resolved cruxes have become stale due to evidence decay?

**Module:** `aragora/epistemic/gardening.py` (new), ~150 LOC. Scheduled by `aragora.scheduler`, same pattern as the existing `confidence_decay_scheduler` (`aragora/knowledge/mound/...`).

**Mechanism:**

- A scheduled pass runs against the archive of closed CruxSets and resolved arbitrations.
- For each resolved crux: re-verify that the evidence it relied on is still fresh (via DIC-14) and that no contradicting claim has entered the ledger (via DIC-26).
- For each outstanding crux: check whether the world-state stress-test (DIC-25) now ranks it as materially more fragile than when it was last examined.
- Emit a gardening report. Optionally feed high-priority re-examination candidates through DIC-17 as bounded follow-ups (gated, same as other DIC-17 output).
- Default cadence: weekly; operator-configurable; disabled by default in the first ship.

**Acceptance shape:**

- deterministic output given the same stored CruxSet archive and claim state
- report-only default; no auto-debate, no auto-issue-creation
- stress-test integration is via explicit function call, not via module-level side effect
- focused tests: (a) a resolved crux whose evidence went stale surfaces; (b) an outstanding crux whose fragility score dropped does not; (c) one new crux emerging on the same family is flagged

## Interaction With The Existing DIC Tranche

This is the exact interaction surface. No existing acceptance criteria change.

| Existing item | What DIC-23..28 adds | Change to existing item |
|---|---|---|
| DIC-13 Executable Claim Manifest | Coherence monitor consumes it | none |
| DIC-14 Claim Verification Runner | Stress-test mutates its outputs in-memory | none |
| DIC-15 CruxSet + crux-finder mode | Runtime loop invokes it on demand; arbitration escalates persistent cruxes | none |
| DIC-16 Receipt/KM provenance | Genealogy ledger reads from it; arbitration writes to it | none |
| DIC-17 Failed-claim / open-crux bridge | Runtime loop, stress-test, coherence, and gardening can file via it (gated) | none — uses existing bounded-work controls |
| DIC-18 Organisational Truth Map | Displays DialecticalEvents, genealogy, stress-test, coherence, arbitration, and gardening reports | additive drill-downs only |
| DIC-19 Proof-carrying code unit | Stress-test and runtime loop consume it | none |
| DIC-20 Epistemic Decay Monitor | Runtime loop subscribes to its signals | none |
| DIC-21 Quarantine Policy | Runtime loop applies it; gardening proposes policy-threshold reviews | none |
| DIC-22 Verified Replacement Pipeline | Runtime loop emits one bounded repair spec per decay event; arbitration pins priors | none — repair remains receipt-gated |

## Activation Gate

DIC-23..28 **cannot** enter active implementation until:

1. DIC-20 decay monitor is production-green on at least three proof-carrying code units.
2. DIC-21 quarantine policy model has landed with at least the `report_only`, `degrade`, `fail_closed`, and `repair_required` classes and inter-class promotion rules.
3. DIC-22 verified-replacement pipeline has shipped at least the "draft-PR" output path, with the "no-hot-swap guardrail" test in place.
4. Proof-first Foreman reliability goals per [`NEXT_STEPS_CANONICAL.md`](../status/NEXT_STEPS_CANONICAL.md) have stabilised — i.e. recurring benchmark truth publication stays complete and fresh on `main` without operator babysitting.

Until all four are true, the DIC-23..28 issues remain planning-only, labelled with `enhancement` and topic labels only (never `boss-ready`). This is the same gating pattern that [`EPISTEMIC_CI_AND_CRUX_ENGINE.md`](EPISTEMIC_CI_AND_CRUX_ENGINE.md) already applies to DIC-13..22.

## Implementation Sequencing (after activation gate opens)

1. **DIC-23 Runtime Loop Orchestrator — report-only.** One module, one event stream, deterministic tests, zero action beyond telemetry and receipt-chain writes. Smallest possible slice.
2. **DIC-24 Epistemic Genealogy.** Read-only lineage view. Wires into DIC-18 truth map drill-downs. No new storage.
3. **DIC-26 Belief Coherence Monitor.** Report-only coherence passes. Feeds DIC-17 bridge only when explicitly enabled.
4. **DIC-25 Adversarial World-State Stress-Test.** Curated perturbation catalog + offline report. Does **not** yet drive DIC-23 actions.
5. **DIC-28 Proactive Crux Gardening.** Scheduled re-examination. Reuses the stress-test and coherence modules. Gated behind a configuration flag.
6. **DIC-27 Operator Crux Arbitration.** The human-in-the-loop surface. Ships last because it is the most behaviour-changing: operator decisions start flowing into priors.

Each step ships individually. Each can be reversed by removing one module and one flag. Nothing depends on a later step landing.

## Dogfood Targets

- **DIC-23** — handle_decay invoked on `scripts.run_proof_first_shift.evaluate_green_shift` emits a DialecticalEvent with full receipt chain under `docs/status/generated/dialectical_events/`.
- **DIC-24** — the genealogy of the same code unit renders in the DIC-18 truth map with all four ancestry kinds visible.
- **DIC-25** — the stress-test catalog contains the four scenarios listed above and runs on ≥3 units without network.
- **DIC-26** — a coherence report on the initial DIC-13 claim set runs without warnings and flags any synthetic-contradiction test.
- **DIC-27** — `aragora crux arbitrate <cruxset-id>` writes a signed arbitration receipt for one dogfood CruxSet (e.g. the soak-equivalence question from `EPISTEMIC_CI_AND_CRUX_ENGINE.md`).
- **DIC-28** — one scheduled gardening pass over closed CruxSets produces a deterministic report under `docs/status/generated/gardening_reports/`.

## Risk And Non-Risk

**Non-risk (on purpose):**

- *No live queue impact in the first ship.* Every module in DIC-23..28 is report-only by default. All issue creation flows through the existing DIC-17 bridge, which is already gated.
- *No new storage schemas.* Genealogy, coherence, stress-test, and gardening are views over existing receipt, claim, and belief-network stores.
- *No in-memory hot-swap.* DIC-21 and DIC-22 already prohibit production hot-swap; DIC-23 explicitly does not add a new hot-swap path.
- *No dependency additions.* Every module uses the existing Python 3.11 standard library plus aragora internals.

**Actual risk and mitigation:**

| Risk | Mitigation |
|---|---|
| Runtime loop becomes a noisy event source that buries real signal | Default `enable_crux_probe=False`; telemetry only; DIC-18 truth map displays dedupe-by-code-unit views |
| Stress-test scenarios drift from plausible perturbations | Scenario catalog is operator-curated and versioned; each scenario carries a rationale and an expiry window |
| Coherence monitor surfaces false contradictions from non-contradictory paraphrases | Use the existing semantic similarity machinery (`aragora/debate/convergence.py`, PR #723 migration) rather than string matching; require `confidence_delta >= 0.4` before flagging |
| Operator arbitration becomes stale and the system trusts outdated judgment | Default 90-day expiry; arbitrations are displayed with age in DIC-18; gardening pass (DIC-28) re-examines them |
| Genealogy ledger grows without bound | Lineage is derived from existing stores; bounded retention is inherited from them, not re-implemented |
| Scope creep back into hot-swap territory | DIC-22's "no-hot-swap guardrail" test remains authoritative; DIC-23's repair-proposal output is strictly a patch/PR/spec, never in-memory replacement |

## Relationship To Current Roadmap

This tranche is a **strict extension** of the existing [`EPISTEMIC_CI_AND_CRUX_ENGINE.md`](EPISTEMIC_CI_AND_CRUX_ENGINE.md) plan. It does not change that plan's sequencing, acceptance criteria, or activation gate. It adds six planning-only issues and one synthesis surface on top.

It also strengthens Pillar 5 of [`CANONICAL_GOALS.md`](../CANONICAL_GOALS.md) — "Cryptographic Receipts and Auditability" — by making the lineage of decisions, cruxes, and code paths into a single inspectable artifact. The existing language in Pillar 5 already references proof-carrying code; this tranche is the loop that connects the proof to its evolution.

It is a **P4 — Strategic Evolution** backlog item on [`FEATURE_GAP_LIST.md`](../FEATURE_GAP_LIST.md), consistent with the existing placement of Epistemic CI / Crux Engine work.

## First Ship — The Smallest Useful Thing

If only one thing from DIC-23..28 lands, it should be this:

**A report-only DialecticalEvent trace (DIC-23) joined with a read-only Epistemic Genealogy view (DIC-24), both visible in the DIC-18 truth map drill-down for one real proof-carrying code unit.**

That single slice gives operators a living answer to the question *"why does this code look like this today, and how has its justification changed?"* without changing any runtime behaviour, without adding any storage, without creating any issues, and without touching the live queue. Everything else in DIC-23..28 is a multiplier on that foundation.

## What This Adds Versus What Already Exists

| Layer | Already on `main` | Added by DIC-23..28 |
|---|---|---|
| Claim manifests | DIC-13 | — |
| Claim verification runner | DIC-14 | — |
| CruxSet contract + crux-finder | DIC-15 | — |
| Receipt/KM provenance | DIC-16 | Genealogy join (DIC-24) |
| Failed-claim / open-crux bridge | DIC-17 | Runtime loop feeds it (DIC-23); coherence feeds it (DIC-26); gardening feeds it (DIC-28) — all gated |
| Truth map report | DIC-18 | Dialectical event drill-down (DIC-23); genealogy drill-down (DIC-24) |
| Proof-carrying code unit schema | DIC-19 | Stress-test consumer (DIC-25); runtime-loop consumer (DIC-23) |
| Decay monitor | DIC-20 | Runtime-loop subscriber (DIC-23) |
| Quarantine policy | DIC-21 | Runtime-loop policy applicator (DIC-23) |
| Verified replacement pipeline | DIC-22 | Runtime-loop proposer (DIC-23) |
| Observable per-unit evolution story | — | **DIC-23, DIC-24** |
| Proactive fragility surface | — | **DIC-25** |
| Cross-claim coherence | — | **DIC-26** |
| Operator arbitration as a receipt | — | **DIC-27** |
| Scheduled re-examination of cruxes | — | **DIC-28** |

The first ten rows already exist or are in-flight. The last five rows are what this plan adds — and every one of them is a thin seam, not a subsystem.

## Closing Frame

DIC-13..22 turns aragora's debate and receipt infrastructure into a decision-integrity discipline. DIC-23..28 turns that discipline into a **loop**: claims evolve, cruxes get examined and re-examined, code paths carry their own proofs, decay becomes observable, repair becomes receipt-carrying, and operator judgment becomes a first-class artifact rather than tribal knowledge.

The output is code that behaves less like static text and more like a continuous, inspectable argument between an organization's intent and the world it operates in. That is the substrate aragora has been building toward. The six items here are the connective tissue — no more, no less.
