---
title: Strategy as a Bounded-Mission Cadence — design
description: Strategy as a Bounded-Mission Cadence — design
---

# Strategy as a Bounded-Mission Cadence — design

**Date:** 2026-06-26
**Status:** design (approved in brainstorming; grounded in a 5-agent repo audit)
**Owner:** founder
**Relation:** consolidation, not greenfield — wires together machinery already on `main`.

## Problem

A strategic assessment produced five moves to make Aragora useful to others and
defensibly unique (narrow the use case, ship a CI Action, spec a portable
DecisionReceipt, quarantine sprawl, publish one external proof). The recurring
failure mode — recorded in founder memory as *"loop producing loop → publish ONE
artifact, exit"* — is that strategy becomes another doc that is never executed, or
becomes an open-ended autonomous loop that produces motion without a shipped
artifact.

The goal of this design is **not** a new execution engine. It is to bind the
strategy to the durable machinery that already exists, as **one canonical mission
decomposed into a short queue of bounded, exit-gated sub-missions**, advanced on a
standing cadence whose progress is gated by Aragora's own product (a verifiable
DecisionReceipt + model quorum).

## Decisions (from brainstorming)

1. **Continuity model = hybrid: bounded missions on a standing cadence.** A
   perpetual scheduler that runs exactly **one** bounded, exit-defined sub-mission
   at a time and **refuses to start the next until the current one's external
   proof artifact exists and verifies.** Continuity means *durable + resumable +
   advanced on a cadence*, **not perpetual motion**.
2. **This session:** write this spec, then produce a decision-complete
   implementation plan. **M1 does not start until the founder approves the plan.**
   M1 is scoped to **ODR v1.0 docs + verification only** — no README,
   `action.yml`, `.github/workflows/`, server verify endpoint, auto-signing-in-gate,
   or any Tier 3-4 surface.

## What already exists (audit, verified on `main`)

| Capability | Where | State |
|---|---|---|
| Portable receipt spec (ODR v0.1) | `docs/specs/OPEN_DECISION_RECEIPT.md`, `aragora/gauntlet/odr_schema.json` | draft v0.1 (epic #8223) |
| Receipt export pipeline | `aragora/gauntlet/odr_export.py` (`decision_receipt_to_odr`) | wired; `aragora receipt export --format odr` |
| Ed25519 detached signing | `aragora/gauntlet/odr_signing.py` | merged; key via Secrets Manager |
| Standalone verifier | `aragora-verify/` (PyPI v0.1.0) | published; 5 checks, zero Aragora dep |
| Native DecisionReceipt (v1.1) | `aragora/gauntlet/receipt_models.py` | internal dataclass, `schema_version` |
| Quorum machinery | `aragora/swarm/quorum_evidence.py` | tier rules, reviewer CLI→API→OpenRouter fallback |
| GitHub Action | `action.yml` | advisory only (`aragora review`); no quorum, no receipt |
| Native mission engine | `aragora/nomic/mission.py`, `aragora/cli/commands/mission.py` | merged (PRs #8465–8485); flag `enable_native_mission` (default OFF) |
| Mission relay / park-and-notify | `aragora/nomic/mission_relay.py` | merged |
| Gate primitives | `aragora/swarm/mission.py` (`GateType`, `GateVerdict`, `GateEvaluation`) | merged; `PUBLISH_READY` gate type exists |
| Non-halting tick loop | `aragora/swarm/boss_loop.py` + `com.aragora.swarm-boss-loop` launchd | proven; **currently frozen (head-freeze)** |
| Intake Register | `docs/status/ROADMAP_INTAKE_REGISTER.md` | on `main`; single-register rule |
| elves-aragora gate | plugin skill: 7-step batch gate, Tier 0-2 auto-settle, Tier 3-4 hard-stop | invokable |
| Operating contract | `docs/AGENT_OPERATING_CONTRACT.md` (§Conductor), `docs/REVIEW_AUTHORITY_PRINCIPLES.md` (Tier 0-4) | binding |

**Implication:** three of the five strategic moves are 80%+ built. The work is
wiring, a versioning contract, a receipt bridge, a thin gated metronome, and one
founder decision — not new subsystems.

## Architecture

```
  Intake Register  ──────────────────────────── durable home + queue
  (docs/status/ROADMAP_INTAKE_REGISTER.md          status per sub-mission
   + a new "Mission Queue" section)                 single active at a time
        │  reads
        ▼
  MissionMetronome  ───────────────────────────  thin standing cadence (NEW, slim)
   (aragora/missions/metronome.py, default-OFF)    each tick:
        │                                            1. read survival/register (disk truth)
        │                                            2. find single active sub-mission
        │                                            3. evaluate mission gate
        │                                            4. PASS → mark DONE, advance, stop
        │                                            5. else → run one elves-aragora batch, stop
        │  invokes per tick
        ▼
  Mission Gate  ───────────────────────────────  GateType.PUBLISH_READY (extend existing)
   (extends aragora/swarm/mission.py                proof-artifact existence + verifiability
    GateEvaluation → mission-proof gate)            DONE only if external proof verifies
        │  delegates the actual work to
        ▼
  elves-aragora batch  ────────────────────────  7-step governed batch (existing skill)
   (implement → local truth → quorum debate →       Tier 0-2 auto-settle
    DecisionReceipt → tier settlement → close)      Tier 3-4 + approval-surface → HARD STOP
        │  produces
        ▼
  DecisionReceipt (ODR)  ──────────────────────  the proof artifact == the product
   verified by aragora-verify                       dogfood: the gate IS a receipt
```

### Components

1. **Intake Register — Mission Queue section (extend existing file).**
   Add a `## Strategy Mission Queue` section to
   `docs/status/ROADMAP_INTAKE_REGISTER.md` (NOT a parallel file — single-register
   rule). One row per sub-mission: `id | title | tier | status | external-proof
   gate | tracking`. `status ∈ \{queued, active, blocked-on-proof,
   blocked-on-human, done\}`. This is the durable cross-session state; any agent or
   human resumes by reading this one file. Backed by a GitHub epic.

2. **Mission Gate (extend `aragora/swarm/mission.py`).** A function that, given a
   sub-mission id, returns a `GateEvaluation` with `gate_type=PUBLISH_READY` and
   `verdict ∈ {PASS, BLOCKED, NEEDS_HUMAN}` by checking the **external proof
   artifact's existence + verifiability** (e.g. M1: run `aragora-verify` on the
   example receipt; M2: the Action ran green and its receipt verifies). `required_evidence`
   lists what is missing. This is the mechanical enforcement of "publish ONE
   artifact, exit": no verifiable artifact → no advancement.

3. **MissionMetronome (new, slim — `aragora/missions/metronome.py`).** A thin
   service. One tick = read register/survival state → find the single active
   sub-mission → call the mission gate → if `PASS`, mark `done` + advance the queue
   pointer + stop; else run **one** bounded `elves-aragora` batch toward the
   sub-mission, then stop. It does **not** modify the frozen `boss_loop`; it reuses
   the launchd tick pattern (`com.aragora.swarm-mission-metronome`, created
   **disabled**). The metronome is a metronome + gatekeeper — the real work is the
   governed batch.

### Data flow

`founder approves spec` → register Mission Queue rows (M0..M3) →
metronome tick reads register → active = first non-done row → mission gate
evaluates proof → not done → elves-aragora batch runs (implement + quorum +
receipt) → receipt written + tier-settled → next tick re-evaluates gate → gate
PASS → row marked `done` → next row becomes active. Repeat until queue empty,
then exit.

## The mission queue (5 moves → M0..M3, dependency-ordered)

| # | Sub-mission | Covers | Tier | Terminal DONE (external proof gate) |
|---|---|---|---|---|
| **M0** | Mission Queue section in the Intake Register + parser test (autonomous). GitHub epic creation parks for the operator. | (infra) | 0 | Register section exists with M1-M3 rows and the parser test passes |
| **M1** | **ODR v1.0 GA (docs + verification only)** — extend epic #8223: add versioning/stability contract to the spec; formalize native `DecisionReceipt` ↔ ODR mapping; checked-in example receipt + verification test | #3 | 0-1 | `pip install aragora-verify` then `aragora-verify <receipt.json>` independently verifies an **unsigned** receipt produced from a native `DecisionReceipt` via `odr_export`, against the published ODR profile; CI verifies a checked-in example receipt |
| **M1-defer** | Production wiring deferred out of M1: auto-sign-in-gate (#8225), server verify endpoint (#8226) | #3 | **3 → HARD STOP** | parked for founder/operator; not part of the autonomous M1 slice |
| **M2** | **Action wedge** — build `CollectOutcome → DecisionReceipt` bridge; rewrite `action.yml` to run the quorum and emit a verifiable receipt artifact + PR comment | #2 | 1-2 bridge; **2-3 workflow change = approval-required → HARD STOP for founder** | The Action runs **green on a real PR in this repo** and uploads a receipt that `aragora-verify` passes |
| **M3** | **Proof corpus + legibility** — run the gate across a window of PRs, publish receipts + a short "what the quorum caught" writeup; trim README to one sentence; quarantine sprawl behind `aragora/experimental/` | #1, #4, #5 | mixed; **README narrative = Tier 3 founder call → HARD STOP** | A public artifact (release/page) with the receipt corpus live, AND a stranger-readable README + ≤5 documented load-bearing modules on `main` |

**The README reframe is entirely M3, not M1.** The one-sentence reframe is the same
act as the narrative decision (you cannot write the sentence until the narrative is
chosen), so it parks for the founder under M3. M1 touches **no** README,
`action.yml`, `.github/workflows/`, server verify endpoint, auto-signing-in-gate,
or any Tier 3-4 surface — it is ODR v1.0 docs + verification only.

## Error handling & guardrails (binding)

These are enforced by the metronome and inherited from the operating contract /
elves-aragora gate:

- **One active sub-mission.** The gate blocks N+1 until N's proof is published.
- **No artifact → no advancement.** The gate is existence + verifiability of the
  external proof, checked in code, not prose.
- **Tier 0-2 auto-settle; Tier 3-4 hard-stop** for human risk acceptance via the
  `aragora/human-settlement` commit status. The metronome never merges or settles
  Tier 3-4.
- **Approval-required surfaces are hard stops.** `.github/workflows/`, `action.yml`,
  secrets, `CLAUDE.md`, `aragora/__init__.py`, `.env`, `scripts/nomic_loop.py`.
  M2's workflow change and M3's README narrative both park for the founder.
- **No busy-poll, no idle.** On a wait, log `waiting-on:<thing>` and exit the tick;
  re-evaluate next cadence. Never sit in a spin loop.
- **Circuit breaker.** No external progress for 3 consecutive cycles → halt and
  emit one operator escalation. Never re-drive a dead lane.
- **Auto-halt / main-red.** Any required check red on `origin/main` >30 min → halt
  roadmap work, fix first (operating contract).
- **Disk truth.** The metronome reads the register/survival guide every tick; it
  never trusts in-memory or prior-transcript state.
- **Default-OFF.** `enable_native_mission` stays OFF; the launchd unit is created
  disabled. Nothing runs unattended until the founder enables it.
- **Single fleet.** Assumes it is the only active orchestrator (no concurrent
  Factory mission + boss-loop + hand-driven Codex on one account).

## Testing

- **Mission gate:** unit tests for `PASS / BLOCKED / NEEDS_HUMAN` given present /
  absent / unverifiable proof artifacts (mirror `tests/swarm/test_mission*.py`).
- **Metronome tick:** unit tests for the tick state machine — advance on PASS,
  run-one-batch on BLOCKED, park on NEEDS_HUMAN, halt on circuit-breaker — with the
  batch runner and gate mocked (mirror `tests/swarm/test_boss_loop_*.py`).
- **Register round-trip:** parse the Mission Queue section, mutate a row status,
  re-serialize; assert single-active invariant.
- **M1 proof:** an end-to-end test that exports a receipt via `odr_export` and
  verifies it with `aragora-verify` against the published ODR profile. The
  example is **unsigned** — the absent-signature check is a warning, not a
  failure. Signing (and any format-version bump to v1.0) is parked Tier 3.
- **No new `mypy` errors vs baseline; test count never decreases** (elves gate
  step 3).

## Out of scope (YAGNI)

- Rewriting `boss_loop` or merging the metronome into it (run alongside, slim).
- A general workflow DSL for missions (the tranche/MissionSpec schema suffices).
- API model transport as a prerequisite (subscriptions-first; API is a later scale
  lever).
- Reducing the 145 top-level modules to ~50 (M3 quarantines behind a boundary;
  full re-architecture is deferred).

## Open founder decision (surfaced, not decided here)

**README narrative (M3, Tier 3).** The README currently asserts two contradictory
stories — "auditable control plane for AI decisions" vs "5-pillar decision
platform." M3 cannot finish legibility until the founder picks the single public
narrative (or explicitly splits current vs roadmap). The metronome will park M3 on
this decision.
