# Steering Leverage Program — Operating Plan v2

> **For agentic workers:** REQUIRED SUB-SKILL: `elves-aragora` (v0.2.0, repo-tracked). One
> phase per run/sprint; re-read this file, the program doc
> (`2026-06-10-steering-leverage-program.md`), and `docs/FOCUS.md` at every phase boundary.
> Operator adoption into FOCUS.md sprints remains the operator's decision.

**Goal:** Convert the approved Steering Leverage Program from authored thesis into operating
infrastructure — sequenced by the empirical findings of the 2026-06-10 two-week steering audit
(`.aragora/run-20260610/OPERATOR_STEERING_AUDIT.md`) — and extend it with two new pillars the
audit proved necessary. Same north star: **leverage ratio** (verified outcomes per
human-minute) × **steering integrity** (the human steers cruxes, within budget, without
later reversal).

**Why v2 exists:** the audit found that every loss in the window — publisher dead 3 weeks,
boss metrics silent 10 days, empty daemon plist, 217-item outbox, ~1.9k local-only branches of
lost writer work, 215:0 issue creation:closure, a 12-day operator-decision slip — shared one
shape: *a cheap signal existed on disk, nothing was contracted to read it, and the human was
the only fallback reader.* v1 named the right pillars; v2 sequences them so the
signal-reading machinery ships first, and adds what v1 missed: the fleet's own
**mortality signals** and its **work-loss accounting**.

---

## New Pillar 6 — Dead Man's Signals (the loop watches its own blind spots)

**What:** a tiny, always-on sentinel that reads the cheap state the incidents lived in:
`automation-publisher-status.json` freshness + auth_ok, `boss_metrics.jsonl` mtime,
launchd plist validity (`plutil -lint`) and daemon liveness, gh + codex auth state,
main-checkout-on-main invariant, outbox depth and age, disk free. Each check has a
**contracted reader**: breach → attention card (P1 exchange) + one loud push channel. Silence
is recorded as an incident ("loop was blind 05-27→06-05"), never mistaken for health.

**Why nobody ships this:** monitoring products watch services; nobody watches *the agent
fleet's own delegation machinery* — outboxes, publisher auth, metrics heartbeats, work queues
— as first-class mortality signals. Every datum here already existed during the May outage;
what was missing is the contract that something reads it.

**MVP (one sprint):** `scripts/fleet_sentinel.py` (stdlib, cron/launchd every 10 min, JSONL
ledger + exit-coded report) + `aragora fleet status` rendering; breach pushes via the existing
notifications subsystem. Acceptance: replaying May-18→Jun-08 state files raises the publisher
alarm on day one. **Falsification:** if the next real incident is again discovered by a human
before the sentinel, the check set is wrong — extend or kill.

## New Pillar 7 — Work-Loss Accounting (the waste ratio)

**What:** the leverage ratio's dual. Measure **agent-work attrition**: branches pushed but
never PR'd, outbox items expired unpublished, lanes that produced no deliverable, PRs closed
unmerged with unique content, local-only branches. Publish a weekly **waste ratio**
(lost-work-units / produced-work-units) beside LR in `docs/status/LEVERAGE.md`, and run
bounded salvage drains (the outbox harvest of 2026-06-10 — 37 PRs recovered from 254 items —
is the manual prototype; ~1.9k local-only branches remain unaudited).

**Why it matters:** in a future of abundant agent labor, unmeasured attrition silently eats
the leverage ratio, and nobody treats agent WIP loss as an accountable quantity. The audit
found three weeks of writer output simply evaporating — invisible because nothing counted it.

**MVP:** `scripts/measure_work_loss.py` (outbox archive + ls-remote + PR refs cross-walk —
the harvest triage logic, productized) + weekly publication + a salvage-drain runbook.
**Falsification:** if two consecutive salvage drains recover nothing worth merging, attrition
is already-priced garbage; downgrade to a counter, stop draining.

---

## Phasing (each = one elves-aragora run; exit metric published before the next starts)

### Phase 0 — Instruments (Sentinel + LR/SI + Waste) — DO FIRST
1. `scripts/fleet_sentinel.py` + launchd/systemd unit + first breach-replay test (P6 MVP).
2. `scripts/measure_leverage_ratio.py` (v1 plan Phase 0, unchanged contract: refuse to invent
   operator-minutes; SI honestly null until instrumented) + `scripts/measure_work_loss.py`
   (P7 MVP) → `docs/status/LEVERAGE.md` with **LR, waste ratio, blind-period log** together.
3. **Net-closure floor** in `generate_boss_issues.py`: weekly closed:created below floor →
   generator throttles (the substrate-cap pattern, applied to appetite). Audit basis: 215:0.
4. Main-checkout invariant assert (SessionStart hook + sentinel check).
   Tier 1-2 throughout; autonomous-settleable. **Exit:** first LEVERAGE.md with real LR,
   waste ratio, and a sentinel that has run ≥48h without a human-discovered incident it missed.

### Phase 1 — Attention Exchange with operator SLAs (P1, hardened by audit rec 2)
- `aragora attention` over: settlement queue, parked PRs, operator-action items, sentinel
  breaches. Every operator-action card carries a **48h SLA and a pre-declared
  default-on-timeout** (decide / delegate-with-authority / defer-with-date). The #7472
  12-day slip is the regression test: replay the window; the card must escalate by hour 48.
- **Exit:** operator runs one full week from `aragora attention` alone; zero SLA breaches
  silently expire.

### Phase 2 — Crux Cards mandatory at Tier 3-4 (P2) + Standing Intents MVP (P3)
- As v1 Phase 2, plus: drift debate consumes sentinel + waste data ("is the loop serving the
  intent, and is it healthy enough for its receipts to mean anything?").
- **Exit:** one real Tier-3/4 settlement decided from a crux card within its SLA.

### Phase 3 — Calibrated Delegation Ledger (P4)
- As v1, plus auto-narrowing rules take sentinel state as input (blind loop ⇒ autonomy
  narrows automatically — receipts from an unobserved fleet earn less trust).
- **Exit:** trust report with ≥4 weeks of data; calibration↔reversal correlation computed.

### Phase 4 — Open Receipt Standard (P5)
- As v1. **Exit:** receipt-check GitHub Action green on one external repo.

**Standing rules:** elves-aragora gate per batch; two-family lineage evidence
(grok + mistral structured disclosures; codex when authenticated); substrate cap stays on;
exit metrics publish whether flattering or not; settlement surfaces (`review_queue.py`,
quorum workflows, settle scripts) remain Tier 4 — spec + failing governance tests first.

## Why these two pillars and not others

Same selection filter as v1 — adjacent-possible (both MVPs are productizations of scripts
this run already executed by hand), underdeveloped elsewhere (fleet-mortality sentinels and
agent-WIP attrition accounting have no owner anywhere), and aimed at the same number: more
verified work per human-minute *with the human's hand on the wheel*. Considered and excluded:
multi-repo federation (premature before one repo's instruments run), human-attention ML
ranking (rules + SLAs first, learn later), any new debate/consensus modes (the existing ones
are under-leveraged, not insufficient).

## Relationship to v1 and FOCUS

v1 (merged, #8105) is the thesis and pillar definitions — unchanged. v2 is its operating
sequence plus P6/P7, justified line-by-line by the audit. FOCUS.md remains the sprint
contract; the operator chooses what enters Sprint 4. Suggested Sprint 4 candidates, in order:
Phase 0 items 1-4 (they are small, Tier 1-2, and every one of them would have caught a real
incident from the last two weeks).

---

## Pillar → implementation crosswalk, P1–P7 (added 2026-06-11, issue #8232)

What each pillar's implementing artifact actually is on `main` as of 2026-06-11 — or an honest
"not started" — so no pillar is re-specced from scratch by a future agent. Statuses verified
against the codebase (paths cited) and merged PRs, not asserted from memory.

| Pillar | Implementing artifact(s) | Status (verified 2026-06-11) |
|--------|--------------------------|------------------------------|
| P1 Attention Exchange | — | **Not started.** No `aragora attention` CLI surface exists in `aragora/cli/parser.py`; the operator queue is still rendered by hand. |
| P2 Crux Cards | Engine: `aragora/reasoning/crux_detector.py`, `aragora/debate/crux_mode.py`; operator CLI (DIC tranche): `aragora crux`/`cruxset`/`crux-arbitrate`/`crux-garden`; external exposure: ODR-4 ([#8227](https://github.com/synaptent/aragora/issues/8227)) | **Engine built; card format not started.** #8227 covers API/CLI-flag/SDK exposure and crux sets in receipts; the crux-card mandatory-escalation format (Phase 2 here) remains unimplemented. |
| P3 Standing Intents | TET intent chain (in build): spec at `docs/specs/TAMPER_EVIDENT_TRAIL.md` (Tier-2 build spec, operator-requested 2026-06-11) | **In build via TET.** TET's anchored intent records + witness/intent reconciliation implement the durable-steering-contract core; `aragora intent set/compile/status`, receipt intent-compliance sections, and the weekly drift debate are not started. |
| P4 Calibrated Delegation Ledger | Calibration exposure: ODR-5 ([#8229](https://github.com/synaptent/aragora/issues/8229)); decision-stakes routing with receipt-recorded rationale: [#8233](https://github.com/synaptent/aragora/issues/8233); adjacent: jury composition optimizer ([#8234](https://github.com/synaptent/aragora/issues/8234)) | **Data collected, ledger not started.** ELO/calibration/outcome stores exist; no `aragora trust` report, no auto-narrowing rule, no spot-audit gauntlet lane. #8229/#8233 build the measurement-and-exposure half this pillar needs first. |
| P5 Open Receipt Standard | ODR epic [#8223](https://github.com/synaptent/aragora/issues/8223): spine #8224 (content profile, JSON Schema + JCS) → #8225 (Ed25519 signing) → #8226 (standalone offline verifier); enrichment #8230 (human-oversight attestation + Art. 14 evidence pack), #8231 (Sigstore Rekor anchoring) | **In flight as the ODR tranche — the program's active direction.** The ODR epic supersedes the v1 MVP sketch: the standard is decision-semantics (rationale, quorum, calibrated confidence, crux, attestation), not just schema extraction. |
| P6 Dead Man's Signals | `scripts/fleet_sentinel.py` + `aragora fleet status`; shipped [#8147](https://github.com/synaptent/aragora/pull/8147), extended with lane-liveness + api-degradation checks, lane janitor, publisher retry in [#8176](https://github.com/synaptent/aragora/pull/8176) | **Shipped.** Sentinel runs with contracted readers; falsification clause (next incident discovered by a human first ⇒ extend or kill) remains standing. |
| P7 Work-Loss Accounting | `scripts/measure_work_loss.py`; waste ratio published beside LR in `docs/status/LEVERAGE.md` (first publication 2026-06-10: 623 lost units / 32 produced, ratio 19.47) | **Shipped (instrument).** Salvage-drain runbook and the ~1.9k local-only-branch audit remain open follow-ups. |

Phase 0 instruments status (all four shipped): `scripts/measure_leverage_ratio.py` publishing
real LR (1.16 on the 2026-06-10 baseline) with SI honestly `null`; the waste instrument per P7
above; net-closure floor live (`apply_net_closure_floor` in `scripts/generate_boss_issues.py`);
main-checkout invariant live as a sentinel check in `scripts/fleet_sentinel.py`. Substrate cap
also live (#8095, composition measured in `docs/status/LEVERAGE.md`).
