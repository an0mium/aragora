# The Steering Leverage Program

> **For agentic workers:** REQUIRED SUB-SKILL: execute via `elves-aragora` (receipt-gated
> batches, tier settlement, parallel lanes). This is a multi-sprint program: run one phase per
> session/sprint, re-reading this file and `docs/FOCUS.md` at each phase boundary. Adoption of
> any phase into FOCUS.md sprint goals is an **operator decision**; this document is the
> program backlog, not the sprint contract.

**Goal:** Make Aragora the *steering layer* for a world of super-capable AIs and overloaded
humans: maximize the **leverage ratio** (verified outcomes per human-minute) while
*strengthening* — not eroding — the human's ability to meaningfully steer work they no longer
have bandwidth to understand in full.

**Thesis (why Aragora, why now):** The binding constraint on AI-assisted work is no longer
model capability or token cost — it is **human attention and human trust**. Every lab ships
more capable agents; almost nobody ships the instrument panel that lets one person govern a
hundred of them *without reading everything*. Aragora already owns the rare primitives this
needs: adversarial debate, cryptographic DecisionReceipts, calibration/ELO track records, a
crux detector, an evidence-gated merge quorum, and a self-improvement loop that dogfoods all
of it. The adjacent possible is to compose these into steering infrastructure. run-20260610
is the proof sketch: ~14 hours of autonomous work, 8 merged PRs, and the *entire* required
human contribution compressed to one login and one design review.

**North-star metric — the Leverage Ratio (LR):**
`LR = verified merged outcomes / human-minutes consumed` — measured per week, per surface,
from data the platform already produces (receipts, settlement records, attention events).
Companion guardrail metric — **Steering Integrity (SI)**: fraction of high-tier decisions
where the human (a) was shown a crux, not a transcript, (b) decided within their stated
attention budget, and (c) later audits did not reverse the decision. LR without SI is
rubber-stamping; SI without LR is babysitting. The program optimizes the pair.

**What this is NOT (anti-goals, standing):**
- Not "lights-out autonomy as the default story" — the README promise holds; autonomy is
  *granted by measured calibration*, never assumed.
- Not new settlement meta-tooling beyond what each phase names (FOCUS anti-goal discipline).
- No Tier-4 self-modification without spec + failing governance tests first.
- No new top-level modules until a phase proves demand: every pillar below composes existing
  subsystems (`debate/`, `gauntlet/`, `explainability/`, `ranking/`, `inbox/`, `knowledge/`,
  `rlm/`, `notifications/`).

---

## The five pillars (each underdeveloped in the wider world)

### P1 — The Attention Exchange ("what actually needs you")

**What it is:** A first-class human-attention router. Every autonomous lane, debate, or
settlement queue emits typed **attention requests** (decide / confirm / be-aware / FYI) with
machine-estimated stakes: irreversibility, value-at-stake, model uncertainty, deadline. The
exchange ranks them against the human's **declared attention budget** (e.g. "20 min/day for
this repo") and renders each as a one-screen card. Everything below the budget line proceeds
under standing policy and remains auditable; nothing silently waits.

**Why nobody else ships it:** inboxes rank by recency and sender; agent frameworks surface
raw logs or block on approval modals. Ranking by *decision-weight under an explicit budget*,
with receipts proving what flowed under the line, requires the governance substrate Aragora
already has.

**Adjacent-possible composition:** `inbox/` (triage wedge) + `notifications/` +
`control_plane/scheduler` + receipts. The boss loop's operator queue is the dogfood: tonight's
queue ("codex login, #7472 review") IS an attention exchange rendered by hand.

**MVP (one sprint):** `aragora attention` CLI: ingest open settlement queue + parked PRs +
filed escalations → score → render top-N cards with stakes and a single action each. Emit an
`attention_event` (shown, decided, deferred, auto-proceeded) to a JSONL ledger. LR/SI computed
from that ledger by `scripts/measure_leverage_ratio.py`.

**Falsification:** if 4 weeks of dogfood shows the operator still opens GitHub to find work
(cards unused), the ranking model is wrong or the surface is — record honestly and stop.

### P2 — Crux Cards (steer by crux, not by transcript)

**What it is:** Every escalation to a human must be compressed to the **crux**: the single
disagreement, uncertainty, or assumption that flips the decision — with the strongest case on
each side (from real debate dissent), a falsification test ("if X, then side A"), and the
default that fires if unanswered. The human steers by answering cruxes — minutes, not hours —
and their answer becomes precedent (P3) for future similar cruxes.

**Why Aragora:** `CruxDetector` exists (crux-mode epic #6035, design doc in repo);
`explainability/` does factor decomposition; debates already record dissent verbatim. The
missing piece is making crux-rendering the *mandatory escalation format* at Tier 3-4 — a
format change with outsized steering value.

**MVP:** crux card renderer over existing debate results (`aragora explain --crux <receipt>`);
wire into Tier-3/4 settlement packets so `settle-status` and the attention card show the crux.
Measure: median human decision time on Tier-3/4 settlements before/after.

**Falsification:** if crux answers don't reduce decision time or get overridden by humans
reading full transcripts anyway (>30% of the time), the compression is lossy where it matters.

### P3 — Standing Intents (durable steering contracts)

**What it is:** The human writes durable intent in natural language ("prefer boring tech;
never trade auditability for speed; cap spend at $X/week; in doubt about user data, stop").
A debate compiles it into machine-checkable policy clauses + test cases (the prompt-engine /
`policy/` pattern). Every receipt gains an **intent-compliance section**; a weekly
**drift debate** adversarially asks "does the accumulated body of work still serve the
stated intent?" and escalates a crux card if not. Intents are versioned; re-confirmation is
itself an attention card.

**Why it matters for the super-capable-AI future:** this is corrigible delegation in
practice — the human steers the *objective function*, the machine proves per-decision
compliance, and drift is detected adversarially rather than discovered in a postmortem. The
alignment community writes papers about this; almost no one ships the working artifact.

**MVP:** `aragora intent set/compile/status`; compliance section in receipts for one surface
(the boss loop's own PRs); first weekly drift debate published as a receipt.

**Falsification:** if compiled clauses are so generic they never block or flag anything in
4 weeks (0 violations, 0 flags on real work), the compilation isn't capturing intent.

### P4 — The Calibrated Delegation Ledger (trust that is earned, measured, and revocable)

**What it is:** Autonomy width per (agent × surface × tier) is set by **measured calibration**:
Brier-scored predictions, post-merge reversal rates, gauntlet survival rates — the data
`ranking/elo.py` + CalibrationTracker + `outcome/` already collect. A public dashboard shows
the trust surface; thresholds auto-widen autonomy where calibration is proven and auto-narrow
it after misses (the session circuit-breaker pattern, generalized). Continuous adversarial
audit: a budgeted always-on gauntlet lane re-attacks a random sample of recent autonomous
decisions and files refutation receipts — drift gets caught *without* anyone asking.

**Why nobody else:** every agent product has fixed permission tiers set by vibes. Trust as a
measured, revocable, per-domain quantity — backed by adversarial spot-audits — does not exist
as a product anywhere.

**MVP:** `aragora trust` report (read-only) from existing ELO/calibration/outcome stores +
one auto-narrowing rule wired into the boss loop (e.g. surface's settlement tier bumps +1 for
a week after a reverted merge). Spot-audit lane: 3 random recent merges re-gauntleted weekly.

**Falsification:** if 6 weeks of ledger data shows calibration scores don't predict reversal
rates (no correlation), the trust math is decorative — stop and redesign before widening
any autonomy on it.

### P5 — The Open Receipt Standard (steering interop beyond Aragora)

**What it is:** Publish the DecisionReceipt schema + verifier as a small open standard:
spec document, JSON Schema, `aragora verify` as reference implementation, a GitHub Action
("receipt-check") any repo can adopt, and emitter shims for other frameworks (LangChain
callback, OpenAI Agents tracing hook) so *non-Aragora* agents can emit verifiable receipts.
Aragora becomes the audit/steering substrate for heterogeneous fleets, not a silo.

**Why now:** EU AI Act Article 12/14 obligations land Aug 2, 2026 — every serious agent
deployment needs exactly this artifact and currently hand-rolls logs. A clean, verifiable,
vendor-neutral receipt format with a working verifier is a wedge no one is driving.

**MVP:** extract schema → `docs/standards/DECISION_RECEIPT_v1.md` + JSON Schema; publish the
GitHub Action; one external-framework emitter (LangChain callback) with a round-trip test
(emit → `aragora verify` → conformity bundle via the existing EU-AI-Act generator).

**Falsification:** if no external project (including our own non-Aragora repos) adopts it in
8 weeks, the wedge framing is wrong; keep the schema internal and revisit.

---

## Phasing (each phase = one elves-aragora run/sprint; ~2 weeks cadence)

### Phase 0 — Instrument the leverage ratio (do FIRST; everything else is steering by feel)
1. `scripts/measure_leverage_ratio.py`: compute LR + SI weekly from receipts dir, settlement
   records, and a new attention-events JSONL; publish to `docs/status/LEVERAGE.md` (the same
   recurring-publication pattern as B0 truth). Baseline week = run-20260610 (operator minutes:
   ~2 known items vs 9 merged PRs — compute honestly, including reading time).
2. Attention-event capture: minimal hooks in `settle_tier4_pr.py --settle-only`, the boss
   operator queue, and `aragora receipt view` (each logs shown/decided events locally).
3. Tier: 1-2. Fully autonomous-settleable. **Exit:** first LEVERAGE.md published with real numbers.

### Phase 1 — Attention Exchange MVP (P1) + crux renderer (P2 read-only half)
- `aragora attention` over live queues; crux card renderer from existing receipts/dissent.
- Dogfood on this repo's own operator queue for 2 weeks before any external claim.
- Tier: 2 (new read-only CLI surfaces). **Exit:** operator runs `aragora attention` instead of
  GitHub triage for one full week; decision-time metric captured.

### Phase 2 — Crux-mandatory escalation (P2 write half) + intent MVP (P3)
- Tier-3/4 packets carry crux cards; `aragora intent` + receipt compliance section on the
  boss-loop surface; first drift debate.
- Tier: 2-3 (touches settlement packet *rendering*, not gating logic — gating changes remain
  Tier 4 and out of scope). **Exit:** one real Tier-3/4 settlement decided from a crux card.

### Phase 3 — Delegation ledger (P4)
- `aragora trust` read-only report; one auto-narrowing rule; weekly spot-audit gauntlet lane.
- Tier: 2 read-only + Tier 3 for the narrowing rule (park for operator settlement).
- **Exit:** first trust report with ≥4 weeks of data; correlation check scheduled.

### Phase 4 — Open standard (P5)
- Spec + schema + Action + LangChain emitter + round-trip conformity test.
- Tier: 1-2 (docs + additive). **Exit:** receipt-check Action runs green on one external repo.

**Standing rules for every phase:** elves-aragora gate on every batch (local truth →
heterogeneous debate → verified receipt → tier settlement); 2-family evidence (grok + mistral
locally; codex when authenticated); substrate cap stays on; each phase's exit metric gets
published whether it flatters or not (the B0 0.0% discipline). Phase N+1 does not start until
Phase N's exit metric is published.

## Relationship to existing strategy

This program is the natural successor to the proof-first shift: B0 proved *bounded execution
with receipts works* (corpus closed 2026-06-10); this proves *one human can govern it at
leverage without losing the wheel*. FOCUS.md remains the sprint contract — the operator
chooses which phase enters Sprint 4+. Crux-mode epic #6035 and the trust-compound track are
absorbed (not duplicated) by P2 and P4 respectively. The agent-civilization planning layer
(AGT-xx) stays gated; nothing here opens it.

## Why these and not other features

Selection filter applied: (a) **adjacent possible** — composes shipped Aragora subsystems
within 1-2 sprints each; (b) **underdeveloped elsewhere** — labs ship capability, startups
ship copilots; attention routing under budget, crux-format escalation, compiled standing
intents, measured-revocable trust, and a vendor-neutral receipt standard have no serious
owner; (c) **worth existing** — each one moves the same number: more verified work per human
minute *with* the human's hand still meaningfully on the wheel. Features that failed the
filter (deliberately excluded): more chat connectors, more verticals, frontend breadth,
blockchain expansion, generic workflow tooling — all breadth, no steering.
