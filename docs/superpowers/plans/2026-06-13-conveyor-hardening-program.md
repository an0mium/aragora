# Conveyor Hardening Program — direction, taxonomy, and runbook

Status: program document (epic #8344). Written 2026-06-13 from the run-20260612
coordination session; intended to outlive any one session, harness, or model.

## 1. The thesis: the pipeline is an inventory system

The autonomous loop (writers → outbox → publisher → draft PR → ready → evidence
→ quorum → arbiter merge) should be reasoned about as a manufacturing funnel,
not a collection of agents. Measured 2026-06-12:

- Merged:rejected over one week: **106:4** — output quality is solved; the
  adversarial gate is calibrated correctly (4% rejection, with real catches:
  e.g. the grok review on #8291 found a genuine math bug — raw 3-outcome
  Shannon entropy wrongly clamped to [0,1]).
- WIP: ~130 draft PRs + ~117 outbox items vs ~15–20 merges/day. Admission
  outran settlement because **stage transports were missing**, not because
  anything was broken.

Consequences that follow from the inventory framing:

- Build transports between stages before building more producers
  (#8312: ready-promoter, stale janitor, backpressure gate, funnel telemetry).
- Watch **time-in-state per stage**, not just stage depths — the 130-draft
  pileup was invisible for days because only depths were monitored.
- Apply backpressure at admission (`.aragora/backpressure.json`; writer-side
  consumer already exists, #8323). Never fix throughput by loosening review.

## 2. The six failure classes

Every incident of the run reduces to one of these. Closing all six is the
program's near-term definition of done (live status table lives in #8344).

**A. Silent lane death.** Lanes die at setup (empty branches — solved by lane
ledger + `lane_liveness` sentinel + `lane_janitor`, #8176) and at dispatch
(prompt pasted into a composer but never submitted; launcher exits 141/SIGPIPE
before reporting — #8317/#8338). Rule: every async hand-off needs a
*delivery receipt*, and the watcher must breach on its absence.

**B. Transport/structural conflation.** External degradation must never be
read as a domain answer. Live reproduction: `auto_evidence_cycle` returned
`{"plan": "empty"}`, exit 0, while two PRs sat evidence-starved — the per-PR
packet probe returned `transport_blocked` and the selector treated that as
"not selectable". An operator must always be able to distinguish "queue is
clear" from "I couldn't see the queue" (#8316, #8324, #8339).

**C. Ghost owner locks.** Owner claims without liveness become permanent
vetoes (#7825 sat fully exact-gated for hours behind a dead lane). Locks need
lease age / heartbeat / terminal-status reporting and a fail-closed override
advisory: any possibility of unpushed work withholds the advisory entirely
(#8318/#8340, codifying the #8125 manual protocol).

**D. Semantic surface collisions.** Three colliding repairs landed on the
`handle_post` surface within 24h because a guard test contradicting its own
merged tree (#8163) was only executed by non-required CI lanes. Lessons:
claim discipline must operate at the *surface* level, not just the PR level;
guard tests must run in a required shard for PRs touching their surface
(#8342); contested surfaces get one declared owner and everyone else steers
via comments.

**E. Quota starvation as architecture.** One GitHub identity served N
concurrent pollers, each re-verifying full state every cycle (the "do not
trust transcript state" doctrine — correct! — multiplied by fleet size).
GraphQL (5,000 pts/hr) exhausted mid-hour while REST sat healthy. Fix layers:
spread identities (App installation token has its own budget); move hot reads
to REST + ETag conditional requests (304s are free — #8339, #8324); share one
read path (cache + single exact-head verify before mutation); long-term,
replace polling with webhooks (§5).

**F. Review-pipeline defects.** The advisory reviewer's renderer truncates
both its input diff (degrading review substance to "undecidable") and its
output (empty CRITICAL stubs, counters contradicting prose) — #8326. Review
prompts also carried persona blocks and untrusted trending content — an
injection surface inside a merge-gating prompt — #8327. A decision-receipts
product must hold its own review output to receipt standards.

## 3. Identity architecture (execution vs oversight)

- **Execution identities**: `an0mium` (legacy) and the GitHub App
  (`aragora-automation-fable[bot]`) — agents may hold these. All lane traffic
  should move to the App token (own quota, honest attribution).
- **Oversight identity**: `scarmani` — its entire value is that **agents can
  never produce its signatures**. Its credentials must not exist in any
  agent-reachable keyring/env; browser approvals + one-shot fine-grained PAT
  from a password manager; ideally a separate device. H1 (CODEOWNERS pin) +
  H2 (settlement-creator pin) land via #8274. Before enabling
  require-code-owner-review, audit existing broad `@an0mium` rules: an author
  cannot approve their own PR, so broad rules would freeze the pipeline.
- **Pins must follow surfaces**: CODEOWNERS entries should be globs
  (`review_queue*`) because modularization (e.g. #8324's
  `review_queue_rest_fallback.py`) otherwise moves merge-authority code out
  from under the pin.
- **Tamper evidence** (full design: `docs/specs/TAMPER_EVIDENT_TRAIL.md`):
  the witness lives outside the blast radius (server-side audit log, external
  collector), is append-only, and the detector is *intent reconciliation* —
  diff what happened against pre-registered, externally-anchored intent.

## 4. Harness comparison — and what aragora should absorb

Observed across Claude Code (local + cloud), Codex goal-loops, and Factory
missions running side-by-side on this repo:

| Harness | Superpower | Weakness |
|---|---|---|
| Aragora loop | governed merge transport: tiers, quorum, receipts, 24/7 | planning/decomposition ergonomics; lane mortality |
| Factory missions | readiness probes, validation contracts, milestone gates, pointed clarifying questions | had to re-research the repo's merge gate from scratch |
| Dynamic sessions | fast, grounded, surgical; live triage | ephemeral; watches die with the session |

Absorb into aragora as repo artifacts: (1) **validation contracts** —
machine-checkable acceptance assertions per goal, evaluated before a lane's
work counts as done; (2) **readiness probes** — a 5-minute dependency check
(auth, keys, CLI surfaces, worktree) before any lane spends budget; (3) a
**mission schema** so `docs/superpowers/plans/*` files are executable by any
harness — the orchestrator becomes swappable. The 2026-06-12 Fable 5
suspension made this concrete: the heterogeneous merge quorum sailed through
unaffected, while anything pinned to a single model (a mission orchestrator)
was exposed. **No load-bearing component should assume a specific model.**

## 5. Medium-term direction (ordered)

1. **Event substrate**: org webhook → local collector → lanes consume pushed
   events. Retires most polling (quota class E at the root), and the same
   collector is the external witness for intent reconciliation (§3).
2. **Mission schema + validation contracts + readiness probes** (§4).
3. **`review_queue.py` decomposition** (mission P5, checkpoint 5): it is the
   Tier-4 bottleneck — #8315/#8316/#8343 all queue behind its 5,400-LOC lint
   ratchet; #8324's extraction is the template.
4. **Funnel autopilot**: promoter + janitor + backpressure + telemetry wired
   into the arbiter pass and sentinel, with funnel time-in-state breaches as
   first-class incidents.
5. **Dogfood loop**: the conveyor's receipts, reviews, and trail are the
   product. Defects found here (#8326/#8327) are product bugs, and fixtures
   from this program should become product test cases.

## 6. Operating lessons (the short list that earns permanence)

1. Fail-closed without truncation/transport detection silently fails open.
2. Every async hand-off needs a delivery receipt; watchers breach on absence.
3. Locks need liveness; overrides fail closed on possible unpushed work.
4. Tier escalations park by default — disclose-and-proceed races the operator.
5. Claim discipline at surface granularity; guard tests in required shards.
6. "A human must decide" ≠ "a human must type": explicit operator
   authorization in any channel is the decision; unforgeability comes from
   credential isolation, not ceremony.
7. Build the transport before the next producer; measure time-in-state.
8. Never loosen the review gate to fix throughput.
9. Distinguish "no work exists" from "I couldn't see the work" in every tool.
10. Multi-model heterogeneity is resilience infrastructure, not a feature.
