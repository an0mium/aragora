# Strategic Plan: Close the Loop (2026-07-01)

**Epic:** [#8762](https://github.com/synaptent/aragora/issues/8762)
**Status:** Active — 30-day horizon (review 2026-07-29)
**Provenance:** Synthesized from three parallel evidence sweeps run 2026-07-01: (1) live repo/pipeline
health via `gh`+`git`, (2) the strategy-doc corpus (`docs/plans/`, STATUS, METRICS, FEATURE_GAP_LIST,
COMMERCIAL_OVERVIEW), (3) a code-level audit of the autonomous-execution substrate.

---

## 1. Diagnosis: the pipeline is open-loop at the back half

Aragora's problem is not throughput and not direction. It is that the improvement loop does not
close: work is produced far faster than it settles, merges, and feeds back into the backlog.

**Evidence (2026-07-01):**

| Signal | Value | Reading |
|---|---|---|
| PRs merged, last 30 days | 593 | Production capacity is high |
| Ready PRs open | 19 — oldest 18d, **all** sampled failing checks, all `reviewDecision=NONE` | Large/valuable work stalls at settlement |
| Issues: boss-stuck / boss-quarantined / boss-ready | 266 / 39 / 16 | The dispatchable frontier is starved while stuck work accumulates 16:1 |
| Remote branches / local branches / worktrees | 983 / 612 / 44 | No harvest: finished and abandoned work is never folded back |
| Open issues total | 1,306 | Backlog grows monotonically |
| CI on main (last 40 push runs) | 35 green / 5 red — failures concentrated in Deploy Frontend, Deploy (Secure), merge-quorum retrigger | Core CI healthy; deploy + retrigger workflows are the flake sources |

**Stage-by-stage verdict on the autonomous pipeline** (code-level audit):

| Stage | Verdict | Key fact |
|---|---|---|
| Seed | WORKS | `aragora mission seed/run/resume`, crash-survivable tick loop (`aragora/missions/orchestrator.py`) |
| Decompose | PARTIAL | `TaskDecomposer` works but the mission engine never calls it — seeded intake features park forever (`aragora/missions/dispatch.py:89-98`) |
| Dispatch | PARTIAL | Lease machinery + worker launcher real; no bridge from mission Features → swarm work orders |
| Implement | WORKS | nomic executors + safety gates real, not decorative |
| Review | WORKS (flaky) | Quorum evidence collection tier-gated and honest; reviewer reliability is the weak point |
| Adjudicate | PARTIAL | `review_adjudicator.py` merged 2026-07-01 (#8749) but flag-off and not wired to the stall path (#8748) |
| Settle | PARTIAL | advisory_settle now **reachable** in the enforcing quorum job (#8741, merged 2026-07-01); M0a records in flight (#8756) |
| Merge | PARTIAL | CI authorizes but never executes; every merge needs a human `--apply` |
| Harvest/learn | MISSING | outcome_feedback/outcome_learner exist as libraries; nothing drives them ("no harvest engine" — 2026-06-30 queue-drain diagnosis) |

**Binding constraint:** settlement-and-feedback throughput. Everything upstream of review works;
everything downstream leaks. The June/July fixes (#8638 tiered gate, #8574 severity-gated dissent,
#8741 advisory-settle reachability, #8749 adjudicator core) attacked exactly the right joint — the
remaining work is wiring, daemonizing, and *exercising* them until the queue actually drains.

## 2. Strategic thrusts (ranked)

Each thrust states its claim, 30-day measurable outcome, and kill-switch metric — the observation
that would prove it wrong and stop the spend.

### T1 — Close the settlement loop (highest leverage)
**Claim:** with adjudication wired and advisory-settle exercised, stuck ready PRs settle without
per-PR human negotiation.
**Work:** wire ReviewAdjudicator into the quorum stall path (#8748); land M0a settlement records
(#8756); build the unattended Tier 0-2 merge executor (#8759); run the drain campaign over the 19
stuck ready PRs (#8761).
**30-day outcome:** ready-PR queue ≤5, none older than 7 days; ≥3 unattended Tier 0-2 merges with
receipts and zero main breakage.
**Kill-switch:** queue fails to shrink for 2 consecutive weeks with the machinery live → the
constraint is not settlement policy; stop and re-diagnose (reviewer reliability / CI flake first).

### T2 — One canonical autonomous harness (mission engine)
**Claim:** the mission engine (`aragora/missions/` + `aragora mission` CLI) is the right spine —
crash-survivable ticks, file-locked ledger, tier-aware BossLoopDispatch — and is one component away
from a demonstrable happy path.
**Work:** build the intake→TaskDecomposer bridge (#8758); then demonstrate ONE full unattended run:
seed → decompose → dispatch → implement → quorum → settle → merge (Tier 0-2 change), receipts attached
to the epic. Document nomic_loop / self_develop / raw boss loop as feeders of the mission spine, not
competing spines (composition over rewrite — per the canonicalization decision of 2026-06-26).
**30-day outcome:** one evidenced closed-circuit run with ≤2 human interventions.
**Kill-switch:** demo needs >2 interventions → park harness work; fix the intervention causes first.

### T3 — Harvest/learning loop
**Claim:** the backlog never drained because outcomes never fed back (root cause per the
2026-06-30 queue-drain diagnosis: history-rewrite orphaned 645 branches, no harvest engine,
dissent-veto churn).
**Work:** harvest engine (#8760) — recurring classification of merged/parked/orphaned work into
learned-pattern / salvage-candidate (WIP-capped issue creation) / write-off, with a durable drain
ledger; execute the queue-drain cleanup **after** G1/G2 sign-offs land (that human gate is
deliberate and stays).
**30-day outcome:** remote branches <400; boss-stuck count declining week-over-week with recorded
dispositions.
**Kill-switch:** harvest generates more new issues than it retires for 2 runs → tighten WIP caps or
halt; the engine must be net-draining by construction.

### T4 — ODR compliance wedge before EU AI Act enforcement (Aug 2)
**Claim:** third-party-verifiable decision receipts are the non-negotiable enterprise wedge, and the
deadline is external and fixed. Art. 14 (human oversight attestation) is the acknowledged gap.
**Work:** ODR-1 vendor-neutral receipt schema, ODR-2 Ed25519 signing, ODR-3 pip-installable offline
verifier (epic #8223). Note #8389 — the 700-line ODR engine PR — is itself in the T1 drain-campaign
set; T1 directly unblocks T4.
**30-day outcome:** `pip install` verifier validates a real production receipt from api.aragora.ai.
**Kill-switch:** ODR-3 not shippable by Jul 25 → cut to schema+signing, publish verifier as a repo
script, keep external claims narrower than measured proof (standing commercial rule).

### Explicitly deprioritized (unchanged from the June steering-leverage filter)
Marketplace, vertical packages, blockchain/ERC-8004 (superseded by the Sigstore Rekor path),
inbox-wedge GUI retest, tier-benchmark corpus. Breadth without steering = defer.

## 3. Operating loop: minimal-human-input execution

**Continuous (daemons / scheduled):**
- Mission engine tick (`aragora mission run/reconcile`) — after #8758 this drains seeded goals
- Quorum evidence collection with all 3 flags on (severity-gated + tiered + advisory), reliable
  reviewer pair claude+openai with OpenRouter fallback
- Tier 0-2 merge executor (after #8759) — dry-run until explicitly armed
- Harvest pass (after #8760) — WIP-capped
- Existing worktree reconciler + PR watch daemons (unchanged)

**Weekly human touchpoint (≤30 min, agenda fixed):**
1. Tier 3-4 settlement sign-offs from prepared evidence packets (never auto-settled)
2. Governance decisions of G1/G2 class (e.g., queue-drain close authority)
3. Kill-switch dashboard review: ready-queue age, branch count, boss-stuck trend, main-green rate
4. Re-arm any auto-halted executor (main-red, protected-file, tier-escalation halts per
   `docs/AGENT_OPERATING_CONTRACT.md`)

**Immediately actionable human items (this week):** G1/G2 sign-off on the queue-drain cleanup plan;
arming decision for the merge executor once #8759 demonstrates its dry-run.

## 4. Sequencing

```
Week 1: #8748 wiring + #8756 → start #8761 drain campaign → #8758 bridge design
Week 2: #8761 continues (incl. #8389 → unblocks ODR) → #8759 executor dry-run → ODR-1/2
Week 3: #8759 armed for Tier 0-2 → #8758 lands → T2 closed-circuit demo → ODR-3
Week 4: #8760 harvest engine → queue-drain execution (if signed off) → 30-day metric review
```

Dependencies: T1 unblocks T4 (#8389) and proves the gates T2's demo relies on. T3's cleanup is
independent but gated on human sign-off. Nothing here waits on new frameworks — every item composes
existing primitives (the standing lesson: remove fragile abstractions, don't add them).

## 5. Metrics to watch (weekly)

| Metric | Now (2026-07-01) | 30-day target |
|---|---|---|
| Ready PRs > 7 days old | ~15 of 19 | 0 |
| boss-stuck issues | 266 | declining WoW, dispositions recorded |
| Remote branches | 983 | <400 |
| Unattended Tier 0-2 merges (cumulative) | 0 | ≥3 with receipts |
| Closed-circuit mission demos | 0 | 1 |
| ODR verifier | none | pip-installable, verifies production receipt |
| Main required-checks green rate | ~88% (deploy flake) | >95%; deploy workflows fixed or quarantined from the signal |

## Appendix: the refined prompt this plan executes

> Produce an evidence-based strategic plan and stand up a minimally-supervised execution process,
> optimizing for merged-and-verified improvements per week of human attention. Phase 1: parallel
> evidence sweeps (pipeline health, strategy-doc state, code-level substrate audit with
> WORKS/PARTIAL/MISSING verdicts). Phase 2: identify the single binding constraint; rank thrusts,
> each with claim, evidence, measurable 30-day outcome, kill-switch metric, and explicit
> deprioritizations; reconcile with active plans rather than duplicating them. Phase 3: durable
> artifacts — committed plan doc via draft PR, GitHub epic with boss-loop-format sub-issues.
> Phase 4: automation contract — one canonical harness, tier boundaries (0-2 auto / 3-4 human),
> weekly ≤30-min human agenda, auto-halt conditions per the operating contract. Constraints:
> compose existing primitives; no standing credentials; claims narrower than measured proof.
