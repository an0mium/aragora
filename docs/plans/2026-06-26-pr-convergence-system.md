# PR Convergence System — Anti-Churn Budget, Scope Contract, and Net-Value Adjudication

**Status:** Proposed (design / Tier-4 pre-approval artifact)
**Author:** Claude (Opus 4.8), with Armand (scarmani)
**Date:** 2026-06-26

## 1. Problem

The merge-quorum gate keeps PRs from regressing `main`, and that part works. But the
surrounding loop does not *converge*. Empirically (measured 2026-06-26):

| PR | commits | span | surface | merges | posted evidence |
|----|---------|------|---------|--------|-----------------|
| #8595 | 42 | 36h | classifier gates | 0 | 0 |
| #8575 | 11 | 53h | evidence-timeout | 0 | 0 |
| #8627 | 9 | 7h | lane-lease | 0 | 0 |
| #8628 | 16 | 6.5h | swarm-lease | 0 | 0 |

`#8595` alone produced **16 near-duplicate "fail closed on <uncertain X>" commits in
under two hours.** Across these four PRs: 78 commits, 0 merges, 0 posted evidence. The
loop converts compute into commits and recursive prompts, not landed value.

## 2. Root cause

The gate optimizes **"are there findings?"** (a defect-absolute, per-round, memoryless
filter) when the merge decision should be **"is landing this now net-positive versus
another round?"** (a value-relative, trajectory-aware judgment).

Two facts make this fatal:

1. **Adversarial review never returns empty.** A frontier model asked "what's wrong with
   this code?" always finds *something* in any non-trivial diff. "Drive findings to zero"
   optimizes toward an unreachable target over an unbounded space — it cannot terminate.
2. **The repair agent's objective is "make findings disappear," not "maximize value."**
   Generator (reviewer) that always produces + follower (repairer) that always chases +
   no cost per round + no cross-round memory = a divergent loop.

This is exactly why FunSearch/AlphaEvolve *converge* (their evaluator is a fixed fitness
function with a target) and this does not (its evaluator is an open-ended critique).

**The reshaping implementation finding:** the gate *already* has a budget —
`max_reruns_per_head=3` in `plan_rerun()` ([merge_quorum_reconcile.py:188]) — but it is
**keyed by head SHA.** Every repair commit is a new head, so the cap resets every round
and never bites. **The budget must be keyed by PR, not head**, to survive head drift.
That single change is the highest-leverage fix in this document.

## 3. Principle

> **Defect-absolute gating diverges; value-relative gating converges.**

Every component below injects *value* and *cost* into a process that today has only
*defect detection*. The merge bar changes from *"zero findings"* (unreachable) to
*"no BLOCKING-in-scope findings, within a bounded budget"* (reachable).

## 4. Architecture — three layers

- **Layer C — Gas + forced-choice adjudication (the spine).** A depleting per-PR round
  budget. When it depletes, the loop *cannot silently continue*: it emits a decision
  verdict, and (later) a frontier panel forces one of `MERGE_AS_IS / ONE_BOUNDED_ROUND /
  CLOSE / RESTRUCTURE`. This alone converts divergent → bounded.
- **Layer A — Finding triage + scope contract (the quality multiplier).** Each finding is
  judged `BLOCKING_IN_SCOPE / DEFER_OUT_OF_SCOPE / NITPICK / RESTRUCTURE_SIGNAL` against a
  machine-readable scope contract set at PR creation. Out-of-scope `[P2]`s stop blocking.
- **Layer B — Trajectory / churn memory (the detector).** Per-PR, per-surface finding
  history. Same surface flagged N rounds → `RESTRUCTURE_SIGNAL`. Mechanizes the #8511
  lesson ("reviewer flagging the same area N rounds = fix the abstraction").

## 5. Grounded integration seams

| Concern | Seam | Notes |
|---|---|---|
| "Rerun vs stop" chokepoint | `plan_rerun()` — `aragora/swarm/merge_quorum_reconcile.py:178` | pure; caller supplies state |
| Per-PR durable state | `~/.aragora/merge_quorum_reconcile_state.json` | atomic write, pruned 500, **keyed by head** |
| Verdict builder | `_build_model_review_quorum()` — `aragora/cli/commands/review_queue.py:3101` | 8-verdict elif cascade; add `net_value_adjudication_required` |
| Severity split | `would_count` (lint) blocks `[P2]` regardless of flag; `dissenting` (`quorum_evidence.py:427`) honors flag | **triage judge plugs into `dissenting`, narrowing `dissenting_views` — never the lint** |
| Scope contract | `SpecBundle` — `aragora/pipeline/backbone_contracts.py:217` | add `non_goals`, `deferred_surfaces`; PR template + parser |
| Tier/surface classify | `_classify_model_review_tier` / `_subsystem_for` — `review_queue.py` | reusable by triage |
| Judge infra | `LLMJudge` (`aragora/evaluation/llm_judge.py:533`), `default_reviewer_runner` (`quorum_evidence.py:1006`), `_reviewer_verdict` parser, `PartialConsensus`, `cost_tracker` | reuse for both judges |

## 6. Phased plan (smallest-first, dependency-ordered)

Each phase is one small landable PR, dogfooded through the merge gate under the same
2-round discipline it encodes.

| # | Phase | Lands | Tier | Depends | Model calls |
|---|---|---|---|---|---|
| 0 | Per-PR convergence ledger + budget logic | visibility + the data model | **2** | — | no |
| 1a | `plan_rerun()` per-PR budget hard-stop | loop cannot exceed N rounds | **2** | 0 | no |
| 1b | `net_value_adjudication_required` verdict | gate surfaces "decide" instead of looping | **4** | 1a | no |
| 2 | Scope contract (`SpecBundle` + PR template + parser) | PRs declare goals/non-goals | 3 | — (parallel) | no |
| 3 | Finding-triage judge → narrows `dissenting_views` | out-of-scope `[P2]`s stop blocking | **4** | 2 | yes |
| 4 | Churn detector (per-surface trajectory) | auto `RESTRUCTURE_SIGNAL` | 2 | 0 | no |
| 5 | Net-value adjudicator (panel + forced choice + signed receipt) | automated MERGE/ROUND/CLOSE/RESTRUCTURE | **4** | 1,3,4 | yes |
| 6 | Real gas (token-cost budget via `cost_tracker`) | budget = cost, not round count | 2 | 5 | — |

**Phases 0 + 1a ship with zero model calls and no Tier-4 surface** — they convert
divergent → bounded *deterministically*. The frontier judges (3, 5) only improve
*decision quality* on top of an already-bounded loop. The dangerous, expensive, Tier-4
parts come last, after the cheap fix has already stopped the bleeding.

## 7. Design decisions (locked)

- **Budget keyed by PR, not head** — defeats head-drift; the core fix.
- **Default-off / additive** — new `plan_rerun` params default to a disabled budget so
  existing callers/tests are byte-for-byte unaffected until a caller opts in.
- **Triage narrows `dissenting`, never the `would_count` lint** — the lint is flag-blind;
  only the dissent layer can be made value-relative.
- **Forced choice, budget-decrementing** — the adjudicator returns one enum value (parsed
  via the proven `_reviewer_verdict` pattern), its own invocation decrements the budget,
  and `ONE_BOUNDED_ROUND` is choosable at most once before the next call must pick
  MERGE/CLOSE/RESTRUCTURE. No "let me think about it" continuation.
- **RESTRUCTURE inherits a lower budget** — the system cannot infinitely restructure.
- **Scope is immutable to the repairer** — set at PR creation from `SpecBundle`; otherwise
  an agent dodges findings by declaring everything out of scope.
- **Every adjudication is a signed receipt** — reuse the gauntlet receipt store; auditable,
  and the policy is tunable from outcomes. Humans can always override.

## 8. Anti-gaming / failure modes

- *A free-text "is this good?" judge waffles and churns* → forced choice + budget
  decrement + panel majority (cut variance via `PartialConsensus`).
- *Restructure can loop too* → must change scope/abstraction (smaller surface) and inherit
  a lower budget.
- *Self-declared scope is a loophole* → scope set at creation, immutable to the repairer.
- *Building the anti-churn system could itself churn* → build it under its own rule: one
  small PR per phase, 2-round cap, land before the next. Once 1a lands, run the remaining
  phases' PRs *through the budget they create.*

## 9. Rollout

1. Land Phase 0 + 1a (this PR) — Tier 2, deterministic. The loop becomes bounded.
2. Wire the live reconcile script (`scripts/reconcile_merge_quorum.py`) to read the ledger
   and pass `pr_rounds_consumed` / `pr_round_budget` into `plan_rerun()` — tight follow-up.
3. Land 1b (Tier 4) — the gate emits the decision verdict.
4. Build 2, then 3/4, then 5 — each small, each dogfooded.
5. The system governs its own construction from Phase 2 onward.

## 10. Out of scope (non-goals for the first PR)

- No frontier-model calls (Phases 0/1a are pure).
- No change to `review_queue.py` verdict logic (that is 1b, separate Tier-4 PR).
- No live reconcile-script wiring (separate follow-up).
- No public API / interface changes.
