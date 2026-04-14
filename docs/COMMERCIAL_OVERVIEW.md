# Aragora Commercial Overview

**Internal positioning snapshot aligned to the current execution gate.**
**Last updated: April 14, 2026**

## Executive Summary

Aragora is currently a control plane for **bounded autonomous software execution**.

The commercial wedge is deliberately narrow:

- take a bounded engineering task
- run it through guarded autonomous execution
- produce code, verification evidence, PR state, and operator-readable receipts
- fail closed with explicit blocker evidence when autonomy should stop

That is the truthful sellable surface today. The current story is not "general AI for every decision" and not "omnichannel enterprise automation for everything." The near-term value is making unattended execution on bounded software work **measurable, inspectable, and auditable**.

## What Is True On `main`

The claims below are the current proof base, not the long-term ambition.

| Area | Current truth | Why it matters |
|---|---|---|
| Guarded execution substrate | Boss, supervisor, tranche, contract, and preflight paths exist on the live swarm lane. | The system can decide when a run is admissible instead of blindly attempting work. |
| Benchmark truth | A fixed benchmark corpus is checked into the repo, and the tracked B0 cohort is running at **86.7%** no-rescue success as of 2026-04-13. | Progress claims are tied to a measured cohort instead of anecdotes. |
| Truth artifacts | The repo has a diffable benchmark truth-artifact path and GitHub-truth reconciliation scripts. | Weekly or recurring reporting can stay tied to issue-level truth. |
| Repair loop progress | Resume-from-state, repair lifecycle persistence, rescue logging, and bounded recovery planning are on `main`. | Repeated failures can increasingly be resumed, diagnosed, and productized instead of re-run cold. |
| Operator truth | Preflight receipts, blocker evidence, and session-state work are materially underway and partially landed, but not yet fully closed across every live path. | This is the remaining gap between an impressive demo and a boring reliable product lane. |

## Current Gate

The current commercial gate is the same as the current execution gate in [status/NEXT_STEPS_CANONICAL.md](status/NEXT_STEPS_CANONICAL.md):

- finish `RS-07` so receipt-backed preflight becomes the default admission truth
- close the remaining truthful repair gaps in `BC-01` and `BC-03`
- keep `TW-01`, `TW-02`, and `TW-03` publishing recurring corpus-linked truth
- keep all external claims narrower than the measured proof

This means Aragora should currently be positioned as:

- a **guarded autonomous execution control plane** for bounded software tasks
- a **truthful operator surface** that says what happened, what failed, and what to try next
- a **measured benchmark system** for proving unattended execution on a fixed corpus

## What Aragora Can Be Sold For Now

### 1. Bounded engineering execution

Aragora is strongest when the work is narrow, reviewable, and single-PR shaped:

- focused bug fixes
- bounded tests and validation work
- safe code-generation loops with explicit verification
- repair and retry on previously seen failure classes

The operator value is not just "an AI wrote code." The value is:

- the run is scoped
- the run is checked before execution
- the result is tied to receipts, verification, and PR truth
- failure is explicit instead of silent

### 2. Benchmark-backed autonomy proof

Aragora can already support design-partner style proof loops where a team wants to know:

- what percentage of bounded tasks complete without rescue
- which failure classes repeat
- whether repairs are becoming more truthful over time
- whether autonomous progress is improving on a fixed corpus rather than a changing sample

This is commercially useful for internal platform teams and founder-led design partners because it turns autonomy from a vibe into an operating metric.

### 3. Operator-facing trust surface

The system is increasingly useful as an operator control plane for:

- admission truth
- session and repair state
- blocker evidence
- benchmark scorecards
- rescue-to-productization tracking

That surface matters even before the system is fully autonomous, because it reduces the cost of supervising automation honestly.

## Who This Is For Right Now

The current wedge is best suited to:

- engineering leaders who want bounded unattended execution on narrow backlogs
- founders or staff engineers running repeated benchmark-style issue queues
- internal platform teams that need receipts, blocker evidence, and fail-closed automation
- design partners willing to operate on a narrow, measurable execution class first

It is not yet best positioned as:

- a broad cross-functional decision platform
- a universal enterprise copilot across every channel
- a generalized memory or knowledge operating system
- a finished regulator-ready compliance suite

## What Not To Claim Yet

To stay truthful, external positioning should avoid these claims in the current tranche:

- "enterprise-ready general decision platform"
- "omnichannel AI operating system"
- "fully autonomous software factory"
- "broad multi-agent superiority across all workflows"
- "compliance-complete platform for regulated industries"

Those may be later-stage directions, but the current proof base is narrower. The commercial overview should stay narrower than the roadmap, not broader.

## Longer-Term Direction

The roadmap still supports a bigger business. The difference is sequencing.

### Stage 1: Trusted bounded execution

Prove that Aragora can repeatedly turn bounded tasks into:

- guarded runs
- truthful receipts
- verification evidence
- mergeable or merged PR outcomes

### Stage 2: Adjacent operator loops

Once the software-execution wedge is boringly reliable, extend the same substrate to:

- inbox and operator action loops
- prompt-to-spec handoff
- thin truthful operator views backed by live receipts and state

### Stage 3: Broader decision integrity platform

Only after the wedge is proven and repeatable should Aragora broaden into the fuller long-horizon vision:

- richer multi-agent decision workflows
- broader channels and connectors
- larger memory and context surfaces
- wider organizational decision operations

The long-term vision is still valid. The near-term positioning has to be earned by measured proof.

## Commercial Positioning Guidance

If a prospective customer asks what Aragora is today, the honest answer is:

> Aragora is a control plane for guarded autonomous software execution on bounded work. It helps teams run narrow issue queues with receipts, verification evidence, blocker truth, and recurring benchmark measurement.

If they ask what it can become, the honest answer is:

> Aragora is being built toward a broader decision integrity platform, but the current go-to-market wedge is intentionally narrower: make bounded unattended execution reliable first, then expand from proved operator loops outward.

## Proof Sources

Use these documents as the canonical backing for commercial claims:

- [status/NEXT_STEPS_CANONICAL.md](status/NEXT_STEPS_CANONICAL.md)
- [status/ACTIVE_EXECUTION_ISSUES.md](status/ACTIVE_EXECUTION_ISSUES.md)
- [plans/ARAGORA_EVOLUTION_ROADMAP.md](plans/ARAGORA_EVOLUTION_ROADMAP.md)
- [benchmarks/corpus.json](benchmarks/corpus.json)
- `scripts/build_benchmark_truth_artifact.py`
- `scripts/reconcile_b0_pr_truth.py`

If a claim cannot be tied back to those sources or to current `main`, it should not be in first-tranche commercial positioning.
