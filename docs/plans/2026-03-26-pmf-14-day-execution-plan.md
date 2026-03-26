# 14-Day PMF Execution Plan

Last updated: 2026-03-26
Window: March 26, 2026 to April 8, 2026
Owner: Founder

Related:
- `ROADMAP.md`
- `docs/plans/PMF_DOGFOOD_EXECUTION_PLAN.md`
- `docs/plans/2026-03-25-six-week-founder-operating-plan.md`
- `docs/plans/2026-03-25-weekly-founder-review-rhythm.md`
- `docs/status/NEXT_STEPS_CANONICAL.md`
- `docs/status/PMF_SCORECARD.md`
- `docs/status/DESIGN_PARTNER_PROGRAM.md`
- `docs/plans/FOUNDER_RISK_REGISTER_2026_03.md`

## Purpose

Use the next 14 days to turn Aragora's proven founder loop into two things that
matter commercially:

1. repeatable daily use of the inbox trust wedge on a real founder inbox
2. repeatable external proof strong enough to start design-partner motion

This is not a broad roadmap. It is the current execution tranche.

## Current Truth On March 26, 2026

- `main` already contains the founder-loop, onboarding, receipt, FastAPI, and
  live-stream repairs needed for the canonical quickstart path.
- The live founder loop is a guardrail, not the primary proof target.
- The next true PMF gate is the second workflow: the inbox trust wedge.
- GitHub's open issues do not currently represent the PMF backlog truthfully;
  they are enterprise-assurance items.
- The backlog for this window must be recreated from observed live failures,
  not from speculative infrastructure work.

## Success At The End Of This Window

The 14-day window is successful only if all of these are true:

1. The founder inbox wedge has been used on real email across at least 6
   business-day sessions with persisted receipts and truthful terminal states.
2. A metrics log exists for inbox quality, latency, cost, overrides, and
   important-email misses.
3. One clean external demo path is frozen and can be run from a checklist with
   no local debugging.
4. At least 5 target accounts are actively qualified, at least 2 discovery
   calls are completed, and at least 1 live demo is run on a real artifact.
5. Every product fix landed during the window ties directly to inbox proof,
   demo readiness, or design-partner activation.

## Non-Goals For The Next 14 Days

- no broad connector expansion beyond the inbox wedge
- no enterprise assurance execution unless a real prospect requires it now
- no new orchestration or autonomy surface that does not unblock a current
  proof path
- no generic platform positioning refresh disconnected from a live wedge
- no backlog gardening that is not tied to a failed live run or a blocked demo

## Proof Metrics That Matter In This Window

Track these every day and review them formally twice during the window:

- founder inbox sessions completed
- real emails processed
- percent of executed actions with persisted receipts
- average latency per email
- average cost per email
- override rate
- important-email recall on the labeled slice
- time from clean setup to first visible receipt in the external demo path
- number of qualified design-partner accounts
- number of completed discovery calls
- number of completed live demos
- number of prospects with PMF score `>= 65`

## Decision Rules

- Proof metrics outrank code volume, test volume, and roadmap breadth.
- No more than 3 active blockers may sit in the `fix now` queue at one time.
- If a task does not improve inbox proof, demo readiness, or partner movement,
  it is out of scope for this window.
- If a blocker is not reproduced on a real path, it does not displace the top
  queue.
- If trust breaks, all other work pauses until trust is restored.

## Execution Order

### 1. Keep The Founder Loop Green As A Guardrail

Run the controlled baseline at the start of the window and after every blocker
fix that touches the default loop:

```bash
python3 -m pytest tests/e2e/test_user_journey.py tests/cli/test_quickstart.py -q
```

Run one live founder-loop proof at least twice per week:

```bash
ARAGORA_USER_ID=an0mium \
python3 -m aragora.cli.main quickstart \
  --question "Should Aragora use its own dogfood pipeline to close the remaining PMF gaps?" \
  --no-browser
```

Acceptance:
- no silent fallback
- persisted receipt visible on the product surface
- runtime stays below 90 seconds
- exact command and receipt are recorded in the weekly packet

### 2. Dogfood The Inbox Trust Wedge On Real Founder Email

This is the primary product objective for the window.

Day 1-3:
- validate readiness with `aragora triage status`
- complete Gmail auth with `aragora triage auth`
- run 3 dry-run founder sessions on real inbox slices
- create the first labeled slice of 25 messages
- record per-run latency, cost, overrides, and misses

Day 4-7:
- run 3 more founder sessions
- process at least 25 real emails cumulative
- keep allowed actions narrow: `ARCHIVE`, `STAR`, `LABEL`, `IGNORE`
- enable auto-approve only if receipt-before-action and quality gates hold

Acceptance for the window:
- at least 6 founder inbox sessions
- at least 25 real emails processed
- `100%` receipt-before-action on executed operations
- no reply, send, or forward autonomy
- a written decision on whether auto-approve remains off, limited, or enabled

### 3. Build The Inbox Proof Packet

By the end of Day 7, produce one short packet that can be used internally and
externally:

- one-page workflow description
- exact founder command transcript
- one labeled evaluation slice summary
- 2-3 representative receipts
- one honest list of remaining constraints and operator boundaries

This packet is the evidence base for design-partner calls. Do not pitch the
wedge without it.

### 4. Freeze One External Demo Path

The demo path for this window is:

`signup/onboarding -> quickstart receipt -> receipts surface -> optional inbox wedge evidence`

Day 6-10:
- freeze one 7-minute demo script
- prove a clean setup checklist that reaches first receipt in under 10 minutes
- complete 3 clean rehearsals with no local surgery during the run
- write one short receipt-backed case study from founder proof

Acceptance:
- one checklist works from a clean environment
- three rehearsals complete cleanly
- one short case-study artifact exists

### 5. Start Design-Partner Motion From Narrow Evidence

Day 8-14:
- shortlist 15 named accounts across the best-fit segments already defined in
  the design-partner program
- move the best 5 into active qualification
- complete at least 2 discovery calls
- run at least 1 live demo on a real artifact
- score every active prospect with `docs/status/PMF_SCORECARD.md`

Operating rule:
- lead with one bounded workflow only
- do not sell the full platform
- do not show surfaces that were not proven live during this window

Acceptance:
- 5 active qualified targets
- 2 completed calls
- 1 completed live demo
- 1 prospect identified as the most likely first weekly pilot

### 6. Run Only Bounded Repair Lanes

If the inbox wedge or the external demo fails, repair only the smallest slice
that restores proof.

Allowed repair categories:
- receipt persistence or visibility
- onboarding or credential-readiness clarity
- latency or timeout regressions on canonical paths
- inbox classification or action-safety bugs
- surface coherence issues that block a demo or first receipt

Disallowed during this window unless directly pulled by a live blocker:
- new connectors
- enterprise compliance packaging
- net-new orchestration surfaces
- speculative infrastructure expansion

## Daily Operating Rhythm

### Monday / Thursday
- run the founder-loop guardrail
- run one inbox session
- update blocker board from live evidence only

### Tuesday / Wednesday
- run inbox dogfood sessions
- land only blockers that stop the next inbox session or demo rehearsal

### Friday
- run one review packet update
- score partner movement
- enforce the stop-doing list for the next 7 days

## Required Artifacts By April 8, 2026

- inbox metrics ledger
- first labeled inbox slice
- inbox proof packet
- frozen 7-minute demo script
- clean setup checklist
- one receipt-backed case study draft
- partner target list with PMF scores
- one weekly review packet using the founder review rhythm

## Stop-Doing List For This Window

Freeze the following until the inbox wedge is repeating and at least one design
partner is moving toward first receipt:

- enterprise assurance execution work
- generic autonomy or swarm expansion
- connector breadth not tied to Gmail/inbox proof
- product storytelling work that outpaces live evidence
- frontend or dashboard polish that is not on the demo path
- broad performance work outside quickstart, onboarding, and inbox triage

## Window-End Review Questions

At the end of the 14 days, answer these in writing:

1. Is the founder inbox wedge useful enough that the founder would miss it if
   it disappeared?
2. Did the external demo path produce at least one trustworthy first impression
   without local rescue?
3. Which prospect is closest to a weekly pilot and why?
4. Which blockers remain real enough to justify the next repair tranche?
5. What work must be cut because it did not improve proof, partner motion, or
   trust?

If the answers are weak, narrow the wedge further. Do not widen the roadmap.
