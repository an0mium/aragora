# Weekly Founder Review Rhythm

Last updated: 2026-03-25

This is the weekly operating rhythm for Aragora's founder review.

It exists to answer four questions, in order:

1. What new proof did Aragora produce this week?
2. Which blockers are directly preventing the next proof?
3. Did design partner motion advance toward real weekly usage?
4. What work should stop immediately because it is not serving the current gate?

Current context:

- The canonical founder loop is already proven repeatable on `main`.
- The current execution gate is the second workflow (inbox trust wedge) plus
  design partner readiness.
- This review is not a general status meeting. It is a weekly reallocation of
  time, attention, and credibility around proof.

## Operating Principle

Run the review from evidence, not narrative.

The meeting is successful only if it produces:

- one clear proof verdict
- one ranked blocker list
- one explicit design partner movement plan
- one enforced stop-doing list

If the review does not remove work, narrow focus, or move a proof surface
forward, it is too soft.

## Weekly Cadence

### Monday morning: assemble the review packet

Prepare a single founder packet before the meeting with four sections:

1. Proof ledger
   - live runs completed this week
   - receipts produced and verified
   - demos or partner-visible artifacts created
   - exact commands, links, or documents that prove the result
2. Blocker board
   - blockers observed from real usage only
   - severity, owner, and age
   - evidence showing the blocker on a live path
3. Design partner movement board
   - current target accounts
   - stage: qualified, demo scheduled, first receipt, weekly active, paid pilot
   - next meeting, next artifact, and current PMF score
4. Stop-doing candidates
   - work that consumed time this week without improving proof, partner motion,
     or blocker removal

The packet should fit on one page or one short memo. Long narrative updates are
not allowed.

### Monday review: 75 minutes, fixed agenda

#### 1. Proof review (20 minutes)

Start with evidence before discussing plans.

Questions:

- What exact new proof landed since the last review?
- Which proof surface advanced: decision review, inbox trust wedge,
  Ralph/swarm, or design partner pilot?
- Is the proof repeatable, visible, and receipt-backed?
- What changed because of the proof: product confidence, partner credibility,
  or execution priority?

Rules:

- A test, benchmark, or demo only counts if it is tied to the current wedge.
- "We built infrastructure" is not proof unless it changed a live workflow.
- No week is green without at least one new proof artifact or a truthful stop
  reason for why proof did not advance.

Output:

- `prove`, `repair`, or `re-scope` for the current weekly proof surface

#### 2. Blocker triage (20 minutes)

Review only blockers that prevent the next proof or the next design partner
step.

Classify each blocker into exactly one bucket:

- `fix now`: directly blocks this week's proof or partner commitment
- `schedule next`: important, but not the immediate gate
- `delegate`: valuable but does not need founder time
- `drop`: real issue, wrong time

Questions:

- Did this blocker show up in a real run, real review, or real partner motion?
- What proof or movement is impossible until it is fixed?
- Can the blocker be reduced to one bounded lane with a file scope and
  validation contract?

Rules:

- Do not carry more than three `fix now` blockers at once.
- If a blocker cannot be tied to a failed or constrained real workflow, it does
  not enter the top queue.
- If the blocker is broad, rewrite it into the smallest proof-preserving slice.

Output:

- top three blockers for the week, each with owner, scope, and validation

#### 3. Design partner movement (20 minutes)

Treat partner motion as weekly evidence generation, not pipeline theater.

Review each active target:

- current stage
- last concrete interaction
- next artifact needed
- blocker to first receipt or next weekly run
- PMF score trend

Questions:

- Which partner is closest to a first receipt this week?
- Which partner can become weekly active fastest on one bounded workflow?
- Which partner is consuming attention without meeting the ICP?
- What single artifact or session would materially change the odds this week?

Rules:

- Optimize for first receipt, then weekly usage, then paid pilot.
- Do not expand one partner into multiple surfaces before one recurring
  workflow is sticky.
- If a prospect cannot provide a champion, a bounded workflow, and a weekly
  operating cadence, move them out of the active set.

Output:

- one primary design partner move for the week
- one secondary backup move
- explicit drops from the active pursuit list

#### 4. Stop-doing enforcement (15 minutes)

This section is mandatory. The founder must explicitly kill or defer work.

Ask:

- What did we spend time on last week that did not improve proof, remove a top
  blocker, or advance a design partner?
- Which activities are generating narrative comfort instead of evidence?
- What should be frozen until the inbox trust wedge and design partner motion
  are visibly advancing?

Default stop-doing list for the current phase:

- enterprise assurance expansion before design partner proof
- broad connector expansion not tied to a live wedge
- generic orchestration or autonomy infrastructure without a current blocker
- multi-surface pilots for the same partner before one workflow repeats weekly
- roadmap polishing that does not change a live founder, partner, or proof path
- claiming "ready" or "operational" without a current receipt-backed example

Output:

- one explicit stop list for the next seven days
- one "not now" list to revisit only after the current gate is cleared

## Review Scoreboard

Track the review with a short scoreboard:

| Area | Weekly question | Green | Yellow | Red |
|------|-----------------|-------|--------|-----|
| Proof | Did new wedge-relevant proof land? | repeatable receipt-backed proof | partial proof or noisy run | no new proof |
| Blockers | Are top blockers sharp and bounded? | <=3 direct blockers with owners | mixed queue, some drift | generic backlog soup |
| Partners | Did a partner move toward weekly use? | first receipt, repeat use, or pilot step | meetings without artifact movement | no real movement |
| Stop-doing | Was work explicitly cut? | at least one meaningful cut | soft defer language only | no work stopped |

The week is only truly green if proof moved and at least one non-essential
track was cut.

## Required Artifacts After The Review

Every weekly review should end with a short written output:

### 1. This week's proof verdict

- what was proven
- what remains unproven
- the next proof target

### 2. This week's top blockers

- ranked 1-3
- owner
- bounded deliverable
- validation command or acceptance test

### 3. This week's design partner moves

- primary account and next step
- backup account and next step
- accounts removed from active focus

### 4. This week's stop-doing list

- work frozen for the next seven days
- reason it is frozen
- condition for re-entry

## Decision Rules

Use these rules to prevent drift:

1. Proof beats plans.
   - If proof and planning conflict, fund proof.
2. Real blockers beat imagined architecture.
   - The blocker queue must come from observed failures, not abstract concerns.
3. First receipt beats broad outreach.
   - A partner closer to a live weekly workflow outranks a more prestigious but
     vague prospect.
4. Narrow repetition beats surface breadth.
   - One workflow repeated weekly is better than three impressive one-off demos.
5. Killed work stays killed for the week.
   - Do not quietly resurrect work outside the review.

## Suggested Founder Memo Template

Use this structure for the weekly note:

```md
# Founder Review - <date>

## Proof Verdict
- New proof:
- Evidence:
- Next proof target:

## Top Blockers
1. <blocker> - owner - validation
2. <blocker> - owner - validation
3. <blocker> - owner - validation

## Design Partner Movement
- Primary move:
- Backup move:
- Dropped:

## Stop Doing
- Frozen this week:
- Re-entry condition:
```

## Assumptions

- Founder time is the scarcest resource and should be allocated against proof
  and partner movement, not generic activity.
- Aragora's current PMF gate is not broad market expansion; it is repeated
  proof on a narrow wedge plus evidence of design partner pull.
- The PMF scorecard should be used weekly on active partners, but it should not
  replace direct artifact review and founder judgment.

## Open Questions

- Should this review happen with a standing partner scorecard appendix or as a
  single merged memo?
- Does the inbox trust wedge need its own sub-scorecard once live dogfood begins
  weekly?
- At what point should "stop-doing enforcement" become a visible public team
  artifact instead of a founder-only control?

## Next Actions

- Use this rhythm as the default weekly founder review starting this week.
- Fold the current proof ledger, blocker board, and design partner board into a
  single pre-read before each Monday review.
- Revisit the default stop-doing list only after the inbox trust wedge and at
  least one design partner workflow are repeating weekly.
