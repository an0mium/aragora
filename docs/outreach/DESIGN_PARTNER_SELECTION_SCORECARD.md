# Design Partner Selection Scorecard

Last updated: 2026-03-25

## Purpose

Use this scorecard to rank design partner prospects for Aragora's current
beachhead:

- auditable multi-model decision review
- receipt-gated bounded execution
- truthful blocker handling for consequential workflows

The goal is not to pick the biggest logo. The goal is to pick the 3-5 partners
most likely to generate fast, credible proof that Aragora's wedge matters.

## Hard Gates

Do not advance a prospect if any of these are false:

1. They have one bounded recurring workflow with a clear trigger, owner, and
   success/failure outcome.
2. The workflow is consequential enough that receipts, provenance, review, or
   explicit approval gates are valuable.
3. They can provide real artifacts during the partner period. Sanitized inputs
   are acceptable.
4. They have a named champion who will join onboarding and a weekly review loop.
5. They can start with a narrow pilot in 30 days or less.

If a prospect fails a hard gate, mark them `not now` even if the weighted score
looks attractive.

## Scoring Model

Score each dimension from 1 to 5 after the first serious discovery call.

| Dimension | Weight | What it means |
|-----------|--------|---------------|
| Urgency | 30% | How painful and time-sensitive the current workflow is |
| Tractability | 30% | How easily Aragora can be piloted on one narrow workflow |
| Credibility | 20% | How likely the prospect is to run a real pilot and become a referenceable customer |
| Learning value | 20% | How much the pilot will teach Aragora about its wedge and ideal customer profile |

Formula:

`weighted score = (0.30 x urgency) + (0.30 x tractability) + (0.20 x credibility) + (0.20 x learning value)`

Multiply by 20 for a 100-point score.

## Rubric

### 1. Urgency

| Score | Signals |
|-------|---------|
| 1 | Curiosity project, no active pain, no deadline, no owner feeling real pressure |
| 2 | Problem is acknowledged, but current workaround is acceptable for now |
| 3 | Recurring pain exists and someone cares, but the workflow is not yet breaking an SLA, audit need, or throughput goal |
| 4 | Pain is acute: review latency, triage load, audit evidence burden, or bounded backlog pressure is affecting the team now |
| 5 | The current process is visibly failing or expensive, and the champion wants a pilot immediately because delay has a real cost |

Questions to ask:

- What happens today when this workflow goes wrong?
- What manual review, audit, or triage cost shows up every week?
- Why is now the right time to run this pilot?

### 2. Tractability

| Score | Signals |
|-------|---------|
| 1 | Prospect wants a broad AI transformation story, not one narrow workflow |
| 2 | A candidate workflow exists, but scope, owner, or success criteria are still fuzzy |
| 3 | One workflow is identifiable, but artifacts, integrations, or operating cadence are still partially unclear |
| 4 | One narrow workflow is clear, artifacts are available, success is measurable, and the pilot can start with limited integration work |
| 5 | Ideal first pilot: one recurring trigger, one operator owner, real artifacts ready, weekly feedback loop agreed, and a receipt clearly improves trust or governance |

Questions to ask:

- Can we start with one workflow before talking about broader rollout?
- What artifact starts the workflow?
- What exact human decision or action sits at the end?
- What would make the first 2 weeks clearly successful or unsuccessful?

### 3. Credibility

| Score | Signals |
|-------|---------|
| 1 | Tire-kicker, student project, consultant without workflow ownership, or no real path to pilot |
| 2 | Friendly interest, but authority, budget, or access to operators is unclear |
| 3 | Real champion exists, but internal alignment or approval path is still uncertain |
| 4 | Champion has workflow ownership or direct influence, can bring real artifacts, and can keep a weekly cadence |
| 5 | Strong operational sponsor with authority, concrete pilot timeline, path to expansion or paid conversion, and credible willingness to be a reference if the pilot works |

Questions to ask:

- Who owns the workflow operationally?
- Who decides whether the pilot continues?
- Can this team actually provide artifacts and operator time every week?
- If the pilot works, what happens next?

### 4. Learning Value

| Score | Signals |
|-------|---------|
| 1 | Bespoke edge case with little transfer to future customers |
| 2 | Some product learning, but mostly one-off implementation detail |
| 3 | Useful learning for one segment or one product surface |
| 4 | Teaches Aragora something important about its wedge, especially around receipts, review, governance, or bounded execution |
| 5 | High-leverage pilot that can validate positioning, generate reusable messaging, expose important failure modes, and transfer directly to multiple future prospects |

Questions to ask:

- Does this workflow test Aragora's real wedge or just generic automation?
- Will success here produce a reusable case study or proof point?
- Does the segment match the current best-fit profile?
- Will we learn something that changes roadmap or positioning decisions?

## Decision Rules

Use the weighted score, but keep simple floor rules:

- `Fast-track`:
  weighted score >= 4.2, with urgency >= 4 and tractability >= 4
- `Qualified`:
  weighted score 3.6-4.19, with no dimension below 3
- `Monitor`:
  weighted score 3.0-3.59, or learning value is high but the pilot is not yet tractable
- `Not now`:
  weighted score < 3.0, any hard gate fails, or tractability/credibility <= 2

Tiebreakers:

1. Higher tractability wins.
2. Then higher urgency.
3. Then higher learning value.

At Aragora's current stage, speed-to-proof matters more than prestige.

## Best-Fit Prospect Pattern

The best design partners usually have most of these traits:

- regulated or high-accountability environment
- recurring review or triage bottleneck
- visible trust gap with single-model workflows
- strong need for audit evidence or explicit approval gates
- one operator champion who wants to start narrow

Examples of strong first workflows:

- design doc or architecture review with decision receipts
- PR or release review for consequential engineering changes
- inbox or queue triage where receipt-before-action matters
- bounded backlog execution with explicit review and stop conditions

## Capture Template

Use this block in notes or CRM:

```md
## Design Partner Scorecard

Prospect:
Segment:
Champion:
Workflow:
Why now:

Hard gates:
- bounded recurring workflow:
- consequential enough for receipts/gates:
- real artifacts available:
- weekly champion available:
- can start in <=30 days:

Scores:
- urgency (30%):
- tractability (30%):
- credibility (20%):
- learning value (20%):

Weighted total:
Recommendation:
Key evidence:
Main risks:
```

## Operating Principle

Do not accept prospects that mainly want "general AI transformation help."
Prefer prospects where Aragora can prove:

- disagreement is useful
- receipts change trust
- review gates reduce risk
- bounded execution can be governed truthfully
