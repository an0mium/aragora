# Aragora Founder Discovery Call Script

Last updated: 2026-03-24

## Objective

Use this call to determine whether a prospect has a narrow, painful, recurring
workflow that Aragora can improve with adversarial AI review and decision
receipts.

The goal is not to pitch every feature. The goal is to leave with evidence on:

- the exact workflow
- how often it happens
- who owns it
- what failure costs today
- why existing AI or manual review is not enough
- whether the team can supply real artifacts and run a 4-6 week design partner loop

## Qualification Standard

Strong discovery calls produce all of the following:

- one concrete workflow, not a vague strategy discussion
- one painful recent example, not a hypothetical future use case
- one named champion, owner, or buyer
- one artifact path for a pilot: inbox batch, PR/spec, policy doc, backlog, or comparable input
- one reason trust, auditability, or dissent visibility matters

## 30-Minute Call Flow

| Segment | Time | Goal |
|---|---:|---|
| Opening | 2 min | Set expectations and de-risk the call |
| Workflow reconstruction | 8 min | Understand the current process in detail |
| Pain and consequence | 7 min | Surface stakes, delays, and failure cost |
| Fit diagnosis | 8 min | Map the workflow to Aragora's best proof surface |
| Close | 5 min | Confirm evidence, next step, and pilot viability |

## Opening Script

Use this opener:

> Thanks for making the time. This is a discovery call, not a product demo.
> I want to understand one workflow where AI-assisted decisions are high stakes,
> slow, or hard to trust. If there is a real fit, we can identify one narrow
> pilot and the evidence we would need to validate it.

Diagnostic questions:

- What is the single decision or review workflow you most want to improve right now?
- Why is that workflow painful enough to discuss today?
- If we only talked about one workflow on this call, which one matters most?

Red flags:

- They want a generic "AI strategy" conversation with no concrete workflow.
- They jump immediately to feature requests before describing the current problem.
- The pain is interesting but not active this quarter.

## Workflow Reconstruction

Ask for the most recent real example, not the ideal process:

> Walk me through the last time this happened from trigger to final decision.
> Pretend I am shadowing the operator.

Diagnostic questions:

- What triggers the workflow?
- What artifacts go in at the start?
- Who touches it before a decision is made?
- Where does disagreement show up today?
- What tools are involved: email, docs, GitHub, ticketing, internal systems?
- How many times per week or month does this happen?
- What is the median time from trigger to decision?
- What does a "done" output look like today?

Red flags:

- The workflow is rare, ad hoc, or dependent on one founder's intuition.
- There is no stable trigger, no repeatable input artifact, or no clear output.
- No one can describe the last real example with specifics.

## Pain and Consequence

Push past "it is inefficient" and get to consequence:

> Where does this workflow actually break today: latency, mistakes, rework,
> trust, audit, or politics?

Diagnostic questions:

- What is the most expensive failure mode?
- What happens when the wrong call is made?
- What is the cost of false positives versus false negatives?
- How much human review is required before anyone trusts the output?
- Where do people ask "why did we decide that?" after the fact?
- Has this caused a customer, compliance, security, legal, or leadership issue?
- If this improved by 50%, what would change in the business?

Red flags:

- The consequence is only "it would be nice to save time."
- Mistakes are cheap and reversable, so rigor is not valuable.
- There is no trust problem, no review bottleneck, and no audit burden.

## Fit Diagnosis: Choose One Proof Surface

Do not pitch all three surfaces equally. Pick the strongest fit and test it.

### 1. Decision Review

Best when the prospect has high-stakes decisions that already require review:
specs, PRs, policies, architecture choices, risk memos, legal analysis.

Diagnostic questions:

- What artifacts already go through approval or sign-off?
- Where do reviewers disagree or miss edge cases?
- When a decision is challenged later, what evidence is missing?
- Would a receipt showing consensus, dissent, and provenance change anything?

Red flags:

- The team does not preserve decision artifacts.
- No one needs an audit trail, rationale, or dissent visibility.
- They really want content generation, not decision vetting.

### 2. Autonomous Repo Improvement

Best when the prospect has a bounded engineering backlog with clear acceptance
criteria and strong merge discipline.

Diagnostic questions:

- What recurring engineering work is boring, bounded, and easy to verify?
- How are issues scoped and accepted today?
- What review gates are non-negotiable before merge?
- Would the team trust autonomous work only if it shipped with a receipt and review trail?

Red flags:

- They want unbounded autonomous coding without review policy.
- There is no issue hygiene, no acceptance criteria, or no owner.
- They want a general coding copilot, not governed execution.

### 3. Inbox Trust Wedge

Best when the prospect triages high-volume inbound items where wrong actions are
costly: founder inbox, support escalations, security intake, partner requests.

Diagnostic questions:

- Who owns inbox or queue triage today?
- What volume arrives each day or week?
- Which actions are safe, and which require high confidence?
- What is the cost of misrouting, deleting, or acting on the wrong item?
- Could the team start with receipt-before-action on a narrow queue?

Red flags:

- The inbox volume is too low to matter.
- Wrong actions are harmless, so trust is not a meaningful problem.
- They want full automation before any review loop or receipt gate.

## Buyer and Design Partner Readiness

Confirm whether they can actually run a design partner process.

Diagnostic questions:

- Who would own a pilot internally?
- Who signs off on security, legal, or procurement questions?
- Can you provide sanitized real artifacts for one bounded workflow?
- Could you support a 30-minute onboarding and weekly 30-minute check-in for 4-6 weeks?
- Is there a deadline or forcing function this quarter?
- If the pilot works, what budget or buying path would exist?

Red flags:

- No internal champion with enough authority to provide artifacts or feedback.
- Procurement is long and heavy before any discovery or pilot can start.
- They want free consulting or roadmap influence without operational commitment.
- There is no near-term forcing function.

## Closing Script

Use this close:

> Based on what you described, the strongest starting point looks like
> [decision review / autonomous repo improvement / inbox trust]. To validate
> that, we would need one real artifact set, one success metric, and one owner.
> Does that sound accurate?

Then ask:

- What artifact could you share first?
- What metric would make this pilot clearly successful?
- Who needs to be in the next conversation?
- What is the right next step: follow-up discovery, live workflow review, or pilot design?

## Evidence To Capture After Every Call

Do not leave a call with only impressions. Capture evidence while the details
are still fresh.

### Minimum Evidence Packet

| Field | What to capture |
|---|---|
| Contact | Name, title, company, email, date |
| Candidate workflow | One sentence naming the workflow |
| Proof surface | Decision review, autonomous repo improvement, or inbox trust |
| Pain statement | One verbatim quote describing the pain |
| Last real example | Specific recent incident with date or timeframe |
| Volume and frequency | Per day, week, or month |
| Decision stakes | Cost of wrong answer, delay, or lack of trust |
| Current process | Tools, handoffs, review path |
| Required artifact | Spec, PR, inbox batch, policy doc, backlog, or equivalent |
| Champion | Person who can drive a pilot |
| Constraints | Security, compliance, deployment, procurement, data sensitivity |
| Success metric | What would prove the pilot worked |
| Next step | Concrete follow-up and owner |

### Evidence Standards

Capture at least:

- one direct quote in the prospect's own words
- one hard number: volume, SLA, review time, backlog size, or failure cost
- one concrete artifact the team can provide
- one concrete date: next meeting, deadline, audit, launch, or budget window

If you do not have those four things, the call probably stayed too abstract.

## Post-Call Scorecard

Score each category 0-3.

| Category | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Problem intensity | Nice to have | Mild pain | Active pain | Acute pain with visible cost |
| Workflow repeatability | Ad hoc | Occasional | Recurring | High-volume recurring |
| Trust or audit need | None | Weak | Clear | Mission-critical |
| Artifact readiness | None | Possible later | Sanitized examples | Real artifacts ready now |
| Champion strength | Curious observer | Informal contact | Operational owner | Owner with budget influence |
| Urgency | No timeline | Someday | This quarter | Immediate forcing function |

Interpretation:

- 14-18: strong design partner candidate
- 9-13: keep warm, gather more evidence
- 0-8: not a current fit

## Hard Disqualifiers

Do not force a pilot when these are true:

- no recurring workflow
- no owner
- no artifact access
- no measurable consequence for wrong decisions
- desire for unbounded autonomous action without governance
- interest driven only by curiosity, not operational pain

## Interviewer Notes

- Ask about the last real instance before asking about the ideal future.
- Prefer exact numbers over adjectives like "a lot" or "slow."
- Keep re-centering on one workflow.
- If the prospect spans multiple use cases, choose one and defer the rest.
- Do not leave the call without a next step or a clear disqualification reason.
