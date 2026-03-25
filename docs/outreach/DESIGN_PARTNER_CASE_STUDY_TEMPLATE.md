# Aragora Design Partner Case Study Template

Use this document as the canonical case-study working file for every design
partner from kickoff onward. It is intentionally not just a polished marketing
outline. It is the evidence ledger, operating log, and final packaging shell in
one place so we do not have to reconstruct proof after the fact.

## How To Use This Template

1. Duplicate this file at partner kickoff and rename it with the partner
   codename or approved company name.
2. Fill the "Partner Snapshot" and "Workflow Under Proof" sections before the
   first live run.
3. Update the "Evidence Log," "Metrics Scorecard," and "Timeline" sections
   after every meaningful milestone or weekly check-in.
4. Treat every external-facing claim as invalid until it is backed by a linked
   artifact, quote, metric source, or receipt.
5. If a claim is directionally true but not yet proven, mark it `Unverified`
   instead of smoothing over the gap.

## Evidence Rules

- Every quantified claim must name the source of truth.
- Every qualitative claim should include a dated quote, call note, or email.
- Every workflow claim should point to at least one concrete artifact:
  receipt, PR, decision log, inbox batch, spec, or screenshot.
- Preserve failures, reversals, and stop conditions. A case study that only
  records wins is not trustworthy sales proof.
- Do not publish company name, logos, or quotes until approval is explicit.

## Document Control

| Field | Value |
|---|---|
| Partner codename | [partner-codename] |
| Approved company name | [company-name or TBD] |
| Industry / segment | [regulated SaaS / fintech / healthtech / platform team / other] |
| Primary workflow | [workflow name] |
| Internal owner | [Aragora owner] |
| Partner champion | [name, role] |
| Kickoff date | [YYYY-MM-DD] |
| Current phase | [kickoff / onboarding / live proof / expansion / published / paused] |
| Publication status | [internal only / anonymized allowed / named allowed] |
| Last updated | [YYYY-MM-DD] |

## 1. Executive Summary

Write this section only after enough evidence exists to support it.

- One-sentence partner situation: [what was broken before Aragora]
- One-sentence Aragora wedge: [what bounded workflow Aragora handled]
- One-sentence outcome: [what measurably improved and over what window]
- Current confidence level: [Anecdotal / Early signal / Strong internal proof / Publishable]

## 2. Partner Snapshot

### Team Context

- Company stage: [seed / growth / enterprise / internal platform team]
- Team size relevant to workflow: [number]
- Compliance or risk context: [none / SOC 2 / HIPAA / EU AI Act / internal audit / other]
- Existing tools in the loop: [Slack, GitHub, Gmail, ticketing, docs, internal systems]
- Why they said yes to a design partnership: [pain + urgency]

### Pain Before Aragora

- Primary bottleneck: [review latency / triage overload / audit evidence gap / bounded backlog throughput / other]
- Frequency of the workflow: [times per day/week]
- What failure looked like before: [missed issue, manual burden, slow approval, no audit trail]
- Why the incumbent process was not enough: [single-model trust gap, fragmented evidence, too much manual review]

## 3. Workflow Under Proof

### Workflow Definition

| Field | Value |
|---|---|
| Workflow name | [short name] |
| Trigger | [what causes a run] |
| Input artifact | [PR / inbox batch / design doc / backlog slice / other] |
| Human decision boundary | [what human still approves] |
| Aragora action boundary | [what Aragora does autonomously] |
| Stop condition | [when Aragora must halt truthfully] |
| Failure cost if wrong | [low / medium / high + note] |

### Success Contract

| Metric | Baseline | Target | Measurement method | Source of truth |
|---|---|---|---|---|
| Time to first vetted outcome | [value] | [value] | [how measured] | [link or system] |
| Manual touches per workflow | [value] | [value] | [how measured] | [link or system] |
| Receipts captured per workflow | [value] | [value] | [how measured] | [link or system] |
| Rework / escalation rate | [value] | [value] | [how measured] | [link or system] |
| Trust signal | [value] | [value] | [survey / quote / adoption] | [link or note] |

### Proof Surface

Mark the primary wedge being validated:

- [ ] Decision review
- [ ] Autonomous repo improvement
- [ ] Inbox trust wedge
- [ ] Other bounded workflow: [name]

## 4. Baseline Before Aragora

### Previous Process

Describe the pre-Aragora workflow in 5-8 lines. Include who touched the work,
where evidence lived, how long it usually took, and where trust broke down.

### Baseline Evidence

| Evidence type | Date | Summary | Link / location | Confidence |
|---|---|---|---|---|
| Call note | [YYYY-MM-DD] | [summary] | [link] | [high/medium/low] |
| Existing KPI export | [YYYY-MM-DD] | [summary] | [link] | [high/medium/low] |
| Email / Slack quote | [YYYY-MM-DD] | [summary] | [link] | [high/medium/low] |
| Screenshot / dashboard | [YYYY-MM-DD] | [summary] | [link] | [high/medium/low] |

## 5. Deployment Timeline

| Date | Milestone | What happened | Evidence link | Notes |
|---|---|---|---|---|
| [YYYY-MM-DD] | Kickoff | [scope agreed] | [link] | [note] |
| [YYYY-MM-DD] | First live run | [what ran] | [link] | [note] |
| [YYYY-MM-DD] | First trusted outcome | [what changed] | [link] | [note] |
| [YYYY-MM-DD] | Expansion / rollback | [what changed] | [link] | [note] |
| [YYYY-MM-DD] | Publication decision | [approved / anonymized / deferred] | [link] | [note] |

## 6. Evidence Log

This is the core section. Add rows continuously. Each row should support a
claim we may want to make later.

| Date | Claim or observation | Artifact type | Link / location | Supports which future claim? | Confidence | Public-safe? |
|---|---|---|---|---|---|---|
| [YYYY-MM-DD] | [example: first receipt used in real approval path] | [receipt / PR / transcript / metric / screenshot / quote] | [link] | [example: Aragora entered the real workflow] | [high/medium/low] | [yes/no] |
| [YYYY-MM-DD] | [example: partner overruled recommendation due to missing context] | [call note / transcript] | [link] | [example: truthfully bounded autonomy] | [high/medium/low] | [yes/no] |
| [YYYY-MM-DD] | [example: review cycle dropped from 2 days to 45 min] | [metric export] | [link] | [example: latency improvement] | [high/medium/low] | [yes/no] |

## 7. Metrics Scorecard

Update this table at a fixed cadence, ideally weekly.

| Metric | Baseline | Current | Delta | Observation window | Source of truth | Notes |
|---|---|---|---|---|---|---|
| Time to vetted decision | [value] | [value] | [value] | [date range] | [link] | [note] |
| Manual review steps | [value] | [value] | [value] | [date range] | [link] | [note] |
| Workflows completed with receipts | [value] | [value] | [value] | [date range] | [link] | [note] |
| False positives / bad recommendations | [value] | [value] | [value] | [date range] | [link] | [note] |
| Human overrides | [value] | [value] | [value] | [date range] | [link] | [note] |
| Operator confidence / trust score | [value] | [value] | [value] | [date range] | [survey or note] | [note] |

## 8. What Aragora Actually Did

Describe the live workflow, not the aspirational one.

- Models or agent types used: [list]
- Trigger and run frequency: [detail]
- Human checkpoints: [detail]
- Output artifacts produced: [receipts / comments / labels / PRs / summaries / other]
- Where results were visible: [dashboard / API / Slack / GitHub / inbox / other]
- What was still manual: [detail]
- What Aragora stopped on truthfully: [detail]

## 9. Trust, Dissent, And Failure Notes

This section is mandatory. It prevents the case study from collapsing into
marketing theater.

### Notable Successes

- [example: partner adopted receipt as default review artifact]
- [example: dissent surfaced a risk the initial recommendation missed]

### Notable Failures Or Stops

- [example: missing context caused a false escalation]
- [example: integration gap forced manual fallback]
- [example: workflow remained too high-risk for autonomous action]

### What We Learned

- [product insight]
- [go-to-market insight]
- [scope boundary insight]

## 10. Quotes

Only include quotes with source and approval status.

| Date | Speaker | Quote | Source | Approval status | Public-safe? |
|---|---|---|---|---|---|
| [YYYY-MM-DD] | [name, role] | "[quote]" | [call / email / Slack / interview] | [pending / approved / rejected] | [yes/no] |

## 11. Draft Case Study Narrative

Fill this section only when the evidence above is strong enough.

### Headline

[One-line measurable transformation]

### Subhead

[Who the partner is, what workflow Aragora handled, and what changed]

### Before Aragora

[3-5 lines]

### Why Aragora Won This Workflow

[3-5 lines on receipts, dissent, bounded autonomy, or trust surface]

### Measured Outcomes

- [outcome with metric and timeframe]
- [outcome with metric and timeframe]
- [qualitative outcome with quote or evidence]

### Truthful Boundaries

- [what Aragora still does not automate here]
- [what still requires a human]
- [what evidence is strong vs early]

## 12. Publication Checklist

- [ ] Every material claim links to evidence in this document
- [ ] Metrics are reproducible from a named source of truth
- [ ] At least one failure, stop, or boundary is documented
- [ ] Quote approvals are explicit
- [ ] Naming / logo permissions are explicit
- [ ] The workflow described is the real live workflow, not roadmap fiction
- [ ] The final story matches the strongest evidence, not the most flattering framing

## Appendix A: Artifact Index

List every raw artifact we may need later.

| Artifact | Type | Date | Owner | Link / location | Included in final story? |
|---|---|---|---|---|---|
| [artifact name] | [receipt / transcript / export / screenshot / PR / email] | [YYYY-MM-DD] | [owner] | [link] | [yes/no] |

## Appendix B: Open Questions

- [What still needs to be measured?]
- [What approval is still missing?]
- [What claim do we suspect is true but cannot yet support?]
