# Design Partner Program (Q1 2026)

Last updated: 2026-02-12

This document defines the design partner program for achieving product-market fit (PMF) for Aragora's **Decision Integrity** wedge (adversarial validation + receipts) with the **SME Starter Pack** as the fastest path to value.

Related:
- Canonical execution order: `docs/status/NEXT_STEPS_CANONICAL.md`
- 12-week backlog: `docs/status/BACKLOG_Q1_2026.md`
- SME wedge spec: `docs/status/SME_STARTER_PACK.md`
- Positioning: `docs/status/COMMERCIAL_POSITIONING.md`, `docs/status/COMPETITIVE_POSITIONING.md`
- PMF scoring rubric: `docs/status/PMF_SCORECARD.md`

---

## Program Goals

### Outcomes (12 weeks)
- 3-5 active design partners running real decisions through Aragora weekly.
- 2 publishable case studies (anonymized if required) with quantified ROI.
- Clear conversion path: paid pilot, LOI, or security/procurement green light.
- Product clarity: one "default path" from onboarding to first recurring workflow.

### Non-Goals (for this program)
- Broad connector expansion beyond the SME starter set (Slack, Gmail, Drive, Outlook).
- Building a general "marketplace" or broad workflow engine surface area unless it directly reduces onboarding time or increases retention for partners.

---

## Ideal Design Partner Profile (ICP)

Use the ICP checklist in `docs/status/POSITIONING.md` as the base.

Minimum criteria:
- Has compliance requirements (SOC 2, HIPAA, GDPR, internal audit) OR a high cost of failure for technical/security decisions.
- Ships software at least monthly and has recurring design/review rituals.
- 20+ engineers OR a security/compliance function that signs off on releases.
- Already experimenting with AI/LLMs but does not trust single-model outputs.

Best-fit segments:
- FinTech (risk, audit trail, change control)
- HealthTech (HIPAA + data access controls)
- Enterprise SaaS (SOC 2, multi-tenant risk, security review bottlenecks)

Target roles:
- Primary: CTO, VP Engineering, Head of Platform, Head of AI Governance
- Secondary: CISO, Security Engineering lead, GRC lead, DPO

---

## Program Structure

### Partner Commitments
- 1 kickoff (60-90 min)
- 1 guided "magic moment" session (60 min)
- Weekly check-in (30 min) for 4-6 weeks
- Provide 2-4 real artifacts (sanitized if needed):
  - architecture or change proposal (markdown)
  - policy or compliance requirement
  - incident postmortem / "we should have caught this" example
  - one real PR or release plan to run through `aragora review` / Gauntlet

### What Aragora Commits To
- Hands-on onboarding support through first receipt and first recurring workflow.
- Weekly iteration loop: capture friction, fix the top 1-2 issues, re-validate.
- A decision receipt package that is:
  - shareable (Slack/email)
  - exportable (PDF/MD)
  - verifiable (receipt integrity checks)

### Confidentiality & Data Handling
- Default to sanitized artifacts.
- If full artifacts are required, use self-hosted deployment and restrict logs.
- Document data boundaries in a one-page "pilot data policy" before kickoff.

---

## Timeline (Calendar Dates)

Current date: 2026-02-12.

Recommended cadence:
- Week of 2026-02-16: outreach + first discovery calls
- Week of 2026-02-23: select 3-5 design partners + kickoff
- Weeks of 2026-03-02 through 2026-04-06: weekly loops + conversions
- Week of 2026-04-13: case study drafting + pricing/procurement closure

---

## Interview Scripts

### Script 1: Discovery Call (45 minutes)

Goal: confirm ICP fit, quantify pain, and identify the first workflow to pilot.

1. Context
   - "What decisions in your org can hurt you if they're wrong?"
   - "Who signs off on these decisions today?"
   - "How often do you make them (weekly/monthly)?"

2. Current workflow
   - "Walk me through the last time you made one of these decisions."
   - "What artifacts exist (docs, PRs, tickets, incident reports)?"
   - "Where does it happen (Slack, docs, meetings, email)?"

3. Failure modes and urgency
   - "When was the last 'we should have caught this' incident?"
   - "What was the cost (time, money, customer impact, audit impact)?"
   - "What changed recently that makes this urgent now?"

4. Existing tooling and constraints
   - "What are you using today (GRC tools, scanners, code review, LLM tools)?"
   - "Can you self-host? Any restrictions on cloud tools?"
   - "Do you require SSO, audit logging, specific data retention?"

5. Success criteria
   - "If Aragora works, what changes in 30 days?"
   - "What would you be willing to pay to make this workflow 10x faster or safer?"

Capture:
- 1 primary workflow for the pilot (e.g., "architecture proposal review", "security change approval", "policy drafting").
- 1-2 artifacts to run through Gauntlet in the demo session.
- Buyer map: champion, economic buyer, security approver.

### Script 2: "Magic Moment" Demo Session (60 minutes)

Goal: run a real artifact end-to-end and produce a receipt the partner can share internally.

Agenda:
- 10 min: confirm workflow, artifact, and the decision they need to make.
- 25 min: run Gauntlet or multi-agent review live (streaming if available).
- 15 min: review findings, dissent, confidence, and what changed their mind.
- 10 min: export + share receipt to their channel (Slack/email) and define next run.

Key questions:
- "Which finding would have surprised your team?"
- "What would you do differently based on this output?"
- "Who else needs to see this receipt for it to matter?"

Artifacts to produce:
- Receipt export (PDF + MD) attached or posted to their chosen channel.
- One screenshot of the receipt summary suitable for a case study (optional).

### Script 3: Pilot Kickoff (60 minutes)

Goal: define pilot scope, measures, and cadence.

1. Pick the first recurring workflow
   - trigger: "new architecture proposal", "release candidate", "policy update"
   - frequency: weekly is ideal
   - owner: one person who runs it every time

2. Define integration + deployment mode
   - hosted vs self-hosted (prefer self-hosted for regulated partners)
   - Slack/Gmail/Drive integration choices

3. Define "success in 4 weeks"
   - quantitative: time saved, fewer escalations, fewer rework loops
   - qualitative: confidence, audit readiness, decision clarity

4. Agree on the weekly loop
   - 30 min: usage + friction review
   - 30 min: pick next improvements (product + docs + deployment)

---

## Program Artifacts (What We Maintain)

- A running score per partner using `docs/status/PMF_SCORECARD.md`.
- A list of top onboarding blockers and the exact reproduction path.
- A weekly changelog of pilot-driven fixes (link to PRs/commits if used).

