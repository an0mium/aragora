# Product-Market Fit (PMF) Scorecard

Last updated: 2026-02-12

This scorecard is the single rubric for:
- selecting design partners,
- measuring pilot success,
- deciding whether to scale, iterate, or narrow scope.

Related:
- Design partner program: `docs/status/DESIGN_PARTNER_PROGRAM.md`
- SME wedge: `docs/status/SME_STARTER_PACK.md`
- Q1 execution backlog: `docs/status/BACKLOG_Q1_2026.md`
- Canonical execution order (stability gates): `docs/status/NEXT_STEPS_CANONICAL.md`

---

## How To Use

1. Score every prospect after the first discovery call.
2. Re-score after the "magic moment" session.
3. Re-score weekly during the pilot.
4. Only scale GTM when you have 3+ partners scoring above the "Scale" threshold for 3 consecutive weeks.

Scoring: 0-4 per dimension, weighted to a 0-100 total.

---

## Scoring Dimensions (0-4)

### 1) ICP Fit (weight 15)
0: hobby/individual, no meaningful decision accountability
1: small team, low stakes, no compliance
2: some accountability, occasional high-stakes decisions
3: clear accountable decision owner + recurring decisions + compliance pressure
4: regulated/high-stakes org where audit trails materially matter

### 2) Pain & Urgency (weight 15)
0: "nice to have"
1: occasional annoyance
2: real cost, but no urgency
3: recent incident or active initiative with real risk
4: high urgency (audit deadline, incident, executive mandate)

### 3) Activation (weight 10)
Definition: time-to-first-value (first usable receipt).
0: cannot activate due to tooling/security constraints
1: >2 hours with hands-on support
2: 30-120 minutes with hands-on support
3: 15-30 minutes mostly self-serve
4: <15 minutes self-serve ("SME Starter Pack" target)

### 4) "Magic Moment" Value (weight 20)
0: output is ignored or distrusted
1: output is interesting but not actionable
2: output leads to minor action; limited internal sharing
3: output changes a decision or escalates correctly; shared with the team
4: output prevents a likely failure or materially changes a high-stakes decision; shared with leadership or compliance

### 5) Repeatability / Retention (weight 15)
0: one-off usage only
1: sporadic; not tied to a workflow trigger
2: repeats monthly or requires heavy prompting
3: weekly use in a defined workflow trigger
4: multiple workflows + >1 internal user running it (not just the champion)

### 6) Willingness To Pay (weight 15)
0: no budget / refuses paid pilot
1: would pay only token amounts
2: willing to pay if procurement is easy and ROI is clear
3: willing to pay for a paid pilot or annual contract discussion
4: clear budget owner + timeline to purchase + strong ROI narrative

### 7) Procurement / Security Friction (weight 10)
Score is inverted: higher is better (less friction).
0: cannot use due to policy
1: requires long vendor onboarding before any pilot
2: heavy review but self-host allows pilot
3: standard security review; pilot can start now
4: can start immediately

---

## Total Score Calculation (0-100)

For each dimension:
- raw score = 0-4
- weighted = (raw / 4) * weight

Total = sum(weighted).

Thresholds:
- 80-100: SCALE (add sales-assist + publish case study)
- 65-79: ITERATE (ship top blockers, keep partner engaged)
- <65: NARROW (change wedge, change ICP, or stop pursuing)

---

## Measurement Hooks (Where To Instrument)

This scorecard should be grounded in data when possible.

Suggested instrumentation sources:
- Onboarding funnel / time-to-first-receipt:
  - Server: `aragora/server/handlers/onboarding.py` (flow + analytics endpoints)
  - UI: `aragora/live/src/components/onboarding/`
- Receipt generation + export:
  - `aragora/export/decision_receipt.py`
  - `aragora/gauntlet/receipt_models.py`
  - CLI receipt tooling: `aragora/cli/commands/receipt.py`, `aragora/cli/commands/verify.py`
- Integrations setup:
  - Wizard API: `aragora/server/handlers/oauth_wizard.py`
  - Slack/Gmail/Drive connectors:
    - `aragora/connectors/chat/slack/`
    - `aragora/connectors/enterprise/communication/gmail/`
    - `aragora/connectors/enterprise/documents/gdrive.py`
- Usage/cost/budget:
  - Budgets + enforcement: `aragora/control_plane/cost_enforcement.py`
  - Pipeline budget tracking: `aragora/pipeline/decision_plan/core.py`
  - Analytics: `aragora/analytics/dashboard.py`, `aragora/analytics/debate_analytics.py`
- Auditability:
  - `aragora/audit/log.py`, `aragora/audit/unified.py`

---

## Scorecard Template (Copy Per Partner)

Partner:
- Company:
- Segment: (FinTech / HealthTech / Enterprise SaaS / Other)
- Deployment: (Hosted / Self-hosted)
- Champion:
- Economic buyer:
- Security approver:

Workflow:
- Primary workflow trigger:
- Artifact types:
- Decision owner:
- Frequency:

Scores:
| Dimension | Raw (0-4) | Notes |
|---|---:|---|
| ICP Fit |  |  |
| Pain & Urgency |  |  |
| Activation |  |  |
| Magic Moment Value |  |  |
| Repeatability / Retention |  |  |
| Willingness To Pay |  |  |
| Procurement / Security Friction |  |  |

Pilot outcomes (weekly):
- Receipts generated (count):
- Time to first receipt (p50/p95):
- Weekly active users:
- Decisions changed / escalations avoided:
- Procurement status:

