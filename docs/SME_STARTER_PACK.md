# SME Starter Pack

**Version:** 1.0
**Status:** Scope Definition
**Target:** First debate + decision receipt in <15 minutes

---

## Overview

The SME (Small/Medium Enterprise) Starter Pack provides a streamlined onboarding experience for teams of 5-50 users. It enables organizations to run their first AI-facilitated debate and receive a decision receipt within 15 minutes of signup.

## Target Audience

- **Company Size:** 5-50 employees
- **Decision Types:** Team decisions, project planning, vendor selection, policy changes
- **Technical Level:** Low to moderate (no engineering required)
- **Budget:** $99-499/month

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to first debate | <5 minutes | Onboarding funnel analytics |
| Time to first receipt | <15 minutes | End-to-end timing |
| Integration setup time | <10 minutes per connector | User testing |
| First-week retention | >60% | Cohort analysis |
| NPS score | >40 | Survey |

---

## Included Components

### 1. Guided Onboarding Wizard

A step-by-step wizard that walks users through:

1. **Workspace Creation** (2 min)
   - Organization name
   - Admin email verification
   - Workspace URL selection

2. **First Integration** (5 min)
   - Slack recommended (most common)
   - OAuth flow with permissions explanation
   - Test message to confirm connection

3. **First Debate** (5 min)
   - Template selection (8-12 pre-built templates)
   - Topic entry
   - Agent selection (auto-recommended)
   - Run debate

4. **Decision Receipt** (3 min)
   - View receipt in UI
   - Export options (PDF, Markdown)
   - Share to connected channel

### 2. Pre-Configured Integrations

| Integration | Status | Auth Method | Setup Time |
|-------------|--------|-------------|------------|
| Slack | Production-ready | OAuth 2.0 | 3 min |
| Gmail | Production-ready | OAuth 2.0 | 5 min |
| Google Drive | Production-ready | OAuth 2.0 | 5 min |
| Outlook | Production-ready | OAuth 2.0 | 5 min |

**Implementation References:**
- `aragora/connectors/chat/slack.py` (2,304 lines, circuit breaker)
- `aragora/connectors/enterprise/communication/gmail.py` (1,605 lines)
- `aragora/connectors/enterprise/documents/gdrive.py`
- `aragora/connectors/email/outlook_sync.py` (1,004 lines)

### 3. Workflow Templates Library

8-12 pre-built templates for common SME decisions:

| Category | Template | Agents | Rounds |
|----------|----------|--------|--------|
| **Team** | Hiring Decision | Claude, GPT-4 | 3 |
| **Team** | Performance Review | Claude, Gemini | 2 |
| **Project** | Feature Prioritization | Claude, GPT-4, Mistral | 3 |
| **Project** | Sprint Planning | Claude, GPT-4 | 2 |
| **Vendor** | Tool Selection | Claude, GPT-4, Gemini | 4 |
| **Vendor** | Contract Review | Claude, GPT-4 | 3 |
| **Policy** | Remote Work Policy | Claude, GPT-4, Gemini | 3 |
| **Policy** | Budget Allocation | Claude, GPT-4 | 2 |

### 4. Usage Dashboard

Real-time visibility into:

- **Debates:** Count, topics, participants
- **Spend:** API costs by model, budget remaining
- **Integrations:** Connected channels, message volume
- **Receipts:** Generated, exported, shared

### 5. Budget Controls

- **Workspace Caps:** Maximum monthly spend
- **Alerts:** 50%, 75%, 90% threshold notifications
- **Per-Debate Limits:** Optional cost ceiling per debate

---

## MVP Features (Sprint 1-2)

| Feature | Priority | Sprint |
|---------|----------|--------|
| Onboarding wizard | P0 | Sprint 2 |
| First debate flow | P0 | Sprint 2 |
| Decision receipt export | P0 | Sprint 2 |
| Slack integration wizard | P0 | Sprint 4 |
| Gmail/Drive integration wizard | P0 | Sprint 4 |
| Usage dashboard | P0 | Sprint 3 |
| Budget caps | P0 | Sprint 3 |
| Template library (8 templates) | P1 | Sprint 3 |

## Post-MVP Features (Sprint 5-6)

| Feature | Priority | Sprint |
|---------|----------|--------|
| Workspace admin UI | P0 | Sprint 5 |
| RBAC-lite (admin/member) | P0 | Sprint 5 |
| ROI dashboard | P0 | Sprint 6 |
| Template library (12 templates) | P1 | Sprint 6 |
| Audit log UI | P1 | Sprint 5 |
| User feedback collection | P1 | Sprint 6 |

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SME Starter Pack                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Onboarding │  │   Debate    │  │   Receipt   │         │
│  │    Wizard   │──│    Flow     │──│   Export    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                  │
│         ▼                ▼                ▼                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Integration Layer                        │   │
│  │  ┌───────┐  ┌───────┐  ┌───────┐  ┌────────┐       │   │
│  │  │ Slack │  │ Gmail │  │ Drive │  │Outlook │       │   │
│  │  └───────┘  └───────┘  └───────┘  └────────┘       │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Core Aragora Engine                      │   │
│  │  • Debate Orchestration (Arena)                      │   │
│  │  • Agent Pool (15+ models)                           │   │
│  │  • Receipt Generation (Gauntlet)                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Pricing Tiers (Proposed)

| Tier | Users | Debates/mo | Price |
|------|-------|------------|-------|
| Starter | 5 | 50 | $99/mo |
| Team | 15 | 200 | $249/mo |
| Business | 50 | Unlimited | $499/mo |

All tiers include:
- All 4 integrations
- All workflow templates
- Decision receipts
- Usage dashboard
- Email support

---

## Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| OAuth UI for integrations | Gap | Medium | Build unified OAuth wizard |
| Onboarding wizard UI | Gap | Low | Use existing Live app patterns |
| Budget controls backend | Exists | Low | Wire to billing module |
| Template library | Partial | Low | Document existing templates |

---

## Related Documents

- [BACKLOG_Q1_2026.md](BACKLOG_Q1_2026.md) - Sprint planning
- [INTEGRATION_AUDIT.md](INTEGRATION_AUDIT.md) - Connector assessment
- [ONBOARDING_FLOW.md](ONBOARDING_FLOW.md) - Detailed flow design

---

*Created: 2026-01-24*
*Next Review: Sprint 2 Planning*
