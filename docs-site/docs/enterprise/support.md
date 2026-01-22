---
title: Enterprise Support Guide
description: Enterprise Support Guide
---

# Enterprise Support Guide

**Version:** 1.0.0
**Effective Date:** January 2026
**Last Updated:** January 14, 2026

---

## Overview

This document defines the enterprise support offering for Aragora, including support tiers, response times, escalation procedures, and engagement models.

---

## Support Tiers

### Free Tier

| Attribute | Value |
|-----------|-------|
| **Price** | $0/month |
| **Support Channels** | Community forums, GitHub issues |
| **Response Time** | Best effort |
| **Coverage Hours** | N/A |
| **Dedicated Support** | No |
| **SLA** | None |

### Pro Tier

| Attribute | Value |
|-----------|-------|
| **Price** | $99/month |
| **Support Channels** | Email, GitHub issues |
| **Response Time** | See matrix below |
| **Coverage Hours** | Business hours (9am-6pm PT, Mon-Fri) |
| **Dedicated Support** | No |
| **SLA** | 99.5% uptime |

### Enterprise Tier

| Attribute | Value |
|-----------|-------|
| **Price** | Custom (contact sales) |
| **Support Channels** | Email, Slack, phone, video |
| **Response Time** | See matrix below |
| **Coverage Hours** | 24/7/365 |
| **Dedicated Support** | Yes - named account manager |
| **SLA** | 99.9% uptime with credits |

---

## Response Time Matrix

### Severity Definitions

| Severity | Definition | Examples |
|----------|------------|----------|
| **P1 - Critical** | Service completely unavailable; business operations stopped | API returning 5xx for all requests; data loss |
| **P2 - High** | Major feature unavailable; significant business impact | Debates not completing; authentication broken |
| **P3 - Medium** | Feature degraded; workaround available | Slow response times; intermittent errors |
| **P4 - Low** | Minor issue; cosmetic or documentation | Typo in UI; unclear error message |

### Response Times

| Severity | Pro Tier | Enterprise Tier |
|----------|----------|-----------------|
| P1 - Critical | 4 hours | 1 hour |
| P2 - High | 8 hours | 2 hours |
| P3 - Medium | 24 hours | 4 hours |
| P4 - Low | 72 hours | 24 hours |

### Resolution Targets

| Severity | Pro Tier | Enterprise Tier |
|----------|----------|-----------------|
| P1 - Critical | 24 hours | 4 hours |
| P2 - High | 72 hours | 24 hours |
| P3 - Medium | 1 week | 72 hours |
| P4 - Low | Best effort | 1 week |

---

## Contact Channels

### Email Support

- **Pro Tier**: support@aragora.ai
- **Enterprise Tier**: enterprise-support@aragora.ai (dedicated queue)

**Email Format for Faster Response:**

```
Subject: [P1/P2/P3/P4] Brief description of issue

Organization: Your Company Name
Account ID: ara_xxx...
Environment: Production / Staging
Affected Users: Number or percentage

Issue Description:
- What happened
- When it started
- Steps to reproduce

Business Impact:
- How this affects operations

Already Tried:
- List troubleshooting steps taken
```

### Slack Connect (Enterprise Only)

Enterprise customers receive a dedicated Slack Connect channel with:
- Direct access to support engineers
- Real-time incident communication
- Proactive notifications for planned maintenance

### Phone Support (Enterprise Only)

- **Emergency Hotline**: +1-XXX-XXX-XXXX
- **Hours**: 24/7 for P1/P2 issues
- **Use for**: Critical production issues requiring immediate human intervention

### Video Calls (Enterprise Only)

- Scheduled via account manager
- Used for: Architecture reviews, onboarding, quarterly business reviews

---

## Escalation Procedures

### Pro Tier Escalation

```
Level 1: Support ticket (0-4 hours)
    ↓ No resolution
Level 2: Senior engineer (4-8 hours)
    ↓ No resolution
Level 3: Engineering lead (8-24 hours)
```

### Enterprise Tier Escalation

```
Level 1: Dedicated support engineer (0-1 hour)
    ↓ No resolution
Level 2: Account manager + senior engineer (1-2 hours)
    ↓ No resolution
Level 3: Engineering leadership (2-4 hours)
    ↓ No resolution
Level 4: Executive escalation (4+ hours)
```

### Escalation Contacts

| Level | Contact | Availability |
|-------|---------|--------------|
| L1 | Support team | Business hours (Pro) / 24/7 (Enterprise) |
| L2 | Senior engineers | Business hours (Pro) / 24/7 (Enterprise) |
| L3 | VP Engineering | 24/7 for P1 |
| L4 | CEO/CTO | 24/7 for P1 |

### When to Escalate

Escalate immediately if:
- P1 issue not acknowledged within 30 minutes (Enterprise)
- P1 issue not resolved within target time
- Multiple related issues indicate systemic problem
- Customer explicitly requests escalation

---

## Enterprise Onboarding

### Week 1: Kickoff

| Day | Activity | Participants |
|-----|----------|--------------|
| 1 | Welcome call, introductions | Account manager, customer team |
| 2-3 | Environment setup assistance | Support engineer |
| 4-5 | Integration review | Technical lead |

### Week 2: Integration

| Day | Activity | Participants |
|-----|----------|--------------|
| 1-2 | API integration support | Support engineer |
| 3-4 | SSO/SAML configuration | Support engineer, customer IT |
| 5 | Initial testing and validation | Joint team |

### Week 3-4: Go-Live

| Day | Activity | Participants |
|-----|----------|--------------|
| 1-5 | Pilot rollout | Support engineer on standby |
| 6-10 | Full production deployment | Joint team |

### Ongoing Support

- **Weekly check-ins** (first month)
- **Bi-weekly check-ins** (months 2-3)
- **Monthly check-ins** (ongoing)
- **Quarterly business reviews** (QBR)

---

## Quarterly Business Reviews (QBR)

### Agenda Template

1. **Usage Review** (15 min)
   - Debates conducted
   - Token consumption
   - API call volume

2. **Performance Metrics** (15 min)
   - Uptime and SLA compliance
   - Response time percentiles
   - Incident summary

3. **Support Review** (10 min)
   - Ticket volume and resolution times
   - Common issues and resolutions
   - Feedback and improvements

4. **Roadmap Preview** (10 min)
   - Upcoming features
   - Planned maintenance
   - Beta opportunities

5. **Action Items** (10 min)
   - Open issues
   - Customer requests
   - Next steps

---

## Included Services

### Pro Tier

- Email support during business hours
- Access to documentation and knowledge base
- Monthly usage reports
- Standard API rate limits

### Enterprise Tier

All Pro features plus:

| Service | Description |
|---------|-------------|
| **Dedicated Account Manager** | Named contact for all needs |
| **24/7 Support** | Round-the-clock coverage |
| **Slack Connect** | Real-time communication channel |
| **Phone Support** | Emergency hotline access |
| **Priority Queue** | Tickets routed to senior engineers |
| **Custom SLA** | Negotiated uptime and response times |
| **Quarterly Reviews** | Strategic planning sessions |
| **Early Access** | Beta features and roadmap input |
| **Custom Integrations** | Professional services available |
| **Training Sessions** | Team onboarding and best practices |
| **Architecture Review** | Annual security and performance review |

---

## Professional Services

Available for Enterprise customers at additional cost:

| Service | Description | Typical Duration |
|---------|-------------|------------------|
| **Custom Integration** | Build integrations with your systems | 2-4 weeks |
| **Custom Personas** | Develop domain-specific agents | 1-2 weeks |
| **On-Site Training** | In-person team training | 1-2 days |
| **Architecture Review** | Deep-dive system assessment | 1 week |
| **Performance Tuning** | Optimize for your workload | 1-2 weeks |
| **Compliance Mapping** | Map to your compliance framework | 2-4 weeks |

Contact your account manager for scoping and pricing.

---

## Support Boundaries

### In Scope

- Aragora platform functionality
- API and SDK usage
- Integration guidance
- Configuration assistance
- Performance troubleshooting
- Security questions
- Billing and account issues

### Out of Scope

- Custom code development (see Professional Services)
- Third-party system debugging
- AI model training or fine-tuning
- General programming questions
- Infrastructure management (customer-managed)

---

## Feedback and Improvement

### Providing Feedback

- **Support ticket ratings**: After each ticket closure
- **NPS surveys**: Quarterly satisfaction surveys
- **QBR discussions**: Strategic feedback during reviews
- **Feature requests**: Via support ticket or account manager

### Continuous Improvement

We track:
- First response time
- Time to resolution
- Customer satisfaction (CSAT)
- Net Promoter Score (NPS)
- Ticket reopen rate

Monthly reviews ensure we meet our commitments.

---

## Contact Information

| Channel | Contact |
|---------|---------|
| **General Support** | support@aragora.ai |
| **Enterprise Support** | enterprise-support@aragora.ai |
| **Sales** | sales@aragora.ai |
| **Security Issues** | security@aragora.ai |
| **Status Page** | status.aragora.ai |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-14 | Initial release |
