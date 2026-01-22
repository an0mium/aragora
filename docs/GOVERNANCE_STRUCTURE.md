# Aragora Governance Structure

**SOC 2 Control:** CC1.1, CC1.2

This document outlines the organizational governance structure, decision-making authority, and accountability framework for Aragora.

## Organizational Structure

### Executive Leadership

| Role | Responsibilities |
|------|------------------|
| CEO/Founder | Strategic direction, investor relations, final authority |
| CTO | Technical architecture, engineering decisions, security oversight |
| Head of Product | Product roadmap, customer requirements, feature prioritization |
| Head of Operations | Infrastructure, SLAs, incident response, compliance |

### Engineering Teams

| Team | Focus Area | Decision Authority |
|------|------------|-------------------|
| Platform | Core debate engine, API | Technical design, performance |
| Infrastructure | DevOps, security, reliability | Deployment, monitoring, SRE |
| Frontend | Live dashboard, user experience | UI/UX decisions |
| AI/ML | Agent integration, RLM, knowledge | Model selection, training |

## Decision-Making Authority

### Technical Decisions

| Decision Type | Authority | Escalation Path |
|---------------|-----------|-----------------|
| Code changes (non-breaking) | Any engineer (PR approval) | Tech lead |
| Breaking API changes | Tech lead + CTO | CEO if customer impact |
| Security-related changes | Security review required | CTO |
| Infrastructure changes | SRE team | Head of Operations |
| Data schema changes | Tech lead + affected teams | CTO |

### Business Decisions

| Decision Type | Authority | Escalation Path |
|---------------|-----------|-----------------|
| Pricing changes | Head of Product + CEO | CEO |
| Customer SLA exceptions | Head of Operations | CEO |
| Vendor selection | Relevant team lead | CTO or CEO if > $10k/year |
| Partnership agreements | CEO | Board (if applicable) |

### Security Decisions

| Decision Type | Authority | Required Review |
|---------------|-----------|-----------------|
| Security policy changes | CTO | Security team review |
| Access grant (production) | SRE on-call + manager | Audit logged |
| Incident response | Incident commander | Post-incident review |
| Compliance exceptions | CTO + CEO | Documented justification |
| Penetration test authorization | CTO | Legal review |

## Escalation Procedures

### Severity Levels

| Level | Definition | Response | Escalation |
|-------|------------|----------|------------|
| P1 - Critical | Service outage, data breach | 15 min response | Immediate to CTO/CEO |
| P2 - High | Major feature down, security vuln | 1 hour response | To tech lead within 2 hours |
| P3 - Medium | Minor feature issue | 4 hour response | Normal triage |
| P4 - Low | Cosmetic, documentation | Next business day | None required |

### Escalation Matrix

```
Engineer → Tech Lead → CTO → CEO
    ↓           ↓        ↓
On-call → Ops Lead → Head of Ops
```

## Change Control Board

### Composition

- CTO (Chair)
- Tech Lead from each team
- Security representative
- QA representative

### Meeting Cadence

- **Weekly:** Review pending changes, sprint planning
- **Ad-hoc:** Emergency changes, security patches
- **Monthly:** Architecture review, tech debt prioritization

### Change Categories

| Category | Review Required | Approval Authority |
|----------|-----------------|-------------------|
| Standard | Automated tests pass | Tech lead |
| Normal | Code review + tests | Change Control Board |
| Emergency | Post-implementation review | CTO + on-call lead |
| Major | Full impact assessment | Change Control Board + CEO |

## Audit and Compliance

### Internal Audit

- **Quarterly:** Access review, permission audit
- **Monthly:** Security scanning results review
- **Weekly:** Vulnerability triage

### External Audit

- **Annual:** SOC 2 Type II audit
- **As needed:** Customer security assessments
- **Bi-annual:** Penetration testing

### Reporting Structure

| Report | Audience | Frequency |
|--------|----------|-----------|
| Security dashboard | Leadership | Daily (automated) |
| Incident report | Affected parties | Per incident |
| Compliance status | Leadership + auditors | Monthly |
| Risk register | Board/investors | Quarterly |

## Policy Ownership

| Policy | Owner | Review Cycle |
|--------|-------|--------------|
| Security Policy | CTO | Annual |
| Privacy Policy | Head of Operations | Annual |
| Acceptable Use | CTO | Annual |
| Incident Response | Head of Operations | Semi-annual |
| Data Classification | CTO | Annual |
| Vendor Management | Head of Operations | Annual |

## Employee Responsibilities

### All Employees

- Complete security awareness training within 30 days of hire
- Acknowledge security policies annually
- Report security incidents immediately
- Follow data classification guidelines
- Use approved tools and services only

### Managers

- Ensure team compliance with policies
- Conduct access reviews for team members
- Participate in incident response as needed
- Approve access requests for reports

### Security Team

- Maintain security controls and monitoring
- Conduct security reviews of changes
- Manage vulnerability response
- Lead incident response
- Maintain compliance documentation

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-21 | Operations | Initial version |

---

*Last updated: January 2026*
*Document owner: Head of Operations*
*Review cycle: Annual*
