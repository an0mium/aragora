# Breach Notification SLA

**Effective Date:** January 14, 2026
**Last Updated:** January 14, 2026
**Version:** 1.0.0
**Owner:** Security Team

---

## Overview

This document defines Aragora's Service Level Agreement (SLA) for breach notifications, ensuring compliance with regulatory requirements and maintaining trust with our customers.

**SOC 2 Control:** CC7-04 - Incident notification procedures

---

## Definitions

| Term | Definition |
|------|------------|
| **Personal Data Breach** | Unauthorized access, disclosure, alteration, or destruction of personal data |
| **Data Subject** | An identified or identifiable natural person whose data is processed |
| **Supervisory Authority** | Government body responsible for data protection (e.g., ICO, CNIL, state AG) |
| **Discovery Time** | When Aragora becomes aware of a breach (T=0) |

---

## Notification Timeline Summary

| Stakeholder | Regulatory Requirement | Aragora SLA | Notes |
|-------------|------------------------|-------------|-------|
| EU/EEA Supervisory Authority | 72 hours (GDPR) | **48 hours** | Lead supervisory authority |
| EU/EEA Data Subjects | "Without undue delay" | **72 hours** | If high risk to rights/freedoms |
| UK ICO | 72 hours (UK GDPR) | **48 hours** | If UK users affected |
| US State AG (varies) | 30-60 days (varies) | **30 days** | See state-specific below |
| California AG | 72 hours (CCPA) | **48 hours** | If >500 CA residents |
| Enterprise Customers | Per contract | **24 hours** | Initial notification |
| All Affected Users | Varies | **7 days** | After assessment complete |

---

## Notification Process

### Phase 1: Internal Response (T+0 to T+4h)

```
[ ] 1. DETECTION & TRIAGE (T+0 to T+1h)
    - Confirm breach occurrence
    - Activate incident response team
    - Initial scope assessment
    - Determine data types affected

[ ] 2. CONTAINMENT (T+1h to T+2h)
    - Stop ongoing breach
    - Preserve evidence
    - Identify affected systems

[ ] 3. INITIAL ASSESSMENT (T+2h to T+4h)
    - Estimate users affected
    - Identify data categories exposed
    - Determine jurisdictions affected
    - Risk assessment (low/medium/high)
```

### Phase 2: Regulatory Notification (T+4h to T+48h)

```
[ ] 4. REGULATORY DETERMINATION (T+4h to T+8h)
    - Identify applicable regulations
    - Determine notification requirements
    - Prepare regulatory submissions
    - Legal review

[ ] 5. REGULATORY NOTIFICATION (T+8h to T+48h)
    - File with lead supervisory authority (EU)
    - Notify state attorneys general (US)
    - Document all submissions
```

### Phase 3: Stakeholder Notification (T+24h to T+7d)

```
[ ] 6. ENTERPRISE CUSTOMER NOTIFICATION (T+24h)
    - Send initial breach notice
    - Provide preliminary impact assessment
    - Schedule detailed briefing

[ ] 7. AFFECTED USER NOTIFICATION (T+48h to T+7d)
    - Prepare notification content
    - Segment by jurisdiction/risk
    - Send notifications
    - Provide remediation guidance
```

---

## Regulatory Requirements by Jurisdiction

### European Union (GDPR)

**Supervisory Authority:**
- **When:** Within 72 hours of becoming aware
- **Threshold:** Unless unlikely to result in risk to individuals
- **What to Include:**
  - Nature of breach
  - Categories and approximate number of data subjects
  - Contact details of DPO
  - Likely consequences
  - Measures taken or proposed

**Data Subjects:**
- **When:** Without undue delay if high risk
- **What to Include:**
  - Plain language description
  - Contact details
  - Likely consequences
  - Remedial measures
  - Advice on protective actions

### United Kingdom (UK GDPR)

- Same requirements as GDPR
- Report to ICO (Information Commissioner's Office)
- Portal: https://ico.org.uk/for-organisations/report-a-breach/

### California (CCPA/CPRA)

**Attorney General:**
- **When:** Within 72 hours
- **Threshold:** If >500 California residents affected
- **What to Include:**
  - Nature of breach
  - Types of personal information involved
  - Number of California residents affected

**Affected Individuals:**
- **When:** "In the most expedient time possible"
- **Method:** Written notice or electronic if consistent with E-SIGN Act

### Other US States

| State | Notification Timeline | AG Notification | Special Requirements |
|-------|----------------------|-----------------|---------------------|
| New York | Expeditiously | If >5,000 | Security program documentation |
| Texas | Within 60 days | If >250 | Third-party notice to credit bureaus |
| Florida | Within 30 days | If >500 | Identity theft prevention guidance |
| Massachusetts | As soon as practicable | All breaches | Breach type and remediation |
| Illinois | As soon as practicable | All breaches | Credit monitoring for SSN exposure |

---

## Notification Templates

### Template 1: Regulatory Authority Initial Notification

```
Subject: Personal Data Breach Notification - [Company Name]

Dear [Supervisory Authority],

Pursuant to [Article 33 GDPR / applicable law], [Company Name] is providing
notification of a personal data breach discovered on [date] at [time] UTC.

1. NATURE OF BREACH
[Description of what occurred]

2. DATA CATEGORIES AFFECTED
- [ ] Names
- [ ] Email addresses
- [ ] [Other categories]

3. APPROXIMATE NUMBER OF DATA SUBJECTS AFFECTED
[Number] individuals across [jurisdictions]

4. DATA PROTECTION OFFICER CONTACT
Name: [DPO Name]
Email: dpo@aragora.ai
Phone: [Phone]

5. LIKELY CONSEQUENCES
[Assessment of potential harm]

6. MEASURES TAKEN
[Containment and remediation steps]

7. MEASURES PROPOSED
[Ongoing remediation plan]

We will provide supplementary information as our investigation progresses.

Respectfully,
[Security Officer Name]
[Title]
[Company Name]
```

### Template 2: Affected Individual Notification

```
Subject: Important Security Notice - Action Required

Dear [Customer Name],

We are writing to inform you of a security incident that may have affected
your personal information.

WHAT HAPPENED
On [date], we discovered [brief description of incident]. Upon discovery, we
immediately [containment actions taken].

WHAT INFORMATION WAS INVOLVED
Based on our investigation, the following types of information may have been
affected:
- [Data type 1]
- [Data type 2]

WHAT WE ARE DOING
We have taken the following steps:
- [Action 1]
- [Action 2]
- [Action 3]

WHAT YOU CAN DO
We recommend that you:
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

FOR MORE INFORMATION
If you have questions, please contact us at:
- Email: security@aragora.ai
- Phone: [Support phone]
- Website: https://aragora.ai/security

We sincerely apologize for this incident and any inconvenience it may cause.

Sincerely,
[Executive Name]
[Title]
Aragora
```

### Template 3: Enterprise Customer Notification

```
Subject: Security Incident Notification - [Incident ID]

Dear [Customer Contact],

This notice is to inform you of a security incident affecting [service/data].

INCIDENT SUMMARY
- Incident ID: [ID]
- Discovery Date: [Date]
- Incident Type: [Type]
- Status: [Active/Contained/Resolved]

YOUR DATA IMPACT
[Specific assessment of customer's data exposure]

ARAGORA'S RESPONSE
[Actions taken and timeline]

NEXT STEPS
We will provide a detailed briefing on [date/time]. In the meantime, we
recommend:
- [Recommendation 1]
- [Recommendation 2]

CONTACTS
- Incident Hotline: [Phone]
- Security Team: security@aragora.ai
- Your Account Manager: [Name/Email]

We are committed to full transparency and will provide updates every
[24/48/72] hours until resolution.

Sincerely,
[Security Officer]
Aragora Security Team
```

---

## Stakeholder Notification Matrix

| Stakeholder | Initial Notice | Update Frequency | Final Report | Method |
|-------------|----------------|------------------|--------------|--------|
| Executive Team | T+1h | Every 4h | T+7d | Slack + Email |
| Legal | T+2h | Every 4h | T+7d | Email + Meeting |
| Board (P1 only) | T+24h | Daily | T+30d | Email + Call |
| Regulators | T+48h | As required | T+30d | Official submission |
| Enterprise Customers | T+24h | Daily | T+14d | Email + Portal |
| Affected Users | T+72h | Weekly | T+30d | Email |
| Media (if required) | T+72h | As needed | T+30d | Press release |

---

## Escalation Thresholds

### P1 - Critical (Board + Executive Notification)

- >10,000 users affected
- Financial data (payment cards, bank accounts) exposed
- Health information exposed
- Government/enterprise customer data
- Ongoing active attack
- Media attention likely

### P2 - High (Executive Notification)

- 1,000-10,000 users affected
- Authentication credentials exposed
- API keys or secrets compromised
- Single enterprise customer affected

### P3 - Medium (Security Team)

- <1,000 users affected
- Limited personal data exposure
- No financial or health data
- Contained within 24 hours

### P4 - Low (Document Only)

- <100 users affected
- Public information only
- No regulatory notification required

---

## Documentation Requirements

For each breach, maintain records of:

1. **Incident Log**
   - Timeline of events
   - Actions taken
   - Decisions made and rationale

2. **Notification Records**
   - All notifications sent
   - Recipient confirmations
   - Response correspondence

3. **Regulatory Submissions**
   - Copies of all filings
   - Acknowledgment receipts
   - Follow-up correspondence

4. **Post-Incident Review**
   - Root cause analysis
   - Lessons learned
   - Remediation plan

**Retention Period:** 7 years minimum (or as required by applicable law)

---

## Compliance Verification

### Quarterly Review

- [ ] Notification templates current
- [ ] Contact lists verified
- [ ] Regulatory requirements reviewed
- [ ] Team training completed
- [ ] Mock notification drill conducted

### Annual Review

- [ ] Full SLA review and update
- [ ] Legal review of requirements
- [ ] Process improvement implementation
- [ ] Metrics analysis and reporting

---

## Metrics and Reporting

Track and report monthly:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to Detection | <24h | From breach to discovery |
| Time to Containment | <4h | From discovery to containment |
| Regulatory Notification | <48h | From discovery to filing |
| User Notification | <7d | From discovery to notification |
| Documentation Complete | <30d | All records finalized |

---

## Contact Information

| Role | Contact | Availability |
|------|---------|--------------|
| Security Lead | security@aragora.ai | 24/7 on-call |
| DPO | dpo@aragora.ai | Business hours |
| Legal | legal@aragora.ai | Business hours |
| VP Engineering | [Phone] | 24/7 for P1 |
| CEO/CTO | [Phone] | 24/7 for P1 |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-14 | Initial release |
