---
title: Data Subject Access Request (DSAR) Workflow
description: Data Subject Access Request (DSAR) Workflow
---

# Data Subject Access Request (DSAR) Workflow

**Effective Date:** January 14, 2026
**Last Updated:** January 14, 2026
**Version:** 1.0.0
**Owner:** Privacy Team

---

## Overview

This document defines Aragora's workflow for handling Data Subject Access Requests (DSARs) in compliance with GDPR, CCPA, and other applicable privacy regulations. All requests must be processed within statutory deadlines while ensuring proper identity verification and data accuracy.

**SOC 2 Control:** P1-02 - Data subject rights procedures

---

## Request Types

| Request Type | GDPR Article | CCPA Section | Response Time |
|--------------|--------------|--------------|---------------|
| **Access** | Art. 15 | 1798.100 | 30 days |
| **Rectification** | Art. 16 | - | 30 days |
| **Erasure** | Art. 17 | 1798.105 | 30 days |
| **Portability** | Art. 20 | - | 30 days |
| **Opt-Out** | - | 1798.120 | 15 business days |
| **Restrict Processing** | Art. 18 | - | 30 days |
| **Object to Processing** | Art. 21 | - | 30 days |
| **Know** | - | 1798.110 | 45 days |
| **Delete** | - | 1798.105 | 45 days |

---

## Response Timeline Summary

| Regulation | Initial Response | Extension | Maximum Total |
|------------|------------------|-----------|---------------|
| GDPR | 30 days | +60 days (complex) | 90 days |
| UK GDPR | 30 days | +60 days (complex) | 90 days |
| CCPA/CPRA | 45 days | +45 days | 90 days |
| Virginia CDPA | 45 days | +45 days | 90 days |
| Colorado CPA | 45 days | +45 days | 90 days |

---

## Request Intake

### Authorized Channels

Requests are accepted through:

| Channel | Address | Response SLA |
|---------|---------|--------------|
| Email | privacy@aragora.ai | Acknowledge within 24h |
| Web Form | aragora.ai/privacy/request | Automated acknowledgment |
| Mail | Aragora Inc., [Address] | 3 business days |
| In-App | Settings > Privacy > My Data | Automated acknowledgment |

### Request Form Fields

**Required:**
- Full name
- Email address (account email)
- Request type
- Description of request

**Optional (may expedite):**
- Account ID
- Date range for data
- Specific data categories

---

## Workflow Stages

### Stage 1: Receipt & Acknowledgment (Day 0-1)

```
[ ] 1. LOG REQUEST
    - Assign unique DSAR ID: DSAR-YYYY-NNNN
    - Record submission channel
    - Record timestamp (UTC)
    - Categorize request type

[ ] 2. ACKNOWLEDGE RECEIPT
    - Send acknowledgment within 24 hours
    - Include DSAR ID for tracking
    - Provide expected timeline
    - Request additional info if needed

[ ] 3. PRELIMINARY REVIEW
    - Verify request is valid DSAR
    - Identify data subject jurisdiction
    - Determine applicable regulations
```

**Acknowledgment Template:**

```
Subject: DSAR Acknowledgment - [DSAR-YYYY-NNNN]

Dear [Name],

Thank you for your data subject request submitted on [date].

Request Details:
- Reference Number: DSAR-YYYY-NNNN
- Request Type: [Type]
- Received: [Date/Time]

Next Steps:
Before we can process your request, we need to verify your identity. Please
respond to this email with:
1. [Verification requirement 1]
2. [Verification requirement 2]

Timeline:
Once verified, we will respond within [30/45] days as required by [GDPR/CCPA].

Questions? Reply to this email or contact privacy@aragora.ai.

Regards,
Aragora Privacy Team
```

### Stage 2: Identity Verification (Day 1-5)

```
[ ] 4. VERIFY IDENTITY
    - Match request to existing account
    - For account holders: email verification + account access
    - For non-account holders: government ID + proof of address
    - Document verification method used

[ ] 5. RECORD VERIFICATION
    - Log verification timestamp
    - Store verification evidence (redacted ID)
    - Note any discrepancies
```

**Verification Methods:**

| Scenario | Method | Evidence Required |
|----------|--------|-------------------|
| Account holder | Email verification | Click link from registered email |
| Account holder | Account login | Session + MFA verification |
| Non-account holder | Document verification | Government ID + utility bill |
| Authorized agent | Written authorization | Power of attorney or signed form |

**Verification Checklist:**

- [ ] Name matches account/ID
- [ ] Email matches account (if applicable)
- [ ] Photo ID valid and unexpired
- [ ] Address verification (if required)
- [ ] Agent authorization verified (if applicable)

### Stage 3: Data Collection (Day 5-20)

```
[ ] 6. IDENTIFY DATA SOURCES
    - Primary database (PostgreSQL)
    - Redis cache
    - Analytics systems
    - Backup systems
    - Third-party processors
    - Audit logs

[ ] 7. COLLECT DATA
    - Query all identified sources
    - Export in structured format
    - Include metadata where applicable
    - Document any data not found

[ ] 8. REVIEW & REDACT
    - Remove third-party personal data
    - Redact system credentials
    - Ensure no proprietary code included
    - Legal review for complex cases
```

**Data Categories to Collect:**

| Category | Source | Export Format |
|----------|--------|---------------|
| Account Information | users table | JSON |
| Profile Data | profiles table | JSON |
| Debate Participation | debates, messages tables | JSON |
| Voting History | votes table | JSON |
| API Usage | api_logs table | JSON |
| Authentication Events | auth_logs table | JSON |
| Payment History | Stripe API | JSON |
| Support Tickets | Support system | JSON |

### Stage 4: Response Preparation (Day 20-28)

```
[ ] 9. PREPARE RESPONSE PACKAGE
    - Compile all collected data
    - Format in machine-readable format (JSON)
    - Include human-readable summary
    - Generate secure download link

[ ] 10. QUALITY ASSURANCE
    - Verify completeness
    - Confirm no third-party data included
    - Ensure redactions applied
    - Manager review sign-off

[ ] 11. PREPARE COVER LETTER
    - Summarize data provided
    - Explain any data not provided (with legal basis)
    - Include contact for questions
```

### Stage 5: Delivery (Day 28-30)

```
[ ] 12. DELIVER RESPONSE
    - Send via secure channel
    - Use encrypted download link (expires 7 days)
    - Require authentication to access
    - Log delivery timestamp

[ ] 13. CLOSE REQUEST
    - Update status to "Complete"
    - Archive documentation
    - Send satisfaction survey (optional)
    - Set reminder for retention deletion
```

---

## Request-Specific Procedures

### Access Request

**Scope:** Provide copy of all personal data processed

**Include:**
- All personal data held
- Categories of data
- Processing purposes
- Recipients/categories of recipients
- Retention periods
- Source of data (if not from subject)
- Automated decision-making details

**Export Script:**

```bash
# Generate data export for user
python scripts/dsar_export.py --user-id USER_ID --type access

# Output:
# - user_data_export_YYYYMMDD.json (machine-readable)
# - user_data_summary_YYYYMMDD.pdf (human-readable)
```

### Erasure Request (Right to be Forgotten)

**Prerequisites Check:**
- [ ] No legal obligation to retain
- [ ] No freedom of expression defense
- [ ] No public health purpose
- [ ] No archiving in public interest
- [ ] No legal claims defense needed

**Procedure:**
1. Soft delete from production (immediate)
2. Remove from analytics (7 days)
3. Remove from backups (per backup rotation)
4. Notify third-party processors
5. Confirm deletion to data subject

**Cannot Delete:**
- Audit logs (legal requirement, 7 years)
- Financial records (7 years)
- Anonymized analytics data

### Portability Request

**Format:** Machine-readable (JSON, CSV)

**Provide:**
- Data provided by the subject
- Data generated from subject's activity
- NOT include inferred or derived data

### Rectification Request

**Procedure:**
1. Verify correct information from subject
2. Update primary database
3. Propagate to all systems
4. Confirm update to subject
5. Notify third parties of correction

---

## Extension Requests

When additional time is needed:

**GDPR (Complex Request):**
- Notify within 30 days of receipt
- Explain reasons for extension
- Maximum additional 60 days

**CCPA:**
- Notify within 45 days of receipt
- Explain reasons for extension
- Maximum additional 45 days

**Extension Template:**

```
Subject: DSAR Extension Notice - [DSAR-YYYY-NNNN]

Dear [Name],

We are writing regarding your data subject request [DSAR-YYYY-NNNN].

Due to [the complexity of your request / the volume of requests received],
we require additional time to complete your request.

Original Deadline: [Date]
Extended Deadline: [Date]
Reason: [Detailed explanation]

We appreciate your patience and will respond as soon as possible.

Regards,
Aragora Privacy Team
```

---

## Refusal Grounds

Requests may be refused if:

| Ground | Regulation | Action Required |
|--------|------------|-----------------|
| Manifestly unfounded | GDPR Art. 12 | Explain reasons |
| Excessive (repetitive) | GDPR Art. 12 | May charge fee |
| Identity not verified | Both | Request verification |
| Rights of others affected | GDPR Art. 15 | Partial response |
| Legal privilege | Both | Explain with legal basis |

**Refusal Template:**

```
Subject: DSAR Response - Unable to Complete [DSAR-YYYY-NNNN]

Dear [Name],

We have reviewed your data subject request [DSAR-YYYY-NNNN].

Unfortunately, we are unable to complete your request because:
[Detailed explanation of legal basis for refusal]

Your Right to Appeal:
You have the right to lodge a complaint with the supervisory authority:
- EU/EEA: [Lead supervisory authority]
- UK: Information Commissioner's Office (ico.org.uk)
- California: California Attorney General

If you believe this decision is in error, please contact us with additional
information at privacy@aragora.ai.

Regards,
Aragora Privacy Team
```

---

## Automation Tools

### DSAR Export Script

```bash
# Location: scripts/dsar_export.py

# Access request - full export
python scripts/dsar_export.py \
  --user-id USER_ID \
  --type access \
  --format json \
  --output /secure/exports/

# Portability request - portable data only
python scripts/dsar_export.py \
  --user-id USER_ID \
  --type portability \
  --format json \
  --output /secure/exports/

# Erasure request - deletion verification
python scripts/dsar_export.py \
  --user-id USER_ID \
  --type erasure \
  --dry-run  # Preview what will be deleted
```

### Deletion Script

```bash
# Location: scripts/dsar_delete.py

# Soft delete (production)
python scripts/dsar_delete.py \
  --user-id USER_ID \
  --type soft \
  --reason "DSAR-2026-0001"

# Hard delete (after retention period)
python scripts/dsar_delete.py \
  --user-id USER_ID \
  --type hard \
  --confirm
```

---

## Tracking & Metrics

### Required Tracking Fields

| Field | Description |
|-------|-------------|
| dsar_id | Unique identifier |
| submission_date | Date request received |
| channel | How request was submitted |
| request_type | Access, erasure, etc. |
| jurisdiction | GDPR, CCPA, etc. |
| status | Received, Verifying, Processing, Complete, Refused |
| verification_date | Date identity verified |
| response_date | Date response sent |
| extended | Whether extension was used |
| notes | Internal notes |

### Monthly Metrics Report

| Metric | Target | Formula |
|--------|--------|---------|
| Total Requests | Track | Count |
| On-Time Completion | >95% | Completed on time / Total |
| Average Response Time | &lt;25 days | Sum of days / Count |
| Extensions Used | &lt;10% | Extended / Total |
| Refusal Rate | &lt;5% | Refused / Total |

---

## Training Requirements

| Role | Training | Frequency |
|------|----------|-----------|
| Privacy Team | Full DSAR procedures | Quarterly |
| Customer Support | Request identification | Quarterly |
| Engineering | Data export tools | Annually |
| Legal | Refusal grounds | Annually |
| All Staff | Privacy awareness | Annually |

---

## Escalation Paths

| Situation | Escalate To | Timeline |
|-----------|-------------|----------|
| Complex legal question | Legal Team | Day 5 |
| Data cannot be found | Engineering Lead | Day 10 |
| Third-party data involved | DPO | Day 5 |
| Media/VIP request | VP Operations | Day 1 |
| Regulatory inquiry | Legal + DPO | Immediate |

---

## Contact Information

| Role | Contact | Responsibility |
|------|---------|----------------|
| Privacy Team | privacy@aragora.ai | Daily operations |
| DPO | dpo@aragora.ai | Complex requests, escalations |
| Legal | legal@aragora.ai | Refusals, regulatory response |
| Engineering | [Internal] | Data extraction support |

---

## Compliance Checklist

### Per Request
- [ ] Request logged within 24 hours
- [ ] Acknowledgment sent within 24 hours
- [ ] Identity verified before processing
- [ ] Response within statutory deadline
- [ ] Documentation complete and archived

### Quarterly Review
- [ ] Metrics reviewed
- [ ] Process improvements identified
- [ ] Training needs assessed
- [ ] Templates updated if needed

### Annual Review
- [ ] Full procedure review
- [ ] Regulatory updates incorporated
- [ ] Tool effectiveness assessed
- [ ] Staff training completed

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-14 | Initial release |
