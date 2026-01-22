---
title: Compliance Documentation
description: Compliance Documentation
---

# Compliance Documentation

Regulatory compliance mappings and controls for Aragora.

## Table of Contents

- [Overview](#overview)
- [GDPR Compliance](#gdpr-compliance)
- [SOC 2 Controls](#soc-2-controls)
- [HIPAA Considerations](#hipaa-considerations)
- [AI Act Compliance](#ai-act-compliance)
- [CCPA Compliance](#ccpa-compliance)
- [Data Processing](#data-processing)
- [Audit Support](#audit-support)

---

## Overview

Aragora is designed with privacy and security as core principles. This document maps Aragora's features to common regulatory frameworks.

### Compliance Summary

| Framework | Status | Notes |
|-----------|--------|-------|
| GDPR | Supported | Full data subject rights |
| SOC 2 Type II | Mappable | Controls documented below |
| HIPAA | Partial | Requires BAA and additional config |
| EU AI Act | Supported | Risk assessment available |
| CCPA | Supported | Full consumer rights |
| ISO 27001 | Mappable | Controls align with Annex A |

---

## GDPR Compliance

### Article Mapping

| GDPR Article | Requirement | Aragora Implementation |
|--------------|-------------|------------------------|
| Art. 5 | Data minimization | Only essential data collected |
| Art. 6 | Lawful basis | Consent and legitimate interest tracking |
| Art. 7 | Consent conditions | Explicit opt-in, withdrawable consent |
| Art. 12-14 | Transparency | Privacy policy, data usage disclosure |
| Art. 15 | Right of access | Data export API |
| Art. 16 | Right to rectification | User profile editing |
| Art. 17 | Right to erasure | Account deletion API |
| Art. 18 | Right to restriction | Processing pause capability |
| Art. 20 | Data portability | JSON/CSV export |
| Art. 25 | Privacy by design | Encryption, access controls |
| Art. 30 | Records of processing | Audit logging |
| Art. 32 | Security measures | See SECURITY.md |
| Art. 33-34 | Breach notification | Incident response procedures |

### Data Subject Rights Implementation

#### Right of Access (Art. 15)

Users can export their data via the self-service API:

```bash
# Export all user data
GET /api/privacy/export
GET /api/v2/users/me/export?format=json

# CSV format
GET /api/privacy/export?format=csv

# Data inventory (categories collected)
GET /api/privacy/data-inventory
```

Exports include:
- Profile information
- Organization membership
- OAuth provider links
- User preferences
- Audit log (90 days)
- Usage summary

For manual requests: privacy@aragora.ai (processed within 30 days).
See [DSAR_WORKFLOW.md](../DSAR_WORKFLOW.md) for detailed procedures.

#### Right to Erasure (Art. 17)

Users can delete their account via the self-service API:

```bash
# Delete account (requires password confirmation)
DELETE /api/privacy/account
DELETE /api/v2/users/me

# Request body
{
  "password": "your_password",
  "confirm": true,
  "reason": "optional reason"
}
```

For manual requests: privacy@aragora.ai with subject "Account Deletion Request"

Deletion process:
1. User profile anonymized (email, name replaced)
2. OAuth provider links removed
3. API keys revoked
4. MFA data cleared
5. Active sessions invalidated
6. Audit logs: PII redacted (retained for compliance)
7. Backups: marked for exclusion in next rotation

#### Data Portability (Art. 20)

```bash
# Export in machine-readable format
curl -X GET /api/v2/users/me/export?format=json \
  -H "Authorization: Bearer $TOKEN"

# Alternative formats
?format=csv  # CSV export
?format=xml  # XML export
```

### Consent Management

```python
# Consent tracking in database
consent_record = {
    "user_id": "uuid",
    "consent_type": "marketing|analytics|third_party",
    "granted": True,
    "granted_at": "2026-01-13T10:00:00Z",
    "ip_address": "hashed",
    "version": "v2.1"  # Policy version
}
```

### Data Retention

| Data Type | Retention Period | Legal Basis |
|-----------|------------------|-------------|
| User accounts | Until deletion | Consent |
| Debate content | Indefinite (anonymizable) | Legitimate interest |
| Audit logs | 1 year | Legal obligation |
| Access logs | 90 days | Security |
| Backup data | 14 days | Legitimate interest |
| Deleted user data | 30 days (soft delete) | Recovery period |

---

## SOC 2 Controls

### Trust Service Criteria Mapping

#### Security (Common Criteria)

| Control | CC# | Aragora Implementation |
|---------|-----|------------------------|
| Logical access | CC6.1 | RBAC, JWT authentication |
| System boundaries | CC6.2 | Network policies, firewalls |
| Access removal | CC6.3 | Session invalidation, token revocation |
| Access restrictions | CC6.6 | Rate limiting, IP allowlisting |
| Transmission security | CC6.7 | TLS 1.2+, WSS |
| Vulnerability management | CC7.1 | Dependency scanning, updates |
| Security monitoring | CC7.2 | Audit logs, Prometheus alerts |
| Incident response | CC7.3 | Documented procedures |
| Incident recovery | CC7.4 | Backup/restore tested |
| Security testing | CC7.5 | Penetration testing (annual) |

#### Availability

| Control | A# | Aragora Implementation |
|---------|-----|------------------------|
| Capacity planning | A1.1 | Resource monitoring, scaling |
| Recovery planning | A1.2 | DR runbook, RTO/RPO targets |
| Backup procedures | A1.2 | Daily backups, 14-day retention |
| Incident recovery | A1.3 | Tested recovery procedures |

#### Processing Integrity

| Control | PI# | Aragora Implementation |
|---------|-----|------------------------|
| Processing accuracy | PI1.1 | Input validation, checksums |
| Processing completeness | PI1.2 | Transaction logging |
| Processing timeliness | PI1.3 | SLO monitoring |

#### Confidentiality

| Control | C# | Aragora Implementation |
|---------|-----|------------------------|
| Confidential info identification | C1.1 | Data classification |
| Confidential info protection | C1.2 | Encryption at rest/transit |

#### Privacy

| Control | P# | Aragora Implementation |
|---------|-----|------------------------|
| Privacy notice | P1.1 | Privacy policy displayed |
| Choice and consent | P2.1 | Opt-in consent |
| Collection limitation | P3.1 | Minimal data collection |
| Use and retention | P4.1 | Defined retention periods |
| Access | P5.1 | Data export available |
| Disclosure | P6.1 | No third-party sale |
| Quality | P7.1 | User can update profile |
| Monitoring | P8.1 | Privacy compliance audits |

### Evidence Collection

```bash
# Generate SOC 2 evidence report
python scripts/compliance_report.py --framework soc2 --output soc2_evidence.json

# Evidence includes:
# - Access control configurations
# - Encryption settings
# - Audit log samples
# - Backup verification results
# - Incident response test results
```

---

## HIPAA Considerations

> **Note:** Aragora is not HIPAA-certified by default. Additional configuration and a Business Associate Agreement (BAA) are required for PHI handling.

### Technical Safeguards

| Safeguard | § | Implementation Status |
|-----------|---|----------------------|
| Access control | 164.312(a)(1) | ✅ RBAC, unique user IDs |
| Audit controls | 164.312(b) | ✅ Comprehensive audit logging |
| Integrity controls | 164.312(c)(1) | ✅ Checksums, validation |
| Transmission security | 164.312(e)(1) | ✅ TLS encryption |
| Authentication | 164.312(d) | ✅ MFA available |

### Administrative Safeguards

| Safeguard | § | Notes |
|-----------|---|-------|
| Security officer | 164.308(a)(2) | Customer responsibility |
| Workforce training | 164.308(a)(5) | Customer responsibility |
| Incident procedures | 164.308(a)(6) | ✅ Documented |
| Contingency plan | 164.308(a)(7) | ✅ DR runbook |
| BAA requirements | 164.308(b)(1) | Contact sales |

### Physical Safeguards

| Safeguard | § | Notes |
|-----------|---|-------|
| Facility access | 164.310(a)(1) | Cloud provider responsibility |
| Workstation use | 164.310(b) | Customer responsibility |
| Device controls | 164.310(d)(1) | Encryption at rest |

### HIPAA-Ready Configuration

```bash
# Enable HIPAA-ready mode
export ARAGORA_HIPAA_MODE=1

# This enables:
# - Enhanced audit logging
# - Automatic session timeout (15 minutes)
# - Mandatory MFA for all users
# - Encrypted local storage
# - PHI data classification warnings
```

---

## AI Act Compliance

### Risk Classification

Aragora's multi-agent debate system is classified as:

| Use Case | Risk Level | Rationale |
|----------|------------|-----------|
| Code review debates | Minimal | Developer tool, no direct user impact |
| Decision support | Limited | Advisory only, human oversight required |
| Content moderation | Limited | Requires human review |
| Autonomous actions | High | Nomic loop has safety guardrails |

### Transparency Requirements

#### Model Documentation

| Requirement | Implementation |
|-------------|----------------|
| Training data | Third-party models (Anthropic, OpenAI) |
| Model capabilities | Documented per agent |
| Limitations | Listed in agent descriptions |
| Intended use | Debate and critique |

#### User Disclosure

```
This system uses AI models from multiple providers:
- Anthropic Claude
- OpenAI GPT
- Google Gemini
- Mistral AI
- xAI Grok

AI-generated content is clearly labeled. Human oversight
is maintained through voting and approval mechanisms.
```

### Human Oversight

| Mechanism | Description |
|-----------|-------------|
| Vote requirements | Consensus requires human votes |
| Approval workflows | Critical changes need human approval |
| Override capability | Humans can override AI decisions |
| Audit trails | All AI actions logged |
| Kill switch | Circuit breaker stops runaway processes |

### Technical Documentation

```bash
# Generate AI Act compliance report
python scripts/compliance_report.py --framework ai-act --output ai_act_report.json

# Report includes:
# - Model inventory
# - Risk assessment
# - Transparency measures
# - Human oversight mechanisms
# - Testing documentation
```

---

## CCPA Compliance

### Consumer Rights

| Right | Implementation |
|-------|----------------|
| Right to know | Data inventory, privacy policy |
| Right to delete | Account deletion API |
| Right to opt-out | Marketing consent toggle |
| Right to non-discrimination | No service degradation |
| Right to correct | Profile editing |

### Data Collection Disclosure

Categories of personal information collected:

| Category | Examples | Purpose |
|----------|----------|---------|
| Identifiers | Email, username | Account management |
| Internet activity | Debate participation | Service provision |
| Geolocation | IP-derived country | Compliance, analytics |
| Inferences | Agent preferences | Personalization |

### Do Not Sell

Aragora does not sell personal information. Third-party sharing is limited to:
- LLM providers (for debate processing)
- Analytics (anonymized, opt-out available)
- Legal requirements (law enforcement)

### CCPA Request Handling

```bash
# Right to know (data inventory)
GET /api/privacy/data-inventory
GET /api/v2/users/me/data-inventory

# Right to know (full export)
GET /api/privacy/export
GET /api/v2/users/me/export

# Right to delete
DELETE /api/privacy/account
DELETE /api/v2/users/me
# Body: {"password": "...", "confirm": true}

# Right to opt-out (Do Not Sell)
GET /api/privacy/preferences  # Get current settings
POST /api/privacy/preferences # Update settings
# Body: {"do_not_sell": true, "marketing_opt_out": true, "analytics_opt_out": true}
```

---

## Data Processing

### Data Flow Diagram

```
[User] --> [Frontend] --> [API Gateway]
                              |
                              v
                      [Authentication]
                              |
                              v
                   [Rate Limiting (Redis)]
                              |
                              v
                      [API Handlers]
                         /      \
                        v        v
              [Debate Engine]  [User Store]
                    |              |
                    v              v
             [LLM Providers]  [SQLite/PostgreSQL]
             (External APIs)     (Encrypted)
```

### Data Processing Activities

| Activity | Data | Legal Basis | Retention |
|----------|------|-------------|-----------|
| Authentication | Email, password hash | Contract | Account lifetime |
| Debate participation | Messages, votes | Consent | Indefinite |
| Agent calls | Prompts, responses | Contract | 30 days |
| Analytics | Usage patterns | Legitimate interest | 90 days |
| Audit logging | Actions, timestamps | Legal obligation | 1 year |

### Third-Party Processors

| Processor | Purpose | Data Shared | Safeguards |
|-----------|---------|-------------|------------|
| Anthropic | Claude API | Debate prompts | DPA, encryption |
| OpenAI | GPT API | Debate prompts | DPA, encryption |
| Google | Gemini API | Debate prompts | DPA, encryption |
| Sentry | Error tracking | Stack traces (no PII) | Data processing agreement |
| Prometheus | Metrics | Aggregated stats only | Self-hosted option |

---

## Audit Support

### Available Reports

```bash
# Generate compliance evidence
python scripts/compliance_report.py --all

# Individual frameworks
python scripts/compliance_report.py --framework gdpr
python scripts/compliance_report.py --framework soc2
python scripts/compliance_report.py --framework hipaa
python scripts/compliance_report.py --framework ccpa
```

### Audit Log Access

```bash
# Export audit logs for external review
sqlite3 .nomic/aragora_audit.db ".mode csv" ".headers on" \
  ".output audit_export.csv" "SELECT * FROM audit_log"

# Filter by date range
sqlite3 .nomic/aragora_audit.db \
  "SELECT * FROM audit_log WHERE timestamp BETWEEN '2026-01-01' AND '2026-01-31'"
```

### Control Evidence

| Control Area | Evidence Location |
|--------------|-------------------|
| Access control | User role assignments, permission logs |
| Authentication | Login audit trail, MFA enrollment |
| Encryption | TLS certificates, encryption config |
| Backup | Backup logs, restore test results |
| Monitoring | Alert configurations, incident logs |
| Change management | Git history, deployment logs |

### Auditor Access

```bash
# Create read-only auditor account
curl -X POST /api/v2/admin/users \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "email": "auditor@example.com",
    "role": "auditor",
    "permissions": ["read:audit_logs", "read:users", "read:config"]
  }'
```

---

## Compliance Checklist

### Pre-Deployment

- [ ] Privacy policy published and accessible
- [ ] Cookie consent banner implemented (if applicable)
- [ ] Data processing agreements with third parties
- [ ] Encryption enabled at rest and in transit
- [ ] Audit logging configured
- [ ] Backup procedures documented and tested

### Ongoing

- [ ] Monthly access review
- [ ] Quarterly vulnerability scan
- [ ] Annual penetration test
- [ ] Annual privacy impact assessment
- [ ] Incident response drill (semi-annual)
- [ ] Employee security training (annual)

### On Request

- [ ] Data subject requests processed within 30 days
- [ ] Breach notification within 72 hours (GDPR)
- [ ] Audit evidence available within 5 business days

---

## Contact

- **Privacy Officer:** privacy@aragora.ai
- **Security Team:** security@aragora.ai
- **Compliance Questions:** compliance@aragora.ai

---

## Related Documentation

- [SECURITY.md](../SECURITY.md) - Security policies and controls
- [DISASTER_RECOVERY.md](../DISASTER_RECOVERY.md) - Backup and recovery procedures
- [DATABASE.md](../DATABASE.md) - Data storage and encryption
- [DATA_RESIDENCY.md](../DATA_RESIDENCY.md) - Data location and transfer policies
- [DATA_CLASSIFICATION.md](../DATA_CLASSIFICATION.md) - Data classification levels
- [PRIVACY_POLICY.md](../PRIVACY_POLICY.md) - Privacy policy
- [DSAR_WORKFLOW.md](../DSAR_WORKFLOW.md) - Data subject access request procedures
- [BREACH_NOTIFICATION_SLA.md](../BREACH_NOTIFICATION_SLA.md) - Breach response timelines
- [REMOTE_WORK_SECURITY.md](../REMOTE_WORK_SECURITY.md) - Remote access security
