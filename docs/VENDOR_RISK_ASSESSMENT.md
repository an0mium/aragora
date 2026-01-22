# Vendor Risk Assessment Framework

**SOC 2 Control:** CC3.1, CC9.2

This document provides the framework and templates for assessing third-party vendor risks.

## Overview

All vendors that process, store, or have access to Aragora data must undergo a risk assessment before engagement and periodic re-assessment thereafter.

## Vendor Categories

### Category 1: Critical (High Risk)
- Process customer data
- Have production system access
- Provide core infrastructure
- Examples: Cloud providers, AI API providers, database services

### Category 2: Standard (Medium Risk)
- Access non-production systems
- Process aggregate/anonymized data
- Provide development tools
- Examples: CI/CD platforms, monitoring services, development tools

### Category 3: Low (Low Risk)
- No data access
- Commodity services
- No system integration
- Examples: Office productivity, design tools, HR systems

## Assessment Requirements by Category

| Category | Initial Assessment | Re-assessment | SOC 2 Required |
|----------|-------------------|---------------|----------------|
| Critical | Full assessment | Annual | Yes |
| Standard | Standard questionnaire | Biennial | Preferred |
| Low | Self-certification | Triennial | No |

---

## Vendor Risk Assessment Template

### 1. Vendor Information

```
Vendor Name: _______________________
Service Description: _______________________
Category: [ ] Critical  [ ] Standard  [ ] Low
Contract Start Date: _______________________
Contract Value (Annual): _______________________
Primary Contact: _______________________
Assessment Date: _______________________
Assessor: _______________________
```

### 2. Data Handling

| Question | Response | Risk Score (1-5) |
|----------|----------|------------------|
| What types of Aragora data will the vendor access? | | |
| Will data be stored by the vendor? | Yes / No | |
| If stored, where (geographic location)? | | |
| Is data encrypted at rest? | Yes / No / N/A | |
| Is data encrypted in transit? | Yes / No | |
| Data retention period? | | |
| Data disposal process documented? | Yes / No | |

**Data Classification:**
- [ ] Public
- [ ] Internal
- [ ] Confidential
- [ ] Restricted

### 3. Security Controls

| Control Area | Question | Response | Risk Score |
|--------------|----------|----------|------------|
| Authentication | MFA supported/required? | | |
| Authorization | Role-based access control? | | |
| Encryption | TLS 1.2+ for all connections? | | |
| Logging | Security events logged? | | |
| Monitoring | 24/7 security monitoring? | | |
| Incident Response | Documented IR plan? | | |
| Vulnerability Management | Regular vulnerability scanning? | | |
| Penetration Testing | Annual pentest performed? | | |

### 4. Compliance & Certifications

| Certification | Status | Expiry Date | Notes |
|---------------|--------|-------------|-------|
| SOC 2 Type II | | | |
| ISO 27001 | | | |
| GDPR Compliant | | | |
| HIPAA (if applicable) | | | |
| PCI DSS (if applicable) | | | |

### 5. Business Continuity

| Question | Response | Risk Score |
|----------|----------|------------|
| Documented BCP/DR plan? | Yes / No | |
| RTO commitment? | | |
| RPO commitment? | | |
| Geographic redundancy? | Yes / No | |
| Last DR test date? | | |

### 6. Subcontractors

| Question | Response |
|----------|----------|
| Does vendor use subcontractors for our data? | Yes / No |
| If yes, are they assessed by vendor? | |
| Flow-down security requirements? | |

### 7. Contractual Requirements

- [ ] Data Processing Agreement (DPA) signed
- [ ] NDA in place
- [ ] SLA documented
- [ ] Right to audit clause
- [ ] Breach notification clause (< 72 hours)
- [ ] Data return/deletion on termination
- [ ] Insurance requirements met

---

## Risk Scoring Matrix

### Risk Score Definitions

| Score | Level | Definition |
|-------|-------|------------|
| 1 | Minimal | Strong controls, low data exposure |
| 2 | Low | Adequate controls, limited data access |
| 3 | Medium | Some gaps, standard data access |
| 4 | High | Significant gaps, sensitive data access |
| 5 | Critical | Major gaps, critical data exposure |

### Overall Risk Calculation

```
Overall Risk = (Data Risk × 0.3) + (Security Risk × 0.4) + (Compliance Risk × 0.2) + (BC Risk × 0.1)
```

### Risk Thresholds

| Overall Score | Decision |
|---------------|----------|
| 1.0 - 2.0 | Approve |
| 2.1 - 3.0 | Approve with conditions |
| 3.1 - 4.0 | Requires CTO approval |
| 4.1 - 5.0 | Reject (or CEO exception) |

---

## Risk Acceptance Criteria

### Acceptable Risks

Risks may be accepted when:
1. Risk score is within approved thresholds
2. Compensating controls are implemented
3. Business justification is documented
4. Acceptance is approved by appropriate authority
5. Risk is added to the risk register

### Risk Acceptance Authority

| Risk Level | Approval Authority |
|------------|-------------------|
| Low (1-2) | Team Lead |
| Medium (2.1-3) | CTO |
| High (3.1-4) | CTO + CEO |
| Critical (4.1-5) | Not acceptable without exception |

### Exception Process

1. Document the risk in detail
2. Identify compensating controls
3. Submit to Security team for review
4. Obtain required approvals
5. Set review date (maximum 6 months)
6. Add to risk register with monitoring plan

---

## Current Vendor Risk Register

| Vendor | Category | Last Assessment | Risk Score | Status |
|--------|----------|-----------------|------------|--------|
| Anthropic | Critical | 2026-01-15 | 1.5 | Approved |
| OpenAI | Critical | 2026-01-15 | 1.8 | Approved |
| AWS | Critical | 2025-12-01 | 1.2 | Approved |
| Supabase | Critical | 2026-01-10 | 2.1 | Approved with conditions |
| Upstash (Redis) | Standard | 2026-01-10 | 2.0 | Approved |
| GitHub | Standard | 2025-11-15 | 1.5 | Approved |
| Vercel | Standard | 2025-11-20 | 1.8 | Approved |

---

## Assessment Schedule

| Vendor | Next Assessment | Owner |
|--------|-----------------|-------|
| Anthropic | 2027-01-15 | Security |
| OpenAI | 2027-01-15 | Security |
| AWS | 2026-12-01 | Infrastructure |
| Supabase | 2027-01-10 | Infrastructure |
| Upstash | 2028-01-10 | Infrastructure |

---

*Last updated: January 2026*
*Document owner: Head of Operations*
*Review cycle: Annual*
