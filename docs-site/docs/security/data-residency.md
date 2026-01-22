---
title: Data Residency Policy
description: Data Residency Policy
---

# Data Residency Policy

**Effective Date:** January 14, 2026
**Last Updated:** January 14, 2026
**Version:** 1.0.0
**Owner:** Compliance Team

---

## Purpose

This document defines Aragora's data residency practices, including where customer data is stored, processed, and transferred. It ensures compliance with regional data protection regulations (GDPR, CCPA, etc.) and provides transparency about data locations.

**SOC 2 Control:** C1-02 - Data residency and geographic controls

---

## Data Storage Locations

### Primary Infrastructure

| Service | Provider | Region | Purpose |
|---------|----------|--------|---------|
| Application Servers | AWS | us-east-1 (N. Virginia) | Primary API and compute |
| Database (PostgreSQL) | Supabase | us-east-1 (N. Virginia) | Customer data storage |
| Cache (Redis) | AWS ElastiCache | us-east-1 (N. Virginia) | Session and rate limiting |
| Object Storage | AWS S3 | us-east-1 (N. Virginia) | Document and artifact storage |
| CDN | Cloudflare | Global (edge) | Static assets, caching |

### Backup Locations

| Backup Type | Location | Retention | Encryption |
|-------------|----------|-----------|------------|
| Database snapshots | us-east-1 | 30 days | AES-256 |
| Cross-region replica | us-west-2 (Oregon) | Real-time | AES-256 |
| Cold storage archive | us-east-1 (Glacier) | 1 year | AES-256 |

### Disaster Recovery

| Failover Region | Activation Time | Data Sync |
|-----------------|-----------------|-----------|
| us-west-2 (Oregon) | < 1 hour | Real-time replication |

---

## Data Processing Locations

### AI Provider Data Flows

When processing debates, user content may be sent to AI providers:

| Provider | Processing Location | Data Sent | Retention |
|----------|---------------------|-----------|-----------|
| Anthropic (Claude) | US | Debate prompts, user queries | Zero retention (API) |
| OpenAI (GPT) | US | Debate prompts, user queries | Zero retention (API) |
| OpenRouter | US | Fallback queries | Zero retention (API) |
| Mistral | EU (France) | Optional - if configured | Zero retention (API) |

**Note:** AI providers receive only debate content, not user PII. All AI API calls use zero-retention settings where available.

### Analytics and Monitoring

| Service | Location | Data Processed |
|---------|----------|----------------|
| Prometheus/Grafana | us-east-1 | Anonymized metrics |
| Sentry | US | Error traces (PII redacted) |
| Uptime Kuma | us-east-2 | Availability metrics |

---

## Regional Compliance

### United States

- **Applicable Laws:** CCPA, state privacy laws
- **Data Location:** us-east-1 (N. Virginia), us-west-2 (Oregon)
- **Controls:**
  - All production data in US AWS regions
  - SOC 2 Type II certified infrastructure
  - Data encryption at rest and in transit

### European Union (GDPR)

- **Applicable Laws:** GDPR, ePrivacy Directive
- **Legal Basis:** Standard Contractual Clauses (SCCs) for transfers
- **Data Location:** Data stored in US with adequate safeguards
- **Controls:**
  - DPA (Data Processing Agreement) available
  - SCCs signed with AWS and Supabase
  - EU-representative appointed (as required)
  - DSAR workflow documented (see DSAR_WORKFLOW.md)

**EU Customer Options:**
1. Standard: US-based storage with SCC protections
2. Enterprise: Contact sales for EU-hosted options

### United Kingdom (UK GDPR)

- **Applicable Laws:** UK GDPR, Data Protection Act 2018
- **Legal Basis:** International Data Transfer Agreement (IDTA)
- **Data Location:** Data stored in US with adequate safeguards
- **Controls:**
  - IDTA addendum available
  - TIA (Transfer Impact Assessment) completed

### California (CCPA/CPRA)

- **Applicable Laws:** CCPA, CPRA
- **Data Location:** US-based
- **Controls:**
  - Do Not Sell My Personal Information honored
  - Consumer rights portal available
  - 45-day response to access requests

### Canada (PIPEDA)

- **Applicable Laws:** PIPEDA, provincial laws
- **Data Location:** US with cross-border safeguards
- **Controls:**
  - Privacy policy compliant with PIPEDA
  - Breach notification to OPC if required

---

## Customer Data Controls

### Data Residency Selection

| Tier | Available Regions | Migration Support |
|------|-------------------|-------------------|
| Free | US only | N/A |
| Starter | US only | N/A |
| Professional | US (EU roadmap) | Contact support |
| Enterprise | US, EU (custom) | Full migration support |

### Data Export

Customers can export their data via the self-service API:

```bash
# JSON export (default)
GET /api/privacy/export
GET /api/v2/users/me/export

# CSV export
GET /api/privacy/export?format=csv

# Data inventory (categories collected)
GET /api/privacy/data-inventory
```

Export includes:
- User profile data
- Organization membership
- OAuth provider links
- Preferences
- Audit log (last 90 days)
- Usage summary

For manual requests: privacy@aragora.ai (processed within 30 days per GDPR)

### Data Deletion

Customers can delete their account via the self-service API:

```bash
# Delete account (requires password confirmation)
DELETE /api/privacy/account
DELETE /api/v2/users/me

# Request body:
{
  "password": "your_password",
  "confirm": true,
  "reason": "optional reason for deletion"
}
```

For manual requests: privacy@aragora.ai with subject "Account Deletion Request"

Deletion timeline:
- Active data: Immediate (anonymized for audit compliance)
- OAuth links: Immediate
- API keys: Immediate
- MFA data: Immediate
- Backups: Within 30 days
- Audit logs: Retained 7 years (legal requirement, PII redacted)

---

## Third-Party Subprocessors

### Authorized Subprocessors

| Subprocessor | Purpose | Location | Data Accessed |
|--------------|---------|----------|---------------|
| Amazon Web Services | Infrastructure | US | All data |
| Supabase | Database | US | Customer data |
| Cloudflare | CDN, DDoS protection | Global | Request data (encrypted) |
| Stripe | Payment processing | US | Billing info only |
| Anthropic | AI processing | US | Debate content |
| OpenAI | AI processing | US | Debate content |
| Google (OAuth) | Authentication | US | Auth tokens only |

### Subprocessor Notification

Customers can subscribe to subprocessor change notifications:
- Email: Sign up at aragora.ai/subprocessor-updates
- API webhook: Configure at Settings > Webhooks

30-day advance notice provided for new subprocessors.

---

## Data Transfer Mechanisms

### Encryption in Transit

| Transfer Type | Protocol | Minimum Version |
|---------------|----------|-----------------|
| Client to API | HTTPS/TLS | TLS 1.2 (prefer 1.3) |
| API to Database | TLS | TLS 1.3 |
| Cross-region replication | TLS + VPN | TLS 1.3 |
| Backup transfer | TLS + SSE-KMS | TLS 1.3 |

### Cross-Border Safeguards

For data transfers from EU/UK to US:

1. **Standard Contractual Clauses (SCCs)**
   - Module 2 (Controller to Processor)
   - Supplementary measures implemented

2. **Technical Measures**
   - End-to-end encryption (AES-256)
   - Pseudonymization where applicable
   - Access logging and monitoring

3. **Organizational Measures**
   - Data access limited to authorized personnel
   - Security training for all staff
   - Regular access reviews

---

## Network Architecture

### Traffic Flow

```
Customer (Global)
       |
       v
[Cloudflare CDN] ---- Global Edge (encrypted)
       |
       v
[AWS ALB] ---- us-east-1 (TLS termination)
       |
       v
[API Servers] ---- us-east-1 (VPC)
       |
       v
[Database] ---- us-east-1 (private subnet)
       |
       v
[Replica] ---- us-west-2 (DR)
```

### Network Isolation

| Zone | Purpose | Access |
|------|---------|--------|
| Public subnet | Load balancers | Internet |
| Private subnet | API servers | VPC only |
| Isolated subnet | Database | Private subnet only |
| DR subnet | Failover | Cross-region VPN |

---

## Compliance Certifications

### Current Certifications

| Certification | Status | Valid Until |
|---------------|--------|-------------|
| SOC 2 Type II | In Progress | - |
| ISO 27001 | Roadmap | - |
| GDPR Compliance | Self-assessed | Ongoing |

### Infrastructure Provider Certifications

| Provider | Certifications |
|----------|----------------|
| AWS | SOC 1/2/3, ISO 27001, FedRAMP, GDPR, HIPAA |
| Supabase | SOC 2 Type II, GDPR |
| Cloudflare | SOC 2 Type II, ISO 27001, GDPR |
| Stripe | PCI DSS Level 1, SOC 2 |

---

## Data Residency FAQ

### Can I choose where my data is stored?
Enterprise customers can request specific regional hosting. Contact sales@aragora.ai for options.

### Is my data encrypted?
Yes. All data is encrypted at rest (AES-256) and in transit (TLS 1.2+).

### Who can access my data?
Only authorized Aragora personnel with legitimate business need. All access is logged and audited.

### Can I get a copy of my data?
Yes. Use the data export feature or contact support@aragora.ai.

### What happens to my data if I delete my account?
Active data is deleted immediately. Backups are purged within 30 days. Audit logs are retained for compliance (7 years).

### Does Aragora sell my data?
No. Aragora never sells customer data. See our Privacy Policy for details.

### How do you handle government data requests?
We follow a strict legal process:
- Requests must be legally valid and properly scoped
- We notify users unless legally prohibited
- We provide only the minimum data required
- All requests are logged in our transparency report

For law enforcement inquiries, contact: legal@aragora.ai

---

## Contact Information

**Privacy Inquiries:** privacy@aragora.ai
**Data Protection Officer:** dpo@aragora.ai
**Legal:** legal@aragora.ai
**Security:** security@aragora.ai

**EU Representative:**
Aragora EU Representative
*(Address to be published upon GDPR Article 27 appointment)*
eu-rep@aragora.ai

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-14 | Initial release |

---

## Related Documents

- [Privacy Policy](./privacy-policy)
- [Data Classification Policy](./data-classification)
- [DSAR Workflow](./dsar)
- [Breach Notification SLA](./breach-notification)
- [Incident Response Plan](./INCIDENT_RESPONSE.md)
