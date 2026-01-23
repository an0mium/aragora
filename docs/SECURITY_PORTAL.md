# Aragora Security Portal

**SOC 2 Control:** CC2.2

This document serves as the public-facing security information portal for Aragora customers and stakeholders.

---

## Security Overview

Aragora is committed to maintaining the highest standards of security for our multi-agent deliberation control plane. This portal provides transparency into our security practices, compliance status, and how we protect your data.

### Our Security Principles

1. **Defense in Depth** - Multiple layers of security controls
2. **Least Privilege** - Access limited to what's necessary
3. **Transparency** - Open communication about security practices
4. **Continuous Improvement** - Regular security assessments and updates

---

## Compliance & Certifications

### Current Certifications

| Certification | Status | Scope |
|---------------|--------|-------|
| SOC 2 Type II | In Progress (Q2 2026) | All production systems |
| GDPR | Compliant | EU customer data |
| CCPA | Compliant | California customer data |

### Cloud Provider Certifications

Our infrastructure runs on certified cloud providers:

| Provider | Certifications |
|----------|----------------|
| AWS | SOC 2, ISO 27001, PCI DSS, HIPAA |
| Supabase | SOC 2, GDPR, HIPAA-ready |

### Audit Reports

SOC 2 reports are available to customers under NDA. Contact security@aragora.ai to request access.

---

## Data Security

### Encryption

| Data State | Encryption Standard |
|------------|---------------------|
| In Transit | TLS 1.3 |
| At Rest | AES-256 |
| Backups | AES-256 |
| API Keys | AES-256-GCM with AAD |

### Data Residency

| Region | Data Center Location | Availability |
|--------|---------------------|--------------|
| US | AWS us-east-1, us-west-2 | Primary |
| EU | AWS eu-west-1 | Available |
| Asia-Pacific | AWS ap-southeast-1 | Planned Q3 2026 |

See [Data Residency Policy](DATA_RESIDENCY.md) for details.

### Data Classification

We classify data into four levels with corresponding protections:

| Level | Examples | Protections |
|-------|----------|-------------|
| Public | Documentation | Standard |
| Internal | Architecture docs | Access control |
| Confidential | Customer data | Encryption + access control |
| Restricted | API keys, credentials | Maximum protection |

See [Data Classification Policy](DATA_CLASSIFICATION.md) for details.

---

## Infrastructure Security

### Architecture Overview

```
                    ┌─────────────┐
                    │   WAF/CDN   │
                    │  (DDoS Prot)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Load        │
                    │ Balancer    │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │   API       │ │   API       │ │   API       │
    │   Server    │ │   Server    │ │   Server    │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    │         ┌────────────▼────────────┐         │
    │         │      VPC (Private)      │         │
    │         │  ┌─────┐    ┌─────┐     │         │
    │         │  │ DB  │    │Cache│     │         │
    │         │  └─────┘    └─────┘     │         │
    │         └─────────────────────────┘         │
    │                                             │
    └─────────────────────────────────────────────┘
```

### Network Security

- All services run in private VPCs
- Internet-facing services behind WAF
- DDoS protection enabled
- Network segmentation enforced
- Regular penetration testing

### Monitoring

- 24/7 security monitoring
- Real-time alerting
- Audit logging with tamper detection
- Anomaly detection

---

## Application Security

### Secure Development

| Practice | Implementation |
|----------|----------------|
| Code Review | Required for all changes |
| Static Analysis | Automated on every PR |
| Dependency Scanning | Daily automated scans |
| Secret Detection | Pre-commit hooks |

### Vulnerability Management

| Severity | Response Time | Resolution Time |
|----------|---------------|-----------------|
| Critical | 15 minutes | 24 hours |
| High | 1 hour | 72 hours |
| Medium | 24 hours | 30 days |
| Low | 5 days | 90 days |

### Penetration Testing

- Annual third-party penetration tests
- Continuous automated scanning
- Bug bounty program (coming Q3 2026)

---

## Access Control

### Authentication

| Feature | Status |
|---------|--------|
| Multi-factor Authentication | Required for all accounts |
| Single Sign-On (SSO) | Available (Enterprise) |
| SAML 2.0 | Supported |
| OIDC | Supported |
| API Key Authentication | Available |
| JWT Tokens | Short-lived (15 min) |

### Authorization

| Feature | Description |
|---------|-------------|
| Role-Based Access Control | Fine-grained permissions |
| Organization Isolation | Complete tenant separation |
| Workspace Permissions | Team-level access control |
| Audit Logging | All access logged |

---

## Incident Response

### Response Process

1. **Detection** - Automated monitoring + manual reporting
2. **Triage** - Severity assessment and team notification
3. **Containment** - Limit impact and preserve evidence
4. **Eradication** - Remove threat and fix vulnerabilities
5. **Recovery** - Restore services safely
6. **Post-Incident** - Review and improve

### Breach Notification

| Notification | Timeline |
|--------------|----------|
| Customer notification | Within 72 hours of confirmation |
| Regulatory notification | As required by law |
| Public disclosure | After containment if appropriate |

See [Breach Notification SLA](BREACH_NOTIFICATION_SLA.md) for details.

### Status Page

Real-time service status: [status.aragora.ai](https://status.aragora.ai)

Includes:
- Current service status
- Planned maintenance
- Incident history
- Uptime metrics

---

## Business Continuity

### Availability

| Metric | Target | Typical |
|--------|--------|---------|
| Uptime | 99.9% | 99.95% |
| RTO | 4 hours | 1 hour |
| RPO | 1 hour | 15 minutes |

### Disaster Recovery

- Multi-region infrastructure
- Automated failover
- Regular DR testing
- Documented recovery procedures

See [Disaster Recovery Plan](DISASTER_RECOVERY.md) for details.

### Backups

| Data Type | Frequency | Retention |
|-----------|-----------|-----------|
| Database | Continuous (WAL) | 30 days |
| User uploads | Daily | 90 days |
| Configuration | On change | 1 year |
| Audit logs | Real-time | 7 years |

---

## Privacy

### Data Handling

| Principle | Implementation |
|-----------|----------------|
| Data Minimization | Collect only necessary data |
| Purpose Limitation | Use data only for stated purposes |
| Storage Limitation | Delete data when no longer needed |
| Accuracy | Allow data correction |

### Your Rights

Under GDPR, CCPA, and other regulations, you have the right to:

- **Access** your data
- **Correct** inaccurate data
- **Delete** your data
- **Export** your data
- **Object** to processing
- **Restrict** processing

See [Privacy Policy](PRIVACY_POLICY.md) for full details.

### Data Subject Requests

Submit requests to: privacy@aragora.ai

Response time: Within 30 days (or less as required by law)

---

## Vendor Security

### Third-Party Vendors

All vendors with access to customer data are:
- Security assessed before engagement
- Required to meet our security standards
- Regularly re-assessed
- Contractually obligated to protect data

### Key Vendors

| Vendor | Purpose | Certification |
|--------|---------|---------------|
| AWS | Infrastructure | SOC 2, ISO 27001 |
| Anthropic | AI API | SOC 2 |
| OpenAI | AI API | SOC 2 |
| Supabase | Database | SOC 2 |

See [Vendor Risk Assessment](VENDOR_RISK_ASSESSMENT.md) for details.

---

## Security Contacts

### Reporting Security Issues

| Contact | Purpose |
|---------|---------|
| security@aragora.ai | Security questions, vulnerability reports |
| privacy@aragora.ai | Privacy questions, data requests |
| abuse@aragora.ai | Report abuse |

### Responsible Disclosure

We welcome security researchers to report vulnerabilities responsibly:

1. Email security@aragora.ai with details
2. Allow reasonable time for fix (90 days)
3. Do not access or modify customer data
4. Do not disclose publicly until fixed

We commit to:
- Acknowledge receipt within 24 hours
- Provide updates on resolution progress
- Credit researchers (with permission)
- Not pursue legal action for good-faith reports

---

## Frequently Asked Questions

### Is my data encrypted?

Yes. All data is encrypted in transit (TLS 1.3) and at rest (AES-256).

### Where is my data stored?

By default, data is stored in US data centers. EU data residency is available upon request.

### Can I get a copy of your SOC 2 report?

Yes, SOC 2 reports are available to customers and prospects under NDA. Contact security@aragora.ai.

### Do you support SSO?

Yes, we support SAML 2.0 and OIDC for Enterprise customers.

### How do you handle AI provider data?

We have Data Processing Agreements with all AI providers. They do not train on your data by default.

### Can I delete my data?

Yes, you can request data deletion at any time. Contact privacy@aragora.ai.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | January 2026 | Initial release |

---

*For security questions, contact: security@aragora.ai*
*For privacy questions, contact: privacy@aragora.ai*

*Last updated: January 2026*
