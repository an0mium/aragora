# Data Classification Policy

**Effective Date:** January 14, 2026
**Last Updated:** January 14, 2026
**Version:** 1.0.0
**Owner:** Security Team

---

## Purpose

This document defines data classification levels, handling requirements, and access controls for all data processed by Aragora. Proper data classification ensures appropriate protection based on sensitivity and regulatory requirements.

**SOC 2 Control:** CC6-01 - Data classification and handling

---

## Classification Levels

### Level 1: Public

| Attribute | Description |
|-----------|-------------|
| Definition | Information intended for public disclosure |
| Impact if Exposed | None - already public |
| Examples | Marketing materials, public documentation, blog posts, API docs |
| Access | Anyone |
| Storage | Standard |
| Transmission | Standard |
| Disposal | No special requirements |

### Level 2: Internal

| Attribute | Description |
|-----------|-------------|
| Definition | Business information for internal use |
| Impact if Exposed | Minor business impact, competitive disadvantage |
| Examples | Internal procedures, meeting notes, non-sensitive metrics, feature roadmaps |
| Access | Authenticated employees |
| Storage | Standard with access controls |
| Transmission | Standard encryption (TLS) |
| Disposal | Delete from storage when no longer needed |

### Level 3: Confidential

| Attribute | Description |
|-----------|-------------|
| Definition | Sensitive business or customer data |
| Impact if Exposed | Significant business/legal impact, customer trust damage |
| Examples | Customer data, API keys, usage analytics, debate content, financial data |
| Access | Need-to-know basis, role-based access |
| Storage | Encrypted at rest (AES-256) |
| Transmission | Encrypted (TLS 1.3) |
| Disposal | Secure deletion, audit trail |

### Level 4: Restricted

| Attribute | Description |
|-----------|-------------|
| Definition | Highly sensitive data requiring maximum protection |
| Impact if Exposed | Severe legal, financial, or reputational damage |
| Examples | PII (email, name), authentication secrets, encryption keys, credentials, payment data |
| Access | Strictly limited, MFA required, audited |
| Storage | Encrypted, access-logged, isolated |
| Transmission | Encrypted, authenticated endpoints only |
| Disposal | Cryptographic erasure, certificate of destruction |

---

## Data Inventory

### Customer Data (Confidential/Restricted)

| Data Element | Classification | Storage Location | Retention |
|--------------|---------------|------------------|-----------|
| User email | Restricted (PII) | users table | Account lifetime + 30 days |
| User name | Restricted (PII) | users table | Account lifetime + 30 days |
| Password hash | Restricted | users table | Account lifetime |
| API keys | Restricted | users table (hashed) | Until revoked |
| MFA secrets | Restricted | users table (encrypted) | Until disabled |
| Organization name | Confidential | organizations table | Account lifetime + 30 days |
| Debate content | Confidential | debates table | Configurable (default 90 days) |
| Usage records | Confidential | usage_events table | 2 years |
| Audit logs | Confidential | audit_events table | 7 years |

### Operational Data (Internal/Confidential)

| Data Element | Classification | Storage Location | Retention |
|--------------|---------------|------------------|-----------|
| Application logs | Internal | Log files/Elasticsearch | 30 days |
| Error logs | Internal | Log files/Sentry | 30 days |
| Performance metrics | Internal | Prometheus/Grafana | 90 days |
| Session tokens | Confidential | Redis/memory | Session duration |
| Request traces | Internal | Jaeger/traces | 7 days |

### Infrastructure Data (Restricted)

| Data Element | Classification | Storage Location | Retention |
|--------------|---------------|------------------|-----------|
| Database credentials | Restricted | Secrets manager | Until rotated |
| API provider keys | Restricted | Secrets manager | Until rotated |
| TLS certificates | Restricted | Certificate store | Until expired |
| SSH keys | Restricted | Secrets manager | Until rotated |
| JWT signing keys | Restricted | Secrets manager | Until rotated |

---

## Handling Requirements

### Level 1: Public

- No special handling required
- May be shared externally without approval
- No encryption requirements

### Level 2: Internal

- Share only with authenticated employees
- May be stored in standard systems
- Basic access logging recommended
- No approval required for internal sharing

### Level 3: Confidential

```
Required Controls:
[ ] Encryption at rest
[ ] TLS for transmission
[ ] Access logging
[ ] Need-to-know access
[ ] Manager approval for external sharing
[ ] Secure deletion when no longer needed
```

### Level 4: Restricted

```
Required Controls:
[ ] AES-256 encryption at rest
[ ] TLS 1.3 for transmission
[ ] MFA for access
[ ] Complete audit trail
[ ] Quarterly access review
[ ] VP approval for any external sharing
[ ] Cryptographic erasure on disposal
[ ] No local copies
[ ] No email transmission (use secure portal)
```

---

## Access Control Matrix

### By Role

| Role | Public | Internal | Confidential | Restricted |
|------|--------|----------|--------------|------------|
| Anonymous | Read | No | No | No |
| User | Read | No | Own data only | Own data only |
| Support | Read | Read | Customer data (cases) | Limited (with audit) |
| Developer | Read | Read/Write | Read (non-PII) | No (without approval) |
| Admin | Read | Read/Write | Read/Write | Read/Write (audited) |
| Security | Read | Read/Write | Read/Write | Full access (audited) |

### By System

| System | Confidential | Restricted | Controls |
|--------|--------------|------------|----------|
| Production Database | Yes | Yes | VPC, encryption, IAM |
| Staging Database | Yes (masked) | No | VPC, separate credentials |
| Development | No | No | Synthetic data only |
| CI/CD | No | Secrets only | Secret injection, no logs |
| Logging | Redacted PII | No | Automatic redaction |
| Backups | Yes | Yes | Encrypted, access-logged |

---

## Labeling and Marking

### Document Labeling

Documents containing Confidential or Restricted data should include:

```
Classification: [CONFIDENTIAL/RESTRICTED]
Owner: [Team/Individual]
Handling: See DATA_CLASSIFICATION.md
```

### Database Field Marking

Sensitive fields are documented in schema:

```sql
-- Classification: Restricted (PII)
email VARCHAR(255) NOT NULL,

-- Classification: Restricted
password_hash VARCHAR(255) NOT NULL,

-- Classification: Confidential
debate_content TEXT,
```

### API Response Handling

Sensitive fields are redacted in logs:

```python
# Automatically redacted in SecurityBarrier
REDACTED_FIELDS = [
    "password", "api_key", "token", "secret",
    "email", "credit_card", "ssn"
]
```

---

## Data Lifecycle

### Creation

1. Classify data before storage
2. Apply appropriate encryption
3. Set retention policy
4. Document in data inventory

### Processing

1. Minimize data collection
2. Use least privilege access
3. Log access to Confidential/Restricted
4. Apply redaction in logs

### Storage

1. Encrypt based on classification
2. Apply access controls
3. Regular access audits
4. Backup according to classification

### Transmission

1. Use TLS for all transmission
2. Additional encryption for Restricted
3. Authenticated endpoints only
4. Log transmission events

### Disposal

| Classification | Disposal Method |
|----------------|-----------------|
| Public | Standard deletion |
| Internal | Standard deletion |
| Confidential | Secure deletion with audit |
| Restricted | Cryptographic erasure, certificate |

---

## Incident Response

### Data Exposure Levels

| Level | Impact | Response |
|-------|--------|----------|
| Public | None | No action required |
| Internal | Low | Security team notification |
| Confidential | Medium | Security team + legal review |
| Restricted | High | Full incident response + breach assessment |

### Breach Notification

For Confidential/Restricted data exposure:

1. **Containment** (immediate)
2. **Assessment** (within 24 hours)
3. **Internal notification** (within 24 hours)
4. **Legal review** (within 48 hours)
5. **Customer notification** (within 72 hours if required)
6. **Regulatory notification** (as required by law)

---

## Training and Awareness

### Required Training

| Role | Training | Frequency |
|------|----------|-----------|
| All Employees | Data handling basics | Annual |
| Developers | Secure coding, data handling | Annual + onboarding |
| Support | Customer data handling | Annual + onboarding |
| Admins | Full classification training | Annual + on change |
| Security | Advanced data protection | Quarterly |

### Acknowledgment

All employees must acknowledge:
- Understanding of classification levels
- Handling requirements for each level
- Incident reporting procedures
- Consequences of mishandling

---

## Compliance Mapping

### SOC 2 Trust Service Criteria

| Criteria | How This Policy Addresses |
|----------|---------------------------|
| CC6.1 | Defines classification levels |
| CC6.2 | Access control matrix |
| CC6.3 | Handling requirements |
| CC6.4 | Transmission controls |
| CC6.5 | Disposal procedures |

### GDPR

| Article | How This Policy Addresses |
|---------|---------------------------|
| Art. 5 | Data minimization, storage limitation |
| Art. 25 | Privacy by design (classification) |
| Art. 32 | Technical measures (encryption) |
| Art. 33 | Breach notification procedures |

---

## Exceptions

### Exception Process

1. Submit request to security@aragora.ai
2. Include business justification
3. Document compensating controls
4. VP approval required
5. Time-limited (max 90 days)
6. Quarterly review

### Approved Exceptions

Document all approved exceptions here:

| Date | Data | Exception | Compensating Control | Expiry |
|------|------|-----------|---------------------|--------|
| - | - | - | - | - |

---

## Review and Updates

- **Quarterly:** Review access patterns and exceptions
- **Annually:** Full policy review and update
- **On Change:** Update for new data types or systems

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-14 | Initial release |
