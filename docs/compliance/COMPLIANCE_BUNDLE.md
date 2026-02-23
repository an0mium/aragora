# Compliance Artifacts Bundle

Unified entry point for all Aragora compliance documentation, controls mappings, and audit-ready artifacts.

## Compliance Overview

| Framework | Status | Primary Artifact |
|-----------|--------|------------------|
| SOC 2 Type II | Production-ready (60+ controls) | [SOC 2 Controls](#soc-2-controls-mapping) |
| GDPR | Supported | [GDPR Section](#gdpr-data-processing-impact-assessment) |
| HIPAA | Configurable | [HIPAA Section](#hipaa-baa-outline) |
| EU AI Act | Supported (Aug 2026 deadline) | [EU AI Act Section](#eu-ai-act) |
| CCPA/CPRA | Supported | [DSAR Workflow](../enterprise/DSAR_WORKFLOW.md) |
| ISO 27001 | Mappable | Controls align with Annex A |

## Existing Compliance Artifacts

### SOC 2

| Document | Location | Content |
|----------|----------|---------|
| SOC 2 Controls | [docs/enterprise/SOC2_CONTROLS.md](../enterprise/SOC2_CONTROLS.md) | Trust Service Criteria implementation mapping (CC1-CC9, A1, PI1, C1, P1) |
| SOC 2 Control Matrix | [docs/enterprise/SOC2_CONTROL_MATRIX.md](../enterprise/SOC2_CONTROL_MATRIX.md) | Detailed control-to-code mapping for audit preparation (60+ controls) |
| SOC 2 Evidence | [docs/enterprise/SOC2_EVIDENCE.md](../enterprise/SOC2_EVIDENCE.md) | Evidence collection procedures |
| Security Audit Checklist | [docs/enterprise/SECURITY_AUDIT_CHECKLIST.md](../enterprise/SECURITY_AUDIT_CHECKLIST.md) | Pre-audit checklist |
| Integration Audit | [docs/enterprise/INTEGRATION_AUDIT.md](../enterprise/INTEGRATION_AUDIT.md) | Third-party integration security audit |

### GDPR and Privacy

| Document | Location | Content |
|----------|----------|---------|
| GDPR Compliance | [docs/enterprise/GDPR_COMPLIANCE.md](../enterprise/GDPR_COMPLIANCE.md) | Full GDPR article mapping and configuration guide |
| Privacy Policy | [docs/enterprise/PRIVACY_POLICY.md](../enterprise/PRIVACY_POLICY.md) | Customer-facing privacy policy |
| DSAR Workflow | [docs/enterprise/DSAR_WORKFLOW.md](../enterprise/DSAR_WORKFLOW.md) | Data Subject Access Request procedures (Art. 15-21) |
| Data Classification | [docs/enterprise/DATA_CLASSIFICATION.md](../enterprise/DATA_CLASSIFICATION.md) | Data sensitivity levels and handling rules |
| Data Residency | [docs/enterprise/DATA_RESIDENCY.md](../enterprise/DATA_RESIDENCY.md) | Data storage locations and sovereignty guidance |
| Breach Notification SLA | [docs/enterprise/BREACH_NOTIFICATION_SLA.md](../enterprise/BREACH_NOTIFICATION_SLA.md) | Breach response timelines |

### EU AI Act

| Document | Location | Content |
|----------|----------|---------|
| EU AI Act Guide | [docs/compliance/EU_AI_ACT_GUIDE.md](EU_AI_ACT_GUIDE.md) | Full compliance guide with artifact generation workflows |
| EU AI Act Checklist | [docs/compliance/EU_AI_ACT_CHECKLIST.md](EU_AI_ACT_CHECKLIST.md) | Article-by-article compliance checklist |
| EU AI Act Sample | [docs/compliance/EU_AI_ACT_SAMPLE.md](EU_AI_ACT_SAMPLE.md) | Example artifact output |
| Compliance Presets | [docs/enterprise/COMPLIANCE_PRESETS.md](../enterprise/COMPLIANCE_PRESETS.md) | Pre-configured compliance profiles |

### Security

| Document | Location | Content |
|----------|----------|---------|
| Security Patterns | [docs/enterprise/SECURITY_PATTERNS.md](../enterprise/SECURITY_PATTERNS.md) | Defense-in-depth patterns, code execution sandboxing |
| Security Audit | [docs/enterprise/SECURITY_AUDIT.md](../enterprise/SECURITY_AUDIT.md) | Security posture assessment |
| Security Testing | [docs/enterprise/SECURITY_TESTING.md](../enterprise/SECURITY_TESTING.md) | Penetration testing and vulnerability scanning |
| Security Deployment | [docs/deployment/SECURITY_DEPLOYMENT.md](../deployment/SECURITY_DEPLOYMENT.md) | Production security hardening |
| CI/CD Security | [docs/enterprise/CI_CD_SECURITY.md](../enterprise/CI_CD_SECURITY.md) | Pipeline security controls |
| Secrets Management | [docs/enterprise/SECRETS_MANAGEMENT.md](../enterprise/SECRETS_MANAGEMENT.md) | Secret rotation and storage |
| Security Runtime | [docs/enterprise/SECURITY_RUNTIME.md](../enterprise/SECURITY_RUNTIME.md) | Runtime security controls |
| Workflow Security | [docs/enterprise/WORKFLOW_SECURITY.md](../enterprise/WORKFLOW_SECURITY.md) | Workflow execution security |

### Access Control

| Document | Location | Content |
|----------|----------|---------|
| RBAC Matrix | [docs/enterprise/RBAC_MATRIX.md](../enterprise/RBAC_MATRIX.md) | 8 roles, 100+ permissions, full matrix |
| RBAC Guide | [docs/enterprise/RBAC_GUIDE.md](../enterprise/RBAC_GUIDE.md) | RBAC implementation guide |
| RBAC Permission Reference | [docs/enterprise/RBAC_PERMISSION_REFERENCE.md](../enterprise/RBAC_PERMISSION_REFERENCE.md) | Complete permission catalog |
| RBAC Role Hierarchy | [docs/enterprise/RBAC_ROLE_HIERARCHY.md](../enterprise/RBAC_ROLE_HIERARCHY.md) | Role inheritance tree |
| RBAC Audit Report | [docs/enterprise/RBAC_AUDIT_REPORT.md](../enterprise/RBAC_AUDIT_REPORT.md) | Authorization audit findings |
| Auth Guide | [docs/enterprise/AUTH_GUIDE.md](../enterprise/AUTH_GUIDE.md) | Authentication setup (OIDC, SAML, MFA) |
| SSO Setup | [docs/enterprise/SSO_SETUP.md](../enterprise/SSO_SETUP.md) | Single Sign-On configuration |
| OAuth Guide | [docs/enterprise/OAUTH_GUIDE.md](../enterprise/OAUTH_GUIDE.md) | OAuth 2.0 integration |
| Session Management | [docs/enterprise/SESSION_MANAGEMENT.md](../enterprise/SESSION_MANAGEMENT.md) | Session lifecycle and security |

### Governance

| Document | Location | Content |
|----------|----------|---------|
| Governance | [docs/enterprise/GOVERNANCE.md](../enterprise/GOVERNANCE.md) | Platform governance framework |
| Governance Structure | [docs/enterprise/GOVERNANCE_STRUCTURE.md](../enterprise/GOVERNANCE_STRUCTURE.md) | Organizational structure |
| Nomic Governance | [docs/enterprise/NOMIC_GOVERNANCE.md](../enterprise/NOMIC_GOVERNANCE.md) | Self-improvement governance controls |
| Vendor Risk Assessment | [docs/enterprise/VENDOR_RISK_ASSESSMENT.md](../enterprise/VENDOR_RISK_ASSESSMENT.md) | Third-party vendor risk |
| Security Policy Acknowledgment | [docs/enterprise/SECURITY_POLICY_ACKNOWLEDGMENT.md](../enterprise/SECURITY_POLICY_ACKNOWLEDGMENT.md) | Employee security acknowledgment |

---

## SOC 2 Controls Mapping

Summary of Aragora implementations mapped to SOC 2 Trust Service Criteria. For the full matrix with file paths and evidence, see [SOC2_CONTROL_MATRIX.md](../enterprise/SOC2_CONTROL_MATRIX.md).

### Common Criteria (CC)

| TSC | Control Area | Aragora Implementation |
|-----|-------------|----------------------|
| CC1.1 | Security policies | RBAC v2 with 8 roles, 100+ permissions (`aragora/rbac/`) |
| CC1.2 | Audit logging | HMAC-SHA256 signed audit events, tamper detection (`aragora/rbac/audit.py`) |
| CC2.1 | Risk monitoring | Circuit breaker metrics, health monitoring (`aragora/resilience/`) |
| CC3.1 | Risk identification | Gauntlet adversarial testing framework (`aragora/gauntlet/`) |
| CC4.1 | Continuous monitoring | Prometheus metrics, Grafana dashboards (`aragora/observability/`) |
| CC5.1 | Control activities | Circuit breakers, rate limiting, retry policies |
| CC6.1 | Access control | OIDC/SAML SSO (`aragora/auth/oidc.py`), MFA (`aragora/auth/mfa.py`) |
| CC6.4 | Tenant isolation | Multi-tenant data isolation (`aragora/tenancy/isolation.py`) |
| CC6.7 | Encryption at rest | AES-256-GCM (`aragora/security/encryption.py`) |
| CC6.8 | Security scanning | Bandit, gitleaks in CI (`.github/workflows/security.yml`) |
| CC7.3 | Change management | CI/CD with 205K+ automated tests |
| CC7.5 | Backup/recovery | Incremental backups with retention policies (`aragora/backup/manager.py`) |
| CC8.1 | Change authorization | Protected file checksums, approval gates |
| CC9.1 | Vendor management | Vendor risk assessment process |

### Availability (A)

| TSC | Control | Implementation |
|-----|---------|----------------|
| A1.1 | Capacity planning | Configurable concurrency limits (debates, proposals, critiques) |
| A1.2 | Recovery objectives | Disaster recovery with < 1 hour RTO |
| A1.3 | Backup testing | DR drill procedures documented |

### Processing Integrity (PI)

| TSC | Control | Implementation |
|-----|---------|----------------|
| PI1.1 | Input validation | Max content length, question length, rate limiting |
| PI1.2 | Processing accuracy | Consensus mechanisms (unanimous, majority, supermajority, judge) |
| PI1.3 | Output validation | Gauntlet receipts with SHA-256 cryptographic verification |

### Confidentiality (C)

| TSC | Control | Implementation |
|-----|---------|----------------|
| C1.1 | Data classification | Four-tier classification (Public, Internal, Confidential, Restricted) |
| C1.2 | Data residency | Configurable region (US default, EU available) |
| C1.3 | Encryption in transit | TLS 1.3 required |
| C1.4 | Encryption at rest | AES-256-GCM for sensitive data |

### Privacy (P)

| TSC | Control | Implementation |
|-----|---------|----------------|
| P1.1 | Privacy notice | Privacy policy with data usage disclosure |
| P1.2 | Data subject rights | Access, rectification, erasure, portability APIs |
| P1.3 | Consent management | Explicit opt-in, withdrawable consent |
| P1.4 | Data minimization | Configurable retention, minimal logging mode |

---

## GDPR Data Processing Impact Assessment

Template for conducting a DPIA when deploying Aragora for processing personal data.

### 1. Processing Description

| Field | Details |
|-------|---------|
| **Controller** | [Your organization name] |
| **Processor** | Aragora (self-hosted or managed) |
| **Purpose** | Multi-agent AI debate for decision support |
| **Data Categories** | User queries, debate content, usage metrics, account data |
| **Data Subjects** | Organization employees, decision stakeholders |
| **Legal Basis** | Legitimate interest (Art. 6(1)(f)) or consent (Art. 6(1)(a)) |
| **Retention** | Configurable: `ARAGORA_DEBATE_RETENTION_DAYS` (default: 90) |

### 2. Necessity and Proportionality

| Assessment | Finding |
|------------|---------|
| Data minimization | Only debate content and account data collected. Metrics anonymizable via `ARAGORA_ANONYMIZE_METRICS=true` |
| Storage limitation | Configurable retention. Memory tiers auto-decay (Fast: 1 min, Medium: 1 hr, Slow: 1 day, Glacial: 1 week) |
| Purpose limitation | Data used only for debate orchestration and organizational learning |
| Accuracy | Consensus mechanisms verify decision quality. Outcome feedback detects errors |

### 3. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Unauthorized access | Low | High | RBAC v2 (8 roles, 100+ permissions), MFA, SSO |
| Data breach | Low | High | AES-256-GCM encryption, tenant isolation, audit logging |
| AI bias in decisions | Medium | Medium | Multi-agent adversarial debate, heterogeneous model consensus |
| Cross-border transfer | Medium | Medium | Configurable data residency (`ARAGORA_DATA_REGION`), EU-only mode |
| Data loss | Low | High | Incremental backups, cross-region replication, < 1 hr RTO |

### 4. Technical Measures

| Measure | Implementation |
|---------|----------------|
| Encryption at rest | AES-256-GCM (`aragora/security/encryption.py`) |
| Encryption in transit | TLS 1.3 |
| Access control | RBAC v2 with role hierarchy, permission checker with caching |
| Audit trail | HMAC-SHA256 signed events, cryptographic receipts |
| Anonymization | `aragora/privacy/anonymization.py` |
| Consent management | `aragora/privacy/consent.py` |
| Data deletion | `aragora/privacy/deletion.py` |
| Retention enforcement | `aragora/privacy/retention.py` |

### 5. GDPR Configuration

```bash
# Enable GDPR mode
ARAGORA_GDPR_MODE=true
ARAGORA_DATA_REGION=eu
ARAGORA_REQUIRE_CONSENT=true
ARAGORA_CONSENT_VERSION=1.0
ARAGORA_MINIMAL_LOGGING=true
ARAGORA_ANONYMIZE_METRICS=true
ARAGORA_DEBATE_RETENTION_DAYS=90
ARAGORA_LOG_RETENTION_DAYS=365
```

See [GDPR_COMPLIANCE.md](../enterprise/GDPR_COMPLIANCE.md) for full configuration reference.

---

## HIPAA Business Associate Agreement Outline

Template outline for organizations deploying Aragora in healthcare contexts. This is a starting point -- consult legal counsel before execution.

### Covered Provisions

| BAA Section | Aragora Capability |
|-------------|-------------------|
| **Permitted Uses** | Decision support for clinical workflows. No direct PHI processing unless configured |
| **Safeguards** | AES-256-GCM encryption, RBAC, MFA, audit logging, tenant isolation |
| **Reporting** | Breach notification within 60 days. Audit trail with HMAC-SHA256 integrity verification |
| **Return/Destruction** | Data deletion API (`aragora/privacy/deletion.py`), configurable retention |
| **Subcontractors** | AI provider list in [DATA_RESIDENCY.md](../enterprise/DATA_RESIDENCY.md) (all zero-retention API) |
| **Access** | Data export API for HIPAA-required accounting of disclosures |

### Technical Safeguards (45 CFR 164.312)

| Safeguard | Implementation |
|-----------|----------------|
| Access control (a)(1) | RBAC v2 with unique user authentication, role-based permissions |
| Audit controls (b) | HMAC-signed audit events, Prometheus metrics, structured logging |
| Integrity controls (c)(1) | SHA-256 Gauntlet receipts, consensus verification |
| Transmission security (e)(1) | TLS 1.3, WebSocket encryption |
| Encryption (a)(2)(iv) | AES-256-GCM at rest, TLS 1.3 in transit |
| Authentication (d) | OIDC/SAML SSO, TOTP/HOTP MFA |
| Automatic logoff | Configurable session timeout (`SESSION_MANAGEMENT.md`) |

### Administrative Safeguards

| Safeguard | Implementation |
|-----------|----------------|
| Risk analysis | Gauntlet adversarial testing, security scanning in CI |
| Workforce training | Security policy acknowledgment workflow |
| Contingency plan | Backup/DR with < 1 hr RTO, DR drill procedures |
| Evaluation | SOC 2 control matrix, security audit checklist |

### HIPAA Deployment Checklist

- [ ] Execute BAA between organization and Aragora deployment operator
- [ ] Enable encryption at rest (`aragora/security/encryption.py`)
- [ ] Configure MFA for all users with PHI access
- [ ] Set data retention to comply with HIPAA minimum (6 years for records)
- [ ] Enable audit logging with HMAC integrity
- [ ] Configure tenant isolation if multi-tenant
- [ ] Restrict AI provider list to HIPAA-compliant providers (see [DATA_RESIDENCY.md](../enterprise/DATA_RESIDENCY.md))
- [ ] Test data deletion and export workflows
- [ ] Document breach notification procedures
- [ ] Schedule annual risk assessment

See [docs/verticals/HEALTHCARE.md](../verticals/HEALTHCARE.md) for healthcare-specific deployment guidance.

---

## Data Residency and Sovereignty Guidance

### Deployment Regions

| Region | Infrastructure | AI Providers Available | Compliance |
|--------|---------------|----------------------|------------|
| **US (default)** | AWS us-east-1 | All providers | SOC 2, CCPA, HIPAA |
| **EU** | AWS eu-west-1 or eu-central-1 | Mistral (France), self-hosted Ollama | GDPR, EU AI Act |
| **UK** | AWS eu-west-2 | All providers | UK GDPR |
| **Self-hosted** | On-premises | Ollama (local) | Full sovereignty |

### Data Flow Controls

```
                    ┌─────────────────────────────────┐
                    │         Data Residency           │
                    │    ARAGORA_DATA_REGION=eu         │
                    └───────┬─────────────────────────┘
                            │
              ┌─────────────┼─────────────────┐
              ▼             ▼                 ▼
     ┌────────────┐  ┌──────────┐   ┌──────────────┐
     │ PostgreSQL  │  │  Redis   │   │    Backups   │
     │ EU region   │  │ EU region│   │  EU region   │
     └────────────┘  └──────────┘   └──────────────┘
              │
              ▼
     ┌──────────────────────────────────────────┐
     │          AI Provider Routing              │
     │  EU mode: Mistral (France) + Ollama only  │
     │  US mode: All providers available          │
     └──────────────────────────────────────────┘
```

### AI Provider Data Handling

| Provider | Processing Location | Data Retention | Notes |
|----------|-------------------|----------------|-------|
| Anthropic (Claude) | US | Zero retention (API) | No training on API data |
| OpenAI (GPT) | US | Zero retention (API) | Opt-out confirmed |
| Mistral | EU (France) | Zero retention (API) | EU-compliant by default |
| OpenRouter | US | Zero retention (API) | Fallback provider |
| Ollama (local) | On-premises | No external transfer | Full data sovereignty |

### Cross-Border Transfer Safeguards

For EU-to-US transfers:
1. Standard Contractual Clauses (SCCs) with AI providers
2. Data Processing Agreements (DPAs) in place
3. Transfer Impact Assessments documented
4. Technical measures: encryption in transit (TLS 1.3), encryption at rest (AES-256-GCM)
5. Organizational measures: access restricted to need-to-know, audit logging

---

## Audit Trail Configuration

### Enabling Full Audit Logging

```bash
# Structured JSON logging for audit
ARAGORA_LOG_FORMAT=json
ARAGORA_LOG_LEVEL=INFO
ARAGORA_LOG_FILE=/var/log/aragora/audit.log
ARAGORA_LOG_MAX_BYTES=10485760   # 10 MB per file
ARAGORA_LOG_BACKUP_COUNT=5

# Sensitive field redaction
ARAGORA_LOG_SENSITIVE_FIELDS=password,token,secret,api_key,authorization,cookie,session

# RBAC audit events
# Automatically logged via aragora/rbac/audit.py:
# - All authorization decisions (allow/deny)
# - Role assignments and changes
# - Permission checks with context
# Events signed with HMAC-SHA256 for tamper detection
```

### Audit Event Types

Events are logged as structured JSON with HMAC-SHA256 signatures:

| Event Category | Events | Source |
|----------------|--------|--------|
| Authentication | Login, logout, MFA challenge, token refresh | `aragora/auth/` |
| Authorization | Permission check, role assignment, access denied | `aragora/rbac/audit.py` |
| Data access | Read, export, search across tenants | `aragora/tenancy/` |
| Data modification | Create, update, delete operations | `@handle_errors` decorator |
| Debate lifecycle | Start, complete, consensus reached | `aragora/debate/orchestrator.py` |
| System operations | Backup, restore, key rotation | `aragora/backup/`, `aragora/security/` |

### Cryptographic Receipts

The Gauntlet system produces cryptographic decision receipts:
- SHA-256 hash of debate inputs, outputs, and agent votes
- Timestamp and participant attestation
- Stored in Knowledge Mound via `ReceiptAdapter`
- Queryable via `aragora/gauntlet/receipts.py`

---

## Compliance Automation

### CLI Commands

```bash
# Generate EU AI Act compliance artifacts
aragora compliance eu-ai-act --output artifacts/

# Run security audit
aragora audit security --output audit-report.json

# Generate DSAR export for a user
aragora privacy export --user-id <id> --format json

# Run adversarial testing (Gauntlet)
aragora gauntlet run --suite standard --output gauntlet-report.json

# Check SLO compliance
aragora observe slo-status
```

### Automated Checks in CI

```yaml
# .github/workflows/compliance.yml
- name: Security scanning
  run: |
    bandit -r aragora/ -f json -o bandit-report.json
    gitleaks detect --report-format json --report-path gitleaks-report.json

- name: Run test suite
  run: pytest tests/ -v --tb=short

- name: Verify audit integrity
  run: python -m aragora.rbac.audit verify --log-file audit.log
```

---

## Quick Reference: Compliance by Framework

| Requirement | SOC 2 | GDPR | HIPAA | EU AI Act |
|-------------|-------|------|-------|-----------|
| Access control | CC6 | Art. 32 | 164.312(a) | Art. 9(4)(b) |
| Encryption at rest | CC6.7 | Art. 32 | 164.312(a)(2)(iv) | - |
| Encryption in transit | CC6.6 | Art. 32 | 164.312(e)(1) | - |
| Audit logging | CC4.1 | Art. 30 | 164.312(b) | Art. 12 |
| Data retention | CC6.5 | Art. 5(1)(e) | 164.530(j) | Art. 12 |
| Data deletion | CC6.5 | Art. 17 | 164.524(e) | - |
| Breach notification | CC7.4 | Art. 33-34 | 164.408 | - |
| Risk assessment | CC3.1 | Art. 35 | 164.308(a)(1) | Art. 9 |
| Vendor management | CC9.1 | Art. 28 | 164.314(a) | Art. 28 |
| Human oversight | - | Art. 22 | - | Art. 14 |
| Transparency | - | Art. 13-14 | - | Art. 13 |
| Bias monitoring | - | - | - | Art. 10 |
