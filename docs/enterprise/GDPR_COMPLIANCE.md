# GDPR Compliance Guide

This document describes Aragora's GDPR compliance measures and how to configure the platform for EU data protection requirements.

## Data Processing Overview

### Data Categories

| Category | Examples | Retention | Legal Basis |
|----------|----------|-----------|-------------|
| **Account Data** | Email, name, organization | Account lifetime + 30 days | Contract performance |
| **Debate Content** | Questions, agent responses | Configurable (default: 90 days) | Legitimate interest |
| **Usage Metrics** | Request counts, latency | 90 days | Legitimate interest |
| **Audit Logs** | Access events, changes | 1 year | Legal obligation |

### Data Flows

```
User Input → Aragora API → AI Providers (Claude/GPT) → Response Storage
                              ↓
                    Third-party processing
                    (see subprocessor list)
```

## Configuration for GDPR Compliance

### Environment Variables

```bash
# Enable GDPR mode (restricts data collection, enables consent flows)
ARAGORA_GDPR_MODE=true

# Data residency (EU-only processing)
ARAGORA_DATA_REGION=eu

# Retention periods (in days)
ARAGORA_DEBATE_RETENTION_DAYS=90
ARAGORA_LOG_RETENTION_DAYS=365
ARAGORA_METRICS_RETENTION_DAYS=90

# Consent management
ARAGORA_REQUIRE_CONSENT=true
ARAGORA_CONSENT_VERSION=1.0

# Data minimization
ARAGORA_MINIMAL_LOGGING=true
ARAGORA_ANONYMIZE_METRICS=true
```

### Database Configuration

For EU data residency, deploy PostgreSQL in an EU region:

```yaml
# docker-compose.production.yml (EU deployment)
services:
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=aragora
      - POSTGRES_USER=aragora
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    # Deploy in EU region only
```

## Data Subject Rights

### Right of Access (Article 15)

Users can export their data via the API:

```bash
# Export user data
curl -X GET "https://api.aragora.ai/api/v2/users/me/export" \
  -H "Authorization: Bearer $TOKEN"
```

**SDK Usage:**
```python
# Python SDK
export_data = await client.auth.export_user_data()

# Includes:
# - Profile information
# - Debate participation history
# - Consent records
# - API key metadata (not secrets)
```

### Right to Erasure (Article 17)

Users can delete their data:

```bash
# Request account deletion
curl -X DELETE "https://api.aragora.ai/api/v2/users/me" \
  -H "Authorization: Bearer $TOKEN"

# Response includes deletion confirmation
# Data is soft-deleted immediately, hard-deleted after 30 days
```

**Automated Deletion:**
```python
# Admin SDK for GDPR requests
await client.admin.process_deletion_request(
    user_id="user-123",
    reason="gdpr_erasure",
    retain_audit_logs=True,  # Required for legal compliance
)
```

### Right to Rectification (Article 16)

```python
# Update profile information
await client.auth.update_profile(
    name="Updated Name",
    email="new-email@example.com",  # Requires verification
)
```

### Right to Data Portability (Article 20)

```bash
# Export in machine-readable format
curl -X GET "https://api.aragora.ai/api/v2/users/me/export?format=json" \
  -H "Authorization: Bearer $TOKEN"
```

Supported formats: `json`, `csv`

### Right to Object (Article 21)

```python
# Opt-out of analytics/profiling
await client.auth.update_preferences(
    analytics_enabled=False,
    profiling_enabled=False,
    marketing_enabled=False,
)
```

## Consent Management

### Collecting Consent

```python
# Record consent
await client.consent.record(
    consent_type="terms_of_service",
    version="1.0",
    granted=True,
    ip_address=request.remote_addr,  # Optional
)

# Check consent status
consent = await client.consent.check("terms_of_service")
if not consent.granted:
    # Prompt for consent
    pass
```

### Consent Types

| Type | Required | Purpose |
|------|----------|---------|
| `terms_of_service` | Yes | Service usage |
| `privacy_policy` | Yes | Data processing |
| `ai_processing` | Yes | AI provider data sharing |
| `analytics` | No | Usage analytics |
| `marketing` | No | Marketing communications |

## Subprocessors

| Subprocessor | Purpose | Location | DPA |
|--------------|---------|----------|-----|
| Anthropic | Claude AI | USA (SCCs) | Yes |
| OpenAI | GPT models | USA (SCCs) | Yes |
| Mistral | Mistral models | EU | Yes |
| AWS/GCP/Azure | Infrastructure | EU option | Yes |
| PostgreSQL Cloud | Database | EU option | Yes |

### AI Provider Data Handling

All AI providers are configured to:
- Not use data for training (where supported)
- Delete input data after processing
- Apply data retention limits

```python
# Configure AI provider settings
from aragora.agents import configure_provider

configure_provider("anthropic", {
    "data_retention": "none",
    "training_opt_out": True,
    "region_preference": "eu",
})
```

## Data Protection Impact Assessment (DPIA)

### When Required

A DPIA is recommended when:
1. Processing sensitive personal data
2. Systematic monitoring of individuals
3. Large-scale profiling

### DPIA Template

```markdown
## Data Protection Impact Assessment

**Project:** [Aragora Deployment]
**Date:** [Date]
**Assessor:** [Name]

### 1. Processing Description
- Nature of processing: AI-assisted decision making
- Scope: [Number of users, data types]
- Context: [Business purpose]
- Purpose: [Specific use case]

### 2. Necessity Assessment
- Legal basis: [Contract/Legitimate interest/Consent]
- Data minimization: [Yes/No - justification]
- Retention limits: [Configured periods]

### 3. Risk Assessment
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data breach | Low | High | Encryption, access controls |
| Profiling harm | Medium | Medium | Human oversight, opt-out |

### 4. Measures
- Technical: [Encryption, access control, audit logging]
- Organizational: [Training, policies, DPO appointment]
```

## Breach Notification

### Detection

Aragora monitors for potential breaches via:
- Failed authentication attempts (`aragora/auth/`)
- Unusual access patterns (`aragora/observability/`)
- Data exfiltration signals (`aragora/rbac/audit.py`)

### Notification Procedure

1. **Detect** - Automated alerts for anomalies
2. **Contain** - Revoke compromised credentials
3. **Assess** - Determine scope and impact
4. **Notify** - DPA within 72 hours, users if high risk
5. **Document** - Record in breach register

```python
# Admin API for breach logging
await client.admin.log_breach_incident(
    incident_type="unauthorized_access",
    affected_users=["user-123"],
    data_categories=["email", "debate_content"],
    detection_time=datetime.now(),
    containment_actions=["token_revoked", "password_reset_required"],
)
```

## DPO Contact

For GDPR inquiries:
- Email: dpo@aragora.ai
- Data Protection Officer: [Name]
- EU Representative: [If applicable]

## Audit Checklist

- [ ] Data processing register maintained
- [ ] Privacy policy published and current
- [ ] Cookie consent implemented (if applicable)
- [ ] Data subject request procedures documented
- [ ] Subprocessor agreements in place
- [ ] Breach notification procedures tested
- [ ] Staff training completed
- [ ] DPIA conducted (if required)
- [ ] Data retention policies configured
- [ ] Cross-border transfer mechanisms in place
