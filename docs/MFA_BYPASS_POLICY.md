# MFA Bypass Policy

> **Last Updated:** 2026-01-31
> **Classification:** Internal Security Policy
> **Compliance:** SOC 2 Control CC5-01

This document defines the Multi-Factor Authentication (MFA) bypass policy for Aragora, including when bypass is permitted, security controls, and audit requirements.

## Table of Contents

- [Overview](#overview)
- [When MFA Bypass is Permitted](#when-mfa-bypass-is-permitted)
- [Service Account Bypass Workflow](#service-account-bypass-workflow)
- [Security Controls](#security-controls)
- [Audit Requirements](#audit-requirements)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Risk Assessment](#risk-assessment)

---

## Overview

MFA is **required by default** for all administrative operations in Aragora. However, certain automated systems (service accounts) may require MFA bypass to function correctly. This policy governs when and how MFA bypass is permitted.

### Key Principles

1. **MFA bypass is an exception, not the rule**
2. **All bypasses require explicit approval from an administrator**
3. **All bypass attempts are audited**
4. **Bypasses expire automatically (default: 90 days)**

---

## When MFA Bypass is Permitted

MFA bypass is **only permitted** for:

| Scenario | Justification | Approval Required |
|----------|---------------|-------------------|
| Service accounts | Automated API integrations cannot perform MFA | Admin approval |
| CI/CD pipelines | Deployment automation | Admin approval |
| Bot accounts | Automated monitoring/alerting | Admin approval |

MFA bypass is **never permitted** for:

- Human user accounts
- Admin accounts (even temporary)
- Accounts with elevated privileges
- Shared accounts

---

## Service Account Bypass Workflow

### Prerequisites

1. Account must be flagged as `is_service_account=True`
2. Account must have a documented business justification
3. Admin must approve the bypass

### Approval Process

```python
from aragora.billing.models import User

# 1. Create service account (during provisioning)
service_account = User(
    email="ci-pipeline@service.aragora.io",
    is_service_account=True,
    name="CI Pipeline Service Account",
)

# 2. Admin approves MFA bypass (after review)
service_account.approve_mfa_bypass(
    approved_by="admin-user-id",
    reason="CI/CD pipeline for production deployments",
    expires_days=90,  # Auto-expire after 90 days
)
```

### Bypass Validation

Three conditions must be met for a valid MFA bypass:

```python
def _has_valid_mfa_bypass(user: User) -> bool:
    """Check if user has valid MFA bypass."""
    # 1. Must be a service account
    if not user.is_service_account:
        return False

    # 2. Must have approved bypass
    if not user.mfa_bypass_approved_at:
        return False

    # 3. Must not be expired
    if user.mfa_bypass_expires_at:
        if datetime.now(timezone.utc) > user.mfa_bypass_expires_at:
            return False

    return True
```

### Revocation

Bypasses can be revoked at any time:

```python
service_account.revoke_mfa_bypass(
    revoked_by="admin-user-id",
    reason="Service decommissioned",
)
```

---

## Security Controls

### 1. Automatic Expiration

All MFA bypasses expire after a configurable period (default: 90 days). This ensures:

- Regular review of bypass necessity
- Automatic cleanup of orphaned service accounts
- Reduced attack surface from stale credentials

### 2. Audit Logging

Every MFA bypass attempt is logged via `audit_security()`:

```python
audit_security(
    event_type="mfa_bypass",
    actor_id=user.id,
    ip_address=request_ip,
    reason="service_account_bypass",
    details={
        "operation": operation_type,
        "bypass_reason": user.mfa_bypass_reason,
        "expires_at": str(user.mfa_bypass_expires_at),
    },
)
```

### 3. Grace Period for New Admins

New admin users have a configurable grace period (default: 7 days) to set up MFA before enforcement begins. This prevents lockout during onboarding.

### 4. Step-Up Authentication

Sensitive operations can require "fresh" MFA verification (within the last 15 minutes), even for users who have already authenticated:

```python
@require_mfa_fresh(max_age_seconds=900)  # 15 minutes
async def delete_organization(...):
    # Service accounts can still bypass, but audit is enhanced
    ...
```

---

## Audit Requirements

### Required Audit Events

| Event | Trigger | Retention |
|-------|---------|-----------|
| `mfa_bypass` | Service account bypasses MFA | 2 years |
| `mfa_bypass_approved` | Admin approves bypass | 2 years |
| `mfa_bypass_revoked` | Admin revokes bypass | 2 years |
| `mfa_bypass_expired` | Bypass auto-expires | 1 year |

### Audit Log Fields

All MFA-related audit logs must include:

- `actor_id`: User/service account ID
- `ip_address`: Request origin IP
- `timestamp`: UTC timestamp
- `operation`: What operation was attempted
- `bypass_reason`: Why bypass was granted
- `approved_by`: Who approved the bypass (if applicable)

### Compliance Reporting

Run periodic reports to identify:

```sql
-- Service accounts with active MFA bypass
SELECT id, email, mfa_bypass_reason, mfa_bypass_expires_at
FROM users
WHERE is_service_account = true
  AND mfa_bypass_approved_at IS NOT NULL
  AND (mfa_bypass_expires_at IS NULL OR mfa_bypass_expires_at > NOW());

-- Bypasses expiring soon (next 30 days)
SELECT id, email, mfa_bypass_expires_at
FROM users
WHERE mfa_bypass_expires_at BETWEEN NOW() AND NOW() + INTERVAL '30 days';
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_SECURITY_ADMIN_MFA_REQUIRED` | `true` | Enable MFA enforcement for admins |
| `ARAGORA_SECURITY_MFA_GRACE_PERIOD_DAYS` | `7` | Grace period for new admins |

### Settings Class

```python
from aragora.config.settings import SecuritySettings

settings = SecuritySettings()
print(settings.admin_mfa_required)      # True
print(settings.admin_mfa_grace_period_days)  # 7
```

---

## API Reference

### User Model Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_service_account` | `bool` | Machine/bot account indicator |
| `mfa_bypass_reason` | `str | None` | 'service_account', 'api_integration' |
| `mfa_bypass_approved_by` | `str | None` | User ID who approved bypass |
| `mfa_bypass_approved_at` | `datetime | None` | When bypass was approved |
| `mfa_bypass_expires_at` | `datetime | None` | Auto-expire timestamp |

### User Model Methods

```python
# Check if bypass is currently valid
user.is_mfa_bypass_valid() -> bool

# Approve bypass (admin action)
user.approve_mfa_bypass(
    approved_by: str,
    reason: str,
    expires_days: int = 90,
) -> None

# Revoke bypass (admin action)
user.revoke_mfa_bypass(
    revoked_by: str,
    reason: str,
) -> None
```

### Middleware Decorators

```python
from aragora.server.middleware.mfa import (
    require_mfa,           # All users must have MFA
    require_admin_mfa,     # Admins/owners must have MFA
    require_admin_with_mfa,  # Must be admin AND have MFA
    require_mfa_fresh,     # Recent MFA verification required
)

# Service accounts with valid bypass can skip MFA checks
# but all attempts are logged for audit
```

---

## Risk Assessment

### Identified Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Compromised service account | HIGH | Automatic expiration, audit logging, least privilege |
| Bypass approval without review | MEDIUM | Require documented justification, dual approval (future) |
| Stale service accounts | MEDIUM | 90-day expiration, periodic review alerts |
| Audit log tampering | HIGH | Immutable audit storage, log forwarding |

### Recommended Mitigations

1. **Quarterly Review**: Review all active MFA bypasses quarterly
2. **Least Privilege**: Service accounts should have minimal required permissions
3. **Monitoring**: Alert on unusual service account activity
4. **Rotation**: Rotate service account credentials regularly

---

## Related Documentation

- [SECURITY.md](./SECURITY.md) - Security overview
- [SECURITY_AUDIT_CHECKLIST.md](./SECURITY_AUDIT_CHECKLIST.md) - Audit checklist
- [SECURITY_RUNTIME.md](./SECURITY_RUNTIME.md) - Runtime monitoring

---

## Change Log

| Date | Author | Change |
|------|--------|--------|
| 2026-01-31 | System | Initial policy documentation |
