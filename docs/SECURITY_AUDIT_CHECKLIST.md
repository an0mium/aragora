# Security Audit Checklist

This document provides a comprehensive security audit checklist for the Aragora codebase. It is designed for security auditors to systematically review the security posture of the application.

**Last Updated:** 2026-01-31
**Codebase Version:** See `git log -1 --format='%H'`

---

## Table of Contents

1. [Authentication Security](#1-authentication-security)
2. [Authorization Security](#2-authorization-security)
3. [Input Validation](#3-input-validation)
4. [Cryptography](#4-cryptography)
5. [API Security](#5-api-security)
6. [Data Protection](#6-data-protection)
7. [Dependency Security](#7-dependency-security)
8. [Logging and Monitoring](#8-logging-and-monitoring)

---

## Risk Level Legend

| Level | Description |
|-------|-------------|
| **CRITICAL** | Immediate security risk requiring urgent remediation |
| **HIGH** | Significant security risk that should be addressed promptly |
| **MEDIUM** | Moderate security risk that should be addressed in planned work |
| **LOW** | Minor security consideration for defense in depth |

---

## 1. Authentication Security

### 1.1 OIDC/OAuth 2.0 Implementation

**Risk Level:** HIGH

**File Locations:**
- `/aragora/auth/oidc.py` - OIDC provider implementation
- `/aragora/auth/sso.py` - Base SSO provider classes

**Current Implementation Status:**
- [x] PKCE (Proof Key for Code Exchange) support enabled by default
- [x] State parameter validation for CSRF protection
- [x] ID token validation via JWKS
- [x] Nonce generation for ID token replay protection
- [x] Algorithm restriction (RS256 by default, rejects insecure HS* algorithms)
- [x] Production mode enforcement (requires PyJWT library)
- [x] Domain restriction support for allowed email domains

**Security Controls:**
```python
# Algorithm validation - rejects insecure algorithms
insecure_algorithms = {"HS256", "HS384", "HS512", "none"}
# PKCE enabled by default
use_pkce: bool = True
# Production mode requires proper token validation
if _is_production_mode() and not HAS_JWT:
    raise SSOConfigurationError(...)
```

**Potential Risks:**
1. If `allowed_algorithms` is misconfigured to include symmetric algorithms
2. Development fallback to userinfo endpoint could bypass signature validation

**Recommendations:**
- [ ] Ensure `allowed_algorithms` only contains asymmetric algorithms (RS256, ES256)
- [ ] Verify `ARAGORA_ALLOW_DEV_AUTH_FALLBACK` is never set in production
- [ ] Review token expiration handling and refresh token rotation

---

### 1.2 SAML 2.0 Implementation

**Risk Level:** HIGH

**File Locations:**
- `/aragora/auth/saml.py` - SAML provider implementation

**Current Implementation Status:**
- [x] Uses defusedxml for XML parsing (XXE protection)
- [x] Signature validation enforced in production (requires python3-saml)
- [x] Strict mode enabled by default in onelogin settings
- [x] Certificate validation for IdP
- [x] Relay state validation

**Security Controls:**
```python
# XXE protection via defusedxml
import defusedxml.ElementTree as ET
# Production mode enforcement
if env in ("production", "prod", "staging", "stage"):
    raise SSOConfigurationError("python3-saml required...")
# Double-confirmation for unsafe mode
allow_unsafe = (
    os.getenv("ARAGORA_ALLOW_UNSAFE_SAML", "").lower() == "true"
    and os.getenv("ARAGORA_ALLOW_UNSAFE_SAML_CONFIRMED", "").lower() == "true"
)
```

**Potential Risks:**
1. Simplified parser does NOT validate signatures (only for testing)
2. Assertion Consumer Service (ACS) URL validation

**Recommendations:**
- [ ] Verify python3-saml is installed in production environments
- [ ] Audit IdP certificate rotation procedures
- [ ] Review assertion encryption settings (`want_assertions_encrypted`)

---

### 1.3 MFA (Multi-Factor Authentication)

**Risk Level:** HIGH

**File Locations:**
- `/aragora/server/middleware/mfa.py` - MFA enforcement middleware

**Current Implementation Status:**
- [x] TOTP/HOTP support via MFA decorators
- [x] Admin MFA enforcement (SOC 2 Control CC5-01)
- [x] MFA freshness checks for step-up authentication
- [x] Service account MFA bypass with audit logging
- [x] Backup codes support
- [x] Grace period for admin MFA adoption

**Security Controls:**
```python
# MFA enforcement decorators
@require_mfa  # All users
@require_admin_mfa  # Admin users
@require_admin_with_mfa  # Admin role + MFA
@require_mfa_fresh(max_age_minutes=15)  # Step-up auth
```

**Potential Risks:**
1. Service account MFA bypass could be exploited if bypass approval is not properly controlled
2. MFA backup codes storage security

**Recommendations:**
- [ ] Review MFA bypass approval workflow
- [ ] Verify backup codes are hashed, not stored in plaintext
- [ ] Ensure MFA secret storage uses encryption at rest

---

### 1.4 Account Lockout

**Risk Level:** MEDIUM

**File Locations:**
- `/aragora/auth/lockout.py` - Brute force protection

**Current Implementation Status:**
- [x] Exponential backoff lockout policy
- [x] Dual tracking by email and IP address
- [x] Redis backend for distributed deployments
- [x] Automatic fallback to in-memory storage
- [x] Admin unlock capability with logging

**Security Controls:**
```python
# Lockout thresholds
THRESHOLD_1 = 5   # 1 minute lockout
THRESHOLD_2 = 10  # 15 minute lockout
THRESHOLD_3 = 15  # 1 hour lockout
```

**Potential Risks:**
1. IP-based lockout can be bypassed with rotating IPs
2. In-memory storage not suitable for distributed deployments

**Recommendations:**
- [ ] Verify Redis is configured for production
- [ ] Consider device fingerprinting in addition to IP
- [ ] Review lockout bypass for legitimate users

---

### 1.5 Token Rotation

**Risk Level:** MEDIUM

**File Locations:**
- `/aragora/auth/token_rotation.py` - Token rotation policy enforcement

**Current Implementation Status:**
- [x] Usage-based rotation (after N uses)
- [x] Time-based rotation (max age)
- [x] IP change detection
- [x] Suspicious activity tracking
- [x] Token binding (IP, user agent)
- [x] Configurable policies (strict/standard/relaxed)

**Security Controls:**
```python
# Strict policy settings
max_uses=50
max_age_seconds=3600  # 1 hour
bind_to_ip=True
bind_to_user_agent=True
```

**Recommendations:**
- [ ] Ensure strict policy is used in production
- [ ] Verify suspicious activity alerts are monitored
- [ ] Review token revocation on security events

---

## 2. Authorization Security

### 2.1 RBAC v2 Implementation

**Risk Level:** HIGH

**File Locations:**
- `/aragora/rbac/models.py` - Permission, Role, RoleAssignment models
- `/aragora/rbac/checker.py` - Permission checking
- `/aragora/rbac/decorators.py` - Route protection decorators
- `/aragora/rbac/middleware.py` - HTTP middleware
- `/aragora/rbac/defaults.py` - Default roles and permissions

**Current Implementation Status:**
- [x] 50+ granular permissions across resource types
- [x] Role hierarchy with inheritance
- [x] Organization-scoped role assignments
- [x] API key scopes with permission restrictions
- [x] Permission caching for performance
- [x] Time-based role assignments with expiration

**Security Controls:**
```python
# Permission checking
@require_permission("debates:create")
@require_role("admin")
# API key scope validation
api_key_scope.allows_permission(permission_key)
```

**Potential Risks:**
1. Wildcard permissions ("*") could grant excessive access
2. Permission caching could delay revocation effects

**Recommendations:**
- [ ] Audit wildcard permission usage
- [ ] Review cache invalidation on permission changes
- [ ] Verify role inheritance doesn't create privilege escalation paths

---

### 2.2 RBAC Audit Logging

**Risk Level:** MEDIUM

**File Locations:**
- `/aragora/rbac/audit.py` - Authorization audit logging

**Current Implementation Status:**
- [x] All authorization decisions logged
- [x] HMAC-SHA256 event signing for integrity
- [x] Persistent storage via PersistentAuditHandler
- [x] Break-glass event tracking
- [x] SOC 2 Type II compliant audit trails

**Security Controls:**
```python
# Event signing
ARAGORA_AUDIT_SIGNING_KEY  # Required in production
compute_event_signature(event_data)  # HMAC-SHA256
verify_event_signature(event_data, signature)
```

**Potential Risks:**
1. Audit signing key must be properly managed
2. In-memory audit buffer could lose events on crash

**Recommendations:**
- [ ] Verify `ARAGORA_AUDIT_SIGNING_KEY` is set in production
- [ ] Review audit log retention and archival
- [ ] Test audit trail tampering detection

---

## 3. Input Validation

### 3.1 Request Validation Middleware

**Risk Level:** HIGH

**File Locations:**
- `/aragora/server/middleware/validation.py` - Request validation
- `/aragora/server/validation/schema.py` - JSON schemas

**Current Implementation Status:**
- [x] Schema-based body validation for critical endpoints
- [x] Query parameter range validation
- [x] Path segment validators
- [x] Max body size limits per route
- [x] Blocking mode enabled by default

**Security Controls:**
```python
# Validation rules
RouteValidation(
    pattern=r"^/api/(v1/)?auth/login$",
    method="POST",
    body_schema=USER_LOGIN_SCHEMA,
    max_body_size=5_000,  # 5KB
)
```

**Validated Endpoints:**
- Authentication (login, register, MFA)
- Organization management
- Billing operations

**Potential Risks:**
1. Not all endpoints may have validation rules
2. Schema validation bypass could lead to injection attacks

**Recommendations:**
- [ ] Audit `get_unvalidated_routes()` output
- [ ] Add validation rules for all POST/PUT endpoints
- [ ] Review JSON schema strictness (additionalProperties: false)

---

### 3.2 XSS Protection

**Risk Level:** HIGH

**File Locations:**
- `/aragora/server/middleware/xss_protection.py` - XSS protection utilities

**Current Implementation Status:**
- [x] HTML escaping via markupsafe
- [x] SafeHTMLBuilder for safe HTML construction
- [x] Attribute escaping for HTML attributes
- [x] Cookie security flags (HttpOnly, Secure, SameSite)
- [x] CSP nonce generation per request

**Security Controls:**
```python
# HTML escaping
escape_html(value)
escape_html_attribute(value)
# Cookie flags
HttpOnly=True, Secure=True, SameSite="Lax"
```

**Potential Risks:**
1. `mark_safe()` usage could introduce XSS if misused
2. CSP policy may need tightening for production

**Recommendations:**
- [ ] Audit all uses of `mark_safe()` in codebase
- [ ] Review CSP policy (`CSP_WEB_UI` vs `CSP_API_STRICT`)
- [ ] Enable CSP violation reporting

---

### 3.3 Security Headers

**Risk Level:** MEDIUM

**File Locations:**
- `/aragora/server/middleware/security.py` - Security headers and DoS protection

**Current Implementation Status:**
- [x] X-Content-Type-Options: nosniff
- [x] X-Frame-Options: DENY
- [x] X-XSS-Protection: 1; mode=block
- [x] Referrer-Policy: strict-origin-when-cross-origin
- [x] Permissions-Policy configured
- [x] HSTS in production
- [x] CSP with multiple modes (api/standard/development)

**Security Controls:**
```python
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}
```

**Recommendations:**
- [ ] Verify HSTS is enabled in production
- [ ] Review CSP for all response types
- [ ] Consider adding `Cross-Origin-*` headers

---

### 3.4 SQL Injection Prevention

**Risk Level:** CRITICAL

**File Locations:**
- Various storage modules in `/aragora/storage/`

**Current Implementation Status:**
- [ ] Requires audit - verify parameterized queries usage
- [ ] Requires audit - ORM usage patterns

**Recommendations:**
- [ ] Audit all raw SQL queries for parameterization
- [ ] Verify ORM configurations prevent raw SQL injection
- [ ] Review stored procedure usage if any

---

### 3.5 SSRF Protection

**Risk Level:** HIGH

**File Locations:**
- `/aragora/server/handlers/social/slack/security.py` - Slack SSRF protection

**Current Implementation Status:**
- [x] URL validation for Slack webhooks
- [x] Domain whitelist enforcement

**Security Controls:**
```python
SLACK_ALLOWED_DOMAINS = frozenset({"hooks.slack.com", "api.slack.com"})
def validate_slack_url(url: str) -> bool:
    # Validates HTTPS and allowed domains
```

**Recommendations:**
- [ ] Audit all external URL fetching for SSRF protection
- [ ] Review file:// and internal IP blocking

---

## 4. Cryptography

### 4.1 Encryption at Rest

**Risk Level:** CRITICAL

**File Locations:**
- `/aragora/security/encryption.py` - Application-level encryption

**Current Implementation Status:**
- [x] AES-256-GCM authenticated encryption
- [x] Key derivation via PBKDF2 (100,000 iterations)
- [x] Key rotation support with version tracking
- [x] Field-level encryption for sensitive data
- [x] Envelope encryption for large data
- [x] Encryption required in production mode

**Security Controls:**
```python
# Algorithm
EncryptionAlgorithm.AES_256_GCM
# Key derivation
kdf_iterations: int = 100000
kdf: PBKDF2_SHA256
# Production enforcement
if ENCRYPTION_REQUIRED or env in ("production", "prod", "staging", "stage"):
    return True
```

**Potential Risks:**
1. Master key management critical
2. Key rotation must be tested

**Recommendations:**
- [ ] Verify `ARAGORA_ENCRYPTION_KEY` is set in production
- [ ] Review key rotation procedures
- [ ] Audit all sensitive fields for encryption

---

### 4.2 Secret Management

**Risk Level:** CRITICAL

**File Locations:**
- `/aragora/config/secrets.py` - AWS Secrets Manager integration

**Current Implementation Status:**
- [x] AWS Secrets Manager integration
- [x] Strict mode in production (env vars blocked for critical secrets)
- [x] Audit logging for secret access (SOC 2)
- [x] Automatic cache expiration
- [x] Thread-safe secret access

**Critical Secrets Protected:**
```python
CRITICAL_SECRETS = frozenset({
    "JWT_SECRET_KEY", "JWT_REFRESH_SECRET",
    "ARAGORA_ENCRYPTION_KEY", "ARAGORA_AUDIT_SIGNING_KEY",
    "DATABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
    "STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET",
})
```

**Security Controls:**
```python
# Strict mode enforcement
if use_strict and is_critical:
    if env_value is not None:
        logger.warning("SECURITY: Critical secret found in env var...")
    raise SecretNotFoundError(name)
```

**Recommendations:**
- [ ] Verify AWS Secrets Manager is configured in production
- [ ] Review IAM permissions for secrets access
- [ ] Test secret rotation procedures

---

### 4.3 Password Hashing

**Risk Level:** HIGH

**Recommendations:**
- [ ] Audit password hashing algorithm (should be Argon2, bcrypt, or scrypt)
- [ ] Verify work factor/iterations are appropriate
- [ ] Review password storage locations

---

## 5. API Security

### 5.1 Rate Limiting

**Risk Level:** HIGH

**File Locations:**
- `/aragora/server/middleware/rate_limit/limiter.py` - Rate limiter

**Current Implementation Status:**
- [x] Per-IP rate limiting
- [x] Per-token rate limiting
- [x] Per-endpoint rate limiting
- [x] Combined rate limiting modes
- [x] LRU eviction for memory management
- [x] Token bucket algorithm
- [x] Trusted proxy configuration

**Security Controls:**
```python
# Default limits
DEFAULT_RATE_LIMIT = 60  # requests/minute
IP_RATE_LIMIT = 100      # requests/minute per IP
# Trusted proxy handling
ARAGORA_TRUSTED_PROXIES = "127.0.0.1,::1,localhost"
```

**Potential Risks:**
1. X-Forwarded-For spoofing if trusted proxies misconfigured
2. Rate limit bypass via distributed attacks

**Recommendations:**
- [ ] Verify trusted proxy configuration
- [ ] Review rate limits for sensitive endpoints
- [ ] Consider Redis-backed rate limiting for distributed deployments

---

### 5.2 Webhook Signature Verification

**Risk Level:** HIGH

**File Locations:**
- `/aragora/connectors/chat/webhook_security.py` - Unified webhook verification

**Current Implementation Status:**
- [x] HMAC-SHA256 verification (Slack, WhatsApp)
- [x] Ed25519 verification (Discord)
- [x] Timestamp validation (replay attack prevention)
- [x] Production enforcement (no bypass allowed)
- [x] Audit logging for verification attempts

**Security Controls:**
```python
# Production enforcement
if is_production_environment():
    return True  # Verification always required
# Timestamp validation
if abs(time.time() - request_time) > 300:  # 5 minute window
    return error("Request timestamp too old")
```

**Recommendations:**
- [ ] Verify all webhook endpoints use signature verification
- [ ] Review signing secret rotation procedures
- [ ] Test replay attack protection

---

### 5.3 API Key Management

**Risk Level:** HIGH

**File Locations:**
- `/aragora/rbac/models.py` - APIKeyScope model
- `/aragora/rbac/audit.py` - API key audit events

**Current Implementation Status:**
- [x] Scoped permissions per API key
- [x] Resource-level restrictions
- [x] Custom rate limits per key
- [x] Expiration support
- [x] IP whitelist support
- [x] Audit logging for key creation/revocation

**Security Controls:**
```python
@dataclass
class APIKeyScope:
    permissions: set[str]           # Limited permissions
    resources: dict[...]            # Specific resources
    rate_limit: int | None          # Custom rate limit
    expires_at: datetime | None     # Expiration
    ip_whitelist: set[str] | None   # IP restrictions
```

**Recommendations:**
- [ ] Review API key storage security
- [ ] Verify key hashing before storage
- [ ] Audit unused/long-lived API keys

---

## 6. Data Protection

### 6.1 PII Handling

**Risk Level:** CRITICAL

**File Locations:**
- `/aragora/rbac/models.py` - ResourceType.PII
- `/aragora/security/encryption.py` - Field-level encryption

**Current Implementation Status:**
- [x] PII resource type defined for RBAC
- [x] Field-level encryption available
- [x] Data classification support

**Recommendations:**
- [ ] Audit all PII data flows
- [ ] Verify PII is encrypted at rest
- [ ] Review data retention policies
- [ ] Test GDPR data export/deletion

---

### 6.2 Encryption in Transit

**Risk Level:** HIGH

**Current Implementation Status:**
- [x] HSTS header in production
- [x] Upgrade-insecure-requests in CSP

**Recommendations:**
- [ ] Verify TLS 1.2+ enforcement
- [ ] Review certificate management
- [ ] Test for SSL/TLS vulnerabilities

---

### 6.3 Database Security

**Risk Level:** HIGH

**Recommendations:**
- [ ] Review database connection encryption
- [ ] Audit database user permissions
- [ ] Verify backup encryption
- [ ] Review database access logging

---

## 7. Dependency Security

### 7.1 Python Dependencies

**Risk Level:** MEDIUM

**Recommendations:**
- [ ] Run `pip-audit` or `safety` scan
- [ ] Review pinned version constraints
- [ ] Check for outdated dependencies
- [ ] Verify cryptography library version

**Key Security Dependencies:**
```
PyJWT - JWT handling
cryptography - Encryption
markupsafe - XSS protection
defusedxml - XXE protection
python3-saml - SAML (optional)
pynacl - Ed25519 (optional)
```

---

### 7.2 Dependency Update Policy

**Recommendations:**
- [ ] Establish regular dependency update schedule
- [ ] Configure Dependabot or similar
- [ ] Review dependency licenses

---

## 8. Logging and Monitoring

### 8.1 Audit Logging

**Risk Level:** HIGH

**File Locations:**
- `/aragora/server/middleware/audit_logger.py` - HTTP audit logging
- `/aragora/rbac/audit.py` - RBAC audit logging
- `/aragora/audit/` - Unified audit system

**Current Implementation Status:**
- [x] Structured audit entries with unique IDs
- [x] Tamper-evident logging with hash chains
- [x] Multiple backends (file, database, memory)
- [x] Context propagation (request ID, session ID)
- [x] Pre-defined audit helpers for common operations
- [x] HMAC signing for integrity verification

**Audit Categories:**
```python
class AuditCategory(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION = "configuration"
    ADMINISTRATIVE = "administrative"
    SECURITY = "security"
    SYSTEM = "system"
```

**Recommendations:**
- [ ] Verify audit logs are protected from tampering
- [ ] Review log retention policies
- [ ] Test hash chain integrity verification
- [ ] Ensure sensitive data is not logged in plaintext

---

### 8.2 Security Event Monitoring

**Risk Level:** MEDIUM

**Recommendations:**
- [ ] Review alerting on security events
- [ ] Verify SIEM integration if applicable
- [ ] Test incident response procedures
- [ ] Review failed authentication monitoring

---

### 8.3 Error Handling

**Risk Level:** MEDIUM

**Recommendations:**
- [ ] Verify errors don't leak sensitive information
- [ ] Review stack trace exposure in production
- [ ] Test error responses for information disclosure

---

## Audit Summary Template

| Category | Items Checked | Pass | Fail | N/A | Notes |
|----------|--------------|------|------|-----|-------|
| Authentication | | | | | |
| Authorization | | | | | |
| Input Validation | | | | | |
| Cryptography | | | | | |
| API Security | | | | | |
| Data Protection | | | | | |
| Dependencies | | | | | |
| Logging | | | | | |

---

## Post-Audit Actions

### Critical Findings
_List any CRITICAL findings requiring immediate action_

### High Priority Findings
_List HIGH priority findings for prompt remediation_

### Remediation Timeline
| Finding | Priority | Owner | Due Date | Status |
|---------|----------|-------|----------|--------|
| | | | | |

---

## Appendix: File Reference

### Authentication Modules
- `/aragora/auth/oidc.py` - OIDC provider
- `/aragora/auth/saml.py` - SAML provider
- `/aragora/auth/sso.py` - Base SSO classes
- `/aragora/auth/lockout.py` - Account lockout
- `/aragora/auth/token_rotation.py` - Token rotation

### RBAC Modules
- `/aragora/rbac/models.py` - Core models
- `/aragora/rbac/checker.py` - Permission checking
- `/aragora/rbac/decorators.py` - Route decorators
- `/aragora/rbac/middleware.py` - HTTP middleware
- `/aragora/rbac/audit.py` - Audit logging
- `/aragora/rbac/defaults.py` - Default roles

### Security Middleware
- `/aragora/server/middleware/security.py` - Security headers, DoS protection
- `/aragora/server/middleware/xss_protection.py` - XSS protection
- `/aragora/server/middleware/validation.py` - Input validation
- `/aragora/server/middleware/mfa.py` - MFA enforcement
- `/aragora/server/middleware/audit_logger.py` - Audit logging
- `/aragora/server/middleware/rate_limit/` - Rate limiting

### Cryptography
- `/aragora/security/encryption.py` - AES-256-GCM encryption
- `/aragora/config/secrets.py` - AWS Secrets Manager

### Webhook Security
- `/aragora/connectors/chat/webhook_security.py` - Signature verification
- `/aragora/server/handlers/social/slack/security.py` - Slack SSRF protection

---

*This checklist should be reviewed and updated regularly as the codebase evolves.*
