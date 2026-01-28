# Security Testing Guide

This document describes security testing procedures for the Aragora platform.

## Automated Security Scans

### CI/CD Pipeline Security

The following security scans run automatically on every pull request and weekly:

| Tool | Purpose | Trigger |
|------|---------|---------|
| **CodeQL** | Static analysis for Python and JavaScript | PR, Push to main, Weekly |
| **Bandit** | Python security linter (OWASP, CWE) | PR, Push to main |
| **Safety** | Python dependency vulnerability scanner | PR, Push to main |
| **pip-audit** | Python package security audit | PR, Push to main |
| **npm audit** | Node.js dependency vulnerabilities | PR, Push to main |
| **Gitleaks** | Secret detection in git history | PR, Push to main |
| **TruffleHog** | Verified secret scanning | PR, Push to main |

### SBOM Generation

Software Bill of Materials (SBOM) is generated on every release:

- **Format**: CycloneDX (JSON and XML), SPDX (for containers)
- **Coverage**: Python packages, Node.js packages, Container images
- **Vulnerability scanning**: Grype scans all SBOMs for known CVEs

## Manual Security Testing

### Pre-Release Checklist

Before each release, perform the following manual security checks:

#### 1. Authentication Testing

```bash
# Test rate limiting on login endpoints
for i in {1..20}; do
  curl -X POST http://localhost:8080/api/v1/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"test@example.com","password":"wrong"}' \
    -w "%{http_code}\n" -o /dev/null -s
done
# Expected: 429 after rate limit threshold

# Test session timeout
# 1. Login and get token
# 2. Wait for session timeout (default: 30 minutes)
# 3. Verify token is rejected

# Test MFA enforcement
# 1. Enable MFA for user
# 2. Verify login requires TOTP code
# 3. Test backup codes
```

#### 2. Authorization Testing

```bash
# Test RBAC permission enforcement
# Attempt access with insufficient permissions
curl -X GET http://localhost:8080/api/v1/admin/users \
  -H "Authorization: Bearer $VIEWER_TOKEN"
# Expected: 403 Forbidden

# Test tenant isolation
# User from tenant A should not access tenant B resources
curl -X GET http://localhost:8080/api/v1/debates/tenant-b-debate-id \
  -H "Authorization: Bearer $TENANT_A_TOKEN"
# Expected: 404 Not Found (not 403, to prevent enumeration)
```

#### 3. Input Validation Testing

```bash
# Test SQL injection prevention
curl -X POST http://localhost:8080/api/v1/debates \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"topic":"SELECT * FROM users--"}'
# Expected: Treated as literal string, no SQL execution

# Test XSS prevention
curl -X POST http://localhost:8080/api/v1/debates \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"topic":"<script>alert(1)</script>"}'
# Expected: Script tags escaped in response

# Test path traversal prevention
curl -X GET "http://localhost:8080/api/v1/files/../../../etc/passwd" \
  -H "Authorization: Bearer $TOKEN"
# Expected: 400 Bad Request or 404 Not Found
```

#### 4. API Security Testing

```bash
# Test CORS configuration
curl -X OPTIONS http://localhost:8080/api/v1/debates \
  -H "Origin: https://malicious-site.com" \
  -H "Access-Control-Request-Method: POST" \
  -v
# Expected: No Access-Control-Allow-Origin for unauthorized origins

# Test security headers
curl -I http://localhost:8080/api/v1/health
# Expected headers:
#   X-Content-Type-Options: nosniff
#   X-Frame-Options: DENY
#   Content-Security-Policy: default-src 'self'
#   Strict-Transport-Security: max-age=31536000; includeSubDomains
```

### Penetration Testing

For production deployments, engage a third-party security firm for annual penetration testing. Focus areas:

1. **API Security**: REST and WebSocket endpoints
2. **Authentication**: Session management, token handling
3. **Authorization**: RBAC bypass attempts
4. **Data Protection**: Encryption at rest and in transit
5. **Infrastructure**: Cloud configuration, network security

## Security Configuration

### Environment Variables

Ensure the following security-related environment variables are set:

```bash
# Authentication
ARAGORA_JWT_SECRET=<32+ character random string>
ARAGORA_JWT_ALGORITHM=HS256
ARAGORA_SESSION_TIMEOUT_MINUTES=30

# Rate Limiting
ARAGORA_RATE_LIMIT_ENABLED=true
ARAGORA_RATE_LIMIT_REQUESTS=100
ARAGORA_RATE_LIMIT_WINDOW_SECONDS=60

# CORS
ARAGORA_ALLOWED_ORIGINS=https://your-domain.com

# Encryption
ARAGORA_ENCRYPTION_KEY=<32-byte base64 encoded key>

# MFA
ARAGORA_MFA_ISSUER=Aragora
ARAGORA_MFA_REQUIRED=true  # For admin roles
```

### Security Headers

The server automatically adds these security headers (see `aragora/server/middleware/security.py`):

```python
{
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Content-Security-Policy": "default-src 'self'",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}
```

## Vulnerability Response

### Reporting Security Issues

Report security vulnerabilities via:
- Email: security@aragora.ai
- GitHub Security Advisories (private)

### Response SLAs

| Severity | Initial Response | Fix Target |
|----------|------------------|------------|
| Critical | 4 hours | 24 hours |
| High | 24 hours | 7 days |
| Medium | 72 hours | 30 days |
| Low | 1 week | 90 days |

### Disclosure Policy

- Coordinated disclosure with 90-day deadline
- Security advisories published after fix is available
- Credit given to reporters (unless they prefer anonymity)

## Compliance

See [SOC2_EVIDENCE.md](./SOC2_EVIDENCE.md) for compliance documentation and evidence collection procedures.
