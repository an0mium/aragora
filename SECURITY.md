# Security Policy

## Reporting Vulnerabilities

**Email:** security@aragora.ai

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

### Responsible Disclosure

We follow responsible disclosure practices. Please allow up to 90 days for us to address vulnerabilities before public disclosure. We commit to:
- Acknowledging your report within 48 hours
- Providing an initial assessment within 7 days
- Keeping you informed of our progress
- Crediting you in our security advisories (unless you prefer anonymity)

## Supported Versions

| Version | Supported | Notes |
|---------|-----------|-------|
| 1.0.x   | Yes       | Current stable release |
| 0.8.x   | Security fixes only | Upgrade recommended |
| < 0.8   | No        | Unsupported |

---

## Security Features

### Authentication

#### JWT Authentication
- Token-based authentication with configurable expiry
- Tokens stored in HTTP-only cookies or Authorization header
- Refresh token rotation for long-lived sessions
- Token revocation via dual-layer blacklist (memory + Redis/SQLite)

#### Multi-Factor Authentication (MFA)
- TOTP-based MFA using RFC 6238 (Google Authenticator compatible)
- 10 backup recovery codes generated at setup
- MFA required option for sensitive operations
- Admin-assisted MFA reset capability

#### Account Lockout
- Automatic lockout after failed login attempts:
  - 5 attempts: 1-minute lockout
  - 10 attempts: 15-minute lockout
  - 15+ attempts: 1-hour lockout
- Independent tracking by email AND IP address
- Redis-backed for distributed deployments
- Admin unlock capability via API

#### Session Management
- Configurable session timeout (default: 24 hours)
- Session invalidation on password change
- Concurrent session limits (configurable)
- Session activity logging

### Authorization

- Role-based access control (RBAC)
- API token scopes for fine-grained permissions
- Endpoint-level authorization checks
- Admin role required for sensitive operations

---

## Encryption

### At Rest

| Storage | Encryption Method | Configuration |
|---------|-------------------|---------------|
| SQLite | OS-level or SQLCipher | `ARAGORA_SQLITE_ENCRYPTION=1` |
| PostgreSQL | TDE (Transparent Data Encryption) | Database-level configuration |
| Redis | AUTH + optional TLS | `REDIS_URL=rediss://...` for TLS |

**Key Management:**
- Encryption keys stored in environment variables (not in code/config files)
- Support for external secret managers (AWS Secrets Manager, HashiCorp Vault)
- Key rotation without service interruption via Kubernetes secrets

### In Transit

- **HTTPS Required**: All API endpoints require TLS 1.2+
- **WebSocket Security**: WSS (WebSocket Secure) for real-time connections
- **Internal Traffic**: mTLS for service mesh in Kubernetes deployments
- **Certificate Management**: cert-manager integration for automatic TLS certificates

### Sensitive Data Handling

- API keys never logged or exposed in error messages
- Password hashing using bcrypt (cost factor 12)
- Secrets redacted in telemetry and observability data
- PII minimization in audit logs

---

## Password Policy

| Requirement | Value |
|-------------|-------|
| Minimum length | 12 characters |
| Complexity | At least one uppercase, lowercase, number |
| Hash algorithm | bcrypt (cost factor 12) |
| History | No reuse of last 5 passwords |
| Expiry | Optional, configurable |

**Password Storage:**
- Never stored in plaintext
- Bcrypt with random salt
- Timing-safe comparison to prevent timing attacks

---

## Rate Limiting

Rate limiting is enabled by default to prevent abuse:

| Endpoint Type | Default Limit | Configuration |
|---------------|---------------|---------------|
| Authentication | 10/minute per IP | `ARAGORA_RATE_LIMIT_AUTH` |
| Debate creation | 30/minute per user | `ARAGORA_RATE_LIMIT_DEBATES` |
| API general | 100/minute per token | `ARAGORA_RATE_LIMIT_DEFAULT` |
| File upload | 5/minute per user | `ARAGORA_RATE_LIMIT_UPLOAD` |

**Response Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704134400
Retry-After: 60  (on 429 response)
```

---

## Input Validation

### Request Limits

| Parameter | Limit |
|-----------|-------|
| Request body | 100 MB |
| JSON payload | 10 MB |
| Multipart parts | 100 maximum |
| WebSocket message | 64 KB |
| JSON parse timeout | 5 seconds |

### Validation

- All user input sanitized before processing
- Path traversal protection for file operations
- SQL injection prevention via parameterized queries
- XSS prevention via output encoding (MarkupSafe)
- SSRF protection for external URL fetching

---

## Security Headers

All responses include security headers:

| Header | Value |
|--------|-------|
| `X-Frame-Options` | `DENY` |
| `X-Content-Type-Options` | `nosniff` |
| `X-XSS-Protection` | `1; mode=block` |
| `Referrer-Policy` | `strict-origin-when-cross-origin` |
| `Content-Security-Policy` | Configurable per deployment |

**CORS Configuration:**
- Whitelist-based origin validation
- No wildcard (`*`) in production
- Configurable via `ARAGORA_ALLOWED_ORIGINS`

---

## Incident Response

### Detection
1. Prometheus alerts for anomalous patterns
2. Sentry error tracking and aggregation
3. Audit log monitoring
4. Rate limit breach notifications

### Containment
1. Automatic rate limiting escalation
2. Account lockout for suspicious activity
3. IP blocking for severe abuse
4. Circuit breaker for failing services

### Recovery
1. Database backups (daily, 14-day retention)
2. Point-in-time recovery capability
3. RTO target: < 4 hours
4. RPO target: < 1 hour

### Post-Incident
1. Post-mortem required within 48 hours
2. Root cause analysis documentation
3. Security advisory publication (if applicable)
4. Process improvement implementation

---

## Audit Logging

All security-relevant events are logged:

| Event | Data Captured |
|-------|---------------|
| Login attempts | User, IP, timestamp, success/failure |
| MFA events | Setup, enable, disable, verification |
| Permission changes | User, role, admin who made change |
| Data access | Resource type, action, user |
| Admin actions | Action type, target, timestamp |

**Log Retention:**
- Security logs: 90 days minimum
- Audit trail: 1 year
- Configurable via `ARAGORA_AUDIT_RETENTION_DAYS`

---

## Data Protection

### GDPR Compliance
- Data minimization: Only collect necessary data
- Right to erasure: User deletion removes all associated data
- Data portability: Export user data via API
- Consent tracking: Explicit opt-in for optional features

### CCPA Compliance
- Do Not Sell: No sale of personal information
- Access rights: Users can request their data
- Deletion rights: Complete data removal on request

### Data Retention
| Data Type | Retention Period |
|-----------|------------------|
| User accounts | Until deletion requested |
| Debate content | Configurable (default: indefinite) |
| Audit logs | 1 year |
| Session data | 30 days after expiry |
| Backup data | 14 days |

---

## Dependency Security

### Scanning
- **Bandit**: Static security analysis for Python
- **pip-audit**: Vulnerability scanning for dependencies
- **npm audit**: Frontend dependency scanning
- **Gitleaks**: Secret detection in code
- **TruffleHog**: Additional secret scanning
- **CodeQL**: GitHub Advanced Security scanning

### Update Policy
- Security patches applied within 7 days
- Critical vulnerabilities addressed within 24 hours
- Dependency updates reviewed weekly

### Known Secure Versions
```toml
# pyproject.toml - security-aware versions
aiohttp>=3.13.3        # CVE fixes
jinja2>=3.1.6          # CVE-2024-56326 fix
urllib3>=2.6.3         # CVE fixes
bcrypt>=4.0            # Secure password hashing
markupsafe>=2.1.0      # XSS prevention
```

---

## Deployment Security

### Container Security
- Non-root user in containers
- Read-only root filesystem
- Security context constraints
- Resource limits enforced

### Kubernetes Security
- Pod Security Standards (restricted profile)
- Network policies for traffic isolation
- RBAC for cluster access
- Secrets encryption at rest

### Environment Variables
- Sensitive values via Kubernetes Secrets
- External secret managers supported
- Never commit secrets to version control
- `.env.example` provided without sensitive values

---

## Security Testing

### Automated
- Unit tests for auth flows
- Integration tests for security features
- Load tests for rate limiting
- Mutation testing for critical modules

### Manual
- Penetration testing (annual)
- Code review for security-sensitive changes
- Threat modeling for new features

---

## Contact

- **Security Issues:** security@aragora.ai
- **General Support:** support@aragora.ai
- **GitHub Issues:** https://github.com/an0mium/aragora/security/advisories
