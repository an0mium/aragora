# Runtime Security Monitoring Guide

This document describes security monitoring, incident response, and secret management for Aragora in production.

## Security Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Layers                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Network Security                                  │
│  ├── TLS 1.3 encryption                                     │
│  ├── CORS policy enforcement                                │
│  └── Rate limiting (IP-based + token-based)                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Authentication & Authorization                    │
│  ├── JWT token validation                                   │
│  ├── Token versioning (revocation support)                  │
│  └── Role-based access control                              │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Input Validation                                  │
│  ├── SQL injection prevention (parameterized queries)       │
│  ├── Path traversal protection                              │
│  └── Request size limits                                    │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Runtime Monitoring                                │
│  ├── Security event logging                                 │
│  ├── Anomaly detection                                      │
│  └── Audit trails                                           │
└─────────────────────────────────────────────────────────────┘
```

## Security Event Logging

### Event Categories

| Category | Log Level | Examples |
|----------|-----------|----------|
| Authentication | INFO/WARN | Login success, login failure, token refresh |
| Authorization | WARN | Access denied, role mismatch |
| Input Validation | WARN | Invalid input, SQL injection attempt |
| Rate Limiting | WARN | Rate limit exceeded, IP blocked |
| System | ERROR | Config error, dependency failure |

### Log Format

```json
{
  "timestamp": "2026-01-14T00:00:00.000Z",
  "level": "WARN",
  "category": "authentication",
  "event": "login_failure",
  "details": {
    "ip": "192.168.1.100",
    "user_agent": "Mozilla/5.0...",
    "reason": "invalid_credentials",
    "attempts": 3
  },
  "trace_id": "abc123",
  "request_id": "req-456"
}
```

### Enabling Security Logging

```python
# In your server configuration
import logging

# Configure security logger
security_logger = logging.getLogger("aragora.security")
security_logger.setLevel(logging.INFO)

# Add structured handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
    '"category":"%(name)s","message":"%(message)s"}'
))
security_logger.addHandler(handler)
```

### Environment Variables

```bash
# Security logging configuration
ARAGORA_SECURITY_LOG_LEVEL=INFO
ARAGORA_SECURITY_LOG_FILE=/var/log/aragora/security.log
ARAGORA_AUDIT_ENABLED=true
ARAGORA_AUDIT_RETENTION_DAYS=90
```

## Security Event Debates

Critical findings can trigger a remediation debate via the security events
emitter. This is managed by `aragora/events/security_events.py` and used by
codebase security scans.

Defaults:
- Auto-debate threshold: critical severity
- Debate timeout: 300 seconds
- Consensus: majority with convergence detection

Disable auto-debate by constructing `SecurityEventEmitter(enable_auto_debate=False)`.

## Secret Management

### API Key Rotation

**Rotation Schedule:**
- Production API keys: Every 90 days
- Service accounts: Every 180 days
- Emergency rotation: Immediate on compromise

**Rotation Procedure:**

1. Generate new key:
   ```bash
   # Generate new API key
   python -c "import secrets; print(f'ara_{secrets.token_hex(32)}')"
   ```

2. Update environment:
   ```bash
   # Add new key (keep old key active)
   export ANTHROPIC_API_KEY_NEW="new-key-here"
   ```

3. Deploy with dual-key support:
   ```bash
   # Gradual rollout
   kubectl set env deployment/aragora ANTHROPIC_API_KEY=$NEW_KEY
   ```

4. Verify and remove old key:
   ```bash
   # After 24h verification period
   kubectl set env deployment/aragora ANTHROPIC_API_KEY_OLD-
   ```

### Secret Storage

| Environment | Storage Method | Access Control |
|-------------|---------------|----------------|
| Development | `.env` file (gitignored) | Developer only |
| Staging | AWS Secrets Manager | IAM roles |
| Production | AWS Secrets Manager + rotation | IAM + MFA |

### AWS Secrets Manager Integration

```python
# aragora/config/secrets.py already supports this
from aragora.config.secrets import get_secret

# Automatically fetches from AWS Secrets Manager in production
api_key = get_secret("ANTHROPIC_API_KEY")
```

## Security SLA Definitions

### Response Time Targets

| Severity | Description | Response Time | Resolution Time |
|----------|-------------|---------------|-----------------|
| Critical | Active exploit, data breach | 15 minutes | 4 hours |
| High | Vulnerability discovered | 1 hour | 24 hours |
| Medium | Security misconfiguration | 4 hours | 72 hours |
| Low | Security enhancement | 24 hours | 1 week |

### Incident Classification

**Critical:**
- Active data exfiltration
- Unauthorized admin access
- Service compromise

**High:**
- Exploitable vulnerability (no active exploit)
- Authentication bypass
- Privilege escalation

**Medium:**
- Information disclosure
- Missing security headers
- Weak encryption

**Low:**
- Security best practice violations
- Documentation gaps
- Minor misconfigurations

## Incident Response Procedures

### Phase 1: Detection (0-15 min)

1. **Alert Triggered**
   - PagerDuty notification
   - Slack #security-alerts channel
   - Email to security@aragora.ai

2. **Initial Assessment**
   ```bash
   # Check recent security events
   kubectl logs -l app=aragora --since=1h | grep -i security

   # Check rate limiting status
   curl http://localhost:8080/api/system/rate-limits
   ```

3. **Severity Classification**
   - Determine impact scope
   - Identify affected systems
   - Classify per SLA definitions

### Phase 2: Containment (15-60 min)

1. **Immediate Actions**
   ```bash
   # Block suspicious IP
   kubectl exec -it aragora-pod -- \
     python -c "from aragora.server.rate_limit import block_ip; block_ip('1.2.3.4')"

   # Revoke compromised tokens
   curl -X POST http://localhost:8080/api/auth/revoke-all \
     -H "Authorization: Bearer $ADMIN_TOKEN"
   ```

2. **Evidence Preservation**
   ```bash
   # Export logs
   kubectl logs -l app=aragora --since=24h > incident-logs.txt

   # Snapshot database
   pg_dump aragora > incident-snapshot.sql
   ```

3. **Communication**
   - Update incident channel
   - Notify stakeholders
   - Prepare status page update

### Phase 3: Eradication (1-4 hours)

1. **Root Cause Analysis**
   - Review security logs
   - Trace attack vector
   - Identify vulnerability

2. **Remediation**
   - Apply security patch
   - Update configurations
   - Rotate compromised credentials

3. **Verification**
   ```bash
   # Security scan
   bandit -r aragora/ -ll -ii --severity-level high

   # Dependency check
   safety check --full-report
   ```

### Phase 4: Recovery (4-24 hours)

1. **Service Restoration**
   - Gradual traffic restoration
   - Monitor for anomalies
   - Verify functionality

2. **Post-Incident Review**
   - Document timeline
   - Identify improvements
   - Update runbooks

## Security Monitoring Checklist

### Daily Checks

- [ ] Review security event logs
- [ ] Check rate limiting metrics
- [ ] Verify backup completion
- [ ] Monitor authentication failures

### Weekly Checks

- [ ] Review access logs for anomalies
- [ ] Check certificate expiration dates
- [ ] Audit user permissions
- [ ] Review dependency vulnerabilities

### Monthly Checks

- [ ] Security patch review
- [ ] Penetration testing results
- [ ] Access control audit
- [ ] Secret rotation verification

## Alerting Rules

### PagerDuty Integration

```yaml
# Alert rules for security events
alerts:
  - name: high_auth_failure_rate
    condition: rate(auth_failures[5m]) > 10
    severity: high
    action: page

  - name: rate_limit_exceeded
    condition: rate(rate_limit_hits[1m]) > 100
    severity: medium
    action: slack

  - name: sql_injection_attempt
    condition: count(sql_injection_blocked[5m]) > 0
    severity: critical
    action: page
```

### Grafana Dashboard

Key metrics to display:
- Authentication success/failure rates
- Rate limiting triggers by IP
- Token refresh frequency
- Security event timeline
- Active sessions count

## Compliance Considerations

### Data Protection

- PII handling: Encrypted at rest and in transit
- Data retention: 90 days for logs, configurable for user data
- Right to erasure: Supported via `/api/user/delete` endpoint

### Audit Requirements

- All API calls logged with user ID and timestamp
- Admin actions require MFA
- Audit logs immutable (append-only)
- Retention: 1 year minimum

## Security Contacts

| Role | Contact | Escalation Time |
|------|---------|-----------------|
| On-call Engineer | PagerDuty | Immediate |
| Security Lead | security@aragora.ai | 15 minutes |
| Infrastructure | infra@aragora.ai | 30 minutes |
| Executive | exec@aragora.ai | 1 hour (critical only) |
