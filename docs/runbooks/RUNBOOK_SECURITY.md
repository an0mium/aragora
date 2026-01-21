# Security Operations Runbook

**Effective Date:** January 21, 2026
**Last Updated:** January 21, 2026
**Version:** 1.0.0
**Owner:** Security Team

---

## Overview

This runbook covers operational procedures for Aragora's security infrastructure including:
- Encryption and key management
- RBAC authorization system
- Approval gates for high-risk operations
- Security incident response specific to these systems

**Related Documents:**
- [General Incident Response](../INCIDENT_RESPONSE.md)
- [Secrets Management](../SECRETS_MANAGEMENT.md)
- [Security Deployment](../SECURITY_DEPLOYMENT.md)

---

## Quick Reference

### Emergency Contacts

| Role | Contact | When to Escalate |
|------|---------|------------------|
| Security On-Call | PagerDuty | Any P1/P2 security alert |
| Security Lead | security@aragora.ai | Key rotation failures, breaches |
| VP Engineering | [phone] | Data breach confirmed |
| Legal | legal@aragora.ai | Data breach with PII |

### Critical Commands

```bash
# Check encryption service health
curl -s http://localhost:8080/api/security/encryption/health | jq .

# Check RBAC cache status
curl -s http://localhost:8080/api/security/rbac/cache/stats | jq .

# List pending approvals
curl -s http://localhost:8080/api/security/approvals/pending | jq .

# Emergency: Disable all API keys for a user
curl -X POST http://localhost:8080/api/admin/users/{id}/revoke-keys \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Emergency: Force key rotation
python -m aragora.security.key_rotation --force --reason="emergency"
```

---

## Section 1: Encryption Operations

### Alert: HighEncryptionLatency

**Severity:** Warning
**Threshold:** p95 > 100ms

**Symptoms:**
- Slow API responses for operations involving secrets
- Increased request timeouts
- High CPU usage on application servers

**Investigation:**

```bash
# 1. Check encryption metrics
curl -s http://localhost:8080/metrics | grep "aragora_security_encryption"

# 2. Check if key derivation is slow (indicates CPU contention)
curl -s http://localhost:8080/api/security/encryption/stats | jq .derivation_latency_p95

# 3. Check active keys count
curl -s http://localhost:8080/api/security/encryption/keys | jq .active_count

# 4. Check system resources
top -bn1 | head -20
free -h
```

**Resolution:**

| Root Cause | Action |
|------------|--------|
| High CPU usage | Scale application instances |
| Too many active keys | Clean up expired keys |
| Key derivation slow | Reduce PBKDF2 iterations in non-production |
| Cold cache | Pre-warm encryption cache on startup |

```bash
# Scale up if CPU-bound
kubectl -n aragora scale deployment/aragora --replicas=5

# Clean up expired keys
python -m aragora.security.key_rotation --cleanup-expired

# Check if hardware acceleration is available
python -c "from cryptography.hazmat.backends import default_backend; print(default_backend())"
```

---

### Alert: EncryptionErrorsIncreasing

**Severity:** High
**Threshold:** >5 errors in 5 minutes

**Symptoms:**
- Failed API operations involving secrets
- "Decryption failed" errors in logs
- Users unable to access integrations

**Investigation:**

```bash
# 1. Check error types
curl -s http://localhost:8080/metrics | grep "encryption_errors" | head -20

# 2. Search logs for encryption errors
kubectl -n aragora logs -l app=aragora --tail=500 | grep -i "encrypt\|decrypt" | grep -i "error\|fail"

# 3. Check if master key is loaded
curl -s http://localhost:8080/api/security/encryption/health | jq .master_key_loaded

# 4. Verify encryption key is set
echo "ARAGORA_ENCRYPTION_KEY length: ${#ARAGORA_ENCRYPTION_KEY}"
```

**Resolution:**

| Root Cause | Action |
|------------|--------|
| Master key not loaded | Restart with correct ARAGORA_ENCRYPTION_KEY |
| Corrupted encrypted data | Restore from backup, re-encrypt |
| Key version mismatch | Run key migration |
| Invalid AAD | Check record IDs for encrypted data |

```bash
# Verify encryption key format (should be 64 hex chars = 32 bytes)
python -c "
import os
key = os.environ.get('ARAGORA_ENCRYPTION_KEY', '')
print(f'Key length: {len(key)} chars ({len(key)//2} bytes)')
print(f'Valid hex: {all(c in \"0123456789abcdef\" for c in key.lower())}')
"

# If key is wrong, redeploy with correct key
# WARNING: This may cause decryption failures for existing data

# Run encryption migration to re-encrypt with new key
python -m aragora.security.migration --store=all --dry-run
python -m aragora.security.migration --store=all
```

---

### Alert: KeyRotationFailed

**Severity:** Critical
**Threshold:** Any failure

**Symptoms:**
- Scheduled key rotation did not complete
- Encryption errors after rotation attempt
- Metrics show rotation failure

**Immediate Actions:**

```bash
# 1. Check current key status
curl -s http://localhost:8080/api/security/encryption/keys | jq .

# 2. Check rotation logs
kubectl -n aragora logs -l app=aragora --tail=200 | grep -i "rotation"

# 3. Verify the old key is still active
curl -s http://localhost:8080/api/security/encryption/health | jq .active_key_id
```

**Resolution:**

```bash
# If rotation failed mid-way, both old and new keys should be active
# Check key status
python -c "
from aragora.security.encryption import get_encryption_service
service = get_encryption_service()
print(f'Active keys: {service.list_active_keys()}')
print(f'Primary key: {service.get_primary_key_id()}')
"

# Re-attempt rotation with verbose logging
ARAGORA_LOG_LEVEL=DEBUG python -m aragora.security.key_rotation \
  --reason="retry after failure"

# If old key is lost, restore from backup
# This requires the key backup stored securely
```

**Prevention:**
- Ensure key backups are stored securely before rotation
- Test rotation in staging first
- Schedule rotations during low-traffic periods

---

### Alert: NoActiveEncryptionKeys

**Severity:** Critical
**Threshold:** master key count = 0

**THIS IS A CRITICAL INCIDENT - All encryption/decryption will fail**

**Immediate Actions:**

```bash
# 1. Verify the alert
curl -s http://localhost:8080/api/security/encryption/health

# 2. Check if ARAGORA_ENCRYPTION_KEY is set
kubectl -n aragora exec deploy/aragora -- printenv | grep ENCRYPTION

# 3. Check secrets manager
aws secretsmanager get-secret-value --secret-id aragora/encryption-key
```

**Resolution:**

```bash
# If key is missing from environment, redeploy with key
kubectl -n aragora set env deployment/aragora \
  ARAGORA_ENCRYPTION_KEY="$(aws secretsmanager get-secret-value \
    --secret-id aragora/encryption-key --query SecretString --output text)"

# Verify key is loaded
kubectl -n aragora exec deploy/aragora -- \
  python -c "from aragora.security.encryption import get_encryption_service; print(get_encryption_service().health())"
```

---

## Section 2: RBAC Authorization

### Alert: HighRBACDenialRate

**Severity:** Warning
**Threshold:** >20% denials

**Symptoms:**
- Users receiving 403 Forbidden errors
- Feature access issues reported
- Spike in support tickets

**Investigation:**

```bash
# 1. Check which permissions are being denied
curl -s http://localhost:8080/metrics | grep "rbac_denied_total" | sort -t'=' -k2 -nr | head -10

# 2. Check which roles are being denied
curl -s http://localhost:8080/api/security/rbac/denials/recent | jq '.[] | {permission, role, count}'

# 3. Check if there was a recent permission change
git log --oneline -20 -- "**/permissions*" "**/rbac*"

# 4. Verify RBAC cache is functioning
curl -s http://localhost:8080/api/security/rbac/cache/stats | jq .
```

**Resolution:**

| Root Cause | Action |
|------------|--------|
| Permission misconfiguration | Review and fix permission definitions |
| Role mapping issue | Check user role assignments |
| Cache stale | Clear RBAC cache |
| New feature without permissions | Add missing permissions |

```bash
# Clear RBAC cache (forces refresh)
curl -X POST http://localhost:8080/api/security/rbac/cache/clear \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Check specific user's permissions
curl -s "http://localhost:8080/api/admin/users/{user_id}/permissions" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq .
```

---

### Alert: RBACDenialSpike / SensitivePermissionDenied

**Severity:** Warning/High
**Threshold:** 3x baseline / Any sensitive permission denial

**This may indicate an attack attempt**

**Immediate Actions:**

```bash
# 1. Identify the source of denials
curl -s http://localhost:8080/api/security/rbac/denials/recent?minutes=15 | jq '
  group_by(.user_id) |
  map({user_id: .[0].user_id, count: length, permissions: [.[].permission] | unique}) |
  sort_by(-.count)'

# 2. Check for patterns (same user, same IP, same permission)
kubectl -n aragora logs -l app=aragora --tail=1000 | \
  grep "permission_denied\|403" | \
  awk '{print $NF}' | sort | uniq -c | sort -rn | head -20

# 3. Check if it's a single bad actor
curl -s http://localhost:8080/api/security/audit/search?action=permission_denied&minutes=15 | jq '
  .entries | group_by(.actor) | map({actor: .[0].actor, attempts: length})'
```

**Resolution:**

| Pattern | Action |
|---------|--------|
| Single user, many denials | Possible compromised account - disable user |
| Many users, same permission | Missing permission for new feature |
| Sensitive permissions only | Possible privilege escalation attempt |
| Random pattern | Likely bot/scanner - rate limit IP |

```bash
# Disable suspicious user
curl -X POST http://localhost:8080/api/admin/users/{user_id}/disable \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"reason": "Suspicious RBAC activity"}'

# Block suspicious IP (if using rate limiter)
curl -X POST http://localhost:8080/api/admin/ratelimit/block \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"ip": "1.2.3.4", "duration": "1h", "reason": "RBAC attack"}'
```

---

### Alert: SlowRBACEvaluation

**Severity:** Warning
**Threshold:** p95 > 25ms

**Investigation:**

```bash
# 1. Check cache hit rate
curl -s http://localhost:8080/api/security/rbac/cache/stats | jq .

# 2. Check permission complexity
curl -s http://localhost:8080/api/security/rbac/permissions | jq 'length'

# 3. Check if Redis is slow (distributed cache)
redis-cli --latency -h $REDIS_HOST
```

**Resolution:**

```bash
# Warm up RBAC cache
curl -X POST http://localhost:8080/api/security/rbac/cache/warmup \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Increase cache TTL if needed
kubectl -n aragora set env deployment/aragora RBAC_CACHE_TTL=600

# If Redis is slow, check Redis health
redis-cli -h $REDIS_HOST INFO | grep -E "used_memory|connected_clients|blocked_clients"
```

---

## Section 3: Approval Gates

### Alert: ApprovalRequestTimeout

**Severity:** Warning
**Threshold:** >5 timeouts in 1 hour

**Symptoms:**
- Operations stuck waiting for approval
- Users reporting blocked workflows
- Backlog of pending approvals

**Investigation:**

```bash
# 1. List pending approvals
curl -s http://localhost:8080/api/security/approvals/pending | jq '
  .[] | {id, operation, risk_level, requested_at, requester_id}'

# 2. Check approval queue age
curl -s http://localhost:8080/api/security/approvals/pending | jq '
  [.[] | .requested_at | fromdateiso8601] |
  map(now - .) |
  {oldest_minutes: (max/60), avg_minutes: (add/length/60)}'

# 3. Check if approvers are available
curl -s http://localhost:8080/api/admin/users?role=approver | jq '.[].last_active_at'
```

**Resolution:**

| Root Cause | Action |
|------------|--------|
| No approvers online | Contact approvers, or use emergency bypass |
| Too many requests | Adjust risk thresholds, auto-approve low-risk |
| System issue | Check approval notification delivery |

```bash
# Emergency: Approve critical pending request (requires admin)
curl -X POST "http://localhost:8080/api/security/approvals/{id}/approve" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"reason": "Emergency override per runbook", "bypass_checklist": true}'

# Adjust timeout for specific operation type
kubectl -n aragora set env deployment/aragora \
  APPROVAL_TIMEOUT_KEY_ROTATION=7200
```

---

### Alert: CriticalOperationPendingTooLong

**Severity:** High
**Threshold:** Critical approval pending >1 hour

**THIS REQUIRES IMMEDIATE ATTENTION**

**Immediate Actions:**

```bash
# 1. Identify the pending critical operation
curl -s http://localhost:8080/api/security/approvals/pending?risk_level=critical | jq .

# 2. Check who can approve
curl -s http://localhost:8080/api/security/approvals/{id}/eligible-approvers | jq .

# 3. Contact approvers directly
# (Use emergency contact list)
```

**Resolution:**

```bash
# Option 1: Approve with proper authorization
curl -X POST "http://localhost:8080/api/security/approvals/{id}/approve" \
  -H "Authorization: Bearer $APPROVER_TOKEN" \
  -d '{"notes": "Approved per escalation procedure"}'

# Option 2: Reject if operation is no longer needed
curl -X POST "http://localhost:8080/api/security/approvals/{id}/reject" \
  -H "Authorization: Bearer $APPROVER_TOKEN" \
  -d '{"reason": "Operation timed out, user should retry"}'

# Option 3: Emergency bypass (requires security lead approval)
curl -X POST "http://localhost:8080/api/security/approvals/{id}/emergency-bypass" \
  -H "Authorization: Bearer $SECURITY_LEAD_TOKEN" \
  -d '{"reason": "Emergency bypass - contacted via phone", "incident_id": "INC-12345"}'
```

---

### Alert: UnauthorizedApprovalAttempt

**Severity:** High
**Threshold:** Any occurrence

**This indicates someone tried to approve without proper authorization**

**Immediate Actions:**

```bash
# 1. Get details of the attempt
curl -s http://localhost:8080/api/security/audit/search?action=unauthorized_approval | jq '
  .entries[-5:] | .[] | {time: .timestamp, actor: .actor, ip: .ip_address, details: .details}'

# 2. Check if the actor's account is compromised
curl -s "http://localhost:8080/api/admin/users/{actor_id}/sessions" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq .

# 3. Check for other suspicious activity from this user
curl -s "http://localhost:8080/api/security/audit/search?actor={actor_id}&minutes=60" \
  -H "Authorization: Bearer $ADMIN_TOKEN" | jq .
```

**Resolution:**

| Scenario | Action |
|----------|--------|
| User misunderstanding | Educate user on approval process |
| Role misconfiguration | Fix role assignments |
| Possible attack | Disable user, investigate further |

---

## Section 4: Authentication Security

### Alert: BruteForceAttemptDetected

**Severity:** High
**Threshold:** >5 failed auth attempts/second

**Immediate Actions:**

```bash
# 1. Identify source IPs
kubectl -n aragora logs -l app=aragora --tail=2000 | \
  grep "auth_failure\|401" | \
  grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | \
  sort | uniq -c | sort -rn | head -20

# 2. Check targeted accounts
curl -s http://localhost:8080/api/security/audit/search?action=auth_failure&minutes=10 | jq '
  .entries | group_by(.details.email) |
  map({email: .[0].details.email, attempts: length}) |
  sort_by(-.attempts) | .[0:10]'

# 3. Check if any succeeded after failures
curl -s http://localhost:8080/api/security/audit/search?action=login_success&minutes=10 | jq '
  [.entries[].actor] as $successes |
  $successes'
```

**Resolution:**

```bash
# Block offending IPs
for ip in 1.2.3.4 5.6.7.8; do
  curl -X POST http://localhost:8080/api/admin/ratelimit/block \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -d "{\"ip\": \"$ip\", \"duration\": \"24h\", \"reason\": \"Brute force\"}"
done

# Lock targeted accounts temporarily
curl -X POST http://localhost:8080/api/admin/users/bulk-lock \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"emails": ["target1@example.com", "target2@example.com"], "duration": "1h"}'

# Enable stricter rate limiting
kubectl -n aragora set env deployment/aragora \
  AUTH_RATE_LIMIT_PER_IP=5 \
  AUTH_RATE_LIMIT_WINDOW=60
```

---

### Alert: JWTValidationErrors

**Severity:** Warning
**Threshold:** >1/second

**This may indicate token forgery attempts**

**Investigation:**

```bash
# 1. Check error types
curl -s http://localhost:8080/metrics | grep "auth_failures.*reason" | head -20

# 2. Check if JWT secret was recently changed
git log --oneline -10 -- "**/*secret*" "**/*jwt*"

# 3. Look for patterns
kubectl -n aragora logs -l app=aragora --tail=500 | \
  grep "invalid.*token\|jwt.*error" | head -20
```

**Resolution:**

| Root Cause | Action |
|------------|--------|
| JWT secret rotation | Expected - old tokens will fail |
| Token forgery attempt | Block IPs, investigate source |
| Clock skew | Check server time synchronization |
| Malformed client | Identify and fix client bug |

---

## Section 5: Secret Access Patterns

### Alert: UnusualSecretAccessPattern / HighSecretDecryptionRate

**Severity:** Warning
**Threshold:** 3x baseline / >100/second

**This may indicate data exfiltration**

**Immediate Actions:**

```bash
# 1. Identify who is accessing secrets
curl -s http://localhost:8080/api/security/audit/search?action=secret_access&minutes=15 | jq '
  .entries | group_by(.actor) |
  map({actor: .[0].actor, count: length, secret_types: [.[].details.secret_type] | unique})'

# 2. Check for bulk exports or API calls
kubectl -n aragora logs -l app=aragora --tail=1000 | \
  grep -E "export|bulk|list.*integration" | head -20

# 3. Correlate with normal business activity
# (Check if there's a legitimate reason like migration)
```

**Resolution:**

```bash
# If suspicious, immediately revoke user's access
curl -X POST http://localhost:8080/api/admin/users/{user_id}/disable \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"reason": "Unusual secret access pattern investigation"}'

# Rotate potentially exposed secrets
python -m aragora.security.rotation --secrets-accessed-by={user_id} --dry-run
python -m aragora.security.rotation --secrets-accessed-by={user_id}

# If confirmed exfiltration, escalate to security incident
# Follow Data Breach playbook in INCIDENT_RESPONSE.md
```

---

## Section 6: Migration Operations

### Alert: MigrationErrorsDetected

**Severity:** High
**Threshold:** Any errors

**Investigation:**

```bash
# 1. Check migration logs
kubectl -n aragora logs -l app=aragora --tail=500 | grep -i "migration"

# 2. Check which records failed
curl -s http://localhost:8080/api/security/migration/status | jq '
  .failed_records | .[0:10]'

# 3. Check error types
curl -s http://localhost:8080/metrics | grep "migration_errors" | head -10
```

**Resolution:**

```bash
# Retry failed records
python -m aragora.security.migration --retry-failed --store={store}

# If specific record is corrupt, skip it
python -m aragora.security.migration --skip-record={record_id}

# Manually fix corrupted record if needed
python -c "
from aragora.storage import get_store
store = get_store('{store}')
record = store.get('{record_id}')
# Inspect and fix record
"
```

---

## Appendix A: Metrics Reference

### Encryption Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `aragora_security_encryption_operations_total` | Total encrypt/decrypt ops | N/A |
| `aragora_security_encryption_latency_seconds` | Operation latency | p95 > 100ms |
| `aragora_security_encryption_errors_total` | Encryption errors | >5 in 5min |
| `aragora_security_active_keys` | Active encryption keys | master = 0 |

### RBAC Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `aragora_security_rbac_decisions_total` | Total auth decisions | N/A |
| `aragora_security_rbac_denied_total` | Denied decisions | >20% rate |
| `aragora_security_rbac_evaluation_latency_seconds` | Evaluation time | p95 > 25ms |

### Authentication Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `aragora_security_auth_attempts_total` | Auth attempts | N/A |
| `aragora_security_auth_failures_total` | Failed attempts | >5/sec |
| `aragora_security_auth_latency_seconds` | Auth latency | p95 > 500ms |

### Approval Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `aragora_approval_requests_total` | Approval requests | N/A |
| `aragora_pending_critical_approvals` | Pending critical | >0 for 1hr |
| `aragora_approval_unauthorized_attempts_total` | Unauthorized attempts | Any |

---

## Appendix B: Emergency Procedures

### Complete Security Lockdown

Use when active breach is suspected:

```bash
# 1. Enable maintenance mode (stops new requests)
kubectl -n aragora set env deployment/aragora MAINTENANCE_MODE=true

# 2. Revoke all user sessions
curl -X POST http://localhost:8080/api/admin/sessions/revoke-all \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# 3. Rotate JWT secret (invalidates all tokens)
NEW_SECRET=$(openssl rand -base64 64)
kubectl -n aragora set env deployment/aragora JWT_SECRET="$NEW_SECRET"

# 4. Document and escalate
# Create incident ticket
# Contact security lead
```

### Emergency Key Recovery

If encryption key is lost:

```bash
# 1. Check if key backup exists
aws secretsmanager get-secret-value --secret-id aragora/encryption-key-backup

# 2. If no backup, data encrypted with that key is unrecoverable
# You must restore from database backup taken before key loss

# 3. Restore database
pg_restore -d aragora_production backup_before_key_loss.dump

# 4. Generate new key and re-encrypt
NEW_KEY=$(openssl rand -hex 32)
aws secretsmanager create-secret --name aragora/encryption-key --secret-string "$NEW_KEY"
kubectl -n aragora set env deployment/aragora ARAGORA_ENCRYPTION_KEY="$NEW_KEY"
```

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-21 | Initial release |
