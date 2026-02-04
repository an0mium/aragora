# Security Incident Runbook

This runbook covers procedures for handling security incidents, breaches, and key rotation in Aragora.

## Incident Classification

| Type | Severity | Example |
|------|----------|---------|
| Critical | SEV-1 | Active breach, data exfiltration, compromised credentials |
| High | SEV-2 | Vulnerability exploitation attempt, suspicious access patterns |
| Medium | SEV-3 | Failed authentication spikes, potential phishing |
| Low | SEV-4 | Security scanner findings, policy violations |

## Immediate Response (First 15 Minutes)

### 1. Assess and Contain

```bash
# Check for active sessions from suspicious IPs
psql $DATABASE_URL -c "
SELECT user_id, ip_address, created_at, last_active
FROM sessions
WHERE ip_address = '{SUSPICIOUS_IP}'
ORDER BY created_at DESC;"

# Block suspicious IP immediately
kubectl exec -it $(kubectl get pod -l app=nginx -o name | head -1) -- \
  nginx -s reload  # After updating blocklist

# Or via API
curl -X POST https://api.aragora.ai/api/internal/security/block-ip \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"ip": "{SUSPICIOUS_IP}", "reason": "Security incident investigation"}'
```

### 2. Preserve Evidence

```bash
# Export relevant logs before rotation
kubectl logs -l app=aragora-api --since=24h > /tmp/incident-logs-$(date +%s).txt

# Snapshot suspicious user activity
psql $DATABASE_URL -c "
COPY (
  SELECT * FROM audit_events
  WHERE actor_id = '{SUSPICIOUS_USER}'
  AND created_at > NOW() - INTERVAL '7 days'
) TO '/tmp/user-audit.csv' CSV HEADER;"

# Preserve session data
redis-cli DUMP "session:{SESSION_ID}" > /tmp/session-dump.rdb
```

### 3. Notify Security Team

```
[SECURITY INCIDENT] {Classification}
- Type: {breach/vulnerability/suspicious_activity}
- Affected systems: {list}
- Potential impact: {description}
- Containment status: {contained/in_progress}
- IC: @{name}

Do NOT share details outside this channel.
```

## Credential Compromise Response

### API Key Compromise

```bash
# 1. Identify affected key
psql $DATABASE_URL -c "
SELECT id, user_id, prefix, last_used, created_at
FROM api_keys
WHERE prefix = '{COMPROMISED_PREFIX}';"

# 2. Revoke immediately
curl -X DELETE https://api.aragora.ai/api/v1/auth/api-keys/{KEY_ID} \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# 3. Check for unauthorized usage
psql $DATABASE_URL -c "
SELECT action, ip_address, created_at
FROM audit_events
WHERE api_key_id = '{KEY_ID}'
AND created_at > NOW() - INTERVAL '30 days'
ORDER BY created_at DESC;"

# 4. Notify affected user
# (via automated notification system or manual contact)
```

### User Password Compromise

```bash
# 1. Force password reset
psql $DATABASE_URL -c "
UPDATE users
SET must_reset_password = true,
    password_hash = NULL
WHERE id = '{USER_ID}';"

# 2. Invalidate all sessions
psql $DATABASE_URL -c "
DELETE FROM sessions WHERE user_id = '{USER_ID}';"

# 3. Revoke all API keys
psql $DATABASE_URL -c "
UPDATE api_keys SET revoked_at = NOW()
WHERE user_id = '{USER_ID}';"

# 4. Clear cached tokens
redis-cli KEYS "user:{USER_ID}:*" | xargs redis-cli DEL
```

### OAuth Token Compromise

```bash
# Revoke OAuth token
curl -X POST https://api.aragora.ai/api/v1/auth/oauth/revoke \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"user_id": "{USER_ID}", "provider": "{PROVIDER}"}'

# Force re-authentication
psql $DATABASE_URL -c "
DELETE FROM oauth_tokens WHERE user_id = '{USER_ID}';"
```

## Key Rotation Procedures

### Database Encryption Keys

```bash
# 1. Generate new key
NEW_KEY=$(openssl rand -base64 32)

# 2. Update secret in Kubernetes
kubectl create secret generic db-encryption-key-new \
  --from-literal=key=$NEW_KEY

# 3. Start re-encryption job
kubectl apply -f k8s/jobs/reencrypt-data.yaml

# 4. Monitor progress
kubectl logs -f job/reencrypt-data

# 5. After completion, swap keys
kubectl patch secret db-encryption-key \
  -p '{"data":{"key":"'$(echo -n $NEW_KEY | base64)'"}}'

# 6. Restart services
kubectl rollout restart deployment/aragora-api
```

### JWT Signing Keys

```bash
# 1. Generate new key pair
openssl genrsa -out jwt-private-new.pem 4096
openssl rsa -in jwt-private-new.pem -pubout -out jwt-public-new.pem

# 2. Update secrets (keep old public key for grace period)
kubectl create secret generic jwt-keys-new \
  --from-file=private=jwt-private-new.pem \
  --from-file=public=jwt-public-new.pem

# 3. Deploy with dual-key support
kubectl set env deployment/aragora-api \
  JWT_NEW_KEY_ID=v2 \
  JWT_GRACE_PERIOD=24h

# 4. After grace period, remove old key
kubectl set env deployment/aragora-api JWT_OLD_KEY_ID-
```

### LLM API Keys

```bash
# 1. Generate new key in provider dashboard
# (Anthropic: console.anthropic.com)
# (OpenAI: platform.openai.com)

# 2. Update Kubernetes secret
kubectl create secret generic llm-api-keys \
  --from-literal=ANTHROPIC_API_KEY=$NEW_ANTHROPIC_KEY \
  --from-literal=OPENAI_API_KEY=$NEW_OPENAI_KEY \
  --dry-run=client -o yaml | kubectl apply -f -

# 3. Rolling restart to pick up new keys
kubectl rollout restart deployment/aragora-api
kubectl rollout restart deployment/aragora-worker

# 4. Revoke old keys in provider dashboard
```

## Vulnerability Response

### Critical Vulnerability (e.g., RCE)

```bash
# 1. Assess exposure
# Check if vulnerable endpoint was accessed
grep -E "vulnerable_endpoint" /var/log/aragora/*.log

# 2. Apply emergency patch
git checkout -b hotfix/security-{CVE}
# Apply fix
git push origin hotfix/security-{CVE}

# 3. Emergency deploy
kubectl set image deployment/aragora-api \
  aragora-api=aragora/api:hotfix-{CVE}

# 4. Verify fix
curl https://api.aragora.ai/api/health | jq .version
```

### Dependency Vulnerability

```bash
# 1. Identify affected packages
pip-audit
npm audit

# 2. Update dependencies
pip install --upgrade {PACKAGE}
# or
npm update {PACKAGE}

# 3. Run security tests
pytest tests/security/ -v

# 4. Deploy update
# (Follow normal deployment process)
```

## Audit Trail Requirements

All security incidents must include:
- [ ] Initial detection timestamp
- [ ] Containment actions taken
- [ ] Affected systems and users
- [ ] Evidence preserved
- [ ] Root cause analysis
- [ ] Remediation steps
- [ ] Timeline of events
- [ ] External notification requirements (GDPR, etc.)

## Compliance Notifications

| Condition | Notify | Timeframe |
|-----------|--------|-----------|
| User data breach (GDPR) | Data Protection Authority | 72 hours |
| User data breach (GDPR) | Affected users | Without undue delay |
| SOC 2 relevant incident | Auditor | Next audit |
| PCI data involved | Payment processor | 24 hours |

## Post-Incident Review

```markdown
## Security Incident Report: {ID}
**Date:** {Date}
**Classification:** {Type}
**Severity:** SEV-{X}

### Summary
{One paragraph description}

### Attack Vector
{How the incident occurred}

### Indicators of Compromise (IOCs)
- IP addresses: {list}
- User agents: {list}
- Patterns: {list}

### Affected Assets
- Systems: {list}
- Data: {classification and scope}
- Users: {count}

### Timeline
| Time | Event |
|------|-------|
| HH:MM | {Event} |

### Response Actions
{What was done}

### Root Cause
{Technical explanation}

### Remediation
| Action | Status | Owner |
|--------|--------|-------|
| {Action} | {Done/In Progress} | @{name} |

### Lessons Learned
{What to improve}

### External Notifications
| Party | Date | Reference |
|-------|------|-----------|
| {Authority/User} | {Date} | {Ref#} |
```

---
*Last updated: 2026-02-03*
