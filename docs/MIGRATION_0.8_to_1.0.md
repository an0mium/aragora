# Migration Guide: Aragora 0.8.x to 1.0.0

This guide covers upgrading from Aragora 0.8.x to 1.0.0. The upgrade is designed to be backwards-compatible, with deprecated features continuing to work until July 2026.

---

## Quick Migration Checklist

- [ ] Update Aragora package to 1.0.0
- [ ] Update TypeScript SDK to @aragora/sdk@1.0.0
- [ ] Migrate API calls from `/api/` to `/api/v2/`
- [ ] Configure Redis for multi-replica deployments
- [ ] Review and update rate limit configuration
- [ ] Enable MFA if required for your security policy
- [ ] Update Kubernetes manifests for HA features
- [ ] Run test suite to verify compatibility

---

## 1. Package Update

### Python

```bash
pip install --upgrade aragora==1.0.0
```

Or update `pyproject.toml`:

```toml
[project]
dependencies = [
    "aragora>=1.0.0,<2.0.0",
]
```

### TypeScript SDK

```bash
npm install @aragora/sdk@1.0.0
```

Or update `package.json`:

```json
{
  "dependencies": {
    "@aragora/sdk": "^1.0.0"
  }
}
```

---

## 2. API Endpoint Migration

### What's Changing

All unversioned API endpoints (`/api/*`) are deprecated in favor of versioned endpoints (`/api/v2/*`).

**Deprecation Timeline:**
- **January 2026**: V2 endpoints available, V1 still works
- **April 2026**: V1 endpoints return deprecation warnings
- **July 2026**: V1 endpoints removed

### Migration Steps

Update your API calls to use the `/api/v2/` prefix:

```diff
# Before (deprecated)
- curl https://aragora.example.com/api/debates
- curl https://aragora.example.com/api/health

# After (recommended)
+ curl https://aragora.example.com/api/v2/debates
+ curl https://aragora.example.com/api/v2/health
```

### TypeScript SDK

The SDK automatically uses V2 endpoints. No changes needed if using the SDK.

```typescript
// SDK handles versioning automatically
const debates = await client.debates.list();
```

### Detecting Deprecated Calls

V1 endpoints return deprecation headers:

```http
HTTP/1.1 200 OK
Deprecation: true
Sunset: Sat, 04 Jul 2026 00:00:00 GMT
Link: </api/v2/debates>; rel="successor-version"
```

Configure your logging to alert on these headers.

---

## 3. Redis Configuration

### When Redis is Required

Redis is **required** for:
- Multi-replica Kubernetes deployments
- Account lockout (distributed tracking)
- Session management (cross-pod)
- Rate limiting (distributed counters)

Redis is **optional** for:
- Single-instance deployments
- Development environments
- Testing

### Configuration

Set the `REDIS_URL` environment variable:

```bash
export REDIS_URL="redis://redis-host:6379/0"
```

Or in Kubernetes:

```yaml
env:
  - name: REDIS_URL
    value: "redis://aragora-redis:6379/0"
```

### Redis Schema

1.0.0 uses the following Redis key prefixes:

| Prefix | Purpose | TTL |
|--------|---------|-----|
| `aragora:lockout:email:*` | Account lockout by email | 24h |
| `aragora:lockout:ip:*` | Account lockout by IP | 24h |
| `aragora:session:*` | User sessions | Configurable |
| `aragora:ratelimit:*` | Rate limit counters | 1min-1h |
| `aragora:blacklist:*` | Token blacklist | Token lifetime |

### Fallback Behavior

If Redis is unavailable:
- In-memory fallback activates automatically
- Warning logged: "Redis unavailable, using in-memory storage"
- Multi-replica deployments will have inconsistent state

---

## 4. Rate Limiting Changes

### Default Behavior

Rate limiting is now **enabled by default**. Previous versions had it disabled.

### Configuration

```bash
# Disable rate limiting (not recommended)
export ARAGORA_ENABLE_RATE_LIMIT=0

# Customize limits
export ARAGORA_RATE_LIMIT_DEFAULT=100    # requests per minute
export ARAGORA_RATE_LIMIT_DEBATES=30     # debate creations per minute
export ARAGORA_RATE_LIMIT_AUTH=10        # auth attempts per minute
```

### Response Headers

Rate-limited responses include:

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1704134400
Retry-After: 60
```

---

## 5. Authentication Changes

### Account Lockout

Account lockout is now enabled by default after failed login attempts:

| Attempts | Lockout Duration |
|----------|------------------|
| 5 | 1 minute |
| 10 | 15 minutes |
| 15+ | 1 hour |

### MFA Support

Multi-factor authentication is available but opt-in:

```bash
# Enable MFA feature
export ARAGORA_ENABLE_MFA=true
```

MFA endpoints:
- `POST /api/v2/auth/mfa/setup` - Initialize MFA
- `POST /api/v2/auth/mfa/enable` - Activate after verification
- `POST /api/v2/auth/mfa/verify` - Verify code at login
- `DELETE /api/v2/auth/mfa` - Disable MFA

### Admin Unlock

Admins can unlock locked accounts:

```bash
curl -X POST \
  https://aragora.example.com/api/v2/admin/users/{user_id}/unlock \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

---

## 6. Kubernetes Deployment Updates

### New Manifests

1.0.0 includes new Kubernetes manifests in `deploy/kubernetes/`:

| File | Purpose |
|------|---------|
| `hpa.yaml` | Horizontal Pod Autoscaler (2-10 replicas) |
| `pdb.yaml` | Pod Disruption Budget (min 1 available) |
| `redis/` | Redis StatefulSet for shared state |
| `cert-manager.yaml` | TLS certificate automation |

### Update Deployment

```bash
# If using kustomize
kubectl apply -k deploy/kubernetes/

# Or apply individual resources
kubectl apply -f deploy/kubernetes/hpa.yaml
kubectl apply -f deploy/kubernetes/pdb.yaml
```

### Anti-Affinity

The deployment now includes pod anti-affinity to spread across nodes:

```yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app.kubernetes.io/name: aragora
          topologyKey: kubernetes.io/hostname
```

---

## 7. Environment Variables

### New Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_ENABLE_MFA` | `false` | Enable MFA feature |
| `ARAGORA_LOCKOUT_ENABLED` | `true` | Enable account lockout |
| `ARAGORA_LOCKOUT_THRESHOLD_1` | `5` | First lockout threshold |
| `ARAGORA_LOCKOUT_DURATION_1` | `60` | First lockout seconds |

### Changed Variables

| Variable | Old Default | New Default |
|----------|-------------|-------------|
| `ARAGORA_ENABLE_RATE_LIMIT` | `0` | `1` |
| `ARAGORA_LOG_LEVEL` | `WARNING` | `INFO` |

### Removed Variables

None. All 0.8.x variables continue to work.

---

## 8. SDK Migration

### Client Initialization

The SDK initialization is backwards-compatible:

```typescript
// Works in both 0.8.x and 1.0.0
import { AragoraClient } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL,
  apiToken: process.env.ARAGORA_API_TOKEN,
});
```

### New APIs

1.0.0 adds several new API namespaces:

```typescript
// Calibration (prediction accuracy)
const scores = await client.calibration.getScores({ limit: 10 });
const history = await client.calibration.getHistory('agent-id');

// Insights (pattern extraction)
const insights = await client.insights.extract(debateId);

// Consensus proofs
const proof = await client.consensus.getProof(consensusId);

// MFA management
await client.auth.mfa.setup();
await client.auth.mfa.enable({ code: '123456' });
```

### Deprecated Methods

| Deprecated | Replacement |
|------------|-------------|
| `client.getDebates()` | `client.debates.list()` |
| `client.createDebate()` | `client.debates.create()` |
| `client.getAgents()` | `client.agents.list()` |

---

## 9. Testing Your Migration

### Run Compatibility Tests

```bash
# Backend tests
pytest tests/ -v --tb=short

# Frontend tests (if using SDK)
cd aragora/live && npm test

# Load tests
locust -f tests/load/locustfile.py --headless -u 10 -r 2 -t 1m
```

### Check Deprecation Warnings

Enable deprecation logging:

```bash
export ARAGORA_LOG_DEPRECATION_WARNINGS=true
```

Monitor logs for:
```
DEPRECATION: /api/debates is deprecated, use /api/v2/debates
```

---

## 10. Rollback Procedure

If you need to rollback:

### Python

```bash
pip install aragora==0.8.1
```

### Kubernetes

```bash
kubectl rollout undo deployment/aragora -n aragora
```

### Database

1.0.0 uses the same database schema as 0.8.x. No rollback needed for data.

---

## Common Issues

### Issue: "Redis connection failed"

**Solution:** Ensure Redis is running and `REDIS_URL` is correct:

```bash
redis-cli -h redis-host ping
# Should return: PONG
```

### Issue: "Account locked"

**Solution:** Wait for lockout to expire, or admin unlock:

```bash
curl -X POST /api/v2/admin/users/{id}/unlock
```

### Issue: "Rate limit exceeded"

**Solution:** Implement exponential backoff or request limit increase:

```bash
export ARAGORA_RATE_LIMIT_DEFAULT=200
```

### Issue: "MFA required but not configured"

**Solution:** Either disable MFA requirement or complete setup:

```bash
# Disable MFA requirement
export ARAGORA_REQUIRE_MFA=false

# Or complete setup via API
POST /api/v2/auth/mfa/setup
POST /api/v2/auth/mfa/enable
```

---

## Getting Help

- **Documentation:** https://aragora.ai/docs
- **GitHub Issues:** https://github.com/an0mium/aragora/issues
- **Migration Support:** Create an issue with label `migration`

---

## Changelog Reference

See [CHANGELOG.md](../CHANGELOG.md) for complete version history.
