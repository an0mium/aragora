# API Migration Guide: v1 to v2

This guide covers migrating from Aragora API v1 to v2. API v1 is deprecated and will be removed on **June 1, 2026**.

## Migration Checklist

- [ ] Update API endpoint URLs from `/api/v1/` to `/api/v2/`
- [ ] Update response parsing to handle new `data`/`meta` wrapper format
- [ ] Update Python SDK client initialization to specify v2
- [ ] Update TypeScript SDK to @aragora/sdk@1.0.0+
- [ ] Monitor deprecation warnings in logs
- [ ] Update webhook endpoints to expect v2 payloads
- [ ] Test all integrations in staging environment

## Timeline

| Date | Milestone |
|------|-----------|
| January 2025 | API v2 released |
| January 2026 | v1 endpoints return deprecation warnings |
| April 2026 | v1 endpoints return critical warnings |
| **June 1, 2026** | **v1 endpoints removed** |

## Response Format Changes

### v1 Response (Deprecated)

```json
{
  "debates": [
    {"id": "d1", "task": "Design a cache"}
  ],
  "count": 1
}
```

### v2 Response (Current)

```json
{
  "data": {
    "debates": [
      {"id": "d1", "task": "Design a cache"}
    ],
    "count": 1
  },
  "meta": {
    "version": "v2",
    "timestamp": "2026-01-19T12:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### Migration Code

```python
# v1 pattern (deprecated)
response = client.get("/api/v1/debates")
debates = response["debates"]

# v2 pattern (recommended)
response = client.get("/api/v2/debates")
debates = response["data"]["debates"]
meta = response["meta"]  # Access version, timestamp, request_id
```

## Endpoint Changes

### Renamed Endpoints

| v1 Endpoint | v2 Endpoint | Notes |
|-------------|-------------|-------|
| `POST /api/v1/debate` | `POST /api/v2/debates` | Pluralized |
| `GET /api/v1/debate/{id}` | `GET /api/v2/debates/{id}` | Pluralized |
| `POST /api/v1/agent/probe` | `POST /api/v2/agents/{id}/probes` | REST nested resource |
| `GET /api/v1/consensus` | `GET /api/v2/debates/{id}/consensus` | Scoped to debate |
| `POST /api/v1/vote` | `POST /api/v2/debates/{id}/votes` | Scoped to debate |

### New v2-Only Endpoints

These endpoints are only available in API v2:

| Endpoint | Description |
|----------|-------------|
| `GET /api/v2/calibration/scores` | Prediction accuracy scores |
| `GET /api/v2/calibration/history/{agent_id}` | Agent calibration history |
| `POST /api/v2/insights/extract/{debate_id}` | Extract patterns from debate |
| `GET /api/v2/consensus/{id}/proof` | Cryptographic consensus proof |
| `POST /api/v2/auth/mfa/setup` | Initialize MFA |
| `POST /api/v2/auth/mfa/verify` | Verify MFA code |
| `GET /api/v2/privacy/data-export` | GDPR data export |
| `DELETE /api/v2/privacy/data-deletion` | GDPR data deletion |
| `GET /api/v2/gallery` | Public debate gallery |

### Removed in v2

| v1 Endpoint | Replacement |
|-------------|-------------|
| `GET /api/v1/status` | `GET /api/v2/health` |
| `POST /api/v1/simple-debate` | `POST /api/v2/debates` with `mode: "quick"` |

## Python SDK Migration

### Client Initialization

```python
from aragora.client import AragoraClient

# v1 (deprecated - uses v1 by default)
client = AragoraClient(base_url="https://api.aragora.io")

# v2 (recommended - explicit version)
client = AragoraClient(
    base_url="https://api.aragora.io",
    api_version="v2"
)
```

### Method Changes

```python
# v1 methods (deprecated)
debates = client.getDebates()
debate = client.createDebate(task="...")
agents = client.getAgents()

# v2 methods (recommended)
debates = client.debates.list()
debate = client.debates.create(task="...")
agents = client.agents.list()
```

### New v2 APIs

```python
# Calibration (v2 only)
scores = await client.calibration.get_scores(limit=10)
history = await client.calibration.get_history("agent-id")

# Insights (v2 only)
insights = await client.insights.extract(debate_id)

# Consensus proofs (v2 only)
proof = await client.consensus.get_proof(consensus_id)

# MFA management (v2 only)
await client.auth.mfa.setup()
await client.auth.mfa.enable(code="123456")

# Privacy compliance (v2 only)
export = await client.privacy.export_data()
await client.privacy.request_deletion()
```

## TypeScript SDK Migration

### Installation

```bash
# Ensure you have SDK 1.0.0+
npm install @aragora/sdk@^1.0.0
```

### Client Initialization

```typescript
import { AragoraClient } from '@aragora/sdk';

// v1 (deprecated)
const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL,
  apiVersion: 'v1'  // Don't do this
});

// v2 (recommended)
const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL,
  apiVersion: 'v2'  // Explicit or omit (v2 is default in SDK 1.0+)
});
```

### Method Changes

```typescript
// v1 methods (deprecated)
const debates = await client.getDebates();
const debate = await client.createDebate({ task: '...' });

// v2 methods (recommended)
const debates = await client.debates.list();
const debate = await client.debates.create({ task: '...' });
```

### New v2 APIs

```typescript
// Calibration (v2 only)
const scores = await client.calibration.getScores({ limit: 10 });
const history = await client.calibration.getHistory('agent-id');

// Insights (v2 only)
const insights = await client.insights.extract(debateId);

// Consensus proofs (v2 only)
const proof = await client.consensus.getProof(consensusId);

// Probes (v2 only)
const probeResult = await client.probes.run({
  agentId: 'agent-id',
  probeTypes: ['contradiction', 'hallucination', 'sycophancy']
});

// MFA management (v2 only)
await client.auth.mfa.setup();
await client.auth.mfa.enable({ code: '123456' });
```

## Webhook Payload Changes

### v1 Webhook Payload

```json
{
  "event": "debate.completed",
  "debate_id": "d1",
  "consensus": "reached",
  "timestamp": 1705600000
}
```

### v2 Webhook Payload

```json
{
  "event": "debate.completed",
  "version": "v2",
  "data": {
    "debate_id": "d1",
    "consensus": {
      "status": "reached",
      "confidence": 0.95,
      "proof_id": "proof_abc123"
    }
  },
  "meta": {
    "timestamp": "2026-01-19T12:00:00Z",
    "delivery_id": "del_xyz789"
  }
}
```

### Webhook Migration

```python
# v1 webhook handler (deprecated)
def handle_webhook(payload):
    debate_id = payload["debate_id"]
    consensus = payload["consensus"]

# v2 webhook handler (recommended)
def handle_webhook(payload):
    if payload.get("version") != "v2":
        raise ValueError("Expected v2 webhook")

    data = payload["data"]
    debate_id = data["debate_id"]
    consensus = data["consensus"]["status"]
    confidence = data["consensus"]["confidence"]
```

## Deprecation Headers

v1 endpoints return deprecation headers:

```http
HTTP/1.1 200 OK
Deprecation: @1748736000
Sunset: 2026-06-01
Link: </api/v2/debates>; rel="successor-version"
X-Deprecation-Level: warning
```

### Monitoring Deprecation

```python
# Log deprecation warnings
import logging

logger = logging.getLogger("aragora.deprecation")

def check_response_headers(response):
    if "Deprecation" in response.headers:
        logger.warning(
            f"Deprecated endpoint called",
            extra={
                "endpoint": response.url,
                "sunset": response.headers.get("Sunset"),
                "replacement": response.headers.get("Link"),
            }
        )
```

## Error Response Changes

### v1 Error Response

```json
{
  "error": "Not found",
  "code": 404
}
```

### v2 Error Response

```json
{
  "error": {
    "code": "DEBATE_NOT_FOUND",
    "message": "Debate with ID 'd123' not found",
    "details": {
      "debate_id": "d123"
    }
  },
  "meta": {
    "version": "v2",
    "request_id": "req_abc123",
    "timestamp": "2026-01-19T12:00:00Z"
  }
}
```

### Error Handling Migration

```python
# v1 error handling
try:
    response = client.get("/api/v1/debates/invalid")
except Exception as e:
    error_msg = str(e)

# v2 error handling
try:
    response = client.get("/api/v2/debates/invalid")
except ApiError as e:
    error_code = e.error.code  # "DEBATE_NOT_FOUND"
    error_msg = e.error.message
    request_id = e.meta.request_id  # For support tickets
```

## Rate Limiting Changes

v2 includes improved rate limiting with clearer headers:

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705600060
X-RateLimit-Policy: 100;w=60
Retry-After: 60
```

### New Rate Limit Tiers

| Tier | v1 Limit | v2 Limit |
|------|----------|----------|
| Debates | 30/min | 30/min |
| Queries | 100/min | 200/min |
| Auth | 10/min | 10/min |
| Webhooks | 50/min | 100/min |

## Testing Migration

### Enable Deprecation Logging

```bash
export ARAGORA_LOG_DEPRECATION_WARNINGS=true
```

### Verify v2 Compatibility

```bash
# Run backend tests
pytest tests/ -v --tb=short

# Check for deprecation warnings
grep -r "DEPRECATION" logs/*.log

# Test v2 endpoints explicitly
curl -H "X-API-Version: 2" https://api.aragora.io/api/debates
```

### SDK Compatibility Test

```python
from aragora.client import AragoraClient

def test_v2_migration():
    client = AragoraClient(api_version="v2")

    # Test debates endpoint
    response = client.debates.list()
    assert "data" in response
    assert "meta" in response

    # Test new v2-only endpoints
    scores = client.calibration.get_scores()
    assert "data" in scores
```

## Common Issues

### Issue: "Invalid API version"

**Cause:** Mixing v1 and v2 endpoints in same request.

**Solution:** Use consistent version:
```python
client = AragoraClient(api_version="v2")
# All subsequent calls use v2
```

### Issue: "Missing 'data' key"

**Cause:** Parsing v1 response as v2.

**Solution:** Check API version in response:
```python
if "meta" in response:
    # v2 response
    data = response["data"]
else:
    # v1 response (deprecated)
    data = response
```

### Issue: "Endpoint not found (404)"

**Cause:** Using removed v1 endpoint.

**Solution:** Check endpoint mappings above and use v2 equivalent.

### Issue: "Webhook signature invalid"

**Cause:** Webhook payload format changed.

**Solution:** Update webhook handler for v2 payload structure (see Webhook section).

## Rollback

If you need to temporarily revert to v1:

```python
# Per-request rollback
response = client.get("/debates", api_version="v1")

# Client-level rollback
client = AragoraClient(api_version="v1")
```

Note: v1 will be removed on June 1, 2026. Plan permanent migration.

## Getting Help

- **Documentation:** https://aragora.ai/docs/migration
- **GitHub Issues:** https://github.com/an0mium/aragora/issues (label: `migration`)
- **API Reference:** https://aragora.ai/api/v2

## Related Documentation

- [API Versioning Strategy](./API_VERSIONING.md)
- [Deprecation Policy](./DEPRECATION_POLICY.md)
- [SDK Documentation](./SDK_GUIDE.md)
