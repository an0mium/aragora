# V1 to V2 API Migration Guide

> **Deadline:** API v1 endpoints will be removed on **June 1, 2026**.

This guide covers migrating from Aragora API v1 to v2. For SDK package migration
(`aragora-client` to `aragora-sdk`), see [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md).

## Current Status

As of February 2026, all `/api/v1/` responses include RFC 8594 deprecation headers.
The deprecation middleware is active in both the unified server and FastAPI factory.

| Component | Status |
|-----------|--------|
| v2 endpoints | Available since January 2025 |
| Deprecation headers on v1 | Active (warning level) |
| Critical warnings (< 30 days) | After May 2, 2026 |
| v1 removal (410 Gone) | June 1, 2026 |
| `ARAGORA_BLOCK_SUNSET_ENDPOINTS` | Becomes default at sunset |

## Quick Checklist

1. Update all endpoint URLs from `/api/v1/` to `/api/v2/`
2. Update response parsing for `data`/`meta` wrapper format
3. Update SDK clients to specify `api_version="v2"`
4. Update webhook handlers for v2 payload structure
5. Monitor deprecation warnings in server logs
6. Test integrations in staging before cutover

## Response Format Changes

### v1 (deprecated)

```json
{
  "debates": [{"id": "d1", "task": "Design a cache"}],
  "count": 1
}
```

### v2 (current)

```json
{
  "data": {
    "debates": [{"id": "d1", "task": "Design a cache"}],
    "count": 1
  },
  "meta": {
    "version": "v2",
    "timestamp": "2026-01-19T12:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### Migration pattern

```python
# v1
response = client.get("/api/v1/debates")
debates = response["debates"]

# v2
response = client.get("/api/v2/debates")
debates = response["data"]["debates"]
```

## Endpoint Mapping

| v1 Endpoint | v2 Endpoint |
|-------------|-------------|
| `POST /api/v1/debate` | `POST /api/v2/debates` |
| `GET /api/v1/debate/{id}` | `GET /api/v2/debates/{id}` |
| `POST /api/v1/agent/probe` | `POST /api/v2/agents/{id}/probes` |
| `GET /api/v1/consensus` | `GET /api/v2/debates/{id}/consensus` |
| `POST /api/v1/vote` | `POST /api/v2/debates/{id}/votes` |
| `GET /api/v1/user` | `GET /api/v2/users/me` |
| `POST /api/v1/auth/login` | `POST /api/v2/auth/token` |
| `GET /api/v1/leaderboard` | `GET /api/v2/agents/leaderboard` |
| `GET /api/v1/rankings` | `GET /api/v2/agents/rankings` |
| `GET /api/v1/metrics` | `GET /api/v2/system/metrics` |
| `GET /api/v1/health` | `GET /api/v2/system/health` |
| `GET /api/v1/status` | `GET /api/v2/health` |

### Removed in v2

| v1 Endpoint | Replacement |
|-------------|-------------|
| `GET /api/v1/status` | `GET /api/v2/health` |
| `POST /api/v1/simple-debate` | `POST /api/v2/debates` with `mode: "quick"` |

### v2-Only Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/v2/calibration/scores` | Prediction accuracy scores |
| `GET /api/v2/calibration/history/{agent_id}` | Agent calibration history |
| `POST /api/v2/insights/extract/{debate_id}` | Extract patterns from debate |
| `GET /api/v2/consensus/{id}/proof` | Cryptographic consensus proof |
| `POST /api/v2/auth/mfa/setup` | Initialize MFA |
| `GET /api/v2/privacy/data-export` | GDPR data export |
| `GET /api/v2/gallery` | Public debate gallery |

## Error Response Changes

### v1

```json
{"error": "Not found", "code": 404}
```

### v2

```json
{
  "error": {
    "code": "DEBATE_NOT_FOUND",
    "message": "Debate with ID 'd123' not found",
    "details": {"debate_id": "d123"}
  },
  "meta": {
    "version": "v2",
    "request_id": "req_abc123",
    "timestamp": "2026-01-19T12:00:00Z"
  }
}
```

## Deprecation Headers

Every v1 response includes these headers:

```http
Sunset: Mon, 01 Jun 2026 00:00:00 GMT
Deprecation: @1748736000
Link: <https://docs.aragora.ai/migration/v1-to-v2>; rel="sunset"
X-API-Version: v1
X-API-Version-Warning: API v1 is deprecated and will be removed on 2026-06-01...
X-API-Sunset: 2026-06-01
X-Deprecation-Level: warning
```

`X-Deprecation-Level` progresses through `warning` -> `critical` (< 30 days) -> `sunset`.

## SDK Migration

### Python

```python
# v1 (deprecated)
client = AragoraClient(base_url="https://api.aragora.io")
debates = client.getDebates()

# v2 (recommended)
client = AragoraClient(base_url="https://api.aragora.io", api_version="v2")
debates = client.debates.list()
```

### TypeScript

```typescript
// v1 (deprecated)
const client = new AragoraClient({ apiVersion: 'v1' });
const debates = await client.getDebates();

// v2 (recommended - default in SDK 1.0+)
const client = new AragoraClient({ apiVersion: 'v2' });
const debates = await client.debates.list();
```

## Webhook Payload Changes

### v1

```json
{
  "event": "debate.completed",
  "debate_id": "d1",
  "consensus": "reached",
  "timestamp": 1705600000
}
```

### v2

```json
{
  "event": "debate.completed",
  "version": "v2",
  "data": {
    "debate_id": "d1",
    "consensus": {"status": "reached", "confidence": 0.95}
  },
  "meta": {"timestamp": "2026-01-19T12:00:00Z", "delivery_id": "del_xyz789"}
}
```

## Operator Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ARAGORA_DISABLE_V1_DEPRECATION` | `false` | Suppress deprecation headers |
| `ARAGORA_BLOCK_SUNSET_ENDPOINTS` | `true` | Return 410 for sunset endpoints |
| `ARAGORA_LOG_DEPRECATED_USAGE` | `true` | Log v1 endpoint usage |

## Monitoring Migration Progress

```bash
# Check v1 usage in logs
grep "v1_api_access" logs/*.log | wc -l

# Top v1 endpoints still used
grep "v1_api_access" logs/*.log | sort | uniq -c | sort -rn

# Deprecation stats API
curl https://api.aragora.io/api/v2/system/deprecation-stats
```

## Architecture Reference

The v1 sunset infrastructure consists of:

| Module | Purpose |
|--------|---------|
| `aragora/server/versioning/constants.py` | Central sunset dates and URLs |
| `aragora/server/versioning/router.py` | `APIVersion` enum and `VersionedRouter` |
| `aragora/server/versioning/deprecation.py` | `DeprecationRegistry` and decorators |
| `aragora/server/middleware/deprecation.py` | Header injection middleware |
| `aragora/server/middleware/deprecation_enforcer.py` | Sunset blocking and pattern matching |
| `aragora/server/response_utils.py` | BaseHTTPRequestHandler header injection |

Deprecations are registered at server startup in `unified_server.py` and `fastapi/factory.py`
via `register_default_deprecations()`.

## Related Documentation

- [SDK Package Migration](./MIGRATION_GUIDE.md) -- `aragora-client` to `aragora-sdk`
- [Full Migration Reference](../status/MIGRATION_V1_TO_V2.md) -- Detailed endpoint mappings
- [API Versioning](../api/API_VERSIONING.md) -- Versioning strategy
