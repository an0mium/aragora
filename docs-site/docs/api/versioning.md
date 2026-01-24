---
title: API Versioning Strategy
description: API Versioning Strategy
---

# API Versioning Strategy

Aragora uses URL prefix versioning with header-based fallback for API version management.

## Version Format

### URL Prefix (Recommended)
```
GET /api/v1/debates
GET /api/v2/debates
```

### Header-Based
```http
GET /api/debates
X-API-Version: 2
```

### Accept Header
```http
GET /api/debates
Accept: application/json; version=2
```

## Current Versions

| Version | Status | Released | Sunset Date |
|---------|--------|----------|-------------|
| v1 | **Deprecated** | 2024-01-01 | **2026-06-01** |
| v2 | Stable (Current) | 2025-01-01 | - |

> **Important:** API v1 will be removed on June 1, 2026. Please migrate to v2 before this date.
> All v1 endpoints now return `Sunset: 2026-06-01` and `Deprecation` headers.

## Version Selection Priority

1. URL path prefix (`/api/v1/...`)
2. `X-API-Version` header
3. `Accept` header version parameter
4. Default to v1

## Deprecation Policy

For the complete deprecation policy covering API endpoints, SDK methods, configuration options, and internal APIs, see **[DEPRECATION_POLICY.md](../contributing/deprecation)**.

### Summary

Aragora follows a **2 minor version grace period** for all deprecations:
- Deprecated in v2.1.0 -> Removed in v2.3.0
- Minimum 6 months notice for public APIs

### Timeline
- **Warning**: 6+ months before sunset (deprecation announced)
- **Critical**: 30 days before sunset (final warning)
- **Sunset**: Endpoint removed

### Headers
Deprecated endpoints include:
- `Deprecation: @<timestamp>` (RFC 8594)
- `Sunset: <date>` (ISO 8601)
- `Link: <replacement>; rel="successor-version"`
- `X-Deprecation-Level: warning|critical|sunset`

### Example Response Headers
```http
HTTP/1.1 200 OK
Deprecation: @1735689600
Sunset: 2026-06-01
Link: </api/v2/users>; rel="successor-version"
X-Deprecation-Level: warning
```

## Migration Guide: v1 → v2

This comprehensive guide helps you migrate from API v1 to v2. The migration involves changes to response formats, endpoint names, authentication, and error handling.

### Summary of Breaking Changes

| Category | Change | Impact |
|----------|--------|--------|
| Response format | Wrapped with `data` and `meta` | All responses |
| Endpoints | Several renamed/restructured | 15 endpoints |
| Authentication | OAuth 2.0 required for some endpoints | Auth endpoints |
| Error format | Standardized error codes | Error handling |
| Rate limits | Per-endpoint limits | High-traffic apps |

### Response Format Changes

**v1** returns data directly:
```json
{
  "debates": [...],
  "total": 100
}
```

**v2** wraps with metadata:
```json
{
  "data": {
    "debates": [...],
    "total": 100
  },
  "meta": {
    "version": "v2",
    "timestamp": "2025-01-18T12:00:00Z",
    "request_id": "req_abc123xyz"
  },
  "links": {
    "self": "/api/v2/debates?limit=20&offset=0",
    "next": "/api/v2/debates?limit=20&offset=20"
  }
}
```

**Migration code (Python):**
```python
# v1: Access data directly
debates = response.json()["debates"]

# v2: Access via data wrapper
debates = response.json()["data"]["debates"]

# Helper function for both versions
def get_data(response, version="v2"):
    data = response.json()
    return data["data"] if version == "v2" else data
```

**Migration code (TypeScript):**
```typescript
// v1: Access data directly
const debates = response.debates;

// v2: Access via data wrapper
const debates = response.data.debates;

// Type definitions
interface ApiResponseV2<T> {
  data: T;
  meta: { version: string; timestamp: string; request_id: string };
  links?: { self: string; next?: string; prev?: string };
}
```

### Endpoint Changes

| v1 Endpoint | v2 Endpoint | Notes |
|-------------|-------------|-------|
| `POST /api/v1/debate` | `POST /api/v2/debates` | Pluralized, different request body |
| `GET /api/v1/debate/:id` | `GET /api/v2/debates/:id` | Pluralized |
| `GET /api/v1/agents` | `GET /api/v2/agents` | Response format changed |
| `GET /api/v1/leaderboard` | `GET /api/v2/agents/leaderboard` | Nested under agents |
| `GET /api/v1/rankings` | `GET /api/v2/agents/rankings` | Nested under agents |
| `POST /api/v1/auth/login` | `POST /api/v2/auth/token` | OAuth 2.0 flow |
| `GET /api/v1/user` | `GET /api/v2/users/me` | Restructured |
| `GET /api/v1/consensus/:id` | `GET /api/v2/debates/:id/consensus` | Nested under debates |
| `GET /api/v1/metrics` | `GET /api/v2/system/metrics` | Nested under system |
| `GET /api/v1/health` | `GET /api/v2/system/health` | Nested under system |

### Request Body Changes

**Creating a Debate:**

v1:
```json
{
  "topic": "Should we use microservices?",
  "agents": ["claude", "gpt-4"],
  "max_rounds": 3
}
```

v2:
```json
{
  "task": "Should we use microservices?",
  "agents": ["claude", "gpt-4"],
  "rounds": 3,
  "consensus": "majority",
  "auto_select": false
}
```

| v1 Field | v2 Field | Notes |
|----------|----------|-------|
| `topic` | `task` | Renamed |
| `max_rounds` | `rounds` | Renamed |
| - | `consensus` | New field (default: "majority") |
| - | `auto_select` | New field (default: true) |
| - | `context` | New field (optional) |

### Authentication Changes

**v1: API Key**
```bash
curl -H "Authorization: Bearer sk_v1_abc123" \
  https://api.aragora.ai/api/v1/debates
```

**v2: OAuth 2.0 or API Key**
```bash
# API Key (still supported)
curl -H "Authorization: Bearer sk_v2_abc123" \
  https://api.aragora.ai/api/v2/debates

# OAuth 2.0 (new)
curl -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..." \
  https://api.aragora.ai/api/v2/debates
```

OAuth 2.0 is now required for:
- Organization management endpoints
- User management endpoints
- Webhook configuration

### Error Format Changes

**v1:**
```json
{
  "error": "Debate not found",
  "status": 404
}
```

**v2:**
```json
{
  "error": "Debate not found",
  "code": "NOT_FOUND",
  "resource_type": "debate",
  "resource_id": "deb_abc123",
  "trace_id": "req_xyz789",
  "support_url": "https://github.com/an0mium/aragora/issues"
}
```

**Error handling migration:**
```python
# v1
if response.status_code == 404:
    print(f"Error: {response.json()['error']}")

# v2
error = response.json()
if response.status_code >= 400:
    print(f"Error [{error['code']}]: {error['error']}")
    print(f"Trace ID: {error.get('trace_id')}")
```

### Rate Limit Changes

v2 introduces per-endpoint rate limits with clear headers:

| Endpoint Category | Free Tier | Pro Tier | Enterprise |
|-------------------|-----------|----------|------------|
| List endpoints | 60/min | 300/min | 1000/min |
| Create debate | 10/day | 100/day | Unlimited |
| Export | 5/hour | 50/hour | 500/hour |
| WebSocket | 1 conn | 5 conn | 20 conn |

**Rate limit headers (v2):**
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705579200
```

### Migration Checklist

- [ ] Update API base URL to include `/v2`
- [ ] Update response parsing to use `.data` wrapper
- [ ] Rename request fields (`topic` → `task`, `max_rounds` → `rounds`)
- [ ] Update error handling for new error format
- [ ] Add rate limit handling with backoff
- [ ] Update authentication for OAuth 2.0 (if using org endpoints)
- [ ] Test all endpoints in staging environment
- [ ] Update client SDK to v2 compatible version
- [ ] Monitor deprecation warnings in production
- [ ] Remove v1 code paths after successful migration

### Gradual Migration Strategy

1. **Phase 1: Dual Support** (Recommended)
   - Update client to support both v1 and v2 responses
   - Use feature flag to switch between versions
   - Test v2 with subset of traffic

2. **Phase 2: v2 Primary**
   - Make v2 the default
   - Keep v1 fallback for edge cases
   - Monitor error rates

3. **Phase 3: v1 Removal**
   - Remove v1 code paths
   - Clean up feature flags
   - Complete migration before sunset date

### SDK Updates

Update your SDK to the latest version for automatic v2 support:

```bash
# Python
pip install aragora>=2.0.0

# Node.js
npm install @aragora/sdk@^2.0.0

# Go (planned)
# Contact support for early access builds.
```

### Getting Help

- **Documentation:** https://docs.aragora.ai/migration
- **Support:** support@aragora.ai
- **GitHub Issues:** https://github.com/an0mium/aragora/issues
- **Discord:** https://discord.gg/aragora

## Usage

### Python Client
```python
from aragora.client import AragoraClient

# Specify version
client = AragoraClient(api_version="v2")

# Or per-request
response = client.get("/debates", api_version="v2")
```

### TypeScript Client
```typescript
import { AragoraClient } from '@aragora/sdk';

// Specify version
const client = new AragoraClient({ apiVersion: 'v2' });

// Or per-request
const debates = await client.get('/debates', { apiVersion: 'v2' });
```

### curl
```bash
# URL prefix (recommended)
curl https://api.aragora.io/api/v2/debates

# Header-based
curl -H "X-API-Version: 2" https://api.aragora.io/api/debates
```

## Monitoring

### Metrics
- `aragora_api_requests_total{version="v1"}` - Requests by version
- `aragora_deprecated_endpoint_calls_total` - Deprecated endpoint usage
- `aragora_version_adoption{version="v2"}` - Version adoption rate

### Alerts
- Warning when sunset endpoint usage > 100/hour
- Critical when sunset < 30 days with active usage

## Best Practices

1. **Always specify version** - Don't rely on defaults
2. **Subscribe to deprecation notices** - Monitor sunset dates
3. **Test with new versions early** - Use beta/alpha in staging
4. **Use semantic versioning** - Major version = breaking changes
5. **Plan migrations** - Start migration 3+ months before sunset
6. **Review deprecation policy** - See [DEPRECATION_POLICY.md](../contributing/deprecation) for complete guidelines
