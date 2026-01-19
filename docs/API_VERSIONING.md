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

### Timeline
- **Warning**: 6+ months before sunset
- **Critical**: 30 days before sunset
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
Sunset: 2025-01-01
Link: </api/v2/users>; rel="successor-version"
X-Deprecation-Level: warning
```

## Migration Guide

### v1 → v2

#### Response Format Changes
v1 returns data directly:
```json
{
  "debates": [...]
}
```

v2 wraps with metadata:
```json
{
  "data": {
    "debates": [...]
  },
  "meta": {
    "version": "v2",
    "timestamp": "2025-01-18T12:00:00Z"
  }
}
```

#### Endpoint Changes
| v1 | v2 | Notes |
|----|----|-------|
| GET /api/v1/debates | GET /api/v2/debates | Response format changed |
| POST /api/v1/debate | POST /api/v2/debates | Endpoint renamed |

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
import { AragoraClient } from 'aragora';

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
