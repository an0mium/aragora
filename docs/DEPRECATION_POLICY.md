# API Deprecation Policy

This document outlines Aragora's API deprecation policy, versioning strategy, and migration guidance.

## Versioning Strategy

Aragora uses **path-based API versioning**:

```
/api/v1/debates   # Version 1 (deprecated)
/api/v2/debates   # Version 2 (current)
```

### Version Detection

The API version is determined in this order:

1. **Path prefix**: `/api/v2/debates` uses V2
2. **Accept header**: `Accept: application/vnd.aragora.v2+json`
3. **Default**: V2 for new requests, V1 for legacy `/api/debates` paths

## Current API Versions

| Version | Status | Sunset Date | Notes |
|---------|--------|-------------|-------|
| V1 | **Deprecated** | 2026-06-01 | Legacy, still functional |
| V2 | Current | - | Recommended for all new integrations |

## Deprecation Timeline

When an API version is deprecated:

1. **Announcement** (6+ months before sunset)
   - Deprecation headers added to all responses
   - Documentation updated
   - Changelog entry published

2. **Warning Period** (3-6 months)
   - V1 endpoints return `X-API-Deprecated: true` header
   - `X-API-Sunset: 2026-06-01` header indicates removal date
   - Console warnings logged for deprecated calls

3. **Migration Assistance** (1-3 months before sunset)
   - Additional logging enabled
   - Email notifications to high-volume API users (if contact available)

4. **Sunset**
   - V1 endpoints return `410 Gone` with migration guidance
   - Traffic automatically redirected to V2 equivalent (best effort)

## Response Headers

All API responses include versioning headers:

```http
X-API-Version: v2
X-API-Supported-Versions: v1,v2
```

For deprecated versions:

```http
X-API-Version: v1
X-API-Deprecated: true
X-API-Sunset: 2026-06-01
X-API-Migration: Use /api/v2/ prefix for versioned endpoints
```

For legacy (unversioned) paths:

```http
X-API-Legacy: true
X-API-Migration: Use /api/v1/ prefix for versioned endpoints
```

## Migration Guide: V1 to V2

### Breaking Changes

V2 introduces the following changes:

1. **Response envelope** (planned)
   - V1: Direct data in response body
   - V2: `{ "data": {...}, "meta": {...} }` envelope

2. **Pagination** (planned)
   - V1: `offset` + `limit` parameters
   - V2: Cursor-based pagination with `cursor` parameter

3. **Error format** (planned)
   - V1: `{ "error": "message" }`
   - V2: `{ "error": { "code": "ERROR_CODE", "message": "...", "details": {...} } }`

### Endpoint Mappings

| V1 Endpoint | V2 Endpoint | Changes |
|-------------|-------------|---------|
| `/api/debates` | `/api/v2/debates` | Response envelope |
| `/api/debates/:id` | `/api/v2/debates/:id` | Response envelope |
| `/api/agents` | `/api/v2/agents` | Response envelope |
| `/api/health` | `/api/v2/health` | No changes |

### Code Migration Example

**Before (V1):**
```python
response = requests.get("https://api.aragora.com/api/debates")
debates = response.json()  # List directly
```

**After (V2):**
```python
response = requests.get("https://api.aragora.com/api/v2/debates")
data = response.json()
debates = data["data"]  # Unwrap from envelope
meta = data.get("meta", {})  # Access pagination, etc.
```

## Checking Your Version

Use the health endpoint to verify which version you're using:

```bash
# Check current version
curl -I https://api.aragora.com/api/v2/health

# Response headers show version info
X-API-Version: v2
X-API-Supported-Versions: v1,v2
```

## When We Deprecate

An API version may be deprecated when:

1. **Security issues** that cannot be fixed without breaking changes
2. **Performance improvements** requiring structural changes
3. **New standards adoption** (e.g., updated OAuth, pagination patterns)
4. **Simplification** to reduce maintenance burden

We commit to:
- **6+ months notice** before any version sunset
- **Overlap period** where both versions are functional
- **Clear migration documentation** for each version transition

## Contact

For API deprecation questions or migration assistance:
- GitHub Issues: https://github.com/an0mium/aragora/issues
- Tag: `api-migration`

---

*Last updated: 2026-01-11*
