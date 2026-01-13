# API Versioning

Aragora uses path-based API versioning to enable backward-compatible evolution of the API.

## Table of Contents

- [Overview](#overview)
- [Version Format](#version-format)
- [Using Versioned Endpoints](#using-versioned-endpoints)
- [Response Headers](#response-headers)
- [Legacy Endpoint Support](#legacy-endpoint-support)
- [Migration Guide](#migration-guide)
- [Breaking Change Policy](#breaking-change-policy)

---

## Overview

All API endpoints support versioning through URL path prefixes:

```
/api/v1/debates     # Versioned (recommended)
/api/debates        # Legacy (deprecated, still works)
```

Version information is included in every API response via headers, allowing clients to detect when they're using legacy endpoints.

---

## Version Format

### Path-Based (Recommended)

```
/api/v{major}/resource
```

Examples:
```
GET /api/v1/debates
POST /api/v1/debates
GET /api/v1/agent/anthropic-api/profile
GET /api/v1/leaderboard
```

### Header-Based (Alternative)

For clients that cannot modify URL paths, version can be specified via Accept header:

```http
GET /api/debates HTTP/1.1
Accept: application/vnd.aragora.v1+json
```

Response will use the specified version.

---

## Using Versioned Endpoints

### Recommended Approach

Always use versioned endpoints in production:

```python
# Python (requests)
import requests

BASE_URL = "https://api.aragora.ai/api/v1"

# List debates
response = requests.get(f"{BASE_URL}/debates")

# Create debate
response = requests.post(f"{BASE_URL}/debates", json={
    "task": "Discuss API design patterns",
    "agents": ["anthropic-api", "openai-api"],
})
```

```javascript
// JavaScript (fetch)
const BASE_URL = 'https://api.aragora.ai/api/v1';

// List debates
const debates = await fetch(`${BASE_URL}/debates`).then(r => r.json());

// Get agent profile
const profile = await fetch(`${BASE_URL}/agent/anthropic-api/profile`).then(r => r.json());
```

```bash
# cURL
curl -X GET https://api.aragora.ai/api/v1/debates \
  -H "Authorization: Bearer $TOKEN"
```

---

## Response Headers

Every API response includes version headers:

### Standard Headers

| Header | Description | Example |
|--------|-------------|---------|
| `X-API-Version` | Version used for this request | `v1` |
| `X-API-Supported-Versions` | All supported versions | `v1` |

### Legacy Path Headers

When using unversioned paths (`/api/debates`), additional headers are included:

| Header | Description | Example |
|--------|-------------|---------|
| `X-API-Legacy` | Indicates legacy path usage | `true` |
| `X-API-Migration` | Suggests migration to versioned path | `Use /api/v1/ prefix...` |

### Deprecated Version Headers

When a version is deprecated (still works but scheduled for removal):

| Header | Description | Example |
|--------|-------------|---------|
| `X-API-Deprecated` | Indicates deprecated version | `true` |
| `X-API-Sunset` | Date when version will be removed | `2026-06-01` |

### Example Response

```http
HTTP/1.1 200 OK
Content-Type: application/json
X-API-Version: v1
X-API-Supported-Versions: v1

{"debates": [...]}
```

Legacy path response:
```http
HTTP/1.1 200 OK
Content-Type: application/json
X-API-Version: v1
X-API-Supported-Versions: v1
X-API-Legacy: true
X-API-Migration: Use /api/v1/ prefix for versioned endpoints

{"debates": [...]}
```

---

## Legacy Endpoint Support

### Current Status

Legacy (unversioned) endpoints are fully supported but emit deprecation warnings:

| Legacy Path | Versioned Path | Status |
|-------------|----------------|--------|
| `/api/debates` | `/api/v1/debates` | Supported with warning |
| `/api/agent/*` | `/api/v1/agent/*` | Supported with warning |
| `/api/leaderboard` | `/api/v1/leaderboard` | Supported with warning |

### Behavior

1. Legacy paths route to the same handlers as versioned paths
2. Response includes `X-API-Legacy: true` header
3. No functional difference in behavior
4. Full deprecation timeline TBD

### Detection

Check for legacy usage in your client:

```python
response = requests.get("https://api.aragora.ai/api/debates")

if response.headers.get("X-API-Legacy") == "true":
    print("Warning: Using legacy endpoint, migrate to /api/v1/")
```

---

## Migration Guide

### Step 1: Update Base URL

```python
# Before
BASE_URL = "https://api.aragora.ai/api"

# After
BASE_URL = "https://api.aragora.ai/api/v1"
```

### Step 2: Update All Endpoint Calls

No other changes needed - all endpoints work identically with the version prefix.

### Step 3: Verify Headers

Check that responses include `X-API-Version: v1` without `X-API-Legacy`.

### Step 4: Update Documentation

Update any API documentation to reference versioned endpoints.

---

## Breaking Change Policy

### Versioning Rules

1. **Major versions (v1 → v2)**: May contain breaking changes
2. **Minor versions**: Add features, never remove
3. **Patch versions**: Bug fixes only

### What Constitutes a Breaking Change

- Removing an endpoint
- Removing a required field from response
- Changing field types
- Changing authentication requirements
- Changing rate limit behavior

### What Is NOT a Breaking Change

- Adding new endpoints
- Adding optional fields to responses
- Adding optional request parameters
- Performance improvements
- Bug fixes

### Deprecation Timeline

When an endpoint or version is deprecated:

1. **Announcement**: Deprecation notice in release notes
2. **Warning Headers**: `X-API-Deprecated: true` and `X-API-Sunset` date
3. **Migration Period**: Minimum 90 days
4. **Sunset**: Endpoint returns `410 Gone`

---

## Supported Versions

| Version | Status | Notes |
|---------|--------|-------|
| `v1` | **Current** | Recommended for all new integrations |

---

## Implementation Details

### Server Configuration

Version configuration is managed in `aragora/server/versioning.py`:

```python
from aragora.server.versioning import (
    APIVersion,
    get_version_config,
    set_version_config,
    VersionConfig,
)

# View current config
config = get_version_config()
print(config.current)  # APIVersion.V1
print(config.supported)  # [APIVersion.V1]

# Custom config (for testing)
set_version_config(VersionConfig(
    current=APIVersion.V1,
    supported=[APIVersion.V1],
    deprecated=[],
    sunset_dates={},
))
```

### Route Matching

The handler registry automatically normalizes versioned paths:

```
/api/v1/debates → /api/debates (for handler matching)
```

Handlers only need to define routes for unversioned paths:

```python
class DebatesHandler(BaseHandler):
    ROUTES = [
        "/api/debates",      # Matches both /api/debates AND /api/v1/debates
        "/api/debates/list",
    ]
```

---

## See Also

- [DEPRECATION_POLICY.md](./DEPRECATION_POLICY.md) - Full deprecation policy
- [API_REFERENCE.md](./API_REFERENCE.md) - Complete API documentation
- [SECURITY.md](./SECURITY.md) - Authentication and security
