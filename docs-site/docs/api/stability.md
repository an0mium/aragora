---
title: API Stability & Versioning Policy
description: API Stability & Versioning Policy
---

# API Stability & Versioning Policy

This document defines Aragora's API versioning strategy, stability guarantees, deprecation policies, and backward compatibility commitments.

## Versioning Scheme

### Semantic Versioning

Aragora follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH

Examples:
- 1.0.0 → 1.0.1  (patch: bug fixes, no API changes)
- 1.0.0 → 1.1.0  (minor: new features, backward compatible)
- 1.0.0 → 2.0.0  (major: breaking changes)
```

### API Version Header

All API responses include version information:

```http
X-API-Version: 1.3.0
X-API-Deprecation-Notice: endpoint-name (sunset: 2026-07-01)
```

### URL-Based Versioning

For major version changes, URL prefixes are supported:

```
/api/v1/debates     # Current stable
/api/v2/debates     # Next major (when available)
```

## Stability Tiers

### Tier Definitions

| Tier | Stability | Breaking Changes | Notice Period |
|------|-----------|------------------|---------------|
| **Stable** | Production-ready | Major versions only | 6 months |
| **Beta** | Feature-complete | Minor versions | 3 months |
| **Alpha** | Experimental | Any release | 2 weeks |
| **Internal** | Not for external use | Any time | None |

### Endpoint Classification

```
Stable Endpoints (v1):
├── /api/debates/*           # Core debate functionality
├── /api/agents/*            # Agent management
├── /api/consensus/*         # Consensus detection
├── /api/memory/*            # Memory operations
├── /api/leaderboard/*       # Rankings and ELO
├── /api/health              # Health checks
└── /api/auth/*              # Authentication

Beta Endpoints:
├── /api/gauntlet/*          # Adversarial testing
├── /api/genesis/*           # Agent evolution
├── /api/belief/*            # Belief networks
└── /api/verification/*      # Formal verification

Alpha Endpoints:
├── /api/features/discover   # Feature discovery
├── /api/pulse/*             # Trending topics
└── /api/experimental/*      # New features
```

## Deprecation Policy

### Timeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Deprecation Timeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Announcement    Active Deprecation    Sunset    Removal    │
│       │                │                  │          │      │
│       ▼                ▼                  ▼          ▼      │
│  ────────────────────────────────────────────────────────   │
│  │   6 months    │   3 months   │   3 months   │           │
│  │               │              │              │           │
│  │  Deprecation  │   Warning    │   Error      │  Gone     │
│  │  notice added │   headers    │   responses  │           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase Details

| Phase | Duration | Behavior |
|-------|----------|----------|
| **Announcement** | T+0 | Deprecation documented, notice headers added |
| **Active Deprecation** | T+6 months | Warning logs, migration guides available |
| **Sunset Warning** | T+9 months | 410 Gone responses with redirect info |
| **Removal** | T+12 months | Endpoint removed completely |

### Deprecation Notice Format

```http
HTTP/1.1 200 OK
X-API-Deprecation: true
X-API-Sunset-Date: 2026-07-01
X-API-Replacement: /api/v2/debates
Link: </docs/migration/v2-debates>; rel="deprecation"

{
  "data": { ... },
  "_meta": {
    "deprecation": {
      "deprecated": true,
      "sunset": "2026-07-01",
      "replacement": "/api/v2/debates",
      "migration_guide": "https://docs.aragora.ai/migration/v2-debates"
    }
  }
}
```

## Breaking Change Policy

### What Constitutes a Breaking Change

**Breaking (requires major version):**
- Removing an endpoint
- Removing a required field from response
- Adding a required field to request
- Changing field types (string → number)
- Changing error codes or status codes
- Changing authentication requirements
- Removing enum values

**Non-Breaking (can be in minor version):**
- Adding new endpoints
- Adding optional fields to requests
- Adding fields to responses
- Adding new enum values
- Adding new error codes (while keeping existing)
- Performance improvements
- Bug fixes that don't change API contract

### Breaking Change Notification

1. **GitHub Release Notes**
   - Tagged with `breaking-change` label
   - Detailed migration instructions

2. **Changelog**
   - CHANGELOG.md entry with `BREAKING:` prefix
   - Links to migration guide

3. **Email Notification** (for registered API users)
   - Sent at announcement, 3 months, and 1 month before sunset

4. **API Response Headers**
   - `X-API-Breaking-Change: <description>`
   - Added 6 months before change

### Example Changelog Entry

```markdown
## [2.0.0] - 2026-07-01

### BREAKING CHANGES

- **`POST /api/debates`**: `agents` field now requires agent IDs instead of names
  - Migration: Use `/api/agents` to look up IDs
  - See: docs/migration/v2-agent-ids.md

- **`GET /api/consensus`**: Response format changed
  - Old: `{ "consensus": true, "value": "..." }`
  - New: `{ "result": { "reached": true, "position": "..." } }`
  - See: docs/migration/v2-consensus-format.md
```

## Version Compatibility Matrix

### Current Compatibility

| Client Version | API v1.0 | API v1.1 | API v1.2 | API v1.3 |
|----------------|----------|----------|----------|----------|
| SDK 1.0.x | Full | Partial | Partial | Partial |
| SDK 1.1.x | Full | Full | Partial | Partial |
| SDK 1.2.x | Full | Full | Full | Partial |
| SDK 1.3.x | Full | Full | Full | Full |

**Legend:**
- **Full**: All features work correctly
- **Partial**: Core features work, new features unavailable
- **None**: Not compatible

### Minimum Supported Versions

| Component | Minimum Version | End of Support |
|-----------|-----------------|----------------|
| API | v1.0.0 | 2027-01-01 |
| Python SDK | 1.0.0 | 2027-01-01 |
| TypeScript SDK | 1.0.0 | 2027-01-01 |
| WebSocket Protocol | v1 | 2027-01-01 |

## Migration Guides

### Accessing Migration Documentation

Migration guides are available at:
- **Docs Site**: https://docs.aragora.ai/migration/
- **GitHub**: /docs/migration/
- **API**: `GET /api/docs/migration/\{from_version\}/\{to_version\}`

### Migration Guide Structure

```markdown
# Migration Guide: v1.x → v2.0

## Overview
Brief description of major changes

## Breaking Changes
1. Change description
   - Old behavior
   - New behavior
   - Code example

## New Features
List of new capabilities

## Deprecated Features
Features marked for future removal

## Step-by-Step Migration
1. Update SDK version
2. Update endpoint calls
3. Update response handling
4. Test thoroughly

## Rollback Plan
How to revert if issues arise
```

## API Lifecycle

### Feature Lifecycle

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Alpha   │───▶│   Beta   │───▶│  Stable  │───▶│Deprecated│
│          │    │          │    │          │    │          │
│ 2-4 weeks│    │ 1-3 months│   │ 1+ years │    │ 6 months │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Promotion Criteria

**Alpha → Beta:**
- Feature complete
- Basic documentation
- Internal testing passed
- No known critical bugs

**Beta → Stable:**
- Production-tested for 1+ month
- Full documentation
- SDK support
- Performance benchmarks met
- Security review passed

**Stable → Deprecated:**
- Replacement available
- Migration guide published
- 6-month notice given

## SDK Version Policy

### SDK-API Compatibility

```python
# Python SDK
from aragora import Client

# Specify API version explicitly
client = Client(api_version="1.3")

# Or use latest stable
client = Client(api_version="stable")
```

### SDK Update Policy

| API Change Type | SDK Update |
|-----------------|------------|
| Patch (1.3.0 → 1.3.1) | Optional |
| Minor (1.3.0 → 1.4.0) | Recommended within 30 days |
| Major (1.x → 2.0) | Required before sunset |

## Error Handling for Deprecated Endpoints

### Warning Phase (T+6 to T+9 months)

```http
HTTP/1.1 200 OK
Warning: 299 - "Deprecated API: migrate to /api/v2/debates by 2026-07-01"
```

### Sunset Phase (T+9 to T+12 months)

```http
HTTP/1.1 410 Gone
Content-Type: application/json

{
  "error": {
    "code": "ENDPOINT_SUNSET",
    "message": "This endpoint has been sunset",
    "replacement": "/api/v2/debates",
    "migration_guide": "https://docs.aragora.ai/migration/v2-debates",
    "support_contact": "support@aragora.ai"
  }
}
```

## Stability Commitments

### What We Guarantee

1. **6-month deprecation notice** for stable endpoints
2. **12-month total lifecycle** from deprecation to removal
3. **Backward compatible** minor and patch releases
4. **Migration guides** for all breaking changes
5. **SDK updates** before API changes take effect

### What We Don't Guarantee

1. **Alpha/experimental endpoints** - may change without notice
2. **Internal endpoints** - not for external use
3. **Performance characteristics** - may vary
4. **Rate limits** - may be adjusted
5. **Undocumented behavior** - may change

## Changelog

All API changes are documented in:
- `CHANGELOG.md` - Full history
- GitHub Releases - Per-version notes
- `/api/changelog` - Programmatic access

### Changelog Format

```markdown
## [1.3.0] - 2026-01-14

### Added
- `GET /api/features/discover` - Feature discovery endpoint
- `GET /api/belief/graph` - Belief network visualization
- WebSocket support for gauntlet streaming

### Changed
- Improved consensus detection accuracy
- Enhanced ELO calculation for team debates

### Deprecated
- `GET /api/rankings` - Use `/api/leaderboard` instead (sunset: 2026-07-14)

### Fixed
- CORS headers on error responses
- WebSocket reconnection handling
```

## Contact & Support

| Need | Contact |
|------|---------|
| API questions | api-support@aragora.ai |
| Breaking change concerns | api-stability@aragora.ai |
| Security issues | security@aragora.ai |
| General support | support@aragora.ai |

## Version History

| Version | Release Date | Status | Sunset Date |
|---------|--------------|--------|-------------|
| 1.3.0 | 2026-01-14 | Current | - |
| 1.2.0 | 2025-11-01 | Supported | - |
| 1.1.0 | 2025-08-01 | Supported | - |
| 1.0.0 | 2025-05-01 | Supported | 2027-01-01 |
