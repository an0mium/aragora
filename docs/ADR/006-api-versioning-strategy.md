# ADR-006: API Versioning Strategy

## Status
Accepted

## Context

As Aragora matures and gains external consumers (TypeScript SDK, third-party integrations), we need a clear API versioning strategy to:
- Support breaking changes without disrupting existing clients
- Communicate deprecation timelines clearly
- Allow gradual migration paths
- Maintain backward compatibility during transitions

Key requirements:
- URL-based versioning for discoverability (`/api/v1/debates`)
- Header-based version negotiation for advanced use cases
- Clear deprecation warnings and sunset dates
- Version information in response headers

## Decision

Implement a comprehensive versioning middleware in `aragora/server/middleware/versioning.py`:

### Path-Based Versioning
```
/api/v1/debates     # Current stable
/api/v2/debates     # Future (when needed)
/debates            # Legacy (maps to v0, eventually deprecated)
```

### Version Configuration
```python
API_VERSIONS = {
    "v0": APIVersion(version="v0", status="deprecated", sunset_date="2026-06-01"),
    "v1": APIVersion(version="v1", status="stable", is_current=True),
}
```

### Response Headers
Every response includes:
- `X-API-Version`: The version used for this request
- `X-API-Deprecated`: Set when using deprecated version
- `Deprecation`: RFC 8594 compliant deprecation header
- `Sunset`: ISO 8601 sunset date for deprecated versions

### Middleware Behavior
1. Extract version from path prefix or `X-API-Version` header
2. Normalize paths (strip version prefix for internal routing)
3. Inject version headers in responses
4. Log version usage for analytics
5. Return warnings for deprecated versions

## Consequences

### Positive
- **Client clarity**: Clients know exactly which API version they're using
- **Graceful deprecation**: Clear timeline and warnings before removal
- **Analytics**: Track version adoption to inform migration timing
- **SDK support**: TypeScript SDK can target specific versions
- **Standards compliance**: RFC 8594 deprecation headers

### Negative
- **URL complexity**: Clients must include version in URLs
- **Documentation**: Must maintain docs for multiple versions
- **Testing**: Each version needs its own test coverage

### Neutral
- Legacy endpoints (without version prefix) continue to work during transition
- Version extraction happens once per request (minimal overhead)

## Related
- `aragora/server/middleware/versioning.py` - Versioning middleware and helpers
- `docs/API_VERSIONING.md` - Client documentation
