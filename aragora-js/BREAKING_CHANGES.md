# TypeScript SDK Breaking Changes

This document tracks breaking changes specific to the Aragora TypeScript SDK (`@aragora/sdk`). For core API breaking changes, see the main [BREAKING_CHANGES.md](../docs/BREAKING_CHANGES.md).

---

## Version 2.x

### v2.0.3 (2026-01-20)

**No breaking changes.** Added ConnectorsAPI and new types.

**New Features:**
- ConnectorsAPI for enterprise data source management
- Full TypeScript types for connector operations

---

### v2.0.0 (2026-01-17)

#### Breaking Changes

| Change | Before | After | Migration |
|--------|--------|-------|-----------|
| Version alignment | Independent versioning | Aligned with core Aragora version | Update to `@aragora/sdk@^2.0.0` |
| API version default | v1 | v2 | Pass `apiVersion: 'v1'` to keep old behavior |
| Response format | Direct data | Wrapped in `data`/`meta` | Access via `response.data` |

#### Migration Example

```typescript
// Before (v1.x)
import { AragoraClient } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL,
});

const debates = await client.getDebates();
const debate = await client.createDebate({ topic: 'GraphQL vs REST', maxRounds: 3 });

// After (v2.x)
import { AragoraClient } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL,
  // apiVersion: 'v2' is now the default
});

const response = await client.debates.list();
const debates = response.data.debates;

const createResponse = await client.debates.create({
  task: 'GraphQL vs REST',  // 'topic' renamed to 'task'
  rounds: 3  // 'maxRounds' renamed to 'rounds'
});
const debate = createResponse.data;
```

#### Response Format Change

```typescript
// Before (v1.x) - Direct data access
const response = await client.debates.list();
const debates = response.debates;
const count = response.count;

// After (v2.x) - Wrapped response
const response = await client.debates.list();
const debates = response.data.debates;
const count = response.data.count;
const meta = response.meta; // { version, timestamp, request_id }
```

#### Type Changes

```typescript
// Before (v1.x)
interface DebateResponse {
  debates: Debate[];
  count: number;
}

// After (v2.x)
interface ApiResponse<T> {
  data: T;
  meta: {
    version: string;
    timestamp: string;
    request_id: string;
  };
  links?: {
    self: string;
    next?: string;
    prev?: string;
  };
}

interface DebateListData {
  debates: Debate[];
  count: number;
}

// Usage: ApiResponse<DebateListData>
```

---

## Version 1.x

### v1.0.0 (2026-01-14)

**Initial stable release.**

#### Breaking Changes from Pre-1.0

| Change | Before | After |
|--------|--------|-------|
| Package name | `aragora-sdk` | `@aragora/sdk` |
| Minimum Node.js | 16.x | 18.x |
| ESM support | CommonJS only | ESM + CommonJS |

**Migration:**

```bash
# Update package name
npm uninstall aragora-sdk
npm install @aragora/sdk@^1.0.0
```

```typescript
// Before
import { AragoraClient } from 'aragora-sdk';

// After
import { AragoraClient } from '@aragora/sdk';
```

---

## Upcoming Breaking Changes

### Scheduled for v3.0.0

No breaking changes currently scheduled.

---

## Migration Guides

- [API v1 to v2 Migration](../docs/MIGRATION_V1_TO_V2.md) - Complete guide for API migration
- [WebSocket Events](./docs/WEBSOCKET_EVENTS.md) - WebSocket streaming documentation

---

## Deprecation Notices

Deprecated methods emit console warnings in development mode. To suppress:

```typescript
const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL,
  suppressDeprecationWarnings: true, // Not recommended
});
```

---

*Last updated: 2026-01-31*
