# TypeScript SDK Consolidation Roadmap

This document outlines the plan to consolidate the two TypeScript packages (`@aragora/sdk` and `@aragora/client`) into a single unified SDK.

## Current State (v2.1.x)

Two packages exist with different focuses:

| Aspect | `@aragora/sdk` | `@aragora/client` |
|--------|---------------|-------------------|
| **Version** | 2.1.15 | 2.1.15 |
| **Location** | `sdk/typescript/` | `aragora-js/` |
| **API Style** | Flat (`client.createDebate()`) | Namespaced (`client.debates.create()`) |
| **Build** | tsup (ESM + CJS) | tsc (CJS only) |
| **Tests** | Minimal | 6 test suites |
| **Best For** | Application developers | Enterprise/Control plane |

### Feature Matrix

| Feature | `@aragora/sdk` | `@aragora/client` |
|---------|:-------------:|:-----------------:|
| Basic debates | Yes | Yes |
| WebSocket streaming | Yes | Yes |
| Workflows | Yes | No |
| Explainability | Yes | No |
| Marketplace | Yes | No |
| Control Plane | No | Yes |
| Graph/Matrix debates | No | Yes |
| Formal verification | No | Yes |
| Team selection | No | Yes |

## Target State (v3.0.0)

A single `@aragora/sdk` package combining the best of both:

### Goals

- **Single source of truth** - One package to install and maintain
- **Namespace API** - Organized, discoverable API surface
- **Full feature set** - All capabilities from both packages
- **Modern build** - ESM + CJS dual output, tree-shakeable
- **Comprehensive tests** - Merged and expanded test coverage

### Target API Structure

```typescript
import { AragoraClient } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: 'https://api.aragora.ai',
  apiKey: 'your-key'
});

// Debates (from both packages)
await client.debates.create({ task: '...' });
await client.debates.list();
await client.graphDebates.create({ ... });
await client.matrixDebates.create({ ... });

// Agents
await client.agents.list();
await client.agents.get('agent-id');

// Control Plane (from @aragora/client)
await client.controlPlane.registerAgent('agent-id', [...]);
await client.controlPlane.submitTask('debate', { ... });
await client.controlPlane.getAgentStatus('agent-id');

// Verification (from @aragora/client)
await client.verification.verifyClaim({ ... });

// Workflows (from @aragora/sdk)
await client.workflows.list();
await client.workflows.execute('template-id', { ... });

// Explainability (from @aragora/sdk)
await client.explainability.getFactors('debate-id');
await client.explainability.getCounterfactuals('debate-id');

// Gauntlet
await client.gauntlet.run({ ... });
await client.gauntlet.getReceipt('receipt-id');

// WebSocket (unified)
const stream = client.createStream();
await stream.connect();
stream.on('message', (event) => { ... });
```

## Migration Timeline

### v2.2.0 (Q1 2026)

**Goal**: Prepare for consolidation

- Add deprecation warnings to `@aragora/client`
- Add namespace aliases to `@aragora/sdk`
- Port test suites from client to sdk
- Document migration path

**Breaking changes**: None

### v2.3.0 (Q1 2026)

**Goal**: Feature parity in `@aragora/sdk`

- Add Control Plane API to sdk
- Add Graph/Matrix debates to sdk
- Add Formal Verification to sdk
- Add Team Selection to sdk
- Client becomes thin wrapper around sdk

**Breaking changes**: None (client still works)

### v3.0.0 (Q2 2026)

**Goal**: Single unified SDK

- `@aragora/client` deprecated (no longer published)
- `@aragora/sdk` is the only package
- Full namespace API
- ESM-first with CJS fallback
- Complete TypeScript definitions

**Breaking changes**:
- `createClient()` -> `new AragoraClient()`
- Flat methods moved to namespaces
- Some type names may change

## Migration Guide

### From `@aragora/sdk` v2.x to v3.0.0

```typescript
// Before (v2.x)
import { createClient } from '@aragora/sdk';
const client = createClient({ baseUrl: '...' });
await client.createDebate({ ... });
await client.listAgents();

// After (v3.0.0)
import { AragoraClient } from '@aragora/sdk';
const client = new AragoraClient({ baseUrl: '...' });
await client.debates.create({ ... });
await client.agents.list();
```

### From `@aragora/client` v2.x to `@aragora/sdk` v3.0.0

```typescript
// Before (client v2.x)
import { AragoraClient } from '@aragora/client';
const client = new AragoraClient({ baseUrl: '...' });
await client.debates.run({ ... });
await client.controlPlane.submitTask({ ... });

// After (sdk v3.0.0)
import { AragoraClient } from '@aragora/sdk';
const client = new AragoraClient({ baseUrl: '...' });
await client.debates.create({ ... });  // Method renamed for consistency
await client.controlPlane.submitTask({ ... });  // Same API
```

## Implementation Phases

### Phase 1: Analysis (1 week)

- [ ] Document all methods in both packages
- [ ] Identify overlapping functionality
- [ ] Design unified namespace structure
- [ ] Plan test migration

### Phase 2: SDK Enhancement (2-3 weeks)

- [ ] Add namespace structure to sdk
- [ ] Port Control Plane API
- [ ] Port Graph/Matrix debates
- [ ] Port Formal Verification
- [ ] Migrate tests from client

### Phase 3: Deprecation (1 week)

- [ ] Add deprecation warnings to client
- [ ] Update documentation
- [ ] Announce migration timeline
- [ ] Publish v2.2.0 of both packages

### Phase 4: Client Wrapper (1 week)

- [ ] Make client a wrapper around sdk
- [ ] Ensure backwards compatibility
- [ ] Test with existing client users
- [ ] Publish v2.3.0

### Phase 5: Final Release (1 week)

- [ ] Remove client source (keep as deprecated npm package)
- [ ] Finalize sdk v3.0.0
- [ ] Update all documentation
- [ ] Announce final migration

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes for sdk users | High | Detailed migration guide, deprecation warnings |
| Breaking changes for client users | Medium | Client wrapper maintains compatibility through v2.x |
| Test coverage gaps | Medium | Merge all tests, add integration tests |
| Missing features during transition | Low | Feature flags, gradual rollout |

## Success Metrics

- [ ] Single npm package with all features
- [ ] Zero breaking changes for users following migration guide
- [ ] Test coverage >= 80%
- [ ] Documentation updated on docs.aragora.ai
- [ ] No issues reported within 2 weeks of v3.0.0 release

## Related Documentation

- [sdk/typescript/README.md](../sdk/typescript/README.md) - SDK documentation
- [aragora-js/README.md](../aragora-js/README.md) - Client documentation
- [CONTRIBUTING.md](../CONTRIBUTING.md#package-naming) - Package naming conventions
