# SDK Migration Guide

This guide helps you migrate from deprecated SDK packages to the current recommended packages.

## Quick Reference

| Language | Deprecated Package | Current Package | Status |
|----------|-------------------|-----------------|--------|
| **Python** | `aragora-client` | `aragora-sdk` | Migrate now |
| **TypeScript** | `@aragora/client` | `@aragora/sdk` | Migrate now |

## Python: `aragora-client` → `aragora-sdk`

The `aragora-client` Python package is deprecated. Use `aragora-sdk` for the full-featured client with 153 namespace modules, sync + async support, and generated types.

### Installation

```bash
# Remove old package
pip uninstall aragora-client

# Install new package
pip install aragora-sdk
```

### Import Changes

```python
# BEFORE (deprecated)
from aragora_client import AragoraClient
from aragora_client.types import Debate, ConsensusResult

# AFTER (recommended)
from aragora_sdk import AragoraClient
from aragora_sdk.types import Debate, ConsensusResult
```

### API Changes

Key improvements in `aragora-sdk`:

```python
# BEFORE: Async-only (aragora-client)
import asyncio
from aragora_client import AragoraClient

async def main():
    client = AragoraClient("http://localhost:8080")
    debate = await client.debates.create(task="...")  # Async only

asyncio.run(main())

# AFTER: Sync + async support (aragora-sdk)
from aragora_sdk import AragoraClient

# Synchronous usage (new!)
client = AragoraClient("http://localhost:8080")
debate = client.debates.create(task="...")  # Sync

# Asynchronous usage
import asyncio

async def main():
    client = AragoraClient("http://localhost:8080")
    debate = await client.debates.create_async(task="...")  # Async

asyncio.run(main())
```

### Method Mapping

| `aragora-client` (deprecated) | `aragora-sdk` (current) |
|-------------------------------|-------------------------|
| `await client.debates.create()` | `client.debates.create()` or `await client.debates.create_async()` |
| `await client.debates.get(id)` | `client.debates.get(id)` or `await client.debates.get_async(id)` |
| `await client.debates.list()` | `client.debates.list()` or `await client.debates.list_async()` |
| `await client.agents.list()` | `client.agents.list()` or `await client.agents.list_async()` |
| `await client.health()` | `client.health()` or `await client.health_async()` |

### New Namespaces in aragora-sdk

The `aragora-sdk` includes 153 namespace modules (vs 26 in aragora-client):

```python
from aragora_sdk import AragoraClient

client = AragoraClient("http://localhost:8080")

# Workflows
client.workflows.list()
client.workflows.execute("template-id", params)

# Gauntlet (stress testing)
client.gauntlet.run(debate_id="...", attacks=["hollow"])
receipt = client.gauntlet.get_receipt("receipt-id")

# Explainability
client.explainability.get_factors("debate-id")
client.explainability.get_counterfactuals("debate-id")

# Knowledge Mound
client.knowledge.search(query="...")
client.knowledge.get_insights("debate-id")

# Control Plane
client.control_plane.register_agent(agent_config)
client.control_plane.get_health()
```

---

## TypeScript: `@aragora/client` → `@aragora/sdk`

The `@aragora/client` package is deprecated. Use `@aragora/sdk` for the unified TypeScript SDK.

### Installation

```bash
# Remove old package
npm uninstall @aragora/client

# Install new package
npm install @aragora/sdk
```

### Import Changes

```typescript
// BEFORE (deprecated)
import { AragoraClient } from '@aragora/client';
import type { Debate, Agent } from '@aragora/client';

// AFTER (recommended)
import { createClient } from '@aragora/sdk';
import type { Debate, Agent } from '@aragora/sdk';
```

### Client Initialization

```typescript
// BEFORE
import { AragoraClient } from '@aragora/client';
const client = new AragoraClient({
  baseUrl: 'https://api.aragora.ai',
  apiKey: 'your-api-key',
});

// AFTER
import { createClient } from '@aragora/sdk';
const client = createClient({
  baseUrl: 'https://api.aragora.ai',
  apiKey: 'your-api-key',
});
```

### API Changes

The SDK uses a namespace-based API. Most methods are similar:

```typescript
// BEFORE (@aragora/client)
await client.debates.create({ task: '...' });
await client.debates.list();
await client.debates.get('debate-id');
await client.agents.list();

// AFTER (@aragora/sdk) - Same API!
await client.debates.create({ task: '...' });
await client.debates.list();
await client.debates.get('debate-id');
await client.agents.list();
```

### Method Mapping

| `@aragora/client` | `@aragora/sdk` |
|------------------|----------------|
| `client.debates.create()` | `client.debates.create()` |
| `client.debates.run()` | `client.debates.create()` + poll |
| `client.debates.list()` | `client.debates.list()` |
| `client.debates.get(id)` | `client.debates.get(id)` |
| `client.agents.list()` | `client.agents.list()` |
| `client.controlPlane.registerAgent()` | `client.controlPlane.registerAgent()` |
| `client.verification.verifyClaim()` | `client.verification.verifyClaim()` |

### New Namespaces in SDK

The `@aragora/sdk` includes additional namespaces not in `@aragora/client`:

```typescript
// Workflows
await client.workflows.list();
await client.workflows.execute('template-id', params);

// Explainability
await client.explainability.getFactors('debate-id');
await client.explainability.getCounterfactuals('debate-id');

// Gauntlet (stress testing)
await client.gauntlet.run({ debate_id: '...', attacks: ['hollow'] });
await client.gauntlet.getReceipt('receipt-id');

// SME features
await client.sme.getDashboard();
await client.sme.getTimeToDecision();

// Billing & Budgets
await client.billing.getInvoices();
await client.budgets.setCap({ monthly_limit: 500 });
```

### WebSocket Streaming

```typescript
// BEFORE
import { AragoraClient } from '@aragora/client';
const client = new AragoraClient({ ... });
const stream = client.createDebateStream('debate-id');
stream.on('message', (event) => console.log(event));

// AFTER
import { createClient } from '@aragora/sdk';
const client = createClient({ ... });
const stream = client.debates.stream('debate-id');
stream.on('message', (event) => console.log(event));
```

### Type Changes

Some types have been renamed for consistency:

```typescript
// BEFORE
import type { DebateResult, AgentInfo } from '@aragora/client';

// AFTER
import type { Debate, Agent } from '@aragora/sdk';
```

| `@aragora/client` Type | `@aragora/sdk` Type |
|----------------------|-------------------|
| `DebateResult` | `Debate` |
| `AgentInfo` | `Agent` |
| `DebateConfig` | `DebateCreateRequest` |
| `ConsensusData` | `ConsensusResult` |

---

## v3.0.0 Preview (Q2 2026)

In v3.0.0, the API will be fully consolidated with these changes:

```typescript
// v3.0.0 syntax (coming Q2 2026)
import { AragoraClient } from '@aragora/sdk';

// Class-based instantiation
const client = new AragoraClient({
  baseUrl: 'https://api.aragora.ai',
  apiKey: 'your-api-key'
});

// Namespaced API (same as current)
await client.debates.create({ task: '...' });
```

### Breaking Changes in v3.0.0

1. `createClient()` → `new AragoraClient()`
2. All flat methods moved to namespaces
3. Some type names standardized

---

## Deprecation Timeline

### Python (`aragora-client` → `aragora-sdk`)

| Date | Event |
|------|-------|
| **v2.6.3** | Both packages coexist, `aragora-sdk` is canonical |
| **v2.7.0** | `aragora-client` becomes a thin wrapper around `aragora-sdk` |
| **v3.0.0 (Q2 2026)** | Single `aragora` package replaces both |

### TypeScript (`@aragora/client` → `@aragora/sdk`)

| Date | Event |
|------|-------|
| **January 2026** | Deprecation warnings added to `@aragora/client` |
| **Q1 2026** | `@aragora/sdk` reaches feature parity |
| **Q2 2026** | `@aragora/client` no longer published |
| **Q3 2026** | `@aragora/client` removed from npm |

---

## Getting Help

- **Documentation**: See [SDK_GUIDE.md](./SDK_GUIDE.md) for full API reference
- **TypeScript SDK**: See [SDK_TYPESCRIPT.md](./SDK_TYPESCRIPT.md)
- **Examples**: Check `sdk/cookbook/` (Python) and `sdk/typescript/examples/` (TypeScript)
- **Issues**: Report migration problems at https://github.com/an0mium/aragora/issues

---

## Checklist

### Python Migration

- [ ] Replace `aragora-client` with `aragora-sdk` in requirements.txt
- [ ] Update imports from `aragora_client` to `aragora_sdk`
- [ ] Convert async-only code to sync (or use `_async` suffixed methods)
- [ ] Test all API calls

### TypeScript Migration

- [ ] Replace `@aragora/client` with `@aragora/sdk` in package.json
- [ ] Update imports from `@aragora/client` to `@aragora/sdk`
- [ ] Replace `new AragoraClient()` with `createClient()`
- [ ] Update type imports if using renamed types
- [ ] Test all API calls
