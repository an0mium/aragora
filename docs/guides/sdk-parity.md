# SDK Parity Guide: Python vs TypeScript

This guide documents the feature parity between the Python and TypeScript SDKs for Aragora.

## Overview

| Metric | Python SDK | TypeScript SDK |
|--------|------------|----------------|
| Package | `aragora` | `@aragora/sdk` |
| Overall Parity | 100% | **100%** |
| Type Definitions | N/A (dynamic) | 3,169+ types |
| Namespace Modules | 56+ | 85 |
| WebSocket Events | 18 | 18 |

> **Note**: As of January 2026, TypeScript SDK has achieved full parity with Python SDK for all documented APIs. TypeScript SDK actually has MORE namespaces (85 vs 56) due to additional enterprise features.

## Feature Comparison

### Core Features (Full Parity)

| Feature | Python | TypeScript | Notes |
|---------|--------|------------|-------|
| Create debate | `client.create_debate()` | `client.debates.create()` | Full parity |
| Get debate | `client.get_debate()` | `client.debates.get()` | Full parity |
| List debates | `client.list_debates()` | `client.debates.list()` | Full parity |
| Run debate (blocking) | Manual polling | `client.debates.run()` | TS has convenience method |
| Stream debate | `async for event in stream` | `for await (event of stream)` | Full parity |
| Wait for completion | Manual | `client.debates.waitForCompletion()` | TS has convenience method |

### Agent Analytics (Full Parity)

| Feature | Python | TypeScript |
|---------|--------|------------|
| Get agent | `client.get_agent()` | `client.agents.get()` |
| List agents | `client.list_agents()` | `client.agents.list()` |
| Get rankings | `client.get_rankings()` | `client.agents.getRankings()` |
| Head-to-head | `client.get_head_to_head()` | `client.agents.getHeadToHead()` |
| Domain ratings | `client.get_domain_ratings()` | `client.agents.getDomainRatings()` |
| Flip history | `client.get_flip_history()` | `client.agents.getFlipHistory()` |
| Network analysis | `client.get_network()` | `client.agents.getNetwork()` |

### Authentication (Full Parity)

| Feature | Python | TypeScript |
|---------|--------|------------|
| Login | `client.login()` | `client.auth.login()` |
| Register | `client.register()` | `client.auth.register()` |
| OAuth | `client.oauth_callback()` | `client.auth.oauthCallback()` |
| MFA setup | `client.setup_mfa()` | `client.auth.setupMfa()` |
| Token refresh | `client.refresh_token()` | `client.auth.refreshToken()` |

### Explainability (Full Parity)

| Feature | Python | TypeScript |
|---------|--------|------------|
| Get explanation | `client.get_explanation()` | `client.explainability.get()` |
| Get factors | `client.get_factors()` | `client.explainability.getFactors()` |
| Get counterfactuals | `client.get_counterfactuals()` | `client.explainability.getCounterfactuals()` |
| Get provenance | `client.get_provenance()` | `client.explainability.getProvenance()` |
| Get narrative | `client.get_narrative()` | `client.explainability.getNarrative()` |

### Workflows (Full Parity)

| Feature | Python | TypeScript |
|---------|--------|------------|
| List templates | `client.list_templates()` | `client.workflows.listTemplates()` |
| Get template | `client.get_template()` | `client.workflows.getTemplate()` |
| Run template | `client.run_template()` | `client.workflows.runTemplate()` |
| List categories | `client.list_categories()` | `client.workflows.listCategories()` |
| Get patterns | `client.get_patterns()` | `client.workflows.getPatterns()` |

### Memory & Analytics (Full Parity)

| Feature | Python | TypeScript |
|---------|--------|------------|
| Store memory | `client.store_memory()` | `client.memory.store()` |
| Retrieve memory | `client.retrieve_memory()` | `client.memory.retrieve()` |
| Search memory | `client.search_memory()` | `client.memory.search()` |
| Consensus stats | `client.get_consensus_stats()` | `client.analytics.getConsensusStats()` |

### RBAC (Full Parity)

| Feature | Python | TypeScript |
|---------|--------|------------|
| List permissions | `client.list_permissions()` | `client.rbac.listPermissions()` |
| List roles | `client.list_roles()` | `client.rbac.listRoles()` |
| Assign role | `client.assign_role()` | `client.rbac.assignRole()` |
| Check permission | `client.check_permission()` | `client.rbac.checkPermission()` |

## Previously Identified Gaps (Now Resolved)

All previously documented gaps have been resolved. The following features now have full parity:

### Marketplace (Full Parity)

| Feature | Python | TypeScript | Status |
|---------|--------|------------|--------|
| Browse | Yes | Yes | Parity |
| Get template | Yes | Yes | Parity |
| Import | Yes | Yes | Parity |
| Featured | Yes | Yes | Parity |
| Publish | Yes | Yes | Parity |
| Rate | Yes | Yes | Parity |
| Review | Yes | Yes | Parity |
| Update | Yes | Yes | Parity |
| Unpublish | Yes | Yes | Parity |

### Batch Operations (Full Parity)

| Feature | Python | TypeScript | Status |
|---------|--------|------------|--------|
| Batch submit | Yes | Yes | Parity |
| Batch status | Yes | Yes | Parity |
| Batch results | Yes | Yes | Parity |
| Batch explanations | Yes | Yes | Parity |

### Codebase Analysis (Full Parity)

| Feature | Python | TypeScript | Status |
|---------|--------|------------|--------|
| Analyze codebase | Yes | Yes | Parity |
| Security scan | Yes | Yes | Parity |
| SBOM generation | Yes | Yes | Parity |
| Vulnerability scan | Yes | Yes | Parity |
| Dependency analysis | Yes | Yes | Parity |
| Secret detection | Yes | Yes | Parity |
| SAST analysis | Yes | Yes | Parity |
| Code metrics | Yes | Yes | Parity |
| Dead code detection | Yes | Yes | Parity |
| Call graph analysis | Yes | Yes | Parity |
| Impact analysis | Yes | Yes | Parity |

### Gauntlet (Full Parity)

| Feature | Python | TypeScript | Status |
|---------|--------|------------|--------|
| Run gauntlet | Yes | Yes | Parity |
| Get status | Yes | Yes | Parity |
| List receipts | Yes | Yes | Parity |
| Get receipt | Yes | Yes | Parity |
| Compare runs | Yes | Yes | Parity |
| Heatmap visualization | Yes | Yes | Parity |
| Export reports | Yes | Yes | Parity |

## WebSocket Events (Full Parity)

Both SDKs support all 18 WebSocket event types:

| Event | Description |
|-------|-------------|
| `debate_start` | Debate has started |
| `round_start` | New round begins |
| `round_end` | Round completed |
| `agent_message` | Agent sends message |
| `propose` | Agent makes proposal |
| `critique` | Critique phase |
| `revision` | Revision phase |
| `synthesis` | Synthesis phase |
| `vote` | Voting occurs |
| `consensus` | Consensus calculation |
| `consensus_reached` | Final consensus |
| `debate_end` | Debate completed |
| `phase_change` | Phase transition |
| `audience_suggestion` | User suggestion |
| `user_vote` | User vote cast |
| `error` | Error occurred |
| `warning` | Warning issued |
| `heartbeat` | Connection alive |

## Choosing Between SDKs

### Use Python SDK When

- Building backend services
- Need codebase analysis features
- Need batch operations
- Integrating with data science workflows
- Using Jupyter notebooks

### Use TypeScript SDK When

- Building web applications
- Need strong type safety
- Building Node.js services
- Building browser extensions
- Need real-time streaming in frontend

## Migration Notes

### Python to TypeScript

```python
# Python
from aragora import AragoraClient

client = AragoraClient(api_key="...")
debate = await client.create_debate("Should we use AI?", ["claude", "gpt-4"])
```

```typescript
// TypeScript
import { createClient } from '@aragora/sdk';

const client = createClient({ apiKey: '...' });
const debate = await client.debates.create({
  task: 'Should we use AI?',
  agents: ['claude', 'gpt-4']
});
```

### Key Differences

1. **Namespace organization**: TypeScript uses `client.namespace.method()` pattern
2. **Request objects**: TypeScript uses typed request objects vs Python kwargs
3. **Streaming**: TypeScript uses `for await...of`, Python uses `async for`
4. **Blocking calls**: TypeScript has `run()` convenience method

## Roadmap

All previously planned parity features have been implemented:

- [x] **Q1 2026**: Marketplace publish/rate/review methods ✅
- [x] **Q1 2026**: Batch explainability methods ✅
- [x] **Q1 2026**: Gauntlet comparison methods ✅
- [x] **Q1 2026**: Codebase analysis namespace (full) ✅

### Future Enhancements

1. **v3.0.0 SDK Consolidation**: Merge `@aragora/sdk` and `@aragora/client` into a single unified package
2. **Performance**: Optimize bundle size with tree-shaking improvements
3. **Developer Experience**: Enhanced TypeScript IntelliSense with JSDoc examples

## Contributing

To contribute SDK improvements:

1. Check existing types in `sdk/typescript/src/types.ts`
2. Add namespace methods in `sdk/typescript/src/namespaces/`
3. Export in `sdk/typescript/src/index.ts`
4. Add tests in `sdk/typescript/src/__tests__/`
5. Update this documentation

See [Contributing Guide](../contributing/first-contribution.md) for details.
