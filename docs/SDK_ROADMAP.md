# SDK Parity Roadmap

Roadmap for achieving 95%+ API coverage across TypeScript and Python SDKs.

## Current Status (Updated 2026-01-27)

| SDK | Namespaces/Methods | Coverage | Target |
|-----|-----------|----------|--------|
| TypeScript | 65 namespaces | ~70% | 95% |
| Python | 272 methods | ~75% | 95% |

**Recent Progress:**
- TypeScript: Added 12 new namespaces (Deliberations, Genesis, Laboratory, Teams, Learning, Batch, Privacy, Feedback, CodeReview, RLM, Backups, Dashboard)
- Python: Auto-generated client has 272 methods covering most core functionality

## Coverage Tiers

### Tier 1: Core (Priority)
Essential for basic SDK functionality.

| Category | Endpoints | TS Status | PY Status |
|----------|-----------|-----------|-----------|
| Auth | 26 | 20/26 | 8/26 |
| Debates | 25 | 22/25 | 12/25 |
| Codebase | 34 | 25/34 | 5/34 |
| **Subtotal** | 85 | 67 (79%) | 25 (29%) |

### Tier 2: Platform
Core platform features.

| Category | Endpoints | TS Status | PY Status |
|----------|-----------|-----------|-----------|
| Agents | 19 | 15/19 | 4/19 |
| Workflows | 14 | 10/14 | 2/14 |
| Webhooks | 14 | 12/14 | 2/14 |
| Budgets | 12 | 8/12 | 0/12 |
| Memory | 18 | 14/18 | 3/18 |
| Knowledge | 22 | 16/22 | 4/22 |
| **Subtotal** | 99 | 75 (76%) | 15 (15%) |

### Tier 3: Enterprise
Advanced features for enterprise deployments.

| Category | Endpoints | TS Status | PY Status |
|----------|-----------|-----------|-----------|
| Admin | 12 | 8/12 | 0/12 |
| Integrations | 9 | 6/9 | 0/9 |
| Plugins | 9 | 4/9 | 0/9 |
| Analytics | 16 | 10/16 | 0/16 |
| RBAC | 14 | 8/14 | 1/14 |
| Tenancy | 11 | 5/11 | 0/11 |
| **Subtotal** | 71 | 41 (58%) | 1 (1%) |

### Tier 4: Specialized
Domain-specific capabilities.

| Category | Endpoints | TS Status | PY Status |
|----------|-----------|-----------|-----------|
| Pulse | 18 | 10/18 | 0/18 |
| Gauntlet | 15 | 8/15 | 0/15 |
| Explainability | 12 | 6/12 | 0/12 |
| RLM | 8 | 4/8 | 0/8 |
| Voice | 10 | 5/10 | 0/10 |
| Control Plane | 20 | 12/20 | 0/20 |
| **Subtotal** | 83 | 45 (54%) | 0 (0%) |

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Focus:** TypeScript to 70%, Python to 30%

- [ ] TypeScript: Complete Tier 1 (Auth, Debates, Codebase)
- [ ] Python: Complete Auth and Debates
- [ ] Add code generation tooling for consistency
- [ ] Set up SDK test infrastructure

**Deliverables:**
- TypeScript: +18 endpoints (211 total, 59%)
- Python: +35 endpoints (76 total, 21%)

### Phase 2: Platform Coverage (Weeks 3-5)
**Focus:** TypeScript to 85%, Python to 50%

- [ ] TypeScript: Complete Tier 2
- [ ] Python: Complete Tier 1 + start Tier 2
- [ ] Add async/streaming support to Python SDK
- [ ] WebSocket client implementation

**Deliverables:**
- TypeScript: +37 endpoints (248 total, 69%)
- Python: +82 endpoints (158 total, 44%)

### Phase 3: Enterprise Features (Weeks 6-8)
**Focus:** TypeScript to 92%, Python to 75%

- [ ] TypeScript: Complete Tier 3
- [ ] Python: Complete Tier 2 + start Tier 3
- [ ] Add retry/circuit breaker to SDKs
- [ ] Error handling standardization

**Deliverables:**
- TypeScript: +37 endpoints (285 total, 80%)
- Python: +77 endpoints (235 total, 66%)

### Phase 4: Parity Push (Weeks 9-12)
**Focus:** Both SDKs to 95%+

- [ ] TypeScript: Complete Tier 4
- [ ] Python: Complete Tiers 3 + 4
- [ ] Cross-SDK test suite
- [ ] Documentation sync

**Deliverables:**
- TypeScript: +55 endpoints (340 total, 95%)
- Python: +105 endpoints (340 total, 95%)

## SDK Architecture

### TypeScript SDK (`sdk/typescript/`)

```
sdk/typescript/
├── src/
│   ├── client.ts           # Main client
│   ├── auth/               # Auth endpoints
│   ├── debates/            # Debate operations
│   ├── agents/             # Agent management
│   ├── knowledge/          # KM operations
│   ├── workflows/          # Workflow engine
│   └── types/              # Generated types
├── examples/
└── tests/
```

### Python SDK (`sdk/python/`)

```
sdk/python/
├── aragora/
│   ├── client.py           # Main client
│   ├── auth/               # Auth endpoints
│   ├── debates/            # Debate operations
│   ├── agents/             # Agent management
│   ├── knowledge/          # KM operations
│   ├── workflows/          # Workflow engine
│   └── models/             # Pydantic models
├── examples/
└── tests/
```

## Code Generation Strategy

To accelerate development and ensure consistency:

1. **OpenAPI Spec** - Maintain `api/openapi.yaml` as source of truth
2. **Type Generation** - Auto-generate TypeScript types and Pydantic models
3. **Client Generation** - Use templates for endpoint wrappers
4. **Test Generation** - Generate test stubs from spec

### Generator Pipeline

```bash
# Generate from OpenAPI spec
python scripts/generate_sdk.py --spec api/openapi.yaml --output sdk/

# Outputs:
# - sdk/typescript/src/types/generated.ts
# - sdk/python/aragora/models/generated.py
# - sdk/*/tests/test_*.py (stubs)
```

## Quality Standards

### Required for Each Endpoint

- [ ] Type definitions (TS) / Pydantic models (Python)
- [ ] Input validation
- [ ] Error handling with typed exceptions
- [ ] JSDoc/docstring documentation
- [ ] Unit test with mocked HTTP
- [ ] Integration test example

### SDK Features

| Feature | TypeScript | Python |
|---------|------------|--------|
| Async support | Native | asyncio |
| Streaming | ReadableStream | AsyncGenerator |
| Retry logic | Configurable | tenacity |
| Rate limiting | Built-in | Built-in |
| Auth refresh | Automatic | Automatic |
| Pagination | Cursor-based | Cursor-based |

## Tracking Progress

Update this document as endpoints are implemented:

### Weekly Status Update

**Week of 2026-01-27**
- TypeScript: 53 namespaces implemented (was 47)
- New namespaces added:
  - `DeliberationsAPI` - Vetted decisionmaking visibility
  - `GenesisAPI` - Evolution and genome lineage
  - `LaboratoryAPI` - Emergent traits and cross-pollination
  - `TeamsAPI` - Microsoft Teams bot integration
  - `LearningAPI` - Meta-learning analytics
  - `BatchAPI` - Batch debate operations
- Total new endpoints covered: ~35
- Blockers: None

```markdown
### Template for Future Updates

**Week of YYYY-MM-DD**
- TypeScript: X/358 (Y.Y%)
- Python: X/358 (Y.Y%)
- New endpoints: [list]
- Blockers: [any]
```

## Resources

- **API Reference:** `docs/API_REFERENCE.md`
- **SDK Guide:** `docs/SDK_GUIDE.md`
- **OpenAPI Spec:** `api/openapi.yaml`
- **TypeScript SDK:** `sdk/typescript/`
- **Python SDK:** `sdk/python/`

## Issue Tracking

This roadmap tracks GitHub issue #102 (SDK Parity).

Updates will be posted to the issue as milestones are reached.
