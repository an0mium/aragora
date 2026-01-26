# SDK Endpoint Audit Report

**Date:** January 25, 2026
**Audited By:** Automated Analysis
**Purpose:** Map API endpoints to SDK coverage for expansion roadmap

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total API Endpoints** | 1,038 |
| **TypeScript SDK Methods** | 323 (31% coverage) |
| **Python SDK Methods** | 213 (21% coverage) |
| **Target Coverage** | 80% TypeScript, 60% Python |

---

## Coverage by Category

### Tier 1: Critical (Must Have)

| Category | Endpoints | TS Coverage | Python Coverage | Gap |
|----------|-----------|-------------|-----------------|-----|
| Debates | 130 | 15 (12%) | 3 (2%) | HIGH |
| Knowledge Management | 100+ | 17 (17%) | 0 (0%) | CRITICAL |
| Agents/Leaderboard | 34 | 16 (47%) | 0 (0%) | HIGH |
| Workflows | 8 | 19 (100%+) | 0 (0%) | HIGH |
| Auth/RBAC | 54 | 29 (54%) | ~5 (9%) | MEDIUM |

### Tier 2: Important (Should Have)

| Category | Endpoints | TS Coverage | Python Coverage | Gap |
|----------|-----------|-------------|-----------------|-----|
| Analytics | 71 | 6 (8%) | 0 (0%) | HIGH |
| Billing/Budgets | 18 | 22 (100%+) | 0 (0%) | HIGH |
| Control Plane | 11 | 15 (100%+) | 0 (0%) | HIGH |
| Explainability | 11 | 9 (82%) | 7 (64%) | LOW |
| Memory | 15 | 11 (73%) | 0 (0%) | MEDIUM |

### Tier 3: Enterprise (Nice to Have)

| Category | Endpoints | TS Coverage | Python Coverage | Gap |
|----------|-----------|-------------|-----------------|-----|
| Social Connectors | 84 | ~10 (12%) | 0 (0%) | MEDIUM |
| Gauntlet | 28 | 10 (36%) | 0 (0%) | MEDIUM |
| DevOps/Support | 46 | 27 (59%) | 0 (0%) | LOW |
| Documents | 21 | ~5 (24%) | 0 (0%) | MEDIUM |

---

## Priority Endpoint Groups for SDK Expansion

### Phase 1 (Weeks 1-2): Knowledge & Core

**Knowledge Management (100 new methods)**
```
GET  /api/v1/knowledge/mound/nodes
POST /api/v1/knowledge/mound/nodes
GET  /api/v1/knowledge/mound/nodes/{id}
PUT  /api/v1/knowledge/mound/nodes/{id}
DELETE /api/v1/knowledge/mound/nodes/{id}
GET  /api/v1/knowledge/mound/relationships
POST /api/v1/knowledge/mound/relationships
GET  /api/v1/knowledge/mound/search
POST /api/v1/knowledge/mound/sync
GET  /api/v1/knowledge/mound/federation
POST /api/v1/knowledge/mound/federation/share
GET  /api/v1/knowledge/mound/analytics
GET  /api/v1/knowledge/mound/governance
POST /api/v1/knowledge/mound/governance/policies
GET  /api/v1/knowledge/graph
POST /api/v1/knowledge/chat
```

**Debates Extended (50 new methods)**
```
POST /api/v1/debates/{id}/fork
GET  /api/v1/debates/{id}/convergence
GET  /api/v1/debates/{id}/evidence
POST /api/v1/debates/{id}/evidence
GET  /api/v1/debates/{id}/provenance
GET  /api/v1/debates/{id}/timeline
GET  /api/v1/debates/{id}/messages
POST /api/v1/debates/{id}/messages
GET  /api/v1/debates/{id}/consensus
GET  /api/v1/debates/{id}/receipts
POST /api/v1/debates/{id}/export
```

### Phase 2 (Weeks 3-4): RBAC & Analytics

**RBAC/Auth (25 new methods)**
```
GET  /api/v1/rbac/roles
POST /api/v1/rbac/roles
GET  /api/v1/rbac/permissions
POST /api/v1/rbac/assignments
DELETE /api/v1/rbac/assignments/{id}
GET  /api/v1/rbac/audit
POST /api/v1/auth/mfa/setup
POST /api/v1/auth/mfa/verify
GET  /api/v1/auth/sso/providers
POST /api/v1/auth/sso/callback
```

**Analytics (40 new methods)**
```
GET  /api/v1/analytics/dashboard
GET  /api/v1/analytics/metrics
GET  /api/v1/analytics/trends
GET  /api/v1/analytics/costs
GET  /api/v1/analytics/tokens
GET  /api/v1/analytics/agents
GET  /api/v1/analytics/debates
GET  /api/v1/analytics/compliance
POST /api/v1/analytics/reports
GET  /api/v1/analytics/cross-platform
```

### Phase 3 (Weeks 5-6): Agents & Connectors

**Agent Management (30 new methods)**
```
GET  /api/v1/agents
GET  /api/v1/agents/{id}
PUT  /api/v1/agents/{id}
GET  /api/v1/agents/{id}/stats
GET  /api/v1/agents/{id}/calibration
POST /api/v1/agents/{id}/calibrate
GET  /api/v1/leaderboard
GET  /api/v1/leaderboard/head-to-head
GET  /api/v1/leaderboard/flips
GET  /api/v1/leaderboard/moments
```

**Social Connectors (40 new methods)**
```
POST /api/v1/connectors/slack/send
POST /api/v1/connectors/slack/channels
POST /api/v1/connectors/teams/send
POST /api/v1/connectors/discord/send
POST /api/v1/connectors/telegram/send
POST /api/v1/connectors/whatsapp/send
POST /api/v1/connectors/email/send
GET  /api/v1/connectors/{type}/status
POST /api/v1/connectors/{type}/webhook
```

### Phase 4 (Weeks 7-8): Enterprise & Polish

**Control Plane (15 new methods)**
```
GET  /api/v1/control-plane/agents
POST /api/v1/control-plane/agents/register
GET  /api/v1/control-plane/health
GET  /api/v1/control-plane/queue
POST /api/v1/control-plane/tasks
GET  /api/v1/control-plane/policies
POST /api/v1/control-plane/policies
```

**Gauntlet (20 new methods)**
```
POST /api/v1/gauntlet/run
GET  /api/v1/gauntlet/results/{id}
POST /api/v1/gauntlet/defend
GET  /api/v1/gauntlet/receipts
POST /api/v1/gauntlet/comparative
GET  /api/v1/gauntlet/findings
POST /api/v1/verification/formal
GET  /api/v1/verification/proofs
```

---

## TypeScript SDK Structure

**Current Namespaces (26):**
```typescript
// sdk/typescript/src/namespaces/
agents.ts        // 16 methods
analytics.ts     // 6 methods
audit.ts         // 13 methods
auth.ts          // 29 methods
billing.ts       // 22 methods
codebase.ts      // 27 methods
control-plane.ts // 15 methods
debates.ts       // 15 methods
explainability.ts // 9 methods
gauntlet.ts      // 10 methods
knowledge.ts     // 17 methods
memory.ts        // 11 methods
onboarding.ts    // 15 methods
tournaments.ts   // 11 methods
workflows.ts     // 19 methods
// ... plus 11 more
```

**New Namespaces Needed:**
```typescript
// To add:
connectors.ts    // Social/enterprise connectors (40+ methods)
analytics-extended.ts  // Full analytics suite (40+ methods)
rbac.ts          // Fine-grained permissions (25+ methods)
knowledge-mound.ts    // Full KM API (50+ methods)
```

---

## Python SDK Structure

**Current Structure (Monolithic):**
```python
# sdk/python/aragora/client.py
class AragoraClient:
    # 213 methods on single class
    def list_debates(self): ...
    def get_debate(self, id): ...
    def create_debate(self, env, protocol): ...
    def get_explanation(self, debate_id): ...
    # etc.
```

**Target Structure (Namespaced):**
```python
# sdk/python/aragora/
client.py          # Main client with namespace accessors
namespaces/
    debates.py     # DebatesNamespace
    knowledge.py   # KnowledgeNamespace
    agents.py      # AgentsNamespace
    workflows.py   # WorkflowsNamespace
    analytics.py   # AnalyticsNamespace
    auth.py        # AuthNamespace
    rbac.py        # RBACNamespace
    connectors.py  # ConnectorsNamespace
    gauntlet.py    # GauntletNamespace
```

---

## SDK Method Generation Strategy

### 1. Extract OpenAPI Spec
```bash
# Generate OpenAPI from handlers
python scripts/generate_openapi.py > openapi.yaml
```

### 2. Generate TypeScript Methods
```bash
# Auto-generate from OpenAPI
npx openapi-typescript openapi.yaml -o sdk/typescript/src/generated/
```

### 3. Generate Python Methods
```bash
# Auto-generate from OpenAPI
openapi-generator generate -i openapi.yaml -g python -o sdk/python/aragora/generated/
```

### 4. Manual Integration
- Add type definitions
- Implement streaming support
- Add authentication handling
- Write tests

---

## Milestone Targets

| Week | TypeScript Coverage | Python Coverage | New Methods |
|------|---------------------|-----------------|-------------|
| 2 | 45% | 15% | +150 |
| 4 | 60% | 30% | +200 |
| 6 | 70% | 45% | +150 |
| 8 | 80% | 60% | +100 |

**Total New Methods:** ~600 (400 TypeScript, 200 Python)

---

## Handler Reference

**Source Files:**
- `aragora/server/handlers/` - 138 handler modules
- `aragora/server/unified_server.py` - Route registration
- `aragora/server/routes/` - Route definitions

**Key Handlers by Endpoint Count:**
1. `handlers/debates/handler.py` - 60+ endpoints
2. `handlers/knowledge_base/mound/handler.py` - 50+ endpoints
3. `handlers/social/slack.py` - 30+ endpoints
4. `handlers/analytics_dashboard.py` - 25+ endpoints
5. `handlers/agents/agents.py` - 20+ endpoints

---

## Next Steps

1. **Create SDK generator script** - `scripts/generate_sdk.py`
2. **Extract OpenAPI spec** from handlers
3. **Generate Knowledge namespace** (TypeScript + Python)
4. **Generate Analytics namespace** (TypeScript + Python)
5. **Add tests for new methods**
6. **Update SDK documentation**

---

*This audit will be updated as SDK expansion progresses.*
