# Core API Surface Analysis

Last updated: 2026-04-03

## Summary

The Aragora server exposes 3,000+ API operations across 208 handler files. This document identifies the maintained core endpoints that serve most users and categorizes the rest for potential extraction.

Verified against the current codebase on `origin/main`:
- Native FastAPI routes now live primarily under `/api/v2/*`.
- A smaller legacy compatibility surface still exists for routes such as `/api/debates` and `/api/health`.
- The tables below prefer the maintained route first and call out legacy aliases only when they are still commonly used by clients or runbooks.

## Core Endpoints (~40 maintained + compatibility aliases)

These are the endpoints every Aragora user needs.

### Debates (10 maintained endpoints)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/v2/debates` | debates | Create a new debate (`/api/debates` compatibility alias also exists) |
| GET | `/api/v2/debates` | debates | List debates |
| GET | `/api/v2/debates/{debate_id}` | debates | Get debate details |
| PATCH | `/api/v2/debates/{debate_id}` | debates | Update debate metadata |
| DELETE | `/api/v2/debates/{debate_id}` | debates | Delete a debate |
| GET | `/api/v2/debates/{debate_id}/messages` | debates | Get debate messages |
| GET | `/api/v2/debates/{debate_id}/convergence` | debates | Get convergence status |
| GET | `/api/v2/debates/{debate_id}/export/{format}` | debates | Export a debate |
| GET | `/api/v2/debates/{debate_id}/argument-graph` | debates | Get argument graph |
| GET | `/api/v2/debates/{debate_id}/stats` | debates | Get debate graph statistics |
| POST | `/api/v1/playground/debate` | playground | Interactive playground |
| GET | `/api/v1/playground/status` | playground | Playground status |

### Receipts (4 endpoints)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/v2/receipts` | receipts | List decision receipts |
| GET | `/api/v2/receipts/:id` | receipts | Get receipt by ID |
| POST | `/api/v2/receipts/:id/verify` | receipts | Verify receipt HMAC |
| POST | `/api/v2/receipts/:id/share` | receipts | Generate share link |

### Agents (8 maintained endpoints)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/api/v2/agents` | agents | List available agents |
| POST | `/api/v2/agents` | agents | Register a new agent |
| GET | `/api/v2/agents/rankings` | agents | ELO rankings |
| GET | `/api/v2/agents/leaderboard` | agents | Agent leaderboard |
| GET | `/api/v2/agents/domains` | agents | List agent domains |
| GET | `/api/v2/agents/{agent_id}` | agents | Get agent details |
| GET | `/api/v2/agents/{agent_id}/capabilities` | agents | Agent capabilities and metadata |
| GET | `/api/v2/agents/{agent_id}/stats` | agents | Agent performance stats |

### Authentication (4 maintained endpoints)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/v2/auth/login` | auth | Login |
| POST | `/api/v2/auth/logout` | auth | Logout |
| GET | `/api/v2/auth/me` | auth | Current user |
| POST | `/api/v2/auth/refresh` | auth | Refresh token |

### Health (5 maintained endpoints)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| GET | `/healthz` | health | Basic health check |
| GET | `/livez` | health | Liveness probe |
| GET | `/readyz` | health | Readiness probe |
| GET | `/api/v2/health` | health | Detailed health status |
| GET | `/api/v2/metrics/summary` | health | Basic metrics summary |

### Gauntlet & Review (8 core endpoints)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/v2/gauntlet/run` | gauntlet | Run adversarial gauntlet |
| GET | `/api/v2/gauntlet/{run_id}/status` | gauntlet | Get gauntlet run status |
| GET | `/api/v2/gauntlet/{run_id}/findings` | gauntlet | Get gauntlet findings |
| POST | `/api/v1/review` | review | Code review |
| GET | `/api/v1/review/:id` | review | Get review result |
| POST | `/api/v1/skills` | skills | Register skill |
| GET | `/api/v1/skills` | skills | List skills |
| GET | `/api/v1/skills/marketplace` | skills | Marketplace browse |

### Webhooks & Events (4 endpoints)

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| POST | `/api/v1/webhooks` | webhooks | Register webhook |
| GET | `/api/v1/webhooks` | webhooks | List webhooks |
| GET | `/api/v1/webhooks/events` | webhooks | Event types |
| GET | `/api/v1/webhooks/dead-letter` | webhooks | Dead letter queue |

**Total core: ~40 maintained endpoints, plus legacy compatibility aliases**

## Extended Endpoints (by category)

| Category | Est. Endpoints | Handler Files | LOC | Priority |
|----------|---------------|---------------|-----|----------|
| Analytics & Dashboard | ~30 | 15 | 8K | Keep |
| Learning & Insights | ~15 | 8 | 5K | Keep |
| Knowledge Mound | ~25 | 12 | 10K | Keep |
| Memory & Continuum | ~20 | 10 | 7K | Keep |
| Pulse (trending) | ~15 | 6 | 4K | Keep |
| Billing & Metering | ~20 | 10 | 6K | Optional |
| Email Services | ~15 | 5 | 5K | Optional |
| Compliance | ~20 | 8 | 6K | Enterprise |
| Notifications | ~10 | 4 | 3K | Keep |
| Integrations (Slack, etc) | ~25 | 15 | 8K | Optional |

## Enterprise Endpoints

| Category | Est. Endpoints | Handler Files | LOC | Notes |
|----------|---------------|---------------|-----|-------|
| RBAC Administration | ~25 | 12 | 8K | Gate behind license |
| Multi-Tenancy | ~15 | 6 | 5K | Gate behind license |
| Control Plane | ~30 | 15 | 10K | Gate behind license |
| Audit & Compliance | ~25 | 10 | 8K | Gate behind license |
| SSO/OIDC/SAML | ~15 | 8 | 6K | Gate behind license |
| Backup/DR | ~10 | 4 | 3K | Gate behind license |

## Experimental Endpoints (Candidates for extraction)

| Category | Est. Endpoints | Handler Files | LOC | Action |
|----------|---------------|---------------|-----|--------|
| Genesis (evolution) | ~10 | 3 | 3K | Extract to aragora-experimental |
| Blockchain/ERC-8004 | ~15 | 5 | 4K | Extract to aragora-experimental |
| Nomic Loop | ~10 | 4 | 3K | Extract to aragora-experimental |
| Workflow Engine | ~20 | 8 | 6K | Extract to aragora-experimental |
| Computer Use | ~10 | 4 | 4K | Extract to aragora-experimental |
| Feature verticals | ~60 | 30 | 20K | Gate behind feature flags |

## Recommendations

### 1. Handler loading tiers

```python
# In server startup, load handlers in tiers:
CORE_HANDLERS = [...]      # Always loaded (~48 endpoints)
EXTENDED_HANDLERS = [...]   # Loaded by default, can disable
ENTERPRISE_HANDLERS = [...]  # Loaded only with license key
EXPERIMENTAL_HANDLERS = [...] # Loaded only with --experimental flag
```

### 2. Estimated impact

| Action | Handlers Removed | Endpoints Reduced | Startup Time |
|--------|-----------------|-------------------|-------------|
| Remove experimental | ~30 files | ~65 endpoints | -15% |
| Gate enterprise | ~55 files | ~120 endpoints | -25% |
| Gate feature verticals | ~30 files | ~60 endpoints | -15% |
| **Total (core only)** | **~115 fewer** | **~245 fewer** | **~55% faster** |

### 3. Path forward

1. **Immediate**: Tag handlers with tier metadata (`TIER = "core"`)
2. **Short-term**: Lazy-load non-core handlers on first request
3. **Medium-term**: Extract experimental handlers to `aragora-experimental` package
4. **Long-term**: Enterprise handlers behind license check, separate `aragora-enterprise` package
