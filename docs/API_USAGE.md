# API Usage Documentation

This document maps API endpoints to their frontend consumers and identifies optimization opportunities.

## Frontend-Used Endpoints

These endpoints are actively used by the Next.js frontend (`aragora/live/`):

### Core Debate API
| Endpoint | Method | Frontend Consumer | Notes |
|----------|--------|------------------|-------|
| `/api/debates` | POST | `/app/page.tsx` | Create new debate |
| `/api/debates` | GET | `/app/page.tsx` | List debates |
| `/api/debates/{id}` | GET | `/app/debate/[id]/page.tsx` | Debate details |
| `/ws` | WS | `useDebateWebSocket.ts` | Real-time streaming (filter by `loop_id`) |

### Agent API
| Endpoint | Method | Frontend Consumer | Notes |
|----------|--------|------------------|-------|
| `/api/agents` | GET | `/components/AgentSelector.tsx` | List agents |
| `/api/agent/{name}` | GET | `/app/agent/[name]/page.tsx` | Agent profile |
| `/api/agent/{name}/history` | GET | Agent page | Match history |
| `/api/agent/{name}/rivals` | GET | Agent page | Top rivals |
| `/api/agent/{name}/allies` | GET | Agent page | Top allies |

### Gauntlet API
| Endpoint | Method | Frontend Consumer | Notes |
|----------|--------|------------------|-------|
| `/api/gauntlet` | POST | `/app/gauntlet/page.tsx` | Start gauntlet |
| `/api/gauntlet/{id}` | GET | `/app/gauntlet/[id]/page.tsx` | Get status |
| `/api/gauntlet/{id}/receipt` | GET | Gauntlet page | Get receipt |
| `/ws` | WS | `useGauntletWebSocket.ts` | Gauntlet streaming (filter by `loop_id`) |

### Replays API
| Endpoint | Method | Frontend Consumer | Notes |
|----------|--------|------------------|-------|
| `/api/replays` | GET | `/app/replays/page.tsx` | List replays |
| `/api/replays/{id}` | GET | Replay detail | Full replay |

### Insights API
| Endpoint | Method | Frontend Consumer | Notes |
|----------|--------|------------------|-------|
| `/api/insights/patterns` | GET | `/app/insights/page.tsx` | Meta-patterns |
| `/api/insights/flip-detection` | GET | Insights page | Position flips |

### A/B Testing API
| Endpoint | Method | Frontend Consumer | Notes |
|----------|--------|------------------|-------|
| `/api/ab-tests` | GET | `/app/ab-testing/page.tsx` | List tests |
| `/api/ab-tests/{id}` | GET | A/B test detail | Test details |

### Billing/Organization API
| Endpoint | Method | Frontend Consumer | Notes |
|----------|--------|------------------|-------|
| `/api/billing/usage` | GET | `/app/billing/page.tsx` | Usage stats |
| `/api/organization` | GET | `/app/organization/page.tsx` | Org details |

## Unused API Endpoints

These endpoints have backend implementations but no frontend consumers:

### Graph Debates (Backend Ready, No UI)
- `POST /api/debates/graph` - Create graph debate
- `GET /api/debates/graph/{id}` - Get graph debate
- `GET /api/debates/graph/{id}/branches` - Get branches

### Matrix Debates (Backend Ready, No UI)
- `POST /api/debates/matrix` - Create matrix debate
- `GET /api/debates/matrix/{id}` - Get matrix debate
- `GET /api/debates/matrix/{id}/conclusions` - Get conclusions

### Tournaments (Backend Ready, No UI)
- `GET /api/tournaments` - List tournaments
- `POST /api/tournaments` - Create tournament
- `GET /api/tournaments/{id}` - Tournament details
- `POST /api/tournaments/{id}/start` - Start tournament
- `GET /api/tournaments/{id}/bracket` - Tournament bracket

### Verification (Backend Ready, Minimal UI)
- `POST /api/verify/claim` - Verify claim
- `GET /api/verify/status` - Backend status
- `POST /api/verify/batch` - Batch verification

### Memory Analytics (Backend Ready, No UI)
- `GET /api/memory/analytics` - Memory tier analytics
- `GET /api/memory/analytics/tier/{name}` - Tier-specific stats
- `POST /api/memory/analytics/snapshot` - Take snapshot

### Pulse/Trending (Backend Ready, No UI)
- `GET /api/pulse/topics` - Trending topics
- `GET /api/pulse/debates` - Scheduled debates
- `POST /api/pulse/trigger` - Trigger debate from topic

### Leaderboard (Partial UI)
- `GET /api/leaderboard` - Agent rankings
- `GET /api/leaderboard/calibration` - Calibration leaderboard

### Cross-Pollination (Observability)
- `GET /api/cross-pollination/stats` - Subscriber statistics and event counts
- `GET /api/cross-pollination/subscribers` - List all registered cross-subscribers
- `GET /api/cross-pollination/bridge` - Arena event bridge status and mappings
- `POST /api/cross-pollination/reset` - Reset statistics (for testing)

See [CROSS_POLLINATION.md](./CROSS_POLLINATION.md) for architecture details.

## Query Optimization Opportunities

The following files contain `SELECT *` queries that should be optimized to select only required columns:

### High Priority (Hot Paths)
1. `aragora/server/handlers/billing.py:524` - Usage events query
2. `aragora/server/handlers/replays.py:231` - Meta patterns query
3. `aragora/server/handlers/evolution_ab_testing.py:208,218` - A/B test queries

### Medium Priority (Repositories)
1. `aragora/persistence/repositories/base.py:400,478` - Generic repository methods
2. `aragora/persistence/repositories/elo.py:290,568,594,604` - ELO queries
3. `aragora/persistence/repositories/debate.py:254` - Debate lookup
4. `aragora/persistence/repositories/memory.py:339,443` - Memory queries

### Lower Priority (Domain Logic)
1. `aragora/storage/user_store.py` - Multiple user/org queries
2. `aragora/agents/positions.py:279,353` - Position tracking
3. `aragora/agents/relationships.py:163,253,283` - Agent relationships
4. `aragora/genesis/*.py` - Genesis breeding/genome queries

**Recommendation**: The repository base class (`base.py`) uses SELECT * generically. Consider adding a `_columns` property to subclasses that explicitly lists columns. This would be a Phase 2 architecture improvement.

## Endpoint Deprecation Candidates

The following endpoints have not been used in frontend development and may be candidates for deprecation or documentation:

1. `/api/plugins/*` - Plugin system (experimental)
2. `/api/probes/*` - Health probes (internal)
3. `/api/broadcast/*` - Audio broadcast generation
4. `/api/genesis/*` - Agent genesis/breeding (experimental)

## Recommended Actions

1. **Create Tournament UI** - Backend is complete, just needs frontend pages
2. **Create Pulse UI** - Trending topics dashboard
3. **Expose Graph/Matrix debates** - Advanced debate modes in UI
4. **Document internal endpoints** - Mark with @internal decorator
5. **Add API versioning** - Before public launch, add /v1/ prefix
