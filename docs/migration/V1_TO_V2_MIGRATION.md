# API v1 to v2 Migration Guide

> **Sunset date: June 1, 2026.**  After this date, v1 endpoints will return
> `410 Gone` responses.  All v1 responses already include RFC 8594 `Deprecation`
> and `Sunset` headers.

## Timeline

| Milestone | Date | What happens |
|-----------|------|--------------|
| Deprecation announced | 2025-01-15 | `Deprecation` header added to all v1 responses |
| Warning phase (current) | Now | v1 works normally, headers warn of upcoming removal |
| Critical phase | 2026-05-02 | `X-Deprecation-Level: critical` (30 days left) |
| Sunset | **2026-06-01** | v1 endpoints return `410 Gone` |
| Removal | 2026-07-01 | v1 route code removed from codebase |

## Quick Start

1. Check your codebase for `/api/v1/` references.
2. Replace them with the corresponding `/api/v2/` path from the table below.
3. Update any response-parsing code for breaking changes listed in the
   "Breaking Changes" section.
4. Verify against a staging environment.

## HTTP Headers You Will See

Every v1 response now includes these headers:

```
Sunset: Mon, 01 Jun 2026 00:00:00 GMT
Deprecation: @1748736000
Link: <https://docs.aragora.ai/migration/v1-to-v2>; rel="sunset",
      </api/v2/debates>; rel="successor-version"
X-API-Version: v1
X-API-Version-Warning: API v1 is deprecated and will be removed on 2026-06-01. ...
X-API-Sunset: 2026-06-01
X-Deprecation-Level: warning
```

**Tip:** Search your logs for `X-Deprecation-Level` or grep HTTP responses for
the `Sunset` header to find v1 usage in production.

## Configuration

| Env var | Default | Effect |
|---------|---------|--------|
| `ARAGORA_DISABLE_V1_DEPRECATION` | `false` | Set `true` to suppress deprecation headers |
| `ARAGORA_BLOCK_SUNSET_ENDPOINTS` | `true` | Set `false` to keep v1 alive past sunset (escape hatch) |
| `ARAGORA_LOG_DEPRECATED_USAGE` | `true` | Set `false` to disable v1 usage logging |

## Endpoint Mapping Table

### Core Debate & Agents

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/debates` | `/api/v2/debates` | GET, POST | Response pagination format changed |
| `/api/v1/debates/{id}` | `/api/v2/debates/{id}` | GET, PUT, DELETE | |
| `/api/v1/debate` | `/api/v2/debates` | POST | Singular form removed in v2 |
| `/api/v1/debate/{id}` | `/api/v2/debates/{id}` | GET | Singular form removed in v2 |
| `/api/v1/agents` | `/api/v2/agents` | GET | |
| `/api/v1/agents/{name}` | `/api/v2/agents/{name}` | GET | |
| `/api/v1/consensus/{id}` | `/api/v2/debates/{id}/consensus` | GET | Nested under debates in v2 |
| `/api/v1/leaderboard` | `/api/v2/agents/leaderboard` | GET | Nested under agents in v2 |
| `/api/v1/rankings` | `/api/v2/agents/rankings` | GET | Nested under agents in v2 |
| `/api/v1/team-selection` | `/api/v2/agents/team-selection` | GET, POST | Nested under agents in v2 |

### Authentication & Users

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/auth/login` | `/api/v2/auth/token` | POST | Returns JWT; field `token` renamed to `access_token` |
| `/api/v1/auth/register` | `/api/v2/auth/register` | POST | |
| `/api/v1/auth/oauth/**` | `/api/v2/auth/oauth/**` | GET, POST | |
| `/api/v1/user` | `/api/v2/users/me` | GET | Path normalized to RESTful plural |
| `/api/v1/rbac/**` | `/api/v2/rbac/**` | ALL | |

### System & Health

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/health` | `/api/v2/system/health` | GET | Moved under `/system` namespace |
| `/api/v1/health/detailed` | `/api/v2/system/health?detail=true` | GET | Query param instead of path |
| `/api/v1/status` | `/api/v2/health` | GET | Simplified path |
| `/api/v1/metrics` | `/api/v2/system/metrics` | GET | Moved under `/system` namespace |

### Analytics & Insights

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/analytics/**` | `/api/v2/analytics/**` | GET | |
| `/api/v1/insights/**` | `/api/v2/analytics/insights/**` | GET | Nested under analytics |
| `/api/v1/flips/**` | `/api/v2/analytics/flips/**` | GET | Nested under analytics |
| `/api/v1/moments/**` | `/api/v2/analytics/moments/**` | GET | Nested under analytics |

### Knowledge & Memory

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/knowledge/**` | `/api/v2/knowledge/**` | ALL | |
| `/api/v1/memory/**` | `/api/v2/memory/**` | GET | |
| `/api/v1/facts/**` | `/api/v2/knowledge/facts/**` | ALL | Nested under knowledge |
| `/api/v1/evidence/**` | `/api/v2/knowledge/evidence/**` | ALL | Nested under knowledge |

### Gauntlet & Verification

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/gauntlet/**` | `/api/v2/gauntlet/**` | GET, POST | |
| `/api/v1/evaluate` | `/api/v2/gauntlet/evaluate` | POST | Nested under gauntlet |
| `/api/v1/verify/**` | `/api/v2/verification/**` | POST | Namespace changed |

### Workflow & Automation

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/workflows` | `/api/v2/workflows` | ALL | |
| `/api/v1/workflow-templates` | `/api/v2/workflows/templates` | ALL | Nested under workflows |
| `/api/v1/workflow-executions` | `/api/v2/workflows/executions` | ALL | Nested under workflows |
| `/api/v1/approvals` | `/api/v2/workflows/approvals` | ALL | Nested under workflows |
| `/api/v1/webhooks` | `/api/v2/webhooks` | ALL | |

### Billing & Usage

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/billing/**` | `/api/v2/billing/**` | ALL | |
| `/api/v1/budgets` | `/api/v2/billing/budgets` | ALL | Nested under billing |
| `/api/v1/costs` | `/api/v2/billing/costs` | GET | Nested under billing |
| `/api/v1/quotas` | `/api/v2/billing/quotas` | GET | Nested under billing |
| `/api/v1/usage/**` | `/api/v2/billing/usage/**` | GET | Nested under billing |
| `/api/v1/accounting/**` | `/api/v2/billing/accounting/**` | ALL | Nested under billing |

### Integrations & Connectors

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/integrations/**` | `/api/v2/integrations/**` | ALL | v2 wizard at `/api/v2/integrations/wizard` |
| `/api/v1/connectors/**` | `/api/v2/connectors/**` | ALL | |
| `/api/v1/bots/**` | `/api/v2/bots/**` | ALL | Slack, Discord, Telegram, etc. |

### Blockchain & ERC-8004

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/blockchain/**` | `/api/v2/blockchain/**` | ALL | |
| `/api/v1/openclaw/**` | `/api/v2/openclaw/**` | ALL | |

### Nomic Loop & Self-Improvement

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/nomic/**` | `/api/v2/nomic/**` | ALL | |
| `/api/v1/genesis/**` | `/api/v2/genesis/**` | ALL | |
| `/api/v1/evolution/**` | `/api/v2/evolution/**` | GET | |

### Gateway & Routing

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/gateway/**` | `/api/v2/gateway/**` | ALL | |
| `/api/v1/routing/**` | `/api/v2/routing/**` | ALL | |

### Security & Compliance

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/privacy/**` | `/api/v2/users/me/**` | GET, DELETE | Restructured under users |
| `/api/v1/audit/**` | `/api/v2/audit/**` | ALL | |
| `/api/v1/threat/**` | `/api/v2/security/threat/**` | ALL | Nested under security |
| `/api/v1/compliance/**` | `/api/v2/compliance/**` | ALL | |

### Marketplace & Skills

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/marketplace` | `/api/v2/marketplace` | ALL | |
| `/api/v1/skills/**` | `/api/v2/skills/**` | ALL | |
| `/api/v1/plugins/**` | `/api/v2/plugins/**` | ALL | |

### Miscellaneous

| v1 Endpoint | v2 Endpoint | Methods | Notes |
|-------------|-------------|---------|-------|
| `/api/v1/canvas/**` | `/api/v2/canvas/**` | ALL | |
| `/api/v1/computer-use/**` | `/api/v2/computer-use/**` | ALL | |
| `/api/v1/slos` | `/api/v2/slo/status` | GET | Path simplified |
| `/api/v1/replays` | `/api/v2/replays` | GET | |
| `/api/v1/tournaments` | `/api/v2/tournaments` | GET | |
| `/api/v1/reviews` | `/api/v2/reviews` | GET | |
| `/api/v1/verticals` | `/api/v2/verticals` | GET | |
| `/api/v1/features` | `/api/v2/features` | GET | |
| `/api/v1/checkpoints` | `/api/v2/checkpoints` | ALL | |
| `/api/v1/dashboard/**` | `/api/v2/dashboard/**` | GET | |
| `/api/v1/cross-pollination/**` | `/api/v2/cross-pollination/**` | GET | |
| `/api/v1/inbox/**` | `/api/v2/inbox/**` | GET, POST | |
| `/api/v1/onboarding/**` | `/api/v2/onboarding/**` | ALL | |
| `/api/v1/graphql` | `/api/v2/graphql` | POST | |
| `/api/v1/docs` | `/api/v2/docs` | GET | OpenAPI docs |

## Breaking Changes

### 1. Authentication Token Response

**v1:**
```json
{"token": "eyJ...", "expires_in": 3600}
```

**v2:**
```json
{"access_token": "eyJ...", "token_type": "Bearer", "expires_in": 3600}
```

**Migration:** Rename `token` to `access_token` in your parsing code.  The
`token_type` field is new; `Bearer` is the only value currently.

### 2. User Endpoint Path

**v1:** `GET /api/v1/user`
**v2:** `GET /api/v2/users/me`

**Migration:** Update the URL.  The response body is unchanged.

### 3. Singular "debate" Path Removed

**v1:** `POST /api/v1/debate` and `GET /api/v1/debate/{id}` (singular)
**v2:** Only `POST /api/v2/debates` and `GET /api/v2/debates/{id}` (plural)

**Migration:** Replace `debate` with `debates` in all paths.

### 4. Consensus Endpoint Restructured

**v1:** `GET /api/v1/consensus/{debate_id}`
**v2:** `GET /api/v2/debates/{debate_id}/consensus`

**Migration:** Consensus is now a sub-resource of the debate.

### 5. Leaderboard and Rankings Nested Under Agents

**v1:** `GET /api/v1/leaderboard`, `GET /api/v1/rankings`
**v2:** `GET /api/v2/agents/leaderboard`, `GET /api/v2/agents/rankings`

### 6. Health and Metrics Moved Under System

**v1:** `GET /api/v1/health`, `GET /api/v1/metrics`
**v2:** `GET /api/v2/system/health`, `GET /api/v2/system/metrics`

### 7. Billing Consolidation

Budget, cost, quota, usage, and accounting endpoints are now nested under
`/api/v2/billing/`.  For example:

- `/api/v1/budgets` becomes `/api/v2/billing/budgets`
- `/api/v1/costs` becomes `/api/v2/billing/costs`
- `/api/v1/quotas` becomes `/api/v2/billing/quotas`

### 8. Privacy Endpoints Restructured

**v1:** `GET /api/v1/privacy/export`, `DELETE /api/v1/privacy/account`
**v2:** `GET /api/v2/users/me/export`, `DELETE /api/v2/users/me`

## Code Examples

### Python (requests)

```python
# Before (v1)
import requests

resp = requests.post(
    "https://api.aragora.ai/api/v1/auth/login",
    json={"email": "user@example.com", "password": "..."},
)
token = resp.json()["token"]

resp = requests.get(
    "https://api.aragora.ai/api/v1/debates",
    headers={"Authorization": f"Bearer {token}"},
)
debates = resp.json()

# After (v2)
resp = requests.post(
    "https://api.aragora.ai/api/v2/auth/token",
    json={"email": "user@example.com", "password": "..."},
)
access_token = resp.json()["access_token"]

resp = requests.get(
    "https://api.aragora.ai/api/v2/debates",
    headers={"Authorization": f"Bearer {access_token}"},
)
debates = resp.json()
```

### Python (aragora-sdk)

```python
# The SDK handles versioning automatically.
# Update to aragora-sdk >= 2.0.0 for v2 endpoints.

from aragora_sdk import AragoraClient

client = AragoraClient(api_key="...", api_version="v2")  # default is now v2
debates = client.debates.list()
```

### TypeScript

```typescript
// Before (v1)
const resp = await fetch('/api/v1/debates');
const debates = await resp.json();

// After (v2)
const resp = await fetch('/api/v2/debates');
const debates = await resp.json();
```

### cURL

```bash
# Before (v1)
curl -H "Authorization: Bearer $TOKEN" \
     https://api.aragora.ai/api/v1/debates

# After (v2)
curl -H "Authorization: Bearer $TOKEN" \
     https://api.aragora.ai/api/v2/debates
```

### Detecting Deprecation Headers

```python
import requests

resp = requests.get("https://api.aragora.ai/api/v1/debates",
                     headers={"Authorization": f"Bearer {token}"})

# Check for deprecation
if "Sunset" in resp.headers:
    print(f"WARNING: This endpoint sunsets on {resp.headers['Sunset']}")
    print(f"Level: {resp.headers.get('X-Deprecation-Level')}")
    print(f"Migrate to: see Link header")
```

## SDK Version Compatibility

| SDK Version | Default API | v1 Support | v2 Support |
|-------------|-------------|------------|------------|
| < 1.0 | v1 | Full | None |
| 1.x | v1 | Full | Opt-in (`api_version="v2"`) |
| >= 2.0 | v2 | Opt-in (`api_version="v1"`) | Full |

## FAQ

**Q: Can I keep using v1 after the sunset date?**
A: By default, v1 endpoints will return `410 Gone` after June 1, 2026.
Operators can set `ARAGORA_BLOCK_SUNSET_ENDPOINTS=false` as a temporary
escape hatch, but this is not recommended for production use.

**Q: How do I find all v1 calls in my codebase?**
A: Search for `/api/v1/` in your source code.  In production, check for the
`X-API-Version: v1` response header or search server logs for
`v1_api_access` entries.

**Q: Will v1 and v2 work side-by-side?**
A: Yes.  Both versions are served from the same server.  You can migrate
endpoints incrementally.

**Q: What about the legacy `/api/` (unversioned) endpoints?**
A: Unversioned `/api/` paths are treated as v1 and follow the same sunset
schedule.

**Q: How do I monitor my migration progress?**
A: The `GET /api/v2/system/deprecation/stats` endpoint returns usage counts
for all deprecated v1 endpoints, broken down by path and method.

## Support

- Migration docs: https://docs.aragora.ai/migration/v1-to-v2
- API reference: https://docs.aragora.ai/api
- Support: support@aragora.ai
