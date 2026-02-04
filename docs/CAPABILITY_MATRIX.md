# Aragora Capability Matrix

> Generated: 2026-02-03
> Purpose: Map features to their exposure across CLI, API, SDK, and UI surfaces

## Executive Summary

| Surface | Features Exposed | Coverage |
|---------|------------------|----------|
| **SDK (Python/TypeScript)** | 134 namespaces | 100% (baseline) |
| **HTTP API** | 60 handlers / 896 endpoints | 45% |
| **CLI** | 45 commands | 8% |
| **UI (docs-site)** | TBD | TBD |

**Critical Gap:** 67 namespaces (50%) are SDK-only with no HTTP API access.

---

## Feature Coverage Matrix

### Legend
- **Full** = SDK + HTTP + CLI + Tests
- **API** = SDK + HTTP + Tests (no CLI)
- **CLI** = SDK + CLI + Tests (local only)
- **SDK** = SDK + Tests only (no HTTP/CLI)
- **Stub** = Defined but not implemented

### Core Decision Making

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Debates (multi-agent) | Y | N | Y (`ask`) | Y | **CLI** |
| Graph Debates | Y | Y | Y (`--graph`) | Y | **Full** |
| Matrix Debates | Y | Y | Y (`--matrix`) | Y | **Full** |
| Consensus Detection | Y | Y | N | Y | **API** |
| Deliberations | Y | Y | N | Y | **API** |
| Decisions Pipeline | Y | Y | Y (`decide`) | Y | **Full** |
| Gauntlet (stress-test) | Y | Y | Y | Y | **Full** |
| Explainability | Y | Y | N | Y | **API** |
| Verification (formal) | Y | Y | N | Y | **API** |
| Receipts | Y | Y | N | Y | **API** |

### Agents & Reasoning

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Agents (CRUD) | Y | Y | Y (`agents`) | Y | **Full** |
| Personas | Y | Y | N | Y | **API** |
| Calibration/ELO | Y | Y | Y (`elo`) | Y | **Full** |
| Training | Y | Y | Y | Y | **Full** |
| A2A Protocol | Y | Y | N | Y | **API** |
| Agent Selection | Y | N | N | Y | **SDK** |
| Verticals | Y | Y | N | Y | **API** |
| Evolution (A/B) | Y | Y | N | Y | **API** |

### Knowledge & Memory

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Knowledge Base | Y | Y | Y (`knowledge`) | Y | **Full** |
| Knowledge Mound | Y | Y | N | Y | **API** |
| Memory (continuum) | Y | Y | Y (`memory`) | Y | **Full** |
| Evidence | Y | Y | N | Y | **API** |
| Belief Network | Y | Y | N | N | **API** |
| Cross-Pollination | Y | Y | N | Y | **API** |
| RLM Context | Y | Y | Y (`rlm`) | Y | **Full** |
| Facts | Y | N | N | Y | **SDK** |

### Documents & Content

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Documents | Y | Y | Y (`documents`) | Y | **Full** |
| Document Audit | Y | Y | Y | Y | **Full** |
| Code Review | Y | Y | Y (`review`) | Y | **Full** |
| Codebase Intel | Y | Y | N | Y | **API** |
| Threat Intel | Y | Y | N | N | **API** |

### Enterprise & Admin

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Admin | Y | N | N | Y | **SDK** |
| Auth | Y | N | N | Y | **SDK** |
| OAuth | Y | Y | N | Y | **API** |
| SSO | Y | Y | N | Y | **API** |
| SCIM | Y | Y | N | Y | **API** |
| RBAC | Y | Y | N | Y | **API** |
| Tenants | Y | N | Y (`tenant`) | Y | **CLI** |
| Workspaces | Y | Y | N | Y | **API** |
| Organizations | Y | Y | N | Y | **API** |
| Compliance | Y | N | N | Y | **SDK** |
| Audit Trails | Y | Y | Y (`audit`) | Y | **Full** |
| Privacy | Y | Y | N | Y | **API** |
| Backup/DR | Y | Y | Y (`backup`) | Y | **Full** |

### Billing & Costs

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Billing | Y | Y | Y (`billing`) | Y | **Full** |
| Costs | Y | Y | N | Y | **API** |
| Budgets | Y | Y | N | Y | **API** |
| Usage Metering | Y | Y | N | Y | **API** |
| Payments | Y | N | N | Y | **SDK** |

### Integrations

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Slack | Y | Y | N | Y | **API** |
| Teams | Y | Y | N | Y | **API** |
| Discord | Y | Y | N | Y | **API** |
| Telegram | Y | Y | N | Y | **API** |
| WhatsApp | Y | Y | N | Y | **API** |
| Gmail | Y | Y | N | Y | **API** |
| Outlook | Y | Y | N | Y | **API** |
| Webhooks | Y | Y | N | Y | **API** |
| Connectors | Y | Y | N | Y | **API** |
| OpenClaw | Y | Y | Y (`openclaw`) | Y | **Full** |

### Workflows & Automation

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Workflows | Y | N | N | Y | **SDK** |
| Workflow Templates | Y | Y | Y (`template`) | Y | **Full** |
| Queue | Y | Y | N | Y | **API** |
| Scheduler | Y | Y | N | Y | **API** |
| Plugins | Y | Y | N | Y | **API** |

### Observability

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Analytics | Y | N | N | Y | **SDK** |
| Metrics | Y | Y | N | Y | **API** |
| SLO | Y | Y | N | Y | **API** |
| Pulse (trending) | Y | Y | N | Y | **API** |
| Dashboard | Y | Y | N | Y | **API** |
| Health | Y | Y | Y (`status`) | Y | **Full** |

### Marketplace & Skills

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Marketplace | Y | Y | Y (`marketplace`) | Y | **Full** |
| Skills | Y | Y | N | N | **API** |
| Templates | Y | Y | Y (`templates`) | Y | **Full** |

---

## Gap Analysis

### Priority 1: SDK-Only Features Needing HTTP API

These 67 namespaces are only accessible via embedded SDK, blocking cloud/SaaS deployments:

**Critical (security/compliance):**
- `admin` - System administration
- `auth` - Authentication flows
- `compliance` - Regulatory requirements

**High (core features):**
- `debates` - Core multi-agent debates (CLI-only currently)
- `analytics` - Business intelligence
- `workflows` - DAG automation
- `notifications` - Alert delivery
- `payments` - Revenue operations

**Medium (enterprise):**
- `control_plane` - Policy governance
- `evolution` - A/B testing
- `integrations` - Third-party connections
- `orchestration` - Task coordination

### Priority 2: HTTP-Only Features Needing CLI

56 handlers have no CLI equivalent, limiting DevOps automation:

**High Value:**
- `dashboard` - Admin dashboard access
- `analytics` - Analytics queries
- `metrics` - Prometheus metrics
- `monitoring` - Health monitoring
- `cross_pollination` - Federation stats

### Priority 3: Missing Tests

12 namespaces lack test coverage:
- belief, evaluation, moments, reviews, skills, threat_intel, workflow_templates
- Plus 5 SDK-only namespaces

---

## Activation Roadmap

### Phase 1: HTTP API Parity (Weeks 1-4)

Add HTTP handlers for critical SDK-only features:

```
aragora/server/handlers/debates_crud.py      # Debates CRUD via HTTP
aragora/server/handlers/admin_api.py         # Admin operations
aragora/server/handlers/auth_api.py          # Auth flows
aragora/server/handlers/compliance_api.py    # Compliance checks
aragora/server/handlers/analytics_api.py     # Analytics queries
```

### Phase 2: CLI Expansion (Weeks 3-6)

Add CLI commands for high-value HTTP handlers:

```bash
aragora dashboard [show|export]
aragora analytics [query|report]
aragora metrics [list|export]
aragora monitoring [status|alerts]
```

### Phase 3: Test Coverage (Ongoing)

Add tests for 12 untested namespaces.

---

## Validation Commands

```bash
# Verify SDK-Handler parity
python scripts/sdk_handler_audit.py --verify

# Check CLI coverage
aragora --help | wc -l

# Count HTTP endpoints
grep -r "@api_endpoint" aragora/server/handlers | wc -l

# Run full test suite
pytest tests/ -v --cov=aragora --cov-fail-under=70
```

---

## Vertical Specialists Activation

The vertical specialists system (Software, Legal, Healthcare, Accounting, Research) is architecturally complete but needs activation:

### Current State
- Registry: Complete (factory pattern, 5 verticals registered)
- HTTP API: Complete (RBAC, circuit breaker, rate limiting)
- SDK: Complete (Python + TypeScript namespaces)
- LLM Integration: Implemented via delegate agent (fallbacks if provider unavailable)
- Tool Connectors: Active (with web fallbacks for unsupported domains)

### Remaining Enhancements (8-16 hours)
1. Add dedicated legal case law and statute connectors (6-10 hrs)
2. Optional: Multi-jurisdiction tax connectors (4-8 hrs)
3. Optional: premium legal sources (Westlaw/Lexis) expansion

### Remaining Connector Gaps
| Vertical | Tool | Suggested Source | Effort |
|----------|------|------------------|--------|
| Accounting | Multi-Jurisdiction Tax | OECD/EU/UK/local tax portals | 4-8 hrs |
| Healthcare | US Guidelines | AHRQ / NIH (optional) | 4-8 hrs |

### Configuration Prereqs
- **FASB GAAP**: Requires licensed API or internal proxy (`FASB_API_BASE`, `FASB_API_KEY`)
- **IRS Tax**: Requires IRS search API or internal proxy (`IRS_API_BASE`, `IRS_API_KEY`)
- **Westlaw/Lexis**: Requires licensed API access (`WESTLAW_API_BASE`, `WESTLAW_API_KEY`, `LEXIS_API_BASE`, `LEXIS_API_KEY`)
- **Multi-Jurisdiction Tax**: `TAX_{JURISDICTION}_API_BASE` / `TAX_{JURISDICTION}_SEARCH_URL` (optional `TAX_{JURISDICTION}_API_KEY`)
- **Proxy flexibility**: `FASB_SEARCH_METHOD`, `FASB_SEARCH_QUERY_PARAM`, `FASB_SEARCH_LIMIT_PARAM`, `IRS_SEARCH_METHOD`, `IRS_SEARCH_QUERY_PARAM`, `IRS_SEARCH_LIMIT_PARAM`

---

## References

- [API Reference](API_REFERENCE.md)
- [CLI Reference](CLI_REFERENCE.md)
- [SDK Documentation](../sdk/python/README.md)
- [Handler Registry](../aragora/server/handler_registry/)
- [STATUS.md](STATUS.md)
