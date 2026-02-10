# Aragora Capability Matrix

> Updated: 2026-02-10
> Purpose: Map features to their exposure across CLI, API, SDK, and UI surfaces

## Executive Summary

| Surface | Features Exposed | Coverage |
|---------|------------------|----------|
| **SDK (Python/TypeScript)** | 153 namespaces | 100% (baseline) |
| **HTTP API** | 613 handler files / 1,635 paths / 1,896 operations | ~85% |
| **CLI** | 45 commands | ~8% |
| **UI (docs-site)** | TBD | TBD |

The HTTP API surface is extensive — most SDK namespaces have corresponding HTTP endpoints.

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
| Debates (multi-agent) | Y | Y (130 ops) | Y (`ask`) | Y | **Full** |
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
| Evolution (A/B) | Y | Y (15 ops) | N | Y | **API** |

### Knowledge & Memory

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Knowledge Base | Y | Y | Y (`knowledge`) | Y | **Full** |
| Knowledge Mound | Y | Y | N | Y | **API** |
| Memory (continuum) | Y | Y | Y (`memory`) | Y | **Full** |
| Evidence | Y | Y | N | Y | **API** |
| Belief Network | Y | Y | N | Y | **API** |
| Cross-Pollination | Y | Y | N | Y | **API** |
| RLM Context | Y | Y | Y (`rlm`) | Y | **Full** |
| Facts | Y | Y (20 ops) | N | Y | **API** |

### Documents & Content

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Documents | Y | Y | Y (`documents`) | Y | **Full** |
| Document Audit | Y | Y | Y | Y | **Full** |
| Code Review | Y | Y | Y (`review`) | Y | **Full** |
| Codebase Intel | Y | Y | N | Y | **API** |
| Threat Intel | Y | Y | N | Y | **API** |

### Enterprise & Admin

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Admin | Y | Y (24 ops) | N | Y | **API** |
| Auth | Y | Y (118 ops) | N | Y | **API** |
| OAuth | Y | Y | N | Y | **API** |
| SSO | Y | Y | N | Y | **API** |
| SCIM | Y | Y | N | Y | **API** |
| RBAC | Y | Y | N | Y | **API** |
| Tenants | Y | N | Y (`tenant`) | Y | **CLI** (no HTTP) |
| Workspaces | Y | Y | N | Y | **API** |
| Organizations | Y | Y | N | Y | **API** |
| Compliance | Y | Y (11 ops) | N | Y | **API** |
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
| Payments | Y | Y (49 ops) | N | Y | **API** |

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
| Workflows | Y | Y (57 ops) | N | Y | **API** |
| Workflow Templates | Y | Y | Y (`template`) | Y | **Full** |
| Queue | Y | Y | N | Y | **API** |
| Scheduler | Y | Y | N | Y | **API** |
| Plugins | Y | Y | N | Y | **API** |

### Observability

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Analytics | Y | Y (117 ops) | N | Y | **API** |
| Metrics | Y | Y | N | Y | **API** |
| SLO | Y | Y | N | Y | **API** |
| Pulse (trending) | Y | Y | N | Y | **API** |
| Dashboard | Y | Y | N | Y | **API** |
| Health | Y | Y | Y (`status`) | Y | **Full** |

### Marketplace & Skills

| Feature | SDK | HTTP | CLI | Tests | Status |
|---------|-----|------|-----|-------|--------|
| Marketplace | Y | Y | Y (`marketplace`) | Y | **Full** |
| Skills | Y | Y | N | Y | **API** |
| Templates | Y | Y | Y (`templates`) | Y | **Full** |

---

## Gap Analysis

The HTTP API surface is far more extensive than previously documented. Most features initially
flagged as "SDK-only" actually have HTTP endpoints (debates: 130 ops, auth: 118 ops, analytics: 117 ops, etc.).

### Priority 1: CLI Coverage

Most features are accessible via SDK and HTTP but lack CLI commands (45 commands vs 1,896 HTTP operations). High-value CLI additions:

- `aragora analytics [query|report]` - Business intelligence from terminal
- `aragora admin [users|config]` - System administration
- `aragora compliance [check|report]` - Compliance status
- `aragora workflows [list|run|status]` - Workflow management

### Priority 2: Test Gaps

Some namespaces have limited or no test coverage:
- evaluation, moments, reviews, workflow_templates

### Priority 3: SDK Namespace Coverage

153 Python SDK namespaces cover the API surface comprehensively. Remaining gaps are
primarily in newer features where HTTP endpoints exist but SDK wrappers haven't been generated yet.

---

## Activation Roadmap

### Phase 1: CLI Expansion (Weeks 1-4)

Add CLI commands for high-value HTTP APIs that currently lack CLI access:

```bash
aragora analytics [query|report]       # 117 HTTP ops, no CLI
aragora admin [users|config|health]    # 24 HTTP ops, no CLI
aragora compliance [check|report]      # 11 HTTP ops, no CLI
aragora workflows [list|run|status]    # 57 HTTP ops, no CLI
```

### Phase 2: Test Coverage (Weeks 3-6)

Add tests for namespaces with limited coverage.

### Phase 3: FastAPI Migration (Ongoing)

Incrementally migrate handlers from legacy server to FastAPI.
Current FastAPI surface: 4 route groups (health, debates, decisions, testfixer).
Target: full API parity with legacy unified_server.

---

## Validation Commands

```bash
# Count SDK namespaces
ls sdk/python/aragora_sdk/namespaces/*.py | wc -l  # Expected: 153

# Count HTTP handler files
find aragora/server/handlers -name "*.py" | wc -l  # Expected: 613

# Count API paths from OpenAPI spec
python3 -c "import json; d=json.load(open('openapi.json')); print(len(d['paths']))"  # Expected: 1,635

# Check CLI coverage
aragora --help | wc -l

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
