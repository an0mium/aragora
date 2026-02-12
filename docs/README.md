# Aragora Documentation

Welcome to Aragora's documentation. The `docs/` directory is the canonical
source. The published site in `docs-site/` is synced from these files via
`docs-site/scripts/sync-docs.js`.

Aragora is the **control plane for multi-agent vetted decisionmaking across
organizational knowledge and channels**.

## Quick Start

| Document | Description |
|----------|-------------|
| [GETTING_STARTED](./guides/GETTING_STARTED.md) | Comprehensive onboarding guide |

## Core Concepts

| Document | Description |
|----------|-------------|
| [ARCHITECTURE](./architecture/ARCHITECTURE.md) | System architecture overview |
| [FEATURES](./status/FEATURES.md) | Complete feature documentation |
| [MODES_GUIDE](./guides/MODES_GUIDE.md) | Debate modes (standard, gauntlet, genesis) |
| [DEBATE_INTERNALS](./debate/DEBATE_INTERNALS.md) | Debate engine internals (Arena, phases, consensus) |
| [REASONING](./workflow/REASONING.md) | Belief networks, provenance, and claims |
| [WORKFLOW_ENGINE](./workflow/WORKFLOW_ENGINE.md) | DAG-based workflow execution |
| [RESILIENCE](./resilience/RESILIENCE.md) | Circuit breaker and fault tolerance |
| [CONTROL_PLANE](./reference/CONTROL_PLANE.md) | Control plane architecture |
| [CONTROL_PLANE_GUIDE](./guides/CONTROL_PLANE_GUIDE.md) | Control plane operations guide |
| [MEMORY](./knowledge/MEMORY.md) | Memory systems overview |
| [DOCUMENTS](./reference/DOCUMENTS.md) | Document ingestion and parsing |

## Using Aragora

### Debates & Gauntlet

| Document | Description |
|----------|-------------|
| [GAUNTLET](./debate/GAUNTLET.md) | Adversarial stress testing (primary) |
| [PROBE_STRATEGIES](./debate/PROBE_STRATEGIES.md) | Probe attack strategies |
| [GENESIS](./workflow/GENESIS.md) | Agent evolution and genesis |
| [GRAPH_DEBATES](./debate/GRAPH_DEBATES.md) | Graph debate mode (experimental) |
| [MATRIX_DEBATES](./debate/MATRIX_DEBATES.md) | Matrix debate mode (experimental) |

### Agents & Memory

| Document | Description |
|----------|-------------|
| [AGENT_SELECTION](./debate/AGENT_SELECTION.md) | Agent selection algorithms |
| [AGENTS](./debate/AGENTS.md) | Agent type catalog and defaults |
| [AGENT_DEVELOPMENT](./debate/AGENT_DEVELOPMENT.md) | Creating custom agents |
| [CUSTOM_AGENTS](./guides/CUSTOM_AGENTS.md) | Custom agent configuration |
| [MEMORY_STRATEGY](./knowledge/MEMORY_STRATEGY.md) | Memory tier architecture |
| [MEMORY_ANALYTICS](./knowledge/MEMORY_ANALYTICS.md) | Memory system analytics |

### Inbox & Channels

| Document | Description |
|----------|-------------|
| [INBOX_GUIDE](./guides/INBOX_GUIDE.md) | Unified inbox setup and triage workflows |
| [EMAIL_PRIORITIZATION](./integrations/EMAIL_PRIORITIZATION.md) | Priority scoring and tiers |
| [SHARED_INBOX](./guides/SHARED_INBOX.md) | Shared inbox routing and team workflows |
| [CHANNELS](./integrations/CHANNELS.md) | Supported channels and delivery |

### Integrations

| Document | Description |
|----------|-------------|
| [MCP_INTEGRATION](./integrations/MCP_INTEGRATION.md) | Model Context Protocol setup |
| [MCP_ADVANCED](./integrations/MCP_ADVANCED.md) | Advanced MCP patterns |
| [INTEGRATIONS](./integrations/INTEGRATIONS.md) | Third-party integrations |
| [TINKER_INTEGRATION](./integrations/TINKER_INTEGRATION.md) | Tinker framework integration |

## API & SDK

| Document | Description |
|----------|-------------|
| [API_REFERENCE](./api/API_REFERENCE.md) | Complete API documentation |
| [API_ENDPOINTS](./api/API_ENDPOINTS.md) | HTTP endpoint reference |
| [API_EXAMPLES](./api/API_EXAMPLES.md) | API usage examples |
| [API_VERSIONING](./api/API_VERSIONING.md) | API version policy |
| [BREAKING_CHANGES](./reference/BREAKING_CHANGES.md) | Breaking changes and migration |
| [MIGRATION_V1_TO_V2](./status/MIGRATION_V1_TO_V2.md) | API v1 to v2 migration guide |
| [WEBSOCKET_EVENTS](./streaming/WEBSOCKET_EVENTS.md) | WebSocket event reference |
| [SDK_TYPESCRIPT](./guides/SDK_TYPESCRIPT.md) | TypeScript SDK guide |
| [SDK_CONSOLIDATION](./guides/SDK_CONSOLIDATION.md) | TypeScript SDK migration (v2 → v3) |
| [SDK_GUIDE](./SDK_GUIDE.md) | Python SDK guide |
| [LIBRARY_USAGE](./reference/LIBRARY_USAGE.md) | Using Aragora as a library |

## Operations & Deployment

| Document | Description |
|----------|-------------|
| [DEPLOYMENT](./deployment/DEPLOYMENT.md) | Deployment guide |
| [OPERATIONS](./OPERATIONS.md) | Operations runbook |
| [RUNBOOK](./deployment/RUNBOOK.md) | Incident response procedures |
| [PRODUCTION_READINESS](./deployment/PRODUCTION_READINESS.md) | Production readiness checklist |
| [OBSERVABILITY](./observability/OBSERVABILITY.md) | Monitoring and telemetry |
| [RATE_LIMITING](./api/RATE_LIMITING.md) | Rate limit configuration |
| [QUEUE](./resilience/QUEUE.md) | Debate queue management |

## Security

| Document | Description |
|----------|-------------|
| [SECURITY](./enterprise/SECURITY.md) | Security overview |
| [SECURITY_DEPLOYMENT](./deployment/SECURITY_DEPLOYMENT.md) | Secure deployment practices |
| [SECRETS_MANAGEMENT](./enterprise/SECRETS_MANAGEMENT.md) | Managing API keys and secrets |
| [SSO_SETUP](./enterprise/SSO_SETUP.md) | SSO configuration |
| [COMPLIANCE_PRESETS](./enterprise/COMPLIANCE_PRESETS.md) | Built-in audit presets |

## Configuration

| Document | Description |
|----------|-------------|
| [ENVIRONMENT](./reference/ENVIRONMENT.md) | Environment variables reference |
| [DATABASE](./reference/DATABASE.md) | Database architecture |
| [CLI_REFERENCE](./reference/CLI_REFERENCE.md) | CLI command reference |

## Development

| Document | Description |
|----------|-------------|
| [FRONTEND_DEVELOPMENT](./debate/FRONTEND_DEVELOPMENT.md) | Frontend contribution guide |
| [FRONTEND_ROUTES](./guides/FRONTEND_ROUTES.md) | Frontend route and feature map |
| [TESTING](./testing/TESTING.md) | Test suite documentation |
| [BREAKING_CHANGES](./reference/BREAKING_CHANGES.md) | Breaking changes by version |
| [DEPRECATION_POLICY](./reference/DEPRECATION_POLICY.md) | Deprecation and migration policy |
| [ERROR_CODES](./reference/ERROR_CODES.md) | Error code reference |

## Features

| Document | Description |
|----------|-------------|
| [PULSE](./resilience/PULSE.md) | Trending topic automation |
| [BROADCAST](./integrations/BROADCAST.md) | Audio broadcast generation |
| [NOMIC_LOOP](./workflow/NOMIC_LOOP.md) | Self-improvement system |
| [FORMAL_VERIFICATION](./workflow/FORMAL_VERIFICATION.md) | Z3/Lean verification |
| [TRICKSTER](./debate/TRICKSTER.md) | Hollow consensus detection |
| [GOVERNANCE](./enterprise/GOVERNANCE.md) | Decision governance |
| [BILLING](./reference/BILLING.md) | Billing and subscriptions |

## Compliance

| Document | Description |
|----------|-------------|
| [A_B_TESTING](./testing/A_B_TESTING.md) | A/B testing framework |
| [EVOLUTION_PATTERNS](./workflow/EVOLUTION_PATTERNS.md) | Evolution pattern library |
| [GITHUB_ACTIONS](./deployment/GITHUB_ACTIONS.md) | CI/CD integration |

## Troubleshooting

| Document | Description |
|----------|-------------|
| [TROUBLESHOOTING](./guides/TROUBLESHOOTING.md) | Common issues and solutions |
| [NOMIC_LOOP_TROUBLESHOOTING](./guides/NOMIC_LOOP_TROUBLESHOOTING.md) | Nomic loop specific issues |

## Case Studies

| Document | Description |
|----------|-------------|
| [case-studies/README](./case-studies/README.md) | Real-world applications and audits |

---

## Archived/Historical Documents

Deprecated and historical documents live under `docs/deprecated/`. These are
kept for reference but are no longer maintained.

See [docs/deprecated/README.md](./deprecated/README.md) for the full index.

---

## API Documentation

- [OpenAPI Specification (YAML)](./api/openapi.yaml)
- [OpenAPI Specification (JSON)](./api/openapi.json)
- [Interactive Docs](./index.html) - Swagger UI
  - `openapi.yaml` is JSON-formatted for compatibility; regenerate with
    `python scripts/export_openapi.py --output-dir docs/api`.

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full contribution guide. For
frontend contributions, also see [FRONTEND_DEVELOPMENT.md](./debate/FRONTEND_DEVELOPMENT.md),
and for new agents, see [AGENT_DEVELOPMENT.md](./debate/AGENT_DEVELOPMENT.md).

## Documentation Maintenance

### Review Schedule

Documentation is reviewed and updated according to this schedule:

| Category | Review Frequency | Last Review |
|----------|------------------|-------------|
| Quick Start / Getting Started | Monthly | 2026-02 |
| API Reference | With each release | 2026-02 |
| Architecture / Core Concepts | Quarterly | 2026-02 |
| Feature Documentation | When features change | 2026-02 |
| Security Documentation | Monthly | 2026-02 |
| Troubleshooting | As issues are reported | Ongoing |

### Documentation Review Checklist

When reviewing documentation:

1. **Accuracy**: Do code examples still work? Are APIs current?
2. **Completeness**: Are all features documented? Missing sections?
3. **Clarity**: Is the writing clear and accessible?
4. **Links**: Do all internal/external links work?
5. **Versioning**: Is version-specific info clearly marked?

### Reporting Issues

Found outdated or incorrect documentation?

1. Open a GitHub issue with the label `documentation`
2. Include the document path and section
3. Describe what's incorrect or outdated
4. Suggest a correction if possible

---

## Support

- [GitHub Issues](https://github.com/an0mium/aragora/issues)
- [Documentation Updates](https://github.com/an0mium/aragora/pulls)
