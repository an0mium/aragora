# Aragora Documentation

Welcome to Aragora's documentation. The `docs/` directory is the canonical
source. The published site in `docs-site/` is synced from these files via
`docs-site/scripts/sync-docs.js`.

Aragora is the **control plane for multi-agent vetted decisionmaking across
organizational knowledge and channels**.

## Quick Start

| Document | Description |
|----------|-------------|
| [GETTING_STARTED](./GETTING_STARTED.md) | Comprehensive onboarding guide |

## Core Concepts

| Document | Description |
|----------|-------------|
| [ARCHITECTURE](./ARCHITECTURE.md) | System architecture overview |
| [FEATURES](./FEATURES.md) | Complete feature documentation |
| [MODES_GUIDE](./MODES_GUIDE.md) | Debate modes (standard, gauntlet, genesis) |
| [DEBATE_INTERNALS](./DEBATE_INTERNALS.md) | Debate engine internals (Arena, phases, consensus) |
| [REASONING](./REASONING.md) | Belief networks, provenance, and claims |
| [WORKFLOW_ENGINE](./WORKFLOW_ENGINE.md) | DAG-based workflow execution |
| [RESILIENCE](./RESILIENCE.md) | Circuit breaker and fault tolerance |
| [CONTROL_PLANE](./CONTROL_PLANE.md) | Control plane architecture |
| [CONTROL_PLANE_GUIDE](./CONTROL_PLANE_GUIDE.md) | Control plane operations guide |
| [MEMORY](./MEMORY.md) | Memory systems overview |
| [DOCUMENTS](./DOCUMENTS.md) | Document ingestion and parsing |

## Using Aragora

### Debates & Gauntlet

| Document | Description |
|----------|-------------|
| [GAUNTLET](./GAUNTLET.md) | Adversarial stress testing (primary) |
| [PROBE_STRATEGIES](./PROBE_STRATEGIES.md) | Probe attack strategies |
| [GENESIS](./GENESIS.md) | Agent evolution and genesis |
| [GRAPH_DEBATES](./GRAPH_DEBATES.md) | Graph debate mode (experimental) |
| [MATRIX_DEBATES](./MATRIX_DEBATES.md) | Matrix debate mode (experimental) |

### Agents & Memory

| Document | Description |
|----------|-------------|
| [AGENT_SELECTION](./AGENT_SELECTION.md) | Agent selection algorithms |
| [AGENTS](./AGENTS.md) | Agent type catalog and defaults |
| [AGENT_DEVELOPMENT](./AGENT_DEVELOPMENT.md) | Creating custom agents |
| [CUSTOM_AGENTS](./CUSTOM_AGENTS.md) | Custom agent configuration |
| [MEMORY_STRATEGY](./MEMORY_STRATEGY.md) | Memory tier architecture |
| [MEMORY_ANALYTICS](./MEMORY_ANALYTICS.md) | Memory system analytics |

### Inbox & Channels

| Document | Description |
|----------|-------------|
| [INBOX_GUIDE](./INBOX_GUIDE.md) | Unified inbox setup and triage workflows |
| [EMAIL_PRIORITIZATION](./EMAIL_PRIORITIZATION.md) | Priority scoring and tiers |
| [SHARED_INBOX](./SHARED_INBOX.md) | Shared inbox routing and team workflows |
| [CHANNELS](./CHANNELS.md) | Supported channels and delivery |

### Integrations

| Document | Description |
|----------|-------------|
| [MCP_INTEGRATION](./MCP_INTEGRATION.md) | Model Context Protocol setup |
| [MCP_ADVANCED](./MCP_ADVANCED.md) | Advanced MCP patterns |
| [INTEGRATIONS](./INTEGRATIONS.md) | Third-party integrations |
| [TINKER_INTEGRATION](./TINKER_INTEGRATION.md) | Tinker framework integration |

## API & SDK

| Document | Description |
|----------|-------------|
| [API_REFERENCE](./API_REFERENCE.md) | Complete API documentation |
| [API_ENDPOINTS](./API_ENDPOINTS.md) | HTTP endpoint reference |
| [API_EXAMPLES](./API_EXAMPLES.md) | API usage examples |
| [API_VERSIONING](./API_VERSIONING.md) | API version policy |
| [WEBSOCKET_EVENTS](./WEBSOCKET_EVENTS.md) | WebSocket event reference |
| [SDK_TYPESCRIPT](./SDK_TYPESCRIPT.md) | TypeScript SDK guide |
| [SDK_CONSOLIDATION](./SDK_CONSOLIDATION.md) | TypeScript SDK migration (v2 → v3) |
| [SDK_GUIDE](./SDK_GUIDE.md) | Python SDK guide |
| [LIBRARY_USAGE](./LIBRARY_USAGE.md) | Using Aragora as a library |

## Operations & Deployment

| Document | Description |
|----------|-------------|
| [DEPLOYMENT](./DEPLOYMENT.md) | Deployment guide |
| [OPERATIONS](./OPERATIONS.md) | Operations runbook |
| [RUNBOOK](./RUNBOOK.md) | Incident response procedures |
| [PRODUCTION_READINESS](./PRODUCTION_READINESS.md) | Production readiness checklist |
| [OBSERVABILITY](./OBSERVABILITY.md) | Monitoring and telemetry |
| [RATE_LIMITING](./RATE_LIMITING.md) | Rate limit configuration |
| [QUEUE](./QUEUE.md) | Debate queue management |

## Security

| Document | Description |
|----------|-------------|
| [SECURITY](./SECURITY.md) | Security overview |
| [SECURITY_DEPLOYMENT](./SECURITY_DEPLOYMENT.md) | Secure deployment practices |
| [SECRETS_MANAGEMENT](./SECRETS_MANAGEMENT.md) | Managing API keys and secrets |
| [SSO_SETUP](./SSO_SETUP.md) | SSO configuration |
| [COMPLIANCE_PRESETS](./COMPLIANCE_PRESETS.md) | Built-in audit presets |

## Configuration

| Document | Description |
|----------|-------------|
| [ENVIRONMENT](./ENVIRONMENT.md) | Environment variables reference |
| [DATABASE](./DATABASE.md) | Database architecture |
| [CLI_REFERENCE](./CLI_REFERENCE.md) | CLI command reference |

## Development

| Document | Description |
|----------|-------------|
| [FRONTEND_DEVELOPMENT](./FRONTEND_DEVELOPMENT.md) | Frontend contribution guide |
| [FRONTEND_ROUTES](./FRONTEND_ROUTES.md) | Frontend route and feature map |
| [TESTING](./TESTING.md) | Test suite documentation |
| [DEPRECATION_POLICY](./DEPRECATION_POLICY.md) | Deprecation and migration policy |
| [ERROR_CODES](./ERROR_CODES.md) | Error code reference |

## Features

| Document | Description |
|----------|-------------|
| [PULSE](./PULSE.md) | Trending topic automation |
| [BROADCAST](./BROADCAST.md) | Audio broadcast generation |
| [NOMIC_LOOP](./NOMIC_LOOP.md) | Self-improvement system |
| [FORMAL_VERIFICATION](./FORMAL_VERIFICATION.md) | Z3/Lean verification |
| [TRICKSTER](./TRICKSTER.md) | Hollow consensus detection |
| [GOVERNANCE](./GOVERNANCE.md) | Decision governance |
| [BILLING](./BILLING.md) | Billing and subscriptions |

## Compliance

| Document | Description |
|----------|-------------|
| [A_B_TESTING](./A_B_TESTING.md) | A/B testing framework |
| [EVOLUTION_PATTERNS](./EVOLUTION_PATTERNS.md) | Evolution pattern library |
| [GITHUB_ACTIONS](./GITHUB_ACTIONS.md) | CI/CD integration |

## Troubleshooting

| Document | Description |
|----------|-------------|
| [TROUBLESHOOTING](./TROUBLESHOOTING.md) | Common issues and solutions |
| [NOMIC_LOOP_TROUBLESHOOTING](./NOMIC_LOOP_TROUBLESHOOTING.md) | Nomic loop specific issues |

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
frontend contributions, also see [FRONTEND_DEVELOPMENT.md](./FRONTEND_DEVELOPMENT.md),
and for new agents, see [AGENT_DEVELOPMENT.md](./AGENT_DEVELOPMENT.md).

## Documentation Maintenance

### Review Schedule

Documentation is reviewed and updated according to this schedule:

| Category | Review Frequency | Last Review |
|----------|------------------|-------------|
| Quick Start / Getting Started | Monthly | 2026-01 |
| API Reference | With each release | 2026-01 |
| Architecture / Core Concepts | Quarterly | 2026-01 |
| Feature Documentation | When features change | Ongoing |
| Security Documentation | Monthly | 2026-01 |
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
