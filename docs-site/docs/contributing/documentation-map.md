---
title: Documentation Map
description: Documentation Map
---

# Documentation Map

This map describes how documentation is organized and which files are canonical.
The `docs/` directory is authoritative; `docs-site/` is a published view synced
from these sources.

For quick navigation, see [INDEX.md](./documentation-index). Deprecated and historical
documents are archived in [deprecated/README.md](../analysis/adr).

## Quick Navigation

| I want to... | Go to |
|--------------|-------|
| Get started | [GETTING_STARTED.md](../getting-started/overview) |
| Understand the architecture | [ARCHITECTURE.md](../core-concepts/architecture) |
| Run the control plane | [CONTROL_PLANE_GUIDE.md](../enterprise/control-plane) |
| Use the API | [API_REFERENCE.md](../api/reference) |
| Deploy to production | [DEPLOYMENT.md](../deployment/overview) |
| Troubleshoot issues | [TROUBLESHOOTING.md](../operations/troubleshooting) |
| Prioritize inbox | [EMAIL_PRIORITIZATION.md](../guides/email-prioritization) |
| Unified inbox guide | [INBOX_GUIDE.md](./INBOX_GUIDE) |
| Manage shared inbox | [SHARED_INBOX.md](../guides/shared-inbox) |
| Understand document ingestion | [DOCUMENTS.md](../guides/documents) |
| Analyze codebase health | [CODEBASE_ANALYSIS.md](../analysis/codebase) |
| Track costs | [COST_VISIBILITY.md](../guides/cost-visibility) |
| Automate accounting workflows | [ACCOUNTING.md](../guides/accounting) |
| Automate PR reviews | [GITHUB_PR_REVIEW.md](../api/github-pr-review) |
| Generate tests | [CODING_ASSISTANCE.md](../guides/coding-assistance) |

## Canonical Areas

### Getting Started

- [GETTING_STARTED.md](../getting-started/overview)
- [DEVELOPER_QUICKSTART.md](../getting-started/quickstart)
- [ENVIRONMENT.md](../getting-started/environment)

### Control Plane

- [CONTROL_PLANE.md](../enterprise/control-plane-overview)
- [CONTROL_PLANE_GUIDE.md](../enterprise/control-plane)

### Core Concepts

- [ARCHITECTURE.md](../core-concepts/architecture)
- [DEBATE_INTERNALS.md](../core-concepts/debate-internals)
- [WORKFLOW_ENGINE.md](../core-concepts/workflow-engine)
- [TEMPLATES.md](../guides/templates)
- [KNOWLEDGE_MOUND.md](../core-concepts/knowledge-mound)
- [MEMORY_TIERS.md](../core-concepts/memory)
- [MEMORY.md](../core-concepts/memory-overview)
- [DOCUMENTS.md](../guides/documents)
- [AGENTS.md](../core-concepts/agents)

### Channels & Inbox

- [CHANNELS.md](../guides/channels)
- [BOT_INTEGRATIONS.md](../guides/bot-integrations)
- [EMAIL_PRIORITIZATION.md](../guides/email-prioritization)
- [INBOX_GUIDE.md](./INBOX_GUIDE)
- [SHARED_INBOX.md](../guides/shared-inbox)

### Costs & Billing

- [BILLING.md](../enterprise/billing)
- [COST_VISIBILITY.md](../guides/cost-visibility)
- [ACCOUNTING.md](../guides/accounting)

### Analysis & Code Health

- [CODEBASE_ANALYSIS.md](../analysis/codebase)
- [ANALYSIS.md](../analysis/overview)

### API & SDK

- [API_REFERENCE.md](../api/reference)
- [API_ENDPOINTS.md](../api/endpoints) (auto-generated)
- [API_EXAMPLES.md](../api/examples)
- [API_VERSIONING.md](../api/versioning)
- [WEBSOCKET_EVENTS.md](../guides/websocket-events)
- [SDK_GUIDE.md](../guides/sdk)
- [SDK_TYPESCRIPT.md](../guides/sdk-typescript)

### Operations & Deployment

- [DEPLOYMENT.md](../deployment/overview)
- [OPERATIONS.md](../operations/overview)
- [PRODUCTION_READINESS.md](../operations/production-readiness)
- [OBSERVABILITY.md](../deployment/observability)
- [SCALING.md](../deployment/scaling)
- [deployment/ASYNC_GATEWAY.md](../deployment/async-gateway)
- [deployment/CONTAINER_VOLUMES.md](../deployment/container-volumes)

### Security & Compliance

- [SECURITY.md](../security/overview)
- [AUTH_GUIDE.md](../security/authentication)
- [COMPLIANCE.md](../enterprise/compliance)
- [DATA_CLASSIFICATION.md](../security/data-classification)
- [PRIVACY_POLICY.md](../security/privacy-policy)
- [COMPLIANCE_PRESETS.md](../security/compliance-presets)

### Enterprise & Commercial

- [COMMERCIAL_OVERVIEW.md](../enterprise/commercial-overview)
- [COMMERCIAL_POSITIONING.md](../enterprise/positioning)
- [ENTERPRISE_FEATURES.md](../enterprise/features)
- [ENTERPRISE_SUPPORT.md](../enterprise/support)

### Development & Contributing

- [CONTRIBUTING.md](./guide)
- [FRONTEND_DEVELOPMENT.md](./frontend-development)
- [FRONTEND_ROUTES.md](./frontend-routes)
- [HANDLERS.md](./handlers)
- [HANDLER_DEVELOPMENT.md](./handler-development)
- [TESTING.md](./testing)
- [DEPRECATION_POLICY.md](./deprecation)

### Coding & Review

- [CODING_ASSISTANCE.md](../guides/coding-assistance)
- [GITHUB_PR_REVIEW.md](../api/github-pr-review)

### Advanced / Research

- [NOMIC_LOOP.md](../admin/nomic-loop)
- [GENESIS.md](../advanced/genesis)
- [CROSS_POLLINATION.md](../advanced/cross-pollination)
- [FORMAL_VERIFICATION.md](../advanced/formal-verification)
- [TRICKSTER.md](../advanced/trickster)

### Architecture Decision Records

- [ADR/README.md](../analysis/adr)

### Case Studies

- [case-studies/README.md](../analysis/case-studies)

## Deprecated & Historical

Deprecated documents are stored under `docs/deprecated/` with notes on
replacement docs. See [deprecated/README.md](../analysis/adr).

## Inventory & Maintenance

- Markdown files under `docs/`: 266 (includes deprecated)
- Sync to docs-site: `node docs-site/scripts/sync-docs.js`
- API endpoint list: `python scripts/generate_api_docs.py --output docs/API_ENDPOINTS.md`
- OpenAPI export: `python scripts/export_openapi.py --output-dir docs/api`

Last updated: 2026-01-23
