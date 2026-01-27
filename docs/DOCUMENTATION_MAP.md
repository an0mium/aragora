# Documentation Map

This map describes how documentation is organized and which files are canonical.
The `docs/` directory is authoritative; `docs-site/` is a published view synced
from these sources.

For quick navigation, see [INDEX.md](INDEX.md). Deprecated and historical
documents are archived in [deprecated/README.md](deprecated/README.md).

## Quick Navigation

| I want to... | Go to |
|--------------|-------|
| Get started | [GETTING_STARTED.md](GETTING_STARTED.md) |
| Understand the architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Run the control plane | [CONTROL_PLANE_GUIDE.md](CONTROL_PLANE_GUIDE.md) |
| Use the API | [API_REFERENCE.md](API_REFERENCE.md) |
| Deploy to production | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Troubleshoot issues | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Prioritize inbox | [EMAIL_PRIORITIZATION.md](EMAIL_PRIORITIZATION.md) |
| Unified inbox guide | [INBOX_GUIDE.md](INBOX_GUIDE.md) |
| Manage shared inbox | [SHARED_INBOX.md](SHARED_INBOX.md) |
| Understand document ingestion | [DOCUMENTS.md](DOCUMENTS.md) |
| Analyze codebase health | [CODEBASE_ANALYSIS.md](CODEBASE_ANALYSIS.md) |
| Track costs | [COST_VISIBILITY.md](COST_VISIBILITY.md) |
| Automate accounting workflows | [ACCOUNTING.md](ACCOUNTING.md) |
| Automate PR reviews | [GITHUB_PR_REVIEW.md](GITHUB_PR_REVIEW.md) |
| Generate tests | [CODING_ASSISTANCE.md](CODING_ASSISTANCE.md) |

## Canonical Areas

### Getting Started

- [GETTING_STARTED.md](GETTING_STARTED.md)
- [DEVELOPER_QUICKSTART.md](DEVELOPER_QUICKSTART.md)
- [ENVIRONMENT.md](ENVIRONMENT.md)

### Control Plane

- [CONTROL_PLANE.md](CONTROL_PLANE.md)
- [CONTROL_PLANE_GUIDE.md](CONTROL_PLANE_GUIDE.md)

### Core Concepts

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [DEBATE_INTERNALS.md](DEBATE_INTERNALS.md)
- [HOOKS.md](HOOKS.md)
- [WORKFLOW_ENGINE.md](WORKFLOW_ENGINE.md)
- [TEMPLATES.md](TEMPLATES.md)
- [KNOWLEDGE_MOUND.md](KNOWLEDGE_MOUND.md)
- [MEMORY_TIERS.md](MEMORY_TIERS.md)
- [MEMORY.md](MEMORY.md)
- [DOCUMENTS.md](DOCUMENTS.md)
- [AGENTS.md](AGENTS.md)
- [SKILLS.md](SKILLS.md)

### Channels & Inbox

- [CHANNELS.md](CHANNELS.md)
- [BOT_INTEGRATIONS.md](BOT_INTEGRATIONS.md)
- [EMAIL_PRIORITIZATION.md](EMAIL_PRIORITIZATION.md)
- [INBOX_GUIDE.md](INBOX_GUIDE.md)
- [SHARED_INBOX.md](SHARED_INBOX.md)

### Costs & Billing

- [BILLING.md](BILLING.md)
- [COST_VISIBILITY.md](COST_VISIBILITY.md)
- [ACCOUNTING.md](ACCOUNTING.md)

### Analysis & Code Health

- [CODEBASE_ANALYSIS.md](CODEBASE_ANALYSIS.md)
- [ANALYSIS.md](ANALYSIS.md)

### API & SDK

- [API_REFERENCE.md](API_REFERENCE.md)
- [ADMIN_API_REFERENCE.md](ADMIN_API_REFERENCE.md)
- [API_ENDPOINTS.md](API_ENDPOINTS.md) (auto-generated)
- [API_EXAMPLES.md](API_EXAMPLES.md)
- [API_VERSIONING.md](API_VERSIONING.md)
- [WEBSOCKET_EVENTS.md](WEBSOCKET_EVENTS.md)
- [SDK_GUIDE.md](SDK_GUIDE.md)
- [SDK_TYPESCRIPT.md](SDK_TYPESCRIPT.md)

### Operations & Deployment

- [DEPLOYMENT.md](DEPLOYMENT.md)
- [OPERATIONS.md](OPERATIONS.md)
- [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md)
- [OBSERVABILITY.md](OBSERVABILITY.md)
- [SCALING.md](SCALING.md)
- [deployment/ASYNC_GATEWAY.md](deployment/ASYNC_GATEWAY.md)
- [deployment/CONTAINER_VOLUMES.md](deployment/CONTAINER_VOLUMES.md)

### Security & Compliance

- [SECURITY.md](SECURITY.md)
- [AUTH_GUIDE.md](AUTH_GUIDE.md)
- [security/rbac-abac-strategy.md](security/rbac-abac-strategy.md)
- [COMPLIANCE.md](COMPLIANCE.md)
- [DATA_CLASSIFICATION.md](DATA_CLASSIFICATION.md)
- [PRIVACY_POLICY.md](PRIVACY_POLICY.md)
- [COMPLIANCE_PRESETS.md](COMPLIANCE_PRESETS.md)

### Enterprise & Commercial

- [COMMERCIAL_OVERVIEW.md](COMMERCIAL_OVERVIEW.md)
- [COMMERCIAL_POSITIONING.md](COMMERCIAL_POSITIONING.md)
- [ENTERPRISE_FEATURES.md](ENTERPRISE_FEATURES.md)
- [ENTERPRISE_SUPPORT.md](ENTERPRISE_SUPPORT.md)

### Development & Contributing

- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [FRONTEND_DEVELOPMENT.md](FRONTEND_DEVELOPMENT.md)
- [FRONTEND_ROUTES.md](FRONTEND_ROUTES.md)
- [HANDLERS.md](HANDLERS.md)
- [HANDLER_DEVELOPMENT.md](HANDLER_DEVELOPMENT.md)
- [TESTING.md](TESTING.md)
- [DEPRECATION_POLICY.md](DEPRECATION_POLICY.md)

### Coding & Review

- [CODING_ASSISTANCE.md](CODING_ASSISTANCE.md)
- [GITHUB_PR_REVIEW.md](GITHUB_PR_REVIEW.md)

### Advanced / Research

- [NOMIC_LOOP.md](NOMIC_LOOP.md)
- [GENESIS.md](GENESIS.md)
- [CROSS_POLLINATION.md](CROSS_POLLINATION.md)
- [FORMAL_VERIFICATION.md](FORMAL_VERIFICATION.md)
- [TRICKSTER.md](TRICKSTER.md)

### Architecture Decision Records

- [ADR/README.md](ADR/README.md)

### Case Studies

- [case-studies/README.md](case-studies/README.md)

## Deprecated & Historical

Deprecated documents are stored under `docs/deprecated/` with notes on
replacement docs. See [deprecated/README.md](deprecated/README.md).

## Inventory & Maintenance

- Markdown files under `docs/`: 266 (includes deprecated)
- Sync to docs-site: `node docs-site/scripts/sync-docs.js`
- API endpoint list: `python scripts/generate_api_docs.py --output docs/API_ENDPOINTS.md`
- OpenAPI export: `python scripts/export_openapi.py --output-dir docs/api`

Last updated: 2026-01-27
