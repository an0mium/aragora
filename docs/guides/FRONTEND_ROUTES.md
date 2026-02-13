# Frontend Routes and Feature Map

This document maps the primary UI routes in `aragora/live` to their purpose and related documentation.
It is the canonical map for the user-facing surface area.

## Canonical UI Domains

- `https://aragora.ai` serves the live dashboard.
- `https://www.aragora.ai` serves the marketing landing page.

Local development uses `http://localhost:3000` with `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_WS_URL` from `.env.local`.

## Core Debate Surfaces

- `/` - Live dashboard (streaming debate view, panels, and controls). Landing page when hosted on `www.aragora.ai`.
  - Related docs: [GETTING_STARTED](./GETTING_STARTED.md), [FEATURES](../status/FEATURES.md), [WEBSOCKET_EVENTS](../streaming/WEBSOCKET_EVENTS.md)
- `/debate/[id]` - Live/archived debate viewer with streaming transcript and metadata.
  - Related docs: [DEBATE_PHASES](../debate/DEBATE_PHASES.md), [WEBSOCKET_EVENTS](../streaming/WEBSOCKET_EVENTS.md)
- `/debates` - Debate archive list with filtering and sharing links.
  - Related docs: [API_USAGE](../api/API_USAGE.md), [CLI_REFERENCE](../reference/CLI_REFERENCE.md)
- `/debates/graph` - Graph debate browser for counterfactual branching.
  - Related docs: [GRAPH_DEBATES](../debate/GRAPH_DEBATES.md)
- `/debates/matrix` - Scenario matrix runner and comparison grid.
  - Related docs: [MATRIX_DEBATES](../debate/MATRIX_DEBATES.md)
- `/deliberations` - Control plane vetted decisionmaking dashboard and live status.
  - Related docs: [CONTROL_PLANE](../reference/CONTROL_PLANE.md), [CONTROL_PLANE_GUIDE](./CONTROL_PLANE_GUIDE.md)
- `/gauntlet/[id]` - Live gauntlet run viewer (adversarial stress-test).
  - Related docs: [GAUNTLET](../debate/GAUNTLET.md), [GAUNTLET_ARCHITECTURE](../architecture/GAUNTLET_ARCHITECTURE.md)
- `/replays` - Replay browser to review historical debates and fork at checkpoints.
  - Related docs: [API_USAGE](../api/API_USAGE.md), [FEATURES](../status/FEATURES.md)
- `/reviews` - Shareable multi-agent code reviews and consensus summaries.
  - Related docs: [CLI_REFERENCE](../reference/CLI_REFERENCE.md), [INDEX](../INDEX.md)
- `/code-review` - Guided code review workspace with review templates.
  - Related docs: [CODING_ASSISTANCE](../architecture/CODING_ASSISTANCE.md), [GITHUB_PR_REVIEW](../integrations/GITHUB_PR_REVIEW.md)
- `/control-plane` - Control plane dashboard (agents, queue, vetted decisionmaking sessions).
  - Related docs: [CONTROL_PLANE](../reference/CONTROL_PLANE.md), [CONTROL_PLANE_GUIDE](./CONTROL_PLANE_GUIDE.md)

## Analysis, Insights, and Memory

- `/insights` - Pattern analysis, position flips, and learning metrics dashboard.
  - Related docs: [FEATURES](../status/FEATURES.md), [MEMORY_ANALYTICS](../knowledge/MEMORY_ANALYTICS.md)
- `/evidence` - Evidence and dissent explorer for rebuttals, warnings, and audit trails.
  - Related docs: [TRICKSTER](../debate/TRICKSTER.md), [FEATURES](../status/FEATURES.md)
- `/memory` - Continuum memory explorer (fast/medium/slow/glacial tiers).
  - Related docs: [MEMORY_STRATEGY](../knowledge/MEMORY_STRATEGY.md), [MEMORY_ANALYTICS](../knowledge/MEMORY_ANALYTICS.md)
- `/pulse` - Pulse scheduler for automated trending-topic debates.
  - Related docs: [PULSE](../resilience/PULSE.md)
- `/repository` - Repository indexing and knowledge graph explorer.
  - Related docs: [KNOWLEDGE_MOUND](../knowledge/KNOWLEDGE_MOUND.md), [ARCHITECTURE](../architecture/ARCHITECTURE.md)
- `/security-scan` - Codebase security scans and dependency intelligence.
  - Related docs: [CODEBASE_ANALYSIS](../architecture/CODEBASE_ANALYSIS.md)
- `/audit` - Audit dashboard for sessions, findings, and exports.
  - Related docs: [API_REFERENCE](../api/API_REFERENCE.md), [SECURITY_RUNTIME](../enterprise/SECURITY_RUNTIME.md)
- `/audit/[id]` - Audit session detail view (findings, events, report export).
  - Related docs: [API_REFERENCE](../api/API_REFERENCE.md)
- `/audit/new` - Launch a new audit session.
  - Related docs: [API_REFERENCE](../api/API_REFERENCE.md)
- `/audit/templates` - Audit templates and presets.
  - Related docs: [GAUNTLET](../debate/GAUNTLET.md), [FEATURES](../status/FEATURES.md)
- `/network` - Agent relationship network visualization.
  - Related docs: [FEATURES](../status/FEATURES.md), [AGENT_SELECTION](../debate/AGENT_SELECTION.md)
- `/tournaments` - Agent tournaments and rankings dashboard.
  - Related docs: [FEATURES](../status/FEATURES.md), [AGENT_SELECTION](../debate/AGENT_SELECTION.md)
- `/agents` - Agent recommender and leaderboard explorer.
  - Related docs: [AGENT_SELECTION](../debate/AGENT_SELECTION.md)
- `/agent/[name]` - Individual agent profile with stats and comparisons.
  - Related docs: [AGENT_DEVELOPMENT](../debate/AGENT_DEVELOPMENT.md), [AGENT_SELECTION](../debate/AGENT_SELECTION.md)

## Experimentation and Governance

- `/laboratory` - Persona laboratory (emergent traits, pollinations, evolution, patterns).
  - Related docs: [EVOLUTION_PATTERNS](../workflow/EVOLUTION_PATTERNS.md), [AGENT_DEVELOPMENT](../debate/AGENT_DEVELOPMENT.md)
- `/evolution` - Evolution dashboard for genetic optimization and breeding.
  - Related docs: [GENESIS](../workflow/GENESIS.md), [EVOLUTION_PATTERNS](../workflow/EVOLUTION_PATTERNS.md)
- `/ab-testing` - A/B test management for prompt/evolution variants.
  - Related docs: [A_B_TESTING](../testing/A_B_TESTING.md)
- `/breakpoints` - Human-in-the-loop approvals triggered by Trickster breakpoints.
  - Related docs: [TRICKSTER](../debate/TRICKSTER.md), [API_REFERENCE](../api/API_REFERENCE.md)
- `/verification` - Formal verification workspace (Z3 and Lean4 proofs).
  - Related docs: [FORMAL_VERIFICATION](../workflow/FORMAL_VERIFICATION.md)
- `/batch` - Batch debate submission and monitoring.
  - Related docs: [WORKFLOWS](../workflow/WORKFLOWS.md), [CLI_REFERENCE](../reference/CLI_REFERENCE.md)
- `/plugins` - Plugin marketplace and integration manager.
  - Related docs: [PLUGIN_GUIDE](./PLUGIN_GUIDE.md), [MCP_INTEGRATION](../integrations/MCP_INTEGRATION.md)

## Account, Org, and Billing

- `/auth/login` - Login UI (email/password + OAuth).
  - Related docs: [SECURITY](../enterprise/SECURITY.md), [SSO_SETUP](../enterprise/SSO_SETUP.md)
- `/auth/register` - Registration UI.
  - Related docs: [SECURITY](../enterprise/SECURITY.md)
- `/developer` - Developer portal for API key management and usage stats.
  - Related docs: [SDK_GUIDE](./SDK_GUIDE.md), [SDK_TYPESCRIPT](./SDK_TYPESCRIPT.md)
- `/settings` - User preferences, integrations, and API configuration.
  - Related docs: [ENVIRONMENT](../reference/ENVIRONMENT.md), [INTEGRATIONS](../integrations/INTEGRATIONS.md)
- `/inbox` - AI smart inbox with prioritization and Gmail/Outlook sync.
  - Related docs: [EMAIL_PRIORITIZATION](../integrations/EMAIL_PRIORITIZATION.md)
- `/command-center` - Unified inbox command center for quick actions.
  - Related docs: [EMAIL_PRIORITIZATION](../integrations/EMAIL_PRIORITIZATION.md)
- `/shared-inbox` - Shared inbox dashboard for team triage and routing rules.
  - Related docs: [SHARED_INBOX](./SHARED_INBOX.md), [EMAIL_PRIORITIZATION](../integrations/EMAIL_PRIORITIZATION.md)
- `/accounting` - QuickBooks accounting dashboard.
  - Related docs: [API_REFERENCE](../api/API_REFERENCE.md), [BILLING](../reference/BILLING.md)
- `/organization` - Organization profile settings (name, tier, limits).
  - Related docs: [BILLING](../reference/BILLING.md), [SSO_SETUP](../enterprise/SSO_SETUP.md)
- `/organization/members` - Member invites, roles, and removal.
  - Related docs: [BILLING](../reference/BILLING.md), [SSO_SETUP](../enterprise/SSO_SETUP.md)
- `/pricing` - Plan and pricing page.
  - Related docs: [BILLING](../reference/BILLING.md)
- `/billing` - Subscription management portal.
  - Related docs: [BILLING](../reference/BILLING.md), [STRIPE_SETUP](./STRIPE_SETUP.md)
- `/billing/success` - Post-checkout confirmation.
  - Related docs: [BILLING](../reference/BILLING.md)
- `/costs` - Cost visibility dashboard (usage, budgets, alerts).
  - Related docs: [COST_VISIBILITY](../observability/COST_VISIBILITY.md), [BILLING](../reference/BILLING.md)
- `/api-explorer` - Interactive API explorer for live endpoint testing.
  - Related docs: [API_REFERENCE](../api/API_REFERENCE.md), [API_ENDPOINTS](../api/API_ENDPOINTS.md)
- `/admin` - Admin console for system health, rate limits, and diagnostics.
  - Related docs: [OPERATIONS](../OPERATIONS.md), [RUNBOOK](../deployment/RUNBOOK.md)

## Informational

- `/about` - Product overview, Nomic loop explanation, and links.
  - Related docs: [ARCHITECTURE](../architecture/ARCHITECTURE.md), [NOMIC_LOOP](../workflow/NOMIC_LOOP.md)
