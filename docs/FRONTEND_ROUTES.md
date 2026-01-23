# Frontend Routes and Feature Map

This document maps every UI route in `aragora/live` to its primary purpose and related documentation.
It is the single source of truth for the frontend surface area.

## Canonical UI Domains

- `https://aragora.ai` serves the live dashboard.
- `https://www.aragora.ai` serves the marketing landing page.

Local development uses `http://localhost:3000` with `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_WS_URL` from `.env.local`.

## Core Debate Surfaces

- `/` - Live dashboard (streaming debate view, panels, and controls). Landing page when hosted on `www.aragora.ai`.
  - Related docs: [GETTING_STARTED](./GETTING_STARTED.md), [FEATURES](./FEATURES.md), [WEBSOCKET_EVENTS](./WEBSOCKET_EVENTS.md)
- `/debate/[id]` - Live/archived debate viewer with streaming transcript and metadata.
  - Related docs: [DEBATE_PHASES](./DEBATE_PHASES.md), [WEBSOCKET_EVENTS](./WEBSOCKET_EVENTS.md)
- `/debates` - Debate archive list with filtering and sharing links.
  - Related docs: [API_USAGE](./API_USAGE.md), [CLI_REFERENCE](./CLI_REFERENCE.md)
- `/debates/graph` - Graph debate browser for counterfactual branching.
  - Related docs: [GRAPH_DEBATES](./GRAPH_DEBATES.md)
- `/debates/matrix` - Scenario matrix runner and comparison grid.
  - Related docs: [MATRIX_DEBATES](./MATRIX_DEBATES.md)
- `/deliberations` - Control plane deliberations dashboard and live status.
  - Related docs: [CONTROL_PLANE](./CONTROL_PLANE.md), [CONTROL_PLANE_GUIDE](./CONTROL_PLANE_GUIDE.md)
- `/gauntlet/[id]` - Live gauntlet run viewer (adversarial stress-test).
  - Related docs: [GAUNTLET](./GAUNTLET.md), [GAUNTLET_ARCHITECTURE](./GAUNTLET_ARCHITECTURE.md)
- `/replays` - Replay browser to review historical debates and fork at checkpoints.
  - Related docs: [API_USAGE](./API_USAGE.md), [FEATURES](./FEATURES.md)
- `/reviews` - Shareable multi-agent code reviews and consensus summaries.
  - Related docs: [CLI_REFERENCE](./CLI_REFERENCE.md), [README](../README.md)
- `/control-plane` - Control plane dashboard (agents, queue, deliberations).
  - Related docs: [CONTROL_PLANE](./CONTROL_PLANE.md), [CONTROL_PLANE_GUIDE](./CONTROL_PLANE_GUIDE.md)

## Analysis, Insights, and Memory

- `/insights` - Pattern analysis, position flips, and learning metrics dashboard.
  - Related docs: [FEATURES](./FEATURES.md), [MEMORY_ANALYTICS](./MEMORY_ANALYTICS.md)
- `/evidence` - Evidence and dissent explorer for rebuttals, warnings, and audit trails.
  - Related docs: [TRICKSTER](./TRICKSTER.md), [FEATURES](./FEATURES.md)
- `/memory` - Continuum memory explorer (fast/medium/slow/glacial tiers).
  - Related docs: [MEMORY_STRATEGY](./MEMORY_STRATEGY.md), [MEMORY_ANALYTICS](./MEMORY_ANALYTICS.md)
- `/pulse` - Pulse scheduler for automated trending-topic debates.
  - Related docs: [PULSE](./PULSE.md)
- `/repository` - Repository indexing and knowledge graph explorer.
  - Related docs: [KNOWLEDGE_MOUND](./KNOWLEDGE_MOUND.md), [ARCHITECTURE](./ARCHITECTURE.md)
- `/network` - Agent relationship network visualization.
  - Related docs: [FEATURES](./FEATURES.md), [AGENT_SELECTION](./AGENT_SELECTION.md)
- `/tournaments` - Agent tournaments and rankings dashboard.
  - Related docs: [FEATURES](./FEATURES.md), [AGENT_SELECTION](./AGENT_SELECTION.md)
- `/agents` - Agent recommender and leaderboard explorer.
  - Related docs: [AGENT_SELECTION](./AGENT_SELECTION.md)
- `/agent/[name]` - Individual agent profile with stats and comparisons.
  - Related docs: [AGENT_DEVELOPMENT](./AGENT_DEVELOPMENT.md), [AGENT_SELECTION](./AGENT_SELECTION.md)

## Experimentation and Governance

- `/laboratory` - Persona laboratory (emergent traits, pollinations, evolution, patterns).
  - Related docs: [EVOLUTION_PATTERNS](./EVOLUTION_PATTERNS.md), [AGENT_DEVELOPMENT](./AGENT_DEVELOPMENT.md)
- `/evolution` - Evolution dashboard for genetic optimization and breeding.
  - Related docs: [GENESIS](./GENESIS.md), [EVOLUTION_PATTERNS](./EVOLUTION_PATTERNS.md)
- `/ab-testing` - A/B test management for prompt/evolution variants.
  - Related docs: [A_B_TESTING](./A_B_TESTING.md)
- `/breakpoints` - Human-in-the-loop approvals triggered by Trickster breakpoints.
  - Related docs: [TRICKSTER](./TRICKSTER.md), [API_REFERENCE](./API_REFERENCE.md)
- `/verification` - Formal verification workspace (Z3 and Lean4 proofs).
  - Related docs: [FORMAL_VERIFICATION](./FORMAL_VERIFICATION.md)
- `/batch` - Batch debate submission and monitoring.
  - Related docs: [WORKFLOWS](./WORKFLOWS.md), [CLI_REFERENCE](./CLI_REFERENCE.md)
- `/plugins` - Plugin marketplace and integration manager.
  - Related docs: [PLUGIN_GUIDE](./PLUGIN_GUIDE.md), [MCP_INTEGRATION](./MCP_INTEGRATION.md)

## Account, Org, and Billing

- `/auth/login` - Login UI (email/password + OAuth).
  - Related docs: [SECURITY](./SECURITY.md), [SSO_SETUP](./SSO_SETUP.md)
- `/auth/register` - Registration UI.
  - Related docs: [SECURITY](./SECURITY.md)
- `/developer` - Developer portal for API key management and usage stats.
  - Related docs: [SDK_GUIDE](./SDK_GUIDE.md), [SDK_TYPESCRIPT](./SDK_TYPESCRIPT.md)
- `/settings` - User preferences, integrations, and API configuration.
  - Related docs: [ENVIRONMENT](./ENVIRONMENT.md), [INTEGRATIONS](./INTEGRATIONS.md)
- `/inbox` - AI smart inbox with prioritization and Gmail sync.
  - Related docs: [EMAIL_PRIORITIZATION](./EMAIL_PRIORITIZATION.md)
- `/organization` - Organization profile settings (name, tier, limits).
  - Related docs: [BILLING](./BILLING.md), [SSO_SETUP](./SSO_SETUP.md)
- `/organization/members` - Member invites, roles, and removal.
  - Related docs: [BILLING](./BILLING.md), [SSO_SETUP](./SSO_SETUP.md)
- `/pricing` - Plan and pricing page.
  - Related docs: [BILLING](./BILLING.md)
- `/billing` - Subscription management portal.
  - Related docs: [BILLING](./BILLING.md), [STRIPE_SETUP](./STRIPE_SETUP.md)
- `/billing/success` - Post-checkout confirmation.
  - Related docs: [BILLING](./BILLING.md)
- `/api-explorer` - Interactive API explorer for live endpoint testing.
  - Related docs: [API_REFERENCE](./API_REFERENCE.md), [API_ENDPOINTS](./API_ENDPOINTS.md)
- `/admin` - Admin console for system health, rate limits, and diagnostics.
  - Related docs: [OPERATIONS](./OPERATIONS.md), [RUNBOOK](./RUNBOOK.md)

## Informational

- `/about` - Product overview, Nomic loop explanation, and links.
  - Related docs: [ARCHITECTURE](./ARCHITECTURE.md), [NOMIC_LOOP](./NOMIC_LOOP.md)
