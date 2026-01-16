Discovering endpoints...
19:14:50 DEBUG    aragora.utils.cache_registry: Registered LRU cache: _compute_domain_from_task
19:14:50 DEBUG    aragora.services.registry: ServiceRegistry initialized
19:14:50 DEBUG    aragora.services.registry: Registered factory: RateLimiterRegistry
19:14:50 DEBUG    aragora.services.registry: Lazily initialized service: RateLimiterRegistry
19:14:50 WARNING  aragora.server.handlers.oauth: [OAuth] GOOGLE_OAUTH_REDIRECT_URI not set, using localhost fallback. This will fail in production.
19:14:50 WARNING  aragora.server.handlers.oauth: [OAuth] OAUTH_SUCCESS_URL not set, using localhost fallback. This will fail in production.
19:14:50 WARNING  aragora.server.handlers.oauth: [OAuth] OAUTH_ERROR_URL not set, using localhost fallback. This will fail in production.
19:14:50 DEBUG    aragora.server.handlers.oauth: [OAuth] Using localhost for allowed redirect hosts (dev mode)
19:14:50 INFO     aragora.server.oauth_state_store: OAuth state store initialized: in-memory
19:14:50 DEBUG    aragora.server.handlers.social.social_media: [Social] Using localhost for OAuth hosts (dev mode)
19:14:50 DEBUG    aragora.server.cors_config: [CORS] Using default origins (dev mode). Set ARAGORA_ALLOWED_ORIGINS to customize.
19:14:50 DEBUG    aragora.server.cors_config: [CORS] Allowed origins: {'http://127.0.0.1:8080', 'http://localhost:8080', 'https://live.aragora.ai', 'https://api.aragora.ai', 'https://www.aragora.ai', 'http://localhost:3000', 'http://127.0.0.1:3000', 'https://aragora.ai'}
19:14:50 WARNING  aragora.server.auth: Authentication disabled (no API token). Set ARAGORA_API_TOKEN for access control.
Found 128 endpoints in 26 groups
# Aragora API Documentation

This document describes the HTTP API endpoints provided by the Aragora server.

## Table of Contents

- [Analytics](#analytics)
- [Auditing](#auditing)
- [Belief](#belief)
- [Breakpoints](#breakpoints)
- [Checkpoints](#checkpoints)
- [Consensus](#consensus)
- [Critique](#critique)
- [Docs](#docs)
- [Gallery](#gallery)
- [Gauntlet](#gauntlet)
- [Genesis](#genesis)
- [Introspection](#introspection)
- [Laboratory](#laboratory)
- [Metrics](#metrics)
- [Moments](#moments)
- [Nomic](#nomic)
- [Oauth](#oauth)
- [Organizations](#organizations)
- [Persona](#persona)
- [Privacy](#privacy)
- [Replays](#replays)
- [Reviews](#reviews)
- [Selection](#selection)
- [Tournament](#tournament)
- [Training](#training)
- [Webhook](#webhook)

---

## Analytics

Analytics and metrics endpoint handlers.

### `GET` `/api/analytics/disagreements`

Get disagreement statistics

### `GET` `/api/analytics/role-rotation`

Get role rotation statistics

### `GET` `/api/analytics/early-stops`

Get early stopping statistics

### `GET` `/api/ranking/stats`

Get ranking statistics

### `GET` `/api/memory/stats`

Get memory statistics

---

## Auditing

Auditing and security analysis endpoint handlers.

### `POST` `/api/debates/capability-probe`

Run capability probes on an agent

### `POST` `/api/debates/deep-audit`

Run deep audit on a task

### `POST` `/api/debates/:id/red-team`

Run red team analysis on a debate

---

## Belief

Belief Network and Reasoning endpoint handlers.

### `GET` `/api/belief-network/:debate_id/cruxes`

Get key claims that impact debate outcome

### `GET` `/api/belief-network/:debate_id/load-bearing-claims`

Get high-centrality claims

### `GET` `/api/provenance/:debate_id/claims/:claim_id/support`

Get claim verification status

### `GET` `/api/laboratory/emergent-traits`

Get emergent traits from agent performance

### `GET` `/api/debate/:debate_id/graph-stats`

Get argument graph statistics

---

## Breakpoints

Breakpoints endpoint handlers for human-in-the-loop intervention.

### `GET` `/api/breakpoints/pending`

List pending breakpoints awaiting resolution

### `POST` `/api/breakpoints/{id}/resolve`

Resolve a pending breakpoint

### `GET` `/api/breakpoints/{id}/status`

Get status of a specific breakpoint

---

## Checkpoints

Checkpoint management endpoint handlers.

### `GET` `/api/checkpoints`

List all checkpoints

### `GET` `/api/checkpoints/{id}`

Get checkpoint details

### `POST` `/api/checkpoints/{id}/resume`

Resume debate from checkpoint

### `DELETE` `/api/checkpoints/{id}`

Delete checkpoint

### `GET` `/api/debates/{id}/checkpoints`

List checkpoints for a debate

### `POST` `/api/debates/{id}/checkpoint`

Create checkpoint for running debate

### `POST` `/api/debates/{id}/pause`

Pause debate and create checkpoint

---

## Consensus

Consensus Memory endpoint handlers.

### `GET` `/api/consensus/similar`

Find debates similar to a topic

### `GET` `/api/consensus/settled`

Get high-confidence settled topics

### `GET` `/api/consensus/stats`

Get consensus memory statistics

### `GET` `/api/consensus/dissents`

Get recent dissenting views

### `GET` `/api/consensus/contrarian-views`

Get contrarian perspectives

### `GET` `/api/consensus/risk-warnings`

Get risk warnings and edge cases

### `GET` `/api/consensus/domain/:domain`

Get domain-specific history

---

## Critique

Critique pattern and reputation endpoint handlers.

### `GET` `/api/critiques/patterns`

Get high-impact critique patterns

### `GET` `/api/critiques/archive`

Get archive statistics

### `GET` `/api/reputation/all`

Get all agent reputations

### `GET` `/api/agent/:name/reputation`

Get specific agent reputation

---

## Docs

API documentation endpoint handlers.

### `GET` `/api/openapi`

OpenAPI 3.0 JSON specification

### `GET` `/api/openapi.json`

OpenAPI 3.0 JSON specification

### `GET` `/api/openapi.yaml`

OpenAPI 3.0 YAML specification

### `GET` `/api/postman.json`

Postman collection export

### `GET` `/api/docs`

Swagger UI interactive documentation

### `GET` `/api/redoc`

ReDoc API documentation viewer

---

## Gallery

Public Gallery endpoint handlers.

### `GET` `/api/gallery`

List public debates

### `GET` `/api/gallery/:debate_id`

Get specific debate with full history

### `GET` `/api/gallery/:debate_id/embed`

Get embeddable debate summary

---

## Gauntlet

Gauntlet endpoint handlers for adversarial stress-testing.

### `POST` `/api/gauntlet/run` 🔒

Start a gauntlet stress-test

### `GET` `/api/gauntlet/{id}` 🔒

Get gauntlet status/results

### `GET` `/api/gauntlet/{id}/receipt` 🔒

Get decision receipt

### `GET` `/api/gauntlet/{id}/heatmap` 🔒

Get risk heatmap

### `GET` `/api/gauntlet/personas` 🔒

List available personas

### `GET` `/api/gauntlet/results` 🔒

List recent results with pagination

### `GET` `/api/gauntlet/{id}/compare/{id2}` 🔒

Compare two gauntlet runs

---

## Genesis

Genesis (evolution visibility) endpoint handlers.

### `GET` `/api/genesis/stats`

Get overall genesis statistics

### `GET` `/api/genesis/events`

Get recent genesis events

### `GET` `/api/genesis/lineage/:genome_id`

Get genome ancestry

### `GET` `/api/genesis/tree/:debate_id`

Get debate tree structure

### `GET` `/api/genesis/genomes`

List all genomes

### `GET` `/api/genesis/genomes/top`

Get top genomes by fitness

### `GET` `/api/genesis/genomes/:genome_id`

Get single genome details

---

## Introspection

Introspection endpoint handlers.

### `GET` `/api/introspection/all`

Get introspection for all agents

### `GET` `/api/introspection/leaderboard`

Get agents ranked by reputation

### `GET` `/api/introspection/agents`

List available agents

### `GET` `/api/introspection/agents/{name}`

Get introspection for specific agent

---

## Laboratory

Persona laboratory endpoint handlers.

### `GET` `/api/laboratory/emergent-traits`

Get emergent traits from agent performance

### `POST` `/api/laboratory/cross-pollinations/suggest`

Suggest beneficial trait transfers

---

## Metrics

Operational metrics endpoint handlers.

### `GET` `/api/metrics`

Get operational metrics for monitoring

### `GET` `/api/metrics/health`

Detailed health check

### `GET` `/api/metrics/cache`

Cache statistics

### `GET` `/api/metrics/verification`

Z3 formal verification statistics

### `GET` `/api/metrics/system`

System information

### `GET` `/metrics`

Prometheus-format metrics (OpenMetrics)

---

## Moments

Moments endpoint handlers.

### `GET` `/api/moments/summary`

Global moments overview

### `GET` `/api/moments/timeline`

Chronological moments (limit, offset)

### `GET` `/api/moments/by-type/{type}`

Filter moments by type

### `GET` `/api/moments/trending`

Most significant recent moments

---

## Nomic

Nomic loop state and monitoring endpoint handlers.

### `GET` `/api/nomic/state`

Get nomic loop state

### `GET` `/api/nomic/health`

Get nomic loop health with stall detection

### `GET` `/api/nomic/metrics`

Get nomic loop Prometheus metrics summary

### `GET` `/api/nomic/log`

Get nomic loop logs

### `GET` `/api/nomic/risk-register`

Get risk register entries

### `GET` `/api/modes`

Get available operational modes

---

## Oauth

OAuth Authentication Handlers.

### `GET` `/api/auth/oauth/google`

Redirect to Google OAuth consent screen

### `GET` `/api/auth/oauth/google/callback`

Handle OAuth callback

### `POST` `/api/auth/oauth/link`

Link OAuth account to existing user

### `DELETE` `/api/auth/oauth/unlink`

Unlink OAuth provider from account

---

## Organizations

Organization Management Handlers.

### `GET` `/api/org/{org_id}`

Get organization details

### `PUT` `/api/org/{org_id}`

Update organization settings

### `GET` `/api/org/{org_id}/members`

List organization members

### `POST` `/api/org/{org_id}/invite`

Invite user to organization

### `GET` `/api/org/{org_id}/invitations`

List pending invitations

### `DELETE` `/api/org/{org_id}/invitations/{invitation_id}`

Revoke invitation

### `DELETE` `/api/org/{org_id}/members/{user_id}`

Remove member

### `PUT` `/api/org/{org_id}/members/{user_id}/role`

Update member role

### `GET` `/api/invitations/pending`

List pending invitations for current user

### `POST` `/api/invitations/{token}/accept`

Accept an invitation

---

## Persona

Persona-related endpoint handlers.

### `GET` `/api/personas`

Get all agent personas

### `GET` `/api/agent/{name}/persona`

Get agent persona

### `GET` `/api/agent/{name}/grounded-persona`

Get truth-grounded persona

### `GET` `/api/agent/{name}/identity-prompt`

Get identity prompt

### `GET` `/api/agent/{name}/performance`

Get agent performance summary

### `GET` `/api/agent/{name}/domains`

Get agent expertise domains

### `GET` `/api/agent/{name}/accuracy`

Get position accuracy stats

---

## Privacy

Privacy Handler - GDPR/CCPA Compliant Data Export and Account Deletion.

### `GET` `/api/privacy/export`

Export all user data (GDPR Article 15, CCPA Right to Know)

### `GET` `/api/privacy/data-inventory`

Get summary of data categories collected

### `DELETE` `/api/privacy/account`

Delete user account (GDPR Article 17, CCPA Right to Delete)

### `POST` `/api/privacy/preferences`

Update privacy preferences (CCPA Do Not Sell)

---

## Replays

Replays and learning evolution endpoint handlers.

### `GET` `/api/replays`

List available replays

### `GET` `/api/replays/:replay_id`

Get specific replay with events

### `GET` `/api/learning/evolution`

Get meta-learning patterns

### `GET` `/api/meta-learning/stats`

Get meta-learning hyperparameters and efficiency stats

---

## Reviews

Reviews Handler - Serve shareable code reviews.

### `GET` `/api/reviews/{id}`

Get a specific review by ID

### `GET` `/api/reviews`

List recent reviews

---

## Selection

Handler for selection plugin endpoints.

### `GET` `/api/selection/plugins` 🔒

List all available selection plugins

### `GET` `/api/selection/defaults` 🔒

Get default plugin configuration

### `GET` `/api/selection/score` 🔒

Get information about a specific scorer

### `GET` `/api/selection/team` 🔒

Get information about a specific team selector

---

## Tournament

Handler for tournament-related endpoints.

### `GET` `/api/tournaments` 🔒

List all available tournaments

### `GET` `/api/tournaments/*`

GET /api/tournaments/*

### `GET` `/api/tournaments/*/standings` 🔒

Get current tournament standings

### `GET` `/api/tournaments/*/bracket` 🔒

Get tournament bracket structure

### `GET` `/api/tournaments/*/matches` 🔒

Get tournament match history

### `GET` `/api/tournaments/*/advance` 🔒

Advance tournament to next round (for elimination brackets)

### `GET` `/api/tournaments/*/matches/*/result` 🔒

Record a match result

---

## Training

Handler for training data export endpoints.

### `GET` `/api/training/export/sft`

Get or create SFT exporter

### `GET` `/api/training/export/dpo`

Get or create DPO exporter

### `GET` `/api/training/export/gauntlet`

Get or create Gauntlet exporter

### `GET` `/api/training/stats`

GET /api/training/stats

### `GET` `/api/training/formats`

GET /api/training/formats

---

## Webhook

Handler for webhook management API endpoints.

### `GET` `/api/webhooks`

Handle DELETE /api/webhooks/:id - delete webhook

### `GET` `/api/webhooks/events`

Handle GET /api/webhooks/events - list available event types

---

## Authentication

Endpoints marked with 🔒 require authentication.

Include an `Authorization` header with your API token:

```
Authorization: Bearer <your-api-token>
```

Set `ARAGORA_API_TOKEN` environment variable to configure the token.

---

*Generated automatically by `scripts/generate_api_docs.py`*
