# Aragora API Documentation

This document describes the HTTP API endpoints provided by the Aragora server.

## Table of Contents

- [Agents](#agents)
- [Analytics](#analytics)
- [Audio](#audio)
- [Auditing](#auditing)
- [Auth](#auth)
- [Belief](#belief)
- [Billing](#billing)
- [Breakpoints](#breakpoints)
- [Broadcast](#broadcast)
- [Calibration](#calibration)
- [Consensus](#consensus)
- [Critique](#critique)
- [Dashboard](#dashboard)
- [Debates](#debates)
- [Documents](#documents)
- [Evolution](#evolution)
- [Evolution Ab Testing](#evolution-ab-testing)
- [Features](#features)
- [Formal Verification](#formal-verification)
- [Gallery](#gallery)
- [Gauntlet](#gauntlet)
- [Genesis](#genesis)
- [Graph Debates](#graph-debates)
- [Insights](#insights)
- [Introspection](#introspection)
- [Laboratory](#laboratory)
- [LeaderboardView](#leaderboardview)
- [Learning](#learning)
- [Matrix Debates](#matrix-debates)
- [Memory](#memory)
- [Memory Analytics](#memory-analytics)
- [Metrics](#metrics)
- [Moments](#moments)
- [Oauth](#oauth)
- [Organizations](#organizations)
- [Persona](#persona)
- [Plugins](#plugins)
- [Probes](#probes)
- [Pulse](#pulse)
- [Relationship](#relationship)
- [Replays](#replays)
- [Reviews](#reviews)
- [Sharing](#sharing)
- [Slack](#slack)
- [Social](#social)
- [System](#system)
- [Tournament](#tournament)
- [Training](#training)
- [Verification](#verification)

---

## Agents

Agent-related endpoint handlers.

### `GET` `/api/leaderboard`

Get agent rankings

### `GET` `/api/rankings`

Get agent rankings (alias)

### `GET` `/api/agents/local`

List detected local LLM servers

### `GET` `/api/agents/local/status`

Get local LLM availability status

### `GET` `/api/agent/{name}/profile`

Get agent profile

### `GET` `/api/agent/{name}/history`

Get agent match history

### `GET` `/api/agent/{name}/calibration`

Get calibration scores

### `GET` `/api/agent/{name}/consistency`

Get consistency score

### `GET` `/api/agent/{name}/flips`

Get agent flip history

### `GET` `/api/agent/{name}/network`

Get relationship network

### `GET` `/api/agent/{name}/rivals`

Get top rivals

### `GET` `/api/agent/{name}/allies`

Get top allies

### `GET` `/api/agent/{name}/moments`

Get significant moments

### `GET` `/api/agent/{name}/positions`

Get position history

### `GET` `/api/agent/compare`

Compare multiple agents

### `GET` `/api/agent/{name}/head-to-head/{opponent}`

Get head-to-head stats

### `GET` `/api/flips/recent`

Get recent flips across all agents

### `GET` `/api/flips/summary`

Get flip summary for dashboard

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

## Audio

Audio and Podcast endpoint handlers.

### `GET` `/audio/{id}.mp3`

Serve audio file

### `GET` `/api/podcast/feed.xml`

iTunes-compatible RSS feed

### `GET` `/api/podcast/episodes`

JSON episode listing

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

## Auth

User Authentication Handlers.

### `POST` `/api/auth/register`

Create a new user account

### `POST` `/api/auth/login`

Authenticate and get tokens

### `POST` `/api/auth/logout`

Invalidate current token (adds to blacklist)

### `POST` `/api/auth/logout-all`

Invalidate all tokens for user (logout all devices)

### `POST` `/api/auth/refresh`

Refresh access token (revokes old refresh token)

### `POST` `/api/auth/revoke`

Explicitly revoke a specific token

### `GET` `/api/auth/me`

Get current user information

### `PUT` `/api/auth/me`

Update current user information

### `POST` `/api/auth/password`

Change password

### `POST` `/api/auth/api-key`

Generate API key

### `DELETE` `/api/auth/api-key`

Revoke API key

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

## Billing

Billing API Handlers.

### `GET` `/api/billing/plans`

List available subscription plans

### `GET` `/api/billing/usage`

Get current usage for authenticated user

### `GET` `/api/billing/subscription`

Get current subscription

### `POST` `/api/billing/checkout`

Create checkout session for subscription

### `POST` `/api/billing/portal`

Create billing portal session

### `POST` `/api/billing/cancel`

Cancel subscription

### `POST` `/api/billing/resume`

Resume canceled subscription

### `POST` `/api/webhooks/stripe`

Handle Stripe webhooks

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

## Broadcast

Broadcast generation handler.

### `POST` `/api/debates/{id}/broadcast`

Generate podcast audio from debate trace

### `POST` `/api/debates/{id}/broadcast/full`

Run full broadcast pipeline

### `GET` `/api/podcast/feed.xml`

Get RSS podcast feed

---

## Calibration

Calibration endpoint handlers.

### `GET` `/api/agent/{name}/calibration-curve`

Get calibration curve (confidence vs accuracy)

### `GET` `/api/agent/{name}/calibration-summary`

Get calibration summary metrics

### `GET` `/api/calibration/leaderboard`

Get top agents by calibration score

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

## Dashboard

Handler for dashboard endpoint.

### `GET` `/api/dashboard/debates` 🔒

Get consolidated debate metrics for dashboard

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `domain` | string | Optional domain filter |
| `limit` | string | Max items per list section |
| `hours` | string | Time window for recent activity |

### `GET` `/api/dashboard/quality-metrics`

GET /api/dashboard/quality-metrics

---

## Debates

Debate-related endpoint handlers.

### `GET` `/api/debates` 🔒

List all debates

### `GET` `/api/debates/{slug}` 🔒

Get debate by slug

### `GET` `/api/debates/slug/{slug}` 🔒

Get debate by slug (alternative)

### `GET` `/api/debates/{id}/export/{format}` 🔒

Export debate

### `GET` `/api/debates/{id}/impasse` 🔒

Detect debate impasse

### `GET` `/api/debates/{id}/convergence` 🔒

Get convergence status

### `GET` `/api/debates/{id}/citations` 🔒

Get evidence citations for debate

### `GET` `/api/debates/{id}/evidence` 🔒

Get comprehensive evidence trail

### `GET` `/api/debate/{id}/meta-critique`

Get meta-level debate analysis

### `GET` `/api/debate/{id}/graph/stats`

Get argument graph statistics

### `POST` `/api/debates/{id}/fork` 🔒

Fork debate at a branch point

### `PATCH` `/api/debates/{id}` 🔒

Update debate metadata (title, tags, status)

### `GET` `/api/search`

Cross-debate search by query

---

## Documents

Document management endpoint handlers.

### `GET` `/api/documents`

List all uploaded documents

### `GET` `/api/documents/formats`

Get supported file formats

### `GET` `/api/documents/{doc_id}`

Get a document by ID

### `POST` `/api/documents/upload`

Upload a document

### `DELETE` `/api/documents/{doc_id}`

Delete a document by ID

---

## Evolution

Prompt evolution endpoint handlers.

### `GET` `/api/evolution/patterns`

Get top patterns across all agents

### `GET` `/api/evolution/summary`

Get evolution summary statistics

### `GET` `/api/evolution/{agent}/history`

Get prompt evolution history for an agent

### `GET` `/api/evolution/{agent}/prompt`

Get current/specific prompt version for an agent

---

## Evolution Ab Testing

Evolution A/B testing endpoint handlers.

### `GET` `/api/evolution/ab-tests` 🔒

List all A/B tests

### `GET` `/api/evolution/ab-tests/{agent}/active` 🔒

Get active test for agent

### `POST` `/api/evolution/ab-tests` 🔒

Start new A/B test

### `GET` `/api/evolution/ab-tests/{id}` 🔒

Get specific test

### `POST` `/api/evolution/ab-tests/{id}/record` 🔒

Record debate result

### `POST` `/api/evolution/ab-tests/{id}/conclude` 🔒

Conclude test

### `DELETE` `/api/evolution/ab-tests/{id}` 🔒

Cancel test

---

## Features

Handler for feature availability endpoints.

### `GET` `/api/features`

Get full feature matrix

### `GET` `/api/features/available`

Get list of available features

### `GET` `/api/features/all`

Get full feature matrix

### `GET` `/api/features/handlers`

GET /api/features/handlers

### `GET` `/api/features/config`

Get user's feature configuration

### `GET` `/api/features/{feature_id}`

GET /api/features/{feature_id}

---

## Formal Verification

Formal Verification API Endpoints.

### `POST` `/api/verify/claim`

Verify a single claim

### `POST` `/api/verify/batch`

Batch verification of multiple claims

### `GET` `/api/verify/status`

Get backend availability status

### `POST` `/api/verify/translate`

Translate claim to formal language only

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

## Graph Debates

Graph debates endpoint handlers.

### `POST` `/api/debates/graph` 🔒

Run a graph-structured debate with branching

### `GET` `/api/debates/graph/{id}` 🔒

Get graph debate by ID

### `GET` `/api/debates/graph/{id}/branches` 🔒

Get all branches for a debate

### `GET` `/api/debates/graph/{id}/nodes` 🔒

Get all nodes in debate graph

---

## Insights

Insights-related endpoint handlers.

### `GET` `/api/insights/recent`

Get recent insights from InsightStore

### `POST` `/api/insights/extract-detailed` 🔒

Extract detailed insights from content

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

## LeaderboardView

Handler for consolidated leaderboard view endpoint.

### `GET` `/api/leaderboard-view`

GET /api/leaderboard-view

---

## Learning

Cross-cycle learning analytics endpoint handlers.

### `GET` `/api/learning/cycles`

Get all cycle summaries

### `GET` `/api/learning/patterns`

Get learned patterns across cycles

### `GET` `/api/learning/agent-evolution`

Get agent performance evolution

### `GET` `/api/learning/insights`

Get aggregated insights from cycles

---

## Matrix Debates

Matrix debates endpoint handlers.

### `POST` `/api/debates/matrix` 🔒

Run parallel scenario debates

### `GET` `/api/debates/matrix/{id}` 🔒

Get matrix debate results

### `GET` `/api/debates/matrix/{id}/scenarios` 🔒

Get all scenario results

### `GET` `/api/debates/matrix/{id}/conclusions` 🔒

Get universal/conditional conclusions

---

## Memory

Memory-related endpoint handlers.

### `GET` `/api/memory/continuum/retrieve`

Retrieve memories from continuum

### `POST` `/api/memory/continuum/consolidate`

Trigger memory consolidation

### `POST` `/api/memory/continuum/cleanup`

Cleanup expired memories

### `GET` `/api/memory/tier-stats`

Get tier statistics

### `GET` `/api/memory/archive-stats`

Get archive statistics

### `GET` `/api/memory/pressure`

Get memory pressure and utilization

### `DELETE` `/api/memory/continuum/{id}`

Delete a memory by ID

### `GET` `/api/memory/tiers`

List all memory tiers with detailed stats

### `GET` `/api/memory/search`

Search memories across tiers

### `GET` `/api/memory/critiques`

Browse critique store entries

---

## Memory Analytics

Memory analytics endpoint handlers.

### `GET` `/api/memory/analytics`

Get comprehensive memory tier analytics

### `GET` `/api/memory/analytics/tier/{tier}`

Get stats for specific tier

### `POST` `/api/memory/analytics/snapshot`

Take a manual snapshot

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

## Plugins

Plugins endpoint handlers.

### `GET` `/api/plugins`

List all available plugins

### `GET` `/api/plugins/{name}`

Get details for a specific plugin

### `POST` `/api/plugins/{name}/run`

Run a plugin with provided input

### `GET` `/api/plugins/installed`

List installed plugins for user/org

### `POST` `/api/plugins/{name}/install`

Install a plugin

### `DELETE` `/api/plugins/{name}/install`

Uninstall a plugin

---

## Probes

Capability probing endpoint handlers.

### `POST` `/api/probes/capability`

Run capability probes on an agent to find vulnerabilities

---

## Pulse

Pulse and trending topics endpoint handlers.

### `GET` `/api/pulse/trending`

Get trending topics from multiple sources

### `GET` `/api/pulse/suggest`

Suggest a trending topic for debate

### `GET` `/api/pulse/analytics`

Get analytics on trending topic debate outcomes

### `POST` `/api/pulse/debate-topic`

Start a debate on a trending topic

---

## Relationship

Relationship endpoint handlers.

### `GET` `/api/relationships/summary`

Global relationship overview

### `GET` `/api/relationships/graph`

Full relationship graph for visualizations

### `GET` `/api/relationships/stats`

Relationship system statistics

### `GET` `/api/relationship/{agent_a}/{agent_b}`

Detailed relationship between two agents

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

## Sharing

Handler for debate sharing endpoints.

### `GET` `/api/debates/*/share`

Generate a secure share token

### `GET` `/api/debates/*/share/revoke` 🔒

Revoke all share links for a debate

### `GET` `/api/shared/*`

GET /api/shared/*

---

## Slack

Slack integration endpoint handlers.

### `POST` `/api/integrations/slack/commands`

Handle Slack slash commands

### `POST` `/api/integrations/slack/interactive`

Handle interactive components

### `POST` `/api/integrations/slack/events`

Handle Slack events

---

## Social

Social Media endpoint handlers for Twitter and YouTube.

### `GET` `/api/youtube/auth`

Get YouTube OAuth authorization URL

### `GET` `/api/youtube/callback`

Handle YouTube OAuth callback

### `GET` `/api/youtube/status`

Get YouTube connector status

### `POST` `/api/debates/{id}/publish/twitter`

Publish debate to Twitter/X

### `POST` `/api/debates/{id}/publish/youtube`

Publish debate to YouTube

---

## System

System and utility endpoint handlers.

### `GET` `/healthz`

Kubernetes liveness probe (lightweight)

### `GET` `/readyz`

Kubernetes readiness probe (checks dependencies)

### `GET` `/api/health`

Health check

### `GET` `/api/health/detailed`

Detailed health check with component status

### `GET` `/api/health/deep`

Deep health check with all external dependencies

### `GET` `/api/nomic/state`

Get nomic loop state

### `GET` `/api/nomic/health`

Get nomic loop health with stall detection

### `GET` `/api/nomic/log`

Get nomic loop logs

### `GET` `/api/nomic/risk-register`

Get risk register entries

### `GET` `/api/modes`

Get available operational modes

### `GET` `/api/history/cycles`

Get cycle history

### `GET` `/api/history/events`

Get event history

### `GET` `/api/history/debates`

Get debate history

### `GET` `/api/history/summary`

Get history summary

### `GET` `/api/system/maintenance?task=<task>`

Run database maintenance (status|vacuum|analyze|checkpoint|full)

### `GET` `/api/openapi`

OpenAPI 3.0 JSON specification

### `GET` `/api/openapi.yaml`

OpenAPI 3.0 YAML specification

### `GET` `/api/docs`

Swagger UI interactive documentation

### `GET` `/api/auth/stats`

Get authentication statistics

### `POST` `/api/auth/revoke`

Revoke a token to invalidate it

### `GET` `/api/circuit-breakers`

Circuit breaker metrics for monitoring cascading failures

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

## Verification

Formal verification endpoint handlers.

### `GET` `/api/verification/status`

Get status of formal verification backends

### `POST` `/api/verification/formal-verify`

Verify a claim using Z3 SMT solver

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