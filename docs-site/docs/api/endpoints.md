---
title: Aragora API Documentation
description: Aragora API Documentation
---

# Aragora API Documentation

This document describes the HTTP API endpoints provided by the Aragora server.

## Table of Contents

- [A2A](#a2a)
- [Analytics](#analytics)
- [AnalyticsDashboard](#analyticsdashboard)
- [Auditing](#auditing)
- [Belief](#belief)
- [Breakpoints](#breakpoints)
- [Checkpoints](#checkpoints)
- [Composite](#composite)
- [Consensus](#consensus)
- [Critique](#critique)
- [Cross Pollination](#cross-pollination)
- [Decision](#decision)
- [Docs](#docs)
- [Email](#email)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [ExternalIntegrations](#externalintegrations)
- [Gallery](#gallery)
- [Gauntlet](#gauntlet)
- [Genesis](#genesis)
- [Introspection](#introspection)
- [Laboratory](#laboratory)
- [Metrics](#metrics)
- [Ml](#ml)
- [Moments](#moments)
- [Nomic](#nomic)
- [Oauth](#oauth)
- [Organizations](#organizations)
- [Persona](#persona)
- [Policy](#policy)
- [Privacy](#privacy)
- [Queue](#queue)
- [Replays](#replays)
- [Repository](#repository)
- [Reviews](#reviews)
- [RLMContext](#rlmcontext)
- [Selection](#selection)
- [TemplateMarketplace](#templatemarketplace)
- [Tournament](#tournament)
- [Training](#training)
- [Transcription](#transcription)
- [Uncertainty](#uncertainty)
- [Verticals](#verticals)
- [Webhook](#webhook)
- [Workflow Templates](#workflow-templates)
- [Workflow](#workflow)
- [Workspace](#workspace)

---

## A2A

A2A Protocol HTTP Handler.

### `GET` `/api/a2a/agents`

List all available agents

### `GET` `/api/a2a/agents/:name`

Get agent card by name

### `POST` `/api/a2a/tasks`

Submit a task

### `GET` `/api/a2a/tasks/:id`

Get task status

### `DELETE` `/api/a2a/tasks/:id`

Cancel task

### `POST` `/api/a2a/tasks/:id/stream`

Stream task (WebSocket upgrade)

### `GET` `/api/a2a/.well-known/agent.json`

Discovery endpoint

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

## AnalyticsDashboard

Handler for analytics dashboard endpoints.

### `GET` `/api/analytics/summary` 🔒

Get flip detection summary for dashboard

### `GET` `/api/analytics/trends/findings`

GET /api/analytics/trends/findings

### `GET` `/api/analytics/remediation` 🔒

Get remediation performance metrics

### `GET` `/api/analytics/agents`

GET /api/analytics/agents

### `GET` `/api/analytics/cost` 🔒

Get cost analysis for audits

### `GET` `/api/analytics/compliance` 🔒

Get compliance scorecard for specified frameworks

### `GET` `/api/analytics/heatmap` 🔒

Get risk heatmap data (category x severity)

### `GET` `/api/analytics/tokens`

GET /api/analytics/tokens

### `GET` `/api/analytics/tokens/trends` 🔒

Get finding trends over time

### `GET` `/api/analytics/tokens/providers`

GET /api/analytics/tokens/providers

### `GET` `/api/analytics/flips/summary` 🔒

Get flip detection summary for dashboard

### `GET` `/api/analytics/flips/recent` 🔒

Get recent flip events

### `GET` `/api/analytics/flips/consistency` 🔒

Get agent consistency scores

### `GET` `/api/analytics/flips/trends` 🔒

Get finding trends over time

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

### `POST` `/api/breakpoints/\{id\}/resolve`

Resolve a pending breakpoint

### `GET` `/api/breakpoints/\{id\}/status`

Get status of a specific breakpoint

---

## Checkpoints

Checkpoint management endpoint handlers.

### `GET` `/api/checkpoints`

List all checkpoints

### `GET` `/api/checkpoints/\{id\}`

Get checkpoint details

### `POST` `/api/checkpoints/\{id\}/resume`

Resume debate from checkpoint

### `DELETE` `/api/checkpoints/\{id\}`

Delete checkpoint

### `GET` `/api/debates/\{id\}/checkpoints`

List checkpoints for a debate

### `POST` `/api/debates/\{id\}/checkpoint`

Create checkpoint for running debate

### `POST` `/api/debates/\{id\}/pause`

Pause debate and create checkpoint

---

## Composite

Handler for composite API endpoints that aggregate multiple data sources.

### `GET` `/api/debates/*/full-context`

GET /api/debates/*/full-context

### `GET` `/api/agents/*/reliability`

Calculate overall reliability score (0-1)

### `GET` `/api/debates/*/compression-analysis`

GET /api/debates/*/compression-analysis

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

## Cross Pollination

Cross-Pollination observability endpoint handlers.

### `GET` `/api/cross-pollination/stats`

Get cross-subscriber statistics

### `GET` `/api/cross-pollination/subscribers`

List all subscribers

### `GET` `/api/cross-pollination/bridge`

Arena event bridge status

### `POST` `/api/cross-pollination/reset`

Reset subscriber statistics

---

## Decision

Handler for unified decision-making API endpoints.

### `GET` `/api/decisions`

List recent decisions

### `GET` `/api/decisions/*`

GET /api/decisions/*

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

## Email

HTTP API Handlers for Email Prioritization.

### `POST` `/api/email/prioritize`

Score a single email

### `POST` `/api/email/rank-inbox`

Rank multiple emails

### `POST` `/api/email/feedback`

Record user action for learning

### `GET` `/api/email/context/:email_address`

Get cross-channel context

### `POST` `/api/email/gmail/oauth/url`

Get Gmail OAuth URL

### `POST` `/api/email/gmail/oauth/callback`

Handle OAuth callback

### `GET` `/api/email/inbox`

Fetch and rank inbox

### `GET` `/api/email/config`

Get prioritization config

### `PUT` `/api/email/config`

Update prioritization config

---

## Evaluation

Handler for LLM-as-Judge evaluation endpoints.

### `GET` `/api/evaluate` 🔒

Compare two responses using pairwise evaluation

### `GET` `/api/evaluate/compare` 🔒

Compare two responses using pairwise evaluation

### `GET` `/api/evaluate/dimensions` 🔒

List available evaluation dimensions

### `GET` `/api/evaluate/profiles` 🔒

List available evaluation weight profiles

---

## Explainability

Handler for debate explainability endpoints.

### `GET` `/api/v1/debates/*/explanation`

Build explanation dictionary based on options

### `GET` `/api/v1/debates/*/evidence`

Handle evidence chain request

### `GET` `/api/v1/debates/*/votes/pivots`

Handle vote pivot analysis request

### `GET` `/api/v1/debates/*/counterfactuals`

Handle counterfactual analysis request

### `GET` `/api/v1/debates/*/summary`

Handle human-readable summary request

### `GET` `/api/v1/explain/*`

GET /api/v1/explain/*

### `GET` `/api/v1/explainability/batch` 🔒

Create a new batch explainability job

### `GET` `/api/v1/explainability/batch/*/status`

Get status of a batch job

### `GET` `/api/v1/explainability/batch/*/results`

Get results of a completed batch job

### `GET` `/api/v1/explainability/compare` 🔒

Compare explanations between multiple debates

### `GET` `/api/debates/*/explanation`

Build explanation dictionary based on options

### `GET` `/api/explain/*`

GET /api/explain/*

---

## ExternalIntegrations

Handler for external integration management.

### `GET` `/api/integrations/zapier/apps`

Handle POST /api/integrations/zapier/apps - create Zapier app

### `GET` `/api/integrations/zapier/triggers`

Handle GET /api/integrations/zapier/triggers - list trigger types

### `GET` `/api/integrations/make/connections`

Handle POST /api/integrations/make/connections - create connection

### `GET` `/api/integrations/make/webhooks`

Handle POST /api/integrations/make/webhooks - register webhook

### `GET` `/api/integrations/make/modules`

Handle GET /api/integrations/make/modules - list available modules

### `GET` `/api/integrations/n8n/credentials`

Handle POST /api/integrations/n8n/credentials - create credential

### `GET` `/api/integrations/n8n/webhooks`

Handle POST /api/integrations/n8n/webhooks - register webhook

### `GET` `/api/integrations/n8n/nodes`

Handle GET /api/integrations/n8n/nodes - get node definitions

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

### `GET` `/api/gauntlet/\{id\}` 🔒

Get gauntlet status/results

### `GET` `/api/gauntlet/\{id\}/receipt` 🔒

Get decision receipt

### `GET` `/api/gauntlet/\{id\}/heatmap` 🔒

Get risk heatmap

### `GET` `/api/gauntlet/personas` 🔒

List available personas

### `GET` `/api/gauntlet/results` 🔒

List recent results with pagination

### `GET` `/api/gauntlet/\{id\}/compare/\{id2\}` 🔒

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

### `GET` `/api/introspection/agents/\{name\}`

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

## Ml

ML (Machine Learning) endpoint handlers.

### `POST` `/api/ml/route`

Get ML-based agent routing for a task

### `POST` `/api/ml/score`

Score response quality

### `POST` `/api/ml/score-batch`

Score multiple responses

### `POST` `/api/ml/consensus`

Predict consensus likelihood

### `POST` `/api/ml/export-training`

Export debate data for training

### `GET` `/api/ml/models`

List available ML models/capabilities

### `GET` `/api/ml/stats`

Get ML module statistics

---

## Moments

Moments endpoint handlers.

### `GET` `/api/moments/summary`

Global moments overview

### `GET` `/api/moments/timeline`

Chronological moments (limit, offset)

### `GET` `/api/moments/by-type/\{type\}`

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

### `WS` `/api/nomic/stream`

Real-time WebSocket event stream

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

### `GET` `/api/org/\{org_id\}`

Get organization details

### `PUT` `/api/org/\{org_id\}`

Update organization settings

### `GET` `/api/org/\{org_id\}/members`

List organization members

### `POST` `/api/org/\{org_id\}/invite`

Invite user to organization

### `GET` `/api/org/\{org_id\}/invitations`

List pending invitations

### `DELETE` `/api/org/\{org_id\}/invitations/\{invitation_id\}`

Revoke invitation

### `DELETE` `/api/org/\{org_id\}/members/\{user_id\}`

Remove member

### `PUT` `/api/org/\{org_id\}/members/\{user_id\}/role`

Update member role

### `GET` `/api/invitations/pending`

List pending invitations for current user

### `POST` `/api/invitations/\{token\}/accept`

Accept an invitation

---

## Persona

Persona-related endpoint handlers.

### `GET` `/api/personas`

Get all agent personas

### `GET` `/api/agent/\{name\}/persona`

Get agent persona

### `GET` `/api/agent/\{name\}/grounded-persona`

Get truth-grounded persona

### `GET` `/api/agent/\{name\}/identity-prompt`

Get identity prompt

### `GET` `/api/agent/\{name\}/performance`

Get agent performance summary

### `GET` `/api/agent/\{name\}/domains`

Get agent expertise domains

### `GET` `/api/agent/\{name\}/accuracy`

Get position accuracy stats

---

## Policy

Handler for policy and compliance endpoints.

### `GET` `/api/policies`

List policies with optional filters

### `GET` `/api/policies/*`

GET /api/policies/*

### `GET` `/api/policies/*/toggle`

Toggle a policy's enabled status

### `GET` `/api/policies/*/violations`

Get violations for a specific policy

### `GET` `/api/compliance/violations`

Get violations for a specific policy

### `GET` `/api/compliance/violations/*`

GET /api/compliance/violations/*

### `GET` `/api/compliance/check`

Run compliance check on content

### `GET` `/api/compliance/stats`

Get compliance statistics

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

## Queue

Queue management endpoint handlers.

### `POST` `/api/queue/jobs`

Submit new job

### `GET` `/api/queue/jobs`

List jobs with filters

### `GET` `/api/queue/jobs/:id`

Get job status

### `POST` `/api/queue/jobs/:id/retry`

Retry failed job

### `DELETE` `/api/queue/jobs/:id`

Cancel job

### `GET` `/api/queue/stats`

Queue statistics

### `GET` `/api/queue/workers`

Worker status

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

## Repository

Repository indexing endpoint handlers.

### `POST` `/api/repository/index`

Start full repository index

### `POST` `/api/repository/incremental`

Incremental update

### `GET` `/api/repository/:id/status`

Get indexing status

### `GET` `/api/repository/:id/entities`

List entities with filters

### `GET` `/api/repository/:id/graph`

Get relationship graph

### `DELETE` `/api/repository/:id`

Remove indexed repository

---

## Reviews

Reviews Handler - Serve shareable code reviews.

### `GET` `/api/reviews/\{id\}`

Get a specific review by ID

### `GET` `/api/reviews`

List recent reviews

---

## RLMContext

Handler for RLM context compression and query endpoints.

### `GET` `/api/rlm/stats`

GET /api/rlm/stats

### `GET` `/api/rlm/strategies`

GET /api/rlm/strategies

### `GET` `/api/rlm/compress`

Get or create the hierarchical compressor using factory

### `GET` `/api/rlm/query`

Simple fallback query when full RLM is not available

### `GET` `/api/rlm/contexts`

GET /api/rlm/contexts

### `GET` `/api/rlm/stream`

GET /api/rlm/stream

### `GET` `/api/rlm/stream/modes`

GET /api/rlm/stream/modes

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

## TemplateMarketplace

Handler for template marketplace API endpoints.

### `GET` `/api/marketplace/templates` 🔒

List marketplace templates with search and filters

### `GET` `/api/marketplace/templates/*`

GET /api/marketplace/templates/*

### `GET` `/api/marketplace/featured` 🔒

Get featured marketplace templates

### `GET` `/api/marketplace/trending` 🔒

Get trending marketplace templates

### `GET` `/api/marketplace/categories` 🔒

Get marketplace categories with counts

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

### `GET` `/api/training/jobs`

GET /api/training/jobs

---

## Transcription

Transcription endpoint handlers for speech-to-text and media processing.

### `POST` `/api/transcription/audio`

Transcribe audio file

### `POST` `/api/transcription/video`

Extract and transcribe audio from video

### `POST` `/api/transcription/youtube`

Transcribe YouTube video

---

## Uncertainty

Uncertainty estimation endpoint handlers.

### `POST` `/api/uncertainty/estimate`

Estimate uncertainty for a debate/response

### `GET` `/api/uncertainty/debate/:id`

Get debate uncertainty metrics

### `GET` `/api/uncertainty/agent/:id`

Get agent calibration profile

### `POST` `/api/uncertainty/followups`

Generate follow-up suggestions from cruxes

---

## Verticals

Vertical specialist endpoint handlers.

### `GET` `/api/verticals`

List available verticals

### `GET` `/api/verticals/:id`

Get vertical config

### `GET` `/api/verticals/:id/tools`

Get vertical tools

### `GET` `/api/verticals/:id/compliance`

Get compliance frameworks

### `POST` `/api/verticals/:id/debate`

Create vertical-specific debate

### `POST` `/api/verticals/:id/agent`

Create specialist agent instance

### `GET` `/api/verticals/suggest`

Suggest vertical for a task

---

## Webhook

Handler for webhook management API endpoints.

### `GET` `/api/webhooks`

Handle DELETE /api/webhooks/:id - delete webhook

### `GET` `/api/webhooks/events`

Handle GET /api/webhooks/events - list available event types

### `GET` `/api/webhooks/slo/status`

Handle GET /api/webhooks/slo/status - get SLO webhook status

---

## Workflow Templates

Workflow Templates API Handler.

### `GET` `/api/workflow/templates`

List available templates

### `GET` `/api/workflow/templates/:id`

Get template details

### `GET` `/api/workflow/templates/:id/package`

Get full package

### `POST` `/api/workflow/templates/run`

Execute a template

---

## Workflow

HTTP request handler for workflow API endpoints.

### `GET` `/api/workflows`

Handle POST /api/workflows

### `GET` `/api/workflows/*`

GET /api/workflows/*

### `GET` `/api/workflow-templates`

Handle GET /api/workflow-templates

### `GET` `/api/workflow-approvals`

Handle GET /api/workflow-approvals

### `GET` `/api/workflow-approvals/*`

GET /api/workflow-approvals/*

### `GET` `/api/workflow-executions`

Handle GET /api/workflow-executions

---

## Workspace

Workspace Handler - Enterprise Privacy and Data Isolation APIs.

### `POST` `/api/workspaces`

Create a new workspace

### `GET` `/api/workspaces`

List workspaces

### `GET` `/api/workspaces/\{id\}`

Get workspace details

### `DELETE` `/api/workspaces/\{id\}`

Delete workspace

### `POST` `/api/workspaces/\{id\}/members`

Add member to workspace

### `DELETE` `/api/workspaces/\{id\}/members/\{user_id\}`

Remove member

### `GET` `/api/retention/policies`

List retention policies

### `POST` `/api/retention/policies`

Create retention policy

### `PUT` `/api/retention/policies/\{id\}`

Update retention policy

### `DELETE` `/api/retention/policies/\{id\}`

Delete retention policy

### `POST` `/api/retention/policies/\{id\}/execute`

Execute retention policy

### `GET` `/api/retention/expiring`

Get items expiring soon

### `POST` `/api/classify`

Classify content sensitivity

### `GET` `/api/classify/policy/\{level\}`

Get policy for sensitivity level

### `GET` `/api/audit/entries`

Query audit entries

### `GET` `/api/audit/report`

Generate compliance report

### `GET` `/api/audit/verify`

Verify audit log integrity

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
