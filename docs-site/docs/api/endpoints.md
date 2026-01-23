---
title: Aragora API Documentation
description: Aragora API Documentation
---

# Aragora API Documentation

This document describes the HTTP API endpoints provided by the Aragora server.

## Table of Contents

- [A2A](#a2a)
- [Accounting](#accounting)
- [Analytics](#analytics)
- [Analytics Dashboard](#analytics-dashboard)
- [Audit Export](#audit-export)
- [Audit Trail](#audit-trail)
- [Auditing](#auditing)
- [Belief](#belief)
- [Breakpoints](#breakpoints)
- [Checkpoints](#checkpoints)
- [Composite](#composite)
- [Consensus](#consensus)
- [Control Plane](#control-plane)
- [Costs](#costs)
- [Critique](#critique)
- [Cross Pollination](#cross-pollination)
- [Decision](#decision)
- [Deliberations](#deliberations)
- [Dependency Analysis](#dependency-analysis)
- [Docs](#docs)
- [Email](#email)
- [EmailDebate](#emaildebate)
- [Email Services](#email-services)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [External Integrations](#external-integrations)
- [Gallery](#gallery)
- [Gauntlet](#gauntlet)
- [Genesis](#genesis)
- [Inbox Command](#inbox-command)
- [Introspection](#introspection)
- [KnowledgeChat](#knowledgechat)
- [Laboratory](#laboratory)
- [Metrics](#metrics)
- [Ml](#ml)
- [Moments](#moments)
- [Nomic](#nomic)
- [Oauth](#oauth)
- [Orchestration](#orchestration)
- [Organizations](#organizations)
- [Partner](#partner)
- [Persona](#persona)
- [Policy](#policy)
- [Privacy](#privacy)
- [Queue](#queue)
- [Replays](#replays)
- [Repository](#repository)
- [Reviews](#reviews)
- [RLMContext](#rlmcontext)
- [Selection](#selection)
- [Shared Inbox](#shared-inbox)
- [Template Marketplace](#template-marketplace)
- [Threat Intel](#threat-intel)
- [Tournaments](#tournaments)
- [Training](#training)
- [Transcription](#transcription)
- [Uncertainty](#uncertainty)
- [Verticals](#verticals)
- [Webhook](#webhook)
- [Workflow Templates](#workflow-templates)
- [Workflow](#workflow)
- [Workspace](#workspace)
- [Intelligence](#intelligence)
- [Metrics](#metrics)
- [Quick Scan](#quick-scan)
- [Security](#security)
- [Audit Bridge](#audit-bridge)
- [Pr Review](#pr-review)

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

## Accounting

Accounting handlers for QuickBooks Online and Gusto payroll integration.

### `GET` `/api/accounting/status`

QuickBooks status + dashboard data

### `GET` `/api/accounting/connect`

Start QuickBooks OAuth

### `GET` `/api/accounting/callback`

QuickBooks OAuth callback

### `POST` `/api/accounting/disconnect`

Disconnect QuickBooks

### `GET` `/api/accounting/customers`

List QuickBooks customers

### `GET` `/api/accounting/transactions`

List QuickBooks transactions

### `POST` `/api/accounting/report`

Generate accounting report

### `GET` `/api/accounting/gusto/status`

Gusto connection status

### `GET` `/api/accounting/gusto/connect`

Start Gusto OAuth

### `GET` `/api/accounting/gusto/callback`

Gusto OAuth callback

### `POST` `/api/accounting/gusto/disconnect`

Disconnect Gusto

### `GET` `/api/accounting/gusto/employees`

List employees

### `GET` `/api/accounting/gusto/payrolls`

List payroll runs

### `GET` `/api/accounting/gusto/payrolls/\{payroll_id\}`

Payroll run details

### `POST` `/api/accounting/gusto/payrolls/\{payroll_id\}/journal-entry`

Generate journal entry

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

## Analytics Dashboard

Analytics Dashboard endpoint handlers.

### `GET` `/api/analytics/summary`

Dashboard summary

### `GET` `/api/analytics/trends/findings`

Finding trends over time

### `GET` `/api/analytics/remediation`

Remediation metrics

### `GET` `/api/analytics/agents`

Agent performance metrics

### `GET` `/api/analytics/cost`

Cost analysis

### `GET` `/api/analytics/compliance`

Compliance scorecard

### `GET` `/api/analytics/heatmap`

Risk heatmap data

### `GET` `/api/analytics/flips/summary`

Flip detection summary

### `GET` `/api/analytics/flips/recent`

Recent flip events

### `GET` `/api/analytics/flips/consistency`

Agent consistency scores

### `GET` `/api/analytics/flips/trends`

Flip trends over time

---

## Audit Export

Audit Export API Handler.

### `GET` `/api/audit/events`

Query audit events

### `GET` `/api/audit/stats`

Audit log statistics

### `POST` `/api/audit/export`

Export audit log (JSON, CSV, SOC2)

### `POST` `/api/audit/verify`

Verify audit log integrity

---

## Audit Trail

Audit Trail HTTP Handlers for Aragora.

### `GET` `/api/v1/audit-trails`

List recent audit trails

### `GET` `/api/v1/audit-trails/:trail_id`

Get specific audit trail

### `GET` `/api/v1/audit-trails/:trail_id/export`

Export (format=json|csv|md)

### `POST` `/api/v1/audit-trails/:trail_id/verify`

Verify integrity checksum

### `GET` `/api/v1/receipts`

List recent decision receipts

### `GET` `/api/v1/receipts/:receipt_id`

Get specific receipt

### `POST` `/api/v1/receipts/:receipt_id/verify`

Verify receipt integrity

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

### `GET` `/api/v1/debates/*/full-context`

GET /api/v1/debates/*/full-context

### `GET` `/api/v1/agents/*/reliability`

Calculate overall reliability score (0-1)

### `GET` `/api/v1/debates/*/compression-analysis`

GET /api/v1/debates/*/compression-analysis

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

## Control Plane

Control Plane HTTP Handlers for Aragora.

### `GET` `/api/control-plane/agents`

List registered agents (also /api/v1/control-plane/agents)

### `POST` `/api/control-plane/agents`

Register an agent (also /api/v1/control-plane/agents)

### `GET` `/api/control-plane/agents/:id`

Get agent info (also /api/v1/control-plane/agents/:id)

### `DELETE` `/api/control-plane/agents/:id`

Unregister agent (also /api/v1/control-plane/agents/:id)

### `POST` `/api/control-plane/agents/:id/heartbeat`

Send heartbeat

---

## Costs

Cost Visibility API Handler.

### `GET` `/api/costs`

Get cost dashboard data

### `GET` `/api/costs/breakdown`

Get detailed cost breakdown

### `GET` `/api/costs/timeline`

Get usage timeline

### `GET` `/api/costs/alerts`

Get budget alerts

### `POST` `/api/costs/budget`

Set budget limits

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

### `GET` `/api/v1/decisions`

List recent decisions

### `GET` `/api/v1/decisions/*`

GET /api/v1/decisions/*

---

## Deliberations

Handler for deliberation dashboard endpoints.

### `GET` `/api/v1/deliberations/active`

Fetch active deliberations from the debate store

### `GET` `/api/v1/deliberations/stats`

Get deliberation statistics

### `GET` `/api/v1/deliberations/stream`

Handle WebSocket stream for real-time updates

### `GET` `/api/v1/deliberations/\{deliberation_id\}`

GET /api/v1/deliberations/\{deliberation_id\}

---

## Dependency Analysis

HTTP API Handlers for Dependency Analysis.

### `POST` `/api/v1/codebase/analyze-dependencies`

Analyze project dependencies

### `GET` `/api/v1/codebase/sbom`

Generate SBOM

### `POST` `/api/v1/codebase/scan-vulnerabilities`

Scan for CVEs

### `POST` `/api/v1/codebase/check-licenses`

Check license compatibility

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

## EmailDebate

Handler for email deliberation API endpoints.

### `GET` `/api/v1/email/prioritize`

Prioritize multiple emails

### `GET` `/api/v1/email/prioritize/batch`

Prioritize multiple emails

### `GET` `/api/v1/email/triage`

Full inbox triage with categorization and sorting

---

## Email Services

HTTP API Handlers for Email Services.

### `POST` `/api/v1/email/followups/mark`

Mark email as awaiting reply

### `GET` `/api/v1/email/followups/pending`

List pending follow-ups

### `POST` `/api/v1/email/followups/\{id\}/resolve`

Resolve a follow-up

### `POST` `/api/v1/email/followups/check-replies`

Check for replies

### `GET` `/api/v1/email/\{id\}/snooze-suggestions`

Get snooze recommendations

### `POST` `/api/v1/email/\{id\}/snooze`

Apply snooze to email

### `DELETE` `/api/v1/email/\{id\}/snooze`

Cancel snooze

### `GET` `/api/v1/email/snoozed`

List snoozed emails

### `GET` `/api/v1/email/categories`

List available categories

### `POST` `/api/v1/email/categories/learn`

Submit category feedback

---

## Evaluation

Handler for LLM-as-Judge evaluation endpoints.

### `GET` `/api/v1/evaluate` 🔒

Evaluate a response using LLM-as-Judge

### `GET` `/api/v1/evaluate/compare` 🔒

Compare two responses using pairwise evaluation

### `GET` `/api/v1/evaluate/dimensions` 🔒

List available evaluation dimensions

### `GET` `/api/v1/evaluate/profiles` 🔒

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

### `GET` `/api/v1/debates/*/explanation`

Build explanation dictionary based on options

### `GET` `/api/v1/explain/*`

GET /api/v1/explain/*

---

## External Integrations

External Integrations API Handler.

### `POST` `/api/integrations/zapier/apps`

Create Zapier app

### `GET` `/api/integrations/zapier/apps`

List Zapier apps

### `DELETE` `/api/integrations/zapier/apps/:id`

Delete Zapier app

### `POST` `/api/integrations/zapier/triggers`

Subscribe to trigger

### `DELETE` `/api/integrations/zapier/triggers/:id`

Unsubscribe trigger

### `GET` `/api/integrations/zapier/triggers`

List trigger types

### `POST` `/api/integrations/make/connections`

Create Make connection

### `GET` `/api/integrations/make/connections`

List Make connections

### `DELETE` `/api/integrations/make/connections/:id`

Delete Make connection

### `POST` `/api/integrations/make/webhooks`

Register webhook

### `DELETE` `/api/integrations/make/webhooks/:id`

Unregister webhook

### `GET` `/api/integrations/make/modules`

List available modules

### `POST` `/api/integrations/n8n/credentials`

Create n8n credential

### `GET` `/api/integrations/n8n/credentials`

List n8n credentials

### `DELETE` `/api/integrations/n8n/credentials/:id`

Delete n8n credential

### `POST` `/api/integrations/n8n/webhooks`

Register webhook

### `DELETE` `/api/integrations/n8n/webhooks/:id`

Unregister webhook

### `GET` `/api/integrations/n8n/nodes`

Get node definitions

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

### `POST` `/api/gauntlet/run`

Start a gauntlet stress-test

### `GET` `/api/gauntlet/\{id\}`

Get gauntlet status/results

### `GET` `/api/gauntlet/\{id\}/receipt`

Get decision receipt

### `GET` `/api/gauntlet/\{id\}/heatmap`

Get risk heatmap

### `GET` `/api/gauntlet/personas`

List available personas

### `GET` `/api/gauntlet/results`

List recent results with pagination

### `GET` `/api/gauntlet/\{id\}/compare/\{id2\}`

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

## Inbox Command

Inbox Command Center API Handler.

### `GET` `/api/inbox/command`

Fetch prioritized inbox

### `POST` `/api/inbox/actions`

Execute quick action

### `POST` `/api/inbox/bulk-actions`

Execute bulk action

### `GET` `/api/inbox/sender-profile`

Get sender profile

### `GET` `/api/inbox/daily-digest`

Get daily digest

### `POST` `/api/inbox/reprioritize`

Trigger AI re-prioritization

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

## KnowledgeChat

HTTP handler for Knowledge + Chat bridge endpoints.

### `GET` `/api/v1/chat/knowledge/search`

GET /api/v1/chat/knowledge/search

### `GET` `/api/v1/chat/knowledge/inject`

GET /api/v1/chat/knowledge/inject

### `GET` `/api/v1/chat/knowledge/store`

GET /api/v1/chat/knowledge/store

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

## Orchestration

Unified Orchestration Handler for Aragora Control Plane.

### `POST` `/api/v1/orchestration/deliberate`

Unified deliberation endpoint

### `GET` `/api/v1/orchestration/status/:id`

Get deliberation status

### `GET` `/api/v1/orchestration/templates`

List available templates

### `POST` `/api/v1/orchestration/deliberate/sync`

Synchronous deliberation

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

## Partner

Partner API HTTP handlers.

### `POST` `/api/partners/register`

Register as a partner

### `GET` `/api/partners/me`

Get current partner profile

### `POST` `/api/partners/keys`

Create API key

### `GET` `/api/partners/keys`

List API keys

### `DELETE` `/api/partners/keys/\{key_id\}`

Revoke API key

### `GET` `/api/partners/usage`

Get usage statistics

### `POST` `/api/partners/webhooks`

Configure webhook

### `GET` `/api/partners/limits`

Get rate limits

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

Policy and Compliance endpoint handlers.

### `GET` `/api/policies`

List policies

### `GET` `/api/policies/:id`

Get policy details

### `POST` `/api/policies`

Create policy

### `PATCH` `/api/policies/:id`

Update policy

### `DELETE` `/api/policies/:id`

Delete policy

### `POST` `/api/policies/:id/toggle`

Toggle policy enabled status

### `GET` `/api/policies/:id/violations`

Get violations for a policy

### `GET` `/api/compliance/violations`

List all violations

### `GET` `/api/compliance/violations/:id`

Get violation details

### `PATCH` `/api/compliance/violations/:id`

Update violation status

### `POST` `/api/compliance/check`

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

### `GET` `/api/v1/rlm/stats`

GET /api/v1/rlm/stats

### `GET` `/api/v1/rlm/strategies`

GET /api/v1/rlm/strategies

### `GET` `/api/v1/rlm/compress`

Get or create the hierarchical compressor using factory

### `GET` `/api/v1/rlm/query`

Simple fallback query when full RLM is not available

### `GET` `/api/v1/rlm/contexts`

GET /api/v1/rlm/contexts

### `GET` `/api/v1/rlm/stream`

GET /api/v1/rlm/stream

### `GET` `/api/v1/rlm/stream/modes`

GET /api/v1/rlm/stream/modes

---

## Selection

Handler for selection plugin endpoints.

### `GET` `/api/v1/selection/plugins` 🔒

List all available selection plugins

### `GET` `/api/v1/selection/defaults` 🔒

Get default plugin configuration

### `GET` `/api/v1/selection/score` 🔒

Get information about a specific scorer

### `GET` `/api/v1/selection/team` 🔒

Get information about a specific team selector

---

## Shared Inbox

HTTP API Handlers for Shared Inbox Management.

### `POST` `/api/v1/inbox/shared`

Create shared inbox

### `GET` `/api/v1/inbox/shared`

List shared inboxes

### `GET` `/api/v1/inbox/shared/:id`

Get shared inbox details

### `GET` `/api/v1/inbox/shared/:id/messages`

Get messages in inbox

### `POST` `/api/v1/inbox/shared/:id/messages/:msg_id/assign`

Assign message

### `POST` `/api/v1/inbox/shared/:id/messages/:msg_id/status`

Update status

### `POST` `/api/v1/inbox/shared/:id/messages/:msg_id/tag`

Add tag

### `POST` `/api/v1/inbox/routing/rules`

Create routing rule

### `GET` `/api/v1/inbox/routing/rules`

List routing rules

### `PATCH` `/api/v1/inbox/routing/rules/:id`

Update routing rule

### `DELETE` `/api/v1/inbox/routing/rules/:id`

Delete routing rule

### `POST` `/api/v1/inbox/routing/rules/:id/test`

Test routing rule

---

## Template Marketplace

Template Marketplace API Handler.

### `GET` `/api/marketplace/templates`

Browse marketplace templates

### `GET` `/api/marketplace/templates/:id`

Get marketplace template details

### `POST` `/api/marketplace/templates`

Publish template to marketplace

### `POST` `/api/marketplace/templates/:id/rate`

Rate a template

### `GET` `/api/marketplace/templates/:id/reviews`

Get template reviews

### `POST` `/api/marketplace/templates/:id/reviews`

Submit a review

### `POST` `/api/marketplace/templates/:id/import`

Import to workspace

### `GET` `/api/marketplace/featured`

Get featured templates

### `GET` `/api/marketplace/trending`

Get trending templates

### `GET` `/api/marketplace/categories`

Get marketplace categories

---

## Threat Intel

Threat Intelligence API Handlers.

### `POST` `/api/v1/threat/url`

Check URL for threats

### `POST` `/api/v1/threat/urls`

Batch check URLs

### `GET` `/api/v1/threat/ip/\{ip_address\}`

Check IP reputation

### `POST` `/api/v1/threat/ips`

Batch check IPs

### `GET` `/api/v1/threat/hash/\{hash_value\}`

Check file hash reputation

### `POST` `/api/v1/threat/hashes`

Batch check hashes

### `POST` `/api/v1/threat/email`

Scan email content

### `GET` `/api/v1/threat/status`

Get service status

---

## Tournaments

Tournament-related endpoint handlers.

### `GET` `/api/tournaments`

List all tournaments

### `POST` `/api/tournaments`

Create new tournament

### `GET` `/api/tournaments/\{id\}`

Get tournament details

### `GET` `/api/tournaments/\{id\}/standings`

Get tournament standings

### `GET` `/api/tournaments/\{id\}/bracket`

Get bracket structure

### `GET` `/api/tournaments/\{id\}/matches`

Get match history

### `POST` `/api/tournaments/\{id\}/advance`

Advance to next round

### `POST` `/api/tournaments/\{id\}/matches/\{match_id\}/result`

Record match result

---

## Training

Handler for training data export endpoints.

### `GET` `/api/v1/training/export/sft`

Get or create SFT exporter

### `GET` `/api/v1/training/export/dpo`

Get or create DPO exporter

### `GET` `/api/v1/training/export/gauntlet`

Get or create Gauntlet exporter

### `GET` `/api/v1/training/stats`

GET /api/v1/training/stats

### `GET` `/api/v1/training/formats`

GET /api/v1/training/formats

### `GET` `/api/v1/training/jobs`

GET /api/v1/training/jobs

---

## Transcription

Transcription endpoint handlers for speech-to-text and media processing.

### `POST` `/api/transcription/audio`

Transcribe audio file

### `POST` `/api/transcription/video`

Extract and transcribe audio from video

### `POST` `/api/transcription/youtube`

Transcribe YouTube video

### `GET` `/api/transcription/status/:id`

Get transcription job status

### `GET` `/api/transcription/config`

Get supported formats and limits

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

### `PUT` `/api/verticals/:id/config`

Update vertical configuration

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

### `GET` `/api/v1/webhooks`

Handle GET /api/webhooks - list all webhooks

### `GET` `/api/v1/webhooks/events`

Handle GET /api/webhooks/events - list available event types

### `GET` `/api/v1/webhooks/slo/status`

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

### `GET` `/api/v1/workflows`

Handle GET /api/workflows

### `GET` `/api/v1/workflows/*`

GET /api/v1/workflows/*

### `GET` `/api/v1/workflow-templates`

GET /api/v1/workflow-templates

### `GET` `/api/v1/workflow-approvals`

GET /api/v1/workflow-approvals

### `GET` `/api/v1/workflow-approvals/*`

GET /api/v1/workflow-approvals/*

### `GET` `/api/v1/workflow-executions`

GET /api/v1/workflow-executions

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

## Intelligence

HTTP API Handlers for Code Intelligence Analysis.

### `POST` `/api/v1/codebase/\{repo\}/analyze`

Analyze codebase structure

### `GET` `/api/v1/codebase/\{repo\}/symbols`

List symbols (classes, functions)

### `GET` `/api/v1/codebase/\{repo\}/callgraph`

Get call graph

### `GET` `/api/v1/codebase/\{repo\}/deadcode`

Find dead/unreachable code

### `POST` `/api/v1/codebase/\{repo\}/impact`

Analyze impact of changes

### `POST` `/api/v1/codebase/\{repo\}/understand`

Answer questions about code

### `POST` `/api/v1/codebase/\{repo\}/audit`

Run comprehensive audit

---

## Metrics

HTTP API Handlers for Codebase Metrics Analysis.

### `POST` `/api/v1/codebase/\{repo\}/metrics/analyze`

Run metrics analysis

### `GET` `/api/v1/codebase/\{repo\}/metrics`

Get latest metrics

### `GET` `/api/v1/codebase/\{repo\}/metrics/\{analysis_id\}`

Get specific analysis

### `GET` `/api/v1/codebase/\{repo\}/hotspots`

Get complexity hotspots

### `GET` `/api/v1/codebase/\{repo\}/duplicates`

Get code duplicates

---

## Quick Scan

Quick Security Scan API Handler.

### `POST` `/api/codebase/quick-scan`

Run quick security scan

### `GET` `/api/codebase/quick-scan/\{scan_id\}`

Get scan result

---

## Security

HTTP API Handlers for Codebase Security Analysis.

### `POST` `/api/v1/codebase/\{repo\}/scan`

Trigger security scan

### `GET` `/api/v1/codebase/\{repo\}/scan/latest`

Get latest scan result

### `GET` `/api/v1/codebase/\{repo\}/scan/\{scan_id\}`

Get specific scan result

### `GET` `/api/v1/codebase/\{repo\}/vulnerabilities`

List all vulnerabilities

### `GET` `/api/v1/cve/\{cve_id\}`

Get CVE details

---

## Audit Bridge

Audit-to-GitHub Bridge Handler.

### `POST` `/api/v1/github/audit/issues`

Create issues from findings

### `POST` `/api/v1/github/audit/issues/bulk`

Bulk create issues

### `POST` `/api/v1/github/audit/pr`

Create PR with fixes

### `GET` `/api/v1/github/audit/sync/\{session_id\}`

Get sync status

### `POST` `/api/v1/github/audit/sync/\{session_id\}`

Sync session to GitHub

---

## Pr Review

HTTP API Handlers for GitHub Pull Request Review.

### `POST` `/api/v1/github/pr/review`

Trigger PR review

### `GET` `/api/v1/github/pr/\{pr_number\}`

Get PR details

### `POST` `/api/v1/github/pr/\{pr_number\}/review`

Submit review

### `GET` `/api/v1/github/pr/\{pr_number\}/reviews`

List reviews

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