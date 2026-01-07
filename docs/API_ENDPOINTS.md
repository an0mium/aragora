# Aragora API Documentation

This document describes the HTTP API endpoints provided by the Aragora server.

## Table of Contents

- [Agents](#agents)
- [Analytics](#analytics)
- [Audio](#audio)
- [Auditing](#auditing)
- [Belief](#belief)
- [Broadcast](#broadcast)
- [Calibration](#calibration)
- [Consensus](#consensus)
- [Critique](#critique)
- [Dashboard](#dashboard)
- [Debates](#debates)
- [Documents](#documents)
- [Evolution](#evolution)
- [Genesis](#genesis)
- [Insights](#insights)
- [Introspection](#introspection)
- [Laboratory](#laboratory)
- [LeaderboardView](#leaderboardview)
- [Memory](#memory)
- [Metrics](#metrics)
- [Moments](#moments)
- [Persona](#persona)
- [Plugins](#plugins)
- [Probes](#probes)
- [Pulse](#pulse)
- [Relationship](#relationship)
- [Replays](#replays)
- [Social](#social)
- [System](#system)
- [Tournaments](#tournaments)
- [Verification](#verification)

---

## Agents

Agent-related endpoint handlers.

### `GET` `/api/leaderboard`

Get agent rankings

### `GET` `/api/rankings`

Get agent rankings (alias)

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

## Broadcast

Broadcast generation handler.

### `POST` `/api/debates/{id}/broadcast`

Generate podcast audio from debate trace

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

### `GET` `/api/debate/{id}/meta-critique`

Get meta-level debate analysis

### `GET` `/api/debate/{id}/graph/stats`

Get argument graph statistics

### `POST` `/api/debates/{id}/fork` 🔒

Fork debate at a branch point

---

## Documents

Document management endpoint handlers.

### `GET` `/api/documents`

List all uploaded documents

### `GET` `/api/documents/formats`

Get supported file formats

### `GET` `/api/documents/{doc_id}`

Get a document by ID

---

## Evolution

Prompt evolution endpoint handlers.

### `GET` `/api/evolution/{agent}/history`

Get prompt evolution history for an agent

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

---

## Metrics

Operational metrics endpoint handlers.

### `GET` `/api/metrics`

Get operational metrics for monitoring

### `GET` `/api/metrics/health`

Detailed health check

### `GET` `/api/metrics/cache`

Cache statistics

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

### `GET` `/api/health`

Health check

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

---

## Tournaments

Tournament-related endpoint handlers.

### `GET` `/api/tournaments`

List all tournaments

### `GET` `/api/tournaments/{id}/standings`

Get tournament standings

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