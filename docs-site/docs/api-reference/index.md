title: Curated API Reference
description: Quickstart-focused reference for the most important Aragora endpoints
sidebar_position: 1
---

# Curated API Reference

Use this page when you need the endpoints most teams touch first. It keeps the
quickstart-critical routes in one place and points to the generated reference
when you need full schemas or less common operations.

## Base URL

```
https://api.aragora.ai/api/v1
```

For self-hosted deployments, replace with your server URL. Legacy `/api/...`
paths are also supported for backward compatibility.

## Authentication

All API requests require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.aragora.ai/api/v1/debates
```

See the [Authentication Guide](/docs/security/authentication) for details on obtaining API keys.

## Essential Endpoints

### Debates

| Method | Endpoint | Use it for |
|--------|----------|------------|
| POST | `/api/v1/debates` | Start a new multi-agent debate |
| GET | `/api/v1/debates` | List recent or filtered debates |
| GET | `/api/v1/debates/:id` | Read one debate with status and metadata |
| GET | `/api/v1/debates/:id/messages` | Inspect the round-by-round transcript |
| GET | `/api/v1/debates/:id/consensus` | Pull the final synthesis and voting outcome |

### Agents

| Method | Endpoint | Use it for |
|--------|----------|------------|
| GET | `/api/v1/agents` | List available agent types and providers |
| GET | `/api/v1/agents/:name` | Inspect one agent configuration |
| GET | `/api/v1/agents/:name/stats` | Check performance and usage stats |
| GET | `/api/v1/leaderboard` | Compare agents by current ranking |

### Knowledge

| Method | Endpoint | Use it for |
|--------|----------|------------|
| POST | `/api/v1/knowledge/mound/query` | Search the Knowledge Mound |
| POST | `/api/v1/knowledge/mound/nodes` | Store a new knowledge node |
| PUT | `/api/v1/knowledge/mound/nodes/:id` | Update a node in place |
| DELETE | `/api/v1/knowledge/mound/nodes/:id` | Remove stale or incorrect knowledge |

### Workflows

| Method | Endpoint | Use it for |
|--------|----------|------------|
| GET | `/api/v1/workflows` | List saved workflows |
| POST | `/api/v1/workflows` | Create a reusable workflow |
| POST | `/api/v1/workflows/:id/execute` | Launch a workflow run |
| GET | `/api/v1/workflows/executions/:id` | Poll execution progress and result |

### Platform Essentials

| Method | Endpoint | Use it for |
|--------|----------|------------|
| POST | `/api/v1/decisions` | Submit one decision request without building a workflow |
| GET | `/api/v1/decisions/:id/status` | Poll an async decision request |
| GET | `/api/v1/receipts/:receipt_id` | Retrieve the audit receipt for a completed action |
| POST | `/api/v1/github/pr/review` | Trigger a GitHub PR review from Aragora |
| GET | `/api/features/discover` | Discover deployed capabilities before deeper integration |

## Example Calls

### Start a debate and read consensus

```bash
curl -X POST https://api.aragora.ai/api/v1/debates \
  -H "Authorization: Bearer $ARAGORA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"topic":"Should we gate deploys on two-model review?","agents":["claude","gpt4"],"rounds":2}'

curl -H "Authorization: Bearer $ARAGORA_API_KEY" \
  https://api.aragora.ai/api/v1/debates/debate_abc123/consensus
```

### Store and query knowledge

```bash
curl -X POST https://api.aragora.ai/api/v1/knowledge/mound/nodes \
  -H "Authorization: Bearer $ARAGORA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"title":"Merge gate policy","content":"Require passing receipts before merge."}'

curl -X POST https://api.aragora.ai/api/v1/knowledge/mound/query \
  -H "Authorization: Bearer $ARAGORA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"merge gate receipts","limit":5}'
```

### Execute a workflow or one-off decision

```bash
curl -X POST https://api.aragora.ai/api/v1/workflows/workflow_123/execute \
  -H "Authorization: Bearer $ARAGORA_API_KEY"

curl -X POST https://api.aragora.ai/api/v1/decisions \
  -H "Authorization: Bearer $ARAGORA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"Ship the retry-queue change today?","mode":"judge"}'
```

## Need Full Schemas?

- [Generated API Reference](/docs/api/reference)
- [Generated Endpoint Catalog](/docs/api/endpoints)
- [OpenAPI JSON](https://api.aragora.ai/api/v1/openapi.json)
- [TypeScript SDK Guide](/docs/guides/sdk-typescript)
- [Python SDK Guide](/docs/guides/sdk)
