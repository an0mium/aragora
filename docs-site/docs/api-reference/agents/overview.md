---
title: Agents API
description: Manage AI agents
sidebar_position: 1
---

# Agents API

The Agents API allows you to configure and manage AI agents.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/agents` | List available agents |
| GET | `/api/agents/:name` | Get agent details |
| GET | `/api/agents/:name/stats` | Get agent statistics |
| GET | `/api/leaderboard` | Get agent leaderboard |

All endpoints also accept the `/api/v1/` prefix. For full schemas and response
examples, see the generated API reference at `/docs/api/reference` or the
OpenAPI spec at `/api/v1/openapi.json`.
