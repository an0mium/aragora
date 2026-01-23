---
title: Workflows API
description: Automate debate workflows
sidebar_position: 1
---

# Workflows API

The Workflows API allows you to create and execute automated debate workflows.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/workflows` | List workflows |
| POST | `/api/workflows` | Create workflow |
| POST | `/api/workflows/:id/execute` | Execute workflow |
| GET | `/api/workflows/executions/:id` | Get execution status |

All endpoints also accept the `/api/v1/` prefix. For full schemas and response
examples, see the generated API reference at `/docs/api/reference` or the
OpenAPI spec at `/api/v1/openapi.json`.
