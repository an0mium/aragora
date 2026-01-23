---
title: Knowledge API
description: Knowledge Mound operations
sidebar_position: 1
---

# Knowledge API

The Knowledge API provides access to the Knowledge Mound.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/knowledge/mound/query` | Query knowledge |
| POST | `/api/knowledge/mound/nodes` | Store knowledge |
| PUT | `/api/knowledge/mound/nodes/:id` | Update knowledge |
| DELETE | `/api/knowledge/mound/nodes/:id` | Delete knowledge |

All endpoints also accept the `/api/v1/` prefix. For full schemas and response
examples, see the generated API reference at `/docs/api/reference` or the
OpenAPI spec at `/api/v1/openapi.json`.
