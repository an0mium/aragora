---
title: API Reference
description: Complete API reference for the Aragora platform
sidebar_position: 1
---

# API Reference

This section contains the complete API reference for the Aragora platform.

## Base URL

```
https://api.aragora.ai/v1
```

For self-hosted deployments, replace with your server URL.

## Authentication

All API requests require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.aragora.ai/v1/debates
```

See the [Authentication Guide](/docs/guides/authentication) for details on obtaining API keys.

## API Sections

### Core APIs

| Section | Description |
|---------|-------------|
| [Debates](/docs/api-reference/debates) | Create, manage, and query debates |
| [Agents](/docs/api-reference/agents) | Agent configuration and management |
| [Knowledge](/docs/api-reference/knowledge) | Knowledge Mound operations |
| [Workflows](/docs/api-reference/workflows) | Automated workflow execution |

### Administration

| Section | Description |
|---------|-------------|
| [Users](/docs/api-reference/users) | User management |
| [Organizations](/docs/api-reference/organizations) | Multi-tenancy |
| [Audit](/docs/api-reference/audit) | Audit logs |

### Integrations

| Section | Description |
|---------|-------------|
| [Webhooks](/docs/api-reference/webhooks) | Event notifications |
| [Connectors](/docs/api-reference/connectors) | Data source integrations |

## Rate Limits

| Tier | Requests/min | Debates/hour |
|------|--------------|--------------|
| Free | 60 | 10 |
| Pro | 300 | 100 |
| Enterprise | Custom | Custom |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1703185200
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests. Please retry after 60 seconds.",
    "details": {
      "retry_after": 60
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

## SDKs

Official SDKs are available for:

- [TypeScript/JavaScript](/docs/guides/sdk-typescript)
- [Python](/docs/guides/sdk-python)

## OpenAPI Specification

The complete OpenAPI specification is available at:

```
https://api.aragora.ai/v1/openapi.json
```

You can import this into tools like Postman or use it to generate clients.
