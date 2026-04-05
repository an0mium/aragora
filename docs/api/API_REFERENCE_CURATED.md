# Curated API Reference

Use this page when you need the endpoints most teams touch first. It keeps the
quickstart-critical routes in one place and points to the generated reference
when you need full schemas or less common operations.

> **New to Aragora?** Start with the [Developer Quickstart](../QUICKSTART_DEVELOPER.md)
> to run your first review, then come back here for API integration.

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

See the [Authentication Guide](../guides/AUTH_GUIDE.md) for details on obtaining API keys.

## Essential Endpoints

### Debates

| Method | Endpoint | Use it for |
|--------|----------|------------|
| `POST` | `/debates` | Start a new debate or review |
| `GET` | `/debates/{debate_id}` | Fetch debate status and receipt data |
| `GET` | `/debates/{debate_id}/stream` | Stream live debate events |

### Agents

| Method | Endpoint | Use it for |
|--------|----------|------------|
| `GET` | `/agents` | List available registered agents |
| `POST` | `/agents/select` | Choose agents for a domain or task |

### Knowledge

| Method | Endpoint | Use it for |
|--------|----------|------------|
| `POST` | `/knowledge/search` | Retrieve supporting evidence before debates |
| `POST` | `/memory/search` | Search prior debates, findings, and receipts |

### Workflows

| Method | Endpoint | Use it for |
|--------|----------|------------|
| `POST` | `/workflows` | Launch a review or orchestration workflow |
| `GET` | `/workflows/{workflow_id}` | Track workflow state |

## When To Use The Generated Reference

Use the [Generated API Reference](./API_REFERENCE.md) when you need:

- full request and response schemas
- less common or specialized endpoints
- exact parameter and response field definitions
- operation-level examples
