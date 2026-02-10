# OpenClaw Gateway API Reference

REST API for the OpenClaw gateway integration. Provides session management, action execution, credential storage, and administrative operations.

**Base path:** `/api/gateway/openclaw`
**Versioned aliases:** `/api/v1/gateway/openclaw`, `/api/v1/openclaw`

All endpoints require authentication unless noted otherwise. Responses use JSON. Timestamps are ISO 8601 in UTC.

---

## Session Endpoints

### POST /gateway/openclaw/sessions

Create a new gateway session.

**Permission:** `gateway:sessions.create`
**Rate limit:** 30 req/min

**Request body:**

```json
{
  "config": { "timeout": 300, "max_actions": 50 },
  "metadata": { "purpose": "code-review" }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `config` | object | No | Session configuration key-value pairs |
| `metadata` | object | No | Arbitrary metadata attached to the session |

**Response (201):**

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "user_id": "user-42",
  "tenant_id": "org-7",
  "status": "active",
  "created_at": "2026-01-31T12:00:00+00:00",
  "updated_at": "2026-01-31T12:00:00+00:00",
  "last_activity_at": "2026-01-31T12:00:00+00:00",
  "config": { "timeout": 300, "max_actions": 50 },
  "metadata": { "purpose": "code-review" }
}
```

**Errors:** `500` Internal server error.

---

### GET /gateway/openclaw/sessions/{id}

Retrieve a single session by ID. Non-admin users can only access their own sessions.

**Permission:** `gateway:sessions.read`
**Rate limit:** 120 req/min

**Response (200):** Session object (same schema as creation response).

**Errors:** `403` Access denied (not owner or admin). `404` Session not found. `500` Internal server error.

---

### DELETE /gateway/openclaw/sessions/{id}

Close a session. Sets the session status to `closed`.

**Permission:** `gateway:sessions.delete`
**Rate limit:** 30 req/min

**Response (200):**

```json
{
  "closed": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**Errors:** `403` Access denied. `404` Session not found. `500` Internal server error.

---

### GET /gateway/openclaw/sessions

List sessions scoped to the authenticated user and tenant.

**Permission:** `gateway:sessions.read`
**Rate limit:** 120 req/min

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | (none) | Filter by status: `active`, `idle`, `closing`, `closed`, `error` |
| `limit` | int | 50 | Max results (up to 500) |
| `offset` | int | 0 | Pagination offset |

**Response (200):**

```json
{
  "sessions": [ { "id": "...", "status": "active", "..." : "..." } ],
  "total": 12,
  "limit": 50,
  "offset": 0
}
```

**Errors:** `400` Invalid parameter. `500` Internal server error.

---

## Action Endpoints

### POST /gateway/openclaw/actions

Execute an action within an active session. The action is created in `pending` status and immediately transitions to `running`.

**Permission:** `gateway:actions.execute`
**Rate limit:** 60 req/min (per-user auth rate limit)

**Request body:**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "action_type": "web_search",
  "input": { "query": "OpenClaw documentation" },
  "metadata": { "priority": "high" }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | **Yes** | ID of the active session to execute within |
| `action_type` | string | **Yes** | Type of action to execute |
| `input` | object | No | Input data for the action |
| `metadata` | object | No | Arbitrary metadata |

**Response (202):**

```json
{
  "id": "f0e1d2c3-b4a5-6789-0fed-cba987654321",
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "action_type": "web_search",
  "status": "running",
  "input_data": { "query": "OpenClaw documentation" },
  "output_data": null,
  "error": null,
  "created_at": "2026-01-31T12:05:00+00:00",
  "started_at": "2026-01-31T12:05:00+00:00",
  "completed_at": null,
  "metadata": { "priority": "high" }
}
```

**Errors:** `400` Missing `session_id` or `action_type`, or session not active. `403` Access denied. `404` Session not found. `500` Internal server error.

---

### GET /gateway/openclaw/actions/{id}

Get the current status of an action. Access is verified through session ownership.

**Permission:** `gateway:actions.read`
**Rate limit:** 120 req/min

**Response (200):** Action object (same schema as execution response).

**Action status values:** `pending`, `running`, `completed`, `failed`, `cancelled`, `timeout`.

**Errors:** `403` Access denied. `404` Action not found. `500` Internal server error.

---

### POST /gateway/openclaw/actions/{id}/cancel

Cancel a pending or running action.

**Permission:** `gateway:actions.cancel`
**Rate limit:** 30 req/min

**Response (200):**

```json
{
  "cancelled": true,
  "action_id": "f0e1d2c3-b4a5-6789-0fed-cba987654321"
}
```

**Errors:** `400` Action is not cancellable (already completed, failed, or cancelled). `403` Access denied. `404` Action not found. `500` Internal server error.

---

## Credential Endpoints

Credentials are stored with secrets separated from metadata. List and get operations never return secret values.

### POST /gateway/openclaw/credentials

Store a new credential.

**Permission:** `gateway:credentials.create`
**Rate limit:** 10 req/min (per-user auth rate limit)

**Request body:**

```json
{
  "name": "github-deploy-key",
  "type": "ssh_key",
  "secret": "ssh-key-placeholder",
  "expires_at": "2027-01-31T00:00:00+00:00",
  "metadata": { "environment": "production" }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | **Yes** | Human-readable credential name |
| `type` | string | **Yes** | One of: `api_key`, `oauth_token`, `password`, `certificate`, `ssh_key`, `service_account` |
| `secret` | string | **Yes** | The secret value (never returned in responses) |
| `expires_at` | string | No | ISO 8601 expiration timestamp |
| `metadata` | object | No | Arbitrary metadata |

**Response (201):**

```json
{
  "id": "cred-1234-5678-abcd",
  "name": "github-deploy-key",
  "credential_type": "ssh_key",
  "user_id": "user-42",
  "tenant_id": "org-7",
  "created_at": "2026-01-31T12:10:00+00:00",
  "updated_at": "2026-01-31T12:10:00+00:00",
  "last_rotated_at": null,
  "expires_at": "2027-01-31T00:00:00+00:00",
  "metadata": { "environment": "production" }
}
```

**Errors:** `400` Missing required field or invalid credential type. `500` Internal server error.

---

### GET /gateway/openclaw/credentials

List credentials (metadata only) scoped to the authenticated user and tenant.

**Permission:** `gateway:credentials.read`
**Rate limit:** 60 req/min

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | (none) | Filter by credential type |
| `limit` | int | 50 | Max results (up to 500) |
| `offset` | int | 0 | Pagination offset |

**Response (200):**

```json
{
  "credentials": [ { "id": "...", "name": "...", "credential_type": "api_key", "..." : "..." } ],
  "total": 3,
  "limit": 50,
  "offset": 0
}
```

**Errors:** `400` Invalid parameter. `500` Internal server error.

---

### DELETE /gateway/openclaw/credentials/{id}

Delete a credential and its stored secret.

**Permission:** `gateway:credentials.delete`
**Rate limit:** 20 req/min

**Response (200):**

```json
{
  "deleted": true,
  "credential_id": "cred-1234-5678-abcd"
}
```

**Errors:** `403` Access denied. `404` Credential not found. `500` Internal server error.

---

### POST /gateway/openclaw/credentials/{id}/rotate

Replace a credential's secret value with a new one.

**Permission:** `gateway:credentials.rotate`
**Rate limit:** 10 req/min (per-user auth rate limit)

**Request body:**

```json
{
  "secret": "new-secret-value-here"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `secret` | string | **Yes** | The new secret value |

**Response (200):**

```json
{
  "rotated": true,
  "credential_id": "cred-1234-5678-abcd",
  "rotated_at": "2026-01-31T14:00:00+00:00"
}
```

**Errors:** `400` Missing `secret`. `403` Access denied. `404` Credential not found. `500` Internal server error.

---

## Admin Endpoints

### GET /gateway/openclaw/health

Health check for the gateway. This is a **public endpoint** -- no authentication or permission required.

**Rate limit:** None

**Response (200):**

```json
{
  "status": "healthy",
  "healthy": true,
  "timestamp": "2026-01-31T12:00:00+00:00",
  "active_sessions": 5,
  "pending_actions": 2,
  "running_actions": 3
}
```

**Status values:**
- `healthy` -- Normal operation.
- `degraded` -- Running actions exceed 100.
- `unhealthy` -- Pending actions exceed 500.
- `error` -- Internal error (returned with HTTP 503).

**Response (503):** Returned when the health check itself fails.

```json
{
  "status": "error",
  "healthy": false,
  "error": "...",
  "timestamp": "2026-01-31T12:00:00+00:00"
}
```

---

### GET /gateway/openclaw/metrics

Detailed gateway metrics broken down by resource type and status.

**Permission:** `gateway:metrics.read`
**Rate limit:** 30 req/min

**Response (200):**

```json
{
  "sessions": {
    "total": 25,
    "active": 5,
    "by_status": { "active": 5, "idle": 3, "closing": 0, "closed": 17, "error": 0 }
  },
  "actions": {
    "total": 142,
    "pending": 2,
    "running": 3,
    "by_status": { "pending": 2, "running": 3, "completed": 130, "failed": 5, "cancelled": 1, "timeout": 1 }
  },
  "credentials": {
    "total": 8,
    "by_type": { "api_key": 4, "oauth_token": 2, "password": 0, "certificate": 1, "ssh_key": 1, "service_account": 0 }
  },
  "audit_log_entries": 312,
  "timestamp": "2026-01-31T12:00:00+00:00"
}
```

**Errors:** `500` Internal server error.

---

### GET /gateway/openclaw/audit

Query the audit log. All gateway mutations are automatically recorded.

**Permission:** `gateway:audit.read`
**Rate limit:** 30 req/min

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | string | (none) | Filter by action (e.g. `session.create`, `credential.rotate`) |
| `actor_id` | string | (none) | Filter by actor user ID |
| `resource_type` | string | (none) | Filter by resource type: `session`, `action`, `credential` |
| `limit` | int | 100 | Max results (up to 1000) |
| `offset` | int | 0 | Pagination offset |

**Response (200):**

```json
{
  "entries": [
    {
      "id": "audit-001",
      "timestamp": "2026-01-31T12:05:00+00:00",
      "action": "action.execute",
      "actor_id": "user-42",
      "resource_type": "action",
      "resource_id": "f0e1d2c3-b4a5-6789-0fed-cba987654321",
      "result": "pending",
      "details": { "action_type": "web_search", "session_id": "a1b2c3d4-..." }
    }
  ],
  "total": 312,
  "limit": 100,
  "offset": 0
}
```

**Audit action values:** `session.create`, `session.close`, `session.end`, `action.execute`, `action.cancel`, `credential.create`, `credential.rotate`, `credential.delete`, `policy.rule.add`, `policy.rule.remove`, `approval.approve`, `approval.deny`.

**Errors:** `500` Internal server error.

---

## Permissions Summary

| Endpoint | Permission |
|----------|------------|
| Create session | `gateway:sessions.create` |
| Get / list sessions | `gateway:sessions.read` |
| Close / delete session | `gateway:sessions.delete` |
| Execute action | `gateway:actions.execute` |
| Get action | `gateway:actions.read` |
| Cancel action | `gateway:actions.cancel` |
| Store credential | `gateway:credentials.create` |
| List credentials | `gateway:credentials.read` |
| Delete credential | `gateway:credentials.delete` |
| Rotate credential | `gateway:credentials.rotate` |
| Health check | _(none -- public)_ |
| Metrics | `gateway:metrics.read` |
| Audit log | `gateway:audit.read` |

Users with `gateway:admin` permission can bypass ownership checks on sessions, actions, and credentials.

## Error Response Format

All error responses follow a consistent structure:

```json
{
  "error": "Description of the error"
}
```

| HTTP Code | Meaning |
|-----------|---------|
| 400 | Bad request -- missing or invalid parameters |
| 403 | Forbidden -- insufficient permissions or not resource owner |
| 404 | Not found -- resource does not exist |
| 500 | Internal server error |
| 503 | Service unavailable (health endpoint only) |

## Rate Limits

Rate limits are enforced per-client. Auth-scoped endpoints (credential store, credential rotate, action execute) apply per-user limits.

| Endpoint group | Limit |
|----------------|-------|
| Session create / close | 30 req/min |
| Session list / get | 120 req/min |
| Action execute | 60 req/min |
| Action get | 120 req/min |
| Action cancel | 30 req/min |
| Credential store / rotate | 10 req/min |
| Credential list | 60 req/min |
| Credential delete | 20 req/min |
| Metrics / audit / stats | 30 req/min |
