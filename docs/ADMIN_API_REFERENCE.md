# Admin API Reference

This document provides a comprehensive reference for Aragora's administrative APIs.

## Overview

The Admin API provides endpoints for:
- User and organization management
- Billing and subscription management
- System health and monitoring
- Security and audit logging
- Dashboard metrics

**Base URL:** `/api/v1/admin/`

**Authentication:** All admin endpoints require authentication with appropriate role-based permissions.

---

## User Management

### List Users

Retrieve a paginated list of all users.

```
GET /api/v1/admin/users
```

**Required Permission:** `admin:users`

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `page` | integer | Page number (default: 1) |
| `limit` | integer | Items per page (default: 50, max: 100) |
| `status` | string | Filter by status: `active`, `inactive`, `locked` |
| `search` | string | Search by email or name |

**Response:**
```json
{
  "users": [
    {
      "user_id": "user-123",
      "email": "user@example.com",
      "name": "John Doe",
      "status": "active",
      "created_at": "2024-01-15T10:00:00Z",
      "last_login": "2024-01-26T08:30:00Z",
      "organization_id": "org-456"
    }
  ],
  "total": 150,
  "page": 1,
  "limit": 50
}
```

### Deactivate User

Deactivate a user account.

```
POST /api/v1/admin/users/{user_id}/deactivate
```

**Required Permission:** `admin:users:write`

**Response:**
```json
{
  "success": true,
  "user_id": "user-123",
  "status": "inactive"
}
```

### Activate User

Reactivate a deactivated user account.

```
POST /api/v1/admin/users/{user_id}/activate
```

**Required Permission:** `admin:users:write`

### Unlock User

Unlock a locked user account (e.g., after too many failed login attempts).

```
POST /api/v1/admin/users/{user_id}/unlock
```

**Required Permission:** `admin:users:write`

### Impersonate User

Temporarily impersonate a user for support purposes. Creates an audit log entry.

```
POST /api/v1/admin/users/{user_id}/impersonate
```

**Required Permission:** `admin:users:impersonate`

**Response:**
```json
{
  "session_token": "temp-session-xyz",
  "expires_at": "2024-01-26T09:00:00Z",
  "audit_id": "audit-789"
}
```

---

## Organization Management

### List Organizations

Retrieve all organizations.

```
GET /api/v1/admin/organizations
```

**Required Permission:** `admin:organizations`

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `page` | integer | Page number |
| `limit` | integer | Items per page |
| `plan` | string | Filter by subscription plan |
| `status` | string | Filter by status |

**Response:**
```json
{
  "organizations": [
    {
      "org_id": "org-456",
      "name": "Acme Corp",
      "plan": "enterprise",
      "status": "active",
      "user_count": 25,
      "created_at": "2023-06-01T00:00:00Z",
      "subscription": {
        "plan_id": "plan-enterprise",
        "status": "active",
        "current_period_end": "2024-02-01T00:00:00Z"
      }
    }
  ],
  "total": 50
}
```

---

## Billing & Usage

### Get Usage

Retrieve usage metrics for an organization or user.

```
GET /api/v1/admin/billing/usage
```

**Required Permission:** `org:billing`

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `org_id` | string | Organization ID |
| `user_id` | string | User ID (optional) |
| `start_date` | string | Start date (ISO 8601) |
| `end_date` | string | End date (ISO 8601) |

**Response:**
```json
{
  "org_id": "org-456",
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "usage": {
    "debates": 150,
    "tokens_in": 2500000,
    "tokens_out": 1200000,
    "api_calls": 5000,
    "storage_bytes": 104857600
  },
  "costs": {
    "total_usd": 125.50,
    "by_provider": {
      "anthropic": 75.00,
      "openai": 45.00,
      "google": 5.50
    }
  }
}
```

### Get Subscription

Retrieve subscription details.

```
GET /api/v1/admin/billing/subscription
```

**Required Permission:** `org:billing`

**Response:**
```json
{
  "subscription_id": "sub-123",
  "plan": {
    "id": "plan-enterprise",
    "name": "Enterprise",
    "features": ["unlimited_debates", "priority_support", "sla_guarantee"]
  },
  "status": "active",
  "current_period_start": "2024-01-01T00:00:00Z",
  "current_period_end": "2024-02-01T00:00:00Z",
  "cancel_at_period_end": false
}
```

### Create Checkout Session

Create a Stripe checkout session for subscription upgrade.

```
POST /api/v1/admin/billing/checkout
```

**Required Permission:** `org:billing`

**Request Body:**
```json
{
  "plan_id": "plan-enterprise",
  "success_url": "https://app.example.com/billing/success",
  "cancel_url": "https://app.example.com/billing/cancel"
}
```

**Response:**
```json
{
  "checkout_url": "https://checkout.stripe.com/...",
  "session_id": "cs_123"
}
```

### Get Invoices

Retrieve invoice history.

```
GET /api/v1/admin/billing/invoices
```

**Required Permission:** `org:billing`

**Response:**
```json
{
  "invoices": [
    {
      "invoice_id": "inv-123",
      "amount_due": 99.00,
      "currency": "usd",
      "status": "paid",
      "created_at": "2024-01-01T00:00:00Z",
      "paid_at": "2024-01-01T00:05:00Z",
      "pdf_url": "https://..."
    }
  ]
}
```

### Get Usage Forecast

Get projected usage and costs.

```
GET /api/v1/admin/billing/forecast
```

**Required Permission:** `org:billing`

**Response:**
```json
{
  "forecast_period": {
    "start": "2024-02-01T00:00:00Z",
    "end": "2024-02-29T23:59:59Z"
  },
  "projected_usage": {
    "debates": 180,
    "tokens_in": 3000000,
    "tokens_out": 1500000
  },
  "projected_cost_usd": 150.00,
  "confidence": 0.85
}
```

### Export Usage CSV

Export usage data as CSV.

```
GET /api/v1/admin/billing/export
```

**Required Permission:** `org:billing`

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `start_date` | string | Start date |
| `end_date` | string | End date |
| `format` | string | Export format (default: csv) |

### Cancel Subscription

Cancel subscription at period end.

```
POST /api/v1/admin/billing/subscription/cancel
```

**Required Permission:** `org:billing`

### Resume Subscription

Resume a cancelled subscription.

```
POST /api/v1/admin/billing/subscription/resume
```

**Required Permission:** `org:billing`

---

## Audit & Compliance

### Get Audit Log

Retrieve audit log entries.

```
GET /api/v1/admin/billing/audit
```

**Required Permission:** `admin:audit`

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `start_date` | string | Start date |
| `end_date` | string | End date |
| `user_id` | string | Filter by user |
| `action` | string | Filter by action type |
| `limit` | integer | Max entries (default: 100) |

**Response:**
```json
{
  "entries": [
    {
      "audit_id": "audit-123",
      "timestamp": "2024-01-26T08:30:00Z",
      "user_id": "user-123",
      "action": "user.login",
      "resource_type": "user",
      "resource_id": "user-123",
      "ip_address": "192.168.1.1",
      "details": {
        "method": "password",
        "mfa_used": true
      }
    }
  ],
  "total": 500
}
```

---

## System Health

### Health Check

Check system health status.

```
GET /api/v1/admin/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.4.0",
  "uptime_seconds": 86400,
  "components": {
    "database": "ok",
    "redis": "ok",
    "agents": "ok"
  }
}
```

### Kubernetes Probes

Liveness and readiness probes for Kubernetes.

```
GET /api/v1/health/live
GET /api/v1/health/ready
```

These endpoints do not require authentication.

---

## System Metrics

### Get System Metrics

Retrieve system performance metrics.

```
GET /api/v1/admin/metrics
```

**Required Permission:** `admin:system`

**Response:**
```json
{
  "cpu_percent": 45.2,
  "memory_percent": 62.5,
  "disk_percent": 35.0,
  "active_debates": 12,
  "active_connections": 150,
  "requests_per_minute": 500,
  "agent_status": {
    "healthy": 10,
    "unhealthy": 0
  }
}
```

### Get Admin Stats

Retrieve administrative statistics.

```
GET /api/v1/admin/stats
```

**Required Permission:** `admin:system`

**Response:**
```json
{
  "total_users": 1500,
  "active_users_24h": 250,
  "total_organizations": 50,
  "total_debates": 15000,
  "debates_today": 150,
  "revenue_mtd_usd": 25000.00
}
```

### Get Revenue Stats

Retrieve revenue statistics.

```
GET /api/v1/admin/revenue
```

**Required Permission:** `admin:billing`

**Response:**
```json
{
  "mtd": {
    "total_usd": 25000.00,
    "mrr_usd": 22000.00,
    "new_subscriptions": 5,
    "churned_subscriptions": 1
  },
  "by_plan": {
    "starter": 5000.00,
    "professional": 10000.00,
    "enterprise": 10000.00
  }
}
```

---

## Security

### Validate Token

Validate an authentication token.

```
POST /api/v1/admin/security/validate-token
```

**Required Permission:** `admin:security`

**Request Body:**
```json
{
  "token": "eyJhbG..."
}
```

**Response:**
```json
{
  "valid": true,
  "user_id": "user-123",
  "expires_at": "2024-01-27T00:00:00Z",
  "scopes": ["read", "write"]
}
```

### Revoke Token

Revoke an authentication token.

```
POST /api/v1/admin/security/revoke-token
```

**Required Permission:** `admin:security`

### Get Security Events

Retrieve security-related events.

```
GET /api/v1/admin/security/events
```

**Required Permission:** `admin:security`

---

## Dashboard

### Get Dashboard Metrics

Retrieve comprehensive dashboard metrics.

```
GET /api/v1/admin/dashboard
```

**Required Permission:** `admin:dashboard`

**Response:**
```json
{
  "overview": {
    "active_debates": 15,
    "completed_debates_24h": 45,
    "consensus_rate": 0.85,
    "avg_debate_duration_minutes": 12
  },
  "usage": {
    "debates_this_month": 450,
    "tokens_used": 5000000,
    "api_calls": 15000
  },
  "agents": {
    "total": 10,
    "healthy": 10,
    "by_provider": {
      "anthropic": 3,
      "openai": 4,
      "google": 3
    }
  },
  "quality": {
    "avg_consensus_score": 0.82,
    "avg_response_time_ms": 250,
    "error_rate": 0.02
  }
}
```

### Get Quality Metrics

Retrieve debate quality metrics.

```
GET /api/v1/admin/dashboard/quality
```

**Response:**
```json
{
  "metrics": {
    "consensus_rate": 0.85,
    "avg_rounds_to_consensus": 2.5,
    "user_satisfaction": 4.2,
    "factual_accuracy": 0.92
  },
  "trends": {
    "consensus_rate_trend": "+5%",
    "satisfaction_trend": "+0.3"
  }
}
```

---

## History & Events

### Get Cycles

Retrieve Nomic loop cycle history.

```
GET /api/v1/admin/history/cycles
```

**Required Permission:** `admin:system`

### Get Events

Retrieve system event history.

```
GET /api/v1/admin/history/events
```

**Required Permission:** `admin:system`

### Get Debates History

Retrieve debate history.

```
GET /api/v1/admin/history/debates
```

**Required Permission:** `admin:system`

---

## Error Responses

All endpoints return standard error responses:

```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired token",
    "details": {}
  }
}
```

**Common Error Codes:**
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Rate Limits

Admin endpoints have the following rate limits:

| Endpoint Category | Rate Limit |
|-------------------|------------|
| User Management | 60 rpm |
| Billing | 30 rpm |
| Metrics | 120 rpm |
| History | 30 rpm |
| Security | 20 rpm |

---

## Permissions Reference

| Permission | Description |
|------------|-------------|
| `admin:users` | Read user information |
| `admin:users:write` | Modify user accounts |
| `admin:users:impersonate` | Impersonate users |
| `admin:organizations` | Read organization data |
| `admin:system` | Access system metrics |
| `admin:security` | Access security features |
| `admin:audit` | View audit logs |
| `admin:billing` | Access billing data |
| `admin:dashboard` | View dashboard metrics |
| `org:billing` | Manage organization billing |

---

## Webhook Events

Admin actions can trigger webhooks:

| Event | Description |
|-------|-------------|
| `user.deactivated` | User account deactivated |
| `user.activated` | User account activated |
| `subscription.created` | New subscription created |
| `subscription.cancelled` | Subscription cancelled |
| `subscription.updated` | Subscription plan changed |
| `invoice.paid` | Invoice payment received |
| `security.alert` | Security event detected |
