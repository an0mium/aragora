# Webhook API Documentation

Aragora provides a comprehensive webhook system for receiving real-time notifications about events in the system. Webhooks allow external services to be notified when debates complete, consensus is reached, SLO violations occur, and more.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Event Types](#event-types)
- [Payload Format](#payload-format)
- [Signature Verification](#signature-verification)
- [Batching](#batching)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Overview

The webhook system delivers HTTP POST requests to registered endpoints when subscribed events occur. Key features:

- **HMAC-SHA256 signatures** for payload verification
- **Automatic retries** with exponential backoff (up to 3 retries by default)
- **Event batching** for high-volume scenarios
- **Distributed tracing** via correlation IDs
- **Prometheus metrics** for monitoring delivery success rates

## Authentication

Webhooks use HMAC-SHA256 signatures for authentication. When you register a webhook, you receive a secret key. Store this securely - it's only shown once.

All webhook requests include the `X-Aragora-Signature` header containing the signature:

```
X-Aragora-Signature: sha256=<hex-encoded-hmac>
```

## Endpoints

### Register Webhook

**POST** `/api/webhooks`

Register a new webhook endpoint.

**Request Body:**
```json
{
  "url": "https://your-service.com/webhook",
  "events": ["debate_end", "consensus"],
  "name": "My Production Webhook",
  "description": "Receives debate completion notifications"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | HTTPS URL to receive webhook payloads |
| `events` | string[] | Yes | Event types to subscribe to (use `*` for all) |
| `name` | string | No | Human-readable name |
| `description` | string | No | Description of the webhook's purpose |

**Response (201 Created):**
```json
{
  "webhook": {
    "id": "wh_abc123xyz",
    "url": "https://your-service.com/webhook",
    "events": ["debate_end", "consensus"],
    "secret": "whsec_xxxxxxxxx",
    "active": true,
    "created_at": 1705776000.0,
    "name": "My Production Webhook"
  },
  "message": "Webhook registered successfully. Save the secret - it won't be shown again."
}
```

> **Important:** The `secret` is only returned on creation. Store it securely.

---

### List Webhooks

**GET** `/api/webhooks`

List all registered webhooks for the current user.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `active_only` | boolean | false | Only return active webhooks |

**Response:**
```json
{
  "webhooks": [
    {
      "id": "wh_abc123xyz",
      "url": "https://your-service.com/webhook",
      "events": ["debate_end", "consensus"],
      "active": true,
      "created_at": 1705776000.0,
      "delivery_count": 42,
      "failure_count": 1,
      "last_delivery_at": 1705862400.0,
      "last_delivery_status": 200
    }
  ],
  "count": 1
}
```

---

### Get Webhook

**GET** `/api/webhooks/:id`

Get details for a specific webhook.

**Response:**
```json
{
  "webhook": {
    "id": "wh_abc123xyz",
    "url": "https://your-service.com/webhook",
    "events": ["debate_end"],
    "active": true,
    "created_at": 1705776000.0,
    "delivery_count": 42,
    "failure_count": 1
  }
}
```

---

### Update Webhook

**PATCH** `/api/webhooks/:id`

Update webhook configuration.

**Request Body:**
```json
{
  "url": "https://new-endpoint.com/webhook",
  "events": ["debate_end", "slo_violation"],
  "active": true,
  "name": "Updated Webhook Name"
}
```

All fields are optional. Only provided fields are updated.

**Response:**
```json
{
  "webhook": {
    "id": "wh_abc123xyz",
    "url": "https://new-endpoint.com/webhook",
    "events": ["debate_end", "slo_violation"],
    "active": true
  }
}
```

---

### Delete Webhook

**DELETE** `/api/webhooks/:id`

Delete a webhook registration.

**Response:**
```json
{
  "deleted": true,
  "webhook_id": "wh_abc123xyz"
}
```

---

### Test Webhook

**POST** `/api/webhooks/:id/test`

Send a test event to verify webhook connectivity.

**Response:**
```json
{
  "success": true,
  "status_code": 200,
  "message": "Test webhook delivered successfully"
}
```

---

### List Event Types

**GET** `/api/webhooks/events`

List all available event types for webhook subscriptions.

**Response:**
```json
{
  "events": [
    "debate_start",
    "debate_end",
    "consensus",
    "slo_violation",
    "..."
  ],
  "count": 28,
  "description": {
    "debate_start": "Fired when a debate begins",
    "debate_end": "Fired when a debate completes",
    "consensus": "Fired when consensus is reached"
  }
}
```

---

### SLO Webhook Status

**GET** `/api/webhooks/slo/status`

Get the status of SLO violation webhooks.

**Response:**
```json
{
  "slo_webhooks": {
    "enabled": true,
    "initialized": true,
    "notifications_sent": 15,
    "recoveries_sent": 12
  },
  "violation_state": {
    "km_query": {"in_violation": false},
    "debate_round": {"in_violation": true, "since": 1705862400.0}
  },
  "active_violations": 1
}
```

---

### Test SLO Webhook

**POST** `/api/webhooks/slo/test`

Send a test SLO violation notification.

**Response:**
```json
{
  "success": true,
  "message": "Test SLO violation notification sent successfully",
  "details": {
    "operation": "test_operation",
    "percentile": "p99",
    "latency_ms": 1500.0,
    "threshold_ms": 500.0,
    "severity": "minor"
  }
}
```

## Event Types

Available event types for webhook subscriptions:

### Debate Events
| Event | Description |
|-------|-------------|
| `debate_start` | Fired when a debate begins |
| `debate_end` | Fired when a debate completes |
| `consensus` | Fired when consensus is reached |
| `round_start` | Fired at the start of each debate round |
| `agent_message` | Fired when an agent sends a message |
| `vote` | Fired when a vote is cast |

### Knowledge Events
| Event | Description |
|-------|-------------|
| `knowledge_indexed` | Knowledge item added to mound |
| `knowledge_queried` | Knowledge mound queried |
| `mound_updated` | Knowledge mound structure changed |
| `insight_extracted` | New insight extracted |

### Agent Events
| Event | Description |
|-------|-------------|
| `agent_elo_updated` | Agent ELO rating changed |
| `agent_calibration_changed` | Agent calibration updated |
| `agent_fallback_triggered` | Agent fallback to backup provider |

### Memory Events
| Event | Description |
|-------|-------------|
| `memory_stored` | Memory item stored |
| `memory_retrieved` | Memory item retrieved |

### Verification Events
| Event | Description |
|-------|-------------|
| `claim_verification_result` | Claim verification completed |
| `formal_verification_result` | Formal verification completed |

### Gauntlet Events
| Event | Description |
|-------|-------------|
| `gauntlet_complete` | Gauntlet stress-test completed |
| `gauntlet_verdict` | Gauntlet verdict determined |

### Graph Debate Events
| Event | Description |
|-------|-------------|
| `graph_branch_created` | Graph debate branched |
| `graph_branch_merged` | Graph branches merged |

### Workflow Events
| Event | Description |
|-------|-------------|
| `breakpoint` | Human intervention breakpoint triggered |
| `breakpoint_resolved` | Breakpoint resolved |

### Other Events
| Event | Description |
|-------|-------------|
| `genesis_evolution` | Agent population evolved |
| `calibration_update` | System calibration updated |
| `evidence_found` | Evidence discovered |
| `explanation_ready` | Explanation generation completed |
| `receipt_ready` | Receipt generated |
| `receipt_exported` | Receipt exported |

## Payload Format

All webhook payloads follow this structure:

```json
{
  "event": "debate_end",
  "timestamp": 1705862400.0,
  "delivery_id": "del_xyz789",
  "data": {
    "debate_id": "deb_abc123",
    "result": {...},
    "correlation_id": "trace_123"
  }
}
```

### Headers

All webhook requests include these headers:

| Header | Description |
|--------|-------------|
| `Content-Type` | `application/json` |
| `User-Agent` | `Aragora-Webhooks/1.0` |
| `X-Aragora-Signature` | HMAC-SHA256 signature |
| `X-Aragora-Event` | Event type |
| `X-Aragora-Delivery` | Unique delivery ID |
| `X-Aragora-Timestamp` | Unix timestamp |
| `X-Aragora-Correlation-ID` | Distributed tracing ID |

## Signature Verification

Verify webhook authenticity using the signature:

### Python Example

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    """Verify webhook signature."""
    expected = "sha256=" + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)

# In your webhook handler:
@app.post("/webhook")
def handle_webhook(request):
    signature = request.headers.get("X-Aragora-Signature")
    payload = request.get_data()

    if not verify_webhook(payload, signature, WEBHOOK_SECRET):
        return "Invalid signature", 401

    # Process the webhook
    data = json.loads(payload)
    print(f"Received {data['event']} event")
    return "OK", 200
```

### Node.js Example

```javascript
const crypto = require('crypto');

function verifyWebhook(payload, signature, secret) {
  const expected = 'sha256=' + crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expected)
  );
}

// Express handler
app.post('/webhook', (req, res) => {
  const signature = req.headers['x-aragora-signature'];
  const payload = JSON.stringify(req.body);

  if (!verifyWebhook(payload, signature, process.env.WEBHOOK_SECRET)) {
    return res.status(401).send('Invalid signature');
  }

  console.log(`Received ${req.body.event} event`);
  res.status(200).send('OK');
});
```

## Batching

For high-volume scenarios, events can be batched together. Batched payloads have a different structure:

```json
{
  "event": "slo_violation_batch",
  "batch_size": 10,
  "window_start": 1705862400.0,
  "window_end": 1705862405.0,
  "events": [
    {"data": {...}, "timestamp": 1705862401.0},
    {"data": {...}, "timestamp": 1705862402.0}
  ],
  "summary": {
    "count": 10,
    "by_operation": {"km_query": 6, "debate_round": 4},
    "by_severity": {"major": 3, "minor": 7},
    "first_event_at": 1705862401.0,
    "last_event_at": 1705862405.0
  }
}
```

### Batching Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_WEBHOOK_BATCH_WINDOW` | `5.0` | Batch window in seconds |
| `ARAGORA_WEBHOOK_MAX_BATCH_SIZE` | `100` | Maximum events per batch |
| `ARAGORA_WEBHOOK_PRIORITY_EVENTS` | `slo_violation,debate_end,consensus,gauntlet_verdict` | Events that bypass batching |

## Error Handling

### Retry Logic

Failed deliveries are automatically retried with exponential backoff:

- **Attempt 1:** Immediate
- **Attempt 2:** After 1 second
- **Attempt 3:** After 2 seconds
- **Attempt 4:** After 4 seconds (max)

Retries occur for:
- Connection failures
- Timeouts (default: 30 seconds)
- 5xx server errors

**No retries** for 4xx client errors (considered permanent failures).

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_WEBHOOK_MAX_RETRIES` | `3` | Maximum retry attempts |
| `ARAGORA_WEBHOOK_RETRY_DELAY` | `1.0` | Initial retry delay (seconds) |
| `ARAGORA_WEBHOOK_MAX_RETRY_DELAY` | `60.0` | Maximum retry delay |
| `ARAGORA_WEBHOOK_TIMEOUT` | `30.0` | Request timeout (seconds) |
| `ARAGORA_WEBHOOK_WORKERS` | `10` | Concurrent delivery workers |

### Response Expectations

Your webhook endpoint should:

1. Return `2xx` status code to acknowledge receipt
2. Respond within 30 seconds (configurable)
3. Handle duplicate deliveries idempotently

## Best Practices

### Security

1. **Always verify signatures** - Never process webhooks without signature verification
2. **Use HTTPS** - Webhook URLs must use HTTPS (except localhost in development)
3. **Store secrets securely** - Use environment variables or secret managers
4. **Implement idempotency** - Use `delivery_id` to handle duplicate deliveries

### Reliability

1. **Respond quickly** - Process webhooks asynchronously; return 200 immediately
2. **Handle failures gracefully** - Log failures for investigation
3. **Monitor delivery rates** - Track success/failure rates via Prometheus metrics

### Monitoring

Available Prometheus metrics:

```
# Delivery counter
aragora_webhook_deliveries_total{event_type, success}

# Delivery duration histogram
aragora_webhook_delivery_duration_seconds{event_type}

# Retry counter
aragora_webhook_delivery_retries_total{event_type, attempt}

# Queue size gauge
aragora_webhook_queue_size

# Active endpoints gauge
aragora_webhook_active_endpoints{event_type}

# Failures by status code
aragora_webhook_failures_by_status_total{event_type, status_code}
```

### Example Grafana Dashboard

Import the SLO tracking dashboard from `deploy/grafana/dashboards/slo-tracking.json` for webhook delivery monitoring.

## Rate Limits

| Operation | Limit |
|-----------|-------|
| Register webhook | 10/minute |
| Test webhook | 5/minute |
| List webhooks | 60/minute |

## Storage Configuration

Webhook configurations are persisted and survive server restarts.

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_WEBHOOK_CONFIG_STORE_BACKEND` | `sqlite` | Storage backend: `sqlite`, `redis`, `memory` |
| `ARAGORA_DATA_DIR` | `./data` | Directory for SQLite database |
| `ARAGORA_REDIS_URL` | - | Redis URL for distributed deployments |
