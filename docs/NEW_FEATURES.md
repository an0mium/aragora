# New Features Guide

This guide covers the new features added to Aragora: Template Marketplace, Decision Explainability, and Webhook System.

## Table of Contents

1. [Template Marketplace](#template-marketplace)
2. [Decision Explainability](#decision-explainability)
3. [Webhook System](#webhook-system)
4. [Background Workers](#background-workers)
5. [Monitoring & Metrics](#monitoring--metrics)

---

## Template Marketplace

The Template Marketplace allows users to share, discover, and import workflow templates.

### Features

- **Publish Templates**: Share your workflow configurations with the community
- **Search & Filter**: Find templates by category, tags, and ratings
- **Reviews & Ratings**: Rate templates and leave reviews
- **Version Control**: Templates support versioning

### API Endpoints

#### List Templates
```bash
GET /api/marketplace/templates?category=workflow&visibility=public&page=1&limit=20
```

#### Publish a Template
```bash
POST /api/marketplace/templates
Content-Type: application/json

{
  "name": "Security Scanner Workflow",
  "description": "Automated security scanning workflow",
  "category": "workflow",
  "visibility": "public",
  "template_data": {
    "nodes": [...],
    "edges": [...]
  },
  "tags": ["security", "automation"]
}
```

#### Download a Template
```bash
POST /api/marketplace/templates/{template_id}/download
```

#### Rate a Template
```bash
POST /api/marketplace/templates/{template_id}/rate
Content-Type: application/json

{
  "rating": 5
}
```

#### Submit a Review
```bash
POST /api/marketplace/templates/{template_id}/reviews
Content-Type: application/json

{
  "rating": 4,
  "title": "Great template!",
  "content": "Easy to use and well documented."
}
```

### Python SDK Example

```python
from aragora import AragonaClient

client = AragonaClient(api_token="your-token")

# List templates
templates = client.marketplace.list_templates(category="workflow")

# Publish a template
template = client.marketplace.publish({
    "name": "My Workflow",
    "category": "workflow",
    "template_data": my_workflow_config,
})

# Download a template
downloaded = client.marketplace.download(template_id="tmpl-123")
```

---

## Decision Explainability

Generate human-readable explanations for debate decisions, including reasoning chains and evidence.

### Features

- **Single Debate Explanations**: Get detailed explanations for any debate
- **Batch Processing**: Process multiple debates in parallel
- **Multiple Formats**: Summary, detailed, or visual explanations
- **Reasoning Chains**: Trace the logic path to conclusions

### API Endpoints

#### Get Explanation for a Debate
```bash
GET /api/explainability/debate/{debate_id}?format=detailed
```

Response:
```json
{
  "debate_id": "deb-123",
  "summary": "The agents reached consensus after 3 rounds...",
  "reasoning_chain": [
    {"step": 1, "agent": "claude", "action": "Proposed security-first approach"},
    {"step": 2, "agent": "gpt4", "action": "Challenged scalability concerns"},
    {"step": 3, "agent": "claude", "action": "Provided evidence from benchmarks"}
  ],
  "key_factors": ["performance data", "security requirements"],
  "consensus_strength": 0.85
}
```

#### Create Batch Job
```bash
POST /api/explainability/batch
Content-Type: application/json

{
  "debate_ids": ["deb-1", "deb-2", "deb-3"],
  "options": {
    "format": "summary",
    "include_evidence": true
  }
}
```

Response:
```json
{
  "job_id": "batch-456",
  "status": "processing",
  "total": 3,
  "processed": 0
}
```

#### Get Batch Job Status
```bash
GET /api/explainability/batch/{job_id}
```

#### Get Batch Results
```bash
GET /api/explainability/batch/{job_id}/results
```

### Python SDK Example

```python
from aragora import AragonaClient

client = AragonaClient(api_token="your-token")

# Single explanation
explanation = client.explainability.explain(
    debate_id="deb-123",
    format="detailed"
)

# Batch processing
batch = client.explainability.create_batch(
    debate_ids=["deb-1", "deb-2", "deb-3"],
    options={"format": "summary"}
)

# Poll for completion
while batch.status != "completed":
    batch = client.explainability.get_batch(batch.job_id)
    time.sleep(1)

# Get results
results = client.explainability.get_batch_results(batch.job_id)
```

---

## Webhook System

Register webhooks to receive real-time notifications about debate events.

### Features

- **Event Subscriptions**: Subscribe to specific event types
- **HMAC Signing**: Secure delivery with signature verification
- **Delivery Receipts**: Track delivery status and history
- **Circuit Breakers**: Automatic failover for unreliable endpoints
- **Retry with Backoff**: Exponential backoff for failed deliveries

### Supported Events

| Event | Description |
|-------|-------------|
| `debate.started` | A new debate has begun |
| `debate.completed` | A debate has finished |
| `consensus.reached` | Agents reached consensus |
| `agent.error` | An agent encountered an error |

### API Endpoints

#### Register a Webhook
```bash
POST /api/webhooks
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/aragora",
  "secret": "your-hmac-secret",
  "event_types": ["debate.completed", "consensus.reached"],
  "description": "Production webhook"
}
```

#### List Webhooks
```bash
GET /api/webhooks
```

#### Update a Webhook
```bash
PUT /api/webhooks/{webhook_id}
Content-Type: application/json

{
  "enabled": false
}
```

#### Test Webhook Delivery
```bash
POST /api/webhooks/{webhook_id}/test
```

#### View Delivery Receipts
```bash
GET /api/webhooks/{webhook_id}/receipts?status=failed&limit=10
```

### Webhook Payload Format

```json
{
  "event_type": "debate.completed",
  "event_id": "evt-789",
  "timestamp": "2024-01-20T12:00:00Z",
  "data": {
    "debate_id": "deb-123",
    "task": "Design a rate limiter",
    "outcome": "consensus",
    "consensus_strength": 0.92
  }
}
```

### Verifying Signatures

Webhooks are signed using HMAC-SHA256. Verify the signature in your handler:

```python
import hmac
import hashlib

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)

# In your webhook handler:
@app.post("/webhooks/aragora")
def handle_webhook(request):
    signature = request.headers.get("X-Signature-SHA256")
    if not verify_signature(request.body, signature, WEBHOOK_SECRET):
        return Response(status=401)

    event = request.json()
    # Process event...
```

### Python SDK Example

```python
from aragora import AragonaClient

client = AragonaClient(api_token="your-token")

# Register webhook
webhook = client.webhooks.register(
    url="https://your-app.com/webhooks",
    secret="your-secret",
    event_types=["debate.completed"]
)

# Test delivery
result = client.webhooks.test(webhook.id)
print(f"Test delivery: {result.status}")

# Check receipts
receipts = client.webhooks.get_receipts(webhook.id, status="failed")
```

---

## Background Workers

The new features include background workers for processing jobs asynchronously.

### Batch Explainability Worker

Processes batch explanation jobs with:
- Concurrent debate processing
- Progress tracking
- Graceful shutdown

### Webhook Delivery Worker

Handles webhook delivery with:
- Per-endpoint circuit breakers
- Exponential backoff retry
- HMAC signature generation
- Delivery tracking

### Starting Workers

Workers are automatically started with the unified server. For manual control:

```python
from aragora.queue.batch_worker import BatchExplainabilityWorker
from aragora.queue.webhook_worker import WebhookDeliveryWorker

# Start batch worker
batch_worker = BatchExplainabilityWorker(
    queue=job_queue,
    worker_id="batch-1",
    max_concurrent_debates=10,
)
await batch_worker.start()

# Start webhook worker
webhook_worker = WebhookDeliveryWorker(
    queue=job_queue,
    worker_id="webhook-1",
    max_concurrent=50,
)
await webhook_worker.start()
```

---

## Monitoring & Metrics

### Prometheus Metrics

New metrics are available for monitoring:

#### Marketplace Metrics
- `aragora_marketplace_templates_total` - Template count by category
- `aragora_marketplace_downloads_total` - Download count
- `aragora_marketplace_ratings_distribution` - Rating histogram

#### Batch Explainability Metrics
- `aragora_explainability_batch_jobs_active` - Active jobs
- `aragora_explainability_batch_jobs_total` - Total jobs by status
- `aragora_explainability_batch_processing_latency_seconds` - Processing time

#### Webhook Metrics
- `aragora_webhook_deliveries_total` - Delivery attempts
- `aragora_webhook_delivery_latency_seconds` - Delivery latency
- `aragora_webhook_failures_by_endpoint_total` - Failures by endpoint
- `aragora_webhook_circuit_breaker_state` - Circuit breaker states

### Grafana Dashboards

Import the provided dashboards from `deploy/monitoring/grafana/`:
- `marketplace-metrics.json`
- `batch-explainability.json`
- `webhooks-receipts.json`

### Alerting

Alerting rules are configured in `deploy/monitoring/alerts.yaml`:
- Marketplace publish failures
- Batch job processing delays
- Webhook delivery failures
- Circuit breaker activations

---

## Database Schema

New tables are created by the migration `v20260120100000_marketplace_webhooks_batch`:

| Table | Purpose |
|-------|---------|
| `marketplace_templates` | Template storage |
| `marketplace_reviews` | Reviews and ratings |
| `webhook_registrations` | Webhook configurations |
| `webhook_delivery_receipts` | Delivery tracking |
| `batch_explainability_jobs` | Batch job status |
| `batch_explainability_results` | Individual results |

Run migrations with:
```bash
python -m aragora.migrations
```

---

## Security Considerations

### SSRF Protection

Webhook URLs are validated to prevent Server-Side Request Forgery:
- Private IP ranges are blocked
- Loopback addresses are blocked
- Cloud metadata endpoints are blocked

### RBAC Permissions

New permissions are required for these features:

| Permission | Description |
|------------|-------------|
| `marketplace:read` | View templates |
| `marketplace:publish` | Publish templates |
| `marketplace:rate` | Rate templates |
| `explainability:read` | View explanations |
| `explainability:batch` | Create batch jobs |
| `webhooks:read` | View webhooks |
| `webhooks:write` | Create/update webhooks |
| `webhooks:delete` | Delete webhooks |

---

## Troubleshooting

### Webhook Delivery Failures

1. Check the delivery receipts for error messages
2. Verify your endpoint is publicly accessible
3. Ensure HMAC secret matches
4. Check circuit breaker state

### Batch Job Stuck

1. Check worker status via `/api/workers/status`
2. Review job progress via batch status endpoint
3. Check for errors in individual results

### Template Not Found

1. Verify template ID is correct
2. Check visibility settings
3. Ensure user has required permissions
