# New Features Guide

This guide covers the new features added to Aragora: Template Marketplace, Decision Explainability, and Webhook System.

## Table of Contents

1. [Template Marketplace](#template-marketplace)
2. [Decision Explainability](#decision-explainability)
3. [Webhook System](#webhook-system)
4. [Background Workers](#background-workers)
5. [Monitoring & Metrics](#monitoring--metrics)
6. [CDC Database Connectors](#cdc-database-connectors)

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
GET /api/v1/debates/{debate_id}/explanation?format=json
```

Response:
```json
{
  "decision_id": "dec-abc123",
  "debate_id": "deb-123",
  "conclusion": "Adopt a tiered rate limiting strategy",
  "confidence": 0.87,
  "consensus_reached": true,
  "evidence_chain": [],
  "vote_pivots": [],
  "counterfactuals": []
}
```

#### Create Batch Job
```bash
POST /api/v1/explainability/batch
Content-Type: application/json

{
  "debate_ids": ["deb-1", "deb-2", "deb-3"],
  "options": {
    "include_evidence": true,
    "include_counterfactuals": false
  }
}
```

Response:
```json
{
  "batch_id": "batch-456",
  "status": "processing",
  "total": 3
}
```

#### Get Batch Job Status
```bash
GET /api/v1/explainability/batch/{batch_id}/status
```

#### Get Batch Results
```bash
GET /api/v1/explainability/batch/{batch_id}/results
```

### Python SDK Example

```python
from aragora.client import AragoraClient

client = AragoraClient(api_key="your-token")

# Single explanation
explanation = client.explainability.get_explanation("deb-123")

# Batch processing
batch = client.explainability.create_batch(
    debate_ids=["deb-1", "deb-2", "deb-3"],
    include_evidence=True,
)

# Poll for completion
while batch.status != "completed":
    batch = client.explainability.get_batch_status(batch.batch_id)
    time.sleep(1)

# Get results
results = client.explainability.get_batch_results(batch.batch_id)
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

---

## CDC Database Connectors

Change Data Capture (CDC) connectors enable real-time synchronization of database changes to Aragora's knowledge systems.

### Supported Databases

| Database | CDC Method | Features |
|----------|------------|----------|
| PostgreSQL | LISTEN/NOTIFY | Real-time triggers, incremental sync |
| MongoDB | Change Streams | Replica set streaming |
| MySQL | Binary Log (binlog) | Row-based replication, GTID support |
| SQL Server | CDC/Change Tracking | Enterprise CDC, lightweight tracking |
| Snowflake | Change Tracking | Time travel, stream-based changes |

### MySQL Connector

The MySQL connector supports both incremental sync and real-time CDC via binary log (binlog) parsing.

#### Configuration

```python
from aragora.connectors.enterprise.database import MySQLConnector

connector = MySQLConnector(
    host="localhost",
    port=3306,
    database="myapp",
    tables=["users", "orders", "products"],
    use_binlog=True,  # Enable binlog CDC
    server_id=100,    # Unique server ID for replication
    credentials={"username": "cdc_user", "password": "secret"},
)
```

#### Binlog Requirements

```sql
-- Enable binlog on MySQL server
SET GLOBAL binlog_format = 'ROW';
SET GLOBAL binlog_row_image = 'FULL';

-- Grant replication privileges
GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'cdc_user'@'%';
GRANT SELECT ON myapp.* TO 'cdc_user'@'%';
```

#### Usage

```python
# Add change event handler
from aragora.connectors.enterprise.database.cdc import CallbackHandler

async def on_change(event):
    print(f"Change: {event.operation} on {event.table}")
    print(f"Data: {event.data}")

connector.add_change_handler(CallbackHandler(on_change))

# Start CDC
await connector.start_binlog_cdc()

# Perform incremental sync
async for row in connector.sync():
    print(f"Synced: {row.evidence_id}")

# Stop CDC
await connector.stop_binlog_cdc()
await connector.close()
```

### SQL Server Connector

The SQL Server connector supports both CDC (Change Data Capture) and Change Tracking modes.

#### Configuration

```python
from aragora.connectors.enterprise.database import SQLServerConnector

# Using CDC (full change history)
cdc_connector = SQLServerConnector(
    host="localhost",
    port=1433,
    database="myapp",
    schema="dbo",
    tables=["users", "orders"],
    use_cdc=True,
    poll_interval_seconds=5,
    credentials={"username": "sa", "password": "secret"},
)

# Using Change Tracking (lightweight, current changes only)
ct_connector = SQLServerConnector(
    host="localhost",
    port=1433,
    database="myapp",
    use_change_tracking=True,
    credentials={"username": "sa", "password": "secret"},
)
```

#### CDC Setup (SQL Server)

```sql
-- Enable CDC on database
USE myapp;
EXEC sys.sp_cdc_enable_db;

-- Enable CDC on table
EXEC sys.sp_cdc_enable_table
    @source_schema = N'dbo',
    @source_name = N'users',
    @role_name = NULL,
    @supports_net_changes = 1;
```

#### Change Tracking Setup

```sql
-- Enable Change Tracking on database
ALTER DATABASE myapp SET CHANGE_TRACKING = ON
    (CHANGE_RETENTION = 2 DAYS, AUTO_CLEANUP = ON);

-- Enable Change Tracking on table
ALTER TABLE dbo.users ENABLE CHANGE_TRACKING
    WITH (TRACK_COLUMNS_UPDATED = ON);
```

#### Usage

```python
# Start CDC polling
await cdc_connector.start_cdc_polling()

# Or start Change Tracking polling
await ct_connector.start_change_tracking_polling()

# Both support the standard sync interface
async for row in cdc_connector.sync():
    print(f"Row: {row.content}")

# Stop polling
await cdc_connector.stop_cdc_polling()
await cdc_connector.close()
```

### CDC Event Model

All connectors emit standardized `ChangeEvent` objects:

```python
@dataclass
class ChangeEvent:
    id: str                        # Unique event ID
    source_type: CDCSourceType     # postgresql, mongodb, mysql, sqlserver
    connector_id: str              # Connector identifier
    operation: ChangeOperation     # INSERT, UPDATE, DELETE
    timestamp: datetime            # Event timestamp
    database: str                  # Database name
    table: str                     # Table/collection name
    data: Dict[str, Any]           # New data (INSERT/UPDATE)
    old_data: Dict[str, Any]       # Old data (UPDATE/DELETE)
    primary_key: Dict[str, Any]    # Primary key values
    resume_token: str              # For resuming streams
    sequence_number: int           # Event sequence
```

### Concurrency Support

The CDC system handles high-volume event processing:

- **100+ events/second** throughput
- **Concurrent handler execution** via `CompositeHandler`
- **Resume token persistence** for crash recovery
- **Knowledge Mound integration** for automatic indexing

```python
from aragora.connectors.enterprise.database.cdc import (
    CompositeHandler,
    KnowledgeMoundHandler,
    CallbackHandler,
)

# Combine multiple handlers
composite = CompositeHandler()
composite.add_handler(KnowledgeMoundHandler(mound))
composite.add_handler(CallbackHandler(log_change))
composite.add_handler(CallbackHandler(send_webhook))

connector.add_change_handler(composite)
```

### Health Monitoring

All database connectors support health checks:

```python
health = await connector.health_check()
# Returns: {"healthy": True, "latency_ms": 5.2, "details": {...}}
```

### Test Coverage

The CDC system includes comprehensive tests:

- **70+ unit tests** for MySQL and SQL Server connectors
- **17 concurrency tests** validating 50+ simultaneous event handling
- **Performance tests** for throughput and latency validation
