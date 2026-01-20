# Streaming Connectors Deployment Guide

This guide covers deploying Aragora with Kafka and RabbitMQ streaming connectors for enterprise event ingestion.

## Overview

Aragora's streaming connectors enable real-time ingestion of events from message queues into the Knowledge Mound for use in debates and decision-making. Both connectors:

- Convert messages to `SyncItem` objects for Knowledge Mound ingestion
- Support SSL/TLS for secure connections
- Provide circuit breaker protection for reliability
- Track consumption statistics for observability

## Prerequisites

### Kafka

```bash
pip install aiokafka
```

Optional for Avro/Protobuf:
```bash
pip install confluent-kafka[avro,protobuf]
```

### RabbitMQ

```bash
pip install aio-pika
```

## Configuration

### Environment Variables

#### Kafka

| Variable | Description | Default |
|----------|-------------|---------|
| `KAFKA_BOOTSTRAP_SERVERS` | Comma-separated broker list | `localhost:9092` |
| `KAFKA_TOPIC` | Topic to consume from | `aragora-events` |
| `KAFKA_GROUP_ID` | Consumer group ID | `aragora-consumer` |
| `KAFKA_AUTO_OFFSET_RESET` | Offset reset policy | `earliest` |
| `KAFKA_SSL_ENABLED` | Enable SSL/TLS | `false` |
| `KAFKA_SSL_CAFILE` | CA certificate path | - |
| `KAFKA_SSL_CERTFILE` | Client certificate path | - |
| `KAFKA_SSL_KEYFILE` | Client key path | - |
| `KAFKA_SASL_MECHANISM` | SASL mechanism | - |
| `KAFKA_SASL_USERNAME` | SASL username | - |
| `KAFKA_SASL_PASSWORD` | SASL password | - |

#### RabbitMQ

| Variable | Description | Default |
|----------|-------------|---------|
| `RABBITMQ_URL` | AMQP connection URL | `amqp://guest:guest@localhost/` |
| `RABBITMQ_QUEUE` | Queue name | `aragora-events` |
| `RABBITMQ_EXCHANGE` | Exchange name (optional) | - |
| `RABBITMQ_EXCHANGE_TYPE` | Exchange type | `direct` |
| `RABBITMQ_ROUTING_KEY` | Routing key | - |
| `RABBITMQ_PREFETCH_COUNT` | QoS prefetch | `10` |
| `RABBITMQ_DLX` | Dead letter exchange | - |
| `RABBITMQ_DLK` | Dead letter routing key | - |

## Docker Compose Setup

Add the following to your `docker-compose.yml`:

```yaml
services:
  # Kafka (using Redpanda for simplicity)
  redpanda:
    image: redpandadata/redpanda:v23.3.5
    command:
      - redpanda
      - start
      - --smp 1
      - --memory 1G
      - --reserve-memory 0M
      - --overprovisioned
      - --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092
      - --advertise-kafka-addr internal://redpanda:9092,external://localhost:19092
    ports:
      - "19092:19092"
    healthcheck:
      test: ["CMD", "rpk", "cluster", "health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # RabbitMQ
  rabbitmq:
    image: rabbitmq:3.12-management
    ports:
      - "5672:5672"   # AMQP
      - "15672:15672" # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: aragora
      RABBITMQ_DEFAULT_PASS: aragora_secret
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "check_running"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Aragora with streaming
  aragora:
    build: .
    depends_on:
      redpanda:
        condition: service_healthy
      rabbitmq:
        condition: service_healthy
    environment:
      # Kafka configuration
      KAFKA_BOOTSTRAP_SERVERS: redpanda:9092
      KAFKA_TOPIC: aragora-decisions
      KAFKA_GROUP_ID: aragora-prod
      # RabbitMQ configuration
      RABBITMQ_URL: amqp://aragora:aragora_secret@rabbitmq:5672/
      RABBITMQ_QUEUE: aragora-events
    ports:
      - "8080:8080"
```

## Kubernetes Deployment

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aragora-streaming-config
data:
  KAFKA_BOOTSTRAP_SERVERS: "kafka-broker-0.kafka:9092,kafka-broker-1.kafka:9092"
  KAFKA_TOPIC: "aragora-decisions"
  KAFKA_GROUP_ID: "aragora-prod"
  KAFKA_AUTO_OFFSET_RESET: "earliest"
  RABBITMQ_QUEUE: "aragora-events"
  RABBITMQ_PREFETCH_COUNT: "50"
```

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: aragora-streaming-secrets
type: Opaque
stringData:
  RABBITMQ_URL: "amqp://aragora:secret@rabbitmq.messaging:5672/"
  KAFKA_SASL_USERNAME: "aragora"
  KAFKA_SASL_PASSWORD: "kafka-secret"
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aragora
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aragora
  template:
    metadata:
      labels:
        app: aragora
    spec:
      containers:
      - name: aragora
        image: aragora:latest
        envFrom:
        - configMapRef:
            name: aragora-streaming-config
        - secretRef:
            name: aragora-streaming-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

## Usage

### Kafka Consumer

```python
from aragora.connectors.enterprise.streaming.kafka import (
    KafkaConnector,
    KafkaConfig,
)

# Configure
config = KafkaConfig(
    bootstrap_servers="kafka:9092",
    topic="decisions",
    group_id="aragora-prod",
)

# Create connector
connector = KafkaConnector(config)

# Connect and consume
await connector.start()

async for message in connector.consume():
    # Convert to SyncItem for Knowledge Mound
    sync_item = message.to_sync_item()
    await knowledge_mound.ingest(sync_item)

await connector.stop()
```

### RabbitMQ Consumer

```python
from aragora.connectors.enterprise.streaming.rabbitmq import (
    RabbitMQConnector,
    RabbitMQConfig,
)

# Configure with dead letter queue
config = RabbitMQConfig(
    url="amqp://user:pass@rabbitmq:5672/",
    queue="decisions",
    exchange="aragora",
    exchange_type="topic",
    routing_key="decisions.*",
    dead_letter_exchange="dlx",
    prefetch_count=50,
)

# Create connector
connector = RabbitMQConnector(config)

# Connect and consume with manual ack
await connector.start()

async for message in connector.consume():
    try:
        sync_item = message.to_sync_item()
        await knowledge_mound.ingest(sync_item)
        await message.ack()
    except Exception as e:
        await message.nack(requeue=True)

await connector.stop()
```

### Batch Sync

```python
# Sync a batch of messages to Knowledge Mound
async for sync_item in connector.sync(batch_size=100):
    await knowledge_mound.ingest(sync_item)
```

## Message Format

Both connectors accept JSON messages with optional metadata:

```json
{
  "type": "decision",
  "title": "Architecture Decision",
  "content": "We should use microservices for the payment system...",
  "source": "architecture-review",
  "timestamp": "2026-01-20T12:00:00Z",
  "metadata": {
    "author": "john@example.com",
    "tags": ["architecture", "payments"]
  }
}
```

The connectors extract:
- `title` - Used as SyncItem title (falls back to `type` or topic/queue name)
- `content` - Main content body
- Other fields stored in `metadata`

## Monitoring

### Prometheus Metrics

Both connectors expose metrics via the `/metrics` endpoint:

```
# Kafka
aragora_kafka_consumed_total{topic="decisions"} 1234
aragora_kafka_consumer_lag{topic="decisions",partition="0"} 56

# RabbitMQ
aragora_rabbitmq_consumed_total{queue="events"} 5678
aragora_rabbitmq_acked_total{queue="events"} 5670
aragora_rabbitmq_nacked_total{queue="events"} 8
```

### Health Checks

```bash
# Check connector health
curl http://localhost:8080/health/streaming

# Response
{
  "kafka": {
    "connected": true,
    "topic": "decisions",
    "consumed": 1234,
    "lag": 56
  },
  "rabbitmq": {
    "connected": true,
    "queue": "events",
    "consumed": 5678,
    "acked": 5670
  }
}
```

### Grafana Dashboard

Import the provided dashboard from `deploy/grafana/streaming-dashboard.json` for:
- Message throughput
- Consumer lag
- Error rates
- Processing latency

## Troubleshooting

### Kafka

**Consumer lag increasing:**
- Increase `batch_size` for higher throughput
- Scale horizontally with more consumer instances (same `group_id`)
- Check Knowledge Mound ingestion performance

**Connection refused:**
- Verify `bootstrap_servers` is correct
- Check firewall rules
- Ensure Kafka is running and accessible

**SSL errors:**
- Verify certificate paths and permissions
- Check certificate expiration
- Ensure correct CA chain

### RabbitMQ

**Queue not receiving messages:**
- Verify exchange binding with correct routing key
- Check publisher is using correct exchange/routing key
- Use RabbitMQ Management UI to inspect bindings

**Messages going to DLQ:**
- Check consumer error logs
- Verify message format matches expected schema
- Review requeue policy

**Connection drops:**
- Enable heartbeats in connection URL
- Check network stability
- Review RabbitMQ server logs

## Security Best Practices

1. **Use SSL/TLS** in production for both Kafka and RabbitMQ
2. **Enable SASL** authentication for Kafka
3. **Use separate credentials** for each environment
4. **Rotate credentials** regularly
5. **Limit topic/queue access** using ACLs
6. **Monitor for anomalies** in message patterns

## Performance Tuning

### Kafka

```python
config = KafkaConfig(
    batch_size=500,           # Increase for higher throughput
    batch_timeout_ms=1000,    # Reduce for lower latency
    # Consumer settings
    fetch_max_bytes=52428800, # 50MB
    max_poll_records=500,
)
```

### RabbitMQ

```python
config = RabbitMQConfig(
    prefetch_count=100,       # Higher for throughput, lower for fairness
    auto_ack=False,           # Manual ack for reliability
    batch_size=100,           # Batch processing size
)
```

## Further Reading

- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html)
- [aiokafka Documentation](https://aiokafka.readthedocs.io/)
- [aio-pika Documentation](https://aio-pika.readthedocs.io/)
