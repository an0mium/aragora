# Aragora Deployment Guide

This guide covers deploying Aragora to production environments.

## Prerequisites

- Docker 20.10+
- Kubernetes 1.25+ (for K8s deployment)
- At least one AI provider API key (Anthropic, OpenAI, etc.)

## Quick Start: Docker Compose

The simplest way to run Aragora in production:

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env with your API keys
vim .env

# 3. Start services
docker compose up -d

# 4. Check health
curl http://localhost:8080/api/health
```

## Kubernetes Deployment

### 1. Prepare Secrets

First, create your secrets file from the template:

```bash
cp deploy/k8s/secret.yaml deploy/k8s/secret-local.yaml
# Edit with your actual values
vim deploy/k8s/secret-local.yaml
```

For production, use [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets):

```bash
kubeseal --format yaml < deploy/k8s/secret-local.yaml > deploy/k8s/sealed-secret.yaml
```

### 2. Build and Push Docker Image

```bash
# Build production image
docker build -t your-registry/aragora:latest .

# Push to registry
docker push your-registry/aragora:latest
```

### 3. Update Kustomization

Edit `deploy/k8s/kustomization.yaml`:

```yaml
images:
  - name: aragora
    newName: your-registry/aragora
    newTag: v1.0.0
```

### 4. Deploy

```bash
# Apply all resources
kubectl apply -k deploy/k8s/

# Watch rollout
kubectl -n aragora rollout status deployment/aragora

# Check pods
kubectl -n aragora get pods
```

### 5. Configure Ingress

Edit `deploy/k8s/ingress.yaml` with your domain:

```yaml
spec:
  tls:
    - hosts:
        - aragora.yourdomain.com
      secretName: aragora-tls
  rules:
    - host: aragora.yourdomain.com
```

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `OPENAI_API_KEY` | OpenAI API key (alternative to Anthropic) |

### Recommended

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis for rate limiting and sessions |
| `JWT_SECRET_KEY` | (required for auth) | 32+ character secret for JWT tokens |
| `ARAGORA_API_TOKEN` | (optional) | API token for authenticated endpoints |

### Optional Providers

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter for fallback |
| `GEMINI_API_KEY` | Google Gemini |
| `XAI_API_KEY` | xAI Grok |
| `MISTRAL_API_KEY` | Mistral AI |

### Billing (Optional)

| Variable | Description |
|----------|-------------|
| `STRIPE_SECRET_KEY` | Stripe API key |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret |

## Resource Requirements

### Minimum (Development)

- CPU: 0.5 cores
- Memory: 512MB
- Storage: 1GB

### Recommended (Production)

- CPU: 2 cores
- Memory: 2GB
- Storage: 10GB
- Redis: 256MB

### Scaling Guidelines

| Concurrent Debates | Replicas | CPU | Memory |
|--------------------|----------|-----|--------|
| 1-5 | 1 | 1 core | 1GB |
| 5-20 | 2-3 | 2 cores | 2GB |
| 20-50 | 3-5 | 4 cores | 4GB |
| 50+ | 5-10 | 8 cores | 8GB |

## Health Checks

### Liveness Probe
```
GET /api/v2/health
```
Returns 200 if server is running.

### Readiness Probe
```
GET /api/v2/health/ready
```
Returns 200 if server can accept requests.

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics` (port 8080):

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'aragora'
    static_configs:
      - targets: ['aragora:8080']
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `aragora_debates_total` | Total debates run |
| `aragora_debate_duration_seconds` | Debate duration histogram |
| `aragora_agent_errors_total` | Agent error count by type |
| `aragora_consensus_rate` | Consensus achievement rate |

## Troubleshooting

### Pod CrashLoopBackOff

1. Check logs: `kubectl -n aragora logs deploy/aragora`
2. Verify secrets: `kubectl -n aragora get secret aragora-secrets -o yaml`
3. Check resource limits

### Redis Connection Failed

1. Verify Redis is running: `kubectl -n aragora get pods -l app.kubernetes.io/name=aragora-redis`
2. Check service: `kubectl -n aragora get svc aragora-redis`
3. Test connection: `kubectl -n aragora exec -it deploy/aragora -- redis-cli -h aragora-redis ping`

### High Memory Usage

1. Check debate queue: Reduce `ARAGORA_MAX_CONCURRENT_DEBATES`
2. Enable memory limits in deployment
3. Consider horizontal scaling via HPA

## Security Recommendations

1. **Use Sealed Secrets or External Secrets** for API keys
2. **Enable TLS** via cert-manager or your ingress controller
3. **Set resource limits** to prevent resource exhaustion
4. **Use NetworkPolicies** to restrict traffic
5. **Enable Pod Security Standards** (restricted profile)

## Backup and Recovery

### Database Backup

```bash
# SQLite backup (if using default storage)
kubectl -n aragora exec deploy/aragora -- sqlite3 /app/data/aragora.db ".backup /tmp/backup.db"
kubectl -n aragora cp aragora-0:/tmp/backup.db ./aragora-backup.db
```

### Redis Backup

```bash
# Trigger RDB snapshot
kubectl -n aragora exec aragora-redis-0 -- redis-cli BGSAVE
```
