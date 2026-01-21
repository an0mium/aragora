# Aragora Deployment

This directory contains deployment configurations for running Aragora in various environments.

## Quick Start

### Docker Compose (Development)

```bash
# Set environment variables
export ANTHROPIC_API_KEY=your-key
export OPENAI_API_KEY=your-key

# Start basic services
docker-compose up -d

# Start with monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Start with PostgreSQL
docker-compose --profile postgres up -d

# View logs
docker-compose logs -f backend
```

Access points:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8080
- Prometheus: http://localhost:9090 (with monitoring profile)
- Grafana: http://localhost:3001 (with monitoring profile)

### Kubernetes (Production)

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create config and secrets (edit secrets.yaml first!)
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Create persistent volumes
kubectl apply -f k8s/pvc.yaml

# Deploy backend and frontend
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml

# Create ingress and HPA
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl -n aragora get pods
kubectl -n aragora get svc
```

## Files

```
deploy/
├── Dockerfile.backend     # Backend container image
├── Dockerfile.frontend    # Frontend container image
├── docker-compose.yml     # Local development stack
├── monitoring/
│   ├── prometheus.yml     # Prometheus configuration
│   └── grafana/           # Grafana provisioning
└── k8s/
    ├── namespace.yaml     # Kubernetes namespace
    ├── configmap.yaml     # Non-sensitive config
    ├── secrets.yaml       # Sensitive config (template)
    ├── backend-deployment.yaml
    ├── frontend-deployment.yaml
    ├── ingress.yaml       # Ingress with TLS
    ├── hpa.yaml           # Auto-scaling
    └── pvc.yaml           # Persistent storage
```

## Environment Variables

### Required
| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `OPENAI_API_KEY` | OpenAI API key for GPT |

### Recommended
| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | Fallback for rate limits |
| `MISTRAL_API_KEY` | Mistral models |
| `ARAGORA_API_TOKEN` | API authentication token |

### Production
| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `ARAGORA_ALLOWED_ORIGINS` | CORS allowed origins |

## Building Images

```bash
# Build backend
docker build -f deploy/Dockerfile.backend -t aragora/backend:latest .

# Build frontend
docker build -f deploy/Dockerfile.frontend -t aragora/frontend:latest .

# Push to registry
docker push your-registry/aragora/backend:latest
docker push your-registry/aragora/frontend:latest
```

## Production Considerations

1. **Secrets Management**: Use sealed-secrets or external-secrets-operator
2. **TLS**: Configure cert-manager for automatic certificate management
3. **Monitoring**: Deploy Prometheus Operator for production monitoring
4. **Logging**: Configure FluentD/Loki for log aggregation
5. **Backup**: Set up PostgreSQL backups with pg_dump or Velero

## Health Checks

- Backend: `GET /api/health`
- Frontend: `GET /`
- Metrics: `GET /metrics` (backend)

## Troubleshooting

```bash
# Check pod logs
kubectl -n aragora logs -f deployment/aragora-backend

# Check events
kubectl -n aragora get events --sort-by=.metadata.creationTimestamp

# Restart deployment
kubectl -n aragora rollout restart deployment/aragora-backend
```
