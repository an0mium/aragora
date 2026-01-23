---
title: Kubernetes Deployment Guide
description: Kubernetes Deployment Guide
---

# Kubernetes Deployment Guide

This guide covers deploying Aragora on Kubernetes for production use.

## Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured
- Helm 3.x (optional, for chart installation)
- PostgreSQL database
- Redis cluster (for horizontal scaling)

## Quick Start

### 1. Create Namespace

```bash
kubectl create namespace aragora
```

### 2. Configure Secrets

```bash
# Create secrets for API keys
kubectl create secret generic aragora-api-keys \
  --namespace aragora \
  --from-literal=ANTHROPIC_API_KEY='your-key' \
  --from-literal=OPENAI_API_KEY='your-key'

# Create secrets for database
kubectl create secret generic aragora-db \
  --namespace aragora \
  --from-literal=DATABASE_URL='postgresql://user:pass@host:5432/aragora'

# Create secrets for Redis
kubectl create secret generic aragora-redis \
  --namespace aragora \
  --from-literal=REDIS_URL='redis://host:6379'
```

### 3. Deploy Application

```bash
kubectl apply -f deploy/kubernetes/
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes (or OPENAI_API_KEY) |
| `OPENAI_API_KEY` | OpenAI API key | Yes (or ANTHROPIC_API_KEY) |
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `ARAGORA_REDIS_URL` | Redis connection string | Recommended |
| `ARAGORA_ENV` | Environment (production/staging) | Yes |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry collector | Optional |

### Resource Requirements

**Minimum (single replica):**
- CPU: 500m
- Memory: 1Gi

**Recommended (production):**
- CPU: 2000m
- Memory: 4Gi
- Replicas: 3+

## Deployment Manifests

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aragora
  namespace: aragora
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
        image: aragora/server:latest
        ports:
        - containerPort: 8080
        envFrom:
        - secretRef:
            name: aragora-api-keys
        - secretRef:
            name: aragora-db
        - secretRef:
            name: aragora-redis
        env:
        - name: ARAGORA_ENV
          value: "production"
        - name: ARAGORA_STATE_BACKEND
          value: "redis"
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: aragora
  namespace: aragora
spec:
  selector:
    app: aragora
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aragora
  namespace: aragora
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.aragora.ai
    secretName: aragora-tls
  rules:
  - host: api.aragora.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: aragora
            port:
              number: 80
```

### HorizontalPodAutoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aragora
  namespace: aragora
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aragora
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring

### Prometheus ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: aragora
  namespace: aragora
spec:
  selector:
    matchLabels:
      app: aragora
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### Grafana Dashboard

Import the dashboard from `deploy/grafana/dashboards/aragora-overview.json`.

## Scaling Considerations

### Horizontal Scaling

Aragora supports horizontal scaling with Redis-backed state:

1. Set `ARAGORA_STATE_BACKEND=redis`
2. Configure `ARAGORA_REDIS_URL`
3. Increase replica count

### Database Connections

Configure connection pooling:

```
DATABASE_URL=postgresql://user:pass@host:5432/aragora?pool_size=20&max_overflow=10
```

### WebSocket Sticky Sessions

For WebSocket connections, configure sticky sessions in your ingress:

```yaml
annotations:
  nginx.ingress.kubernetes.io/affinity: "cookie"
  nginx.ingress.kubernetes.io/session-cookie-name: "aragora-affinity"
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n aragora
kubectl describe pod <pod-name> -n aragora
kubectl logs <pod-name> -n aragora
```

### Health Check

```bash
kubectl exec -it <pod-name> -n aragora -- python -m aragora.cli.doctor
```

### Database Migration

```bash
kubectl exec -it <pod-name> -n aragora -- python -m aragora.migrations.run
```

## Security Best Practices

1. **Network Policies**: Restrict pod-to-pod communication
2. **RBAC**: Use service accounts with minimal permissions
3. **Secrets**: Use external secrets operator or vault
4. **TLS**: Enable TLS for all ingress traffic
5. **Pod Security**: Use restricted security context

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
```
