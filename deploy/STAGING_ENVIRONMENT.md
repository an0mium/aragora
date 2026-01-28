# Staging Environment Setup Guide

This guide covers setting up and managing the Aragora staging environment.

## Overview

The staging environment mirrors production but with:
- **Reduced resources** for cost efficiency
- **Debug logging** enabled
- **Faster deployments** (no PDB, no approval gates)
- **Isolated data** from production
- **More permissive rate limits** for testing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Staging Cluster (EKS)                    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Ingress    │  │   Aragora    │  │   Workers    │     │
│  │   (nginx)    │  │   (2 pods)   │  │  (optional)  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘     │
│         │                  │                               │
│         ▼                  ▼                               │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │   Cert-      │  │   External   │                       │
│  │   Manager    │  │   Secrets    │                       │
│  └──────────────┘  └──────────────┘                       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │  PostgreSQL  │  │    Redis     │                       │
│  │  (staging)   │  │  (staging)   │                       │
│  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
         │                     │
         ▼                     ▼
┌─────────────────┐  ┌─────────────────┐
│ AWS Secrets     │  │  S3 (staging)   │
│ Manager         │  │                 │
└─────────────────┘  └─────────────────┘
```

## Prerequisites

1. **Kubernetes cluster** (EKS recommended)
2. **kubectl** configured with cluster access
3. **Helm 3** installed
4. **AWS CLI** configured (for secrets)
5. **ArgoCD** installed (optional, for GitOps)

## Quick Start

### Option 1: Helm Installation

```bash
# Create namespace
kubectl create namespace aragora-staging

# Install with staging values
helm install aragora ./deploy/helm/aragora \
  -n aragora-staging \
  -f deploy/helm/aragora/values-staging.yaml
```

### Option 2: ArgoCD (GitOps)

The staging environment is defined in `deploy/argocd/applications/applicationset.yaml`:

```bash
# Apply ArgoCD project
kubectl apply -f deploy/argocd/projects/aragora-project.yaml

# The ApplicationSet will auto-create the staging application
```

## Step-by-Step Setup

### Step 1: Create Namespace

```bash
kubectl create namespace aragora-staging

# Add labels for Kyverno policy selection
kubectl label namespace aragora-staging \
  environment=staging \
  aragora.ai/managed=true
```

### Step 2: Configure Secrets

#### Option A: AWS Secrets Manager (Recommended)

```bash
# Create secret in AWS Secrets Manager
aws secretsmanager create-secret \
  --name aragora/staging/config \
  --secret-string '{
    "database-url": "postgresql://aragora:password@postgres:5432/aragora",
    "redis-url": "redis://:password@redis:6379/0",
    "anthropic-api-key": "sk-ant-...",
    "openai-api-key": "sk-..."
  }'

# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets --create-namespace

# Apply ClusterSecretStore
kubectl apply -f deploy/k8s/external-secrets/cluster-secret-store.yaml
```

#### Option B: Kubernetes Secrets (Development)

```bash
# Create secrets directly (not recommended for production)
kubectl create secret generic aragora-staging-secrets \
  -n aragora-staging \
  --from-literal=database-url='postgresql://aragora:password@postgres:5432/aragora' \
  --from-literal=redis-url='redis://:password@redis:6379/0' \
  --from-literal=anthropic-api-key='sk-ant-...'
```

### Step 3: Install Dependencies

```bash
# Add Helm repos
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install PostgreSQL
helm install postgres bitnami/postgresql \
  -n aragora-staging \
  --set auth.database=aragora \
  --set auth.existingSecret=aragora-staging-secrets \
  --set primary.persistence.size=10Gi

# Install Redis
helm install redis bitnami/redis \
  -n aragora-staging \
  --set architecture=standalone \
  --set auth.existingSecret=aragora-staging-secrets \
  --set master.persistence.size=5Gi
```

### Step 4: Install Aragora

```bash
# Install Aragora with staging values
helm install aragora ./deploy/helm/aragora \
  -n aragora-staging \
  -f deploy/helm/aragora/values-staging.yaml

# Wait for rollout
kubectl rollout status deployment/aragora -n aragora-staging
```

### Step 5: Configure Ingress

```bash
# Install nginx-ingress (if not already installed)
helm install ingress-nginx ingress-nginx/ingress-nginx \
  -n ingress-nginx --create-namespace

# Install cert-manager (if not already installed)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.0/cert-manager.yaml

# Apply staging ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-staging
spec:
  acme:
    server: https://acme-staging-v02.api.letsencrypt.org/directory
    email: admin@aragora.ai
    privateKeySecretRef:
      name: letsencrypt-staging
    solvers:
      - http01:
          ingress:
            class: nginx
EOF
```

### Step 6: Verify Installation

```bash
# Check pods
kubectl get pods -n aragora-staging

# Check services
kubectl get svc -n aragora-staging

# Check ingress
kubectl get ingress -n aragora-staging

# Test health endpoint
kubectl port-forward svc/aragora 8080:8080 -n aragora-staging
curl http://localhost:8080/health
```

## Environment Differences

| Aspect | Staging | Production |
|--------|---------|------------|
| Replicas | 2 | 3+ per region |
| Resources | 500m/512Mi | 1000m/1Gi |
| Autoscaling | Disabled | Enabled (2-10) |
| PDB | Disabled | minAvailable: 1 |
| Logging | Debug | Info |
| Rate Limits | 1000/min | 100/min |
| TLS Issuer | LE Staging | LE Production |
| Database | Shared staging | Dedicated per region |
| Secrets Refresh | 5 min | 1 hour |

## Deploying Updates

### Manual Deployment

```bash
# Update image tag
helm upgrade aragora ./deploy/helm/aragora \
  -n aragora-staging \
  -f deploy/helm/aragora/values-staging.yaml \
  --set image.tag=v1.2.3

# Or force a redeployment
kubectl rollout restart deployment/aragora -n aragora-staging
```

### CI/CD Pipeline

The staging environment auto-deploys on:
- Push to `develop` branch
- Pull request merge to `main` (before production)

```yaml
# .github/workflows/deploy-staging.yaml
name: Deploy to Staging
on:
  push:
    branches: [develop]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to Staging
        run: |
          helm upgrade --install aragora ./deploy/helm/aragora \
            -n aragora-staging \
            -f deploy/helm/aragora/values-staging.yaml \
            --set image.tag=${{ github.sha }}
```

### ArgoCD Sync

If using ArgoCD:

```bash
# Manual sync
argocd app sync aragora-staging

# Check status
argocd app get aragora-staging
```

## Testing in Staging

### Run Integration Tests

```bash
# Port forward for local testing
kubectl port-forward svc/aragora 8080:8080 -n aragora-staging &

# Run test suite against staging
ARAGORA_API_URL=http://localhost:8080 \
  pytest tests/integration/ -v
```

### Load Testing

```bash
# Install k6
brew install k6

# Run load test
k6 run tests/load/staging-load-test.js \
  -e STAGING_URL=https://staging.aragora.ai
```

### Debug Access

```bash
# Get shell in pod
kubectl exec -it deploy/aragora -n aragora-staging -- /bin/sh

# View logs
kubectl logs -f deploy/aragora -n aragora-staging

# View debug logs
kubectl logs -f deploy/aragora -n aragora-staging --tail=100

# Port forward for pprof
kubectl port-forward svc/aragora 6060:6060 -n aragora-staging
# Then access http://localhost:6060/debug/pprof/
```

## Monitoring

### Prometheus Metrics

Staging exports metrics at `/metrics`:

```bash
# Port forward to Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring

# Query staging metrics
# rate(http_requests_total{namespace="aragora-staging"}[5m])
```

### Grafana Dashboards

Access Grafana for staging dashboards:

```bash
kubectl port-forward svc/grafana 3000:3000 -n monitoring
# Dashboard: "Aragora Staging Overview"
```

### Log Aggregation

Logs are shipped to Loki/CloudWatch:

```bash
# Using Loki
logcli query '{namespace="aragora-staging"}'

# Using CloudWatch
aws logs tail /aws/eks/aragora-staging --follow
```

## Troubleshooting

### Common Issues

#### Pods Not Starting

```bash
# Check events
kubectl get events -n aragora-staging --sort-by='.lastTimestamp'

# Check pod details
kubectl describe pod -l app=aragora -n aragora-staging
```

#### Database Connection Failed

```bash
# Test database connectivity
kubectl run pg-test --rm -it --image=postgres:15 \
  -n aragora-staging \
  -- psql $DATABASE_URL -c "SELECT 1"
```

#### Secrets Not Loading

```bash
# Check ExternalSecret status
kubectl get externalsecret -n aragora-staging
kubectl describe externalsecret aragora-staging-secrets -n aragora-staging
```

#### High Memory Usage

Staging has debug logging enabled which can increase memory:

```bash
# Reduce log level temporarily
kubectl set env deployment/aragora LOG_LEVEL=warning -n aragora-staging

# Or restart pods
kubectl rollout restart deployment/aragora -n aragora-staging
```

### Reset Staging

To completely reset the staging environment:

```bash
# Delete namespace (removes everything)
kubectl delete namespace aragora-staging

# Recreate from scratch
kubectl create namespace aragora-staging
helm install aragora ./deploy/helm/aragora \
  -n aragora-staging \
  -f deploy/helm/aragora/values-staging.yaml
```

## Security Considerations

1. **Secrets**: Use External Secrets Operator, never commit secrets to git
2. **Network**: Consider NetworkPolicies even in staging
3. **Access**: Limit kubectl access with RBAC
4. **Data**: Never use production data in staging
5. **API Keys**: Use separate API keys for staging (not production keys)

## Cost Optimization

Staging is designed for cost efficiency:

- **Right-sized resources**: 50% of production
- **No autoscaling**: Fixed 2 replicas
- **Standalone databases**: No replication
- **Spot instances**: Consider using spot nodes

Estimated monthly cost: ~$150-200/month (AWS)

## Related Documentation

- [Production Deployment Guide](../docs/DEPLOYMENT.md)
- [ArgoCD Configuration](./argocd/README.md)
- [Disaster Recovery](./DISASTER_RECOVERY.md)
- [Deployment Checklist](./DEPLOYMENT_CHECKLIST.md)
- [Secrets Management](../docs-site/docs/deployment/secrets-management.md)
